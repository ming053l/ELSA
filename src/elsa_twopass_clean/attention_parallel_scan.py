from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import triton
import triton.language as tl

from .attention import _canonical_bias, _next_power_of_2, _phase1_summary_kernel, _validate_warps


@triton.jit
def _summary_scan_step_kernel(
    M_IN,
    Z_IN,
    S_IN,
    M_OUT,
    Z_OUT,
    S_OUT,
    STEP: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BH: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_kb = tl.program_id(0)
    pid_qb = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D_HEAD

    cur_base = ((pid_kb * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m
    left_kb = pid_kb - STEP
    left_base = ((left_kb * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m
    has_left = pid_kb >= STEP

    m_cur = tl.load(M_IN + cur_base).to(tl.float32)
    z_cur = tl.load(Z_IN + cur_base).to(tl.float32)
    s_cur = tl.load(S_IN + cur_base[:, None] * D_HEAD + offs_d[None, :], mask=d_mask[None, :], other=0.0).to(
        tl.float32
    )

    m_left = tl.load(M_IN + left_base, mask=has_left, other=-float("inf")).to(tl.float32)
    z_left = tl.load(Z_IN + left_base, mask=has_left, other=0.0).to(tl.float32)
    s_left = tl.load(
        S_IN + left_base[:, None] * D_HEAD + offs_d[None, :],
        mask=has_left & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    m = tl.maximum(m_left, m_cur)
    alpha = tl.where(z_left > 0.0, tl.exp2(m_left - m), 0.0)
    beta = tl.where(z_cur > 0.0, tl.exp2(m_cur - m), 0.0)
    z = z_left * alpha + z_cur * beta
    s = s_left * alpha[:, None] + s_cur * beta[:, None]
    m = tl.where(z > 0.0, m, -float("inf"))

    tl.store(M_OUT + cur_base, m)
    tl.store(Z_OUT + cur_base, z)
    tl.store(S_OUT + cur_base[:, None] * D_HEAD + offs_d[None, :], s, mask=d_mask[None, :])


@triton.jit
def _summary_blelloch_upsweep_kernel(
    M_IN,
    Z_IN,
    S_IN,
    M_OUT,
    Z_OUT,
    S_OUT,
    N_BLOCKS_IN: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BH: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_ob = tl.program_id(0)
    pid_qb = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D_HEAD
    left = pid_ob * 2
    right = left + 1

    left_base = ((left * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m
    right_base = ((right * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m
    out_base = ((pid_ob * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m

    m_a = tl.load(M_IN + left_base).to(tl.float32)
    z_a = tl.load(Z_IN + left_base).to(tl.float32)
    s_a = tl.load(S_IN + left_base[:, None] * D_HEAD + offs_d[None, :], mask=d_mask[None, :], other=0.0).to(
        tl.float32
    )

    right_active = right < N_BLOCKS_IN
    m_b = tl.load(M_IN + right_base, mask=right_active, other=-float("inf")).to(tl.float32)
    z_b = tl.load(Z_IN + right_base, mask=right_active, other=0.0).to(tl.float32)
    s_b = tl.load(
        S_IN + right_base[:, None] * D_HEAD + offs_d[None, :],
        mask=right_active & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    m = tl.maximum(m_a, m_b)
    alpha = tl.where(z_a > 0.0, tl.exp2(m_a - m), 0.0)
    beta = tl.where(z_b > 0.0, tl.exp2(m_b - m), 0.0)
    z = z_a * alpha + z_b * beta
    s = s_a * alpha[:, None] + s_b * beta[:, None]
    m = tl.where(z > 0.0, m, -float("inf"))

    tl.store(M_OUT + out_base, m)
    tl.store(Z_OUT + out_base, z)
    tl.store(S_OUT + out_base[:, None] * D_HEAD + offs_d[None, :], s, mask=d_mask[None, :])


@triton.jit
def _phase1_d64_fp16_nomask_kernel(
    Q,
    K,
    V,
    M_BUF,
    Z_BUF,
    S_BUF,
    Q_START: tl.constexpr,
    N_CTX: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG2_SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 64)

    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    k_idx = pid_kb * BLOCK_N + offs_n

    if NEEDS_MASK:
        k_mask = k_idx < N_CTX
        q = (tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :],
                     mask=(q_idx[:, None] < N_CTX), other=0.0) * LOG2_SCALE).to(tl.float16)
        k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :], mask=k_mask[:, None], other=0.0)
        v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :], mask=k_mask[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        scores = tl.where(k_mask[None, :], scores, -float("inf"))
    else:
        q = (tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :]) * LOG2_SCALE).to(tl.float16)
        k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :])
        v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :])
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
    local_m = tl.max(scores, axis=1)
    p = tl.exp2(scores - local_m[:, None])
    local_z = tl.sum(p, axis=1)
    local_s = tl.dot(p.to(tl.float16), v, input_precision=INPUT_PRECISION)

    base = ((pid_kb * Q_BLOCKS + pid_qb) * tl.num_programs(1) + pid_bh) * BLOCK_M + offs_m
    tl.store(M_BUF + base, local_m)
    tl.store(Z_BUF + base, local_z)
    tl.store(S_BUF + base[:, None] * 64 + offs_d[None, :], local_s)


@triton.jit
def _phase1_d64_fp32_nomask_kernel(
    Q,
    K,
    V,
    M_BUF,
    Z_BUF,
    S_BUF,
    Q_START: tl.constexpr,
    N_CTX: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG2_SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 64)

    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    k_idx = pid_kb * BLOCK_N + offs_n

    if NEEDS_MASK:
        k_mask = k_idx < N_CTX
        q = tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :],
                    mask=(q_idx[:, None] < N_CTX), other=0.0).to(tl.float32) * LOG2_SCALE
        k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :], mask=k_mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :], mask=k_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        scores = tl.where(k_mask[None, :], scores, -float("inf"))
    else:
        q = tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32) * LOG2_SCALE
        k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32)
        v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32)
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
    local_m = tl.max(scores, axis=1)
    p = tl.exp2(scores - local_m[:, None])
    local_z = tl.sum(p, axis=1)
    local_s = tl.dot(p, v, input_precision=INPUT_PRECISION)

    base = ((pid_kb * Q_BLOCKS + pid_qb) * tl.num_programs(1) + pid_bh) * BLOCK_M + offs_m
    tl.store(M_BUF + base, local_m)
    tl.store(Z_BUF + base, local_z)
    tl.store(S_BUF + base[:, None] * 64 + offs_d[None, :], local_s)


@triton.jit
def _phase1_d64_tiled_fp16_nomask_kernel(
    Q,
    K,
    V,
    M_BUF,
    Z_BUF,
    S_BUF,
    Q_START: tl.constexpr,
    N_CTX: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOGICAL_BLOCK_N: tl.constexpr,
    MICRO_BLOCK_N: tl.constexpr,
    LOG2_SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MICRO_BLOCK_N)
    offs_d = tl.arange(0, 64)

    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    # NEEDS_MASK: see fp32 kernel — keep partial q loads in-bounds and drop out-of-range keys.
    if NEEDS_MASK:
        q = (tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :],
                     mask=(q_idx[:, None] < N_CTX), other=0.0) * LOG2_SCALE).to(tl.float16)
    else:
        q = (tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :]) * LOG2_SCALE).to(tl.float16)

    m_run = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    z_run = tl.zeros((BLOCK_M,), dtype=tl.float32)
    s_run = tl.zeros((BLOCK_M, 64), dtype=tl.float32)

    logical_start = pid_kb * LOGICAL_BLOCK_N
    for micro_start in tl.range(0, LOGICAL_BLOCK_N, MICRO_BLOCK_N):
        k_idx = logical_start + micro_start + offs_n
        if NEEDS_MASK:
            k_mask = k_idx < N_CTX
            k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :],
                        mask=k_mask[:, None], other=0.0)
            v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :],
                        mask=k_mask[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
            scores = tl.where(k_mask[None, :], scores, -float("inf"))
        else:
            k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :])
            v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :])
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        local_m = tl.max(scores, axis=1)
        p = tl.where(local_m[:, None] == float("-inf"), 0.0, tl.exp2(scores - local_m[:, None]))
        local_z = tl.sum(p, axis=1)
        local_s = tl.dot(p.to(tl.float16), v, input_precision=INPUT_PRECISION)

        m_new = tl.maximum(m_run, local_m)
        alpha = tl.where(z_run > 0.0, tl.exp2(m_run - m_new), 0.0)
        beta = tl.exp2(local_m - m_new)
        z_run = z_run * alpha + local_z * beta
        s_run = s_run * alpha[:, None] + local_s * beta[:, None]
        m_run = m_new

    base = ((pid_kb * Q_BLOCKS + pid_qb) * tl.num_programs(1) + pid_bh) * BLOCK_M + offs_m
    tl.store(M_BUF + base, m_run)
    tl.store(Z_BUF + base, z_run)
    tl.store(S_BUF + base[:, None] * 64 + offs_d[None, :], s_run)


@triton.jit
def _phase1_d64_tiled_fp32_nomask_kernel(
    Q,
    K,
    V,
    M_BUF,
    Z_BUF,
    S_BUF,
    Q_START: tl.constexpr,
    N_CTX: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOGICAL_BLOCK_N: tl.constexpr,
    MICRO_BLOCK_N: tl.constexpr,
    LOG2_SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    NEEDS_MASK: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MICRO_BLOCK_N)
    offs_d = tl.arange(0, 64)

    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    # NEEDS_MASK guards partial last q/k blocks (non-divisible seq, e.g. ViT (img/patch)^2+1):
    # out-of-range q rows are discarded by the final kernel, but their loads must stay in-bounds,
    # and out-of-range KEYS must be excluded from the softmax (scores -> -inf) or they corrupt it.
    if NEEDS_MASK:
        q = tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :],
                    mask=(q_idx[:, None] < N_CTX), other=0.0).to(tl.float32) * LOG2_SCALE
    else:
        q = tl.load(Q + pid_bh * N_CTX * 64 + q_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32) * LOG2_SCALE

    m_run = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    z_run = tl.zeros((BLOCK_M,), dtype=tl.float32)
    s_run = tl.zeros((BLOCK_M, 64), dtype=tl.float32)

    logical_start = pid_kb * LOGICAL_BLOCK_N
    for micro_start in tl.range(0, LOGICAL_BLOCK_N, MICRO_BLOCK_N):
        k_idx = logical_start + micro_start + offs_n
        if NEEDS_MASK:
            k_mask = k_idx < N_CTX
            k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :],
                        mask=k_mask[:, None], other=0.0).to(tl.float32)
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
            scores = tl.where(k_mask[None, :], scores, -float("inf"))
        else:
            k = tl.load(K + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32)
            v = tl.load(V + pid_bh * N_CTX * 64 + k_idx[:, None] * 64 + offs_d[None, :]).to(tl.float32)
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        local_m = tl.max(scores, axis=1)
        # A fully-masked micro-block (all keys past N_CTX) has local_m=-inf; exp2(-inf-(-inf))
        # would be NaN, and NaN*beta(0) poisons the running state. Force its p (hence local_z,
        # local_s, beta) to 0 so the block contributes nothing. Partial blocks need no guard
        # (masked lanes get exp2(-inf-finite)=0). No-op for the divisible/unmasked path.
        p = tl.where(local_m[:, None] == float("-inf"), 0.0, tl.exp2(scores - local_m[:, None]))
        local_z = tl.sum(p, axis=1)
        local_s = tl.dot(p, v, input_precision=INPUT_PRECISION)

        m_new = tl.maximum(m_run, local_m)
        alpha = tl.where(z_run > 0.0, tl.exp2(m_run - m_new), 0.0)
        beta = tl.exp2(local_m - m_new)
        z_run = z_run * alpha + local_z * beta
        s_run = s_run * alpha[:, None] + local_s * beta[:, None]
        m_run = m_new

    base = ((pid_kb * Q_BLOCKS + pid_qb) * tl.num_programs(1) + pid_bh) * BLOCK_M + offs_m
    tl.store(M_BUF + base, m_run)
    tl.store(Z_BUF + base, z_run)
    tl.store(S_BUF + base[:, None] * 64 + offs_d[None, :], s_run)


@triton.jit
def _scan_last_to_out_kernel(
    M_BUF,
    Z_BUF,
    S_BUF,
    OUT,
    ROW_M,
    ROW_Z,
    Q_START: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    FINAL_KB: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_STATE: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    q_mask = q_idx < N_CTX
    d_mask = offs_d < D_HEAD

    base = ((FINAL_KB * Q_BLOCKS + pid_qb) * BH + pid_bh) * BLOCK_M + offs_m
    m = tl.load(M_BUF + base, mask=q_mask, other=-float("inf")).to(tl.float32)
    z = tl.load(Z_BUF + base, mask=q_mask, other=0.0).to(tl.float32)
    s = tl.load(
        S_BUF + base[:, None] * D_HEAD + offs_d[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    out = tl.where(z[:, None] > 0.0, s / z[:, None], 0.0)
    out_ptrs = OUT + pid_bh * N_CTX * D_HEAD + q_idx[:, None] * D_HEAD + offs_d[None, :]
    tl.store(out_ptrs, out, mask=q_mask[:, None] & d_mask[None, :])
    if STORE_STATE:
        tl.store(ROW_M + pid_bh * N_CTX + q_idx, m, mask=q_mask)
        tl.store(ROW_Z + pid_bh * N_CTX + q_idx, z, mask=q_mask)


def _can_use_d64_parallel_final(
    *,
    q: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_m: int,
    block_n: int,
    q_chunk_size: int,
    summary_dtype: torch.dtype,
) -> bool:
    # Partial last q/k blocks are now correctly masked inside the phase1 kernels (NEEDS_MASK
    # path, 2026-06-10) — out-of-range keys get scores=-inf so they're excluded from the
    # softmax, and a fully-masked micro-block is guarded against NaN. This lets the fast d64
    # parallel-final (real multi-partition 2-pass) run on ViT's (img/patch)^2+1 NON-divisible
    # sequences instead of the slow generic fallback (verified correct: max_abs ~1e-6 fp32).
    # block_n must stay a multiple of 64 (micro-tile width); q_chunk is already a block_m multiple.
    return (
        bias is None
        and q.dtype in (torch.float16, torch.float32)
        and q.shape[-1] == 64
        and int(block_n) % 64 == 0
        and int(q_chunk_size) % int(block_m) == 0
        and summary_dtype == q.dtype
    )


def _twopass_attention_d64_parallel_final(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    q_chunk_size: int,
    phase1_warps: int,
    phase1_stages: int,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"],
    return_row_state: bool,
    return_contiguous_inputs: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    from .paper_scan_d64_cuda import paper_scan_d64_final_reduce

    batch, heads, seq_len, _ = map(int, q.shape)
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    bh = batch * heads
    k_blocks = triton.cdiv(seq_len, block_n)
    log2_scale = (1.0 / math.sqrt(64)) * 1.4426950408889634
    micro_block_n = 64
    use_tiled_k_block = block_n > 512
    # Partial last q/k block occurs when seq isn't a multiple of block_m/block_n (e.g. ViT
    # (img/patch)^2+1). The phase1 kernels then mask out-of-range keys (scores->-inf) and keep
    # partial q loads in-bounds; for the common divisible case this is compiled away (fast path).
    needs_mask = 1 if (seq_len % block_n != 0 or seq_len % block_m != 0) else 0
    max_q_len = min(q_chunk_size, seq_len)
    max_q_blocks = triton.cdiv(max_q_len, block_m)
    max_summary_elems = k_blocks * max_q_blocks * bh * block_m
    m_buf = torch.empty((max_summary_elems,), device=q.device, dtype=torch.float32)
    z_buf = torch.empty_like(m_buf)
    s_buf = torch.empty((max_summary_elems, 64), device=q.device, dtype=q.dtype)
    row_m = torch.empty((bh, seq_len), device=q.device, dtype=torch.float32) if return_row_state else None
    row_z = torch.empty_like(row_m) if row_m is not None else None
    dummy_state = torch.empty(1, device=q.device, dtype=torch.float32)

    for q_start in range(0, seq_len, q_chunk_size):
        q_len = min(q_chunk_size, seq_len - q_start)
        q_blocks = triton.cdiv(q_len, block_m)
        if use_tiled_k_block and q.dtype == torch.float16:
            _phase1_d64_tiled_fp16_nomask_kernel[(q_blocks, bh, k_blocks)](
                q_c,
                k_c,
                v_c,
                m_buf,
                z_buf,
                s_buf,
                q_start,
                seq_len,
                q_blocks,
                block_m,
                block_n,
                micro_block_n,
                log2_scale,
                input_precision,
                needs_mask,
                num_warps=phase1_warps,
                num_stages=phase1_stages,
            )
        elif use_tiled_k_block:
            _phase1_d64_tiled_fp32_nomask_kernel[(q_blocks, bh, k_blocks)](
                q_c,
                k_c,
                v_c,
                m_buf,
                z_buf,
                s_buf,
                q_start,
                seq_len,
                q_blocks,
                block_m,
                block_n,
                micro_block_n,
                log2_scale,
                input_precision,
                needs_mask,
                num_warps=phase1_warps,
                num_stages=phase1_stages,
            )
        elif q.dtype == torch.float16:
            _phase1_d64_fp16_nomask_kernel[(q_blocks, bh, k_blocks)](
                q_c,
                k_c,
                v_c,
                m_buf,
                z_buf,
                s_buf,
                q_start,
                seq_len,
                q_blocks,
                block_m,
                block_n,
                log2_scale,
                input_precision,
                needs_mask,
                num_warps=phase1_warps,
                num_stages=phase1_stages,
            )
        else:
            _phase1_d64_fp32_nomask_kernel[(q_blocks, bh, k_blocks)](
                q_c,
                k_c,
                v_c,
                m_buf,
                z_buf,
                s_buf,
                q_start,
                seq_len,
                q_blocks,
                block_m,
                block_n,
                log2_scale,
                input_precision,
                needs_mask,
                num_warps=phase1_warps,
                num_stages=phase1_stages,
            )
        paper_scan_d64_final_reduce(
            m_buf,
            z_buf,
            s_buf,
            out,
            row_m if row_m is not None else dummy_state,
            row_z if row_z is not None else dummy_state,
            q_start=q_start,
            seq_len=seq_len,
            k_blocks=k_blocks,
            q_blocks=q_blocks,
            bh=bh,
            block_m=block_m,
            store_state=row_m is not None,
        )

    if return_row_state and return_contiguous_inputs:
        return out, row_m, row_z, q_c, k_c, v_c
    if return_row_state:
        return out, row_m, row_z
    return out


def twopass_attention_hillis_steele_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 128,
    q_chunk_size: int = 1024,
    summary_dtype: Optional[torch.dtype] = None,
    phase1_warps: int = 4,
    scan_warps: int = 4,
    phase1_stages: int = 1,
    scan_stages: int = 1,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    return_row_state: bool = False,
    return_contiguous_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Strict ELSA forward using a log-depth Hillis-Steele prefix scan.

    Phase 1 emits one monoid summary per K block. Phase 2 performs an inclusive
    parallel prefix scan over those summaries. For non-causal attention the last
    prefix is the full-row result; for causal attention future K blocks contain
    empty row summaries due to the phase-1 mask, so the last prefix is still the
    correct row-local causal result.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [B,H,N,D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.device.type != "cuda":
        raise ValueError("twopass_attention_hillis_steele_scan requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are fp16, bf16, and fp32")

    batch, heads, seq_len, head_dim = map(int, q.shape)
    if head_dim > 128:
        raise ValueError("head_dim > 128 is not supported")
    if seq_len <= 0:
        out_empty = torch.empty_like(q)
        state_empty = torch.empty((batch * heads, 0), device=q.device, dtype=torch.float32)
        if return_row_state and return_contiguous_inputs:
            return out_empty, state_empty, state_empty, q.contiguous(), k.contiguous(), v.contiguous()
        if return_row_state:
            return out_empty, state_empty, state_empty
        return out_empty

    if input_precision == "auto":
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
    if input_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("input_precision must be 'auto', 'ieee', 'tf32', or 'tf32x3'")
    if summary_dtype is None:
        summary_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype
    if summary_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("summary_dtype must be fp16, bf16, or fp32")

    block_m = max(16, int(block_m))
    block_n = max(16, int(block_n))
    q_chunk_size = max(block_m, int(q_chunk_size))
    q_chunk_size = max(block_m, (q_chunk_size // block_m) * block_m)
    phase1_warps = _validate_warps("phase1_warps", phase1_warps)
    scan_warps = _validate_warps("scan_warps", scan_warps)
    phase1_stages = max(1, int(phase1_stages))
    scan_stages = max(1, int(scan_stages))

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    row_m = torch.empty((batch * heads, seq_len), device=q.device, dtype=torch.float32) if return_row_state else None
    row_z = torch.empty_like(row_m) if row_m is not None else None
    dummy_state = torch.empty(1, device=q.device, dtype=torch.float32)
    bias_meta = _canonical_bias(bias, batch=batch, heads=heads, seq_len=seq_len, device=q.device)

    block_d = _next_power_of_2(head_dim)
    k_blocks = triton.cdiv(seq_len, block_n)
    bh = batch * heads
    scale = 1.0 / math.sqrt(head_dim)

    max_q_len = min(q_chunk_size, seq_len)
    max_q_blocks = triton.cdiv(max_q_len, block_m)
    max_summary_elems = k_blocks * max_q_blocks * bh * block_m
    m_buf_a = torch.empty((max_summary_elems,), device=q.device, dtype=torch.float32)
    z_buf_a = torch.empty_like(m_buf_a)
    s_buf_a = torch.empty((max_summary_elems, head_dim), device=q.device, dtype=summary_dtype)
    m_buf_b = torch.empty_like(m_buf_a)
    z_buf_b = torch.empty_like(z_buf_a)
    s_buf_b = torch.empty_like(s_buf_a)

    for q_start in range(0, seq_len, q_chunk_size):
        q_len = min(q_chunk_size, seq_len - q_start)
        q_blocks = triton.cdiv(q_len, block_m)
        _phase1_summary_kernel[(q_blocks, bh, k_blocks)](
            q_c,
            k_c,
            v_c,
            bias_meta.tensor,
            m_buf_a,
            z_buf_a,
            s_buf_a,
            q_start,
            heads,
            seq_len,
            head_dim,
            k_blocks,
            q_blocks,
            block_m,
            block_n,
            block_d,
            scale,
            input_precision,
            bias_meta.has_bias,
            is_causal,
            q.dtype != torch.float16 or summary_dtype == torch.float32,
            bias_meta.stride_b,
            bias_meta.stride_h,
            bias_meta.stride_q,
            bias_meta.stride_k,
            num_warps=phase1_warps,
            num_stages=phase1_stages,
        )

        step = 1
        src_m, src_z, src_s = m_buf_a, z_buf_a, s_buf_a
        dst_m, dst_z, dst_s = m_buf_b, z_buf_b, s_buf_b
        while step < k_blocks:
            _summary_scan_step_kernel[(k_blocks, q_blocks, bh)](
                src_m,
                src_z,
                src_s,
                dst_m,
                dst_z,
                dst_s,
                step,
                q_blocks,
                bh,
                head_dim,
                block_m,
                block_d,
                num_warps=scan_warps,
                num_stages=scan_stages,
            )
            src_m, dst_m = dst_m, src_m
            src_z, dst_z = dst_z, src_z
            src_s, dst_s = dst_s, src_s
            step *= 2

        _scan_last_to_out_kernel[(q_blocks, bh)](
            src_m,
            src_z,
            src_s,
            out,
            row_m if row_m is not None else dummy_state,
            row_z if row_z is not None else dummy_state,
            q_start,
            seq_len,
            head_dim,
            k_blocks - 1,
            q_blocks,
            bh,
            block_m,
            block_d,
            row_m is not None,
            num_warps=scan_warps,
            num_stages=scan_stages,
        )

    if return_row_state and return_contiguous_inputs:
        return out, row_m, row_z, q_c, k_c, v_c
    if return_row_state:
        return out, row_m, row_z
    return out


def twopass_attention_blelloch_final_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    block_m: int = 64,
    block_n: int = 256,
    q_chunk_size: int = 1024,
    summary_dtype: Optional[torch.dtype] = None,
    phase1_warps: int = 4,
    scan_warps: int = 4,
    phase1_stages: int = 1,
    scan_stages: int = 1,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    return_row_state: bool = False,
    return_contiguous_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Strict non-causal ELSA using the Blelloch upsweep final prefix.

    Non-causal attention needs only the final prefix over K-block summaries.
    This is the root produced by the Blelloch upsweep stage of the paper scan,
    without a serial fold or multi-block phase-1 coalescing.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [B,H,N,D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.device.type != "cuda":
        raise ValueError("twopass_attention_blelloch_final_scan requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are fp16, bf16, and fp32")

    batch, heads, seq_len, head_dim = map(int, q.shape)
    if head_dim > 128:
        raise ValueError("head_dim > 128 is not supported")
    if seq_len <= 0:
        out_empty = torch.empty_like(q)
        state_empty = torch.empty((batch * heads, 0), device=q.device, dtype=torch.float32)
        if return_row_state and return_contiguous_inputs:
            return out_empty, state_empty, state_empty, q.contiguous(), k.contiguous(), v.contiguous()
        if return_row_state:
            return out_empty, state_empty, state_empty
        return out_empty

    if input_precision == "auto":
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
    if input_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("input_precision must be 'auto', 'ieee', 'tf32', or 'tf32x3'")
    if summary_dtype is None:
        summary_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype
    if summary_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("summary_dtype must be fp16, bf16, or fp32")

    block_m = max(16, int(block_m))
    block_n = max(16, int(block_n))
    q_chunk_size = max(block_m, int(q_chunk_size))
    q_chunk_size = max(block_m, (q_chunk_size // block_m) * block_m)
    phase1_warps = _validate_warps("phase1_warps", phase1_warps)
    scan_warps = _validate_warps("scan_warps", scan_warps)
    phase1_stages = max(1, int(phase1_stages))
    scan_stages = max(1, int(scan_stages))

    if _can_use_d64_parallel_final(
        q=q,
        bias=bias,
        block_m=block_m,
        block_n=block_n,
        q_chunk_size=q_chunk_size,
        summary_dtype=summary_dtype,
    ):
        return _twopass_attention_d64_parallel_final(
            q,
            k,
            v,
            block_m=block_m,
            block_n=block_n,
            q_chunk_size=q_chunk_size,
            phase1_warps=phase1_warps,
            phase1_stages=phase1_stages,
            input_precision=input_precision,
            return_row_state=return_row_state,
            return_contiguous_inputs=return_contiguous_inputs,
        )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    row_m = torch.empty((batch * heads, seq_len), device=q.device, dtype=torch.float32) if return_row_state else None
    row_z = torch.empty_like(row_m) if row_m is not None else None
    dummy_state = torch.empty(1, device=q.device, dtype=torch.float32)
    bias_meta = _canonical_bias(bias, batch=batch, heads=heads, seq_len=seq_len, device=q.device)

    block_d = _next_power_of_2(head_dim)
    k_blocks = triton.cdiv(seq_len, block_n)
    bh = batch * heads
    scale = 1.0 / math.sqrt(head_dim)

    max_q_len = min(q_chunk_size, seq_len)
    max_q_blocks = triton.cdiv(max_q_len, block_m)
    max_summary_elems = k_blocks * max_q_blocks * bh * block_m
    m_buf_a = torch.empty((max_summary_elems,), device=q.device, dtype=torch.float32)
    z_buf_a = torch.empty_like(m_buf_a)
    s_buf_a = torch.empty((max_summary_elems, head_dim), device=q.device, dtype=summary_dtype)
    # Op-fusion: the Blelloch upsweep (`while in_blocks > 1`) only runs when k_blocks>1.
    # For short sequences that fit one K partition (k_blocks==1, e.g. ViT seq=197) the
    # double-buffer is never touched — skip those 3 allocations (they were a measurable
    # per-layer CPU overhead in the full-model drop-in).
    if k_blocks > 1:
        m_buf_b = torch.empty_like(m_buf_a)
        z_buf_b = torch.empty_like(z_buf_a)
        s_buf_b = torch.empty_like(s_buf_a)
    else:
        m_buf_b = z_buf_b = s_buf_b = None

    for q_start in range(0, seq_len, q_chunk_size):
        q_len = min(q_chunk_size, seq_len - q_start)
        q_blocks = triton.cdiv(q_len, block_m)
        _phase1_summary_kernel[(q_blocks, bh, k_blocks)](
            q_c,
            k_c,
            v_c,
            bias_meta.tensor,
            m_buf_a,
            z_buf_a,
            s_buf_a,
            q_start,
            heads,
            seq_len,
            head_dim,
            k_blocks,
            q_blocks,
            block_m,
            block_n,
            block_d,
            scale,
            input_precision,
            bias_meta.has_bias,
            False,
            q.dtype != torch.float16 or summary_dtype == torch.float32,
            bias_meta.stride_b,
            bias_meta.stride_h,
            bias_meta.stride_q,
            bias_meta.stride_k,
            num_warps=phase1_warps,
            num_stages=phase1_stages,
        )

        in_blocks = k_blocks
        src_m, src_z, src_s = m_buf_a, z_buf_a, s_buf_a
        dst_m, dst_z, dst_s = m_buf_b, z_buf_b, s_buf_b
        while in_blocks > 1:
            out_blocks = triton.cdiv(in_blocks, 2)
            _summary_blelloch_upsweep_kernel[(out_blocks, q_blocks, bh)](
                src_m,
                src_z,
                src_s,
                dst_m,
                dst_z,
                dst_s,
                in_blocks,
                q_blocks,
                bh,
                head_dim,
                block_m,
                block_d,
                num_warps=scan_warps,
                num_stages=scan_stages,
            )
            src_m, dst_m = dst_m, src_m
            src_z, dst_z = dst_z, src_z
            src_s, dst_s = dst_s, src_s
            in_blocks = out_blocks

        _scan_last_to_out_kernel[(q_blocks, bh)](
            src_m,
            src_z,
            src_s,
            out,
            row_m if row_m is not None else dummy_state,
            row_z if row_z is not None else dummy_state,
            q_start,
            seq_len,
            head_dim,
            0,
            q_blocks,
            bh,
            block_m,
            block_d,
            row_m is not None,
            num_warps=scan_warps,
            num_stages=scan_stages,
        )

    if return_row_state and return_contiguous_inputs:
        return out, row_m, row_z, q_c, k_c, v_c
    if return_row_state:
        return out, row_m, row_z
    return out


def twopass_attention_paper_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 128,
    q_chunk_size: int = 1024,
    summary_dtype: Optional[torch.dtype] = None,
    phase1_warps: int = 4,
    scan_warps: int = 4,
    phase1_stages: int = 1,
    scan_stages: int = 1,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    return_row_state: bool = False,
    return_contiguous_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if is_causal:
        return twopass_attention_hillis_steele_scan(
            q,
            k,
            v,
            bias=bias,
            is_causal=True,
            block_m=block_m,
            block_n=block_n,
            q_chunk_size=q_chunk_size,
            summary_dtype=summary_dtype,
            phase1_warps=phase1_warps,
            scan_warps=scan_warps,
            phase1_stages=phase1_stages,
            scan_stages=scan_stages,
            input_precision=input_precision,
            return_row_state=return_row_state,
            return_contiguous_inputs=return_contiguous_inputs,
        )
    return twopass_attention_blelloch_final_scan(
        q,
        k,
        v,
        bias=bias,
        block_m=block_m,
        block_n=block_n,
        q_chunk_size=q_chunk_size,
        summary_dtype=summary_dtype,
        phase1_warps=phase1_warps,
        scan_warps=scan_warps,
        phase1_stages=phase1_stages,
        scan_stages=scan_stages,
        input_precision=input_precision,
        return_row_state=return_row_state,
        return_contiguous_inputs=return_contiguous_inputs,
    )
