from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import triton
import triton.language as tl

from .attention_parallel_scan import twopass_attention_paper_scan


@dataclass(frozen=True)
class _BiasMeta:
    tensor: torch.Tensor
    stride_b: int
    stride_h: int
    stride_q: int
    stride_k: int
    has_bias: bool


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _validate_warps(name: str, value: int) -> int:
    value = int(value)
    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1, 2, 4, or 8")
    return value


def _canonical_bias(
    bias: Optional[torch.Tensor],
    *,
    batch: int,
    heads: int,
    seq_len: int,
    device: torch.device,
) -> _BiasMeta:
    if bias is None:
        dummy = torch.empty(1, device=device, dtype=torch.float32)
        return _BiasMeta(dummy, 0, 0, 0, 0, False)

    if bias.device != device:
        raise ValueError("bias must live on the same device as q/k/v")
    if bias.shape[-2:] != (seq_len, seq_len):
        raise ValueError(f"bias last dims must be {(seq_len, seq_len)}, got {tuple(bias.shape)}")

    if bias.dim() == 2:
        view = bias.contiguous().view(1, 1, seq_len, seq_len)
    elif bias.dim() == 3:
        if bias.shape[0] not in (1, heads):
            raise ValueError("3D bias must have shape [1,N,N] or [H,N,N]")
        view = bias.contiguous().view(1, bias.shape[0], seq_len, seq_len)
    elif bias.dim() == 4:
        if bias.shape[0] not in (1, batch) or bias.shape[1] not in (1, heads):
            raise ValueError("4D bias must broadcast as [B,H,N,N]")
        view = bias.contiguous()
    else:
        raise ValueError("bias must have shape [N,N], [H,N,N], or [B,H,N,N]")

    stride_b = 0 if view.shape[0] == 1 else view.stride(0)
    stride_h = 0 if view.shape[1] == 1 else view.stride(1)
    return _BiasMeta(
        view,
        int(stride_b),
        int(stride_h),
        int(view.stride(2)),
        int(view.stride(3)),
        True,
    )


def twopass_attention_with_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    q_chunk_size: Optional[int] = None,
    summary_dtype: Optional[torch.dtype] = None,
    phase1_warps: Optional[int] = None,
    phase2_warps: Optional[int] = None,
    phase1_stages: Optional[int] = None,
    phase2_stages: Optional[int] = None,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    algorithm: Literal["auto", "paper_scan"] = "auto",
    return_contiguous_inputs: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Training forward through the same strict paper-scan path used by fwd."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [B,H,N,D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes")
    if q.device.type != "cuda":
        raise ValueError("twopass_attention_with_state currently requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are fp16, bf16, and fp32")
    batch, heads, seq_len, head_dim = map(int, q.shape)
    if seq_len <= 0:
        empty_state = torch.empty((batch * heads, 0), device=q.device, dtype=torch.float32)
        out_empty = torch.empty_like(q)
        if return_contiguous_inputs:
            return out_empty, empty_state, empty_state, q.contiguous(), k.contiguous(), v.contiguous()
        return out_empty, empty_state, empty_state
    if head_dim > 128:
        raise ValueError("head_dim > 128 is not supported by the current Triton kernels")
    if algorithm not in ("auto", "paper_scan"):
        raise ValueError("only algorithm='paper_scan' is available in this strict release")
    if input_precision == "auto":
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
    if input_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("input_precision must be 'auto', 'ieee', 'tf32', or 'tf32x3'")
    if summary_dtype is None:
        if q.dtype == torch.float16:
            summary_dtype = torch.float16
        elif q.dtype == torch.bfloat16:
            summary_dtype = torch.bfloat16
        else:
            summary_dtype = torch.float32
    if summary_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("summary_dtype must be fp16, bf16, or fp32")

    if q.dtype == torch.float32:
        block_m = 32 if block_m is None else block_m
        # Mirror the inference _paper_scan_defaults adaptive tiled routing so the TRAIN
        # forward also uses the fast micro-tiled register fold (block_n>512 -> tiled) on
        # ViT's odd seqs, instead of slow non-tiled block_n=128. Partial blocks are masked.
        if block_n is None and head_dim <= 64:
            if seq_len <= 512:
                block_n = 256
            elif seq_len < 2048:
                block_n = ((seq_len + 63) // 64) * 64
            else:
                block_n = 2048
        elif block_n is None:
            block_n = 64
        if q_chunk_size is None:
            q_chunk_size = 512 if seq_len >= 4096 else 1024
        phase1_warps = 4 if phase1_warps is None else phase1_warps
        phase2_warps = 4 if phase2_warps is None else phase2_warps
        phase1_stages = 2 if phase1_stages is None else phase1_stages
        phase2_stages = 1 if phase2_stages is None else phase2_stages
    else:
        block_m = (64 if head_dim <= 64 else 32) if block_m is None else block_m
        block_n = (256 if head_dim <= 64 else 64) if block_n is None else block_n
        if q_chunk_size is None:
            q_chunk_size = 1024 if seq_len >= 4096 else min(seq_len, 1024)
        phase1_warps = 4 if phase1_warps is None else phase1_warps
        phase2_warps = 4 if phase2_warps is None else phase2_warps
        phase1_stages = 2 if phase1_stages is None else phase1_stages
        phase2_stages = 1 if phase2_stages is None else phase2_stages

    block_m = max(16, int(block_m))
    block_n = max(16, int(block_n))
    q_chunk_size = max(block_m, int(q_chunk_size))
    q_chunk_size = max(block_m, (q_chunk_size // block_m) * block_m)
    phase1_warps = _validate_warps("phase1_warps", phase1_warps)
    phase2_warps = _validate_warps("phase2_warps", phase2_warps)
    phase1_stages = max(1, int(phase1_stages))
    phase2_stages = max(1, int(phase2_stages))
    return twopass_attention_paper_scan(
        q,
        k,
        v,
        bias=bias,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        q_chunk_size=q_chunk_size,
        summary_dtype=summary_dtype,
        phase1_warps=phase1_warps,
        scan_warps=phase2_warps,
        phase1_stages=phase1_stages,
        scan_stages=phase2_stages,
        input_precision=input_precision,
        return_row_state=True,
        return_contiguous_inputs=return_contiguous_inputs,
    )


@triton.jit
def _bwd_preprocess_kernel(
    OUT,
    DOUT,
    M_BUF,
    Z_BUF,
    D_BUF,
    LSE_BUF,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N_CTX
    d_mask = offs_d < D_HEAD
    base = pid_bh * N_CTX * D_HEAD

    out = tl.load(
        OUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dout = tl.load(
        DOUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    d_row = tl.sum(out * dout, axis=1)
    m_row = tl.load(M_BUF + pid_bh * N_CTX + offs_m, mask=m_mask, other=-float("inf")).to(tl.float32)
    z_row = tl.load(Z_BUF + pid_bh * N_CTX + offs_m, mask=m_mask, other=1.0).to(tl.float32)
    tl.store(D_BUF + pid_bh * N_CTX + offs_m, d_row, mask=m_mask)
    tl.store(LSE_BUF + pid_bh * N_CTX + offs_m, m_row + tl.log2(tl.maximum(z_row, 1.0e-30)), mask=m_mask)


@triton.jit
def _row_stats_kernel(
    Q,
    K,
    BIAS,
    M_BUF,
    Z_BUF,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_STRIDE_B: tl.constexpr,
    BIAS_STRIDE_H: tl.constexpr,
    BIAS_STRIDE_Q: tl.constexpr,
    BIAS_STRIDE_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N_CTX
    d_mask = offs_d < D_HEAD
    base = pid_bh * N_CTX * D_HEAD

    q = tl.load(
        Q + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_run = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    z_run = tl.zeros((BLOCK_M,), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for k_start in tl.range(0, N_CTX, BLOCK_N):
        offs_k = k_start + offs_n
        k_mask = offs_k < N_CTX
        k = tl.load(
            K + base + offs_k[:, None] * D_HEAD + offs_d[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION).to(tl.float32) * SCALE
        valid = m_mask[:, None] & k_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_k[None, :])

        if HAS_BIAS:
            b_idx = pid_bh // H
            h_idx = pid_bh - b_idx * H
            bias_ptrs = (
                BIAS
                + b_idx * BIAS_STRIDE_B
                + h_idx * BIAS_STRIDE_H
                + offs_m[:, None] * BIAS_STRIDE_Q
                + offs_k[None, :] * BIAS_STRIDE_K
            )
            bias = tl.load(bias_ptrs, mask=valid, other=0.0).to(tl.float32)
            scores = scores + bias

        scores = tl.where(valid, scores * log2e, -float("inf"))
        local_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m_run, local_m)
        alpha = tl.where(z_run > 0.0, tl.exp2(m_run - m_new), 0.0)
        beta = tl.where(local_m > -float("inf"), tl.exp2(local_m - m_new), 0.0)
        local_z = tl.sum(tl.where(valid, tl.exp2(scores - local_m[:, None]), 0.0), axis=1)
        z_run = z_run * alpha + local_z * beta
        m_run = tl.where(z_run > 0.0, m_new, -float("inf"))

    idx = pid_bh * N_CTX + offs_m
    tl.store(M_BUF + idx, m_run, mask=m_mask)
    tl.store(Z_BUF + idx, z_run, mask=m_mask)


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    BIAS,
    DOUT,
    LSE_BUF,
    D_BUF,
    DQ,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_STRIDE_B: tl.constexpr,
    BIAS_STRIDE_H: tl.constexpr,
    BIAS_STRIDE_Q: tl.constexpr,
    BIAS_STRIDE_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N_CTX
    d_mask = offs_d < D_HEAD
    base = pid_bh * N_CTX * D_HEAD
    row_base = pid_bh * N_CTX

    q = tl.load(
        Q + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    dout = tl.load(
        DOUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    lse_row = tl.load(LSE_BUF + row_base + offs_m, mask=m_mask, other=-float("inf")).to(tl.float32)
    d_row = tl.load(D_BUF + row_base + offs_m, mask=m_mask, other=0.0).to(tl.float32)

    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    log2_scale: tl.constexpr = SCALE * 1.4426950408889634

    for k_start in tl.range(0, N_CTX, BLOCK_N):
        active_block = True
        if IS_CAUSAL:
            active_block = k_start < (pid_m + 1) * BLOCK_M
        if active_block:
            offs_k = k_start + offs_n
            k_mask = offs_k < N_CTX
            k = tl.load(
                K + base + offs_k[:, None] * D_HEAD + offs_d[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                V + base + offs_k[:, None] * D_HEAD + offs_d[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION).to(tl.float32)
            valid = m_mask[:, None] & k_mask[None, :]
            if IS_CAUSAL:
                full_past_block = k_start + BLOCK_N <= pid_m * BLOCK_M
                if not full_past_block:
                    valid = valid & (offs_m[:, None] >= offs_k[None, :])

            if HAS_BIAS:
                b_idx = pid_bh // H
                h_idx = pid_bh - b_idx * H
                bias_ptrs = (
                    BIAS
                    + b_idx * BIAS_STRIDE_B
                    + h_idx * BIAS_STRIDE_H
                    + offs_m[:, None] * BIAS_STRIDE_Q
                    + offs_k[None, :] * BIAS_STRIDE_K
                )
                bias = tl.load(bias_ptrs, mask=valid, other=0.0).to(tl.float32)
                scores = (scores * SCALE + bias) * log2e
            else:
                scores = scores * log2_scale

            scores = tl.where(valid, scores, -float("inf"))
            p = tl.where(valid, tl.exp2(scores - lse_row[:, None]), 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=INPUT_PRECISION).to(tl.float32)
            ds = (p * (dp - d_row[:, None])).to(k.dtype)
            dq_acc += tl.dot(ds, k, input_precision=INPUT_PRECISION)

    dq_acc *= SCALE
    tl.store(
        DQ + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
        dq_acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    BIAS,
    DOUT,
    LSE_BUF,
    D_BUF,
    DK,
    DV,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_STRIDE_B: tl.constexpr,
    BIAS_STRIDE_H: tl.constexpr,
    BIAS_STRIDE_Q: tl.constexpr,
    BIAS_STRIDE_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    n_mask = offs_n < N_CTX
    d_mask = offs_d < D_HEAD
    base = pid_bh * N_CTX * D_HEAD
    row_base = pid_bh * N_CTX

    k = tl.load(
        K + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        V + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    log2_scale: tl.constexpr = SCALE * 1.4426950408889634

    for q_start in tl.range(0, N_CTX, BLOCK_M):
        active_block = True
        if IS_CAUSAL:
            active_block = q_start + BLOCK_M > pid_n * BLOCK_N
        if active_block:
            offs_m = q_start + offs_m_base
            m_mask = offs_m < N_CTX
            q = tl.load(
                Q + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            dout = tl.load(
                DOUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            lse_row = tl.load(LSE_BUF + row_base + offs_m, mask=m_mask, other=-float("inf")).to(tl.float32)
            d_row = tl.load(D_BUF + row_base + offs_m, mask=m_mask, other=0.0).to(tl.float32)

            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION).to(tl.float32)
            valid = m_mask[:, None] & n_mask[None, :]
            if IS_CAUSAL:
                full_past_block = q_start >= (pid_n + 1) * BLOCK_N
                if not full_past_block:
                    valid = valid & (offs_m[:, None] >= offs_n[None, :])

            if HAS_BIAS:
                b_idx = pid_bh // H
                h_idx = pid_bh - b_idx * H
                bias_ptrs = (
                    BIAS
                    + b_idx * BIAS_STRIDE_B
                    + h_idx * BIAS_STRIDE_H
                    + offs_m[:, None] * BIAS_STRIDE_Q
                    + offs_n[None, :] * BIAS_STRIDE_K
                )
                bias = tl.load(bias_ptrs, mask=valid, other=0.0).to(tl.float32)
                scores = (scores * SCALE + bias) * log2e
            else:
                scores = scores * log2_scale

            scores = tl.where(valid, scores, -float("inf"))
            p = tl.where(valid, tl.exp2(scores - lse_row[:, None]), 0.0)
            dv_acc += tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=INPUT_PRECISION).to(tl.float32)
            dp = tl.dot(dout, tl.trans(v), input_precision=INPUT_PRECISION).to(tl.float32)
            ds = (p * (dp - d_row[:, None])).to(q.dtype)
            dk_acc += tl.dot(tl.trans(ds), q, input_precision=INPUT_PRECISION).to(tl.float32)

    dk_acc *= SCALE
    tl.store(
        DK + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
        dk_acc,
        mask=n_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        DV + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
        dv_acc,
        mask=n_mask[:, None] & d_mask[None, :],
    )

@triton.jit
def _dbias_dense_kernel(
    Q,
    K,
    V,
    OUT,
    DOUT,
    M_BUF,
    Z_BUF,
    BIAS,
    DBIAS,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BIAS_BATCH_ONE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)
    q_blocks = tl.cdiv(N_CTX, BLOCK_M)
    pid_m = pid_t % q_blocks
    pid_n = pid_t // q_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < N_CTX
    n_mask = offs_n < N_CTX
    d_mask = offs_d < D_HEAD

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_start = 0
    b_stop = B
    if not BIAS_BATCH_ONE:
        b_start = pid_b
        b_stop = pid_b + 1

    for b_idx in tl.range(b_start, b_stop):
        bh = b_idx * H + pid_h
        base = bh * N_CTX * D_HEAD
        q = tl.load(
            Q + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        k = tl.load(
            K + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            V + base + offs_n[:, None] * D_HEAD + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        out = tl.load(
            OUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        dout = tl.load(
            DOUT + base + offs_m[:, None] * D_HEAD + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        m_row = tl.load(M_BUF + bh * N_CTX + offs_m, mask=m_mask, other=-float("inf")).to(tl.float32)
        z_row = tl.load(Z_BUF + bh * N_CTX + offs_m, mask=m_mask, other=1.0).to(tl.float32)
        inv_z = 1.0 / tl.maximum(z_row, 1.0e-30)
        bias_b = 0
        if not BIAS_BATCH_ONE:
            bias_b = b_idx
        bias_ptrs = BIAS + ((bias_b * H + pid_h) * N_CTX + offs_m[:, None]) * N_CTX + offs_n[None, :]
        bias = tl.load(bias_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION).to(tl.float32) * SCALE + bias
        valid = m_mask[:, None] & n_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        scores = tl.where(valid, scores * 1.4426950408889634, -float("inf"))
        p = tl.where(valid, tl.exp2(scores - m_row[:, None]) * inv_z[:, None], 0.0)
        dp = tl.dot(dout, tl.trans(v), input_precision=INPUT_PRECISION).to(tl.float32)
        d_row = tl.sum(out * dout.to(tl.float32), axis=1)
        acc += p * (dp - d_row[:, None])

    db_b = 0
    if not BIAS_BATCH_ONE:
        db_b = pid_b
    dbias_ptrs = DBIAS + ((db_b * H + pid_h) * N_CTX + offs_m[:, None]) * N_CTX + offs_n[None, :]
    tl.store(dbias_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _auto_tile_bwd(head_dim: int, dtype: torch.dtype) -> tuple[int, int, int, int]:
    if dtype == torch.float32:
        if head_dim <= 64:
            # block_n=32 (was 64) halves the _bwd_dkdv dk/dv register accumulator
            # ([BLOCK_N,64]x2) and KILLS a register-spill cliff: clean-GPU sweep on
            # ViT (8,3,2305,64) fwd+bwd vs SDPA went 4.71x -> 1.73x (2.7x faster),
            # gradients still exact (max err 1.6e-7). dq is unaffected (its accumulator
            # is [BLOCK_M,64], independent of block_n). 2026-06-10.
            return 32, 32, 4, 1
        return 32, 32, 4, 1
    if head_dim >= 128:
        return 64, 32, 4, 1
    if head_dim <= 64:
        # block_n 64->32 + stages 2->1: same _bwd_dkdv register-spill kill as fp32 (the
        # dk/dv accumulators are fp32 [BLOCK_N,64]x2 = 32KB regardless of input dtype).
        # clean-GPU sweep vs FlashAttention bwd @ (1,8,8192,64): 2.05x -> 1.58x, grads exact
        # (max err 1.5e-5). 2026-06-10.
        return 64, 32, 4, 1
    return 64, 32, 4, 2


def _auto_tile_bwd_pair(
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    common = _auto_tile_bwd(head_dim, dtype)
    if dtype == torch.float32:
        return common, common
    if head_dim >= 128 and is_causal:
        return (64, 32, 4, 1), (32, 64, 4, 3)
    return common, common


def twopass_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    row_m: Optional[torch.Tensor] = None,
    row_z: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute-based tiled backward for two-pass ELSA attention.

    This computes gradients for q/k/v without materializing the full attention
    matrix. Dense bias participates in the softmax if supplied, but dbias is not
    produced by this low-level helper.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [B,H,N,D]")
    if q.shape != k.shape or q.shape != v.shape or q.shape != out.shape or q.shape != grad_out.shape:
        raise ValueError("q, k, v, out, and grad_out must have identical shapes")
    if q.device.type != "cuda":
        raise ValueError("twopass_attention_backward currently requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are fp16, bf16, and fp32")
    batch, heads, seq_len, head_dim = map(int, q.shape)
    if head_dim > 128:
        raise ValueError("head_dim > 128 is not supported by the current Triton kernels")
    if seq_len <= 0:
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    if input_precision == "auto":
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
    if input_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("input_precision must be 'auto', 'ieee', 'tf32', or 'tf32x3'")

    auto_dq, auto_dkv = _auto_tile_bwd_pair(head_dim, q.dtype, is_causal)
    if block_m is None and block_n is None and num_warps is None and num_stages is None:
        dq_block_m, dq_block_n, dq_warps, dq_stages = auto_dq
        dkv_block_m, dkv_block_n, dkv_warps, dkv_stages = auto_dkv
    else:
        auto_m, auto_n, auto_w, auto_s = _auto_tile_bwd(head_dim, q.dtype)
        shared_m = auto_m if block_m is None else max(16, int(block_m))
        shared_n = auto_n if block_n is None else max(16, int(block_n))
        shared_w = auto_w if num_warps is None else _validate_warps("num_warps", num_warps)
        shared_s = auto_s if num_stages is None else max(1, int(num_stages))
        dq_block_m = dkv_block_m = shared_m
        dq_block_n = dkv_block_n = shared_n
        dq_warps = dkv_warps = shared_w
        dq_stages = dkv_stages = shared_s

    block_d = _next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    bh = batch * heads

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out_c = out.contiguous()
    grad_out_c = grad_out.contiguous()
    bias_meta = _canonical_bias(bias, batch=batch, heads=heads, seq_len=seq_len, device=q.device)

    if row_m is None or row_z is None:
        row_m = torch.empty((bh, seq_len), dtype=torch.float32, device=q.device)
        row_z = torch.empty_like(row_m)
        needs_row_stats = True
    else:
        if row_m.shape != (bh, seq_len) or row_z.shape != (bh, seq_len):
            raise ValueError("row_m and row_z must have shape [B*H,N]")
        row_m = row_m.contiguous()
        row_z = row_z.contiguous()
        needs_row_stats = False
    preprocess_block_m = max(dq_block_m, dkv_block_m)
    grid_pre = (bh, triton.cdiv(seq_len, preprocess_block_m))
    if needs_row_stats:
        _row_stats_kernel[grid_pre](
            q_c,
            k_c,
            bias_meta.tensor,
            row_m,
            row_z,
            heads,
            seq_len,
            head_dim,
            preprocess_block_m,
            dq_block_n,
            block_d,
            scale,
            input_precision,
            bias_meta.has_bias,
            is_causal,
            bias_meta.stride_b,
            bias_meta.stride_h,
            bias_meta.stride_q,
            bias_meta.stride_k,
            num_warps=max(dq_warps, dkv_warps),
            num_stages=1,
        )
    row_d = torch.empty_like(row_m)
    row_lse = torch.empty_like(row_m)
    _bwd_preprocess_kernel[grid_pre](
        out_c,
        grad_out_c,
        row_m,
        row_z,
        row_d,
        row_lse,
        seq_len,
        head_dim,
        preprocess_block_m,
        block_d,
        num_warps=max(dq_warps, dkv_warps),
        num_stages=1,
    )
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)
    grid_dq = (bh, triton.cdiv(seq_len, dq_block_m))
    grid_dkv = (bh, triton.cdiv(seq_len, dkv_block_n))
    dq = torch.empty_like(q_c)
    _bwd_dq_kernel[grid_dq](
        q_c,
        k_c,
        v_c,
        bias_meta.tensor,
        grad_out_c,
        row_lse,
        row_d,
        dq,
        heads,
        seq_len,
        head_dim,
        dq_block_m,
        dq_block_n,
        block_d,
        scale,
        input_precision,
        bias_meta.has_bias,
        is_causal,
        bias_meta.stride_b,
        bias_meta.stride_h,
        bias_meta.stride_q,
        bias_meta.stride_k,
        num_warps=dq_warps,
        num_stages=dq_stages,
    )
    _bwd_dkdv_kernel[grid_dkv](
        q_c,
        k_c,
        v_c,
        bias_meta.tensor,
        grad_out_c,
        row_lse,
        row_d,
        dk,
        dv,
        heads,
        seq_len,
        head_dim,
        dkv_block_m,
        dkv_block_n,
        block_d,
        scale,
        input_precision,
        bias_meta.has_bias,
        is_causal,
        bias_meta.stride_b,
        bias_meta.stride_h,
        bias_meta.stride_q,
        bias_meta.stride_k,
        num_warps=dkv_warps,
        num_stages=dkv_stages,
    )
    return dq, dk, dv


class _TwopassAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor],
        is_causal: bool,
        fwd_kwargs: dict,
        bwd_kwargs: dict,
    ) -> torch.Tensor:
        out, row_m, row_z, q_saved, k_saved, v_saved = twopass_attention_with_state(
            q,
            k,
            v,
            bias=bias,
            is_causal=is_causal,
            return_contiguous_inputs=True,
            **fwd_kwargs,
        )
        if bias is None:
            saved_bias = torch.empty(1, device=q.device, dtype=torch.float32)
            has_bias = False
        else:
            saved_bias = bias
            has_bias = True
        ctx.save_for_backward(q_saved, k_saved, v_saved, out, saved_bias, row_m, row_z)
        ctx.has_bias = has_bias
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        ctx.is_causal = bool(is_causal)
        ctx.bwd_kwargs = dict(bwd_kwargs)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, saved_bias, row_m, row_z = ctx.saved_tensors
        bias = saved_bias if ctx.has_bias else None
        dq, dk, dv = twopass_attention_backward(
            q,
            k,
            v,
            out,
            grad_out,
            bias=bias,
            is_causal=ctx.is_causal,
            row_m=row_m,
            row_z=row_z,
            **ctx.bwd_kwargs,
        )
        dbias = None
        if ctx.bias_requires_grad:
            dbias = _dense_bias_backward(
                q,
                k,
                v,
                out,
                grad_out,
                bias=bias,
                is_causal=ctx.is_causal,
                row_m=row_m,
                row_z=row_z,
            )
        return dq, dk, dv, dbias, None, None, None


def _dense_bias_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    bias: torch.Tensor,
    is_causal: bool,
    row_m: Optional[torch.Tensor] = None,
    row_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute dbias for dense-bias training uses."""
    seq_len = int(q.shape[-2])
    score_elems = q.shape[0] * q.shape[1] * seq_len * seq_len
    if score_elems > 64 * 1024 * 1024 and (bias.dim() != 4 or row_m is None or row_z is None):
        raise NotImplementedError("dbias materialization limit exceeded; use a dedicated dbias kernel")

    if (
        bias.dim() == 4
        and row_m is not None
        and row_z is not None
        and q.device.type == "cuda"
        and q.shape[-1] <= 128
    ):
        bias_c = bias.contiguous()
        row_m_c = row_m.contiguous()
        row_z_c = row_z.contiguous()
        batch = int(q.shape[0])
        heads = int(q.shape[1])
        head_dim = int(q.shape[-1])
        block_m = 16 if seq_len <= 128 else 32
        block_n = 16 if seq_len <= 128 else 32
        block_d = _next_power_of_2(head_dim)
        bias_batch_one = bias_c.shape[0] == 1
        dbias_acc = torch.empty_like(bias_c, dtype=torch.float32)
        grid_b = 1 if bias_batch_one else batch
        grid = (grid_b, heads, triton.cdiv(seq_len, block_m) * triton.cdiv(seq_len, block_n))
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
        _dbias_dense_kernel[grid](
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out.contiguous(),
            grad_out.contiguous(),
            row_m_c,
            row_z_c,
            bias_c,
            dbias_acc,
            batch,
            heads,
            seq_len,
            head_dim,
            block_m,
            block_n,
            block_d,
            1.0 / math.sqrt(head_dim),
            input_precision,
            bias_batch_one,
            is_causal,
            num_warps=4,
            num_stages=1,
        )
        return dbias_acc.to(dtype=bias.dtype)

    qf = q.float()
    kf = k.float()
    vf = v.float()
    do = grad_out.float()
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    scores = scores + bias.float()
    if is_causal:
        causal_mask = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    dp = torch.matmul(do, vf.transpose(-2, -1))
    d_row = (out.float() * do).sum(dim=-1, keepdim=True)
    ds = p * (dp - d_row)

    if bias.dim() == 2:
        return ds.sum(dim=(0, 1)).to(bias.dtype)
    if bias.dim() == 3:
        if bias.shape[0] == 1:
            return ds.sum(dim=(0, 1), keepdim=False).unsqueeze(0).to(bias.dtype)
        return ds.sum(dim=0).to(bias.dtype)
    if bias.dim() == 4:
        grad = ds
        if bias.shape[0] == 1:
            grad = grad.sum(dim=0, keepdim=True)
        if bias.shape[1] == 1:
            grad = grad.sum(dim=1, keepdim=True)
        return grad.to(bias.dtype)
    raise ValueError("bias must have shape [N,N], [H,N,N], or [B,H,N,N]")


def twopass_attention_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    bwd_block_m: Optional[int] = None,
    bwd_block_n: Optional[int] = None,
    bwd_num_warps: Optional[int] = None,
    bwd_num_stages: Optional[int] = None,
    bwd_input_precision: Literal["auto", "ieee", "tf32", "tf32x3"] = "auto",
    **fwd_kwargs,
) -> torch.Tensor:
    """Autograd wrapper using frozen two-pass forward plus tiled bwd."""
    if bwd_input_precision == "auto":
        fwd_input_precision = fwd_kwargs.get("input_precision")
        if fwd_input_precision in ("ieee", "tf32", "tf32x3"):
            bwd_input_precision = fwd_input_precision
    bwd_kwargs = {
        "block_m": bwd_block_m,
        "block_n": bwd_block_n,
        "num_warps": bwd_num_warps,
        "num_stages": bwd_num_stages,
        "input_precision": bwd_input_precision,
    }
    bwd_kwargs = {key: value for key, value in bwd_kwargs.items() if value is not None}
    return _TwopassAttentionAutograd.apply(q, k, v, bias, is_causal, dict(fwd_kwargs), bwd_kwargs)
