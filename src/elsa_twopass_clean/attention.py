from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class _BiasMeta:
    tensor: torch.Tensor
    stride_b: int
    stride_h: int
    stride_q: int
    stride_k: int
    has_bias: bool


@triton.jit
def _phase1_summary_kernel(
    Q,
    K,
    V,
    BIAS,
    M_BUF,
    Z_BUF,
    S_BUF,
    Q_START: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SUMMARY_DOT_FP32: tl.constexpr,
    BIAS_STRIDE_B: tl.constexpr,
    BIAS_STRIDE_H: tl.constexpr,
    BIAS_STRIDE_Q: tl.constexpr,
    BIAS_STRIDE_K: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_kb = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_idx = Q_START + pid_qb * BLOCK_M + offs_m
    k_idx = pid_kb * BLOCK_N + offs_n
    d_mask = offs_d < D_HEAD
    q_mask = q_idx < N_CTX
    k_mask = k_idx < N_CTX

    q_ptrs = Q + pid_bh * N_CTX * D_HEAD + q_idx[:, None] * D_HEAD + offs_d[None, :]
    k_ptrs = K + pid_bh * N_CTX * D_HEAD + k_idx[:, None] * D_HEAD + offs_d[None, :]
    v_ptrs = V + pid_bh * N_CTX * D_HEAD + k_idx[:, None] * D_HEAD + offs_d[None, :]

    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

    scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * SCALE
    valid = q_mask[:, None] & k_mask[None, :]
    if IS_CAUSAL:
        valid = valid & (q_idx[:, None] >= k_idx[None, :])

    if HAS_BIAS:
        b_idx = pid_bh // H
        h_idx = pid_bh - b_idx * H
        bias_ptrs = (
            BIAS
            + b_idx * BIAS_STRIDE_B
            + h_idx * BIAS_STRIDE_H
            + q_idx[:, None] * BIAS_STRIDE_Q
            + k_idx[None, :] * BIAS_STRIDE_K
        )
        bias = tl.load(bias_ptrs, mask=valid, other=0.0).to(tl.float32)
        scores = scores + bias

    scores = tl.where(valid, scores * 1.4426950408889634, -float("inf"))
    row_has = tl.max(tl.where(valid, 1, 0), axis=1) > 0
    local_m_raw = tl.max(scores, axis=1)
    local_m_safe = tl.where(row_has, local_m_raw, 0.0)
    p = tl.where(valid, tl.exp2(scores - local_m_safe[:, None]), 0.0)
    local_z = tl.sum(p, axis=1)
    if SUMMARY_DOT_FP32:
        local_s = tl.dot(p, v.to(tl.float32), input_precision=INPUT_PRECISION)
    else:
        local_s = tl.dot(p.to(tl.float16), v, input_precision=INPUT_PRECISION)
    local_m = tl.where(row_has, local_m_raw, -float("inf"))

    base = ((pid_kb * Q_BLOCKS + pid_qb) * tl.num_programs(1) + pid_bh) * BLOCK_M + offs_m
    tl.store(M_BUF + base, local_m, mask=q_mask)
    tl.store(Z_BUF + base, local_z, mask=q_mask)
    s_ptrs = S_BUF + base[:, None] * D_HEAD + offs_d[None, :]
    tl.store(s_ptrs, local_s, mask=q_mask[:, None] & d_mask[None, :])


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


def _paper_scan_defaults(
    *,
    dtype: torch.dtype,
    seq_len: int,
    head_dim: int,
    block_m: Optional[int],
    block_n: Optional[int],
    q_chunk_size: Optional[int],
    phase1_warps: Optional[int],
    phase2_warps: Optional[int],
    phase1_stages: Optional[int],
    phase2_stages: Optional[int],
) -> tuple[int, int, int, int, int, int, int]:
    if dtype == torch.float32:
        # Paper mode: phase-1 computes one exact (m,z,s) summary per K PARTITION
        # via the in-partition monoid fold (tiled kernel, block_n>512 → paper's
        # intra-block scan, written sequentially but order-independent by monoid
        # associativity), then phase-2 parallel-reduces the few partition
        # summaries. The fold keeps the running (m,z,s) in REGISTERS (fast, high
        # occupancy) — this is what makes it match the paper trend on A100.
        # Verified (clean GPU, true fp32): beats SDPA-math 2.3-3.1x AND
        # mem-efficient from ~32K (0.97→0.91, 越長越贏). block_m=32 keeps fp32
        # register pressure low.
        # Paper-mode tiled register fold. The tiled kernel MICRO-tiles block_n (64-wide
        # SMEM) so a wide LOGICAL block_n no longer OOMs, and partial last blocks are now
        # masked (NEEDS_MASK) so non-divisible ViT seqs ((img/patch)^2+1) use this fast path
        # too. Pick block_n ~ seq (one tiled k-block) for mid seq, 2048 (multi-block scan)
        # for long seq, 256 (non-tiled) only for very short. block_m=32 avoids the fp32
        # register-spill cliff.
        if head_dim <= 64:
            default_block_m = 32
            if seq_len <= 512:
                default_block_n = 256
            elif seq_len < 2048:
                default_block_n = ((seq_len + 63) // 64) * 64  # round up to 64, >512 -> tiled
            else:
                default_block_n = 2048
        else:
            default_block_m = 32
            default_block_n = 64
        default_q_chunk = 512 if seq_len >= 4096 else 1024
        default_phase1_warps = 4
        default_phase2_warps = 4
        default_phase1_stages = 2
    elif dtype == torch.float16 and head_dim == 64 and seq_len >= 8192:
        # Paper mode (long fp16): tiled in-partition REGISTER monoid fold + parallel reduce.
        # vs FlashAttention this gives the paper's "approaches FA at long N" trend (narrows
        # 2.43→1.71 over 8K→262K, ~58% FA2 TC eff), O(n) mem. Paper claims only "approaches".
        # NOTE (2026-06-10): tried extending tiled to fp16 MID-seq (ViT range) like the fp32
        # fix — it REGRESSED (seq2305 3.49→5.79× FA2). fp16's direct matmul is already TC-
        # efficient and block_m=64 doesn't spill (unlike fp32), so the non-tiled path below is
        # faster for fp16 mid-seq. fp16 ViT loss is the FA2 ceiling, not a routing bug.
        default_block_m = 64
        default_block_n = 2048
        default_q_chunk = 4096 if seq_len >= 16384 else 2048
        default_phase1_warps = 4
        default_phase2_warps = 4
        default_phase1_stages = 1
    else:
        default_block_m = 64 if head_dim <= 64 else 32
        default_block_n = 256 if head_dim <= 64 else 64
        default_q_chunk = 1024 if seq_len >= 4096 else min(seq_len, 1024)
        default_phase1_warps = 4
        default_phase2_warps = 4
        default_phase1_stages = 2

    return (
        max(16, int(default_block_m if block_m is None else block_m)),
        max(16, int(default_block_n if block_n is None else block_n)),
        max(16, int(default_q_chunk if q_chunk_size is None else q_chunk_size)),
        _validate_warps("phase1_warps", default_phase1_warps if phase1_warps is None else phase1_warps),
        _validate_warps("phase2_warps", default_phase2_warps if phase2_warps is None else phase2_warps),
        max(1, int(default_phase1_stages if phase1_stages is None else phase1_stages)),
        max(1, int(1 if phase2_stages is None else phase2_stages)),
    )


def twopass_attention(
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
) -> torch.Tensor:
    """Strict paper-scan ELSA attention.

    Public release execution is restricted to the paper scan path. Phase 1
    writes one exact `(m, z, s)` summary per K block. Phase 2 performs a
    log-depth parallel prefix scan over those summaries and emits the final
    prefix.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must have shape [B,H,N,D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes")
    if q.device.type != "cuda":
        raise ValueError("twopass_attention currently requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are fp16, bf16, and fp32")
    if algorithm not in ("auto", "paper_scan"):
        raise ValueError("only algorithm='paper_scan' is available in this strict release")

    batch, heads, seq_len, head_dim = map(int, q.shape)
    if seq_len <= 0:
        return torch.empty_like(q)
    if head_dim > 128:
        raise ValueError("head_dim > 128 is not supported by the current Triton kernels")

    if input_precision == "auto":
        input_precision = "tf32x3" if q.dtype == torch.float32 else "ieee"
    if input_precision not in ("ieee", "tf32", "tf32x3"):
        raise ValueError("input_precision must be 'auto', 'ieee', 'tf32', or 'tf32x3'")
    if summary_dtype is None:
        summary_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype
    if summary_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("summary_dtype must be fp16, bf16, or fp32")

    block_m, block_n, q_chunk_size, phase1_warps, phase2_warps, phase1_stages, phase2_stages = _paper_scan_defaults(
        dtype=q.dtype,
        seq_len=seq_len,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        q_chunk_size=q_chunk_size,
        phase1_warps=phase1_warps,
        phase2_warps=phase2_warps,
        phase1_stages=phase1_stages,
        phase2_stages=phase2_stages,
    )
    q_chunk_size = max(block_m, (q_chunk_size // block_m) * block_m)

    from .attention_parallel_scan import twopass_attention_paper_scan

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
    )
