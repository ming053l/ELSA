import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
from contextlib import contextmanager
import math, os

_SHORT_ATTENTION_COMPILED = None


@contextmanager
def _tf32_context(enabled: Optional[bool]):
    if enabled is None:
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def _short_attention_base(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    B, H, N, D = q.shape
    dv = v.shape[-1]
    q_flat = q.reshape(B * H, N, D)
    k_flat = k.reshape(B * H, N, D)
    v_flat = v.reshape(B * H, N, dv)
    scores = torch.bmm(q_flat, k_flat.transpose(-1, -2))

    scale = logit_scale.exp().clamp_min(1e-6)
    scores = scores.view(B, H, N, N) * scale.view(1, H, 1, 1)

    attn_bias = None
    if relative_position_bias is not None:
        attn_bias = relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1)
    if mask is not None:
        mask_bias = mask
        if mask_bias.dim() == 4 and mask_bias.size(1) == 1:
            mask_bias = mask_bias.view(B, 1, N, N)
        attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias
    if attn_bias is not None:
        scores = scores + attn_bias

    m = scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores - m)
    denom = weights.sum(dim=-1, keepdim=True)
    attn = weights / denom.clamp_min(1e-6)
    out = torch.bmm(attn.reshape(B * H, N, N), v_flat)
    return out.view(B, H, N, dv)


def _short_attention_compiled():
    global _SHORT_ATTENTION_COMPILED
    if _SHORT_ATTENTION_COMPILED is not None:
        return _SHORT_ATTENTION_COMPILED
    if not bool(int(os.environ.get("ELSA_SWIN_SHORT_COMPILE", "1"))):
        return None
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return None

    def _impl(q, k, v, logit_scale, relative_position_bias, mask):
        return _short_attention_base(q, k, v, logit_scale, relative_position_bias, mask)

    try:
        _SHORT_ATTENTION_COMPILED = compile_fn(_impl, mode="reduce-overhead", fullgraph=True)
    except Exception:
        _SHORT_ATTENTION_COMPILED = None
    return _SHORT_ATTENTION_COMPILED

def _choose_tile(N: int, dev_prop, prefer_large=True):
    """
    根據序列長度 N 與 GPU 性能，選出自適應 BLOCK_M/N。
    - dev_prop: torch.cuda.get_device_properties(device)
    - prefer_large: 是否優先選大 tile (對資料中心卡較好)
    """
    # 桌機卡頻寬 < 400GB/s 視為 bandwidth-bound
    is_bandwidth_bound = getattr(dev_prop, "memoryBusWidth", 0) * \
                         getattr(dev_prop, "memoryClockRate", 0) < 400_000

    # 排序策略：資料中心卡 128→96→64；桌機卡 96→64→128
    candidate = [128, 64] if prefer_large and not is_bandwidth_bound else [64, 128]
    for blk in candidate:
        if N % blk == 0:
            return blk
    # 仍無法整除：依 N 大小決定
    return 128 if N > 8192 else 64

_ELSA_FP32_TUNE_CACHE = {}
_ELSA_FP32_FAST_TUNE_CACHE = {}
_ELSA_FP32_INFER_TUNE_CACHE = {}
_ELSA_FP32_SPLITD_TUNE_CACHE = {}
_ELSA_FP32_TRAIN_TUNE_CACHE = {}


def _elsa_fp32_candidates(D: int) -> list[tuple[int, int, int, int]]:
    wide = os.environ.get("ELSA_TRITON_FP32_TUNE_WIDE") == "1"
    if D >= 256:
        candidates = [
            (8, 128, 4, 2),
            (8, 256, 4, 2),
            (16, 32, 2, 1),
            (16, 64, 4, 2),
            (16, 128, 4, 2),
            (32, 32, 2, 1),
            (32, 64, 4, 2),
            (32, 128, 4, 2),
            (32, 256, 4, 2),
            (32, 512, 4, 2),
            (64, 32, 4, 2),
            (64, 64, 4, 2),
            (64, 128, 4, 2),
            (64, 256, 4, 2),
            (64, 512, 4, 2),
            (128, 32, 4, 2),
            (128, 64, 4, 2),
            (128, 128, 4, 2),
            (128, 64, 8, 2),
            (128, 128, 8, 2),
            (128, 128, 8, 3),
            (32, 128, 8, 2),
            (64, 128, 8, 2),
            (32, 256, 8, 2),
            (64, 256, 8, 2),
            (32, 128, 8, 3),
            (64, 128, 8, 3),
            (32, 256, 8, 3),
            (64, 256, 8, 3),
        ]
        if wide:
            extra = []
            for block_q in (16, 32, 48, 64):
                for block_n in (64, 96, 128, 160, 192, 256):
                    extra.append((block_q, block_n, 4, 2))
                    extra.append((block_q, block_n, 8, 2))
            candidates.extend(extra)
        return candidates
    candidates = [
        (32, 32, 2, 1),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
        (64, 64, 8, 3),
        (64, 128, 4, 2),
        (64, 256, 4, 2),
        (128, 32, 4, 2),
        (128, 64, 4, 2),
        (128, 128, 4, 2),
        (128, 128, 8, 3),
    ]
    if wide:
        extra = []
        for block_q in (32, 48, 64):
            for block_n in (64, 96, 128, 160):
                extra.append((block_q, block_n, 4, 2))
                extra.append((block_q, block_n, 8, 2))
        candidates.extend(extra)
    return candidates

def _tune_elsa_fp32_kernel(
    kernel,
    q_,
    k_,
    v_,
    out_s,
    out_z,
    out_m,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    for block_q, block_n, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        block_d = 32 * ((D + 31) // 32)
        grid = (triton.cdiv(N, block_q), B * H)
        # Warmup to avoid compile time in timing.
        try:
            kernel[grid](
                q_, k_, v_,
                out_s, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_s.stride(0), out_s.stride(1), out_s.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_,
                out_s, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_s.stride(0), out_s.stride(1), out_s.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_infer_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    for block_q, block_n, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_q), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_splitd_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = [
        (16, 64, 64, 4, 2),
        (16, 128, 64, 4, 2),
        (32, 64, 64, 4, 2),
        (32, 128, 64, 4, 2),
        (32, 128, 64, 8, 2),
        (32, 256, 64, 8, 2),
        (32, 128, 128, 4, 2),
        (32, 256, 128, 8, 2),
        (64, 64, 64, 4, 2),
        (64, 128, 64, 8, 2),
        (64, 128, 128, 8, 2),
        (64, 256, 64, 8, 2),
    ]
    if os.environ.get("ELSA_TRITON_FP32_TUNE_WIDE") == "1":
        extra = []
        for block_q in (16, 32, 48, 64):
            for block_n in (64, 96, 128, 160, 192, 256):
                for block_d in (64, 128):
                    extra.append((block_q, block_n, block_d, 4, 2))
                    extra.append((block_q, block_n, block_d, 8, 2))
        candidates.extend(extra)
    best = None
    best_ms = None
    for block_q, block_n, block_d, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_q), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, block_d, num_wp, num_stages)
    return best


def _tune_elsa_fp32_fast_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
):
    block_d = 32 * ((D + 31) // 32)
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    candidates = [
        (16, 32, 2, 1),
        (16, 64, 4, 2),
        (16, 128, 4, 2),
        (32, 32, 2, 1),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (32, 256, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
        (64, 128, 4, 2),
        (64, 256, 4, 2),
        (32, 128, 8, 2),
        (64, 128, 8, 2),
        (32, 256, 8, 2),
        (64, 256, 8, 2),
    ]
    best = None
    best_ms = None
    for block_m, block_n, num_wp, num_stages in candidates:
        if block_m > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_m), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_m, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_fast_mz_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    out_m,
    out_z,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    *,
    allow_tf32: bool,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    block_d = 32 * ((D + 31) // 32)
    for block_m, block_n, num_wp, num_stages in candidates:
        if block_m > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_m), B * H)
        try:
            kernel[grid](
                q_,
                k_,
                v_,
                out,
                out_m,
                out_z,
                q_.stride(0),
                q_.stride(1),
                q_.stride(2),
                k_.stride(0),
                k_.stride(1),
                k_.stride(2),
                v_.stride(0),
                v_.stride(1),
                v_.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out_m.stride(0),
                out_m.stride(1),
                out_z.stride(0),
                out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_,
                k_,
                v_,
                out,
                out_m,
                out_z,
                q_.stride(0),
                q_.stride(1),
                q_.stride(2),
                k_.stride(0),
                k_.stride(1),
                k_.stride(2),
                v_.stride(0),
                v_.stride(1),
                v_.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out_m.stride(0),
                out_m.stride(1),
                out_z.stride(0),
                out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_m, block_n, num_wp, num_stages)
    return best


@triton.jit
def elsa_swinv2_kernel_short(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D, DV,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    base_q = Q + b * stride_qb + h * stride_qh
    base_k = K + b * stride_kb + h * stride_kh
    base_v = V + b * stride_vb + h * stride_vh
    base_out = Out + b * stride_ob + h * stride_oh

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    scale = tl.exp(tl.load(LogitScale + h)).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for d0 in tl.static_range(0, 64, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_ptrs = base_q + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
        k_ptrs = base_k + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn

        q_chunk = tl.load(q_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k_chunk = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(q_chunk, k_chunk, allow_tf32=ALLOW_TF32)

    acc = acc * scale

    if HAS_BIAS:
        bias = tl.load(
            RelBias + h * stride_rb_h + offs_n[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += bias

    if HAS_MASK:
        mask_vals = tl.load(
            Mask
            + b * stride_mask_b
            + h * stride_mask_h
            + offs_n[:, None] * stride_mask_n
            + offs_n[None, :] * stride_mask_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += mask_vals

    acc = tl.where(mask_n[None, :], acc, float("-inf"))
    m = tl.max(acc, axis=1)
    acc = acc - m[:, None]
    p = tl.exp(acc)
    l = tl.sum(p, axis=1)
    attn = p / tl.maximum(l[:, None], 1e-6)

    for d0 in tl.static_range(0, 64, BLOCK_D):
        offs_dv = d0 + tl.arange(0, BLOCK_D)
        mask_dv = offs_dv < DV
        v_ptrs = base_v + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)
        out_chunk = tl.dot(attn, v_chunk, allow_tf32=ALLOW_TF32).to(tl.float32)
        tl.store(
            base_out + offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od,
            out_chunk.to(Out.dtype.element_ty),
            mask=mask_n[:, None] & mask_dv[None, :],
        )


@triton.jit
def elsa_swinv2_kernel_short_fused(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D, DV,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    base_q = Q + b * stride_qb + h * stride_qh
    base_k = K + b * stride_kb + h * stride_kh
    base_v = V + b * stride_vb + h * stride_vh
    base_out = Out + b * stride_ob + h * stride_oh

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q = tl.load(
        base_q + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        base_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    scale = tl.exp(tl.load(LogitScale + h)).to(tl.float32)
    q = q * scale

    scores = tl.dot(q, k, trans_b=True, allow_tf32=ALLOW_TF32).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(
            RelBias + h * stride_rb_h + offs_n[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += bias

    if HAS_MASK:
        mask_vals = tl.load(
            Mask
            + b * stride_mask_b
            + h * stride_mask_h
            + offs_n[:, None] * stride_mask_n
            + offs_n[None, :] * stride_mask_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += mask_vals

    scores = tl.where(mask_n[None, :], scores, float("-inf"))
    m = tl.max(scores, axis=1)
    scores = scores - m[:, None]
    p = tl.exp(scores)
    l = tl.sum(p, axis=1)
    attn = p / tl.maximum(l[:, None], 1e-6)

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < DV
    v = tl.load(
        base_v + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
        mask=mask_n[:, None] & mask_dv[None, :],
        other=0.0,
    ).to(tl.float32)

    out = tl.dot(attn, v, allow_tf32=ALLOW_TF32).to(tl.float32)

    tl.store(
        base_out + offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od,
        out.to(Out.dtype.element_ty),
        mask=mask_n[:, None] & mask_dv[None, :],
    )


@triton.jit
def kernel_elsa_attention_fwd_fixed(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    修正的 ELSA Attention kernel - 兼容 Triton 3.2.0
    - 正確的 dtype 處理以使用 Tensor Core
    - 移除不支援的 acc_dtype 參數
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 基礎偏移
    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh
    
    # M 維度範圍
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    # D 維度範圍
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # ===== 載入 Q block ===== #
    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    # 保持 FP16 以利用 Tensor Core
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # 縮放 Q - 保持 FP16
    q = q * scale
    
    # ===== 初始化累積變量 (使用 FP32) ===== #
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # ===== 主循環 ===== #
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Causal mask
        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])
        
        # ===== 載入 K block (保持 FP16) ===== #
        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        
        # ===== 計算 QK^T ===== #
        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        
        # 轉換為 FP32 進行 softmax 計算
        qk = qk.to(tl.float32)
        
        # 應用 mask
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # ===== Online softmax ===== #
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        
        # 穩定的指數計算
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        
        # 更新累積值
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # ===== 載入 V block (保持 FP16) ===== #
        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # ===== 累積輸出 ===== #
        p_cast = p.to(v.dtype)
        pv = tl.dot(p_cast, v)
        
        # 累積到 FP32
        acc = acc * alpha[:, None] + pv.to(tl.float32)
        
        # 更新 m_i
        m_i = m_new
    
    # ===== 最終歸一化 ===== #
    acc = acc / tl.maximum(l_i[:, None], 1e-6)
    
    # ===== 寫回結果 ===== #
    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    # 轉回 FP16
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fwd_fixed_mz(
    Q, K, V, Out, Out_M, Out_Z,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_zb, stride_zh, stride_zn,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """Training variant that also writes per-row max (M) and sum-exp (Z)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)

        m_i = m_new

    acc = acc / tl.maximum(l_i[:, None], 1e-6)

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = Out_M + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mn
    z_ptrs = Out_Z + pid_b * stride_zb + pid_h * stride_zh + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fwd_qknorm(
    Q, K, V, Out,
    Q_norm_w, Q_norm_b, K_norm_w, K_norm_b,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """支援 QK normalization 的版本"""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # 載入 norm weights/bias
    q_w = tl.load(Q_norm_w + offs_d, mask=mask_d, other=1.0)
    q_b = tl.load(Q_norm_b + offs_d, mask=mask_d, other=0.0)
    k_w = tl.load(K_norm_w + offs_d, mask=mask_d, other=1.0)
    k_b = tl.load(K_norm_b + offs_d, mask=mask_d, other=0.0)
    
    # 載入並正規化 Q
    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # 簡化的 LayerNorm (假設已預先計算 mean/std)
    q = q * q_w[None, :] + q_b[None, :]
    q = q * scale
    
    # 初始化
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])
        
        # 載入並正規化 K
        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        k = k * k_w[:, None] + k_b[:, None]
        
        # QK^T
        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # Softmax
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # V
        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 累積
        p_cast = p.to(v.dtype)
        pv = tl.dot(p_cast, v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new
    
    # 歸一化並存儲
    acc = acc / tl.maximum(l_i[:, None], 1e-6)
    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fwd_qknorm_mz(
    Q, K, V, Out, Out_M, Out_Z,
    Q_norm_w, Q_norm_b, K_norm_w, K_norm_b,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_zb, stride_zh, stride_zn,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_w = tl.load(Q_norm_w + offs_d, mask=mask_d, other=1.0)
    q_b = tl.load(Q_norm_b + offs_d, mask=mask_d, other=0.0)
    k_w = tl.load(K_norm_w + offs_d, mask=mask_d, other=1.0)
    k_b = tl.load(K_norm_b + offs_d, mask=mask_d, other=0.0)

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * q_w[None, :] + q_b[None, :]
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        k = k * k_w[:, None] + k_b[:, None]

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)

        m_i = m_new

    acc = acc / tl.maximum(l_i[:, None], 1e-6)

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = Out_M + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mn
    z_ptrs = Out_Z + pid_b * stride_zb + pid_h * stride_zh + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)

@triton.jit
def kernel_elsa_attention_fp32_fast(
    Q, K, V, OUT,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fp32_fast_mz(
    Q, K, V, OUT, OUT_M, OUT_Z,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    stride_mb, stride_mn,
    stride_zb, stride_zn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = OUT_M + pid_bh * stride_mb + offs_m * stride_mn
    z_ptrs = OUT_Z + pid_bh * stride_zb + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)

def _mem_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_mem_autotune_configs(), key=["N_CTX", "D_HEAD"])
@triton.jit
def kernel_elsa_attention_fp32_fast_tuned(
    Q, K, V, OUT,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])
    
@triton.jit
def kernel_integral_mhsa_stable(
    Q, K, V,
    OUT_S, OUT_Z, OUT_M,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_sbh, stride_sn, stride_sd,
    stride_zh, stride_z0, stride_z1,
    stride_mh, stride_m0, stride_m1,
    BLOCK_Q : tl.constexpr,   
    BLOCK_N : tl.constexpr,   
    D_HEAD  : tl.constexpr,   
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q  = tl.program_id(0)               
    pid_bh = tl.program_id(1)               

    offs_q  = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)      
    offs_n  = tl.arange(0, BLOCK_N)                        
    offs_d  = tl.arange(0, D_HEAD)                         
    mask_q  = offs_q < N_CTX

    # ---- load Q (保持原始邏輯) ----------------------------------- #
    q_ptrs = Q + pid_bh*stride_qbh + offs_q[:,None]*stride_qn + offs_d[None,:]*stride_qd
    q      = tl.load(q_ptrs, mask=mask_q[:,None]).to(tl.float32)

    # ---- running stats (使用 D_HEAD 而非 PAD_D) ------------------------------------ #
    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    # ---- sweep over sequence (保持原始邏輯) ------------------------------ #
    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX
        
        k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        v_ptrs = V + pid_bh*stride_vbh + (start_n+offs_n)[:,None]*stride_vn + offs_d[None,:]*stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:,None], other=0.).to(tl.float32)
        
        m_prev = m_q
        scores = tl.dot(q, k, allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)  # 使用更小的值
        
        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_prev, cur_m)
        
        alpha = tl.exp(m_prev - new_m)
        beta = tl.exp(scores - new_m[:,None])
        beta = tl.where(mask_n[None,:], beta, 0.0)  # 確保masked位置為0
        
        z_q = z_q * alpha + tl.sum(beta, 1)
        s_q = s_q * alpha[:,None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)
        
        m_q = new_m
    # ---- 關鍵修復：添加數值穩定性保護 ------- #
    # z_q_safe = tl.maximum(z_q, 1e-8)
    # result = s_q / z_q_safe[:,None]
    
    s_ptrs = OUT_S + pid_bh*stride_sbh + offs_q[:,None]*stride_sn + offs_d[None,:]*stride_sd
    z_ptrs = OUT_Z + pid_bh*stride_z0 + offs_q*stride_z1
    m_ptrs = OUT_M + pid_bh*stride_m0 + offs_q*stride_m1
    
    mask_sd = mask_q[:,None]
    tl.store(s_ptrs, s_q, mask=mask_sd)  # 存儲 s_q，不是 z_q！
    tl.store(z_ptrs, z_q, mask=mask_q)   # 存儲 z_q
    tl.store(m_ptrs, m_q, mask=mask_q)   # 存儲 m_q


@triton.jit
def kernel_integral_mhsa_stable_infer(
    Q, K, V,
    OUT,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    mask_q = offs_q < N_CTX

    q_ptrs = Q + pid_bh * stride_qbh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q[:, None]).to(tl.float32)

    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        v_ptrs = V + pid_bh * stride_vbh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        m_prev = m_q
        scores = tl.dot(q, k, allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)

        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_prev, cur_m)

        alpha = tl.exp(m_prev - new_m)
        beta = tl.exp(scores - new_m[:, None])
        beta = tl.where(mask_n[None, :], beta, 0.0)

        z_q = z_q * alpha + tl.sum(beta, 1)
        s_q = s_q * alpha[:, None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)

        m_q = new_m

    inv_z = 1.0 / tl.maximum(z_q, 1e-6)
    out = s_q * inv_z[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_q[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_q[:, None])


@triton.jit
def kernel_integral_mhsa_splitd_infer(
    Q, K, V,
    OUT,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_N)
    mask_q = offs_q < N_CTX

    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX
        scores = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)
        for start_d in range(0, D_HEAD, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_HEAD
            q_ptrs = Q + pid_bh * stride_qbh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
            k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
            q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            scores += tl.dot(q, k, allow_tf32=ALLOW_TF32)

        scores = scores * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)

        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_q, cur_m)

        alpha = tl.exp(m_q - new_m)
        beta = tl.exp(scores - new_m[:, None])
        beta = tl.where(mask_n[None, :], beta, 0.0)

        z_q = z_q * alpha + tl.sum(beta, 1)

        v_ptrs = V + pid_bh * stride_vbh + (start_n + offs_n)[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        s_q = s_q * alpha[:, None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)

        m_q = new_m

    inv_z = 1.0 / tl.maximum(z_q, 1e-6)
    out = s_q * inv_z[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_q[:, None] * stride_on + tl.arange(0, D_HEAD)[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_q[:, None])


@triton.jit
def kernel_elsa_bwd_delta(
    Q, K, V, DO, M, Z, DELTA,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do_ptrs = DO + pid_bh * stride_dobh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_ptrs = M + pid_bh * stride_mbh + offs_m * stride_mn
    z_ptrs = Z + pid_bh * stride_zbh + offs_m * stride_zn
    m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
    z = tl.maximum(z, 1e-6)

    delta = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        delta += tl.sum(dp * p, axis=1)

    delta_ptrs = DELTA + pid_bh * stride_dbh + offs_m * stride_dn
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def kernel_elsa_bwd_dq(
    Q, K, V, DO, M, Z, DELTA, DQ,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    stride_dqbh, stride_dqn, stride_dqd,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do_ptrs = DO + pid_bh * stride_dobh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_ptrs = M + pid_bh * stride_mbh + offs_m * stride_mn
    z_ptrs = Z + pid_bh * stride_zbh + offs_m * stride_zn
    d_ptrs = DELTA + pid_bh * stride_dbh + offs_m * stride_dn
    m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
    d = tl.load(d_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.maximum(z, 1e-6)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        ds = (dp - d[:, None]) * p
        acc += tl.dot(ds, k, allow_tf32=ALLOW_TF32) * SCALE

    dq_ptrs = DQ + pid_bh * stride_dqbh + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_bwd_dkv(
    Q, K, V, DO, M, Z, DELTA, DK, DV,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    stride_dkbh, stride_dkn, stride_dkd,
    stride_dvbh, stride_dvn, stride_dvd,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N_CTX
    mask_d = offs_d < D_HEAD

    k_ptrs = K + pid_bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_bh * stride_vbh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc_k = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m_block = start_m + offs_m
        mask_m = offs_m_block < N_CTX

        q_ptrs = Q + pid_bh * stride_qbh + offs_m_block[:, None] * stride_qn + offs_d[None, :] * stride_qd
        do_ptrs = DO + pid_bh * stride_dobh + offs_m_block[:, None] * stride_don + offs_d[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        m_ptrs = M + pid_bh * stride_mbh + offs_m_block * stride_mn
        z_ptrs = Z + pid_bh * stride_zbh + offs_m_block * stride_zn
        d_ptrs = DELTA + pid_bh * stride_dbh + offs_m_block * stride_dn
        m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        d = tl.load(d_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        z = tl.maximum(z, 1e-6)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        ds = (dp - d[:, None]) * p

        acc_k += tl.dot(tl.trans(ds), q, allow_tf32=ALLOW_TF32) * SCALE
        acc_v += tl.dot(tl.trans(p), do, allow_tf32=ALLOW_TF32)

    dk_ptrs = DK + pid_bh * stride_dkbh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + pid_bh * stride_dvbh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, acc_k, mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, acc_v, mask=mask_n[:, None] & mask_d[None, :])

class ELSA_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, qk_norm_weights=None, is_causal=False):
        B, H, N, D = q.shape

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), bool(is_causal))

        needs_grad = q.requires_grad or k.requires_grad or v.requires_grad
        if needs_grad and qk_norm_weights is not None:
            raise RuntimeError("ELSA_triton backward does not support qk_norm.")
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        out = torch.empty_like(q)
        use_tf32 = bool(
            q.dtype == torch.float32
            and k.dtype == torch.float32
            and v.dtype == torch.float32
            and torch.backends.cuda.matmul.allow_tf32
        )
        
        # Block 大小
        BLOCK_D = 16 * ((D + 15) // 16)
        
        # 1. 取得序列長度與 GPU 性能
        dev_prop = torch.cuda.get_device_properties(q.device)
        blk = _choose_tile(N, dev_prop, prefer_large=True)

        stream_env = os.environ.get("ELSA_TRITON_STREAM", "0") == "1"
        if stream_env:
            try:
                stream_q = int(os.environ.get("ELSA_STREAM_Q_BLOCK", "0"))
            except ValueError:
                stream_q = 0
            try:
                stream_kv = int(os.environ.get("ELSA_STREAM_KV_BLOCK", "0"))
            except ValueError:
                stream_kv = 0
        
        # 2. 根據 blk 選擇對應 warp/stage（保持簡易）
        if blk == 128:
            BLOCK_M = BLOCK_N = 128 if D <= 64 else 64   # 128×128 tile 時 M 可 128、N 固定 128
            num_warps = 4 if D <= 64 else 8
        elif blk == 96:
            BLOCK_M = BLOCK_N = 96
            num_warps = 4
        else:  # 64
            BLOCK_M = BLOCK_N = 64
            num_warps = 4
        if stream_env:
            if stream_q > 0:
                BLOCK_M = max(16, (stream_q // 16) * 16)
            if stream_kv > 0:
                BLOCK_N = max(16, (stream_kv // 16) * 16)

        
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        
        # 選擇 kernel
        if qk_norm_weights is not None:
            q_norm_w, q_norm_b, k_norm_w, k_norm_b = qk_norm_weights
            if needs_grad:
                out_m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
                out_z = torch.empty_like(out_m)
                kernel_elsa_attention_fwd_qknorm_mz[grid](
                    q, k, v, out, out_m, out_z,
                    q_norm_w, q_norm_b, k_norm_w, k_norm_b,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    out_m.stride(0), out_m.stride(1), out_m.stride(2),
                    out_z.stride(0), out_z.stride(1), out_z.stride(2),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=2,
                )
                ctx.save_for_backward(q, k, v, out_m, out_z)
            else:
                kernel_elsa_attention_fwd_qknorm[grid](
                    q, k, v, out,
                    q_norm_w, q_norm_b, k_norm_w, k_norm_b,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=2,
                )
        else:
            if needs_grad:
                out_m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
                out_z = torch.empty_like(out_m)
                kernel_elsa_attention_fwd_fixed_mz[grid](
                    q, k, v, out, out_m, out_z,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    out_m.stride(0), out_m.stride(1), out_m.stride(2),
                    out_z.stride(0), out_z.stride(1), out_z.stride(2),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=2,
                )
                ctx.save_for_backward(q, k, v, out_m, out_z)
            else:
                kernel_elsa_attention_fwd_fixed[grid](
                    q, k, v, out,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=2,
                )

        ctx.scale = scale
        ctx.use_tf32 = use_tf32
        ctx.needs_grad = needs_grad
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        if not getattr(ctx, "needs_grad", False):
            return None, None, None, None, None, None
        q, k, v, out_m, out_z = ctx.saved_tensors
        scale = ctx.scale
        allow_tf32 = bool(getattr(ctx, "use_tf32", False))

        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        do = grad_out.contiguous()

        B, H, N, D = q_.shape
        qh = q_.view(B * H, N, D)
        kh = k_.view(B * H, N, D)
        vh = v_.view(B * H, N, D)
        doh = do.view(B * H, N, D)
        mh = out_m.view(B * H, N)
        zh = out_z.view(B * H, N)

        block_m = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_M", "64"))
        block_n = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_N", "64"))
        block_m = 128 if block_m >= 128 else 64
        block_n = 128 if block_n >= 128 else 64
        num_warps = int(os.environ.get("ELSA_TRITON_BWD_WARPS", "4"))
        num_stages = int(os.environ.get("ELSA_TRITON_BWD_STAGES", "2"))
        block_d = 16 * ((D + 15) // 16)

        delta = torch.empty_like(zh)
        dq = torch.empty_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        grid_q = (triton.cdiv(N, block_m), B * H)
        kernel_elsa_bwd_delta[grid_q](
            qh, kh, vh, doh, mh, zh, delta,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        kernel_elsa_bwd_dq[grid_q](
            qh, kh, vh, doh, mh, zh, delta, dq,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid_k = (triton.cdiv(N, block_n), B * H)
        kernel_elsa_bwd_dkv[grid_k](
            qh, kh, vh, doh, mh, zh, delta, dk, dv,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq.view(B, H, N, D), dk.view(B, H, N, D), dv.view(B, H, N, D), None, None, None

class ELSA_triton_fp32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), False)

        use_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        kernel = kernel_integral_mhsa_stable
        block_n = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_N", "64"))
        block_q = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_Q", "64"))
        num_wp = int(os.environ.get("ELSA_TRITON_FWD_WARPS", "4"))
        num_stages = int(os.environ.get("ELSA_TRITON_FWD_STAGES", "2"))
        auto_tune = bool(int(os.environ.get("ELSA_TRITON_FWD_AUTOTUNE", "0")))
        manual_override = any(
            key in os.environ
            for key in (
                "ELSA_TRITON_FWD_BLOCK_N",
                "ELSA_TRITON_FWD_BLOCK_Q",
                "ELSA_TRITON_FWD_WARPS",
                "ELSA_TRITON_FWD_STAGES",
            )
        )
        stream_env = os.environ.get("ELSA_TRITON_FP32_STREAM", "0") == "1"
        if stream_env:
            try:
                stream_q = int(os.environ.get("ELSA_STREAM_Q_BLOCK", "0"))
            except ValueError:
                stream_q = 0
            try:
                stream_kv = int(os.environ.get("ELSA_STREAM_KV_BLOCK", "0"))
            except ValueError:
                stream_kv = 0
            if stream_q > 0:
                block_q = max(16, (stream_q // 16) * 16)
            if stream_kv > 0:
                block_n = max(16, (stream_kv // 16) * 16)
            auto_tune = False
            manual_override = True

        B, H, N, D = q.shape
        needs_grad = q.requires_grad or k.requires_grad or v.requires_grad
        fast_env = os.environ.get("ELSA_TRITON_FP32_FAST")
        fast_autotune = os.environ.get("ELSA_TRITON_FP32_FAST_AUTOTUNE", "1") == "1"
        infer_env = os.environ.get("ELSA_TRITON_FP32_INFER", "1")
        splitd_env = os.environ.get("ELSA_TRITON_FP32_SPLITD", "0")
        if fast_env is None:
            use_fast = (not needs_grad) and D >= 256
        else:
            use_fast = (not needs_grad) and fast_env == "1"
        use_infer_kernel = (not needs_grad) and infer_env != "0"
        use_splitd_kernel = use_infer_kernel and splitd_env == "1" and D == 256
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_ = q.view(B * H, N, D)
        k_ = k.view(B * H, N, D)
        v_ = v.view(B * H, N, D)

        if use_fast:
            out = torch.empty_like(q_, dtype=q.dtype)
            block_d = 32 * ((D + 31) // 32)
            cfg = None
            if fast_autotune:
                tune_key = (q.device.index or -1, N, D)
                cfg = _ELSA_FP32_FAST_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_fast_kernel(
                        kernel_elsa_attention_fp32_fast,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                    )
                    if cfg:
                        _ELSA_FP32_FAST_TUNE_CACHE[tune_key] = cfg
            if cfg:
                block_m, block_n, num_wp, num_stages = cfg
                grid = (triton.cdiv(N, block_m), B * H)
                kernel_elsa_attention_fp32_fast[grid](
                    q_, k_, v_, out,
                    q_.stride(0), q_.stride(1), q_.stride(2),
                    k_.stride(0), k_.stride(1), k_.stride(2),
                    v_.stride(0), v_.stride(1), v_.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BH=B * H,
                    N_CTX=N,
                    D_HEAD=D,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_D=block_d,
                    SCALE=scale,
                    IS_CAUSAL=False,
                    ALLOW_TF32=use_tf32,
                    num_warps=num_wp,
                    num_stages=num_stages,
                )
            else:
                block_m = 64 if D <= 128 else 32
                block_n_fast = 64 if N < 256 else 128
                grid = (triton.cdiv(N, block_m), B * H)
                kernel_elsa_attention_fp32_fast[grid](
                    q_, k_, v_, out,
                    q_.stride(0), q_.stride(1), q_.stride(2),
                    k_.stride(0), k_.stride(1), k_.stride(2),
                    v_.stride(0), v_.stride(1), v_.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BH=B * H,
                    N_CTX=N,
                    D_HEAD=D,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n_fast,
                    BLOCK_D=block_d,
                    SCALE=scale,
                    IS_CAUSAL=False,
                    ALLOW_TF32=use_tf32,
                    num_warps=4,
                    num_stages=2,
                )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        if use_splitd_kernel:
            out = torch.empty_like(q_, dtype=q.dtype)
            if auto_tune and not manual_override:
                tune_key = (q.device.index or -1, N, D, "splitd")
                cfg = _ELSA_FP32_SPLITD_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_splitd_kernel(
                        kernel_integral_mhsa_splitd_infer,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=use_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_SPLITD_TUNE_CACHE[tune_key] = cfg
                if cfg:
                    block_q, block_n, block_d, num_wp, num_stages = cfg

            grid = (triton.cdiv(N, block_q), B * H)
            kernel_integral_mhsa_splitd_infer[grid](
                q_, k_, v_,
                out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=use_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        if use_infer_kernel:
            out = torch.empty_like(q_, dtype=q.dtype)
            if auto_tune and not manual_override:
                tune_key = (q.device.index or -1, N, D, "infer")
                cfg = _ELSA_FP32_INFER_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_infer_kernel(
                        kernel_integral_mhsa_stable_infer,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=use_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_INFER_TUNE_CACHE[tune_key] = cfg
                if cfg:
                    block_q, block_n, num_wp, num_stages = cfg

            grid = (triton.cdiv(N, block_q), B * H)
            kernel_integral_mhsa_stable_infer[grid](
                q_, k_, v_,
                out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=use_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        out = torch.empty_like(q_, dtype=q.dtype)
        out_z = torch.empty(B * H, N, dtype=q.dtype, device=q.device)
        out_m = torch.empty(B * H, N, dtype=q.dtype, device=q.device)

        train_fast = os.environ.get("ELSA_TRITON_FP32_TRAIN_FAST", "1") == "1"

        if auto_tune and not manual_override:
            tune_key = (q.device.index or -1, N, D)
            if train_fast:
                cfg = _ELSA_FP32_TRAIN_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_fast_mz_kernel(
                        kernel_elsa_attention_fp32_fast_mz,
                        q_,
                        k_,
                        v_,
                        out,
                        out_m,
                        out_z,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=use_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TRAIN_TUNE_CACHE[tune_key] = cfg
            else:
                cfg = _ELSA_FP32_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_kernel(
                        kernel,
                        q_,
                        k_,
                        v_,
                        out,
                        out_z,
                        out_m,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=use_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TUNE_CACHE[tune_key] = cfg
            if cfg:
                block_q, block_n, num_wp, num_stages = cfg
        # BLOCK_N = 64

        block_d = 32 * ((D + 31) // 32)
        grid = (triton.cdiv(N, block_q), B * H)

        if train_fast:
            kernel_elsa_attention_fp32_fast_mz[grid](
                q_, k_, v_, out, out_m, out_z,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_m.stride(0), out_m.stride(1),
                out_z.stride(0), out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_q,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=use_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        else:
            kernel[grid](
                q_, k_, v_,
                out, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=use_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )

        ctx.save_for_backward(q, k, v, out_m, out_z)
        ctx.scale = scale
        ctx.use_tf32 = use_tf32
        return out.view(B, H, N, D).to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out_m, out_z = ctx.saved_tensors
        scale = ctx.scale
        allow_tf32 = bool(getattr(ctx, "use_tf32", False))
        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        do = grad_out.contiguous()

        B, H, N, D = q_.shape
        qh = q_.view(B * H, N, D)
        kh = k_.view(B * H, N, D)
        vh = v_.view(B * H, N, D)
        doh = do.view(B * H, N, D)
        mh = out_m.view(B * H, N)
        zh = out_z.view(B * H, N)

        block_m = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_M", "64"))
        block_n = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_N", "64"))
        block_m = 128 if block_m >= 128 else 64
        block_n = 128 if block_n >= 128 else 64
        block_d = 16 * ((D + 15) // 16)

        delta = torch.empty_like(zh)
        dq = torch.empty_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        grid_q = (triton.cdiv(N, block_m), B * H)
        kernel_elsa_bwd_delta[grid_q](
            qh, kh, vh, doh, mh, zh, delta,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=4,
            num_stages=2,
        )

        kernel_elsa_bwd_dq[grid_q](
            qh, kh, vh, doh, mh, zh, delta, dq,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=4,
            num_stages=2,
        )

        grid_k = (triton.cdiv(N, block_n), B * H)
        kernel_elsa_bwd_dkv[grid_k](
            qh, kh, vh, doh, mh, zh, delta, dk, dv,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=4,
            num_stages=2,
        )

        return dq.view(B, H, N, D), dk.view(B, H, N, D), dv.view(B, H, N, D), None


class ELSA_triton_mem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal=False):
        B, H, N, D = q.shape

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), bool(is_causal))

        q_ = q.contiguous().view(B * H, N, D)
        k_ = k.contiguous().view(B * H, N, D)
        v_ = v.contiguous().view(B * H, N, D)

        BLOCK_D = 32 * ((D + 31) // 32)
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        use_autotune = bool(int(os.environ.get("ELSA_MEM_AUTOTUNE", "0")))
        if use_autotune:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
        else:
            BLOCK_M = 64 if D <= 128 else 32
            base_block_n = 128 if N >= 128 else 64
            try:
                env_eta = float(os.environ.get("ELSA_MEM_ETA", "1.0"))
            except ValueError:
                env_eta = 1.0
            env_eta = max(0.125, min(1.0, env_eta))
            scaled_block_n = int(base_block_n * env_eta)
            granularity = 32 if base_block_n >= 64 else 16
            BLOCK_N = granularity * max(1, math.ceil(scaled_block_n / granularity))
            num_warps = 4 if D <= 128 else 8
            grid = (triton.cdiv(N, BLOCK_M), B * H)

        out = torch.empty_like(q_)

        if use_autotune:
            kernel_elsa_attention_fp32_fast_tuned[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_D=BLOCK_D,
                SCALE=scale,
                IS_CAUSAL=is_causal,
                ALLOW_TF32=allow_tf32,
            )
        else:
            kernel_elsa_attention_fp32_fast[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                SCALE=scale,
                IS_CAUSAL=is_causal,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=2,
            )

        return out.view(B, H, N, D).to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        return None, None, None, None, None
        
# 優化的 PyTorch 實作
def ELSA_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """優化的 PyTorch 實作 - 正確處理混合精度以使用 Tensor Core"""
    B, H, N, D = q.shape
    
    # 保持原始 dtype (FP16)
    dtype = q.dtype
    device = q.device
    

    # 初始化 (FP32 累積)
    m = torch.full((B, H, N, 1), -torch.inf, dtype=torch.float32, device=device)
    l = torch.zeros((B, H, N, 1), dtype=torch.float32, device=device)
    o = torch.zeros((B, H, N, D), dtype=torch.float32, device=device)
    
    # 縮放 Q 但保持 FP16
    q_scaled = q * scale
    
    # 優化的 chunk 大小
    chunk_size = 256 if N > 1024 else 128
    
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        
        # 使用自動混合精度來最大化 Tensor Core 使用
        with torch.amp.autocast(enabled=True, dtype=dtype, device_type=q.device.type):
            # 這會使用 Tensor Core (FP16 × FP16 → FP32)
            scores = torch.matmul(q_scaled, k[:, :, i:j].transpose(-2, -1))
            scores = scores.float()  # 確保是 FP32 for softmax
        
        # Causal mask
        if is_causal:
            mask = torch.triu(
                torch.ones(N, j-i, dtype=torch.bool, device=device),
                diagonal=i+1-torch.arange(N, device=device).unsqueeze(1)
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
        # Online softmax (FP32)
        m_curr = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, m_curr)
        
        p = torch.exp(scores - m_new)
        alpha = torch.exp(m - m_new)
        
        # Dropout
        if dropout_p > 0 and q.requires_grad:
            p = F.dropout(p, p=dropout_p, training=True)
        
        # 更新 l
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        
        # 對於 o 的更新，使用混合精度以利用 Tensor Core
        with torch.amp.autocast(enabled=True, dtype=dtype, device_type=q.device.type):
            # p 會自動轉為 FP16，v 保持 FP16
            # 內部使用 Tensor Core: FP16 × FP16 → FP32 累積
            o_update = torch.matmul(p.to(dtype), v[:, :, i:j])
        
        # FP32 累積
        o = o * alpha + o_update.float()
        m = m_new
    
    # 歸一化並轉回原始 dtype
    return (o / l.clamp(min=1e-6)).to(dtype)



    
# ========= 0-A  PyTorch 版 ELSA -– 帶 bias / mask =========
@torch.jit.script
def _softmax_row(scores: torch.Tensor) -> torch.Tensor:  # (B,H,N,N)
    max_v, _ = scores.max(-1, keepdim=True)
    exp_v = (scores - max_v).exp()
    return exp_v / exp_v.sum(-1, keepdim=True)
    
def ELSA_swin_pytorch(q, k, v,
                           log_scale,         # (H,1,1) fp32
                           rel_bias=None,     # (H,N,N) fp32 / None
                           attn_mask=None):   # (B_,1,N,N) fp32 / None
    # q = F.normalize(q.to(torch.float32), dim=-1)
    # k = F.normalize(k.to(torch.float32), dim=-1)
    scores = torch.matmul(q, k.transpose(-1, -2))        # (B,H,N,N) fp32
    scores *= log_scale                                  # broadcast

    if rel_bias is not None:
        scores += rel_bias.unsqueeze(0)                  # (1,H,N,N)
    if attn_mask is not None:
        scores += attn_mask                              # (B_,1,N,N)

    attn = _softmax_row(scores)                          # bit-wise = baseline
    out  = torch.matmul(attn, v.to(torch.float32))
    return out.to(v.dtype)
             # (B,H,N,D)


import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


def ELSA_swinv2_pytorch(q, k, v, logit_scale, relative_position_bias=None, mask=None, chunk_size=128):
    """
    ELSA attention PyTorch implementation for Swin Transformer v2
    
    Args:
        q, k, v: (B, H, N, D) - already normalized for cosine attention
        logit_scale: (H, 1, 1) - learnable temperature parameter
        relative_position_bias: (H, N, N) - relative position bias
        mask: (B, 1, N, N) - attention mask, 0 or -inf
        chunk_size: chunk size for memory-efficient computation
    
    Returns:
        output: (B, H, N, D)
    """
    B, H, N, D = q.shape
    
    # Initialize running statistics
    m = torch.full((B, H, N, 1), -1e10, dtype=torch.float32, device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(B, H, N, D, dtype=torch.float32, device=q.device)
    
    # Apply logit scale
    scale = logit_scale.exp()
    
    # Process in chunks
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        
        # Get key and value chunks
        k_chunk = k[:, :, start_idx:end_idx]
        v_chunk = v[:, :, start_idx:end_idx]
        
        # Compute cosine similarity scores
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        
        # Add relative position bias if provided
        if relative_position_bias is not None:
            scores = scores + relative_position_bias[:, :, start_idx:end_idx].unsqueeze(0)
        
        # Add mask if provided
        if mask is not None:
            scores = scores + mask[:, :, :, start_idx:end_idx]
        
        # Stable softmax update
        scores_max = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, scores_max)
        
        # Update accumulator
        correction = torch.exp(m - m_new)
        scores_exp = torch.exp(scores - m_new)
        
        l = l * correction + scores_exp.sum(dim=-1, keepdim=True)
        acc = acc * correction + torch.matmul(scores_exp, v_chunk.to(torch.float32))
        
        m = m_new
    
    # Normalize
    output = acc / l
    
    return output.to(q.dtype)


def ELSA_swinv2_pytorch_short(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Specialized fast-path for short Swin windows (N ≤ 256).
    Windows are batched across heads to maximise GEMM throughput.
    """
    B, H, N, D = q.shape
    dv = v.shape[-1]

    triton_enabled = (
        q.is_cuda
        and bool(int(os.environ.get("ELSA_SWIN_SHORT_TRITON", "0")))
        and elsa_swinv2_triton is not None
    )
    if triton_enabled:
        try:
            allow_tf32 = bool(int(os.environ.get("ELSA_SWIN_SHORT_TF32", "1")))
            if N <= 64 and D <= 64:
                out_short = elsa_swinv2_triton_short_kernel(
                    q,
                    k,
                    v,
                    logit_scale=logit_scale,
                    relative_position_bias=relative_position_bias,
                    mask=mask,
                    allow_tf32=allow_tf32,
                )
                return out_short
            q_contig = q.to(torch.float32).contiguous()
            k_contig = k.to(torch.float32).contiguous()
            v_contig = v.to(torch.float32).contiguous()

            rel_bias_contig = relative_position_bias
            if relative_position_bias is not None:
                rel_bias_contig = relative_position_bias.to(torch.float32).contiguous()
            mask_contig = mask
            if mask is not None:
                mask_contig = mask.to(torch.float32).contiguous()

            out_triton = elsa_swinv2_triton(
                q_contig,
                k_contig,
                v_contig,
                logit_scale=logit_scale.to(torch.float32).contiguous(),
                relative_position_bias=rel_bias_contig,
                mask=mask_contig,
                use_half_qk=allow_tf32,
            )
            return out_triton.to(q.dtype)
        except Exception:
            triton_enabled = False

    sdpa_enabled = (
        hasattr(F, "scaled_dot_product_attention")
        and q.is_cuda
        and bool(int(os.environ.get("ELSA_SWIN_USE_SDPA_SHORT", "0")))
    )
    if sdpa_enabled:
        try:
            q_flat = q.to(torch.float32).view(B * H, N, D)
            k_flat = k.to(torch.float32).view(B * H, N, D)
            v_flat = v.to(torch.float32).view(B * H, N, dv)

            scale = logit_scale.exp().clamp_min(1e-6)
            scale_flat = scale.view(1, H, 1, 1).expand(B, -1, -1, -1).reshape(B * H, 1, 1)
            sqrt_scale = torch.sqrt(scale_flat)
            q_scaled = q_flat * sqrt_scale
            k_scaled = k_flat * sqrt_scale

            attn_bias = None
            if relative_position_bias is not None:
                bias = relative_position_bias.to(torch.float32)
                bias = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * H, N, N)
                attn_bias = bias
            if mask is not None:
                mask_bias = mask.to(torch.float32)
                if mask_bias.dim() == 4 and mask_bias.size(1) == 1:
                    mask_bias = mask_bias.view(B, 1, N, N).expand(-1, H, -1, -1)
                mask_bias = mask_bias.reshape(B * H, N, N)
                attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias

            sdp_ctx = torch.backends.cuda.sdp_kernel
            with sdp_ctx(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(
                    q_scaled,
                    k_scaled,
                    v_flat,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    is_causal=False,
                )
            return out.view(B, H, N, dv).to(q.dtype)
        except RuntimeError:
            sdpa_enabled = False

    q_fp32 = q.to(torch.float32).contiguous()
    k_fp32 = k.to(torch.float32).contiguous()
    v_fp32 = v.to(torch.float32).contiguous()
    rel_bias_fp32 = relative_position_bias.to(torch.float32) if relative_position_bias is not None else None
    mask_fp32 = mask.to(torch.float32) if mask is not None else None
    logit_scale_fp32 = logit_scale.to(torch.float32)

    compiled_fn = None
    if q_fp32.is_cuda:
        compiled_fn = _short_attention_compiled()
    if compiled_fn is not None:
        try:
            out_compiled = compiled_fn(q_fp32, k_fp32, v_fp32, logit_scale_fp32, rel_bias_fp32, mask_fp32)
            return out_compiled.to(q.dtype)
        except Exception:
            pass

    use_half_qk = (
        q_fp32.is_cuda
        and bool(int(os.environ.get("ELSA_SWIN_SHORT_HALF_QK", "0")))
    )
    if not use_half_qk:
        out = _short_attention_base(q_fp32, k_fp32, v_fp32, logit_scale_fp32, rel_bias_fp32, mask_fp32)
        return out.to(q.dtype)

    q_half = q_fp32.to(torch.float16)
    k_half = k_fp32.to(torch.float16)
    q_half_flat = q_half.view(B * H, N, D)
    k_half_flat = k_half.view(B * H, N, D)
    q_flat = q_fp32.view(B * H, N, D)
    k_flat = k_fp32.view(B * H, N, D)

    scores_main = torch.bmm(q_half_flat, k_half_flat.transpose(1, 2)).to(torch.float32)
    dq = (q_fp32 - q_half.to(torch.float32)).view(B * H, N, D)
    dk = (k_fp32 - k_half.to(torch.float32)).view(B * H, N, D)
    corr1 = torch.bmm(dq, k_flat.transpose(1, 2))
    corr2 = torch.bmm(q_flat, dk.transpose(1, 2))
    scores = scores_main + corr1 + corr2

    scale = logit_scale_fp32.exp().view(1, H, 1, 1).expand(B, -1, -1, -1)
    scores = scores.view(B, H, N, N) * scale
    if rel_bias_fp32 is not None:
        scores = scores + rel_bias_fp32.unsqueeze(0)
    if mask_fp32 is not None:
        temp_mask = mask_fp32
        if temp_mask.dim() == 4 and temp_mask.size(1) == 1:
            temp_mask = temp_mask.view(B, 1, N, N)
        scores = scores + temp_mask

    scores = scores.reshape(B * H, N, N)
    max_scores = scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores - max_scores)
    denom = weights.sum(dim=-1, keepdim=True)
    attn = weights / denom

    out_flat = torch.bmm(attn, v_fp32.view(B * H, N, dv))
    out = out_flat.view(B, H, N, dv)
    return out.to(q.dtype)


import triton
import triton.language as tl
import torch

# ... (你檔案中的其他程式碼) ...

@triton.jit
def elsa_swinv2_kernel(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HALF_QK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask_m_compute = offs_m < N
    mask_d_compute = offs_d < D
    
    logit_scale_val = tl.load(LogitScale + pid_h)
    scale = tl.exp(logit_scale_val.to(tl.float32))
    
    m_i = tl.full([BLOCK_M], -1e10, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m_compute[:, None] & mask_d_compute[None, :], other=0.0)
    if HALF_QK:
        q_tc = q.to(tl.float16)
        q_fp32 = q.to(tl.float32)
    else:
        q = q.to(tl.float32)  # Cast Q to float32 for high-precision scores

    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N
        
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d_compute[:, None], other=0.0)
        if HALF_QK:
            k_tc = k.to(tl.float16)
            k_fp32 = k.to(tl.float32)
            main_scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False)
            dq = q_fp32 - q_tc.to(tl.float32)
            dk = k_fp32 - k_tc.to(tl.float32)
            corr1 = tl.dot(dq, k_fp32, allow_tf32=False)
            corr2 = tl.dot(q_tc.to(tl.float32), dk, allow_tf32=False)
            scores = (main_scores + corr1 + corr2) * scale
        else:
            scores = tl.dot(q, k.to(tl.float32), allow_tf32=False) * scale
        
        if HAS_BIAS:
            bias_ptrs = RelBias + pid_h * stride_rb_h + offs_m[:, None] * stride_rb_n + offs_n_curr[None, :] * stride_rb_m
            bias = tl.load(bias_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=0.0)
            scores += bias.to(tl.float32)
            
        if HAS_MASK:
            mask_ptrs = Mask + pid_b * stride_mask_b + pid_h * stride_mask_h + offs_m[:, None] * stride_mask_n + offs_n_curr[None, :] * stride_mask_m
            mask_vals = tl.load(mask_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=-1e10)
            scores += mask_vals.to(tl.float32)
            
        scores = tl.where(mask_n[None, :], scores, -1e10)
        
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        correction = tl.exp(m_i - m_i_new)
        scores_exp = tl.exp(scores - m_i_new[:, None])
        
        l_i = l_i * correction + tl.sum(scores_exp, axis=1)
        
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_compute[None, :], other=0.0)
        
        # --- THIS IS THE FIX ---
        # Ensure accumulation happens in high precision by up-casting v
        # Accumulate via Tensor Core path with FP16 inputs and FP32 outputs
        scores_comp = scores_exp.to(v.dtype)
        acc = acc * correction[:, None] + tl.dot(scores_comp, v, out_dtype=tl.float32, allow_tf32=True)
        m_i = m_i_new

    output = acc / (l_i[:, None] + 1e-6)
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, output.to(Out.dtype.element_ty), mask=mask_m_compute[:, None] & mask_d_compute[None, :])

def elsa_swinv2_triton(q, k, v, logit_scale, relative_position_bias=None, mask=None, use_half_qk: bool = False):
    B, H, N, D = q.shape
    
    # Ensure inputs are contiguous
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    
    out = torch.empty_like(q)
    
    has_bias = relative_position_bias is not None
    if has_bias:
        relative_position_bias = relative_position_bias.contiguous()
        rel_bias_strides = relative_position_bias.stride()
    else:
        # Use dummy tensor and strides if not provided
        relative_position_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)

    has_mask = mask is not None
    if has_mask:
        # Robustly handle mask broadcasting and get strides
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        mask = mask.expand(B, H, N, N)
        mask_strides = mask.stride()
    else:
        mask = torch.empty(0, device=q.device)
        mask_strides = (0, 0, 0, 0)
    
    if N <= 64:
        BLOCK_M = BLOCK_N = 32
        num_warps = 4
    elif N <= 128:
        BLOCK_M = BLOCK_N = 64
        num_warps = 4
    elif N <= 256:
        BLOCK_M = BLOCK_N = 128
        num_warps = 8
    else:
        BLOCK_M = BLOCK_N = 64
        num_warps = 4

    BLOCK_D = min(128, triton.next_power_of_2(max(D, 16)))
    
    grid = (B, H, triton.cdiv(N, BLOCK_M))

    elsa_swinv2_kernel[grid](
        q, k, v, out,
        logit_scale.contiguous(), relative_position_bias, mask,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        *rel_bias_strides,
        *mask_strides,
        B, H, N, D,
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        HALF_QK=use_half_qk,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=1 if N <= 64 else 2,
    )
    
    return out


def elsa_swinv2_triton_short_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    allow_tf32: bool = True,
) -> torch.Tensor:
    B, H, N, D = q.shape
    dv = v.shape[-1]
    out = torch.empty_like(q)

    q32 = q.to(torch.float32).contiguous()
    k32 = k.to(torch.float32).contiguous()
    v32 = v.to(torch.float32).contiguous()
    logit_scale32 = logit_scale.to(torch.float32).contiguous()

    if relative_position_bias is not None:
        rel_bias = relative_position_bias.to(torch.float32).contiguous()
        rel_bias_strides = rel_bias.stride()
    else:
        rel_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)

    if mask is not None:
        mask_t = mask.to(torch.float32)
        while mask_t.ndim < 4:
            mask_t = mask_t.unsqueeze(0)
        mask_t = mask_t.expand(B, H, N, N).contiguous()
        mask_strides = mask_t.stride()
    else:
        mask_t = torch.empty(0, device=q.device)
        mask_strides = (0, 0, 0, 0)

    grid = (B * H,)
    BLOCK_N = 64 if N > 32 else 32
    BLOCK_D = 64 if D > 32 else 32
    BLOCK_DV = 64 if dv > 32 else 32
    elsa_swinv2_kernel_short_fused[grid](
        q32,
        k32,
        v32,
        out,
        logit_scale32,
        rel_bias,
        mask_t,
        *q32.stride(),
        *k32.stride(),
        *v32.stride(),
        *out.stride(),
        *rel_bias_strides,
        *mask_strides,
        B,
        H,
        N,
        D,
        dv,
        HAS_BIAS=relative_position_bias is not None,
        HAS_MASK=mask is not None,
        ALLOW_TF32=allow_tf32,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
        num_stages=1,
    )
    return out.to(q.dtype)


def elsa_triton_vit_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        )
    dtype = q.dtype
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_mem.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale, is_causal)
    return out.to(dtype)


def elsa_triton_mem_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_fp32.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale)
    return out.to(dtype)


def elsa_triton_baseline_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expose baseline Triton implementation for benchmarks (FP32 path)."""
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    return elsa_triton_vit_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_tensor_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_fp32.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale)
    return out.to(dtype)


def elsa_triton_new_fp32_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    attn_mask = bias
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton.apply(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        scale,
        None,
        is_causal,
    )
    if attn_mask is not None or is_causal:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
    return out.to(dtype)


def elsa_triton_new_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Default FP32 path uses the faster streaming kernel."""
    return elsa_triton_mem_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_new_fp32_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alias of the default FP32 path."""
    return elsa_triton_new_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_new(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
    precision: str = "auto",
) -> torch.Tensor:
    """Precision-aware wrapper for ELSA Triton kernels.

    precision: auto|fp32|tf32|fp16|bf16
    """
    precision = precision.lower()
    orig_dtype = q.dtype
    if precision == "auto":
        target_dtype = orig_dtype
        tf32_override = None
    elif precision == "fp32":
        target_dtype = torch.float32
        tf32_override = False
    elif precision == "tf32":
        target_dtype = torch.float32
        tf32_override = True
    elif precision == "fp16":
        target_dtype = torch.float16
        tf32_override = None
    elif precision == "bf16":
        target_dtype = torch.bfloat16
        tf32_override = None
    else:
        raise ValueError(f"Unsupported precision '{precision}'.")

    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("ELSA_triton requires fp16/bf16/fp32 inputs.")

    q_t = q.to(target_dtype)
    k_t = k.to(target_dtype)
    v_t = v.to(target_dtype)
    attn_mask = bias
    scale = 1.0 / math.sqrt(q_t.shape[-1])

    out_dtype = orig_dtype if precision == "auto" else target_dtype

    with _tf32_context(tf32_override):
        if attn_mask is not None or is_causal:
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=0.0,
            )
        else:
            if target_dtype == torch.float32:
                out = elsa_triton_mem_fp32(q_t, k_t, v_t, is_causal=is_causal, bias=None)
            else:
                out = ELSA_triton.apply(q_t, k_t, v_t, scale, None, is_causal)
    return out.to(out_dtype)


def elsa_triton_new_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="fp16")


def elsa_triton_new_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="bf16")


def elsa_triton_new_tf32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="tf32")
