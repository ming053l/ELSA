import os
from typing import Optional

import torch
import triton
import triton.language as tl


def _maybe_contig_last(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@triton.jit
def elsa_swinv2_kernel_fused(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    RelBiasTable, RelBiasIndex,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_rbt_h, stride_rbt_r,
    stride_rbi_n, stride_rbi_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D,
    NUM_WINDOWS,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HALF_QK: tl.constexpr,
    MASK_IS_COMPACT: tl.constexpr,
    MASK_HAS_HEADS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_CORRECTION: tl.constexpr,
    NORM_QK: tl.constexpr,
    USE_BIAS_TABLE: tl.constexpr,
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
    if NORM_QK:
        q_fp32 = q.to(tl.float32)
        q_norm = tl.sum(q_fp32 * q_fp32, axis=1)
        q_scale = tl.rsqrt(q_norm + 1e-6)
        q_fp32 = q_fp32 * q_scale[:, None]
        if HALF_QK:
            q_tc = q_fp32.to(tl.float16)
        else:
            q = q_fp32
    else:
        if HALF_QK:
            q_tc = q.to(tl.float16)
            if USE_CORRECTION:
                q_fp32 = q.to(tl.float32)
        else:
            q = q.to(tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N

        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d_compute[:, None], other=0.0)
        if NORM_QK:
            k_fp32 = k.to(tl.float32)
            k_norm = tl.sum(k_fp32 * k_fp32, axis=0)
            k_scale = tl.rsqrt(k_norm + 1e-6)
            k_fp32 = k_fp32 * k_scale[None, :]
            if HALF_QK:
                k_tc = k_fp32.to(tl.float16)
                if USE_CORRECTION:
                    main_scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False)
                    dq = q_fp32 - q_tc.to(tl.float32)
                    dk = k_fp32 - k_tc.to(tl.float32)
                    corr1 = tl.dot(dq, k_fp32, allow_tf32=False)
                    corr2 = tl.dot(q_tc.to(tl.float32), dk, allow_tf32=False)
                    scores = (main_scores + corr1 + corr2) * scale
                else:
                    scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False) * scale
            else:
                scores = tl.dot(q, k_fp32, allow_tf32=ALLOW_TF32) * scale
        else:
            if HALF_QK:
                k_tc = k.to(tl.float16)
                if USE_CORRECTION:
                    k_fp32 = k.to(tl.float32)
                    main_scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False)
                    dq = q_fp32 - q_tc.to(tl.float32)
                    dk = k_fp32 - k_tc.to(tl.float32)
                    corr1 = tl.dot(dq, k_fp32, allow_tf32=False)
                    corr2 = tl.dot(q_tc.to(tl.float32), dk, allow_tf32=False)
                    scores = (main_scores + corr1 + corr2) * scale
                else:
                    scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False) * scale
            else:
                scores = tl.dot(q, k.to(tl.float32), allow_tf32=ALLOW_TF32) * scale

        if HAS_BIAS:
            if USE_BIAS_TABLE:
                idx_ptrs = RelBiasIndex + offs_m[:, None] * stride_rbi_n + offs_n_curr[None, :] * stride_rbi_m
                idx = tl.load(idx_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=0)
                bias_ptrs = RelBiasTable + pid_h * stride_rbt_h + idx * stride_rbt_r
                bias = tl.load(bias_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=0.0)
                scores += bias.to(tl.float32)
            else:
                bias_ptrs = RelBias + pid_h * stride_rb_h + offs_m[:, None] * stride_rb_n + offs_n_curr[None, :] * stride_rb_m
                bias = tl.load(bias_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=0.0)
                scores += bias.to(tl.float32)

        if HAS_MASK:
            mask_b = pid_b
            if MASK_IS_COMPACT:
                mask_b = pid_b % NUM_WINDOWS
            mask_h = pid_h if MASK_HAS_HEADS else 0
            mask_ptrs = Mask + mask_b * stride_mask_b + mask_h * stride_mask_h + offs_m[:, None] * stride_mask_n + offs_n_curr[None, :] * stride_mask_m
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

        scores_comp = scores_exp.to(v.dtype)
        acc = acc * correction[:, None] + tl.dot(scores_comp, v, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)
        m_i = m_i_new

    output = acc / (l_i[:, None] + 1e-6)
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, output.to(Out.dtype.element_ty), mask=mask_m_compute[:, None] & mask_d_compute[None, :])


@triton.jit
def elsa_swinv2_proj_kernel(
    X, W, Bias, Out,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    M, K, N,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask_k = k < K
        x_ptrs = X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk
        w_ptrs = W + offs_n[None, :] * stride_wn + k[:, None] * stride_wk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, allow_tf32=ALLOW_TF32)

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def elsa_swinv2_triton_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    relative_position_bias_table: Optional[torch.Tensor] = None,
    relative_position_index: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    use_half_qk: bool = False,
    normalize_qk: bool = False,
    use_bias_table: bool = False,
    out_layout: str = "HND",
) -> torch.Tensor:
    B, H, N, D = q.shape

    q = _maybe_contig_last(q)
    k = _maybe_contig_last(k)
    v = _maybe_contig_last(v)

    layout = out_layout.upper()
    if layout not in ("HND", "NH"):
        raise ValueError(f"Unsupported out_layout '{out_layout}'. Use 'HND' or 'NH'.")
    if layout == "NH":
        out = torch.empty((B, N, H, D), device=q.device, dtype=q.dtype)
        stride_ob = out.stride(0)
        stride_on = out.stride(1)
        stride_oh = out.stride(2)
        stride_od = out.stride(3)
    else:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        stride_ob, stride_oh, stride_on, stride_od = out.stride()

    has_bias = relative_position_bias is not None or relative_position_bias_table is not None
    if has_bias:
        if use_bias_table:
            if relative_position_bias_table is None or relative_position_index is None:
                raise RuntimeError("Bias table path requires bias table and index.")
            relative_position_bias_table = _maybe_contig_last(relative_position_bias_table)
            relative_position_index = _maybe_contig_last(relative_position_index)
            rel_bias_strides = (0, 0, 0)
            rel_bias_table_strides = relative_position_bias_table.stride()
            rel_bias_index_strides = relative_position_index.stride()
        else:
            relative_position_bias = _maybe_contig_last(relative_position_bias)
            rel_bias_strides = relative_position_bias.stride()
            rel_bias_table_strides = (0, 0)
            rel_bias_index_strides = (0, 0)
    else:
        relative_position_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)
        relative_position_bias_table = torch.empty(0, device=q.device)
        relative_position_index = torch.empty(0, device=q.device, dtype=torch.int32)
        rel_bias_table_strides = (0, 0)
        rel_bias_index_strides = (0, 0)

    has_mask = mask is not None
    mask_is_compact = False
    mask_has_heads = False
    num_windows = 0
    if has_mask:
        if mask.ndim == 3:
            mask_has_heads = False
        elif mask.ndim == 4:
            mask_has_heads = True
        else:
            raise RuntimeError(f"Mask must be 3D or 4D, got {mask.ndim}D.")
        mask_is_compact = mask.size(0) != B
        num_windows = int(mask.size(0))
        mask = _maybe_contig_last(mask)
        if mask_has_heads:
            stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m = mask.stride()
        else:
            stride_mask_b, stride_mask_n, stride_mask_m = mask.stride()
            stride_mask_h = 0
    else:
        mask = torch.empty(0, device=q.device)
        stride_mask_b = stride_mask_h = stride_mask_n = stride_mask_m = 0

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

    allow_tf32 = (
        q.dtype == torch.float32
        and torch.backends.cuda.matmul.allow_tf32
        and not use_half_qk
        and bool(int(os.environ.get("ELSA_SWIN_USE_TF32", "1")))
    )
    use_correction = use_half_qk and q.dtype == torch.float32
    half_qk = use_half_qk or q.dtype == torch.float16

    grid = (B, H, triton.cdiv(N, BLOCK_M))

    stride_qb, stride_qh, stride_qn, stride_qd = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = v.stride()

    elsa_swinv2_kernel_fused[grid](
        q, k, v, out,
        logit_scale.contiguous(), relative_position_bias, mask,
        relative_position_bias_table, relative_position_index,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        *rel_bias_strides,
        *rel_bias_table_strides,
        *rel_bias_index_strides,
        stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
        B, H, N, D,
        num_windows,
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        HALF_QK=half_qk,
        MASK_IS_COMPACT=mask_is_compact,
        MASK_HAS_HEADS=mask_has_heads,
        ALLOW_TF32=allow_tf32,
        USE_CORRECTION=use_correction,
        NORM_QK=normalize_qk,
        USE_BIAS_TABLE=use_bias_table,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=1 if N <= 64 else 2,
    )
    return out


def elsa_swinv2_triton_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # x: [B,N,H,D] or [B,N,C], weight: [C_out, C_in]
    if x.ndim == 4:
        B, N, H, D = x.shape
        x2d = x.reshape(B * N, H * D)
        out_shape = (B, N, weight.shape[0])
    elif x.ndim == 3:
        B, N, C = x.shape
        x2d = x.reshape(B * N, C)
        out_shape = (B, N, weight.shape[0])
    else:
        raise RuntimeError(f"Unexpected proj input rank: {x.ndim}")

    x2d = _maybe_contig_last(x2d)
    w = _maybe_contig_last(weight)
    out2d = torch.empty((x2d.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
    else:
        bias = torch.empty(0, device=x.device, dtype=x.dtype)

    M, K = x2d.shape
    Nout = w.shape[0]

    if Nout <= 128:
        block_n = 64
    else:
        block_n = 128
    block_m = 64
    block_k = 32 if K <= 128 else 64
    num_warps = 4 if block_n <= 64 else 8
    allow_tf32 = x.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32

    grid = (triton.cdiv(M, block_m), triton.cdiv(Nout, block_n))
    elsa_swinv2_proj_kernel[grid](
        x2d, w, bias, out2d,
        *x2d.stride(),
        *w.stride(),
        *out2d.stride(),
        M, K, Nout,
        HAS_BIAS=has_bias,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return out2d.view(*out_shape)
