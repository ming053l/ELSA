"""Full-model ELSA kernels wrapping the locked sic_triton baseline."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[3]
_STABLE_SIC_PATH = (
    _ROOT
    / "timm"
    / "elsa_cuda"
    / "versions"
    / "original_20251021_195305"
    / "elsa_cuda"
    / "versions"
    / "original_20251021_195305"
    / "sic_triton.py"
)
_STABLE_MODULE = None
_NEW_SWIN_KERNEL = None
_TUNED_SWIN_KERNEL = None

_SWIN_TUNED_CFG = {
    ("fp16", 64): dict(block_m=64, block_n=64, block_d=32, num_warps=4, num_stages=1),
    ("fp16", 256): dict(block_m=64, block_n=64, block_d=32, num_warps=4, num_stages=1),
    ("fp32", 64): dict(block_m=64, block_n=64, block_d=32, num_warps=4, num_stages=1),
    ("fp32", 256): dict(block_m=64, block_n=64, block_d=32, num_warps=4, num_stages=2),
}


def _pick_swin_cfg(n: int, dtype: torch.dtype) -> Optional[dict]:
    key = "fp16" if dtype == torch.float16 else "fp32"
    if n <= 64:
        return _SWIN_TUNED_CFG.get((key, 64))
    if n <= 256:
        return _SWIN_TUNED_CFG.get((key, 256))
    return None


def _load_stable():
    global _STABLE_MODULE
    if _STABLE_MODULE is not None:
        return _STABLE_MODULE
    if not _STABLE_SIC_PATH.is_file():
        raise FileNotFoundError(f"Stable sic_triton path not found: {_STABLE_SIC_PATH}")
    spec = importlib.util.spec_from_file_location("_elsa_full_stable", _STABLE_SIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stable module at {_STABLE_SIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STABLE_MODULE = module
    return module


def _load_new_swin_kernel():
    global _NEW_SWIN_KERNEL
    if _NEW_SWIN_KERNEL is not None:
        return _NEW_SWIN_KERNEL
    try:
        from .elsa_triton import elsa_swinv2_triton
    except Exception:
        elsa_swinv2_triton = None
    _NEW_SWIN_KERNEL = elsa_swinv2_triton
    return _NEW_SWIN_KERNEL


def _swinv2_triton_tuned(
    mod,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    B, H, N, D = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)

    has_bias = relative_position_bias is not None
    if has_bias:
        relative_position_bias = relative_position_bias.contiguous()
        rel_bias_strides = relative_position_bias.stride()
    else:
        relative_position_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)

    has_mask = mask is not None
    if has_mask:
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        mask = mask.expand(B, H, N, N).contiguous()
        mask_strides = mask.stride()
    else:
        mask = torch.empty(0, device=q.device)
        mask_strides = (0, 0, 0, 0)

    cfg = _pick_swin_cfg(N, q.dtype)
    if cfg is None:
        return mod.can_swinv2_triton(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias if has_bias else None,
            mask=mask if has_mask else None,
        )
    block_m = cfg["block_m"]
    block_n = cfg["block_n"]
    block_d = cfg["block_d"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]
    grid = (B, H, mod.triton.cdiv(N, block_m))

    mod.can_swinv2_kernel[grid](
        q,
        k,
        v,
        out,
        logit_scale.contiguous(),
        relative_position_bias,
        mask,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *rel_bias_strides,
        *mask_strides,
        B,
        H,
        N,
        D,
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def elsa_full_triton_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    qk_norm_weights=None,
    is_causal: bool = False,
) -> torch.Tensor:
    if qk_norm_weights is not None:
        raise RuntimeError("ELSA full-kernel does not support qk_norm.")
    if not q.is_cuda or is_causal:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)
    if q.dtype == torch.bfloat16:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
    # Full-kernel CAN_triton path in stable module does not implement backward.
    # Route training to differentiable Triton autograd path.
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        from .elsa_triton import ELSA_triton
        return ELSA_triton.apply(q, k, v, scale, None, is_causal)
    mod = _load_stable()
    return mod.CAN_triton.apply(q, k, v, scale, None, is_causal)


def elsa_full_triton_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    if not q.is_cuda or is_causal:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)
    if q.dtype != torch.float32:
        raise RuntimeError("ELSA full-kernel FP32 expects float32 inputs.")
    # Full-kernel can_triton_new_fp32 currently uses CAN_triton.apply, whose backward
    # is not implemented in stable module. Use differentiable fp32 Triton path in training.
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        from .elsa_triton import ELSA_triton_fp32
        return ELSA_triton_fp32.apply(q, k, v, scale)
    mod = _load_stable()
    return mod.can_triton_new_fp32(q, k, v, is_causal=is_causal, bias=None)


def elsa_full_triton_turbo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    return elsa_full_triton_fp32(q, k, v, scale, is_causal=is_causal)


def elsa_swinv2_triton_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    use_half_qk: bool = False,
) -> torch.Tensor:
    del use_half_qk
    mod = _load_stable()
    if not q.is_cuda or torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return mod.CAN_swinv2_pytorch(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias,
            mask=mask,
        )
    if q.dtype == torch.float16:
        impl = os.environ.get("ELSA_SWIN_FULL_FP16_IMPL", "stable").lower()
    else:
        impl = os.environ.get("ELSA_SWIN_FULL_IMPL", "stable").lower()
    if impl == "stable":
        return mod.can_swinv2_triton(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias,
            mask=mask,
        )
    if impl == "tuned":
        try:
            return _swinv2_triton_tuned(
                mod,
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
            )
        except Exception:
            impl = "new"
    if impl == "new":
        new_kernel = _load_new_swin_kernel()
        if new_kernel is not None:
            allow_half = bool(int(os.environ.get("ELSA_SWIN_FULL_HALF_QK", "1")))
            if q.dtype == torch.float16:
                allow_half = True
            elif not torch.backends.cuda.matmul.allow_tf32:
                allow_half = False
            return new_kernel(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
                use_half_qk=allow_half,
            )
    return mod.can_swinv2_triton(
        q,
        k,
        v,
        logit_scale=logit_scale,
        relative_position_bias=relative_position_bias,
        mask=mask,
    )
