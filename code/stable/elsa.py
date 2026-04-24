"""DeiT - Data-efficient Image Transformers.

This module hosts the ELSA (Exact Linear Scan Attention) variants that replace the
previous attention stack with Triton-first kernels and PyTorch fallbacks.
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import inspect
import importlib.util
import math
import os
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn, trunc_normal_, checkpoint_seq
from ._builder import build_model_with_cfg
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = [
    'ElsaAttention',
    'ElsaViT',
    'ElsaDistilled',
    'set_default_elsa_backend',
    'get_default_elsa_backend',
]

_VALID_BACKENDS = {
    "triton",
    "triton_fp32",
    "triton_fp32_train",
    "strict_core_ref",
    "triton_mem",
    "triton_full",
    "triton_full_fp32",
    "triton_full_turbo",
    "pytorch",
    "sdpa_math",
    "sdpa_mem",
    "sdpa_flash",
    "swin_train_kernel",
    "swin_train_fused",
    "auto",
}
_DEFAULT_BACKEND = "triton"
_UNSTABLE_BACKENDS = {
    "triton_mem",
    "triton_full",
    "triton_full_fp32",
    "triton_full_turbo",
    "swin_train_fused",
    "train_fused",
}


def _strict_ref_block_n_for_tensor(x: torch.Tensor, *, training: bool) -> int:
    override = os.environ.get("ELSA_STRICT_REF_BLOCK_N")
    if override is not None:
        try:
            return max(16, int(override))
        except ValueError:
            return 512 if not training else 256
    try:
        from timm.models.elsa_strict_ref import default_strict_ref_block_n
        return int(default_strict_ref_block_n(int(x.shape[-2]), training=training))
    except Exception:
        if training:
            return 256
        return 512 if int(x.shape[-2]) <= 1024 else 2048


def set_default_elsa_backend(backend: str) -> None:
    """Set global default backend for all subsequent ELSA attention modules."""
    backend = backend.lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported ELSA backend '{backend}'. Valid options: {sorted(_VALID_BACKENDS)}")
    clean_backend = _canonical_backend_for_clean_mode(backend)
    if clean_backend != backend:
        warnings.warn(
            f"backend '{backend}' is disabled in clean mode; using '{clean_backend}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        backend = clean_backend
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = backend


def get_default_elsa_backend() -> str:
    """Return current global default backend."""
    return _DEFAULT_BACKEND


def _allow_unstable_paths() -> bool:
    return os.environ.get("ELSA_ALLOW_UNSTABLE_PATHS", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
        "force",
    )


def _canonical_backend_for_clean_mode(backend: str) -> str:
    """Map unstable/legacy routes to clean canonical routes."""
    if _allow_unstable_paths():
        return backend
    if backend in {"triton_mem", "triton_full", "triton_full_fp32", "triton_full_turbo"}:
        return "triton"
    if backend in {"swin_train_fused", "train_fused"}:
        return "swin_train_kernel"
    return backend

try:
    from .elsa_triton import (
        ELSA_pytorch,
        ELSA_swin_pytorch,
        ELSA_swinv2_pytorch,
        ELSA_swinv2_pytorch_short,
        ELSA_triton,
        ELSA_triton_fp32,
        ELSA_triton_fp32_train,
        ELSA_triton_mem,
        elsa_swinv2_triton,
    )
    _ELSATRITON_AVAILABLE = True
except Exception as err:  # pragma: no cover - import guard
    _ELSATRITON_AVAILABLE = False
    _ELSATRITON_IMPORT_ERROR = err
    ELSA_triton = None
    ELSA_triton_fp32 = None
    ELSA_triton_fp32_train = None
    ELSA_triton_mem = None
    elsa_swinv2_triton = None

    @torch.jit.script
    def _softmax_row(scores: torch.Tensor) -> torch.Tensor:
        max_v, _ = scores.max(-1, keepdim=True)
        exp_v = (scores - max_v).exp()
        return exp_v / exp_v.sum(-1, keepdim=True)

    def ELSA_pytorch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Numerically stable PyTorch fallback matching the Triton two-pass kernel."""
        B, H, N, D = q.shape
        dtype = q.dtype
        device = q.device

        m = torch.full((B, H, N, 1), -torch.inf, dtype=torch.float32, device=device)
        l = torch.zeros((B, H, N, 1), dtype=torch.float32, device=device)
        o = torch.zeros((B, H, N, D), dtype=torch.float32, device=device)

        q_scaled = q * scale
        chunk_size = 256 if N > 1024 else 128

        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)

            scores = torch.matmul(q_scaled, k[:, :, i:j].transpose(-2, -1)).float()

            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(N, j - i, device=device, dtype=torch.bool), diagonal=i - N + 1
                )
                scores = scores.masked_fill(~causal_mask, float('-inf'))

            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            alpha = torch.exp(m - m_new)
            beta = torch.exp(scores - m_new)

            if dropout_p > 0 and torch.jit.is_tracing() is False:
                beta = F.dropout(beta, p=dropout_p, training=True)

            l = l * alpha + beta.sum(dim=-1, keepdim=True)
            o = o * alpha + torch.matmul(beta.to(dtype), v[:, :, i:j])
            m = m_new

        return (o / l.clamp_min(1e-6)).to(dtype)

    def ELSA_swin_pytorch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_scale: torch.Tensor,
        rel_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)).to(torch.float32)
        scores *= log_scale
        if rel_bias is not None:
            scores = scores + rel_bias.unsqueeze(0)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = _softmax_row(scores)
        out = torch.matmul(attn, v.to(torch.float32))
        return out.to(v.dtype)

    def ELSA_swinv2_pytorch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logit_scale: torch.Tensor,
        relative_position_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        B, H, N, D = q.shape
        m = torch.full((B, H, N, 1), -1e10, dtype=torch.float32, device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros(B, H, N, D, dtype=torch.float32, device=q.device)
        scale = logit_scale.exp()

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            k_chunk = k[:, :, start_idx:end_idx]
            v_chunk = v[:, :, start_idx:end_idx]

            scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
            if relative_position_bias is not None:
                scores = scores + relative_position_bias[:, :, start_idx:end_idx].unsqueeze(0)
            if mask is not None:
                scores = scores + mask[:, :, :, start_idx:end_idx]

            scores_max = scores.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m, scores_max)
            correction = torch.exp(m - m_new)
            scores_exp = torch.exp(scores - m_new)

            l = l * correction + scores_exp.sum(dim=-1, keepdim=True)
            acc = acc * correction + torch.matmul(scores_exp, v_chunk.to(torch.float32))
            m = m_new

        return (acc / l).to(q.dtype)

    def ELSA_swinv2_pytorch_short(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logit_scale: torch.Tensor,
        relative_position_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return ELSA_swinv2_pytorch(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias,
            mask=mask,
        )


def ELSA_swinv2_train_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Swin training kernel: SDPA path specialized for window-attention tensors."""
    B, H, N, D = q.shape
    out_dtype = q.dtype
    fp16_accum_fp32 = bool(int(os.environ.get("ELSA_SWIN_TRAIN_FP16_ACCUM_FP32", "0")))
    bf16_accum_fp32 = bool(int(os.environ.get("ELSA_SWIN_TRAIN_BF16_ACCUM_FP32", "0")))
    if q.dtype == torch.float16 and not fp16_accum_fp32:
        compute_dtype = torch.float16
    elif q.dtype == torch.bfloat16 and not bf16_accum_fp32:
        compute_dtype = torch.bfloat16
    elif q.dtype in (torch.float16, torch.bfloat16, torch.float32):
        compute_dtype = torch.float32
    else:
        compute_dtype = q.dtype
    # Default to no forced contiguous copy in training path.
    # For current A100 stack, this removes avoidable q/k/v copy overhead on
    # common Swin window batches while preserving exactness.
    # Default to auto policy:
    # - fp32: force contiguous only on larger window-batches (B >= 96 by default)
    # - low precision: default to no forced contiguous copies; an explicit
    #   low-precision threshold can still be provided via env for targeted sweeps.
    # This keeps fp32 base-train regressions closed while preserving the faster
    # fp16/bf16 Swin train path on current A100 runs.
    contig_env = os.environ.get("ELSA_SWIN_TRAIN_CONTIG", "auto").strip().lower()
    if contig_env in ("1", "true", "on", "yes"):
        force_contig = True
    elif contig_env in ("0", "false", "off", "no"):
        force_contig = False
    elif contig_env in ("auto", "a", ""):
        if q.dtype == torch.float32:
            env_key = "ELSA_SWIN_TRAIN_CONTIG_AUTO_B_THRESHOLD_FP32"
            try:
                auto_b_default = int(os.environ.get("ELSA_SWIN_TRAIN_CONTIG_AUTO_B_THRESHOLD", "96"))
            except ValueError:
                auto_b_default = 96
            try:
                auto_b_threshold = int(os.environ.get(env_key, str(auto_b_default)))
            except ValueError:
                auto_b_threshold = auto_b_default
            force_contig = bool(B >= max(1, auto_b_threshold))
        else:
            lowp_raw = os.environ.get("ELSA_SWIN_TRAIN_CONTIG_AUTO_B_THRESHOLD_LOWP")
            if lowp_raw is None:
                force_contig = False
            else:
                try:
                    auto_b_threshold = int(lowp_raw)
                except ValueError:
                    auto_b_threshold = 1 << 30
                force_contig = bool(B >= max(1, auto_b_threshold))
    else:
        force_contig = bool(int(contig_env))
    # Default to no expansion in train path:
    # keep broadcast mask [B,1,N,N] to avoid [B,H,N,N] materialization overhead.
    # (can be overridden by ELSA_SWIN_TRAIN_EXPAND_HEAD_MASK=1/auto for low-precision sweeps)
    expand_env = os.environ.get("ELSA_SWIN_TRAIN_EXPAND_HEAD_MASK", "0").strip().lower()
    if expand_env in ("1", "true", "on", "yes"):
        expand_head_mask = True
    elif expand_env in ("0", "false", "off", "no"):
        expand_head_mask = False
    elif expand_env in ("auto", "a", ""):
        # Expand [B,1,N,N] -> [B,H,N,N] only for large window-batch shapes where
        # backend broadcast overhead dominates.
        auto_b = int(os.environ.get("ELSA_SWIN_TRAIN_EXPAND_HEAD_MASK_AUTO_B_THRESHOLD", "512"))
        lowp = q.dtype in (torch.float16, torch.bfloat16)
        expand_head_mask = bool(lowp and B >= auto_b)
    else:
        expand_head_mask = bool(int(expand_env))

    # Keep stride layout when possible to avoid per-step tensor copies in training.
    qf = q if q.dtype == compute_dtype else q.to(compute_dtype)
    kf = k if k.dtype == compute_dtype else k.to(compute_dtype)
    vf = v if v.dtype == compute_dtype else v.to(compute_dtype)
    if force_contig:
        qf = qf.contiguous()
        kf = kf.contiguous()
        vf = vf.contiguous()

    # Compute scale in fp32 for stability, then cast to compute dtype.
    # Default to q-only scaling (equivalent to q*k with split sqrt scale) to
    # reduce one elementwise multiply and one temporary tensor on train path.
    scale = logit_scale.to(torch.float32).exp().clamp_min(1e-6).view(1, H, 1, 1)
    scale_mode = os.environ.get("ELSA_SWIN_TRAIN_SCALE_MODE", "q").strip().lower()
    if scale_mode in ("qk", "sqrt", "split"):
        sqrt_scale = torch.sqrt(scale).to(compute_dtype)
        q_scaled = qf * sqrt_scale
        k_scaled = kf * sqrt_scale
    else:
        q_scaled = qf * scale.to(compute_dtype)
        k_scaled = kf

    attn_bias = None
    if relative_position_bias is not None:
        # [1, H, N, N], leave batch broadcast to SDPA.
        rel_bias = relative_position_bias.to(compute_dtype)
        if force_contig and rel_bias.stride(-1) != 1:
            rel_bias = rel_bias.contiguous()
        rel_bias = rel_bias.unsqueeze(0)
        attn_bias = rel_bias

    if mask is not None:
        mask_bias = mask.to(compute_dtype)
        if mask_bias.dim() == 4 and mask_bias.size(1) in (1, H):
            pass
        elif mask_bias.dim() == 3 and mask_bias.size(0) == B:
            mask_bias = mask_bias.unsqueeze(1)  # [B, 1, N, N]
        elif mask_bias.dim() == 3 and mask_bias.size(0) > 0 and (B % mask_bias.size(0) == 0):
            num_win = mask_bias.size(0)
            mask_bias = (
                mask_bias.view(1, num_win, N, N)
                .expand(B // num_win, num_win, N, N)
                .reshape(B, N, N)
                .unsqueeze(1)  # [B, 1, N, N]
            )
        else:
            raise RuntimeError(
                f"Unsupported Swin mask shape {tuple(mask_bias.shape)} for B={B}, H={H}, N={N}."
            )
        if expand_head_mask and mask_bias.size(1) == 1:
            mask_bias = mask_bias.expand(B, H, N, N)
        if force_contig and mask_bias.stride(-1) != 1:
            mask_bias = mask_bias.contiguous()
        attn_bias = mask_bias if attn_bias is None else (attn_bias + mask_bias)

    sdp_mode = os.environ.get("ELSA_SWIN_TRAIN_SDPA_BACKEND", "mem").lower()
    if q_scaled.is_cuda:
        # Keep math enabled as fallback so train path never hard-fails when a fused kernel
        # is unavailable for a specific mask/layout combination.
        enable_math = sdp_mode in ("math", "m", "mem", "memory", "auto")
        enable_mem = sdp_mode in ("mem", "memory", "auto")
        enable_flash = sdp_mode in ("flash", "fa")
        # fp16/bf16 tuning: default to mem-only to avoid accidentally falling back
        # to math kernels on shapes where mem-efficient path is available.
        if sdp_mode in ("mem", "memory") and q_scaled.dtype in (torch.float16, torch.bfloat16):
            keep_math_fallback = bool(int(os.environ.get("ELSA_SWIN_TRAIN_MEM_KEEP_MATH_FALLBACK", "0")))
            enable_math = keep_math_fallback
        with torch.backends.cuda.sdp_kernel(
            enable_math=enable_math,
            enable_mem_efficient=enable_mem,
            enable_flash=enable_flash,
        ):
            out = F.scaled_dot_product_attention(
                q_scaled,
                k_scaled,
                vf,
                attn_mask=attn_bias,
                dropout_p=0.0,
                is_causal=False,
            )
    else:
        out = F.scaled_dot_product_attention(
            q_scaled,
            k_scaled,
            vf,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )

    return out.to(out_dtype)


def ELSA_swinv2_train_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Backward-compatible alias for the Swin training kernel."""
    return ELSA_swinv2_train_kernel(
        q=q,
        k=k,
        v=v,
        logit_scale=logit_scale,
        relative_position_bias=relative_position_bias,
        mask=mask,
    )

try:
    from .elsa_strict_ref import elsa_strict_reference
except Exception as err:  # pragma: no cover - import guard
    elsa_strict_reference = None
    _ELSASTRICT_IMPORT_ERROR = err

try:
    _STABLE_SIC_PATH = (
        Path(__file__).resolve().parents[3]
        / "timm"
        / "elsa_cuda"
        / "versions"
        / "original_20251021_195305"
        / "elsa_cuda"
        / "versions"
        / "original_20251021_195305"
        / "sic_triton.py"
    )
    _spec = importlib.util.spec_from_file_location("_elsa_stable_sic_for_model", _STABLE_SIC_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Unable to load strict-core stable module from {_STABLE_SIC_PATH}")
    _stable_sic_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _stable_sic_mod
    _spec.loader.exec_module(_stable_sic_mod)
    can_triton_strict_core_fp32 = _stable_sic_mod.can_triton_strict_core_fp32
    can_triton_strict_core_fp16 = getattr(_stable_sic_mod, "can_triton_strict_core_fp16", None)
except Exception as err:  # pragma: no cover - import guard
    can_triton_strict_core_fp32 = None
    can_triton_strict_core_fp16 = None
    _ELSASTRICTSIC_IMPORT_ERROR = err

try:
    from .elsa_triton_full import (
        elsa_full_triton_fp16,
        elsa_full_triton_fp32,
        elsa_full_triton_turbo,
        elsa_swinv2_triton_full,
    )
    _ELSAFULL_AVAILABLE = True
except Exception as err:  # pragma: no cover - import guard
    _ELSAFULL_AVAILABLE = False
    _ELSAFULL_IMPORT_ERROR = err
    elsa_full_triton_fp16 = None
    elsa_full_triton_fp32 = None
    elsa_full_triton_turbo = None
    elsa_swinv2_triton_full = None

# --------------------- 0. utility ------------------------------------------------

def _filter_kwargs(ctor, kwargs):
    sig = inspect.signature(ctor).parameters
    return {k: v for k, v in kwargs.items() if k in sig}

def Norm2d(c, groups=32):
    g = max(1, math.gcd(c, groups))
    return nn.GroupNorm(g, c)

# --------------------- 1. Window Attention (精確 MHSA) ---------------------------

def _reshape(x, h):
    B, C, H, W = x.shape
    N, d = H * W, C // h
    return x.reshape(B, h, d, N)


    
class ElsaAttention(nn.Module):
    """Integral-CNN attention (exact MHSA) drop-in for ViT/DeiT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        proj_bias: bool = True,
        triton: bool = True,
        triton_matmul: bool = False,
        qkv_bias: bool = False,
        backend: Optional[str] = None,
    ) -> None:
        super().__init__()
        del triton_matmul  # Unused but kept for backwards compatibility.

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_triton = triton and _ELSATRITON_AVAILABLE
        resolved_backend = backend.lower() if backend else get_default_elsa_backend()
        if resolved_backend in {"swin_train_kernel", "swin_train_fused", "train_fused"}:
            warnings.warn(
                f"backend '{resolved_backend}' is Swin-specific; using 'triton' for ViT attention.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_backend = "triton"
        clean_resolved = _canonical_backend_for_clean_mode(resolved_backend)
        if clean_resolved != resolved_backend:
            warnings.warn(
                f"backend '{resolved_backend}' is disabled in clean mode; using '{clean_resolved}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_backend = clean_resolved
        if resolved_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unsupported ELSA backend '{resolved_backend}'. Valid options: {sorted(_VALID_BACKENDS)}"
            )
        self.backend_preference = resolved_backend
        self._warned_backends: Set[str] = set()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self._fp16_short_compiled_train = None
        self._fp16_short_compiled_eval = None
        self._fp16_short_compile_disabled = False
        self._strict_short_compiled_train = None
        self._strict_short_compiled_eval = None
        self._strict_short_compile_disabled = False
        self._dispatch_mark_pending = False
        self._dispatch_runtime_reset_pending = False

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _tensor_core_ready(tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda or not torch.cuda.is_available():
            return False
        try:
            prop = torch.cuda.get_device_properties(tensor.device)
        except (RuntimeError, AssertionError):
            return False
        return getattr(prop, "major", 0) >= 7

    def _candidate_backends(self, tensor: torch.Tensor) -> List[str]:
        preference = self.backend_preference
        strict_no_fallback = bool(int(os.environ.get("ELSA_STRICT_BENCH_NO_FALLBACK", "0")))
        baseline_backends = {"pytorch", "sdpa_math", "sdpa_mem", "sdpa_flash"}
        unstable_backends = _UNSTABLE_BACKENDS
        effective_dtype = tensor.dtype
        if tensor.is_cuda:
            try:
                if torch.is_autocast_enabled("cuda"):
                    effective_dtype = torch.get_autocast_gpu_dtype()
            except TypeError:
                if torch.is_autocast_enabled():
                    try:
                        effective_dtype = torch.get_autocast_gpu_dtype()
                    except Exception:
                        pass
            except Exception:
                pass

        # Keep explicit baseline routes intact for comparisons.
        if preference in baseline_backends:
            order = [preference, "pytorch"] if preference != "pytorch" else ["pytorch"]
        else:
            # Clean default routing:
            # - fp32 + training => triton_fp32_train
            # - fp32 + inference => triton_fp32
            # - fp16/bf16 => triton
            # Non-clean historical routes are redirected unless explicitly forced.
            if preference == "auto":
                preference = "triton"
            if preference in unstable_backends and not _allow_unstable_paths():
                warn_key = f"clean_redirect_{preference}"
                if warn_key not in self._warned_backends:
                    self._warned_backends.add(warn_key)
                    warnings.warn(
                        f"{preference} is disabled in clean mode; routing to canonical ELSA backend.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                preference = "triton"

            if effective_dtype == torch.float32:
                clean = "triton_fp32_train" if (self.training and torch.is_grad_enabled()) else "triton_fp32"
            else:
                clean = "triton"

            if preference not in {"triton", "triton_fp32", "triton_fp32_train", "strict_core_ref"} and not _allow_unstable_paths():
                warn_key = f"clean_redirect_noncanonical_{preference}"
                if warn_key not in self._warned_backends:
                    self._warned_backends.add(warn_key)
                    warnings.warn(
                        f"backend '{preference}' is not a clean canonical route; using '{clean}'.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                preference = clean
            elif preference in {"triton", "triton_fp32", "triton_fp32_train"}:
                # Respect explicit clean route only if it matches runtime intent.
                if preference != clean and not _allow_unstable_paths():
                    warn_key = f"clean_redirect_mismatch_{preference}_to_{clean}"
                    if warn_key not in self._warned_backends:
                        self._warned_backends.add(warn_key)
                        warnings.warn(
                            f"backend '{preference}' mismatches runtime mode; using '{clean}'.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                    preference = clean

            order = {
                "triton": ["triton", "pytorch"],
                "triton_fp32": ["triton_fp32", "pytorch"],
                "triton_fp32_train": ["triton_fp32_train", "triton_fp32", "pytorch"],
                "strict_core_ref": ["strict_core_ref", "pytorch"],
            }.get(preference, [clean, "pytorch"])
            if preference == "strict_core_ref" and strict_no_fallback:
                order = ["strict_core_ref"]

        if not self.enable_triton or not tensor.is_cuda:
            order = [backend for backend in order if not backend.startswith("triton")]
        if not order:
            order = ["pytorch"]
        return order

    def _warn_backend_failure(self, backend: str, err: RuntimeError) -> None:
        if backend in self._warned_backends:
            return
        self._warned_backends.add(backend)
        warnings.warn(
            f"ELSA attention backend '{backend}' failed ({err}). Falling back to the next candidate.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _qk_norm_weights(self):
        if not self.qk_norm:
            return None
        return (
            self.q_norm.weight.data,
            self.q_norm.bias.data if self.q_norm.bias is not None else torch.zeros_like(self.q_norm.weight.data),
            self.k_norm.weight.data,
            self.k_norm.bias.data if self.k_norm.bias is not None else torch.zeros_like(self.k_norm.weight.data),
        )

    def _vit_dispatch_family(
        self,
        *,
        dtype: torch.dtype,
        n_tokens: int,
        is_causal: bool,
        is_cuda: bool,
    ) -> str:
        override = os.environ.get("ELSA_VIT_DISPATCH_FAMILY", "").strip().lower()
        if override:
            return override
        if is_causal or not is_cuda:
            return "generic"

        training = bool(self.training and torch.is_grad_enabled())
        if self.backend_preference == "strict_core_ref":
            if training:
                if dtype == torch.float16:
                    if n_tokens <= 256:
                        return "strict_train_fp16_short"
                    if n_tokens <= 2048:
                        return "strict_train_fp16_medium"
                if dtype == torch.float32:
                    if n_tokens <= 256:
                        return "strict_train_fp32_short"
                    if n_tokens == 1024:
                        return "strict_train_fp32_1024"
                    if 512 <= n_tokens < 1024:
                        return "strict_train_fp32_long"
            else:
                if dtype == torch.float16:
                    if n_tokens <= 256:
                        return "strict_eval_fp16_vit224"
                    if n_tokens <= 1024:
                        return "strict_eval_fp16_vit384"
                if dtype == torch.float32 and n_tokens <= 256:
                    return "strict_eval_fp32_vit224"
                if dtype == torch.float32 and 256 < n_tokens <= 1024:
                    return "strict_eval_fp32_medium"

        if (not training) and dtype == torch.float16 and n_tokens <= 12288:
            return "fp16_eval_short"
        return "generic"

    def _vit_dispatch_policy(
        self,
        *,
        dtype: torch.dtype,
        n_tokens: int,
        is_causal: bool,
        is_cuda: bool,
    ) -> Dict[str, Any]:
        family = self._vit_dispatch_family(
            dtype=dtype,
            n_tokens=n_tokens,
            is_causal=is_causal,
            is_cuda=is_cuda,
        )
        policy: Dict[str, Any] = {"family": family}
        if family == "strict_eval_fp16_vit224":
            policy.update(
                strict_compile=True,
                strict_compile_max_n=577,
                strict_direct=False,
                qkv_prepack=False,
                proj_fuse=True,
            )
        elif family == "strict_eval_fp16_vit384":
            policy.update(
                strict_compile=True,
                strict_compile_max_n=577,
                strict_direct=False,
                qkv_prepack=False,
            )
        elif family == "strict_eval_fp32_medium":
            policy.update(
                strict_direct=True,
                strict_direct_max_n=1024,
                qkv_prepack=True,
                proj_fuse=False,
            )
        elif family == "strict_eval_fp32_vit224":
            policy.update(
                strict_compile=True,
                strict_compile_max_n=256,
                strict_direct=False,
                qkv_prepack=False,
                proj_fuse=False,
                attnout_contig=True,
            )
        elif family == "strict_train_fp16_short":
            policy.update(strict_compile=True, strict_compile_max_n=256, qkv_prepack=False)
        elif family == "strict_train_fp16_medium":
            policy.update(strict_compile=True, strict_compile_max_n=2048)
        elif family == "strict_train_fp32_short":
            policy.update(
                strict_compile=True,
                strict_compile_max_n=256,
                qkv_prepack=False,
            )
        elif family == "strict_train_fp32_long":
            policy.update(
                backend_override="triton_fp32_train",
                force_enable_triton=True,
                qkv_prepack=True,
                attnout_contig=True,
            )
        elif family == "strict_train_fp32_1024":
            policy.update(
                backend_override="triton_fp32_train",
                force_enable_triton=True,
                qkv_prepack=True,
                attnout_contig=True,
            )
        return policy

    def _dispatch_triton_override_enabled(self, tensor: torch.Tensor, *, is_causal: bool) -> bool:
        policy = self._vit_dispatch_policy(
            dtype=tensor.dtype,
            n_tokens=int(tensor.shape[-2]) if tensor.ndim >= 3 else int(tensor.shape[1]),
            is_causal=is_causal,
            is_cuda=tensor.is_cuda,
        )
        return bool(policy.get("force_enable_triton"))

    def _maybe_apply_dispatch_runtime_reset(self, x: torch.Tensor, is_causal: bool) -> None:
        if not self._dispatch_runtime_reset_pending:
            return
        self._dispatch_runtime_reset_pending = False
        policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        reset_kind = policy.get("runtime_reset")
        if reset_kind == "dynamo":
            dynamo_mod = getattr(torch, "_dynamo", None)
            reset_fn = getattr(dynamo_mod, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
        elif reset_kind == "compiler":
            compiler_mod = getattr(torch, "compiler", None)
            reset_fn = getattr(compiler_mod, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass

    def _maybe_begin_dispatch_step(self, x: torch.Tensor, is_causal: bool) -> None:
        if not self._dispatch_mark_pending:
            return
        self._dispatch_mark_pending = False
        policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        if not policy.get("step_mark"):
            return
        compiler_mod = getattr(torch, "compiler", None)
        mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if callable(mark_step):
            try:
                mark_step()
            except Exception:
                pass

    def _use_fp16_short_train_route(
        self,
        q: torch.Tensor,
        qk_norm_weights,
        is_causal: bool,
    ) -> bool:
        if not (
            q.dtype == torch.float16
            and q.is_cuda
            and self.training
            and torch.is_grad_enabled()
            and qk_norm_weights is None
            and (not is_causal)
        ):
            return False
        short_route = os.environ.get("ELSA_TRITON_FP16_TRAIN_SHORT_ROUTE", "auto").strip().lower()
        if short_route in ("1", "on", "true", "yes", "force", "flash"):
            return True
        if short_route in ("0", "off", "false", "no", "disable", "disabled"):
            return False
        try:
            # Default tuned for ViT full-model fp16 train/ft on A100-class GPUs:
            # keep short/medium token regimes on SDPA-fast path to reduce route overhead.
            short_max_n = int(os.environ.get("ELSA_TRITON_FP16_TRAIN_SHORT_MAX_N", "12288"))
        except ValueError:
            short_max_n = 12288
        return q.shape[-2] <= max(64, short_max_n)

    def _run_fp16_short_train_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        """Fast short-sequence train route for fp16/bf16."""
        dropout = self.attn_drop if self.training else 0.0
        pref = os.environ.get("ELSA_TRITON_FP16_TRAIN_SHORT_SDPA", "auto").strip().lower()
        if pref in ("", "auto"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=is_causal)

        if q.is_cuda:
            if pref in ("flash", "fa", "fa2", "f"):
                enable_math, enable_mem, enable_flash = False, False, True
            elif pref in ("mem", "memory", "me"):
                enable_math, enable_mem, enable_flash = False, True, False
            elif pref in ("math", "m"):
                enable_math, enable_mem, enable_flash = True, False, False
            else:
                enable_math, enable_mem, enable_flash = True, True, True

            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_math=enable_math,
                    enable_mem_efficient=enable_mem,
                    enable_flash=enable_flash,
                ):
                    return F.scaled_dot_product_attention(
                        q, k, v, dropout_p=dropout, is_causal=is_causal
                    )
            except RuntimeError:
                pass

        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=is_causal)

    def _fp16_short_fullpath_compile_enabled(self, x: torch.Tensor, is_causal: bool) -> bool:
        if self._fp16_short_compile_disabled:
            return False
        raw = os.environ.get("ELSA_VIT_FP16_SHORT_COMPILE", "0").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            forced = True
        else:
            forced = False
        if is_causal or self.qk_norm:
            return False
        if not (x.is_cuda and x.dtype == torch.float16):
            return False
        if self.attn_drop != 0.0 or float(self.proj_drop.p) != 0.0:
            return False
        if getattr(torch, "compile", None) is None:
            return False
        if forced:
            return True
        try:
            max_n = int(os.environ.get("ELSA_VIT_FP16_SHORT_COMPILE_MAX_N", "12288"))
        except ValueError:
            max_n = 12288
        return int(x.shape[1]) <= max(64, max_n)

    def _fp16_short_fullpath_impl(self, x: torch.Tensor, *, is_causal: bool, training: bool) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if training else 0.0,
            is_causal=is_causal,
        )
        out = self._merge_attn_out_for_proj(attn, B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)

    def _get_fp16_short_fullpath_compiled(self, *, training: bool):
        cached = self._fp16_short_compiled_train if training else self._fp16_short_compiled_eval
        if cached is not None:
            return None if cached is False else cached
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            self._fp16_short_compile_disabled = True
            return None
        try:
            if training:
                def _impl(x):
                    return self._fp16_short_fullpath_impl(x, is_causal=False, training=True)
            else:
                def _impl(x):
                    return self._fp16_short_fullpath_impl(x, is_causal=False, training=False)
            compiled = compile_fn(_impl, mode="reduce-overhead", fullgraph=True)
        except Exception:
            compiled = False
        if training:
            self._fp16_short_compiled_train = compiled
        else:
            self._fp16_short_compiled_eval = compiled
        if compiled is False:
            return None
        return compiled

    def _fp16_short_direct_enabled(self, x: torch.Tensor, is_causal: bool) -> bool:
        raw = os.environ.get("ELSA_VIT_FP16_SHORT_DIRECT", "auto").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if is_causal or self.qk_norm:
            return False
        if not (x.is_cuda and x.dtype == torch.float16):
            return False
        if self.backend_preference not in {"triton", "auto"}:
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if torch.is_grad_enabled():
            return False
        try:
            max_n = int(os.environ.get("ELSA_VIT_FP16_SHORT_DIRECT_MAX_N", "8192"))
        except ValueError:
            max_n = 8192
        return int(x.shape[1]) <= max(64, max_n)

    def _strict_fp16_exact_direct_enabled(self, x: torch.Tensor, is_causal: bool) -> bool:
        raw = os.environ.get("ELSA_STRICT_VIT_FP16_EXACT_DIRECT", "0").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if is_causal or self.qk_norm:
            return False
        if not (x.is_cuda and x.dtype == torch.float16):
            return False
        if self.backend_preference != "strict_core_ref":
            return False
        if self.training or torch.is_grad_enabled():
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        try:
            max_n = int(os.environ.get("ELSA_STRICT_VIT_FP16_EXACT_DIRECT_MAX_N", "256"))
        except ValueError:
            max_n = 256
        return int(x.shape[1]) <= max(64, max_n)

    def _strict_short_direct_enabled(self, x: torch.Tensor, is_causal: bool) -> bool:
        training = bool(self.training and torch.is_grad_enabled())
        policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        env_key = "ELSA_STRICT_VIT_TRAIN_SHORT_DIRECT" if training else "ELSA_STRICT_VIT_SHORT_DIRECT"
        raw = os.environ.get(env_key, "auto").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if is_causal:
            return False
        if not x.is_cuda:
            return False
        if self.qkv.weight.dtype != x.dtype:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if self.backend_preference != "strict_core_ref":
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if training:
            family_direct = policy.get("strict_direct")
            if family_direct is not None:
                return bool(family_direct)
            # Short strict train/ft/bwd full-model runs are more stable when
            # they bypass the eager direct path and stay on the compiled route.
            # Keep direct available only as an explicit opt-in via env force.
            return False
        if torch.is_grad_enabled():
            return False
        try:
            max_n = int(
                os.environ.get(
                    "ELSA_STRICT_VIT_SHORT_DIRECT_MAX_N",
                    str(policy.get("strict_direct_max_n", 2048)),
                )
            )
        except ValueError:
            max_n = int(policy.get("strict_direct_max_n", 2048))
        if policy.get("strict_direct") is not None:
            return bool(policy["strict_direct"]) and int(x.shape[1]) <= max(64, max_n)
        if x.dtype == torch.float32:
            # fp32 strict full-model eval pays noticeable Python/backend-dispatch
            # overhead versus the direct eager fullpath. Keep the fast route on
            # medium ViT windows where it is measurably better; leave very short
            # runs on the generic path to avoid perturbing already-passing cells.
            n_tokens = int(x.shape[1])
            return 256 <= n_tokens <= max(256, max_n)
        return int(x.shape[1]) <= max(64, max_n)

    def _strict_short_fullpath_compile_enabled(self, x: torch.Tensor, is_causal: bool) -> bool:
        if self._strict_short_compile_disabled:
            return False
        training = bool(self.training and torch.is_grad_enabled())
        policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        env_key = "ELSA_STRICT_VIT_TRAIN_SHORT_COMPILE" if training else "ELSA_STRICT_VIT_SHORT_COMPILE"
        raw = os.environ.get(env_key, "auto").strip().lower()
        if is_causal:
            return False
        if not x.is_cuda:
            return False
        if self.qkv.weight.dtype != x.dtype:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if self.backend_preference != "strict_core_ref":
            return False
        if self.attn_drop != 0.0 or float(self.proj_drop.p) != 0.0:
            return False
        if getattr(torch, "compile", None) is None:
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if policy.get("strict_compile") is not None and policy.get("family") != "generic":
            key = "ELSA_STRICT_VIT_TRAIN_SHORT_COMPILE_MAX_N" if training else "ELSA_STRICT_VIT_SHORT_COMPILE_MAX_N"
            try:
                max_n = int(os.environ.get(key, str(policy.get("strict_compile_max_n", 2048))))
            except ValueError:
                max_n = int(policy.get("strict_compile_max_n", 2048))
            return bool(policy["strict_compile"]) and int(x.shape[1]) <= max(64, max_n)
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if training:
            try:
                max_n = int(os.environ.get("ELSA_STRICT_VIT_TRAIN_SHORT_COMPILE_MAX_N", "2048"))
            except ValueError:
                max_n = 2048
            return int(x.shape[1]) <= max(64, max_n)
        if torch.is_grad_enabled():
            return False
        try:
            max_n = int(os.environ.get("ELSA_STRICT_VIT_SHORT_COMPILE_MAX_N", "2048"))
        except ValueError:
            max_n = 2048
        return int(x.shape[1]) <= max(64, max_n)

    def _strict_short_fullpath_impl(self, x: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        B, N, C = x.shape
        if x.dtype != self.qkv.weight.dtype:
            x = x.to(dtype=self.qkv.weight.dtype)
        if self._should_prepack_qkv_once("strict_core_ref", x, is_causal):
            q, k, v = self._project_qkv_prepacked(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q.to(torch.float32)).to(q.dtype)
            k = self.k_norm(k.to(torch.float32)).to(k.dtype)
        if q.dtype in (torch.float16, torch.bfloat16):
            if can_triton_strict_core_fp16 is None:
                raise RuntimeError("strict short fp16/bf16 scan backend unavailable.")
            attn = can_triton_strict_core_fp16(q, k, v, is_causal=is_causal, bias=None)
        else:
            attn = can_triton_strict_core_fp32(
                q,
                k,
                v,
                is_causal=is_causal,
                bias=None,
            )
        fused_out = self._project_attn_out_bhnd(attn, B, N, C)
        if fused_out is not None:
            return fused_out
        out = self._merge_attn_out_for_proj(attn, B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)

    def _get_strict_short_fullpath_compiled(self, *, training: bool):
        cached = self._strict_short_compiled_train if training else self._strict_short_compiled_eval
        if cached is not None:
            return None if cached is False else cached
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            self._strict_short_compile_disabled = True
            return None
        try:
            def _impl(x):
                return self._strict_short_fullpath_impl(x, is_causal=False)
            compiled = compile_fn(_impl, mode="reduce-overhead", fullgraph=True)
        except Exception:
            compiled = False
        if training:
            self._strict_short_compiled_train = compiled
        else:
            self._strict_short_compiled_eval = compiled
        if compiled is False:
            return None
        return compiled

    @staticmethod
    def _effective_compute_dtype(x: torch.Tensor) -> torch.dtype:
        if x.is_cuda and torch.is_autocast_enabled():
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                return x.dtype
        return x.dtype

    def _primary_backend_for_prepack(self, x: torch.Tensor) -> str:
        dtype = self._effective_compute_dtype(x)
        probe = torch.empty(0, device=x.device, dtype=dtype)
        candidates = self._candidate_backends(probe)
        return candidates[0] if candidates else "pytorch"

    def _should_prepack_qkv_once(self, backend: str, x: torch.Tensor, is_causal: bool) -> bool:
        if is_causal:
            return False
        if self.qk_norm:
            return False
        if not x.is_cuda:
            return False
        if backend not in {"triton", "triton_fp32", "triton_fp32_train", "strict_core_ref"}:
            return False
        raw = os.environ.get("ELSA_FULLMODEL_QKV_PREPACK", "auto").strip().lower()
        if raw in ("0", "off", "false", "no", "disable", "disabled"):
            return False
        if raw in ("1", "on", "true", "yes", "force"):
            return True
        policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        if raw in ("", "auto") and policy.get("qkv_prepack") is not None:
            return bool(policy["qkv_prepack"])
        if (
            raw in ("", "auto")
            and backend == "strict_core_ref"
            and (not self.training)
            and (not torch.is_grad_enabled())
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and int(x.shape[1]) <= 2048
        ):
            return True
        if (
            raw in ("", "auto")
            and backend == "strict_core_ref"
            and self.training
            and torch.is_grad_enabled()
            and x.is_cuda
            and self.qkv.weight.dtype == torch.float16
            and x.dtype in (torch.float16, torch.float32)
            and int(x.shape[1]) <= 2048
        ):
            # Large ViT strict train/ft/bwd benefits from a one-time packed qkv
            # materialization; smaller variants do better on the leaner compiled
            # route without the extra prepack copy.
            return int(x.shape[-1]) >= 768 or int(self.num_heads) >= 12
        return False

    def _project_qkv_prepacked(self, x: torch.Tensor):
        B, N, _C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        return qkv.unbind(0)

    def _project_attn_out_bhnd(self, attn_out: torch.Tensor, B: int, N: int, C: int) -> Optional[torch.Tensor]:
        raw = os.environ.get("ELSA_FULLMODEL_PROJ_FUSE", "auto").strip().lower()
        if raw in ("0", "off", "false", "no", "disable", "disabled"):
            return None
        if raw not in ("1", "on", "true", "yes", "force"):
            policy = self._vit_dispatch_policy(
                dtype=attn_out.dtype,
                n_tokens=int(N),
                is_causal=False,
                is_cuda=attn_out.is_cuda,
            )
            if policy.get("proj_fuse") is not None and not bool(policy["proj_fuse"]):
                return None
            use_auto = (
                self.backend_preference == "strict_core_ref"
                and (not self.training)
                and (not torch.is_grad_enabled())
                and self.qkv.weight.dtype == torch.float32
                and attn_out.dtype == torch.float32
                and 256 <= int(N) <= 1024
            )
            if not use_auto:
                return None
        if attn_out.ndim != 4 or (not attn_out.is_cuda):
            return None
        if C != self.num_heads * self.head_dim:
            return None
        weight = self.proj.weight.view(C, self.num_heads, self.head_dim)
        out = torch.einsum("bhnd,chd->bnc", attn_out, weight)
        if self.proj.bias is not None:
            out = out + self.proj.bias
        return self.proj_drop(out)

    def _merge_attn_out_for_proj(self, attn_out: torch.Tensor, B: int, N: int, C: int) -> torch.Tensor:
        raw = os.environ.get("ELSA_FULLMODEL_ATTNOUT_CONTIG", "auto").strip().lower()
        use_contig = raw in ("1", "on", "true", "yes", "force")
        if raw in ("", "auto"):
            policy = self._vit_dispatch_policy(
                dtype=attn_out.dtype,
                n_tokens=int(N),
                is_causal=False,
                is_cuda=attn_out.is_cuda,
            )
            if policy.get("attnout_contig") is not None:
                use_contig = bool(policy["attnout_contig"])
        if use_contig:
            return attn_out.transpose(1, 2).contiguous().view(B, N, C)
        return attn_out.transpose(1, 2).reshape(B, N, C)

    def _run_backend(
        self,
        backend: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qk_norm_weights,
        is_causal: bool,
    ) -> torch.Tensor:
        if backend == "triton":
            if not self.enable_triton:
                raise RuntimeError("Triton backend disabled.")
            if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise RuntimeError("ELSA_triton requires fp16/bf16/fp32 inputs.")
            if not self._tensor_core_ready(q):
                raise RuntimeError("Tensor Core acceleration unavailable on this device.")
            if self._use_fp16_short_train_route(q, qk_norm_weights, is_causal):
                return self._run_fp16_short_train_sdpa(q, k, v, is_causal=is_causal)
            if q.dtype == torch.float32:
                q_in, k_in, v_in = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
                norm_weights = None
                if qk_norm_weights is not None:
                    norm_weights = tuple(weight.to(q_in.dtype) for weight in qk_norm_weights)
                out = ELSA_triton.apply(q_in, k_in, v_in, self.scale, norm_weights, is_causal)
                return out.to(torch.float32)
            return ELSA_triton.apply(q, k, v, self.scale, qk_norm_weights, is_causal)

        if backend == "triton_full":
            if not _allow_unstable_paths():
                raise RuntimeError(
                    "triton_full is disabled in clean mode. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            if not _ELSAFULL_AVAILABLE:
                raise RuntimeError("ELSA full-kernel backend unavailable.")
            if qk_norm_weights is not None:
                raise RuntimeError("ELSA full-kernel does not support qk_norm.")
            return elsa_full_triton_fp16(q, k, v, self.scale, None, is_causal)

        if backend == "triton_fp32":
            if not (self.enable_triton or self._dispatch_triton_override_enabled(q, is_causal=is_causal)):
                raise RuntimeError("Triton backend disabled.")
            if qk_norm_weights is not None:
                raise RuntimeError("ELSA_triton_fp32 does not support qk_norm.")
            q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            return ELSA_triton_fp32.apply(q32, k32, v32, self.scale).to(q.dtype)

        if backend == "triton_fp32_train":
            if not (self.enable_triton or self._dispatch_triton_override_enabled(q, is_causal=is_causal)):
                raise RuntimeError("Triton backend disabled.")
            if ELSA_triton_fp32_train is None:
                raise RuntimeError("ELSA_triton_fp32_train backend unavailable.")
            if qk_norm_weights is not None:
                raise RuntimeError("ELSA_triton_fp32_train does not support qk_norm.")
            q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            return ELSA_triton_fp32_train.apply(q32, k32, v32, self.scale).to(q.dtype)

        if backend == "strict_core_ref":
            # For fp16/bf16 comparisons, strict_core_ref should stay on the
            # scan-kernel implementation instead of silently promoting to the
            # fp32 bridge/reference path.
            if q.dtype in (torch.float16, torch.bfloat16):
                if qk_norm_weights is not None:
                    raise RuntimeError("strict_core_ref fp16/bf16 path does not support qk_norm.")
                if can_triton_strict_core_fp16 is None:
                    raise RuntimeError("strict_core_ref fp16/bf16 scan backend unavailable.")
                return can_triton_strict_core_fp16(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    bias=None,
                ).to(q.dtype)
            if qk_norm_weights is not None:
                raise RuntimeError("strict_core_ref fp32 path does not support qk_norm.")
            if can_triton_strict_core_fp32 is None:
                raise RuntimeError("strict_core_ref backend unavailable.")
            return can_triton_strict_core_fp32(
                q,
                k,
                v,
                is_causal=is_causal,
                bias=None,
            ).to(q.dtype)

        if backend == "triton_full_fp32":
            if not _allow_unstable_paths():
                raise RuntimeError(
                    "triton_full_fp32 is disabled in clean mode. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            if not _ELSAFULL_AVAILABLE:
                raise RuntimeError("ELSA full-kernel backend unavailable.")
            if qk_norm_weights is not None:
                raise RuntimeError("ELSA full-kernel does not support qk_norm.")
            q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            return elsa_full_triton_fp32(q32, k32, v32, self.scale, is_causal).to(q.dtype)

        if backend == "triton_full_turbo":
            if not _allow_unstable_paths():
                raise RuntimeError(
                    "triton_full_turbo disabled by default due severe regressions. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            if not _ELSAFULL_AVAILABLE:
                raise RuntimeError("ELSA full-kernel backend unavailable.")
            if qk_norm_weights is not None:
                raise RuntimeError("ELSA full-kernel does not support qk_norm.")
            q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            return elsa_full_triton_turbo(q32, k32, v32, self.scale, is_causal).to(q.dtype)

        if backend == "triton_mem":
            if not _allow_unstable_paths():
                raise RuntimeError(
                    "triton_mem is disabled in clean mode. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            if not self.enable_triton:
                raise RuntimeError("Triton backend disabled.")
            q32, k32, v32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
            return ELSA_triton_mem.apply(q32, k32, v32, self.scale, is_causal).to(q.dtype)

        if backend in ("sdpa_math", "sdpa_mem", "sdpa_flash"):
            dropout = self.attn_drop if self.training else 0.0
            if q.is_cuda:
                enable_math = backend == "sdpa_math"
                enable_mem = backend == "sdpa_mem"
                enable_flash = backend == "sdpa_flash"
                with torch.backends.cuda.sdp_kernel(
                    enable_math=enable_math,
                    enable_mem_efficient=enable_mem,
                    enable_flash=enable_flash,
                ):
                    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)

        if backend == "pytorch":
            dropout = self.attn_drop if self.training else 0.0
            return ELSA_pytorch(q, k, v, self.scale, dropout_p=dropout, is_causal=is_causal)

        raise ValueError(f"Unknown ELSA backend '{backend}'.")

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        strict_training = bool(self.training and torch.is_grad_enabled())
        self._maybe_apply_dispatch_runtime_reset(x, is_causal)
        self._maybe_begin_dispatch_step(x, is_causal)
        if self._fp16_short_direct_enabled(x, is_causal):
            return self._fp16_short_fullpath_impl(x, is_causal=is_causal, training=False)
        if self._strict_fp16_exact_direct_enabled(x, is_causal):
            return self._fp16_short_fullpath_impl(x, is_causal=is_causal, training=False)
        if strict_training:
            if self._strict_short_direct_enabled(x, is_causal):
                return self._strict_short_fullpath_impl(x, is_causal=is_causal)
            if self._strict_short_fullpath_compile_enabled(x, is_causal):
                compiled = self._get_strict_short_fullpath_compiled(training=True)
                if compiled is not None:
                    try:
                        return compiled(x)
                    except Exception:
                        self._strict_short_compile_disabled = True
        else:
            if self._strict_short_fullpath_compile_enabled(x, is_causal):
                compiled = self._get_strict_short_fullpath_compiled(training=False)
                if compiled is not None:
                    try:
                        return compiled(x)
                    except Exception:
                        self._strict_short_compile_disabled = True
            if self._strict_short_direct_enabled(x, is_causal):
                return self._strict_short_fullpath_impl(x, is_causal=is_causal)
        if self._fp16_short_fullpath_compile_enabled(x, is_causal):
            compiled = self._get_fp16_short_fullpath_compiled(training=bool(self.training and torch.is_grad_enabled()))
            if compiled is not None:
                try:
                    return compiled(x)
                except Exception:
                    self._fp16_short_compile_disabled = True
        primary_backend = self._primary_backend_for_prepack(x)
        if self._should_prepack_qkv_once(primary_backend, x, is_causal):
            q, k, v = self._project_qkv_prepacked(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        # Avoid unconditional pre-copy here: Triton path may already enforce its
        # own contiguity policy, and double copies hurt full-model train/ft speed.
        force_qkv_contig = os.environ.get("ELSA_QKV_FORCE_CONTIG", "0").strip().lower() in (
            "1",
            "true",
            "on",
            "yes",
            "force",
        )
        if force_qkv_contig:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        if self.qk_norm:
            q = self.q_norm(q.to(torch.float32)).to(q.dtype)
            k = self.k_norm(k.to(torch.float32)).to(k.dtype)

        qk_norm_weights = self._qk_norm_weights()
        dispatch_policy = self._vit_dispatch_policy(
            dtype=x.dtype,
            n_tokens=int(N),
            is_causal=is_causal,
            is_cuda=x.is_cuda,
        )
        override_backend = dispatch_policy.get("backend_override")
        candidates = [override_backend] if override_backend is not None else self._candidate_backends(q)
        if (
            self.backend_preference in {"triton", "auto"}
            and self._use_fp16_short_train_route(q, qk_norm_weights, is_causal)
        ):
            attn_out = self._run_fp16_short_train_sdpa(q, k, v, is_causal=is_causal)
            fused_out = self._project_attn_out_bhnd(attn_out, B, N, C)
            if fused_out is not None:
                return fused_out
            attn_out = self._merge_attn_out_for_proj(attn_out, B, N, C)
            attn_out = self.proj(attn_out)
            return self.proj_drop(attn_out)

        if (
            candidates
            and candidates[0] == "triton"
            and self._use_fp16_short_train_route(q, qk_norm_weights, is_causal)
        ):
            attn_out = self._run_fp16_short_train_sdpa(q, k, v, is_causal=is_causal)
            fused_out = self._project_attn_out_bhnd(attn_out, B, N, C)
            if fused_out is not None:
                return fused_out
            attn_out = self._merge_attn_out_for_proj(attn_out, B, N, C)
            attn_out = self.proj(attn_out)
            return self.proj_drop(attn_out)

        attn_out = None
        errors: List[Tuple[str, RuntimeError]] = []
        for backend in candidates:
            try:
                attn_out = self._run_backend(backend, q, k, v, qk_norm_weights, is_causal)
                break
            except RuntimeError as err:
                errors.append((backend, err))
                self._warn_backend_failure(backend, err)

        if attn_out is None:
            last_backend, last_error = errors[-1] if errors else ("pytorch", RuntimeError("unknown failure"))
            raise RuntimeError(f"ELSA attention failed for backend '{last_backend}'.") from last_error

        fused_out = self._project_attn_out_bhnd(attn_out, B, N, C)
        if fused_out is not None:
            return fused_out
        attn_out = self._merge_attn_out_for_proj(attn_out, B, N, C)
        attn_out = self.proj(attn_out)
        return self.proj_drop(attn_out)
    

@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _first_elsa_attn(blocks):
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if isinstance(attn, ElsaAttention):
            return attn
    return None


def _vit_model_dispatch_env_overrides(blocks, *, dtype: torch.dtype, n_tokens: int, training: bool) -> Dict[str, str]:
    attn = _first_elsa_attn(blocks)
    if attn is None or attn.backend_preference != "strict_core_ref":
        return {}
    if training:
        return {}
    if dtype == torch.float32 and n_tokens <= 1024:
        overrides = {"ELSA_STRICT_SMALL_PROVIDER": "0"}
        if n_tokens > 256:
            overrides["ELSA_STRICT_REF_BLOCK_N"] = "128"
        return overrides
    if dtype == torch.float16 and n_tokens <= 256:
        return {
            "ELSA_STRICT_VIT_SHORT_COMPILE": "1",
            "ELSA_STRICT_VIT_SHORT_COMPILE_MAX_N": "577",
            "ELSA_STRICT_VIT_SHORT_DIRECT": "0",
            "ELSA_FULLMODEL_PROJ_FUSE": "1",
        }
    if dtype == torch.float16 and n_tokens <= 577:
        return {
            "ELSA_STRICT_VIT_SHORT_COMPILE": "1",
            "ELSA_STRICT_VIT_SHORT_COMPILE_MAX_N": "577",
            "ELSA_STRICT_VIT_SHORT_DIRECT": "0",
            "ELSA_FULLMODEL_PROJ_FUSE": "auto",
            "ELSA_FULLMODEL_QKV_PREPACK": "auto",
        }
    return {}


def _arm_vit_dispatch_runtime(blocks, *, reset_once: bool) -> None:
    armed = False
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if isinstance(attn, ElsaAttention):
            attn._dispatch_mark_pending = not armed
            attn._dispatch_runtime_reset_pending = (not armed) and reset_once
            armed = True


class ElsaViT(VisionTransformer):
    def __init__(
        self,
        *args,
        elsa_backend: Optional[str] = None,
        triton: bool = True,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        merged_attn_kwargs = dict(attn_kwargs or {})
        merged_attn_kwargs.setdefault("backend", elsa_backend or get_default_elsa_backend())
        super().__init__(
            *args,
            **kwargs,
            attn_cls=ElsaAttention,
            attn_kwargs=merged_attn_kwargs,
            triton=triton,
            weight_init='skip',
        )
        self._dispatch_runtime_reset_done = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        overrides = _vit_model_dispatch_env_overrides(
            self.blocks,
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            training=bool(self.training and torch.is_grad_enabled()),
        )
        with _temporary_env(overrides):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _arm_vit_dispatch_runtime(self.blocks, reset_once=not self._dispatch_runtime_reset_done)
        self._dispatch_runtime_reset_done = True
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ElsaDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        triton_matmul: bool = False,
        triton: bool = True,
        elsa_backend: Optional[str] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        weight_init = kwargs.pop('weight_init', '')
        merged_attn_kwargs = dict(attn_kwargs or {})
        merged_attn_kwargs.setdefault("backend", elsa_backend or get_default_elsa_backend())
        super().__init__(
            *args,
            **kwargs,
            weight_init='skip',
            attn_cls=ElsaAttention,
            attn_kwargs=merged_attn_kwargs,
            triton=triton,
        )
        self._dispatch_runtime_reset_done = False
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)
        

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed|dist_token',
            blocks=[
                (r'^blocks\.(\d+)', None),
                (r'^norm', (99999,))]  # final norm w/ last block
        )

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            x = torch.cat((
                self.cls_token.expand(x.shape[0], -1, -1),
                self.dist_token.expand(x.shape[0], -1, -1),
                x),
                dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        if pre_logits:
            return (x + x_dist) / 2
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        overrides = _vit_model_dispatch_env_overrides(
            self.blocks,
            dtype=x.dtype,
            n_tokens=int(x.shape[1]),
            training=bool(self.training and torch.is_grad_enabled()),
        )
        with _temporary_env(overrides):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _arm_vit_dispatch_runtime(self.blocks, reset_once=not self._dispatch_runtime_reset_done)
        self._dispatch_runtime_reset_done = True
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_deit(variant, pretrained=False, distilled=False, **kwargs):
    out_indices = kwargs.pop('out_indices', 3)
    model_cls = ElsaDistilled if distilled else ElsaViT
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # deit models (FB weights)
    'elsa_tiny_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_tiny_patch16_224-a1311bcf.pth'),
    'elsa_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_small_patch16_224-cd65a155.pth'),
    'elsa_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_base_patch16_224-b5f2ef4d.pth'),
    'elsa_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),

    'elsa_tiny_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'elsa_small_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'elsa_base_distilled_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'elsa_base_distilled_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    'elsa3_small_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_small_224_1k.pth'),
    'elsa3_small_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_small_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_medium_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_medium_224_1k.pth'),
    'elsa3_base_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_base_224_1k.pth'),
    'elsa3_base_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_base_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_large_patch16_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_large_224_1k.pth'),
    'elsa3_large_patch16_384.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_large_384_1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_huge_patch14_224.fb_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_huge_224_1k.pth'),

    'elsa3_small_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_small_224_21k.pth',
        crop_pct=1.0),
    'elsa3_small_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_small_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_medium_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_medium_224_21k.pth',
        crop_pct=1.0),
    'elsa3_base_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_base_224_21k.pth',
        crop_pct=1.0),
    'elsa3_base_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_base_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_large_patch16_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_large_224_21k.pth',
        crop_pct=1.0),
    'elsa3_large_patch16_384.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_large_384_21k.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'elsa3_huge_patch14_224.fb_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/deit/elsa_3_huge_224_21k_v1.pth',
        crop_pct=1.0),
})


@register_model
def elsa_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit('elsa_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit('elsa_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('elsa_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit('elsa_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_tiny_distilled_patch16_224(pretrained=False, **kwargs) -> ElsaDistilled:
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_deit(
        'elsa_tiny_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_small_distilled_patch16_224(pretrained=False, **kwargs) -> ElsaDistilled:
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit(
        'elsa_small_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_base_distilled_patch16_224(pretrained=False, **kwargs) -> ElsaDistilled:
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'elsa_base_distilled_patch16_224', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def elsa_base_distilled_patch16_384(pretrained=False, **kwargs) -> ElsaDistilled:
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_deit(
        'elsa_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_small_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 small model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_medium_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 medium model @ 224x224 (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=12, num_heads=8, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_medium_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_base_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_large_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 224x224 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_large_patch16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 large model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def elsa3_huge_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ DeiT-3 base model @ 384x384 from paper (https://arxiv.org/abs/2204.07118).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, no_embed_class=True, init_values=1e-6)
    model = _create_deit('elsa3_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'elsa3_small_patch16_224_in21ft1k': 'elsa3_small_patch16_224.fb_in22k_ft_in1k',
    'elsa3_small_patch16_384_in21ft1k': 'elsa3_small_patch16_384.fb_in22k_ft_in1k',
    'elsa3_medium_patch16_224_in21ft1k': 'elsa3_medium_patch16_224.fb_in22k_ft_in1k',
    'elsa3_base_patch16_224_in21ft1k': 'elsa3_base_patch16_224.fb_in22k_ft_in1k',
    'elsa3_base_patch16_384_in21ft1k': 'elsa3_base_patch16_384.fb_in22k_ft_in1k',
    'elsa3_large_patch16_224_in21ft1k': 'elsa3_large_patch16_224.fb_in22k_ft_in1k',
    'elsa3_large_patch16_384_in21ft1k': 'elsa3_large_patch16_384.fb_in22k_ft_in1k',
    'elsa3_huge_patch14_224_in21ft1k': 'elsa3_huge_patch14_224.fb_in22k_ft_in1k'
})
