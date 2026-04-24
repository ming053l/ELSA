""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
import os
import warnings
from typing import Callable, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_, ClassifierHead,\
    resample_patch_embed, ndgrid, get_act_layer, LayerType
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._features_fx import register_notrace_function
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from . import elsa as elsa_core

__all__ = ['SwinELSA', 'set_default_elsa_backend', 'get_default_elsa_backend']

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

ELSA_swinv2_pytorch = elsa_core.ELSA_swinv2_pytorch
ELSA_swinv2_train_kernel = getattr(elsa_core, "ELSA_swinv2_train_kernel", ELSA_swinv2_pytorch)
ELSA_swinv2_train_fused = getattr(elsa_core, "ELSA_swinv2_train_fused", ELSA_swinv2_pytorch)
elsa_swinv2_triton = getattr(elsa_core, "elsa_swinv2_triton", None)
elsa_swinv2_triton_full = getattr(elsa_core, "elsa_swinv2_triton_full", None)
_ELSATRITON_AVAILABLE = getattr(elsa_core, "_ELSATRITON_AVAILABLE", False)


def set_default_elsa_backend(backend: str) -> None:
    elsa_core.set_default_elsa_backend(backend)


def get_default_elsa_backend() -> str:
    return elsa_core.get_default_elsa_backend()


def _allow_unstable_paths() -> bool:
    return os.environ.get("ELSA_ALLOW_UNSTABLE_PATHS", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
        "force",
    )


def _canonical_swin_backend(backend: str) -> str:
    if _allow_unstable_paths():
        return backend
    if backend in {"swin_train_fused", "train_fused"}:
        return "swin_train_kernel"
    if backend in {"triton_full", "triton_full_fp32", "triton_full_turbo", "triton_mem"}:
        return "triton"
    return backend

# --------------------- 1. Window Attention (精確 MHSA) ---------------------------

def _reshape(x, h):
    B, C, H, W = x.shape
    N, d = H * W, C // h
    return x.reshape(B, h, d, N)


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            dim: int,
            window_size: Tuple[int, int],
            num_heads: int,
            configured_window_size: Optional[Tuple[int, int]] = None,
            qkv_bias: bool = True,
            qkv_bias_separate: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            pretrained_window_size: Tuple[int, int] = (0, 0),
        triton_matmul = False,
        triton = True,
        backend: Optional[str] = None,
        strict_launch_cfg: Optional[dict] = None,
        strict_launch_cfg_train: Optional[dict] = None,
        strict_launch_cfg_eval: Optional[dict] = None,
        strict_compact_mask_max_n: Optional[int] = None,
        strict_use_compact_mask: Optional[bool] = None,
        strict_use_compact_mask_train: Optional[bool] = None,
        strict_use_compact_mask_eval: Optional[bool] = None,
        strict_force_out_nh: Optional[bool] = None,
        strict_fuse_compact_bias: Optional[bool] = None,
    ) -> None:
        super().__init__()
        backend = backend or get_default_elsa_backend()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        # Keep the originally requested window around for dispatch decisions.
        # Later stages can shrink the live window size (e.g. 24 -> 12 when the
        # feature map becomes smaller), but those reduced windows still belong
        # to the large-window family and must not re-enter the short-window
        # direct qblock path.
        self.configured_window_size = to_2tuple(configured_window_size or window_size)
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.num_heads = num_heads
        self.qkv_bias_separate = qkv_bias_separate
        self.triton_matmul = triton_matmul
        self.triton = triton
        resolved_backend = backend.lower() if backend else get_default_elsa_backend()
        clean_backend = _canonical_swin_backend(resolved_backend)
        if clean_backend != resolved_backend:
            warnings.warn(
                f"Swin backend '{resolved_backend}' is disabled in clean mode; using '{clean_backend}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_backend = clean_backend
        self.backend_preference = resolved_backend
        self.enable_triton = triton
        self.strict_launch_cfg = dict(strict_launch_cfg) if strict_launch_cfg is not None else None
        self.strict_launch_cfg_train = dict(strict_launch_cfg_train) if strict_launch_cfg_train is not None else None
        self.strict_launch_cfg_eval = dict(strict_launch_cfg_eval) if strict_launch_cfg_eval is not None else None
        self.strict_compact_mask_max_n = strict_compact_mask_max_n
        self.strict_use_compact_mask = strict_use_compact_mask
        self.strict_use_compact_mask_train = strict_use_compact_mask_train
        self.strict_use_compact_mask_eval = strict_use_compact_mask_eval
        self.strict_force_out_nh = strict_force_out_nh
        self.strict_fuse_compact_bias = strict_fuse_compact_bias
        self._warned_backends: Set[str] = set()
        self.short_seq_threshold = int(os.environ.get("ELSA_SWIN_SHORT_THRESHOLD", "256"))
        self.enable_short_path = bool(int(os.environ.get("ELSA_SWIN_ENABLE_SHORT", "1")))
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self._make_pair_wise_relative_positions()
        self._rel_bias_cache = {}
        self._logit_scale_cache = {}
        self._mask_cache = {}
        self._strict_eval_compiled_nomask = None
        self._strict_eval_compiled_mask = None
        self._strict_eval_compile_disabled = False

    def _strict_out_nh_enabled(self, n_tokens: int, dtype: Optional[torch.dtype] = None) -> bool:
        if self.strict_force_out_nh is not None:
            return bool(self.strict_force_out_nh)
        raw = os.environ.get("ELSA_SWIN_STRICT_OUT_NH", "auto").strip().lower()
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        policy = self._swin_dispatch_policy(dtype=dtype or self.qkv.weight.dtype, n_tokens=n_tokens, is_cuda=self.qkv.weight.is_cuda)
        if policy.get("out_nh") is not None:
            return bool(policy["out_nh"])
        if dtype == torch.float32 and n_tokens <= 64:
            return True
        return n_tokens >= 128

    def _strict_fuse_compact_bias_enabled(self) -> bool:
        raw = os.environ.get("ELSA_SWIN_STRICT_FUSE_COMPACT_BIAS")
        if raw is None:
            if self.strict_fuse_compact_bias is not None:
                return bool(self.strict_fuse_compact_bias)
            policy = self._swin_dispatch_policy(dtype=self.qkv.weight.dtype, n_tokens=self.window_size[0] * self.window_size[1], is_cuda=self.qkv.weight.is_cuda)
            if policy.get("fuse_compact_bias") is not None:
                return bool(policy["fuse_compact_bias"])
            return True
        return bool(int(raw))

    def _strict_use_compact_mask_enabled(self) -> bool:
        raw = os.environ.get("ELSA_SWIN_COMPACT_MASK")
        if raw is None:
            raw = os.environ.get("ELSA_SWIN_STRICT_COMPACT_MASK")
        if raw is None:
            if self.training and self.strict_use_compact_mask_train is not None:
                return bool(self.strict_use_compact_mask_train)
            if (not self.training) and self.strict_use_compact_mask_eval is not None:
                return bool(self.strict_use_compact_mask_eval)
            if self.strict_use_compact_mask is not None:
                return bool(self.strict_use_compact_mask)
            policy = self._swin_dispatch_policy(dtype=self.qkv.weight.dtype, n_tokens=self.window_size[0] * self.window_size[1], is_cuda=self.qkv.weight.is_cuda)
            if policy.get("use_compact_mask") is not None:
                return bool(policy["use_compact_mask"])
            return True
        return bool(int(raw))

    def _strict_eval_fused_mask_cache_enabled(self) -> bool:
        raw = os.environ.get("ELSA_SWIN_STRICT_FUSED_MASK_CACHE_EVAL", "0").strip().lower()
        return raw not in ("0", "false", "off", "no", "disable", "disabled")

    def _swin_dispatch_family(
        self,
        *,
        dtype: torch.dtype,
        n_tokens: int,
        is_cuda: bool,
    ) -> str:
        override = os.environ.get("ELSA_SWIN_DISPATCH_FAMILY", "").strip().lower()
        if override:
            return override
        dispatch_tokens = max(
            int(n_tokens),
            int(self.configured_window_size[0] * self.configured_window_size[1]),
        )
        if not is_cuda or self.backend_preference != "strict_core_ref":
            return "generic"
        training = bool(self.training and torch.is_grad_enabled())
        if training:
            if dispatch_tokens <= 64:
                return "strict_train_w8"
            if dispatch_tokens <= 256:
                if dtype in (torch.float16, torch.bfloat16):
                    return "strict_train_fp16_w16"
                if dtype == torch.float32:
                    return "strict_train_fp32_w16"
                return "strict_train_w16"
            return "strict_train_generic"
        if dtype in (torch.float16, torch.bfloat16) and dispatch_tokens <= 64:
            return "strict_eval_fp16_w8"
        if dtype in (torch.float16, torch.bfloat16) and 128 <= dispatch_tokens <= 256:
            return "strict_eval_fp16_w16"
        if dtype in (torch.float16, torch.bfloat16) and dispatch_tokens > 256:
            return "strict_eval_fp16_largewin"
        if dtype == torch.float32 and dispatch_tokens <= 64:
            return "strict_eval_fp32_w8"
        if dtype == torch.float32 and 128 <= dispatch_tokens <= 256:
            return "strict_eval_fp32_w16"
        if dtype == torch.float32 and dispatch_tokens > 256:
            return "strict_eval_fp32_largewin"
        return "generic"

    def _swin_dispatch_policy(
        self,
        *,
        dtype: torch.dtype,
        n_tokens: int,
        is_cuda: bool,
    ) -> dict:
        family = self._swin_dispatch_family(dtype=dtype, n_tokens=n_tokens, is_cuda=is_cuda)
        policy = {"family": family}
        if family == "strict_train_w8":
            policy.update(use_compact_mask=False, fuse_compact_bias=False, out_nh=True)
        elif family == "strict_train_fp16_w16":
            policy.update(use_compact_mask=False, fuse_compact_bias=False, out_nh=False, launch_cfg={})
        elif family == "strict_train_fp32_w16":
            policy.update(use_compact_mask=False, fuse_compact_bias=False, out_nh=True, launch_cfg={})
        elif family == "strict_train_w16":
            policy.update(use_compact_mask=False, fuse_compact_bias=False, out_nh=True, launch_cfg={})
        elif family == "strict_eval_fp16_w8":
            policy.update(
                strict_direct=False,
                strict_compile=False,
                use_compact_mask=True,
                fuse_compact_bias=True,
                out_nh=True,
            )
        elif family == "strict_eval_fp16_w16":
            policy.update(
                strict_direct=True,
                strict_compile=False,
                use_compact_mask=True,
                fuse_compact_bias=False,
                out_nh=True,
                launch_cfg={"block_q": 16, "block_n": 64, "num_warps": 8, "num_stages": 1},
            )
        elif family == "strict_eval_fp32_w8":
            policy.update(
                strict_direct=True,
                strict_compile=False,
                use_compact_mask=True,
                fuse_compact_bias=True,
                out_nh=True,
            )
        elif family == "strict_eval_fp32_w16":
            policy.update(
                strict_direct=False,
                use_compact_mask=True,
                fuse_compact_bias=False,
                out_nh=True,
                launch_cfg={"block_q": 16, "block_n": 64, "num_warps": 8, "num_stages": 1},
            )
        elif family in ("strict_eval_fp16_largewin", "strict_eval_fp32_largewin"):
            # Large biased windows still need the exact strict reference path,
            # but the direct/compact qblock specializations only support
            # short-window reductions. Keep these regimes on the safer
            # non-direct fallback route.
            policy.update(
                strict_direct=False,
                strict_compile=False,
                use_compact_mask=True,
                fuse_compact_bias=False,
                out_nh=False,
            )
        return policy

    def _strict_launch_cfg_enabled(self) -> Optional[dict]:
        if self.training and self.strict_launch_cfg_train is not None:
            return self.strict_launch_cfg_train
        if (not self.training) and self.strict_launch_cfg_eval is not None:
            return self.strict_launch_cfg_eval
        if self.strict_launch_cfg is not None:
            return self.strict_launch_cfg
        policy = self._swin_dispatch_policy(dtype=self.qkv.weight.dtype, n_tokens=self.window_size[0] * self.window_size[1], is_cuda=self.qkv.weight.is_cuda)
        return policy.get("launch_cfg")

    def _strict_qkv_prepack_enabled(self, *, dtype: torch.dtype, n_tokens: int, is_cuda: bool) -> bool:
        raw = os.environ.get("ELSA_SWIN_STRICT_QKV_PREPACK_TRAIN", "auto").strip().lower()
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        policy = self._swin_dispatch_policy(dtype=dtype, n_tokens=n_tokens, is_cuda=is_cuda)
        return bool(policy.get("qkv_prepack", False))

    def _make_pair_wise_relative_positions(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0]).to(torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1]).to(torch.float32)
        relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        try:
            compact_rel_bias_min_n = max(0, int(os.environ.get("ELSA_SWIN_COMPACT_REL_BIAS_MIN_N", "8192")))
        except ValueError:
            compact_rel_bias_min_n = 8192
        window_area = int(self.window_size[0] * self.window_size[1])
        if (
            compact_rel_bias_min_n > 0
            and window_area >= compact_rel_bias_min_n
            and self.window_size[0] == self.window_size[1]
        ):
            # Huge Swin windows cannot afford the classic [N, N] relative index
            # buffer. The strict qblock backend consumes the compact CPB table
            # directly and reconstructs tile-local offsets in-kernel.
            self.register_buffer("relative_position_index", torch.empty(0, dtype=torch.long), persistent=False)
            return

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(ndgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _compact_relative_position_enabled(self) -> bool:
        rel_index = getattr(self, "relative_position_index", None)
        return rel_index is not None and rel_index.numel() == 0

    def _relative_position_bias(self, *, compact_ok: bool) -> torch.Tensor:
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        if compact_ok and self._compact_relative_position_enabled():
            relative_position_bias = relative_position_bias_table.squeeze(0).permute(2, 0, 1).contiguous()
            return 16 * torch.sigmoid(relative_position_bias)
        if self._compact_relative_position_enabled():
            # Non-strict fallback/debug paths retain the original semantics, but
            # may OOM for huge windows. The strict path should pass compact_ok.
            coords_h = torch.arange(self.window_size[0], device=self.relative_coords_table.device)
            coords_w = torch.arange(self.window_size[1], device=self.relative_coords_table.device)
            coords = torch.stack(ndgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
        else:
            relative_position_index = self.relative_position_index
        relative_position_bias_table = relative_position_bias_table.view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return 16 * torch.sigmoid(relative_position_bias)

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        """Update window size & interpolate position embeddings
        Args:
            window_size (int): New window size
        """
        window_size = to_2tuple(window_size)
        if window_size != self.window_size:
            self.window_size = window_size
            self._make_pair_wise_relative_positions()
            self._rel_bias_cache.clear()
            self._logit_scale_cache.clear()
            self._mask_cache.clear()

    def _cache_enabled(self) -> bool:
        return (
            (not self.training)
            and (not torch.is_grad_enabled())
            and bool(int(os.environ.get("ELSA_SWIN_CACHE", "1")))
        )

    def _strict_eval_direct_enabled(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> bool:
        policy = self._swin_dispatch_policy(dtype=x.dtype, n_tokens=int(x.shape[1]), is_cuda=x.is_cuda)
        raw = os.environ.get("ELSA_SWIN_STRICT_EVAL_DIRECT", "auto").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if self.backend_preference != "strict_core_ref":
            return False
        if self.training or torch.is_grad_enabled():
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        try:
            max_n = int(os.environ.get("ELSA_SWIN_STRICT_EVAL_DIRECT_MAX_N", "256"))
        except ValueError:
            max_n = 256
        n = int(x.shape[1])
        if n > max(64, max_n):
            return False
        dispatch_tokens = max(
            n,
            int(self.configured_window_size[0] * self.configured_window_size[1]),
        )
        if dispatch_tokens > 256:
            # The strict direct path is only validated for short biased Swin
            # windows. Larger windows should stay on the reference-compatible
            # fallback until the biased qblock path supports them.
            return False
        if raw in ("1", "true", "on", "yes", "force"):
            return True
        if raw == "auto" and policy.get("strict_direct") is not None:
            return bool(policy["strict_direct"])
        if raw == "auto":
            # The direct path is a measured win for strict fp16/bf16 W16 eval,
            # and for strict fp32 W8 eval once compact-bias materialization is
            # no longer kept alive across warmup. Keep larger fp32 windows on
            # the non-direct path for now.
            if x.dtype in (torch.float16, torch.bfloat16):
                return n >= 128
            return x.dtype == torch.float32 and n <= 64
        return False

    def _strict_eval_compile_enabled(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> bool:
        if self._strict_eval_compile_disabled:
            return False
        policy = self._swin_dispatch_policy(dtype=x.dtype, n_tokens=int(x.shape[1]), is_cuda=x.is_cuda)
        raw = os.environ.get("ELSA_SWIN_STRICT_EVAL_COMPILE", "0").strip().lower()
        if raw in ("0", "false", "off", "no", "disable", "disabled"):
            return False
        if not self._strict_eval_direct_enabled(x, mask):
            return False
        if getattr(torch, "compile", None) is None:
            return False
        if raw == "auto" and policy.get("strict_compile") is not None:
            return bool(policy["strict_compile"])
        return raw in ("1", "true", "on", "yes", "force")

    def _strict_eval_fullpath_impl(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B_, N, C = x.shape
        strict_out_nh = self._strict_out_nh_enabled(N, x.dtype)

        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        cache_ok = self._cache_enabled()
        rel_bias_key = (q.device, q.dtype)
        relative_position_bias = None
        if cache_ok:
            relative_position_bias = self._rel_bias_cache.get(rel_bias_key)
        if relative_position_bias is None:
            relative_position_bias = self._relative_position_bias(compact_ok=True)
            if cache_ok:
                self._rel_bias_cache[rel_bias_key] = relative_position_bias

        logit_scale = None
        if cache_ok:
            logit_scale = self._logit_scale_cache.get(rel_bias_key)
        if logit_scale is None:
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01))
            if cache_ok:
                # Cache a detached clone so compiled eval paths do not reuse a
                # graph-owned tensor that may be invalidated by later runs.
                logit_scale = logit_scale.detach().clone()
                self._logit_scale_cache[rel_bias_key] = logit_scale

        backend_bias = relative_position_bias
        backend_mask = None
        if mask is not None:
            num_win = mask.shape[0]
            try:
                strict_compact_max_n = int(
                    os.environ.get(
                        "ELSA_SWIN_STRICT_COMPACT_MASK_MAX_N",
                        str(self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64),
                    )
                )
            except ValueError:
                strict_compact_max_n = self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64
            configured_window_tokens = int(self.configured_window_size[0] * self.configured_window_size[1])
            use_compact = self._strict_use_compact_mask_enabled() and (
                N <= max(64, strict_compact_max_n) or configured_window_tokens > 256
            )
            if use_compact:
                fused_key = None
                fused_cache_ok = cache_ok and self._strict_eval_fused_mask_cache_enabled()
                if fused_cache_ok:
                    fused_key = (
                        "strict_direct_compact_add",
                        mask.data_ptr(),
                        relative_position_bias.data_ptr(),
                        q.device,
                        q.dtype,
                        self.num_heads,
                        N,
                    )
                    backend_mask = self._mask_cache.get(fused_key)
                if backend_mask is None:
                    if mask.ndim == 2:
                        mask_compact = mask
                    else:
                        mask_compact = mask.view(num_win, N, N).unsqueeze(1).expand(num_win, self.num_heads, N, N)
                    # The strict-core bias kernels already support repeated
                    # compact window tensors shaped as (num_win, H, N, N).
                    # Keep this compact form instead of materializing the
                    # batched (B_, H, N, N) expansion on eval.
                    if mask_compact.ndim != 2 and self._strict_fuse_compact_bias_enabled():
                        backend_mask = mask_compact + relative_position_bias.unsqueeze(0)
                        backend_bias = None
                    else:
                        backend_mask = mask_compact
                    if fused_cache_ok and fused_key is not None and backend_bias is None:
                        self._mask_cache[fused_key] = backend_mask
            else:
                backend_mask = mask.view(1, num_win, 1, N, N)
                backend_mask = backend_mask.expand(B_ // num_win, num_win, 1, N, N)
                backend_mask = backend_mask.reshape(B_, 1, N, N)

        attn_output = self._run_backend(
            "strict_core_ref",
            q,
            k,
            v,
            logit_scale,
            backend_bias,
            backend_mask,
            out_layout="NH" if strict_out_nh else "HND",
        )
        if self.training and self.attn_drop.p:
            attn_output = self.attn_drop(attn_output)
        if strict_out_nh:
            x = attn_output.reshape(B_, N, C)
        else:
            x = attn_output.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        if self.training and self.proj_drop.p:
            x = self.proj_drop(x)
        return x

    def _get_strict_eval_compiled(self, has_mask: bool):
        cached = self._strict_eval_compiled_mask if has_mask else self._strict_eval_compiled_nomask
        if cached is not None:
            return None if cached is False else cached
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            self._strict_eval_compile_disabled = True
            return None
        try:
            if has_mask:
                def _impl(x, mask):
                    return self._strict_eval_fullpath_impl(x, mask)
            else:
                def _impl(x):
                    return self._strict_eval_fullpath_impl(x, None)
            compiled = compile_fn(_impl, mode="reduce-overhead", fullgraph=True)
        except Exception:
            compiled = False
        if has_mask:
            self._strict_eval_compiled_mask = compiled
        else:
            self._strict_eval_compiled_nomask = compiled
        if compiled is False:
            return None
        return compiled

    def _candidate_backends(self, tensor: torch.Tensor) -> List[str]:
        preference = self.backend_preference
        strict_no_fallback = bool(int(os.environ.get("ELSA_STRICT_BENCH_NO_FALLBACK", "0")))
        if preference == "auto":
            preference = "triton"

        # Clean routing:
        # - training + grad: always prefer dedicated swin_train_kernel
        # - eval/inference: triton fast path
        # Historical/unstable routes are disabled unless forced.
        if self.training and torch.is_grad_enabled():
            if preference == "strict_core_ref" and strict_no_fallback:
                order = ["strict_core_ref"]
            else:
                if preference not in ("swin_train_kernel", "pytorch") and not _allow_unstable_paths():
                    key = f"clean_redirect_train_{preference}"
                    if key not in self._warned_backends:
                        self._warned_backends.add(key)
                        warnings.warn(
                            f"Swin backend '{preference}' is disabled in clean mode; using 'swin_train_kernel'.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                    preference = "swin_train_kernel"
                order = [preference] if preference == "pytorch" else ["swin_train_kernel", "pytorch"]
        else:
            allow_unstable = _allow_unstable_paths()
            if preference not in ("pytorch", "triton", "strict_core_ref") and not allow_unstable:
                key = f"clean_redirect_eval_{preference}"
                if key not in self._warned_backends:
                    self._warned_backends.add(key)
                    warnings.warn(
                        f"Swin backend '{preference}' is disabled in clean mode; using 'triton'.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                preference = "triton"
            if allow_unstable and preference not in ("pytorch", "triton", "strict_core_ref"):
                order = [preference, "triton", "pytorch"]
            else:
                if preference == "pytorch":
                    order = ["pytorch"]
                elif preference == "strict_core_ref":
                    order = ["strict_core_ref", "pytorch"]
                else:
                    order = ["triton", "pytorch"]
            if preference == "strict_core_ref" and strict_no_fallback:
                order = ["strict_core_ref"]

        # Swin Triton kernels in this branch are forward-only; keep train path differentiable by default.
        if self.training and torch.is_grad_enabled() and bool(int(os.environ.get("ELSA_SWIN_TRAIN_SAFE", "1"))):
            order = [backend for backend in order if not backend.startswith("triton")] or ["pytorch"]
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
            f"Swin ELSA backend '{backend}' failed ({err}). Falling back to the next candidate.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _run_backend(
        self,
        backend: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logit_scale: torch.Tensor,
        relative_position_bias: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        out_layout: str = "HND",
    ) -> torch.Tensor:
        def _format_strict_out(out: torch.Tensor) -> torch.Tensor:
            if out_layout.strip().upper() == "NH":
                return out.transpose(1, 2).contiguous()
            return out

        if backend == "swin_train_kernel":
            return ELSA_swinv2_train_kernel(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
            )
        if backend in ("swin_train_fused", "train_fused"):
            if not _allow_unstable_paths():
                raise RuntimeError(
                    "swin_train_fused is disabled in clean mode. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            return ELSA_swinv2_train_fused(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
            )
        if backend == "triton":
            if not (_ELSATRITON_AVAILABLE and elsa_swinv2_triton is not None):
                raise RuntimeError("ELSA Triton kernels unavailable.")
            use_half_qk = (
                q.dtype == torch.float32
                and torch.backends.cuda.matmul.allow_tf32
                and not self.training
                and bool(int(os.environ.get("ELSA_SWIN_FP32_TURBO", "1")))
            )
            return elsa_swinv2_triton(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
                use_half_qk=use_half_qk,
            )
        if backend == "strict_core_ref":
            scale = logit_scale.to(torch.float32).exp().clamp_min(1e-6).view(1, q.shape[1], 1, 1)
            strict_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else q.dtype
            q_scale = (scale * math.sqrt(q.shape[-1])).to(q.dtype)
            if (not torch.is_grad_enabled()) and strict_dtype == q.dtype:
                # Eval-only fast path: q is not reused after strict-core launch,
                # so scale it in place and avoid an extra transient allocation.
                q_scaled = q.mul_(q_scale)
            else:
                q_scaled = q * q_scale
            if strict_dtype in (torch.float16, torch.bfloat16):
                q_scaled = q_scaled.to(strict_dtype)
                k = k.to(strict_dtype)
                v = v.to(strict_dtype)
            strict_rel_bias = relative_position_bias
            strict_mask = mask
            configured_window_tokens = int(self.configured_window_size[0] * self.configured_window_size[1])
            force_largewin_ref = os.environ.get("ELSA_SWIN_FORCE_LARGEWIN_STRICT_REF", "0").strip().lower() in {
                "1", "true", "yes", "on", "force"
            }
            try:
                huge_masked_ref_min_n = max(
                    0,
                    int(os.environ.get("ELSA_SWIN_LARGEWIN_MASK_REF_MIN_N", "0")),
                )
            except ValueError:
                huge_masked_ref_min_n = 0
            auto_huge_masked_ref = (
                strict_mask is not None
                and huge_masked_ref_min_n > 0
                and configured_window_tokens >= huge_masked_ref_min_n
            )
            if (force_largewin_ref or auto_huge_masked_ref) and (strict_rel_bias is not None or strict_mask is not None) and configured_window_tokens > 256:
                # Large-window Swin blocks can shrink to smaller live windows in
                # deep stages (e.g. 24x24 -> 12x12), but they still carry the
                # same relative-bias semantics. The qblock bias fast path only
                # used to support only short-window direct reductions. Keep an
                # opt-in exact-reference escape hatch for debugging, but let
                # large-window eval hit the strict qblock path by default now
                # that biased non-direct summaries are implemented. Very large
                # shifted windows still materialize an enormous compact-mask +
                # rel-bias tensor if we force them through the single-bias
                # qblock interface, so route those masked regimes through the
                # chunked exact reference path instead.
                largewin_ref_block_default = "64" if configured_window_tokens >= 4096 else "128"
                strict_ref_block_n = max(
                    16,
                    int(os.environ.get("ELSA_SWIN_LARGEWIN_STRICT_REF_BLOCK_N", largewin_ref_block_default)),
                )
                default_chunk_b = "1" if configured_window_tokens >= 4096 else "8"
                strict_ref_chunk_b = max(
                    1,
                    int(os.environ.get("ELSA_SWIN_LARGEWIN_STRICT_REF_CHUNK_B", default_chunk_b)),
                )
                q_ref = q_scaled.to(torch.float32)
                k_ref = k.to(torch.float32)
                v_ref = v.to(torch.float32)
                ref_chunks = []
                for start in range(0, q_ref.shape[0], strict_ref_chunk_b):
                    end = min(start + strict_ref_chunk_b, q_ref.shape[0])
                    if strict_mask is None:
                        mask_chunk = None
                    elif strict_mask.ndim == 4 and strict_mask.shape[0] == q_ref.shape[0]:
                        mask_chunk = strict_mask[start:end]
                    else:
                        mask_chunk = strict_mask
                    if strict_rel_bias is None:
                        bias_chunk = mask_chunk
                    elif mask_chunk is None:
                        bias_chunk = strict_rel_bias
                    else:
                        bias_chunk = strict_rel_bias
                    ref_kwargs = {
                        "block_n": strict_ref_block_n,
                        "scale": 1.0,
                        "is_causal": False,
                        "attn_bias": bias_chunk,
                    }
                    if strict_rel_bias is not None and mask_chunk is not None:
                        # Keep huge shifted-window mask and relative bias as
                        # separate sources. elsa_strict_reference slices them
                        # per query/key tile, avoiding a full [B,H,N,N]
                        # materialization.
                        ref_kwargs["attn_bias_extra"] = mask_chunk
                    ref_chunks.append(
                        elsa_core.elsa_strict_reference(
                            q_ref[start:end],
                            k_ref[start:end],
                            v_ref[start:end],
                            **ref_kwargs,
                        )
                    )
                ref_out = ref_chunks[0] if len(ref_chunks) == 1 else torch.cat(ref_chunks, dim=0)
                return _format_strict_out(ref_out.to(strict_dtype))
            strict_bias = strict_rel_bias
            if strict_mask is not None:
                separate_bias_mask_env = os.environ.get("ELSA_SWIN_SEPARATE_BIAS_MASK", "auto").strip().lower()
                compact_mask_like = (
                    strict_mask.ndim == 2
                    or (
                        strict_mask.ndim == 4
                        and strict_mask.shape[1] == q.shape[1]
                    )
                )
                separate_bias_mask = separate_bias_mask_env in {"1", "true", "yes", "on", "force"}
                if separate_bias_mask_env == "auto":
                    separate_bias_mask = (
                        compact_mask_like
                        and strict_rel_bias is not None
                        and strict_mask.ndim in (2, 4)
                    )
                if strict_rel_bias is not None and strict_dtype in (torch.float16, torch.bfloat16):
                    strict_rel_bias = strict_rel_bias.to(strict_dtype)
                if (
                    strict_rel_bias is not None
                    and (configured_window_tokens >= 8192 or separate_bias_mask)
                    and strict_mask.ndim in (2, 4)
                ):
                    # Pass shifted masks as a separate compact source. The
                    # strict qblock kernels slice relative bias and mask
                    # independently per tile, avoiding a full [B,H,N,N] fusion.
                    mask_compact_nohead = strict_mask if strict_mask.ndim == 2 else strict_mask[:, 0]
                    if (
                        mask_compact_nohead.is_floating_point()
                        and strict_dtype in (torch.float16, torch.bfloat16)
                    ):
                        mask_compact_nohead = mask_compact_nohead.to(strict_dtype)
                    strict_bias = (strict_rel_bias, mask_compact_nohead)
                else:
                    if strict_mask.ndim != 2 and strict_dtype in (torch.float16, torch.bfloat16):
                        strict_mask = strict_mask.to(strict_dtype)
                    strict_bias = strict_mask if strict_bias is None else (strict_bias + strict_mask)
            elif strict_rel_bias is not None and strict_dtype in (torch.float16, torch.bfloat16):
                strict_bias = strict_rel_bias.to(strict_dtype)
            if strict_dtype in (torch.float16, torch.bfloat16):
                strict_fp16 = getattr(elsa_core, "can_triton_strict_core_fp16", None)
                if strict_fp16 is None:
                    raise RuntimeError("Strict fp16/bf16 Swin scan backend unavailable.")
                launch_cfg = self._strict_launch_cfg_enabled() or {}
                block_q = launch_cfg.get("block_q", launch_cfg.get("block_m"))
                block_n = launch_cfg.get("block_n")
                return _format_strict_out(strict_fp16(
                    q_scaled,
                    k,
                    v,
                    is_causal=False,
                    bias=strict_bias,
                    block_q=block_q,
                    block_n=block_n,
                ))
            launch_cfg = self._strict_launch_cfg_enabled() or {}
            block_q = launch_cfg.get("block_q", launch_cfg.get("block_m"))
            block_n = launch_cfg.get("block_n")
            return _format_strict_out(elsa_core.can_triton_strict_core_fp32(
                q_scaled,
                k,
                v,
                is_causal=False,
                bias=strict_bias,
                block_q=block_q,
                block_n=block_n,
            ))
        if backend in ("triton_full", "triton_full_fp32", "triton_full_turbo"):
            if not _allow_unstable_paths():
                raise RuntimeError(
                    f"{backend} is disabled in clean mode. "
                    "Set ELSA_ALLOW_UNSTABLE_PATHS=1 to force."
                )
            if elsa_swinv2_triton_full is None:
                raise RuntimeError("ELSA full-kernel Swin backend unavailable.")
            return elsa_swinv2_triton_full(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
            )
        if (
            self.enable_short_path
            and q.size(2) <= self.short_seq_threshold
            and backend == "pytorch"
        ):
            return elsa_core.ELSA_swinv2_pytorch_short(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                mask=mask,
            )
        return ELSA_swinv2_pytorch(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias,
            mask=mask,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        
        Returns:
            output features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        strict_out_nh = False
        if self.backend_preference == "strict_core_ref" and (not self.training) and (not torch.is_grad_enabled()):
            strict_out_nh = self._strict_out_nh_enabled(N, x.dtype)

        if self._strict_eval_compile_enabled(x, mask):
            compiled = self._get_strict_eval_compiled(has_mask=mask is not None)
            if compiled is not None:
                try:
                    if mask is None:
                        return compiled(x)
                    return compiled(x, mask)
                except Exception:
                    self._strict_eval_compile_disabled = True
        if self._strict_eval_direct_enabled(x, mask):
            return self._strict_eval_fullpath_impl(x, mask)
        
        # QKV projection with bias
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        
        # Reshape to (B_, H, N, D). For strict train kernels, optionally pack
        # all QKV lanes once instead of letting the backend make three separate
        # contiguous copies after normalization.
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        if (
            self.training
            and torch.is_grad_enabled()
            and self.backend_preference == "strict_core_ref"
            and self._strict_qkv_prepack_enabled(dtype=x.dtype, n_tokens=N, is_cuda=x.is_cuda)
        ):
            qkv = qkv.contiguous()
        q, k, v = qkv.unbind(0)  # Each is (B_, H, N, D)
        
        # Normalize Q and K for cosine attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Compute relative position bias
        cache_ok = self._cache_enabled()
        rel_bias_key = (q.device, q.dtype)
        relative_position_bias = None
        if cache_ok:
            relative_position_bias = self._rel_bias_cache.get(rel_bias_key)
        if relative_position_bias is None:
            compact_ok = self.backend_preference == "strict_core_ref"
            relative_position_bias = self._relative_position_bias(compact_ok=compact_ok)
            if cache_ok:
                self._rel_bias_cache[rel_bias_key] = relative_position_bias
        
        # Clamp logit scale
        logit_scale = None
        if cache_ok:
            logit_scale = self._logit_scale_cache.get(rel_bias_key)
        if logit_scale is None:
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01))
            if cache_ok:
                # Cache a detached clone so compiled eval paths do not reuse a
                # graph-owned tensor that may be invalidated by later runs.
                logit_scale = logit_scale.detach().clone()
                self._logit_scale_cache[rel_bias_key] = logit_scale
        
        # Handle window mask
        mask_full = None
        mask_compact = None
        mask_compact_batched = None
        mask_full_key = None
        num_win = None
        mask_cache_ok = False
        if mask is not None:
            num_win = mask.shape[0]
            mask_cache_ok = cache_ok or bool(int(os.environ.get("ELSA_SWIN_MASK_CACHE_TRAIN", "1")))
            try:
                strict_compact_max_n = int(
                    os.environ.get(
                        "ELSA_SWIN_STRICT_COMPACT_MASK_MAX_N",
                        str(self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64),
                    )
                )
            except ValueError:
                strict_compact_max_n = self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64
            configured_window_tokens = int(self.configured_window_size[0] * self.configured_window_size[1])
            if mask.ndim == 2:
                mask_compact = mask
                use_compact = True
            else:
                use_compact = (
                    (
                        (self.enable_triton and self.backend_preference.startswith("triton"))
                        or (
                            self.backend_preference == "strict_core_ref"
                            and (
                                N <= max(64, strict_compact_max_n)
                                or configured_window_tokens > 256
                            )
                        )
                    )
                    and self._strict_use_compact_mask_enabled()
                )
            if use_compact:
                if mask.ndim != 2:
                    mask_key = ("compact", mask.data_ptr(), q.device, q.dtype, self.num_heads, N)
                    if mask_cache_ok:
                        mask_compact = self._mask_cache.get(mask_key)
                    if mask_compact is None:
                        # (num_win, N, N) -> (num_win, H, N, N) without materializing repeats
                        mask_compact = mask.view(num_win, N, N).unsqueeze(1).expand(num_win, self.num_heads, N, N)
                        if mask_cache_ok:
                            self._mask_cache[mask_key] = mask_compact
            else:
                mask_full_key = ("full", mask.data_ptr(), q.device, q.dtype, B_, N)
        
        def _mask_for_backend(backend_name: str) -> Optional[torch.Tensor]:
            nonlocal mask_full, mask_compact_batched
            if mask is None:
                return None
            try:
                strict_compact_max_n = int(
                    os.environ.get(
                        "ELSA_SWIN_STRICT_COMPACT_MASK_MAX_N",
                        str(self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64),
                    )
                )
            except ValueError:
                strict_compact_max_n = self.strict_compact_mask_max_n if self.strict_compact_mask_max_n is not None else 64
            configured_window_tokens = int(self.configured_window_size[0] * self.configured_window_size[1])
            if (
                backend_name.startswith("triton")
                or (
                    backend_name == "strict_core_ref"
                    and (
                        N <= max(64, strict_compact_max_n)
                        or configured_window_tokens > 256
                    )
                )
            ) and mask_compact is not None:
                if backend_name == "strict_core_ref":
                    # Strict-core can consume repeated compact windows directly.
                    # Avoid expanding to (B_, H, N, N), which only increases
                    # peak memory on shifted-window full-model eval.
                    return mask_compact
                if B_ == num_win:
                    return mask_compact
                if mask_compact_batched is None:
                    if mask_compact.ndim == 2:
                        return mask_compact
                    mask_compact_batched = (
                        mask_compact.unsqueeze(0)
                        .expand(B_ // num_win, num_win, self.num_heads, N, N)
                        .reshape(B_, self.num_heads, N, N)
                    )
                return mask_compact_batched
            if mask_full is None:
                if mask_full_key is not None and mask_cache_ok:
                    mask_full = self._mask_cache.get(mask_full_key)
                if mask_full is None:
                    if mask.ndim == 2:
                        raise RuntimeError("Label masks are only supported by strict compact backends.")
                    mask_full = mask.view(1, num_win, 1, N, N)
                    mask_full = mask_full.expand(B_ // num_win, num_win, 1, N, N)
                    mask_full = mask_full.reshape(B_, 1, N, N)
                    if mask_full_key is not None and mask_cache_ok:
                        self._mask_cache[mask_full_key] = mask_full
            return mask_full

        def _bias_and_mask_for_backend(backend_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            backend_bias = relative_position_bias
            backend_mask = _mask_for_backend(backend_name)
            if (
                backend_name == "strict_core_ref"
                and backend_mask is mask_compact
                and backend_mask is not None
                and backend_mask.ndim != 2
                and backend_bias is not None
                and self._strict_fuse_compact_bias_enabled()
            ):
                fused = None
                fused_key = None
                fused_cache_ok = mask_cache_ok and not (
                    torch.is_grad_enabled() and backend_bias.requires_grad
                )
                if cache_ok:
                    # The compact mask itself is a cheap cached view. The
                    # materialized compact+bias tensor is not: keeping it alive
                    # across warmup inflates eval peak memory for shifted-window
                    # full-model runs. Rebuild it each iteration unless the
                    # cache is explicitly re-enabled.
                    fused_cache_ok = fused_cache_ok and self._strict_eval_fused_mask_cache_enabled()
                if fused_cache_ok:
                    fused_key = (
                        "compact_add",
                        mask.data_ptr(),
                        backend_bias.data_ptr(),
                        q.device,
                        q.dtype,
                        self.num_heads,
                        N,
                    )
                    fused = self._mask_cache.get(fused_key)
                if fused is None:
                    fused = backend_mask + backend_bias.unsqueeze(0)
                    if fused_cache_ok and fused_key is not None:
                        self._mask_cache[fused_key] = fused
                backend_bias = None
                backend_mask = fused
            return backend_bias, backend_mask

        attn_output = None
        used_backend = None
        used_out_layout = "HND"
        errors: List[RuntimeError] = []
        direct_backend = None
        if self.training and torch.is_grad_enabled():
            if self.backend_preference == "strict_core_ref":
                direct_backend = "strict_core_ref"
            else:
                direct_backend = "pytorch" if self.backend_preference == "pytorch" else "swin_train_kernel"
        if direct_backend is not None:
            try:
                out_layout = "NH" if (direct_backend == "strict_core_ref" and strict_out_nh) else "HND"
                backend_bias, backend_mask = _bias_and_mask_for_backend(direct_backend)
                attn_output = self._run_backend(
                    direct_backend,
                    q,
                    k,
                    v,
                    logit_scale,
                    backend_bias,
                    backend_mask,
                    out_layout=out_layout,
                )
                used_backend = direct_backend
                used_out_layout = out_layout
            except RuntimeError as err:
                errors.append(err)
                self._warn_backend_failure(direct_backend, err)

        if attn_output is None:
            for backend in self._candidate_backends(q):
                if backend == direct_backend:
                    continue
                try:
                    out_layout = "NH" if (backend == "strict_core_ref" and strict_out_nh) else "HND"
                    backend_bias, backend_mask = _bias_and_mask_for_backend(backend)
                    attn_output = self._run_backend(
                        backend,
                        q,
                        k,
                        v,
                        logit_scale,
                        backend_bias,
                        backend_mask,
                        out_layout=out_layout,
                    )
                    used_backend = backend
                    used_out_layout = out_layout
                    break
                except RuntimeError as err:
                    errors.append(err)
                    self._warn_backend_failure(backend, err)

        if attn_output is None:
            raise RuntimeError("ELSA Swin attention failed.") from (errors[-1] if errors else None)
        
        # Apply dropout
        if self.training and self.attn_drop.p:
            attn_output = self.attn_drop(attn_output)

        # Reshape back to (B_, N, C)
        if used_backend == "strict_core_ref" and used_out_layout == "NH":
            x = attn_output.reshape(B_, N, C)
        else:
            x = attn_output.transpose(1, 2).reshape(B_, N, C)
        
        # Output projection
        x = self.proj(x)
        if self.training and self.proj_drop.p:
            x = self.proj_drop(x)
        
        return x




class SwinELSABlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: _int_or_tuple_2_t = 0,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: LayerType = "gelu",
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
        triton_matmul=False,
        triton= False,
        backend: Optional[str] = None,
        strict_launch_cfg: Optional[dict] = None,
        strict_launch_cfg_train: Optional[dict] = None,
        strict_launch_cfg_eval: Optional[dict] = None,
        strict_compact_mask_max_n: Optional[int] = None,
        strict_use_compact_mask: Optional[bool] = None,
        strict_use_compact_mask_train: Optional[bool] = None,
        strict_use_compact_mask_eval: Optional[bool] = None,
        strict_force_out_nh: Optional[bool] = None,
        strict_fuse_compact_bias: Optional[bool] = None,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            always_partition: Always partition into full windows and shift
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        backend = backend or get_default_elsa_backend()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.target_shift_size = to_2tuple(shift_size)  # store for later resize
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        act_layer = get_act_layer(act_layer)

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            configured_window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            triton_matmul=triton_matmul,
            triton=triton,
            backend=backend,
            strict_launch_cfg=strict_launch_cfg,
            strict_launch_cfg_train=strict_launch_cfg_train,
            strict_launch_cfg_eval=strict_launch_cfg_eval,
            strict_compact_mask_max_n=strict_compact_mask_max_n,
            strict_use_compact_mask=strict_use_compact_mask,
            strict_use_compact_mask_train=strict_use_compact_mask_train,
            strict_use_compact_mask_eval=strict_use_compact_mask_eval,
            strict_force_out_nh=strict_force_out_nh,
            strict_fuse_compact_bias=strict_fuse_compact_bias,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def get_attn_mask(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is None:
                img_mask = torch.zeros((1, *self.input_resolution, 1))  # 1 H W 1
            else:
                img_mask = torch.zeros((1, x.shape[1], x.shape[2], 1), dtype=x.dtype, device=x.device)  # 1 H W 1
            cnt = 0
            for h in (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
            ):
                for w in (
                        (0, -self.window_size[1]),
                        (-self.window_size[1], -self.shift_size[1]),
                        (-self.shift_size[1], None),
                ):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            try:
                label_mask_min_n = max(0, int(os.environ.get("ELSA_SWIN_LABEL_MASK_MIN_N", "8192")))
            except ValueError:
                label_mask_min_n = 8192
            if label_mask_min_n > 0 and self.window_area >= label_mask_min_n:
                return mask_windows.to(torch.int16)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def _calc_window_shift(
            self,
            target_window_size: _int_or_tuple_2_t,
            target_shift_size: Optional[_int_or_tuple_2_t] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was active
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                # if there was previously a non-zero shift, recalculate based on current window_size
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: Tuple[int, int],
            always_partition: Optional[bool] = None,
    ):
        """ Updates the input resolution, window size.

        Args:
            feat_size (Tuple[int, int]): New input resolution
            window_size (int): New window size
            always_partition: Change always_partition attribute if not None
        """
        # Update input resolution
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.set_window_size(self.window_size)
        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
        )

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, (Hp, Wp))  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x + self.drop_path1(self.norm1(self._attn(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape

        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinELSAStage(nn.Module):
    """ A Swin Transformer V2 Stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int,
            num_heads: int,
            window_size: _int_or_tuple_2_t,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            downsample: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_2_t = 0,
            output_nchw: bool = False,
        triton_matmul=False,
        triton= False,
        backend: Optional[str] = None,
        strict_launch_cfg: Optional[dict] = None,
        strict_launch_cfg_train: Optional[dict] = None,
        strict_launch_cfg_eval: Optional[dict] = None,
        strict_compact_mask_max_n: Optional[int] = None,
        strict_use_compact_mask: Optional[bool] = None,
        strict_use_compact_mask_train: Optional[bool] = None,
        strict_use_compact_mask_eval: Optional[bool] = None,
        strict_force_out_nh: Optional[bool] = None,
        strict_fuse_compact_bias: Optional[bool] = None,
    ) -> None:
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            always_partition: Always partition into full windows and shift
            dynamic_mask: Create attention mask in forward based on current input size
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer type.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinELSABlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                always_partition=always_partition,
                dynamic_mask=dynamic_mask,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                triton_matmul=triton_matmul,
                triton=triton,
                backend=backend,
                strict_launch_cfg=strict_launch_cfg,
                strict_launch_cfg_train=strict_launch_cfg_train,
                strict_launch_cfg_eval=strict_launch_cfg_eval,
                strict_compact_mask_max_n=strict_compact_mask_max_n,
                strict_use_compact_mask=strict_use_compact_mask,
                strict_use_compact_mask_train=strict_use_compact_mask_train,
                strict_use_compact_mask_eval=strict_use_compact_mask_eval,
                strict_force_out_nh=strict_force_out_nh,
                strict_fuse_compact_bias=strict_fuse_compact_bias,
            )
            for i in range(depth)])

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ):
        """ Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            assert isinstance(self.downsample, PatchMerging)
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                checkpoint_fn = checkpoint.checkpoint if hasattr(checkpoint, "checkpoint") else checkpoint
                x = checkpoint_fn(blk, x)
            else:
                x = blk(x)
        return x

    def _init_respostnorm(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinELSA(nn.Module):
    """ Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2, 6, 2),
            num_heads: Tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 7,
            always_partition: bool = False,
            strict_img_size: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Callable = nn.LayerNorm,
            pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
        triton_matmul = False,
        triton = False,
        elsa_backend: Optional[str] = None,
            strict_launch_cfg: Optional[dict] = None,
            strict_launch_cfg_train: Optional[dict] = None,
            strict_launch_cfg_eval: Optional[dict] = None,
            strict_compact_mask_max_n: Optional[int] = None,
            strict_use_compact_mask: Optional[bool] = None,
            strict_use_compact_mask_train: Optional[bool] = None,
            strict_use_compact_mask_eval: Optional[bool] = None,
            strict_force_out_nh: Optional[bool] = None,
            strict_fuse_compact_bias: Optional[bool] = None,
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer stage (layer).
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Head dropout rate.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            act_layer: Activation layer type.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
        """
        super().__init__()

        self.num_classes = num_classes
        assert global_pool in ('', 'avg')
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = self.head_hidden_size = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            strict_img_size=strict_img_size,
            output_fmt='NHWC',
        )
        grid_size = self.patch_embed.grid_size

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinELSAStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(grid_size[0] // scale, grid_size[1] // scale),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                window_size=window_size,
                always_partition=always_partition,
                dynamic_mask=not strict_img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_sizes[i],
                triton_matmul=triton_matmul,
                triton=triton,
                backend=elsa_backend or get_default_elsa_backend(),
                strict_launch_cfg=strict_launch_cfg,
                strict_launch_cfg_train=strict_launch_cfg_train,
                strict_launch_cfg_eval=strict_launch_cfg_eval,
                strict_compact_mask_max_n=strict_compact_mask_max_n,
                strict_use_compact_mask=strict_use_compact_mask,
                strict_use_compact_mask_train=strict_use_compact_mask_train,
                strict_use_compact_mask_eval=strict_use_compact_mask_eval,
                strict_force_out_nh=strict_force_out_nh,
                strict_fuse_compact_bias=strict_fuse_compact_bias,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim, reduction=4 * scale, module=f'layers.{i}')]

        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
            window_size: Optional[Tuple[int, int]] = None,
            window_ratio: Optional[int] = 8,
            always_partition: Optional[bool] = None,
    ):
        """Updates the image resolution, window size, and so the pair-wise relative positions.

        Args:
            img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            patch_size (Optional[Tuple[int, int]): New patch size, if None use current patch size
            window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            window_ratio (int): divisor for calculating window size from patch grid size
            always_partition: always partition / shift windows even if feat size is < window
        """
        if img_size is not None or patch_size is not None:
            self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
            grid_size = self.patch_embed.grid_size

        if window_size is None and window_ratio is not None:
            window_size = tuple([s // window_ratio for s in grid_size])

        for index, stage in enumerate(self.layers):
            stage_scale = 2 ** max(index - 1, 0)
            stage.set_input_size(
                feat_size=(grid_size[0] // stage_scale, grid_size[1] // stage_scale),
                window_size=window_size,
                always_partition=always_partition,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale")]):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^absolute_pos_embed|patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.layers), indices)

        # forward pass
        x = self.patch_embed(x)

        num_stages = len(self.layers)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.layers
        else:
            stages = self.layers[:max_index + 1]
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in take_indices:
                if norm and i == num_stages - 1:
                    x_inter = self.norm(x)  # applying final norm last intermediate
                else:
                    x_inter = x
                x_inter = x_inter.permute(0, 3, 1, 2).contiguous()
                intermediates.append(x_inter)

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.layers), indices)
        self.layers = self.layers[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    native_checkpoint = 'head.fc.weight' in state_dict
    out_dict = {}
    import re
    for k, v in state_dict.items():
        if any([n in k for n in ('relative_position_index', 'relative_coords_table', 'attn_mask')]):
            continue  # skip buffers that should not be persistent

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation='bicubic',
                    antialias=True,
                    verbose=True,
                )

        if not native_checkpoint:
            # skip layer remapping for updated checkpoints
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = k.replace('head.', 'head.fc.')
        out_dict[k] = v

    return out_dict


def _create_swin_elsa(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(kwargs.get('depths', (1, 1, 1, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinELSA, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256), 'pool_size': (8, 8),
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'elsa_base_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_base_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'elsa_base_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_base_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'elsa_large_window12to16_192to256.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_large_patch4_window12to16_192to256_22kto1k_ft.pth',
    ),
    'elsa_large_window12to24_192to384.ms_in22k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_large_patch4_window12to24_192to384_22kto1k_ft.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),

    'elsa_tiny_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_tiny_patch4_window8_256.pth',
    ),
    'elsa_tiny_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_tiny_patch4_window16_256.pth',
    ),
    'elsa_small_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_small_patch4_window8_256.pth',
    ),
    'elsa_small_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_small_patch4_window16_256.pth',
    ),
    'elsa_base_window8_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_base_patch4_window8_256.pth',
    ),
    'elsa_base_window16_256.ms_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_base_patch4_window16_256.pth',
    ),

    'elsa_base_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_base_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192), pool_size=(6, 6)
    ),
    'elsa_large_window12_192.ms_in22k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/SwinTransformer/storage/releases/download/v2.0.0/elsa_large_patch4_window12_192_22k.pth',
        num_classes=21841, input_size=(3, 192, 192), pool_size=(6, 6)
    ),
})


@register_model
@register_model
def elsa_tiny_window16_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=16,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strict_compact_mask_max_n=256,
        strict_use_compact_mask_train=None,
        strict_use_compact_mask_eval=True,
        strict_fuse_compact_bias=False,
        strict_launch_cfg_train={},
        strict_launch_cfg_eval={
            "block_q": 16,
            "block_n": 64,
            "num_warps": 8,
            "num_stages": 1,
        },
    )
    return _create_swin_elsa(
        'elsa_tiny_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_tiny_window8_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=8,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strict_force_out_nh=True,
        # Tiny W8 train/eval regimes use different mask materialization policies.
        # Leave train compactness policy-owned so fwd/bwd can keep shifted masks
        # as a separate compact source instead of materializing full BxHxNxN bias.
        strict_use_compact_mask_train=None,
        strict_use_compact_mask_eval=True,
        strict_fuse_compact_bias=None,
    )
    return _create_swin_elsa(
        'elsa_tiny_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_small_window16_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=16,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        strict_compact_mask_max_n=256,
        strict_force_out_nh=True,
        strict_fuse_compact_bias=True,
        strict_launch_cfg={
            "block_q": 32,
            "block_n": 128,
            "num_warps": 8,
            "num_stages": 1,
        },
    )
    return _create_swin_elsa(
        'elsa_small_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_small_window8_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(window_size=8, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
    return _create_swin_elsa(
        'elsa_small_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_base_window16_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_elsa(
        'elsa_base_window16_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_base_window8_256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=8,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        strict_force_out_nh=True,
        strict_use_compact_mask=False,
        strict_fuse_compact_bias=False,
    )
    return _create_swin_elsa(
        'elsa_base_window8_256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_base_window12_192(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    return _create_swin_elsa(
        'elsa_base_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_base_window12to16_192to256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=16, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_elsa(
        'elsa_base_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_base_window12to24_192to384(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=24, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_elsa(
        'elsa_base_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_large_window12_192(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
    return _create_swin_elsa(
        'elsa_large_window12_192', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_large_window12to16_192to256(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=16, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_elsa(
        'elsa_large_window12to16_192to256', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
@register_model
def elsa_large_window12to24_192to384(pretrained=False, **kwargs) -> SwinELSA:
    """
    """
    model_args = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_elsa(
        'elsa_large_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'elsa_base_window12_192_22k': 'elsa_base_window12_192.ms_in22k',
    'elsa_base_window12to16_192to256_22kft1k': 'elsa_base_window12to16_192to256.ms_in22k_ft_in1k',
    'elsa_base_window12to24_192to384_22kft1k': 'elsa_base_window12to24_192to384.ms_in22k_ft_in1k',
    'elsa_large_window12_192_22k': 'elsa_large_window12_192.ms_in22k',
    'elsa_large_window12to16_192to256_22kft1k': 'elsa_large_window12to16_192to256.ms_in22k_ft_in1k',
    'elsa_large_window12to24_192to384_22kft1k': 'elsa_large_window12to24_192to384.ms_in22k_ft_in1k',
})
