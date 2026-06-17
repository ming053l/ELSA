from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .attention import twopass_attention


@dataclass(frozen=True)
class PatchReport:
    vit_attention: int = 0
    swin_window_attention: int = 0

    @property
    def total(self) -> int:
        return self.vit_attention + self.swin_window_attention


class ElsaTimmVitAttention(nn.Module):
    def __init__(self, source: nn.Module, *, elsa_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        self.num_heads = int(source.num_heads)
        self.head_dim = int(source.head_dim)
        # newer timm exposes attn_dim; older ViT Attention does not -> derive it.
        self.attn_dim = int(getattr(source, "attn_dim", self.num_heads * self.head_dim))
        self.qkv = copy.deepcopy(source.qkv)
        self.q_norm = copy.deepcopy(source.q_norm)
        self.k_norm = copy.deepcopy(source.k_norm)
        # newer timm has a pre-proj norm; older ViT does not -> Identity.
        self.norm = copy.deepcopy(source.norm) if hasattr(source, "norm") else nn.Identity()
        self.proj = copy.deepcopy(source.proj)
        self.proj_drop = copy.deepcopy(source.proj_drop)
        self.elsa_kwargs = dict(elsa_kwargs or {})

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError("ELSA ViT drop-in currently supports unmasked attention only")
        batch, seq_len, _ = x.shape
        # Op-fusion: make the stacked qkv contiguous ONCE so each unbound q/k/v slice is
        # already contiguous and twopass_attention's internal .contiguous() become no-ops
        # (was 3 separate per-layer copies — a measurable full-model overhead at short seq).
        qkv = (
            self.qkv(x)
            .reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        y = twopass_attention(q, k, v, **self.elsa_kwargs)
        y = y.transpose(1, 2).reshape(batch, seq_len, self.attn_dim)
        y = self.norm(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class ElsaTimmSwinWindowAttention(nn.Module):
    def __init__(self, source: nn.Module, *, elsa_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        self.dim = int(source.dim)
        self.window_size = tuple(int(x) for x in source.window_size)
        self.window_area = int(source.window_area)
        self.num_heads = int(source.num_heads)
        self.qkv = copy.deepcopy(source.qkv)
        self.attn_drop = copy.deepcopy(source.attn_drop)
        self.proj = copy.deepcopy(source.proj)
        self.proj_drop = copy.deepcopy(source.proj_drop)
        self.softmax = copy.deepcopy(source.softmax)
        self.scale = float(source.scale)
        self.head_dim = int(self.qkv.out_features // (3 * self.num_heads))
        self.attn_dim = self.head_dim * self.num_heads
        self.relative_position_bias_table = nn.Parameter(source.relative_position_bias_table.detach().clone())
        self.register_buffer(
            "relative_position_index",
            source.relative_position_index.detach().clone(),
            persistent=False,
        )
        self.elsa_kwargs = dict(elsa_kwargs or {})

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area, self.window_area, self.num_heads
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def set_window_size(self, window_size: tuple[int, int]) -> None:
        from timm.models.swin_transformer import get_relative_position_index, resize_rel_pos_bias_table
        from timm.models.swin_transformer import to_2tuple

        window_size = to_2tuple(window_size)
        if window_size == self.window_size:
            return
        self.window_size = tuple(int(x) for x in window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        with torch.no_grad():
            new_bias_shape = (2 * win_h - 1) * (2 * win_w - 1), self.num_heads
            self.relative_position_bias_table = nn.Parameter(
                resize_rel_pos_bias_table(
                    self.relative_position_bias_table,
                    new_window_size=self.window_size,
                    new_bias_shape=new_bias_shape,
                )
            )
            self.register_buffer(
                "relative_position_index",
                get_relative_position_index(win_h, win_w, device=self.relative_position_bias_table.device),
                persistent=False,
            )

    def _elsa_kwargs_for_window(self, seq_len: int) -> dict:
        kwargs = dict(self.elsa_kwargs)
        kwargs["algorithm"] = "paper_scan"
        kwargs.setdefault("block_m", 64)
        kwargs.setdefault("block_n", 64)
        kwargs.setdefault("q_chunk_size", seq_len)
        return kwargs

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.training and self.attn_drop.p:
            raise ValueError("ELSA Swin drop-in forward is inference-only when attention dropout is nonzero")

        batch_windows, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_windows, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        canonical_scale = self.head_dim ** -0.5
        if abs(self.scale - canonical_scale) > 1e-12:
            q = q * (self.scale / canonical_scale)

        bias = self._get_rel_pos_bias()
        if mask is not None:
            num_win = int(mask.shape[0])
            mask_bias = mask.view(1, num_win, 1, seq_len, seq_len).expand(
                batch_windows // num_win, -1, self.num_heads, -1, -1
            )
            mask_bias = mask_bias.reshape(-1, self.num_heads, seq_len, seq_len)
            bias = bias + mask_bias

        y = twopass_attention(q, k, v, bias=bias, **self._elsa_kwargs_for_window(seq_len))
        y = y.transpose(1, 2).reshape(batch_windows, seq_len, self.attn_dim)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


def _is_vit_attention(module: nn.Module) -> bool:
    # 'norm' and 'attn_dim' exist only in newer timm ViT Attention; derive/Identity
    # them in the wrapper so older forks (e.g. SICNet timm) still patch.
    required = ("qkv", "q_norm", "k_norm", "proj", "num_heads", "head_dim")
    return all(hasattr(module, name) for name in required)


def _is_swin_v1_window_attention(module: nn.Module) -> bool:
    required = (
        "qkv",
        "proj",
        "relative_position_bias_table",
        "relative_position_index",
        "window_area",
        "window_size",
        "scale",
        "num_heads",
        "dim",
    )
    return all(hasattr(module, name) for name in required) and not hasattr(module, "logit_scale")


def patch_timm_attention(model: nn.Module, *, elsa_kwargs: Optional[dict] = None) -> PatchReport:
    vit_count = 0
    swin_count = 0

    def visit(parent: nn.Module) -> None:
        nonlocal vit_count, swin_count
        for name, child in list(parent.named_children()):
            if _is_vit_attention(child):
                setattr(parent, name, ElsaTimmVitAttention(child, elsa_kwargs=elsa_kwargs))
                vit_count += 1
            elif _is_swin_v1_window_attention(child):
                setattr(parent, name, ElsaTimmSwinWindowAttention(child, elsa_kwargs=elsa_kwargs))
                swin_count += 1
            else:
                visit(child)

    visit(model)
    if vit_count + swin_count == 0:
        raise ValueError("no compatible timm ViT or Swin v1 attention modules were patched")
    return PatchReport(vit_attention=vit_count, swin_window_attention=swin_count)
