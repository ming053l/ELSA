from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .attention import twopass_attention


AttentionBackend = Literal["elsa", "sdpa"]
SdpaKind = Literal["auto", "math", "flash", "mem"]


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool, kind: SdpaKind) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    if kind == "auto":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale)
    if kind == "math":
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale)
    if kind == "flash":
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale)
    if kind == "mem":
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale)
    raise ValueError(f"unknown SDPA kind: {kind}")


def _full_model_elsa_kwargs(q: torch.Tensor, explicit: dict) -> dict:
    kwargs = dict(explicit)
    kwargs["algorithm"] = "paper_scan"
    return kwargs


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        backend: AttentionBackend,
        sdpa_kind: SdpaKind = "auto",
        is_causal: bool = False,
        qkv_bias: bool = True,
        elsa_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.backend = backend
        self.sdpa_kind = sdpa_kind
        self.is_causal = bool(is_causal)
        self.elsa_kwargs = dict(elsa_kwargs or {})
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.backend == "elsa":
            y = twopass_attention(q, k, v, is_causal=self.is_causal, **_full_model_elsa_kwargs(q, self.elsa_kwargs))
        elif self.backend == "sdpa":
            y = _sdpa(q, k, v, is_causal=self.is_causal, kind=self.sdpa_kind)
        else:
            raise ValueError(f"unknown attention backend: {self.backend}")

        y = y.transpose(1, 2).reshape(batch, seq_len, channels)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        mlp_ratio: float = 4.0,
        backend: AttentionBackend,
        sdpa_kind: SdpaKind = "auto",
        is_causal: bool = False,
        elsa_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(round(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim,
            heads,
            backend=backend,
            sdpa_kind=sdpa_kind,
            is_causal=is_causal,
            elsa_kwargs=elsa_kwargs,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Minimal full-model Transformer encoder for fair SDPA-vs-ELSA studies."""

    def __init__(
        self,
        *,
        dim: int = 192,
        depth: int = 1,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        backend: AttentionBackend,
        sdpa_kind: SdpaKind = "auto",
        is_causal: bool = False,
        elsa_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    mlp_ratio=mlp_ratio,
                    backend=backend,
                    sdpa_kind=sdpa_kind,
                    is_causal=is_causal,
                    elsa_kwargs=elsa_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def make_model_pair(
    *,
    dim: int = 192,
    depth: int = 1,
    heads: int = 3,
    mlp_ratio: float = 4.0,
    sdpa_kind: SdpaKind = "auto",
    is_causal: bool = False,
    elsa_kwargs: Optional[dict] = None,
) -> tuple[TransformerEncoder, TransformerEncoder]:
    baseline = TransformerEncoder(
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        backend="sdpa",
        sdpa_kind=sdpa_kind,
        is_causal=is_causal,
    )
    elsa = TransformerEncoder(
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        backend="elsa",
        is_causal=is_causal,
        elsa_kwargs=elsa_kwargs,
    )
    elsa.load_state_dict(baseline.state_dict())
    return baseline, elsa
