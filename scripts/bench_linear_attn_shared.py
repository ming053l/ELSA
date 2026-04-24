#!/usr/bin/env python3
"""Benchmark linear-time attention families vs ELSA/SDPA (FP32 only).

Runs either attention-only or shared-block (full-model) comparisons with a
fixed MLP/norm stack for fairness. Outputs CSV per run.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Callable, Dict, Optional

import requests
import torch
import torch.nn as nn

from timm.models.elsa import ElsaAttention

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.layers.gla import GatedLinearAttention
from fla.layers.delta_net import DeltaNet
from fla.layers.gsa import GatedSlotAttention
from fla.layers.hgrn2 import HGRN2Attention
from fla.layers.multiscale_retention import MultiScaleRetention
from fla.modules import RMSNorm

try:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_v2  # type: ignore
    FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_v2 = None  # type: ignore
    FLASH_ATTN_AVAILABLE = False


FAMILY_DEFAULT_MODEL = {
    "gated_deltanet": "Idiap/gated-deltanet-attn-0.4B-10B",
    "gated_deltaproduct": "msj19/gated_deltaproduct",
    "gla": "fla-hub/gla-340M-15B",
    "delta_net": "fla-hub/delta_net-1.3B-100B",
    "gsa": "fla-hub/gsa-7B-mistral-20B",
    "hgrn2": "fla-hub/hgrn2-1.3B-100B",
    "retnet": "fla-hub/retnet-1.3B-100B",
}


def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, int]:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, iters), torch.cuda.max_memory_allocated()


def _run_backend(name: str, fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Dict[str, object]:
    try:
        ms, mem = _bench(fn, warmup, iters)
        return {
            "backend": name,
            "status": "ok",
            "latency_ms": ms,
            "peak_gb": mem / (1024 ** 3),
        }
    except RuntimeError as err:
        msg = str(err).lower()
        status = "oom" if "out of memory" in msg else "error"
        torch.cuda.empty_cache()
        return {
            "backend": name,
            "status": status,
            "latency_ms": None,
            "peak_gb": None,
        }


def _load_config(model_id: str) -> Dict[str, object]:
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _call_attn(attn: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    try:
        out = attn(hidden, attention_mask=None, use_cache=False)
    except TypeError:
        out = attn(hidden)
    if isinstance(out, tuple):
        return out[0]
    return out


class SharedBlock(nn.Module):
    def __init__(
        self,
        attn: nn.Module,
        hidden_size: int,
        mlp_ratio: float,
        norm_eps: float,
        norm_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps, dtype=norm_dtype)
        self.attn = attn
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps, dtype=norm_dtype)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size, bias=False),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attn_out = _call_attn(self.attn, self.norm1(hidden))
        hidden = hidden + attn_out
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class FlashAttnBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if flash_attn_v2 is None:
            raise RuntimeError("flash-attn v2 is unavailable")
        bsz, seq_len, dim = hidden.shape
        qkv = self.qkv(hidden)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn_v2(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        out = out.reshape(bsz, seq_len, dim)
        return self.proj(out)


def _build_elsa(
    backend: str,
    hidden_size: int,
    num_heads: int,
) -> ElsaAttention:
    return ElsaAttention(
        dim=hidden_size,
        num_heads=num_heads,
        qkv_bias=False,
        proj_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        backend=backend,
    )


def _build_linear_attn(
    family: str,
    cfg: Dict[str, object],
    layer_idx: int,
) -> nn.Module:
    hidden_size = int(cfg["hidden_size"])
    attn_mode = cfg.get("attn_mode") or "chunk"
    if family == "gated_deltanet":
        num_heads = int(cfg["num_heads"])
        head_dim = int(cfg.get("head_dim") or (hidden_size // num_heads))
        return GatedDeltaNet(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            expand_v=float(cfg.get("expand_v", 1.0)),
            mode=attn_mode,
            use_gate=bool(cfg.get("use_gate", True)),
            use_short_conv=bool(cfg.get("use_short_conv", True)),
            conv_size=int(cfg.get("conv_size", 4)),
            layer_idx=layer_idx,
        )
    if family == "gated_deltaproduct":
        return GatedDeltaProduct(
            hidden_size=hidden_size,
            head_dim=int(cfg.get("head_dim", hidden_size // int(cfg["num_heads"]))),
            num_heads=int(cfg["num_heads"]),
            num_v_heads=cfg.get("num_v_heads"),
            expand_v=float(cfg.get("expand_v", 1.0)),
            mode=attn_mode,
            use_output_gate=bool(cfg.get("use_output_gate", True)),
            use_short_conv=bool(cfg.get("use_short_conv", True)),
            conv_size=int(cfg.get("conv_size", 4)),
            conv_bias=bool(cfg.get("conv_bias", False)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            use_forget_gate=bool(cfg.get("use_forget_gate", False)),
            allow_neg_eigval=bool(cfg.get("allow_neg_eigval", False)),
            num_householder=int(cfg.get("num_householder", 1)),
            layer_idx=layer_idx,
        )
    if family == "gla":
        return GatedLinearAttention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=float(cfg.get("expand_k", 0.5)),
            expand_v=float(cfg.get("expand_v", 1.0)),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=cfg.get("num_kv_heads"),
            feature_map=cfg.get("feature_map"),
            use_short_conv=bool(cfg.get("use_short_conv", False)),
            conv_size=int(cfg.get("conv_size", 4)),
            use_output_gate=bool(cfg.get("use_output_gate", True)),
            gate_fn=cfg.get("gate_fn", "swish"),
            elementwise_affine=bool(cfg.get("elementwise_affine", True)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            gate_logit_normalizer=int(cfg.get("gate_logit_normalizer", 16)),
            gate_low_rank_dim=int(cfg.get("gate_low_rank_dim", 16)),
            clamp_min=cfg.get("clamp_min"),
            fuse_norm=bool(cfg.get("fuse_norm", True)),
            layer_idx=layer_idx,
        )
    if family == "delta_net":
        return DeltaNet(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=float(cfg.get("expand_k", 1.0)),
            expand_v=float(cfg.get("expand_v", 1.0)),
            num_heads=int(cfg["num_heads"]),
            use_beta=bool(cfg.get("use_beta", True)),
            use_gate=bool(cfg.get("use_gate", False)),
            use_short_conv=bool(cfg.get("use_short_conv", True)),
            conv_size=int(cfg.get("conv_size", 4)),
            qk_activation=str(cfg.get("qk_activation", "silu")),
            qk_norm=str(cfg.get("qk_norm", "l2")),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            layer_idx=layer_idx,
        )
    if family == "gsa":
        return GatedSlotAttention(
            mode=attn_mode if attn_mode is not None else "chunk",
            hidden_size=hidden_size,
            expand_k=float(cfg.get("expand_k", 1.0)),
            expand_v=float(cfg.get("expand_v", 1.0)),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=cfg.get("num_kv_heads"),
            use_short_conv=bool(cfg.get("use_short_conv", False)),
            conv_size=int(cfg.get("conv_size", 4)),
            num_slots=cfg.get("num_slots"),
            feature_map=str(cfg.get("feature_map", "swish")),
            use_output_gate=bool(cfg.get("use_output_gate", False)),
            use_norm=bool(cfg.get("use_norm", True)),
            layer_idx=layer_idx,
        )
    if family == "hgrn2":
        return HGRN2Attention(
            mode=attn_mode,
            hidden_size=hidden_size,
            num_heads=cfg.get("num_heads"),
            expand_ratio=cfg.get("expand_ratio", 128),
            use_short_conv=bool(cfg.get("use_short_conv", False)),
            conv_size=int(cfg.get("conv_size", 4)),
            elementwise_affine=bool(cfg.get("elementwise_affine", True)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            layer_idx=layer_idx,
        )
    if family == "retnet":
        return MultiScaleRetention(
            mode=attn_mode,
            hidden_size=hidden_size,
            expand_k=float(cfg.get("expand_k", 1.0)),
            expand_v=float(cfg.get("expand_v", 2.0)),
            num_heads=int(cfg["num_heads"]),
            num_kv_heads=cfg.get("num_kv_heads"),
            feature_map=cfg.get("feature_map"),
            use_short_conv=bool(cfg.get("use_short_conv", False)),
            conv_size=int(cfg.get("conv_size", 4)),
            conv_bias=bool(cfg.get("conv_bias", False)),
            use_output_gate=bool(cfg.get("use_output_gate", True)),
            gate_fn=str(cfg.get("gate_fn", "swish")) if cfg.get("gate_fn") is not None else "swish",
            elementwise_affine=bool(cfg.get("elementwise_affine", True)),
            norm_eps=float(cfg.get("norm_eps", 1e-5)),
            fuse_norm=bool(cfg.get("fuse_norm", True)),
            layer_idx=layer_idx,
        )
    raise ValueError(f"Unknown family '{family}'")


def _resolve_num_heads(cfg: Dict[str, object]) -> int:
    num_heads = cfg.get("num_heads")
    if num_heads is None:
        expand_ratio = int(cfg.get("expand_ratio", 128))
        hidden = int(cfg["hidden_size"])
        num_heads = hidden // expand_ratio
    return int(num_heads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear attention vs ELSA/SDPA (FP32/FP16).")
    parser.add_argument("--family", type=str, choices=sorted(FAMILY_DEFAULT_MODEL), required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["attn_only", "full_model"], default="attn_only")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    _disable_tf32()
    torch.manual_seed(0)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model_id = args.model_id or FAMILY_DEFAULT_MODEL[args.family]
    cfg = _load_config(model_id)
    max_len = int(cfg.get("max_position_embeddings", 2048))
    seq_len = args.seq_len or min(max_len, 2048)
    if seq_len > max_len:
        raise SystemExit(f"seq_len ({seq_len}) exceeds max_position_embeddings ({max_len}) for {model_id}")

    hidden_size = int(cfg["hidden_size"])
    num_heads = _resolve_num_heads(cfg)
    norm_eps = float(cfg.get("norm_eps", 1e-5))
    head_dim = hidden_size // num_heads

    # ELSA FP32 Triton kernel needs smaller blocks for large head dims.
    use_autotune = os.environ.get("ELSA_TRITON_FWD_AUTOTUNE", "0") == "1"
    if head_dim >= 256 and not use_autotune:
        os.environ.setdefault("ELSA_TRITON_FWD_BLOCK_Q", "32")
        os.environ.setdefault("ELSA_TRITON_FWD_BLOCK_N", "32")
        os.environ.setdefault("ELSA_TRITON_FWD_WARPS", "2")
        os.environ.setdefault("ELSA_TRITON_FWD_STAGES", "1")

    hidden = torch.randn(args.batch, seq_len, hidden_size, device=device, dtype=dtype)

    results = []

    def make_linear(layer_idx: int) -> nn.Module:
        return _build_linear_attn(args.family, cfg, layer_idx).eval().to(device, dtype=dtype)

    def make_elsa(backend: str) -> nn.Module:
        return _build_elsa(backend, hidden_size, num_heads).eval().to(device, dtype=dtype)

    elsa_backend = "triton_fp32" if dtype == torch.float32 else "triton"
    elsa_label = "ELSA-triton-fp32" if dtype == torch.float32 else "ELSA-triton-fp16"
    if dtype == torch.float32 and use_autotune:
        elsa_label += "-autotune"
    if dtype == torch.float32 and head_dim >= 256:
        fast_env = os.environ.get("ELSA_TRITON_FP32_FAST")
        if fast_env is None or fast_env == "1":
            elsa_label += "-fast"
    if dtype == torch.float32:
        infer_env = os.environ.get("ELSA_TRITON_FP32_INFER", "1")
        if infer_env != "0" and "-fast" not in elsa_label:
            elsa_label += "-infer"
        splitd_env = os.environ.get("ELSA_TRITON_FP32_SPLITD", "0")
        if splitd_env == "1":
            elsa_label += "-splitd"

    if args.mode == "attn_only":
        backends = [
            (elsa_label, elsa_backend),
            ("SDPA-math", "sdpa_math"),
        ]
        if dtype == torch.float16 and FLASH_ATTN_AVAILABLE:
            backends.append(("FA2", "fa2"))
        else:
            backends.append(("SDPA-mem", "sdpa_mem"))

        for label, backend in backends:
            attn = FlashAttnBlock(hidden_size, num_heads).eval().to(device, dtype=dtype) if backend == "fa2" else make_elsa(backend)

            def run_elsa() -> torch.Tensor:
                with torch.inference_mode():
                    return attn(hidden)

            out = _run_backend(label, run_elsa, args.warmup, args.iters)
            out.update(
                {
                    "family": args.family,
                    "mode": args.mode,
                    "dtype": args.dtype,
                    "model_id": model_id,
                    "seq_len": seq_len,
                    "batch": args.batch,
                    "dim": hidden_size,
                    "heads": num_heads,
                    "layers": 0,
                    "mlp_ratio": args.mlp_ratio,
                }
            )
            results.append(out)

        linear = make_linear(0)

        def run_linear() -> torch.Tensor:
            with torch.inference_mode():
                return _call_attn(linear, hidden)

        out = _run_backend(args.family, run_linear, args.warmup, args.iters)
        out.update(
            {
                "family": args.family,
                "mode": args.mode,
                "dtype": args.dtype,
                "model_id": model_id,
                "seq_len": seq_len,
                "batch": args.batch,
                "dim": hidden_size,
                "heads": num_heads,
                "layers": 0,
                "mlp_ratio": args.mlp_ratio,
            }
        )
        results.append(out)
    else:
        def build_model(attn_factory: Callable[[int], nn.Module]) -> nn.Module:
            blocks = []
            for idx in range(args.layers):
                blocks.append(
                    SharedBlock(
                        attn_factory(idx),
                        hidden_size,
                        args.mlp_ratio,
                        norm_eps,
                        norm_dtype=dtype,
                    )
                )
            return nn.Sequential(*blocks)

        backends = [
            (elsa_label, elsa_backend),
            ("SDPA-math", "sdpa_math"),
        ]
        if dtype == torch.float16 and FLASH_ATTN_AVAILABLE:
            backends.append(("FA2", "fa2"))
        else:
            backends.append(("SDPA-mem", "sdpa_mem"))

        for label, backend in backends:
            if backend == "fa2":
                model = build_model(lambda _idx: FlashAttnBlock(hidden_size, num_heads)).eval().to(device, dtype=dtype)
            else:
                model = build_model(lambda _idx: make_elsa(backend)).eval().to(device, dtype=dtype)

            def run_model() -> torch.Tensor:
                with torch.inference_mode():
                    return model(hidden)

            out = _run_backend(label, run_model, args.warmup, args.iters)
            out.update(
                {
                    "family": args.family,
                    "mode": args.mode,
                    "dtype": args.dtype,
                    "model_id": model_id,
                    "seq_len": seq_len,
                    "batch": args.batch,
                    "dim": hidden_size,
                    "heads": num_heads,
                    "layers": args.layers,
                    "mlp_ratio": args.mlp_ratio,
                }
            )
            results.append(out)

        model = build_model(make_linear).eval().to(device, dtype=dtype)

        def run_linear_model() -> torch.Tensor:
            with torch.inference_mode():
                return model(hidden)

        out = _run_backend(args.family, run_linear_model, args.warmup, args.iters)
        out.update(
            {
                "family": args.family,
                "mode": args.mode,
                "dtype": args.dtype,
                "model_id": model_id,
                "seq_len": seq_len,
                "batch": args.batch,
                "dim": hidden_size,
                "heads": num_heads,
                "layers": args.layers,
                "mlp_ratio": args.mlp_ratio,
            }
        )
        results.append(out)

    out_path = args.out
    if out_path is None:
        out_path = (
            f"artifacts/current/rerun/linear_attn_{args.family}_{args.mode}_{args.dtype}_seq{seq_len}.csv"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "mode",
                "dtype",
                "model_id",
                "backend",
                "status",
                "seq_len",
                "batch",
                "dim",
                "heads",
                "layers",
                "mlp_ratio",
                "latency_ms",
                "peak_gb",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote: {out_path}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
