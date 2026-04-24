#!/usr/bin/env python3
"""Benchmark Gated DeltaNet (2025) vs ELSA attention (FP32)."""
from __future__ import annotations

import argparse
import csv
import os
from typing import Callable, Dict, List, Tuple

import requests
import torch

from timm.models.elsa import ElsaAttention
from fla.layers.gated_deltanet import GatedDeltaNet


def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> Tuple[float, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gated DeltaNet vs ELSA attention (FP32).")
    parser.add_argument("--model-id", type=str, default="Idiap/gated-deltanet-attn-1.4B-30B")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=str, default="artifacts/current/rerun/gated_deltanet_vs_elsa_fp32.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    _disable_tf32()
    torch.manual_seed(0)

    cfg = _load_config(args.model_id)
    max_len = int(cfg.get("max_position_embeddings", args.seq_len))
    if args.seq_len > max_len:
        raise SystemExit(f"seq_len ({args.seq_len}) exceeds max_position_embeddings ({max_len})")

    hidden_size = int(cfg["hidden_size"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    expand_v = float(cfg.get("expand_v", 1.0))
    use_gate = bool(cfg.get("use_gate", True))
    use_short_conv = bool(cfg.get("use_short_conv", True))
    conv_size = int(cfg.get("conv_size", 4))
    mode = str(cfg.get("attn_mode", "chunk"))

    # ELSA FP32 Triton kernel needs smaller blocks for head_dim>=256 on A100.
    if head_dim >= 256:
        os.environ.setdefault("ELSA_TRITON_FWD_BLOCK_Q", "32")
        os.environ.setdefault("ELSA_TRITON_FWD_BLOCK_N", "32")
        os.environ.setdefault("ELSA_TRITON_FWD_WARPS", "2")
        os.environ.setdefault("ELSA_TRITON_FWD_STAGES", "1")

    hidden = torch.randn(args.batch, args.seq_len, hidden_size, device=device, dtype=torch.float32)

    elsa_backends = [
        ("ELSA-triton-fp32", "triton_fp32"),
        ("SDPA-math", "sdpa_math"),
        ("SDPA-mem", "sdpa_mem"),
    ]

    results: List[Dict[str, object]] = []

    for label, backend in elsa_backends:
        attn = ElsaAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            qk_norm=False,
            backend=backend,
        ).eval().to(device)

        def run_elsa() -> torch.Tensor:
            with torch.inference_mode():
                return attn(hidden)

        out = _run_backend(label, run_elsa, args.warmup, args.iters)
        out.update({"seq_len": args.seq_len, "batch": args.batch, "dim": hidden_size, "heads": num_heads})
        results.append(out)

    gdn = GatedDeltaNet(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        expand_v=expand_v,
        mode=mode,
        use_gate=use_gate,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        layer_idx=0,
    ).eval().to(device)

    def run_gdn() -> torch.Tensor:
        with torch.inference_mode():
            return gdn(hidden, attention_mask=None)[0]

    out = _run_backend("GatedDeltaNet", run_gdn, args.warmup, args.iters)
    out.update({"seq_len": args.seq_len, "batch": args.batch, "dim": hidden_size, "heads": num_heads})
    results.append(out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["backend", "status", "seq_len", "batch", "dim", "heads", "latency_ms", "peak_gb"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Wrote: {args.out}")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
