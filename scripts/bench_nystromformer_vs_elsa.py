#!/usr/bin/env python3
"""Benchmark Nyströmformer attention vs ELSA attention (FP32, no TF32)."""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Callable, Dict, List, Tuple

import torch

from timm.models.elsa import ElsaAttention

try:
    from transformers import NystromformerConfig
    from transformers.models.nystromformer.modeling_nystromformer import NystromformerAttention
except Exception as exc:  # pragma: no cover
    raise SystemExit("transformers is required to run this benchmark") from exc


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Nyströmformer vs ELSA attention (FP32).")
    parser.add_argument("--model-id", type=str, default="LennartKeller/nystromformer-gottbert-base-8192")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=str, default="artifacts/current/rerun/nystromformer_vs_elsa_fp32.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    _disable_tf32()

    torch.manual_seed(0)
    # Load Nyströmformer config so we match official dimensions.
    config = NystromformerConfig.from_pretrained(args.model_id)
    if args.seq_len != config.max_position_embeddings:
        raise SystemExit(
            f"seq_len must equal max_position_embeddings ({config.max_position_embeddings}) for Nyströmformer"
        )
    # HF Nyströmformer expects segment_means_seq_len to match the input length.
    config.segment_means_seq_len = args.seq_len

    nystrom_attn = NystromformerAttention(config).eval().to(device)

    dim = config.hidden_size
    heads = config.num_attention_heads
    hidden = torch.randn(args.batch, args.seq_len, dim, device=device, dtype=torch.float32)

    elsa_backends = [
        ("ELSA-triton-fp32", "triton_fp32"),
        ("SDPA-math", "sdpa_math"),
        ("SDPA-mem", "sdpa_mem"),
    ]

    results: List[Dict[str, object]] = []

    for label, backend in elsa_backends:
        attn = ElsaAttention(
            dim=dim,
            num_heads=heads,
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
        out.update({"seq_len": args.seq_len, "batch": args.batch, "dim": dim, "heads": heads})
        results.append(out)

    def run_nystrom() -> torch.Tensor:
        with torch.inference_mode():
            return nystrom_attn(hidden, attention_mask=None)[0]

    out = _run_backend("Nyströmformer-Attn", run_nystrom, args.warmup, args.iters)
    out.update({"seq_len": args.seq_len, "batch": args.batch, "dim": dim, "heads": heads})
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
