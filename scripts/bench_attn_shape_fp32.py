#!/usr/bin/env python3
"""Shape-controlled FP32 attention benchmark (ELSA vs SDPA)."""
from __future__ import annotations

import argparse
import csv
import os
from contextlib import contextmanager
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from timm.models.elsa_triton import ELSA_triton_fp32


def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def _bench(fn, warmup: int, iters: int) -> Tuple[float, float]:
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
    ms = start.elapsed_time(end) / max(1, iters)
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return ms, peak_gb


@contextmanager
def _sdpa_context(kind: str):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if kind == "math":
            ctx = sdpa_kernel([SDPBackend.MATH])
        elif kind == "mem":
            ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
        else:
            ctx = sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])
        with ctx:
            yield
        return
    except Exception:
        pass

    if kind == "math":
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            yield
    elif kind == "mem":
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
            yield
    else:
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=True):
            yield


def _run_backend(name: str, fn, warmup: int, iters: int) -> Dict[str, object]:
    try:
        ms, peak = _bench(fn, warmup, iters)
        return {"backend": name, "status": "ok", "latency_ms": ms, "peak_gb": peak}
    except RuntimeError as err:
        status = "oom" if "out of memory" in str(err).lower() else "error"
        torch.cuda.empty_cache()
        return {"backend": name, "status": status, "latency_ms": None, "peak_gb": None}


def _parse_seq_lens(values: str) -> Iterable[int]:
    return [int(x) for x in values.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Shape-controlled FP32 attention benchmark.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-lens", type=str, default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    _disable_tf32()
    torch.manual_seed(0)
    device = torch.device("cuda")

    b = args.batch
    h = args.heads
    d = args.head_dim
    seq_lens = list(_parse_seq_lens(args.seq_lens))

    out_path = args.out or f"artifacts/current/rerun/shape_fp32_b{b}_h{h}_d{d}.csv"
    rows = []

    for n in seq_lens:
        q = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
        k = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
        v = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
        scale = 1.0 / (d ** 0.5)

        def run_elsa():
            with torch.inference_mode():
                return ELSA_triton_fp32.apply(q, k, v, scale)

        def run_sdpa_math():
            with torch.inference_mode(), _sdpa_context("math"):
                return F.scaled_dot_product_attention(q, k, v)

        def run_sdpa_mem():
            with torch.inference_mode(), _sdpa_context("mem"):
                return F.scaled_dot_product_attention(q, k, v)

        for name, fn in [
            ("ELSA-triton-fp32", run_elsa),
            ("SDPA-math", run_sdpa_math),
            ("SDPA-mem", run_sdpa_mem),
        ]:
            out = _run_backend(name, fn, args.warmup, args.iters)
            out.update(
                {
                    "batch": b,
                    "heads": h,
                    "head_dim": d,
                    "seq_len": n,
                    "dim": h * d,
                }
            )
            rows.append(out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "status",
                "latency_ms",
                "peak_gb",
                "batch",
                "heads",
                "head_dim",
                "seq_len",
                "dim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
