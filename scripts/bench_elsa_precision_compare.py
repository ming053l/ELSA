#!/usr/bin/env python3
import argparse
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from timm.models import elsa_triton as elsa_ops

PRECISIONS: Sequence[str] = ("fp32", "tf32", "fp16", "bf16")


def _parse_tokens_from_csv(
    path: str,
    *,
    min_tokens: int,
    max_tokens: int,
    num_tokens: int,
) -> List[int]:
    if pd is None:
        raise RuntimeError("pandas is required for --tokens-csv")
    df = pd.read_csv(path)
    if "tokens" not in df.columns:
        raise ValueError(f"{path} is missing a 'tokens' column")
    tokens = sorted(set(int(x) for x in df["tokens"].tolist()))
    if min_tokens:
        tokens = [t for t in tokens if t >= min_tokens]
    if max_tokens:
        tokens = [t for t in tokens if t <= max_tokens]
    if num_tokens > 0:
        tokens = tokens[-num_tokens:]
    return tokens


def _parse_tokens(args: argparse.Namespace) -> List[int]:
    if args.tokens:
        tokens = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    elif args.tokens_csv:
        tokens = _parse_tokens_from_csv(
            args.tokens_csv,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            num_tokens=args.num_tokens,
        )
    else:
        default_csv = "fair_bench_all_CAN.csv"
        if os.path.exists(default_csv) and pd is not None:
            tokens = _parse_tokens_from_csv(
                default_csv,
                min_tokens=args.min_tokens or 65536,
                max_tokens=args.max_tokens,
                num_tokens=args.num_tokens or 4,
            )
        else:
            tokens = [65536, 131072]
    if not tokens:
        raise ValueError("No tokens selected; please check --tokens/--tokens-csv filters.")
    return tokens


def _precision_dtype(precision: str) -> torch.dtype:
    if precision in ("fp32", "tf32"):
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision '{precision}'")


def _benchmark(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) / max(1, iters)


def _format_ms(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(torch.device("cuda"))
    return prop.major >= 8


@contextmanager
def _tf32_context(enabled: bool):
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ELSA precision modes (before vs after).")
    parser.add_argument("--tokens", type=str, default="", help="Comma-separated token counts.")
    parser.add_argument("--tokens-csv", type=str, default="", help="CSV with a 'tokens' column.")
    parser.add_argument("--min-tokens", type=int, default=0, help="Filter tokens >= this value.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Filter tokens <= this value.")
    parser.add_argument("--num-tokens", type=int, default=0, help="Take last N tokens after filtering.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--heads", type=int, default=2, help="Attention heads.")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension.")
    parser.add_argument("--precisions", type=str, default=",".join(PRECISIONS), help="Precisions to test.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations.")
    parser.add_argument("--causal", action="store_true", help="Use causal masking.")
    args = parser.parse_args()

    tokens = _parse_tokens(args)
    precisions = [p.strip().lower() for p in args.precisions.split(",") if p.strip()]
    if not precisions:
        raise ValueError("No precisions selected.")

    if "bf16" in precisions and not _supports_bf16():
        precisions = [p for p in precisions if p != "bf16"]

    device = torch.device("cuda")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")

    print("ELSA precision compare")
    print(f"  tokens={tokens}")
    print(f"  B={args.batch} H={args.heads} D={args.head_dim} causal={args.causal}")
    print(f"  precisions={precisions}")

    results: List[Dict[str, float]] = []

    for n in tokens:
        for precision in precisions:
            dtype = _precision_dtype(precision)
            q = torch.randn(args.batch, args.heads, n, args.head_dim, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            scale = 1.0 / math.sqrt(args.head_dim)

            legacy_fp32 = getattr(elsa_ops, "elsa_triton_new_fp32_legacy", elsa_ops.elsa_triton_new_fp32)

            def run_old():
                with torch.no_grad():
                    if precision == "fp32":
                        legacy_fp32(q, k, v, is_causal=args.causal, bias=None)
                    elif precision == "tf32":
                        with _tf32_context(True):
                            legacy_fp32(q, k, v, is_causal=args.causal, bias=None)
                    elif precision in ("fp16", "bf16"):
                        elsa_ops.ELSA_triton.apply(q, k, v, scale, None, args.causal)
                    else:
                        raise ValueError(f"Unsupported precision '{precision}'")

            def run_new():
                with torch.no_grad():
                    if precision == "fp32":
                        elsa_ops.elsa_triton_new_fp32(q, k, v, is_causal=args.causal, bias=None)
                    elif precision == "tf32":
                        with _tf32_context(True):
                            elsa_ops.elsa_triton_new_fp32(q, k, v, is_causal=args.causal, bias=None)
                    else:
                        elsa_ops.elsa_triton_new(
                            q, k, v, is_causal=args.causal, bias=None, precision=precision
                        )

            old_ms = _benchmark(run_old, warmup=args.warmup, iters=args.iters)
            new_ms = _benchmark(run_new, warmup=args.warmup, iters=args.iters)
            speedup = old_ms / new_ms if new_ms else float("inf")

            tokens_per_s = (args.batch * args.heads * n) / (new_ms / 1000.0)

            results.append(
                {
                    "tokens": n,
                    "precision": precision,
                    "old_ms": old_ms,
                    "new_ms": new_ms,
                    "speedup": speedup,
                    "after_tokens_per_s": tokens_per_s,
                }
            )

            print(
                f"N={n} prec={precision} "
                f"old={_format_ms(old_ms)}ms new={_format_ms(new_ms)}ms "
                f"speedup={speedup:.2f}x"
            )

    if args.tokens_csv:
        base = os.path.splitext(os.path.basename(args.tokens_csv))[0]
    else:
        base = "elsa_precision_compare"
    out_path = f"{base}_compare.csv"
    with open(out_path, "w", newline="") as f:
        headers = ["tokens", "precision", "old_ms", "new_ms", "speedup", "after_tokens_per_s"]
        f.write(",".join(headers) + "\n")
        for row in results:
            f.write(",".join(str(row[h]) for h in headers) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
