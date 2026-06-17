from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elsa_twopass_clean import twopass_attention


def _sync():
    torch.cuda.synchronize()


def bench(fn, *, warmup: int, iters: int):
    torch.cuda.empty_cache()
    for _ in range(warmup):
        fn()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 1024**2


BASELINES = ("auto", "flash", "mem", "math")


def _skip_baseline(kind: str, *, dtype: torch.dtype, batch: int, heads: int, seq_len: int) -> bool:
    if kind != "math":
        return False
    score_bytes = batch * heads * seq_len * seq_len * torch.empty((), dtype=dtype).element_size()
    return score_bytes > 4 * 1024**3


def sdpa(q, k, v, kind: str):
    if kind == "auto":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(q.shape[-1]))
    if kind == "math":
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(q.shape[-1]))
    if kind == "flash":
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(q.shape[-1]))
    if kind == "mem":
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(q.shape[-1]))
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--seq", type=int, nargs="+", default=[196, 1024, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--q-chunk", type=int, default=None)
    parser.add_argument("--summary-dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--phase1-warps", type=int, default=None)
    parser.add_argument("--phase2-warps", type=int, default=None)
    parser.add_argument("--phase1-stages", type=int, default=None)
    parser.add_argument("--phase2-stages", type=int, default=None)
    parser.add_argument("--input-precision", choices=["auto", "ieee", "tf32", "tf32x3"], default="auto")
    parser.add_argument(
        "--algorithm",
        choices=["auto", "paper_scan"],
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--baseline", choices=["best", "auto", "math", "flash", "mem"], default="best")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    summary_dtype = None if args.summary_dtype == "auto" else {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.summary_dtype]
    torch.manual_seed(1234)
    print("dtype,summary_dtype,seq,baseline,base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio,max_abs")
    for n in args.seq:
        shape = (args.batch, args.heads, n, args.dim)
        q = torch.randn(shape, device="cuda", dtype=dtype)
        k = torch.randn(shape, device="cuda", dtype=dtype)
        v = torch.randn(shape, device="cuda", dtype=dtype)

        def run_elsa():
            return twopass_attention(
                q,
                k,
                v,
                block_m=args.block_m,
                block_n=args.block_n,
                q_chunk_size=args.q_chunk,
                summary_dtype=summary_dtype,
                phase1_warps=args.phase1_warps,
                phase2_warps=args.phase2_warps,
                phase1_stages=args.phase1_stages,
                phase2_stages=args.phase2_stages,
                input_precision=args.input_precision,
                algorithm=args.algorithm,
            )

        elsa_ms, elsa_mb = bench(run_elsa, warmup=args.warmup, iters=args.iters)

        baseline_kinds = BASELINES if args.baseline == "best" else (args.baseline,)
        base_candidates = []
        base_errors = []
        for baseline_kind in baseline_kinds:
            if _skip_baseline(baseline_kind, dtype=dtype, batch=args.batch, heads=args.heads, seq_len=n):
                base_errors.append(f"{baseline_kind}:skipped")
                continue

            def run_base(kind=baseline_kind):
                return sdpa(q, k, v, kind)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base_ms_i, base_mb_i = bench(run_base, warmup=args.warmup, iters=args.iters)
                base_candidates.append((base_ms_i, base_mb_i, baseline_kind))
            except Exception as err:
                base_errors.append(f"{baseline_kind}:{type(err).__name__}")
            finally:
                torch.cuda.empty_cache()

        if base_candidates:
            base_ms, base_mb, baseline = min(base_candidates, key=lambda item: item[0])
        else:
            base_ms, base_mb = float("nan"), float("nan")
            baseline = "|".join(base_errors)

        if args.skip_check or not base_candidates:
            max_abs = float("nan")
        else:
            base_out = sdpa(q, k, v, baseline)
            elsa_out = run_elsa()
            max_abs = (elsa_out.float() - base_out.float()).abs().max().item()
        lat_ratio = elsa_ms / base_ms if base_ms == base_ms else float("nan")
        mem_ratio = elsa_mb / base_mb if base_mb == base_mb and base_mb > 0 else float("nan")
        print(
            f"{args.dtype},{args.summary_dtype},{n},{baseline},"
            f"{base_ms:.6f},{base_mb:.3f},{elsa_ms:.6f},{elsa_mb:.3f},"
            f"{lat_ratio:.4f},{mem_ratio:.4f},{max_abs:.6g}"
        )


if __name__ == "__main__":
    main()
