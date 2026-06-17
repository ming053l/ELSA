from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elsa_twopass_clean import make_model_pair


BASELINES = ("auto", "flash", "mem", "math")


def _sync() -> None:
    torch.cuda.synchronize()


def bench(fn, *, warmup: int, iters: int) -> tuple[float, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--baseline", choices=["best", "auto", "math", "flash", "mem"], default="best")
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
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
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
    elsa_kwargs = {
        "block_m": args.block_m,
        "block_n": args.block_n,
        "q_chunk_size": args.q_chunk,
        "summary_dtype": summary_dtype,
        "phase1_warps": args.phase1_warps,
        "phase2_warps": args.phase2_warps,
        "phase1_stages": args.phase1_stages,
        "phase2_stages": args.phase2_stages,
        "input_precision": args.input_precision,
        "algorithm": args.algorithm,
    }
    elsa_kwargs = {key: value for key, value in elsa_kwargs.items() if value is not None}
    baseline_kinds = BASELINES if args.baseline == "best" else (args.baseline,)
    print(
        "dtype,seq,batch,dim,heads,depth,baseline,"
        "base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio,max_abs"
    )

    for seq_len in args.seq:
        torch.manual_seed(1234)
        baseline_ref, elsa = make_model_pair(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            sdpa_kind="auto",
            is_causal=args.causal,
            elsa_kwargs=elsa_kwargs,
        )
        elsa = elsa.eval().to(device="cuda", dtype=dtype)
        x = torch.randn((args.batch, seq_len, args.dim), device="cuda", dtype=dtype)

        def run_elsa():
            with torch.no_grad():
                return elsa(x)

        elsa_ms, elsa_mb = bench(run_elsa, warmup=args.warmup, iters=args.iters)

        base_candidates = []
        base_errors = []
        for kind in baseline_kinds:
            baseline = baseline_ref
            if kind != "auto":
                baseline, _ = make_model_pair(
                    dim=args.dim,
                    depth=args.depth,
                    heads=args.heads,
                    mlp_ratio=args.mlp_ratio,
                    sdpa_kind=kind,
                    is_causal=args.causal,
                )
                baseline.load_state_dict(baseline_ref.state_dict())
            baseline = baseline.eval().to(device="cuda", dtype=dtype)

            def run_base(model=baseline):
                with torch.no_grad():
                    return model(x)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base_ms_i, base_mb_i = bench(run_base, warmup=args.warmup, iters=args.iters)
                base_candidates.append((base_ms_i, base_mb_i, kind))
            except Exception as err:
                base_errors.append(f"{kind}:{type(err).__name__}")
            finally:
                del baseline
                torch.cuda.empty_cache()

        if base_candidates:
            base_ms, base_mb, baseline_name = min(base_candidates, key=lambda item: item[0])
        else:
            base_ms, base_mb = float("nan"), float("nan")
            baseline_name = "|".join(base_errors)

        if args.skip_check or not base_candidates:
            max_abs = float("nan")
        else:
            if baseline_name == "auto":
                baseline = baseline_ref
            else:
                baseline, _ = make_model_pair(
                    dim=args.dim,
                    depth=args.depth,
                    heads=args.heads,
                    mlp_ratio=args.mlp_ratio,
                    sdpa_kind=baseline_name,
                    is_causal=args.causal,
                )
                baseline.load_state_dict(baseline_ref.state_dict())
            baseline = baseline.eval().to(device="cuda", dtype=dtype)
            with torch.no_grad():
                base_out = baseline(x)
            elsa_out = run_elsa()
            max_abs = (elsa_out.float() - base_out.float()).abs().max().item()
        lat_ratio = elsa_ms / base_ms if base_ms == base_ms else float("nan")
        mem_ratio = elsa_mb / base_mb if base_mb == base_mb and base_mb > 0 else float("nan")
        print(
            f"{args.dtype},{seq_len},{args.batch},{args.dim},{args.heads},{args.depth},{baseline_name},"
            f"{base_ms:.6f},{base_mb:.3f},{elsa_ms:.6f},{elsa_mb:.3f},"
            f"{lat_ratio:.4f},{mem_ratio:.4f},{max_abs:.6g}"
        )


if __name__ == "__main__":
    main()
