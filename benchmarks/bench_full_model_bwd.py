from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elsa_twopass_clean.full_model_bwd import make_train_model_pair


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


BASELINES = ("auto", "flash", "mem", "math")


def _skip_baseline(kind: str, *, dtype: torch.dtype, batch: int, heads: int, seq_len: int) -> bool:
    if kind != "math":
        return False
    score_bytes = batch * heads * seq_len * seq_len * torch.empty((), dtype=dtype).element_size()
    return score_bytes > 4 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--baseline", choices=["best", "auto", "math", "flash", "mem"], default="best")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--input-precision", choices=["auto", "ieee", "tf32", "tf32x3"], default="auto")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    torch.manual_seed(9090)

    print("dtype,seq,depth,causal,baseline,base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio")
    for seq_len in args.seq:
        shape = (args.batch, seq_len, args.dim)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        grad = torch.randn(shape, device="cuda", dtype=dtype)

        elsa_kwargs = {"input_precision": args.input_precision}
        baseline_kinds = BASELINES if args.baseline == "best" else (args.baseline,)
        base_candidates = []
        base_errors = []

        base_template, elsa_template = make_train_model_pair(
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            is_causal=args.causal,
            elsa_kwargs=elsa_kwargs,
        )
        elsa_model = copy.deepcopy(elsa_template).train().to(device="cuda", dtype=dtype)

        def run_elsa():
            elsa_model.zero_grad(set_to_none=True)
            x_i = x.detach().requires_grad_(True)
            out = elsa_model(x_i)
            out.backward(grad)
            return x_i.grad

        elsa_ms, elsa_mb = bench(run_elsa, warmup=args.warmup, iters=args.iters)
        del elsa_model
        torch.cuda.empty_cache()

        for baseline_kind in baseline_kinds:
            if _skip_baseline(
                baseline_kind,
                dtype=dtype,
                batch=args.batch,
                heads=args.heads,
                seq_len=seq_len,
            ):
                base_errors.append(f"{baseline_kind}:skipped")
                continue

            base_model = copy.deepcopy(base_template)
            for block in base_model.blocks:
                block.attn.sdpa_kind = baseline_kind
            base_model = base_model.train().to(device="cuda", dtype=dtype)

            def run_base():
                base_model.zero_grad(set_to_none=True)
                x_i = x.detach().requires_grad_(True)
                out = base_model(x_i)
                out.backward(grad)
                return x_i.grad

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base_ms_i, base_mb_i = bench(run_base, warmup=args.warmup, iters=args.iters)
                base_candidates.append((base_ms_i, base_mb_i, baseline_kind))
            except Exception as err:
                base_errors.append(f"{baseline_kind}:{type(err).__name__}")
            finally:
                del base_model
                torch.cuda.empty_cache()

        if base_candidates:
            base_ms, base_mb, baseline = min(base_candidates, key=lambda item: item[0])
        else:
            base_ms, base_mb = float("nan"), float("nan")
            baseline = "|".join(base_errors)
        lat_ratio = elsa_ms / base_ms if base_ms == base_ms else float("nan")
        mem_ratio = elsa_mb / base_mb if base_mb == base_mb and base_mb > 0 else float("nan")
        print(
            f"{args.dtype},{seq_len},{args.depth},{int(args.causal)},{baseline},"
            f"{base_ms:.6f},{base_mb:.3f},{elsa_ms:.6f},{elsa_mb:.3f},"
            f"{lat_ratio:.4f},{mem_ratio:.4f}",
            flush=True,
        )
        del x, grad, base_template, elsa_template
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
