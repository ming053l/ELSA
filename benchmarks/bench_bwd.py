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

from elsa_twopass_clean.attention_bwd import twopass_attention_train


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


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kind: str, *, is_causal: bool) -> torch.Tensor:
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
    raise ValueError(kind)


BASELINES = ("auto", "flash", "mem", "math")


def _skip_baseline(kind: str, *, dtype: torch.dtype, batch: int, heads: int, seq_len: int) -> bool:
    if kind != "math":
        return False
    score_bytes = batch * heads * seq_len * seq_len * torch.empty((), dtype=dtype).element_size()
    return score_bytes > 4 * 1024**3


def _make_inputs(shape: tuple[int, int, int, int], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    dout = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v, dout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--baseline", choices=["best", "auto", "math", "flash", "mem"], default="best")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--input-precision", choices=["auto", "ieee", "tf32", "tf32x3"], default="auto")
    parser.add_argument("--bwd-block-m", type=int, default=None)
    parser.add_argument("--bwd-block-n", type=int, default=None)
    parser.add_argument("--bwd-warps", type=int, default=None)
    parser.add_argument("--bwd-stages", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required")
        return 1

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    torch.manual_seed(2026)

    print("dtype,seq,causal,baseline,base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio")
    for seq_len in args.seq:
        shape = (args.batch, args.heads, seq_len, args.dim)
        q, k, v, dout = _make_inputs(shape, dtype)

        def run_elsa():
            q_i = q.detach().requires_grad_(True)
            k_i = k.detach().requires_grad_(True)
            v_i = v.detach().requires_grad_(True)
            out = twopass_attention_train(
                q_i,
                k_i,
                v_i,
                is_causal=args.causal,
                bwd_block_m=args.bwd_block_m,
                bwd_block_n=args.bwd_block_n,
                bwd_num_warps=args.bwd_warps,
                bwd_num_stages=args.bwd_stages,
                bwd_input_precision=args.input_precision,
                input_precision=args.input_precision,
            )
            out.backward(dout)
            return q_i.grad, k_i.grad, v_i.grad

        elsa_ms, elsa_mb = bench(run_elsa, warmup=args.warmup, iters=args.iters)

        baseline_kinds = BASELINES if args.baseline == "best" else (args.baseline,)
        base_candidates = []
        base_errors = []
        for baseline_kind in baseline_kinds:
            if _skip_baseline(baseline_kind, dtype=dtype, batch=args.batch, heads=args.heads, seq_len=seq_len):
                base_errors.append(f"{baseline_kind}:skipped")
                continue

            def run_base(kind=baseline_kind):
                q_i = q.detach().requires_grad_(True)
                k_i = k.detach().requires_grad_(True)
                v_i = v.detach().requires_grad_(True)
                out = sdpa(q_i, k_i, v_i, kind, is_causal=args.causal)
                out.backward(dout)
                return q_i.grad, k_i.grad, v_i.grad

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
        lat_ratio = elsa_ms / base_ms if base_ms == base_ms else float("nan")
        mem_ratio = elsa_mb / base_mb if base_mb == base_mb and base_mb > 0 else float("nan")
        print(
            f"{args.dtype},{seq_len},{int(args.causal)},{baseline},"
            f"{base_ms:.6f},{base_mb:.3f},{elsa_ms:.6f},{elsa_mb:.3f},"
            f"{lat_ratio:.4f},{mem_ratio:.4f}",
            flush=True,
        )
        del q, k, v, dout
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
