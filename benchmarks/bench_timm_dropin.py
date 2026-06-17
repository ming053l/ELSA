from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elsa_twopass_clean import patch_timm_attention


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


def _set_input_size(model: torch.nn.Module, image_size: int, *, window_size: int | None, window_ratio: int) -> None:
    if not hasattr(model, "set_input_size"):
        return
    kwargs = {"img_size": (image_size, image_size)}
    if window_size is not None:
        kwargs["window_size"] = (window_size, window_size)
    else:
        kwargs["window_ratio"] = window_ratio
    try:
        model.set_input_size(**kwargs)
    except TypeError:
        model.set_input_size(img_size=(image_size, image_size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="swin_tiny_patch4_window7_224.ms_in1k")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--image-size", type=int, nargs="+", default=[224])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-ratio", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--input-precision", choices=["auto", "ieee", "tf32", "tf32x3"], default="auto")
    parser.add_argument("--summary-dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--q-chunk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    summary_dtype = None if args.summary_dtype == "auto" else {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.summary_dtype]
    elsa_kwargs = {
        "input_precision": args.input_precision,
        "summary_dtype": summary_dtype,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "q_chunk_size": args.q_chunk,
    }
    elsa_kwargs = {key: value for key, value in elsa_kwargs.items() if value is not None}

    print(
        "model,dtype,image,batch,patched_attn,patched_vit,patched_swin,"
        "base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio,max_abs"
    )
    for image_size in args.image_size:
        torch.manual_seed(1234)
        base = timm.create_model(args.model, pretrained=args.pretrained).eval()
        _set_input_size(base, image_size, window_size=args.window_size, window_ratio=args.window_ratio)
        elsa = copy.deepcopy(base).eval()
        report = patch_timm_attention(elsa, elsa_kwargs=elsa_kwargs)
        base = base.to(device="cuda", dtype=dtype)
        elsa = elsa.to(device="cuda", dtype=dtype)
        x = torch.randn((args.batch, 3, image_size, image_size), device="cuda", dtype=dtype)

        def run_base():
            with torch.no_grad():
                return base(x)

        def run_elsa():
            with torch.no_grad():
                return elsa(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_ms, base_mb = bench(run_base, warmup=args.warmup, iters=args.iters)
            elsa_ms, elsa_mb = bench(run_elsa, warmup=args.warmup, iters=args.iters)
            if args.skip_check:
                max_abs = float("nan")
            else:
                base_out = run_base()
                elsa_out = run_elsa()
                max_abs = (elsa_out.float() - base_out.float()).abs().max().item()

        print(
            f"{args.model},{args.dtype},{image_size},{args.batch},{report.total},"
            f"{report.vit_attention},{report.swin_window_attention},"
            f"{base_ms:.6f},{base_mb:.3f},{elsa_ms:.6f},{elsa_mb:.3f},"
            f"{elsa_ms / base_ms:.4f},{elsa_mb / base_mb:.4f},{max_abs:.6g}"
        )


if __name__ == "__main__":
    main()
