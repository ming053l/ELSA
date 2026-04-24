#!/usr/bin/env python3
"""Benchmark ViT full-model throughput for ELSA vs FlashAttention."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

import torch
import timm

from scripts.benchmark_model_throughput import measure_model, patch_vit_flash_attention, toggle_tf32
from timm.models.elsa import set_default_elsa_backend


def run_vi_transformers(
    device: torch.device,
    variants: Iterable[str],
    warmup: int,
    trials: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    def run_model(model_name: str, label: str, dtype: torch.dtype, ctx=None, autocast_dtype: Optional[torch.dtype] = None):
        model = timm.create_model(model_name, pretrained=False).to(device)
        model.eval()
        stats = measure_model(
            model=model,
            batch=1,
            input_size=224,
            dtype=dtype,
            device=device,
            warmup=warmup,
            trials=trials,
            autocast_dtype=autocast_dtype,
            ctx_factories=ctx,
        )
        results.append(
            dict(
                model=model_name,
                backend=label,
                dtype="fp16" if dtype == torch.float16 else "fp32",
                lat_ms_avg=stats["lat_ms_avg"],
                lat_ms_med=stats["lat_ms_med"],
                lat_ms_p95=stats["lat_ms_p95"],
                peak_gb=stats["peak_gb"],
                imgs_per_s=stats["imgs_per_s"],
            )
        )
        del model
        torch.cuda.empty_cache()

    model_pairs = {
        "tiny": ("elsa_tiny_patch16_224", "deit_tiny_patch16_224"),
        "small": ("elsa_small_patch16_224", "deit_small_patch16_224"),
        "medium": ("elsa3_medium_patch16_224", "deit3_medium_patch16_224"),
        "base": ("elsa3_base_patch16_224", "deit3_base_patch16_224"),
    }

    for variant in variants:
        if variant not in model_pairs:
            raise RuntimeError(f"Unsupported ViT variant '{variant}'.")
        elsa_model_name, baseline_model = model_pairs[variant]

        # FP16 - ELSA
        set_default_elsa_backend("triton")
        run_model(elsa_model_name, "ELSA-triton", torch.float16)

        # FP16 - FlashAttention v2/v3
        for kind in ("fa2", "fa3"):
            try:
                model = timm.create_model(baseline_model, pretrained=False).to(device)
                patch_vit_flash_attention(model, kind)
                stats = measure_model(
                    model=model,
                    batch=1,
                    input_size=224,
                    dtype=torch.float16,
                    device=device,
                    warmup=warmup,
                    trials=trials,
                    autocast_dtype=torch.float16,
                )
                results.append(
                    dict(
                        model=f"deit_{variant}",
                        backend=f"Flash-{kind.upper()}",
                        dtype="fp16",
                        lat_ms_avg=stats["lat_ms_avg"],
                        lat_ms_med=stats["lat_ms_med"],
                        lat_ms_p95=stats["lat_ms_p95"],
                        peak_gb=stats["peak_gb"],
                        imgs_per_s=stats["imgs_per_s"],
                    )
                )
            except RuntimeError:
                continue
            finally:
                torch.cuda.empty_cache()

        # FP32 - ELSA (strict)
        set_default_elsa_backend("triton_fp32")
        with toggle_tf32(False):
            run_model(elsa_model_name, "ELSA-triton_fp32", torch.float32)

        # FP32 - ELSA (TF32 turbo)
        set_default_elsa_backend("triton")
        with toggle_tf32(True):
            run_model(elsa_model_name, "ELSA-turbo", torch.float32)

        # FP32 - SDPA math/mem
        baseline_ctx = [
            ("SDPA-math", False, [lambda: torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)]),
            ("SDPA-mem", False, [lambda: torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)]),
            ("SDPA-math-tf32", True, [lambda: torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)]),
            ("SDPA-mem-tf32", True, [lambda: torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)]),
        ]
        for label, enable_tf32, ctx in baseline_ctx:
            model = timm.create_model(baseline_model, pretrained=False).to(device)
            with toggle_tf32(enable_tf32):
                stats = measure_model(
                    model=model,
                    batch=1,
                    input_size=224,
                    dtype=torch.float32,
                    device=device,
                    warmup=warmup,
                    trials=trials,
                    ctx_factories=ctx,
                )
            results.append(
                dict(
                    model=baseline_model,
                    backend=label,
                    dtype="fp32",
                    lat_ms_avg=stats["lat_ms_avg"],
                    lat_ms_med=stats["lat_ms_med"],
                    lat_ms_p95=stats["lat_ms_p95"],
                    peak_gb=stats["peak_gb"],
                    imgs_per_s=stats["imgs_per_s"],
                )
            )
            del model
            torch.cuda.empty_cache()

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["tiny", "small", "medium", "base"],
        help="ViT model aliases: tiny/small/medium/base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/current/results/vit_full_model_bench.csv"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    run_vi_transformers(device, args.variants, args.warmup, args.trials, args.output)


if __name__ == "__main__":
    main()
