#!/usr/bin/env python3
"""Benchmark Swin full-model throughput for fused ELSA vs SDPA baselines."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import timm

from scripts.benchmark_model_throughput import measure_model, toggle_tf32
from timm.models.elsa_swin import set_default_elsa_backend
from timm.models.elsa_swin_fused import patch_elsa_window_attention


MODEL_MAP = {
    "swin_t_8": ("elsa_tiny_window8_256", "swinv2_tiny_window8_256"),
    "swin_t_16": ("elsa_tiny_window16_256", "swinv2_tiny_window16_256"),
    "swin_s_8": ("elsa_small_window8_256", "swinv2_small_window8_256"),
    "swin_s_16": ("elsa_small_window16_256", "swinv2_small_window16_256"),
}


def run_swin(
    device: torch.device,
    warmup: int,
    trials: int,
    input_size: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def run_model(model_name: str, label: str, dtype: torch.dtype, ctx=None, autocast=None):
        model = timm.create_model(model_name, pretrained=False, img_size=input_size).to(device)
        model.eval()
        stats = measure_model(
            model=model,
            batch=1,
            input_size=input_size,
            dtype=dtype,
            device=device,
            warmup=warmup,
            trials=trials,
            autocast_dtype=autocast,
            ctx_factories=ctx,
        )
        rows.append(
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

    patch_elsa_window_attention()

    for _, (elsa_model, baseline_model) in MODEL_MAP.items():
        # FP16 ELSA
        set_default_elsa_backend("triton")
        run_model(elsa_model, "ELSA-triton-fused", torch.float16)

        # Baseline FP16 using SDPA mem (flash not supported with Swin biases)
        run_model(
            baseline_model,
            "SDPA-mem",
            torch.float16,
            ctx=[lambda: torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)],
            autocast=torch.float16,
        )

        # FP32 strict
        set_default_elsa_backend("triton_fp32")
        with toggle_tf32(False):
            run_model(elsa_model, "ELSA-triton_fp32-fused", torch.float32)

        # FP32 turbo (TF32)
        set_default_elsa_backend("triton")
        with toggle_tf32(True):
            run_model(elsa_model, "ELSA-turbo-fused", torch.float32)

        # Baseline FP32 math/mem
        for label, ctx in (
            ("SDPA-math", [lambda: torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)]),
            ("SDPA-mem", [lambda: torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)]),
        ):
            with toggle_tf32(False):
                run_model(baseline_model, label, torch.float32, ctx=ctx)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--fused-qknorm", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/current/results/swin_full_model_bench_fused.csv"))
    args = parser.parse_args()

    if args.fused_qknorm:
        os.environ["ELSA_SWIN_FUSED_QKNORM"] = "1"

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    run_swin(device, args.warmup, args.trials, args.input_size, args.output)


if __name__ == "__main__":
    main()
