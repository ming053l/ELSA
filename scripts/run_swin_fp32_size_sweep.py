#!/usr/bin/env python3
"""Swin full-model FP32 size sweep: ELSA vs baseline (WindowAttn or SDPA)."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import torch
import timm

from scripts.benchmark_model_throughput import measure_model, sdp_context, toggle_tf32
from timm.models.elsa_swin import set_default_elsa_backend
from timm.models.elsa_swin_fused import patch_elsa_window_attention
from timm.models.swin_sdpa_patch import patch_swin_v2_sdpa

MODEL_MAP = {
    "tiny_w8": ("elsa_tiny_window8_256", "swinv2_tiny_window8_256"),
    "small_w8": ("elsa_small_window8_256", "swinv2_small_window8_256"),
    "base_w8": ("elsa_base_window8_256", "swinv2_base_window8_256"),
}


def _parse_sizes(raw: str) -> list[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("No sizes parsed.")
    return vals


def _run_one(
    model_name: str,
    backend: str,
    size: int,
    device: torch.device,
    warmup: int,
    trials: int,
    ctx_factories: Iterable | None = None,
) -> dict[str, object]:
    row: dict[str, object] = dict(
        model=model_name,
        backend=backend,
        size=size,
        dtype="fp32",
        status="ok",
        error="",
        lat_ms_avg=float("nan"),
        lat_ms_med=float("nan"),
        lat_ms_p95=float("nan"),
        peak_gb=float("nan"),
        imgs_per_s=float("nan"),
    )
    model = None
    try:
        model_kwargs = dict(pretrained=False, img_size=size)
        # SwinELSA defaults to triton=False; force Triton path for ELSA runs.
        if model_name.startswith("elsa_"):
            model_kwargs.update(dict(triton=True, elsa_backend="triton"))
        model = timm.create_model(model_name, **model_kwargs).to(device)
        model.eval()
        stats = measure_model(
            model=model,
            batch=1,
            input_size=size,
            dtype=torch.float32,
            device=device,
            warmup=warmup,
            trials=trials,
            ctx_factories=ctx_factories,
        )
        row.update(stats)
    except Exception as err:
        row["status"] = "error"
        row["error"] = str(err).split("\n")[0][:240]
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", default="256,512,768,1024,1280,1536,1792,2048")
    parser.add_argument("--models", default="tiny_w8,small_w8")
    parser.add_argument(
        "--baseline",
        choices=("window", "sdpa"),
        default="window",
        help="Baseline backend for SwinV2 models.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/current/results/swin_fp32_size_sweep_256_2048.csv"),
    )
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    for k in model_keys:
        if k not in MODEL_MAP:
            raise ValueError(f"Unknown model key: {k}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Keep using the current stable fused attention configuration.
    os.environ.setdefault("ELSA_SWIN_FUSED_QKNORM", "1")
    os.environ.setdefault("ELSA_SWIN_FUSED_RELBIAS", "1")
    os.environ.setdefault("ELSA_SWIN_FUSED_OUT_NH", "1")
    os.environ.setdefault("ELSA_SWIN_FUSED_COMPACT_MASK", "1")
    os.environ.setdefault("ELSA_SWIN_FUSED_PROJ", "off")

    patch_elsa_window_attention()
    if args.baseline == "sdpa":
        patch_swin_v2_sdpa()

    rows: list[dict[str, object]] = []
    for key in model_keys:
        elsa_name, baseline_name = MODEL_MAP[key]
        for size in sizes:
            with toggle_tf32(False):
                set_default_elsa_backend("triton_fp32")
                rows.append(
                    _run_one(
                        model_name=elsa_name,
                        backend="ELSA-fp32",
                        size=size,
                        device=device,
                        warmup=args.warmup,
                        trials=args.trials,
                    )
                )

            with toggle_tf32(False):
                if args.baseline == "sdpa":
                    rows.append(
                        _run_one(
                            model_name=baseline_name,
                            backend="SDPA-math",
                            size=size,
                            device=device,
                            warmup=args.warmup,
                            trials=args.trials,
                            ctx_factories=[lambda: sdp_context(math=True, mem=False, flash=False)],
                        )
                    )

                else:
                    rows.append(
                        _run_one(
                            model_name=baseline_name,
                            backend="WindowAttn-Orig",
                            size=size,
                            device=device,
                            warmup=args.warmup,
                            trials=args.trials,
                            ctx_factories=None,
                        )
                    )

            if args.baseline == "sdpa":
                with toggle_tf32(False):
                    rows.append(
                        _run_one(
                            model_name=baseline_name,
                            backend="SDPA-mem",
                            size=size,
                            device=device,
                            warmup=args.warmup,
                            trials=args.trials,
                            ctx_factories=[lambda: sdp_context(math=False, mem=True, flash=False)],
                        )
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
