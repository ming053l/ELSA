#!/usr/bin/env python3
"""Verify Swin ELSA exactness: PyTorch vs Triton-FP32 backend."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional
from collections import Counter

import torch
import timm

from scripts.benchmark_model_throughput import toggle_tf32
from timm.models.elsa_swin import set_default_elsa_backend
from timm.models.elsa_swin_fused import patch_elsa_window_attention

MODEL_MAP = {
    "tiny_w8": "elsa_tiny_window8_256",
    "small_w8": "elsa_small_window8_256",
    "base_w8": "elsa_base_window8_256",
}


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("No integers parsed.")
    return vals


def _find_first_shifted_attn_name(model: torch.nn.Module) -> Optional[str]:
    for li, layer in enumerate(getattr(model, "layers", [])):
        blocks = getattr(layer, "blocks", [])
        for bi, block in enumerate(blocks):
            shift = getattr(block, "shift_size", (0, 0))
            if isinstance(shift, tuple) and any(int(x) > 0 for x in shift):
                return f"layers.{li}.blocks.{bi}.attn"
    return None


def _get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    return cur


def _tensor_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    diff = (test - ref).abs()
    ref_abs = ref.abs()
    denom = torch.maximum(ref_abs, torch.full_like(ref_abs, 1e-8))
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def _run_one(
    model_name: str,
    size: int,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    set_default_elsa_backend("pytorch")
    model_ref = timm.create_model(
        model_name,
        pretrained=False,
        img_size=size,
        triton=False,
        elsa_backend="pytorch",
    ).to(device)
    model_ref.eval()

    state = {k: v.detach().clone() for k, v in model_ref.state_dict().items()}

    set_default_elsa_backend("triton")
    model_test = timm.create_model(
        model_name,
        pretrained=False,
        img_size=size,
        triton=True,
        elsa_backend="triton",
    ).to(device)
    model_test.load_state_dict(state, strict=True)
    model_test.eval()

    # Track actual backend dispatch to ensure Triton path is really used.
    backend_calls: Counter[str] = Counter()
    for mod in model_test.modules():
        if hasattr(mod, "_run_backend"):
            orig_run_backend = mod._run_backend

            def _wrapped(backend, *args, _orig=orig_run_backend, **kwargs):
                backend_calls[str(backend)] += 1
                return _orig(backend, *args, **kwargs)

            mod._run_backend = _wrapped

    shifted_name = _find_first_shifted_attn_name(model_ref)
    cap_ref: dict[str, torch.Tensor] = {}
    cap_test: dict[str, torch.Tensor] = {}
    h_ref = None
    h_test = None
    if shifted_name is not None:
        mod_ref = _get_module_by_name(model_ref, shifted_name)
        mod_test = _get_module_by_name(model_test, shifted_name)

        def _hook_ref(_m, _i, o):
            cap_ref["out"] = o.detach()

        def _hook_test(_m, _i, o):
            cap_test["out"] = o.detach()

        h_ref = mod_ref.register_forward_hook(_hook_ref)
        h_test = mod_test.register_forward_hook(_hook_test)

    x = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
    with torch.no_grad():
        with toggle_tf32(False):
            y_ref = model_ref(x).detach()
        with toggle_tf32(False):
            y_test = model_test(x).detach()

    if h_ref is not None:
        h_ref.remove()
    if h_test is not None:
        h_test.remove()

    global_m = _tensor_metrics(y_ref, y_test)
    global_ok = bool(torch.allclose(y_ref, y_test, atol=atol, rtol=rtol))

    attn_max_abs = float("nan")
    attn_mean_abs = float("nan")
    attn_max_rel = float("nan")
    attn_ok = False
    if "out" in cap_ref and "out" in cap_test:
        attn_ref = cap_ref["out"]
        attn_test = cap_test["out"]
        attn_m = _tensor_metrics(attn_ref, attn_test)
        attn_max_abs = attn_m["max_abs"]
        attn_mean_abs = attn_m["mean_abs"]
        attn_max_rel = attn_m["max_rel"]
        attn_ok = bool(torch.allclose(attn_ref, attn_test, atol=atol, rtol=rtol))

    del model_ref
    del model_test
    del state
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "size": size,
        "seed": seed,
        "shifted_attn_module": shifted_name or "",
        "global_max_abs": global_m["max_abs"],
        "global_mean_abs": global_m["mean_abs"],
        "global_max_rel": global_m["max_rel"],
        "global_allclose": int(global_ok),
        "attn_max_abs": attn_max_abs,
        "attn_mean_abs": attn_mean_abs,
        "attn_max_rel": attn_max_rel,
        "attn_allclose": int(attn_ok),
        "triton_backend_calls": int(sum(v for k, v in backend_calls.items() if k.startswith("triton"))),
        "pytorch_backend_calls": int(backend_calls.get("pytorch", 0)),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="tiny_w8,small_w8,base_w8")
    parser.add_argument("--sizes", default="512,1024,1536,2048")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/current/results/swin_elsa_exactness_fp32.csv"),
    )
    args = parser.parse_args()

    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    model_names: list[str] = []
    for k in model_keys:
        if k not in MODEL_MAP:
            raise ValueError(f"Unknown model key: {k}")
        model_names.append(MODEL_MAP[k])
    sizes = _parse_int_list(args.sizes)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    patch_elsa_window_attention()

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for size in sizes:
            rows.append(
                _run_one(
                    model_name=model_name,
                    size=size,
                    device=device,
                    seed=args.seed,
                    atol=args.atol,
                    rtol=args.rtol,
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
