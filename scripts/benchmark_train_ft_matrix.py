#!/usr/bin/env python3
"""Unified matrix benchmark for full-model train / backward-only / finetune.

Matrix axes:
- precision: fp32 / tf32 / fp16
- mode: train / backward / finetune
- family: vit / swin
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

import timm
from timm.models.elsa import set_default_elsa_backend as set_default_vit_backend
from timm.models.elsa_swin import set_default_elsa_backend as set_default_swin_backend
from timm.models.elsa_swin_fused import patch_elsa_window_attention

PRECISIONS = ("fp32", "tf32", "fp16")
MODES = ("train", "backward", "finetune")
FAMILIES = ("vit", "swin")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    elsa_model: str
    baseline_model: str
    img_size: int
    batch_size: int
    num_classes: int = 1000


MODEL_SPECS: Dict[str, ModelSpec] = {
    "vit_small_512": ModelSpec(
        key="vit_small_512",
        family="vit",
        elsa_model="elsa3_small_patch16_384",
        baseline_model="deit3_small_patch16_384",
        img_size=512,
        batch_size=2,
    ),
    "vit_base_512": ModelSpec(
        key="vit_base_512",
        family="vit",
        elsa_model="elsa3_base_patch16_384",
        baseline_model="deit3_base_patch16_384",
        img_size=512,
        batch_size=1,
    ),
    "vit_large_512": ModelSpec(
        key="vit_large_512",
        family="vit",
        elsa_model="elsa3_large_patch16_384",
        baseline_model="deit3_large_patch16_384",
        img_size=512,
        batch_size=1,
    ),
    "swin_tiny_w8_256": ModelSpec(
        key="swin_tiny_w8_256",
        family="swin",
        elsa_model="elsa_tiny_window8_256",
        baseline_model="swinv2_tiny_window8_256",
        img_size=256,
        batch_size=2,
    ),
    "swin_small_w8_256": ModelSpec(
        key="swin_small_w8_256",
        family="swin",
        elsa_model="elsa_small_window8_256",
        baseline_model="swinv2_small_window8_256",
        img_size=256,
        batch_size=2,
    ),
    "swin_base_w8_256": ModelSpec(
        key="swin_base_w8_256",
        family="swin",
        elsa_model="elsa_base_window8_256",
        baseline_model="swinv2_base_window8_256",
        img_size=256,
        batch_size=1,
    ),
}


@contextlib.contextmanager
def temp_environ(overrides: Dict[str, Optional[str]]) -> Iterable[None]:
    prev = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def tf32_guard(enabled: bool) -> Iterable[None]:
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_env = os.environ.get("NVIDIA_TF32_OVERRIDE")
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1" if enabled else "0"
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        if prev_env is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        else:
            os.environ["NVIDIA_TF32_OVERRIDE"] = prev_env


def sdpa_ctx(kind: str) -> contextlib.AbstractContextManager[None]:
    if kind == "math":
        return torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)
    if kind == "mem":
        return torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)
    if kind == "flash":
        return torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False, enable_flash=True)
    raise ValueError(f"Unsupported SDPA kind: {kind}")


def _backend_plan(family: str, precision: str) -> List[str]:
    if family == "vit":
        if precision == "fp16":
            return ["elsa", "sdpa_math", "sdpa_flash"]
        return ["elsa", "sdpa_math", "sdpa_mem"]
    return ["elsa", "swin_window"]


def _elsa_backend_for_precision(precision: str, family: str) -> str:
    if family == "swin":
        # Swin train path defaults to the dedicated training kernel route.
        return os.environ.get("ELSA_TRAIN_SWIN_BACKEND", "swin_train_kernel").lower()
    if precision == "fp32":
        # Dedicated training path: keep inference fp32 route untouched.
        return os.environ.get("ELSA_TRAIN_FP32_BACKEND", "triton_fp32_train").lower()
    if precision == "tf32":
        # Training path also uses dedicated fp32-train route; TF32 behavior is
        # controlled by tf32_guard + kernel policy inside elsa_triton.
        return os.environ.get("ELSA_TRAIN_TF32_BACKEND", "triton_fp32_train").lower()
    # fp16 full kernel can exceed Triton launch-resource limits on 512-image ViT specs.
    # Use train-stable triton backend by default; allow override for ablations.
    return os.environ.get("ELSA_TRAIN_FP16_BACKEND", "triton").lower()


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    return torch.float16 if precision == "fp16" else None


def _optimizer_for_mode(
    model: torch.nn.Module,
    mode: str,
    family: str,
    vit_train_lr: float,
    swin_train_lr: float,
):
    if mode == "finetune":
        # Keep full-model finetuning but use conservative SGD to preserve loss-path comparability.
        return torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    train_lr = vit_train_lr if family == "vit" else swin_train_lr
    return torch.optim.SGD(model.parameters(), lr=train_lr, momentum=0.9)


def _run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    warmup: int,
    mode: str,
    autocast_dtype: Optional[torch.dtype],
    scaler: Optional[torch.amp.GradScaler],
    record_losses: bool,
) -> Dict[str, object]:
    assert mode in MODES
    model.train()
    amp_enabled = autocast_dtype is not None
    losses_gpu: List[torch.Tensor] = []

    def do_forward():
        with torch.amp.autocast(device_type=inputs.device.type, dtype=autocast_dtype, enabled=amp_enabled):
            logits = model(inputs)
            return F.cross_entropy(logits, targets)

    def warm_step():
        optimizer.zero_grad(set_to_none=True)
        loss = do_forward()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    for _ in range(warmup):
        warm_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    if mode in ("train", "finetune"):
        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = do_forward()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if record_losses:
                losses_gpu.append(loss.detach())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    else:
        # backward-only: exclude forward time by timing only backward pass; keep parameters fixed.
        elapsed = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = do_forward()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            if record_losses:
                losses_gpu.append(loss.detach())

    step_ms = (elapsed / steps) * 1e3
    imgs_per_s = (inputs.shape[0] * steps) / elapsed if elapsed > 0 else float("nan")
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    losses = torch.stack(losses_gpu).float().cpu().tolist() if losses_gpu else []
    return {
        "step_ms": step_ms,
        "imgs_per_s": imgs_per_s,
        "peak_gb": peak_gb,
        "epoch_s": elapsed,
        "losses": losses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified train/finetune/backward-only matrix benchmark.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    parser.add_argument("--specs", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vit-train-lr", type=float, default=1e-3)
    parser.add_argument("--swin-train-lr", type=float, default=3e-4)
    parser.add_argument(
        "--img-size-override",
        type=int,
        default=0,
        help="Override img_size for selected specs (0 = use spec default).",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=0,
        help="Override batch size for selected specs (0 = use spec default).",
    )
    parser.add_argument("--swin-fused", action="store_true", help="Enable fused Swin window-attention patch.")
    parser.add_argument(
        "--vit-fp16-backends",
        nargs="+",
        choices=("elsa", "sdpa_math", "sdpa_flash"),
        default=None,
        help="Optional backend subset for ViT FP16 (default keeps full plan: elsa/sdpa_math/sdpa_flash).",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each case this many times.")
    parser.add_argument(
        "--alternate-backend-order",
        action="store_true",
        help="Reverse backend order on odd repeats to reduce order bias.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/current/results/train_ft_matrix.csv"))
    parser.add_argument("--loss-output", type=Path, default=Path("artifacts/current/results/train_ft_matrix_loss.csv"))
    return parser.parse_args()


def _build_model(
    spec: ModelSpec,
    backend_id: str,
    precision: str,
    device: torch.device,
    runtime_img_size: Optional[int] = None,
) -> torch.nn.Module:
    img_size = int(runtime_img_size) if runtime_img_size is not None else int(spec.img_size)
    if spec.family == "vit":
        if backend_id == "elsa":
            elsa_backend = _elsa_backend_for_precision(precision, spec.family)
        else:
            elsa_backend = backend_id
        set_default_vit_backend(elsa_backend)
        model = timm.create_model(
            spec.elsa_model,
            pretrained=False,
            img_size=img_size,
            num_classes=spec.num_classes,
            elsa_backend=elsa_backend,
        ).to(device)
        return model

    # swin family
    if backend_id == "elsa":
        elsa_backend = _elsa_backend_for_precision(precision, spec.family)
        set_default_swin_backend(elsa_backend)
        model = timm.create_model(
            spec.elsa_model,
            pretrained=False,
            img_size=img_size,
            num_classes=spec.num_classes,
            triton=True,
            elsa_backend=elsa_backend,
        ).to(device)
        return model

    # baseline window attention
    model = timm.create_model(
        spec.baseline_model,
        pretrained=False,
        img_size=img_size,
        num_classes=spec.num_classes,
    ).to(device)
    return model


def _backend_label(spec: ModelSpec, backend_id: str, precision: str) -> Tuple[str, str]:
    if spec.family == "vit":
        if backend_id == "elsa":
            return f"ELSA-{precision.upper()}", _elsa_backend_for_precision(precision, spec.family)
        if backend_id == "sdpa_math":
            return "SDPA-Math", "sdpa_math"
        if backend_id == "sdpa_mem":
            return "SDPA-Mem", "sdpa_mem"
        if backend_id == "sdpa_flash":
            return "SDPA-Flash", "sdpa_flash"
    else:
        if backend_id == "elsa":
            return f"ELSA-Swin-{precision.upper()}", _elsa_backend_for_precision(precision, spec.family)
        if backend_id == "swin_window":
            return "Swin-Window", "window_attn"
    raise ValueError(f"Unknown backend_id {backend_id} for family {spec.family}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    if args.swin_fused:
        patch_elsa_window_attention()

    rows: List[Dict[str, object]] = []
    loss_rows: List[Dict[str, object]] = []

    for spec_key in args.specs:
        spec = MODEL_SPECS[spec_key]
        if spec.family not in args.families:
            continue
        runtime_img_size = int(args.img_size_override) if int(args.img_size_override) > 0 else int(spec.img_size)
        runtime_batch = int(args.batch_size_override) if int(args.batch_size_override) > 0 else int(spec.batch_size)

        for precision in args.precisions:
            input_dtype = torch.float16 if precision == "fp16" else torch.float32
            autocast_dtype = _autocast_dtype(precision)
            use_tf32 = precision == "tf32"

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            inputs = torch.randn(
                runtime_batch,
                3,
                runtime_img_size,
                runtime_img_size,
                device=device,
                dtype=input_dtype,
            )
            targets = torch.randint(0, spec.num_classes, (runtime_batch,), device=device)

            backends = _backend_plan(spec.family, precision)
            if spec.family == "vit" and precision == "fp16" and args.vit_fp16_backends:
                backends = list(args.vit_fp16_backends)
            repeats = max(1, int(args.repeats))
            for rep in range(repeats):
                run_backends = list(backends)
                if args.alternate_backend_order and (rep % 2 == 1):
                    run_backends = list(reversed(run_backends))

                for mode in args.modes:
                    for backend_id in run_backends:
                        label, backend_name = _backend_label(spec, backend_id, precision)
                        status = "ok"
                        stats: Dict[str, object] = {}
                        model = None
                        try:
                            torch.manual_seed(args.seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(args.seed)
                            model = _build_model(
                                spec,
                                backend_id,
                                precision,
                                device,
                                runtime_img_size=runtime_img_size,
                            )
                            optimizer = _optimizer_for_mode(
                                model=model,
                                mode=mode,
                                family=spec.family,
                                vit_train_lr=args.vit_train_lr,
                                swin_train_lr=args.swin_train_lr,
                            )
                            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

                            ctx_factory = contextlib.nullcontext
                            if spec.family == "vit" and backend_id.startswith("sdpa_"):
                                kind = backend_id.split("_", 1)[1]
                                ctx_factory = lambda k=kind: sdpa_ctx(k)

                            with tf32_guard(use_tf32), ctx_factory():
                                stats = _run_epoch(
                                    model=model,
                                    optimizer=optimizer,
                                    inputs=inputs,
                                    targets=targets,
                                    steps=args.steps,
                                    warmup=args.warmup,
                                    mode=mode,
                                    autocast_dtype=autocast_dtype,
                                    scaler=scaler,
                                    record_losses=True,
                                )
                        except Exception as exc:
                            if os.environ.get("ELSA_BENCH_DEBUG_EXC", "0") == "1":
                                import traceback
                                traceback.print_exc()
                            status = f"error:{exc.__class__.__name__}"
                        finally:
                            if model is not None:
                                del model
                            torch.cuda.empty_cache()

                        rows.append(
                            {
                                "family": spec.family,
                                "variant": spec.key,
                                "mode": mode,
                                "precision": precision,
                                "backend": label,
                                "backend_id": backend_id,
                                "backend_kernel": backend_name,
                                "seed": args.seed,
                                "rep": rep,
                                "img_size": runtime_img_size,
                                "batch": runtime_batch,
                                "steps": args.steps,
                                "warmup": args.warmup,
                                "status": status,
                                "step_ms": float(stats.get("step_ms", float("nan"))),
                                "imgs_per_s": float(stats.get("imgs_per_s", float("nan"))),
                                "peak_gb": float(stats.get("peak_gb", float("nan"))),
                                "epoch_s": float(stats.get("epoch_s", float("nan"))),
                            }
                        )

                        if stats.get("losses"):
                            for i, loss_val in enumerate(stats["losses"], start=1):
                                loss_rows.append(
                                    {
                                        "family": spec.family,
                                        "variant": spec.key,
                                        "mode": mode,
                                        "precision": precision,
                                        "backend": label,
                                        "backend_id": backend_id,
                                        "seed": args.seed,
                                        "rep": rep,
                                        "img_size": runtime_img_size,
                                        "batch": runtime_batch,
                                        "step": i,
                                        "loss": float(loss_val),
                                    }
                                )

    if not rows:
        raise RuntimeError("No rows produced.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    args.loss_output.parent.mkdir(parents=True, exist_ok=True)
    if loss_rows:
        with args.loss_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(loss_rows[0].keys()))
            writer.writeheader()
            writer.writerows(loss_rows)

    print(f"[OK] wrote {args.output}")
    print(f"[OK] wrote {args.loss_output}")


if __name__ == "__main__":
    main()
