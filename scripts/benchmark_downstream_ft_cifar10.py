#!/usr/bin/env python3
"""Downstream 1-epoch finetune matrix on CIFAR-10.

Goals:
- verify ELSA finetuning viability on a real task (Top-1)
- compare efficiency (step latency / epoch time / peak VRAM)
- cover fp32 / tf32 / fp16 precision combinations
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

import timm
from timm.models.elsa import set_default_elsa_backend as set_default_vit_backend
from timm.models.elsa_swin import set_default_elsa_backend as set_default_swin_backend

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    elsa_model: str
    baseline_model: str
    img_size: int
    batch_size: int
    local_init_ckpt: Optional[str] = None


MODEL_SPECS: Dict[str, ModelSpec] = {
    "vit_small_224": ModelSpec(
        key="vit_small_224",
        family="vit",
        elsa_model="elsa3_small_patch16_224",
        baseline_model="deit3_small_patch16_224",
        img_size=224,
        batch_size=32,
        local_init_ckpt=None,
    ),
    "swin_small_w8_256": ModelSpec(
        key="swin_small_w8_256",
        family="swin",
        elsa_model="elsa_small_window8_256",
        baseline_model="swinv2_small_window8_256",
        img_size=256,
        batch_size=32,
        local_init_ckpt=None,
    ),
}

PRECISIONS = ("fp32", "tf32", "fp16")


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


def _elsa_backend_for_precision(precision: str, family: str, mode: str) -> str:
    if mode == "legacy_full":
        if family == "swin":
            return "triton"
        if precision == "fp32":
            return "triton_full_fp32"
        if precision == "tf32":
            return "triton_full_turbo"
        return "triton_full"

    # train_safe: choose backends that preserve training gradients.
    if family == "swin":
        return os.environ.get("ELSA_FT_SWIN_BACKEND", "triton").lower()
    if precision in ("fp32", "tf32"):
        # Keep downstream FT aligned with proven stable-fwd route used by prior
        # reproducible top1 runs in this repo.
        return os.environ.get("ELSA_FT_VIT_FP32_BACKEND", "triton_fp32").lower()
    return os.environ.get("ELSA_FT_VIT_FP16_BACKEND", "triton").lower()


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    return torch.float16 if precision == "fp16" else None


def _unwrap_state_dict(ckpt: dict) -> dict:
    for key in ("state_dict", "model", "model_state", "net", "module"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _load_matching_state(model: torch.nn.Module, state: dict) -> Tuple[int, int]:
    target = model.state_dict()
    matched = {}
    for k, v in state.items():
        if k in target and target[k].shape == v.shape:
            matched[k] = v
    target.update(matched)
    model.load_state_dict(target, strict=False)
    return len(matched), len(target)


def _build_model(
    spec: ModelSpec,
    backend_id: str,
    precision: str,
    elsa_backend_mode: str,
    num_classes: int,
    device: torch.device,
    use_pretrained_init: bool,
) -> Tuple[torch.nn.Module, str]:
    init_info = "random"
    if spec.family == "vit":
        if backend_id == "elsa":
            elsa_backend = _elsa_backend_for_precision(precision, spec.family, elsa_backend_mode)
            set_default_vit_backend(elsa_backend)
            model = timm.create_model(
                spec.elsa_model,
                pretrained=False,
                img_size=spec.img_size,
                num_classes=num_classes,
                elsa_backend=elsa_backend,
            )
            if use_pretrained_init:
                donor = timm.create_model(
                    spec.baseline_model,
                    pretrained=True,
                    img_size=spec.img_size,
                    num_classes=1000,
                )
                matched, total = _load_matching_state(model, donor.state_dict())
                init_info = f"donor:{spec.baseline_model}:{matched}/{total}"
                del donor
        elif backend_id in ("sdpa_math", "sdpa_mem", "sdpa_flash"):
            model = timm.create_model(
                spec.baseline_model,
                pretrained=use_pretrained_init,
                img_size=spec.img_size,
                num_classes=num_classes,
            )
            init_info = "pretrained" if use_pretrained_init else "random"
        else:
            raise ValueError(f"Unknown backend for vit: {backend_id}")
        return model.to(device), init_info

    # Swin
    if backend_id == "elsa":
        elsa_backend = _elsa_backend_for_precision(precision, spec.family, elsa_backend_mode)
        set_default_swin_backend(elsa_backend)
        model = timm.create_model(
            spec.elsa_model,
            pretrained=False,
            img_size=spec.img_size,
            num_classes=num_classes,
            triton=True,
            elsa_backend=elsa_backend,
        )
        if use_pretrained_init:
            loaded = False
            if spec.local_init_ckpt:
                ckpt_path = Path(spec.local_init_ckpt)
                if ckpt_path.exists():
                    raw = torch.load(ckpt_path, map_location="cpu")
                    state = _unwrap_state_dict(raw)
                    matched, total = _load_matching_state(model, state)
                    init_info = f"local:{ckpt_path.name}:{matched}/{total}"
                    loaded = True
            if not loaded:
                donor = timm.create_model(
                    spec.baseline_model,
                    pretrained=True,
                    img_size=spec.img_size,
                    num_classes=1000,
                )
                matched, total = _load_matching_state(model, donor.state_dict())
                init_info = f"donor:{spec.baseline_model}:{matched}/{total}"
                del donor
    elif backend_id == "swin_window":
        model = timm.create_model(
            spec.baseline_model,
            pretrained=use_pretrained_init,
            img_size=spec.img_size,
            num_classes=num_classes,
        )
        init_info = "pretrained" if use_pretrained_init else "random"
    else:
        raise ValueError(f"Unknown backend for swin: {backend_id}")

    return model.to(device), init_info


def _get_classifier_linear(model: torch.nn.Module) -> Optional[nn.Linear]:
    # Common timm heads
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return head
    if head is not None and hasattr(head, "fc") and isinstance(head.fc, nn.Linear):
        return head.fc
    cls = getattr(model, "classifier", None)
    if isinstance(cls, nn.Linear):
        return cls
    # Fallback to get_classifier API if available
    getter = getattr(model, "get_classifier", None)
    if callable(getter):
        out = getter()
        if isinstance(out, nn.Linear):
            return out
    return None


def _align_classifier_init(model: torch.nn.Module, seed: int) -> bool:
    """Re-init classifier head with a fixed seed for backend-fair FT comparison."""
    fc = _get_classifier_linear(model)
    if fc is None:
        return False
    devices = []
    if fc.weight.is_cuda:
        devices = [fc.weight.device]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        nn.init.trunc_normal_(fc.weight, std=0.02)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
    return True


def _backend_label(spec: ModelSpec, backend_id: str, precision: str) -> str:
    if spec.family == "vit":
        if backend_id == "elsa":
            return f"ELSA-{precision.upper()}"
        if backend_id == "sdpa_math":
            return "SDPA-Math"
        if backend_id == "sdpa_mem":
            return "SDPA-Mem"
        if backend_id == "sdpa_flash":
            return "SDPA-Flash"
    else:
        if backend_id == "elsa":
            return f"ELSA-Swin-{precision.upper()}"
        if backend_id == "swin_window":
            return "Swin-Window"
    raise ValueError(f"Unknown backend {backend_id}")


def _get_real_torchvision():
    # This repo injects a lightweight torchvision stub via sitecustomize.
    # For downstream finetune we need the real torchvision datasets/transforms.
    try:
        from torchvision.datasets import CIFAR10 as _CIFAR10  # type: ignore
        if getattr(_CIFAR10, "__module__", "").startswith("sitecustomize"):
            for mod_name in list(sys.modules):
                if mod_name == "torchvision" or mod_name.startswith("torchvision."):
                    del sys.modules[mod_name]
    except Exception:
        pass
    import torchvision  # type: ignore
    from torchvision.datasets import CIFAR10  # type: ignore
    from torchvision import transforms as T  # type: ignore
    return CIFAR10, T


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = int(base_seed + worker_id)
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)


def _make_loaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    train_indices: List[int],
    val_indices: List[int],
    workers: int,
    seed: int,
    deterministic: bool,
    disable_train_augment: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    CIFAR10, T = _get_real_torchvision()
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf_ops = [T.Resize((img_size, img_size), antialias=True)]
    if not disable_train_augment:
        train_tf_ops.append(T.RandomHorizontalFlip())
    train_tf_ops.extend([T.ToTensor(), normalize])
    train_tf = T.Compose(
        train_tf_ops
    )
    val_tf = T.Compose(
        [
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            normalize,
        ]
    )
    train_ds = CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
    val_ds = CIFAR10(root=str(data_dir), train=False, download=True, transform=val_tf)

    train_sub = torch.utils.data.Subset(train_ds, train_indices)
    val_sub = torch.utils.data.Subset(val_ds, val_indices)

    train_gen = torch.Generator()
    train_gen.manual_seed(int(seed))
    val_gen = torch.Generator()
    val_gen.manual_seed(int(seed) + 1)
    worker_init_fn = partial(_seed_worker, base_seed=seed) if deterministic else None

    train_loader = torch.utils.data.DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=train_gen,
        worker_init_fn=worker_init_fn,
        persistent_workers=bool(workers > 0 and not deterministic),
    )
    val_loader = torch.utils.data.DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        generator=val_gen,
        worker_init_fn=worker_init_fn,
        persistent_workers=bool(workers > 0 and not deterministic),
    )
    return train_loader, val_loader


def _evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    ctx_factory,
) -> Tuple[float, float]:
    model.eval()
    amp_enabled = autocast_dtype is not None
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with ctx_factory():
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, targets)
            pred = logits.argmax(dim=1)
            correct += int((pred == targets).sum().item())
            total += targets.numel()
            loss_sum += float(loss.item()) * targets.size(0)
    top1 = 100.0 * correct / max(1, total)
    val_loss = loss_sum / max(1, total)
    return top1, val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream 1-epoch finetune matrix on CIFAR-10.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--specs", nargs="+", default=list(MODEL_SPECS.keys()), choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--vit-lr", type=float, default=5e-4)
    parser.add_argument("--swin-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict reproducibility (deterministic kernels, fixed loader RNG per backend).",
    )
    parser.add_argument(
        "--disable-train-augment",
        action="store_true",
        help="Disable random train-time augmentation (e.g., RandomHorizontalFlip).",
    )
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=None,
        help="Seed for data order/augmentation RNG. Defaults to --seed.",
    )
    parser.add_argument(
        "--elsa-backend-mode",
        choices=("train_safe", "legacy_full"),
        default="train_safe",
        help="ELSA backend policy for finetune. train_safe avoids non-differentiable full kernels.",
    )
    parser.add_argument(
        "--align-head-init",
        action="store_true",
        help="Re-initialize classifier head with fixed seed across backends for fair 1-epoch FT.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/cifar10"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/current/results/downstream_ft_cifar10_matrix.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        # Needed for deterministic CuBLAS GEMM paths on CUDA >= 10.2.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    train_perm = torch.randperm(50000, generator=g).tolist()
    val_perm = torch.randperm(10000, generator=g).tolist()
    train_indices = train_perm[: args.train_samples]
    val_indices = val_perm[: args.val_samples]

    rows: List[Dict[str, object]] = []
    loader_cache: Dict[Tuple[int, int], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]] = {}
    loader_seed = args.seed if args.loader_seed is None else args.loader_seed

    for spec_key in args.specs:
        spec = MODEL_SPECS[spec_key]
        if not args.deterministic:
            if (spec.img_size, spec.batch_size) not in loader_cache:
                loader_cache[(spec.img_size, spec.batch_size)] = _make_loaders(
                    data_dir=args.data_dir,
                    img_size=spec.img_size,
                    batch_size=spec.batch_size,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    workers=args.workers,
                    seed=loader_seed,
                    deterministic=False,
                    disable_train_augment=args.disable_train_augment,
                )
            train_loader, val_loader = loader_cache[(spec.img_size, spec.batch_size)]
        else:
            train_loader, val_loader = (None, None)  # created per-backend for strict fairness

        for precision in args.precisions:
            backends = _backend_plan(spec.family, precision)
            use_tf32 = precision == "tf32"
            autocast_dtype = _autocast_dtype(precision)
            amp_enabled = autocast_dtype is not None

            for backend_id in backends:
                label = _backend_label(spec, backend_id, precision)
                status = "ok"
                top1 = float("nan")
                val_loss = float("nan")
                train_loss_last = float("nan")
                epoch_s = float("nan")
                step_ms = float("nan")
                imgs_per_s = float("nan")
                peak_gb = float("nan")
                init_info = "unknown"

                model = None
                effective_backend = backend_id
                try:
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)

                    if backend_id == "elsa":
                        effective_backend = _elsa_backend_for_precision(precision, spec.family, args.elsa_backend_mode)

                    if args.deterministic:
                        train_loader, val_loader = _make_loaders(
                            data_dir=args.data_dir,
                            img_size=spec.img_size,
                            batch_size=spec.batch_size,
                            train_indices=train_indices,
                            val_indices=val_indices,
                            workers=args.workers,
                            seed=loader_seed,
                            deterministic=True,
                            disable_train_augment=args.disable_train_augment,
                        )

                    model, init_info = _build_model(
                        spec=spec,
                        backend_id=backend_id,
                        precision=precision,
                        elsa_backend_mode=args.elsa_backend_mode,
                        num_classes=10,
                        device=device,
                        use_pretrained_init=True,
                    )
                    if args.align_head_init:
                        aligned = _align_classifier_init(model, args.seed + 1009)
                        if aligned:
                            init_info = f"{init_info}|head_aligned"

                    lr = args.vit_lr if spec.family == "vit" else args.swin_lr
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
                    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
                    criterion = nn.CrossEntropyLoss()

                    if spec.family == "vit" and backend_id.startswith("sdpa_"):
                        kind = backend_id.split("_", 1)[1]
                        ctx_factory = lambda k=kind: sdpa_ctx(k)
                    else:
                        ctx_factory = contextlib.nullcontext

                    model.train()
                    step_times: List[float] = []
                    step_count = 0

                    with tf32_guard(use_tf32):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        epoch_start = time.perf_counter()
                        for _ in range(args.epochs):
                            for images, targets in train_loader:
                                images = images.to(device, non_blocking=True)
                                targets = targets.to(device, non_blocking=True)

                                optimizer.zero_grad(set_to_none=True)
                                t0 = torch.cuda.Event(enable_timing=True)
                                t1 = torch.cuda.Event(enable_timing=True)
                                t0.record()
                                with ctx_factory():
                                    with torch.amp.autocast(
                                        device_type="cuda",
                                        dtype=autocast_dtype,
                                        enabled=amp_enabled,
                                    ):
                                        logits = model(images)
                                        loss = criterion(logits, targets)
                                if amp_enabled:
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    loss.backward()
                                    optimizer.step()
                                t1.record()
                                torch.cuda.synchronize()

                                step_count += 1
                                train_loss_last = float(loss.item())
                                if step_count > args.warmup_steps:
                                    step_times.append(float(t0.elapsed_time(t1)))

                        epoch_s = time.perf_counter() - epoch_start
                        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

                        measured_steps = max(1, len(step_times))
                        step_ms = sum(step_times) / measured_steps
                        imgs_per_s = (len(train_loader.dataset) * args.epochs) / max(epoch_s, 1e-9)

                        top1, val_loss = _evaluate(
                            model=model,
                            loader=val_loader,
                            device=device,
                            autocast_dtype=autocast_dtype,
                            ctx_factory=ctx_factory,
                        )
                except Exception as exc:
                    status = f"error:{exc.__class__.__name__}"
                finally:
                    if model is not None:
                        del model
                    torch.cuda.empty_cache()

                rows.append(
                    {
                        "family": spec.family,
                        "variant": spec.key,
                        "precision": precision,
                        "backend": label,
                        "backend_id": backend_id,
                        "effective_backend": effective_backend,
                        "elsa_backend_mode": args.elsa_backend_mode,
                        "status": status,
                        "init_info": init_info,
                        "train_samples": args.train_samples,
                        "val_samples": args.val_samples,
                        "epochs": args.epochs,
                        "batch_size": spec.batch_size,
                        "img_size": spec.img_size,
                        "warmup_steps": args.warmup_steps,
                        "deterministic": int(args.deterministic),
                        "disable_train_augment": int(args.disable_train_augment),
                        "loader_seed": int(loader_seed),
                        "train_loss_last": train_loss_last,
                        "val_loss": val_loss,
                        "top1": top1,
                        "epoch_s": epoch_s,
                        "step_ms": step_ms,
                        "imgs_per_s": imgs_per_s,
                        "peak_gb": peak_gb,
                    }
                )

                print(
                    f"[{spec.key}][{precision}][{label}] status={status} "
                    f"top1={top1:.3f} step_ms={step_ms:.3f} peak_gb={peak_gb:.3f}"
                )

    if not rows:
        raise RuntimeError("No results generated.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
