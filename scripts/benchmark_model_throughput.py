#!/usr/bin/env python
"""Full-model throughput & peak memory benchmarks for ViT / Swin ELSA variants."""
from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

import timm
from timm.models.vision_transformer import Attention

from experiments.a100_run.run_all import (  # noqa: E402
    FLASH_ATTN_AVAILABLE,
    flash_attn_v2,
    flash_attn_v3,
)


@dataclass(frozen=True)
class ModelSpec:
    family: str
    size: str
    elsa_name: str
    baseline_name: str
    input_size: int = 224


VIT_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("vit", "tiny", "elsa_tiny_patch16_224", "deit_tiny_patch16_224"),
    ModelSpec("vit", "small", "elsa_small_patch16_224", "deit_small_patch16_224"),
    ModelSpec("vit", "medium", "elsa3_medium_patch16_224", "deit3_medium_patch16_224"),
    ModelSpec("vit", "base", "elsa3_base_patch16_224", "deit3_base_patch16_224"),
)

SWIN_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec("swin", "tiny_w8", "elsa_tiny_window8_256", "swinv2_tiny_window8_256", input_size=256),
    ModelSpec("swin", "tiny_w16", "elsa_tiny_window16_256", "swinv2_tiny_window16_256", input_size=256),
    ModelSpec("swin", "small_w8", "elsa_small_window8_256", "swinv2_small_window8_256", input_size=256),
    ModelSpec("swin", "small_w16", "elsa_small_window16_256", "swinv2_small_window16_256", input_size=256),
)


def make_inputs(batch: int, channels: int, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, channels, size, size, device=device, dtype=dtype)


def _flash_forward_factory(kind: str) -> Callable[[Attention, torch.Tensor], torch.Tensor]:
    if kind not in {"fa2", "fa3"}:
        raise ValueError(f"Unsupported flash kind {kind}")

    def forward(self: Attention, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if kind == "fa2":
            if not FLASH_ATTN_AVAILABLE or flash_attn_v2 is None:
                raise RuntimeError("flash-attn v2 unavailable")
            q_, k_, v_ = (tensor.contiguous().half() for tensor in (q, k, v))
            out = flash_attn_v2(
                q_.permute(0, 2, 1, 3).contiguous(),
                k_.permute(0, 2, 1, 3).contiguous(),
                v_.permute(0, 2, 1, 3).contiguous(),
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )
            out = out.permute(0, 2, 1, 3).contiguous()
        else:
            if not FLASH_ATTN_AVAILABLE or flash_attn_v3 is None:
                raise RuntimeError("flash-attn v3 unavailable")
            q_ = q.permute(0, 2, 1, 3).contiguous().half()
            k_ = k.permute(0, 2, 1, 3).contiguous().half()
            v_ = v.permute(0, 2, 1, 3).contiguous().half()
            B_, H, N_, D = q_.shape
            qkv = torch.stack([q_, k_, v_], dim=2).view(B_ * N_, 3, H, D)
            cu = torch.arange(0, (B_ + 1) * N_, N_, dtype=torch.int32, device=q.device)
            out = flash_attn_v3(qkv, cu, N_, causal=False)
            out = out.view(B_, N_, H, D).permute(0, 2, 1, 3).contiguous()

        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    return forward


def patch_vit_flash_attention(model: torch.nn.Module, kind: str) -> None:
    import types

    for module in model.modules():
        if isinstance(module, Attention):
            module.fused_attn = False
            module.forward = types.MethodType(_flash_forward_factory(kind), module)


def toggle_tf32(enable: bool) -> contextlib.AbstractContextManager[None]:
    class _TF32Ctx(contextlib.AbstractContextManager[None]):
        def __enter__(self):
            self.prev_matmul = torch.backends.cuda.matmul.allow_tf32
            self.prev_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = enable
            torch.backends.cudnn.allow_tf32 = enable
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.backends.cuda.matmul.allow_tf32 = self.prev_matmul
            torch.backends.cudnn.allow_tf32 = self.prev_cudnn
            return False

    return _TF32Ctx()


def sdp_context(math: bool = False, mem: bool = False, flash: bool = False):
    return torch.backends.cuda.sdp_kernel(
        enable_math=math,
        enable_mem_efficient=mem,
        enable_flash=flash,
    )


def measure_model(
    model: torch.nn.Module,
    batch: int,
    input_size: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    trials: int,
    autocast_dtype: Optional[torch.dtype] = None,
    ctx_factories: Optional[Iterable[Callable[[], contextlib.AbstractContextManager[None]]]] = None,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    if dtype == torch.float16:
        model = model.half()
    torch.cuda.empty_cache()
    x = make_inputs(batch, 3, input_size, dtype if autocast_dtype is None else torch.float32, device)

    def run_once():
        with torch.no_grad():
            cm = contextlib.ExitStack()
            if ctx_factories:
                for factory in ctx_factories:
                    cm.enter_context(factory())
            if autocast_dtype is not None:
                cm.enter_context(torch.cuda.amp.autocast(dtype=autocast_dtype))
            with cm:
                out = model(x.to(dtype=dtype, device=device))
        return out

    with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
        for _ in range(warmup):
            run_once()
    torch.cuda.synchronize()
    # Reset peak after warmup so Triton JIT/autotune and one-time setup
    # memory are excluded from steady-state VRAM measurement.
    torch.cuda.reset_peak_memory_stats(device)

    latencies: List[float] = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.time()
        run_once()
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1e3)

    lat_ms_avg = float(statistics.mean(latencies))
    lat_ms_med = float(statistics.median(latencies))
    if len(latencies) >= 20:
        lat_ms_p95 = float(statistics.quantiles(latencies, n=20)[18])
    elif len(latencies) >= 2:
        lat_ms_p95 = float(statistics.quantiles(latencies, n=4)[-1])
    else:
        lat_ms_p95 = lat_ms_med
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    imgs_per_s = (batch / (lat_ms_med / 1e3)) if lat_ms_med > 0 else float("nan")

    return {
        "lat_ms_avg": lat_ms_avg,
        "lat_ms_med": lat_ms_med,
        "lat_ms_p95": lat_ms_p95,
        "peak_gb": peak_gb,
        "imgs_per_s": imgs_per_s,
    }


def run_benchmarks(device: torch.device, warmup: int, trials: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_fp16: List[Dict[str, object]] = []
    results_fp32: List[Dict[str, object]] = []

    # ViT FP16: ELSA / FA2 / FA3
    for spec in VIT_SPECS:
        # ELSA
        model = timm.create_model(spec.elsa_name, pretrained=False)
        stats = measure_model(model, batch=1, input_size=spec.input_size, dtype=torch.float16, device=device, warmup=warmup, trials=trials)
        results_fp16.append(
            dict(
                family=spec.family,
                size=spec.size,
                backend="ELSA",
                dtype="fp16",
                status="ok",
                **stats,
            )
        )
        del model

        # FlashAttention v2 / v3
        for kind, label in (("fa2", "FA2"), ("fa3", "FA3")):
            if kind == "fa3" and (not FLASH_ATTN_AVAILABLE or flash_attn_v3 is None):
                results_fp16.append(
                    dict(
                        family=spec.family,
                        size=spec.size,
                        backend=label,
                        dtype="fp16",
                        status="unavailable",
                        lat_ms_avg=float("nan"),
                        lat_ms_med=float("nan"),
                        lat_ms_p95=float("nan"),
                        peak_gb=float("nan"),
                        imgs_per_s=float("nan"),
                    )
                )
                continue
            model = timm.create_model(spec.baseline_name, pretrained=False)
            patch_vit_flash_attention(model, kind=kind)
            stats = measure_model(
                model,
                batch=1,
                input_size=spec.input_size,
                dtype=torch.float16,
                device=device,
                warmup=warmup,
                trials=trials,
                autocast_dtype=torch.float16,
            )
            results_fp16.append(
                dict(
                    family=spec.family,
                    size=spec.size,
                    backend=label,
                    dtype="fp16",
                    status="ok",
                    **stats,
                )
            )
            del model

    # Swin FP16: ELSA / FA2 (+ optional FA3)
    for spec in SWIN_SPECS:
        model = timm.create_model(spec.elsa_name, pretrained=False)
        stats = measure_model(model, batch=1, input_size=spec.input_size, dtype=torch.float16, device=device, warmup=warmup, trials=trials)
        results_fp16.append(
            dict(
                family=spec.family,
                size=spec.size,
                backend="ELSA",
                dtype="fp16",
                status="ok",
                **stats,
            )
        )
        del model

        # PyTorch Flash (FA2)
        base_model = timm.create_model(spec.baseline_name, pretrained=False)
        stats = measure_model(
            base_model,
            batch=1,
            input_size=spec.input_size,
            dtype=torch.float16,
            device=device,
            warmup=warmup,
            trials=trials,
            autocast_dtype=torch.float16,
            ctx_factories=[lambda: sdp_context(flash=True)],
        )
        results_fp16.append(
            dict(
                family=spec.family,
                size=spec.size,
                backend="FA2",
                dtype="fp16",
                status="ok",
                **stats,
            )
        )
        del base_model

        # FA3 unavailable for Swin by default
        results_fp16.append(
            dict(
                family=spec.family,
                size=spec.size,
                backend="FA3",
                dtype="fp16",
                status="unavailable",
                lat_ms_avg=float("nan"),
                lat_ms_med=float("nan"),
                lat_ms_p95=float("nan"),
                peak_gb=float("nan"),
                imgs_per_s=float("nan"),
            )
        )

    # FP32 (TF32) runs
    with toggle_tf32(True):
        for spec in VIT_SPECS:
            # ELSA Turbo
            model = timm.create_model(spec.elsa_name, pretrained=False)
            stats = measure_model(
                model,
                batch=1,
                input_size=spec.input_size,
                dtype=torch.float32,
                device=device,
                warmup=warmup,
                trials=trials,
            )
            results_fp32.append(
                dict(
                    family=spec.family,
                    size=spec.size,
                    backend="ELSA",
                    dtype="fp32_tf32",
                    status="ok",
                    **stats,
                )
            )
            del model

            # SDPA math / mem
            for label, ctx_factory in (("SDPA", lambda: sdp_context(math=True)), ("ME", lambda: sdp_context(mem=True))):
                model = timm.create_model(spec.baseline_name, pretrained=False)
                stats = measure_model(
                    model,
                    batch=1,
                    input_size=spec.input_size,
                    dtype=torch.float32,
                    device=device,
                    warmup=warmup,
                    trials=trials,
                    ctx_factories=[ctx_factory],
                )
                results_fp32.append(
                    dict(
                        family=spec.family,
                        size=spec.size,
                        backend=label,
                        dtype="fp32_tf32",
                        status="ok",
                        **stats,
                    )
                )
                del model

        for spec in SWIN_SPECS:
            model = timm.create_model(spec.elsa_name, pretrained=False)
            stats = measure_model(
                model,
                batch=1,
                input_size=spec.input_size,
                dtype=torch.float32,
                device=device,
                warmup=warmup,
                trials=trials,
            )
            results_fp32.append(
                dict(
                    family=spec.family,
                    size=spec.size,
                    backend="ELSA",
                    dtype="fp32_tf32",
                    status="ok",
                    **stats,
                )
            )
            del model

            for label, ctx_factory in (("SDPA", lambda: sdp_context(math=True)), ("ME", lambda: sdp_context(mem=True))):
                base_model = timm.create_model(spec.baseline_name, pretrained=False)
                stats = measure_model(
                    base_model,
                    batch=1,
                    input_size=spec.input_size,
                    dtype=torch.float32,
                    device=device,
                    warmup=warmup,
                    trials=trials,
                    ctx_factories=[ctx_factory],
                )
                results_fp32.append(
                    dict(
                        family=spec.family,
                        size=spec.size,
                        backend=label,
                        dtype="fp32_tf32",
                        status="ok",
                        **stats,
                    )
                )
                del base_model

    import csv

    fp16_path = out_dir / "model_throughput_fp16.csv"
    fp32_path = out_dir / "model_throughput_fp32.csv"
    if results_fp16:
        keys = sorted({k for row in results_fp16 for k in row})
        with fp16_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_fp16)
    if results_fp32:
        keys = sorted({k for row in results_fp32 for k in row})
        with fp32_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_fp32)
    print(f"[OK] wrote FP16 results to {fp16_path}")
    print(f"[OK] wrote FP32 results to {fp32_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-model throughput benchmark for ELSA vs baselines.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/current/results"),
        help="Directory to place CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_benchmarks(device=device, warmup=args.warmup, trials=args.trials, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
