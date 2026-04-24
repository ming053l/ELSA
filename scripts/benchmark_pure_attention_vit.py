#!/usr/bin/env python
"""Benchmark ViT-style pure attention across ELSA / FlashAttention / SDPA variants."""
from __future__ import annotations

import argparse
import csv
import os
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from experiments.a100_run.run_all import (
    FLASH_ATTN_AVAILABLE,
    ElsaPolicies,
    flash_attn_v2,
    flash_attn_v3,
    latency_stats,
    load_kernels,
    make_inputs,
)


VariantFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@contextmanager
def tf32_guard(enabled: bool) -> Iterable[None]:
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    prev_env = os.environ.get("NVIDIA_TF32_OVERRIDE")
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1" if enabled else "0"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
        if prev_env is None:
            os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        else:
            os.environ["NVIDIA_TF32_OVERRIDE"] = prev_env


def benchmark(
    fn: VariantFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    trials: int,
) -> Dict[str, float]:
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(q, k, v)
            del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(trials):
            start.record()
            out = fn(q, k, v)
            end.record()
            del out
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    stats = latency_stats(latencies)
    stats["peak_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    return stats


def run_flash_attn_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not FLASH_ATTN_AVAILABLE or flash_attn_v2 is None:
        raise RuntimeError("flash-attn v2 unavailable")
    q_ = q.permute(0, 2, 1, 3).contiguous().half()
    k_ = k.permute(0, 2, 1, 3).contiguous().half()
    v_ = v.permute(0, 2, 1, 3).contiguous().half()
    out = flash_attn_v2(q_, k_, v_, dropout_p=0.0, softmax_scale=None, causal=False)
    return out.permute(0, 2, 1, 3).contiguous()


def run_flash_attn_v3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not FLASH_ATTN_AVAILABLE or flash_attn_v3 is None:
        raise RuntimeError("flash-attn v3 unavailable")
    B, H, N, D = q.shape
    q_ = q.permute(0, 2, 1, 3).contiguous().half()
    k_ = k.permute(0, 2, 1, 3).contiguous().half()
    v_ = v.permute(0, 2, 1, 3).contiguous().half()
    qkv = torch.stack([q_, k_, v_], dim=2)  # (B, N, 3, H, D)
    qkv = qkv.view(B * N, 3, H, D)
    cu = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=q.device)
    out = flash_attn_v3(qkv, cu, N, causal=False)
    out = out.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
    return out


def run_elsa_strict(policies: ElsaPolicies, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with tf32_guard(False):
        return policies.strict_fp32(q, k, v)


def run_elsa_turbo(policies: ElsaPolicies, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with tf32_guard(True):
        return policies.turbo_tf32(q, k, v)


def run_elsa_mem(
    policies: ElsaPolicies,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eta: float,
    allow_tf32: bool,
) -> torch.Tensor:
    with tf32_guard(allow_tf32):
        return policies.elsa_mem(q, k, v, eta, dtype=torch.float32)


def run_sdpa_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str, allow_tf32: bool) -> torch.Tensor:
    with tf32_guard(allow_tf32), torch.backends.cuda.sdp_kernel(
        enable_math=(backend == "math"),
        enable_mem_efficient=(backend == "mem"),
        enable_flash=False,
    ):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), dropout_p=0.0, is_causal=False)


def run_benchmark(
    device: str,
    kernel_file: str,
    seq_lens: Iterable[int],
    batch: int,
    heads: int,
    head_dim: int,
    warmup: int,
    trials: int,
    output: str,
    eta_values: Iterable[float],
    dtype_names: Iterable[str],
) -> None:
    torch.cuda.set_device(device)
    kernels = load_kernels(kernel_file)
    policies = ElsaPolicies(kernels)

    rows: List[Dict[str, object]] = []

    base_inputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for seq in seq_lens:
        base_inputs[seq] = make_inputs(batch, heads, seq, head_dim, dtype=torch.float16)

    dtype_map = {"fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}
    requested = [dtype_map[name.lower()] for name in dtype_names]

    for dtype in requested:
        if dtype == torch.float16:
            variants: List[Tuple[str, VariantFn, Dict[str, object]]] = [
                ("ELSA-FP16", policies.fp16, {}),
                ("SDPA-FP16-math", lambda q, k, v: ElsaPolicies.sdpa_fp16(q, k, v, "math"), {"sdpa_backend": "math"}),
                ("SDPA-FP16-mem", lambda q, k, v: ElsaPolicies.sdpa_fp16(q, k, v, "mem"), {"sdpa_backend": "mem"}),
            ]
            for eta in eta_values:
                variants.append(
                    (
                        f"ELSA-mem-FP16(eta={eta:.2f})",
                        lambda q, k, v, eta=eta: policies.elsa_mem(q, k, v, eta, dtype=torch.float16),
                        {"eta": eta},
                    )
                )
            if FLASH_ATTN_AVAILABLE and flash_attn_v2 is not None:
                variants.append(("FA2-FP16", run_flash_attn_v2, {}))
            if FLASH_ATTN_AVAILABLE and flash_attn_v3 is not None:
                variants.append(("FA3-FP16", run_flash_attn_v3, {}))
        else:  # FP32
            variants = [
                ("ELSA-FP32-strict", lambda q, k, v: run_elsa_strict(policies, q, k, v), {"tf32": "off"}),
                ("ELSA-Turbo", lambda q, k, v: run_elsa_turbo(policies, q, k, v), {"tf32": "on"}),
            ]
            for eta in eta_values:
                variants.append(
                    (
                        f"ELSA-mem-FP32-strict(eta={eta:.2f})",
                        lambda q, k, v, eta=eta: run_elsa_mem(policies, q, k, v, eta, allow_tf32=False),
                        {"eta": eta, "tf32": "off"},
                    )
                )
            for eta in eta_values:
                variants.append(
                    (
                        f"ELSA-mem-FP32-turbo(eta={eta:.2f})",
                        lambda q, k, v, eta=eta: run_elsa_mem(policies, q, k, v, eta, allow_tf32=True),
                        {"eta": eta, "tf32": "on"},
                    )
                )
            variants.extend(
                [
                    (
                        "SDPA-FP32-math(tf32_off)",
                        lambda q, k, v: run_sdpa_fp32(q, k, v, "math", allow_tf32=False),
                        {"sdpa_backend": "math", "tf32": "off"},
                    ),
                    (
                        "SDPA-FP32-math(tf32_on)",
                        lambda q, k, v: run_sdpa_fp32(q, k, v, "math", allow_tf32=True),
                        {"sdpa_backend": "math", "tf32": "on"},
                    ),
                    (
                        "SDPA-FP32-mem(tf32_off)",
                        lambda q, k, v: run_sdpa_fp32(q, k, v, "mem", allow_tf32=False),
                        {"sdpa_backend": "mem", "tf32": "off"},
                    ),
                    (
                        "SDPA-FP32-mem(tf32_on)",
                        lambda q, k, v: run_sdpa_fp32(q, k, v, "mem", allow_tf32=True),
                        {"sdpa_backend": "mem", "tf32": "on"},
                    ),
                ]
            )
            if FLASH_ATTN_AVAILABLE and flash_attn_v2 is not None:
                variants.append(
                    (
                        "FA2-TF32",
                        run_flash_attn_v2,
                        {"tf32": "on", "effective_dtype": "fp16"},
                    )
                )
            if FLASH_ATTN_AVAILABLE and flash_attn_v3 is not None:
                variants.append(
                    (
                        "FA3-TF32",
                        run_flash_attn_v3,
                        {"tf32": "on", "effective_dtype": "fp16"},
                    )
                )

        for seq in seq_lens:
            q_base, k_base, v_base = base_inputs[seq]

            for label, fn, extra in variants:
                q = q_base.to(dtype=dtype).clone()
                k = k_base.to(dtype=dtype).clone()
                v = v_base.to(dtype=dtype).clone()

                status = "ok"
                try:
                    stats = benchmark(fn, q, k, v, warmup, trials)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                        stats = {"lat_ms_avg": float("nan"), "lat_ms_med": float("nan"), "lat_ms_p95": float("nan"), "peak_gb": float("nan")}
                        status = "oom"
                        print(f"[{label}] dtype={dtype} seq={seq:,} | OOM ({exc})")
                    else:
                        raise

                lat_med = stats.get("lat_ms_med", float("nan"))
                tokens_per_s = float("nan")
                if lat_med and not torch.isnan(torch.tensor(lat_med)):
                    tokens_per_s = (batch * seq) / (lat_med / 1e3)

                row: Dict[str, object] = {
                    "variant": label,
                    "batch": batch,
                    "heads": heads,
                    "head_dim": head_dim,
                    "seq_len": seq,
                    "warmup": warmup,
                    "trials": trials,
                    "dtype": str(dtype),
                    "lat_ms_avg": stats.get("lat_ms_avg"),
                    "lat_ms_med": lat_med,
                    "lat_ms_p95": stats.get("lat_ms_p95"),
                    "peak_gb": stats.get("peak_gb"),
                    "tokens_per_s": tokens_per_s,
                    "status": status,
                }
                row.update(extra)
                row.setdefault("tf32", "n/a")
                row.setdefault("effective_dtype", str(dtype))
                rows.append(row)
                print(
                    f"[{label}] dtype={dtype} seq={seq:,} | median {lat_med:.3f} ms | "
                    f"peak {stats.get('peak_gb', float('nan')):.3f} GB | tokens/s {tokens_per_s:,.0f}"
                )

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        out_path = os.path.abspath(output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] wrote results to {out_path}")
    else:
        print("[WARN] no rows were collected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pure attention (ViT-style) across multiple kernels.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device identifier (default: cuda:0).")
    parser.add_argument("--kernel-file", type=str, default="elsa_triton_entry.py", help="Path to kernel entry module.")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384],
        help="Sequence lengths to benchmark.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention heads (default: 12).")
    parser.add_argument("--head-dim", type=int, default=64, help="Per-head dimension (default: 64).")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warm-up iterations (default: 20).")
    parser.add_argument("--trials", type=int, default=120, help="Number of timed iterations (default: 120).")
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        default=[1.00, 0.50, 0.25],
        help="eta values for ELSA-mem variants (default: 1.00 0.50 0.25).",
    )
    parser.add_argument(
        "--disable-elsa-mem",
        action="store_true",
        help="Skip running ELSA-mem variants.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["fp16"],
        help="Which data types to benchmark (choices: fp16, fp32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/current/results/pure_attention_vit_fp16.csv",
        help="Destination CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        device=args.device,
        kernel_file=args.kernel_file,
        seq_lens=args.seq_lens,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        trials=args.trials,
        output=args.output,
        eta_values=[] if args.disable_elsa_mem else args.eta,
        dtype_names=args.dtypes,
    )


if __name__ == "__main__":
    main()
