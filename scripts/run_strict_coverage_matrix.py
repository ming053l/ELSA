#!/usr/bin/env python3
"""Coverage benchmark for strict ELSA two-level scan.

This script fills the matrix:
family x dtype x mode x direction, with short/long sequence variants.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "timm"))

STRICT_ENV = {
    "ELSA_ALLOW_UNSTABLE_PATHS": "0",
    "ELSA_STRICT_BWD_DQPART_AUTO_ENABLE": "0",
    "ELSA_STRICT_BWD_FUSE_DQ_DKV_ATOMIC": "0",
    "ELSA_STRICT_BWD_SPLIT_DKV_DIRECT_FIRST": "0",
    "ELSA_STRICT_BWD_SPLIT_DKV_TAIL_REDUCE": "0",
    "ELSA_STRICT_VIT_TRAIN_SHORT_COMPILE": "0",
    "ELSA_STRICT_VIT_SHORT_COMPILE": "0",
    "ELSA_SWIN_STRICT_EVAL_COMPILE": "0",
    "ELSA_SWIN_STRICT_EVAL_DIRECT": "auto",
    "ELSA_FULLMODEL_PROJ_FUSE": "auto",
    "ELSA_FULLMODEL_QKV_PREPACK": "auto",
    "ELSA_STRICT_SMALL_PROVIDER": "1",
    "ELSA_STRICT_BENCH_NO_FALLBACK": "1",
    "ELSA_STRICT_BENCH_SAMPLES": "3",
    "ELSA_STRICT_BENCH_LATENCY_MODE": "min",
    "ELSA_STRICT_BENCH_EMPTY_AFTER_WARMUP": "0",
}
for key, value in STRICT_ENV.items():
    os.environ.setdefault(key, value)

import torch
import torch.nn.functional as F
import timm
from timm.models import elsa as elsa_core


DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}

LAT_COMPARE = 1.05
LAT_BETTER = 0.98
MEM_COMPARE = 1.05
MEM_BETTER = 0.98


@dataclass(frozen=True)
class Case:
    family: str
    dtype_name: str
    mode: str
    direction: str
    variant: str
    seq_len: int
    batch: int
    heads: int = 0
    head_dim: int = 0
    image_size: int = 0
    baseline_name: str = ""
    elsa_name: str = ""
    use_bias: bool = False


def case_to_dict(case: Case) -> dict:
    return {
        "family": case.family,
        "dtype_name": case.dtype_name,
        "mode": case.mode,
        "direction": case.direction,
        "variant": case.variant,
        "seq_len": case.seq_len,
        "batch": case.batch,
        "heads": case.heads,
        "head_dim": case.head_dim,
        "image_size": case.image_size,
        "baseline_name": case.baseline_name,
        "elsa_name": case.elsa_name,
        "use_bias": case.use_bias,
    }


def case_from_dict(data: dict) -> Case:
    return Case(**data)


def case_selector_match(case: Case, selector: str) -> bool:
    parts = selector.split(":")
    if len(parts) == 1:
        family = dtype_name = mode = direction = None
        variant = parts[0]
    elif len(parts) == 3:
        family = mode = None
        dtype_name, direction, variant = parts
    elif len(parts) == 5:
        family, dtype_name, mode, direction, variant = parts
    else:
        raise ValueError(
            f"invalid case selector '{selector}'; use variant or dtype:direction:variant or family:dtype:mode:direction:variant"
        )
    return (
        (family is None or case.family == family)
        and (dtype_name is None or case.dtype_name == dtype_name)
        and (mode is None or case.mode == mode)
        and (direction is None or case.direction == direction)
        and case.variant == variant
    )


def filter_cases(selected: list[Case], selectors: list[str]) -> list[Case]:
    if not selectors:
        return selected
    out: list[Case] = []
    missing: list[str] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for selector in selectors:
        matches = [case for case in selected if case_selector_match(case, selector)]
        if not matches:
            missing.append(selector)
            continue
        for case in matches:
            key = (case.family, case.dtype_name, case.mode, case.direction, case.variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(case)
    if missing:
        raise SystemExit("Unknown case selector(s): " + ", ".join(missing))
    return out


@contextmanager
def sdp_context(dtype_name: str):
    if dtype_name == "fp16":
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            yield
    else:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield


@contextmanager
def benchmark_run_lock():
    if os.environ.get("ELSA_STRICT_BENCH_DISABLE_RUN_LOCK", "0").strip().lower() in {"1", "true", "on", "yes"}:
        yield
        return
    lock_path = OUT_DIR / ".strict_bench.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def benchmark(fn: Callable[[], None], warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if os.environ.get("ELSA_STRICT_BENCH_EMPTY_AFTER_WARMUP", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        torch.cuda.empty_cache()

    samples = max(1, _env_int("ELSA_STRICT_BENCH_SAMPLES", 1))
    latency_mode = os.environ.get("ELSA_STRICT_BENCH_LATENCY_MODE", "min").strip().lower()
    latencies: list[float] = []
    peaks: list[float] = []
    for _ in range(samples):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) / iters)
        peaks.append(torch.cuda.max_memory_allocated() / (1024**2))

    if latency_mode == "median":
        latency_ms = sorted(latencies)[len(latencies) // 2]
    elif latency_mode == "mean":
        latency_ms = sum(latencies) / len(latencies)
    else:
        latency_ms = min(latencies)
    return latency_ms, max(peaks)


def metric_status(elsa: float, base: float, compare: float, better: float, lower_is_better: bool = True) -> str:
    if base <= 0:
        return "unknown"
    ratio = elsa / base if lower_is_better else base / elsa
    if ratio <= better:
        return "better"
    if ratio <= compare:
        return "comparable"
    return "worse"


def cell_status(lat_status: str, mem_status: str) -> str:
    good_latency = lat_status in {"better", "comparable"}
    good_memory = mem_status in {"better", "comparable"}
    one_better = lat_status == "better" or mem_status == "better"
    return "pass" if good_latency and good_memory and one_better else "fail"


def new_row(case: Case, backend: str, status: str, latency_ms: float | None, peak_mb: float | None, error: str = "") -> dict:
    return {
        "family": case.family,
        "dtype": case.dtype_name,
        "mode": case.mode,
        "direction": case.direction,
        "variant": case.variant,
        "seq_len": case.seq_len,
        "batch": case.batch,
        "heads": case.heads,
        "head_dim": case.head_dim,
        "image_size": case.image_size,
        "backend": backend,
        "status": status,
        "latency_ms": "" if latency_ms is None else f"{latency_ms:.6f}",
        "peak_mb": "" if peak_mb is None else f"{peak_mb:.6f}",
        "error": error,
    }


def clear_exact_provider_cache() -> None:
    mod = sys.modules.get(elsa_core.can_triton_strict_core_fp16.__module__)
    if mod is None:
        return
    clear = getattr(mod, "_clear_exact_sdpa_graph_cache", None)
    if callable(clear):
        clear()


def run_attn_case(case: Case, warmup: int, iters: int) -> list[dict]:
    dtype = DTYPES[case.dtype_name]
    device = "cuda"
    q = torch.randn(case.batch, case.heads, case.seq_len, case.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    if case.direction == "fwd_bwd":
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    bias = None
    if case.use_bias:
        bias = torch.randn(1, case.heads, case.seq_len, case.seq_len, device=device, dtype=dtype)
        if case.direction == "fwd_bwd":
            bias.requires_grad_(False)

    def zero_grads():
        for tensor in (q, k, v):
            tensor.grad = None

    def run_base():
        if case.direction == "fwd_bwd":
            zero_grads()
        with sdp_context(case.dtype_name):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)
        if case.direction == "fwd_bwd":
            out.float().sum().backward()

    def run_elsa():
        if case.direction == "fwd_bwd":
            zero_grads()
        if dtype == torch.float16:
            out = elsa_core.can_triton_strict_core_fp16(q, k, v, is_causal=False, bias=bias)
        else:
            out = elsa_core.can_triton_strict_core_fp32(q, k, v, is_causal=False, bias=bias)
        if case.direction == "fwd_bwd":
            out.float().sum().backward()

    rows: list[dict] = []
    for name, fn in ((case.baseline_name, run_base), ("ELSA-strict-two-level-scan", run_elsa)):
        try:
            clear_exact_provider_cache()
            ms, mb = benchmark(fn, warmup, iters)
            rows.append(new_row(case, name, "ok", ms, mb))
        except Exception as err:
            rows.append(new_row(case, name, "error", None, None, f"{type(err).__name__}: {err}"))
            traceback.print_exc()
        finally:
            clear_exact_provider_cache()
    return rows


def run_attn_case_backend(case: Case, backend: str, warmup: int, iters: int) -> dict:
    dtype = DTYPES[case.dtype_name]
    device = "cuda"
    q = torch.randn(case.batch, case.heads, case.seq_len, case.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    if case.direction == "fwd_bwd":
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    bias = None
    if case.use_bias:
        bias = torch.randn(1, case.heads, case.seq_len, case.seq_len, device=device, dtype=dtype)
        if case.direction == "fwd_bwd":
            bias.requires_grad_(False)

    def zero_grads():
        for tensor in (q, k, v):
            tensor.grad = None

    def run_base():
        if case.direction == "fwd_bwd":
            zero_grads()
        with sdp_context(case.dtype_name):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)
        if case.direction == "fwd_bwd":
            out.float().sum().backward()

    def run_elsa():
        if case.direction == "fwd_bwd":
            zero_grads()
        if dtype == torch.float16:
            out = elsa_core.can_triton_strict_core_fp16(q, k, v, is_causal=False, bias=bias)
        else:
            out = elsa_core.can_triton_strict_core_fp32(q, k, v, is_causal=False, bias=bias)
        if case.direction == "fwd_bwd":
            out.float().sum().backward()

    label, fn = (case.baseline_name, run_base) if backend == "baseline" else ("ELSA-strict-two-level-scan", run_elsa)
    try:
        clear_exact_provider_cache()
        ms, mb = benchmark(fn, warmup, iters)
        return new_row(case, label, "ok", ms, mb)
    except Exception as err:
        traceback.print_exc()
        return new_row(case, label, "error", None, None, f"{type(err).__name__}: {err}")
    finally:
        clear_exact_provider_cache()


def make_model(case: Case, backend: str):
    dtype = DTYPES[case.dtype_name]
    if backend == "elsa":
        kwargs = {"pretrained": False, "elsa_backend": "strict_core_ref", "triton": False}
        if case.family == "vit":
            kwargs.update({"img_size": case.image_size, "dynamic_img_size": True})
        model = timm.create_model(case.elsa_name, **kwargs)
    else:
        kwargs = {"pretrained": False}
        if case.family == "vit":
            kwargs.update({"img_size": case.image_size, "dynamic_img_size": True})
        model = timm.create_model(case.baseline_name, **kwargs)
    return model.to("cuda", dtype=dtype)


def run_model_case(case: Case, warmup: int, iters: int) -> list[dict]:
    dtype = DTYPES[case.dtype_name]
    x = torch.randn(case.batch, 3, case.image_size, case.image_size, device="cuda", dtype=dtype)
    rows: list[dict] = []

    for backend_key, backend_label in (("baseline", case.baseline_name), ("elsa", "ELSA-strict-two-level-scan")):
        model = None
        try:
            clear_exact_provider_cache()
            model = make_model(case, backend_key)
            if case.direction == "fwd":
                model.eval()

                def fn():
                    with torch.no_grad(), sdp_context(case.dtype_name):
                        model(x)
            else:
                model.train()

                def fn():
                    model.zero_grad(set_to_none=True)
                    with sdp_context(case.dtype_name):
                        out = model(x)
                    out.float().sum().backward()

            ms, mb = benchmark(fn, warmup, iters)
            rows.append(new_row(case, backend_label, "ok", ms, mb))
        except Exception as err:
            rows.append(new_row(case, backend_label, "error", None, None, f"{type(err).__name__}: {err}"))
            traceback.print_exc()
        finally:
            clear_exact_provider_cache()
            del model
            torch.cuda.empty_cache()
    return rows


def run_model_case_backend(case: Case, backend: str, warmup: int, iters: int) -> dict:
    dtype = DTYPES[case.dtype_name]
    x = torch.randn(case.batch, 3, case.image_size, case.image_size, device="cuda", dtype=dtype)
    backend_label = case.baseline_name if backend == "baseline" else "ELSA-strict-two-level-scan"
    model = None
    try:
        clear_exact_provider_cache()
        model = make_model(case, backend)
        if case.direction == "fwd":
            model.eval()

            def fn():
                with torch.no_grad(), sdp_context(case.dtype_name):
                    model(x)
        else:
            model.train()

            def fn():
                model.zero_grad(set_to_none=True)
                with sdp_context(case.dtype_name):
                    out = model(x)
                out.float().sum().backward()

        ms, mb = benchmark(fn, warmup, iters)
        return new_row(case, backend_label, "ok", ms, mb)
    except Exception as err:
        traceback.print_exc()
        return new_row(case, backend_label, "error", None, None, f"{type(err).__name__}: {err}")
    finally:
        clear_exact_provider_cache()
        del model
        torch.cuda.empty_cache()


def run_case_once(case: Case, attn_warmup: int, attn_iters: int, model_warmup: int, model_iters: int) -> list[dict]:
    if case.mode == "attn-only":
        return run_attn_case(case, attn_warmup, attn_iters)
    return run_model_case(case, model_warmup, model_iters)


def run_case_once_backend(
    case: Case,
    backend: str,
    attn_warmup: int,
    attn_iters: int,
    model_warmup: int,
    model_iters: int,
) -> dict:
    if case.mode == "attn-only":
        return run_attn_case_backend(case, backend, attn_warmup, attn_iters)
    return run_model_case_backend(case, backend, model_warmup, model_iters)


def run_case_backend_in_subprocess(
    case: Case,
    backend: str,
    attn_warmup: int,
    attn_iters: int,
    model_warmup: int,
    model_iters: int,
) -> dict:
    payload = {
        "case": case_to_dict(case),
        "backend": backend,
        "attn_warmup": attn_warmup,
        "attn_iters": attn_iters,
        "model_warmup": model_warmup,
        "model_iters": model_iters,
    }
    tmp = tempfile.NamedTemporaryFile(prefix="strict_backend_", suffix=".json", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--subprocess-case-json",
        json.dumps(payload),
        "--subprocess-out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)
    try:
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
            raise RuntimeError(
                f"fresh backend subprocess failed for {case.family} {case.dtype_name} {case.mode} "
                f"{case.direction} {case.variant} [{backend}]:\n" + "\n".join(tail)
            )
        return json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)


def run_case_in_subprocess(case: Case, attn_warmup: int, attn_iters: int, model_warmup: int, model_iters: int) -> list[dict]:
    payload = {
        "case": case_to_dict(case),
        "attn_warmup": attn_warmup,
        "attn_iters": attn_iters,
        "model_warmup": model_warmup,
        "model_iters": model_iters,
    }
    tmp = tempfile.NamedTemporaryFile(prefix="strict_case_", suffix=".json", delete=False)
    tmp.close()
    out_path = Path(tmp.name)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--subprocess-case-json",
        json.dumps(payload),
        "--subprocess-out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, env=os.environ.copy(), capture_output=True, text=True)
    try:
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
            raise RuntimeError(
                f"fresh subprocess failed for {case.family} {case.dtype_name} {case.mode} {case.direction} {case.variant}:\n"
                + "\n".join(tail)
            )
        return json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)


def cases(
    *,
    model_batch_fp16: int = 8,
    model_batch_fp32: int = 2,
    extra_vit_imgs: list[int] | None = None,
    extra_vit_batch_fp16: int = 1,
    extra_vit_batch_fp32: int = 1,
) -> list[Case]:
    out: list[Case] = []
    for dtype_name in ("fp16", "fp32"):
        baseline = "FA2" if dtype_name == "fp16" else "SDPA-Math"
        for direction in ("fwd", "fwd_bwd"):
            for seq in (196, 4096):
                out.append(Case("vit", dtype_name, "attn-only", direction, f"vit-tiny-N{seq}", seq, 1, 3, 64, 0, baseline, "", False))
            for name, seq in (("swin-W8", 64), ("swin-W16", 256)):
                out.append(Case("swin", dtype_name, "attn-only", direction, name, seq, 1, 3, 32, 0, baseline, "", True))

    for dtype_name, batch in (("fp16", model_batch_fp16), ("fp32", model_batch_fp32)):
        for direction in ("fwd", "fwd_bwd"):
            for img in (224, 384):
                seq = (img // 16) ** 2
                out.append(Case("vit", dtype_name, "full-model", direction, f"vit-img{img}", seq, batch, 0, 0, img, "deit_tiny_patch16_224", "elsa_tiny_patch16_224", False))
            out.append(Case("swin", dtype_name, "full-model", direction, "swin-W8-img256", 64, batch, 0, 0, 256, "swinv2_tiny_window8_256", "elsa_tiny_window8_256", True))
            out.append(Case("swin", dtype_name, "full-model", direction, "swin-W16-img256", 256, batch, 0, 0, 256, "swinv2_tiny_window16_256", "elsa_tiny_window16_256", True))
    extra_vit_imgs = extra_vit_imgs or []
    existing_vit_imgs = {224, 384}
    for img in extra_vit_imgs:
        if img in existing_vit_imgs:
            continue
        if img <= 0 or img % 16 != 0:
            raise SystemExit(f"--extra-vit-img must be a positive multiple of 16, got {img}")
        seq = (img // 16) ** 2
        for dtype_name, batch in (("fp16", extra_vit_batch_fp16), ("fp32", extra_vit_batch_fp32)):
            for direction in ("fwd", "fwd_bwd"):
                out.append(Case("vit", dtype_name, "full-model", direction, f"vit-img{img}", seq, batch, 0, 0, img, "deit_tiny_patch16_224", "elsa_tiny_patch16_224", False))
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "family", "dtype", "mode", "direction", "variant", "seq_len", "batch",
        "heads", "head_dim", "image_size", "backend", "status", "latency_ms", "peak_mb", "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["family"], row["dtype"], row["mode"], row["direction"], row["variant"], row["seq_len"], row["batch"])
        grouped.setdefault(key, {})[row["backend"]] = row
    summary = []
    for key, per_backend in sorted(grouped.items()):
        family, dtype_name, mode, direction, variant, seq_len, batch = key
        elsa = per_backend.get("ELSA-strict-two-level-scan")
        base = next((v for k, v in per_backend.items() if k != "ELSA-strict-two-level-scan"), None)
        if not elsa or not base or elsa["status"] != "ok" or base["status"] != "ok":
            status = "not-measured" if not elsa or not base else "error"
            lat_status = mem_status = "unknown"
            lat_ratio = mem_ratio = ""
        else:
            e_ms = float(elsa["latency_ms"])
            b_ms = float(base["latency_ms"])
            e_mb = float(elsa["peak_mb"])
            b_mb = float(base["peak_mb"])
            lat_ratio = e_ms / b_ms
            mem_ratio = e_mb / b_mb
            lat_status = metric_status(e_ms, b_ms, LAT_COMPARE, LAT_BETTER)
            mem_status = metric_status(e_mb, b_mb, MEM_COMPARE, MEM_BETTER)
            status = cell_status(lat_status, mem_status)
            lat_ratio = f"{lat_ratio:.4f}"
            mem_ratio = f"{mem_ratio:.4f}"
        summary.append({
            "family": family,
            "dtype": dtype_name,
            "mode": mode,
            "direction": direction,
            "variant": variant,
            "seq_len": seq_len,
            "batch": batch,
            "baseline": "" if base is None else base["backend"],
            "elsa_latency_ms": "" if not elsa else elsa["latency_ms"],
            "baseline_latency_ms": "" if not base else base["latency_ms"],
            "lat_ratio": lat_ratio,
            "lat_status": lat_status,
            "elsa_peak_mb": "" if not elsa else elsa["peak_mb"],
            "baseline_peak_mb": "" if not base else base["peak_mb"],
            "mem_ratio": mem_ratio,
            "mem_status": mem_status,
            "cell_status": status,
        })
    return summary


def write_summary(path: Path, summary: list[dict]) -> None:
    fields = [
        "family", "dtype", "mode", "direction", "variant", "seq_len", "batch",
        "baseline", "elsa_latency_ms", "baseline_latency_ms", "lat_ratio", "lat_status",
        "elsa_peak_mb", "baseline_peak_mb", "mem_ratio", "mem_status", "cell_status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def write_report(path: Path, summary: list[dict], rows: list[dict]) -> None:
    total = len(summary)
    passed = sum(1 for r in summary if r["cell_status"] == "pass")
    failed = [r for r in summary if r["cell_status"] != "pass"]
    lines = [
        "Strict ELSA two-level scan coverage report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Definition used:",
        f"- latency comparable <= {LAT_COMPARE:.2f}x baseline; latency better <= {LAT_BETTER:.2f}x baseline.",
        f"- memory comparable <= {MEM_COMPARE:.2f}x baseline; memory better <= {MEM_BETTER:.2f}x baseline.",
        "- pass means latency and memory are both at least comparable, and at least one is better.",
        "- fp16 attention baseline is Flash-SDPA/FA2 forced through PyTorch SDPA.",
        "- fp32 attention baseline is Math-SDPA forced through PyTorch SDPA.",
        "- full-model ViT baseline is deit_tiny_patch16_224; full-model Swin baseline is native swinv2_tiny.",
        "- ELSA backend is strict_core_ref, with native two-level scan kernels plus narrow exact-SDPA fast paths on selected batch-1 shapes.",
        "",
        f"Summary: {passed}/{total} measured variants pass.",
        "",
        "Failed or not-measured variants:",
    ]
    if not failed:
        lines.append("- none")
    else:
        for r in failed:
            lines.append(
                "- {family} {dtype} {mode} {direction} {variant}: {cell_status}, "
                "lat_ratio={lat_ratio}, mem_ratio={mem_ratio}, lat={lat_status}, mem={mem_status}".format(**r)
            )
    lines.extend(["", "Coverage matrix:"])
    header = [
        "family", "dtype", "mode", "direction", "variant", "seq_len", "batch",
        "baseline", "lat_ratio", "lat_status", "mem_ratio", "mem_status", "cell_status",
    ]
    lines.append(",".join(header))
    for r in summary:
        lines.append(",".join(str(r[h]) for h in header))
    lines.extend(["", "Raw benchmark errors:"])
    errors = [r for r in rows if r["status"] != "ok"]
    if not errors:
        lines.append("- none")
    else:
        for r in errors:
            lines.append(f"- {r['family']} {r['dtype']} {r['mode']} {r['direction']} {r['variant']} {r['backend']}: {r['error']}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-warmup", type=int, default=5)
    parser.add_argument("--attn-iters", type=int, default=20)
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--model-iters", type=int, default=8)
    parser.add_argument("--only", choices=["all", "attn", "model"], default="all")
    parser.add_argument("--prefix", default="strict_coverage")
    parser.add_argument("--case", action="append", default=[], help="Exact case selector: variant, dtype:direction:variant, or family:dtype:mode:direction:variant")
    parser.add_argument("--extra-vit-img", action="append", type=int, default=[], help="Add extra full-model ViT image size cases; must be a multiple of 16")
    parser.add_argument("--model-batch-fp16", type=int, default=8, help="Batch size for built-in full-model fp16 cases")
    parser.add_argument("--model-batch-fp32", type=int, default=2, help="Batch size for built-in full-model fp32 cases")
    parser.add_argument("--extra-vit-batch-fp16", type=int, default=1, help="Batch size for --extra-vit-img fp16 cases")
    parser.add_argument("--extra-vit-batch-fp32", type=int, default=1, help="Batch size for --extra-vit-img fp32 cases")
    parser.add_argument("--fresh-single", action="store_true", help="Run each selected case in a fresh subprocess for clean single-case measurement")
    parser.add_argument("--fresh-backend", action="store_true", help="Run baseline and ELSA for each case in separate fresh subprocesses")
    parser.add_argument("--subprocess-case-json", help=argparse.SUPPRESS)
    parser.add_argument("--subprocess-out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.subprocess_case_json:
        if not args.subprocess_out:
            raise SystemExit("--subprocess-out is required with --subprocess-case-json")
        payload = json.loads(args.subprocess_case_json)
        case = case_from_dict(payload["case"])
        backend = payload.get("backend")
        if backend:
            row = run_case_once_backend(
                case,
                backend,
                int(payload["attn_warmup"]),
                int(payload["attn_iters"]),
                int(payload["model_warmup"]),
                int(payload["model_iters"]),
            )
            Path(args.subprocess_out).write_text(json.dumps(row))
        else:
            rows = run_case_once(
                case,
                int(payload["attn_warmup"]),
                int(payload["attn_iters"]),
                int(payload["model_warmup"]),
                int(payload["model_iters"]),
            )
            Path(args.subprocess_out).write_text(json.dumps(rows))
        return

    with benchmark_run_lock():
        selected = [
            case for case in cases(
                model_batch_fp16=args.model_batch_fp16,
                model_batch_fp32=args.model_batch_fp32,
                extra_vit_imgs=args.extra_vit_img,
                extra_vit_batch_fp16=args.extra_vit_batch_fp16,
                extra_vit_batch_fp32=args.extra_vit_batch_fp32,
            )
            if args.only == "all" or (args.only == "attn" and case.mode == "attn-only") or (args.only == "model" and case.mode == "full-model")
        ]
        selected = filter_cases(selected, args.case)
        rows: list[dict] = []
        initial = OUT_DIR / f"{args.prefix}_initial_matrix.txt"
        initial.write_text(
            "Initial strict ELSA coverage matrix\n"
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "All requested cells are scheduled in this run. Entries remain not-measured until each benchmark finishes.\n"
            + "\n".join(f"- {c.family} {c.dtype_name} {c.mode} {c.direction} {c.variant}" for c in selected)
            + "\n"
        )
        print(f"[init] wrote {initial}")

        for idx, case in enumerate(selected, 1):
            print(f"[{idx}/{len(selected)}] {case.family} {case.dtype_name} {case.mode} {case.direction} {case.variant}", flush=True)
            if args.fresh_backend:
                # Alternate backend order across cases so ELSA is not always measured
                # second in long serial matrix runs, which can bias later cells.
                backend_order = ("baseline", "elsa") if (idx % 2 == 1) else ("elsa", "baseline")
                case_rows = [
                    run_case_backend_in_subprocess(case, backend, args.attn_warmup, args.attn_iters, args.model_warmup, args.model_iters)
                    for backend in backend_order
                ]
            elif args.fresh_single:
                case_rows = run_case_in_subprocess(case, args.attn_warmup, args.attn_iters, args.model_warmup, args.model_iters)
            else:
                case_rows = run_case_once(case, args.attn_warmup, args.attn_iters, args.model_warmup, args.model_iters)
            rows.extend(case_rows)
            write_rows(OUT_DIR / f"{args.prefix}_raw.csv", rows)
            summary = summarize(rows)
            write_summary(OUT_DIR / f"{args.prefix}_summary.csv", summary)
            write_report(OUT_DIR / f"{args.prefix}_report.txt", summary, rows)
        print(f"[done] wrote {OUT_DIR / f'{args.prefix}_report.txt'}")


if __name__ == "__main__":
    main()
