#!/usr/bin/env python3
"""16-point release spot check: 2 correctness + 14 performance-matrix cells.

Runs the packaged benchmarks (reduced sizes/iters) and judges each cell against a
LOOSE trend bound (generous vs the full-matrix numbers, to tolerate run noise while
still catching any real regression). Exit 0 only if all 16 checks PASS.
Run via scripts/bench_clean.sh on a clean GPU.
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = sys.executable
results: list[tuple[str, bool, str]] = []


def sh(args: list[str]) -> str:
    r = subprocess.run([PY] + args, capture_output=True, text=True, timeout=1500)
    return r.stdout + r.stderr


def rows(out: str, prefix: str) -> list[list[str]]:
    return [l.split(",") for l in out.splitlines() if l.startswith(prefix)]


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name:24} {detail}", flush=True)


# ---- C1: forward correctness (odd + pow2 seqs, both dtypes) ----
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from elsa_twopass_clean import twopass_attention  # noqa: E402

errs = []
for (B, H, N, D, dt) in [(8, 3, 1025, 64, torch.float32), (8, 3, 2305, 64, torch.float16),
                          (1, 8, 4096, 64, torch.float32)]:
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k, v = torch.randn_like(q), torch.randn_like(q)
    o = twopass_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
    e = (o.float() - ref.float()).abs().max().item()
    errs.append((dt, e))
    del q, k, v, o, ref
    torch.cuda.empty_cache()
ok = all(e < (5e-3 if dt == torch.float16 else 1e-5) for dt, e in errs)
check("C1 fwd correctness", ok, " ".join(f"{str(dt)[6:]}:{e:.1e}" for dt, e in errs))

# ---- C2: backward gradient correctness ----
out = sh(["benchmarks/bwd_grad_correctness.py"])
ok = "WRONG" not in out and out.count(" OK") >= 5
check("C2 bwd grad correctness", ok, f"{out.count(' OK')} OK rows")

# ---- attn-only perf cells ----
HD = ["--heads", "8", "--dim", "64"]

out = sh(["benchmarks/bench_attn.py", "--dtype", "fp32", "--baseline", "math", *HD,
          "--seq", "1024", "4096", "--warmup", "3", "--iters", "10"])
r = rows(out, "fp32")
lats = [float(x[8]) for x in r if x[3] == "math"]
check("P1 attn fp32 fwd math", bool(lats) and all(l < 1.0 for l in lats), f"lat={lats}")

out = sh(["benchmarks/bench_attn.py", "--dtype", "fp32", "--baseline", "mem", *HD,
          "--seq", "4096", "32768", "--warmup", "3", "--iters", "10"])
r = rows(out, "fp32")
lats = [float(x[8]) for x in r]
check("P2 attn fp32 fwd mem", len(lats) == 2 and lats[1] < lats[0] and lats[1] < 1.35,
      f"lat={lats} (narrows, long<1.35)")

out = sh(["benchmarks/bench_attn.py", "--dtype", "fp16", "--baseline", "flash", *HD,
          "--seq", "4096", "16384", "--warmup", "3", "--iters", "10"])
r = rows(out, "fp16")
lats = [float(x[8]) for x in r]
check("P3 attn fp16 fwd flash", len(lats) == 2 and lats[1] < lats[0] and lats[1] < 2.0,
      f"lat={lats} (approaches)")

out = sh(["benchmarks/bench_bwd.py", "--dtype", "fp32", "--baseline", "math", *HD,
          "--seq", "2048", "8192", "--warmup", "3", "--iters", "8"])
r = rows(out, "fp32")
lats = [float(x[8]) for x in r]; mems = [float(x[9]) for x in r]
check("P4 attn fp32 bwd math", len(lats) == 2 and lats[1] < 1.2 and max(mems) < 0.15,
      f"lat={lats} mem={mems}")

out = sh(["benchmarks/bench_bwd.py", "--dtype", "fp32", "--baseline", "mem", *HD,
          "--seq", "8192", "16384", "--warmup", "3", "--iters", "8"])
r = rows(out, "fp32")
lats = [float(x[8]) for x in r]; mems = [float(x[9]) for x in r]
check("P5 attn fp32 bwd mem", len(lats) == 2 and max(lats) < 1.8 and max(mems) < 0.9,
      f"lat={lats} mem={mems}")

out = sh(["benchmarks/bench_bwd.py", "--dtype", "fp16", "--baseline", "flash", *HD,
          "--seq", "8192", "16384", "--warmup", "3", "--iters", "8"])
r = rows(out, "fp16")
lats = [float(x[8]) for x in r]; mems = [float(x[9]) for x in r]
check("P6 attn fp16 bwd flash", len(lats) == 2 and max(lats) < 3.2 and max(mems) < 0.85,
      f"lat={lats} mem={mems}")

# ---- full-model fwd (CUDA graph) ----
VIT = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
SWIN = "swin_tiny_patch4_window7_224.ms_in1k"


def graph_cell(name, model, dtype, img, lat_bound, mem_bound=1.40, retries=3):
    # full-model cells are short single-shot measurements on a shared node: a burst of
    # contention mid-run skews one side's timing. Retry in a FRESH process (fresh
    # measurement window) before declaring a real regression.
    last = "no data"
    for att in range(retries):
        out = sh(["benchmarks/full_model_graph.py", "--model", model, "--dtype", dtype,
                  "--batch", "8", "--image-size", str(img), "--iters", "20"])
        r = rows(out, "graph,") + rows(out, "eagermin,")
        if r and "FAILED" not in out:
            lat, mem = float(r[0][9]), float(r[0][10])
            last = f"lat={lat:.3f}(<{lat_bound}) mem={mem:.3f} try{att+1}"
            if lat < lat_bound and mem < mem_bound:
                check(name, True, last)
                return
    check(name, False, last)


graph_cell("P7 vit fp16 fwd@224", VIT, "fp16", 224, 1.5, 1.10)
graph_cell("P8 vit fp32 fwd@512", VIT, "fp32", 512, 1.7, 1.10)
graph_cell("P9 swin fp16 fwd@384", SWIN, "fp16", 384, 1.10)
graph_cell("P10 swin fp32 fwd@384", SWIN, "fp32", 384, 1.10)


# ---- full-model training ----
def train_cell(name, model, dtype, img, lat_bound, retries=3):
    last = "no data"
    for att in range(retries):
        out = sh(["benchmarks/full_model_bwd_min.py", "--model", model, "--dtype", dtype,
                  "--batch", "8", "--image-size", str(img), "--iters", "20"])
        r = rows(out, "bwdmin,")
        if r and "FAILED" not in out:
            lat = float(r[0][9])
            last = f"lat={lat:.3f}(<{lat_bound}) try{att+1}"
            if lat < lat_bound:
                check(name, True, last)
                return
    check(name, False, last)


train_cell("P11 vit fp16 train@224", VIT, "fp16", 224, 1.55)
train_cell("P12 vit fp32 train@512", VIT, "fp32", 512, 1.30)
train_cell("P13 swin fp16 train@384", SWIN, "fp16", 384, 1.15)
train_cell("P14 swin fp32 train@384", SWIN, "fp32", 384, 1.25)

npass = sum(ok for _, ok, _ in results)
print(f"\nSPOT CHECK: {npass}/16 PASS")
print("SPOTCHECK_" + ("ALL_PASS" if npass == 16 else "FAILED"))
sys.exit(0 if npass == 16 else 1)
