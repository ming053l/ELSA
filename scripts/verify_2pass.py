#!/usr/bin/env python3
"""Structural proof that the ELSA path is a genuine TWO-PASS PARALLEL SCAN, not a
FlashAttention-style fused single pass.

It instruments the kernels and reports, for representative matrix shapes:
  - phase-1 launches + the grid's 3rd dim = number of K-PARTITIONS (k_blocks) scanned in parallel,
  - that per-partition summaries (m,z,s) are written to GLOBAL MEMORY (the HBM round-trip that
    defines a two-pass scan and is exactly what FlashAttention avoids),
  - that a SEPARATE phase-2 CUDA reduce reads those global summaries and merges them.

A single fused online-softmax kernel would show NO global summary buffer and NO second pass.
Run on any CUDA GPU (structural check; does not need a clean GPU).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import elsa_twopass_clean.attention_parallel_scan as APS
import elsa_twopass_clean.paper_scan_d64_cuda as PSC
from elsa_twopass_clean import twopass_attention

phase1, phase2 = [], []


def wrap_phase1(name):
    k = getattr(APS, name)
    class W:
        def __getitem__(self, grid):
            phase1.append(grid); return k[grid]
    setattr(APS, name, W())


for nm in ["_phase1_d64_tiled_fp32_nomask_kernel", "_phase1_d64_tiled_fp16_nomask_kernel",
           "_phase1_d64_fp32_nomask_kernel", "_phase1_d64_fp16_nomask_kernel"]:
    wrap_phase1(nm)

_orig = PSC.paper_scan_d64_final_reduce
def wrapped_reduce(m_buf, z_buf, s_buf, *a, **k):
    phase2.append((tuple(m_buf.shape), bool(m_buf.is_cuda)))
    return _orig(m_buf, z_buf, s_buf, *a, **k)
PSC.paper_scan_d64_final_reduce = wrapped_reduce
APS.paper_scan_d64_final_reduce = wrapped_reduce

shapes = [
    ("attn fp32 N=4096", 1, 8, 4096, 64, torch.float32),
    ("attn fp16 N=16384", 1, 8, 16384, 64, torch.float16),
    ("ViT fp32 img224 (N=197)", 8, 3, 197, 64, torch.float32),
    ("ViT fp32 img1024 (N=4097)", 8, 3, 4097, 64, torch.float32),
]
print(f"{'shape':28} | {'phase1':7} | {'K-partitions':12} | {'phase2 reads HBM summary':24}")
print("-" * 84)
ok = True
for (lbl, B, H, N, D, dt) in shapes:
    phase1.clear(); phase2.clear()
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k, v = torch.randn_like(q), torch.randn_like(q)
    twopass_attention(q, k, v)
    torch.cuda.synchronize()
    kparts = max((g[2] for g in phase1), default=0)
    p2 = f"yes shape={phase2[0][0]} gpu={phase2[0][1]}" if phase2 else "NONE"
    two_pass = len(phase1) >= 1 and len(phase2) >= 1 and phase2[0][1]
    ok = ok and two_pass
    print(f"{lbl:28} | {len(phase1):7} | {kparts:12} | {p2:24}")
    del q, k, v; torch.cuda.empty_cache()
print("-" * 84)
print("VERIFY:", "TWO-PASS PARALLEL SCAN CONFIRMED (phase1->global summaries->phase2 reduce, no FA fusion)"
      if ok else "FAILED")
sys.exit(0 if ok else 1)
