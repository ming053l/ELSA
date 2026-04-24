# Release Notes (2026-03-01)

## Positioning
This release is the cleaned, reproducible candidate where attention-only and full-model paths are both available, with FP32 training-backward stability fixes integrated into the stable line.

## Key updates
1. Stable vs experimental split
- Stable: `code/stable/`
- Experimental: `code/future_exp/`

2. FP32 backward stability hardening (`code/stable/elsa_triton.py`)
- Added guard for `ELSA_TRITON_FP32_TRAIN_BWD=triton`.
- Forcing this path now requires `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1`.
- Added `ELSA_TRITON_FP32_MEM_SAVE_OUT` to expose speed/VRAM trade-off.

3. Added release-level summaries
- `docs/summary_core_metrics_20260301.csv`
- `docs/summary_accuracy_consistency_20260301.csv`
- `docs/STATUS_MATRIX_20260301.md`

## Core metrics (excerpt)
Source: `results/latest_fixes/tmp_20260301_vit_fp32_math512_ab.csv`
- ViT FP32 (train + backward + finetune, batch=1/2, img=1024) vs SDPA-Math:
  - Median speedup: `1.72x`
  - Median VRAM ratio: `0.23x`
- Same setup vs SDPA-Mem:
  - Median speedup: `1.09x`
  - Median VRAM ratio: `1.05x`

Source: `results/latest_fixes/tmp_20260301_vit_fp32_memsaveout0_ab.csv`
- With `ELSA_TRITON_FP32_MEM_SAVE_OUT=0`:
  - VRAM ratio vs SDPA-Mem improves to `~1.00x`
  - Speed drops to `~0.94x`

Source: `results/latest_fixes/tmp_20260301_vit_fp32_bwd_triton_ab.csv` and `..._guard.csv`
- Unguarded triton-bwd can collapse to ~`0.02x` vs SDPA-Math.
- Guarded/default path restores normal behavior (~`1.57x` in this probe).

## Known limitations
- `code/future_exp/` is intentionally not production-grade and not part of final claims.
- Downstream top-1 interpretation should be done under fixed seed and consistent split protocol.
