# Full Report (2026-03-01, ELSA Release Candidate)

## Scope
This report summarizes reproducible evidence in this release bundle across attention-only, full-model forward/backward/train/finetune, downstream checks, and LLM streaming probes.

## Sources
- Core metrics: `docs/summary_core_metrics_20260301.csv`
- Accuracy/consistency: `docs/summary_accuracy_consistency_20260301.csv`
- Status matrix: `docs/STATUS_MATRIX_20260301.md`

## 1) Attention-only
- ViT FP32 long-sequence sweep is included in `results/attn_only/tmp_20260228_ao_fp32_rebuild_1025_2305_4097.csv`.
- ViT FP16 ultra-long sequence runability is included in `results/attn_only/tmp_20260228_ao_fp16_ultralong_rebuild*.csv`.
- Conclusion: long-sequence scaling trend matches historical ELSA expectations on the stable path.

## 2) Full-model highlights

### 2.1 ViT FP32 (train + backward + finetune)
From `results/latest_fixes/tmp_20260301_vit_fp32_math512_ab.csv`:
- vs SDPA-Math: median speedup `1.7247x`, median VRAM ratio `0.2261x`
- vs SDPA-Mem: median speedup `1.0888x`, median VRAM ratio `1.0458x`

### 2.2 ViT FP32 low-VRAM toggle
From `results/latest_fixes/tmp_20260301_vit_fp32_memsaveout0_ab.csv`:
- vs SDPA-Mem: VRAM ratio improves to `~0.9998x`
- speed becomes `~0.9356x`
- This is a controlled trade-off option, not the default profile.

### 2.3 Swin FP32
From `results/full_model/tmp_20260228_swin_fp32fp16_regress.csv`:
- ELSA-Swin-FP32 vs native Swin Window Attention:
  - median speedup `1.0353x`
  - median VRAM ratio `0.9494x`

### 2.4 ViT FP16
From `results/full_model/tmp_20260228_fp16_vit_img1536_fair4rep.csv`:
- ELSA-FP16 vs SDPA-Flash: median speedup `1.0003x`, near parity.

## 3) Backward-path hardening
- Problematic path: forcing `ELSA_TRITON_FP32_TRAIN_BWD=triton` can severely degrade on this stack (`~0.02x` in probe).
- Fix: guard this path by default; require `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1` to force.
- After guard: backward returns to expected trend (`~1.57x` vs SDPA-Math in probe).

## 4) Numerical consistency
- ViT FP32 train loss alignment: absolute loss diff median `0`, p95 `~1.18e-7`.
- LLM PPL alignment (ELSA vs ELSA-stream): absolute diff `~7.48e-06`.
- Downstream top-1 should be interpreted under fixed seed and fixed split protocol.

## 5) Release conclusion
1. This bundle is publish-ready as a clean release candidate. 2. ViT FP32 remains the strongest headline path: clear win vs SDPA-Math and mostly win/near-win vs SDPA-Mem with close VRAM. 3. Swin FP32 is stable with slight advantage; ViT FP16 is near parity to flash baseline in included probes. 4. Unsafe/degraded paths are isolated and guarded to avoid contaminating release claims.
