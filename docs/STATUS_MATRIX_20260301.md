# Status Matrix (2026-03-01)

| Area | Item | Status | Evidence |
|---|---|---|---|
| Attn-only / ViT / FP32 | ELSA strict vs SDPA-Math/Mem | Solved | `results/attn_only/tmp_20260228_ao_fp32_rebuild_1025_2305_4097.csv` |
| Full-model / ViT / FP32 | train+backward+finetune speed | Solved | `results/latest_fixes/tmp_20260301_vit_fp32_math512_ab.csv` |
| Full-model / ViT / FP32 | VRAM vs SDPA-Mem | Solved (acceptable gap) | same file, `vram_ratio ~= 1.04~1.05` |
| Full-model / ViT / FP32 | low-VRAM mode | Solved (trade-off) | `results/latest_fixes/tmp_20260301_vit_fp32_memsaveout0_ab.csv` |
| Full-model / ViT / FP32 | triton-bwd degraded path | Solved (guarded) | `tmp_20260301_vit_fp32_bwd_triton_ab.csv`, `..._guard.csv` |
| Full-model / Swin / FP32 | ELSA-Swin vs Window Attention | Solved | `results/full_model/tmp_20260228_swin_fp32fp16_regress.csv` |
| Full-model / ViT / FP16 | ELSA vs SDPA-Flash | Solved (near parity) | `results/full_model/tmp_20260228_fp16_vit_img1536_fair4rep.csv` |
| Full-model / Swin / FP16 | ELSA-Swin vs SDPA-Flash | Solved | `results/full_model/tmp_20260228_swin_fp32fp16_regress.csv` |
| Train/FT consistency | loss-curve agreement | Solved | `results/train_ft/tmp_20260228_vit_fp32_regress_after_autobwd_fix_loss.csv` |
| Downstream FT | top-1 / val loss reproducibility | Solved | `results/downstream/tmp_20260228_downstream_vit_fp32_auto_afterfix_probe.csv` |
| LLM streaming | long-sequence runability + PPL alignment | Solved | `results/llm_linear/llm_4gb_oom_*.csv`, `llama_ppl_streaming_compare_full.csv` |

## Status legend
- `Solved`: reproducible on stable path, aligned with target trend.
- `Solved (acceptable gap)`: small residual gap, considered acceptable for this release.
- `Solved (trade-off)`: optional speed/VRAM trade-off path is available.
