# Status Matrix (2026-03-01)

| 面向 | 子項 | 狀態 | 證據 |
|---|---|---|---|
| Attn-only / ViT / FP32 | ELSA strict vs SDPA-Math/Mem | Solved | `results/attn_only/tmp_20260228_ao_fp32_rebuild_1025_2305_4097.csv` |
| Full-model / ViT / FP32 | train+bwd+finetune 速率 | Solved | `results/latest_fixes/tmp_20260301_vit_fp32_math512_ab.csv` |
| Full-model / ViT / FP32 | 對 SDPA-Mem 的 VRAM | Solved (可接受) | 同上，`vram_ratio ~= 1.04~1.05` |
| Full-model / ViT / FP32 | 低 VRAM 模式 | Solved (trade-off) | `tmp_20260301_vit_fp32_memsaveout0_ab.csv` |
| Full-model / ViT / FP32 | triton-bwd 退化路徑 | Solved (guarded) | `tmp_20260301_vit_fp32_bwd_triton_ab.csv`, `..._guard.csv` |
| Full-model / Swin / FP32 | ELSA-Swin vs Window Attn | Solved | `results/full_model/tmp_20260228_swin_fp32fp16_regress.csv` |
| Full-model / ViT / FP16 | ELSA vs SDPA-Flash | Solved (近乎打平) | `results/full_model/tmp_20260228_fp16_vit_img1536_fair4rep.csv` |
| Full-model / Swin / FP16 | ELSA-Swin vs SDPA-Flash | Solved | `results/full_model/tmp_20260228_swin_fp32fp16_regress.csv` |
| Train/FT 一致性 | loss 曲線差異 | Solved | `results/train_ft/tmp_20260228_vit_fp32_regress_after_autobwd_fix_loss.csv` |
| 下游 FT | top-1 / val loss | Solved (可重現) | `results/downstream/tmp_20260228_downstream_vit_fp32_auto_afterfix_probe.csv` |
| LLM streaming | 可跑長序列 + PPL 對齊 | Solved | `results/llm_linear/llm_4gb_oom_*.csv`, `llama_ppl_streaming_compare_full.csv` |

## 判定準則
- `Solved`: 目前主線路徑可重現，且結果與既有目標趨勢一致。
- `Solved (可接受)`: 與理想值有小幅差距，但在理論/工程可接受範圍。
- `Solved (trade-off)`: 提供可控交換（例如速度換 VRAM）。
