# Benchmark Summary

This summary uses only curated clean report logs in `docs/clean_logs/`. Raw CSV files are intentionally excluded from the release package.

## Acceptance Rule

Pass means both latency and memory are at least comparable to the baseline, and at least one is better.

- Latency comparable: `<= 1.05x` baseline.
- Latency better: `<= 0.98x` baseline.
- Memory comparable: `<= 1.05x` baseline.
- Memory better: `<= 0.98x` baseline.

## Clean Evidence Snapshot

| Area | Clean report | Result | Interpretation |
| --- | --- | ---: | --- |
| Attention-only | `attn_only_16x16_gpu0_20260413_report.txt` | 16/16 pass once | Achieved full matrix pass once; reruns are matrix-flaky, so this is evidence of capability, not yet a stability guarantee. |
| ViT fp32 full-model | `vit_fp32_fullmodel_4x4_gpu4_20260421_report.txt` | 4/4 pass | Current strongest full-model result. |
| Full-model 16-cell matrix | `fullmodel_16cell_clean_gpu0_20260419_report.txt` | 5/16 pass | Clean snapshot before later ViT fp32 improvements; Swin bwd and ViT fp16 remain the main full-model gaps. |
| ViT fp32 pair | `vit_fp32_fullmodel_pair_gpu0_20260419_report.txt` | 1/2 pass | Pre-latest-policy pair rerun; kept for traceability. |
| Long-token diagnostic | `long_token_and_swin_w24_clean_gpu0_20260420_report.txt` | ViT fp32 fwd pass at 16K/36K tokens | Long-token trend is favorable for ViT fp32 forward. Forward-backward long baseline OOMed on the tested 40GB GPU. |

## ViT fp32 Full-Model Clean 4/4

From `vit_fp32_fullmodel_4x4_gpu4_20260421_report.txt`:

| Direction | Variant | Tokens | Latency ratio | Memory ratio | Status |
| --- | --- | ---: | ---: | ---: | --- |
| fwd | `vit-img224` | 196 | 0.7773 | 0.9879 | pass |
| fwd | `vit-img384` | 576 | 0.7833 | 0.7720 | pass |
| fwd_bwd | `vit-img224` | 196 | 0.8183 | 0.7413 | pass |
| fwd_bwd | `vit-img384` | 576 | 0.9854 | 0.7109 | pass |

## Attention-Only Clean 16/16 Best Run

From `attn_only_16x16_gpu0_20260413_report.txt`:

- Swin fp16 W8/W16 fwd and fwd_bwd all passed.
- Swin fp32 W8/W16 fwd and fwd_bwd all passed.
- ViT fp16 N196/N4096 fwd and fwd_bwd all passed.
- ViT fp32 N196/N4096 fwd and fwd_bwd all passed.

The important long-token cells in that run:

| Cell | Latency ratio | Memory ratio | Status |
| --- | ---: | ---: | --- |
| ViT fp16 attn-only fwd N4096 | 0.9317 | 0.9922 | pass |
| ViT fp16 attn-only fwd_bwd N4096 | 0.9891 | 0.8291 | pass |
| ViT fp32 attn-only fwd N4096 | 0.7803 | 0.0445 | pass |
| ViT fp32 attn-only fwd_bwd N4096 | 0.2620 | 0.0548 | pass |

## Full-Model Gaps

The release candidate is not claiming full universal kill-all stability yet.

- ViT fp32 full-model short-token cells are currently strong and clean.
- ViT fp32 long-token forward shows the expected stronger win trend at 16K and 36K tokens.
- Swin short-window full-model backward remains latency-sensitive because mask/bias/front-end overhead dominates at W8/W16 token counts.
- ViT fp16 full-model still needs additional front-end or launch-policy work before it is a stable full-matrix win.

## Fairness Notes

- Use one clean GPU and one benchmark process at a time.
- Run baseline and ELSA through fresh subprocesses.
- Do not mix multiple benchmark matrices in the same Python process.
- Treat short-token results separately from 4K/16K+ token trend checks.
