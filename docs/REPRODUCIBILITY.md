# Reproducibility

## Environment

Verify your environment before running:

```bash
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Benchmark Discipline

For fair ELSA vs baseline measurement:

- Use a clean GPU whenever possible.
- Run only one benchmark job at a time, even across GPUs, unless the goal is stress testing.
- Use fresh subprocesses for baseline and ELSA.
- Do not alternate many configurations inside one long-lived Python process.
- Keep raw CSVs outside the release package.

## Full-Model Coverage

```bash
CUDA_VISIBLE_DEVICES=0 python \
  scripts/run_strict_coverage_matrix.py \
  --mode full-model \
  --families vit swin \
  --dtypes fp16 fp32 \
  --directions fwd fwd_bwd \
  --fresh-backend \
  --out /tmp/elsa_fullmodel_summary.csv
```

## Attention-Only Coverage

```bash
CUDA_VISIBLE_DEVICES=0 python \
  scripts/run_strict_coverage_matrix.py \
  --mode attn-only \
  --families vit swin \
  --dtypes fp16 fp32 \
  --directions fwd fwd_bwd \
  --fresh-backend \
  --out /tmp/elsa_attn_summary.csv
```

## Long-Token Trend

Use ViT image sizes that correspond to meaningful token counts:

- `img384`: 576 tokens, still short.
- `img1024`: 4096 tokens, medium.
- `img2048`: 16384 tokens, long.
- `img3072`: 36864 tokens, very long.

The current clean long-token report in `docs/clean_logs/long_token_and_swin_w24_clean_gpu0_20260420_report.txt` shows ViT fp32 forward pass at 16K and 36K tokens. Forward-backward long baselines exceeded the tested 40GB GPU memory.
