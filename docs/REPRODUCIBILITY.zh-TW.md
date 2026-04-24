# Reproducibility Guide

## 1) 環境
- 建議環境：`conda activate <your-env>`（需已安裝 PyTorch ≥ 2.1 CUDA 版本及 Triton ≥ 2.2）
- CUDA 驅動：A100 測試環境（與主實驗一致）
- 執行入口：`scripts/run_benchmark.sh`（會自動挑選空閒 GPU 並轉發參數給任何 benchmark script）

## 2) 一致性設定
建議在同一輪比較固定以下條件：
- 相同 `seed`
- 相同 `warmup/trials`
- 相同 `batch/img_size/spec`
- 相同 TF32 設定（嚴格 FP32 時關閉 TF32）

可選 deterministic：
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=123
```
註：deterministic 可能稍微影響速度，不建議與非 deterministic 結果混比。

## 3) 最小重現命令

### Attn-only（ViT）
```bash
bash scripts/run_benchmark.sh python scripts/benchmark_pure_attention_vit.py \
  --device cuda:0 --dtype fp32 --seq-lens 1025,2305,4097 --warmup 6 --trials 30 \
  --output results/attn_only/recheck_vit_fp32.csv
```

### Full-model（ViT+Swin）
```bash
bash scripts/run_benchmark.sh python scripts/benchmark_model_throughput.py \
  --device cuda:0 --warmup 8 --trials 32 \
  --output results/full_model/recheck_full_model.csv
```

### Train/Finetune/Backward
```bash
bash scripts/run_benchmark.sh python scripts/benchmark_train_ft_matrix.py \
  --device cuda:0 --families vit swin --modes backward train finetune \
  --precisions fp32 tf32 fp16 --specs vit_small_512 swin_tiny_w8_256 \
  --steps 20 --warmup 8 --seed 123 \
  --output results/train_ft/recheck_train_ft.csv \
  --loss-output results/train_ft/recheck_train_ft_loss.csv
```

## 4) FP32 訓練路徑開關
- `ELSA_TRITON_FP32_TRAIN_BWD=auto`（預設）
- `ELSA_TRITON_FP32_TRAIN_BWD=triton`（不建議，除非另開 `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1`）
- `ELSA_TRITON_FP32_MEM_SAVE_OUT=1`（速度優先）
- `ELSA_TRITON_FP32_MEM_SAVE_OUT=0`（更省 VRAM，可能降速）