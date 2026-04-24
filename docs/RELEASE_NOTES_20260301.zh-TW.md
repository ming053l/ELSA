# Release Notes (2026-03-01)

## 版本定位
本版是「attn-only 與 full-model 共同可用」的整理版，重點在修復 FP32 訓練 backward 路徑穩定性，並將可用路徑與未收斂路徑分離。

## 主要變更
1. 穩定主線與實驗路徑分離
- 主線：`code/stable/`
- 未定版：`code/future_exp/`

2. FP32 backward 穩定性修補（`code/stable/elsa_triton.py`）
- 新增 `ELSA_TRITON_FP32_TRAIN_BWD=triton` 防呆 guard。
- 預設需 `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1` 才可強制 triton-bwd。
- 新增 `ELSA_TRITON_FP32_MEM_SAVE_OUT`，支援速度/記憶體取捨。

3. 新增本版核心彙整
- `docs/摘要_核心指標_20260301.csv`
- `docs/摘要_精度一致性_20260301.csv`
- `docs/STATUS_MATRIX_20260301.md`

## 核心數據（節錄）
來源：`tmp_20260301_vit_fp32_math512_ab.csv`
- ViT FP32（train+bwd+ft, batch=1/2, img=1024）對 SDPA-Math：ELSA 中位速率 `1.72x`，VRAM 約 `0.23x`。
- 同條件對 SDPA-Mem：ELSA 中位速率 `1.09x`，VRAM 約 `1.05x`。

來源：`tmp_20260301_vit_fp32_memsaveout0_ab.csv`
- 啟用 `ELSA_TRITON_FP32_MEM_SAVE_OUT=0` 後，對 SDPA-Mem 的 VRAM 約 `1.00x`，但速度降到約 `0.94x`。

來源：`tmp_20260301_vit_fp32_bwd_triton_ab.csv` 與 `tmp_20260301_vit_fp32_bwd_triton_guard.csv`
- 未加 guard 的 triton-bwd 可退化到約 `0.02x`（相對 SDPA-Math）。
- guard 啟用後恢復正常（約 `1.57x`）。

## 已知限制
- `code/future_exp/` 不保證效能或穩定性，不應納入主結論。
- 下游任務 top-1 請搭配固定 seed 與一致資料切分解讀（見 `docs/REPRODUCIBILITY.md`）。
