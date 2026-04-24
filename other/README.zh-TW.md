> **注意：** 此繁體中文版本為初版翻譯，內容可能未反映最新功能（如 `elsa_strict_ref.py`、`elsa_ext_pack/`、`run_strict_coverage_matrix.py` 等）。最新完整說明請參閱 [English README](../README.md)。

# ELSA：用於快速、輕量記憶體視覺 Transformer 的精確線性掃描注意力機制

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[專案頁面]</a> &nbsp;|&nbsp;
  <a href="#">[論文 (CVPR 2026)]</a> &nbsp;|&nbsp;
  <a href="#引用">[引用]</a>
</p>

<p align="center">
  <a href="../README.md">English</a> &nbsp;|&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;|&nbsp;
  <a href="README.ja.md">日本語</a> &nbsp;|&nbsp;
  <a href="README.ko.md">한국어</a>
</p>

---

<div align="center">

**<a href="https://cchsu.info/wordpress/">Chih-Chung Hsu</a>, Xin-Di Ma, Wo-Ting Liao, <a href="https://ming053l.github.io/">Chia-Ming Lee</a>**

Advanced Computer Vision Laboratory, National Yang Ming Chiao Tung University

CVPR Findings 2026

*"Can't use FA on your device? Try ELSA on!"**

</div>

---

## 概述

ELSA 將 softmax 注意力機制重新表述為一個基於關聯幺半群（associative monoid）狀態三元組 *(m, S, W)* 的**平行前綴掃描**，實現以下特性：

- **精確 softmax 語義** — 具有可證明的 FP32 相對誤差上界，無需重新訓練
- **O(log n) 平行深度** — 透過兩層掃描（塊內 Hillis–Steele + 塊間 Blelloch）
- **O(n) 額外記憶體** — 無需 O(n²) 分數矩陣，每個查詢僅需單次 I/O
- **不依賴 Tensor Core** — 以 Triton 及 CUDA C++ 實作，支援 A100、L4、Jetson TX2

---

## 功能與應用

### 本專案提供的內容

| | |
|---|---|
| **Triton / CUDA kernel** | `ELSA_triton`（FP16/BF16）、`ELSA_triton_fp32`（推論）、`ELSA_triton_fp32_train`（訓練含反向傳播） |
| **PyTorch 模組** | `ElsaAttention` — 可取代任何標準 attention block 的 `nn.Module` |
| **完整模型類別** | `ElsaViT`、`ElsaSwinTransformerV2` — 可直接訓練的架構 |
| **Patch 工具** | `patch_vit_attention`、`patch_swin_attention` — 加速預訓練 timm 模型，無需重寫程式碼 |
| **基準測試框架** | `fairbench_driver` / `fairbench_worker` — 可重現的多後端速度與 VRAM 比較 |

### 潛在應用場景

- **高解析度視覺** — 長序列 ViT 推論（醫療影像、衛星影像、高光譜分析），需要 FP32 精度的場合
- **記憶體受限的部署** — 在消費級 GPU 上執行 32K+ token 的 LLM 推論，無需擔心標準 SDPA 的 OOM 問題
- **嵌入式 / 邊緣 AI** — Jetson TX2 等裝置受益於不依賴 Tensor Core 的設計
- **機器人與自動駕駛** — 在預算有限的車載或機器人計算平台（AGX Orin、Drive AGX）上執行即時感知；O(n²) 注意力往往成為延遲或功耗瓶頸，ELSA 的 O(n) 記憶體佔用允許在相同 VRAM 空間內使用更大的上下文窗口，無需升級硬體即可獲得更豐富的場景表示
- **3D 場景理解** — 加速 VGGT 等多幀模型，精度零損失
- **任意自訂 Transformer** — ELSA kernel 接受原始 Q/K/V 張量，可整合至任何架構

### 已提供的使用示例

| 示例 | 位置 |
|---|---|
| Raw Q/K/V kernel 替換 | [使用方式 — 層級 1](#層級-1--raw-kernel自訂-attention-class) |
| 自訂 `TransformerBlock` + `ElsaAttention` | [使用方式 — 層級 2](#層級-2--elsaattention-模組替換單一層) |
| 預訓練 timm ViT / Swin patch | [使用方式 — 層級 3](#層級-3--patch-預訓練-timm-vit--swin不修改模型程式碼) |
| 從頭建立 `ElsaViT` | [使用方式 — 層級 4](#層級-4--從頭建立-elsavit) |
| HuggingFace LLaMA patch + 長序列 FP32 | [使用方式 — 層級 5](#層級-5--llama--huggingface-語言模型) |

---

## 效能亮點

### FP32 推論 vs. SDPA-Math（CLIP ViT，A100）

| 模型 | 解析度 | 加速比 | 記憶體節省 |
|---|---|---|---|
| ViT-B/16 | 224→560 px | 最高 **1.98×** | 最高 **36.1%** |
| ViT-L/14 | 224→560 px | 最高 **2.15×** | 最高 **39.6%** |

### FP32 訓練（ViT，A100，batch=1/2，image=1024）

| 比較對象 | 中位加速比 | 中位 VRAM 比 |
|---|---|---|
| ELSA vs SDPA-Math | **1.72×** | **0.23×**（−77%） |
| ELSA vs SDPA-Mem | **1.09×** | **1.05×** |

### 高光譜影像分類（HSI-MAE，FP32）

| 模型 | 資料集 | vs ME-SDPA 加速比 |
|---|---|---|
| HSI-MAE-B | Pavia / Salinas / WHU | +37–40% |
| HSI-MAE-L | Pavia / Salinas / WHU | **+60–62%** |

### 嵌入式裝置（Jetson TX2，FP16）

在所有 token 長度（64–900 tokens）下，相較 Math-SDPA 穩定達到 **~37% 延遲降低**。

### 3D 重建（VGGT，FP32 vs xFormers）

| 幀數 | 加速比 |
|---|---|
| 50 | **1.46×** |
| 100 | **2.09×** |
| 150 | **2.34×** |

---

## 方法

ELSA 將線上 softmax 轉化為幺半群 `(m, S, W) ∈ ℝ × ℝ × ℝ^dv` 上的前綴掃描：

```
m  = 運行最大 logit
S  = 歸一化累積 exp 權重和
W  = exp 加權值累加器
```

合併運算子 ⊕ 透過三個步驟組合兩個塊：**反歸一化 → 聚合 → 重新歸一化**，得到：

| 特性 | 數值 |
|---|---|
| 平行深度 | O(log n) |
| 額外記憶體 | O(n) |
| 每次查詢 I/O | 1 次（K、V 各串流一次） |
| FP32 誤差上界 | O(u · log n) |

---

## 支援的模型與硬體

**視覺模型：**
ViT（Tiny / Small / Medium / Base / Large）、Swin Transformer、CLIP、SAM、VGGT、HSI-MAE

**語言模型：**
LLaMA（8B、13B）、BERT

**硬體平台：**
NVIDIA A100、L4、Jetson TX2 · 任何支援 CUDA 的 GPU（不依賴 Tensor Core）

---

## 與其他注意力核的比較

| 方法 | 精確 | FP32 原生 | GPU 通用 | 免重新訓練 | 深度 | 額外記憶體 |
|---|---|---|---|---|---|---|
| Standard SDPA | ✓ | ✗ | ✓ | ✓ | O(n) | O(n²) |
| FlashAttention-2/3 | ✓ | ✗ | ✗ | ✓ | O(n/Tk) | O(Tk·d) |
| Linear Attention | ✗ | ✓ | ✓ | ✗ | O(log n)† | O(n) |
| **ELSA（本文）** | **✓** | **✓** | **✓** | **✓** | **O(log n)** | **O(n)** |

---

## 程式庫結構

```
code/
  stable/          # 正式版 ELSA 核心與模型整合
  future_exp/      # 實驗性路徑（不納入主要效能聲明）
scripts/           # 可重現的基準測試與驗證腳本
docs/              # 發布說明、狀態矩陣、可重現性指南、完整報告
results/           # 精選 CSV 輸出（attn_only、full_model、train_ft、downstream、llm）
validation/        # 快速驗證輸出
manifests/         # 檔案清單與 SHA256 校驗碼
```

主要原始碼檔案：

| 檔案 | 用途 |
|---|---|
| `code/stable/elsa_triton.py` | Triton 注意力核（tile 設定、FP32 反向傳播、save-out 控制） |
| `code/stable/elsa.py` | `ElsaAttention` / `ElsaViT` 模型類別 |
| `code/stable/elsa_swin.py` | Swin Transformer 整合 |
| `code/stable/elsa_swin_fused.py` | 融合 Swin 核路徑 |

---

## 安裝

```bash
git clone https://github.com/your-org/elsa.git
cd elsa
pip install -e .           # 以可編輯模式安裝 elsa 套件
```

> 依賴套件（`torch`、`triton`、`timm`）會自動安裝。
> 若需基準測試工具，加上 `[benchmark]` 附加選項：
> `pip install -e ".[benchmark]"`

---

## 使用方式

### 層級 1 — Raw kernel（自訂 attention class）

已有 Q、K、V 張量，只想替換注意力計算本身：

```python
import torch
from elsa import ELSA_triton, ELSA_triton_fp32, ELSA_pytorch

B, H, N, D = 2, 12, 1024, 64   # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = D ** -0.5

# FP16 / BF16 — 最快路徑
out = ELSA_triton.apply(q, k, v, scale)          # (B, H, N, D)

# FP32 推論 — 省記憶體，可證明精確
out = ELSA_triton_fp32.apply(q.float(), k.float(), v.float(), scale)

# FP32 訓練（支援反向傳播）
out = ELSA_triton_fp32_train.apply(q.float(), k.float(), v.float(), scale)

# 純 PyTorch fallback — 不需要 Triton，完整 autograd 支援
out = ELSA_pytorch(q, k, v, scale)
```

以上皆可直接取代：

```python
# 原本的寫法
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
```

---

### 層級 2 — `ElsaAttention` 模組（替換單一層）

`ElsaAttention` 是完整的 `nn.Module`（QKV 投影 → ELSA kernel → 輸出投影），介面與 `timm.models.vision_transformer.Attention` 相同：

```python
import torch
import torch.nn as nn
from elsa import ElsaAttention

class MyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 直接取代 timm Attention / nn.MultiheadAttention
        self.attn = ElsaAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            backend="auto",   # 建議：自動選擇最佳 kernel
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):            # x: (B, N, dim)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

model = MyTransformerBlock(dim=768, num_heads=12).cuda()
x = torch.randn(2, 196, 768, device="cuda")
out = model(x)    # (2, 196, 768)
```

可用的 `backend` 選項：

| 值 | 說明 |
|---|---|
| `"auto"` | **建議預設**。依 dtype 與硬體自動選擇最佳 kernel |
| `"triton"` | ELSA Triton kernel — FP16 / BF16 |
| `"triton_fp32"` | ELSA Triton kernel — FP32 推論 |
| `"triton_fp32_train"` | ELSA Triton kernel — FP32 訓練含反向傳播 |
| `"sdpa_math"` / `"sdpa_mem"` / `"sdpa_flash"` | PyTorch SDPA 各後端（基線比較用） |
| `"pytorch"` | 純 PyTorch fallback |

也可以在執行時全域切換 backend：

```python
from elsa import set_default_elsa_backend
set_default_elsa_backend("triton_fp32")   # 之後建立的 ElsaAttention 都使用此設定
```

---

### 層級 3 — Patch 預訓練 timm ViT / Swin（不修改模型程式碼）

只需一行呼叫，熱替換所有注意力層：

```python
import timm
from elsa import ElsaAttention
from fairbench_worker import patch_vit_attention, patch_swin_attention

# ── ViT ──────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda().eval()

# 就地替換所有 attention layer 為 ELSA kernel
patch_vit_attention(model, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = model(images)   # 以 ELSA 注意力執行

# ── Swin Transformer ─────────────────────────────────────────────────────────
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).cuda().eval()
patch_swin_attention(swin, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = swin(images)
```

---

### 層級 4 — 從頭建立 `ElsaViT`

```python
import torch
from elsa import ElsaViT

model = ElsaViT(
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_classes=1000,
    elsa_backend="auto",
).cuda()

images = torch.randn(4, 3, 224, 224, device="cuda")
logits = model(images)   # (4, 1000)
```

---

### 層級 5 — LLaMA / HuggingFace 語言模型

ELSA kernel 與架構無關，可用於任何暴露 Q、K、V 張量的 Transformer，包括 decoder-only LLM。以下示例就地 patch 一個 HuggingFace LLaMA 模型：

```python
import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from elsa import ELSA_triton, ELSA_triton_fp32

def patch_llama_attention(model):
    """將每個 LLaMA attention 層的 scaled_dot_product_attention 替換為 ELSA。"""

    def _make_forward(original_forward):
        def elsa_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, **kwargs):
            _real_sdpa = torch.nn.functional.scaled_dot_product_attention

            def _elsa_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, **kw):
                head_dim = q.shape[-1]
                s = scale if scale is not None else head_dim ** -0.5
                if q.dtype == torch.float32:
                    return ELSA_triton_fp32.apply(q, k, v, s)
                return ELSA_triton.apply(q, k, v, s)

            torch.nn.functional.scaled_dot_product_attention = _elsa_sdpa
            try:
                return original_forward(
                    self, hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache, **kwargs
                )
            finally:
                torch.nn.functional.scaled_dot_product_attention = _real_sdpa

        return elsa_forward

    for layer in model.model.layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(_make_forward(attn.__class__.forward), attn)

    return model


# ── 使用方式 ────────────────────────────────────────────────────────────────
model_id = "meta-llama/Llama-2-7b-hf"   # 任意 HuggingFace LLaMA 變體均可
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

patch_llama_attention(model)   # 熱替換注意力，無需重新訓練

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**FP32 長序列推論（記憶體受限的 GPU）**

在長序列（>16K tokens）下，FP32 SDPA 需要 O(n²) 的分數矩陣，在消費級 GPU 上會 OOM。ELSA 的 O(n) 記憶體用量讓這些序列得以正常執行：

```python
# 精度敏感任務（醫療、科學）使用 FP32 載入
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cuda"
)
patch_llama_attention(model)

# 使用標準 SDPA 會 OOM 的 32K token context
long_input = tokenizer(
    very_long_document, return_tensors="pt", max_length=32768, truncation=True
).to("cuda")

with torch.no_grad():
    out = model(**long_input)
```

> **Perplexity 等價：** ELSA 保留精確的 softmax 語義——
> 在 WikiText-2 上量測的 perplexity 與原始模型一致，精確到浮點精度。

---

### 如何選擇層級

| 情境 | 建議 |
|---|---|
| 自訂 `nn.Module` 並手動計算 Q、K、V | **層級 1** — `ELSA_triton` / `ELSA_triton_fp32` |
| 替換模型中的單一 attention block | **層級 2** — `ElsaAttention` |
| 預訓練 timm ViT / Swin，不想改模型 | **層級 3** — `patch_vit_attention` |
| 從頭訓練新的 ViT | **層級 4** — `ElsaViT` |
| HuggingFace LLaMA / decoder LLM | **層級 5** — 手動替換 `F.scaled_dot_product_attention` |

---

## 環境需求

- Python ≥ 3.9
- PyTorch ≥ 2.1（CUDA 版本）
- Triton ≥ 2.2
- timm ≥ 0.9

```bash
pip install -r requirements.txt
```


---

## 快速開始

```bash
# 僅注意力基準測試（ViT FP32）
python scripts/benchmark_pure_attention_vit.py \
  --device cuda:0 --dtype fp32 --seq-lens 1025,2305,4097 \
  --warmup 6 --trials 30 \
  --output results/attn_only/recheck_vit_fp32.csv

# 完整模型吞吐量
python scripts/benchmark_model_throughput.py \
  --device cuda:0 --warmup 8 --trials 32 \
  --output results/full_model/recheck_full_model.csv

# 訓練 / 微調 / 反向傳播矩陣
python scripts/benchmark_train_ft_matrix.py \
  --device cuda:0 --families vit swin \
  --modes backward train finetune \
  --precisions fp32 tf32 fp16 \
  --specs vit_small_512 swin_tiny_w8_256 \
  --steps 20 --warmup 8 --seed 123 \
  --output results/train_ft/recheck_train_ft.csv
```

使用 `scripts/run_benchmark.sh` 可自動選取空閒 GPU 並啟動 conda 環境：

```bash
export ELSA_CONDA_ENV=elsa
bash scripts/run_benchmark.sh python scripts/benchmark_model_throughput.py --device cuda:0
```

---

## 環境變數

| 變數 | 預設值 | 說明 |
|---|---|---|
| `ELSA_TRITON_FP32_TRAIN_BWD` | `auto` | FP32 訓練反向路徑（`triton` 需搭配 `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1`） |
| `ELSA_TRITON_FP32_MEM_SAVE_OUT` | `1` | `1` = 速度優先；`0` = 降低 VRAM |
| `ELSA_TRITON_ALLOW_UNSTABLE_PATHS` | `0` | 設為 `1` 以啟用實驗性路徑 |
| `ELSA_FORCE_ALLOW_TF32` | `0` | 覆蓋 TF32 策略 |

---

## 技術文件

| 文件 | 說明 |
|---|---|
| [`docs/FULL_REPORT_20260301.md`](docs/FULL_REPORT_20260301.md) | 所有設定下的完整基準測試結果 |
| [`docs/RELEASE_NOTES_20260301.md`](docs/RELEASE_NOTES_20260301.md) | 本版本主要更新與穩定性修正 |
| [`docs/STATUS_MATRIX_20260301.md`](docs/STATUS_MATRIX_20260301.md) | 各設定支援狀態 |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 含公平性控制的重現指南 |

---

## 引用

若您在研究中使用了 ELSA，請引用：

```bibtex
@inproceedings{hsu2026elsa,
  title={ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers},
  author={Hsu, Chih-Chung and Ma, Xin-Di and Liao, Wo-Ting and Lee, Chia-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 授權

本專案僅供**學術研究與非商業用途**使用。
詳見 [LICENSE](LICENSE)。
