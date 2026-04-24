> **注記：** この日本語版は初版翻訳であり、最新機能（`elsa_strict_ref.py`、`elsa_ext_pack/`、`run_strict_coverage_matrix.py` 等）が反映されていない場合があります。最新の情報は [英語版 README](../README.md) をご参照ください。

# ELSA: 高速・軽量メモリな Vision Transformer のための精確線形スキャンアテンション

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[プロジェクトページ]</a> &nbsp;|&nbsp;
  <a href="#">[論文 (CVPR 2026)]</a> &nbsp;|&nbsp;
  <a href="#引用">[引用]</a>
</p>

<p align="center">
  <a href="../README.md">English</a> &nbsp;|&nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp;|&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;|&nbsp;
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

## 概要

ELSA は softmax アテンションを、状態トリプル *(m, S, W)* の結合モノイド上の**並列プレフィックススキャン**として再定式化することで、以下を実現します：

- **精確な softmax セマンティクス** — 証明可能な FP32 相対誤差上界、再学習不要
- **O(log n) 並列深度** — 2 段スキャン（ブロック内 Hillis–Steele + ブロック間 Blelloch）
- **O(n) 追加メモリ** — O(n²) スコア行列不要、クエリあたり 1 パス I/O
- **Tensor Core 非依存** — Triton および CUDA C++ で実装、A100・L4・Jetson TX2 で動作

---

## 機能と応用

### 本リポジトリが提供するもの

| | |
|---|---|
| **Triton / CUDA カーネル** | `ELSA_triton`（FP16/BF16）、`ELSA_triton_fp32`（推論）、`ELSA_triton_fp32_train`（訓練＋逆伝播） |
| **PyTorch モジュール** | `ElsaAttention` — 任意の標準アテンションブロックを置き換える `nn.Module` |
| **完全モデルクラス** | `ElsaViT`、`ElsaSwinTransformerV2` — すぐに訓練可能なアーキテクチャ |
| **パッチユーティリティ** | `patch_vit_attention`、`patch_swin_attention` — コード変更なしで timm モデルを高速化 |
| **ベンチマークハーネス** | `fairbench_driver` / `fairbench_worker` — 再現可能なマルチバックエンド速度・VRAM 比較 |

### 想定される応用シーン

- **高解像度ビジョン** — FP32 精度が必須な長シーケンス ViT 推論（医療画像・衛星画像・ハイパースペクトル解析）
- **メモリ制約のある展開** — 標準 SDPA では OOM になる 32K+ トークンの LLM 推論をコンシューマ GPU で実行
- **組み込み / エッジ AI** — Tensor Core 非依存設計により Jetson TX2 等のデバイスでも動作
- **ロボティクス・自動運転** — 予算制約のある車載・ロボット向け計算プラットフォーム（AGX Orin、Drive AGX）上でのリアルタイム認識；O(n²) アテンションはレイテンシや消費電力のボトルネックになりやすく、ELSA の O(n) メモリフットプリントにより同一 VRAM 容量内でより大きなコンテキストウィンドウが利用可能になり、ハードウェアを増強せずに豊かな場面表現を実現
- **3D シーン理解** — VGGT 等のマルチフレームモデルを精度ゼロ損失で高速化
- **任意のカスタム Transformer** — ELSA カーネルは生の Q/K/V テンソルを受け取り、あらゆるアーキテクチャに組み込み可能

### 提供されている使用例

| 使用例 | 場所 |
|---|---|
| Raw Q/K/V カーネル置き換え | [使用方法 — レベル 1](#レベル-1--raw-カーネルカスタム-attention-クラス) |
| `ElsaAttention` を使ったカスタム `TransformerBlock` | [使用方法 — レベル 2](#レベル-2--elsaattention-モジュール単一レイヤーの置き換え) |
| 事前学習済み timm ViT / Swin のパッチ | [使用方法 — レベル 3](#レベル-3--事前学習済み-timm-vit--swin-のパッチモデルコード変更なし) |
| `ElsaViT` をゼロから構築 | [使用方法 — レベル 4](#レベル-4--新しい-elsavit-をゼロから構築) |
| HuggingFace LLaMA パッチ + 長コンテキスト FP32 | [使用方法 — レベル 5](#レベル-5--llama--huggingface-言語モデル) |

---

## 性能ハイライト

### FP32 推論 vs. SDPA-Math（CLIP ViT、A100）

| モデル | 解像度 | 速度向上 | メモリ削減 |
|---|---|---|---|
| ViT-B/16 | 224→560 px | 最大 **1.98×** | 最大 **36.1%** |
| ViT-L/14 | 224→560 px | 最大 **2.15×** | 最大 **39.6%** |

### FP32 訓練（ViT、A100、batch=1/2、image=1024）

| 比較対象 | 中央値速度向上 | 中央値 VRAM 比 |
|---|---|---|
| ELSA vs SDPA-Math | **1.72×** | **0.23×**（−77%） |
| ELSA vs SDPA-Mem | **1.09×** | **1.05×** |

### ハイパースペクトル画像分類（HSI-MAE、FP32）

| モデル | データセット | ME-SDPA との速度比 |
|---|---|---|
| HSI-MAE-B | Pavia / Salinas / WHU | +37–40% |
| HSI-MAE-L | Pavia / Salinas / WHU | **+60–62%** |

### 組み込みデバイス（Jetson TX2、FP16）

全トークン長（64–900 tokens）にわたり、Math-SDPA 比で安定して **~37% のレイテンシ削減**。

### 3D 再構成（VGGT、FP32 vs xFormers）

| フレーム数 | 速度向上 |
|---|---|
| 50 | **1.46×** |
| 100 | **2.09×** |
| 150 | **2.34×** |

---

## 手法

ELSA はオンライン softmax をモノイド `(m, S, W) ∈ ℝ × ℝ × ℝ^dv` 上のプレフィックススキャンに変換します：

```
m  = 走行中の最大 logit
S  = 正規化累積 exp 重みの和
W  = exp 重み付き値の累積器
```

マージ演算子 ⊕ は 3 ステップでブロックを合成します：**逆正規化 → 集約 → 再正規化**。これにより：

| 特性 | 値 |
|---|---|
| 並列深度 | O(log n) |
| 追加メモリ | O(n) |
| クエリあたり I/O | 1 パス（K、V を各 1 回ストリーム） |
| FP32 誤差上界 | O(u · log n) |

---

## 対応モデルとハードウェア

**ビジョン：**
ViT（Tiny / Small / Medium / Base / Large）、Swin Transformer、CLIP、SAM、VGGT、HSI-MAE

**言語：**
LLaMA（8B、13B）、BERT

**ハードウェア：**
NVIDIA A100、L4、Jetson TX2 · 任意の CUDA 対応 GPU（Tensor Core 非依存）

---

## 他のアテンションカーネルとの比較

| 手法 | 精確 | FP32 ネイティブ | GPU 非依存 | 再学習不要 | 深度 | 追加メモリ |
|---|---|---|---|---|---|---|
| Standard SDPA | ✓ | ✗ | ✓ | ✓ | O(n) | O(n²) |
| FlashAttention-2/3 | ✓ | ✗ | ✗ | ✓ | O(n/Tk) | O(Tk·d) |
| Linear Attention | ✗ | ✓ | ✓ | ✗ | O(log n)† | O(n) |
| **ELSA（提案手法）** | **✓** | **✓** | **✓** | **✓** | **O(log n)** | **O(n)** |

---

## リポジトリ構成

```
code/
  stable/          # 本番環境対応 ELSA カーネルとモデル統合
  future_exp/      # 実験的パス（最終性能主張には使用しない）
scripts/           # 再現可能なベンチマーク・検証スクリプト
docs/              # リリースノート、ステータスマトリクス、再現性ガイド、全報告書
results/           # キュレーション済み CSV 出力
validation/        # クイック検証出力
manifests/         # ファイルマニフェストと SHA256 チェックサム
```

主要ソースファイル：

| ファイル | 用途 |
|---|---|
| `code/stable/elsa_triton.py` | Triton アテンションカーネル |
| `code/stable/elsa.py` | `ElsaAttention` / `ElsaViT` モデルクラス |
| `code/stable/elsa_swin.py` | Swin Transformer 統合 |
| `code/stable/elsa_swin_fused.py` | 融合 Swin カーネルパス |

---

## インストール

```bash
git clone https://github.com/your-org/elsa.git
cd elsa
pip install -e .           # elsa パッケージを編集可能モードでインストール
```

> 依存関係（`torch`、`triton`、`timm`）は自動的にインストールされます。
> ベンチマークユーティリティが必要な場合は `[benchmark]` オプションを追加：
> `pip install -e ".[benchmark]"`

---

## 使用方法

### レベル 1 — Raw カーネル（カスタム attention クラス）

Q、K、V テンソルが手元にあり、アテンション計算のみを置き換えたい場合：

```python
import torch
from elsa import ELSA_triton, ELSA_triton_fp32, ELSA_pytorch

B, H, N, D = 2, 12, 1024, 64   # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = D ** -0.5

# FP16 / BF16 — 最速パス
out = ELSA_triton.apply(q, k, v, scale)

# FP32 推論 — メモリ効率的、証明可能な精確さ
out = ELSA_triton_fp32.apply(q.float(), k.float(), v.float(), scale)

# FP32 訓練（逆伝播サポート）
out = ELSA_triton_fp32_train.apply(q.float(), k.float(), v.float(), scale)

# 純 PyTorch フォールバック — Triton 不要、完全な autograd サポート
out = ELSA_pytorch(q, k, v, scale)
```

以下の標準 PyTorch 実装の直接代替品として使用できます：

```python
# 置き換え前
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
```

---

### レベル 2 — `ElsaAttention` モジュール（単一レイヤーの置き換え）

`ElsaAttention` は完全な `nn.Module`（QKV 投影 → ELSA カーネル → 出力投影）で、`timm.models.vision_transformer.Attention` と同じインターフェースを持ちます：

```python
import torch
import torch.nn as nn
from elsa import ElsaAttention

class MyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ElsaAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            backend="auto",
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
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

利用可能な `backend` オプション：

| 値 | 説明 |
|---|---|
| `"auto"` | **推奨。** dtype とハードウェアに応じて最適なカーネルを自動選択 |
| `"triton"` | ELSA Triton カーネル — FP16 / BF16 |
| `"triton_fp32"` | ELSA Triton カーネル — FP32 推論 |
| `"triton_fp32_train"` | ELSA Triton カーネル — 逆伝播付き FP32 訓練 |
| `"sdpa_math"` / `"sdpa_mem"` / `"sdpa_flash"` | PyTorch SDPA バックエンド（ベースライン比較用） |
| `"pytorch"` | 純 PyTorch フォールバック |

ランタイムでグローバルにバックエンドを切り替えることも可能です：

```python
from elsa import set_default_elsa_backend
set_default_elsa_backend("triton_fp32")
```

---

### レベル 3 — 事前学習済み timm ViT / Swin のパッチ（モデルコード変更なし）

```python
import timm
from fairbench_worker import patch_vit_attention, patch_swin_attention

# ── ViT ──────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda().eval()
patch_vit_attention(model, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = model(images)

# ── Swin Transformer ─────────────────────────────────────────────────────────
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).cuda().eval()
patch_swin_attention(swin, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = swin(images)
```

---

### レベル 4 — 新しい `ElsaViT` をゼロから構築

```python
import torch
from elsa import ElsaViT

model = ElsaViT(
    img_size=224, patch_size=16, embed_dim=768,
    depth=12, num_heads=12, num_classes=1000,
    elsa_backend="auto",
).cuda()

logits = model(torch.randn(4, 3, 224, 224, device="cuda"))   # (4, 1000)
```

---

### レベル 5 — LLaMA / HuggingFace 言語モデル

```python
import types, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from elsa import ELSA_triton, ELSA_triton_fp32

def patch_llama_attention(model):
    def _make_forward(original_forward):
        def elsa_forward(self, hidden_states, attention_mask=None,
                         position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, **kwargs):
            _real_sdpa = torch.nn.functional.scaled_dot_product_attention
            def _elsa_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, **kw):
                s = scale if scale is not None else q.shape[-1] ** -0.5
                if q.dtype == torch.float32:
                    return ELSA_triton_fp32.apply(q, k, v, s)
                return ELSA_triton.apply(q, k, v, s)
            torch.nn.functional.scaled_dot_product_attention = _elsa_sdpa
            try:
                return original_forward(self, hidden_states,
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache, **kwargs)
            finally:
                torch.nn.functional.scaled_dot_product_attention = _real_sdpa
        return elsa_forward
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.forward = types.MethodType(_make_forward(attn.__class__.forward), attn)
    return model

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)
patch_llama_attention(model)

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**FP32 長シーケンス推論（メモリ制約 GPU）**

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cuda"
)
patch_llama_attention(model)

long_input = tokenizer(
    very_long_document, return_tensors="pt", max_length=32768, truncation=True
).to("cuda")
with torch.no_grad():
    out = model(**long_input)
```

> **パープレキシティ等価性：** ELSA は精確な softmax セマンティクスを保持します。
> WikiText-2 で測定したパープレキシティは元のモデルと浮動小数点精度で一致します。

---

### レベルの選び方

| 状況 | 推奨 |
|---|---|
| 手動で Q、K、V を計算するカスタム `nn.Module` | **レベル 1** — `ELSA_triton` / `ELSA_triton_fp32` |
| モデル内の単一アテンションブロックを置き換える | **レベル 2** — `ElsaAttention` |
| 事前学習済み timm ViT / Swin、モデル変更なし | **レベル 3** — `patch_vit_attention` |
| 新しい ViT をゼロから訓練 | **レベル 4** — `ElsaViT` |
| HuggingFace LLaMA / デコーダ型 LLM | **レベル 5** — `F.scaled_dot_product_attention` の手動置き換え |

---

## 動作要件

- Python ≥ 3.9
- PyTorch ≥ 2.1（CUDA ビルド）
- Triton ≥ 2.2
- timm ≥ 0.9

```bash
pip install -r requirements.txt
```


---

## 環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `ELSA_TRITON_FP32_TRAIN_BWD` | `auto` | FP32 訓練逆伝播パス（`triton` は `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1` が必要） |
| `ELSA_TRITON_FP32_MEM_SAVE_OUT` | `1` | `1` = 速度優先；`0` = VRAM 削減 |
| `ELSA_TRITON_ALLOW_UNSTABLE_PATHS` | `0` | `1` で実験的パスを有効化 |
| `ELSA_FORCE_ALLOW_TF32` | `0` | TF32 ポリシーの上書き |

---

## ドキュメント

| ドキュメント | 説明 |
|---|---|
| [`docs/FULL_REPORT_20260301.md`](docs/FULL_REPORT_20260301.md) | 全設定の完全なベンチマーク結果 |
| [`docs/RELEASE_NOTES_20260301.md`](docs/RELEASE_NOTES_20260301.md) | 主要な更新と安定性修正 |
| [`docs/STATUS_MATRIX_20260301.md`](docs/STATUS_MATRIX_20260301.md) | 設定別サポートステータス |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 公平性制御付き再現ガイド |

---

## 引用

ELSA を研究に使用する場合は、以下を引用してください：

```bibtex
@inproceedings{hsu2026elsa,
  title={ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers},
  author={Hsu, Chih-Chung and Ma, Xin-Di and Liao, Wo-Ting and Lee, Chia-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## ライセンス

本プロジェクトは**学術研究および非商業目的のみ**でリリースされています。
詳細は [LICENSE](LICENSE) をご覧ください。
