> **注意：** 此简体中文版本为初版翻译，内容可能未反映最新功能（如 `elsa_strict_ref.py`、`elsa_ext_pack/`、`run_strict_coverage_matrix.py` 等）。最新完整说明请参阅 [English README](../README.md)。

# ELSA：用于快速、轻量内存视觉 Transformer 的精确线性扫描注意力机制

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[项目主页]</a> &nbsp;|&nbsp;
  <a href="#">[论文 (CVPR 2026)]</a> &nbsp;|&nbsp;
  <a href="#引用">[引用]</a>
</p>

<p align="center">
  <a href="../README.md">English</a> &nbsp;|&nbsp;
  <a href="README.zh-TW.md">繁體中文</a> &nbsp;|&nbsp;
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

ELSA 将 softmax 注意力机制重新表述为一个基于关联幺半群（associative monoid）状态三元组 *(m, S, W)* 的**并行前缀扫描**，实现以下特性：

- **精确 softmax 语义** — 具有可证明的 FP32 相对误差上界，无需重新训练
- **O(log n) 并行深度** — 通过两层扫描（块内 Hillis–Steele + 块间 Blelloch）
- **O(n) 额外内存** — 无需 O(n²) 分数矩阵，每个查询仅需单次 I/O
- **不依赖 Tensor Core** — 以 Triton 及 CUDA C++ 实现，支持 A100、L4、Jetson TX2

---

## 功能与应用

### 本项目提供的内容

| | |
|---|---|
| **Triton / CUDA kernel** | `ELSA_triton`（FP16/BF16）、`ELSA_triton_fp32`（推理）、`ELSA_triton_fp32_train`（训练含反向传播） |
| **PyTorch 模块** | `ElsaAttention` — 可替换任何标准 attention block 的 `nn.Module` |
| **完整模型类** | `ElsaViT`、`ElsaSwinTransformerV2` — 可直接训练的架构 |
| **Patch 工具** | `patch_vit_attention`、`patch_swin_attention` — 加速预训练 timm 模型，无需重写代码 |
| **基准测试框架** | `fairbench_driver` / `fairbench_worker` — 可重现的多后端速度与 VRAM 对比 |

### 潜在应用场景

- **高分辨率视觉** — 长序列 ViT 推理（医疗影像、卫星图像、高光谱分析），需要 FP32 精度的场合
- **内存受限的部署** — 在消费级 GPU 上运行 32K+ token 的 LLM 推理，无需担心标准 SDPA 的 OOM 问题
- **嵌入式 / 边缘 AI** — Jetson TX2 等设备受益于不依赖 Tensor Core 的设计
- **机器人与自动驾驶** — 在预算有限的车载或机器人计算平台（AGX Orin、Drive AGX）上执行实时感知；O(n²) 注意力往往成为延迟或功耗瓶颈，ELSA 的 O(n) 内存占用允许在相同 VRAM 空间内使用更大的上下文窗口，无需升级硬件即可获得更丰富的场景表示
- **3D 场景理解** — 加速 VGGT 等多帧模型，精度零损失
- **任意自定义 Transformer** — ELSA kernel 接受原始 Q/K/V 张量，可集成至任何架构

### 已提供的使用示例

| 示例 | 位置 |
|---|---|
| Raw Q/K/V kernel 替换 | [使用方式 — 层级 1](#层级-1--raw-kernel自定义-attention-class) |
| 自定义 `TransformerBlock` + `ElsaAttention` | [使用方式 — 层级 2](#层级-2--elsaattention-模块替换单一层) |
| 预训练 timm ViT / Swin patch | [使用方式 — 层级 3](#层级-3--patch-预训练-timm-vit--swin不修改模型代码) |
| 从头构建 `ElsaViT` | [使用方式 — 层级 4](#层级-4--从头构建-elsavit) |
| HuggingFace LLaMA patch + 长序列 FP32 | [使用方式 — 层级 5](#层级-5--llama--huggingface-语言模型) |

---

## 性能亮点

### FP32 推理 vs. SDPA-Math（CLIP ViT，A100）

| 模型 | 分辨率 | 加速比 | 内存节省 |
|---|---|---|---|
| ViT-B/16 | 224→560 px | 最高 **1.98×** | 最高 **36.1%** |
| ViT-L/14 | 224→560 px | 最高 **2.15×** | 最高 **39.6%** |

### FP32 训练（ViT，A100，batch=1/2，image=1024）

| 对比对象 | 中位加速比 | 中位 VRAM 比 |
|---|---|---|
| ELSA vs SDPA-Math | **1.72×** | **0.23×**（−77%） |
| ELSA vs SDPA-Mem | **1.09×** | **1.05×** |

### 高光谱图像分类（HSI-MAE，FP32）

| 模型 | 数据集 | vs ME-SDPA 加速比 |
|---|---|---|
| HSI-MAE-B | Pavia / Salinas / WHU | +37–40% |
| HSI-MAE-L | Pavia / Salinas / WHU | **+60–62%** |

### 嵌入式设备（Jetson TX2，FP16）

在所有 token 长度（64–900 tokens）下，相较 Math-SDPA 稳定实现 **~37% 延迟降低**。

### 3D 重建（VGGT，FP32 vs xFormers）

| 帧数 | 加速比 |
|---|---|
| 50 | **1.46×** |
| 100 | **2.09×** |
| 150 | **2.34×** |

---

## 方法

ELSA 将在线 softmax 转化为幺半群 `(m, S, W) ∈ ℝ × ℝ × ℝ^dv` 上的前缀扫描：

```
m  = 运行最大 logit
S  = 归一化累积 exp 权重和
W  = exp 加权值累加器
```

合并运算子 ⊕ 通过三个步骤组合两个块：**反归一化 → 聚合 → 重新归一化**，得到：

| 特性 | 数值 |
|---|---|
| 并行深度 | O(log n) |
| 额外内存 | O(n) |
| 每次查询 I/O | 1 次（K、V 各流式传输一次） |
| FP32 误差上界 | O(u · log n) |

---

## 支持的模型与硬件

**视觉模型：**
ViT（Tiny / Small / Medium / Base / Large）、Swin Transformer、CLIP、SAM、VGGT、HSI-MAE

**语言模型：**
LLaMA（8B、13B）、BERT

**硬件平台：**
NVIDIA A100、L4、Jetson TX2 · 任何支持 CUDA 的 GPU（不依赖 Tensor Core）

> **注意：** 本次 release 的所有 benchmark 结果均在 NVIDIA GPU（A100-40GB）上收集。
> AMD/ROCm 支持计划于后续版本提供。

---

## 与其他注意力核的比较

| 方法 | 精确 | FP32 原生 | GPU 通用 | 免重新训练 | 深度 | 额外内存 |
|---|---|---|---|---|---|---|
| Standard SDPA | ✓ | ✗ | ✓ | ✓ | O(n) | O(n²) |
| FlashAttention-2/3 | ✓ | ✗ | ✗ | ✓ | O(n/Tk) | O(Tk·d) |
| Linear Attention | ✗ | ✓ | ✓ | ✗ | O(log n)† | O(n) |
| **ELSA（本文）** | **✓** | **✓** | **✓** | **✓** | **O(log n)** | **O(n)** |

---

## 代码库结构

```
code/
  stable/          # 正式版 ELSA 核心与模型集成
  future_exp/      # 实验性路径（不纳入主要性能声明）
scripts/           # 可重现的基准测试与验证脚本
docs/              # 发布说明、状态矩阵、可重现性指南、完整报告
results/           # 精选 CSV 输出（attn_only、full_model、train_ft、downstream、llm）
validation/        # 快速验证输出
manifests/         # 文件清单与 SHA256 校验码
```

主要源码文件：

| 文件 | 用途 |
|---|---|
| `code/stable/elsa_triton.py` | Triton 注意力核（tile 配置、FP32 反向传播、save-out 控制） |
| `code/stable/elsa.py` | `ElsaAttention` / `ElsaViT` 模型类 |
| `code/stable/elsa_swin.py` | Swin Transformer 集成 |
| `code/stable/elsa_swin_fused.py` | 融合 Swin 核路径 |

---

## 安装

```bash
git clone https://github.com/your-org/elsa.git
cd elsa
pip install -e .           # 以可编辑模式安装 elsa 包
```

> 依赖项（`torch`、`triton`、`timm`）会自动安装。
> 若需基准测试工具，添加 `[benchmark]` 附加选项：
> `pip install -e ".[benchmark]"`

---

## 使用方式

### 层级 1 — Raw kernel（自定义 attention class）

已有 Q、K、V 张量，只想替换注意力计算本身：

```python
import torch
from elsa import ELSA_triton, ELSA_triton_fp32, ELSA_pytorch

B, H, N, D = 2, 12, 1024, 64   # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = D ** -0.5

# FP16 / BF16 — 最快路径
out = ELSA_triton.apply(q, k, v, scale)          # (B, H, N, D)

# FP32 推理 — 省内存，可证明精确
out = ELSA_triton_fp32.apply(q.float(), k.float(), v.float(), scale)

# FP32 训练（支持反向传播）
out = ELSA_triton_fp32_train.apply(q.float(), k.float(), v.float(), scale)

# 纯 PyTorch fallback — 不需要 Triton，完整 autograd 支持
out = ELSA_pytorch(q, k, v, scale)
```

以上均可直接替代：

```python
# 原来的写法
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
```

---

### 层级 2 — `ElsaAttention` 模块（替换单一层）

`ElsaAttention` 是完整的 `nn.Module`（QKV 投影 → ELSA kernel → 输出投影），接口与 `timm.models.vision_transformer.Attention` 相同：

```python
import torch
import torch.nn as nn
from elsa import ElsaAttention

class MyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 直接替换 timm Attention / nn.MultiheadAttention
        self.attn = ElsaAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=0.0,
            proj_drop=0.0,
            backend="auto",   # 推荐：自动选择最佳 kernel
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

可用的 `backend` 选项：

| 值 | 说明 |
|---|---|
| `"auto"` | **推荐默认**。依 dtype 与硬件自动选择最佳 kernel |
| `"triton"` | ELSA Triton kernel — FP16 / BF16 |
| `"triton_fp32"` | ELSA Triton kernel — FP32 推理 |
| `"triton_fp32_train"` | ELSA Triton kernel — FP32 训练含反向传播 |
| `"sdpa_math"` / `"sdpa_mem"` / `"sdpa_flash"` | PyTorch SDPA 各后端（基线对比用） |
| `"pytorch"` | 纯 PyTorch fallback |

也可在运行时全局切换 backend：

```python
from elsa import set_default_elsa_backend
set_default_elsa_backend("triton_fp32")   # 之后创建的 ElsaAttention 都使用此设置
```

---

### 层级 3 — Patch 预训练 timm ViT / Swin（不修改模型代码）

只需一行调用，热替换所有注意力层：

```python
import timm
from elsa import ElsaAttention
from fairbench_worker import patch_vit_attention, patch_swin_attention

# ── ViT ──────────────────────────────────────────────────────────────────────
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda().eval()

# 就地替换所有 attention layer 为 ELSA kernel
patch_vit_attention(model, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = model(images)   # 以 ELSA 注意力运行

# ── Swin Transformer ─────────────────────────────────────────────────────────
swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).cuda().eval()
patch_swin_attention(swin, method="elsa", precision="fp32", full_model_mode=True)

with torch.no_grad():
    out = swin(images)
```

---

### 层级 4 — 从头构建 `ElsaViT`

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

### 层级 5 — LLaMA / HuggingFace 语言模型

ELSA kernel 与架构无关，可用于任何暴露 Q、K、V 张量的 Transformer，包括 decoder-only LLM。以下示例就地 patch 一个 HuggingFace LLaMA 模型：

```python
import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from elsa import ELSA_triton, ELSA_triton_fp32

def patch_llama_attention(model):
    """将每个 LLaMA attention 层的 scaled_dot_product_attention 替换为 ELSA。"""

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
model_id = "meta-llama/Llama-2-7b-hf"   # 任意 HuggingFace LLaMA 变体均可
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="cuda"
)

patch_llama_attention(model)   # 热替换注意力，无需重新训练

inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**FP32 长序列推理（内存受限的 GPU）**

在长序列（>16K tokens）下，FP32 SDPA 需要 O(n²) 的分数矩阵，在消费级 GPU 上会 OOM。ELSA 的 O(n) 内存用量让这些序列得以正常运行：

```python
# 精度敏感任务（医疗、科学）使用 FP32 加载
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cuda"
)
patch_llama_attention(model)

# 使用标准 SDPA 会 OOM 的 32K token context
long_input = tokenizer(
    very_long_document, return_tensors="pt", max_length=32768, truncation=True
).to("cuda")

with torch.no_grad():
    out = model(**long_input)
```

> **Perplexity 等价：** ELSA 保留精确的 softmax 语义——
> 在 WikiText-2 上测量的 perplexity 与原始模型一致，精确到浮点精度。

---

### 如何选择层级

| 情境 | 建议 |
|---|---|
| 自定义 `nn.Module` 并手动计算 Q、K、V | **层级 1** — `ELSA_triton` / `ELSA_triton_fp32` |
| 替换模型中的单一 attention block | **层级 2** — `ElsaAttention` |
| 预训练 timm ViT / Swin，不想改模型 | **层级 3** — `patch_vit_attention` |
| 从头训练新的 ViT | **层级 4** — `ElsaViT` |
| HuggingFace LLaMA / decoder LLM | **层级 5** — 手动替换 `F.scaled_dot_product_attention` |

---

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.1（CUDA 版本）
- Triton ≥ 2.2
- timm ≥ 0.9

```bash
pip install -r requirements.txt
```


---

## 快速开始

```bash
# 仅注意力基准测试（ViT FP32）
python scripts/benchmark_pure_attention_vit.py \
  --device cuda:0 --dtype fp32 --seq-lens 1025,2305,4097 \
  --warmup 6 --trials 30 \
  --output results/attn_only/recheck_vit_fp32.csv

# 完整模型吞吐量
python scripts/benchmark_model_throughput.py \
  --device cuda:0 --warmup 8 --trials 32 \
  --output results/full_model/recheck_full_model.csv

# 训练 / 微调 / 反向传播矩阵
python scripts/benchmark_train_ft_matrix.py \
  --device cuda:0 --families vit swin \
  --modes backward train finetune \
  --precisions fp32 tf32 fp16 \
  --specs vit_small_512 swin_tiny_w8_256 \
  --steps 20 --warmup 8 --seed 123 \
  --output results/train_ft/recheck_train_ft.csv
```

使用 `scripts/run_benchmark.sh` 可自动选取空闲 GPU 并激活 conda 环境：

```bash
export ELSA_CONDA_ENV=elsa
bash scripts/run_benchmark.sh python scripts/benchmark_model_throughput.py --device cuda:0
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ELSA_TRITON_FP32_TRAIN_BWD` | `auto` | FP32 训练反向路径（`triton` 需配合 `ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1`） |
| `ELSA_TRITON_FP32_MEM_SAVE_OUT` | `1` | `1` = 速度优先；`0` = 降低 VRAM |
| `ELSA_TRITON_ALLOW_UNSTABLE_PATHS` | `0` | 设为 `1` 以启用实验性路径 |
| `ELSA_FORCE_ALLOW_TF32` | `0` | 覆盖 TF32 策略 |

---

## 技术文档

| 文档 | 说明 |
|---|---|
| [`docs/FULL_REPORT_20260301.md`](docs/FULL_REPORT_20260301.md) | 所有配置下的完整基准测试结果 |
| [`docs/RELEASE_NOTES_20260301.md`](docs/RELEASE_NOTES_20260301.md) | 本版本主要更新与稳定性修复 |
| [`docs/STATUS_MATRIX_20260301.md`](docs/STATUS_MATRIX_20260301.md) | 各配置支持状态 |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 含公平性控制的重现指南 |

---

## 引用

若您在研究中使用了 ELSA，请引用：

```bibtex
@inproceedings{hsu2026elsa,
  title={ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers},
  author={Hsu, Chih-Chung and Ma, Xin-Di and Liao, Wo-Ting and Lee, Chia-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 许可证

本项目仅供**学术研究与非商业用途**使用。
详见 [LICENSE](LICENSE)。
