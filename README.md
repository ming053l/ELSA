# ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[Project Page]</a> &nbsp;|&nbsp;
  <a href="#">[Paper (CVPR 2026)]</a> &nbsp;|&nbsp;
  <a href="#citation">[Citation]</a>
</p>

<p align="center">
  <a href="other/README.zh-TW.md">繁體中文</a> &nbsp;|&nbsp;
  <a href="other/README.zh-CN.md">简体中文</a> &nbsp;|&nbsp;
  <a href="other/README.ja.md">日本語</a> &nbsp;|&nbsp;
  <a href="other/README.ko.md">한국어</a>
</p>

---

<div align="center">

**<a href="https://cchsu.info/wordpress/">Chih-Chung Hsu</a>, Xin-Di Ma, Wo-Ting Liao, <a href="https://ming053l.github.io/">Chia-Ming Lee</a>**

Advanced Computer Vision Laboratory, National Yang Ming Chiao Tung University

CVPR Findings 2026

*"Can't use FA on your device? Try ELSA on!"*

</div>

---

## Overview

ELSA reformulates softmax attention as a **parallel prefix scan** over an associative monoid of state triples *(m, S, W)*, achieving:

- **Exact softmax semantics** — provable FP32 relative error bound, no retraining required
- **O(log n) parallel depth** via two-level scan (intra-block Hillis–Steele + inter-block Blelloch)
- **O(n) extra memory** — no O(n²) score matrix, single-pass I/O per query
- **Tensor-Core independent** — implemented in Triton and CUDA C++, runs on A100, L4, Jetson TX2

---

## Features & Applications

### What this repository provides

| | |
|---|---|
| **Triton / CUDA kernels** | `ELSA_triton` (FP16/BF16), `ELSA_triton_fp32` (inference), `ELSA_triton_fp32_train` (training + backward) |
| **Native CUDA extension** | `elsa_ext_pack/` — optional C++/CUDA extension for performance-critical paths |
| **Strict reference impl** | `ElsaStrictState` — provably-correct associative state triple with `merge_states()` |
| **PyTorch module** | `ElsaAttention` — drop-in `nn.Module` replacing any standard attention block |
| **Full model classes** | `ElsaViT`, `ElsaSwinTransformerV2` — ready-to-train architectures |
| **Patching utilities** | `patch_vit_attention`, `patch_swin_attention` — accelerate pretrained timm models without rewriting them |
| **Benchmark harness** | `fairbench_driver` / `fairbench_worker` + strict coverage matrix runner |

### Potential applications

- **High-resolution vision** — long-sequence ViT inference (medical imaging, satellite imagery, hyperspectral analysis) where FP32 precision is mandatory
- **Memory-constrained deployment** — run 32K+ token LLM inference on consumer GPUs that OOM with standard SDPA
- **Embedded / edge AI** — Jetson TX2 and similar devices benefit from the Tensor-Core-independent design
- **Robotics & autonomous driving** — real-time perception on budget compute (AGX Orin, Drive AGX) where O(n²) attention is a latency or power bottleneck; ELSA's O(n) memory footprint allows larger context windows within the same VRAM envelope, enabling richer scene representations without upgrading hardware
- **3D scene understanding** — accelerate multi-frame models such as VGGT with no accuracy loss
- **Any custom Transformer** — ELSA kernels accept raw Q/K/V tensors and integrate into any architecture

### Included usage examples

| Example | Location |
|---|---|
| Raw Q/K/V kernel replacement | [Usage — Level 1](#level-1--raw-kernel-custom-attention-class) |
| Custom `TransformerBlock` with `ElsaAttention` | [Usage — Level 2](#level-2--elsaattention-module-replace-one-layer) |
| Pretrained timm ViT / Swin patch | [Usage — Level 3](#level-3--patch-a-pretrained-timm-vit--swin-zero-model-code-changes) |
| Build `ElsaViT` from scratch | [Usage — Level 4](#level-4--build-a-new-elsavit-from-scratch) |
| HuggingFace LLaMA patch + long-context FP32 | [Usage — Level 5](#level-5--llama--huggingface-language-models) |

---

## Performance Highlights

### FP32 Inference vs. SDPA-Math (CLIP ViT attention-only, A100) — Table 2

| Model | Resolution | Speedup | VRAM Saving |
|---|---|---|---|
| ViT-B/16 | 224→560 px | up to **1.98×** | up to **36.1%** |
| ViT-L/14 | 224→560 px | up to **2.12×** | up to **39.6%** |
| ViT-L/14-336 | 224→560 px | up to **2.15×** | up to **39.6%** |

Gains scale with resolution. FP32 comparisons exclude FA2/FA3 as their FP32 fallbacks revert to
untuned SIMD paths—not comparable on our hardware.

### FP16 Full-Model Throughput (ImageNet-1K, A100-40GB, batch=8, strict scan) — Table 3

ELSA / W-ELSA is faster than FA2 on every reported ViT and Swin configuration.

| Model | ELSA (img/s) | vs FA2 | VRAM vs FA2 |
|---|---|---|---|
| ViT-T | **1309** | +65.5% | −6.1% |
| ViT-S | **1276** | +48.5% | −7.3% |
| ViT-M | **1204** | +45.4% | −13.0% |
| ViT-B | **1064** | +32.5% | −11.5% |
| Swin-T/W8 | **597** | +13.4% | +3.9% |
| Swin-T/W16 | **520** | +7.2% | 0.0% |
| Swin-S/W16 | **305** | +13.9% | −8.1% |

### FP32 Long-Sequence vs. ME-SDPA (A100)

Across all FP32 benchmarks ELSA delivers **1.3–3.5× speedup** over ME-SDPA while using
comparable peak memory. Gains increase with sequence length:

| Tokens | FP32 Speedup vs ME-SDPA (approx.) |
|---|---|
| 1K | ~1.3× |
| 4K | ~2× |
| 16K | ~3.5× |

### NLP Benchmarks — BERT (FP32, bucketed) — Table 7

| Task | ELSA Speedup vs ME-SDPA |
|---|---|
| SST-2 sentiment | **1.97×** |
| IMDB sentiment | **2.27×** |

### Hyperspectral Image Classification (HSI-MAE, FP32) — Table 4

| Model | Dataset | Throughput gain vs ME-SDPA |
|---|---|---|
| HSI-MAE-B | Pavia / Salinas / WHU | **+37–40%** |
| HSI-MAE-L | Pavia / Salinas / WHU | **+60–62%** |

Memory overhead remains negligible (sub-gigabyte) across all datasets.

### Embedded Device (Jetson TX2, FP16 and FP32) — Table 6

Consistent **~35–38% latency reduction** (≈ **1.5–1.6×**) vs PyTorch Math kernel across all
token lengths (64–900 tokens), with no Tensor Core dependency.

### 3D Reconstruction (VGGT, FP32 vs xFormers) — Table 11

| Frames | Speedup |
|---|---|
| 50 | **1.46×** |
| 100 | **2.09×** |
| 150 | **2.34×** |
| 400 | **1.38×** (FastVGGT scaling) |

### LLaMA-13B Host-Device Offloading (FP32) — Table 13

At ≥ 32K tokens, ELSA's lower memory footprint hides PCIe transfer latency, delivering
**17.8–20.2% throughput gains** over SDPA with no weight modification.

### Strict Coverage Benchmark (Clean A100, 2026-04-21)

The following results use the strict acceptance rule: both latency and VRAM must be at least
comparable (`≤ 1.05×` baseline), with at least one metric better (`≤ 0.98×` baseline).

| Area | Result | Key numbers |
|---|---|---|
| **Attention-only 16/16** | ✅ All pass (best run) | See `docs/clean_logs/attn_only_16x16_gpu0_20260413_report.txt` |
| **ViT fp32 full-model 4/4** | ✅ All pass (stable) | `img224 fwd: 0.777×lat / 0.988×mem` |
| Long-token trend (ViT fp32 fwd) | Confirmed | `N=4096 → 0.286×lat`, `N=16384 → pass` |

See [docs/benchmark_summary.md](docs/benchmark_summary.md) for detailed numbers and methodology.

---

## Method

ELSA casts online softmax into a prefix scan over the monoid `(m, S, W) ∈ ℝ × ℝ × ℝ^dv`:

```
m  = running max logit
S  = normalized cumulative sum of exp weights
W  = exp-weighted value accumulator
```

The merge operator ⊕ composes two blocks in three steps: **unnormalize → aggregate → renormalize**. This yields:

| Property | Value |
|---|---|
| Parallel depth | O(log n) |
| Extra memory | O(n) |
| I/O per query | 1 pass (K, V streamed once) |
| FP32 error bound | O(u · log n) |

The strict reference implementation in `code/stable/elsa_strict_ref.py` provides a CPU-verifiable
proof of the `merge_states()` associativity and serves as the correctness baseline for all kernel
validation.

---

## Tested Models & Hardware

The models below were explicitly benchmarked in the paper. Because ELSA kernels accept raw Q/K/V
tensors, **any standard Transformer architecture** can be patched with `patch_vit_attention` or
`patch_swin_attention` in the same way — no retraining required.

**Vision (paper benchmarks):**
ViT (Tiny / Small / Medium / Base / Large), Swin Transformer, CLIP, SAM, VGGT, HSI-MAE

**Language (paper benchmarks):**
LLaMA (8B, 13B), BERT

**Hardware:**
NVIDIA A100, L4, Jetson TX2 · any CUDA-capable GPU (Tensor-Core independent)

> **Note:** All benchmark results in this release were collected on NVIDIA GPUs (A100-40GB).
> AMD/ROCm support is planned for a future release.

---

## Comparison vs. Other Attention Kernels

| Method | Exact | FP32-native | GPU-Agnostic | Retrain-free | Depth | Extra Mem |
|---|---|---|---|---|---|---|
| Standard SDPA | ✓ | ✗ | ✓ | ✓ | O(n) | O(n²) |
| FlashAttention-2/3 | ✓ | ✗ | ✗ | ✓ | O(n/Tk) | O(Tk·d) |
| Linear Attention | ✗ | ✓ | ✓ | ✗ | O(log n)† | O(n) |
| GatedDeltaNet | ✗ | ✓ | ✓ | ✗ | O(log n) | O(n) |
| Nyströmformer | ✗ | ✓ | ✓ | ✗ | O(m·n) | O(m·n) |
| **ELSA (Ours)** | **✓** | **✓** | **✓** | **✓** | **O(log n)** | **O(n)** |

Comparison scripts against GatedDeltaNet and Nyströmformer are in `scripts/`.

---

## Repository Layout

```
code/
  stable/          # Production-ready ELSA kernels and model integrations
    elsa_triton.py          # Triton kernels (tile config, FP32 backward, save-out control)
    elsa.py                 # ElsaAttention / ElsaViT model classes
    elsa_swin.py            # Swin Transformer integration
    elsa_swin_fused.py      # Fused Swin kernel path
    elsa_strict_ref.py      # Strict reference: ElsaStrictState + merge_states() proof  ← new
  future_exp/      # Experimental paths (not used for headline claims)
elsa_ext_pack/     # Optional native CUDA extension (C++ binding + .cu kernel)  ← new
scripts/
  run_strict_coverage_matrix.py   # Unified strict coverage harness (16-cell matrix)  ← new
  bench_gated_deltanet_vs_elsa.py # vs GatedDeltaNet  ← new
  bench_nystromformer_vs_elsa.py  # vs Nyströmformer  ← new
  bench_truefp32_vs_elsa_triton.py                    ← new
  bench_elsa_precision_compare.py                     ← new
  bench_attn_shape_fp32.py                            ← new
  benchmark_pure_attention_vit.py
  benchmark_model_throughput.py
  benchmark_train_ft_matrix.py
  benchmark_downstream_ft_cifar10.py
  run_vit_sota.py / run_swin_sota_fused.py / run_swin_fp32_size_sweep.py
  verify_swin_elsa_exactness.py
docs/
  benchmark_summary.md      # Strict coverage pass/fail summary  ← new
  clean_logs/               # Curated clean benchmark reports    ← new
  FULL_REPORT_20260301.md
  RELEASE_NOTES_20260301.md
  STATUS_MATRIX_20260301.md
  REPRODUCIBILITY.md
results/           # Curated CSV outputs
validation/        # Smoke-check outputs
manifests/         # File manifests and SHA256 checksums
```

---

## Installation

### Basic (Triton kernels only)

```bash
git clone https://github.com/ming053l/ELSA.git
cd ELSA
pip install -e .
```

> Dependencies (`torch`, `triton`, `timm`) are installed automatically.
> For benchmark utilities: `pip install -e ".[benchmark]"`

### With native CUDA extension (optional, higher performance on some paths)

```bash
cd elsa_ext_pack
python setup.py bdist_wheel
pip install dist/*.whl
```

> Requires a CUDA toolkit matching your PyTorch installation.
> The Triton-only install works without this extension.

---

## Usage

### Level 1 — Raw kernel (custom attention class)

For when you already have Q, K, V tensors and just want to replace the attention computation:

```python
import torch
from elsa import ELSA_triton, ELSA_triton_fp32, ELSA_pytorch

B, H, N, D = 2, 12, 1024, 64   # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = D ** -0.5

# FP16 / BF16 — fastest path
out = ELSA_triton.apply(q, k, v, scale)

# FP32 inference — memory-efficient, provably exact
out = ELSA_triton_fp32.apply(q.float(), k.float(), v.float(), scale)

# FP32 training (supports backward)
out = ELSA_triton_fp32_train.apply(q.float(), k.float(), v.float(), scale)

# Pure-PyTorch fallback — no Triton required, full autograd support
out = ELSA_pytorch(q, k, v, scale)
```

---

### Level 2 — `ElsaAttention` module (replace one layer)

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
        self.mlp  = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

Available `backend` options:

| Value | Description |
|---|---|
| `"auto"` | **Recommended.** Auto-selects best kernel per dtype & hardware |
| `"triton"` | ELSA Triton kernel — FP16 / BF16 |
| `"triton_fp32"` | ELSA Triton kernel — FP32 inference |
| `"triton_fp32_train"` | ELSA Triton kernel — FP32 training with backward |
| `"strict_core_ref"` | Strict reference path (correctness baseline) |
| `"sdpa_math"` / `"sdpa_mem"` / `"sdpa_flash"` | PyTorch SDPA backends |
| `"pytorch"` | Pure-PyTorch fallback |

---

### Level 3 — Patch a pretrained timm ViT / Swin (zero model-code changes)

```python
import timm
from fairbench_worker import patch_vit_attention, patch_swin_attention

model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda().eval()
patch_vit_attention(model, method="elsa", precision="fp32", full_model_mode=True)

swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True).cuda().eval()
patch_swin_attention(swin, method="elsa", precision="fp32", full_model_mode=True)
```

---

### Level 4 — Build a new `ElsaViT` from scratch

```python
from elsa import ElsaViT
model = ElsaViT(img_size=224, patch_size=16, embed_dim=768,
                depth=12, num_heads=12, num_classes=1000,
                elsa_backend="auto").cuda()
```

---

### Level 5 — LLaMA / HuggingFace language models

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

model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
patch_llama_attention(model)
```

> **Perplexity equivalence:** ELSA preserves exact softmax semantics —
> measured perplexity on WikiText-2 matches the original model to floating-point precision.

---

### Choosing the right level

| Situation | Recommended |
|---|---|
| Custom `nn.Module` with manual Q, K, V | **Level 1** |
| Replace one attention block | **Level 2** — `ElsaAttention` |
| Pretrained timm ViT / Swin, no model changes | **Level 3** — `patch_vit_attention` |
| Train a new ViT from scratch | **Level 4** — `ElsaViT` |
| HuggingFace LLaMA / decoder LLM | **Level 5** |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1 (CUDA build)
- Triton ≥ 2.2
- timm ≥ 0.9

```bash
pip install -r requirements.txt
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ELSA_TRITON_FP32_TRAIN_BWD` | `auto` | FP32 training backward path |
| `ELSA_TRITON_FP32_MEM_SAVE_OUT` | `1` | `1` = speed-first; `0` = lower VRAM |
| `ELSA_TRITON_ALLOW_UNSTABLE_PATHS` | `0` | Set to `1` to opt into experimental paths |
| `ELSA_FORCE_ALLOW_TF32` | `0` | Override TF32 policy |
| `ELSA_STRICT_SMALL_PROVIDER` | `0` | Enable narrow exact-SDPA provider for short-sequence paths |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/benchmark_summary.md`](docs/benchmark_summary.md) | Strict coverage pass/fail summary with acceptance rules |
| [`docs/clean_logs/`](docs/clean_logs/) | Curated clean benchmark reports (attn-only 16/16, ViT fp32 4/4, long-token trend) |
| [`docs/FULL_REPORT_20260301.md`](docs/FULL_REPORT_20260301.md) | Complete benchmark results |
| [`docs/RELEASE_NOTES_20260301.md`](docs/RELEASE_NOTES_20260301.md) | Key updates and stability fixes |
| [`docs/STATUS_MATRIX_20260301.md`](docs/STATUS_MATRIX_20260301.md) | Per-configuration support status |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Reproduction guide with fairness controls |

Traditional Chinese versions available as `*.zh-TW.md`.

---

## Citation

If you use ELSA in your research, please cite:

```bibtex
@inproceedings{hsu2026elsa,
  title={ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers},
  author={Hsu, Chih-Chung and Ma, Xin-Di and Liao, Wo-Ting and Lee, Chia-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

---

## Acknowledgments

This study was supported in part by the National Science and Technology Council (NSTC), Taiwan, under grants 112-2221-E-006-157-MY3, 114-2627-M-A49-003, 114-2218-E-035-001, and 114-2119-M-006-007. We thank the National Center for High-performance Computing (NCHC) of National Applied Research Laboratories (NARLabs) in Taiwan for providing computational and storage resources.

## License

This project is released for **academic research and non-commercial use only**.
See [LICENSE](LICENSE) for full terms.
