# ELSA Repository — File Guide

This document describes the purpose of every file and directory in the repository.
Numbers in parentheses refer to tables or sections in the CVPR 2026 paper.

---

## Root

| File | Purpose |
|------|---------|
| `README.md` | Main project introduction: overview, method, performance highlights (paper Tables 2–13), installation, usage examples (Level 1–5), environment variables, documentation index, citation. |
| `setup.py` | `pip install -e .` entry point. Declares the `elsa` package, maps `code/stable/` as the importable root, and lists runtime dependencies (`torch`, `triton`, `timm`). |
| `requirements.txt` | Pinned dependency list for reproducible installs. |
| `LICENSE` | Academic non-commercial license terms. |

---

## `code/`

Core ELSA source code, split into production-ready (`stable/`) and experimental (`future_exp/`) sub-trees.

### `code/stable/` — Production kernels and model classes

| File | Purpose |
|------|---------|
| `elsa_triton.py` | **Primary Triton kernel file** (≈ 7 500 lines). Implements all ELSA attention variants in Triton DSL: FP16/BF16 forward, FP32 inference, FP32 training with backward, save-out control, and tile/launch configuration. This is the innermost compute layer that all higher-level APIs call. |
| `elsa.py` | **ViT integration layer** (≈ 2 100 lines). Defines `ElsaAttention` (`nn.Module` drop-in), `ElsaViT`, `ElsaDistilled`, backend selection (`set_default_elsa_backend` / `get_default_elsa_backend`), and strict-mode dispatch helpers (`_arm_vit_dispatch_runtime`, `_vit_model_dispatch_env_overrides`). |
| `elsa_swin.py` | **Swin Transformer integration** (≈ 2 300 lines). Defines `ElsaSwinTransformerV2` with window-based attention, compact-mask / fuse-bias front-end switches (`strict_use_compact_mask_train/eval`, `strict_fuse_compact_bias`), and W-ELSA launch policy. |
| `elsa_strict_ref.py` | **Strict reference implementation** (573 lines). CPU-verifiable proof of ELSA correctness. Defines `ElsaStrictState` (the `(m, S, W)` triple), `merge_states()` (associativity proof), and a sequential scan that can be used to bit-check kernel outputs. Absent from the earlier public release; added as part of the SICNet validation effort. |
| `elsa_swin_fused.py` | Fused Swin kernel path — combines QKV projection and ELSA scan into a single CUDA launch to reduce kernel-call overhead for small window sizes. |
| `elsa_triton_full.py` | Full-sequence (non-windowed) Triton variant for Swin in global-attention mode. Wraps `elsa_triton.py` for models that switch between window and global attention. |
| `elsa_triton_swin_fused.py` | Triton version of the fused Swin path. Mirrors `elsa_swin_fused.py` but written in Triton rather than CUDA C++. |
| `sic_triton.py` | Legacy 16-line shim. Re-exports the main Triton kernel under the historical `sic_triton` module name for backward compatibility with older checkpoints and scripts. |
| `sic_triton_baseline.py` | 5-line baseline shim. Redirects to the reference PyTorch SDPA path so scripts that import `sic_triton_baseline` still run without modification. |
| `__init__.py` | Package init. Re-exports the public API: `ELSA_triton`, `ELSA_triton_fp32`, `ELSA_triton_fp32_train`, `ELSA_pytorch`, `ElsaAttention`, `ElsaViT`, `set_default_elsa_backend`, `patch_vit_attention`, `patch_swin_attention`. |
| `_builder.py` | `timm`-compatible model builder helpers (`build_model_with_cfg`, `build_pretrained_copy`). Copied from `timm` and lightly patched to route ELSA model families through the correct constructor. |
| `_features.py` | Feature-extraction helpers (`FeatureInfo`, `FeatureHooks`) from `timm`. Allows ELSA ViT/Swin to act as feature-pyramid backbones without changes to downstream task heads. |
| `_features_fx.py` | `torch.fx`-based feature extraction. Enables tracing ELSA models with `torch.fx` for export and graph-level optimizations. |
| `_manipulate.py` | Weight-manipulation utilities (`checkpoint_seq`, `named_apply`, `group_parameters`). Used when loading pretrained `timm` weights into ELSA model instances. |
| `_registry.py` | Model registry (`register_model`, `model_entrypoint`). Registers ELSA variants under `timm`-compatible names so `timm.create_model("elsa_vit_base_patch16_224")` resolves correctly. |

### `code/future_exp/` — Experimental / historical paths

| File | Purpose |
|------|---------|
| `elsa_triton.py` | Earlier experimental Triton kernel snapshot. Kept for reference and regression testing. **Not used by the production API.** |
| `elsa_triton_entry.py` | Thin entry-point wrapper used during early development to hot-swap kernel versions without editing callers. Superseded by the backend selection in `code/stable/elsa.py`. |
| `sic_triton.py` | Experimental SIC (Strict I/O Conscious) Triton path from early development. Evolved into `code/stable/elsa_strict_ref.py`. **Not part of headline claims.** |

---

## `elsa_ext_pack/` — Optional native CUDA extension

Provides a compiled C++/CUDA extension for platforms where Triton JIT is unavailable or when
finer memory-hierarchy control is needed (e.g., host-device offloading on Jetson TX2, Table 6).

| File | Purpose |
|------|---------|
| `elsa_kernel.cu` | Native CUDA kernel implementing the two-level prefix scan (Hillis–Steele intra-block + Blelloch inter-block). Template-parameterized on block size and precision. Avoids HMMA/GMMA instructions, enabling FP32 on Tensor-Core-free devices. |
| `elsa.cpp` | PyTorch C++ binding layer (`torch::Tensor elsa_forward(...)`). Dispatches to `elsa_kernel.cu`, handles tensor contiguity checks, and registers the op with `PYBIND11_MODULE`. |
| `setup.py` | `torch.utils.cpp_extension.CUDAExtension` build script. Targets `sm_80` (A100) by default; edit `extra_compile_args` to add other architectures. |
| `README.md` | Build instructions, supported architectures, and usage notes for the extension. |

---

## `scripts/` — Benchmark and validation scripts

### Strict coverage (SICNet addition)

| File | Purpose |
|------|---------|
| `run_strict_coverage_matrix.py` | **Unified strict coverage harness** (804 lines). Runs the full 16-cell matrix (ViT/Swin × fp16/fp32 × fwd/fwd_bwd × attention-only/full-model) with `--fresh-backend` subprocess isolation and alternating backend order to prevent order bias. Primary tool for the pass/fail evidence in `docs/clean_logs/`. |

### Comparison benchmarks (SICNet addition)

| File | Purpose |
|------|---------|
| `bench_gated_deltanet_vs_elsa.py` | Head-to-head throughput comparison of ELSA vs GatedDeltaNet across sequence lengths and dtypes. |
| `bench_nystromformer_vs_elsa.py` | Head-to-head comparison vs Nyströmformer (landmark-token approximation). Demonstrates ELSA's exact-semantics advantage at no throughput cost. |
| `bench_truefp32_vs_elsa_triton.py` | Compares ELSA's Triton FP32 path against a numerically identical "true FP32" reference, verifying that the scan formulation does not introduce latency overhead relative to a naive FP32 SDPA. |
| `bench_elsa_precision_compare.py` | Benchmarks all ELSA precision variants (FP16, BF16, FP32-strict, TF32-turbo) side-by-side on the same hardware, reproducing the data behind Table 15 of the paper. |
| `bench_attn_shape_fp32.py` | Sweeps over (B, H, N, D) shape space in FP32 to characterize where ELSA's memory advantage becomes dominant. Useful for profiling custom architectures before deployment. |
| `bench_linear_attn_shared.py` | Shared harness used by multiple comparison scripts. Provides a unified `run_benchmark(method, ...)` API that handles warmup, timing, and memory tracking in a fresh subprocess. |

### Standard benchmarks (from original ELSA-main)

| File | Purpose |
|------|---------|
| `benchmark_pure_attention_vit.py` | Attention-module-only throughput benchmark for ViT across sequence lengths and dtypes (paper Figure 4, Table 2). |
| `benchmark_model_throughput.py` | Full-model forward throughput on ImageNet-1K image sizes (paper Table 3). |
| `benchmark_train_ft_matrix.py` | Training / fine-tuning / backward matrix benchmark across families, precisions, and modes. |
| `benchmark_downstream_ft_cifar10.py` | Downstream fine-tuning benchmark on CIFAR-10 to verify ELSA does not degrade task accuracy. |
| `run_vit_sota.py` | Reproduces ViT SOTA throughput and memory numbers from the paper (Table 3, ViT family). |
| `run_swin_sota_fused.py` | Reproduces Swin SOTA numbers using the fused W-ELSA path (Table 3, Swin family). |
| `run_swin_fp32_size_sweep.py` | Window-size and resolution sweep for Swin in FP32, characterizing the short-window overhead discussed in §J (Limitations). |
| `verify_swin_elsa_exactness.py` | Smoke-test that W-ELSA output matches the reference Math-SDPA output to within the FP32 error bound of Theorem 1. |
| `run_benchmark.sh` | Shell wrapper: selects a free GPU (`nvidia-smi`), activates the conda environment, and forwards arguments to any benchmark script. |

---

## `docs/`

### New additions (SICNet)

| File / Directory | Purpose |
|------|---------|
| `benchmark_summary.md` | **Compact pass/fail summary** with strict acceptance criteria (≤1.05× comparable, ≤0.98× better). Lists the five curated clean-log reports with their cell-level results. |
| `clean_logs/attn_only_16x16_gpu0_20260413_report.txt` | Best attention-only run: **16/16 pass** on clean GPU 0. The only run achieving full matrix pass; used as capability evidence. |
| `clean_logs/vit_fp32_fullmodel_4x4_gpu4_20260421_report.txt` | ViT fp32 full-model **4/4 pass** on clean GPU 4. Strongest and most stable full-model result; used as headline evidence. Numbers: img224 fwd 0.777×lat/0.988×mem, img384 fwd_bwd 0.985×lat/0.711×mem. |
| `clean_logs/fullmodel_16cell_clean_gpu0_20260419_report.txt` | Full 16-cell full-model snapshot (5/16 pass) before the latest ViT fp32 training-path patch. Kept for traceability. |
| `clean_logs/vit_fp32_fullmodel_pair_gpu0_20260419_report.txt` | Pre-patch ViT fp32 pair rerun (1/2 pass). Kept for traceability. |
| `clean_logs/long_token_and_swin_w24_clean_gpu0_20260420_report.txt` | Long-token ViT fp32 forward pass at 16K and 36K tokens (paper trend), plus Swin W24 diagnostic. Forward-backward long baseline OOMed on 40GB GPU. |
| `reproducibility.md` | Environment setup, benchmark discipline, and exact CLI commands for reproducing all five clean-log reports. |

### From original ELSA-main

| File | Purpose |
|------|---------|
| `FULL_REPORT_20260301.md` | Complete historical benchmark results from the March 2026 release candidate. |
| `FULL_REPORT_20260301.zh-TW.md` | Traditional Chinese version of the full report. |
| `RELEASE_NOTES_20260301.md` | Key changes and stability fixes in the March 2026 release. |
| `RELEASE_NOTES_20260301.zh-TW.md` | Traditional Chinese release notes. |
| `STATUS_MATRIX_20260301.md` | Per-configuration (model × dtype × direction) support and stability status as of March 2026. |
| `STATUS_MATRIX_20260301.zh-TW.md` | Traditional Chinese status matrix. |
| `REPRODUCIBILITY.md` | Original reproduction guide for the March 2026 benchmark set. |
| `REPRODUCIBILITY.zh-TW.md` | Traditional Chinese reproduction guide. |

---

## `validation/` — Quick smoke-check outputs

| File | Purpose |
|------|---------|
| `VALIDATION_SUMMARY_20260301.md` | Narrative summary of the March 2026 validation pass/fail status for all three quick checks. |
| `VALIDATION_SUMMARY_20260301.zh-TW.md` | Traditional Chinese version. |

---

## `other/` — Multilingual README files

| File | Purpose |
|------|---------|
| `README.zh-TW.md` | Traditional Chinese README. Note at top indicates English README is authoritative for latest features. |
| `README.zh-CN.md` | Simplified Chinese README. Same caveat applies. |
| `README.ja.md` | Japanese README. Same caveat applies. |
| `README.ko.md` | Korean README. Same caveat applies. |

---

## Key import paths

```python
# Recommended: install in editable mode
pip install -e .

# Then import from the stable package
from elsa import (
    ELSA_triton,           # FP16/BF16 kernel
    ELSA_triton_fp32,      # FP32 inference kernel
    ELSA_triton_fp32_train,# FP32 training kernel (with backward)
    ELSA_pytorch,          # Pure-PyTorch fallback
    ElsaAttention,         # Drop-in nn.Module
    ElsaViT,               # Full ViT model
    set_default_elsa_backend,
    patch_vit_attention,
    patch_swin_attention,
)

# Strict reference (correctness baseline)
from elsa.elsa_strict_ref import ElsaStrictState, merge_states
```
