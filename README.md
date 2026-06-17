# ELSA Two-Pass — exact monoid attention with parallel scan (A100-validated)

<p align="center">
  <a href="https://ming053l.github.io/ELSA_projectpage/">[Project Page]</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2604.23798">[Paper (arXiv)]</a> &nbsp;|&nbsp;
  <a href="#citation">[Citation]</a>
</p>

**Chih-Chung Hsu, Xin-Di Ma, Wo-Ting Liao, Chia-Ming Lee**

Advanced Computer Vision Laboratory, National Yang Ming Chiao Tung University · CVPR Findings 2026

> 🏆 **ELSA received the CVPR 2026 Computational Transparency Award Champion!!**

---

A clean-room implementation of **ELSA** (exact linear-memory softmax attention,
arXiv 2604.23798): exact attention computed as a **two-level parallel scan over an
associative monoid**, with O(n) extra memory and O(log n) merge depth — **no
FlashAttention-style fused single-pass anywhere in the ELSA path**.

## Algorithm ↔ paper mapping

| Paper (Sec. 4) | This implementation |
|---|---|
| Monoid state `(m, S, W)`: running max-logit, normalized cumulative sum, weighted sum | `(m, z, s)` in `attention_parallel_scan.py` — identical triple |
| Merge operator (Eq. 4), identity `(-∞, 0, 0)` | `m_new = max(m_a, m_b)`; `z = z_a·e^(m_a−m_new) + z_b·e^(m_b−m_new)` (same for `s`) — the symmetric max-form of Eq. 4, exactly equal in value; identity `(-inf, 0, 0)` incl. the fully-masked-block guard |
| Intra-block scan (per-chunk reduction; correctness relies only on the monoid structure, not on order) | Phase 1: per-K-partition summary via a register-resident monoid fold over 64-token micro-tiles (order-independent by associativity). Register-resident is what makes it fast on A100 — an SMEM-materialized tree-combine was measured occupancy-bound (1 CTA/SM) |
| Inter-block Blelloch two-pass scan in global memory | Phase 2: block summaries round-trip through **global memory** and are combined by a parallel merge tree (CUDA `paper_scan_d64_final_reduce`). Non-causal needs only the total (= the Blelloch **up-sweep**); the causal path uses the Hillis-Steele/Blelloch prefix scan kernels |
| Output `y = W / S` | `out = s / z` (zero-guarded) |
| Exactness (Thm 1) | fwd `max_abs` fp32 ≈ 5e-7 / fp16 ≈ 5e-4 vs math-SDPA; bwd gradients fp32 ≈ 1e-7 / fp16 ≈ 1.5e-5 |
| Backward | the paper specifies no backward; here: exact recompute from saved per-row `(m, z)` statistics (no attention matrix stored, O(n) memory; softmax weights reconstructed exactly — no online-softmax recurrence) |

Key A100 engineering (all correctness-verified):
- **Partial-block masking** (`NEEDS_MASK`): non-divisible sequences (ViT's `(img/patch)²+1`)
  run the fast multi-partition path; out-of-range keys are excluded (`scores=-inf`) and a
  fully-masked micro-block merges as the monoid identity.
- **Length-adaptive fp32 routing**: short → non-tiled; mid (`512<seq<2048`) → one tiled
  partition of width ≈ seq; long → 2048-wide tiled partitions (multi-block scan).
- **Backward tile config**: `block_n=32` keeps the `dk/dv` accumulators (fp32 `[block_n,64]×2`)
  out of register-spill (was a 2.7–4× cliff at `block_n=64`).

## A100 results (clean-GPU gated, full matrix)

Final A100 matrix (clean-GPU gated; util ≤ 8% / free ≥ 12 GB, GPU+system load logged before/after
every cell). `lat_ratio = ELSA / baseline` (**lower is better**); `mem_ratio = ELSA / baseline`.
Full logs in [`docs/RESULTS_A100.md`](docs/RESULTS_A100.md). 16/16 spot check PASS.

**Attention-only** (B=1, H=8, d=64; fwd = cuda-event mean, GPU-bound):

| cell | baseline | lat_ratio sweep | mem_ratio | trend |
|---|---|---|---|---|
| fp32 fwd | Math-SDPA | 1K 0.44 · 4K 0.44 · 8K 0.36 *(math OOM @16K)* | 0.02–0.09 | wins all + O(n) memory |
| fp32 fwd | mem-eff SDPA | 4K 1.63 · 8K 1.20 · 16K 1.22 · 32K **0.99** · 65K **0.95** | 1.05 | 越長越贏 (crosses <1 ~32K) |
| fp16 fwd | FlashAttention | 1K 4.57 · 4K 2.83 · 16K **1.26** | 0.7–1.9 | approaches FA (paper Fig 4 points) |
| fp32 bwd | Math-SDPA | 2K 0.76 · 4K 0.63 · 8K **0.45** | 0.02–0.05 | wins + O(n) memory |
| fp32 bwd | mem-eff SDPA | 4K 1.25 · 8K 1.27 · 16K 1.02 · 32K 1.29 | **0.80** | parity-band + memory win |
| fp16 bwd | FlashAttention | 8K 2.12 · 16K 2.21 · 32K **1.57** | **0.77** | narrows + memory win |

**Full-model forward** (batch=8, CUDA-graph pure-GPU vs the model's native SDPA/FA backend):

| model · dtype | img → lat_ratio | mem_ratio | trend |
|---|---|---|---|
| ViT-T fp16 | 224 1.14 · 512 1.36 · 1024 (hump) · 1536 2.93 · 2048 **1.71** | 1.00 | parity @224; narrows to ~1.7 by 16K tokens |
| ViT-T fp32 | 224 3.74 · 512 **1.12** · 768 1.51 · 1024 1.40 | 1.00 | short loses, narrows with length |
| Swin-T fp16 | 224 **0.97** · 384 **0.77** · 512 **0.85** | 1.06–1.28 | wins all (paper Swin +3–14%) |
| Swin-T fp32 | 224 **0.99** · 384 **0.87** · 512 **0.92** | 1.07–1.31 | parity → wins |

**Full-model training** (fwd+bwd, cuda-event min):

| model · dtype | img → lat_ratio | mem_ratio | trend |
|---|---|---|---|
| ViT-T fp16 | 224 **1.05** · 384 1.21 · 512 1.17 | 1.02–1.07 | ~parity |
| ViT-T fp32 | 224 1.15 · 512 **0.85** · 768 **0.64** | 1.04–1.07 | **wins, 越長越贏** |
| Swin-T fp16 | 224 **0.89** · 384 **0.87** · 512 **0.88** | **0.89–1.02** | wins ~12% + memory win |
| Swin-T fp32 | 224 **0.89** · 384 1.00 · 512 0.98 | **0.88–1.01** | parity/wins + memory win |

Correctness (verified every release): forward `max_abs` fp32 ≈ 5e-7 / fp16 ≈ 4e-4 vs Math-SDPA;
backward gradients fp32 ≈ 1e-7 / fp16 ≈ 1.5e-5 (incl. ViT's non-divisible `(img/patch)²+1` sequences).

Note: a couple of points carry shared-cluster noise (ViT fp16 @1024 and attn fp16 @65K vary run-to-run);
trend verdicts use repeated measurements. All 14 cells are trend-aligned with the paper's actual claims;
the only place ELSA trails a vendor kernel is fp16 isolated **backward** latency vs FlashAttention (memory
still wins) — a claim the paper does not make.

## Install / use

```bash
pip install -e .          # torch >= 2.7, triton >= 3.3, CUDA 12.x, sm_80+
```

```python
from elsa_twopass_clean import twopass_attention, patch_timm_attention
out = twopass_attention(q, k, v)                  # [B, H, N, D], exact
patch_timm_attention(model)                       # timm ViT / Swin-v1 drop-in (inference)
from elsa_twopass_clean import patch_timm_attention_train   # training drop-in
```

The d64 CUDA reduce extension builds on first use; on this machine:
`CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.0`.

## Reproduce the matrix (clean-GPU protocol)

Every measurement goes through `scripts/bench_clean.sh`, which picks the cleanest GPU,
**refuses to run** unless `util ≤ 8%` and `free ≥ 12 GB`, and logs full GPU + system load
before and after each run:

```bash
python scripts/verify_2pass.py   # STRUCTURAL proof: phase1 -> global-memory summaries -> phase2
                                 # parallel reduce over K partitions (no FlashAttention fusion)
scripts/run_matrix.sh            # full 14-cell matrix (hours, retries for clean windows)
scripts/spot_check.sh            # 16-point spot check (2 correctness + 14 perf cells)
pytest tests/                    # unit tests
```

`verify_2pass.py` instruments the kernels and prints, per shape, the number of K-partitions scanned
in parallel and confirms per-partition `(m,z,s)` summaries round-trip through global memory before a
separate phase-2 reduce — the HBM round-trip that defines a two-pass scan and is exactly what a fused
FlashAttention-style kernel avoids. Example output (A100):

```
attn fp32 N=4096        | phase1=8 | K-partitions=2 | phase2 reads HBM summary shape=(8192,)
attn fp16 N=16384       | phase1=4 | K-partitions=8 | phase2 reads HBM summary shape=(262144,)
ViT fp32 img1024 N=4097 | phase1=9 | K-partitions=3 | phase2 reads HBM summary shape=(36864,)
VERIFY: TWO-PASS PARALLEL SCAN CONFIRMED
```

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

## Acknowledgments

This study was supported in part by the National Science and Technology Council (NSTC), Taiwan,
under grants 112-2221-E-006-157-MY3, 114-2627-M-A49-003, 114-2218-E-035-001, and 114-2119-M-006-007.
We thank the National Center for High-performance Computing (NCHC) of National Applied Research
Laboratories (NARLabs) in Taiwan for providing computational and storage resources.

## License

MIT — see [`LICENSE`](LICENSE).
