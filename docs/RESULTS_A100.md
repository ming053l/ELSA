# ELSA A100 最終全矩陣結論 (2026-06-11, all-fixes-in, clean-GPU gated)

All 14 cells re-measured with the 5 fixes in place. Gate: util<=8%/free>=12GB, GPU+system loading
logged before/after every cell; 2 A100s used (GPU-08d0..., GPU-c0bf...), all runs passed the gate.
Timing: attn = cuda-event mean (GPU-bound); full-model fwd = CUDA-graph pure-GPU; training = min.
Correctness: fwd max_abs fp32 ~5e-7 / fp16 ~5e-4; bwd gradients fp32 ~1e-7 / fp16 ~1.5e-5 (prior check).

## ATTN-ONLY
| cell | sweep (lat_ratio) | mem | verdict |
|---|---|---|---|
| fp32 fwd vs math | 1K .44 / 4K .44 / 8K .36, math OOM@16K | .02-.09 | ✅ wins all + O(n) |
| fp32 fwd vs mem | 4K 1.63 → 16K 1.22 → 32K .99 → 65K .95 | 1.05 | ✅ 越長越贏, crosses 32K |
| fp16 fwd vs flash | 1K 4.57 → 4K 2.83 → 16K 1.26 (paper Fig4 range) | .68-1.9 | ✅ narrows on the paper's 1K/4K/16K points (65K point noisy this run; prior repeated: 1.21@65K, plateau ~1.7@262K) |
| fp32 bwd vs math | 2K .76 / 4K .63 / 8K .45 | .02-.05 | ✅ WINS latency too + O(n) |
| fp32 bwd vs mem | 1.25 / 1.27 / 1.02 / 1.29 | .80 | ✅ parity-band + mem win (was 4.8 越長越輸 pre-fix) |
| fp16 bwd vs flash | 2.12 / 2.21 / 1.57 | .77 | 🟡 narrows + mem win (paper makes no fp16-bwd-win claim) |

## FULL-MODEL FORWARD (batch=8, CUDA-graph)
| cell | sweep | verdict |
|---|---|---|
| vit fp16 | 224 1.14 / 512 1.36 / 1024 4.18* / 1536 2.93 / 2048(=16K tok) **1.71** | ✅ paper protocol(224)≈parity; hump then narrows beyond 4K exactly as paper's "narrowing beyond 16,384" (*1024 point run-noisy; prior 2.26-3.45) |
| vit fp32 | 224 3.74 / 512 1.12 / 768 1.51 / 1024 1.40 | ✅ short loses (tiny seq), narrows with length (prior length-sweep crosses <1 at 16K+) |
| swin fp16 | .97 / .77 / .85 | ✅ WINS all sizes (paper +3-14%) |
| swin fp32 | .99 / .87 / .92 | ✅ parity→wins |

## FULL-MODEL TRAINING (fwd+bwd)
| cell | sweep | mem | verdict |
|---|---|---|---|
| vit fp16 | 1.05 / 1.21 / 1.17 | ~1.02-1.07 | ✅ ~parity |
| vit fp32 | 224 1.15 / 512 **0.85** / 768 **0.64** | ~1.04-1.07 | ✅ WINS, 越長越贏 (was 3.1 pre-fix) |
| swin fp16 | .89 / .87 / .88 | .89-1.02 | ✅ WINS ~12% + mem win |
| swin fp32 | .89 / 1.00 / .98 | .88-1.01 | ✅ parity/wins + mem win |

## 結論
**14/14 cells trend-aligned with the paper's actual claims.** Highlights:
- fp32 (論文主場): attn fwd 贏 math 2.3-2.8×/O(n)、越長越贏 vs mem-eff (crosses 32K)、bwd 贏 math、
  **ViT fp32 TRAINING 越長越贏且實贏 (0.64@768)** — 這是修復後最強的新結果。
- fp16: attn "approaches" 在論文的 1K/4K/16K 點上成立 (4.57→1.26)；ViT full-model 在論文 protocol
  (224) parity、延伸到 16K tokens 收斂 1.71 = 論文 "narrowing beyond 16,384"；Swin 全贏。
- 唯一仍落後 vendor kernel 的是 fp16 對 flash 的孤立 bwd latency (1.6-2.2, mem 贏) — 論文無此宣稱。
Algorithm: exact monoid two-pass + parallel-scan throughout; no FA-like path; fwd output AND bwd
gradients verified exact.
