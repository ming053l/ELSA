#!/usr/bin/env bash
# Full 14-cell ELSA coverage matrix on clean, gated GPUs.
# Cells: attn-only {fp32 fwd vs math/mem, fp16 fwd vs flash, fp32 bwd vs math/mem, fp16 bwd vs flash}
#      + full-model fwd (CUDA-graph) {vit,swin}x{fp16,fp32}
#      + full-model training (fwd+bwd min) {vit,swin}x{fp16,fp32}.
# Each cell runs via bench_clean.sh (clean-GPU gate + loading logged); retries for clean windows.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/matrix_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
VIT=vit_tiny_patch16_224.augreg_in21k_ft_in1k
SWIN=swin_tiny_patch4_window7_224.ms_in1k
HD="--heads 8 --dim 64"
runc() { local name="$1"; shift; local rc=3 a=0
  echo "#### CELL $name $(date -u +%H:%M:%S)"
  while [ "$rc" -eq 3 ] && [ "$a" -lt 60 ]; do a=$((a+1))
    scripts/bench_clean.sh "$OUT/${name}.log" "$@"; rc=$?
    [ "$rc" -eq 3 ] && { echo "  [$name] no clean gpu $a/60"; sleep 60; }
  done; echo "CELL_DONE $name rc=$rc"; }

runc attn_fp32_fwd_math  benchmarks/bench_attn.py --dtype fp32 --baseline math  $HD --seq 1024 4096 8192 16384 --warmup 5 --iters 20
runc attn_fp32_fwd_mem   benchmarks/bench_attn.py --dtype fp32 --baseline mem   $HD --seq 4096 8192 16384 32768 65536 --warmup 5 --iters 20
runc attn_fp16_fwd_flash benchmarks/bench_attn.py --dtype fp16 --baseline flash $HD --seq 1024 4096 16384 65536 --warmup 5 --iters 20
runc attn_fp32_bwd_math  benchmarks/bench_bwd.py  --dtype fp32 --baseline math  $HD --seq 2048 4096 8192 --warmup 4 --iters 10
runc attn_fp32_bwd_mem   benchmarks/bench_bwd.py  --dtype fp32 --baseline mem   $HD --seq 4096 8192 16384 32768 --warmup 4 --iters 10
runc attn_fp16_bwd_flash benchmarks/bench_bwd.py  --dtype fp16 --baseline flash $HD --seq 8192 16384 32768 --warmup 4 --iters 10
runc vit_fp16_fwd  benchmarks/full_model_graph.py --model "$VIT"  --dtype fp16 --batch 8 --image-size 224 512 1024 1536 2048 --iters 25
runc vit_fp32_fwd  benchmarks/full_model_graph.py --model "$VIT"  --dtype fp32 --batch 8 --image-size 224 512 768 1024 --iters 30
runc swin_fp16_fwd benchmarks/full_model_graph.py --model "$SWIN" --dtype fp16 --batch 8 --image-size 224 384 512 --iters 40
runc swin_fp32_fwd benchmarks/full_model_graph.py --model "$SWIN" --dtype fp32 --batch 8 --image-size 224 384 512 --iters 30
runc vit_fp16_train  benchmarks/full_model_bwd_min.py --model "$VIT"  --dtype fp16 --batch 8 --image-size 224 384 512 --iters 30
runc vit_fp32_train  benchmarks/full_model_bwd_min.py --model "$VIT"  --dtype fp32 --batch 8 --image-size 224 512 768 --iters 25
runc swin_fp16_train benchmarks/full_model_bwd_min.py --model "$SWIN" --dtype fp16 --batch 8 --image-size 224 384 512 --iters 25
runc swin_fp32_train benchmarks/full_model_bwd_min.py --model "$SWIN" --dtype fp32 --batch 8 --image-size 224 384 512 --iters 25
echo "MATRIX ALL DONE -> $OUT"
