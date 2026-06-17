#!/usr/bin/env bash
# Clean-GPU measurement wrapper for ELSA benchmarks.
# - Picks the cleanest A100 (lowest util, then most free mem; excludes DGX Display).
# - REFUSES to run if the chosen GPU's util > MAX_UTIL or free mem < MIN_FREE_MIB.
# - Logs full GPU + system loading BEFORE and AFTER the run, alongside results.
# Usage: scripts/bench_clean.sh <logfile> <python-bench-cmd...>
set -uo pipefail
DGX_UUID="GPU-8156aaaf-501e-3339-d527-6891a9cc6df5"
MAX_UTIL="${MAX_UTIL:-8}"            # require <= this % util to call it "clean"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
LOG="$1"; shift
PYBIN="/home/jess/anaconda3/envs/sicnet/bin/python"

pick() {
  nvidia-smi --query-gpu=uuid,utilization.gpu,memory.free --format=csv,noheader \
   | awk -F', ' -v dgx="$DGX_UUID" -v minf="$MIN_FREE_MIB" '
      $1!=dgx { gsub(/ %| MiB/,"",$2); gsub(/ MiB/,"",$3);
                if ($3+0 >= minf) print $2+0, $3+0, $1 }' \
   | sort -k1,1n -k2,2nr | head -1
}
SEL="$(pick)"
UTIL="$(echo "$SEL" | awk '{print $1}')"
FREE="$(echo "$SEL" | awk '{print $2}')"
UUID="$(echo "$SEL" | awk '{print $3}')"

{
  echo "================ CLEAN-GPU RUN $(date -u +%Y-%m-%dT%H:%M:%SZ) ================"
  echo "chosen UUID=$UUID  util=${UTIL}%  free=${FREE}MiB  (gate: util<=${MAX_UTIL}%, free>=${MIN_FREE_MIB})"
  echo "--- system load --- $(uptime)"
  echo "--- GPU table BEFORE ---"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
  echo "--- compute apps on chosen GPU BEFORE ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$UUID" || echo "(none)"
} | tee -a "$LOG"

if [ -z "$UUID" ] || [ "${UTIL:-99}" -gt "$MAX_UTIL" ]; then
  echo "!! NO CLEAN GPU (best util=${UTIL}% free=${FREE}MiB > gate). ABORTING to avoid contaminated numbers." | tee -a "$LOG"
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda-12.2 TORCH_CUDA_ARCH_LIST=8.0
export PYTHONPATH="src" CUBLAS_WORKSPACE_CONFIG=":4096:8"
echo ">>> RUN on $UUID : $*" | tee -a "$LOG"
"$PYBIN" "$@" 2>&1 | grep -vE "llama-can|No checkpoint" | tee -a "$LOG"
RC=$?
{
  echo "--- GPU util on chosen GPU AFTER ---"
  nvidia-smi --query-gpu=uuid,utilization.gpu,memory.free --format=csv,noheader | grep "$UUID"
  echo "--- compute apps on chosen GPU AFTER ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$UUID" || echo "(none)"
  echo "<<< done rc=$RC"
} | tee -a "$LOG"
exit $RC
