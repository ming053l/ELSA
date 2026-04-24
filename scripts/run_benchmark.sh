#!/usr/bin/env bash
# run_benchmark.sh — activate the ELSA conda environment, select a free GPU,
# then forward all arguments to the given command.
#
# Configuration (via environment variables):
#   ELSA_CONDA_ENV   Name of the conda environment to activate (default: elsa)
#   ELSA_CONDA_SH    Path to conda's init script (default: auto-detected)
#   ELSA_RESPECT_CUDA_VISIBLE_DEVICES
#                    Set to 1 to keep any CUDA_VISIBLE_DEVICES already set in
#                    the outer shell (default: 0, i.e., auto-select a free GPU)
set -eo pipefail

CONDA_ENV="${ELSA_CONDA_ENV:-elsa}"

# Locate the conda init script.
if [[ -n "${ELSA_CONDA_SH:-}" ]]; then
  CONDA_SH="${ELSA_CONDA_SH}"
else
  # Try conda info --base first, then fall back to common install paths.
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[run_benchmark] ERROR: Could not locate conda init script." >&2
    echo "  Set ELSA_CONDA_SH=/path/to/conda.sh and retry." >&2
    exit 1
  fi
fi

# conda activate may read unset vars; relax nounset temporarily.
set +u
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# GPU selection: by default, ignore any inherited CUDA_VISIBLE_DEVICES to
# avoid stale pinning from outer shells.
if [[ "${ELSA_RESPECT_CUDA_VISIBLE_DEVICES:-0}" != "1" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  pick_free_gpu() {
    # Prefer an A100 with low utilization and <1 GB used.
    local picked
    picked="$(
      nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' '
          /A100/ {
            uuid=$1; used=$3+0; util=$4+0
            if (used < 1024 && util < 10) { print uuid; exit }
          }' \
      | head -n 1
    )"
    if [[ -n "${picked}" ]]; then
      echo "${picked}"
      return
    fi
    # Fallback 1: any A100.
    picked="$(
      nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null \
      | awk -F', *' '/A100/ {print $1; exit}'
    )"
    if [[ -n "${picked}" ]]; then
      echo "${picked}"
      return
    fi
    # Fallback 2: first available GPU index.
    echo "0"
  }
  export CUDA_VISIBLE_DEVICES="$(pick_free_gpu)"
fi

echo "[run_benchmark] env=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2

exec "$@"
