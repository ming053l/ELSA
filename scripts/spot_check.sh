#!/usr/bin/env bash
# 16-point release spot check on a clean, gated GPU (retries for a clean window).
set -uo pipefail
cd "$(dirname "$0")/.."
rc=3 a=0
while [ "$rc" -eq 3 ] && [ "$a" -lt 60 ]; do a=$((a+1))
  scripts/bench_clean.sh "${1:-results/spot_check.log}" scripts/spot_check.py; rc=$?
  [ "$rc" -eq 3 ] && { echo "no clean gpu $a/60"; sleep 60; }
done
exit $rc
