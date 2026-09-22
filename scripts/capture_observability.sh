#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DURATION="${DURATION:-60}"
GPU_ID="${GPU_ID:-0}"
SESSIONS_ROOT="${SESSIONS_ROOT:-dataset/real}"
DRY_RUN="${DRY_RUN:-0}"

cd "$ROOT"

for workload in baseline swma; do
  for interval in 10 50 100 1000; do
    cmd=(
      "$PYTHON_BIN" -m dataset.run_capture
      --workload "$workload"
      --duration "$DURATION"
      --interval-ms "$interval"
      --gpu-id "$GPU_ID"
      --confirm-shared-gpu-safe
    )
    if [[ "$workload" == "swma" ]]; then
      cmd+=(--period 1 --duty-cycle 0.5)
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done
done

analysis=(
  "$PYTHON_BIN" -m dataset.observability "$SESSIONS_ROOT"
  --output dataset/real/observability/summary.md
)
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${analysis[@]}"
  printf '\n'
else
  "${analysis[@]}"
fi
