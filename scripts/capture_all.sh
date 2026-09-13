#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DURATION="${DURATION:-120}"
INTERVAL_MS="${INTERVAL_MS:-100}"
GPU_ID="${GPU_ID:-0}"
CONFIRM_FLAG="${CONFIRM_FLAG:---confirm-shared-gpu-safe}"

run_capture() {
  local workload="$1"
  local repeat="$2"
  shift 2
  for idx in $(seq 1 "$repeat"); do
    "${PYTHON_BIN}" -m dataset.run_capture \
      --workload "$workload" \
      --duration "$DURATION" \
      --interval-ms "$INTERVAL_MS" \
      --gpu-id "$GPU_ID" \
      --session-id "real-${workload}-${idx}" \
      ${CONFIRM_FLAG} \
      "$@"
  done
}

run_capture baseline 3
run_capture hpo 3
run_capture checkpoint 3
run_capture dataloader_stall 3
run_capture eval_train_switch 3
run_capture distributed 3 --gres-req gpu:2
run_capture swma 3 --period 1 --duty-cycle 0.5
run_capture swma_multi 3 --period 1 --duty-cycle 0.5 --gres-req gpu:2
run_capture ltma 3
run_capture cryptojacking 3
