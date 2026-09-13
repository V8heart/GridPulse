#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DURATION="${DURATION:-60}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-dataset/real/observability}"
CONFIRM_FLAG="${CONFIRM_FLAG:---confirm-shared-gpu-safe}"

mkdir -p "$OUT_DIR"
inputs=()

for interval in 10 50 100 1000; do
  idle_out="${OUT_DIR}/idle-${interval}ms.csv"
  "${PYTHON_BIN}" -m bit2watt_impl.collect_telemetry \
    --output "$idle_out" \
    --duration "$DURATION" \
    --interval-ms "$interval" \
    --gpu-ids "$GPU_ID" \
    --session-id "observability-idle-${interval}ms" \
    --label normal_baseline \
    --job-type observability_idle \
    --gres-req gpu:1
  inputs+=("$idle_out")
done

for interval in 10 50 100 1000; do
  load_out="${OUT_DIR}/swma-${interval}ms.csv"
  "${PYTHON_BIN}" -m dataset.run_capture \
    --workload swma \
    --duration "$DURATION" \
    --interval-ms "$interval" \
    --gpu-id "$GPU_ID" \
    --session-id "observability-swma-${interval}ms" \
    --output "$load_out" \
    --period 1 \
    --duty-cycle 0.5 \
    ${CONFIRM_FLAG}
  inputs+=("$load_out")
done

"${PYTHON_BIN}" dataset/observability.py "${inputs[@]}" \
  --output "${OUT_DIR}/summary.md"
