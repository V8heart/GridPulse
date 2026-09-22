#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DRY_RUN=0
CONFIRMED=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --confirm-shared-gpu-safe) CONFIRMED=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$DRY_RUN" != "1" && "$CONFIRMED" != "1" ]]; then
  echo "--confirm-shared-gpu-safe is required for GPU smoke" >&2
  exit 2
fi

cd "$ROOT"
commands=(
  "$PYTHON_BIN -m dataset.run_capture --workload gpt_tiny_finetune --duration 120 --warmup-s 0 --confirm-shared-gpu-safe"
  "$PYTHON_BIN -m dataset.run_capture --workload swma --duration 120 --warmup-s 0 --period 1 --duty-cycle 0.5 --confirm-shared-gpu-safe"
)

for command in "${commands[@]}"; do
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$command"
  else
    read -r -a argv <<<"$command"
    "${argv[@]}"
  fi
done
