#!/usr/bin/env bash
# Synthetic v2 regen + cal-only refit. Writes ONLY under dataset/eval/v2_regen/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/dataset/eval/v2_regen}"
RUN_TEST=0

for arg in "$@"; do
  case "$arg" in
    --run-test) RUN_TEST=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$ROOT"

if ! git rev-parse --verify stage1-mondrian-baseline >/dev/null 2>&1; then
  echo "missing git tag stage1-mondrian-baseline; seal current Mondrian baseline first" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
SYNTH_ROOT="$OUT_ROOT/synthetic"
mkdir -p "$SYNTH_ROOT"

"$PYTHON_BIN" -m dataset.build_synthetic \
  --sessions-per-class 20 \
  --seed 7 \
  --output-root "$SYNTH_ROOT"

"$PYTHON_BIN" -m dataset.audit_metadata_leakage \
  --telemetry "$SYNTH_ROOT/all_v3.csv" \
  --labels "$SYNTH_ROOT/index/labels.csv" \
  --out "$OUT_ROOT/metadata_leakage_audit.json"

"$PYTHON_BIN" -m pipeline.fit_stage1_v2 \
  --telemetry "$SYNTH_ROOT/all_v3.csv" \
  --split-manifest "$SYNTH_ROOT/split_manifest.json" \
  --sessions-root "$SYNTH_ROOT/sessions" \
  --labels "$SYNTH_ROOT/index/labels.csv" \
  --baseline-out "$OUT_ROOT/cohort_baseline_v2.json" \
  --calibration-out "$OUT_ROOT/stage1_v2_calibration.json" \
  --stage1-config config/stage1_v2.yaml

"$PYTHON_BIN" -m pipeline.eval_stage1 \
  --telemetry "$SYNTH_ROOT/all_v3.csv" \
  --split-manifest "$SYNTH_ROOT/split_manifest.json" \
  --sessions-root "$SYNTH_ROOT/sessions" \
  --labels "$SYNTH_ROOT/index/labels.csv" \
  --baseline-model "$OUT_ROOT/cohort_baseline_v2.json" \
  --calibration "$OUT_ROOT/stage1_v2_calibration.json" \
  --stage1-config config/stage1_v2.yaml \
  --split cal \
  --out "$OUT_ROOT/stage1_v2_report_cal.json"

"$PYTHON_BIN" -m pipeline.report_stratified \
  --input "$OUT_ROOT/stage1_v2_report_cal.json" \
  --split cal \
  --out-dir "$OUT_ROOT"

if [[ "$RUN_TEST" == "1" ]]; then
  "$PYTHON_BIN" -m pipeline.eval_stage1 \
    --telemetry "$SYNTH_ROOT/all_v3.csv" \
    --split-manifest "$SYNTH_ROOT/split_manifest.json" \
    --sessions-root "$SYNTH_ROOT/sessions" \
    --labels "$SYNTH_ROOT/index/labels.csv" \
    --baseline-model "$OUT_ROOT/cohort_baseline_v2.json" \
    --calibration "$OUT_ROOT/stage1_v2_calibration.json" \
    --stage1-config config/stage1_v2.yaml \
    --split test \
    --out "$OUT_ROOT/stage1_v2_report_test.json"
  echo "test 2회차: 데이터 재생성에 따른 재평가, 채점 로직 무변경" >>"$OUT_ROOT/synthetic_regen_comparison.md"
else
  echo "cal freeze complete; re-run with --run-test after confirming freeze" >&2
fi
