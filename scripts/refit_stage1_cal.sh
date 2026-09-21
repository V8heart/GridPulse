#!/usr/bin/env bash
# Stage1 refit after code changes. Cal only for alpha; test once at end.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"

echo "[1/6] regenerate synthetic with progress-log policy"
"$PY" dataset/build_synthetic.py \
  --output-dir dataset/synthetic \
  --sessions-per-class 20 \
  --seed 7 \
  --nvml-avg-window-s 1.0 \
  --progress-log-drop-prob 0.4

echo "[2/6] fit baseline + Mondrian calibration"
"$PY" pipeline/fit_stage1_v2.py

echo "[3/6] cal alpha sweep"
"$PY" pipeline/eval_stage1_auc_alpha.py

echo "[4/6] cal evaluation report"
"$PY" pipeline/eval_stage1.py --split cal \
  --out dataset/eval/stage1_v2_report_cal.json

echo "[5/6] detailed cal diagnostics"
"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.baseline import CohortBaseline
from pipeline.stage1_v2 import build_windows, load_config, score_window, impact_components

root = Path('.')
cfg = load_config(root/'config/stage1_v2.yaml')
cal = json.loads((root/'dataset/eval/stage1_v2_calibration.json').read_text())
# freeze selected alpha from sweep if present
sweep = json.loads((root/'dataset/eval/stage1_alpha_sweep.json').read_text()) if (root/'dataset/eval/stage1_alpha_sweep.json').exists() else {}
selected = (sweep.get('selected') or (cal.get('config') or {}).get('alpha_selection', {}).get('selected'))
if selected and 'alpha' in selected:
    cfg['alpha'] = selected['alpha']
    cfg['alpha_high_impact'] = selected.get('alpha_high_impact', min(1.0, selected['alpha']*2))
if cal.get('evidence_surprisal_weights'):
    cfg['evidence_weights'] = {**cfg.get('evidence_weights', {}), **cal['evidence_surprisal_weights']}
base = CohortBaseline.load(root/'dataset/eval/cohort_baseline_v2.json')
man = json.loads((root/'dataset/synthetic/split_manifest.json').read_text())
df = pd.read_csv(root/'dataset/synthetic/all_v3.csv', low_memory=False)
sub = df[df['session_id'].astype(str).isin(set(map(str, man['cal'])))]
w = build_windows(sub, window_s=30, stride_s=15, progress_log_dir=root/'dataset/synthetic/steps', config=cfg)
rows = []
for _, row in w.iterrows():
    data = row.to_dict()
    full = score_window(data, base, cal, cfg, profile='full')
    none = score_window(data, base, cal, cfg, profile='no_evidence')
    rows.append({
        'gt_label': str(row['gt_label']),
        'family': str(row.get('declared_job_family')),
        'is_attack': not str(row['gt_label']).startswith('normal'),
        'cand_full': full['is_candidate'],
        'cand_none': none['is_candidate'],
        'max_abs_z': full['max_abs_z'],
        'driving': full.get('driving_feature'),
        'impact_raw': full['impact_raw'],
        'multi_gpu_sync_index': float(data.get('multi_gpu_sync_index') or 0.0),
        'cross_job_sync_index': float(data.get('cross_job_sync_index') or 0.0),
        'sync_applicable': bool(data.get('sync_applicable')),
    })
frame = pd.DataFrame(rows)
report = {
    'alpha': cfg.get('alpha'),
    'alpha_high_impact': cfg.get('alpha_high_impact'),
    'by_family': {},
    'evidence_ablation': {
        'full_attack_recall': float(frame.loc[frame.is_attack, 'cand_full'].mean()) if frame.is_attack.any() else None,
        'no_evidence_attack_recall': float(frame.loc[frame.is_attack, 'cand_none'].mean()) if frame.is_attack.any() else None,
        'delta_pp': None,
    },
    'max_abs_z': {
        'p50': float(frame['max_abs_z'].quantile(0.50)),
        'p99': float(frame['max_abs_z'].quantile(0.99)),
        'max': float(frame['max_abs_z'].max()),
        'top_driving_features': frame['driving'].value_counts().head(8).to_dict(),
    },
    'impact_raw_by_label': {},
    'sync': {
        'n_applicable': int(frame['sync_applicable'].sum()),
        'multi_gpu_gt_0_rate': float((frame['multi_gpu_sync_index'] > 0).mean()),
        'cross_job_gt_0_rate': float((frame['cross_job_sync_index'] > 0).mean()),
    },
}
for fam, g in frame.groupby('family'):
    n = g[~g.is_attack]
    a = g[g.is_attack]
    report['by_family'][fam] = {
        'normal_candidate_rate': float(n['cand_full'].mean()) if len(n) else None,
        'attack_recall': float(a['cand_full'].mean()) if len(a) else None,
        'n_normal': int(len(n)),
        'n_attack': int(len(a)),
    }
for label, g in frame.groupby('gt_label'):
    report['impact_raw_by_label'][label] = {
        'p50': float(g['impact_raw'].quantile(0.50)),
        'p90': float(g['impact_raw'].quantile(0.90)),
        'p99': float(g['impact_raw'].quantile(0.99)),
        'n': int(len(g)),
    }
fr = report['evidence_ablation']['full_attack_recall']
nr = report['evidence_ablation']['no_evidence_attack_recall']
if fr is not None and nr is not None:
    report['evidence_ablation']['delta_pp'] = float(fr - nr)
out = root/'dataset/eval/stage1_cal_diagnostics.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "[6/6] unit tests"
"$PY" -m pytest -q
echo "DONE cal refit. Run test split only after reviewing cal diagnostics."
