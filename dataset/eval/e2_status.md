# E2 status

## Gate results (train+cal only)

| check | result |
|-------|--------|
| Separability / constant exclusions | `cross_job_sync` excluded (index never computed). `sustained_high_load`, `flat_power`, `high_ramp`, `util_power_decoupled` disabled (feature AUC < 0.6). |
| Active evidence after attack>normal gate | **1** (`progress_log_missing` only) |
| Active ≥ 4 gate | **FAIL** → stop before E3 |
| Cal ablation \|Δ attack_recall\| full vs no_evidence | **0.0** with only `progress_log_missing` weighted |
| Quantile loosening | **not done** (per plan) |
| Alpha reselection rule | unchanged: `cal normal candidate_rate <= 0.10, maximize attack recall; tie -> smaller alpha` → α=0.10 |
| Corpus | untouched |

## Interpretation

Synthetic class-conditional AUC for power/util features is weak overall. After excluding non-separating axes and refusing to loosen thresholds when attack firing ≤ normal firing, only `progress_log_missing` remains as a Stage1 evidence weight.

That is below the **4 active evidence** stop line. E2 therefore **does not claim success** for Stage2 corpus matching. Next work needs new discriminative axes and/or real capture + step logs (not corpus edits yet).

## Artifacts

- `dataset/eval/feature_separability.md`
- `dataset/eval/evidence_activation.json`
- `dataset/eval/evidence_firing_counts.md` (E1_before / E1_after / E2_after)
- `dataset/eval/stage1_alpha_sweep.md`
- `dataset/eval/stage1_v2_report_cal.json`
- `config/stage1_v2.yaml` (thresholds + weights + alpha)
- `dataset/eval/e2_status.md` (this file)
