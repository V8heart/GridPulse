# E2.5 status

evidence_gate: `{'prune_auc_min': 0.6, 'activate_attack_rate_min': 0.2, 'activate_rate_ratio_min': 2.0}`  
(operating points from `config/stage1_v2.yaml`; not retuned from these results)

- n_active=6 (need ≥4): **PASS**
- n_threshold_based=4 (need ≥2): **PASS** (`strong_peak`, `sustained_high_load`, `util_power_decoupled`, `declared_family_mismatch`)
- stop_before_e3: **False** → E3 is allowed by this gate
- cal ablation `|Δ attack_recall|` full vs no_evidence: **0.0437** (>1%p)

## Active evidence

| evidence | activating_classes | direction | best_auc |
|----------|-------------------|-----------|----------|
| period_mismatch | swma | — | — |
| progress_log_missing | cryptojacking, swma | — | — |
| strong_peak | swma | higher_is_attack | 0.744 |
| sustained_high_load | cryptojacking | higher_is_attack | 0.952 |
| util_power_decoupled | ltma | higher_is_attack | 0.99997 |
| declared_family_mismatch | cryptojacking, ltma, swma | higher_is_attack | 0.929 |

## Inactive (recorded)

- `flat_power`, `high_ramp`, `unexplained_changepoint`, `cross_job_sync` — see `evidence_activation.json` per-class rates
- SWMA `high_ramp` recovery was **not** a success criterion; it did not activate

## Fixes applied

- Sticky bool strip: `BUILD_DEFERRED_BOOL_KEYS` / `SCORE_RECOMPUTE_BOOL_KEYS`
- `declared_family_mismatch` now fires (435/2058 on train+cal)
- `dominant_peak_prominence` → log10; `tau_peak`≈5.69 on normal q95
- `flat_power` = low swing ∧ low ramp ∧ high mean (still inactive under bool gate)
- `progress_log_missing` co-fires with another active evidence (+ training family gate)

## Artifacts

- `dataset/eval/feature_separability.md` / `.json`
- `dataset/eval/evidence_activation.json`
- `dataset/eval/evidence_firing_counts.md` (E1_before / E1_after / E2.5 column as E2_after)
- `dataset/eval/stage1_v2_report_cal.json`
- `config/stage1_v2.yaml`
