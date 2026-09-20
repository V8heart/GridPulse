# Feature separability (train+cal, E2.5)

evidence_gate: `{'prune_auc_min': 0.6, 'activate_attack_rate_min': 0.2, 'activate_rate_ratio_min': 2.0}`
windows=2058 normal=1176 attack=882

## Features

| feature | best_class | best_auc | best_auc_dir | direction | decision |
|---------|------------|----------|--------------|-----------|----------|
| high_load_fraction | cryptojacking | 0.9523809523809523 | 0.9523809523809523 | higher_is_attack | keep |
| longest_high_seconds | cryptojacking | 0.9523809523809523 | 0.9523809523809523 | higher_is_attack | keep |
| swing_abs_w | cryptojacking | 0.06274295432458697 | 0.937257045675413 | lower_is_attack | keep |
| mean_w | ltma | 0.07853498542274052 | 0.9214650145772595 | lower_is_attack | keep |
| ramp_p95_w_per_s | cryptojacking | 0.027453838678328474 | 0.9725461613216715 | lower_is_attack | keep |
| util_residual_mad_w | ltma | 0.9999652922393447 | 0.9999652922393447 | higher_is_attack | keep |
| dominant_peak_prominence_log | ltma | 0.7438306955435235 | 0.7438306955435235 | higher_is_attack | keep |
| cross_job_sync_index | None | None | None | None | exclude_constant |
| declared_family_mean_abs_z | ltma | 0.9290573372206026 | 0.9290573372206026 | higher_is_attack | keep |
| best_other_family_mean_abs_z | swma | 0.7481691656254339 | 0.7481691656254339 | higher_is_attack | keep |
