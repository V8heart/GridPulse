# Feature separability (train+cal)

windows=2058 normal=1176 attack=882

## Class sample sizes

| label | n_sessions | n_windows | verdict |
|-------|------------|-----------|---------|
| cryptojacking | 14 | 98 | ok |
| ltma | 14 | 98 | ok |
| normal_baseline | 14 | 98 | ok |
| normal_checkpoint | 14 | 98 | ok |
| normal_dataloader_stall | 14 | 98 | ok |
| normal_distributed_training | 14 | 98 | ok |
| normal_eval_train_switch | 14 | 98 | ok |
| normal_flat_pretrain | 14 | 98 | ok |
| normal_fsdp_deep_trough | 14 | 98 | ok |
| normal_hpo_search | 14 | 98 | ok |
| normal_inference_bursty | 14 | 98 | ok |
| normal_mixed_tenants | 14 | 196 | ok |
| normal_sync_ddp | 14 | 98 | ok |
| swma | 84 | 686 | ok |

## Features

| feature | normal_q50 | normal_q95 | auc_overall | nunique | decision |
|---------|------------|------------|-------------|---------|----------|
| high_load_fraction | 0.45 | 1.0 | 0.5577185239175035 | 276 | disable_auc_lt_0.6 |
| longest_high_seconds | 3.3 | 30.0 | 0.41661460502568376 | 147 | disable_auc_lt_0.6 |
| swing_abs_w | 136.55442285724698 | 307.4428727032688 | 0.5179516250944822 | 1999 | disable_auc_lt_0.6 |
| mean_w | 182.91977214726163 | 413.0546933756743 | 0.4656007527727644 | 2002 | disable_auc_lt_0.6 |
| ramp_p95_w_per_s | 141.10039737344158 | 330.77933735782045 | 0.5730608002838323 | 1956 | disable_auc_lt_0.6 |
| util_residual_mad_w | 1.9312288914827818 | 6.8192332869201735 | 0.2716441451864192 | 1921 | disable_auc_lt_0.6 |
| dominant_peak_prominence | 64307.13829747442 | 485074.03553837846 | 0.5550628981751431 | 2002 | disable_auc_lt_0.6 |
| cross_job_sync_index | 0.0 | 0.0 | None | 1 | exclude_constant |
| declared_family_mean_abs_z | 0.8187566796720384 | 2.509415901064057 | 0.8967559813040863 | 2002 | keep |
| best_other_family_mean_abs_z | 1.2277532071690502 | 3.517267981994097 | 0.7219802320020979 | 2002 | keep |

## Per-attack-class AUC (deferred if sessions < 3)

### high_load_fraction

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 1.0 | 14 | 98 | 0.9523809523809523 | ok |
| ltma | 0.005 | 14 | 98 | 0.20237661391087047 | ok |
| swma | 0.45 | 84 | 686 | 0.5521013069951013 | ok |

### longest_high_seconds

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 30.0 | 14 | 98 | 0.9523809523809523 | ok |
| ltma | 0.15000000000000002 | 14 | 98 | 0.2318348257670415 | ok |
| swma | 0.9 | 84 | 686 | 0.36647366672616566 | ok |

### swing_abs_w

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 3.8814727827593742 | 14 | 98 | 0.06274295432458697 | ok |
| ltma | 146.12806956965102 | 14 | 98 | 0.5358791475773983 | ok |
| swma | 238.69291635046702 | 84 | 686 | 0.5804203605640507 | ok |

### mean_w

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 285.0878733172219 | 14 | 98 | 0.6776516729140636 | ok |
| ltma | 120.33718686790522 | 14 | 98 | 0.07853498542274052 | ok |
| swma | 184.1224326944178 | 84 | 686 | 0.4906028738025823 | ok |

### ramp_p95_w_per_s

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 3.79206981461408 | 14 | 98 | 0.027453838678328474 | ok |
| ltma | 125.06780427494263 | 14 | 98 | 0.3773514507843954 | ok |
| swma | 297.5632930264997 | 84 | 686 | 0.678963130441681 | ok |

### util_residual_mad_w

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 0.7843751380254105 | 14 | 98 | 0.2431452172705817 | ok |
| ltma | 14.084769669089336 | 14 | 98 | 0.9999652922393447 | ok |
| swma | 0.6702605785423934 | 84 | 686 | 0.17166954245254953 | ok |

### dominant_peak_prominence

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 71710.6845848275 | 14 | 98 | 0.5215188116062751 | ok |
| ltma | 156327.6962778925 | 14 | 98 | 0.7438306955435235 | ok |
| swma | 58848.5789684334 | 84 | 686 | 0.5328880823466413 | ok |

### cross_job_sync_index

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 0.0 | 14 | 98 | None | ok |
| ltma | 0.0 | 14 | 98 | None | ok |
| swma | 0.0 | 84 | 686 | None | ok |

### declared_family_mean_abs_z

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 1.3664429415025432 | 14 | 98 | 0.779701166180758 | ok |
| ltma | 3.7216356389813288 | 14 | 98 | 0.9290573372206026 | ok |
| swma | 3.4049022205730184 | 84 | 686 | 0.9088636183336308 | ok |

### best_other_family_mean_abs_z

| attack_label | q50 | n_sessions | n_windows | auc | verdict |
|--------------|-----|------------|-----------|-----|---------|
| cryptojacking | 1.4471908356194936 | 14 | 98 | 0.5222303206997084 | ok |
| ltma | 1.5674646989168957 | 14 | 98 | 0.7384076079411357 | ok |
| swma | 1.8644728928304508 | 84 | 686 | 0.7481691656254339 | ok |
