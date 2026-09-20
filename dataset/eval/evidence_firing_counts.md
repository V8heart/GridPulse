# Evidence firing counts

E2.5 after sticky-bool fix, log-prominence tau, flat_power redefine, class-wise gate. train+cal only.

**E2.5 gate:** n_active=6, threshold_based=4 → stop_before_e3=False (see `e25_status.md`).

| evidence | E1_before | E1_after | E2_after | normal_rate | attack_rate |
|----------|-----------|----------|----------|-------------|-------------|
| cross_job_sync | 0 | 0 | 0 | 0.0000 | 0.0000 |
| declared_family_mismatch | 0 | 0 | 435 | 0.0136 | 0.4751 |
| flat_power | 0 | 111 | 98 | 0.0833 | 0.0000 |
| high_ramp | 0 | 0 | 59 | 0.0502 | 0.0000 |
| period_match | 146 | 522 | 522 | 0.2832 | 0.2143 |
| period_mismatch | 108 | 262 | 344 | 0.0417 | 0.3345 |
| progress_log_available | 260 | 1078 | 1078 | 0.6667 | 0.3333 |
| progress_log_missing | 315 | 980 | 980 | 0.3333 | 0.6667 |
| strong_peak | 0 | 2058 | 354 | 0.0502 | 0.3345 |
| sustained_high_load | 0 | 224 | 224 | 0.0952 | 0.1270 |
| unexplained_changepoint | 0 | 1452 | 1452 | 0.7611 | 0.6315 |
| util_power_decoupled | 0 | 31 | 157 | 0.0502 | 0.1111 |
