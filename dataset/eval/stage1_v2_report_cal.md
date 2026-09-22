# Stage 1 v2 evaluation

Scope: **cal_split_only**. Test results are not used for tuning.

- legacy normal candidate rate: 0.6041666666666666
- v2 normal candidate rate: 0.07738095238095238
- legacy attack recall: 0.8690476190476191
- v2 attack recall: 0.6944444444444444
- ablation keys: ['full', 'no_cohort', 'no_evidence', 'no_progress_log']
- full vs no_evidence |Δ attack_recall|: 0.003968253968253954
- holdout attack recall: None
