# Stage 1 v2 evaluation

Scope: **test_split_only**. Test results are not used for tuning.

- legacy normal candidate rate: 0.6111111111111112
- v2 normal candidate rate: 0.09126984126984126
- legacy attack recall: 0.8716707021791767
- v2 attack recall: 0.6440677966101694
- ablation keys: ['full', 'no_cohort', 'no_evidence', 'no_progress_log']
- full vs no_evidence |Δ attack_recall|: 0.0581113801452785
- holdout attack recall: 0.6
