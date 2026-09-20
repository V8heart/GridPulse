# Stage1 alpha sweep (cal only)

cal u_score AUC overall: **0.8496728552532123**

selected: `{'alpha': 0.1, 'cal_normal_candidate_rate': 0.09523809523809523, 'cal_attack_recall': 0.4126984126984127}`

| alpha | cal_normal_candidate_rate | cal_attack_recall |
|-------|---------------------------|-------------------|
| 0.05 | 0.044642857142857144 | 0.3492063492063492 |
| 0.1 | 0.09523809523809523 | 0.4126984126984127 |
| 0.2 | 0.19940476190476192 | 0.7738095238095238 |

## cal u_score AUC by attack label

| label | auc | n | root_cause |
|-------|-----|---|------------|
| cryptojacking | 0.7596726190476191 | 28 | alpha_may_help |
| ltma | 0.9140093537414966 | 28 | alpha_may_help |
| swma | 0.8533391034985423 | 196 | alpha_may_help |

Note: Under conformal scoring, FPR often tracks alpha; low-AUC classes need feature/evidence work, not alpha alone.
