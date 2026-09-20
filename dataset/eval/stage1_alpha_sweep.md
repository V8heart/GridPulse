# Stage1 alpha sweep (cal only)

cal u_score AUC overall: **0.8524718915343915**

selected: `{'alpha': 0.1, 'cal_normal_candidate_rate': 0.09523809523809523, 'cal_attack_recall': 0.4603174603174603}`

| alpha | cal_normal_candidate_rate | cal_attack_recall |
|-------|---------------------------|-------------------|
| 0.05 | 0.044642857142857144 | 0.376984126984127 |
| 0.1 | 0.09523809523809523 | 0.4603174603174603 |
| 0.2 | 0.19642857142857142 | 0.7222222222222222 |

## cal u_score AUC by attack label

| label | auc | n | root_cause |
|-------|-----|---|------------|
| cryptojacking | 0.8321641156462585 | 28 | alpha_may_help |
| ltma | 0.9168792517006803 | 28 | alpha_may_help |
| swma | 0.8461719509232264 | 196 | alpha_may_help |

Note: Under conformal scoring, FPR often tracks alpha; low-AUC classes need feature/evidence work, not alpha alone.
