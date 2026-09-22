# Stage1 alpha sweep (cal only)

cal u_score AUC overall: **0.8524896069538926**

selected: `{'alpha': 0.1, 'cal_normal_candidate_rate': 0.07738095238095238, 'cal_attack_recall': 0.6944444444444444}`

| alpha | cal_normal_candidate_rate | cal_attack_recall |
|-------|---------------------------|-------------------|
| 0.05 | 0.026785714285714284 | 0.5753968253968254 |
| 0.1 | 0.07738095238095238 | 0.6944444444444444 |
| 0.2 | 0.19047619047619047 | 0.7023809523809523 |

## cal u_score AUC by attack label

| label | auc | n | root_cause |
|-------|-----|---|------------|
| cryptojacking | 0.9007227891156463 | 28 | alpha_may_help |
| ltma | 0.9466411564625851 | 28 | alpha_may_help |
| swma | 0.8321489310009719 | 196 | alpha_may_help |

Note: Under conformal scoring, FPR often tracks alpha; low-AUC classes need feature/evidence work, not alpha alone.
