# Stage1 alpha sweep (cal only)

cal u_score AUC overall: **0.8406852324263039**

selected: `{'alpha': 0.1, 'cal_normal_candidate_rate': 0.09523809523809523, 'cal_attack_recall': 0.4246031746031746}`

| alpha | cal_normal_candidate_rate | cal_attack_recall |
|-------|---------------------------|-------------------|
| 0.05 | 0.044642857142857144 | 0.3531746031746032 |
| 0.1 | 0.09523809523809523 | 0.4246031746031746 |
| 0.2 | 0.20238095238095238 | 0.7261904761904762 |

## cal u_score AUC by attack label

| label | auc | n | root_cause |
|-------|-----|---|------------|
| cryptojacking | 0.8306760204081632 | 28 | alpha_may_help |
| ltma | 0.9130527210884354 | 28 | alpha_may_help |
| swma | 0.831776907191448 | 196 | alpha_may_help |

Note: Under conformal scoring, FPR often tracks alpha; low-AUC classes need feature/evidence work, not alpha alone.
