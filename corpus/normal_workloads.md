---
threat_id: B-NORMAL-001
category: benign
synthetic_source_labels: ["normal_baseline", "normal_checkpoint", "normal_dataloader_stall", "normal_distributed_training", "normal_eval_train_switch", "normal_hpo_search"]
swing_ratio: [0.0663, 1.5494]
periodicity_strength: [0.6788, 0.968]
duty_regularity: [0.0, 0.8617]
mean_power_level: any
ramp_level: high
multi_gpu_sync: optional
min_duration_s: null
ramp_max_w_per_s: [24.052, 2252.9626]
high_load_fraction: [0.0, 0.7275]
longest_high_seconds: [0.0, 11.35]
---

# 정상 워크로드 프로파일 (오탐 방지용 기준)

## 목적
이상 신호가 감지됐을 때, 그것이 아래의 "정상적으로 전력이 출렁이는 상황"과 일치하는지
확인하기 위한 기준 문서. 아래 패턴과 유사하면 공격이 아니라 정상일 가능성이 높다.

## 정상적으로 변동폭이 큰 워크로드들
- **대규모 분산 학습(정상)**: compute-heavy 위상과 communication-heavy(all-reduce) 위상이
  번갈아 나타나 전력이 출렁임. 단, 주기가 배치 크기·모델·네트워크에 종속되어 불규칙하고,
  그 사용자의 과거 학습 패턴과 일치함.
- **하이퍼파라미터 탐색**: 짧은 학습을 여러 번 반복 → 전력이 오르내림. 하지만 각 구간이 짧고
  job_type/사용자 이력으로 설명 가능.
- **체크포인트 저장**: 주기적으로 GPU가 잠깐 유휴(저장 중) → 전력 급락 후 복귀. 일회성 스파이크.
- **데이터 로딩 병목**: I/O 대기로 GPU가 잠깐 놀았다가 재개 → 불규칙한 전력 변동.
- **학습-평가(eval) 교차**: 학습 구간과 평가 구간의 부하가 달라 전력 프로파일이 두 레벨로 나뉨.

## 정상 판정의 핵심 근거
- 변동의 원인이 맥락으로 설명됨 (예정된 작업, 알려진 사용자 패턴, 표준적 학습 동역학).
- 이 사용자가 과거에도 유사한 패턴을 보였음.
- job_type이 명확하고, 시간대·GPU 조합이 이 사용자에게 통상적임.

## 유지보수·운영성 변동 (정상)
- 드라이버 업데이트, 정기 유지보수, 벤치마크 실행 등은 일시적 전력 변동을 유발하나
  운영 일정과 대조하면 정상으로 확인 가능.
