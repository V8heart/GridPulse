---
threat_id: T-BURST-RAMP-001
category: cyber_physical_attack
mitre_technique: null
evidence:
  - name: feature_range_match
    necessity: supporting
    description: "train split에서 적합된 feature range와 근접"
  - name: context_inconsistency
    necessity: supporting
    description: "선언 컨텍스트와 관측 evidence가 불일치"
benign_lookalikes: []
grid_relevance:
  mechanism: unsupported
  tier: unsupported
  notes: "# TODO(review): public test-system evidence only; do not overstate real-grid impact"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/corpus_feature_ranges.json
---
# 급격 부하 변동 공격 (Burst / Ramp Load Attack)

## 관측 요약
순간 전력 상승·하강 기울기가 매우 크다. 짧은 시간에 전력이 급격히 치솟거나
급락하며, 주기성은 약하거나 없다. 전력 변동폭이 크고 high/low 전환이 급격하다.
평균 전력 자체는 정상 범위일 수 있어 평균만 보면 놓치기 쉽다.

## 전력·워크로드 특성
- 평균 전력: 정상 범위일 수 있음. 평균 기반 탐지로는 포착 어려움
- 변동폭(swing ratio): 큼 (1.5 이상)
- 램프: 매우 큼. 이 공격의 핵심 지표
- 주기성: 약함. 단발 또는 산발적 반복
- duty 전환: 불규칙
- 지속시간: 개별 구간은 짧음(수 초~수십 초)
- 다수 GPU가 동시에 동일 방향으로 급변하면 위험도가 크게 상승

## 정상 워크로드와의 구분 포인트
정상 상황에서도 job 시작·종료, 체크포인트 저장, 데이터로더 병목에서 급격한
전력 변화가 발생한다. 따라서 램프 크기만으로 단정할 수 없다. 스케줄러 이벤트
(job 제출·종료 시각)와 시간적으로 정렬되지 않는 급변이 의심 대상이다.

## 물리적 관련성
높음. 급격한 부하 증감은 계통 주파수 변화율(RoCoF)에 직접 영향을 준다.
계통 관성이 낮을수록 동일한 부하 계단에 대한 주파수 편차가 커진다.
대규모 부하가 동시에 이탈하거나 투입되는 상황은 계통 운영 관점에서
가장 직접적인 위험 시나리오에 해당한다.

## 근거
북미 계통에서 데이터센터 부하가 전압 과도에 반응해 대규모로 동시 이탈한 사례가
보고되었으며, 계통 운영기관은 이를 신뢰도 현안으로 다루고 있다. 데이터센터의
보호 설계(UPS·발전기 이중화)가 오히려 단기 전압 강하에 민감하게 반응하도록
만든다는 분석이 함께 제시되었다.
