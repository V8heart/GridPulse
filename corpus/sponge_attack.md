---
threat_id: T-SPONGE-001
category: cyber_physical_attack
mitre_technique: null
evidence:
  - name: sustained_high_load
    necessity: required
    description: "비정상적으로 비싼 입력으로 고부하 지속"
  - name: period_match
    necessity: exclusion
    description: "정상 step 주기 설명이면 sponge보다 정상/학습 쪽"
  - name: flat_power
    necessity: supporting
    description: "고비용 입력이 변동을 줄인 채 고전력을 유지할 수 있음"
benign_lookalikes: []
grid_relevance:
  mechanism: frequency_response_stress
  tier: unsupported
  notes: "DRAFT pending human approval. 에너지/지연을 노린 sponge 입력 — 지속 부하 스트레스"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/evidence_thresholds.json
---
# 에너지·지연 증폭 공격 (Sponge / Energy-Latency Attack)

## 관측 요약
평균 전력이 정상 추론 대비 크게 높게 지속된다. 전력 변동폭은 작고 평평하며,
고부하 상태가 입력이 유입되는 동안 계속 유지된다. 주기성은 요청 주입 주기를
따르며, 공격자가 주입 간격을 조절하면 뚜렷한 주기성이 나타난다.

## 전력·워크로드 특성
- 평균 전력: 매우 높음. 동일 모델의 정상 추론보다 수 배 이상
- 변동폭(swing ratio): 작음. 고부하가 평탄하게 유지됨
- 주기성: 공격자가 제어 가능. 요청 주입 간격이 곧 주기가 됨
- duty 전환: 요청 주입 패턴에 따라 규칙적일 수 있음
- 램프: 요청 시작 시 급격한 상승
- 지속시간: 입력 공급이 지속되는 한 계속
- 동일 모델·동일 배치 크기인데 전력만 비정상적으로 높은 것이 핵심 단서

## 정상 추론과의 구분 포인트
모델과 배치 크기가 같은데도 요청당 소비 전력과 지연이 이례적으로 크다면 의심
대상이다. 입력 길이·토큰 수 대비 전력 효율이 급격히 나빠지는 것이 특징이다.

## 물리적 관련성
중간~높음. 권한 침해 없이 외부 입력만으로 GPU 전력을 끌어올릴 수 있고,
요청 주입 시점을 조절하면 주기적 부하 변조로 전환 가능하다. 다수 노드를
동시에 겨냥하면 협조적 부하 조작에 근접한다.

## 근거
Shumailov 등은 에너지 소비와 지연을 최대화하도록 설계한 입력(sponge examples)이
비전·언어 모델의 에너지 소비를 10~200배 증가시킬 수 있음을 보였다. 실제 상용
번역 서비스에서 응답시간이 크게 늘어난 사례도 보고되었다. 저자들은 NVML로
에너지를 측정했으며, 이는 본 파이프라인의 관측 수단과 동일하다.
(Sponge Examples: Energy-Latency Attacks on Neural Networks, IEEE EuroS&P 2021, arXiv:2006.03463)

## Evidence 근거 (초안)
에너지/지연을 노린 sponge 입력
