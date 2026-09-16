---
threat_id: T-COORD-GPU-001
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
# 다중 GPU 협조 부하 변조 (Coordinated Multi-GPU Modulation)

## 관측 요약
개별 GPU만 보면 전력 변동폭이 크지 않으나, 여러 GPU의 전력 변화가 같은 위상으로
동시에 움직인다. 지속적이고 뚜렷한 주기성이 존재하며 high/low 전환이 규칙적이다.
합산 전력에서는 변동폭이 개별 GPU 대비 크게 증폭되어 나타난다.

## 전력·워크로드 특성
- 평균 전력: 개별 GPU는 정상 범위. 합산 시 변동이 두드러짐
- 변동폭(swing ratio): 개별로는 작으나(1.1 내외) 합산 시 선형 증가
- 주기성: 강하고 안정적. 여러 GPU가 동일 주파수를 공유
- 다중 GPU 상관: 매우 높음. 위상차가 거의 0에 수렴
- duty 전환: 규칙적
- 지속시간: 김. 공격이 유지되는 동안 계속

## 정상 분산학습과의 구분 포인트 (가장 중요)
정상 대규모 분산학습도 all-reduce 동기화 때문에 여러 GPU가 동시에 움직인다.
따라서 동기화 자체는 악의성의 근거가 되지 못한다. 판별 기준은 그 동기화가
정당한 스케줄링 관계로 설명되는가이다.
- 같은 job_id / 같은 분산학습 그룹에 속한 GPU가 동기화 → 정상
- 서로 다른 id_user / job_type에 속해 독립적이어야 할 job이 동기화 → 의심
- 스케줄러가 함께 배치하지 않은 노드 간 위상 일치 → 강한 의심

## 물리적 관련성
높음. 개별 부하 변동은 계통에 무의미하지만 다수가 협조하면 합산 부하 변조가
되어 물리적 의미를 갖는다. 인버터 비중이 높고 계통 관성이 낮은 조건에서
동일한 변조의 응답이 더 커진다.

## 근거
Load-Altering Attack 연구는 인터넷 연결 부하를 조직적으로 조작해 전력계통에
영향을 줄 수 있음을 보였고, BlackIoT(MadIoT)는 다수 고전력 기기의 동시 조작으로
이를 구체화했다. Bit2Watt는 이 개념을 GPU workload 수준으로 확장했으며 대표
결과를 다수 GPU·고DER 조건에서 제시했다.
