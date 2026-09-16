---
threat_id: T-HIDDEN-ML-001
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
# 비인가 은닉 ML 학습 (Hidden / Unauthorized ML Training)

## 관측 요약
평균 전력이 정상 기준선보다 높게 오래 지속되며, 전력 변동폭은 중간 수준이다.
주기성이 존재하지만 duty 전환이 완전히 규칙적이지는 않다. 순간 전력 상승·하강
기울기는 중간이며, 고부하 구간이 수십 분 이상 끊기지 않고 이어진다.

## 전력·워크로드 특성
- 평균 전력: 높음, 장시간 유지 (학습 job과 유사)
- 변동폭(swing ratio): 중간 (1.1~1.4). 크립토재킹보다 크고 SWMA보다 작음
- 주기성: 존재하나 강하지 않음 (iteration 경계에서 약한 반복)
- duty 전환: 반규칙적. 배치 크기·데이터로더 상태에 따라 흔들림
- 램프: 중간. 커널 전환 시점에 완만한 상승·하강
- 지속시간: 매우 김 (수십 분~수 시간)
- 메모리 사용률과 GPU 사용률이 동시에 높게 유지됨

## 정상 학습과의 구분 포인트
전력·사용률 패턴만으로는 정상 학습과 거의 구분되지 않는다. 판단은 맥락 증거에
의존한다: 스케줄러에 등록되지 않은 job, 승인되지 않은 사용자(id_user), 신고된
job_type과 실제 연산 특성의 불일치, 예약되지 않은 시간대 실행.

## 물리적 관련성
단독으로는 전력 인프라 위협이 낮다. 다만 다수 노드에서 동시에 발생하면 시설
단위 기저부하를 증가시켜 피크 여유를 잠식할 수 있다.

## 근거
NVML 9종 카운터(사용률·메모리·전력·온도·클럭·PCIe)를 1Hz로 수집해 은닉 학습을
식별할 수 있음이 보고되었다. 애플리케이션 코드나 모델 가중치 접근 없이 운영
텔레메트리만으로 탐지 가능하다는 점이 본 파이프라인의 관측 조건과 일치한다.
(arXiv:2606.19262, Detecting Hidden ML Training With Zero-Overhead Telemetry)
MITRE ATT&CK T1496.001 Compute Hijacking과도 연결된다.
