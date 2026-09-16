---
threat_id: T-CRYPTO-001
category: cyber_physical_attack
mitre_technique: null
evidence:
  - name: persistent_high_load
    necessity: supporting
    description: "평탄하고 높은 부하가 장시간 지속"
  - name: progress_log_present_training
    necessity: exclusion
    description: "정상 학습 로그가 지속 고부하를 설명하면 배제 방향"
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
# GPU 크립토재킹 (Cryptojacking)

## 출처
GPU 텔레메트리 기반 크립토재킹 탐지 선행연구 다수 (nvidia-smi 기반 분류기, RF/DT로 고정확도 보고).

## 공격 개요
공격자가 무단으로 GPU 자원을 암호화폐 채굴에 사용. 침해된 계정이나 컨테이너에서 실행되거나,
정상 ML 파이프라인에 은닉되기도 함.

## 관측 가능한 정성적 특성 (탐지 근거로 사용)
- **평균 전력**: 지속적으로 높음. 채굴은 GPU를 꾸준히 고부하로 사용 → 평균 전력·사용률이 계속 상승.
- **변동폭(swing)**: 작음. on/off 반복이 아니라 지속적 고사용률이라 변동이 적고 평평함.
- **duty cycle 규칙성**: 해당 없음 (거의 상시 active).
- **지속시간**: 매우 김. 채굴은 오래 돌수록 이득이라 장시간 지속.
- **메모리 패턴**: 채굴 알고리즘 특유의 메모리 접근 패턴 (해시 연산 위주).

## 저해상도 텔레메트리에서 남는 흔적
- 평균 전력·GPU 사용률이 지속적으로 높은 것이 가장 뚜렷한 신호.
- Bit2Watt류와 정반대: 크립토재킹은 "높고 평평", Bit2Watt는 "출렁임".

## 정상 워크로드와의 구분 포인트 / 다른 공격과의 차이
- SWMA/LTMA와의 결정적 차이: 크립토재킹은 변동성이 낮고 평균이 지속적으로 높음.
  반면 SWMA/LTMA는 변동성이 크고(특히 SWMA), 평균은 정상일 수도 있음(특히 LTMA).
- 정상 대규모 학습과는 평균 전력만으로는 구분이 어려울 수 있어, 지속시간·작업 유형·사용자 맥락 결합 필요.
