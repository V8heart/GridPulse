---
threat_id: T-SWMA-001
category: cyber_physical_attack
mitre_technique: null
evidence:
  - name: period_mismatch
    necessity: required
    description: "전력 지배 주기가 진행 로그 step 주기와 불일치"
  - name: strong_peak
    necessity: required
    description: "규칙적 on/off가 스펙트럼에 뚜렷한 peak로 남음"
  - name: period_match
    necessity: exclusion
    description: "로그 주기와 일치하면 정상 학습 위상으로 배제"
  - name: progress_log_missing
    necessity: supporting
    description: "위장 커널은 training progress log가 빈약한 경우가 많음"
benign_lookalikes: []
grid_relevance:
  mechanism: electromechanical_oscillation
  tier: B
  notes: "DRAFT pending human approval. Bit2Watt SWMA: 별도 CUDA 커널의 규칙적 active/passive 전환 → 저주파 전기기계 외란 후보"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/evidence_thresholds.json
---
# SWMA (Synthetic Workload Modulation Attack)

## 출처
Bit2Watt (Ji, Pan, Xu, arXiv:2607.05993, 2026), 4.2.1절.

## 공격 개요
공격자가 별도 CUDA 커널(active/passive 두 모드)을 올려, 호스트 컨트롤러가
unified-memory control flag로 두 모드를 정밀한 스케줄에 따라 전환한다. 권한 상승 불필요.

## 관측 가능한 정성적 특성 (탐지 근거로 사용)
- **평균 전력**: 정상 워크로드 범위를 벗어날 수 있음 (active 모드가 SM을 포화시켜 순간 전력이 TDP 근처까지 도달).
- **변동폭(swing)**: 매우 큼. active↔passive 전이가 뚜렷.
- **duty cycle 규칙성**: 매우 규칙적. 호스트가 정해진 스케줄로 전환하므로 주기가 일정하고 예측 가능.
- **주파수 제어 가능성**: 있음. 공격자가 명시적으로 조정 가능 (Table 1 기준 1.5~6kHz 대역).
- **워크로드 위장성**: 낮음. 정상 학습/추론과 무관한 별도 합성 커널이라, 프로파일링 시 정체불명의 커널로 보임.
- **지속성**: 공격이 실행되는 동안 지속적으로 반복.

## 저해상도 텔레메트리에서 남는 흔적 (우리가 실제로 관측하는 것)
- 원 신호(kHz)는 직접 관측 불가하나, 다운샘플링된 신호에서도 변동폭(분산)이 정상 대비 크게 증가.
- SM 사용률이 규칙적으로 100%↔0% 부근을 오가는 패턴.
- 정상 학습의 data-loading/compute 위상 전환보다 훨씬 규칙적이고 기계적임.

## 정상 워크로드와의 구분 포인트
- 정상 대규모 학습도 compute/communication 위상 전환으로 전력이 출렁이지만, 그 주기는
  배치 처리·동기화에 종속되어 불규칙하고 맥락(모델 크기, 배치)으로 설명 가능하다.
- SWMA는 "기계적으로 규칙적인 on/off"가 핵심 차별점.

## Evidence 근거 (초안)
Bit2Watt SWMA: 별도 CUDA 커널의 규칙적 active/passive 전환
