---
threat_id: B-PERIODIC-001
category: benign
mitre_technique: null
evidence:
  - name: period_match
    necessity: required
    description: "규칙적 전력이 정상 step/동기화 주기로 설명됨"
  - name: period_mismatch
    necessity: exclusion
    description: "주기 불일치가 있으면 정상이 아니다"
  - name: progress_log_available
    necessity: supporting
    description: "DDP/체크포인트 등 정상 주기성의 설명 로그"
benign_lookalikes: []
grid_relevance:
  mechanism: none
  tier: unsupported
  notes: "DRAFT pending human approval. 정상 분산학습의 규칙적 출렁임"
observability:
  min_sample_hz: 1
  notes: "NVML 1초 평균을 전제로 해석"
thresholds_provenance: dataset/eval/evidence_thresholds.json
---
# 정상이지만 공격과 유사한 주기적 워크로드 (Benign Periodic Workload)

## 관측 요약
지속적이고 뚜렷한 주기성이 존재하고 전력 변동폭도 크지만, 정상 운영 활동이다.
high/low 전환이 규칙적으로 보일 수 있고 순간 전력 상승·하강 기울기도 크다.
즉 주기성·변동폭·램프만으로는 공격과 구분되지 않는 정상 사례들이다.

## 해당하는 정상 활동
- 대규모 분산학습: compute phase와 all-reduce 통신 phase가 교대로 반복되어
  강한 주기성이 나타난다. 여러 GPU가 동시에 움직이는 것도 정상이다.
- 주기적 체크포인트 저장: 일정 step마다 GPU가 유휴로 떨어졌다 복귀한다.
- 하이퍼파라미터 탐색(HPO): 짧은 학습이 반복되어 규칙적 패턴을 만든다.
- 학습·평가 교대 실행: 두 개의 부하 수준을 오간다.
- 데이터로더 병목: I/O 대기로 전력이 불규칙하게 출렁인다.

## 정상으로 판정하기 위한 근거
아래 맥락 증거가 확인되면 주기성이 강해도 정상으로 판정한다.
- 스케줄러에 정상 등록된 job_id가 존재하고 실행 시간대가 일치
- 승인된 id_user와 신고된 job_type이 실제 연산 특성과 부합
- 동기화된 GPU들이 같은 분산학습 그룹에 속함
- 주기가 학습 iteration 시간·체크포인트 주기와 설명 가능하게 대응

## 운영상 취급
이 문서와 높은 유사도로 매칭되는 이벤트는 후보로 선별되었더라도 정상으로
분류하고 경보를 발생시키지 않는다. 본 문서는 오경보 억제를 목적으로 하는
hard negative 기준 문서이다.

## 주의
주기성이 존재한다는 사실만으로 공격을 단정해서는 안 된다. 공격 판정은 주기성에
더해 맥락적 불일치(설명되지 않는 스케줄러 관계, 비인가 사용자, 독립적이어야 할
job 간 위상 일치)가 함께 확인될 때만 이루어진다.

## Evidence 근거 (초안)
정상 분산학습의 규칙적 출렁임
