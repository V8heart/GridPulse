---
threat_id: T-LLMJACKING-001
category: attack
synthetic_source_labels: ["normal_hpo_search"]
swing_ratio: [0.988, 1.5494]
periodicity_strength: [0.7616, 0.9225]
duty_regularity: [0.0, 0.4442]
mean_power_level: any
ramp_level: high
multi_gpu_sync: optional
min_duration_s: null
ramp_max_w_per_s: [1636.246, 2252.9626]
high_load_fraction: [0.0725, 0.6925]
longest_high_seconds: [1.45, 11.35]
---

# LLM 서비스 탈취 (LLMJacking)

## 관측 요약
평균 전력이 중간~높은 수준에서 오르내리며, 전력 변동폭이 크고 불규칙하다.
주기성은 약하거나 없다. 순간 전력 상승·하강 기울기가 크고 짧은 고부하 구간이
산발적으로 반복된다. high/low 전환이 불규칙하고 예측하기 어렵다.

## 전력·워크로드 특성
- 평균 전력: 중간~높음. 요청량에 따라 시간대별로 크게 변동
- 변동폭(swing ratio): 큼 (1.3~2.0). 요청 도착이 불규칙하기 때문
- 주기성: 약함. 배치 추론이 아니면 뚜렷한 반복 주기가 없음
- duty 전환: 매우 불규칙. 외부 요청 도착 분포를 그대로 따름
- 램프: 큼. 추론 요청 시작·종료가 급격함
- 지속시간: 개별 구간은 짧으나 전체 활동은 장시간 이어짐
- 메모리 사용량은 모델 적재 상태로 높게 고정, 사용률만 출렁임

## 정상 추론 서비스와의 구분 포인트
전력 패턴만으로는 정상 추론 트래픽과 구분이 어렵다. 핵심 단서는 맥락이다:
승인되지 않은 API 자격증명 사용, 정상 서비스 시간대를 벗어난 요청 급증,
과금 계정과 실제 사용 주체의 불일치.

## 물리적 관련성
낮음~중간. 요청이 외부에서 조작 가능하므로, 공격자가 요청 발생 시점을 의도적으로
동기화하면 부하 변조 수단으로 전용될 여지가 있다.

## 근거
MITRE ATT&CK T1496.004 Cloud Service Hijacking에 해당한다. 리버스 프록시로
클라우드 호스팅 AI 모델의 연산 자원을 탈취하는 공격이 실제로 보고되었다.
정상 요청과 동일한 실행 경로를 사용하므로 권한 상승이 필요하지 않다.
