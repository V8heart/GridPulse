# Corpus evidence draft — human review (v2)

Test metrics were **not** used to choose these rows.

Mechanisms: `electromechanical_oscillation` | `frequency_response_stress` | `converter_resonance` | `none`.

| doc | required / required_any | exclusion | supporting | mechanism | tier | 이 증거가 없으면 주장 불가인 이유 |
|-----|-------------------------|-----------|------------|-----------|------|----------------------------------|
| swma | period_mismatch, strong_peak | period_match | progress_log_missing | electromechanical_oscillation | B | 로그로 설명되는 정상 위상이면 SWMA(별도 스케줄 커널)를 주장할 수 없다 / peak 없으면 기계적 변조 주장을 스펙트럼으로 뒷받침할 수 없다 |
| ltma | util_power_decoupled, ltma_period_or_mismatch=(period_mismatch|unexplained_changepoint) | flat_power | - | electromechanical_oscillation | B | util에 묶인 정상 학습 동력이면 LTMA식 평균유지 변조라고 주장할 수 없다 / 둘 다 없으면 학습 내부 변조의 시간 구조를 주장할 근거가 없다 |
| cryptojacking | crypto_high_or_flat=(sustained_high_load|flat_power) | period_match | progress_log_missing | none | unsupported | 고부하 지속/평탄 둘 다 없으면 채굴형 무단 부하를 주장할 수 없다 |
| coordinated_multi_gpu | cross_job_sync | period_match | period_mismatch | electromechanical_oscillation | B | cross-job 동기가 없으면 협조형 다중 GPU 공격을 주장할 수 없다 |
| burst_ramp_attack | high_ramp | flat_power | unexplained_changepoint | frequency_response_stress | B | 급램프가 없으면 burst/ramp 외란을 주장할 수 없다 |
| hidden_ml_training | hidden_context_break=(declared_family_mismatch|progress_log_missing) | period_match | period_mismatch | none | unsupported | 둘 다 없으면 '숨은/비인가 학습'을 주장할 컨텍스트 근거가 없다 |
| llmjacking | high_ramp | flat_power | declared_family_mismatch, progress_log_missing | none | unsupported | 버스트 램프가 없으면 탈취된 추론 트래픽 주장을 관측으로 못 박기 어렵다 |
| sponge_attack | sustained_high_load | period_match | flat_power | frequency_response_stress | unsupported | 지속 고부하가 없으면 sponge식 에너지/지연 공격을 주장할 수 없다 |
| normal_workloads | period_match | period_mismatch | progress_log_available | none | unsupported | 주기 일치가 없으면 '설명 가능한 정상 워크로드' 문서를 주장할 수 없다 |
| benign_periodic | period_match | period_mismatch | progress_log_available | none | unsupported | period_match가 없으면 benign 주기성이라고 주장할 수 없다 |
