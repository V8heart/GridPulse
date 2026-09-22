# GridPulse — GPU 워크로드 이상탐지 파이프라인

새 공격이 나올 때마다 탐지 모델을 재학습하는 대신, **범용 이상 필터로 의심 활동을
선별하고, 알려진 공격 시그니처 문서와 RAG로 대조**하는 실용 파이프라인.

## 설계 근거 (검증된 2-tier IDS 패턴)

이 구조는 침입탐지(IDS) 분야에서 확립된 정석입니다:
- **LLM-Enhanced Two-Tier IDS** (Springer 2026): 가벼운 통계 모니터링으로 상시 감시하다가,
  이상 감지 시에만 무거운 LLM 분석을 on-demand로 촉발 → 우리의 "의심될 때만 고해상도 분석"과 동일.
- **POSEIDON 2-tier IDS**: 이상탐지(1단계) + 시그니처매칭(2단계) 결합.
- 이 패턴은 "하루 1,000개 알람을 40개, 다시 5개로 압축"하는 것으로 실증됨.

**우리가 하지 않는 것**: Bit2Watt의 kHz 원신호를 직접 탐지 (물리 계측 영역, 우선순위 제외).
**우리가 하는 것**: 관측 가능한 해상도에서 의심 활동을 선별 → 알려진 공격과 대조 → 설명 생성.

## 파이프라인 3단계

```
DCGM/NVML 텔레메트리
   ↓
[1단계] 상시 저비용 스크리닝 — 윈도우별 변동폭 z-score, 의심 후보만 선별
   ↓ (후보만)
[2단계] 고해상도 특성 분석 — 주기성/duty 규칙성/평균유지 여부를 정성적 설명으로 변환
   ↓
[3단계] RAG 대조 + LLM 판정 — 알려진 공격 시그니처와 의미 기반 매칭, 위험도+근거 생성
```

## 서버 요구사항 (검증된 환경)

- Ubuntu 22.04, Python 3.10+, CUDA 13.0, RTX 4090 (실측 환경: 4090 x2, 48코어, 125GB RAM)
- GPU 한 장으로 공격/워크로드 재현, 다른 한 장으로 로컬 LLM 서빙 가능

## 설치 및 실행

```bash
bash setup_env.sh
source .venv/bin/activate

# (선택) 로컬 LLM: 이 서버에는 Ollama 0.32.1이 이미 설치되어 있음
#   ollama serve   # 미실행 시
#   검증된 모델: gemma3:12b  (qwen3:14b는 format=json에서 빈 응답이 나와 비권장)

# 파이프라인 실행 (수집된 텔레메트리 CSV 대상)
python pipeline/run_pipeline.py \
    --telemetry dataset/synthetic/all_v3.csv \
    --baseline-mean 120 \
    --stage1 v2 \
    --stage2 v2 \
    --rag-backend tfidf \
    --llm-backend ollama \
    --llm-model gemma3:12b
```

오프라인/빠른 검증만 할 때는 `--llm-backend stub`을 쓰면 됩니다.
### 합성 데이터셋과 평가

```bash
python dataset/build_synthetic.py
python pipeline/fit_corpus_ranges.py
python pipeline/fit_stage1_v2.py
python pipeline/eval_retrieval.py --telemetry dataset/synthetic/all_v3.csv \
  --split-manifest dataset/synthetic/split_manifest.json --split test
python pipeline/eval_stage1.py
python pipeline/eval_stage2.py --backend stub
pytest -q
```

합성 데이터는 `session_id + gpu_id` 단위로 처리하므로 서로 다른 클래스의
경계가 하나의 분석 윈도우에 섞이지 않는다.
`all_v2.csv`는 하위 호환용 산출물이며 공식 재현 대상은 `all_v3.csv`와
`split_manifest.json`이다.

### 실측 텔레메트리 (gp-telemetry/1.2)

표준: [`dataset/dataset_standard_v1_2.md`](dataset/dataset_standard_v1_2.md).

Collector는 observed만 기록하고, declared/gt는 finalize가 붙인다. 경로는 opaque
`s-<hex>` 세션 ID만 쓰며 라벨 이름을 넣지 않는다.

```bash
# dry-run (GPU 없음): 명령·역할·declared만 확인
python -m dataset.run_capture --workload gpt_tiny_finetune --dry-run
python -m dataset.capture_matrix --dry-run

# 120초 vertical-slice smoke (공유 GPU 안전 확인 후)
bash scripts/gpu_smoke_v2.sh --dry-run
bash scripts/gpu_smoke_v2.sh --confirm-shared-gpu-safe

# 전체 matrix (~22–26h, resume 가능; 승인 후 night shard로 분할)
bash scripts/capture_all.sh --dry-run
bash scripts/capture_all.sh --confirm-shared-gpu-safe --shard 0 --shards 3

# 관측성 (alias / boxcar / period_retention.csv)
python -m dataset.observability dataset/real/sessions --output dataset/real/observability/summary.md
```

이번 매트릭스에는 서로 다른 정상 잡이 한 세션에서 동시에 도는 `mixed_tenants`가 없다.
그래서 `cross_job_sync`의 정상 기준은 없고, 그 지표는 coordinated 공격에서만 관측된다.

`--interval-ms`는 API 호출 요청 주기이지 센서의 물리 측정률이 아니다.
`actual_interval_ms`, `value_changed`와 observability summary로 실효 갱신률을
별도로 확인해야 한다. Nyquist 초과 성분은 `aliased`로 기록하고 folded frequency와
boxcar `sinc` 감쇠를 함께 본다.

### 통제된 실측 워크로드

다른 사용자의 GPU 작업이 없는지 먼저 확인한 다음에만 실행한다.
`--confirm-shared-gpu-safe` 없이는 실부하가 거부된다.

```bash
python -m dataset.run_capture --workload baseline --confirm-shared-gpu-safe
python -m dataset.run_capture --workload swma --period 1 --duty-cycle 0.5 \
  --confirm-shared-gpu-safe
python -m dataset.run_capture --workload gpt_tiny_finetune --duration 120 \
  --confirm-shared-gpu-safe
```

기본 duration은 클래스 공통 pool `{480,600,720}`에서 라벨과 독립적으로 뽑는다.
SWMA 코드는 Bit2Watt persistent kernel의 동일 재현이 아니라, PyTorch CUDA
연산과 sleep으로 소프트웨어 telemetry에서 보이는 규칙적 변동을 만드는 근사다.
2-GPU 캡처는 `distributed` 또는 `swma_multi` workload를 사용한다.

합성 재적합은 기존 `dataset/eval/`을 덮지 않고 `dataset/eval/v2_regen/`에만 쓴다.
먼저 Mondrian baseline을 커밋·태그(`stage1-mondrian-baseline`)한 뒤:

```bash
bash scripts/refit_stage1_v2_regen.sh          # cal only
bash scripts/refit_stage1_v2_regen.sh --run-test  # 동결 후 1회
```

pytest 기본: `pytest -q -m "not gpu"`.

## 구성요소

| 경로 | 역할 |
|---|---|
| `corpus/*.md` | 공격 시그니처 지식베이스. **새 공격은 여기 문서만 추가하면 대응** |
| `pipeline/features.py` | 1·2단계: 텔레메트리 → 통계 피처 → 정성적 자연어 설명 |
| `pipeline/rag_analyzer.py` | 3단계: RAG 검색(sbert/tfidf) + LLM 판정(ollama/stub) |
| `pipeline/run_pipeline.py` | 전체 오케스트레이션 (+ 선택 `--physics-validate` 이벤트 트리거) |
| `pipeline/fit_corpus_ranges.py` | Part A: train split only로 corpus feature range 산출 |
| `pipeline/fit_stage1_v2.py`, `pipeline/eval_stage1.py` | Part B: cohort baseline/conformal calibration 및 Stage1 v2 평가 |
| `pipeline/stage2_evidence.py`, `pipeline/stage2_llm.py`, `pipeline/eval_stage2.py` | Part C: structured evidence bundle과 Stage2 v2 평가 |
| `pipeline/physics_correlation.py` | **§8 실험 전용**: 고정 시나리오 Physics CSV와 cyber JSON을 `attack_id`로 사후 결합(운영 트리거 아님) |
| `dataset/` | 공통 스키마, 합성 데이터, 캡처·평가 도구 |
| `bit2watt_impl/physics/` | 공개 Kundur/WECC 동적 응답 + WECC IBR(교체) 침투율 스윕 |
| `bit2watt_impl/` | NVML 수집과 SWMA/LTMA-like 검증 워크로드 |
| `workloads/` | 정상 PyTorch hard-negative 워크로드 |

## 백엔드 옵션 (점진적 구축용)

- **RAG**: `sbert`(권장, 의미 임베딩) / `tfidf`(오프라인 폴백, 모델 다운로드 불필요)
- **LLM**: `ollama`(로컬 경량 모델) / `stub`(LLM 없이 규칙 기반, 파이프라인 검증용)

→ LLM 없이 `--rag-backend tfidf --llm-backend stub`으로 먼저 파이프라인 전체를
   검증하고, 그다음 sbert + ollama로 성능을 올리는 순서를 권장.

## 데이터 누설 방지와 평가 상태

Part A 변경으로 추론 입력에서는 `gt_label`, `gt_attack_id`, `attack_id`,
`waveform_*`, `power_phys_w`, legacy `label`을 제거한다. `declared_job_type`과
`declared_job_family`는 scheduler가 제공하는 선언 context로만 사용되며, 공격명과
같은 oracle 문자열을 넣지 않는다. 예전 `top1_accuracy=1.0` 주장은 문서 질의와
라벨 context를 함께 쓰던 **누설 포함 상한**으로만 취급한다. 공식 Retrieval 평가는
`--query-context off`에 해당하는 no-context/test-split 결과만 README나 보고서 수치로
사용한다.

## Stage 1 v2 설계 근거 (R1-R10)

- R1: `all_v3.csv`는 `declared_*`와 `gt_*`를 분리해 inference와 evaluation 경계를 명시한다.
- R2: `strip_ground_truth()`가 정답/파형/oracle 컬럼을 제거해 파이프라인 추론 경로 누설을 차단한다.
- R3: 합성 데이터는 NVML 1초 trailing average를 에뮬레이션해 100ms polling과 센서 smoothing 차이를 반영한다.
- R4: 정상 hard-negative는 DDP/FSDP trough, flat pretraining, bursty inference, mixed tenants를 포함한다.
- R5: 공격 variant는 shallow, jitter, piggyback, mimicry, coordinated multi-GPU를 포함해 evasive case를 만든다.
- R6: feature v2는 PSD band power, spectral entropy, Theil-Sen power-util residual, CUSUM changepoint를 포함한다.
- R7: baseline은 `declared_job_family + gpu_model` cohort의 robust median/MAD에서 시작하고 부족하면 family/global로 fallback한다.
- R8: threshold는 calibration split conformal p-value로 산출하며 test split을 tuning에 쓰지 않는다.
- R9: progress log의 step/checkpoint/eval/request event가 주기와 변화점을 설명하면 오탐을 낮춘다.
- R10: 높은 grid impact지만 conformal p-value가 후보 기준을 넘지 않으면 `grid_watch`로 남겨 운영 관찰 대상으로 분리한다.

## Corpus v2와 Stage 2 v2 (R11, R12)

- R11: `corpus/*.md` front-matter는 evidence checklist schema를 사용한다. 숫자 range는 문서에 박지 않고 `pipeline/fit_corpus_ranges.py`가 train split에서 `dataset/eval/corpus_feature_ranges.json`으로 산출한다.
- R12: Stage2 v2는 raw time series나 ANDES 로그를 LLM에 넣지 않고 `stage2_evidence.py`가 만든 structured evidence bundle만 전달한다. LLM 출력은 `known|partial|unknown`, matched/contradicting evidence, fallback 여부를 포함하는 JSON schema로 검증한다.
- Evidence 이름은 `pipeline/evidence_vocab.py`에 고정되어 있으며 (`feature_range_match` 금지), bool 임계값은 train/cal만으로 `fit_evidence_thresholds.py`가 맞춘다.
- Corpus evidence 초안 검토표: `dataset/eval/corpus_evidence_draft_review.md` (**사람 승인 전 공식 Stage2 수치로 주장하지 말 것**).
- Stage2 평가는 `split_manifest`의 test 세션만 사용하고, fallback은 retriever top-1을 쓰며 정상→`benign_periodic` 자동정답 특례는 없다.
- 누설 전/후 표: `dataset/eval/leakage_before_after.json` (`oracle_label` = 누설 포함 상한).
- Stage1 cal AUC / alpha 스윕: `stage1_cal_score_auc.json`, `stage1_alpha_sweep.json` (test는 선택 후 1회).

## 검증 상태

- ✅ features.py: 규칙적 사각파(SWMA류) vs 평평한 고부하(크립토재킹류)를 정성적으로 정확히 구분
- ✅ 합성 v3: 3개 기본 공격 + evasive SWMA 변종 + 6개 기존 정상 + AI datacenter 정상 hard-negative를 세션 경계 없이 생성
- ✅ Stage1 v2: train/cal만 사용해 cohort baseline과 conformal calibration 산출
- ✅ Retrieval: 공식 수치는 no-context/test split 기준으로 보고하며, 구 `top1_accuracy=1.0`은 누설 포함 상한으로만 표기
- ✅ Stage2 v2: corpus v2 evidence checklist, structured evidence bundle, deterministic stub/fallback 평가 경로 추가
- ✅ NVML idle smoke capture와 observability summary 검증
- ✅ Ollama `gemma3:12b`로 실측 SWMA multi-GPU 후보에 대해 설명형 판정 검증 (`_backend: ollama:gemma3:12b`)
- (선택) `--rag-backend sbert`로 임베딩 검색 품질 추가 향상 가능
- ✅ Physics Tier B Phase 1: Kundur 7개 프로필 중 6개 수렴, 30% burst는
  `converged=false`로 기록
- ✅ Kundur 스윕: 28개 중 26개 수렴. 전역 `osc_std` peak는 0.1 Hz로,
  약 0.6 Hz 전역 봉우리는 확인되지 않음
- ✅ WECC 179 GENCLS 스윕: 28/28 수렴. 전역 peak는 1.05 Hz로,
  별도 공식 gallery 사례의 0.37 Hz 전역 봉우리를 재현하지 못함
- ⚠ Cyber–physics join의 사용 가능 공격군은 3개뿐이므로 상관계수는 기술통계이며,
  PTPS 수치 가중치는 확정하지 않음

### RTX 4090 제약

- `DCGM_FI_PROF_*`, NVLink, MIG는 사용하지 않는다.
- 메모리 사용량은 NVML을 우선 사용한다.
- 시스템 nvcc 11.5는 Ada `sm_89` 커널 컴파일에 부적합하므로, 실측 workload는
  `.venv`의 CUDA 지원 PyTorch를 사용한다.
- `dataset/real/`은 용량과 개인정보 가능성 때문에 Git에서 제외된다.

## Physics Tier B — 공개 동적 테스트 계통 검증

Cyber Stage 1은 GPU 텔레메트리 후보 선별이다. Physics Tier B에는 세 갈래가 있다.

### 운영 모드 (이벤트 트리거)

Stage 1이 후보로 잡은 **바로 그 윈도우**의 관측 `power(t)`를 공개 테스트계통에
온디맨드로 재생한다. 사후 `attack_id` 매칭이 아니라, 한 번의
`run_pipeline.py` 실행 결과 dict에 `verdict`와 `physics_*`가 함께 들어간다.
기본값은 off이며, ANDES가 있는 `.venv-physics`에서 켠다.

```bash
source .venv-physics/bin/activate
python pipeline/run_pipeline.py \
  --telemetry dataset/synthetic/all_v3.csv --baseline-mean 120 \
  --rag-backend tfidf --llm-backend stub \
  --physics-validate --physics-timeout-s 60 \
  --out dataset/eval/cyber_physics_ops.json
```

### IBR 침투율 스윕 (교체, WECC)

소량 IBR **추가**는 용량 비중이 무시할 수준(~0.01%)이라 폐기한다. 대신 공개
WECC 179 GENCLS 동기기를 Sn 내림차순으로 REGCA1(+REECA1)로 **교체**해 용량 기준
침투율(0/30/50/70%)과 관성(`M` 합) 감소를 만든 뒤, 공격 주파수 × 침투율 2D
응답을 본다. KPG-193(Tier A)과 섞어 서술하지 않는다.

```bash
source .venv-physics/bin/activate
python -m bit2watt_impl.physics.ibr_baseline
python -m bit2watt_impl.physics.ibr_frequency_sweep
```

산출: `dataset/eval/ibr_penetration_baseline.csv`,
`ibr_frequency_2d_sweep.csv`, heatmap/overlay PNG, summary JSON.
고침투율에서 PQ 공격 TDS가 실패하면 해당 칸은 `converged=false`로 남긴다.

공간도(지리 좌표 없음 → 선로 그래프 spring layout):

```bash
python -m bit2watt_impl.physics.plot_wecc_ibr_map
# → dataset/eval/wecc_ibr_topology_map.png
# → dataset/eval/wecc_ibr_replacement_order.png
```

### §8 연구 실험 모드 (고정 시나리오 + 스윕)

사전 정의 AttackProfile / 주파수 스윕을 배치로 돌린 뒤, cyber 결과와
`attack_id`로 조인한다. 두 파이프라인의 ID 체계가 다르면 unmatched가 나오며,
이는 운영 트리거 실패가 아니라 실험 설계의 한계다.
`pipeline/physics_correlation.py`는 이 실험 모드 전용이다.

배치 스윕도 GPU W→계통 MW 환산이 아니라 상대 부하 파형 주입이다.
`cyber_amplitude_frac`은 관측 GPU 파형 설명용이고 `amplitude_frac`은 공개
테스트계통 민감도 주입값이며, 둘을 실제 설비 환산으로 해석하면 안 된다.

ANDES 2.0은 Python 3.11 이상이 필요하므로 기존 `.venv`와 분리한다.

```bash
./setup_physics_env.sh
source .venv-physics/bin/activate

# Phase 1: Kundur 정상/주기/버스트/램프 및 GridPulse 공격 프로필
python -m bit2watt_impl.physics.stage1_kundur

# Phase 2: Kundur 0.1~1.45 Hz 스윕
python -m bit2watt_impl.physics.frequency_sweep

# Phase 3: WECC smoke 후 주파수 스윕 (IBR 교체 없음)
python -m bit2watt_impl.physics.stage3_wecc
```

Cyber 결과와 결합 (**§8 실험 모드만**; 운영은 `--physics-validate` 사용):

```bash
source .venv/bin/activate
python dataset/build_synthetic.py
python pipeline/run_pipeline.py \
  --telemetry dataset/synthetic/all_v3.csv --baseline-mean 120 \
  --rag-backend tfidf --llm-backend stub \
  --out dataset/eval/cyber_pipeline_results.json

python pipeline/physics_correlation.py \
  --cyber-json dataset/eval/cyber_pipeline_results.json \
  --physics-csv dataset/eval/kundur_attack_features.csv \
  --out dataset/eval/cyber_physics_joined.csv

source .venv-physics/bin/activate
python pipeline/eval_physics_correlation.py
```

Tier B 결과의 `anomaly_score`는 검증된 물리 위험도가 아니라 Cyber Stage 1의
투명한 0~1 스크리닝 휴리스틱이다. 표본이 부족하면 PTPS 수치 가중치를 확정하지
않고 방향성 제안만 저장한다. 수렴 실패는 누락하지 않고 `converged=false`와
`failure_reason`으로 남긴다.

### 해석 제한

- 결과는 반드시 “공개 동적 테스트 계통(Kundur/WECC)에서”로 한정한다.
- 한국 실제 전력망의 주파수 하락이나 피해 규모로 일반화하지 않는다.
- KPG-193 Tier A(정적·위치별 민감도)와 Tier B(동적·범용 테스트 계통)는 별개다.
- Kundur 내장 케이스의 기존 t=2초 선로 Toggle은 실험 파형과 원인을 분리하기 위해
  비활성화하며, 이 사실을 결과 CSV에 기록한다.

## 확장 (제품 비전)

- `corpus/`에 문서 추가만으로 신종 위협 대응 (재학습 불필요) — MA-IDS의 "경험 라이브러리" 패턴
- 커넥터 아키텍처로 향후 운영 매뉴얼/위협 인텔리전스/외부 데이터 소스를 RAG 컨텍스트에 연결 가능
