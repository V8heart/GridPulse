# GridPulse 전력 텔레메트리 데이터셋 표준 v1

- 표준 ID: `gp-telemetry/1.1` (v1.0 대비 변경: progress log 정책, §5·§6.1·§11·§12)
- 적용 대상: 실측(`source=real`)과 합성(`source=synthetic`) **모두**. 합성 생성기도 이 표준을 따른다.
- 레포 위치(권장): `docs/dataset_standard_v1.md`
- 이 문서가 코드와 충돌하면 이 문서가 우선이다. 표준을 바꿀 때는 버전을 올리고 §12 변경 이력에 남긴다.

---

## 1. 핵심 원칙: 정보는 세 계층으로 분리한다

| 계층 | 뜻 | 예 | 파이프라인 추론에서 사용 |
|---|---|---|---|
| **observed** | 센서·OS가 실제로 관측한 값 | 전력, 사용률, 실제 GPU 사용 프로세스 | 가능 |
| **declared** | 사용자·스케줄러가 신고한 값. 거짓일 수 있다 | 선언 작업 유형, 선언 사용자, 요청 GPU 수 | 가능 |
| **ground truth (gt)** | 채점에만 쓰는 정답 | 공격 여부, 공격 종류, 공격 파라미터 | **금지** |

규칙:

1. gt 정보는 `gt_` 접두사 컬럼, 그리고 §6의 정답 색인 파일에만 존재한다.
2. observed와 declared는 **절대 같은 컬럼을 공유하지 않는다.** (예: 실제 프로세스 이름 `observed_process_name` vs 선언 프로세스 이름 `declared_process_name`)
3. 추론 경로에 들어가는 모든 문자열(식별자, 파일 경로, 선언값)은 gt를 암시하면 안 된다. 공격 세션의 declared 값은 정상 작업 풀에서 뽑는다(§7).

---

## 2. 식별자 규칙

| 식별자 | 형식 | 규칙 |
|---|---|---|
| `session_id` | `s-` + 16자리 소문자 hex (예: `s-3f9a0c1d7e2b4a65`) | 무작위 생성. 워크로드 이름·라벨·날짜·순번 등 **의미 있는 단어 금지** |
| `run_id` | `r-` + `YYYYMMDD` + `-` + 6자리 hex | 한 번의 캡처 배치(여러 세션)를 묶는 ID. 날짜는 허용, 워크로드 이름은 금지 |
| `window_id` | `{session_id}:gpu{gpu_id}:{start_idx}-{end_idx}` | 파이프라인이 파생 생성 |
| `gpu_id` | 정수 0부터 | NVML 인덱스 |
| `gpu_uuid` | NVML UUID 문자열 | 서로 다른 호스트·재부팅 간 GPU 식별용 |

합성 세션도 같은 형식을 쓴다. 기존 `syn-normal-sync-ddp-80600` 같은 ID는 폐기한다.

---

## 3. 디렉터리 구조

```
dataset/
  real/                          # git 제외 (용량·개인정보)
    sessions/
      s-3f9a0c1d7e2b4a65/
        telemetry.csv            # §4
        progress.raw.jsonl       # §5, 워크로드 원본 로그. 모든 세션이 가짐. 수정 금지, 파이프라인은 읽지 않음
        progress.jsonl           # §5, 마스크를 통과한 세션에만 존재. `t`는 finalize가 계산
        session.json             # §6.1, 공개 메타데이터 (gt 없음)
        workload_stdout.log      # 워크로드 원본 출력 (gt 포함 가능, 추론 금지)
    index/
      labels.csv                 # §6.2, 정답 색인 (채점 전용)
      runs.csv                   # run_id별 캡처 조건 요약
    splits/
      split_manifest.json        # §9
  synthetic/
    sessions/ ...                # 같은 구조
    index/ ...
    splits/ ...
```

- **라벨 이름으로 된 폴더를 만들지 않는다.** (`dataset/real/swma/...` 금지) 라벨별로 보고 싶으면 `index/labels.csv`로 필터링한다.
- 여러 세션을 합친 분석용 CSV(`all_v3.csv` 등)는 파생물이다. 원본은 항상 `sessions/` 아래 세션별 파일이다.

---

## 4. `telemetry.csv` 스키마

형식: long format. **한 행 = (한 시점, 한 GPU)**. 세션에 GPU가 2개면 매 폴링마다 2행이 생긴다.
인코딩 UTF-8, 구분자 쉼표, 헤더 1줄. 결측은 빈 칸(`NaN`). **결측을 0으로 채우지 않는다.**

### 4.1 시간·식별 (필수)

| 컬럼 | 단위/타입 | 계층 | 설명 |
|---|---|---|---|
| `timestamp` | s, float | meta | 세션 시작(t0) 기준 경과 시간. `time.monotonic()` 차이로 계산. 세션·GPU 내에서 단조 증가 |
| `t_epoch` | s, float | meta | `t0_epoch + timestamp`. 벽시계(UNIX epoch). progress log와 맞추는 기준 |
| `session_id` | str | meta | §2 |
| `gpu_id` | int | meta | §2 |
| `gpu_uuid` | str | meta | §2 |
| `gpu_model` | str | observed | NVML 장치 이름 그대로 (예: `NVIDIA GeForce RTX 4090`). 합성은 `synthetic-<모델>` |
| `sample_hz` | Hz, float | meta | 세션·GPU별 **실제** 폴링 간격 중앙값의 역수. 요청값이 아님. 세션·GPU 내 단일 값 |
| `requested_interval_ms` | ms | meta | 요청한 폴링 간격 |
| `actual_interval_ms` | ms | meta | 직전 샘플과의 실제 간격. 첫 행은 결측 |
| `value_changed` | bool | meta | 직전 샘플 대비 수치 필드가 하나라도 바뀌었는지 |
| `warmup` | bool | meta | 워크로드 시작 후 `warmup_s`(기본 120초) 이내인지. finalize 단계에서 계산 |

### 4.2 observed 텔레메트리

| 컬럼 | 단위 | 필수 | NVML 출처 / 비고 |
|---|---|---|---|
| `power_w` | W | 필수 | `nvmlDeviceGetPowerUsage` / 1000. Ada 세대는 약 1초 평균값 |
| `power_instant_w` | W | 선택 | `nvmlDeviceGetFieldValues(NVML_FI_DEV_POWER_INSTANT)`. 미지원이면 컬럼은 두고 전부 결측 |
| `util_gpu_pct` | % | 필수 | `nvmlDeviceGetUtilizationRates().gpu` |
| `mem_copy_util_pct` | % | 필수 | `nvmlDeviceGetUtilizationRates().memory` |
| `sm_clock_mhz` | MHz | 필수 | `nvmlDeviceGetClockInfo(SM)` |
| `mem_clock_mhz` | MHz | 선택 | `nvmlDeviceGetClockInfo(MEM)` |
| `temp_c` | °C | 필수 | `nvmlDeviceGetTemperature(GPU)` |
| `fb_used_mb` | MiB | 필수 | `nvmlDeviceGetMemoryInfo().used` |
| `fan_speed_pct` | % | 선택 | `nvmlDeviceGetFanSpeed` |
| `pstate` | int | 선택 | `nvmlDeviceGetPerformanceState` |
| `power_limit_w` | W | 선택 | `nvmlDeviceGetEnforcedPowerLimit` / 1000 |
| `throttle_reasons` | int (bitmask) | 선택 | `nvmlDeviceGetCurrentClocksThrottleReasons` (신규 API명은 EventReasons) |
| `observed_pid` | int | 선택 | 이 GPU에서 메모리를 가장 많이 쓰는 compute 프로세스 PID |
| `observed_process_name` | str | 선택 | 위 PID의 `/proc/<pid>/comm` |
| `observed_n_procs` | int | 선택 | 이 GPU의 compute 프로세스 수 |

프로세스 조회는 비용이 크므로 매 폴링이 아니라 `proc_poll_s`(기본 1초)마다 갱신하고 사이 행에는 직전 값을 채운다. 이 경우 채운 값임을 나타내는 컬럼은 두지 않고 `session.json`에 `proc_poll_s`를 기록한다.

### 4.3 declared 컨텍스트

| 컬럼 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `declared_job_type` | str | 필수 | §7의 풀에서 선택 (예: `ddp_training`) |
| `declared_job_family` | str | 필수 | `training` / `inference` / `evaluation` / `interactive` |
| `declared_process_name` | str | 필수 | 선언 명령 (예: `torchrun train.py`) |
| `declared_user` | str | 필수 | `user_###` 형식 가명 |
| `declared_gres` | str | 필수 | `gpu:N` |

한 세션 안에서 GPU별로 declared 값이 다를 수 있다(다중 테넌트: GPU0은 학습, GPU1은 추론). 이 경우 GPU별로 다른 값을 행에 기록한다.

### 4.4 ground truth (채점 전용, 추론 전 제거)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `gt_label` | str | §8의 라벨 어휘 |
| `gt_is_attack` | bool | 공격이면 True |
| `gt_variant` | str | 세부 변종 (예: `swma_jitter`) |
| `gt_params_json` | str(JSON) | 생성 파라미터 (주기, duty, 진폭, 모델 크기 등) |

- 다중 테넌트 세션에서 GPU별 gt가 다르면 GPU별로 다르게 기록한다. `gt_label`은 (session_id, gpu_id) 안에서 단일 값이어야 한다.
- `pipeline` 코드는 입력 직후 `dataset.schema.strip_ground_truth()`로 `gt_` 컬럼과 레거시 금지 컬럼을 제거한다.

### 4.5 폐기 컬럼 (새 데이터에 쓰지 않음)

`label`, `attack_id`, `job_type`, `id_user`, `gres_req`, `process_name`, `waveform_*`, `raw_timestamp`, `collection_timestamp`.
기존 데이터 읽기 호환을 위해 `normalize_frame()`의 매핑은 남기되, 매핑 대상은 다음과 같다.

| 레거시 | 새 컬럼 |
|---|---|
| `label` | `gt_label` |
| `attack_id` | `gt_variant` |
| `job_type` | `declared_job_type` |
| `id_user` | `declared_user` |
| `gres_req` | `declared_gres` |
| `process_name` | **`observed_process_name`** (기존처럼 declared로 보내지 않는다) |
| `waveform_*` | `gt_params_json`에 병합 |

---

## 5. `progress.jsonl` 스키마

워크로드가 스스로 남기는 진행 로그. 한 줄 = 한 이벤트 JSON 객체.

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `t_epoch` | float | 필수 | 이벤트 시각. `time.time()` |
| `t` | float | finalize 후 필수 | `t_epoch - t0_epoch`. 텔레메트리 `timestamp`와 같은 축. finalize 단계에서 채운다 |
| `gpu_id` | int | 필수 | 이벤트를 낸 GPU (DDP면 rank → GPU 매핑) |
| `event` | str | 필수 | 아래 어휘 |
| `step` | int | step 이벤트 필수 | 학습 step 번호 |
| `extra` | object | 선택 | 배치 크기, 토큰 수, request_id 등 |

이벤트 어휘 (이 외의 이름 금지):

| event | 발생 시점 | 필수 여부 |
|---|---|---|
| `workload_start` / `workload_end` | 워크로드 본 루프 시작/종료 | 로그를 쓰는 모든 워크로드 |
| `step_end` | 옵티마이저 step 완료 (`torch.cuda.synchronize()` 후) | 학습 워크로드 |
| `eval_start` / `eval_end` | 평가 구간 | 해당 시 |
| `checkpoint_start` / `checkpoint_end` | 체크포인트 저장 | 해당 시 |
| `request_in` / `request_out` | 추론 요청 도착/응답 완료 (`extra.request_id`, `extra.tokens`) | 추론 워크로드 |
| `phase_change` | 학습률 변경, 배치 변경 등 | 선택 |

규칙:

1. **로그 존재 여부는 gt와 독립이어야 한다.** 캡처 시점에는 **모든 세션이 원본(`progress.raw.jsonl`)을 기록한다.** 파이프라인에 노출할지는 finalize/평가 시점의 결정론적 마스크(`dataset/progress_log_policy.py`, sha256(seed:session_id) 기반, 기본 drop 0.4)로 정하며, 이 확률은 모든 라벨에 동일하다. 원본은 항상 보존한다. (원본 로그가 원래 없는 워크로드를 마스크 대상에서 제외하는 방식은 클래스별 로그 보유율 차이를 만들므로 **금지**한다.)
2-1. **워크로드는 이벤트에 `t`를 쓰지 않는다.** `t_epoch`만 기록하고, `t = t_epoch - t0_epoch`는 finalize가 계산한다. 파이프라인은 워크로드가 쓴 `t`를 신뢰하지 않는다.
2. 학습 루프 안에 숨는 공격(`ltma`, `swma_piggyback`, `swma_mimicry`)은 숙주 정상 작업의 로그를 그대로 남긴다. 공격 코드 자체는 로그를 남기지 않는다.
3. 공격이 단독 프로세스로 도는 경우(`swma`, `cryptojacking` 등)도 로그 정책은 동일하게 적용한다. **모든 세션에서** 원본 로그가 있어야 하므로, 선언된 작업 유형에 맞는 **위장 로그**(예: 학습이라고 선언했으면 `step_end`를 일정 간격으로 기록)를 남긴다. 위장 로그의 주기와 실제 전력 주기가 어긋나는 것이 탐지 대상 증거다.

---

## 6. 세션 메타데이터와 정답 색인

### 6.1 `session.json` (공개 메타, gt 금지)

```json
{
  "schema_version": "gp-telemetry/1.0",
  "session_id": "s-3f9a0c1d7e2b4a65",
  "run_id": "r-20260925-a1b2c3",
  "source": "real",
  "t0_epoch": 1790000000.123456,
  "duration_s": 912.4,
  "warmup_s": 120,
  "host": {"hostname_hash": "sha256:...", "os": "Ubuntu 22.04"},
  "gpus": [
    {"gpu_id": 0, "gpu_uuid": "GPU-...", "gpu_model": "NVIDIA GeForce RTX 4090",
     "power_limit_w": 450.0, "vbios": "..."}
  ],
  "software": {"driver": "550.xx", "cuda": "12.x", "torch": "2.x", "nvml": "12.x"},
  "collector": {"requested_interval_ms": 100, "proc_poll_s": 1.0,
                "power_instant_supported": true},
  "progress_log": {"present": true, "masked": false, "policy_version": "v1_eligible_uniform_drop", "drop_prob": 0.4, "seed": 7},
  "preflight": {"other_gpu_processes": [], "idle_power_w": {"0": 22.1, "1": 21.8}},
  "declared": {"0": {"declared_job_type": "ddp_training", "...": "..."}}
}
```

실행 명령(`workload_cmd`)은 워크로드 이름을 포함하므로 여기 넣지 않고 정답 색인에 넣는다.

### 6.2 `index/labels.csv` (채점 전용)

| 컬럼 | 설명 |
|---|---|
| `session_id`, `gpu_id` | 키 |
| `gt_label`, `gt_is_attack`, `gt_variant`, `gt_params_json` | §4.4와 동일 |
| `workload_module`, `workload_cmd` | 실제 실행한 모듈과 명령 |
| `host_workload` | piggyback/ltma의 숙주 정상 워크로드 라벨 (없으면 빈 칸) |
| `run_id`, `source`, `created_epoch` | 추적용 |
| `valid` | 검증(§10) 통과 여부. 실패 세션은 삭제하지 않고 `valid=false`로 남긴다 |
| `invalid_reason` | 실패 사유 |

---

## 7. declared 컨텍스트 부여 정책

- 모든 세션은 `dataset.declared_context.sample_declared(true_family, rng, disguise=...)`로 declared 값을 정한다. (합성과 실측이 같은 함수를 쓴다.)
- 정상 세션: `disguise=False`, 단 `mismatch_rate=0.1`로 일부는 실제와 다른 계열을 선언한다 (현실의 부정확한 신고를 반영).
- 공격 세션: `disguise=True`. 정상 작업 풀에서 무작위로 선언한다.
- 숙주에 기생하는 공격(piggyback, ltma)은 숙주 작업의 declared 값을 그대로 쓴다.
- `declared_user`는 정상과 공격이 같은 사용자 풀을 공유한다.

---

## 8. 라벨 어휘 (`gt_label`)

정상 (`gt_is_attack=False`):

| gt_label | 설명 |
|---|---|
| `normal_idle` | 부하 없음 |
| `normal_mlp_small` | 소형 MLP 학습 (기존 뼈대 워크로드) |
| `normal_resnet_train` | ResNet 학습 (단일/DDP) |
| `normal_llm_pretrain_ddp` | GPT류 사전학습, 2 GPU DDP |
| `normal_llm_pretrain_fsdp` | GPT류 사전학습, 2 GPU FSDP |
| `normal_llm_flat_pretrain` | 단일 GPU 대배치 연속 학습 (평탄 고부하) |
| `normal_llm_finetune` | 미세조정 + 주기적 eval/checkpoint |
| `normal_llm_inference_serving` | 포아송 도착 요청 처리 |
| `normal_llm_inference_batch` | 일괄 추론 |
| `normal_hpo_search` | 하이퍼파라미터 탐색 |
| `normal_dataloader_bound` | 데이터로더 병목 학습 |
| `normal_mixed_tenants` | GPU별로 다른 정상 작업 동시 실행 |

공격 (`gt_is_attack=True`):

| gt_label | gt_variant 예 |
|---|---|
| `swma` | `swma_basic`, `swma_shallow`, `swma_jitter`, `swma_piggyback`, `swma_mimicry`, `swma_coordinated` |
| `ltma` | `ltma_basic` |
| `cryptojacking` | `crypto_flat` |

새 라벨을 추가할 때는 이 표와 `pipeline/label_to_doc.py`를 함께 갱신한다.

---

## 9. 분할(split) 규칙

- 분할 단위는 **세션**이다. 같은 세션의 창이 여러 분할에 걸치면 안 된다.
- 같은 `run_id` 안에서 파라미터만 바꿔 반복한 세션들은 서로 매우 비슷하므로, 가능하면 `run_id` 단위로 묶어 분할한다(group split).
- 분할: `train` / `cal` / `test` / `holdout_*`. `holdout_*`은 학습에 없는 파라미터나 변종(예: `holdout_unseen_period`, `holdout_mimicry`).
- `split_manifest.json`에 각 분할의 session_id 목록, 생성 seed, 생성 시각, 그리고 `test`와 `holdout_*` 목록의 SHA-256 해시를 기록한다. 해시가 바뀌면 평가 스크립트가 경고한다.
- 임계값·가중치·corpus 증거·α는 `train`과 `cal`에서만 정한다. `test`는 최종 평가 때 한 번만 쓴다.
- 실측과 합성은 분할을 따로 만든다. 보고할 때 "합성으로 적합 → 실측 test"와 "실측 train/cal → 실측 test"를 구분해 표기한다.

---

## 10. 검증 규칙 (`dataset/validate_standard.py`)

세션 하나가 다음을 모두 통과해야 `valid=true`다.

1. 필수 컬럼 존재, 타입 변환 가능.
2. `session_id`가 §2 정규식 `^s-[0-9a-f]{16}$`를 만족.
3. 추론 가능 컬럼(observed, declared, meta)의 문자열 값에 금지 토큰이 없음: 라벨 어휘(§8)의 모든 단어, `attack`, `swma`, `ltma`, `crypto`, `hash`, `unknown_cuda`, `malicious`, `synthetic_` 등. (대소문자 무시)
4. `timestamp` 단조 증가, `t_epoch - timestamp`가 세션 내 상수(허용 오차 1 ms).
5. `sample_hz`가 세션·GPU 내 단일 값이고, `actual_interval_ms` 중앙값과 5% 이내로 일치.
6. `power_w` ≥ 0, `util_*` ∈ [0, 100], `temp_c` ∈ [0, 110].
7. `(session_id, gpu_id)` 안에서 `gt_label`, declared 값이 단일.
8. progress log가 있으면: 이벤트 어휘 준수, `t`가 `[0, duration_s + 5]` 범위, 학습 계열 declared면 `step_end`가 2개 이상.
9. `session.json`에 gt 관련 키가 없음 (검증 3의 금지 토큰 검사를 JSON 전체 문자열에 적용).
10. 워크로드 구간(`workload_start`~`workload_end`, 로그가 없으면 캡처 구간 전체)이 `warmup_s`보다 길다.

---

## 11. 합성 데이터 준수 사항

- `session_id` 형식(§2) 준수. 라벨을 포함한 기존 ID 폐기.
- `t0_epoch`는 세션마다 서로 다른 무작위 값(예: 기준일 ± 30일 범위)으로 두어, 서로 다른 합성 세션의 시간축이 우연히 정렬되지 않게 한다. 세션 간 상관은 원칙적으로 계산하지 않는다.
- 모든 합성 세션이 원본 progress 로그를 갖는다(원본이 없던 생성기에는 위장 로그 추가). 노출 여부는 §5 규칙 1의 마스크를 따른다. 예외는 `interactive` 선언의 `normal_idle`뿐이며 리포트에 명시한다.
- 공격 세션의 declared는 `pool_random`과 `host_family_matched`(training 계열만 선언)를 절반씩 생성하고 `gt_params_json.declared_policy`에 기록한다.
- `gpu_model`은 `synthetic-rtx4090` 등으로 명시해 실측과 섞이지 않게 한다.

---

## 12. 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| 1.0 | 2026-09-21 | 최초 작성. 3계층 분리, 불투명 식별자, 시간축, progress log 어휘, 정답 색인 분리 |
| 1.1 | 2026-09-21 | progress log: 캡처 시 항상 기록 + finalize 마스크로 노출 결정(원본 보존), 원본 없는 워크로드 제외 금지, 위장 로그 필수, 워크로드의 `t` 기록 금지. 공격 `declared_policy` 층화 추가 |
