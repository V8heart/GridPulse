# GridPulse 전력 텔레메트리 데이터셋 표준 v1.2

- 표준 ID: `gp-telemetry/1.2`
- 적용 대상: 실측(`source=real`)과 합성(`source=synthetic`) **모두**. 합성 생성기도 이 표준을 따른다.
- 레포 위치: `dataset/dataset_standard_v1_2.md` (v1.1 역사본: `dataset/dataset_standard_v1_1.md`)
- 이 문서가 코드와 충돌하면 이 문서가 우선이다. 표준을 바꿀 때는 버전을 올리고 §12 변경 이력에 남긴다.

---

## 1. 핵심 원칙: 정보는 세 계층으로 분리한다

| 계층 | 뜻 | 예 | 파이프라인 추론에서 사용 |
|---|---|---|---|
| **observed** | 센서·OS가 실제로 관측한 값 | 전력, 사용률, 실제 GPU 사용 프로세스 | 가능 |
| **declared** | 사용자·스케줄러가 신고한 값. 거짓일 수 있다 | 선언 작업 유형, 선언 사용자, 요청 GPU 수 | 가능 |
| **ground truth (gt)** | 채점에만 쓰는 정답 | 공격 여부, 공격 종류, 공격 파라미터, 공격 활성 구간, GPU 역할 | **금지** |

규칙:

1. gt 정보는 `gt_` 접두사 컬럼, §6의 정답 색인, 그리고 `private/`·`.staging/` 비공개 기록에만 존재한다.
2. observed와 declared는 **절대 같은 컬럼을 공유하지 않는다.**
3. 추론 경로에 들어가는 모든 문자열(식별자, 파일 경로, 선언값)은 gt를 암시하면 안 된다.
4. 파이프라인·검증기의 세션 탐색은 `sessions/`와 `index/`만 본다. `.staging/`과 `private/`는 절대 읽지 않는다.

---

## 2. 식별자 규칙

| 식별자 | 형식 | 규칙 |
|---|---|---|
| `session_id` | `s-` + 16자리 소문자 hex | 무작위(실측) 또는 seed 결정론(합성). 워크로드·라벨·날짜·순번 금지 |
| `run_id` | `r-` + `YYYYMMDD` + `-` + 6자리 hex | 운영 배치 ID. **분할 그룹으로 쓰지 않는다** |
| `group_id` | `g-` + 16자리 소문자 hex | matrix entry + 파라미터 군집 해시. repeat에 의존하지 않음. 같은 group은 항상 같은 split/fold |
| `capture_key` | 문자열(JSON에 기록) | config hash + entry index + repeat + params. resume 키 |
| `window_id` | `{session_id}:gpu{gpu_id}:{start_idx}-{end_idx}` | 파이프라인 파생 |
| `gpu_id` | 정수 0부터 | NVML 인덱스 |
| `gpu_uuid` | NVML UUID 문자열 | 호스트·재부팅 간 GPU 식별 |

---

## 3. 디렉터리 구조

```
dataset/
  real/                          # git 제외
    sessions/
      s-3f9a0c1d7e2b4a65/
        telemetry.csv
        progress.raw.jsonl       # 원본. 수정 금지. 파이프라인은 읽지 않음
        progress.jsonl           # 마스크 통과 세션만. t는 finalize가 계산
        session.json             # 공개 메타 (gt 없음)
    .staging/                    # 캡처 중 임시. sessions 밖. 권한 0600
      s-3f9a0c1d7e2b4a65/
        capture_private.json
        collector_result.json
    private/                     # 영구 비공개. git 제외. 파이프라인 금지
      capture/{session_id}.json  # finalize 성공 후 staging에서 원자 이동
      stdout/{session_id}.log    # 워크로드 원본 출력 (gt 단서 가능)
      probes/                    # micro-batch probe 캐시
    index/
      labels.csv
      runs.csv
    splits/
      split_manifest.json
  synthetic/
    sessions/ ...
    index/ ...
    splits/ ...
```

- **라벨 이름 폴더를 만들지 않는다.**
- v1.1의 `sessions/*/workload_stdout.log`는 누설 통로였다 폐기한다. v1.2 canonical 경로는 `private/stdout/`.
- valid 세션: labels 기록 + 검증 통과 후 staging → `private/capture/` 원자 이동.
- invalid/실패 세션: staging 유지 + 가능하면 private snapshot 보존.
- `python -m dataset.remask`는 raw log와 session.json 공개 메타만으로 visible progress를 다시 만든다.

---

## 4. `telemetry.csv` 스키마

형식: long format. **한 행 = (한 시점, 한 GPU)**.
인코딩 UTF-8, 구분자 쉼표, 헤더 1줄. 결측은 빈 칸. **결측을 0으로 채우지 않는다.**

### 4.1 시간·식별 (필수)

| 컬럼 | 단위/타입 | 계층 | 설명 |
|---|---|---|---|
| `timestamp` | s, float | meta | 세션 시작(t0) 기준. `time.monotonic()` 차이 |
| `t_epoch` | s, float | meta | `t0_epoch + timestamp` |
| `session_id` | str | meta | §2 |
| `gpu_id` | int | meta | §2 |
| `gpu_uuid` | str | meta | §2 |
| `gpu_model` | str | observed | NVML 이름. 합성은 `synthetic-rtx4090` |
| `sample_hz` | Hz, float | meta | GPU별 실제 폴링 간격 중앙값의 역수 |
| `requested_interval_ms` | ms | meta | 요청 폴링 간격 |
| `actual_interval_ms` | ms | meta | 직전 샘플과의 실제 간격. 첫 행 결측 |
| `value_changed` | bool | meta | 직전 대비 수치 변경 여부 |
| `warmup` | bool | meta | 워크로드 시작 후 `warmup_s`(기본 **180초**, smoke/관측 후 고정) 이내 |

### 4.2 observed 텔레메트리

| 컬럼 | 단위 | 필수 | NVML 출처 / 비고 |
|---|---|---|---|
| `power_w` | W | 필수 | `nvmlDeviceGetPowerUsage` / 1000 |
| `power_instant_w` | W | 선택 | `NVML_FI_DEV_POWER_INSTANT`. 미지원이면 전결측 |
| `util_gpu_pct` | % | 필수 | utilization.gpu |
| `mem_copy_util_pct` | % | 필수 | utilization.memory |
| `sm_clock_mhz` | MHz | 필수 | SM clock |
| `mem_clock_mhz` | MHz | 선택 | MEM clock |
| `temp_c` | °C | 필수 | GPU temperature |
| `fb_used_mb` | MiB | 필수 | memory.used |
| `fan_speed_pct` | % | 선택 | fan speed |
| `pstate` | int | 선택 | performance state |
| `power_limit_w` | W | 선택 | enforced power limit / 1000 |
| `throttle_reasons` | int | 선택 | clocks throttle/event reasons bitmask |
| `observed_pid` | int | 선택 | 최대 메모리 compute PID |
| `observed_process_name` | str | 선택 | `/proc/<pid>/comm` |
| `observed_n_procs` | int | 선택 | compute 프로세스 수 |

프로세스 조회는 `proc_poll_s`(기본 1초)마다 갱신하고 사이 행은 직전 값을 채운다.

### 4.3 declared 컨텍스트

| 컬럼 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `declared_job_type` | str | 필수 | §7 풀 |
| `declared_job_family` | str | 필수 | `training` / `inference` / `evaluation` / `interactive` |
| `declared_process_name` | str | 필수 | 선언 명령 |
| `declared_user` | str | 필수 | `user_###` |
| `declared_gres` | str | 필수 | `gpu:N` |

GPU별로 다를 수 있다(다중 테넌트).

### 4.4 ground truth (채점 전용)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `gt_label` | str | §8 |
| `gt_is_attack` | bool | 창/행 단위 공격 여부(창 정답은 §4.6) |
| `gt_variant` | str | 세부 변종 |
| `gt_params_json` | str(JSON) | 생성 파라미터 |

telemetry 행의 `gt_*`는 finalize가 labels와 맞춰 붙인다. **공격 활성 구간 자체는 telemetry에 넣지 않고** labels/private에 둔다.

### 4.5 폐기 컬럼

`label`, `attack_id`, `job_type`, `id_user`, `gres_req`, `process_name`, `pid`, `waveform_*`, `raw_timestamp`, `collection_timestamp`.

읽기 호환 매핑:

| 레거시 | 새 컬럼 |
|---|---|
| `label` | `gt_label` |
| `attack_id` | `gt_variant` |
| `job_type` | `declared_job_type` |
| `id_user` | `declared_user` |
| `gres_req` | `declared_gres` |
| `process_name` | **`observed_process_name`** |
| `pid` | `observed_pid` |
| `waveform_*` | `gt_params_json`에 병합 |

### 4.6 창 단위 정답

세션 라벨을 모든 창에 복사하지 않는다.

1. 공격 워크로드는 GPU별 활성 구간을 `[start_epoch, end_epoch]` 목록(`gt_attack_intervals_epoch`)으로 기록한다.
2. SWMA: 각 active duty 구간. LTMA: 각 삽입 구간. crypto: 실제 연산 구간.
3. piggyback/mimicry의 host-only 선행 구간은 공격이 아니다.
4. 창과 공격 구간 합집합의 duration overlap ≥ 창 길이의 50%이면 창을 공격으로 라벨링한다.
5. hosted 공격의 비활성 창 → host 정상 label. standalone 비활성/pre/post → `normal_idle`.
6. interval이 없는 레거시만 `truth_source=session_legacy` fallback.
7. 창 샘플의 ≥50%가 warmup이면 `warmup=True`. 대표 지표는 warmup 제외, inclusive는 별도 열.

---

## 5. `progress.jsonl` 스키마

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `t_epoch` | float | 필수 | `time.time()` |
| `t` | float | finalize 후 필수 | `t_epoch - t0_epoch` |
| `gpu_id` | int | 필수 | |
| `event` | str | 필수 | 아래 어휘 |
| `step` | int | step 이벤트 필수 | |
| `extra` | object | 선택 | |

이벤트 어휘: `workload_start`, `workload_end`, `step_end`, `eval_start`, `eval_end`, `checkpoint_start`, `checkpoint_end`, `request_in`, `request_out`, `phase_change`.

규칙:

1. 캡처 시 모든 비-idle 세션이 `progress.raw.jsonl`을 갖는다. 노출은 finalize 마스크(`v1_1_uniform_all_sessions`, sha256(seed:session_id), 기본 drop 0.4)로 결정. 예외는 `normal_idle`(interactive)뿐.
2. 워크로드는 `t`를 쓰지 않는다. 파이프라인은 워크로드가 쓴 `t`를 신뢰하지 않는다.
3. hosted 공격은 숙주 로그만 남긴다. standalone 공격은 선언에 맞는 위장 로그를 남긴다(전력 주기와 독립).

---

## 6. 세션 메타데이터와 정답 색인

### 6.1 `session.json` (공개 메타, gt 금지)

```json
{
  "schema_version": "gp-telemetry/1.2",
  "session_id": "s-3f9a0c1d7e2b4a65",
  "run_id": "r-20260925-a1b2c3",
  "source": "real",
  "t0_epoch": 1790000000.123456,
  "duration_s": 600.0,
  "warmup_s": 180,
  "host": {"hostname_hash": "sha256:...", "os": "Ubuntu 22.04"},
  "gpus": [
    {"gpu_id": 0, "gpu_uuid": "GPU-...", "gpu_model": "NVIDIA GeForce RTX 4090",
     "power_limit_w": 450.0, "vbios": "..."}
  ],
  "software": {"driver": "550.xx", "cuda": "12.x", "torch": "2.x", "nvml": "12.x", "env": {}},
  "collector": {"requested_interval_ms": 100, "proc_poll_s": 1.0,
                "power_instant_supported": true},
  "progress_log": {"present": true, "masked": false,
                   "policy_version": "v1_1_uniform_all_sessions",
                   "drop_prob": 0.4, "seed": 7},
  "preflight": {"other_gpu_processes": [], "idle_power_w": {"0": 22.1, "1": 21.8}},
  "declared": {"0": {"declared_job_type": "ddp_training", "declared_job_family": "training"}}
}
```

### 6.2 `index/labels.csv` (채점 전용)

| 컬럼 | 설명 |
|---|---|
| `session_id`, `gpu_id` | 키 |
| `gt_label`, `gt_is_attack`, `gt_variant`, `gt_params_json` | §4.4 |
| `gt_attack_intervals_epoch` | JSON 목록 `[[start,end], ...]` (GPU별) |
| `gpu_role` | `target` 또는 `companion_idle` |
| `group_id`, `capture_key` | §2 |
| `declared_policy` | `honest` / `pool_random` / `host_family_matched` / `host_inherited` |
| `declared_job_type` | GPU별 선언 작업 유형 (§7 풀). 정상 honest는 워크로드 정직 매핑 |
| `workload_module`, `workload_cmd`, `host_workload` | 실행 정보 |
| `run_id`, `source`, `created_epoch` | 추적 |
| `valid`, `invalid_reason` | §10 |

### 6.3 companion GPU

- 워크로드 사용 GPU: `target`
- 미사용 GPU: `companion_idle` → `gt_label=normal_idle`, declared family=`interactive`
- DDP/FSDP/coordinated/mixed_tenants 참여 GPU는 전부 `target`
- `gpu_role`은 telemetry/session.json에 넣지 않는다
- 대표 정상 후보율은 target만. companion-inclusive는 별도 열

### 6.4 비공개 `capture_private.json`

- `gt_label`, `gt_variant`, `gt_params_json`, `declared_policy`
- `workload_module`, `workload_cmd`, `host_workload`
- GPU별 `gt_attack_intervals_epoch`, declared, `gpu_role`
- `group_id`, `capture_key`
- 권한 `0600`. session.json/telemetry/pipeline에 복사하지 않음

---

## 7. declared 컨텍스트 부여 정책

- `dataset.declared_context.sample_declared(..., policy=...)`
- 정상 honest: `HONEST_WORKLOAD_TO_JOB_TYPE` 정직 매핑 (family 안 무작위 금지). 풀에 `vision_training`, `dataloader_bound` 포함
- 합성 레거시 `sample_declared(..., policy=honest)`는 여전히 family 풀에서 샘플할 수 있음. 실측 캡처/`run_capture` 정상 경로는 매핑만 사용
- standalone 공격: `pool_random` 또는 `host_family_matched` (위장, 재선언하지 않음)
- hosted 공격(piggyback/mimicry/ltma): `host_inherited` (숙주 declared 복사, 재선언하지 않음)
- 정책은 `gt_params_json.declared_policy` / labels에만 기록. `declared_job_type`은 labels에도 기록

---

## 8. 라벨 어휘 (`gt_label`)

정상: `normal_idle`, `normal_mlp_small`, `normal_resnet_train`, `normal_llm_pretrain_ddp`, `normal_llm_pretrain_fsdp`, `normal_llm_flat_pretrain`, `normal_llm_finetune`, `normal_llm_inference_serving`, `normal_llm_inference_batch`, `normal_hpo_search`, `normal_dataloader_bound`, `normal_mixed_tenants`

공격: `swma`(`swma_basic`/`shallow`/`jitter`/`piggyback`/`mimicry`/`coordinated`), `ltma`(`ltma_basic`), `cryptojacking`(`crypto_flat`)

### 8.1 금지 토큰 (누설 검사)

금지: 전체 gt label/variant 문자열, `attack`, `swma`, `ltma`, `crypto`, `malicious`, `unknown_cuda`, `synthetic_`, `syn-`, `real-`(레거시 session prefix)

허용(정상 declared 어휘): `llm`, `training`, `inference`, `evaluation`, `interactive`, `pretrain`, `finetune`, `ddp`, `hpo`

단순 단어 분해로 정상 declared를 거부하지 않는다.

---

## 9. 분할(split) 규칙

- 분할 단위는 **세션**이다. 같은 세션의 창이 여러 분할에 걸치면 안 된다.
- **`run_id`는 분할 그룹이 아니다.** 밤샘 배치가 하나의 run_id를 공유할 수 있다.
- 분할 그룹은 opaque `group_id`(matrix entry + 파라미터 군집)다. 같은 group_id는 항상 같은 split/fold.
- 분할: `train` / `cal` / `test` / `holdout_*`
- `split_manifest.json`에 session 목록, seed, 생성 시각, test/holdout SHA-256을 기록한다.
- 임계값·가중치·α는 train/cal에서만. test는 동결 후 1회.
- 실측은 group-stratified nested CV(기본 outer 3-fold, inner train/cal)를 기본으로 한다. 필수 stratum group이 부족하면 fold를 줄이지 말고 캡처 부족으로 실패한다.
- 모든 클래스는 동일 duration pool `{480, 600, 720}`초에서 label 독립 seed로 뽑는다.

---

## 10. 검증 규칙 (`dataset/validate_standard.py`)

1. 필수 컬럼·타입.
2. `session_id` ~ `^s-[0-9a-f]{16}$`.
3. observed/declared/meta 문자열에 §8.1 금지 토큰 없음.
4. `timestamp` 단조 증가, `t_epoch - timestamp` 세션 내 상수(±1 ms).
5. `sample_hz`가 GPU 내 단일 값이고 actual_interval 중앙값과 5% 이내.
6. `power_w` ≥ 0, `util_*` ∈ [0,100], `temp_c` ∈ [0,110].
7. `(session_id, gpu_id)` 안 gt_label·declared 단일.
8. progress 있으면 어휘·`t` 범위·학습 declared면 `step_end` ≥ 2.
9. `session.json`에 gt/금지 토큰 없음.
10. 워크로드 구간이 `warmup_s`보다 김.
11. 세션에 `gpu_role=target` GPU가 최소 1개(labels/private 기준).
12. `.staging/`·`private/`를 세션 트리로 오인하지 않음.

---

## 11. 합성 데이터 준수 사항

- opaque `s-` session_id, `synthetic-rtx4090` gpu_model.
- 세션별 고유 `t0_epoch`, event `t_epoch`.
- 모든 비-idle 세션 raw progress. idle 예외만 리포트에 명시.
- 공격 interval + 50% window truth를 실측과 공유.
- declared_policy 50:50 (`pool_random`/`host_family_matched`), hosted는 `host_inherited`.
- 공통 duration pool. metadata-only leakage audit AUC ≤ 0.6 hard gate.

---

## 12. 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| 1.0 | 2026-09-21 | 최초 작성 |
| 1.1 | 2026-09-21 | progress always-raw + finalize mask, 위장 로그 필수, `t` 기록 금지, declared_policy 층화 |
| 1.2 | 2026-09-22 | private/staging 경로, gpu_role, group_id(≠run_id), gt_attack_intervals·창 overlap 정답, stdout을 private로 이전(v1.1 sessions/*/workload_stdout.log는 누설 통로로 폐기), warmup 기본 180s, 공통 duration pool, remask CLI, 금지 토큰 정의 명확화, companion/target 평가 규칙. 실측 정상은 honest + mismatch_rate=0 (합성 기본 10% mismatch는 실측 캡처에 쓰지 않음). 실측 pool_random은 training/inference 선언만 사용 |
| 1.2.1 | 2026-09-26 | labels.csv에 `declared_job_type`. honest 정상은 워크로드 정직 매핑. 풀에 `vision_training`, `dataloader_bound` |
