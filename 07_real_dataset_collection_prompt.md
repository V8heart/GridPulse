# 실측 데이터셋 수집 — 부족한 코드와 구현 계획 (Cursor 프롬프트)

## 현재 리포에 이미 있는 것 (재작성 금지)

| 파일 | 역할 | 상태 |
|---|---|---|
| `bit2watt_impl/collect_telemetry.py` (222줄) | NVML 필드를 long-format CSV로 수집. `requested_interval_ms`, `actual_interval_ms`, `value_changed` 컬럼까지 이미 기록 | ✅ 관측성(Observability) 측정 기반이 이미 있음 |
| `bit2watt_impl/swma_workload.py` | PyTorch 연산/sleep 토글로 SWMA-like 부하 생성 | ✅ |
| `bit2watt_impl/crypto_workload.py` | 지속 고부하 (채굴 네트워크 미접속) | ✅ |
| `bit2watt_impl/ltma_inject.py` | 학습 루프에 불규칙 보조 연산 삽입 | ✅ |

즉 **공격 3종 워크로드와 수집기는 이미 구현되어 있다.** 아래는 빠진 것만 만든다.

## 빠진 것 1 — 정상 워크로드 생성기 5종 (최우선)

합성 데이터셋에는 정상 5패턴이 있으나 **실측용 실행 스크립트가 없다.**
`dataset/synthetic/all_v2.csv`의 라벨과 1:1 대응하도록 만든다.

`bit2watt_impl/normal_workloads.py` 하나에 서브커맨드로 구현:

| 서브커맨드 | 대응 라벨 | 구현 |
|---|---|---|
| `distributed` | `normal_distributed_training` | `torch.distributed` DDP, GPU 2장. compute/all-reduce 위상 전환이 실제로 생기게 |
| `hpo` | `normal_hpo_search` | 배치크기·lr을 바꿔가며 짧은 학습 반복 |
| `checkpoint` | `normal_checkpoint` | N step마다 `torch.save()`로 실제 디스크 쓰기 |
| `dataloader_stall` | `normal_dataloader_stall` | DataLoader에 인위적 `time.sleep()` 삽입 |
| `eval_switch` | `normal_eval_train_switch` | train N epoch / eval M epoch 교대 |

주의: **모두 실제 연산을 수행해야 한다.** `sleep`만으로 전력 패턴을 흉내내면
실측의 의미가 없다(`dataloader_stall`의 의도적 지연은 예외).

## 빠진 것 2 — 수집 세션 오케스트레이터

지금은 워크로드와 수집기를 사람이 따로 실행해야 한다. 한 번에 돌리는
`bit2watt_impl/run_capture.py`를 만든다.

1. 수집기를 백그라운드로 먼저 시작 → 3초 안정화 대기
2. 지정한 워크로드를 실행
3. 워크로드 종료 후 3초 더 수집하고 종료
4. `session_id`, `label`, `id_user`, `job_type`, `gres_req`를 **CLI 인자로 받아
   모든 행에 기록** (이 값들은 NVML이 주지 않는 그라운드트루스 메타데이터다)
5. 결과를 `dataset/real/{label}/{session_id}.csv`로 저장
6. 세션 메타(시작·종료 시각, GPU 모델, 드라이버 버전, 실제 실행 명령)를
   같은 폴더에 `.json`으로 남긴다

`--dry-run` 옵션으로 GPU 없이 인자 검증만 가능하게 할 것(CI용).

## 빠진 것 3 — 캡처 매트릭스 스크립트

`scripts/capture_all.sh`: 아래 조합을 순차 실행한다.

```
정상 5종 x 각 3회 반복  (GPU 1장)
정상 distributed x 3회  (GPU 2장)
swma x 3회              (GPU 1장)
swma x 3회              (GPU 2장 동시 — 다중 GPU 동기화 케이스)
ltma x 3회              (GPU 1장)
crypto x 3회            (GPU 1장)
```

각 세션은 5분 이내로 제한한다(공유 GPU 환경 배려). 전체 소요 약 2~3시간.

## 빠진 것 4 — 관측성(Observability) 측정 실행

`collect_telemetry.py`가 이미 `requested_interval_ms` / `actual_interval_ms` /
`value_changed`를 기록하므로 **새 코드 없이 실행만 하면 된다.**

`scripts/capture_observability.sh`:
- idle 상태에서 polling 주기 10 / 50 / 100 / 1000 ms로 각 60초 수집
- 부하 상태(swma 실행 중)에서 동일하게 반복
- 결과로 timestamp cadence 분포와 value refresh ratio를 계산해
  `dataset/real/observability/summary.json`에 저장

이는 작품소개서 §5.2의 관측 가능성 지표를 **실측으로 채우는 유일한 경로**다.

## 빠진 것 5 — 다중 GPU 동기화 피처

`pipeline/features.py`에 GPU 간 상관 피처가 없다. 다중 GPU 캡처가 확보된 뒤:

```python
def compute_sync_features(power_by_gpu: dict[int, np.ndarray]) -> dict:
    """GPU 쌍 간 cross-correlation 최댓값, 위상차, 동기화 지수를 반환."""
```

작품소개서 §4.1의 Coordination(Synchronization) 특성군이 코드에 구현되지 않은
유일한 항목이므로 반드시 채운다.

## 검증 기준

1. `dataset/real/` 아래에 9개 라벨 폴더가 모두 생성되고 각 3세션 이상
2. 실측 CSV의 컬럼이 `dataset/synthetic/all_v2.csv`와 **완전히 동일**
   (`run_pipeline.py`가 수정 없이 실측 데이터를 읽을 수 있어야 함)
3. `python pipeline/run_pipeline.py --telemetry dataset/real/...` 실행 성공
4. 실측 swma의 `swing_ratio` / `duty_regularity`가 합성 swma와 같은 대소 관계인지
   비교표를 `dataset/eval/synth_vs_real.json`으로 저장
   (**수치가 정확히 일치할 필요는 없다. 방향성만 확인하고, 다르면 그대로 보고**)

## 안전·환경 주의

- GPU는 공유 자원이므로 세션당 5분 이내, GPU 인덱스를 명시적으로 지정
- cryptojacking 워크로드는 실제 채굴 풀에 절대 접속하지 않는다(기존 구현 유지)
- `nvidia-smi -am 1`(accounting)은 sudo가 필요하므로 사용하지 않는다.
  PID/프로세스명은 `pynvml.nvmlDeviceGetComputeRunningProcesses` + `/proc/<pid>/comm`
  경로를 쓴다(기존 `collect_telemetry.py` 방식 유지)
- `DCGM_FI_PROF_*` 계열은 RTX 4090에서 미지원이므로 시도하지 않는다
