"""합성·실측 텔레메트리의 공통 스키마와 검증 함수 (gp-telemetry/1.2)."""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "gp-telemetry/1.2"

# Canonical columns for new writes (standard §4). Legacy columns are still
# accepted on read via LEGACY_TO_CANONICAL / normalize_frame.
CORE_COLUMNS = [
    "timestamp",
    "t_epoch",
    "session_id",
    "gpu_id",
    "gpu_uuid",
    "gpu_model",
    "sample_hz",
    "requested_interval_ms",
    "actual_interval_ms",
    "value_changed",
    "warmup",
    "power_w",
    "power_instant_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "fan_speed_pct",
    "pstate",
    "power_limit_w",
    "throttle_reasons",
    "observed_pid",
    "observed_process_name",
    "observed_n_procs",
    "declared_job_type",
    "declared_job_family",
    "declared_process_name",
    "declared_user",
    "declared_gres",
    "gt_label",
    "gt_is_attack",
    "gt_variant",
    "gt_params_json",
]

REQUIRED_COLUMNS = {
    "timestamp",
    "session_id",
    "gpu_id",
    "sample_hz",
    "power_w",
}

INFERENCE_FORBIDDEN_COLUMNS = {
    "gt_label",
    "gt_is_attack",
    "gt_attack_id",
    "gt_variant",
    "gt_params_json",
    "gt_attack_intervals_epoch",
    "power_phys_w",
    "label",
    "attack_id",
    "waveform_kind",
    "waveform_frequency_hz",
    "waveform_amplitude_frac",
    "waveform_duty_cycle",
    "gpu_role",
    "group_id",
    "capture_key",
    "declared_policy",
    "workload_module",
    "workload_cmd",
    "host_workload",
    "truth_source",
}

NUMERIC_COLUMNS = {
    "timestamp",
    "t_epoch",
    "raw_timestamp",
    "waveform_frequency_hz",
    "waveform_amplitude_frac",
    "waveform_duty_cycle",
    "gpu_id",
    "sample_hz",
    "power_w",
    "power_instant_w",
    "power_phys_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "fan_speed_pct",
    "pstate",
    "power_limit_w",
    "throttle_reasons",
    "requested_interval_ms",
    "actual_interval_ms",
    "observed_pid",
    "observed_n_procs",
    "pid",
}

LEGACY_TO_CANONICAL = {
    "label": "gt_label",
    "attack_id": "gt_variant",
    "job_type": "declared_job_type",
    "id_user": "declared_user",
    "gres_req": "declared_gres",
    "process_name": "observed_process_name",
    "pid": "observed_pid",
}

# Backward-compatible alias used by older call sites / docs.
LEGACY_TO_DECLARED = LEGACY_TO_CANONICAL


def strip_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with labels/attack params removed for inference."""
    drop = set(INFERENCE_FORBIDDEN_COLUMNS)
    drop.update(c for c in df.columns if str(c).startswith("gt_"))
    return df.drop(columns=[c for c in drop if c in df], errors="ignore").copy()


@dataclass(frozen=True)
class SessionManifest:
    """한 캡처/합성 세션의 재현에 필요한 최소 메타데이터."""

    session_id: str
    label: str
    source: str
    sample_hz: float
    gpu_ids: list[int]
    rows: int
    seed: int | None = None
    workload: str | None = None
    notes: str | None = None
    attack_id: str | None = None
    split: str | None = None
    progress_log_path: str | None = None
    gt_variant: str | None = None
    native_progress_available: bool | None = None
    progress_log_masked: bool | None = None
    progress_log_policy_version: str | None = None
    progress_log_drop_prob: float | None = None
    progress_log_policy_seed: int | None = None
    group_id: str | None = None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """누락된 선택 컬럼을 추가하고 공통 컬럼 순서로 정렬한다."""
    out = df.copy()
    for legacy, new_name in LEGACY_TO_CANONICAL.items():
        if legacy in out and new_name not in out:
            out[new_name] = out[legacy]
            warnings.warn(
                f"legacy column {legacy!r} mapped to {new_name!r}; "
                "new datasets should write declared/gt/observed columns directly",
                RuntimeWarning,
                stacklevel=2,
            )
    if "gt_is_attack" not in out:
        if "gt_label" in out:
            labels = out["gt_label"].astype(str)
            out["gt_is_attack"] = ~labels.str.startswith("normal") & labels.ne("nan") & labels.ne("<NA>")
        elif "label" in out:
            labels = out["label"].astype(str)
            out["gt_is_attack"] = ~labels.str.startswith("normal") & labels.ne("nan") & labels.ne("<NA>")
    for column in CORE_COLUMNS:
        if column not in out:
            if column == "gt_is_attack":
                out[column] = False
            elif column == "warmup":
                out[column] = False
            elif column in NUMERIC_COLUMNS:
                out[column] = np.nan
            else:
                out[column] = pd.NA
    ordered = CORE_COLUMNS + [c for c in out.columns if c not in CORE_COLUMNS]
    return out.loc[:, ordered]


def validate_frame(df: pd.DataFrame, *, require_nonempty: bool = True) -> None:
    """스키마, 범위, 세션 내부 일관성을 검사한다."""
    if require_nonempty and df.empty:
        raise ValueError("텔레메트리 데이터가 비어 있습니다.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")
    if "label" not in df and "gt_label" not in df:
        raise ValueError("필수 컬럼 누락: ['label' 또는 'gt_label']")

    for column in NUMERIC_COLUMNS & set(df.columns):
        values = pd.to_numeric(df[column], errors="coerce")
        if column in REQUIRED_COLUMNS and values.isna().any():
            raise ValueError(f"{column}에 숫자가 아닌 값 또는 결측값이 있습니다.")

    for column in ("util_gpu_pct", "mem_copy_util_pct"):
        if column in df:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if ((values < 0) | (values > 100)).any():
                raise ValueError(f"{column} 값은 0~100 범위여야 합니다.")
    if "power_w" in df and (pd.to_numeric(df["power_w"], errors="coerce") < 0).any():
        raise ValueError("power_w는 음수일 수 없습니다.")
    if "sample_hz" in df and (pd.to_numeric(df["sample_hz"], errors="coerce") <= 0).any():
        raise ValueError("sample_hz는 양수여야 합니다.")

    label_col = "label" if "label" in df and not df["label"].isna().all() else "gt_label"
    grouped = df.groupby(["session_id", "gpu_id"], dropna=False)
    for key, group in grouped:
        if group[label_col].nunique(dropna=False) != 1:
            raise ValueError(f"세션 {key} 안에 여러 label이 섞였습니다.")
        hz = pd.to_numeric(group["sample_hz"], errors="coerce").dropna()
        if hz.nunique() != 1:
            raise ValueError(f"세션 {key} 안에 여러 sample_hz가 섞였습니다.")
        timestamps = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError(f"세션 {key} timestamp가 단조 증가하지 않습니다.")


def write_manifest(manifests: list[SessionManifest], path: str | Path) -> None:
    """세션 manifest를 사람이 읽을 수 있는 JSON으로 저장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(item) for item in manifests], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def manifest_from_dict(data: dict[str, Any]) -> SessionManifest:
    """JSON 등에서 읽은 사전을 검증 가능한 manifest로 변환한다."""
    return SessionManifest(**data)
