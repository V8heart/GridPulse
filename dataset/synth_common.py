"""합성 파형을 공통 GridPulse 스키마로 변환하는 유틸리티."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset.schema import normalize_frame, validate_frame

DEFAULT_SEED = 42
BASELINE_POWER_W = 120.0


def rng_for(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def smooth_noise(rng: np.random.Generator, size: int, scale: float, width: int = 9) -> np.ndarray:
    """백색잡음을 이동평균해 실제 텔레메트리에 가까운 완만한 잡음을 만든다."""
    noise = rng.normal(0.0, scale, size + width - 1)
    return np.convolve(noise, np.ones(width) / width, mode="valid")


def trailing_moving_average(values: np.ndarray, sample_hz: float, window_s: float) -> np.ndarray:
    """Emulate NVML-style trailing average while preserving array length."""
    data = np.asarray(values, dtype=float)
    if window_s <= 0:
        return data.copy()
    width = max(1, int(round(window_s * sample_hz)))
    cumsum = np.cumsum(np.r_[0.0, data])
    result = np.empty_like(data, dtype=float)
    for idx in range(len(data)):
        start = max(0, idx + 1 - width)
        result[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx + 1 - start)
    return result


def _derived_util(power: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    low, high = float(np.percentile(power, 5)), float(np.percentile(power, 95))
    span = max(high - low, 10.0)
    return np.clip(8.0 + 86.0 * (power - low) / span + rng.normal(0, 2.0, len(power)), 0, 100)


def build_frame(
    power_w: np.ndarray,
    *,
    label: str,
    session_id: str,
    sample_hz: float,
    seed: int = DEFAULT_SEED,
    gpu_id: int = 0,
    util_gpu_pct: np.ndarray | None = None,
    declared_user: str | None = None,
    declared_job_type: str | None = None,
    declared_job_family: str | None = None,
    declared_process_name: str | None = None,
    id_user: str | None = None,
    job_type: str | None = None,
    declared_gres: str | None = None,
    gres_req: str | None = None,
    attack_id: str | None = None,
    gt_variant: str | None = None,
    gt_params_json: str | None = None,
    power_phys_w: np.ndarray | None = None,
    gpu_model: str = "RTX4090-synthetic",
    waveform_kind: str | None = None,
    waveform_frequency_hz: float | None = None,
    waveform_amplitude_frac: float | None = None,
    waveform_duty_cycle: float | None = None,
) -> pd.DataFrame:
    """전력 파형과 메타데이터를 물리적으로 일관된 합성 프레임으로 만든다."""
    if sample_hz <= 0:
        raise ValueError("sample_hz는 양수여야 합니다.")
    rng = rng_for(seed)
    power = np.asarray(power_w, dtype=float)
    if len(power) < 8 or not np.isfinite(power).all():
        raise ValueError("power_w에는 8개 이상의 유한한 값이 필요합니다.")
    if declared_job_type is None or declared_job_family is None or declared_process_name is None:
        raise ValueError("declared_job_type/family/process_name must be provided explicitly")
    if declared_user is None:
        raise ValueError("declared_user must be provided explicitly")
    if declared_gres is None:
        declared_gres = gres_req or "gpu:1"
    if id_user is None:
        id_user = declared_user
    if job_type is None:
        job_type = declared_job_type

    util = _derived_util(power, rng) if util_gpu_pct is None else np.asarray(util_gpu_pct, dtype=float)
    if len(util) != len(power):
        raise ValueError("util_gpu_pct 길이는 power_w와 같아야 합니다.")
    phys = power if power_phys_w is None else np.asarray(power_phys_w, dtype=float)
    if len(phys) != len(power):
        raise ValueError("power_phys_w 길이는 power_w와 같아야 합니다.")
    util = np.clip(util, 0, 100)
    mem_copy = np.clip(util * 0.55 + rng.normal(0, 4, len(power)), 0, 100)
    sm_clock = np.clip(210 + util * 25 + rng.normal(0, 35, len(power)), 210, 2520)
    temp = np.clip(28 + 0.28 * util + smooth_noise(rng, len(power), 1.2), 20, 90)
    fb_used = np.clip(450 + 70 * mem_copy + rng.normal(0, 100, len(power)), 0, 24564)
    timestamp = np.arange(len(power), dtype=float) / sample_hz
    origin = pd.Timestamp("2026-01-01T00:00:00Z")
    collection = origin + pd.to_timedelta(timestamp, unit="s")
    interval_ms = 1000.0 / sample_hz

    changed = np.ones(len(power), dtype=bool)
    if len(power) > 1:
        changed[1:] = (
            (np.diff(np.round(power, 3)) != 0)
            | (np.diff(np.round(util, 3)) != 0)
            | (np.diff(np.round(mem_copy, 3)) != 0)
        )

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "collection_timestamp": collection.astype(str),
            "raw_timestamp": np.nan,
            "session_id": session_id,
            "gt_attack_id": attack_id or session_id,
            "gt_label": label,
            "gt_variant": gt_variant,
            "gt_params_json": gt_params_json,
            "attack_id": attack_id or session_id,
            "waveform_kind": waveform_kind,
            "waveform_frequency_hz": (
                np.nan if waveform_frequency_hz is None else waveform_frequency_hz
            ),
            "waveform_amplitude_frac": (
                np.nan if waveform_amplitude_frac is None else waveform_amplitude_frac
            ),
            "waveform_duty_cycle": (
                np.nan if waveform_duty_cycle is None else waveform_duty_cycle
            ),
            "gpu_id": gpu_id,
            "sample_hz": float(sample_hz),
            "power_w": np.clip(power, 0, 500),
            "power_phys_w": np.clip(phys, 0, 500),
            "util_gpu_pct": util,
            "mem_copy_util_pct": mem_copy,
            "sm_clock_mhz": sm_clock,
            "temp_c": temp,
            "fb_used_mb": fb_used,
            "requested_interval_ms": interval_ms,
            "actual_interval_ms": interval_ms,
            "value_changed": changed,
            "pid": np.nan,
            "declared_process_name": declared_process_name,
            "declared_user": declared_user,
            "declared_job_type": declared_job_type,
            "declared_job_family": declared_job_family,
            "declared_gres": declared_gres,
            "gpu_model": gpu_model,
            "process_name": declared_process_name,
            "id_user": id_user,
            "job_type": job_type,
            "gres_req": declared_gres,
            "label": label,
        }
    )
    frame = normalize_frame(frame)
    validate_frame(frame)
    return frame


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """스키마를 검증한 뒤 CSV를 저장한다."""
    validate_frame(df)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target

