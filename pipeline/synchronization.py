"""다중 GPU 시계열의 정렬·상관·coherence 피처."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import coherence


def align_gpu_series(
    df: pd.DataFrame,
    *,
    value_col: str = "power_w",
    sample_hz: float | None = None,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    """timestamp 공통 구간에 GPU별 시계열을 선형 보간한다.

    Optional ``t0``/``t1`` clip the overlap to a window. Callers must pass a
    single-session DataFrame; cross-session alignment is forbidden.
    """
    required = {"timestamp", "gpu_id", value_col}
    if not required.issubset(df.columns):
        raise ValueError(f"동기화 분석 필수 컬럼: {sorted(required)}")
    if "session_id" in df.columns and df["session_id"].nunique(dropna=False) > 1:
        raise ValueError("동기화 분석은 단일 session_id만 허용합니다 (교차 세션 정렬 금지).")

    groups = {
        int(gpu): group.sort_values("timestamp")
        for gpu, group in df.groupby("gpu_id")
    }
    if len(groups) < 2:
        raise ValueError("동기화 분석에는 GPU가 2개 이상 필요합니다.")

    starts = [pd.to_numeric(g["timestamp"], errors="coerce").min() for g in groups.values()]
    ends = [pd.to_numeric(g["timestamp"], errors="coerce").max() for g in groups.values()]
    start, stop = max(starts), min(ends)
    if t0 is not None:
        start = max(start, float(t0))
    if t1 is not None:
        stop = min(stop, float(t1))
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("GPU 시계열의 공통 시간 구간이 없습니다.")

    if sample_hz is None:
        candidates = []
        for group in groups.values():
            ts = pd.to_numeric(group["timestamp"], errors="coerce").dropna().to_numpy()
            delta = np.diff(ts)
            delta = delta[delta > 0]
            if len(delta):
                candidates.append(1.0 / np.median(delta))
        sample_hz = min(candidates) if candidates else 1.0
    timeline = np.arange(start, stop, 1.0 / sample_hz)
    if len(timeline) < 16:
        raise ValueError("공통 시간 구간 샘플이 16개 미만입니다.")
    aligned: dict[int, np.ndarray] = {}
    for gpu, group in groups.items():
        ts = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy()
        mask = np.isfinite(ts) & np.isfinite(values)
        aligned[gpu] = np.interp(timeline, ts[mask], values[mask])
    return timeline, aligned, float(sample_hz)


def pair_synchronization(x: np.ndarray, y: np.ndarray, sample_hz: float) -> dict[str, float]:
    """두 시계열의 zero-lag 상관, 최대 상관, coherence를 계산한다."""
    if len(x) != len(y) or len(x) < 16:
        raise ValueError("길이가 같은 16개 이상의 샘플이 필요합니다.")
    xz, yz = x - np.mean(x), y - np.mean(y)
    denom = np.std(xz) * np.std(yz)
    zero_corr = float(np.mean(xz * yz) / denom) if denom > 0 else 0.0
    corr = np.correlate(xz, yz, mode="full")
    corr_denom = np.linalg.norm(xz) * np.linalg.norm(yz)
    corr = corr / corr_denom if corr_denom > 0 else corr
    best = int(np.argmax(np.abs(corr)))
    lag_samples = best - (len(x) - 1)
    max_corr = float(np.abs(corr[best]))
    _, coh = coherence(x, y, fs=sample_hz, nperseg=min(256, len(x)))
    mean_coherence = float(np.nanmean(coh)) if len(coh) else 0.0
    sync_index = float(np.clip((abs(zero_corr) + max_corr + mean_coherence) / 3.0, 0, 1))
    return {
        "zero_lag_correlation": zero_corr,
        "max_cross_correlation": max_corr,
        "lag_seconds": lag_samples / sample_hz,
        "mean_coherence": mean_coherence,
        "synchronization_index": sync_index,
    }


def synchronization_features(df: pd.DataFrame, value_col: str = "power_w") -> list[dict]:
    """세션 내부 모든 GPU 쌍의 동기화 피처를 반환한다."""
    _, aligned, hz = align_gpu_series(df, value_col=value_col)
    results = []
    for left, right in itertools.combinations(sorted(aligned), 2):
        results.append({
            "gpu_pair": [left, right],
            "sample_hz": hz,
            **pair_synchronization(aligned[left], aligned[right], hz),
        })
    return results


def session_window_sync_metrics(
    session_df: pd.DataFrame,
    t0: float,
    t1: float,
    *,
    value_col: str = "power_w",
    declared_job_col: str = "declared_job_type",
) -> dict:
    """Same-session window sync indices for all GPU pairs and cross-job pairs."""
    empty = {
        "multi_gpu_sync_index": 0.0,
        "cross_job_sync_index": 0.0,
        "sync_applicable": False,
        "sync_lag_s": 0.0,
        "sync_pair": None,
        "cross_job_sync_pair": None,
    }
    if session_df["gpu_id"].nunique() < 2:
        return empty
    try:
        _, aligned, hz = align_gpu_series(session_df, value_col=value_col, t0=t0, t1=t1)
    except ValueError:
        return empty

    declared_by_gpu: dict[int, str] = {}
    if declared_job_col in session_df.columns:
        for gpu, group in session_df.groupby("gpu_id"):
            declared_by_gpu[int(gpu)] = str(group[declared_job_col].iloc[0])

    best_all = None
    best_cross = None
    for left, right in itertools.combinations(sorted(aligned), 2):
        metrics = pair_synchronization(aligned[left], aligned[right], hz)
        candidate = {
            "gpu_pair": [left, right],
            **metrics,
        }
        if best_all is None or candidate["synchronization_index"] > best_all["synchronization_index"]:
            best_all = candidate
        if declared_by_gpu.get(left) != declared_by_gpu.get(right):
            if best_cross is None or candidate["synchronization_index"] > best_cross["synchronization_index"]:
                best_cross = candidate

    if best_all is None:
        return empty
    return {
        "multi_gpu_sync_index": float(best_all["synchronization_index"]),
        "cross_job_sync_index": float(best_cross["synchronization_index"]) if best_cross else 0.0,
        "sync_applicable": True,
        "sync_lag_s": float(best_all["lag_seconds"]),
        "sync_pair": best_all["gpu_pair"],
        "cross_job_sync_pair": best_cross["gpu_pair"] if best_cross else None,
    }
