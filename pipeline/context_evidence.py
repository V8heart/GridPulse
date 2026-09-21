"""Structured context evidence for Stage 1 v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset.progress_log_policy import eval_mask_hides_log


def event_time_s(event: dict) -> float | None:
    """Prefer session-relative ``t``; fall back to ``t_epoch`` when needed."""
    if "t" in event and event["t"] is not None:
        return float(event["t"])
    if "t_epoch" in event and event["t_epoch"] is not None:
        return float(event["t_epoch"])
    return None


def filter_events_for_gpu(events: list[dict], gpu_id: int | None = None) -> list[dict]:
    if gpu_id is None:
        return list(events)
    out = []
    for event in events:
        if "gpu_id" not in event or event["gpu_id"] is None:
            out.append(event)
            continue
        if int(event["gpu_id"]) == int(gpu_id):
            out.append(event)
    return out


def read_progress_log(
    path: str | Path | None,
    *,
    session_id: str | None = None,
    policy_seed: int | None = None,
    drop_prob: float | None = None,
    native_progress_available: bool = True,
    apply_eval_mask: bool = False,
) -> list[dict]:
    if not path:
        return []
    if (
        apply_eval_mask
        and session_id is not None
        and policy_seed is not None
        and drop_prob is not None
        and eval_mask_hides_log(
            session_id,
            seed=policy_seed,
            drop_prob=drop_prob,
            native_progress_available=native_progress_available,
        )
    ):
        return []
    target = Path(path)
    if not target.exists():
        return []
    events = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def step_period_from_log(
    events: list[dict],
    t0: float,
    t1: float,
    *,
    gpu_id: int | None = None,
) -> tuple[float | None, int, float | None]:
    scoped = filter_events_for_gpu(events, gpu_id)
    times = []
    for event in scoped:
        if event.get("event") != "step_end":
            continue
        t = event_time_s(event)
        if t is None:
            continue
        if t0 <= t <= t1:
            times.append(t)
    times = sorted(times)
    if len(times) < 2:
        return None, len(times), None
    intervals = np.diff(times)
    period = float(np.median(intervals))
    cv = float(np.std(intervals) / (np.mean(intervals) + 1e-12))
    return period, len(times), cv


def period_match(dominant_freq_hz: float, step_period_s: float | None, rel_tol: float = 0.15) -> bool | None:
    if not step_period_s or dominant_freq_hz <= 0:
        return None
    base = 1.0 / step_period_s
    candidates = [base, base / 2.0, base * 2.0, base * 3.0]
    return any(abs(dominant_freq_hz - freq) / max(freq, 1e-12) <= rel_tol for freq in candidates)


def explained_changepoints(
    changepoints: list[float],
    events: list[dict],
    *,
    tol_s: float = 5.0,
    gpu_id: int | None = None,
) -> dict:
    scoped = filter_events_for_gpu(events, gpu_id)
    explaining = []
    for event in scoped:
        if event.get("event") not in {
            "checkpoint_start",
            "checkpoint_end",
            "eval_start",
            "eval_end",
            "request_in",
            "request_out",
        }:
            continue
        t = event_time_s(event)
        if t is not None:
            explaining.append(t)
    unexplained = []
    for point in changepoints:
        if not any(abs(point - event_time) <= tol_s for event_time in explaining):
            unexplained.append(point)
    return {"explained": len(changepoints) - len(unexplained), "unexplained": unexplained}
