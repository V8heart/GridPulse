"""Structured context evidence for Stage 1 v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_progress_log(path: str | Path | None) -> list[dict]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        return []
    events = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def step_period_from_log(events: list[dict], t0: float, t1: float) -> tuple[float | None, int, float | None]:
    times = sorted(float(e["t"]) for e in events if t0 <= float(e.get("t", -1)) <= t1 and e.get("event") == "step_end")
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


def explained_changepoints(changepoints: list[float], events: list[dict], *, tol_s: float = 5.0) -> dict:
    explaining = [
        float(e["t"])
        for e in events
        if e.get("event") in {"checkpoint_start", "checkpoint_end", "eval_start", "eval_end", "request_in", "request_out"}
    ]
    unexplained = []
    for point in changepoints:
        if not any(abs(point - event_time) <= tol_s for event_time in explaining):
            unexplained.append(point)
    return {"explained": len(changepoints) - len(unexplained), "unexplained": unexplained}
