"""Window-level ground-truth helpers (gp-telemetry/1.2 §4.6)."""
from __future__ import annotations

import json
from typing import Iterable, Sequence


Interval = tuple[float, float]


def parse_intervals(raw) -> list[Interval]:
    """Parse JSON list of [start_epoch, end_epoch] pairs."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        raw = json.loads(text)
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"intervals must be a list, got {type(raw)!r}")
    out: list[Interval] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"interval must be [start, end], got {item!r}")
        start, end = float(item[0]), float(item[1])
        if end < start:
            start, end = end, start
        out.append((start, end))
    return out


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_overlap_seconds(window: Interval, intervals: Sequence[Interval]) -> float:
    """Union overlap between window and intervals (no double-count)."""
    w0, w1 = float(window[0]), float(window[1])
    if w1 <= w0:
        return 0.0
    total = 0.0
    for start, end in merge_intervals(intervals):
        lo = max(w0, start)
        hi = min(w1, end)
        if hi > lo:
            total += hi - lo
    return total


def window_is_attack(
    window_start_epoch: float,
    window_end_epoch: float,
    intervals: Sequence[Interval] | None,
    *,
    threshold: float = 0.5,
) -> bool:
    duration = float(window_end_epoch) - float(window_start_epoch)
    if duration <= 0:
        return False
    if not intervals:
        return False
    overlap = interval_overlap_seconds((window_start_epoch, window_end_epoch), intervals)
    return (overlap / duration) >= threshold


def label_window(
    *,
    window_start_epoch: float,
    window_end_epoch: float,
    intervals: Sequence[Interval] | None,
    session_gt_label: str,
    host_workload: str | None = None,
    hosted: bool = False,
    legacy_session_label: bool = False,
) -> tuple[str, bool, str]:
    """Return (gt_label, gt_is_attack, truth_source)."""
    if str(session_gt_label).startswith("normal"):
        return str(session_gt_label), False, "session_normal"
    if legacy_session_label or intervals is None:
        is_attack = not str(session_gt_label).startswith("normal")
        return str(session_gt_label), is_attack, "session_legacy"
    if window_is_attack(window_start_epoch, window_end_epoch, intervals):
        return str(session_gt_label), True, "interval_overlap"
    if hosted:
        host = host_workload or "normal_mlp_small"
        return str(host), False, "interval_inactive_host"
    return "normal_idle", False, "interval_inactive"
