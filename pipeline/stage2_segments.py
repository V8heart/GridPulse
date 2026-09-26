"""Merge consecutive Stage1 candidate windows into Stage2 segments."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd


def merge_candidate_segments(candidates: pd.DataFrame) -> list[list[int]]:
    """Return lists of candidate index labels that form contiguous segments."""
    if candidates.empty:
        return []
    frame = candidates
    if "session_id" not in frame.columns or "gpu_id" not in frame.columns or "start" not in frame.columns:
        return [[idx] for idx in frame.index.tolist()]
    work = frame.sort_values(["session_id", "gpu_id", "start"], kind="mergesort")
    segments: list[list[int]] = []
    current: list[int] = []
    prev = None
    for idx, row in work.iterrows():
        key = (str(row["session_id"]), row["gpu_id"])
        start = int(row["start"])
        end = int(row["end"]) if "end" in row and pd.notna(row.get("end")) else start
        if prev is None:
            current = [idx]
            prev = (key, end)
            continue
        prev_key, prev_end = prev
        if key == prev_key and start <= int(prev_end):
            current.append(idx)
            prev = (key, max(int(prev_end), end))
        else:
            segments.append(current)
            current = [idx]
            prev = (key, end)
    if current:
        segments.append(current)
    return segments


def aggregate_bool_evidence(
    evidence_rows: list[dict],
    *,
    rate_min: float = 0.6,
    drop_keys: tuple[str, ...] = (),
) -> tuple[dict[str, bool], dict[str, float]]:
    """Window firing rates; True only when rate >= rate_min. Not a boolean OR."""
    counts: dict[str, int] = defaultdict(int)
    present: dict[str, int] = defaultdict(int)
    dropped = set(drop_keys)
    n = max(len(evidence_rows), 1)
    for row in evidence_rows:
        for key, value in (row or {}).items():
            if key in dropped:
                continue
            if not isinstance(value, bool):
                continue
            present[key] += 1
            if value:
                counts[key] += 1
    rates = {key: counts[key] / n for key in present}
    bools = {key: rates[key] >= rate_min for key in rates}
    return bools, rates
