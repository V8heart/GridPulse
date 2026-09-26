"""Classify attack windows as detectable vs unobservable from NVML-scale power."""
from __future__ import annotations

import json
from typing import Any


def parse_period_s(params: Any) -> float | None:
    if params is None:
        return None
    data = params
    if isinstance(params, str) and params.strip():
        try:
            data = json.loads(params)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    for key in ("period_actual_s", "period", "period_requested_s"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    summary = data.get("workload_summary")
    if isinstance(summary, dict):
        for key in ("period_actual_s", "period"):
            value = summary.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


def attack_visibility_group(
    *,
    gt_label: str | None,
    gt_variant: str | None = None,
    period_s: float | None = None,
) -> str | None:
    """Return detectable, unobservable, or None for normals/other."""
    label = str(gt_label or "").lower()
    variant = str(gt_variant or "").lower()
    if label.startswith("normal"):
        return None
    if "crypto" in label or variant == "crypto_flat":
        return "detectable"
    if "ltma" in label or "ltma" in variant:
        return "unobservable"
    if "piggyback" in variant or "mimicry" in variant:
        return "unobservable"
    if "swma" in label or "swma" in variant:
        if period_s is None:
            return None
        if period_s <= 1.0 + 1e-9:
            return "unobservable"
        if 2.0 - 1e-9 <= period_s <= 4.0 + 1e-9:
            return "detectable"
        return None
    return None
