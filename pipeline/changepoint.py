"""Lightweight CUSUM changepoint utilities for Stage 1 v2."""
from __future__ import annotations

import numpy as np


def cusum_changepoints(values, *, sample_hz: float, k: float = 0.5, h: float = 5.0) -> list[float]:
    """Return approximate changepoint times using a two-sided CUSUM."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return []
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = max(1.4826 * mad, 1e-6)
    pos = neg = 0.0
    points: list[float] = []
    last = -10**9
    refractory = max(1, int(sample_hz))
    for idx, value in enumerate((x - center) / scale):
        pos = max(0.0, pos + value - k)
        neg = min(0.0, neg + value + k)
        if (pos > h or neg < -h) and idx - last >= refractory:
            points.append(idx / sample_hz)
            last = idx
            pos = neg = 0.0
    return points
