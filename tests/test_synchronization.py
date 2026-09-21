from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.synchronization import (
    align_gpu_series,
    pair_synchronization,
    session_window_sync_metrics,
    synchronization_features,
)


def test_identical_series_have_high_synchronization():
    t = np.arange(500) / 10
    signal = np.sin(2 * np.pi * 0.5 * t)
    result = pair_synchronization(signal, signal, sample_hz=10)
    assert result["synchronization_index"] > 0.95
    assert abs(result["lag_seconds"]) < 1e-9


def test_long_format_multi_gpu_alignment():
    t = np.arange(500) / 10
    signal = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)
    frame = pd.concat(
        [
            pd.DataFrame({"timestamp": t, "gpu_id": 0, "power_w": signal, "session_id": "s1"}),
            pd.DataFrame({"timestamp": t + 0.01, "gpu_id": 1, "power_w": signal, "session_id": "s1"}),
        ],
        ignore_index=True,
    )
    result = synchronization_features(frame)
    assert result[0]["gpu_pair"] == [0, 1]
    assert result[0]["synchronization_index"] > 0.9


def test_window_sync_and_cross_job():
    t = np.arange(400) / 10
    signal = 100 + 40 * np.sin(2 * np.pi * 0.5 * t)
    frame = pd.concat(
        [
            pd.DataFrame({
                "timestamp": t,
                "gpu_id": 0,
                "power_w": signal,
                "session_id": "mix",
                "declared_job_type": "hpo_sweep",
            }),
            pd.DataFrame({
                "timestamp": t,
                "gpu_id": 1,
                "power_w": signal,
                "session_id": "mix",
                "declared_job_type": "batch_inference",
            }),
        ],
        ignore_index=True,
    )
    metrics = session_window_sync_metrics(frame, 5.0, 25.0)
    assert metrics["sync_applicable"] is True
    assert metrics["multi_gpu_sync_index"] > 0.9
    assert metrics["cross_job_sync_index"] > 0.9


def test_cross_session_alignment_rejected():
    t = np.arange(100) / 10
    frame = pd.concat(
        [
            pd.DataFrame({"timestamp": t, "gpu_id": 0, "power_w": t, "session_id": "a"}),
            pd.DataFrame({"timestamp": t, "gpu_id": 1, "power_w": t, "session_id": "b"}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="단일 session_id"):
        align_gpu_series(frame)

