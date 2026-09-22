from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset.observability import (
    boxcar_attenuation,
    fold_frequency,
    period_retention_rows,
    write_summary,
)


def test_known_frequency_alias_and_boxcar_fixture(tmp_path: Path):
    sample_hz = 10.0
    true_hz = 7.0
    averaging_s = 0.1
    times = np.arange(400) / sample_hz
    instant = 150 + 40 * np.sin(2 * np.pi * true_hz * times)
    attenuation = boxcar_attenuation(true_hz, averaging_s)
    averaged = 150 + attenuation * 40 * np.sin(2 * np.pi * true_hz * times)
    frame = pd.DataFrame({
        "timestamp": times,
        "session_id": "s-0123456789abcdef",
        "gpu_id": 0,
        "sample_hz": sample_hz,
        "requested_interval_ms": 100.0,
        "actual_interval_ms": 100.0,
        "value_changed": True,
        "true_frequency_hz": true_hz,
        "power_w": averaged,
        "power_instant_w": instant,
    })
    path = tmp_path / "sessions" / "s-0123456789abcdef" / "telemetry.csv"
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)

    row = period_retention_rows(path, frame)[0]
    assert fold_frequency(true_hz, sample_hz) == 3.0
    assert row["aliased"] is True
    assert abs(row["averaged_peak_hz"] - 3.0) < 0.05
    assert abs(row["period_retention"] - attenuation) < 0.02
    assert abs(row["effective_averaging_window_s"] - averaging_s) < 0.01

    write_summary([tmp_path], tmp_path / "observability" / "summary.md")
    retention = pd.read_csv(tmp_path / "observability" / "period_retention.csv")
    assert len(retention) == 1
    assert bool(retention.iloc[0]["aliased"]) is True
import math

import numpy as np
import pandas as pd

from dataset.observability import boxcar_attenuation, fold_frequency, period_retention_rows


def test_fold_frequency_nyquist():
    assert abs(fold_frequency(6.0, sample_hz=10.0) - 4.0) < 1e-9  # 6 aliases to 4
    assert abs(fold_frequency(1.0, sample_hz=10.0) - 1.0) < 1e-9


def test_boxcar_sinc_null():
    # sinc(1) = 0 => f * T = 1
    assert abs(boxcar_attenuation(1.0, 1.0)) < 1e-12
    assert abs(boxcar_attenuation(0.5, 1.0) - abs(np.sinc(0.5))) < 1e-9


def test_period_retention_fixture(tmp_path):
    fs = 10.0
    f0 = 1.0
    t = np.arange(0, 20, 1 / fs)
    power = 120 + 30 * np.sin(2 * math.pi * f0 * t)
    df = pd.DataFrame(
        {
            "timestamp": t,
            "sample_hz": fs,
            "power_w": power,
            "power_instant_w": power,
            "true_frequency_hz": f0,
            "session_id": "s-0000000000000001",
            "gpu_id": 0,
        }
    )
    path = tmp_path / "telemetry.csv"
    df.to_csv(path, index=False)
    rows = period_retention_rows(path, df)
    assert rows
    assert rows[0]["aliased"] is False
    assert rows[0]["true_frequency_hz"] == f0
    assert rows[0]["folded_frequency_hz"] == f0
