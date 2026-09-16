from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.synth_common import trailing_moving_average
from pipeline.baseline import CohortBaseline
from pipeline.context_evidence import period_match
from pipeline.stage1_v2 import conformal_pvalue, score_window


def test_conformal_pvalue_monotonic_and_bounded():
    scores = [0.1, 0.2, 0.4, 0.8]
    low = conformal_pvalue(0.2, scores)
    high = conformal_pvalue(0.7, scores)
    assert 0 < high <= low <= 1


def test_period_match_harmonics():
    assert period_match(1.0, 1.0) is True
    assert period_match(2.0, 1.0) is True
    assert period_match(0.5, 1.0) is True
    assert period_match(1.5, 1.0) is False


def test_nvml_average_attenuates_1hz_square():
    sample_hz = 100
    t = np.arange(0, 4, 1 / sample_hz)
    square = np.where((t % 1.0) < 0.5, 300.0, 100.0)
    averaged = trailing_moving_average(square, sample_hz, 1.0)[sample_hz:]
    assert np.percentile(averaged, 95) - np.percentile(averaged, 5) < 0.4 * (
        np.percentile(square, 95) - np.percentile(square, 5)
    )


def test_cohort_baseline_fit_uses_train_normals_only():
    rows = pd.DataFrame({
        "declared_job_family": ["training"] * 40,
        "gpu_model": ["RTX"] * 40,
        "mean_w": np.linspace(100, 120, 40),
        "swing_abs_w": np.linspace(10, 20, 40),
        "band_frac_0.1_0.7": np.linspace(0.1, 0.2, 40),
        "band_frac_0.7_2": np.linspace(0.1, 0.2, 40),
        "dominant_freq_hz": np.ones(40),
        "util_slope_w_per_pct": np.ones(40),
        "util_residual_mad_w": np.ones(40),
        "spectral_entropy": np.ones(40),
    })
    baseline = CohortBaseline().fit(rows, min_windows=10)
    z = baseline.robust_z(rows.iloc[0])
    assert z["baseline_level"] == "full"


def test_grid_watch_flag_for_high_impact_explained_window():
    baseline = CohortBaseline(stats={"global": {"features": {"mean_w": {"median": 100, "mad": 10}}}})
    row = {"mean_w": 101, "impact_components": {}, "evidence": {}, "swing_frac_tdp": 0.9}
    scored = score_window(row, baseline, {"normal_u_scores": [10, 11, 12], "impact_quantiles": {"p50": 0.2, "p90": 0.5}}, {"alpha": 0.05, "alpha_high_impact": 0.10, "tdp_w": 450, "grid_weights": {}, "evidence_weights": {}})
    assert scored["grid_watch"] is True
    assert scored["is_candidate"] is False
