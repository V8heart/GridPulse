from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset.synth_common import trailing_moving_average
from pipeline.baseline import CohortBaseline
from pipeline.context_evidence import period_match
from pipeline.stage1_v2 import (
    CALIBRATION_SCHEMA_VERSION,
    combine_u_score,
    conformal_pvalue,
    empirical_tail_p,
    score_window,
)


def _mini_calibration(**overrides):
    cal = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "feature_abs_z": {
            "training": {"mean_w": [0.5, 1.0, 1.5, 2.0] * 10},
            "global": {"mean_w": [0.5, 1.0, 1.5, 2.0] * 20},
        },
        "normal_u_scores": {
            "full": {"training": [0.5, 1.0, 1.5, 2.0] * 10, "global": [0.5, 1.0, 1.5, 2.0] * 20},
            "no_evidence": {"training": [0.4, 0.8, 1.2, 1.6] * 10, "global": [0.4, 0.8, 1.2, 1.6] * 20},
        },
        "evidence_surprisal_weights": {"declared_family_mismatch": 1.5, "progress_log_missing": 0.0},
        "progress_log_policy_ok": True,
        "impact_quantiles": {"p50": 0.2, "p90": 0.5},
    }
    cal.update(overrides)
    return cal


def test_conformal_pvalue_monotonic_and_bounded():
    scores = [0.1, 0.2, 0.4, 0.8]
    low = conformal_pvalue(0.2, scores)
    high = conformal_pvalue(0.7, scores)
    assert 0 < high <= low <= 1


def test_empirical_tail_monotonic():
    refs = [0.5, 1.0, 2.0, 3.0]
    assert empirical_tail_p(0.1, refs) > empirical_tail_p(2.5, refs)


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
    baseline = CohortBaseline().fit(rows, min_windows=30)
    z = baseline.robust_z(rows.iloc[0])
    assert z["baseline_level"] == "full"


def test_mad_floor_bounds_near_constant_feature():
    from pipeline.baseline import mad_floor_for

    rows = pd.DataFrame({
        "declared_job_family": ["interactive"] * 40,
        "gpu_model": ["RTX"] * 40,
        "mean_w": np.full(40, 100.0),
        "swing_abs_w": np.full(40, 10.0),
        "band_frac_0.1_0.7": np.full(40, 0.2),
        "band_frac_0.7_2": np.linspace(0.10, 0.1001, 40),
        "dominant_freq_hz": np.ones(40),
        "util_slope_w_per_pct": np.linspace(0.0, 0.001, 40),
        "util_residual_mad_w": np.full(40, 0.14),
        "spectral_entropy": np.full(40, 0.5),
    })
    baseline = CohortBaseline().fit(rows, min_windows=30)
    feats = baseline.stats["full:interactive|RTX"]["features"]
    assert feats["band_frac_0.7_2"]["mad"] >= mad_floor_for("band_frac_0.7_2", 0.1)
    z = baseline.robust_z({**rows.iloc[-1].to_dict(), "band_frac_0.7_2": 0.5})
    assert abs(z["band_frac_0.7_2"]) < 50


def test_util_slope_nan_when_util_range_small():
    from pipeline.features import compute_window_features_v2

    power = 120 + np.linspace(0, 5, 64)
    util = np.full(64, 50.0) + np.linspace(0, 1.0, 64)
    feats = compute_window_features_v2(power, util, sample_hz=10.0, util_min_range_pct=5.0)
    assert feats["util_range_pct"] < 5.0
    assert np.isnan(feats["util_slope_w_per_pct"])
    baseline = CohortBaseline(stats={"global": {"features": {
        "util_slope_w_per_pct": {"median": 1.0, "mad": 0.05, "mad_floor": 0.05},
        "mean_w": {"median": 120.0, "mad": 5.0, "mad_floor": 5.0},
    }}})
    z = baseline.robust_z({"util_slope_w_per_pct": feats["util_slope_w_per_pct"], "mean_w": 121.0})
    assert "util_slope_w_per_pct" not in z
    assert "mean_w" in z


def test_old_calibration_schema_rejected():
    baseline = CohortBaseline(stats={"global": {"features": {"mean_w": {"median": 100, "mad": 10, "mad_floor": 5}}}})
    with pytest.raises(ValueError, match="schema_version"):
        score_window(
            {"mean_w": 101, "declared_job_family": "training", "evidence": {}},
            baseline,
            {"normal_u_scores": [1, 2, 3]},
            {"alpha": 0.1, "tdp_w": 450, "grid_weights": {}, "evidence_weights": {}},
        )


def test_evidence_increases_u_at_moderate_z():
    cal = _mini_calibration()
    row = {"declared_job_family": "training", "evidence": {}}
    base = combine_u_score(
        {"mean_w": 1.2},
        {"declared_family_mismatch": False},
        row,
        cal,
        {"mondrian_min_n": 30, "evidence_weights": {}},
        include_evidence=True,
    )
    with_ev = combine_u_score(
        {"mean_w": 1.2},
        {"declared_family_mismatch": True},
        row,
        cal,
        {"mondrian_min_n": 30, "evidence_weights": {}},
        include_evidence=True,
    )
    assert with_ev["u_score"] > base["u_score"]


def test_mondrian_family_fallback_and_loo():
    baseline = CohortBaseline(stats={"global": {"features": {"mean_w": {"median": 100, "mad": 10, "mad_floor": 5}}}})
    cal = _mini_calibration()
    # Rare family → global conformal bucket.
    scored = score_window(
        {
            "mean_w": 130,
            "declared_job_family": "interactive",
            "evidence": {},
            "band_power_w2_0.1_0.7": 0.0,
        },
        baseline,
        cal,
        {"alpha": 0.1, "alpha_high_impact": 0.2, "tdp_w": 450, "grid_weights": {"0.1_0.7": 1.0}, "evidence_weights": {}, "mondrian_min_n": 30},
    )
    assert scored["mondrian_level"] == "global"
    assert 0 < scored["p_value"] <= 1


def test_grid_watch_flag_for_high_impact_explained_window():
    baseline = CohortBaseline(stats={"global": {"features": {"mean_w": {"median": 100, "mad": 10, "mad_floor": 5}}}})
    row = {
        "mean_w": 101,
        "declared_job_family": "training",
        "evidence": {},
        "swing_frac_tdp": 0.01,
        "band_power_w2_0.1_0.7": (0.9 * 450) ** 2,
        "band_reliability_0.1_0.7": 1.0,
    }
    scored = score_window(
        row,
        baseline,
        _mini_calibration(),
        {
            "alpha": 0.05,
            "alpha_high_impact": 0.10,
            "tdp_w": 450,
            "grid_weights": {"0.1_0.7": 1.0},
            "evidence_weights": {},
            "mondrian_min_n": 30,
        },
    )
    assert scored["impact_raw"] >= 0.5
    assert scored["impact_level"] == "high"
    assert scored["grid_watch"] is True
    assert scored["is_candidate"] is False


def test_impact_ignores_swing_alone():
    from pipeline.stage1_v2 import impact_components

    comps = impact_components(
        {"swing_frac_tdp": 0.99, "ramp_p95_w_per_s": 400.0, "band_power_w2_0.1_0.7": 0.0},
        {"tdp_w": 450.0, "grid_weights": {"0.1_0.7": 1.0}},
    )
    assert comps["impact_raw"] == 0.0
