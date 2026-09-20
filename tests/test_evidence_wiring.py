"""E1 wiring: v2 features and declared_context must reach pipeline result rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_V2_KEYS = (
    "swing_abs_w",
    "ramp_p95_w_per_s",
    "util_residual_mad_w",
    "dominant_peak_prominence",
)


def _write_one_session_csv(tmp_path: Path) -> Path:
    src = ROOT / "dataset/synthetic/all_v3.csv"
    if not src.exists():
        pytest.skip("all_v3.csv missing")
    df = pd.read_csv(src, low_memory=False)
    session = str(df["session_id"].iloc[0])
    subset = df[df["session_id"].astype(str) == session]
    # Prefer a session that has declared_job_type
    with_declared = df[df["declared_job_type"].notna()]
    if not with_declared.empty:
        session = str(with_declared["session_id"].iloc[0])
        subset = df[df["session_id"].astype(str) == session]
    out = tmp_path / "one_session.csv"
    subset.to_csv(out, index=False)
    return out


def test_pipeline_row_has_v2_features_and_declared_context(tmp_path):
    from pipeline.run_pipeline import run

    telemetry = _write_one_session_csv(tmp_path)
    out = tmp_path / "results.json"
    args = type(
        "Args",
        (),
        {
            "telemetry": str(telemetry),
            "baseline_mean": 120.0,
            "sample_hz": None,
            "window": 30,
            "stride": 15,
            "z_threshold": 0.0,  # force candidates so we get rows
            "stage1": "legacy",
            "stage2": "v2",
            "baseline_model": str(ROOT / "dataset/eval/cohort_baseline_v2.json"),
            "stage1_config": str(ROOT / "config/stage1_v2.yaml"),
            "calibration": str(ROOT / "dataset/eval/stage1_v2_calibration.json"),
            "window_s": 30.0,
            "stride_s": 15.0,
            "progress_log_dir": str(ROOT / "dataset/synthetic/steps"),
            "query_context": "off",
            "rag_backend": "tfidf",
            "llm_backend": "stub",
            "llm_model": "stub",
            "physics_validate": False,
            "physics_timeout_s": 60.0,
            "physics_test_system": "kundur_ieeest",
            "out": str(out),
        },
    )()
    rows = run(args)
    assert rows, "expected at least one candidate row"
    row = rows[0]
    feats = row["features"]
    missing = [k for k in REQUIRED_V2_KEYS if k not in feats]
    assert not missing, f"missing v2 feature keys: {missing}"
    assert row.get("features_version") == "v1+v2"
    bundle = row.get("stage2_evidence_bundle") or {}
    declared = bundle.get("declared_context") or {}
    assert declared.get("job_type") is not None, f"declared_context was {declared}"


def test_merge_window_features_includes_v2_keys():
    import numpy as np
    from pipeline.run_pipeline import merge_window_features

    t = np.linspace(0, 30, 30)
    power = 100 + 80 * (np.sin(2 * np.pi * t / 3) > 0)
    util = 50 + 20 * (np.sin(2 * np.pi * t / 3) > 0)
    feats = merge_window_features(power, util, sample_hz=1.0)
    for key in REQUIRED_V2_KEYS:
        assert key in feats, key


def test_build_evidence_bundle_prefers_declared_context():
    from pipeline.stage2_evidence import build_evidence_bundle

    bundle = build_evidence_bundle(
        {
            "window_id": "w1",
            "features": {
                "swing_abs_w": 10.0,
                "ramp_p95_w_per_s": 1.0,
                "util_residual_mad_w": 1.0,
                "dominant_peak_prominence": 2.0,
                "mean_w": 200.0,
            },
            "declared_context": {
                "job_type": "training",
                "job_family": "ml",
                "user": "u1",
            },
            "declared_job_type": None,
        }
    )
    assert bundle["declared_context"]["job_type"] == "training"
