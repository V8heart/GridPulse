from __future__ import annotations

import pandas as pd

from dataset.audit_metadata_leakage import (
    FORBIDDEN_FEATURE_TOKENS,
    aggregate_metadata,
    audit_metadata_leakage,
)


def _frames(*, leaky: bool, n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    telemetry_rows = []
    label_rows = []
    for index in range(n):
        attack = index % 2 == 1
        session_id = f"s-{index:016x}"
        group_id = f"g-{index:016x}"
        for timestamp in (0.0, 1.0):
            telemetry_rows.append(
                {
                    "session_id": session_id,
                    "gpu_id": 0,
                    "timestamp": timestamp,
                    "sample_hz": 20.0 if leaky and attack else 10.0,
                    "requested_interval_ms": 100.0,
                    "gpu_model": "same-model",
                    "declared_job_family": "training",
                    "power_w": 400.0 if attack else 40.0,
                    "gt_label": "secret-attack" if attack else "normal",
                }
            )
        label_rows.append(
            {
                "session_id": session_id,
                "gpu_id": 0,
                "gt_is_attack": attack,
                "group_id": group_id,
            }
        )
    return pd.DataFrame(telemetry_rows), pd.DataFrame(label_rows)


def test_metadata_aggregation_excludes_waveforms_and_truth():
    telemetry, labels = _frames(leaky=False)
    aggregated = aggregate_metadata(telemetry, labels)
    feature_columns = set(aggregated) - {
        "session_id",
        "gpu_id",
        "group_id",
        "gt_is_attack",
    }
    assert "power_w" not in feature_columns
    assert "gt_label" not in feature_columns
    assert not any(
        token in column.lower()
        for column in feature_columns
        for token in FORBIDDEN_FEATURE_TOKENS
    )


def test_metadata_audit_passes_constant_metadata_and_is_group_aware():
    telemetry, labels = _frames(leaky=False)
    report = audit_metadata_leakage(telemetry, labels, folds=3)
    assert report["passed"] is True
    assert report["group_aware"] is True
    assert report["max_oof_auc"] <= 0.60
    assert "session_id" not in report["features"]


def test_metadata_audit_fails_predictive_metadata():
    telemetry, labels = _frames(leaky=True)
    report = audit_metadata_leakage(telemetry, labels, folds=3)
    assert report["passed"] is False
    assert report["max_oof_auc"] > 0.60
