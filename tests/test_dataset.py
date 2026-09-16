from __future__ import annotations

import pandas as pd
import pytest

from dataset.build_synthetic import build
from dataset.schema import CORE_COLUMNS, validate_frame
from dataset.synth_attacks import cryptojacking, ltma, swma
from pipeline.eval_retrieval import build_cases, evaluate


def test_build_is_deterministic(tmp_path):
    first = build(tmp_path / "a", rows=400, sample_hz=10)
    second = build(tmp_path / "b", rows=400, sample_hz=10)
    pd.testing.assert_frame_equal(first, second)
    assert set(CORE_COLUMNS).issubset(first.columns)
    assert first["session_id"].nunique() >= 14


def test_attack_profiles_have_expected_shapes():
    swma_df = swma()
    ltma_df = ltma()
    crypto_df = cryptojacking()
    assert swma_df["power_w"].max() - swma_df["power_w"].min() > 250
    assert abs(ltma_df["power_w"].mean() - 120) < 10
    assert crypto_df["power_w"].mean() > 250
    assert crypto_df["power_w"].std() < 5


def test_schema_rejects_mixed_labels_in_one_session():
    frame = swma(n=100)
    frame.loc[0, "label"] = "normal"
    with pytest.raises(ValueError, match="여러 label"):
        validate_frame(frame)


def test_synthetic_retrieval_and_open_set(tmp_path):
    build(tmp_path / "synthetic", rows=400, sample_hz=10)
    cases = build_cases(tmp_path / "synthetic" / "all_v2.csv")
    report = evaluate(cases, backend="tfidf", unknown_threshold=0.28)
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    assert report["unknown_rejection_rate"] == 1.0

