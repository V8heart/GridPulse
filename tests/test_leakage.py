from __future__ import annotations

import json
from argparse import Namespace

import pandas as pd
import pytest

from dataset.build_synthetic import build
from dataset.schema import INFERENCE_FORBIDDEN_COLUMNS, strip_ground_truth
from dataset.synth_common import build_frame
from pipeline.fit_corpus_ranges import fit_ranges
from pipeline.run_pipeline import context_to_query, run


def test_strip_ground_truth_removes_forbidden_columns(tmp_path):
    df = build(tmp_path / "synthetic", rows=120, sessions_per_class=1)
    stripped = strip_ground_truth(df)
    assert not (set(stripped.columns) & INFERENCE_FORBIDDEN_COLUMNS)
    assert "declared_job_type" in stripped


def test_context_query_contains_no_corpus_doc_names():
    text = context_to_query({"declared_job_type": "ddp_training", "declared_job_family": "training"})
    for forbidden in ("swma", "ltma", "cryptojacking", "llmjacking"):
        assert forbidden not in text.lower()


def test_build_frame_requires_declared_context():
    with pytest.raises(ValueError, match="declared_job_type"):
        build_frame([120.0] * 10, label="x", session_id="s", sample_hz=1)


def test_split_disjoint_sessions(tmp_path):
    build(tmp_path / "synthetic", rows=120, sessions_per_class=3)
    manifest = json.loads((tmp_path / "synthetic/split_manifest.json").read_text())
    splits = [set(manifest[name]) for name in ("train", "cal", "test")]
    assert not (splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])


def test_unseen_holdout_is_test_only(tmp_path):
    build(tmp_path / "synthetic", rows=120, sessions_per_class=4)
    manifest = json.loads((tmp_path / "synthetic/split_manifest.json").read_text())
    holdout = set(manifest["unseen_param_holdout"]["swma_period_2_8_3_2"])
    assert holdout
    assert holdout.isdisjoint(set(manifest["train"]) | set(manifest["cal"]))
    assert holdout.issubset(set(manifest["test"]))


def test_corpus_ranges_fit_on_train_only(tmp_path):
    build(tmp_path / "synthetic", rows=320, sessions_per_class=3)
    report = fit_ranges(tmp_path / "synthetic/all_v3.csv", tmp_path / "synthetic/split_manifest.json")
    assert report["fit_split"] == "train"
    assert report["n_sessions"] == len(json.loads((tmp_path / "synthetic/split_manifest.json").read_text())["train"])


def test_permuting_ground_truth_does_not_change_outputs(tmp_path, capsys):
    df = build(tmp_path / "synthetic", rows=220, sessions_per_class=1)
    src = tmp_path / "synthetic/all_v3.csv"
    permuted = df.copy()
    permuted["gt_label"] = list(reversed(permuted["gt_label"].tolist()))
    permuted["gt_attack_id"] = list(reversed(permuted["gt_attack_id"].tolist()))
    dst = tmp_path / "synthetic/permuted.csv"
    permuted.to_csv(dst, index=False)
    args = dict(
        baseline_mean=120.0,
        sample_hz=None,
        window=200,
        stride=200,
        z_threshold=2.5,
        rag_backend="tfidf",
        llm_backend="stub",
        llm_model="unused",
        physics_validate=False,
        physics_timeout_s=60.0,
        physics_test_system="kundur_ieeest",
        out=None,
        stage1="legacy",
        stage2="legacy",
        query_context="off",
    )
    first = run(Namespace(telemetry=str(src), **args))
    capsys.readouterr()
    second = run(Namespace(telemetry=str(dst), **args))
    capsys.readouterr()
    comparable = ["window_id", "candidate_reasons", "rag_top", "threat_id", "anomaly_score"]
    assert [{k: row[k] for k in comparable} for row in first] == [
        {k: row[k] for k in comparable} for row in second
    ]
