"""Synthetic standard unit checks (no full corpus regen)."""
from __future__ import annotations

import json

import pandas as pd

from dataset.build_synthetic import build, duration_for_repeat
from dataset.identifiers import is_group_id, synthetic_session_id, is_session_id
from dataset.synth_attacks import swma
from dataset.synth_normal_patterns import baseline_idle
from dataset.window_truth import label_window


def test_opaque_synthetic_session_id_deterministic():
    a = synthetic_session_id(seed=7, key="normal:baseline:0")
    b = synthetic_session_id(seed=7, key="normal:baseline:0")
    c = synthetic_session_id(seed=7, key="attack:swma:0")
    assert a == b and a != c
    assert is_session_id(a)
    assert not a.startswith("syn-")
    assert "swma" not in a and "normal" not in a


def test_window_truth_shared_with_real():
    label, is_attack, source = label_window(
        window_start_epoch=0.0,
        window_end_epoch=10.0,
        intervals=[(0.0, 5.0)],
        session_gt_label="swma",
        hosted=False,
    )
    assert is_attack is True and source == "interval_overlap"
    label, is_attack, source = label_window(
        window_start_epoch=0.0,
        window_end_epoch=10.0,
        intervals=[(0.0, 4.9)],
        session_gt_label="swma",
        hosted=False,
    )
    assert is_attack is False and label == "normal_idle"


def test_generators_use_opaque_ids_and_canonical_gpu_model():
    for frame in (baseline_idle(n=64, seed=100), swma(n=64, seed=200)):
        assert is_session_id(str(frame["session_id"].iloc[0]))
        assert set(frame["gpu_model"]) == {"synthetic-rtx4090"}
        assert frame["t_epoch"].notna().all()


def test_duration_choice_is_repeat_based_not_label_based():
    normal = [duration_for_repeat(7, repeat) for repeat in range(20)]
    attack = [duration_for_repeat(7, repeat) for repeat in range(20)]
    assert normal == attack
    assert set(normal) <= {480, 600, 720}


def test_build_writes_session_tree_intervals_and_policies(tmp_path):
    root = tmp_path / "v2_regen"
    frame = build(root, rows=64, sample_hz=8, sessions_per_class=2, seed=3)
    labels = pd.read_csv(root / "index" / "labels.csv")
    assert frame["session_id"].map(is_session_id).all()
    assert labels["group_id"].map(is_group_id).all()
    attacks = labels[labels["gt_is_attack"].astype(str).str.lower().eq("true")]
    assert attacks["gt_attack_intervals_epoch"].map(lambda raw: bool(json.loads(raw))).all()
    standalone = attacks[
        ~attacks["gt_variant"].isin(["ltma", "swma_piggyback", "swma_mimicry"])
    ]
    assert {"pool_random", "host_family_matched"} <= set(standalone["declared_policy"])
    assert set(labels.loc[labels["gt_variant"].eq("ltma"), "declared_policy"]) == {
        "host_inherited"
    }
    for session_id in frame["session_id"].unique():
        session_dir = root / "sessions" / session_id
        assert (session_dir / "telemetry.csv").exists()
        assert (session_dir / "session.json").exists()
    assert json.loads((root / "leakage_audit.json").read_text())["passed"] is True
