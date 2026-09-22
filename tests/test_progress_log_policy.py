from __future__ import annotations

import json
from pathlib import Path

from dataset.build_synthetic import build
from dataset.progress_log_policy import (
    DEFAULT_DROP_PROB,
    POLICY_VERSION,
    decide_progress_log,
    eval_mask_hides_log,
)
from dataset.synth_attacks import cryptojacking, ltma, swma, swma_piggyback
from dataset.synth_normal_patterns import normal_checkpoint
from pipeline.context_evidence import (
    filter_events_for_gpu,
    read_progress_log,
    step_period_from_log,
)


def test_decide_progress_log_deterministic_and_class_independent():
    events = [{"t": 1.0, "event": "step_end", "gpu_id": 0}]
    a = decide_progress_log("sess-a", native_events=events, seed=7, drop_prob=0.4)
    b = decide_progress_log("sess-a", native_events=events, seed=7, drop_prob=0.4)
    assert a == b
    assert a.native_progress_available is True
    assert a.policy_version == POLICY_VERSION

    # Empty native events remain eligible under v1.2; idle alone is the exception.
    empty = decide_progress_log("sess-a", native_events=[], seed=7, drop_prob=0.4)
    assert empty.native_progress_available is True
    idle = decide_progress_log("sess-a", native_events=[], seed=7, drop_prob=0.4, idle_exception=True)
    assert idle.native_progress_available is False
    assert idle.progress_log_masked is False
    assert idle.write_progress_log is False


def test_eval_mask_hides_without_deleting_original(tmp_path: Path):
    path = tmp_path / "steps" / "sess.jsonl"
    path.parent.mkdir(parents=True)
    original = [{"t": 1.0, "gpu_id": 0, "event": "step_end"}]
    path.write_text(json.dumps(original[0]) + "\n", encoding="utf-8")

    assert eval_mask_hides_log("any", seed=1, drop_prob=1.0) is True
    hidden = read_progress_log(
        path,
        session_id="any",
        policy_seed=1,
        drop_prob=1.0,
        native_progress_available=True,
        apply_eval_mask=True,
    )
    assert hidden == []
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8").strip())["event"] == "step_end"


def test_gpu_filter_and_step_period():
    events = [
        {"t": 1.0, "gpu_id": 0, "event": "step_end"},
        {"t": 2.0, "gpu_id": 0, "event": "step_end"},
        {"t": 1.5, "gpu_id": 1, "event": "step_end"},
        {"t": 3.5, "gpu_id": 1, "event": "step_end"},
    ]
    assert [e["gpu_id"] for e in filter_events_for_gpu(events, 0)] == [0, 0]
    period0, n0, _ = step_period_from_log(events, 0.0, 10.0, gpu_id=0)
    period1, n1, _ = step_period_from_log(events, 0.0, 10.0, gpu_id=1)
    assert n0 == 2 and abs(period0 - 1.0) < 1e-9
    assert n1 == 2 and abs(period1 - 2.0) < 1e-9


def test_build_applies_uniform_drop_policy(tmp_path: Path):
    """Until S2 decoys land, some generators may still ship empty native events.

    Policy version and eligible masking semantics are what this gate checks.
    """
    out = tmp_path / "synth"
    build(out, rows=200, sample_hz=10, sessions_per_class=2, seed=7, progress_log_drop_prob=DEFAULT_DROP_PROB)
    manifests = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["progress_log_policy"]["version"] == POLICY_VERSION
    assert split["progress_log_policy"]["drop_prob"] == DEFAULT_DROP_PROB

    eligible = [m for m in manifests if m.get("native_progress_available")]
    assert eligible, "expected some sessions with native progress"

    for item in eligible:
        if item["progress_log_masked"]:
            assert item["progress_log_path"] is None
            assert not (out / "steps" / f"{item['session_id']}.jsonl").exists()
        else:
            assert item["progress_log_path"] is not None
            assert Path(item["progress_log_path"]).exists()


def test_generators_native_availability_truth_table():
    # Documents current synth gaps until Part S2 adds decoy logs for standalone attacks.
    assert (normal_checkpoint(n=1200).attrs.get("progress_events") or [])
    assert (ltma(n=200).attrs.get("progress_events") or [])
    assert (swma_piggyback(n=200).attrs.get("progress_events") or [])
    assert not (swma(n=200).attrs.get("progress_events") or [])
    assert not (cryptojacking(n=200).attrs.get("progress_events") or [])
