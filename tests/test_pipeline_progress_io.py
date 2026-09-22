from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.window_truth import window_is_attack
from pipeline.context_evidence import event_time_s, read_session_progress
from pipeline.stage1_v2 import build_windows, load_config


def test_pipeline_reads_session_dir_progress_log(tmp_path: Path):
    session = tmp_path / "s-0123456789abcdef"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"t0_epoch": 1000.0}))
    (session / "progress.jsonl").write_text(
        json.dumps({"event": "step_end", "t": 999.0, "t_epoch": 1002.5}) + "\n"
    )
    (session / "progress.raw.jsonl").write_text(
        json.dumps({"event": "step_end", "t_epoch": 1009.0}) + "\n"
    )
    events = read_session_progress(session)
    assert len(events) == 1
    assert events[0]["t"] == 2.5


def test_session_reader_never_uses_raw_log(tmp_path: Path):
    session = tmp_path / "s-0123456789abcdef"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"t0_epoch": 1000.0}))
    (session / "progress.raw.jsonl").write_text(
        json.dumps({"event": "step_end", "t_epoch": 1001.0}) + "\n"
    )
    assert read_session_progress(session) == []
    staging = tmp_path / ".staging" / "s-0123456789abcdef"
    staging.mkdir(parents=True)
    (staging / "progress.jsonl").write_text(
        json.dumps({"event": "step_end", "t": 1.0}) + "\n"
    )
    assert read_session_progress(staging) == []


def test_event_time_uses_epoch_and_ignores_workload_t():
    event = {"t": 700.0, "t_epoch": 1012.0}
    assert event_time_s(event, t0_epoch=1000.0) == 12.0
    assert event_time_s(event) is None
    assert event_time_s({"t": 7.0}) == 7.0


def test_window_overlap_49_9_vs_50_percent():
    assert window_is_attack(0.0, 10.0, [(0.0, 4.99)]) is False
    assert window_is_attack(0.0, 10.0, [(0.0, 5.0)]) is True


def _telemetry() -> pd.DataFrame:
    timestamp = np.arange(10, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "t_epoch": 1000.0 + timestamp,
            "session_id": ["s-0123456789abcdef"] * 10,
            "gpu_id": [0] * 10,
            "sample_hz": [1.0] * 10,
            "power_w": 100.0 + np.sin(timestamp),
            "util_gpu_pct": np.linspace(20, 80, 10),
            "warmup": [True] * 4 + [False] * 6,
            "declared_job_family": ["training"] * 10,
            "gpu_model": ["fake"] * 10,
            # These must never become inference output truth.
            "gt_label": ["swma"] * 10,
        }
    )


def test_build_windows_truth_overlap_warmup_and_inference_no_gt():
    frame = _telemetry()
    config = load_config()
    truth = {
        ("s-0123456789abcdef", 0): {
            "intervals": [(1000.0, 1004.5)],
            "session_gt_label": "swma",
            "host_workload": None,
            "hosted": False,
            "gpu_role": "target",
        }
    }
    evaluated = build_windows(
        frame,
        window_s=8.0,
        stride_s=8.0,
        config=config,
        truth_index=truth,
    )
    assert bool(evaluated.iloc[0]["gt_is_attack"]) is True
    assert evaluated.iloc[0]["truth_source"] == "interval_overlap"
    assert bool(evaluated.iloc[0]["warmup"]) is True
    assert evaluated.iloc[0]["window_id"] == "s-0123456789abcdef:0:0:8"

    inference = build_windows(
        frame,
        window_s=8.0,
        stride_s=8.0,
        config=config,
        truth_index=None,
    )
    assert not any(column.startswith("gt_") for column in inference.columns)
    assert "truth_source" not in inference.columns
