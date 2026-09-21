from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.run_pipeline import _write_stage1_envelope


def test_stage1_envelope_includes_grid_watch_and_zero_candidates(tmp_path: Path):
    wdf = pd.DataFrame(
        [
            {
                "session_id": "s1",
                "gpu_id": 0,
                "start": 0,
                "end": 30,
                "window_start_s": 0.0,
                "window_end_s": 30.0,
                "declared_job_family": "training",
                "baseline_level": "full",
                "u_score": 1.0,
                "p_value": 0.5,
                "impact_raw": 0.9,
                "impact_level": "high",
                "multi_gpu_sync_index": 0.0,
                "cross_job_sync_index": 0.0,
                "is_candidate": False,
                "grid_watch": True,
                "candidate_reasons": "",
                "impact_components": {"impact_raw": 0.9},
            },
            {
                "session_id": "s1",
                "gpu_id": 0,
                "start": 15,
                "end": 45,
                "window_start_s": 15.0,
                "window_end_s": 45.0,
                "declared_job_family": "training",
                "baseline_level": "full",
                "u_score": 3.0,
                "p_value": 0.01,
                "impact_raw": 0.1,
                "impact_level": "low",
                "multi_gpu_sync_index": 0.0,
                "cross_job_sync_index": 0.0,
                "is_candidate": True,
                "grid_watch": False,
                "candidate_reasons": "mean_w",
                "impact_components": {"impact_raw": 0.1},
            },
        ]
    )
    path = tmp_path / "stage1.json"
    envelope = _write_stage1_envelope(path, wdf, telemetry="x.csv", stage1_version="v2")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["n_windows"] == 2
    assert loaded["summary"]["n_candidates"] == 1
    assert loaded["summary"]["n_grid_watch"] == 1
    assert loaded["grid_watch"][0]["grid_watch"] is True
    assert loaded["candidates"][0]["is_candidate"] is True
    assert envelope["all_windows"][0]["impact_raw"] == 0.9


def test_stage1_envelope_writes_even_when_empty(tmp_path: Path):
    path = tmp_path / "empty.json"
    envelope = _write_stage1_envelope(path, pd.DataFrame(), telemetry="x.csv", stage1_version="v2")
    assert envelope["summary"]["n_windows"] == 0
    assert path.exists()
