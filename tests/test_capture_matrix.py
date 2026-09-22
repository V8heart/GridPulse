"""Capture matrix dry-run / group_id / duration tests."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dataset.capture_matrix import build_matrix, command_for, load_config, summarize
from dataset.identifiers import is_group_id

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_seed_reproducible_and_group_independent_of_repeat():
    config = load_config(ROOT / "config" / "capture_matrix.yaml")
    a = build_matrix(config)
    b = build_matrix(config)
    assert [x["capture_key"] for x in a] == [x["capture_key"] for x in b]
    assert all(is_group_id(x["group_id"]) for x in a)
    # Identical resolved params share a group; different seq/period params do not.
    by_params = {}
    for item in a:
        key = (item["workload"], json.dumps(item["params"], sort_keys=True))
        by_params.setdefault(key, set()).add(item["group_id"])
    assert all(len(groups) == 1 for groups in by_params.values())
    finetune = [item for item in a if item["workload"] == "llm_finetune"]
    assert len({item["group_id"] for item in finetune}) >= 2
    assert all(item["duration_s"] in {480, 600, 720} for item in a)


def test_attack_odd_repeats_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
schema: kdn-capture-matrix/1
seed: 1
duration_pool_s: [480, 600, 720]
cooldown: {min_seconds: 60, max_wait_seconds: 900, idle_temp_delta_c: 3}
entries:
  - workload: swma
    repeats: 3
""",
        encoding="utf-8",
    )
    try:
        load_config(bad)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "odd" in str(exc).lower()


def test_matrix_balance_policies_periods_and_planning_budget():
    config = load_config(ROOT / "config" / "capture_matrix.yaml")
    items = build_matrix(config)
    summary = summarize(items, 60)
    assert summary["session_count"] == 116
    assert summary["normal_sessions"] == 68
    assert summary["attack_sessions"] == 48
    assert summary["planning_hours"] > summary["estimated_hours"]
    hosted = {"swma_piggyback", "swma_mimicry", "ltma"}
    standalone = {"swma_basic", "swma_shallow", "swma_jitter", "swma_coordinated", "cryptojacking"}
    for name in standalone:
        policies = Counter(
            item["params"]["declared_policy"] for item in items if item["workload"] == name
        )
        assert policies["pool_random"] == policies["host_family_matched"]
    for name in hosted:
        policies = {item["params"]["declared_policy"] for item in items if item["workload"] == name}
        assert policies == {"host_inherited"}
    periods = {
        item["params"]["period"]
        for item in items
        if item["workload"] == "swma_basic"
    }
    assert periods <= {0.5, 1.0, 2.0, 4.0}
    command = command_for(items[0], allow_busy=False, confirm=False)
    assert "--params-grid" not in command
    llm = next(item for item in items if item["workload"] == "llm_finetune")
    llm_cmd = command_for(llm, allow_busy=False, confirm=False)
    assert "--preset" in llm_cmd and "small" in llm_cmd
    hosted_cmd = command_for(
        next(item for item in items if item["workload"] == "swma_piggyback"),
        allow_busy=False,
        confirm=False,
    )
    assert "host_inherited" not in hosted_cmd
    assert "--params-grid" not in hosted_cmd
    assert summary["hosted_vs_standalone"]["hosted"] == 18
    assert summary["hosted_vs_standalone"]["standalone"] == 30


def test_reports_use_actual_period_and_mark_missing_real_power():
    import pandas as pd

    from dataset.report_synth_vs_real import compare_synth_vs_real
    from pipeline.report_stratified import stratified_report

    frame = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "is_candidate": [True, False],
            "gt_is_attack": [True, True],
            "gt_label": ["swma", "swma"],
            "declared_job_family": ["evaluation", "evaluation"],
            "gpu_role": ["target", "target"],
            "warmup": [False, False],
            "gt_params_json": [
                json.dumps({"period_requested_s": 1.0, "period_actual_s": 0.4}),
                json.dumps({"period_requested_s": 1.0, "period_actual_s": 2.0}),
            ],
        }
    )
    report = stratified_report(frame)
    assert report["no_normal_cohort"] == ["evaluation"]
    periods = {row["value"] for row in report["rows"] if row["axis"] == "period_actual_s"}
    assert periods == {"0.4", "2.0"}
    windows = pd.DataFrame(
        {
            "gt_label": ["normal_mlp", "swma"],
            "mean_w": [100.0, 220.0],
            "gt_is_attack": [False, True],
        }
    )
    compared = compare_synth_vs_real(windows, windows)
    assert compared["normal_llm_training"]["status"] == "pending_real_capture"
