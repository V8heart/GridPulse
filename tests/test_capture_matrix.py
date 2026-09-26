"""Capture matrix dry-run / group_id / duration tests."""
from __future__ import annotations

import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

from bit2watt_impl.attack_variants import hosted_scale
from dataset.capture_matrix import (
    ATTACK_WORKLOADS,
    NORMAL_WORKLOADS,
    build_matrix,
    cleanup_incomplete_capture,
    command_for,
    execute_captures,
    load_config,
    summarize,
)
from dataset.identifiers import is_group_id
from dataset.run_capture import build_plan

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
    matrices = {
        item["params"]["matrix_size"]
        for item in items
        if item["workload"] == "swma_basic"
    }
    assert matrices <= {2048, 4096}
    crypto_sizes = {
        item["params"]["matrix_size"]
        for item in items
        if item["workload"] == "cryptojacking"
    }
    assert crypto_sizes == {2048, 3072, 4096}
    crypto_cmd = command_for(
        next(item for item in items if item["workload"] == "cryptojacking"),
        allow_busy=False,
        confirm=False,
    )
    assert "--matrix-size" in crypto_cmd
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
    floored = pd.DataFrame(
        {
            "session_id": ["s3"],
            "is_candidate": [True],
            "gt_is_attack": [True],
            "gt_label": ["swma"],
            "gt_variant": ["swma_mimicry"],
            "declared_job_family": ["training"],
            "gpu_role": ["target"],
            "warmup": [False],
            "gt_params_json": [
                json.dumps(
                    {
                        "period_requested_s": 0.01,
                        "period_actual_s": 0.05,
                        "period_floored": True,
                        "matrix_size": 4096,
                    }
                )
            ],
        }
    )
    floored_report = stratified_report(floored)
    variants = {
        row["value"]
        for row in floored_report["rows"]
        if row["axis"] == "gt_variant"
    }
    assert "swma_floored" in variants
    assert "swma_mimicry" not in variants
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


def _plan_for(item: dict) -> dict:
    params = item["params"]
    return build_plan(Namespace(
        workload=item["workload"],
        gpu_id=item["gpu_id"],
        gpu_ids_override="0,1",
        duration=item["duration_s"],
        interval_ms=100.0,
        period=params.get("period", 1.0),
        duty_cycle=params.get("duty_cycle", 0.5),
        session_id=None,
        run_id=None,
        group_id=item["group_id"],
        capture_key=item["capture_key"],
        seed=item["seed"],
        pre_capture_s=10.0,
        post_capture_s=10.0,
        warmup_s=180.0,
        proc_poll_s=1.0,
        progress_mask_seed=7,
        progress_mask_drop_prob=0.4,
        declared_policy=params.get("declared_policy", "pool_random"),
        dry_run=True,
        probe_cmd=None,
        preset=params.get("preset"),
        seq_len=params.get("seq_len"),
        grad_accum=params.get("grad_accum"),
        batch_size=params.get("batch_size"),
        rps=params.get("rps"),
        dataloader=params.get("dataloader"),
        host=params.get("host", "llm"),
    ))


def _primary_user(plan: dict) -> str:
    target_ids = sorted(
        (gpu_id for gpu_id, role in plan["gpu_roles"].items() if role == "target"),
        key=int,
    )
    return plan["declared_by_gpu"][target_ids[0]]["declared_user"]


def test_capture_seeds_are_unique_reproducible_and_spread_declarations():
    config = load_config(ROOT / "config" / "capture_matrix.yaml")
    first = build_matrix(config)
    second = build_matrix(config)
    assert [item["seed"] for item in first] == [item["seed"] for item in second]
    assert len({item["seed"] for item in first}) == len(first) == 116
    for item in first:
        command = command_for(item, allow_busy=False, confirm=False)
        assert command[command.index("--seed") + 1] == str(item["seed"])

    normal_users = []
    attack_users = []
    for item in first:
        user = _primary_user(_plan_for(item))
        if item["workload"] in NORMAL_WORKLOADS:
            normal_users.append(user)
        elif item["workload"] in ATTACK_WORKLOADS:
            attack_users.append(user)
    for users in (normal_users, attack_users):
        counts = Counter(users)
        assert max(counts.values()) / len(users) <= 0.20

    scales = [
        hosted_scale(item["seed"], "piggyback")
        for item in first
        if item["workload"] == "swma_piggyback"
    ]
    assert len(scales) == 6
    assert len(set(scales)) == len(scales)


def test_execute_captures_continues_after_a_failed_cell(tmp_path):
    import subprocess

    remaining = [
        {"capture_key": "c-fail", "workload": "resnet_ddp", "duration_s": 480, "gpu_id": 0},
        {"capture_key": "c-ok", "workload": "baseline", "duration_s": 480, "gpu_id": 0},
    ]
    seen = []

    def runner(item):
        seen.append(item["capture_key"])
        code = 1 if item["capture_key"] == "c-fail" else 0
        return subprocess.CompletedProcess(args=["x"], returncode=code)

    cooled = []
    failures = execute_captures(
        remaining,
        runner=runner,
        cooldown_fn=lambda *a, **k: cooled.append(a[0] if a else True),
        cooldown_cfg={
            "min_seconds": 0,
            "max_wait_seconds": 0,
            "idle_temp_c": None,
            "idle_temp_delta_c": 3,
            "poll_seconds": 1,
        },
        failures_path=tmp_path / "failures.jsonl",
        cleanup_fn=lambda key: [key],
    )
    assert seen == ["c-fail", "c-ok"]
    assert [row["capture_key"] for row in failures] == ["c-fail"]
    assert cooled == [0]
    lines = (tmp_path / "failures.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["capture_key"] == "c-fail"


def test_cleanup_incomplete_capture_only_touches_matching_key(tmp_path):
    staging = tmp_path / "dataset" / "real" / ".staging" / "s-deadbeefdeadbeef"
    staging.mkdir(parents=True)
    (staging / "capture_private.json").write_text(
        json.dumps({"capture_key": "c-target"}), encoding="utf-8"
    )
    session = tmp_path / "dataset" / "real" / "sessions" / "s-deadbeefdeadbeef"
    session.mkdir(parents=True)
    (session / "telemetry.csv").write_text("x\n", encoding="utf-8")
    other = tmp_path / "dataset" / "real" / ".staging" / "s-aaaaaaaaaaaaaaaa"
    other.mkdir(parents=True)
    (other / "capture_private.json").write_text(
        json.dumps({"capture_key": "c-keep"}), encoding="utf-8"
    )
    removed = cleanup_incomplete_capture("c-target", root=tmp_path)
    assert not session.exists()
    assert not staging.exists()
    assert other.exists()
    assert any("s-deadbeefdeadbeef" in path for path in removed)
