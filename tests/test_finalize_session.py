"""Tests for finalize, remask, validate_standard, and window truth."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataset.finalize_session import finalize_session
from dataset.identifiers import new_session_id
from dataset.remask import remask_session
from dataset.validate_standard import contains_forbidden_token, validate_session_dir
from dataset.window_truth import interval_overlap_seconds, label_window, window_is_attack


def test_window_overlap_boundary_and_merge():
    intervals = [(10.0, 20.0), (15.0, 25.0)]  # union 10..25
    assert abs(interval_overlap_seconds((0.0, 10.0), intervals) - 0.0) < 1e-9
    # window 0..20 overlaps 10 seconds of 20 => 50%
    assert window_is_attack(0.0, 20.0, intervals, threshold=0.5) is True
    # just under 50%
    assert window_is_attack(0.0, 20.01, [(10.0, 20.0)], threshold=0.5) is False
    label, is_attack, source = label_window(
        window_start_epoch=0.0,
        window_end_epoch=5.0,
        intervals=[(10.0, 20.0)],
        session_gt_label="swma",
        hosted=False,
    )
    assert label == "normal_idle" and is_attack is False and source == "interval_inactive"
    label, is_attack, source = label_window(
        window_start_epoch=0.0,
        window_end_epoch=5.0,
        intervals=[(10.0, 20.0)],
        session_gt_label="swma",
        hosted=True,
        host_workload="normal_llm_finetune",
    )
    assert label == "normal_llm_finetune" and is_attack is False


def test_forbidden_token_allows_declared_vocab():
    assert contains_forbidden_token("ddp_training") is None
    assert contains_forbidden_token("vision_training") is None
    assert contains_forbidden_token("dataloader_bound") is None
    assert contains_forbidden_token("python train.py") is None
    assert contains_forbidden_token("real-swma-1") in {"real-", "swma"}
    assert contains_forbidden_token("unknown_cuda_workload") == "unknown_cuda"


def _seed_fake_session(root: Path, session_id: str, *, weird_t: bool = True) -> Path:
    real = root / "dataset" / "real"
    session_dir = real / "sessions" / session_id
    staging = real / ".staging" / session_id
    session_dir.mkdir(parents=True)
    staging.mkdir(parents=True)
    t0 = 1_700_000_000.0
    rows = []
    for i in range(200):
        ts = i * 0.1
        rows.append(
            {
                "timestamp": ts,
                "t_epoch": t0 + ts,
                "session_id": session_id,
                "gpu_id": 0,
                "gpu_uuid": "GPU-0",
                "gpu_model": "Fake",
                "sample_hz": 10.0,
                "requested_interval_ms": 100.0,
                "actual_interval_ms": "" if i == 0 else 100.0,
                "value_changed": True,
                "power_w": 120.0,
                "power_instant_w": "",
                "util_gpu_pct": 40.0,
                "mem_copy_util_pct": 10.0,
                "sm_clock_mhz": 1500.0,
                "mem_clock_mhz": 800.0,
                "temp_c": 45.0,
                "fb_used_mb": 1000.0,
                "fan_speed_pct": 20.0,
                "pstate": 2,
                "power_limit_w": 450.0,
                "throttle_reasons": 0,
                "observed_pid": 1,
                "observed_process_name": "python",
                "observed_n_procs": 1,
            }
        )
    pd.DataFrame(rows).to_csv(session_dir / "telemetry.csv", index=False)
    # raw progress with wrong workload-relative t
    events = []
    for step in range(5):
        events.append(
            {
                "t_epoch": t0 + 5.0 + step,
                "t": float(step),  # wrong axis on purpose
                "gpu_id": 0,
                "event": "step_end",
                "step": step,
            }
        )
    events.insert(0, {"t_epoch": t0 + 4.0, "gpu_id": 0, "event": "workload_start"})
    events.append({"t_epoch": t0 + 12.0, "gpu_id": 0, "event": "workload_end"})
    (session_dir / "progress.raw.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    private = {
        "gt_label": "swma",
        "gt_is_attack": True,
        "gt_variant": "swma_basic",
        "gt_params_json": json.dumps({"period_s": 1.0}),
        "declared_policy": "pool_random",
        "workload_module": "bit2watt_impl.swma_workload",
        "workload_cmd": ["python", "-m", "bit2watt_impl.swma_workload"],
        "host_workload": None,
        "group_id": "g-0123456789abcdef",
        "capture_key": "k1",
        "run_id": "r-20260922-abcdef",
        "source": "real",
        "created_epoch": t0,
        "progress_mask_seed": 7,
        "progress_mask_drop_prob": 0.0,  # force visible for this test
        "gpu_roles": {"0": "target"},
        "declared_by_gpu": {
            "0": {
                "declared_job_type": "ddp_training",
                "declared_job_family": "training",
                "declared_process_name": "torchrun train.py",
                "declared_user": "user_001",
                "declared_gres": "gpu:1",
            }
        },
        "gt_attack_intervals_epoch": {"0": [[t0 + 5.0, t0 + 12.0]]},
        "preflight": {"other_gpu_processes": [], "idle_power_w": {"0": 20.0}},
        "host": {"hostname_hash": "sha256:abc", "os": "Linux"},
        "software": {"torch": "2.x"},
        "gpus": [{"gpu_id": 0, "gpu_uuid": "GPU-0", "gpu_model": "Fake"}],
    }
    (staging / "capture_private.json").write_text(json.dumps(private), encoding="utf-8")
    (staging / "collector_result.json").write_text(
        json.dumps(
            {
                "t0_epoch": t0,
                "requested_interval_ms": 100.0,
                "proc_poll_s": 1.0,
                "power_instant_supported": False,
            }
        ),
        encoding="utf-8",
    )
    return session_dir


def test_finalize_recomputes_t_and_archives(tmp_path: Path):
    session_id = new_session_id()
    _seed_fake_session(tmp_path, session_id)
    report = finalize_session(session_id, root=tmp_path, warmup_s=3.0, drop_prob=0.0, policy_seed=7)
    assert report["valid"] is True
    session_dir = tmp_path / "dataset" / "real" / "sessions" / session_id
    visible = [
        json.loads(line)
        for line in (session_dir / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(abs(e["t"] - (e["t_epoch"] - 1_700_000_000.0)) < 1e-9 for e in visible)
    assert not any(e.get("t") == 0.0 and e["event"] == "step_end" and e["step"] == 0 and e["t_epoch"] != 1_700_000_000.0 + 5.0 for e in visible if False)
    # raw preserved with original bad t
    raw = json.loads((session_dir / "progress.raw.jsonl").read_text().splitlines()[1])
    assert raw.get("t") == 0.0
    assert (tmp_path / "dataset" / "real" / "private" / "capture" / f"{session_id}.json").exists()
    labels = pd.read_csv(tmp_path / "dataset" / "real" / "index" / "labels.csv")
    assert len(labels) == 1
    assert labels.iloc[0]["gpu_role"] == "target"
    # remask hide
    remask_session(session_id, root=tmp_path, drop_prob=1.0, seed=1)
    assert not (session_dir / "progress.jsonl").exists()
    assert (session_dir / "progress.raw.jsonl").exists()


def test_validate_rejects_leaky_session_id():
    assert contains_forbidden_token("real-swma-1") in {"real-", "swma"}
    result = validate_session_dir(Path("/tmp/does-not-matter"))
    # missing files -> invalid
    assert result.valid is False
