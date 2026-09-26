"""Foundation tests for gp-telemetry/1.2 identifiers, schema, declared, progress log."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset.declared_context import (
    FAMILY_TO_DECLARED,
    honest_declared_for_workload,
    honest_job_type_for_workload,
    sample_declared,
    sample_split_partner,
)
from dataset.identifiers import (
    group_id_from_parts,
    is_group_id,
    is_run_id,
    is_session_id,
    new_run_id,
    new_session_id,
    require_session_id,
    synthetic_session_id,
)
from dataset.progress_log_policy import (
    POLICY_VERSION,
    decide_progress_log,
    eval_mask_hides_log,
)
from dataset.schema import (
    CORE_COLUMNS,
    INFERENCE_FORBIDDEN_COLUMNS,
    normalize_frame,
    strip_ground_truth,
)
from workloads.progress_log import DecoyStepper, ProgressLog


def test_opaque_ids_and_regex():
    sid = new_session_id()
    assert is_session_id(sid)
    assert require_session_id(sid) == sid
    with pytest.raises(ValueError):
        require_session_id("real-swma-1")
    syn_a = synthetic_session_id(seed=7, key="attack:swma:0")
    syn_b = synthetic_session_id(seed=7, key="attack:swma:0")
    syn_c = synthetic_session_id(seed=7, key="attack:swma:1")
    assert syn_a == syn_b and syn_a != syn_c and is_session_id(syn_a)
    rid = new_run_id()
    assert is_run_id(rid)
    gid = group_id_from_parts("entry", 3, {"model": "gpt-tiny"})
    assert is_group_id(gid)
    assert group_id_from_parts("entry", 3, {"model": "gpt-tiny"}) == gid


def test_honest_workload_mapping_is_deterministic():
    assert honest_job_type_for_workload("llm_finetune") == "llm_finetune"
    assert honest_job_type_for_workload("pretrain_fsdp") == "ddp_training"
    assert honest_job_type_for_workload("serving") == "online_inference"
    assert honest_job_type_for_workload("resnet_single") == "vision_training"
    assert honest_job_type_for_workload("dataloader_stall") == "dataloader_bound"
    declared = honest_declared_for_workload("llm_finetune", user="user_009")
    assert declared["declared_job_type"] == "llm_finetune"
    assert declared["declared_user"] == "user_009"
    random_honest = sample_declared("training", np.random.default_rng(0), policy="pool_random")
    assert random_honest["declared_job_type"] in set(FAMILY_TO_DECLARED["training"]) | {
        "batch_inference",
        "online_inference",
        "evaluation",
        "notebook",
    }


def test_sample_declared_policies():
    rng = np.random.default_rng(0)
    honest = sample_declared("training", rng, policy="honest", mismatch_rate=0.0)
    assert honest["declared_job_family"] == "training"
    matched = {sample_declared("training", np.random.default_rng(i), policy="host_family_matched")["declared_job_type"] for i in range(40)}
    assert matched <= set(FAMILY_TO_DECLARED["training"])
    pool = sample_declared(None, rng, policy="pool_random")
    restricted = {
        sample_declared(
            "training",
            np.random.default_rng(i),
            policy="pool_random",
            allowed_families=("training", "inference"),
        )["declared_job_family"]
        for i in range(40)
    }
    assert restricted <= {"training", "inference"}
    assert pool["declared_job_type"] in {
        "llm_pretrain",
        "llm_finetune",
        "ddp_training",
        "hpo_sweep",
        "batch_inference",
        "online_inference",
        "evaluation",
        "notebook",
        "vision_training",
        "dataloader_bound",
    }
    host = sample_declared(
        "training",
        rng,
        policy="host_inherited",
        host_declared=honest,
    )
    assert host == honest
    legacy = sample_declared("training", rng, disguise=True)
    assert "declared_job_type" in legacy
    anchor = sample_declared(
        "training",
        np.random.default_rng(1),
        policy="pool_random",
        mismatch_rate=0.0,
        allowed_families=("training", "inference"),
    )
    forced = sample_split_partner(
        anchor,
        np.random.default_rng(2),
        true_family="training",
        policy="pool_random",
        mismatch_rate=0.0,
        allowed_families=("training", "inference"),
        attempts=0,
    )
    assert forced["declared_user"] != anchor["declared_user"]
    assert forced["declared_job_type"] != anchor["declared_job_type"]
    assert forced["declared_job_family"] in {"training", "inference"}


def test_progress_policy_uniform_and_idle_exception():
    assert POLICY_VERSION == "v1_1_uniform_all_sessions"
    a = decide_progress_log("s-aaaaaaaaaaaaaaaa", native_events=[], seed=7, drop_prob=0.4)
    b = decide_progress_log("s-aaaaaaaaaaaaaaaa", native_events=[{"x": 1}], seed=7, drop_prob=0.4)
    assert a == b
    assert a.native_progress_available is True
    idle = decide_progress_log("s-aaaaaaaaaaaaaaaa", seed=7, idle_exception=True)
    assert idle.native_progress_available is False
    assert idle.write_progress_log is False
    assert eval_mask_hides_log("any", seed=1, drop_prob=1.0) is True
    with pytest.raises(ValueError):
        decide_progress_log("s", seed=1, drop_prob=1.5)


def test_schema_legacy_maps_and_strip():
    frame = pd.DataFrame(
        {
            "timestamp": [0.0, 0.1],
            "session_id": ["s-0123456789abcdef", "s-0123456789abcdef"],
            "gpu_id": [0, 0],
            "sample_hz": [10.0, 10.0],
            "power_w": [100.0, 110.0],
            "label": ["swma", "swma"],
            "attack_id": ["swma_basic", "swma_basic"],
            "process_name": ["python", "python"],
            "pid": [42, 42],
            "job_type": ["ddp_training", "ddp_training"],
            "id_user": ["user_001", "user_001"],
            "gres_req": ["gpu:1", "gpu:1"],
        }
    )
    with pytest.warns(RuntimeWarning):
        out = normalize_frame(frame)
    assert out["observed_process_name"].iloc[0] == "python"
    assert out["observed_pid"].iloc[0] == 42
    assert out["gt_variant"].iloc[0] == "swma_basic"
    assert bool(out["gt_is_attack"].iloc[0]) is True
    assert set(CORE_COLUMNS).issubset(out.columns)
    stripped = strip_ground_truth(out)
    assert "gt_label" not in stripped.columns
    assert "gt_is_attack" not in stripped.columns
    assert not (set(stripped.columns) & INFERENCE_FORBIDDEN_COLUMNS)


def _writer(path: str, gpu_id: int, n: int, queue: mp.Queue) -> None:
    log = ProgressLog(path, gpu_id)
    for i in range(n):
        log.emit("step_end", step=i)
    log.close()
    queue.put(gpu_id)


def test_progress_log_streaming_vocab_and_concurrent(tmp_path: Path):
    path = tmp_path / "progress.raw.jsonl"
    with ProgressLog(path, 0) as log:
        log.emit("workload_start")
        log.emit("step_end", step=0)
        log.emit("checkpoint_start", step=1)
        log.emit("checkpoint_end", step=1)
        log.emit("eval_start", step=2)
        log.emit("eval_end", step=2)
        log.emit("workload_end")
    with pytest.raises(ValueError, match="unknown progress event"):
        ProgressLog(path, 0).emit("not_an_event")
    with pytest.raises(ValueError, match="must not record 't'"):
        ProgressLog(path, 0).emit("step_end", step=1, t=1.0)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7
    for line in lines:
        obj = json.loads(line)
        assert "t" not in obj
        assert "t_epoch" in obj

    # Force-kill mid-write still keeps prior lines (separate process simulation).
    kill_path = tmp_path / "kill.jsonl"
    log = ProgressLog(kill_path, 0)
    log.emit("step_end", step=0)
    log.emit("step_end", step=1)
    # do not close; drop reference after flush via emit
    del log
    kept = kill_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(kept) >= 2

    multi = tmp_path / "multi.jsonl"
    queue: mp.Queue = mp.Queue()
    procs = [
        mp.Process(target=_writer, args=(str(multi), gid, 20, queue))
        for gid in (0, 1)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0
    done = {queue.get(timeout=1), queue.get(timeout=1)}
    assert done == {0, 1}
    rows = [json.loads(line) for line in multi.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 40
    assert {r["gpu_id"] for r in rows} == {0, 1}


def test_decoy_stepper_does_not_block(tmp_path: Path):
    path = tmp_path / "decoy.jsonl"
    log = ProgressLog(path, 0)
    stepper = DecoyStepper(log, period_s=0.05, jitter_frac=0.0)
    fired = 0
    t0 = 1000.0
    for i in range(10):
        if stepper.tick(now_mono=t0 + i * 0.06):
            fired += 1
    log.close()
    assert fired >= 5
    assert all(json.loads(line)["event"] == "step_end" for line in path.read_text().splitlines() if line)
