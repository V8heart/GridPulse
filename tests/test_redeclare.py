"""Redeclare honest normals without rewriting progress.raw."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dataset.finalize_session import LABEL_COLUMNS
from dataset.identifiers import new_session_id
from dataset.redeclare import (
    cohort_inventory,
    redeclare_session,
    should_redeclare,
    write_label_declared_types,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _write_session(
    root: Path,
    session_id: str,
    *,
    gt_label: str,
    gt_is_attack: bool,
    policy: str,
    workload: str,
    declared_job_type: str,
    family: str,
    process: str,
) -> Path:
    real = root / "dataset" / "real"
    session_dir = real / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (real / "private" / "capture").mkdir(parents=True, exist_ok=True)
    (real / "index").mkdir(parents=True, exist_ok=True)
    t0 = 1_700_000_000.0
    pd.DataFrame(
        [
            {
                "timestamp": 0.0,
                "t_epoch": t0,
                "session_id": session_id,
                "gpu_id": 0,
                "gpu_model": "Fake",
                "declared_job_type": declared_job_type,
                "declared_job_family": family,
                "declared_process_name": process,
                "declared_user": "user_009",
            },
            {
                "timestamp": 0.1,
                "t_epoch": t0 + 0.1,
                "session_id": session_id,
                "gpu_id": 0,
                "gpu_model": "Fake",
                "declared_job_type": declared_job_type,
                "declared_job_family": family,
                "declared_process_name": process,
                "declared_user": "user_009",
            },
        ]
    ).to_csv(session_dir / "telemetry.csv", index=False)
    raw = session_dir / "progress.raw.jsonl"
    raw.write_text('{"t_epoch": 1700000001.0, "event": "step_end"}\n', encoding="utf-8")
    (session_dir / "progress.jsonl").write_text('{"t": 1.0, "event": "step_end"}\n', encoding="utf-8")
    declared = {
        "declared_job_type": declared_job_type,
        "declared_job_family": family,
        "declared_process_name": process,
        "declared_user": "user_009",
    }
    (session_dir / "session.json").write_text(
        json.dumps({"t0_epoch": t0, "warmup_s": 0.0, "declared": {"0": declared}}),
        encoding="utf-8",
    )
    private = {
        "gt_label": gt_label,
        "gt_is_attack": gt_is_attack,
        "gt_variant": workload,
        "gt_params_json": json.dumps({"workload": workload}),
        "declared_policy": policy,
        "gpu_roles": {"0": "target"},
        "declared_by_gpu": {"0": declared},
    }
    (real / "private" / "capture" / f"{session_id}.json").write_text(
        json.dumps(private), encoding="utf-8"
    )
    return session_dir


def test_should_redeclare_honest_normals_only():
    assert should_redeclare({"declared_policy": "honest", "gt_is_attack": False, "gt_label": "normal_hpo_search"})
    assert not should_redeclare({"declared_policy": "pool_random", "gt_is_attack": True, "gt_label": "swma"})
    assert not should_redeclare({"declared_policy": "host_inherited", "gt_is_attack": True, "gt_label": "ltma"})
    assert not should_redeclare({"declared_policy": "honest", "gt_is_attack": True, "gt_label": "swma"})


def test_redeclare_patches_honest_target_and_keeps_raw(tmp_path: Path):
    honest_id = new_session_id()
    attack_id = new_session_id()
    honest_dir = _write_session(
        tmp_path,
        honest_id,
        gt_label="normal_hpo_search",
        gt_is_attack=False,
        policy="honest",
        workload="hpo",
        declared_job_type="ddp_training",
        family="training",
        process="torchrun train.py",
    )
    attack_dir = _write_session(
        tmp_path,
        attack_id,
        gt_label="swma",
        gt_is_attack=True,
        policy="pool_random",
        workload="swma",
        declared_job_type="llm_finetune",
        family="training",
        process="python finetune.py",
    )
    honest_raw = _hash(honest_dir / "progress.raw.jsonl")
    honest_visible = (honest_dir / "progress.jsonl").read_text(encoding="utf-8")
    attack_before = json.loads((tmp_path / "dataset/real/private/capture" / f"{attack_id}.json").read_text())
    honest_rows = redeclare_session(honest_id, root=tmp_path)
    attack_rows = redeclare_session(attack_id, root=tmp_path)
    assert honest_rows[0]["before"] == "ddp_training"
    assert honest_rows[0]["after"] == "hpo_sweep"
    assert honest_rows[0]["changed"] is True
    assert attack_rows[0]["changed"] is False
    assert attack_rows[0]["after"] == "llm_finetune"
    assert _hash(honest_dir / "progress.raw.jsonl") == honest_raw
    assert (honest_dir / "progress.jsonl").read_text(encoding="utf-8") == honest_visible
    assert _hash(attack_dir / "progress.raw.jsonl")
    private = json.loads((tmp_path / "dataset/real/private/capture" / f"{honest_id}.json").read_text())
    assert private["declared_by_gpu"]["0"]["declared_job_type"] == "hpo_sweep"
    assert private["declared_by_gpu"]["0"]["declared_user"] == "user_009"
    telemetry = pd.read_csv(honest_dir / "telemetry.csv")
    assert telemetry["declared_job_type"].iloc[0] == "hpo_sweep"
    attack_after = json.loads((tmp_path / "dataset/real/private/capture" / f"{attack_id}.json").read_text())
    assert attack_after["declared_by_gpu"] == attack_before["declared_by_gpu"]


def test_labels_declared_job_type_and_inventory(tmp_path: Path):
    sid = new_session_id()
    _write_session(
        tmp_path,
        sid,
        gt_label="normal_hpo_search",
        gt_is_attack=False,
        policy="honest",
        workload="hpo",
        declared_job_type="hpo_sweep",
        family="training",
        process="python sweep.py",
    )
    labels = tmp_path / "dataset/real/index/labels.csv"
    pd.DataFrame(
        [
            {
                "session_id": sid,
                "gpu_id": 0,
                "gt_label": "normal_hpo_search",
                "gt_is_attack": False,
                "gpu_role": "target",
                "valid": True,
            }
        ]
    ).to_csv(labels, index=False)
    write_label_declared_types(root=tmp_path)
    frame = pd.read_csv(labels)
    assert "declared_job_type" in frame.columns
    assert set(LABEL_COLUMNS).issubset(set(frame.columns))
    assert frame["declared_job_type"].iloc[0] == "hpo_sweep"
    report = cohort_inventory(frame)
    assert report["rows"][0]["n_normal"] == 1
    assert report["rows"][0]["est_train"] == 0.3
    assert report["recommend_decision"] is True
