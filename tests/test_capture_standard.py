"""Part T capture-orchestration and matrix safety checks."""
from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from dataset.capture_matrix import build_matrix, load_config, shard
from dataset.run_capture import DURATION_POOL, build_plan

ROOT = Path(__file__).resolve().parent.parent


def _args(**overrides) -> Namespace:
    values = {
        "workload": "swma",
        "gpu_id": 0,
        "gpu_ids_override": "0,1",
        "duration": None,
        "interval_ms": 100.0,
        "period": 1.0,
        "duty_cycle": 0.5,
        "session_id": "s-0123456789abcdef",
        "run_id": None,
        "group_id": None,
        "capture_key": None,
        "seed": 7,
        "pre_capture_s": 10.0,
        "post_capture_s": 10.0,
        "warmup_s": 180.0,
        "proc_poll_s": 1.0,
        "progress_mask_seed": 7,
        "progress_mask_drop_prob": 0.4,
        "declared_policy": "pool_random",
        "dry_run": True,
        "probe_cmd": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dry_plan_has_opaque_public_paths_shared_duration_and_companions():
    plan = build_plan(_args())
    public = json.dumps({
        "session_id": plan["session_id"],
        "session_dir": plan["session_dir"],
        "telemetry": plan["telemetry"],
        "progress_raw": plan["progress_raw"],
    }).lower()
    assert not any(token in public for token in ("attack", "swma", "ltma", "crypto", "real-"))
    assert plan["duration_s"] in DURATION_POOL
    assert plan["gpu_roles"] == {"0": "target", "1": "companion_idle"}
    assert plan["private"]["warmup_s"] == 180.0


def test_real_capture_requires_explicit_shared_gpu_safety_flag():
    result = subprocess.run(
        [
            sys.executable, "-m", "dataset.run_capture",
            "--workload", "baseline", "--duration", "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "confirm-shared-gpu-safe" in result.stderr


def test_matrix_keys_shards_and_even_attack_repeats(tmp_path: Path):
    config = load_config(ROOT / "config" / "capture_matrix.yaml")
    matrix = build_matrix(config)
    assert len(matrix) == 140
    assert 22 <= (sum(x["duration_s"] for x in matrix) + 60 * 139) / 3600 <= 26
    assert len({x["capture_key"] for x in matrix}) == len(matrix)
    left, right = shard(matrix, 2, 0), shard(matrix, 2, 1)
    assert {x["capture_key"] for x in left}.isdisjoint(x["capture_key"] for x in right)
    assert {x["capture_key"] for x in left + right} == {x["capture_key"] for x in matrix}

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema: kdn-capture-matrix/1\nseed: 1\nduration_pool_s: [480, 600, 720]\n"
        "cooldown: {min_seconds: 60, max_wait_seconds: 900, idle_temp_delta_c: 3}\n"
        "entries:\n  - {workload: swma, repeats: 3}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="odd repeats"):
        load_config(bad)
import re

from dataset.identifiers import SESSION_ID_RE
from dataset.run_capture import ATTACK_WORKLOADS, DURATION_POOL, NORMAL_MODES, build_plan
from dataset.validate_standard import contains_forbidden_token


class _Args:
    def __init__(self, **kwargs):
        defaults = dict(
            workload="baseline",
            gpu_id=0,
            gpu_ids_override=None,
            duration=None,
            interval_ms=100.0,
            period=1.0,
            duty_cycle=0.5,
            session_id=None,
            run_id=None,
            group_id=None,
            capture_key=None,
            seed=7,
            pre_capture_s=10.0,
            post_capture_s=10.0,
            proc_poll_s=1.0,
            progress_mask_seed=7,
            progress_mask_drop_prob=0.4,
            declared_policy="pool_random",
            dry_run=True,
            allow_busy=False,
            confirm_shared_gpu_safe=False,
            skip_finalize=False,
            warmup_s=180.0,
            preflight_s=5.0,
            probe_cmd=None,
        )
        defaults.update(kwargs)
        self.__dict__.update(defaults)


def test_dry_run_opaque_paths_and_shared_duration_pool():
    plan = build_plan(_Args(workload="swma", seed=11))
    assert SESSION_ID_RE.fullmatch(plan["session_id"])
    assert "swma" not in plan["session_id"]
    assert "attack" not in plan["session_dir"].lower()
    assert plan["duration_s"] in DURATION_POOL
    blob = json.dumps(plan)
    assert contains_forbidden_token(plan["session_id"]) is None
    assert "/sessions/" in plan["session_dir"]
    assert ".staging" in plan["staging_dir"]
    assert "private/stdout" in plan["private_stdout"]
    assert "workload_stdout" not in blob or "private/stdout" in plan["private_stdout"]


def test_companion_roles_and_attack_progress_log():
    plan = build_plan(_Args(workload="baseline", gpu_ids_override="0,1"))
    roles = plan["gpu_roles"]
    assert roles["0"] == "target"
    assert roles["1"] == "companion_idle"
    attack = build_plan(_Args(workload="swma", gpu_ids_override="0,1"))
    assert "--progress-log" in attack["workload"]
    assert any("progress.raw.jsonl" in str(x) for x in attack["workload"])


def test_duration_distribution_label_independent():
    normals = [build_plan(_Args(workload="baseline", seed=s, session_id=f"s-{s:016x}"))["duration_s"] for s in range(30)]
    attacks = [build_plan(_Args(workload="swma", seed=s, session_id=f"s-{(s+100):016x}"))["duration_s"] for s in range(30)]
    assert set(normals) <= set(DURATION_POOL)
    assert set(attacks) <= set(DURATION_POOL)
    # Both classes can draw every pool member with enough seeds.
    assert len(set(normals)) >= 2
    assert len(set(attacks)) >= 2


def test_workload_registry_covers_matrix_core():
    for name in (*NORMAL_MODES, *ATTACK_WORKLOADS):
        plan = build_plan(_Args(workload=name, dry_run=True))
        assert plan["workload"]
        assert re.fullmatch(r"s-[0-9a-f]{16}", plan["session_id"])
