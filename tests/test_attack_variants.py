from __future__ import annotations

import argparse
import json

from bit2watt_impl import attack_variants


def _args(tmp_path, variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        gpu_id=0,
        duration=10.0,
        period=None,
        duty_cycle=None,
        host="llm",
        progress_log=str(tmp_path / "progress.jsonl"),
        seed=7,
        dry_plan=False,
    )


def test_swma_argv_snapshots_progress_log_policy(tmp_path):
    args = _args(tmp_path, "basic")
    standalone = attack_variants._swma_argv(
        args,
        gpu_id=0,
        progress_log=args.progress_log,
        duration=args.duration,
        params=attack_variants.SWMA_VARIANTS["basic"],
    )
    hosted = attack_variants._swma_argv(
        args,
        gpu_id=0,
        progress_log=None,
        duration=args.duration,
        params=attack_variants.SWMA_VARIANTS["basic"],
    )
    assert standalone[:3] == [
        attack_variants.sys.executable,
        "-m",
        "bit2watt_impl.swma_workload",
    ]
    assert "--progress-log" in standalone
    assert "--progress-log" not in hosted
    assert "--matrix-size" in standalone


def test_hosted_starts_after_30_steps_and_excludes_prefix(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path, "piggyback")
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        "".join(
            json.dumps(
                {"event": "step_end", "gpu_id": 0, "t_epoch": float(index)}
            )
            + "\n"
            for index in range(30)
        ),
        encoding="utf-8",
    )

    class FakeProcess:
        instances = []

        def __init__(self, argv, **_kwargs):
            self.argv = argv
            self.returncode = None
            self.index = len(self.instances)
            self.instances.append(self)

        def poll(self):
            return self.returncode

        def communicate(self):
            if self.index == 1:
                self.returncode = 0
                return (
                    json.dumps(
                        {"gt_attack_intervals_epoch": {"0": [[10.0, 40.0]]}}
                    ),
                    None,
                )
            self.returncode = 0
            return ("", None)

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(attack_variants.subprocess, "Popen", FakeProcess)
    attack_variants.run(args)
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["ok"] is True
    assert result["plan"]["passive_host_steps"] == 30
    assert result["gt_attack_intervals_epoch"]["0"][0][0] == 29.0
    assert "--progress-log" in FakeProcess.instances[0].argv
    assert "--progress-log" not in FakeProcess.instances[1].argv

import subprocess
import sys

from bit2watt_impl.attack_variants import (
    SWMA_VARIANTS,
    build_variant_plan,
    host_warmup_wait_s,
    hosted_period,
)
from bit2watt_impl.ltma_inject import mean_preserving_repeats
from bit2watt_impl.swma_workload import allocate_independent_streams


def test_variant_plan_hosted_vs_standalone():
    class A:
        variant = "piggyback"
        period = None
        duty_cycle = None

    plan = build_variant_plan(A())
    assert plan["hosted"] is True
    assert plan["attack_progress_log"] is False
    assert plan["passive_host_steps"] == 30

    class B:
        variant = "basic"
        period = None
        duty_cycle = None

    plan2 = build_variant_plan(B())
    assert plan2["hosted"] is False
    assert plan2["attack_progress_log"] is True


def test_dry_plan_argv_snapshots():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "piggyback",
            "--dry-plan",
            "--progress-log",
            "/tmp/host.jsonl",
        ],
        cwd="/home/agew1597/KDN",
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(proc.stdout)
    assert "--progress-log" in plan["argv_host"]
    assert "--progress-log" not in plan["argv_attack"]
    assert "--preset" in plan["argv_host"]
    assert "small" in plan["argv_host"]
    assert "tiny" not in plan["argv_host"]
    host = plan["argv_host"]
    assert host[host.index("--batch-size") + 1] == "64"
    assert host[host.index("--seq-len") + 1] == "512"

    proc2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "bit2watt_impl.attack_variants",
            "--variant",
            "basic",
            "--dry-plan",
            "--progress-log",
            "/tmp/swma.jsonl",
        ],
        cwd="/home/agew1597/KDN",
        capture_output=True,
        text=True,
        check=True,
    )
    plan2 = json.loads(proc2.stdout)
    assert "--progress-log" in plan2["argv_attack"]


def test_merge_records_requested_and_actual_period():
    from dataset.run_capture import merge_workload_summary

    params = {"period_requested_s": 1.0, "period_actual_s": None}
    merge_workload_summary(
        params,
        {"period_requested_s": 1.0, "period_actual_s": 0.42, "ok": True},
    )
    assert params["period_requested_s"] == 1.0
    assert params["period_actual_s"] == 0.42


def test_swma_variant_params():
    assert SWMA_VARIANTS["basic"]["matrix_size"] == 4096
    assert SWMA_VARIANTS["basic"]["active_streams"] == 2
    assert SWMA_VARIANTS["shallow"]["matrix_size"] < SWMA_VARIANTS["basic"]["matrix_size"]
    assert SWMA_VARIANTS["jitter"]["jitter_frac"] > 0


def test_hosted_period_floor_and_wait():
    floored = hosted_period(0.01, 0.3)
    assert floored["period_requested_s"] == 0.003
    assert floored["period_actual_s"] == 0.05
    assert floored["period_floored"] is True
    natural = hosted_period(0.4, 0.3)
    assert natural["period_floored"] is False
    assert natural["period_actual_s"] == 0.12
    assert host_warmup_wait_s(120) == 180.0
    assert host_warmup_wait_s(200) == 260.0
    assert host_warmup_wait_s(400) == 300.0


def test_swma_independent_streams_and_ltma_controller():
    import torch

    streams = allocate_independent_streams(torch, "cpu", 8, 2, torch.float32)
    assert len(streams) == 2
    left0, right0, out0 = streams[0]
    left1, _right1, _out1 = streams[1]
    torch.mm(left0, right0, out=out0)
    assert not torch.equal(left0, left1)
    assert mean_preserving_repeats(
        baseline_w=200.0, recent_w=220.0, phase_high=True, max_aux_repeats=4
    ) == 0
    assert mean_preserving_repeats(
        baseline_w=200.0, recent_w=180.0, phase_high=False, max_aux_repeats=4
    ) == 4
    assert mean_preserving_repeats(
        baseline_w=200.0, recent_w=201.0, phase_high=True, max_aux_repeats=4
    ) == 4
    assert mean_preserving_repeats(
        baseline_w=None, recent_w=None, phase_high=False, max_aux_repeats=4
    ) == 0
