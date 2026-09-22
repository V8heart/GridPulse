"""GPU-free tests for the observed-only NVML collector."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from bit2watt_impl import collect_telemetry as ct
from dataset.identifiers import new_session_id


class FakeBackend:
    def __init__(self, *, instant_ok: bool = True, temps: list[float] | None = None):
        self.instant_ok = instant_ok
        self.temps = list(temps or [40.0, 41.0, 42.0, 43.0])
        self._tick = 0
        self.proc_calls = 0
        self.inited = False
        self.shut = False

    def init(self) -> None:
        self.inited = True

    def shutdown(self) -> None:
        self.shut = True

    def device_count(self) -> int:
        return 2

    def resolve_gpu_ids(self, spec: str) -> list[int]:
        if spec.strip().lower() == "all":
            return [0, 1]
        return [int(x) for x in spec.split(",")]

    def probe_power_instant(self, gpu_id: int, errors: dict[str, int]) -> bool:
        return self.instant_ok

    def read_gpu(
        self,
        gpu_id: int,
        *,
        errors: dict[str, int],
        power_instant_supported: bool,
        include_processes: bool,
        cached_proc: dict | None,
    ) -> dict:
        temp = self.temps[min(self._tick, len(self.temps) - 1)]
        if include_processes:
            self.proc_calls += 1
            proc = {"observed_pid": 100 + gpu_id, "observed_process_name": "python", "observed_n_procs": 1}
        else:
            proc = {
                "observed_pid": None if cached_proc is None else cached_proc.get("observed_pid"),
                "observed_process_name": None if cached_proc is None else cached_proc.get("observed_process_name"),
                "observed_n_procs": None if cached_proc is None else cached_proc.get("observed_n_procs"),
            }
        row = {
            "gpu_id": gpu_id,
            "gpu_uuid": f"GPU-{gpu_id}",
            "gpu_model": "Fake GPU",
            "power_w": 100.0 + gpu_id,
            "power_instant_w": (101.0 + gpu_id) if power_instant_supported else None,
            "util_gpu_pct": 50.0,
            "mem_copy_util_pct": 20.0,
            "sm_clock_mhz": 1800.0,
            "mem_clock_mhz": 900.0,
            "temp_c": temp,
            "fb_used_mb": 1024.0,
            "fan_speed_pct": 30.0,
            "pstate": 2,
            "power_limit_w": 450.0,
            "throttle_reasons": 0,
            **proc,
        }
        if gpu_id == 0:
            self._tick += 1
        return row


def test_collector_cli_has_no_label_args():
    import bit2watt_impl.collect_telemetry as mod

    assert "label" not in ct.collect.__code__.co_varnames
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert 'add_argument("--label"' not in src
    assert 'add_argument("--job-type"' not in src
    assert 'add_argument("--id-user"' not in src
    assert 'add_argument("--gres-req"' not in src


def test_collector_stop_file_and_sample_hz(tmp_path: Path):
    session_id = new_session_id()
    output = tmp_path / "telemetry.csv"
    stop = tmp_path / "stop"
    result_json = tmp_path / "collector_result.json"
    mono = {"t": 0.0}

    def mono_fn() -> float:
        return mono["t"]

    def sleep_fn(dt: float) -> None:
        mono["t"] += max(dt, 0.05)
        if mono["t"] >= 0.25:
            stop.write_text("stop\n", encoding="utf-8")

    backend = FakeBackend(instant_ok=True)
    result = ct.collect(
        output,
        session_id=session_id,
        interval_ms=50.0,
        gpu_ids="all",
        max_duration_s=5.0,
        stop_file=stop,
        flush_every_s=0.05,
        proc_poll_s=0.2,
        abort_temp_c=88.0,
        result_json=result_json,
        backend=backend,
        sleep_fn=sleep_fn,
        mono_fn=mono_fn,
        wall_fn=lambda: 1_700_000_000.0,
    )
    assert output.exists()
    assert result["power_instant_supported"] is True
    assert result["aborted_reason"] == "stop_file"
    assert result["t0_epoch"] == 1_700_000_000.0
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) >= 2
    assert {r["gpu_id"] for r in rows} == {"0", "1"}
    assert all(r["session_id"] == session_id for r in rows)
    assert all("label" not in r or r.get("label", "") == "" for r in rows)
    assert all(r["power_instant_w"] != "" for r in rows)
    # sample_hz filled from actual intervals
    assert float(rows[1]["sample_hz"]) > 0
    meta = json.loads(result_json.read_text(encoding="utf-8"))
    assert meta["session_id"] == session_id
    assert backend.proc_calls >= 1
    # process polling should be less frequent than every tick
    ticks = len(rows) // 2
    assert backend.proc_calls < ticks


def test_collector_instant_unsupported_and_temp_abort(tmp_path: Path):
    session_id = new_session_id()
    output = tmp_path / "telemetry.csv"
    stop = tmp_path / "stop"
    mono = {"t": 0.0}

    def mono_fn() -> float:
        return mono["t"]

    def sleep_fn(dt: float) -> None:
        mono["t"] += 0.05

    backend = FakeBackend(instant_ok=False, temps=[40.0, 50.0, 95.0])
    result = ct.collect(
        output,
        session_id=session_id,
        interval_ms=50.0,
        gpu_ids="0",
        max_duration_s=5.0,
        stop_file=stop,
        flush_every_s=0.05,
        proc_poll_s=10.0,
        abort_temp_c=88.0,
        result_json=tmp_path / "out.json",
        backend=backend,
        sleep_fn=sleep_fn,
        mono_fn=mono_fn,
        wall_fn=lambda: 10.0,
    )
    assert result["power_instant_supported"] is False
    assert result["aborted_reason"].startswith("abort_temp_c")
    assert stop.exists()
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert all(r["power_instant_w"] == "" for r in rows)
