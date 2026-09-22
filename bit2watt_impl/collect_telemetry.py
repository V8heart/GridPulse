"""Observed-only NVML telemetry collector (gp-telemetry/1.2)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.identifiers import new_session_id, require_session_id

# Spool / final CSV columns written by the collector (no declared/gt).
COLLECTOR_COLUMNS = [
    "timestamp",
    "t_epoch",
    "session_id",
    "gpu_id",
    "gpu_uuid",
    "gpu_model",
    "sample_hz",
    "requested_interval_ms",
    "actual_interval_ms",
    "value_changed",
    "power_w",
    "power_instant_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "fan_speed_pct",
    "pstate",
    "power_limit_w",
    "throttle_reasons",
    "observed_pid",
    "observed_process_name",
    "observed_n_procs",
]

FINGERPRINT_KEYS = (
    "power_w",
    "power_instant_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temp_c",
    "fb_used_mb",
    "fan_speed_pct",
    "pstate",
    "power_limit_w",
    "throttle_reasons",
    "observed_pid",
    "observed_n_procs",
)


@dataclass
class CollectorState:
    stop_requested: bool = False
    aborted_reason: str | None = None
    field_errors: dict[str, int] = field(default_factory=dict)
    power_instant_supported: bool | None = None
    interval_warnings: list[str] = field(default_factory=list)
    t0_epoch: float | None = None
    t0_mono: float | None = None
    rows_written: int = 0


def _process_name(pid: int) -> str | None:
    for suffix in ("comm", "cmdline"):
        try:
            text = Path(f"/proc/{pid}/{suffix}").read_text(errors="replace").replace("\0", " ").strip()
            if text:
                return text.split()[0] if suffix == "cmdline" else text
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass
    return None


class NvmlBackend:
    """Thin pynvml wrapper so unit tests can inject a fake backend."""

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        self._handles: dict[int, Any] = {}

    def init(self) -> None:
        self._nvml.nvmlInit()

    def shutdown(self) -> None:
        self._nvml.nvmlShutdown()

    def device_count(self) -> int:
        return int(self._nvml.nvmlDeviceGetCount())

    def resolve_gpu_ids(self, spec: str) -> list[int]:
        count = self.device_count()
        if spec.strip().lower() == "all":
            return list(range(count))
        ids = [int(item) for item in spec.split(",") if item.strip() != ""]
        if any(gpu < 0 or gpu >= count for gpu in ids):
            raise ValueError(f"GPU ID 범위 오류: 장치 수={count}, 요청={ids}")
        return ids

    def handle(self, gpu_id: int):
        if gpu_id not in self._handles:
            self._handles[gpu_id] = self._nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        return self._handles[gpu_id]

    def _safe(self, call: Callable[[], Any], errors: dict[str, int], field: str) -> Any:
        try:
            return call()
        except self._nvml.NVMLError:
            errors[field] = errors.get(field, 0) + 1
            return None

    def probe_power_instant(self, gpu_id: int, errors: dict[str, int]) -> bool:
        handle = self.handle(gpu_id)
        field_id = getattr(self._nvml, "NVML_FI_DEV_POWER_INSTANT", None)
        if field_id is None or not hasattr(self._nvml, "nvmlDeviceGetFieldValues"):
            return False
        try:
            values = self._nvml.nvmlDeviceGetFieldValues(handle, [field_id])
            if not values:
                return False
            status = getattr(values[0], "nvmlReturn", 0)
            return int(status) == 0
        except Exception:
            errors["power_instant_probe"] = errors.get("power_instant_probe", 0) + 1
            return False

    def read_processes(self, gpu_id: int, errors: dict[str, int]) -> list[dict[str, Any]]:
        handle = self.handle(gpu_id)
        procs = self._safe(
            lambda: self._nvml.nvmlDeviceGetComputeRunningProcesses(handle),
            errors,
            "processes",
        ) or []
        result = []
        for proc in procs:
            used = getattr(proc, "usedGpuMemory", None)
            result.append(
                {
                    "pid": int(proc.pid),
                    "process_name": _process_name(int(proc.pid)),
                    "used_gpu_memory_mb": None if used is None else float(used / 1024**2),
                }
            )
        return sorted(result, key=lambda item: item["used_gpu_memory_mb"] or -1, reverse=True)

    def read_gpu(
        self,
        gpu_id: int,
        *,
        errors: dict[str, int],
        power_instant_supported: bool,
        include_processes: bool,
        cached_proc: dict[str, Any] | None,
    ) -> dict[str, Any]:
        nvml = self._nvml
        handle = self.handle(gpu_id)
        util = self._safe(lambda: nvml.nvmlDeviceGetUtilizationRates(handle), errors, "utilization")
        memory = self._safe(lambda: nvml.nvmlDeviceGetMemoryInfo(handle), errors, "memory")
        uuid = self._safe(lambda: nvml.nvmlDeviceGetUUID(handle), errors, "gpu_uuid")
        name = self._safe(lambda: nvml.nvmlDeviceGetName(handle), errors, "gpu_model")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8", errors="replace")

        power_instant = None
        if power_instant_supported:
            field_id = getattr(nvml, "NVML_FI_DEV_POWER_INSTANT", None)

            def _instant():
                values = nvml.nvmlDeviceGetFieldValues(handle, [field_id])
                raw = getattr(values[0], "value", None)
                # Field value is typically milliwatts in value.ullVal / uiVal.
                if raw is None:
                    return None
                for attr in ("dVal", "ullVal", "uiVal", "ull_val"):
                    if hasattr(raw, attr):
                        val = getattr(raw, attr)
                        return float(val) / 1000.0 if attr != "dVal" else float(val)
                if isinstance(raw, (int, float)):
                    return float(raw) / 1000.0
                return None

            power_instant = self._safe(_instant, errors, "power_instant_w")

        throttle = None
        if hasattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons"):
            throttle = self._safe(
                lambda: int(nvml.nvmlDeviceGetCurrentClocksEventReasons(handle)),
                errors,
                "throttle_reasons",
            )
        if throttle is None and hasattr(nvml, "nvmlDeviceGetCurrentClocksThrottleReasons"):
            throttle = self._safe(
                lambda: int(nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)),
                errors,
                "throttle_reasons",
            )

        if include_processes:
            processes = self.read_processes(gpu_id, errors)
            primary = processes[0] if processes else {}
            proc_fields = {
                "observed_pid": primary.get("pid"),
                "observed_process_name": primary.get("process_name"),
                "observed_n_procs": len(processes),
            }
        else:
            proc_fields = {
                "observed_pid": None if cached_proc is None else cached_proc.get("observed_pid"),
                "observed_process_name": None if cached_proc is None else cached_proc.get("observed_process_name"),
                "observed_n_procs": None if cached_proc is None else cached_proc.get("observed_n_procs"),
            }

        return {
            "gpu_id": gpu_id,
            "gpu_uuid": uuid,
            "gpu_model": name,
            "power_w": self._safe(lambda: nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, errors, "power_w"),
            "power_instant_w": power_instant,
            "util_gpu_pct": None if util is None else float(util.gpu),
            "mem_copy_util_pct": None if util is None else float(util.memory),
            "sm_clock_mhz": self._safe(
                lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)),
                errors,
                "sm_clock_mhz",
            ),
            "mem_clock_mhz": self._safe(
                lambda: float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)),
                errors,
                "mem_clock_mhz",
            ),
            "temp_c": self._safe(
                lambda: float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)),
                errors,
                "temp_c",
            ),
            "fb_used_mb": None if memory is None else float(memory.used / 1024**2),
            "fan_speed_pct": self._safe(lambda: float(nvml.nvmlDeviceGetFanSpeed(handle)), errors, "fan_speed_pct"),
            "pstate": self._safe(lambda: int(nvml.nvmlDeviceGetPerformanceState(handle)), errors, "pstate"),
            "power_limit_w": self._safe(
                lambda: float(nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0),
                errors,
                "power_limit_w",
            ),
            "throttle_reasons": throttle,
            **proc_fields,
        }


def _empty_row(session_id: str, interval_ms: float) -> dict[str, Any]:
    return {col: "" for col in COLLECTOR_COLUMNS} | {
        "session_id": session_id,
        "requested_interval_ms": interval_ms,
        "sample_hz": "",
    }


def _write_result_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _finalize_spool_to_csv(
    spool: Path,
    output: Path,
    *,
    requested_interval_ms: float,
) -> dict[str, Any]:
    """Rewrite spool with GPU-wise median sample_hz; atomic rename to output."""
    if not spool.exists():
        raise FileNotFoundError(f"spool missing: {spool}")
    by_gpu: dict[int, list[float]] = {}
    with spool.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    for row in rows:
        gpu_id = int(float(row["gpu_id"]))
        actual = row.get("actual_interval_ms") or ""
        if actual.strip() == "":
            continue
        by_gpu.setdefault(gpu_id, []).append(float(actual))

    sample_hz_by_gpu: dict[int, float] = {}
    warnings: list[str] = []
    for gpu_id, intervals in by_gpu.items():
        median = statistics.median(intervals)
        if median <= 0:
            hz = 1000.0 / requested_interval_ms
        else:
            hz = 1000.0 / median
        sample_hz_by_gpu[gpu_id] = float(hz)
        if intervals:
            p95 = sorted(intervals)[max(0, math.ceil(0.95 * len(intervals)) - 1)]
            if p95 > 2.0 * requested_interval_ms:
                warnings.append(
                    f"gpu{gpu_id}: actual_interval_ms p95={p95:.3f} > 2x requested={requested_interval_ms}"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLLECTOR_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            gpu_id = int(float(row["gpu_id"]))
            row = dict(row)
            row["sample_hz"] = sample_hz_by_gpu.get(gpu_id, 1000.0 / requested_interval_ms)
            writer.writerow({k: row.get(k, "") for k in COLLECTOR_COLUMNS})
    os.replace(temporary, output)
    try:
        spool.unlink()
    except FileNotFoundError:
        pass
    return {"sample_hz_by_gpu": sample_hz_by_gpu, "interval_warnings": warnings, "rows": len(rows)}


def collect(
    output: Path,
    *,
    session_id: str,
    interval_ms: float,
    gpu_ids: list[int] | str = "all",
    max_duration_s: float = 4000.0,
    stop_file: Path | None = None,
    flush_every_s: float = 10.0,
    proc_poll_s: float = 1.0,
    abort_temp_c: float = 88.0,
    result_json: Path | None = None,
    backend: NvmlBackend | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    mono_fn: Callable[[], float] = time.monotonic,
    wall_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Collect observed telemetry until stop-file / max-duration / SIGTERM / abort."""
    session_id = require_session_id(session_id)
    if interval_ms <= 0 or max_duration_s <= 0:
        raise ValueError("interval_ms and max_duration_s must be positive")

    state = CollectorState()
    owns_backend = backend is None
    backend = backend or NvmlBackend()

    def _request_stop(signum=None, frame=None) -> None:  # noqa: ARG001
        state.stop_requested = True
        if signum is not None and state.aborted_reason is None:
            state.aborted_reason = f"signal_{signum}"

    previous = signal.signal(signal.SIGTERM, _request_stop)
    previous_int = signal.signal(signal.SIGINT, _request_stop)

    spool = output.with_suffix(output.suffix + ".spool.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if spool.exists():
        spool.unlink()

    previous_values: dict[int, tuple[Any, ...]] = {}
    previous_times: dict[int, float] = {}
    cached_proc: dict[int, dict[str, Any]] = {}
    last_proc_poll = -1e18
    last_flush = 0.0
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal buffer, last_flush
        if not buffer:
            return
        write_header = not spool.exists()
        with spool.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLLECTOR_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in buffer:
                writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in COLLECTOR_COLUMNS})
                state.rows_written += 1
        buffer = []
        last_flush = mono_fn()

    try:
        backend.init()
        ids = backend.resolve_gpu_ids(gpu_ids if isinstance(gpu_ids, str) else ",".join(map(str, gpu_ids)))
        if not ids:
            raise ValueError("no GPU ids to collect")

        state.t0_epoch = float(wall_fn())
        state.t0_mono = float(mono_fn())
        state.power_instant_supported = backend.probe_power_instant(ids[0], state.field_errors)

        deadline = state.t0_mono + float(max_duration_s)
        next_tick = state.t0_mono
        last_flush = state.t0_mono

        while True:
            now_mono = mono_fn()
            if state.stop_requested or now_mono >= deadline:
                break
            if stop_file is not None and stop_file.exists():
                state.stop_requested = True
                if state.aborted_reason is None:
                    state.aborted_reason = "stop_file"
                break

            poll_procs = (now_mono - last_proc_poll) >= proc_poll_s
            if poll_procs:
                last_proc_poll = now_mono

            for gpu_id in ids:
                sampled_mono = mono_fn()
                values = backend.read_gpu(
                    gpu_id,
                    errors=state.field_errors,
                    power_instant_supported=bool(state.power_instant_supported),
                    include_processes=poll_procs,
                    cached_proc=cached_proc.get(gpu_id),
                )
                if poll_procs:
                    cached_proc[gpu_id] = {
                        "observed_pid": values.get("observed_pid"),
                        "observed_process_name": values.get("observed_process_name"),
                        "observed_n_procs": values.get("observed_n_procs"),
                    }
                temp = values.get("temp_c")
                if temp is not None and float(temp) > abort_temp_c:
                    state.aborted_reason = f"abort_temp_c:{temp}"
                    state.stop_requested = True
                    if stop_file is not None:
                        stop_file.parent.mkdir(parents=True, exist_ok=True)
                        stop_file.write_text("abort_temp\n", encoding="utf-8")

                fingerprint = tuple(values.get(k) for k in FINGERPRINT_KEYS)
                timestamp = sampled_mono - float(state.t0_mono)
                row = {
                    "timestamp": timestamp,
                    "t_epoch": float(state.t0_epoch) + timestamp,
                    "session_id": session_id,
                    "sample_hz": "",  # filled on finalize rewrite
                    "requested_interval_ms": interval_ms,
                    "actual_interval_ms": (
                        None
                        if gpu_id not in previous_times
                        else (sampled_mono - previous_times[gpu_id]) * 1000.0
                    ),
                    "value_changed": (
                        True if gpu_id not in previous_values else fingerprint != previous_values[gpu_id]
                    ),
                    **values,
                }
                buffer.append(row)
                previous_values[gpu_id] = fingerprint
                previous_times[gpu_id] = sampled_mono

            if mono_fn() - last_flush >= flush_every_s:
                flush_buffer()
            if state.stop_requested:
                break

            next_tick += interval_ms / 1000.0
            sleep_fn(max(0.0, next_tick - mono_fn()))

        flush_buffer()
        summary = _finalize_spool_to_csv(spool, output, requested_interval_ms=interval_ms)
        state.interval_warnings.extend(summary["interval_warnings"])
        result = {
            "schema_version": "gp-telemetry/1.2",
            "session_id": session_id,
            "t0_epoch": state.t0_epoch,
            "t0_mono": state.t0_mono,
            "requested_interval_ms": interval_ms,
            "proc_poll_s": proc_poll_s,
            "flush_every_s": flush_every_s,
            "max_duration_s": max_duration_s,
            "abort_temp_c": abort_temp_c,
            "gpu_ids": ids,
            "rows": summary["rows"],
            "sample_hz_by_gpu": summary["sample_hz_by_gpu"],
            "power_instant_supported": bool(state.power_instant_supported),
            "field_errors": state.field_errors,
            "interval_warnings": state.interval_warnings,
            "aborted_reason": state.aborted_reason,
            "output": str(output),
        }
        _write_result_json(result_json, result)
        return result
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_int)
        if owns_backend:
            try:
                backend.shutdown()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed-only NVML collector")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True, help="opaque s-<16 hex>")
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--gpu-ids", default="all", help="'all' or comma-separated ids")
    parser.add_argument("--max-duration", type=float, default=4000.0)
    parser.add_argument("--stop-file", type=Path, default=None)
    parser.add_argument("--flush-every-s", type=float, default=10.0)
    parser.add_argument("--proc-poll-s", type=float, default=1.0)
    parser.add_argument("--abort-temp-c", type=float, default=88.0)
    parser.add_argument("--result-json", type=Path, default=None)
    args = parser.parse_args()
    result = collect(
        args.output,
        session_id=args.session_id,
        interval_ms=args.interval_ms,
        gpu_ids=args.gpu_ids,
        max_duration_s=args.max_duration,
        stop_file=args.stop_file,
        flush_every_s=args.flush_every_s,
        proc_poll_s=args.proc_poll_s,
        abort_temp_c=args.abort_temp_c,
        result_json=args.result_json,
    )
    print(json.dumps({"ok": True, "rows": result["rows"], "session_id": result["session_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
