"""Telemetry refresh, aliasing, and averaging-window observability diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = [
    "power_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "temp_c",
    "fb_used_mb",
]


def fold_frequency(frequency_hz: float, sample_hz: float) -> float:
    """Fold a frequency into the sampled Nyquist interval."""
    if sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    return abs(((float(frequency_hz) + sample_hz / 2.0) % sample_hz) - sample_hz / 2.0)


def boxcar_attenuation(frequency_hz: float, averaging_window_s: float) -> float:
    """Magnitude response of a boxcar averager: abs(sinc(f*T_avg))."""
    return float(abs(np.sinc(float(frequency_hz) * float(averaging_window_s))))


def dominant_frequency(values: np.ndarray, sample_hz: float) -> tuple[float | None, float | None]:
    data = np.asarray(values, dtype=float)
    finite = np.isfinite(data)
    if finite.sum() < 8:
        return None, None
    if not finite.all():
        data = np.interp(np.arange(len(data)), np.flatnonzero(finite), data[finite])
    centered = data - np.mean(data)
    spectrum = np.abs(np.fft.rfft(centered)) * 2.0 / len(centered)
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_hz)
    if len(spectrum) <= 1:
        return None, None
    index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[index]), float(spectrum[index])


def _true_frequency(frame: pd.DataFrame) -> float | None:
    for column in ("true_frequency_hz", "waveform_frequency_hz"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                return float(values.iloc[0])
    if "gt_params_json" in frame:
        for raw in frame["gt_params_json"].dropna().astype(str):
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if params.get("frequency_hz") is not None:
                return float(params["frequency_hz"])
            if params.get("period_s"):
                return 1.0 / float(params["period_s"])
    return None


def _sample_hz(frame: pd.DataFrame) -> float:
    if "sample_hz" in frame:
        values = pd.to_numeric(frame["sample_hz"], errors="coerce").dropna()
        if len(values):
            return float(values.median())
    times = pd.to_numeric(frame["timestamp"], errors="coerce").dropna().to_numpy()
    delta = np.diff(times)
    delta = delta[delta > 0]
    if not len(delta):
        raise ValueError("cannot infer sample rate")
    return float(1.0 / np.median(delta))


def estimate_averaging_window(
    frequency_hz: float, observed_attenuation: float, *, max_window_s: float = 5.0
) -> float | None:
    if not np.isfinite(observed_attenuation) or not 0 <= observed_attenuation <= 1.1:
        return None
    grid = np.linspace(0.0, max_window_s, 20001)
    response = np.abs(np.sinc(float(frequency_hz) * grid))
    return float(grid[int(np.argmin(np.abs(response - min(1.0, observed_attenuation))))])


def period_retention_rows(path: Path, frame: pd.DataFrame) -> list[dict]:
    rows = []
    group_columns = ["gpu_id"] if "gpu_id" in frame else []
    groups = frame.groupby(group_columns, dropna=False) if group_columns else [(0, frame)]
    for gpu_key, gpu_frame in groups:
        gpu_id = int(gpu_key[0] if isinstance(gpu_key, tuple) else gpu_key)
        sample_hz = _sample_hz(gpu_frame)
        true_hz = _true_frequency(gpu_frame)
        averaged_hz, averaged_amplitude = dominant_frequency(
            pd.to_numeric(gpu_frame["power_w"], errors="coerce").to_numpy(), sample_hz
        )
        instant_hz = instant_amplitude = None
        if "power_instant_w" in gpu_frame:
            instant_hz, instant_amplitude = dominant_frequency(
                pd.to_numeric(gpu_frame["power_instant_w"], errors="coerce").to_numpy(),
                sample_hz,
            )
        reference = instant_amplitude
        if reference is None and "power_phys_w" in gpu_frame:
            _, reference = dominant_frequency(
                pd.to_numeric(gpu_frame["power_phys_w"], errors="coerce").to_numpy(),
                sample_hz,
            )
        retention = (
            None
            if reference is None or averaged_amplitude is None or reference <= 0
            else float(averaged_amplitude / reference)
        )
        folded = None if true_hz is None else fold_frequency(true_hz, sample_hz)
        effective_window = (
            None
            if true_hz is None or retention is None
            else estimate_averaging_window(true_hz, retention)
        )
        rows.append(
            {
                "file": str(path),
                "session_id": (
                    str(gpu_frame["session_id"].iloc[0]) if "session_id" in gpu_frame else path.parent.name
                ),
                "gpu_id": gpu_id,
                "sample_hz": sample_hz,
                "nyquist_hz": sample_hz / 2.0,
                "true_frequency_hz": true_hz,
                "folded_frequency_hz": folded,
                "aliased": bool(true_hz is not None and true_hz > sample_hz / 2.0),
                "averaged_peak_hz": averaged_hz,
                "instant_peak_hz": instant_hz,
                "averaged_amplitude": averaged_amplitude,
                "instant_amplitude": instant_amplitude,
                "period_retention": retention,
                "effective_averaging_window_s": effective_window,
                "theoretical_boxcar_attenuation": (
                    None
                    if true_hz is None or effective_window is None
                    else boxcar_attenuation(true_hz, effective_window)
                ),
            }
        )
    return rows


def discover_inputs(inputs: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            direct = item / "telemetry.csv"
            if direct.exists():
                discovered.add(direct)
            discovered.update(item.glob("sessions/*/telemetry.csv"))
            discovered.update(item.glob("*/telemetry.csv"))
        elif item.name == "telemetry.csv" or item.suffix.lower() == ".csv":
            discovered.add(item)
    return sorted(discovered)


def summarize_period_retention(inputs: list[Path], output: Path) -> list[dict]:
    files = discover_inputs(inputs)
    rows = [row for path in files for row in period_retention_rows(path, pd.read_csv(path))]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path)
    actual = pd.to_numeric(df.get("actual_interval_ms"), errors="coerce").dropna()
    result = {
        "file": str(path),
        "rows": len(df),
        "sessions": int(df["session_id"].nunique()) if "session_id" in df else 0,
        "requested_interval_ms": (
            float(pd.to_numeric(df["requested_interval_ms"], errors="coerce").dropna().iloc[0])
            if "requested_interval_ms" in df
            and len(pd.to_numeric(df["requested_interval_ms"], errors="coerce").dropna())
            else None
        ),
        "actual_interval_ms": {
            "mean": float(actual.mean()) if len(actual) else None,
            "p50": float(actual.quantile(0.5)) if len(actual) else None,
            "p95": float(actual.quantile(0.95)) if len(actual) else None,
            "max": float(actual.max()) if len(actual) else None,
        },
        "row_refresh_ratio": (
            float(df["value_changed"].astype(str).str.lower().eq("true").mean())
            if "value_changed" in df else None
        ),
        "power_modes": {
            "averaged_present_ratio": float(pd.to_numeric(df.get("power_w"), errors="coerce").notna().mean()),
            "instant_present_ratio": (
                float(pd.to_numeric(df["power_instant_w"], errors="coerce").notna().mean())
                if "power_instant_w" in df else 0.0
            ),
        },
        "fields": {},
    }
    for field in FIELDS:
        if field not in df:
            continue
        values = pd.to_numeric(df[field], errors="coerce")
        valid = values.dropna()
        changes = values.ne(values.shift()) & values.notna() & values.shift().notna()
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
        change_times = timestamps[changes].to_numpy()
        intervals = np.diff(change_times) if len(change_times) > 1 else np.array([])
        result["fields"][field] = {
            "missing_ratio": float(values.isna().mean()),
            "unique_ratio": float(valid.nunique() / len(valid)) if len(valid) else 0.0,
            "refresh_ratio": float(changes.mean()),
            "change_interval_s_p50": float(np.median(intervals)) if len(intervals) else None,
        }
    return result


def write_summary(inputs: list[Path], output: Path) -> list[dict]:
    files = discover_inputs(inputs)
    summaries = [summarize_file(path) for path in files]
    summarize_period_retention(files, output.parent / "period_retention.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Telemetry observability summary", ""]
    for item in summaries:
        cadence = item["actual_interval_ms"]
        lines.extend(
            [
                f"## {item['file']}",
                f"- rows: {item['rows']}",
                f"- requested interval: {item['requested_interval_ms']} ms",
                f"- actual interval mean/p95: {cadence['mean']} / {cadence['p95']} ms",
                f"- row refresh ratio: {item['row_refresh_ratio']}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV files or sessions-tree roots")
    parser.add_argument("--output", type=Path, default=Path("dataset/real/observability/summary.md"))
    args = parser.parse_args()
    write_summary(args.inputs, args.output)
    print(f"요약 저장: {args.output}")


if __name__ == "__main__":
    main()

