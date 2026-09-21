"""Fit evidence bool thresholds on train normals only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.baseline import CohortBaseline
from pipeline.evidence_vocab import merge_thresholds, prominence_log
from pipeline.stage1_v2 import build_windows, load_config


def _update_yaml_thresholds(config_path: Path, thresholds: dict) -> None:
    try:
        import yaml
    except ImportError:
        return
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data["evidence_thresholds"] = thresholds
    if "tau_peak" in thresholds:
        data["tau_peak"] = thresholds["tau_peak"]
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def fit_thresholds_on_normals(
    normals: pd.DataFrame,
    config: dict,
    *,
    baseline: CohortBaseline | None = None,
) -> dict:
    """Quantile thresholds from normal windows only (prefer train)."""
    if normals.empty:
        raise ValueError("no normal windows for evidence threshold fitting")

    swing = normals["swing_abs_w"].dropna() if "swing_abs_w" in normals else pd.Series(dtype=float)
    mean_w = normals["mean_w"].dropna() if "mean_w" in normals else pd.Series(dtype=float)
    high_frac = normals["high_load_fraction"].dropna() if "high_load_fraction" in normals else pd.Series(dtype=float)
    longest = normals["longest_high_seconds"].dropna() if "longest_high_seconds" in normals else pd.Series(dtype=float)
    ramp = normals["ramp_p95_w_per_s"].dropna() if "ramp_p95_w_per_s" in normals else pd.Series(dtype=float)
    util_mad = normals["util_residual_mad_w"].dropna() if "util_residual_mad_w" in normals else pd.Series(dtype=float)
    if "dominant_peak_prominence_log" in normals:
        peak_log = normals["dominant_peak_prominence_log"].dropna()
    else:
        peak_log = (
            normals["dominant_peak_prominence"].dropna().map(prominence_log)
            if "dominant_peak_prominence" in normals
            else pd.Series(dtype=float)
        )

    fitted = merge_thresholds(config)
    if len(peak_log):
        fitted["tau_peak"] = float(peak_log.quantile(0.95))
    if len(high_frac):
        fitted["sustained_high_load"]["high_load_fraction_min"] = float(high_frac.quantile(0.95))
    if len(longest):
        fitted["sustained_high_load"]["longest_high_seconds_min"] = float(longest.quantile(0.95))
    if len(swing):
        fitted["flat_power"]["swing_abs_w_max"] = float(swing.quantile(0.20))
    if len(ramp):
        fitted["flat_power"]["ramp_p95_w_per_s_max"] = float(ramp.quantile(0.20))
        fitted["high_ramp"]["ramp_p95_w_per_s_min"] = float(ramp.quantile(0.95))
    if len(mean_w):
        fitted["flat_power"]["mean_w_min"] = float(mean_w.quantile(0.80))
    if len(util_mad):
        fitted["util_power_decoupled"]["util_residual_mad_w_min"] = float(util_mad.quantile(0.95))

    if baseline is not None:
        declared_z = []
        for _, row in normals.iterrows():
            stats = baseline.declared_family_mismatch_stats(row.to_dict())
            z = stats.get("declared_family_mean_abs_z")
            if z is not None:
                declared_z.append(float(z))
        if declared_z:
            fitted.setdefault("declared_family_mismatch", {})
            fitted["declared_family_mismatch"]["declared_mean_abs_z_min"] = float(
                pd.Series(declared_z).quantile(0.95)
            )
            fitted["declared_family_mismatch"].setdefault("other_family_margin", 0.5)
    else:
        fitted.setdefault(
            "declared_family_mismatch",
            {"declared_mean_abs_z_min": 3.0, "other_family_margin": 0.5},
        )
    return fitted


def fit_thresholds(
    telemetry: Path,
    split_manifest: Path,
    *,
    config_path: Path,
    progress_log_dir: Path,
    window_s: float,
    stride_s: float,
    out: Path,
    update_yaml: bool = True,
    train_only: bool = True,
) -> dict:
    df = pd.read_csv(telemetry, low_memory=False)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    config = load_config(config_path)
    if train_only:
        fit_sessions = set(map(str, manifest["train"]))
        fit_split = "train"
    else:
        fit_sessions = set(map(str, manifest["train"])) | set(map(str, manifest["cal"]))
        fit_split = "train+cal"
    subset = df[df["session_id"].astype(str).isin(fit_sessions)]
    windows = build_windows(
        subset,
        window_s=window_s,
        stride_s=stride_s,
        progress_log_dir=progress_log_dir,
        config=config,
    )
    normals = windows[windows["gt_label"].astype(str).str.startswith("normal")]
    baseline = None
    baseline_path = ROOT / "dataset/eval/cohort_baseline_v2.json"
    if baseline_path.exists():
        baseline = CohortBaseline.load(baseline_path)
    fitted = fit_thresholds_on_normals(normals, config, baseline=baseline)
    report = {
        "fit_split": fit_split,
        "n_normal_windows": int(len(normals)),
        "n_sessions": len(fit_sessions),
        "evidence_thresholds": fitted,
        "note": (
            "Quantile-only fit on train normals by default; no attack labels. "
            "tau_peak fitted on log10(1+prominence)."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if update_yaml:
        _update_yaml_thresholds(config_path, fitted)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/evidence_thresholds.json")
    parser.add_argument("--no-update-yaml", action="store_true")
    parser.add_argument("--include-cal", action="store_true", help="Legacy: fit on train+cal normals")
    args = parser.parse_args()
    report = fit_thresholds(
        args.telemetry,
        args.split_manifest,
        config_path=args.stage1_config,
        progress_log_dir=args.progress_log_dir,
        window_s=args.window_s,
        stride_s=args.stride_s,
        out=args.out,
        update_yaml=not args.no_update_yaml,
        train_only=not args.include_cal,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
