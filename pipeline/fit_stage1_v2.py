"""Fit Stage 1 v2 cohort baseline and calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.baseline import CohortBaseline
from pipeline.stage1_v2 import build_windows, impact_components, load_config, score_window


def fit(
    telemetry: Path,
    split_manifest: Path,
    *,
    config_path: Path,
    baseline_out: Path,
    calibration_out: Path,
    progress_log_dir: Path,
    window_s: float,
    stride_s: float,
) -> dict:
    df = pd.read_csv(telemetry)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    config = load_config(config_path)
    train = set(manifest["train"])
    cal = set(manifest["cal"])
    train_df = df[df["session_id"].astype(str).isin(train)]
    cal_df = df[df["session_id"].astype(str).isin(cal)]
    train_windows = build_windows(train_df, window_s=window_s, stride_s=stride_s, progress_log_dir=progress_log_dir, config=config)
    cal_windows = build_windows(cal_df, window_s=window_s, stride_s=stride_s, progress_log_dir=progress_log_dir, config=config)
    train_normals = train_windows[train_windows["gt_label"].astype(str).str.startswith("normal")]
    baseline = CohortBaseline().fit(train_normals, min_windows=10)
    normal_scores = []
    all_impacts = []
    for _, row in cal_windows.iterrows():
        scored = score_window(row.to_dict(), baseline, {"normal_u_scores": []}, config)
        all_impacts.append(max(impact_components(row.to_dict(), config).values(), default=0.0))
        if str(row["gt_label"]).startswith("normal"):
            normal_scores.append(scored["u_score"])
    calibration = {
        "fit_split": "cal",
        "normal_u_scores": normal_scores,
        "impact_quantiles": {
            "p50": float(np.quantile(all_impacts, 0.5)) if all_impacts else 0.0,
            "p90": float(np.quantile(all_impacts, 0.9)) if all_impacts else 0.0,
        },
        "n_cal_windows": int(len(cal_windows)),
        "n_cal_normal_windows": int(len(normal_scores)),
        "config": config,
    }
    baseline.save(baseline_out)
    calibration_out.parent.mkdir(parents=True, exist_ok=True)
    calibration_out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train_windows": len(train_windows), "cal_windows": len(cal_windows), **calibration}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--baseline-out", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration-out", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    args = parser.parse_args()
    report = fit(
        args.telemetry,
        args.split_manifest,
        config_path=args.stage1_config,
        baseline_out=args.baseline_out,
        calibration_out=args.calibration_out,
        progress_log_dir=args.progress_log_dir,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "normal_u_scores"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
