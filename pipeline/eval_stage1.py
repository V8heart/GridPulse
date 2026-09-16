"""Evaluate legacy Stage 1 versus Stage 1 v2 on the test split."""
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
from pipeline.run_pipeline import infer_sample_hz, stage1_screen_legacy
from pipeline.stage1_v2 import build_windows, load_config, score_window


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    normal = [r for r in rows if str(r["gt_label"]).startswith("normal")]
    attack = [r for r in rows if not str(r["gt_label"]).startswith("normal")]
    by_label = {}
    for label in sorted({r["gt_label"] for r in rows}):
        subset = [r for r in rows if r["gt_label"] == label]
        by_label[label] = {
            "windows": len(subset),
            "candidate_rate": sum(r["is_candidate"] for r in subset) / len(subset),
        }
    return {
        "windows": len(rows),
        "normal_candidate_rate": sum(r["is_candidate"] for r in normal) / len(normal) if normal else None,
        "attack_recall": sum(r["is_candidate"] for r in attack) / len(attack) if attack else None,
        "grid_watch_rate": sum(r.get("grid_watch", False) for r in rows) / len(rows),
        "by_label": by_label,
    }


def evaluate(
    telemetry: Path,
    split_manifest: Path,
    *,
    baseline_model: Path,
    calibration_path: Path,
    config_path: Path,
    progress_log_dir: Path,
    window_s: float,
    stride_s: float,
) -> dict:
    df = pd.read_csv(telemetry)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    test_sessions = set(manifest["test"])
    test = df[df["session_id"].astype(str).isin(test_sessions)]
    legacy_rows = []
    for (_, _), group in test.groupby(["session_id", "gpu_id"], sort=False):
        hz = infer_sample_hz(group)
        screened = stage1_screen_legacy(group.reset_index(drop=True), baseline_mean_w=120.0, sample_hz=hz, window=max(8, int(window_s * hz)), stride=max(1, int(stride_s * hz)))
        for _, row in screened.iterrows():
            legacy_rows.append({"gt_label": row.get("label"), "is_candidate": bool(row["is_candidate"]), "grid_watch": False})

    config = load_config(config_path)
    baseline = CohortBaseline.load(baseline_model)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    windows = build_windows(test, window_s=window_s, stride_s=stride_s, progress_log_dir=progress_log_dir, config=config)
    v2_rows = []
    for _, row in windows.iterrows():
        scored = score_window(row.to_dict(), baseline, calibration, config)
        v2_rows.append({"gt_label": row["gt_label"], **scored})
    report = {
        "scope": "test_split_only",
        "test_sessions": len(test_sessions),
        "legacy": _metrics(legacy_rows),
        "v2": _metrics(v2_rows),
        "ablation": {
            "no_progress_log": "not_run_in_default_eval",
            "no_cohort": "not_run_in_default_eval",
            "no_evidence": "not_run_in_default_eval",
            "nvml_avg_0": "run by regenerating all_v3 with --nvml-avg-window-s 0",
        },
        "notes": "Parameters are fit on train/cal only; test split is not used for tuning.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage1_v2_report.json")
    args = parser.parse_args()
    report = evaluate(
        args.telemetry,
        args.split_manifest,
        baseline_model=args.baseline_model,
        calibration_path=args.calibration,
        config_path=args.stage1_config,
        progress_log_dir=args.progress_log_dir,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = args.out.with_suffix(".md")
    md.write_text(
        "# Stage 1 v2 evaluation\n\n"
        "Official metrics use the test split only. Test results were not used for tuning.\n\n"
        f"- legacy normal candidate rate: {report['legacy'].get('normal_candidate_rate')}\n"
        f"- v2 normal candidate rate: {report['v2'].get('normal_candidate_rate')}\n"
        f"- legacy attack recall: {report['legacy'].get('attack_recall')}\n"
        f"- v2 attack recall: {report['v2'].get('attack_recall')}\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"legacy", "v2"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
