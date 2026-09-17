"""Break down holdout sessions by declared_job_family × progress_log × grid_watch."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.baseline import CohortBaseline
from pipeline.stage1_v2 import build_windows, load_config, score_window


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
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/holdout_breakdown.json")
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    holdout = list(manifest.get("unseen_param_holdout", {}).get("swma_period_2_8_3_2", []))
    df = pd.read_csv(args.telemetry, low_memory=False)
    subset = df[df["session_id"].astype(str).isin(set(map(str, holdout)))]
    config = load_config(args.stage1_config)
    baseline = CohortBaseline.load(args.baseline_model)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    windows = build_windows(
        subset,
        window_s=args.window_s,
        stride_s=args.stride_s,
        progress_log_dir=args.progress_log_dir,
        config=config,
    )

    session_rows = []
    for session_id, group in windows.groupby("session_id"):
        scores = [score_window(row.to_dict(), baseline, calibration, config) for _, row in group.iterrows()]
        first = group.iloc[0]
        evidence = first["evidence"] if isinstance(first.get("evidence"), dict) else {}
        progress_missing = bool(evidence.get("progress_log_missing", True))
        family = str(first.get("declared_job_family", "unknown"))
        grid_watch = any(bool(s.get("grid_watch")) for s in scores)
        session_rows.append({
            "session_id": str(session_id),
            "declared_job_family": family,
            "progress_log": "missing" if progress_missing else "present",
            "grid_watch": bool(grid_watch),
            "is_candidate_any": any(bool(s.get("is_candidate")) for s in scores),
            "n_windows": int(len(group)),
        })

    counter = Counter(
        (r["declared_job_family"], r["progress_log"], r["grid_watch"]) for r in session_rows
    )
    table = [
        {
            "declared_job_family": fam,
            "progress_log": prog,
            "grid_watch": gw,
            "n_sessions": n,
        }
        for (fam, prog, gw), n in sorted(counter.items())
    ]
    report = {
        "holdout_sessions": list(map(str, holdout)),
        "n_holdout": len(holdout),
        "n_scored_sessions": len(session_rows),
        "sessions": session_rows,
        "breakdown": table,
        "note": "Descriptive breakdown only; not used to retune corpus.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Holdout session breakdown",
        "",
        f"n_holdout={len(holdout)} (scored={len(session_rows)})",
        "",
        "| declared_job_family | progress_log | grid_watch | n_sessions |",
        "|---------------------|--------------|------------|------------|",
    ]
    for row in table:
        md.append(
            f"| {row['declared_job_family']} | {row['progress_log']} | {row['grid_watch']} | {row['n_sessions']} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"n_holdout": len(holdout), "breakdown": table}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
