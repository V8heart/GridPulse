"""Feature separability on train+cal only (no test tuning)."""
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
from pipeline.stage1_v2 import build_windows, load_config

FEATURES = [
    "high_load_fraction",
    "longest_high_seconds",
    "swing_abs_w",
    "mean_w",
    "ramp_p95_w_per_s",
    "util_residual_mad_w",
    "dominant_peak_prominence",
    "cross_job_sync_index",
    "declared_family_mean_abs_z",
    "best_other_family_mean_abs_z",
]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    correct = 0.0
    for p in pos:
        correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(correct / (len(pos) * len(neg)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/feature_separability.md")
    parser.add_argument("--json-out", type=Path, default=ROOT / "dataset/eval/feature_separability.json")
    args = parser.parse_args()

    df = pd.read_csv(args.telemetry, low_memory=False)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    fit_ids = set(map(str, manifest["train"])) | set(map(str, manifest["cal"]))
    subset = df[df["session_id"].astype(str).isin(fit_ids)]
    config = load_config(args.stage1_config)
    baseline = CohortBaseline.load(args.baseline_model)
    windows = build_windows(
        subset,
        window_s=30.0,
        stride_s=15.0,
        progress_log_dir=args.progress_log_dir,
        config=config,
    )

    # Attach family distance stats from baseline
    fam_declared = []
    fam_other = []
    for _, row in windows.iterrows():
        stats = baseline.declared_family_mismatch_stats(row.to_dict())
        fam_declared.append(stats.get("declared_family_mean_abs_z"))
        fam_other.append(stats.get("best_other_family_mean_abs_z"))
    windows = windows.copy()
    windows["declared_family_mean_abs_z"] = fam_declared
    windows["best_other_family_mean_abs_z"] = fam_other
    if "cross_job_sync_index" not in windows.columns:
        windows["cross_job_sync_index"] = 0.0

    normals = windows[windows["gt_label"].astype(str).str.startswith("normal")]
    attacks = windows[~windows["gt_label"].astype(str).str.startswith("normal")]

    class_stats = []
    for label, group in windows.groupby(windows["gt_label"].astype(str)):
        n_sessions = group["session_id"].nunique()
        class_stats.append(
            {
                "label": str(label),
                "n_sessions": int(n_sessions),
                "n_windows": int(len(group)),
                "verdict": "판정 보류" if (not str(label).startswith("normal") and n_sessions < 3) else "ok",
            }
        )

    feature_rows = []
    decisions = {}
    for feat in FEATURES:
        series = windows[feat] if feat in windows.columns else pd.Series([np.nan] * len(windows))
        nuniq = series.nunique(dropna=True)
        is_constant = nuniq <= 1
        n_q50 = float(normals[feat].median()) if feat in normals and len(normals) else None
        n_q95 = float(normals[feat].quantile(0.95)) if feat in normals and len(normals) else None
        overall = None
        if not is_constant and len(normals) and len(attacks):
            y = (~windows["gt_label"].astype(str).str.startswith("normal")).astype(int).to_numpy()
            overall = _auc(windows[feat].fillna(0).to_numpy(dtype=float), y)

        by_class = {}
        deferred_classes = []
        for label, group in attacks.groupby(attacks["gt_label"].astype(str)):
            n_sessions = int(group["session_id"].nunique())
            q50 = float(group[feat].median()) if feat in group else None
            entry = {"q50": q50, "n_sessions": n_sessions, "n_windows": int(len(group))}
            if n_sessions < 3:
                entry["verdict"] = "판정 보류"
                deferred_classes.append(str(label))
            else:
                subset = pd.concat([group, normals], ignore_index=True)
                y = (~subset["gt_label"].astype(str).str.startswith("normal")).astype(int).to_numpy()
                auc = None if is_constant else _auc(subset[feat].fillna(0).to_numpy(dtype=float), y)
                entry["auc"] = auc
                entry["verdict"] = "ok"
            by_class[str(label)] = entry

        # Decision: constant -> exclude; overall AUC < 0.6 with enough attack sessions -> disable
        attack_sessions = int(attacks["session_id"].nunique()) if len(attacks) else 0
        if is_constant:
            decision = "exclude_constant"
        elif attack_sessions < 3:
            decision = "deferred_small_n"
        elif overall is not None and overall < 0.6:
            decision = "disable_auc_lt_0.6"
        else:
            decision = "keep"
        decisions[feat] = decision
        feature_rows.append(
            {
                "feature": feat,
                "normal_q50": n_q50,
                "normal_q95": n_q95,
                "auc_overall": overall,
                "nunique": int(nuniq),
                "decision": decision,
                "deferred_classes": deferred_classes,
                "by_class": by_class,
            }
        )

    report = {
        "fit_split": "train+cal",
        "n_windows": int(len(windows)),
        "n_normal_windows": int(len(normals)),
        "n_attack_windows": int(len(attacks)),
        "class_stats": class_stats,
        "features": feature_rows,
        "decisions": decisions,
        "note": "AUC on small session counts is unstable; classes with <3 sessions are deferred.",
    }
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Feature separability (train+cal)",
        "",
        f"windows={len(windows)} normal={len(normals)} attack={len(attacks)}",
        "",
        "## Class sample sizes",
        "",
        "| label | n_sessions | n_windows | verdict |",
        "|-------|------------|-----------|---------|",
    ]
    for row in class_stats:
        lines.append(f"| {row['label']} | {row['n_sessions']} | {row['n_windows']} | {row['verdict']} |")
    lines.extend(
        [
            "",
            "## Features",
            "",
            "| feature | normal_q50 | normal_q95 | auc_overall | nunique | decision |",
            "|---------|------------|------------|-------------|---------|----------|",
        ]
    )
    for row in feature_rows:
        lines.append(
            f"| {row['feature']} | {row['normal_q50']} | {row['normal_q95']} | "
            f"{row['auc_overall']} | {row['nunique']} | {row['decision']} |"
        )
    lines.extend(["", "## Per-attack-class AUC (deferred if sessions < 3)", ""])
    for row in feature_rows:
        lines.append(f"### {row['feature']}")
        lines.append("")
        lines.append("| attack_label | q50 | n_sessions | n_windows | auc | verdict |")
        lines.append("|--------------|-----|------------|-----------|-----|---------|")
        for label, info in sorted(row["by_class"].items()):
            lines.append(
                f"| {label} | {info.get('q50')} | {info.get('n_sessions')} | "
                f"{info.get('n_windows')} | {info.get('auc')} | {info.get('verdict')} |"
            )
        lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decisions": decisions, "class_stats": class_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
