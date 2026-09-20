"""E2.5 feature separability: class-wise max(auc,1-auc) prune + direction (train+cal)."""
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
from pipeline.evidence_vocab import merge_evidence_gate, prominence_log
from pipeline.stage1_v2 import build_windows, load_config

FEATURES = [
    "high_load_fraction",
    "longest_high_seconds",
    "swing_abs_w",
    "mean_w",
    "ramp_p95_w_per_s",
    "util_residual_mad_w",
    "dominant_peak_prominence_log",
    "cross_job_sync_index",
    "declared_family_mean_abs_z",
    "best_other_family_mean_abs_z",
]

FEATURE_TO_EVIDENCE = {
    "high_load_fraction": ["sustained_high_load"],
    "longest_high_seconds": ["sustained_high_load"],
    "swing_abs_w": ["flat_power"],
    "mean_w": ["flat_power"],
    "ramp_p95_w_per_s": ["flat_power", "high_ramp"],
    "util_residual_mad_w": ["util_power_decoupled"],
    "dominant_peak_prominence_log": ["strong_peak"],
    "cross_job_sync_index": ["cross_job_sync"],
    "declared_family_mean_abs_z": ["declared_family_mismatch"],
    "best_other_family_mean_abs_z": ["declared_family_mismatch"],
}


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
    gate = merge_evidence_gate(config)
    prune_min = float(gate["prune_auc_min"])
    baseline = CohortBaseline.load(args.baseline_model)
    windows = build_windows(
        subset,
        window_s=30.0,
        stride_s=15.0,
        progress_log_dir=args.progress_log_dir,
        config=config,
    )
    windows = windows.copy()
    fam_declared, fam_other = [], []
    for _, row in windows.iterrows():
        stats = baseline.declared_family_mismatch_stats(row.to_dict())
        fam_declared.append(stats.get("declared_family_mean_abs_z"))
        fam_other.append(stats.get("best_other_family_mean_abs_z"))
    windows["declared_family_mean_abs_z"] = fam_declared
    windows["best_other_family_mean_abs_z"] = fam_other
    if "cross_job_sync_index" not in windows.columns:
        windows["cross_job_sync_index"] = 0.0
    if "dominant_peak_prominence_log" not in windows.columns:
        windows["dominant_peak_prominence_log"] = windows["dominant_peak_prominence"].map(prominence_log)

    normals = windows[windows["gt_label"].astype(str).str.startswith("normal")]
    attacks = windows[~windows["gt_label"].astype(str).str.startswith("normal")]

    class_stats = []
    for label, group in windows.groupby(windows["gt_label"].astype(str)):
        n_sessions = int(group["session_id"].nunique())
        class_stats.append(
            {
                "label": str(label),
                "n_sessions": n_sessions,
                "n_windows": int(len(group)),
                "verdict": "판정 보류" if (not str(label).startswith("normal") and n_sessions < 3) else "ok",
            }
        )

    feature_rows = []
    decisions = {}
    for feat in FEATURES:
        series = windows[feat] if feat in windows.columns else pd.Series([np.nan] * len(windows))
        nuniq = int(series.nunique(dropna=True))
        is_constant = nuniq <= 1
        n_q50 = float(normals[feat].median()) if feat in normals and len(normals) else None
        n_q95 = float(normals[feat].quantile(0.95)) if feat in normals and len(normals) else None

        by_class = {}
        best_class, best_dir_auc, best_raw_auc = None, -1.0, None
        for label, group in attacks.groupby(attacks["gt_label"].astype(str)):
            n_sessions = int(group["session_id"].nunique())
            q50 = float(group[feat].median()) if feat in group else None
            entry = {"q50": q50, "n_sessions": n_sessions, "n_windows": int(len(group))}
            if n_sessions < 3 or is_constant:
                entry["verdict"] = "판정 보류" if n_sessions < 3 else "constant"
                entry["auc"] = None
                entry["auc_dir"] = None
            else:
                subset_df = pd.concat([group, normals], ignore_index=True)
                y = (~subset_df["gt_label"].astype(str).str.startswith("normal")).astype(int).to_numpy()
                auc = _auc(subset_df[feat].fillna(0).to_numpy(dtype=float), y)
                auc_dir = None if auc is None else max(auc, 1.0 - auc)
                entry["auc"] = auc
                entry["auc_dir"] = auc_dir
                entry["verdict"] = "ok"
                if auc_dir is not None and auc_dir > best_dir_auc:
                    best_dir_auc = auc_dir
                    best_raw_auc = auc
                    best_class = str(label)
            by_class[str(label)] = entry

        if is_constant:
            decision = "exclude_constant"
            direction = None
        elif best_dir_auc < 0:
            decision = "deferred_small_n"
            direction = None
        elif best_dir_auc < prune_min:
            decision = "prune_auc_dir_lt_min"
            direction = None
        else:
            decision = "keep"
            direction = "lower_is_attack" if (best_raw_auc is not None and best_raw_auc < 0.5) else "higher_is_attack"

        decisions[feat] = decision
        feature_rows.append(
            {
                "feature": feat,
                "normal_q50": n_q50,
                "normal_q95": n_q95,
                "best_class": best_class,
                "best_auc": best_raw_auc,
                "best_auc_dir": None if best_dir_auc < 0 else best_dir_auc,
                "direction": direction,
                "nunique": nuniq,
                "decision": decision,
                "maps_to_evidence": FEATURE_TO_EVIDENCE.get(feat, []),
                "by_class": by_class,
            }
        )

    report = {
        "fit_split": "train+cal",
        "evidence_gate": gate,
        "n_windows": int(len(windows)),
        "n_normal_windows": int(len(normals)),
        "n_attack_windows": int(len(attacks)),
        "swma_attack_share": float(len(attacks[attacks["gt_label"] == "swma"]) / len(attacks)) if len(attacks) else None,
        "class_stats": class_stats,
        "features": feature_rows,
        "decisions": decisions,
        "note": "Stage-1 prune uses class-wise max(auc,1-auc) vs evidence_gate.prune_auc_min (operating point).",
    }
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Feature separability (train+cal, E2.5)",
        "",
        f"evidence_gate: `{gate}`",
        f"windows={len(windows)} normal={len(normals)} attack={len(attacks)}",
        "",
        "## Features",
        "",
        "| feature | best_class | best_auc | best_auc_dir | direction | decision |",
        "|---------|------------|----------|--------------|-----------|----------|",
    ]
    for row in feature_rows:
        lines.append(
            f"| {row['feature']} | {row['best_class']} | {row['best_auc']} | "
            f"{row['best_auc_dir']} | {row['direction']} | {row['decision']} |"
        )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decisions": decisions, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
