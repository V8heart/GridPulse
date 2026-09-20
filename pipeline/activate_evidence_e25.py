"""E2.5 stage-2 boolean activation gate (train+cal). Records activating_classes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.baseline import CohortBaseline
from pipeline.evidence_vocab import (
    THRESHOLD_BASED_EVIDENCE,
    merge_evidence_gate,
)
from pipeline.stage1_v2 import build_windows, load_config, score_window

# Evidence evaluated for activation (weights may be zeroed).
CANDIDATE_EVIDENCE = [
    "period_mismatch",
    "unexplained_changepoint",
    "progress_log_missing",
    "strong_peak",
    "sustained_high_load",
    "flat_power",
    "high_ramp",
    "util_power_decoupled",
    "declared_family_mismatch",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument(
        "--separability",
        type=Path,
        default=ROOT / "dataset/eval/feature_separability.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/evidence_activation.json")
    parser.add_argument("--status-out", type=Path, default=ROOT / "dataset/eval/e25_status.md")
    args = parser.parse_args()

    config = load_config(args.stage1_config)
    gate = merge_evidence_gate(config)
    rate_min = float(gate["activate_attack_rate_min"])
    ratio_min = float(gate["activate_rate_ratio_min"])

    sep = json.loads(args.separability.read_text(encoding="utf-8")) if args.separability.exists() else {}
    feat_meta = {f["feature"]: f for f in sep.get("features", [])}
    decisions = sep.get("decisions", {})

    # Evidence pruned if all mapped features are pruned/excluded (except period/progress which have no feature map).
    pruned_evidence: set[str] = {"cross_job_sync"}
    for feat, decision in decisions.items():
        if decision in {"exclude_constant", "prune_auc_dir_lt_min"}:
            for ev in FEATURE_TO_EVIDENCE.get(feat, []):
                # Keep evidence if any mapped feature is kept
                pass
    # Recompute: evidence kept if ANY mapped feature is keep; else if all mapped are bad -> prune
    evidence_feature_hits: dict[str, list[str]] = defaultdict(list)
    for feat, evs in FEATURE_TO_EVIDENCE.items():
        for ev in evs:
            evidence_feature_hits[ev].append(feat)
    for ev, feats in evidence_feature_hits.items():
        keeps = [decisions.get(f) == "keep" for f in feats]
        if feats and not any(keeps):
            pruned_evidence.add(ev)

    df = pd.read_csv(args.telemetry, low_memory=False)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    fit_ids = set(map(str, manifest["train"])) | set(map(str, manifest["cal"]))
    subset = df[df["session_id"].astype(str).isin(fit_ids)]
    baseline = CohortBaseline.load(args.baseline_model)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    windows = build_windows(
        subset,
        window_s=30.0,
        stride_s=15.0,
        progress_log_dir=args.progress_log_dir,
        config=config,
    )

    # Count firings with all candidate weights temporarily positive for bool measurement
    measure_cfg = json.loads(json.dumps(config))
    for k in CANDIDATE_EVIDENCE:
        measure_cfg.setdefault("evidence_weights", {})[k] = 1.0

    totals = defaultdict(int)
    true_counts: dict[str, dict[str, int]] = {e: defaultdict(int) for e in CANDIDATE_EVIDENCE}
    for _, row in windows.iterrows():
        lab = str(row["gt_label"])
        totals[lab] += 1
        scored = score_window(row.to_dict(), baseline, calibration, measure_cfg)
        eb = scored.get("evidence_bool") or {}
        for ev in CANDIDATE_EVIDENCE:
            if eb.get(ev):
                true_counts[ev][lab] += 1

    n_normal = sum(v for k, v in totals.items() if k.startswith("normal"))
    attack_labels = sorted(k for k in totals if not k.startswith("normal"))

    evidence_report = {}
    active = []
    for ev in CANDIDATE_EVIDENCE:
        normal_true = sum(true_counts[ev][k] for k in totals if k.startswith("normal"))
        normal_rate = (normal_true / n_normal) if n_normal else 0.0
        per_class = {}
        activating = []
        for lab in attack_labels:
            n = totals[lab]
            rate = (true_counts[ev][lab] / n) if n else 0.0
            per_class[lab] = {
                "attack_rate": rate,
                "n_windows": n,
                "n_true": true_counts[ev][lab],
                "normal_rate": normal_rate,
                "passes": bool(rate >= rate_min and rate >= ratio_min * normal_rate),
            }
            if per_class[lab]["passes"]:
                activating.append(lab)

        # Attach feature direction/best_auc from separability when available
        direction = None
        best_auc = None
        best_class = None
        for feat, meta in feat_meta.items():
            if ev in meta.get("maps_to_evidence", []):
                if meta.get("decision") == "keep":
                    direction = meta.get("direction")
                    if best_auc is None or (meta.get("best_auc_dir") or 0) > (best_auc or 0):
                        best_auc = meta.get("best_auc")
                        best_class = meta.get("best_class")

        is_active = bool(activating) and ev not in pruned_evidence
        if ev == "cross_job_sync":
            is_active = False
        evidence_report[ev] = {
            "active": is_active,
            "pruned_by_stage1": ev in pruned_evidence,
            "activating_classes": activating,
            "direction": direction,
            "best_auc": best_auc,
            "best_class": best_class,
            "normal_rate": normal_rate,
            "per_class_rates": per_class,
        }
        if is_active:
            active.append(ev)

    # Update yaml weights: active keep prior positive defaults; inactive -> 0
    cfg = yaml.safe_load(args.stage1_config.read_text(encoding="utf-8")) or {}
    weights = dict(cfg.get("evidence_weights") or {})
    default_pos = {
        "period_mismatch": 1.0,
        "unexplained_changepoint": 0.8,
        "cross_job_sync": 0.0,
        "progress_log_missing": 0.4,
        "sustained_high_load": 1.0,
        "flat_power": 0.8,
        "high_ramp": 0.8,
        "util_power_decoupled": 1.0,
        "declared_family_mismatch": 1.0,
    }
    for ev, default in default_pos.items():
        if ev in evidence_report and evidence_report[ev]["active"]:
            weights[ev] = float(default)
        else:
            weights[ev] = 0.0
    cfg["evidence_weights"] = weights
    cfg["evidence_gate"] = gate
    args.stage1_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    threshold_active = [e for e in active if e in THRESHOLD_BASED_EVIDENCE]
    n_active = len(active)
    n_thr = len(threshold_active)
    stop = not (n_active >= 4 and n_thr >= 2)
    report = {
        "evidence_gate": gate,
        "evidence": evidence_report,
        "active": active,
        "n_active": n_active,
        "threshold_based_active": threshold_active,
        "n_threshold_based_active": n_thr,
        "stop_before_e3": stop,
        "gate_active_ge_4": n_active >= 4,
        "gate_threshold_based_ge_2": n_thr >= 2,
        "note": (
            "Boolean activation on train+cal after refit/recalibration. "
            "Do not retune gate params from these results."
        ),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# E2.5 status",
        "",
        f"evidence_gate: `{gate}`",
        "",
        f"- n_active={n_active} (need ≥4): **{'PASS' if n_active >= 4 else 'FAIL'}**",
        f"- n_threshold_based={n_thr} (need ≥2): **{'PASS' if n_thr >= 2 else 'FAIL'}**",
        f"- stop_before_e3: **{stop}**",
        "",
        "## Active evidence",
        "",
        "| evidence | activating_classes | direction | best_auc |",
        "|----------|-------------------|-----------|----------|",
    ]
    for ev in active:
        info = evidence_report[ev]
        lines.append(
            f"| {ev} | {', '.join(info['activating_classes']) or '-'} | "
            f"{info.get('direction')} | {info.get('best_auc')} |"
        )
    lines.extend(["", "## All candidates", ""])
    for ev, info in evidence_report.items():
        lines.append(
            f"- `{ev}` active={info['active']} classes={info['activating_classes']} "
            f"pruned={info['pruned_by_stage1']}"
        )
    args.status_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n_active", "active", "n_threshold_based_active", "threshold_based_active", "stop_before_e3", "evidence_gate"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
