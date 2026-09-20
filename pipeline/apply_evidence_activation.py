"""Apply separability decisions to evidence_weights (train+cal only; no test tuning)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Map continuous features -> evidence weight keys that depend on them.
FEATURE_TO_EVIDENCE = {
    "high_load_fraction": ["sustained_high_load"],
    "longest_high_seconds": ["sustained_high_load"],
    "swing_abs_w": ["flat_power"],
    "mean_w": ["flat_power"],
    "ramp_p95_w_per_s": ["high_ramp"],
    "util_residual_mad_w": ["util_power_decoupled"],
    "cross_job_sync_index": ["cross_job_sync"],
    # family z feeds declared_family_mismatch; keep when decision is keep
    "declared_family_mean_abs_z": ["declared_family_mismatch"],
    "best_other_family_mean_abs_z": ["declared_family_mismatch"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--separability",
        type=Path,
        default=ROOT / "dataset/eval/feature_separability.json",
    )
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument(
        "--out-report",
        type=Path,
        default=ROOT / "dataset/eval/evidence_activation.json",
    )
    args = parser.parse_args()

    sep = json.loads(args.separability.read_text(encoding="utf-8"))
    decisions = sep["decisions"]
    disable: set[str] = set()
    keep: set[str] = set()
    for feat, decision in decisions.items():
        for ev in FEATURE_TO_EVIDENCE.get(feat, []):
            if decision in {"exclude_constant", "disable_auc_lt_0.6"}:
                disable.add(ev)
            elif decision == "keep":
                keep.add(ev)
    # If any dependent feature disables, disable the evidence.
    # declared_family_mismatch kept only if at least one family feature is keep and none disable it.
    if "declared_family_mismatch" in disable and "declared_family_mismatch" in keep:
        # both family features should be keep; if one disable wins, drop
        fam_decisions = [
            decisions.get("declared_family_mean_abs_z"),
            decisions.get("best_other_family_mean_abs_z"),
        ]
        if all(d == "keep" for d in fam_decisions):
            disable.discard("declared_family_mismatch")

    cfg = yaml.safe_load(args.stage1_config.read_text(encoding="utf-8")) or {}
    weights = dict(cfg.get("evidence_weights") or {})
    # Always zero out known-broken sync.
    disable.add("cross_job_sync")
    for name in list(weights):
        if name in disable:
            weights[name] = 0.0
    cfg["evidence_weights"] = weights
    # flat_power: both swing and mean failed AUC -> already disabled; record mode
    cfg.setdefault("evidence_notes", {})
    cfg["evidence_notes"]["flat_power"] = "disabled: swing_abs_w and mean_w both AUC<0.6 on train+cal"
    cfg["evidence_notes"]["sustained_high_load"] = (
        "disabled: high_load_fraction and longest_high_seconds both AUC<0.6"
    )
    cfg["evidence_notes"]["cross_job_sync"] = "disabled: cross_job_sync_index never computed (constant 0)"
    args.stage1_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    active = sorted(k for k, v in weights.items() if float(v) > 0)
    report = {
        "disabled": sorted(disable),
        "active_weights": {k: v for k, v in weights.items() if float(v) > 0},
        "n_active": len(active),
        "active": active,
        "gate_active_ge_4": len(active) >= 4,
        "note": "If n_active < 4, stop before claiming E2 success; need new axes / real capture.",
    }
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
