"""Disable active evidence whose attack firing rate does not beat normal rate.

Does not loosen quantiles. train+cal observation only; does not re-enter alpha selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.report_evidence_firing import _count_from_stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/evidence_activation.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.stage1_config.read_text(encoding="utf-8")) or {}
    weights = dict(cfg.get("evidence_weights") or {})
    _, rates, n_windows = _count_from_stage1(
        args.telemetry,
        args.split_manifest,
        splits=["train", "cal"],
        baseline_model=args.baseline_model,
        calibration=args.calibration,
        config_path=args.stage1_config,
        progress_log_dir=args.progress_log_dir,
    )

    newly_disabled = []
    notes = dict(cfg.get("evidence_notes") or {})
    for name, weight in list(weights.items()):
        if float(weight) <= 0:
            continue
        info = rates.get(name) or {}
        attack_rate = float(info.get("attack_rate") or 0.0)
        normal_rate = float(info.get("normal_rate") or 0.0)
        # Require attack fires strictly more often than normal; else deactivate.
        if attack_rate <= normal_rate or attack_rate <= 0.0:
            weights[name] = 0.0
            newly_disabled.append(name)
            notes[name] = (
                f"disabled: attack_rate={attack_rate:.4f} <= normal_rate={normal_rate:.4f} "
                "(no quantile loosening)"
            )

    cfg["evidence_weights"] = weights
    cfg["evidence_notes"] = notes
    args.stage1_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    active = sorted(k for k, v in weights.items() if float(v) > 0)
    report = {
        "n_windows": n_windows,
        "rates": rates,
        "newly_disabled": newly_disabled,
        "active": active,
        "n_active": len(active),
        "gate_active_ge_4": len(active) >= 4,
        "stop_before_e3": len(active) < 4,
        "note": (
            "If n_active < 4, E2 does not claim success for Stage2 matching; "
            "need new discriminative axes / real capture + step logs."
        ),
    }
    # merge with prior activation file if present
    if args.out.exists():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        report["disabled"] = sorted(set(prev.get("disabled", [])) | set(newly_disabled) | {
            k for k, v in weights.items() if float(v) <= 0
        })
        report["prior"] = {k: prev[k] for k in ("active", "n_active") if k in prev}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
