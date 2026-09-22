"""Run nested group-stratified Stage 1 evaluation on real captures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.make_real_split import make_real_split, write_real_split
from pipeline.eval_stage1 import evaluate
from pipeline.fit_stage1_v2 import fit
from pipeline.report_stratified import _write_markdown, stratified_report


def require_leakage_audit(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"metadata leakage audit is required: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(
            f"metadata leakage audit did not pass (max_oof_auc={report.get('max_oof_auc')})"
        )
    return report


def run_real_cv(
    *,
    telemetry: Path,
    labels: Path,
    sessions_root: Path,
    config_path: Path,
    audit_path: Path,
    output_root: Path,
    outer_folds: int = 3,
    seed: int = 7,
    window_s: float = 30.0,
    stride_s: float = 15.0,
) -> dict:
    audit = require_leakage_audit(audit_path)
    split_report = make_real_split(
        pd.read_csv(labels, low_memory=False),
        outer_folds=outer_folds,
        seed=seed,
    )
    write_real_split(split_report, output_root)
    folds = []
    for fold in split_report["folds"]:
        fold_dir = output_root / f"fold_{int(fold['fold']):02d}"
        manifest_path = fold_dir / "split_manifest.json"
        baseline_path = fold_dir / "cohort_baseline_v2.json"
        calibration_path = fold_dir / "stage1_v2_calibration.json"
        fit_report = fit(
            telemetry,
            manifest_path,
            config_path=config_path,
            baseline_out=baseline_path,
            calibration_out=calibration_path,
            progress_log_dir=output_root / "_unused_legacy_steps",
            labels_csv=labels,
            sessions_root=sessions_root,
            window_s=window_s,
            stride_s=stride_s,
        )
        eval_report = evaluate(
            telemetry,
            manifest_path,
            baseline_model=baseline_path,
            calibration_path=calibration_path,
            config_path=config_path,
            progress_log_dir=output_root / "_unused_legacy_steps",
            labels_csv=labels,
            sessions_root=sessions_root,
            window_s=window_s,
            stride_s=stride_s,
            split="test",
        )
        (fold_dir / "stage1_v2_report.json").write_text(
            json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stratified = stratified_report(
            pd.DataFrame(eval_report.get("scored_rows", [])),
            input_scope="all_windows",
        )
        (fold_dir / "stage1_stratified_test.json").write_text(
            json.dumps(stratified, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_markdown(stratified, fold_dir / "stage1_stratified_test.md")
        folds.append(
            {
                "fold": int(fold["fold"]),
                "n_train_sessions": len(fold["train"]),
                "n_cal_sessions": len(fold["cal"]),
                "n_test_sessions": len(fold["test"]),
                "fit": fit_report,
                "test_metrics": eval_report.get("v2", {}),
            }
        )
    summary = {
        "schema_version": "real_nested_cv_v1",
        "leakage_audit": {
            "path": str(audit_path),
            "max_oof_auc": audit.get("max_oof_auc"),
            "passed": True,
        },
        "fold_sha256": split_report["fold_sha256"],
        "folds": folds,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "dataset/real/index/labels.csv"
    )
    parser.add_argument(
        "--sessions-root", type=Path, default=ROOT / "dataset/real/sessions"
    )
    parser.add_argument(
        "--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml"
    )
    parser.add_argument(
        "--leakage-audit",
        type=Path,
        default=ROOT / "dataset/eval/v2_regen/metadata_leakage_audit.json",
    )
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "dataset/eval/real_cv"
    )
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    args = parser.parse_args()
    report = run_real_cv(
        telemetry=args.telemetry,
        labels=args.labels,
        sessions_root=args.sessions_root,
        config_path=args.stage1_config,
        audit_path=args.leakage_audit,
        output_root=args.out_root,
        outer_folds=args.outer_folds,
        seed=args.seed,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    print(json.dumps({"folds": len(report["folds"]), "audit_passed": True}, indent=2))


if __name__ == "__main__":
    main()
