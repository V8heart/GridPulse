"""Fit Stage 1 v2 cohort baseline and Mondrian calibration (single refit)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.progress_log_policy import POLICY_VERSION
from pipeline.baseline import DEFAULT_MAD_FLOORS, CohortBaseline, FEATURES_V2
from pipeline.fit_evidence_thresholds import fit_thresholds_on_normals
from pipeline.stage1_v2 import (
    CALIBRATION_SCHEMA_VERSION,
    EVIDENCE_WEIGHT_CAP,
    SUSPICIOUS_EVIDENCE_KEYS,
    build_windows,
    impact_components,
    load_config,
    load_truth_index,
    score_window,
)
from pipeline.evidence_vocab import strip_recompute_bools, to_bool_evidence


def _session_equalized_rate(windows: pd.DataFrame, key: str) -> float:
    if windows.empty or "session_id" not in windows:
        return 0.0
    rates = []
    for _, group in windows.groupby("session_id"):
        fired = 0
        n = 0
        for _, row in group.iterrows():
            evidence = row.get("evidence_bool") or row.get("evidence") or {}
            if isinstance(evidence, dict) and evidence.get(key):
                fired += 1
            n += 1
        if n:
            rates.append(fired / n)
    if not rates:
        return 0.0
    return float((1.0 + sum(rates)) / (len(rates) + 2.0))


def fit_evidence_surprisal_weights(
    train_normals: pd.DataFrame,
    baseline: CohortBaseline,
    config: dict,
) -> dict:
    """Train-normal surprisal weights; attack labels never used."""
    # Recompute bools with current thresholds/baseline.
    rows = []
    for _, row in train_normals.iterrows():
        data = row.to_dict()
        raw = dict(data.get("evidence") or {})
        raw.update({k: v for k, v in baseline.declared_family_mismatch_stats(data).items() if v is not None})
        merged = strip_recompute_bools({**data, **raw})
        bools = to_bool_evidence(merged, config)
        rows.append({**data, "evidence_bool": bools})
    frame = pd.DataFrame(rows)
    weights = {}
    rates = {}
    for key in SUSPICIOUS_EVIDENCE_KEYS:
        rate = _session_equalized_rate(frame, key)
        rates[key] = rate
        weights[key] = float(min(EVIDENCE_WEIGHT_CAP, -math.log(max(rate, 1e-12))))
    # Never score benign availability signals.
    weights["period_match"] = 0.0
    weights["progress_log_available"] = 0.0
    return {
        "evidence_surprisal_weights": weights,
        "evidence_normal_firing_rates": rates,
        "provenance": "train_normal_session_equalized_surprisal",
    }


def _collect_feature_abs_z(cal_normals: pd.DataFrame, baseline: CohortBaseline) -> dict:
    by_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_bucket: dict[str, list[float]] = defaultdict(list)
    per_row_z: list[dict[str, float]] = []
    for _, row in cal_normals.iterrows():
        z = baseline.robust_z(row.to_dict())
        abs_z = {k: abs(v) for k, v in z.items() if k != "baseline_level" and k in FEATURES_V2}
        per_row_z.append(abs_z)
        family = str(row.get("declared_job_family", "unknown"))
        for feature, value in abs_z.items():
            by_family[family][feature].append(float(value))
            global_bucket[feature].append(float(value))
    out = {family: dict(feats) for family, feats in by_family.items()}
    out["global"] = dict(global_bucket)
    return out, per_row_z


def fit(
    telemetry: Path,
    split_manifest: Path,
    *,
    config_path: Path,
    baseline_out: Path,
    calibration_out: Path,
    progress_log_dir: Path,
    labels_csv: Path,
    sessions_root: Path | None,
    window_s: float,
    stride_s: float,
) -> dict:
    df = pd.read_csv(telemetry, low_memory=False)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    config = load_config(config_path)
    train = set(map(str, manifest["train"]))
    cal = set(map(str, manifest["cal"]))
    train_df = df[df["session_id"].astype(str).isin(train)]
    cal_df = df[df["session_id"].astype(str).isin(cal)]
    truth_index = load_truth_index(labels_csv, sessions_root=sessions_root)
    train_windows = build_windows(
        train_df,
        window_s=window_s,
        stride_s=stride_s,
        progress_log_dir=progress_log_dir,
        sessions_root=sessions_root,
        truth_index=truth_index,
        config=config,
    )
    cal_windows = build_windows(
        cal_df,
        window_s=window_s,
        stride_s=stride_s,
        progress_log_dir=progress_log_dir,
        sessions_root=sessions_root,
        truth_index=truth_index,
        config=config,
    )
    train_normals = train_windows[train_windows["gt_label"].astype(str).str.startswith("normal")]
    cal_normals = cal_windows[cal_windows["gt_label"].astype(str).str.startswith("normal")]

    mad_floors = dict(config.get("mad_floors") or DEFAULT_MAD_FLOORS)
    baseline = CohortBaseline().fit(
        train_normals,
        min_windows=int(config.get("min_windows", 30)),
        mad_floors=mad_floors,
    )

    thresholds = fit_thresholds_on_normals(train_normals, config, baseline=baseline)
    config = dict(config)
    config["evidence_thresholds"] = thresholds
    if "tau_peak" in thresholds:
        config["tau_peak"] = thresholds["tau_peak"]

    weight_report = fit_evidence_surprisal_weights(train_normals, baseline, config)
    config["evidence_weights"] = {
        **config.get("evidence_weights", {}),
        **weight_report["evidence_surprisal_weights"],
    }

    policy = (manifest.get("progress_log_policy") or {})
    policy_ok = (
        policy.get("version") == config.get("progress_log_policy_required", POLICY_VERSION)
        or policy.get("version") == POLICY_VERSION
    )
    if not policy_ok:
        config["evidence_weights"]["progress_log_missing"] = 0.0

    feature_abs_z, _ = _collect_feature_abs_z(cal_normals, baseline)

    # Partial calibration used while scoring cal normals (LOO feature tails).
    partial_cal = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "feature_abs_z": feature_abs_z,
        "evidence_surprisal_weights": weight_report["evidence_surprisal_weights"],
        "progress_log_policy_ok": policy_ok,
        "normal_u_scores": {"full": {"global": []}, "no_evidence": {"global": []}},
        "impact_quantiles": {},
    }

    full_by_family: dict[str, list[float]] = defaultdict(list)
    none_by_family: dict[str, list[float]] = defaultdict(list)
    full_global: list[float] = []
    none_global: list[float] = []
    for _, row in cal_normals.iterrows():
        data = row.to_dict()
        z = baseline.robust_z(data)
        abs_z = {k: abs(v) for k, v in z.items() if k != "baseline_level" and k in FEATURES_V2}
        full = score_window(
            data, baseline, partial_cal, config, profile="full", leave_out_feature_z=abs_z
        )
        none = score_window(
            data, baseline, partial_cal, config, profile="no_evidence", leave_out_feature_z=abs_z
        )
        family = str(data.get("declared_job_family", "unknown"))
        full_by_family[family].append(float(full["u_score"]))
        none_by_family[family].append(float(none["u_score"]))
        full_global.append(float(full["u_score"]))
        none_global.append(float(none["u_score"]))

    all_impacts = [
        float(impact_components(row.to_dict(), config).get("impact_raw", 0.0))
        for _, row in cal_windows.iterrows()
    ]

    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "fit_split": "cal",
        "progress_log_policy_ok": policy_ok,
        "progress_log_policy": policy,
        "feature_abs_z": {
            family: {feat: [float(x) for x in values] for feat, values in feats.items()}
            for family, feats in feature_abs_z.items()
        },
        "normal_u_scores": {
            "full": {**{k: v for k, v in full_by_family.items()}, "global": full_global},
            "no_evidence": {**{k: v for k, v in none_by_family.items()}, "global": none_global},
        },
        "evidence_surprisal_weights": weight_report["evidence_surprisal_weights"],
        "evidence_normal_firing_rates": weight_report["evidence_normal_firing_rates"],
        "evidence_weight_provenance": weight_report["provenance"],
        "evidence_thresholds": thresholds,
        "impact_quantiles": {
            "p50": float(np.quantile(all_impacts, 0.5)) if all_impacts else 0.0,
            "p90": float(np.quantile(all_impacts, 0.9)) if all_impacts else 0.0,
        },
        "n_cal_windows": int(len(cal_windows)),
        "n_cal_normal_windows": int(len(cal_normals)),
        "n_train_normal_windows": int(len(train_normals)),
        "baseline_fallback_note": (
            "min_windows=30; evaluation family falls back to global when n<30"
        ),
        "mondrian_min_n": int(config.get("mondrian_min_n", 30)),
        "config": config,
    }
    baseline.save(baseline_out)
    calibration_out.parent.mkdir(parents=True, exist_ok=True)
    calibration_out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "train_windows": len(train_windows),
        "cal_windows": len(cal_windows),
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "progress_log_policy_ok": policy_ok,
        "impact_quantiles": calibration["impact_quantiles"],
        "n_cal_normal_windows": int(len(cal_normals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--stage1-config", "--config", dest="stage1_config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--baseline-out", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration-out", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument("--sessions-root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/synthetic/index/labels.csv")
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
        labels_csv=args.labels,
        sessions_root=args.sessions_root,
        window_s=args.window_s,
        stride_s=args.stride_s,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
