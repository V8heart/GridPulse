"""Cal-only Stage1 score AUC and alpha sweep (test sealed until selection)."""
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

from pipeline.attack_visibility import attack_visibility_group, parse_period_s
from pipeline.baseline import CohortBaseline
from pipeline.stage1_v2 import build_windows, load_config, load_truth_index, score_window


def _cohort_key(row) -> str:
    family = str(row["declared_job_family"] if "declared_job_family" in row else "unknown")
    band = row["expected_band"] if "expected_band" in row else None
    if band is None or str(band) in {"", "nan", "None"}:
        return family
    return f"{family}|{band}"


def _target_normals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[frame["gt_label"].astype(str).str.startswith("normal")]
    if "gpu_role" in out.columns:
        out = out[out["gpu_role"].astype(str).eq("target") | out["gpu_role"].isna()]
    return out


def cohort_weighted_fp(
    cal_normals: pd.DataFrame,
    *,
    train_normals: pd.DataFrame | None = None,
    missing_cal_cohorts: list[str] | None = None,
    weight_sessions: dict[str, int] | None = None,
) -> tuple[float | None, list[str]]:
    """Session-weighted normal FP. Missing cal cohorts use train rates."""
    substituted = []
    rates: dict[str, float] = {}
    if cal_normals is not None and not cal_normals.empty:
        for cohort, group in cal_normals.groupby(cal_normals.apply(_cohort_key, axis=1)):
            rates[str(cohort)] = float(group["is_candidate"].mean())
    if missing_cal_cohorts and train_normals is not None and not train_normals.empty:
        train_rates = {
            str(cohort): float(group["is_candidate"].mean())
            for cohort, group in train_normals.groupby(train_normals.apply(_cohort_key, axis=1))
        }
        for cohort in missing_cal_cohorts:
            if cohort in train_rates:
                rates[cohort] = train_rates[cohort]
                substituted.append(cohort)
    if not rates:
        return None, substituted
    if weight_sessions is None:
        weight_sessions = {}
        frames = [cal_normals]
        if train_normals is not None:
            frames.append(train_normals)
        combined = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
        for cohort, group in combined.groupby(combined.apply(_cohort_key, axis=1)):
            weight_sessions[str(cohort)] = int(group["session_id"].nunique()) if "session_id" in group.columns else int(len(group))
    numer = 0.0
    denom = 0.0
    for cohort, rate in rates.items():
        weight = float(weight_sessions.get(cohort) or 0)
        if weight <= 0:
            continue
        numer += rate * weight
        denom += weight
    if denom <= 0:
        return None, substituted
    return float(numer / denom), substituted


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney / ROC-AUC
    correct = 0.0
    for p in pos:
        correct += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(correct / (len(pos) * len(neg)))


def run(
    telemetry: Path,
    split_manifest: Path,
    *,
    baseline_model: Path,
    calibration_path: Path,
    config_path: Path,
    progress_log_dir: Path,
    window_s: float,
    stride_s: float,
    alphas: list[float],
    labels_csv: Path | None = None,
    sessions_root: Path | None = None,
    progress_source: str = "visible",
    write_config: bool = True,
) -> dict:
    df = pd.read_csv(telemetry, low_memory=False)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    cal_ids = set(map(str, manifest["cal"]))
    train_ids = set(map(str, manifest.get("train") or []))
    cal = df[df["session_id"].astype(str).isin(cal_ids)]
    train = df[df["session_id"].astype(str).isin(train_ids)] if train_ids else cal.iloc[0:0]
    config = load_config(config_path)
    baseline = CohortBaseline.load(baseline_model)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("evidence_surprisal_weights"):
        config = dict(config)
        config["evidence_weights"] = {
            **config.get("evidence_weights", {}),
            **calibration["evidence_surprisal_weights"],
        }
    if calibration.get("evidence_thresholds"):
        config["evidence_thresholds"] = calibration["evidence_thresholds"]
    truth_index = None
    if labels_csv is not None:
        truth_index = load_truth_index(labels_csv, sessions_root=sessions_root)
    windows = build_windows(
        cal,
        window_s=window_s,
        stride_s=stride_s,
        progress_log_dir=progress_log_dir,
        sessions_root=sessions_root,
        truth_index=truth_index,
        config=config,
        progress_source=progress_source,
    )
    train_windows = (
        build_windows(
            train,
            window_s=window_s,
            stride_s=stride_s,
            progress_log_dir=progress_log_dir,
            sessions_root=sessions_root,
            truth_index=truth_index,
            config=config,
            progress_source=progress_source,
        )
        if not train.empty
        else windows.iloc[0:0]
    )

    scored = []
    for _, row in windows.iterrows():
        out = score_window(row.to_dict(), baseline, calibration, config)
        scored.append({
            "gt_label": str(row.get("gt_label", "")),
            "gt_variant": row.get("gt_variant"),
            "period_s": row.get("period_s"),
            "visibility_group": row.get("visibility_group") or attack_visibility_group(
                gt_label=str(row.get("gt_label", "")),
                gt_variant=row.get("gt_variant"),
                period_s=row.get("period_s") if pd.notna(row.get("period_s")) else parse_period_s(row.get("gt_params_json")),
            ),
            "u_score": out["u_score"],
            "p_value": out["p_value"],
            "is_attack": not str(row.get("gt_label", "")).startswith("normal"),
        })
    frame = pd.DataFrame(scored)
    y = frame["is_attack"].astype(int).to_numpy()
    auc_overall = _auc(frame["u_score"].to_numpy(), y)
    by_label = {}
    for label, group in frame.groupby("gt_label"):
        if str(label).startswith("normal"):
            continue
        # one-vs-normals AUC
        normals = frame[~frame["is_attack"]]
        subset = pd.concat([group, normals], ignore_index=True)
        auc = _auc(subset["u_score"].to_numpy(), subset["is_attack"].astype(int).to_numpy())
        by_label[str(label)] = {
            "auc": auc,
            "n": int(len(group)),
            "root_cause": "feature_or_evidence" if (auc is not None and auc < 0.6) else "alpha_may_help",
        }

    missing_cal = list(manifest.get("missing_cal_cohorts") or [])
    sweep = []
    for alpha in alphas:
        cfg = dict(config)
        cfg["alpha"] = alpha
        cfg["alpha_high_impact"] = alpha
        rows = []
        for _, row in windows.iterrows():
            rows.append(score_window(row.to_dict(), baseline, calibration, cfg))
        labels = frame["gt_label"].tolist()
        normal_rate = np.mean([
            r["is_candidate"] for r, lab in zip(rows, labels) if str(lab).startswith("normal")
        ]) if any(str(lab).startswith("normal") for lab in labels) else None
        attack_recall = np.mean([
            r["is_candidate"] for r, lab in zip(rows, labels) if not str(lab).startswith("normal")
        ]) if any(not str(lab).startswith("normal") for lab in labels) else None
        det = [
            r["is_candidate"]
            for r, vis in zip(rows, frame["visibility_group"].tolist())
            if vis == "detectable"
        ]
        cal_scored = windows.copy()
        cal_scored["is_candidate"] = [r["is_candidate"] for r in rows]
        train_scored = train_windows.copy()
        if not train_scored.empty:
            train_scored["is_candidate"] = [
                score_window(row.to_dict(), baseline, calibration, cfg)["is_candidate"]
                for _, row in train_windows.iterrows()
            ]
        weighted, substituted = cohort_weighted_fp(
            _target_normals(cal_scored),
            train_normals=_target_normals(train_scored) if not train_scored.empty else None,
            missing_cal_cohorts=missing_cal,
        )
        fit_fp = None
        fit_frames = [_target_normals(cal_scored)]
        if not train_scored.empty:
            fit_frames.append(_target_normals(train_scored))
        fit_all = pd.concat(fit_frames, ignore_index=True)
        if not fit_all.empty:
            fit_fp = float(fit_all["is_candidate"].mean())
        sweep.append({
            "alpha": alpha,
            "cal_normal_candidate_rate": float(normal_rate) if normal_rate is not None else None,
            "cal_cohort_weighted_fp": weighted,
            "cal_attack_recall": float(attack_recall) if attack_recall is not None else None,
            "cal_detectable_recall": float(np.mean(det)) if det else None,
            "fit_target_normal_fp": fit_fp,
            "substituted_from_train": substituted,
        })

    selected, selection_failed, rule = select_alpha(sweep, config)
    selected, stepdown = maybe_stepdown_alpha(selected, sweep)
    return {
        "fit_split": "cal",
        "u_score_auc_overall": auc_overall,
        "u_score_auc_by_label": by_label,
        "alpha_sweep": sweep,
        "selection_rule": rule,
        "selection_failed": selection_failed,
        "selected_alpha": selected,
        "alpha_stepdown": stepdown,
        "missing_cal_cohorts": missing_cal,
        "limitation": "Under conformal scoring, FPR often tracks alpha; low-AUC classes need feature/evidence work, not alpha alone.",
        "test_sealed": True,
    }


def select_alpha(sweep: list[dict], config: dict) -> tuple[dict | None, bool, str]:
    del config
    use_weighted = any(s.get("cal_cohort_weighted_fp") is not None for s in sweep)
    key = "cal_cohort_weighted_fp" if use_weighted else "cal_normal_candidate_rate"
    eligible = [
        s for s in sweep
        if s.get(key) is not None and s[key] <= 0.10
    ]
    selection_failed = False
    if eligible:
        selected = max(eligible, key=lambda s: (s["cal_attack_recall"] or 0.0, -s["alpha"]))
    else:
        selected = min(sweep, key=lambda s: (s.get(key) if s.get(key) is not None else 1.0))
    rule = (
        f"{key} <= 0.10, maximize attack recall; tie -> smaller alpha; "
        "alpha_high_impact = alpha (no impact relaxation); "
        "missing cal cohorts substituted from train"
    )
    return selected, selection_failed, rule


def maybe_stepdown_alpha(selected: dict | None, sweep: list[dict]) -> tuple[dict | None, dict]:
    if not selected:
        return selected, {"applied": False, "reason": "no_selection"}
    weighted = selected.get("cal_cohort_weighted_fp")
    fit_fp = selected.get("fit_target_normal_fp")
    if weighted is None or fit_fp is None or weighted <= 0:
        return selected, {"applied": False, "reason": "missing_rates"}
    if fit_fp <= 2.0 * weighted:
        return selected, {
            "applied": False,
            "cal_cohort_weighted_fp": weighted,
            "fit_target_normal_fp": fit_fp,
        }
    lower = [s for s in sweep if s["alpha"] < selected["alpha"]]
    if not lower:
        return selected, {
            "applied": False,
            "reason": "blocked",
            "cal_cohort_weighted_fp": weighted,
            "fit_target_normal_fp": fit_fp,
        }
    nxt = max(lower, key=lambda s: s["alpha"])
    return nxt, {
        "applied": True,
        "from_alpha": selected["alpha"],
        "to_alpha": nxt["alpha"],
        "cal_cohort_weighted_fp": weighted,
        "fit_target_normal_fp": fit_fp,
    }


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
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--auc-out", type=Path, default=ROOT / "dataset/eval/stage1_cal_score_auc.json")
    parser.add_argument("--sweep-out", type=Path, default=ROOT / "dataset/eval/stage1_alpha_sweep.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/synthetic/index/labels.csv")
    parser.add_argument("--sessions-root", type=Path, default=None)
    parser.add_argument("--progress-source", choices=["visible", "raw"], default="visible")
    parser.add_argument("--write-config", action="store_true", default=False)
    args = parser.parse_args()
    report = run(
        args.telemetry,
        args.split_manifest,
        baseline_model=args.baseline_model,
        calibration_path=args.calibration,
        config_path=args.stage1_config,
        progress_log_dir=args.progress_log_dir,
        window_s=args.window_s,
        stride_s=args.stride_s,
        alphas=list(args.alphas),
        labels_csv=args.labels,
        sessions_root=args.sessions_root,
        progress_source=args.progress_source,
    )
    args.auc_out.write_text(json.dumps({
        "u_score_auc_overall": report["u_score_auc_overall"],
        "u_score_auc_by_label": report["u_score_auc_by_label"],
        "limitation": report["limitation"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    args.sweep_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.write_config and report.get("selected_alpha") and not report.get("selection_failed"):
        try:
            import yaml

            cfg_path = args.stage1_config
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            selected = report["selected_alpha"]
            data["alpha"] = float(selected["alpha"])
            data["alpha_high_impact"] = float(selected["alpha"])
            data.setdefault("alpha_selection", {})
            data["alpha_selection"]["selected_on"] = "cal"
            data["alpha_selection"]["rule"] = report["selection_rule"]
            data["alpha_selection"]["selected"] = selected
            cfg_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        except Exception as exc:
            print(f"warning: could not update yaml alpha: {exc}")
    md_path = args.sweep_out.with_suffix(".md")
    md = [
        "# Stage1 alpha sweep (cal only)",
        "",
        f"cal u_score AUC overall: **{report['u_score_auc_overall']}**",
        "",
        f"selected: `{report['selected_alpha']}`",
        "",
        "| alpha | cal_normal_candidate_rate | cal_attack_recall | cal_detectable_recall |",
        "|-------|---------------------------|-------------------|-----------------------|",
    ]
    for row in report["alpha_sweep"]:
        md.append(
            f"| {row['alpha']} | {row['cal_normal_candidate_rate']} | "
            f"{row['cal_attack_recall']} | {row.get('cal_detectable_recall')} |"
        )
    md.extend(
        [
            "",
            "## cal u_score AUC by attack label",
            "",
            "| label | auc | n | root_cause |",
            "|-------|-----|---|------------|",
        ]
    )
    for label, info in sorted(report["u_score_auc_by_label"].items()):
        md.append(f"| {label} | {info.get('auc')} | {info.get('n')} | {info.get('root_cause')} |")
    md.append("")
    md.append(f"Note: {report['limitation']}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "auc_overall": report["u_score_auc_overall"],
        "selected_alpha": report["selected_alpha"],
        "n_low_auc_labels": sum(1 for v in report["u_score_auc_by_label"].values() if v.get("root_cause") == "feature_or_evidence"),
        "md": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
