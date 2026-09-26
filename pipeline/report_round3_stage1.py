"""Round3 Stage1 report: expected_band fallback, weighted alpha, idle unknown."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.declared_context import apply_expected_band_column, expected_band_for_job_type
from pipeline.attack_visibility import attack_visibility_group
from pipeline.baseline import CohortBaseline
from pipeline.report_round2_stage1 import (
    _fmt_ci,
    _fmt_rate,
    _md_table,
    _pack_comparison,
    published_round1,
    rate_with_ci,
    session_recall,
    target_only,
)

_CMP_METRICS = [
    ("전체 정상 후보율 (target GPU)", "normal_target"),
    ("학습 계열 정상 후보율 (DDP / finetune)", "train_series"),
    ("serving 후보율", "serving"),
    ("high-band 위장 recall", "high_band"),
    ("low-band 위장 recall", "low_band"),
]


def _cmp_side(rows: list[dict], metric: str, side: str) -> dict:
    empty = {"rate": float("nan"), "n": 0, "k": 0, "wilson_95": None}
    for row in rows or []:
        if row.get("metric") == metric:
            return row.get(side) or empty
    return empty


def _pool_band_recall(disguise_rows: list[dict], band: str) -> dict:
    n = 0
    k = 0
    for row in disguise_rows or []:
        if expected_band_for_job_type(row.get("declared_job_type")) != band:
            continue
        win = row.get("window") or {}
        n += int(win.get("n") or 0)
        k += int(win.get("k") or 0)
    if not n:
        return {"rate": float("nan"), "n": 0, "k": 0, "wilson_95": None}
    return rate_with_ci([True] * k + [False] * (n - k))
from pipeline.stage1_v2 import build_windows, load_config, load_truth_index, score_window


def _score(windows, baseline, calibration, config) -> pd.DataFrame:
    rows = []
    for _, row in windows.iterrows():
        data = row.to_dict()
        out = score_window(data, baseline, calibration, config)
        rows.append({
            "session_id": data.get("session_id"),
            "gpu_id": data.get("gpu_id"),
            "gpu_role": data.get("gpu_role"),
            "gt_label": data.get("gt_label"),
            "declared_job_type": data.get("declared_job_type"),
            "declared_job_family": data.get("declared_job_family"),
            "expected_band": data.get("expected_band") or expected_band_for_job_type(data.get("declared_job_type")),
            "observed_n_procs": data.get("observed_n_procs"),
            "mean_w": data.get("mean_w"),
            "u_score": out["u_score"],
            "is_candidate": out["is_candidate"],
            "idle_status": out.get("idle_status") or "pass",
            "baseline_level": out.get("baseline_level"),
            "visibility_group": data.get("visibility_group") or attack_visibility_group(
                gt_label=str(data.get("gt_label", "")),
                gt_variant=data.get("gt_variant"),
                period_s=data.get("period_s"),
            ),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty and "expected_band" not in frame.columns:
        frame = apply_expected_band_column(frame)
    return frame


def _idle_nprocs_cause(idle_like: pd.DataFrame) -> dict:
    n = pd.to_numeric(idle_like.get("observed_n_procs"), errors="coerce") if not idle_like.empty else pd.Series(dtype=float)
    return {
        "n": int(len(idle_like)),
        "n_missing": int(n.isna().sum()) if len(n) else 0,
        "n_zero": int((n == 0).sum()) if len(n) else 0,
        "n_positive": int((n > 0).sum()) if len(n) else 0,
        "candidate_rate": float(idle_like["is_candidate"].mean()) if not idle_like.empty else float("nan"),
    }


def write_md(report: dict, path: Path) -> None:
    cmp_rows = [
        [row["metric"], _fmt_rate(row["round1"]), _fmt_rate(row["round2"]), _fmt_rate(row["round3"])]
        for row in report["comparison"]
    ]
    fallback = [
        [r["cohort"], str(r["full"]), str(r["band"]), str(r["family"]), str(r["global"]), str(r["n"])]
        for r in report["fallback_by_cohort"]
    ]
    fitted = [
        [r["level"], r["key"], str(r["n_sessions"]), str(r["n_windows"])]
        for r in report["fitted_keys"]
    ] or [["—", "none", "0", "0"]]
    cohort_fp = [
        [r["cohort"], str(r["n_sessions"]), str(r["n_windows"]), f"{r['fp']:.3f}"]
        for r in report["cohort_fp"]
    ]
    workload = [[r["label"], str(r["n"]), f"{r['rate']:.3f}"] for r in report["workload_rates"]]
    idle = [[r["idle_status"], str(r["n"])] for r in report["idle_gate"]]
    disguise = [
        [r["band"], str(r["n"]), _fmt_rate(r["recall"]), _fmt_ci(r["recall"]), r.get("note") or ""]
        for r in report["disguise_band"]
    ]
    sweep = [
        [
            f"{r['alpha']:.2f}",
            f"{r.get('cal_normal_candidate_rate')}",
            f"{r.get('cal_cohort_weighted_fp')}",
            f"{r.get('cal_attack_recall')}",
            f"{r.get('fit_target_normal_fp')}",
        ]
        for r in report["alpha_sweep"]
    ]
    cause = report["normal_idle_nprocs"]
    lines = [
        "# Stage1 real refit round3",
        "",
        "expected_band keys, fallback full → band → family → global. "
        "`alpha_high_impact = alpha`. Session recall is appendix-only.",
        "",
        f"selected: `{report['selected_alpha']}`",
        f"alpha_stepdown: `{report['alpha_stepdown']}`",
        f"missing_cal_cohorts: `{report['missing_cal_cohorts']}`",
        f"substituted_from_train: `{report['substituted_from_train']}`",
        "",
        f"cal_cohort_weighted_fp: {report['cal_cohort_weighted_fp']}",
        f"fit_target_normal_fp (train+cal): {report['fit_target_normal_fp']}",
        f"all_target_normal_fp: {report['all_target_normal_fp']}",
        "",
        "## Round1 / round2 / round3 (target GPU)",
        "",
        *_md_table(["metric", "round1", "round2", "round3"], cmp_rows),
        "",
        "## Fitted keys",
        "",
        *_md_table(["level", "key", "n_sessions", "n_windows"], fitted),
        "",
        "## Fallback distribution",
        "",
        *_md_table(["cohort", "full", "band", "family", "global", "n"], fallback),
        "",
        "## Cohort target-normal FP",
        "",
        *_md_table(["cohort", "n_sessions", "n_windows", "fp"], cohort_fp),
        "",
        f"train counts (target normals): `{report['train_counts']}`",
        f"cal counts (target normals): `{report['cal_counts']}`",
        "",
        "## Workload candidate rates (target)",
        "",
        *_md_table(["gt_label", "n", "rate"], workload),
        "",
        "## Idle gate",
        "",
        *_md_table(["status", "n"], idle),
        "",
        f"normal_idle target windows: n={cause['n']} missing={cause['n_missing']} "
        f"zero={cause['n_zero']} positive={cause['n_positive']} cand={cause['candidate_rate']}",
        "",
        "## Recall (window, target)",
        "",
        f"high-band disguise: {_fmt_rate(report['high_band_recall'])} {_fmt_ci(report['high_band_recall'])}",
        f"low-band disguise: {_fmt_rate(report['low_band_recall'])} {_fmt_ci(report['low_band_recall'])}",
        f"detectable: {_fmt_rate(report['detectable'])} {_fmt_ci(report['detectable'])}",
        f"unobservable: {_fmt_rate(report['unobservable'])} {_fmt_ci(report['unobservable'])}",
        "",
        "## Disguise by expected_band",
        "",
        *_md_table(["band", "n", "recall", "Wilson", "note"], disguise),
        "",
        "## Alpha sweep",
        "",
        *_md_table(["alpha", "cal window fp", "cal weighted fp", "attack recall", "fit target fp"], sweep),
        "",
        "## Appendix: session recall (any window hit = 1)",
        "",
        f"high-band session: {_fmt_rate(report['high_band_session'])}",
        f"low-band session: {_fmt_rate(report['low_band_session'])}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    config = load_config(args.stage1_config)
    baseline = CohortBaseline.load(args.baseline)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("evidence_surprisal_weights"):
        config = dict(config)
        config["evidence_weights"] = {**config.get("evidence_weights", {}), **calibration["evidence_surprisal_weights"]}
    if calibration.get("evidence_thresholds"):
        config["evidence_thresholds"] = calibration["evidence_thresholds"]
    labels = pd.read_csv(args.labels, low_memory=False)
    truth = load_truth_index(args.labels, sessions_root=args.sessions_root)
    telemetry = pd.read_csv(args.telemetry, low_memory=False)
    windows = build_windows(
        telemetry, window_s=args.window_s, stride_s=args.stride_s, config=config,
        sessions_root=args.sessions_root, truth_index=truth, progress_source=args.progress_source,
    )
    scored = _score(windows, baseline, calibration, config)
    target = target_only(scored)
    normals = target[target["gt_label"].astype(str).str.startswith("normal")]
    attacks = target[~target["gt_label"].astype(str).str.startswith("normal")]
    high = attacks[attacks["expected_band"].astype(str).eq("high")]
    low = attacks[attacks["expected_band"].astype(str).eq("low")]
    mid = target[target["expected_band"].astype(str).eq("mid") & target["gt_label"].astype(str).str.startswith("normal")]
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    sweep = json.loads(Path(args.alpha_sweep).read_text(encoding="utf-8")) if args.alpha_sweep.exists() else {}
    selected = sweep.get("selected_alpha") or {}
    fit_ids = set(map(str, (manifest.get("train") or []) + (manifest.get("cal") or [])))
    fit_normals = normals[normals["session_id"].astype(str).isin(fit_ids)]
    fitted = []
    for key, stat in (baseline.stats or {}).items():
        level = "global" if key == "global" else str(key).split(":", 1)[0]
        fitted.append({
            "level": level,
            "key": key,
            "n_sessions": int(stat.get("n_sessions") or 0),
            "n_windows": int(stat.get("n_windows") or stat.get("n") or 0),
        })
    fallback_rows = []
    for cohort, group in target.groupby(target["declared_job_family"].astype(str) + "|" + target["expected_band"].astype(str)):
        levels = group["baseline_level"].astype(str).value_counts().to_dict()
        fallback_rows.append({
            "cohort": str(cohort),
            "full": int(levels.get("full", 0)),
            "band": int(levels.get("band", 0)),
            "family": int(levels.get("family", 0)),
            "global": int(levels.get("global", 0)),
            "n": int(len(group)),
        })
    cohort_fp = []
    for cohort, group in normals.groupby(normals["declared_job_family"].astype(str) + "|" + normals["expected_band"].astype(str)):
        cohort_fp.append({
            "cohort": str(cohort),
            "n_sessions": int(group["session_id"].nunique()),
            "n_windows": int(len(group)),
            "fp": float(group["is_candidate"].mean()),
        })
    workload = [
        {"label": str(lab), "n": int(len(group)), "rate": float(group["is_candidate"].mean())}
        for lab, group in target.groupby(target["gt_label"].astype(str))
    ]
    idle_counts = scored["idle_status"].fillna("pass").astype(str).value_counts().to_dict()
    idle_gate = [{"idle_status": name, "n": int(idle_counts.get(name, 0))} for name in ("idle", "undeclared_load", "unknown", "pass")]
    window_n = {row["label"]: row["n"] for row in workload}
    round1 = published_round1(args.round1_json, window_n)
    round2 = json.loads(Path(args.round2_json).read_text(encoding="utf-8")) if args.round2_json.exists() else {}
    r2_cmp = round2.get("comparison") or []
    r2_pack = {
        "normal_target": _cmp_side(r2_cmp, "전체 정상 후보율 (target GPU)", "round2") or (round2.get("normal_target_rate") or {}),
        "train_series": _cmp_side(r2_cmp, "학습 계열 정상 후보율 (DDP / finetune)", "round2"),
        "serving": _cmp_side(r2_cmp, "serving 후보율", "round2"),
        "high_band": _pool_band_recall(round2.get("disguise_job_type") or [], "high"),
        "low_band": _pool_band_recall(round2.get("disguise_job_type") or [], "low"),
    }
    r1_from_cmp = {
        "normal_target": _cmp_side(r2_cmp, "전체 정상 후보율 (target GPU)", "round1") or round1.get("normal_target"),
        "train_series": _cmp_side(r2_cmp, "학습 계열 정상 후보율 (DDP / finetune)", "round1") or round1.get("train_series"),
        "serving": _cmp_side(r2_cmp, "serving 후보율", "round1") or round1.get("serving"),
        # remapping-era job_type is not expected_band; keep n/a
        "high_band": {"rate": float("nan"), "n": 0, "k": 0, "wilson_95": None},
        "low_band": {"rate": float("nan"), "n": 0, "k": 0, "wilson_95": None},
    }
    r3 = _pack_comparison(scored)
    r3["high_band"] = rate_with_ci(high["is_candidate"])
    r3["low_band"] = rate_with_ci(low["is_candidate"])
    comparison = [
        {"metric": title, "round1": r1_from_cmp[key], "round2": r2_pack[key], "round3": r3[key]}
        for title, key in _CMP_METRICS
    ]
    def _counts(ids):
        subset = normals[normals["session_id"].astype(str).isin(set(map(str, ids)))]
        out = {}
        if subset.empty:
            return out
        for name, group in subset.groupby(subset["declared_job_family"].astype(str) + "|" + subset["expected_band"].astype(str)):
            out[str(name)] = {"n_sessions": int(group["session_id"].nunique()), "n_windows": int(len(group))}
        return out

    report = {
        "selected_alpha": selected,
        "alpha_stepdown": sweep.get("alpha_stepdown"),
        "alpha_sweep": sweep.get("alpha_sweep") or [],
        "missing_cal_cohorts": manifest.get("missing_cal_cohorts") or sweep.get("missing_cal_cohorts") or [],
        "substituted_from_train": selected.get("substituted_from_train") or [],
        "cal_cohort_weighted_fp": selected.get("cal_cohort_weighted_fp"),
        "fit_target_normal_fp": float(fit_normals["is_candidate"].mean()) if not fit_normals.empty else None,
        "all_target_normal_fp": float(normals["is_candidate"].mean()) if not normals.empty else None,
        "comparison": comparison,
        "fitted_keys": fitted,
        "fallback_by_cohort": fallback_rows,
        "cohort_fp": cohort_fp,
        "train_counts": _counts(manifest.get("train") or []),
        "cal_counts": _counts(manifest.get("cal") or []),
        "workload_rates": workload,
        "idle_gate": idle_gate,
        "normal_idle_nprocs": _idle_nprocs_cause(target[target["gt_label"].astype(str).eq("normal_idle")]),
        "high_band_recall": rate_with_ci(high["is_candidate"]),
        "low_band_recall": rate_with_ci(low["is_candidate"]),
        "detectable": rate_with_ci(attacks[attacks["visibility_group"].astype(str).eq("detectable")]["is_candidate"]),
        "unobservable": rate_with_ci(attacks[attacks["visibility_group"].astype(str).eq("unobservable")]["is_candidate"]),
        "disguise_band": [
            {"band": "high", "n": int(len(high)), "recall": rate_with_ci(high["is_candidate"]), "note": "representative"},
            {"band": "low", "n": int(len(low)), "recall": rate_with_ci(low["is_candidate"]), "note": "separate row"},
            {"band": "mid", "n": int(len(mid)), "recall": rate_with_ci(mid["is_candidate"]), "note": "no attacks; FP only"},
        ],
        "high_band_session": session_recall(high),
        "low_band_session": session_recall(low),
        "n_windows": int(len(scored)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_md(report, args.out_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/real/pipeline/telemetry_with_procs.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/real/index/labels.csv")
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "dataset/real/sessions")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/fold_00/split_manifest.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/fold_00/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/fold_00/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/stage1_v2.yaml")
    parser.add_argument("--alpha-sweep", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/fold_00/stage1_alpha_sweep.json")
    parser.add_argument("--round1-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/synth_vs_real.json")
    parser.add_argument("--round2-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/stage1_report.json")
    parser.add_argument("--progress-source", default="raw")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--out-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/stage1_report.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "dataset/real/pipeline/refit/round3/stage1_report.md")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"n_windows": report["n_windows"], "selected": report["selected_alpha"], "md": str(args.out_md)}, indent=2, default=str))


if __name__ == "__main__":
    main()
