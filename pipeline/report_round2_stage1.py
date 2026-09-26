"""Round2 Stage1 report. Mid-group fallback, disguise recall, fold_00 comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.declared_context import apply_mid_group_column, mid_group_for_job_type
from pipeline.attack_visibility import attack_visibility_group, parse_period_s
from pipeline.baseline import CohortBaseline
from pipeline.report_stratified import wilson_interval
from pipeline.stage1_v2 import build_windows, load_config, load_truth_index, score_window

TRAIN_SERIES_LABELS = {"normal_llm_pretrain_ddp", "normal_llm_finetune"}
SERVING_LABEL = "normal_llm_inference_serving"
RECALL_EXCLUDE_MID = frozenset({"vision_train"})


def target_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "gpu_role" not in frame.columns:
        return frame
    return frame[frame["gpu_role"].astype(str) == "target"].copy()


def rate_with_ci(flags) -> dict:
    series = pd.Series(list(flags)).astype(bool)
    n = int(len(series))
    k = int(series.sum())
    interval = wilson_interval(k, n) if n else None
    return {
        "n": n,
        "k": k,
        "rate": float(k / n) if n else float("nan"),
        "wilson_95": interval,
    }


def session_recall(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return rate_with_ci([])
    grouped = frame.groupby("session_id")["is_candidate"].any()
    return rate_with_ci(grouped.tolist())


def _fmt_ci(stat: dict) -> str:
    interval = stat.get("wilson_95")
    if not interval:
        return "—"
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _fmt_rate(stat: dict) -> str:
    if stat.get("n", 0) == 0 or pd.isna(stat.get("rate")):
        return "n/a"
    return f"{stat['rate']:.3f}"


def idle_gate_table(frame: pd.DataFrame) -> list[dict]:
    status = frame["idle_status"].fillna("pass") if "idle_status" in frame.columns else pd.Series(["pass"] * len(frame))
    rows = []
    for name, group in frame.groupby(status.astype(str)):
        rows.append({"idle_status": str(name), "n": int(len(group))})
    return rows


def disguise_recall_table(frame: pd.DataFrame) -> list[dict]:
    attack = target_only(frame)
    attack = attack[~attack["gt_label"].astype(str).str.startswith("normal")]
    rows = []
    if attack.empty:
        return rows
    for (job, label), group in attack.groupby(
        [attack["declared_job_type"].astype(str), attack["gt_label"].astype(str)]
    ):
        rows.append(
            {
                "declared_job_type": str(job),
                "gt_label": str(label),
                "window": rate_with_ci(group["is_candidate"]),
                "session": session_recall(group),
            }
        )
    return rows


def _label_mid_counts(labels: pd.DataFrame | None) -> dict[str, dict[str, int]]:
    if labels is None or labels.empty:
        return {}
    frame = labels.copy()
    if "valid" in frame.columns:
        frame = frame[frame["valid"].astype(str).str.lower().isin({"true", "1"})]
    if "gpu_role" in frame.columns:
        frame = frame[frame["gpu_role"].astype(str) == "target"]
    if "declared_job_type" not in frame.columns:
        return {}
    frame["declared_mid_group"] = frame["declared_job_type"].map(mid_group_for_job_type)
    attack = frame["gt_is_attack"].astype(str).str.lower().isin({"true", "1"})
    out: dict[str, dict[str, int]] = {}
    for mid, group in frame.groupby(frame["declared_mid_group"].astype(str)):
        mask = attack.reindex(group.index, fill_value=False)
        out[str(mid)] = {
            "n_normal_sessions": int(group.loc[~mask, "session_id"].nunique()),
            "n_attack_sessions": int(group.loc[mask, "session_id"].nunique()),
        }
    return out


def mid_group_disguise_table(frame: pd.DataFrame, labels: pd.DataFrame | None = None) -> list[dict]:
    target = target_only(frame)
    if "declared_mid_group" not in target.columns:
        target = apply_mid_group_column(target)
    label_counts = _label_mid_counts(labels)
    rows = []
    for mid, group in target.groupby(target["declared_mid_group"].astype(str)):
        normals = group[group["gt_label"].astype(str).str.startswith("normal")]
        attacks = group[~group["gt_label"].astype(str).str.startswith("normal")]
        counts = label_counts.get(str(mid), {})
        row = {
            "mid_group": str(mid),
            "n_normal_sessions": int(counts.get("n_normal_sessions", normals["session_id"].nunique() if not normals.empty else 0)),
            "n_attack_sessions": int(counts.get("n_attack_sessions", attacks["session_id"].nunique() if not attacks.empty else 0)),
            "normal_fp": rate_with_ci(normals["is_candidate"]) if not normals.empty else rate_with_ci([]),
            "normal_fp_session": session_recall(normals) if not normals.empty else rate_with_ci([]),
        }
        if str(mid) in RECALL_EXCLUDE_MID:
            row["attack_recall"] = None
            row["attack_recall_session"] = None
            row["recall_excluded"] = True
            row["note"] = "vision_train has no attacks; FP only"
        else:
            row["attack_recall"] = rate_with_ci(attacks["is_candidate"]) if not attacks.empty else rate_with_ci([])
            row["attack_recall_session"] = session_recall(attacks) if not attacks.empty else rate_with_ci([])
            row["recall_excluded"] = False
            row["note"] = ""
            if str(mid) == "llm_train" and row["n_attack_sessions"] > row["n_normal_sessions"]:
                row["note"] = (
                    "more attacks than normals in this disguise cohort; "
                    "interpret recall against a thin normal baseline"
                )
        rows.append(row)
    return rows


def evidence_firing(frame: pd.DataFrame, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        if key in frame.columns:
            flags = frame[key].astype(bool)
        elif "evidence_bool" in frame.columns:
            flags = frame["evidence_bool"].apply(lambda item: bool((item or {}).get(key)))
        else:
            flags = pd.Series([False] * len(frame))
        out[key] = {
            "rate": float(flags.mean()) if len(flags) else float("nan"),
            "n": int(len(flags)),
        }
    if "u_score" in frame.columns and not frame.empty:
        q = frame["u_score"].quantile([0.25, 0.5, 0.75])
        out["u_score_iqr"] = [float(q.iloc[0]), float(q.iloc[1]), float(q.iloc[2])]
    return out


def crypto_vs_finetune_evidence(frame: pd.DataFrame) -> dict:
    target = target_only(frame)
    keys = ("flat_power", "sustained_high_load", "declared_family_mismatch")
    crypto = target[target["gt_label"].astype(str).str.contains("crypto", case=False, na=False)]
    finetune = target[
        target["declared_job_type"].astype(str).eq("llm_finetune")
        & target["gt_label"].astype(str).str.startswith("normal")
    ]
    return {
        "crypto": evidence_firing(crypto, keys),
        "finetune_target": evidence_firing(finetune, keys),
        "note": "weights and formulas unchanged; reporting only",
    }


def fitted_full_keys(baseline: CohortBaseline) -> list[dict]:
    rows = []
    for key, stat in (baseline.stats or {}).items():
        if not str(key).startswith("full:"):
            continue
        parts = str(key).split(":", 1)[1].split("|")
        mid = parts[1] if len(parts) > 1 else ""
        rows.append(
            {
                "key": key,
                "mid_group": mid,
                "n_sessions": int(stat.get("n_sessions") or 0),
                "n_windows": int(stat.get("n_windows") or stat.get("n") or 0),
                "level": "full",
            }
        )
    return rows


def fallback_by_mid_group(frame: pd.DataFrame, fitted: list[dict]) -> list[dict]:
    target = target_only(frame)
    if "declared_mid_group" not in target.columns:
        target = apply_mid_group_column(target)
    fitted_mids = {row["mid_group"] for row in fitted}
    rows = []
    for mid, group in target.groupby(target["declared_mid_group"].astype(str)):
        levels = group["baseline_level"].astype(str).value_counts().to_dict() if "baseline_level" in group.columns else {}
        fitted_full = str(mid) in fitted_mids
        dominant = max(levels, key=levels.get) if levels else "family"
        rows.append(
            {
                "mid_group": str(mid),
                "fitted_full": fitted_full,
                "score_level": dominant if fitted_full or dominant != "full" else "family",
                "n_windows": int(len(group)),
                "full": int(levels.get("full", 0)),
                "family": int(levels.get("family", 0)),
                "global": int(levels.get("global", 0)),
            }
        )
    return rows


def _pack_comparison(frame: pd.DataFrame) -> dict:
    target = target_only(frame)
    normals = target[target["gt_label"].astype(str).str.startswith("normal")]
    train = normals[normals["gt_label"].astype(str).isin(TRAIN_SERIES_LABELS)]
    serving = normals[normals["gt_label"].astype(str) == SERVING_LABEL]
    det = target[target["visibility_group"].astype(str) == "detectable"]
    unobs = target[target["visibility_group"].astype(str) == "unobservable"]
    return {
        "normal_target": rate_with_ci(normals["is_candidate"]),
        "train_series": rate_with_ci(train["is_candidate"]),
        "serving": rate_with_ci(serving["is_candidate"]),
        "detectable": rate_with_ci(det["is_candidate"]),
        "unobservable": rate_with_ci(unobs["is_candidate"]),
    }


def published_round1(path: Path, window_n: dict[str, int] | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    real = data.get("real") or {}
    by_label = real.get("by_label") or {}
    n_ddp = (window_n or {}).get("normal_llm_pretrain_ddp", 1)
    n_ft = (window_n or {}).get("normal_llm_finetune", 1)
    ddp = float(by_label.get("normal_llm_pretrain_ddp", float("nan")))
    finetune = float(by_label.get("normal_llm_finetune", float("nan")))
    train_rate = (ddp * n_ddp + finetune * n_ft) / (n_ddp + n_ft) if n_ddp + n_ft else float("nan")
    return {
        "normal_target": {"rate": None, "n": 0, "k": 0, "wilson_95": None, "note": "see envelope"},
        "train_series": {"rate": train_rate, "n": n_ddp + n_ft, "k": None, "wilson_95": None},
        "serving": {"rate": float(by_label.get(SERVING_LABEL, float("nan"))), "n": (window_n or {}).get(SERVING_LABEL, 0), "k": None, "wilson_95": None},
        "detectable": {"rate": float(real.get("detectable_recall", float("nan"))), "n": int(real.get("n_detectable") or 0), "k": None, "wilson_95": None},
        "unobservable": {"rate": float(real.get("unobservable_recall", float("nan"))), "n": int(real.get("n_unobservable") or 0), "k": None, "wilson_95": None},
        "ddp": ddp,
        "finetune": finetune,
    }


def comparison_rows(round1: dict, round2: pd.DataFrame) -> list[dict]:
    left = dict(round1)
    envelope = left.pop("_envelope", None)
    if envelope is not None and not envelope.empty:
        packed = _pack_comparison(envelope)
        if left.get("normal_target", {}).get("rate") is None:
            left["normal_target"] = packed["normal_target"]
    right = _pack_comparison(round2)
    names = [
        ("전체 정상 후보율 (target GPU)", "normal_target"),
        ("학습 계열 정상 후보율 (DDP / finetune)", "train_series"),
        ("serving 후보율", "serving"),
        ("detectable recall", "detectable"),
        ("unobservable recall", "unobservable"),
    ]
    rows = []
    for title, key in names:
        rows.append({"metric": title, "round1": left[key], "round2": right[key]})
    return rows


def _score_frame(windows: pd.DataFrame, baseline: CohortBaseline, calibration: dict, config: dict) -> pd.DataFrame:
    rows = []
    for _, row in windows.iterrows():
        data = row.to_dict()
        out = score_window(data, baseline, calibration, config)
        vis = data.get("visibility_group") or attack_visibility_group(
            gt_label=str(data.get("gt_label", "")),
            gt_variant=data.get("gt_variant"),
            period_s=data.get("period_s"),
        )
        rows.append(
            {
                **{k: data.get(k) for k in (
                    "session_id",
                    "gpu_id",
                    "gpu_role",
                    "gt_label",
                    "gt_variant",
                    "declared_job_family",
                    "declared_job_type",
                    "declared_mid_group",
                    "mean_w",
                )},
                "u_score": out["u_score"],
                "is_candidate": out["is_candidate"],
                "idle_status": out.get("idle_status") or "pass",
                "baseline_level": out.get("baseline_level"),
                "visibility_group": vis,
                "evidence_bool": out.get("evidence_bool") or {},
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and "declared_mid_group" not in frame.columns:
        frame = apply_mid_group_column(frame)
    elif not frame.empty:
        missing = frame["declared_mid_group"].isna() if "declared_mid_group" in frame.columns else True
        if isinstance(missing, bool) or (hasattr(missing, "any") and missing.any()):
            frame["declared_mid_group"] = frame["declared_job_type"].map(mid_group_for_job_type)
    return frame


def _round1_from_synth(path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a compact scored frame from the published synth_vs_real rates is not enough.
    Join fold_00 envelope windows to labels when the envelope exists; else empty.
    """
    envelope = ROOT / "dataset/real/pipeline/stage1_real_refit.json"
    if not envelope.exists():
        return pd.DataFrame()
    data = json.loads(envelope.read_text(encoding="utf-8"))
    windows = pd.DataFrame(data.get("all_windows") or [])
    if windows.empty:
        return windows
    key = labels[["session_id", "gpu_id", "gpu_role", "gt_label", "gt_variant", "gt_params_json", "declared_job_type"]].copy()
    key["session_id"] = key["session_id"].astype(str)
    key["gpu_id"] = pd.to_numeric(key["gpu_id"], errors="coerce")
    windows["session_id"] = windows["session_id"].astype(str)
    windows["gpu_id"] = pd.to_numeric(windows["gpu_id"], errors="coerce")
    merged = windows.merge(key, on=["session_id", "gpu_id"], how="left")
    merged["period_s"] = merged.get("gt_params_json", pd.Series([None] * len(merged))).map(parse_period_s)
    merged["visibility_group"] = [
        attack_visibility_group(gt_label=str(lab), gt_variant=var, period_s=per)
        for lab, var, per in zip(merged.get("gt_label", []), merged.get("gt_variant", []), merged["period_s"])
    ]
    if "declared_job_type" in merged.columns:
        merged["declared_mid_group"] = merged["declared_job_type"].map(mid_group_for_job_type)
    return merged


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_evidence_md(evidence: dict, path: Path) -> None:
    def pack(name: str, blob: dict) -> list[str]:
        iqr = blob.get("u_score_iqr") or [float("nan")] * 3
        return [
            name,
            f"{blob.get('flat_power', {}).get('rate', float('nan')):.3f}",
            f"{blob.get('sustained_high_load', {}).get('rate', float('nan')):.3f}",
            f"{blob.get('declared_family_mismatch', {}).get('rate', float('nan')):.3f}",
            f"{iqr[0]:.3f} / {iqr[1]:.3f} / {iqr[2]:.3f}",
        ]

    lines = [
        "# Crypto vs finetune evidence (round2, target)",
        "",
        evidence.get("note", ""),
        "",
        *_md_table(
            ["group", "flat_power", "sustained_high_load", "declared_family_mismatch", "u_score IQR"],
            [pack("crypto", evidence["crypto"]), pack("finetune target", evidence["finetune_target"])],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(report: dict, path: Path) -> None:
    cmp_rows = [
        [
            row["metric"],
            _fmt_rate(row["round1"]),
            _fmt_rate(row["round2"]),
        ]
        for row in report["comparison"]
    ]
    fallback_rows = [
        [
            row["mid_group"],
            "full" if row["fitted_full"] else "family/global",
            str(row["full"]),
            str(row["family"]),
            str(row["global"]),
            str(row["n_windows"]),
        ]
        for row in report["fallback_by_mid_group"]
    ]
    fitted_rows = [
        [row["mid_group"], row["key"], str(row["n_sessions"]), str(row["n_windows"])]
        for row in report["fitted_full_keys"]
    ] or [["—", "none", "0", "0"]]
    disguise_job = [
        [
            row["declared_job_type"],
            row["gt_label"],
            _fmt_rate(row["window"]),
            _fmt_ci(row["window"]),
            _fmt_rate(row["session"]),
            _fmt_ci(row["session"]),
        ]
        for row in report["disguise_job_type"]
    ]
    disguise_mid = []
    for row in report["disguise_mid_group"]:
        recall = row.get("attack_recall")
        sess = row.get("attack_recall_session")
        disguise_mid.append(
            [
                row["mid_group"],
                str(row["n_normal_sessions"]),
                str(row["n_attack_sessions"]),
                _fmt_rate(row["normal_fp"]),
                _fmt_rate(recall) if recall is not None else "excluded",
                _fmt_rate(sess) if sess is not None else "excluded",
                row.get("note") or "",
            ]
        )
    workload = [
        [row["label"], str(row["n"]), f"{row['rate']:.3f}"]
        for row in report["workload_rates"]
    ]
    idle = [[row["idle_status"], str(row["n"])] for row in report["idle_gate"]]
    lines = [
        "# Stage1 real refit round2",
        "",
        "Decision: mid-group cohort keys, `min_sessions=3`. "
        "Thin groups (inference_online / inference_batch / interactive) keep family fallback; "
        "full keys are not forced.",
        "",
        "이번 라운드 `alpha_high_impact = alpha` (impact 완화 미적용).",
        "",
        f"selected alpha: `{report['selected_alpha']}`",
        f"missing_train_job_types: `{report['missing_train_job_types']}`",
        "",
        "## Round1 fold_00 vs round2 (target GPU)",
        "",
        "Round1 recall/serving/DDP/finetune are the published fold_00 table "
        f"(DDP {report.get('round1_ddp')}, finetune {report.get('round1_finetune')}). "
        "학습 계열 is the window-weighted mix of those two labels.",
        "",
        *_md_table(["metric", "round1", "round2"], cmp_rows),
        "",
        "## Fitted full keys",
        "",
        *_md_table(["mid_group", "key", "n_sessions", "n_windows"], fitted_rows),
        "",
        "## Mid-group fallback (full / family / global)",
        "",
        *_md_table(["mid_group", "fitted", "full", "family", "global", "n"], fallback_rows),
        "",
        "## Train / cal cohort counts",
        "",
        f"train: `{report['train_counts']}`",
        f"cal: `{report['cal_counts']}`",
        "",
        "## Workload candidate rates (target GPU)",
        "",
        *_md_table(["gt_label", "n", "candidate rate"], workload),
        "",
        "## Idle gate",
        "",
        *_md_table(["status", "n"], idle),
        "",
        "## Recall (target)",
        "",
        f"detectable window: {_fmt_rate(report['detectable_window'])} {_fmt_ci(report['detectable_window'])}",
        f"unobservable window: {_fmt_rate(report['unobservable_window'])} {_fmt_ci(report['unobservable_window'])}",
        f"detectable session: {_fmt_rate(report['detectable_session'])} {_fmt_ci(report['detectable_session'])}",
        f"unobservable session: {_fmt_rate(report['unobservable_session'])} {_fmt_ci(report['unobservable_session'])}",
        "",
        "## Alpha sweep (cal only; 0.70 unused)",
        "",
        *_md_table(
            ["alpha", "normal cand", "attack recall", "detectable recall"],
            [
                [
                    f"{row['alpha']:.2f}",
                    f"{row['cal_normal_candidate_rate']}",
                    f"{row['cal_attack_recall']}",
                    f"{row.get('cal_detectable_recall')}",
                ]
                for row in report["alpha_sweep"]
            ],
        ),
        "",
        "## Declared remap summary",
        "",
        report.get("remap_summary", ""),
        "",
        "## Attack disguise by declared_job_type",
        "",
        *_md_table(
            ["declared_job_type", "gt_label", "window recall", "Wilson", "session recall", "Wilson"],
            disguise_job,
        ),
        "",
        "## Attack disguise by mid_group",
        "",
        "vision_train is FP-only and excluded from recall. "
        "llm_train has more attacks than normals; interpret recall cautiously.",
        "",
        *_md_table(
            ["mid_group", "n_normal sess", "n_attack sess", "normal FP", "attack recall", "session recall", "note"],
            disguise_mid,
        ),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    config = load_config(args.stage1_config)
    baseline = CohortBaseline.load(args.baseline)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("evidence_surprisal_weights"):
        config = dict(config)
        config["evidence_weights"] = {
            **config.get("evidence_weights", {}),
            **calibration["evidence_surprisal_weights"],
        }
    if calibration.get("evidence_thresholds"):
        config["evidence_thresholds"] = calibration["evidence_thresholds"]
    labels = pd.read_csv(args.labels, low_memory=False)
    truth = load_truth_index(args.labels, sessions_root=args.sessions_root)
    telemetry = pd.read_csv(args.telemetry, low_memory=False)
    windows = build_windows(
        telemetry,
        window_s=args.window_s,
        stride_s=args.stride_s,
        config=config,
        sessions_root=args.sessions_root,
        truth_index=truth,
        progress_source=args.progress_source,
    )
    scored = _score_frame(windows, baseline, calibration, config)
    target = target_only(scored)
    normals = target[target["gt_label"].astype(str).str.startswith("normal")]
    attacks = target[~target["gt_label"].astype(str).str.startswith("normal")]
    det = attacks[attacks["visibility_group"].astype(str) == "detectable"]
    unobs = attacks[attacks["visibility_group"].astype(str) == "unobservable"]
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    sweep = json.loads(Path(args.alpha_sweep).read_text(encoding="utf-8")) if args.alpha_sweep.exists() else {}
    remap = pd.read_csv(args.remap) if args.remap.exists() else pd.DataFrame()
    changed = remap[remap["changed"] == True] if not remap.empty and "changed" in remap.columns else pd.DataFrame()
    remap_summary = (
        f"remap rows {len(remap)}; honest changed {int(len(changed))}; "
        f"attack-policy changed 0."
    )
    if not changed.empty:
        pairs = changed.groupby(["before", "after"]).size().to_dict()
        remap_summary += f" changed pairs: `{pairs}`"
    workload = []
    for label, group in target.groupby(target["gt_label"].astype(str)):
        workload.append({"label": str(label), "n": int(len(group)), "rate": float(group["is_candidate"].mean())})
    evidence = crypto_vs_finetune_evidence(scored)
    write_evidence_md(evidence, args.evidence_md)
    fitted = fitted_full_keys(baseline)
    window_n = {row["label"]: row["n"] for row in workload}
    round1 = published_round1(args.round1_json, window_n)
    envelope = _round1_from_synth(args.round1_json, labels)
    if not envelope.empty:
        packed = _pack_comparison(envelope)
        round1["normal_target"] = packed["normal_target"]
    report = {
        "decision": "mid_group",
        "min_sessions": 3,
        "alpha_high_impact_equals_alpha": True,
        "selected_alpha": sweep.get("selected_alpha"),
        "alpha_sweep": sweep.get("alpha_sweep") or [],
        "missing_train_job_types": manifest.get("missing_train_job_types") or [],
        "train_job_types": manifest.get("train_job_types") or [],
        "train_counts": cohort_counts(target, manifest.get("train") or [], ("declared_mid_group",)),
        "cal_counts": cohort_counts(target, manifest.get("cal") or [], ("declared_mid_group",)),
        "fitted_full_keys": fitted,
        "fallback_by_mid_group": fallback_by_mid_group(scored, fitted),
        "fallback_counts": dict(baseline.fallback_counts),
        "workload_rates": workload,
        "idle_gate": idle_gate_table(scored),
        "detectable_window": rate_with_ci(det["is_candidate"]),
        "unobservable_window": rate_with_ci(unobs["is_candidate"]),
        "detectable_session": session_recall(det),
        "unobservable_session": session_recall(unobs),
        "disguise_job_type": disguise_recall_table(scored),
        "disguise_mid_group": mid_group_disguise_table(scored, labels),
        "comparison": comparison_rows(round1, scored),
        "round1_ddp": round1.get("ddp"),
        "round1_finetune": round1.get("finetune"),
        "remap_summary": remap_summary,
        "evidence_crypto_vs_finetune": evidence,
        "n_windows": int(len(scored)),
        "n_target": int(len(target)),
        "normal_target_rate": rate_with_ci(normals["is_candidate"]),
        "attack_recall_target": rate_with_ci(attacks["is_candidate"]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(report, args.out_md)
    return report


def cohort_counts(windows: pd.DataFrame, session_ids: list[str], keys: tuple[str, ...] = ("declared_job_type",)) -> dict:
    subset = windows[windows["session_id"].astype(str).isin(set(map(str, session_ids)))]
    if subset.empty:
        return {}
    present = [k for k in keys if k in subset.columns]
    if not present:
        return {"all": {"n_sessions": int(subset["session_id"].nunique()), "n_windows": int(len(subset))}}
    by = {}
    for name, group in subset.groupby([subset[k].astype(str) for k in present]):
        key = name if isinstance(name, str) else "|".join(map(str, name))
        by[str(key)] = {
            "n_sessions": int(group["session_id"].nunique()),
            "n_windows": int(len(group)),
        }
    return by


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/real/pipeline/telemetry_with_procs.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/real/index/labels.csv")
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "dataset/real/sessions")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/fold_00/split_manifest.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/fold_00/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/fold_00/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/stage1_v2.yaml")
    parser.add_argument("--alpha-sweep", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/fold_00/stage1_alpha_sweep.json")
    parser.add_argument("--remap", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/declared_remap.csv")
    parser.add_argument("--round1-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/synth_vs_real.json")
    parser.add_argument("--progress-source", default="raw")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--out-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/stage1_report.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/stage1_report.md")
    parser.add_argument("--evidence-md", type=Path, default=ROOT / "dataset/real/pipeline/refit/round2/evidence_crypto_vs_finetune.md")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "n_windows": report["n_windows"],
        "selected_alpha": report["selected_alpha"],
        "md": str(args.out_md),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
