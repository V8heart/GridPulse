"""Produce denominator-safe stratified Stage 1 metrics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _is_attack(frame: pd.DataFrame) -> pd.Series:
    if "gt_is_attack" in frame:
        return frame["gt_is_attack"].astype(str).str.lower().isin({"true", "1"})
    return ~frame["gt_label"].astype(str).str.startswith("normal")


def _metric(frame: pd.DataFrame, attack: bool) -> dict:
    subset = frame[_is_attack(frame) == attack]
    n_windows = int(len(subset))
    window_hits = int(subset["is_candidate"].astype(bool).sum())
    sessions = (
        subset.groupby("session_id", sort=False)["is_candidate"].any()
        if n_windows
        else pd.Series(dtype=bool)
    )
    n_sessions = int(len(sessions))
    session_hits = int(sessions.sum())
    return {
        "metric": "attack_recall" if attack else "normal_candidate_rate",
        "n_windows": n_windows,
        "n_sessions": n_sessions,
        "window_rate": window_hits / n_windows if n_windows else None,
        "window_wilson95": wilson_interval(window_hits, n_windows),
        "session_rate": session_hits / n_sessions if n_sessions else None,
        "session_wilson95": wilson_interval(session_hits, n_sessions),
        "small_n": bool(n_sessions < 5),
        "flag": "n<5" if n_sessions < 5 else None,
    }


def _scope(frame: pd.DataFrame, *, target_only: bool, include_warmup: bool) -> pd.DataFrame:
    scoped = frame
    if target_only and "gpu_role" in scoped:
        role = scoped["gpu_role"].fillna("target").astype(str)
        scoped = scoped[role == "target"]
    if not include_warmup and "warmup" in scoped:
        warmup = scoped["warmup"].astype(str).str.lower().isin({"true", "1"})
        scoped = scoped[~warmup]
    return scoped


def stratified_report(frame: pd.DataFrame, *, input_scope: str = "all_windows") -> dict:
    required = {"session_id", "is_candidate"}
    missing = required - set(frame.columns)
    reasons = []
    if missing:
        reasons.append(f"missing required columns: {sorted(missing)}")
    if "gt_is_attack" not in frame and "gt_label" not in frame:
        reasons.append("missing ground truth")
    if input_scope != "all_windows":
        reasons.append("survival bias: input is not all_windows")
    if frame.empty:
        reasons.append("empty input")
    if reasons:
        return {"valid": False, "invalid_reasons": reasons, "rows": []}

    work = frame.copy()
    work["is_candidate"] = work["is_candidate"].astype(str).str.lower().isin({"true", "1"})
    views = {
        "primary_target_only": _scope(work, target_only=True, include_warmup=False),
        "secondary_companion_inclusive": _scope(work, target_only=False, include_warmup=False),
        "secondary_warmup_inclusive": _scope(work, target_only=True, include_warmup=True),
    }
    axes = {
        "declared_job_family": "declared_job_family",
        "declared_policy": "declared_policy",
        "progress_log": "progress_log",
        "gt_variant": "gt_variant",
    }
    rows = []
    for view_name, view in views.items():
        for attack in (False, True):
            rows.append({"view": view_name, "axis": "overall", "value": "all", **_metric(view, attack)})
        for axis_name, column in axes.items():
            if column not in view:
                continue
            for value, group in view.fillna({column: "unknown"}).groupby(column, sort=True):
                for attack in (False, True):
                    metric = _metric(group, attack)
                    if metric["n_windows"]:
                        rows.append(
                            {
                                "view": view_name,
                                "axis": axis_name,
                                "value": str(value),
                                **metric,
                            }
                        )
    primary = views["primary_target_only"]
    representative = primary[_is_attack(primary)]
    if "declared_job_family" in representative:
        representative = representative[
            representative["declared_job_family"].astype(str) == "training"
        ]
    if "declared_policy" in representative:
        representative = representative[
            representative["declared_policy"].astype(str) == "host_family_matched"
        ]
    return {
        "valid": True,
        "primary_definition": "target GPU only; warmup excluded",
        "secondary_definitions": [
            "companion-inclusive; warmup excluded",
            "target-only; warmup inclusive",
        ],
        "representative_metric": _metric(representative, True),
        "representative_definition": "training declared + host_family_matched attack recall",
        "rows": rows,
    }


def _read_input(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False), "all_windows"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload), "unknown"
    for key in ("all_windows", "scored_rows", "rows", "windows"):
        if isinstance(payload.get(key), list):
            return pd.DataFrame(payload[key]), (
                "all_windows" if key in {"all_windows", "scored_rows"} else key
            )
    if isinstance(payload.get("candidates"), list):
        return pd.DataFrame(payload["candidates"]), "candidates"
    raise ValueError("JSON does not contain window rows")


def _write_markdown(report: dict, path: Path) -> None:
    lines = ["# Stage 1 stratified report", ""]
    if not report["valid"]:
        lines.extend(["**INVALID**: " + "; ".join(report["invalid_reasons"]), ""])
    else:
        rep = report["representative_metric"]
        lines.extend(
            [
                f"Representative session recall: `{rep['session_rate']}` "
                f"(n={rep['n_sessions']}, 95% CI={rep['session_wilson95']})",
                "",
                "| view | axis | value | metric | windows | sessions | window rate | session rate | flag |",
                "|---|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in report["rows"]:
            lines.append(
                f"| {row['view']} | {row['axis']} | {row['value']} | {row['metric']} | "
                f"{row['n_windows']} | {row['n_sessions']} | {row['window_rate']} | "
                f"{row['session_rate']} | {row['flag'] or ''} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=["cal", "test"], default="cal")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "dataset/eval/v2_regen"
    )
    parser.add_argument(
        "--input-scope",
        choices=["auto", "all_windows", "candidates"],
        default="auto",
    )
    args = parser.parse_args()
    frame, detected_scope = _read_input(args.input)
    scope = detected_scope if args.input_scope == "auto" else args.input_scope
    report = stratified_report(frame, input_scope=scope)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stage1_stratified_{args.split}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(report, stem.with_suffix(".md"))
    print(json.dumps({"valid": report["valid"], "rows": len(report["rows"])}, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
