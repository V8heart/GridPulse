"""Count evidence bool firings for E1 wiring verification (train+cal, not test-tuned)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.baseline import CohortBaseline
from pipeline.evidence_vocab import ALLOWED_EVIDENCE_NAMES
from pipeline.stage1_v2 import build_windows, load_config, score_window


def _count_from_pipeline_results(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", [])
    counts: Counter[str] = Counter()
    for row in rows:
        evidence = (
            (row.get("stage2_evidence_bundle") or {})
            .get("unexplainedness", {})
            .get("evidence")
        )
        if not isinstance(evidence, dict):
            evidence = row.get("evidence") or {}
        for name in ALLOWED_EVIDENCE_NAMES:
            if evidence.get(name) is True:
                counts[name] += 1
    return dict(counts)


def _count_from_stage1(
    telemetry: Path,
    split_manifest: Path,
    *,
    splits: list[str],
    baseline_model: Path,
    calibration: Path,
    config_path: Path,
    progress_log_dir: Path,
) -> tuple[dict[str, int], dict[str, dict[str, float]], int]:
    df = pd.read_csv(telemetry, low_memory=False)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    session_ids: set[str] = set()
    for split in splits:
        session_ids |= set(map(str, manifest.get(split, [])))
    subset = df[df["session_id"].astype(str).isin(session_ids)]
    config = load_config(config_path)
    baseline = CohortBaseline.load(baseline_model)
    cal = json.loads(calibration.read_text(encoding="utf-8"))
    windows = build_windows(
        subset,
        window_s=30.0,
        stride_s=15.0,
        progress_log_dir=progress_log_dir,
        config=config,
    )
    counts: Counter[str] = Counter()
    normal_true: Counter[str] = Counter()
    attack_true: Counter[str] = Counter()
    n_normal = n_attack = 0
    for _, row in windows.iterrows():
        scored = score_window(row.to_dict(), baseline, cal, config)
        evidence = scored.get("evidence_bool") or {}
        is_normal = str(row["gt_label"]).startswith("normal")
        if is_normal:
            n_normal += 1
        else:
            n_attack += 1
        for name in ALLOWED_EVIDENCE_NAMES:
            if evidence.get(name) is True:
                counts[name] += 1
                if is_normal:
                    normal_true[name] += 1
                else:
                    attack_true[name] += 1
    rates = {}
    for name in sorted(ALLOWED_EVIDENCE_NAMES):
        rates[name] = {
            "count": int(counts.get(name, 0)),
            "normal_rate": (normal_true[name] / n_normal) if n_normal else 0.0,
            "attack_rate": (attack_true[name] / n_attack) if n_attack else 0.0,
        }
    return dict(counts), rates, int(len(windows))


def write_markdown(
    out: Path,
    *,
    before: dict[str, int] | None,
    after_e1: dict[str, int] | None,
    after_e2: dict[str, int] | None,
    rates: dict[str, dict[str, float]] | None = None,
    note: str = "",
) -> None:
    lines = [
        "# Evidence firing counts",
        "",
        note or "E1 wiring snapshot. Zeros may remain until E2 threshold refit.",
        "",
        "| evidence | E1_before | E1_after | E2_after | normal_rate | attack_rate |",
        "|----------|-----------|----------|----------|-------------|-------------|",
    ]
    names = sorted(ALLOWED_EVIDENCE_NAMES)
    for name in names:
        b = before.get(name, 0) if before else "-"
        a1 = after_e1.get(name, 0) if after_e1 else "-"
        a2 = after_e2.get(name, 0) if after_e2 else "-"
        nr = rates[name]["normal_rate"] if rates and name in rates else "-"
        ar = rates[name]["attack_rate"] if rates and name in rates else "-"
        if isinstance(nr, float):
            nr = f"{nr:.4f}"
        if isinstance(ar, float):
            ar = f"{ar:.4f}"
        lines.append(f"| {name} | {b} | {a1} | {a2} | {nr} | {ar} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--calibration", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--stage1-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--progress-log-dir", type=Path, default=ROOT / "dataset/synthetic/steps")
    parser.add_argument(
        "--legacy-results",
        type=Path,
        default=ROOT / "dataset/eval/cyber_pipeline_results_v3.json",
        help="Pre-E1 pipeline results for E1_before column",
    )
    parser.add_argument("--phase", choices=["e1", "e2"], default="e1")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/evidence_firing_counts.md")
    parser.add_argument("--json-out", type=Path, default=ROOT / "dataset/eval/evidence_firing_counts.json")
    args = parser.parse_args()

    before = _count_from_pipeline_results(args.legacy_results) if args.legacy_results.exists() else {}
    counts, rates, n_windows = _count_from_stage1(
        args.telemetry,
        args.split_manifest,
        splits=["train", "cal"],
        baseline_model=args.baseline_model,
        calibration=args.calibration,
        config_path=args.stage1_config,
        progress_log_dir=args.progress_log_dir,
    )
    hist = {}
    if args.json_out.exists():
        try:
            hist = json.loads(args.json_out.read_text(encoding="utf-8"))
        except Exception:
            hist = {}
    if args.phase == "e1":
        hist["e1_before_pipeline_results"] = before
        hist["e1_after_stage1_train_cal"] = counts
        hist["e1_rates"] = rates
        hist["e1_n_windows"] = n_windows
        write_markdown(
            args.out,
            before=before,
            after_e1=counts,
            after_e2=hist.get("e2_after_stage1_train_cal"),
            rates=rates,
            note="E1 wiring complete. Stage1 counts use train+cal; E1_before is from legacy pipeline results.",
        )
    else:
        hist["e2_after_stage1_train_cal"] = counts
        hist["e2_rates"] = rates
        hist["e2_n_windows"] = n_windows
        write_markdown(
            args.out,
            before=hist.get("e1_before_pipeline_results") or before,
            after_e1=hist.get("e1_after_stage1_train_cal"),
            after_e2=counts,
            rates=rates,
            note="E2 after threshold refit + recalibration. train+cal only.",
        )
    args.json_out.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    zeros = [k for k, v in counts.items() if v == 0]
    print(json.dumps({"phase": args.phase, "n_windows": n_windows, "zeros": zeros, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
