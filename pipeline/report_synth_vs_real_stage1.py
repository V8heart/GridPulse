"""Score the same real windows with synth vs real-fitted Stage1 models."""
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

from pipeline.attack_visibility import attack_visibility_group
from pipeline.baseline import CohortBaseline
from pipeline.stage1_v2 import build_windows, load_config, load_truth_index, score_window


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    correct = 0.0
    for value in pos:
        correct += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(correct / (len(pos) * len(neg)))


def _score_all(windows: pd.DataFrame, baseline: CohortBaseline, calibration: dict, config: dict) -> pd.DataFrame:
    rows = []
    for _, row in windows.iterrows():
        data = row.to_dict()
        out = score_window(data, baseline, calibration, config)
        vis = data.get("visibility_group") or attack_visibility_group(
            gt_label=str(data.get("gt_label", "")),
            gt_variant=data.get("gt_variant"),
            period_s=data.get("period_s"),
        )
        rows.append({
            "session_id": data.get("session_id"),
            "gpu_id": data.get("gpu_id"),
            "gpu_role": data.get("gpu_role"),
            "gt_label": data.get("gt_label"),
            "gt_variant": data.get("gt_variant"),
            "declared_job_family": data.get("declared_job_family"),
            "declared_job_type": data.get("declared_job_type"),
            "mean_w": data.get("mean_w"),
            "u_score": out["u_score"],
            "is_candidate": out["is_candidate"],
            "idle_status": out.get("idle_status"),
            "visibility_group": vis,
        })
    return pd.DataFrame(rows)


def _rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    return float(frame["is_candidate"].mean())


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def run(args) -> dict:
    config_synth = load_config(args.synth_config)
    config_real = load_config(args.real_config)
    truth = load_truth_index(args.labels, sessions_root=args.sessions_root)
    df = pd.read_csv(args.telemetry, low_memory=False)
    windows = build_windows(
        df,
        window_s=args.window_s,
        stride_s=args.stride_s,
        config=config_real,
        sessions_root=args.sessions_root,
        truth_index=truth,
        progress_source=args.progress_source,
    )
    synth_base = CohortBaseline.load(args.synth_baseline)
    real_base = CohortBaseline.load(args.real_baseline)
    synth_cal = json.loads(Path(args.synth_calibration).read_text(encoding="utf-8"))
    real_cal = json.loads(Path(args.real_calibration).read_text(encoding="utf-8"))
    synth = _score_all(windows, synth_base, synth_cal, config_synth)
    real = _score_all(windows, real_base, real_cal, config_real)

    def pack(name: str, frame: pd.DataFrame) -> dict:
        attack = frame[~frame["gt_label"].astype(str).str.startswith("normal")]
        normal = frame[frame["gt_label"].astype(str).str.startswith("normal")]
        det = frame[frame["visibility_group"] == "detectable"]
        unobs = frame[frame["visibility_group"] == "unobservable"]
        crypto = frame[frame["gt_label"].astype(str).str.contains("crypto", case=False, na=False)]
        finetune = frame[frame["gt_variant"].astype(str).eq("finetune")]
        auc = None
        if not crypto.empty and not finetune.empty:
            scores = pd.concat([crypto, finetune], ignore_index=True)
            labels = scores["gt_label"].astype(str).str.contains("crypto", case=False, na=False).astype(int).to_numpy()
            auc = _auc(scores["u_score"].to_numpy(), labels)
        by_label = (
            frame.groupby(frame["gt_label"].astype(str))["is_candidate"].mean().to_dict()
            if not frame.empty
            else {}
        )
        by_role = (
            frame.groupby(frame["gpu_role"].astype(str))["is_candidate"].mean().to_dict()
            if "gpu_role" in frame and frame["gpu_role"].notna().any()
            else {}
        )
        return {
            "name": name,
            "n": int(len(frame)),
            "candidate_rate": _rate(frame),
            "normal_rate": _rate(normal),
            "attack_recall": _rate(attack),
            "detectable_recall": _rate(det),
            "unobservable_recall": _rate(unobs),
            "n_detectable": int(len(det)),
            "n_unobservable": int(len(unobs)),
            "by_label": {str(k): float(v) for k, v in by_label.items()},
            "by_role": {str(k): float(v) for k, v in by_role.items()},
            "crypto_n": int(len(crypto)),
            "finetune_n": int(len(finetune)),
            "crypto_candidate_rate": _rate(crypto),
            "finetune_candidate_rate": _rate(finetune),
            "crypto_u_score_median": float(crypto["u_score"].median()) if not crypto.empty else None,
            "finetune_u_score_median": float(finetune["u_score"].median()) if not finetune.empty else None,
            "crypto_mean_w_median": float(crypto["mean_w"].median()) if not crypto.empty else None,
            "finetune_mean_w_median": float(finetune["mean_w"].median()) if not finetune.empty else None,
            "crypto_vs_finetune_auc": auc,
            "idle_gated": int((frame["idle_status"] == "idle").sum()),
            "undeclared_load": int((frame["idle_status"] == "undeclared_load").sum()),
            "baseline_fallback": getattr(real_base if name == "real" else synth_base, "fallback_counts", {}),
        }

    report = {
        "n_windows": int(len(windows)),
        "job_type_counts": windows["declared_job_type"].value_counts().to_dict() if "declared_job_type" in windows else {},
        "synth": pack("synth", synth),
        "real": pack("real", real),
        "real_baseline_keys": list(getattr(real_base, "cohort_keys", ())),
        "real_fallback_counts": real_base.fallback_counts,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Synth vs real Stage1",
        "",
        f"windows: {report['n_windows']}",
        "",
        * _md_table(
            ["model", "cand rate", "normal rate", "attack recall", "detectable recall", "unobservable recall"],
            [
                [
                    pack_name,
                    f"{report[pack_name]['candidate_rate']:.3f}",
                    f"{report[pack_name]['normal_rate']:.3f}",
                    f"{report[pack_name]['attack_recall']:.3f}",
                    f"{report[pack_name]['detectable_recall']:.3f}",
                    f"{report[pack_name]['unobservable_recall']:.3f}",
                ]
                for pack_name in ("synth", "real")
            ],
        ),
        "",
        "## candidate rate by gt_label",
        "",
        * _md_table(
            ["label", "synth", "real"],
            [
                [
                    label,
                    f"{report['synth']['by_label'].get(label, float('nan')):.3f}",
                    f"{report['real']['by_label'].get(label, float('nan')):.3f}",
                ]
                for label in sorted(set(report["synth"]["by_label"]) | set(report["real"]["by_label"]))
            ],
        ),
        "",
        f"job_type window counts: `{report['job_type_counts']}`",
        "",
        "## gpu_role stratification (report only)",
        "",
        * _md_table(
            ["model", "role", "candidate rate"],
            [
                [name, role, f"{rate:.3f}"]
                for name in ("synth", "real")
                for role, rate in report[name]["by_role"].items()
            ],
        ),
        "",
        "## crypto vs finetune",
        "",
        * _md_table(
            ["model", "crypto cand", "finetune cand", "crypto u median", "finetune u median", "AUC"],
            [
                [
                    name,
                    f"{report[name]['crypto_candidate_rate']:.3f}",
                    f"{report[name]['finetune_candidate_rate']:.3f}",
                    f"{report[name]['crypto_u_score_median']}",
                    f"{report[name]['finetune_u_score_median']}",
                    f"{report[name]['crypto_vs_finetune_auc']}",
                ]
                for name in ("synth", "real")
            ],
        ),
        "",
        f"real cohort_keys: `{report['real_baseline_keys']}`",
        f"real fallback_counts: `{report['real_fallback_counts']}`",
        f"real idle gated / undeclared_load: {report['real']['idle_gated']} / {report['real']['undeclared_load']}",
        "",
    ]
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/real/pipeline/telemetry_with_procs.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/real/index/labels.csv")
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "dataset/real/sessions")
    parser.add_argument("--synth-baseline", type=Path, default=ROOT / "dataset/eval/cohort_baseline_v2.json")
    parser.add_argument("--synth-calibration", type=Path, default=ROOT / "dataset/eval/stage1_v2_calibration.json")
    parser.add_argument("--synth-config", type=Path, default=ROOT / "config/stage1_v2.yaml")
    parser.add_argument("--real-baseline", type=Path, default=ROOT / "dataset/real/pipeline/refit/fold_00/cohort_baseline_v2.json")
    parser.add_argument("--real-calibration", type=Path, default=ROOT / "dataset/real/pipeline/refit/fold_00/stage1_v2_calibration.json")
    parser.add_argument("--real-config", type=Path, default=ROOT / "config/stage1_v2_real.yaml")
    parser.add_argument("--progress-source", default="raw")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--stride-s", type=float, default=15.0)
    parser.add_argument("--out-json", type=Path, default=ROOT / "dataset/real/pipeline/refit/synth_vs_real.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "dataset/real/pipeline/refit/synth_vs_real.md")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"n_windows": report["n_windows"], "real_cand": report["real"]["candidate_rate"], "md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
