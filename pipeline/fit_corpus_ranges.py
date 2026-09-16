"""Fit corpus numeric ranges from train split only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import compute_window_features
from pipeline.rag_analyzer import CORPUS_DIR, parse_front_matter
from pipeline.run_pipeline import infer_sample_hz

DOC_LABELS = {
    "swma": ["swma"],
    "ltma": ["ltma"],
    "cryptojacking": ["cryptojacking"],
    "normal_workloads": ["normal_baseline", "normal_checkpoint", "normal_dataloader_stall",
                         "normal_distributed_training", "normal_eval_train_switch",
                         "normal_hpo_search", "normal_sync_ddp", "normal_fsdp_deep_trough",
                         "normal_flat_pretrain", "normal_inference_bursty", "normal_mixed_tenants"],
    "benign_periodic": ["normal_checkpoint", "normal_dataloader_stall",
                        "normal_distributed_training", "normal_eval_train_switch",
                        "normal_hpo_search", "normal_sync_ddp", "normal_fsdp_deep_trough"],
    "hidden_ml_training": ["normal_sync_ddp", "normal_flat_pretrain", "normal_hpo_search"],
    "llmjacking": ["normal_inference_bursty", "normal_hpo_search"],
    "sponge_attack": ["cryptojacking", "normal_flat_pretrain"],
    "coordinated_multi_gpu": ["swma", "normal_mixed_tenants"],
    "burst_ramp_attack": ["normal_checkpoint", "normal_dataloader_stall", "normal_hpo_search"],
}

RANGE_KEYS = [
    "swing_ratio",
    "periodicity_strength",
    "duty_regularity",
    "ramp_max_w_per_s",
    "high_load_fraction",
    "longest_high_seconds",
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _window_features(df: pd.DataFrame, session_ids: set[str]) -> pd.DataFrame:
    rows = []
    df = df[df["session_id"].astype(str).isin(session_ids)]
    label_col = "gt_label" if "gt_label" in df else "label"
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        hz = infer_sample_hz(group)
        window = max(8, int(round(30 * hz)))
        stride = max(1, int(round(15 * hz)))
        for start in range(0, len(group) - window + 1, stride):
            subset = group.iloc[start:start + window]
            feats = compute_window_features(
                subset["power_w"].to_numpy(),
                subset["util_gpu_pct"].to_numpy() if "util_gpu_pct" in subset else None,
                sample_hz=hz,
            )
            if feats:
                feats["session_id"] = str(session_id)
                feats["gt_label"] = str(subset[label_col].iloc[0])
                rows.append(feats)
    return pd.DataFrame(rows)


def fit_ranges(telemetry: Path, split_manifest: Path, corpus_dir: Path = CORPUS_DIR) -> dict:
    df = pd.read_csv(telemetry)
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    train_sessions = set(map(str, manifest["train"]))
    windows = _window_features(df, train_sessions)
    docs = {}
    for path in sorted(corpus_dir.glob("*.md")):
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        labels = DOC_LABELS.get(path.stem)
        if not labels:
            continue
        selected = windows[windows["gt_label"].isin(labels)]
        ranges = {}
        for key in RANGE_KEYS:
            if key in selected and len(selected):
                ranges[key] = [
                    round(float(selected[key].quantile(0.05)), 6),
                    round(float(selected[key].quantile(0.95)), 6),
                ]
        docs[path.stem] = {
            "threat_id": meta.get("threat_id", path.stem),
            "category": meta.get("category", "attack"),
            "source_labels": labels,
            "n_windows": int(len(selected)),
            "ranges": ranges,
        }
    return {
        "fit_split": "train",
        "n_sessions": len(train_sessions),
        "n_windows": int(len(windows)),
        "quantiles": [5, 95],
        "data": str(telemetry),
        "git_commit": _git_commit(),
        "documents": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/corpus_feature_ranges.json")
    args = parser.parse_args()
    report = fit_ranges(args.telemetry, args.split_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "documents"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
