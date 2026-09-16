"""윈도우 후보 생성과 RAG 판정의 데이터셋 수준 지표."""
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

from pipeline.features import compute_window_features, features_to_description
from pipeline.rag_analyzer import SignatureRetriever
from pipeline.run_pipeline import context_to_query, infer_sample_hz, stage1_screen


def evaluate(path: Path, *, baseline: float, window: int, stride: int,
             split_manifest: Path | None = None, split: str = "test") -> dict:
    df = pd.read_csv(path)
    if split_manifest and split != "all":
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        df = df[df["session_id"].astype(str).isin(set(map(str, manifest[split])))]
    retriever = SignatureRetriever(backend="tfidf")
    windows = []
    filter_counts: Counter[str] = Counter()

    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        hz = infer_sample_hz(group)
        screened = stage1_screen(
            group,
            baseline_mean_w=baseline,
            sample_hz=hz,
            window=window,
            stride=stride,
        )
        for _, item in screened.iterrows():
            label_col = "gt_label" if "gt_label" in group else "label"
            label_value = str(group[label_col].iloc[0])
            normal = label_value.startswith("normal")
            record = {"normal": normal, "candidate": bool(item["is_candidate"]), "rag_correct": None}
            if item["is_candidate"]:
                reasons = [reason for reason in str(item["candidate_reasons"]).split(",") if reason]
                filter_counts.update(reasons)
                subset = group.iloc[int(item["start"]) : int(item["end"])]
                feats = compute_window_features(
                    subset["power_w"].to_numpy(),
                    subset["util_gpu_pct"].to_numpy(),
                    sample_hz=hz,
                )
                description = features_to_description(feats, baseline_mean_w=baseline)
                context = {
                    key: str(subset[key].iloc[0])
                    for key in ("id_user", "job_type", "gres_req")
                    if key in subset
                }
                top = retriever.search(
                    f"{description} {context_to_query(context)}".strip(), top_k=1
                )[0][0]
                expected = "normal_workloads" if normal else ("swma" if label_value.startswith("swma") else label_value)
                record["rag_correct"] = top == expected
                record["top1"] = top
            windows.append(record)

    attack = [row for row in windows if not row["normal"]]
    normal = [row for row in windows if row["normal"]]
    candidates = [row for row in windows if row["candidate"]]
    normal_candidates = [row for row in normal if row["candidate"]]
    return {
        "windows": len(windows),
        "candidate_rate": sum(row["candidate"] for row in windows) / len(windows),
        "attack_candidate_recall": sum(row["candidate"] for row in attack) / len(attack),
        "normal_candidate_rate": sum(row["candidate"] for row in normal) / len(normal),
        "rag_top1_on_candidates": (
            sum(bool(row["rag_correct"]) for row in candidates) / len(candidates)
            if candidates else 0.0
        ),
        "normal_hard_negative_rejection": (
            sum(bool(row["rag_correct"]) for row in normal_candidates) / len(normal_candidates)
            if normal_candidates else 1.0
        ),
        "filter_trigger_counts": dict(filter_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=Path("dataset/synthetic/all_v2.csv"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "cal", "test", "all"], default="test")
    parser.add_argument("--baseline-mean", type=float, default=120)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("dataset/eval/detection_report.json"))
    args = parser.parse_args()
    report = evaluate(
        args.telemetry,
        baseline=args.baseline_mean,
        window=args.window,
        stride=args.stride,
        split_manifest=args.split_manifest,
        split=args.split,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

