"""D0~D4 합성 데이터셋을 결정론적으로 생성한다."""
from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.schema import SessionManifest, validate_frame, write_manifest
from dataset.synth_attacks import ATTACK_GENERATORS
from dataset.synth_common import write_csv
from dataset.synth_normal_patterns import NORMAL_GENERATORS
from pipeline.features import compute_window_features


def _manifest(df: pd.DataFrame, source: str, seed: int, progress_log_path: str | None = None,
              split: str | None = None) -> SessionManifest:
    return SessionManifest(
        session_id=str(df["session_id"].iloc[0]),
        label=str(df["gt_label"].iloc[0] if "gt_label" in df else df["label"].iloc[0]),
        source=source,
        sample_hz=float(df["sample_hz"].iloc[0]),
        gpu_ids=sorted(int(x) for x in df["gpu_id"].unique()),
        rows=len(df),
        seed=seed,
        workload=str(df["declared_job_type"].iloc[0] if "declared_job_type" in df else df["job_type"].iloc[0]),
        attack_id=str(df["gt_attack_id"].iloc[0] if "gt_attack_id" in df else df["attack_id"].iloc[0]),
        split=split,
        progress_log_path=progress_log_path,
        gt_variant=str(df["gt_variant"].iloc[0]) if "gt_variant" in df else None,
    )


def _split_sessions(sessions: dict[str, list[str]], seed: int, ratios: tuple[float, float, float]) -> dict:
    rng = np.random.default_rng(seed)
    split = {"train": [], "cal": [], "test": []}
    for label, ids in sorted(sessions.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * ratios[0]))) if n >= 3 else max(0, n - 2)
        n_cal = max(1, int(round(n * ratios[1]))) if n >= 3 else min(1, n - n_train)
        n_train = min(n_train, n)
        n_cal = min(n_cal, max(0, n - n_train))
        split["train"].extend(ids[:n_train])
        split["cal"].extend(ids[n_train:n_train + n_cal])
        split["test"].extend(ids[n_train + n_cal:])
    return split


def _write_progress(output_dir: Path, session_id: str, events: list[dict]) -> str | None:
    if not events:
        return None
    path = output_dir / "steps" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return str(path)


def build(
    output_dir: Path,
    *,
    rows: int = 1200,
    sample_hz: float = 10,
    sessions_per_class: int = 1,
    seed: int = 7,
    split_ratios: tuple[float, float, float] = (0.5, 0.2, 0.3),
    nvml_avg_window_s: float = 0.0,
    write_session_csvs: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pending_manifests: list[tuple[pd.DataFrame, int, str | None]] = []
    stats: dict[str, dict[str, float]] = {}
    sessions_by_label: dict[str, list[str]] = {}

    for class_index, (label, generator) in enumerate(NORMAL_GENERATORS.items(), start=100):
        for repeat in range(sessions_per_class):
            gen_seed = seed * 10000 + class_index * 100 + repeat
            frame = generator(n=rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            session_id = str(frame["session_id"].iloc[0])
            if write_session_csvs:
                write_csv(frame, output_dir / "normal" / f"{session_id}.csv")
            progress = _write_progress(output_dir, session_id, frame.attrs.get("progress_events", []))
            frames.append(frame)
            pending_manifests.append((frame, gen_seed, progress))
            sessions_by_label.setdefault(label, []).append(session_id)

    for class_index, (label, generator) in enumerate(ATTACK_GENERATORS.items(), start=200):
        for repeat in range(sessions_per_class):
            gen_seed = seed * 10000 + class_index * 100 + repeat
            frame = generator(n=rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            session_id = str(frame["session_id"].iloc[0])
            if write_session_csvs:
                write_csv(frame, output_dir / "attacks" / f"{session_id}.csv")
            progress = _write_progress(output_dir, session_id, frame.attrs.get("progress_events", []))
            frames.append(frame)
            pending_manifests.append((frame, gen_seed, progress))
            sessions_by_label.setdefault(label, []).append(session_id)

    combined = pd.concat(frames, ignore_index=True)
    validate_frame(combined)
    legacy = output_dir / "all_v2.csv"
    if legacy.exists() and not (output_dir / "all_v2_legacy.csv").exists():
        shutil.copy2(legacy, output_dir / "all_v2_legacy.csv")
    write_csv(combined, output_dir / "all_v3.csv")
    if sessions_per_class == 1:
        write_csv(combined, output_dir / "all_v2.csv")
    splits = _split_sessions(sessions_by_label, seed, split_ratios)
    session_to_split = {session_id: name for name, values in splits.items() for session_id in values}
    manifests = [
        _manifest(frame, "synthetic", gen_seed, progress, session_to_split.get(str(frame["session_id"].iloc[0])))
        for frame, gen_seed, progress in pending_manifests
    ]
    write_manifest(manifests, output_dir / "manifest.json")
    split_manifest = {
        "seed": seed,
        "split_ratios": list(split_ratios),
        "data": "all_v3.csv",
        "train": splits["train"],
        "cal": splits["cal"],
        "test": splits["test"],
        "unseen_param_holdout": {"swma": {"period_s": [2.8, 3.2]}},
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for frame in frames:
        label = str(frame["label"].iloc[0])
        feats = compute_window_features(
            frame["power_w"].to_numpy(),
            frame["util_gpu_pct"].to_numpy(),
            sample_hz=sample_hz,
        )
        stats[label] = {
            key: round(float(feats[key]), 5)
            for key in ("mean_w", "swing_ratio", "duty_regularity", "periodicity_strength")
        }
    (output_dir / "feature_summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset" / "synthetic")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--sessions-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.5, 0.2, 0.3))
    parser.add_argument("--nvml-avg-window-s", type=float, default=1.0)
    parser.add_argument("--write-session-csvs", action="store_true")
    args = parser.parse_args()
    frame = build(
        args.output_dir,
        rows=args.rows,
        sample_hz=args.sample_hz,
        sessions_per_class=args.sessions_per_class,
        seed=args.seed,
        split_ratios=tuple(args.split_ratios),
        nvml_avg_window_s=args.nvml_avg_window_s,
        write_session_csvs=args.write_session_csvs,
    )
    print(f"생성 완료: {args.output_dir / 'all_v3.csv'} ({len(frame):,}행)")


if __name__ == "__main__":
    main()

