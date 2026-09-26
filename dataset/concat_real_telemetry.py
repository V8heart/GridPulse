"""Concat session telemetry while keeping observed_n_procs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

KEEP = [
    "timestamp",
    "t_epoch",
    "session_id",
    "gpu_id",
    "gpu_model",
    "sample_hz",
    "power_w",
    "util_gpu_pct",
    "fb_used_mb",
    "observed_n_procs",
    "declared_job_type",
    "declared_job_family",
    "declared_process_name",
    "declared_user",
    "gt_label",
    "gt_is_attack",
    "gt_variant",
    "warmup",
]


def concat_sessions(
    sessions_root: Path,
    labels: pd.DataFrame,
    *,
    out_path: Path,
) -> pd.DataFrame:
    valid = labels
    if "valid" in valid.columns:
        valid = valid[valid["valid"].astype(str).str.lower().isin({"true", "1"})]
    session_ids = sorted(set(valid["session_id"].astype(str)))
    frames = []
    for session_id in session_ids:
        path = sessions_root / session_id / "telemetry.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        cols = [c for c in KEEP if c in frame.columns]
        frames.append(frame[cols])
    if not frames:
        raise FileNotFoundError(f"no session telemetry under {sessions_root}")
    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "dataset/real/sessions")
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/real/index/labels.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dataset/real/pipeline/telemetry_with_procs.csv",
    )
    args = parser.parse_args()
    frame = concat_sessions(args.sessions_root, pd.read_csv(args.labels, low_memory=False), out_path=args.out)
    print({"rows": int(len(frame)), "sessions": int(frame["session_id"].nunique()), "out": str(args.out)})


if __name__ == "__main__":
    main()
