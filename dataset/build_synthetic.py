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
from dataset.synth_attacks import ATTACK_GENERATORS, swma
from dataset.synth_common import write_csv
from dataset.synth_normal_patterns import NORMAL_GENERATORS
from dataset.progress_log_policy import (
    DEFAULT_DROP_PROB,
    POLICY_VERSION,
    decide_progress_log,
)
from pipeline.features import compute_window_features

HOLDOUT_PERIOD_S = (2.8, 3.2)


def _manifest(
    df: pd.DataFrame,
    source: str,
    seed: int,
    progress_log_path: str | None = None,
    split: str | None = None,
    *,
    native_progress_available: bool | None = None,
    progress_log_masked: bool | None = None,
    progress_log_drop_prob: float | None = None,
    progress_log_policy_seed: int | None = None,
) -> SessionManifest:
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
        native_progress_available=native_progress_available,
        progress_log_masked=progress_log_masked,
        progress_log_policy_version=POLICY_VERSION if native_progress_available is not None else None,
        progress_log_drop_prob=progress_log_drop_prob,
        progress_log_policy_seed=progress_log_policy_seed,
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


def _period_s_from_frame(frame: pd.DataFrame) -> float | None:
    if "waveform_frequency_hz" in frame.columns:
        freq = pd.to_numeric(frame["waveform_frequency_hz"], errors="coerce").dropna()
        if len(freq) and float(freq.iloc[0]) > 0:
            return 1.0 / float(freq.iloc[0])
    if "gt_params_json" in frame.columns:
        raw = frame["gt_params_json"].iloc[0]
        if isinstance(raw, str) and raw.strip():
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                params = {}
            if "period_s" in params:
                return float(params["period_s"])
            if "frequency_hz" in params and float(params["frequency_hz"]) > 0:
                return 1.0 / float(params["frequency_hz"])
            if "base_period_s" in params:
                return float(params["base_period_s"])
    return None


def _apply_holdout(splits: dict[str, list[str]], holdout_ids: list[str]) -> dict[str, list[str]]:
    holdout = set(holdout_ids)
    cleaned = {
        name: [session_id for session_id in values if session_id not in holdout]
        for name, values in splits.items()
    }
    cleaned["test"] = sorted(set(cleaned["test"]) | holdout)
    return cleaned


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


def _apply_progress_policy(
    output_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    seed: int,
    drop_prob: float,
) -> tuple[str | None, object]:
    decision = decide_progress_log(
        session_id,
        native_events=events,
        seed=seed,
        drop_prob=drop_prob,
    )
    path = None
    if decision.write_progress_log:
        path = _write_progress(output_dir, session_id, events)
    return path, decision


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
    progress_log_drop_prob: float = DEFAULT_DROP_PROB,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    # (frame, gen_seed, progress_path, decision)
    pending_manifests: list[tuple[pd.DataFrame, int, str | None, object]] = []
    stats: dict[str, dict[str, float]] = {}
    sessions_by_label: dict[str, list[str]] = {}

    for class_index, (label, generator) in enumerate(NORMAL_GENERATORS.items(), start=100):
        for repeat in range(sessions_per_class):
            gen_seed = seed * 10000 + class_index * 100 + repeat
            frame = generator(n=rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            session_id = str(frame["session_id"].iloc[0])
            if write_session_csvs:
                write_csv(frame, output_dir / "normal" / f"{session_id}.csv")
            events = list(frame.attrs.get("progress_events", []) or [])
            progress, decision = _apply_progress_policy(
                output_dir, session_id, events, seed=seed, drop_prob=progress_log_drop_prob
            )
            frames.append(frame)
            pending_manifests.append((frame, gen_seed, progress, decision))
            sessions_by_label.setdefault(label, []).append(session_id)

    for class_index, (label, generator) in enumerate(ATTACK_GENERATORS.items(), start=200):
        for repeat in range(sessions_per_class):
            gen_seed = seed * 10000 + class_index * 100 + repeat
            frame = generator(n=rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            session_id = str(frame["session_id"].iloc[0])
            if write_session_csvs:
                write_csv(frame, output_dir / "attacks" / f"{session_id}.csv")
            events = list(frame.attrs.get("progress_events", []) or [])
            progress, decision = _apply_progress_policy(
                output_dir, session_id, events, seed=seed, drop_prob=progress_log_drop_prob
            )
            frames.append(frame)
            pending_manifests.append((frame, gen_seed, progress, decision))
            sessions_by_label.setdefault(label, []).append(session_id)

    # Unseen-period SWMA holdout: always generated and forced into test only.
    holdout_ids: list[str] = []
    n_holdout = max(2, max(1, sessions_per_class // 4))
    holdout_rng = np.random.default_rng(seed + 999)
    for repeat in range(n_holdout):
        period_s = float(holdout_rng.uniform(*HOLDOUT_PERIOD_S))
        gen_seed = seed * 10000 + 299 * 100 + repeat
        frame = swma(
            n=rows,
            sample_hz=sample_hz,
            seed=gen_seed,
            frequency_hz=1.0 / period_s,
            duty_cycle=0.5,
            nvml_avg_window_s=nvml_avg_window_s,
            variant="swma_holdout",
        )
        session_id = str(frame["session_id"].iloc[0])
        if write_session_csvs:
            write_csv(frame, output_dir / "attacks" / f"{session_id}.csv")
        events = list(frame.attrs.get("progress_events", []) or [])
        progress, decision = _apply_progress_policy(
            output_dir, session_id, events, seed=seed, drop_prob=progress_log_drop_prob
        )
        frames.append(frame)
        pending_manifests.append((frame, gen_seed, progress, decision))
        holdout_ids.append(session_id)

    # Also treat any accidentally in-range SWMA periods as holdout.
    for frame, _, _, _ in pending_manifests:
        session_id = str(frame["session_id"].iloc[0])
        if session_id in holdout_ids:
            continue
        label = str(frame["gt_label"].iloc[0] if "gt_label" in frame else frame["label"].iloc[0])
        if not str(label).startswith("swma"):
            continue
        period_s = _period_s_from_frame(frame)
        if period_s is not None and HOLDOUT_PERIOD_S[0] <= period_s <= HOLDOUT_PERIOD_S[1]:
            holdout_ids.append(session_id)

    combined = pd.concat(frames, ignore_index=True)
    validate_frame(combined)
    legacy = output_dir / "all_v2.csv"
    if legacy.exists() and not (output_dir / "all_v2_legacy.csv").exists():
        shutil.copy2(legacy, output_dir / "all_v2_legacy.csv")
    write_csv(combined, output_dir / "all_v3.csv")
    if sessions_per_class == 1:
        write_csv(combined, output_dir / "all_v2.csv")
    splits = _split_sessions(sessions_by_label, seed, split_ratios)
    splits = _apply_holdout(splits, holdout_ids)
    session_to_split = {session_id: name for name, values in splits.items() for session_id in values}
    manifests = [
        _manifest(
            frame,
            "synthetic",
            gen_seed,
            progress,
            session_to_split.get(str(frame["session_id"].iloc[0])),
            native_progress_available=decision.native_progress_available,
            progress_log_masked=decision.progress_log_masked,
            progress_log_drop_prob=decision.drop_prob,
            progress_log_policy_seed=seed,
        )
        for frame, gen_seed, progress, decision in pending_manifests
    ]
    write_manifest(manifests, output_dir / "manifest.json")
    split_manifest = {
        "seed": seed,
        "split_ratios": list(split_ratios),
        "data": "all_v3.csv",
        "train": splits["train"],
        "cal": splits["cal"],
        "test": splits["test"],
        "progress_log_policy": {
            "version": POLICY_VERSION,
            "drop_prob": float(progress_log_drop_prob),
            "seed": int(seed),
        },
        "unseen_param_holdout": {
            "swma_period_2_8_3_2": sorted(set(holdout_ids)),
            "period_s": list(HOLDOUT_PERIOD_S),
        },
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
    parser.add_argument("--progress-log-drop-prob", type=float, default=DEFAULT_DROP_PROB)
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
        progress_log_drop_prob=args.progress_log_drop_prob,
    )
    print(f"생성 완료: {args.output_dir / 'all_v3.csv'} ({len(frame):,}행)")


if __name__ == "__main__":
    main()
