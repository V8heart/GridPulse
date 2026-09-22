"""D0~D4 합성 데이터셋을 결정론적으로 생성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.declared_context import sample_declared
from dataset.identifiers import group_id_from_parts, is_session_id
from dataset.schema import SCHEMA_VERSION, SessionManifest, validate_frame, write_manifest
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
DURATION_POOL = (480, 600, 720)


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
        group_id=str(df.attrs.get("group_id")) if df.attrs.get("group_id") else None,
    )


def duration_for_repeat(seed: int, repeat: int) -> int:
    """Choose duration without incorporating class/label information."""
    digest = hashlib.sha256(f"synthetic-duration:{seed}:{repeat}".encode()).digest()
    return DURATION_POOL[int.from_bytes(digest[:4], "big") % len(DURATION_POOL)]


def _contiguous_intervals(times: np.ndarray, active: np.ndarray, step_s: float) -> list[list[float]]:
    intervals: list[list[float]] = []
    start: float | None = None
    for index, enabled in enumerate(active.astype(bool)):
        if enabled and start is None:
            start = float(times[index])
        if start is not None and (not enabled or index == len(active) - 1):
            stop = float(times[index] if not enabled else times[index] + step_s)
            intervals.append([start, stop])
            start = None
    return intervals


def _migrate_frame(frame: pd.DataFrame, *, label: str, seed: int, repeat: int) -> pd.DataFrame:
    """Attach v1.2-only private metadata while keeping telemetry opaque."""
    if not is_session_id(str(frame["session_id"].iloc[0])):
        raise ValueError("synthetic generator produced a non-opaque session id")
    is_attack = label in ATTACK_GENERATORS
    variant = str(frame["gt_variant"].dropna().iloc[0]) if frame["gt_variant"].notna().any() else label
    hosted = label == "ltma" or variant in {"swma_piggyback", "swma_mimicry"}
    if is_attack:
        policy = "host_inherited" if hosted else (
            "pool_random" if repeat % 2 == 0 else "host_family_matched"
        )
        rng = np.random.default_rng(seed + 17)
        if hosted:
            host = sample_declared("training", rng, policy="honest", mismatch_rate=0.0)
            declared = sample_declared("training", rng, policy="host_inherited", host_declared=host)
        else:
            declared = sample_declared("training", rng, policy=policy)
    else:
        policy = "honest"
        family = str(frame["declared_job_family"].iloc[0])
        declared = sample_declared(family, np.random.default_rng(seed + 17), policy="honest")
    declared.setdefault("declared_gres", "gpu:2" if frame["gpu_id"].nunique() > 1 else "gpu:1")
    for key, value in declared.items():
        frame[key] = value

    params_raw = frame["gt_params_json"].dropna()
    params = json.loads(str(params_raw.iloc[0])) if len(params_raw) else {}
    params["declared_policy"] = policy
    frame["gt_params_json"] = json.dumps(params, sort_keys=True)
    frame.attrs["declared_policy"] = policy
    cluster = {
        key: (round(float(value), 1) if isinstance(value, (int, float)) else value)
        for key, value in params.items()
        if key != "declared_policy"
    }
    frame.attrs["group_id"] = group_id_from_parts(label, variant, cluster)
    intervals: dict[str, list[list[float]]] = {}
    if is_attack:
        for gpu_id, group in frame.groupby("gpu_id"):
            times = pd.to_numeric(group["t_epoch"], errors="raise").to_numpy()
            step = 1.0 / float(group["sample_hz"].iloc[0])
            if label == "swma":
                physical = pd.to_numeric(group.get("power_phys_w", group["power_w"]), errors="raise").to_numpy()
                threshold = (float(np.nanpercentile(physical, 20)) + float(np.nanpercentile(physical, 80))) / 2
                intervals[str(int(gpu_id))] = _contiguous_intervals(times, physical >= threshold, step)
            else:
                intervals[str(int(gpu_id))] = [[float(times[0]), float(times[-1] + step)]]
    frame.attrs["gt_attack_intervals_epoch"] = intervals
    return frame


def _write_standard_tree(
    frames: list[pd.DataFrame],
    output_dir: Path,
    *,
    session_to_split: dict[str, str],
    policy_seed: int,
    drop_prob: float,
) -> None:
    labels: list[dict] = []
    for frame in frames:
        session_id = str(frame["session_id"].iloc[0])
        session_dir = output_dir / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        write_csv(frame, session_dir / "telemetry.csv")
        events = list(frame.attrs.get("progress_events", []) or [])
        label = str(frame["gt_label"].iloc[0])
        if not events and label != "normal_baseline":
            end_s = float(frame["timestamp"].max())
            events = [
                {"t": float(t), "gpu_id": 0, "event": "step_end", "step": index}
                for index, t in enumerate(np.linspace(max(0.1, end_s / 10), end_s, 10))
            ]
        raw_events = []
        t0 = float(frame["t_epoch"].iloc[0] - frame["timestamp"].iloc[0])
        for event in events:
            item = dict(event)
            relative = float(item.pop("t", 0.0))
            item["t_epoch"] = t0 + relative
            raw_events.append(item)
        if raw_events:
            raw_path = session_dir / "progress.raw.jsonl"
            raw_path.write_text(
                "\n".join(json.dumps(event, sort_keys=True) for event in raw_events) + "\n",
                encoding="utf-8",
            )
            decision = decide_progress_log(
                session_id, native_events=raw_events, seed=policy_seed, drop_prob=drop_prob
            )
            if decision.write_progress_log:
                visible = [{**event, "t": event["t_epoch"] - t0} for event in raw_events]
                (session_dir / "progress.jsonl").write_text(
                    "\n".join(json.dumps(event, sort_keys=True) for event in visible) + "\n",
                    encoding="utf-8",
                )
        else:
            decision = decide_progress_log(
                session_id, native_events=[], seed=policy_seed, drop_prob=drop_prob,
                idle_exception=True,
            )
        declared = {
            str(int(gpu_id)): {
                key: str(group[key].iloc[0])
                for key in (
                    "declared_job_type", "declared_job_family", "declared_process_name",
                    "declared_user", "declared_gres",
                )
            }
            for gpu_id, group in frame.groupby("gpu_id")
        }
        public = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "source": "synthetic",
            "t0_epoch": t0,
            "duration_s": float(frame["timestamp"].max()),
            "warmup_s": 180.0,
            "gpus": [
                {"gpu_id": int(gpu_id), "gpu_model": "synthetic-rtx4090"}
                for gpu_id in sorted(frame["gpu_id"].unique())
            ],
            "progress_log": {
                "present": (session_dir / "progress.jsonl").exists(),
                "masked": bool(decision.progress_log_masked),
                "policy_version": decision.policy_version,
                "drop_prob": decision.drop_prob,
                "seed": policy_seed,
            },
            "declared": declared,
        }
        (session_dir / "session.json").write_text(
            json.dumps(public, indent=2, sort_keys=True), encoding="utf-8"
        )
        intervals = frame.attrs.get("gt_attack_intervals_epoch") or {}
        for gpu_id in sorted(int(x) for x in frame["gpu_id"].unique()):
            labels.append({
                "session_id": session_id,
                "gpu_id": gpu_id,
                "gt_label": label,
                "gt_is_attack": label in ATTACK_GENERATORS,
                "gt_variant": frame["gt_variant"].iloc[0],
                "gt_params_json": frame["gt_params_json"].iloc[0],
                "gt_attack_intervals_epoch": json.dumps(intervals.get(str(gpu_id), [])),
                "gpu_role": "target",
                "group_id": frame.attrs["group_id"],
                "declared_policy": frame.attrs["declared_policy"],
                "source": "synthetic",
                "split": session_to_split.get(session_id),
                "valid": True,
            })
    index = output_dir / "index"
    index.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(index / "labels.csv", index=False)


def leakage_audit(output_dir: Path) -> dict:
    forbidden = ("attack", "swma", "ltma", "crypto", "malicious", "unknown_cuda", "synthetic_", "syn-", "real-")
    findings = []
    for path in (output_dir / "sessions").glob("*"):
        if any(token in path.name.lower() for token in forbidden):
            findings.append(str(path))
        public = path / "session.json"
        if public.exists():
            text = public.read_text(encoding="utf-8").lower()
            findings.extend(f"{public}:{token}" for token in forbidden if token in text)
    report = {"passed": not findings, "findings": findings, "sessions_checked": len(list((output_dir / "sessions").glob("*")))}
    (output_dir / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if findings:
        raise ValueError(f"synthetic public-tree leakage: {findings[:3]}")
    return report


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
    # Until Part S2 adds decoy logs for every non-idle class, empty native event
    # lists are treated as the idle exception so existing synth fixtures stay valid.
    decision = decide_progress_log(
        session_id,
        native_events=events,
        seed=seed,
        drop_prob=drop_prob,
        idle_exception=not bool(events),
    )
    path = None
    if decision.write_progress_log:
        path = _write_progress(output_dir, session_id, events)
    return path, decision


def build(
    output_dir: Path,
    *,
    rows: int | None = 1200,
    sample_hz: float = 10,
    sessions_per_class: int = 1,
    seed: int = 7,
    split_ratios: tuple[float, float, float] = (0.5, 0.2, 0.3),
    nvml_avg_window_s: float = 0.0,
    write_session_csvs: bool = False,
    progress_log_drop_prob: float = DEFAULT_DROP_PROB,
    run_leakage_audit: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    # (frame, gen_seed, progress_path, decision)
    pending_manifests: list[tuple[pd.DataFrame, int, str | None, object]] = []
    stats: dict[str, dict[str, float]] = {}
    sessions_by_label: dict[str, list[str]] = {}

    for class_index, (label, generator) in enumerate(NORMAL_GENERATORS.items(), start=100):
        for repeat in range(sessions_per_class):
            gen_seed = seed * 10000 + class_index * 100 + repeat
            session_rows = int(rows) if rows is not None else int(round(duration_for_repeat(seed, repeat) * sample_hz))
            frame = generator(n=session_rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            frame = _migrate_frame(frame, label=label, seed=gen_seed, repeat=repeat)
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
            session_rows = int(rows) if rows is not None else int(round(duration_for_repeat(seed, repeat) * sample_hz))
            frame = generator(n=session_rows, sample_hz=sample_hz, seed=gen_seed, nvml_avg_window_s=nvml_avg_window_s)
            frame = _migrate_frame(frame, label=label, seed=gen_seed, repeat=repeat)
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
        session_rows = int(rows) if rows is not None else int(round(duration_for_repeat(seed, repeat) * sample_hz))
        frame = swma(
            n=session_rows,
            sample_hz=sample_hz,
            seed=gen_seed,
            frequency_hz=1.0 / period_s,
            duty_cycle=0.5,
            nvml_avg_window_s=nvml_avg_window_s,
            variant="swma_holdout",
        )
        frame = _migrate_frame(frame, label="swma", seed=gen_seed, repeat=repeat)
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
    _write_standard_tree(
        frames,
        output_dir,
        session_to_split=session_to_split,
        policy_seed=seed,
        drop_prob=progress_log_drop_prob,
    )
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
    if run_leakage_audit:
        leakage_audit(output_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", type=Path, default=ROOT / "dataset" / "synthetic")
    parser.add_argument("--rows", type=int, default=None, help="override rows; default uses 480/600/720s pool")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--sessions-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.5, 0.2, 0.3))
    parser.add_argument("--nvml-avg-window-s", type=float, default=1.0)
    parser.add_argument("--write-session-csvs", action="store_true")
    parser.add_argument("--progress-log-drop-prob", type=float, default=DEFAULT_DROP_PROB)
    parser.add_argument("--skip-leakage-audit", action="store_true")
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
        run_leakage_audit=not args.skip_leakage_audit,
    )
    print(f"생성 완료: {args.output_dir / 'all_v3.csv'} ({len(frame):,}행)")


if __name__ == "__main__":
    main()
