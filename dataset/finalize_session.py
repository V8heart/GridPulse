"""Finalize a captured session into gp-telemetry/1.2 public artifacts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from dataset.progress_log_policy import DEFAULT_DROP_PROB, POLICY_VERSION, decide_progress_log
from dataset.schema import SCHEMA_VERSION
from dataset.validate_standard import validate_session_dir

DEFAULT_WARMUP_S = 180.0
LABEL_COLUMNS = [
    "session_id",
    "gpu_id",
    "gt_label",
    "gt_is_attack",
    "gt_variant",
    "gt_params_json",
    "gt_attack_intervals_epoch",
    "gpu_role",
    "group_id",
    "capture_key",
    "seed",
    "declared_policy",
    "workload_module",
    "workload_cmd",
    "host_workload",
    "run_id",
    "source",
    "created_epoch",
    "valid",
    "invalid_reason",
]


def _atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_paths(session_id: str, *, root: Path = ROOT) -> dict[str, Path]:
    real = root / "dataset" / "real"
    return {
        "session_dir": real / "sessions" / session_id,
        "staging_dir": real / ".staging" / session_id,
        "private_capture": real / "private" / "capture" / f"{session_id}.json",
        "private_stdout": real / "private" / "stdout" / f"{session_id}.log",
        "labels": real / "index" / "labels.csv",
        "labels_lock": real / "index" / "labels.csv.lock",
    }


def _read_private(paths: dict[str, Path]) -> dict[str, Any]:
    staging = paths["staging_dir"] / "capture_private.json"
    if staging.exists():
        return _load_json(staging)
    if paths["private_capture"].exists():
        return _load_json(paths["private_capture"])
    raise FileNotFoundError(f"no private capture for {paths['session_dir'].name}")


def _rewrite_progress(
    raw_path: Path,
    visible_path: Path,
    *,
    t0_epoch: float,
    masked: bool,
) -> None:
    if masked:
        if visible_path.exists():
            visible_path.unlink()
        return
    events = []
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            obj.pop("t", None)  # never trust workload t
            t_epoch = float(obj["t_epoch"])
            obj["t"] = t_epoch - float(t0_epoch)
            events.append(obj)
    tmp = visible_path.with_suffix(visible_path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + ("\n" if events else ""),
        encoding="utf-8",
    )
    os.replace(tmp, visible_path)


def _attach_columns(
    telemetry: pd.DataFrame,
    private: dict[str, Any],
    *,
    t0_epoch: float,
    warmup_s: float,
    workload_start_epoch: float | None,
) -> pd.DataFrame:
    out = telemetry.copy()
    declared_by_gpu = private.get("declared_by_gpu") or {}
    roles = private.get("gpu_roles") or {}
    intervals = private.get("gt_attack_intervals_epoch") or {}
    warm_origin = float(workload_start_epoch if workload_start_epoch is not None else t0_epoch)

    for col in (
        "declared_job_type",
        "declared_job_family",
        "declared_process_name",
        "declared_user",
        "declared_gres",
        "gt_label",
        "gt_is_attack",
        "gt_variant",
        "gt_params_json",
        "warmup",
    ):
        if col not in out:
            out[col] = pd.NA

    for idx, row in out.iterrows():
        gpu_id = str(int(row["gpu_id"]))
        role = roles.get(gpu_id) or roles.get(int(gpu_id)) or "target"
        declared = declared_by_gpu.get(gpu_id) or declared_by_gpu.get(int(gpu_id)) or {}
        if role == "companion_idle":
            out.at[idx, "gt_label"] = "normal_idle"
            out.at[idx, "gt_is_attack"] = False
            out.at[idx, "gt_variant"] = pd.NA
            out.at[idx, "declared_job_type"] = declared.get("declared_job_type", "notebook")
            out.at[idx, "declared_job_family"] = "interactive"
            out.at[idx, "declared_process_name"] = declared.get("declared_process_name", "jupyter-kernel")
            out.at[idx, "declared_user"] = declared.get("declared_user", "user_000")
            out.at[idx, "declared_gres"] = declared.get("declared_gres", "gpu:1")
        else:
            out.at[idx, "gt_label"] = private.get("gt_label")
            out.at[idx, "gt_is_attack"] = bool(private.get("gt_is_attack", not str(private.get("gt_label", "")).startswith("normal")))
            out.at[idx, "gt_variant"] = private.get("gt_variant")
            for key in (
                "declared_job_type",
                "declared_job_family",
                "declared_process_name",
                "declared_user",
                "declared_gres",
            ):
                if key in declared:
                    out.at[idx, key] = declared[key]
        out.at[idx, "gt_params_json"] = private.get("gt_params_json")
        t_epoch = float(row["t_epoch"])
        out.at[idx, "warmup"] = (t_epoch - warm_origin) < float(warmup_s)
    _ = intervals  # intervals stay in labels/private only
    return out


def _upsert_labels(labels_path: Path, lock_path: Path, rows: list[dict[str, Any]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if labels_path.exists():
                existing = pd.read_csv(labels_path)
            else:
                existing = pd.DataFrame(columns=LABEL_COLUMNS)
            frame = pd.DataFrame(rows)
            for col in LABEL_COLUMNS:
                if col not in existing:
                    existing[col] = pd.NA
                if col not in frame:
                    frame[col] = pd.NA
            key = ["session_id", "gpu_id"]
            existing["gpu_id"] = pd.to_numeric(existing["gpu_id"], errors="coerce")
            frame["gpu_id"] = pd.to_numeric(frame["gpu_id"], errors="coerce")
            # drop old keys then append
            mask = pd.Series(False, index=existing.index)
            for _, row in frame.iterrows():
                mask = mask | (
                    (existing["session_id"].astype(str) == str(row["session_id"]))
                    & (existing["gpu_id"] == row["gpu_id"])
                )
            kept = existing.loc[~mask, LABEL_COLUMNS]
            merged = pd.concat([kept, frame[LABEL_COLUMNS]], ignore_index=True)
            tmp = labels_path.with_suffix(".csv.tmp")
            merged.to_csv(tmp, index=False)
            os.replace(tmp, labels_path)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def finalize_session(
    session_id: str,
    *,
    root: Path = ROOT,
    warmup_s: float = DEFAULT_WARMUP_S,
    drop_prob: float = DEFAULT_DROP_PROB,
    policy_seed: int = 7,
) -> dict[str, Any]:
    paths = _resolve_paths(session_id, root=root)
    session_dir = paths["session_dir"]
    private = _read_private(paths)
    collector_result_path = paths["staging_dir"] / "collector_result.json"
    if not collector_result_path.exists():
        # allow re-finalize from private archive meta if present
        collector_result = private.get("collector_result") or {}
    else:
        collector_result = _load_json(collector_result_path)

    telemetry_path = session_dir / "telemetry.csv"
    if not telemetry_path.exists():
        raise FileNotFoundError(telemetry_path)
    telemetry = pd.read_csv(telemetry_path)
    t0_epoch = float(collector_result.get("t0_epoch") or private.get("t0_epoch") or telemetry["t_epoch"].iloc[0] - telemetry["timestamp"].iloc[0])

    raw_progress = session_dir / "progress.raw.jsonl"
    visible_progress = session_dir / "progress.jsonl"
    idle_exception = bool(private.get("idle_exception", False))
    decision = decide_progress_log(
        session_id,
        native_events=[{"ok": True}] if raw_progress.exists() else [],
        seed=int(private.get("progress_mask_seed", policy_seed)),
        drop_prob=float(private.get("progress_mask_drop_prob", drop_prob)),
        idle_exception=idle_exception or not raw_progress.exists(),
    )
    _rewrite_progress(raw_progress, visible_progress, t0_epoch=t0_epoch, masked=decision.progress_log_masked)

    workload_start_epoch = None
    if raw_progress.exists():
        for line in raw_progress.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("event") == "workload_start":
                workload_start_epoch = float(obj["t_epoch"])
                break

    enriched = _attach_columns(
        telemetry,
        private,
        t0_epoch=t0_epoch,
        warmup_s=warmup_s,
        workload_start_epoch=workload_start_epoch,
    )
    tmp_tel = telemetry_path.with_suffix(".csv.tmp")
    enriched.to_csv(tmp_tel, index=False)
    os.replace(tmp_tel, telemetry_path)

    duration_s = float(enriched["timestamp"].max()) if len(enriched) else 0.0
    session_json = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "run_id": private.get("run_id"),
        "source": private.get("source", "real"),
        "t0_epoch": t0_epoch,
        "duration_s": duration_s,
        "warmup_s": float(warmup_s),
        "host": private.get("host") or {},
        "gpus": private.get("gpus") or collector_result.get("gpus") or [],
        "software": private.get("software") or {},
        "collector": {
            "requested_interval_ms": collector_result.get("requested_interval_ms"),
            "proc_poll_s": collector_result.get("proc_poll_s"),
            "power_instant_supported": collector_result.get("power_instant_supported"),
            "interval_warnings": collector_result.get("interval_warnings") or [],
            "aborted_reason": collector_result.get("aborted_reason"),
        },
        "progress_log": {
            "present": bool(visible_progress.exists()),
            "masked": bool(decision.progress_log_masked),
            "policy_version": decision.policy_version,
            "drop_prob": decision.drop_prob,
            "seed": int(private.get("progress_mask_seed", policy_seed)),
        },
        "preflight": private.get("preflight") or {},
        "declared": private.get("declared_by_gpu") or {},
    }
    _atomic_write_json(session_dir / "session.json", session_json)

    # labels rows
    roles = private.get("gpu_roles") or {}
    gpu_ids = sorted({int(x) for x in enriched["gpu_id"].tolist()})
    label_rows = []
    for gpu_id in gpu_ids:
        role = roles.get(str(gpu_id)) or roles.get(gpu_id) or "target"
        intervals = (private.get("gt_attack_intervals_epoch") or {}).get(str(gpu_id))
        if intervals is None:
            intervals = (private.get("gt_attack_intervals_epoch") or {}).get(gpu_id) or []
        if role == "companion_idle":
            gt_label = "normal_idle"
            gt_is_attack = False
            gt_variant = None
        else:
            gt_label = private.get("gt_label")
            gt_is_attack = bool(private.get("gt_is_attack", not str(gt_label).startswith("normal")))
            gt_variant = private.get("gt_variant")
        label_rows.append(
            {
                "session_id": session_id,
                "gpu_id": gpu_id,
                "gt_label": gt_label,
                "gt_is_attack": gt_is_attack,
                "gt_variant": gt_variant,
                "gt_params_json": private.get("gt_params_json"),
                "gt_attack_intervals_epoch": json.dumps(intervals),
                "gpu_role": role,
                "group_id": private.get("group_id"),
                "capture_key": private.get("capture_key"),
                "seed": private.get("seed"),
                "declared_policy": private.get("declared_policy"),
                "workload_module": private.get("workload_module"),
                "workload_cmd": private.get("workload_cmd"),
                "host_workload": private.get("host_workload"),
                "run_id": private.get("run_id"),
                "source": private.get("source", "real"),
                "created_epoch": private.get("created_epoch"),
                "valid": True,
                "invalid_reason": "",
            }
        )

    validation = validate_session_dir(
        session_dir,
        labels_row=label_rows[0] if label_rows else None,
        private_capture=private,
    )
    for row in label_rows:
        row["valid"] = validation.valid
        row["invalid_reason"] = "; ".join(validation.reasons)
    _upsert_labels(paths["labels"], paths["labels_lock"], label_rows)

    # archive private capture
    archive = dict(private)
    archive["collector_result"] = collector_result
    archive["finalize_validation"] = {"valid": validation.valid, "reasons": validation.reasons}
    _atomic_write_json(paths["private_capture"], archive, mode=0o600)
    if validation.valid and paths["staging_dir"].exists():
        staging_private = paths["staging_dir"] / "capture_private.json"
        if staging_private.exists():
            staging_private.unlink()
        # leave collector_result for debugging only if invalid; on valid remove staging dir if empty-ish
        try:
            shutil.rmtree(paths["staging_dir"])
        except OSError:
            pass

    return {
        "session_id": session_id,
        "valid": validation.valid,
        "reasons": validation.reasons,
        "progress_masked": decision.progress_log_masked,
        "labels_rows": len(label_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--warmup-s", type=float, default=DEFAULT_WARMUP_S)
    parser.add_argument("--drop-prob", type=float, default=DEFAULT_DROP_PROB)
    parser.add_argument("--policy-seed", type=int, default=7)
    args = parser.parse_args()
    report = finalize_session(
        args.session_id,
        root=args.root,
        warmup_s=args.warmup_s,
        drop_prob=args.drop_prob,
        policy_seed=args.policy_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
