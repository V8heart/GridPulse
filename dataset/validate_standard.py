"""Standard compliance checks for gp-telemetry/1.2 sessions."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset.identifiers import SESSION_ID_RE
from workloads.progress_log import ALLOWED_EVENTS

SCHEMA_VERSION = "gp-telemetry/1.2"

FORBIDDEN_EXACT = {
    "attack",
    "swma",
    "ltma",
    "crypto",
    "malicious",
    "unknown_cuda",
    "synthetic_",
    "syn-",
    "real-",
    "cryptojacking",
}

# Full label / variant strings from §8 (matched case-insensitively as whole tokens
# only when they appear as standalone forbidden phrases in free text).
FORBIDDEN_LABELS = {
    "normal_idle",
    "normal_mlp_small",
    "normal_resnet_train",
    "normal_llm_pretrain_ddp",
    "normal_llm_pretrain_fsdp",
    "normal_llm_flat_pretrain",
    "normal_llm_finetune",
    "normal_llm_inference_serving",
    "normal_llm_inference_batch",
    "normal_hpo_search",
    "normal_dataloader_bound",
    "normal_mixed_tenants",
    "swma_basic",
    "swma_shallow",
    "swma_jitter",
    "swma_piggyback",
    "swma_mimicry",
    "swma_coordinated",
    "ltma_basic",
    "crypto_flat",
    "swma",
    "ltma",
    "cryptojacking",
}

REQUIRED_TELEMETRY = {
    "timestamp",
    "t_epoch",
    "session_id",
    "gpu_id",
    "sample_hz",
    "power_w",
    "util_gpu_pct",
    "mem_copy_util_pct",
    "sm_clock_mhz",
    "temp_c",
    "fb_used_mb",
}


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    session_id: str | None = None

    def fail(self, reason: str) -> None:
        self.valid = False
        self.reasons.append(reason)


def contains_forbidden_token(text: str) -> str | None:
    lower = str(text).lower()
    for token in FORBIDDEN_EXACT:
        if token in lower:
            return token
    for label in FORBIDDEN_LABELS:
        if label.lower() in lower:
            return label
    return None


def _check_strings(frame: pd.DataFrame, columns: list[str], result: ValidationResult) -> None:
    for col in columns:
        if col not in frame:
            continue
        for value in frame[col].dropna().astype(str).unique():
            hit = contains_forbidden_token(value)
            if hit:
                result.fail(f"forbidden token {hit!r} in column {col}")


def validate_session_dir(
    session_dir: Path,
    *,
    labels_row: dict[str, Any] | None = None,
    private_capture: dict[str, Any] | None = None,
) -> ValidationResult:
    result = ValidationResult(valid=True, session_id=session_dir.name)
    if session_dir.name in {".staging", "private"} or ".staging" in session_dir.parts or "private" in session_dir.parts:
        result.fail("session path must not be under .staging or private")
        return result

    telemetry_path = session_dir / "telemetry.csv"
    session_json_path = session_dir / "session.json"
    if not telemetry_path.exists():
        result.fail("missing telemetry.csv")
        return result
    if not session_json_path.exists():
        result.fail("missing session.json")
        return result

    try:
        frame = pd.read_csv(telemetry_path)
    except Exception as exc:  # noqa: BLE001
        result.fail(f"telemetry unreadable: {exc}")
        return result

    missing = REQUIRED_TELEMETRY - set(frame.columns)
    if missing:
        result.fail(f"missing columns: {sorted(missing)}")

    session_id = str(frame["session_id"].iloc[0]) if "session_id" in frame and len(frame) else session_dir.name
    result.session_id = session_id
    if not SESSION_ID_RE.fullmatch(session_id):
        result.fail(f"bad session_id {session_id!r}")

    meta = json.loads(session_json_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        result.fail(f"session.json schema_version must be {SCHEMA_VERSION}")
    blob = json.dumps(meta, ensure_ascii=False)
    hit = contains_forbidden_token(blob)
    if hit:
        result.fail(f"forbidden token {hit!r} in session.json")
    for key in meta:
        if str(key).startswith("gt_") or key in {"gpu_role", "workload_cmd", "label"}:
            result.fail(f"gt/leak key in session.json: {key}")

    if "timestamp" in frame and "t_epoch" in frame:
        for (sid, gid), group in frame.groupby(["session_id", "gpu_id"]):
            ts = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
            te = pd.to_numeric(group["t_epoch"], errors="coerce").to_numpy()
            if len(ts) > 1 and np.any(np.diff(ts) < 0):
                result.fail(f"timestamp not monotonic for {(sid, gid)}")
            delta = te - ts
            if len(delta) and np.nanmax(np.abs(delta - np.nanmedian(delta))) > 1e-3:
                result.fail(f"t_epoch-timestamp not constant for {(sid, gid)}")
            hz = pd.to_numeric(group["sample_hz"], errors="coerce").dropna()
            if hz.nunique() != 1:
                result.fail(f"sample_hz not unique for {(sid, gid)}")
            actual = pd.to_numeric(group["actual_interval_ms"], errors="coerce").dropna()
            if len(actual) and float(hz.iloc[0]) > 0:
                median = float(actual.median())
                expected = 1000.0 / float(hz.iloc[0])
                if median > 0 and abs(median - expected) / expected > 0.05:
                    result.fail(f"sample_hz mismatch for {(sid, gid)}")

    for col, lo, hi in (("util_gpu_pct", 0, 100), ("mem_copy_util_pct", 0, 100), ("temp_c", 0, 110)):
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if ((values < lo) | (values > hi)).any():
                result.fail(f"{col} out of range")
    if "power_w" in frame and (pd.to_numeric(frame["power_w"], errors="coerce") < 0).any():
        result.fail("power_w negative")

    _check_strings(
        frame,
        [
            "declared_job_type",
            "declared_process_name",
            "declared_user",
            "observed_process_name",
            "gpu_model",
            "session_id",
        ],
        result,
    )

    progress_path = session_dir / "progress.jsonl"
    if progress_path.exists():
        events = []
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(json.loads(line))
        duration = float(meta.get("duration_s") or frame["timestamp"].max())
        for event in events:
            if event.get("event") not in ALLOWED_EVENTS:
                result.fail(f"bad progress event {event.get('event')!r}")
            t = event.get("t")
            if t is not None and not (0 <= float(t) <= duration + 5):
                result.fail(f"progress t out of range: {t}")
        # training declared => need step_end >= 2 when present
        if "declared_job_family" in frame:
            families = set(frame["declared_job_family"].dropna().astype(str))
            if "training" in families:
                n_step = sum(1 for e in events if e.get("event") == "step_end")
                if n_step < 2:
                    result.fail("training declared but fewer than 2 step_end events")

    # target GPU requirement from labels/private when provided
    roles = []
    if labels_row and "gpu_role" in labels_row:
        roles.append(str(labels_row["gpu_role"]))
    if private_capture and "gpu_roles" in private_capture:
        roles.extend(str(v) for v in private_capture["gpu_roles"].values())
    if roles and "target" not in roles:
        result.fail("session has no target GPU")

    raw_warmup = meta.get("warmup_s")
    warmup_s = 180.0 if raw_warmup is None else float(raw_warmup)
    if "timestamp" in frame and float(frame["timestamp"].max()) <= warmup_s:
        result.fail("session duration not longer than warmup_s")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="session dir or sessions/ root")
    args = parser.parse_args()
    targets = []
    if (args.path / "telemetry.csv").exists():
        targets = [args.path]
    else:
        targets = sorted(p for p in args.path.glob("s-*") if p.is_dir())
    rows = []
    for target in targets:
        result = validate_session_dir(target)
        rows.append(
            {
                "session_id": result.session_id,
                "valid": result.valid,
                "reasons": "; ".join(result.reasons),
            }
        )
        status = "OK" if result.valid else "FAIL"
        reason_text = "; ".join(result.reasons)
        print(f"{status}\t{result.session_id}\t{reason_text}")
    invalid = sum(1 for r in rows if not r["valid"])
    raise SystemExit(1 if invalid else 0)


if __name__ == "__main__":
    main()
