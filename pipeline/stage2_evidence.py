"""Build structured Stage 2 evidence bundles from Stage 1 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RAW_FORBIDDEN = {"power_w", "util_gpu_pct", "power", "util", "timeseries", "andes_log", "embeddings"}


def build_evidence_bundle(row: dict, *, drop_progress_log_missing: bool = False) -> dict:
    """Create an LLM-safe evidence bundle without raw telemetry arrays."""
    from pipeline.evidence_vocab import to_bool_evidence

    physics = None
    if any(str(key).startswith("physics_") for key in row):
        physics = {
            "tier": "B",
            "converged": row.get("physics_converged"),
            "osc_std": row.get("physics_osc_std"),
            "rocof_hz_s": row.get("physics_rocof_hz_s"),
            "dominant_freq_hz": row.get("physics_dominant_freq_hz"),
            "note": row.get("physics_failure_reason") or "event-triggered public test-system replay",
        }

    features = dict(row.get("features") or {})
    raw_evidence = row.get("evidence_bool") or row.get("evidence", {})
    # Prefer merged features + row fields so v2 keys reach to_bool_evidence.
    merged_for_bool = {**features, **row}
    if raw_evidence and not all(isinstance(v, bool) for v in raw_evidence.values() if v is not None):
        bool_evidence = to_bool_evidence({**merged_for_bool, **dict(raw_evidence)})
    elif raw_evidence and all(isinstance(v, bool) for v in raw_evidence.values() if v is not None):
        bool_evidence = dict(raw_evidence)
    else:
        bool_evidence = to_bool_evidence(merged_for_bool)
    if drop_progress_log_missing:
        bool_evidence.pop("progress_log_missing", None)

    declared = row.get("declared_context") if isinstance(row.get("declared_context"), dict) else {}
    bundle = {
        "window_id": row.get("window_id"),
        "declared_context": {
            "job_type": declared.get("job_type", row.get("declared_job_type")),
            "job_family": declared.get("job_family", row.get("declared_job_family")),
            "user": declared.get("user", row.get("declared_user")),
        },
        "impact": {
            "level": row.get("impact_level"),
            "components": row.get("impact_components", {}),
        },
        "unexplainedness": {
            "u_score": row.get("u_score"),
            "p_value": row.get("p_value"),
            "evidence": bool_evidence,
        },
        "physics": physics,
    }
    return _strip_forbidden(bundle)


def _strip_forbidden(value):
    if isinstance(value, dict):
        return {k: _strip_forbidden(v) for k, v in value.items() if k not in RAW_FORBIDDEN}
    if isinstance(value, list):
        return [_strip_forbidden(item) for item in value if not isinstance(item, (bytes, bytearray))]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", [])
    bundles = [build_evidence_bundle(row) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(bundles)} evidence bundles: {args.out}")


if __name__ == "__main__":
    main()
