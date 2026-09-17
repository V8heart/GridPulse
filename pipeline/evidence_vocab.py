"""Single source of truth for Stage1/Stage2 evidence names."""
from __future__ import annotations

from typing import Any

# Bool names allowed in corpus checklists and Stage2 matched_evidence.
ALLOWED_EVIDENCE_NAMES: frozenset[str] = frozenset(
    {
        "period_match",
        "period_mismatch",
        "unexplained_changepoint",
        "cross_job_sync",
        "progress_log_missing",
        "progress_log_available",
        "strong_peak",
        "sustained_high_load",
        "flat_power",
        "high_ramp",
        "util_power_decoupled",
        "declared_family_mismatch",
    }
)

# Explicitly forbidden (must not appear in corpus).
FORBIDDEN_EVIDENCE_NAMES: frozenset[str] = frozenset(
    {
        "feature_range_match",
        "progress_log_present_training",
        "context_inconsistency",
        "declared_context_consistent",
        "util_residual_anomaly",
        "persistent_high_load",
        "explained_changepoint",
        "impedance_resonance",
        "inertia_reduction",
    }
)

DEFAULT_EVIDENCE_THRESHOLDS: dict[str, Any] = {
    "tau_peak": 8.0,
    "tau_sync": 0.8,
    "sustained_high_load": {
        "high_load_fraction_min": 0.85,
        "longest_high_seconds_min": 8.0,
    },
    "flat_power": {
        "swing_abs_w_max": 40.0,
        "mean_w_min": 250.0,
    },
    "high_ramp": {
        "ramp_p95_w_per_s_min": 800.0,
    },
    "util_power_decoupled": {
        "util_residual_mad_w_min": 25.0,
    },
    "declared_family_mismatch": {
        "declared_mean_abs_z_min": 3.0,
        "other_family_margin": 0.5,
    },
}


def merge_thresholds(config: dict | None = None) -> dict[str, Any]:
    """Merge defaults with stage1 config / fitted thresholds."""
    out = json_deepcopy(DEFAULT_EVIDENCE_THRESHOLDS)
    if not config:
        return out
    nested = config.get("evidence_thresholds", config)
    for key, value in nested.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    if "tau_peak" in config:
        out["tau_peak"] = config["tau_peak"]
    if "tau_sync" in config:
        out["tau_sync"] = config["tau_sync"]
    return out


def json_deepcopy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_deepcopy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_deepcopy(v) for v in value]
    return value


def to_bool_evidence(raw: dict, config: dict | None = None) -> dict[str, bool]:
    """Convert Stage1 raw evidence/features into checklist bools."""
    thr = merge_thresholds(config)
    peak = float(raw.get("dominant_peak_prominence", raw.get("strong_peak_score", 0.0)) or 0.0)
    period_match = raw.get("period_match")
    if period_match is not None and not isinstance(period_match, bool):
        period_match = bool(period_match)

    unexplained_count = int(raw.get("unexplained_changepoint_count", 0) or 0)
    cross_sync = float(raw.get("cross_job_sync_index", 0.0) or 0.0)
    progress_missing = bool(raw.get("progress_log_missing", False))
    progress_available = bool(raw.get("progress_log_available", not progress_missing))

    high_frac = float(raw.get("high_load_fraction", 0.0) or 0.0)
    longest_high = float(raw.get("longest_high_seconds", 0.0) or 0.0)
    swing_abs = float(
        raw.get(
            "swing_abs_w",
            (raw.get("swing_ratio", 0.0) or 0.0) * max(float(raw.get("mean_w", 0.0) or 0.0), 1.0),
        )
        or 0.0
    )
    mean_w = float(raw.get("mean_w", 0.0) or 0.0)
    ramp_p95 = float(raw.get("ramp_p95_w_per_s", raw.get("ramp_max_w_per_s", 0.0)) or 0.0)
    util_mad = float(raw.get("util_residual_mad_w", 0.0) or 0.0)

    sh = thr["sustained_high_load"]
    flat = thr["flat_power"]
    ramp = thr["high_ramp"]
    decoupled = thr["util_power_decoupled"]
    mismatch_thr = thr["declared_family_mismatch"]
    tau_peak = float(thr["tau_peak"])
    tau_sync = float(thr["tau_sync"])

    strong_peak = peak >= tau_peak
    period_mismatch = bool(raw.get("period_mismatch")) if "period_mismatch" in raw else (
        period_match is False and strong_peak
    )

    if "declared_family_mismatch" in raw and isinstance(raw.get("declared_family_mismatch"), bool):
        family_mismatch = bool(raw["declared_family_mismatch"])
    else:
        declared_z = float(raw.get("declared_family_mean_abs_z", 0.0) or 0.0)
        best_other = float(raw.get("best_other_family_mean_abs_z", 1e9) or 1e9)
        family_mismatch = (
            declared_z >= float(mismatch_thr["declared_mean_abs_z_min"])
            and best_other + float(mismatch_thr["other_family_margin"]) < declared_z
        )

    return {
        "period_match": bool(period_match) if period_match is not None else False,
        "period_mismatch": bool(period_mismatch),
        "unexplained_changepoint": unexplained_count >= 1,
        "cross_job_sync": cross_sync >= tau_sync,
        "progress_log_missing": progress_missing,
        "progress_log_available": progress_available,
        "strong_peak": strong_peak,
        "sustained_high_load": (
            high_frac >= float(sh["high_load_fraction_min"])
            and longest_high >= float(sh["longest_high_seconds_min"])
        ),
        "flat_power": (
            swing_abs <= float(flat["swing_abs_w_max"]) and mean_w >= float(flat["mean_w_min"])
        ),
        "high_ramp": ramp_p95 >= float(ramp["ramp_p95_w_per_s_min"]),
        "util_power_decoupled": util_mad >= float(decoupled["util_residual_mad_w_min"]),
        "declared_family_mismatch": family_mismatch,
    }


def assert_allowed_evidence_name(name: str) -> None:
    if name in FORBIDDEN_EVIDENCE_NAMES:
        raise ValueError(f"forbidden evidence name: {name}")
    if name not in ALLOWED_EVIDENCE_NAMES:
        raise ValueError(f"unknown evidence name: {name}")
