"""Contextual Stage 1 v2 screening for AI datacenter telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.baseline import CohortBaseline, FEATURES_V2
from pipeline.changepoint import cusum_changepoints
from pipeline.context_evidence import explained_changepoints, period_match, read_progress_log, step_period_from_log
from pipeline.evidence_vocab import to_bool_evidence
from pipeline.features import compute_window_features_v2

DEFAULT_CONFIG = {
    "alpha": 0.05,
    "alpha_high_impact": 0.10,
    "tdp_w": 450.0,
    "nvml_avg_window_s": 1.0,
    "tau_peak": 8.0,
    "tau_sync": 0.8,
    "evidence_weights": {
        "period_mismatch": 1.0,
        "unexplained_changepoint": 0.8,
        "cross_job_sync": 1.0,
        "progress_log_missing": 0.25,
        "sustained_high_load": 1.0,
        "flat_power": 0.8,
        "high_ramp": 0.8,
        "util_power_decoupled": 1.0,
        "declared_family_mismatch": 1.0,
    },
    "evidence_thresholds": {},
    "grid_weights": {"0.05_0.1": 0.5, "0.1_0.7": 1.0, "0.7_2": 0.7, "2_nyq": 0.2},
}


def load_config(path: str | Path | None = None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and Path(path).exists():
        try:
            import yaml

            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            loaded = {}
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def flatten_features(feats: dict) -> dict:
    out = {k: v for k, v in feats.items() if not isinstance(v, dict)}
    for prefix in ("band_frac", "band_power_w2", "band_reliability"):
        for key, value in feats.get(prefix, {}).items():
            out[f"{prefix}_{key}"] = value
    return out


def conformal_pvalue(score: float, calibration_scores: list[float]) -> float:
    values = np.asarray(calibration_scores, dtype=float)
    values = values[np.isfinite(values)]
    return float((1 + np.sum(values >= score)) / (len(values) + 1)) if len(values) else 1.0


def impact_level(impact_raw: float, quantiles: dict | None) -> str:
    if not quantiles:
        return "high" if impact_raw >= 0.5 else "medium" if impact_raw >= 0.2 else "low"
    if impact_raw >= quantiles.get("p90", np.inf):
        return "high"
    if impact_raw >= quantiles.get("p50", np.inf):
        return "medium"
    return "low"


def impact_components(row: dict, config: dict) -> dict:
    weights = config.get("grid_weights", {})
    band_frac = {k.removeprefix("band_frac_"): v for k, v in row.items() if k.startswith("band_frac_")}
    band_rel = {k.removeprefix("band_reliability_"): v for k, v in row.items() if k.startswith("band_reliability_")}
    weighted = sum(float(band_frac.get(k, 0.0)) * float(weights.get(k, 0.0)) * float(band_rel.get(k, 1.0)) for k in weights)
    return {
        "swing_frac_tdp": float(row.get("swing_frac_tdp", 0.0)),
        "ramp_frac_tdp_s": float(row.get("ramp_p95_w_per_s", 0.0)) / float(config.get("tdp_w", 450.0)),
        "impact_raw": float(np.sqrt(max(weighted, 0.0))),
        "multi_gpu_sync_index": float(row.get("multi_gpu_sync_index", 0.0)),
        "cross_job_sync_index": float(row.get("cross_job_sync_index", 0.0)),
    }


def score_window(row: dict, baseline: CohortBaseline, calibration: dict, config: dict) -> dict:
    z = baseline.robust_z(row)
    z_values = {k: abs(v) for k, v in z.items() if k != "baseline_level" and k in FEATURES_V2}
    raw_evidence = dict(row.get("evidence", {}))
    family_stats = baseline.declared_family_mismatch_stats(row)
    raw_evidence.update({k: v for k, v in family_stats.items() if v is not None})
    bool_evidence = to_bool_evidence({**row, **raw_evidence}, config)
    weights = config.get("evidence_weights", {})
    evidence_score = 0.0
    for key in (
        "period_mismatch",
        "unexplained_changepoint",
        "cross_job_sync",
        "progress_log_missing",
        "sustained_high_load",
        "flat_power",
        "high_ramp",
        "util_power_decoupled",
        "declared_family_mismatch",
    ):
        if bool_evidence.get(key):
            if key == "progress_log_missing" and row.get("declared_job_family") != "training":
                continue
            evidence_score += float(weights.get(key, 0.5))
    u_score = max(z_values.values(), default=0.0) + evidence_score
    p_value = conformal_pvalue(u_score, calibration.get("normal_u_scores", []))
    comps = impact_components(row, config)
    raw_impact = max(comps.values(), default=0.0)
    level = impact_level(raw_impact, calibration.get("impact_quantiles"))
    is_candidate = p_value <= config.get("alpha", 0.05) or (
        level == "high" and p_value <= config.get("alpha_high_impact", 0.10)
    )
    return {
        "u_score": float(u_score),
        "p_value": p_value,
        "baseline_level": z.get("baseline_level", "global"),
        "robust_z": z_values,
        "impact_components": comps,
        "impact_level": level,
        "is_candidate": bool(is_candidate),
        "grid_watch": bool(level == "high" and not is_candidate),
        "candidate_reasons": sorted(z_values, key=z_values.get, reverse=True)[:3],
        "evidence_bool": bool_evidence,
        "declared_family_mean_abs_z": family_stats.get("declared_family_mean_abs_z"),
        "best_other_family_mean_abs_z": family_stats.get("best_other_family_mean_abs_z"),
    }


def build_windows(
    df: pd.DataFrame,
    *,
    window_s: float,
    stride_s: float,
    progress_log_dir: str | Path | None,
    config: dict,
) -> pd.DataFrame:
    rows = []
    label_col = "gt_label" if "gt_label" in df else "label"
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        hz = float(group["sample_hz"].dropna().iloc[0]) if "sample_hz" in group else 1.0
        window = max(8, int(round(window_s * hz)))
        stride = max(1, int(round(stride_s * hz)))
        events = read_progress_log(Path(progress_log_dir) / f"{session_id}.jsonl" if progress_log_dir else None)
        for start in range(0, len(group) - window + 1, stride):
            subset = group.iloc[start:start + window]
            t0, t1 = float(subset["timestamp"].iloc[0]), float(subset["timestamp"].iloc[-1])
            feats = compute_window_features_v2(
                subset["power_w"].to_numpy(),
                subset["util_gpu_pct"].to_numpy() if "util_gpu_pct" in subset else None,
                sample_hz=hz,
                tdp_w=float(config.get("tdp_w", 450.0)),
                nvml_avg_window_s=float(config.get("nvml_avg_window_s", 1.0)),
            )
            flat = flatten_features(feats)
            cps = cusum_changepoints(subset["power_w"].to_numpy(), sample_hz=hz)
            step_period, n_steps, step_cv = step_period_from_log(events, t0, t1)
            matched = period_match(float(flat.get("dominant_freq_hz", 0.0)), step_period)
            explained = explained_changepoints([t0 + cp for cp in cps], events)
            raw_evidence = {
                "period_match": matched,
                "period_mismatch": matched is False and flat.get("dominant_peak_prominence", 0.0) >= config.get("tau_peak", 8.0),
                "step_period_s": step_period,
                "step_count": n_steps,
                "step_period_cv": step_cv,
                "changepoints": [t0 + cp for cp in cps],
                "unexplained_changepoint_count": len(explained["unexplained"]),
                "progress_log_available": bool(events),
                "progress_log_missing": not bool(events),
                "cross_job_sync_index": flat.get("cross_job_sync_index", 0.0),
                "dominant_peak_prominence": flat.get("dominant_peak_prominence", 0.0),
                "high_load_fraction": flat.get("high_load_fraction", 0.0),
                "longest_high_seconds": flat.get("longest_high_seconds", 0.0),
                "swing_abs_w": flat.get("swing_abs_w", 0.0),
                "mean_w": flat.get("mean_w", 0.0),
                "ramp_p95_w_per_s": flat.get("ramp_p95_w_per_s", flat.get("ramp_max_w_per_s", 0.0)),
                "util_residual_mad_w": flat.get("util_residual_mad_w", 0.0),
            }
            bool_evidence = to_bool_evidence(raw_evidence, config)
            flat.update({
                "session_id": str(session_id),
                "gpu_id": int(gpu_id),
                "sample_hz": hz,
                "start": start,
                "end": start + window,
                "window_start_s": t0,
                "window_end_s": t1,
                "gt_label": str(subset[label_col].iloc[0]) if label_col in subset else None,
                "declared_job_type": str(subset.get("declared_job_type", pd.Series(["unknown"])).iloc[0]),
                "declared_job_family": str(subset.get("declared_job_family", pd.Series(["unknown"])).iloc[0]),
                "gpu_model": str(subset.get("gpu_model", pd.Series(["unknown"])).iloc[0]),
                "evidence": {**raw_evidence, **bool_evidence},
                "evidence_bool": bool_evidence,
            })
            rows.append(flat)
    return pd.DataFrame(rows)
