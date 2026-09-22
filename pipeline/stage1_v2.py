"""Contextual Stage 1 v2 screening for AI datacenter telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.window_truth import label_window, parse_intervals
from pipeline.baseline import CohortBaseline, FEATURES_V2
from pipeline.changepoint import cusum_changepoints
from pipeline.context_evidence import (
    explained_changepoints,
    period_match,
    read_progress_log,
    read_session_progress,
    step_period_from_log,
)
from pipeline.evidence_vocab import (
    BUILD_DEFERRED_BOOL_KEYS,
    strip_recompute_bools,
    to_bool_evidence,
)
from pipeline.features import compute_window_features_v2
from pipeline.synchronization import session_window_sync_metrics

DEFAULT_CONFIG = {
    "alpha": 0.05,
    "alpha_high_impact": 0.10,
    "tdp_w": 450.0,
    "nvml_avg_window_s": 1.0,
    "tau_peak": 4.0,
    "tau_sync": 0.8,
    "util_min_range_pct": 5.0,
    "score_mode": "tail_min_plus_evidence",
    "mondrian_min_n": 30,
    "evidence_weights": {
        "period_mismatch": 1.0,
        "unexplained_changepoint": 0.8,
        "cross_job_sync": 0.0,
        "progress_log_missing": 0.0,
        "sustained_high_load": 1.0,
        "flat_power": 0.8,
        "high_ramp": 0.8,
        "util_power_decoupled": 1.0,
        "declared_family_mismatch": 1.0,
    },
    "evidence_thresholds": {},
    "evidence_gate": {
        "prune_auc_min": 0.60,
        "activate_attack_rate_min": 0.20,
        "activate_rate_ratio_min": 2.0,
    },
    "grid_weights": {"0.05_0.1": 0.5, "0.1_0.7": 1.0, "0.7_2": 0.7, "2_nyq": 0.2},
    "grid_weights_provenance": "provisional_placeholder",
    "progress_log_policy_required": "v1_eligible_uniform_drop",
}

CALIBRATION_SCHEMA_VERSION = "mondrian_v1"
EVIDENCE_WEIGHT_CAP = 3.0
# Occurrence = suspicion; benign availability signals never add to u_score.
SUSPICIOUS_EVIDENCE_KEYS = (
    "period_mismatch",
    "unexplained_changepoint",
    "cross_job_sync",
    "progress_log_missing",
    "sustained_high_load",
    "flat_power",
    "high_ramp",
    "util_power_decoupled",
    "declared_family_mismatch",
    "strong_peak",
)


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


def load_truth_index(
    labels_csv: str | Path,
    *,
    sessions_root: str | Path | None = None,
) -> dict[tuple[str, int], dict]:
    """Load evaluation-only labels keyed by ``(session_id, gpu_id)``."""
    labels = pd.read_csv(labels_csv, low_memory=False)
    if "valid" in labels.columns:
        valid = labels["valid"].astype(str).str.lower().isin({"true", "1"})
        labels = labels[valid]
    index: dict[tuple[str, int], dict] = {}
    roots = Path(sessions_root) if sessions_root else None
    for _, row in labels.iterrows():
        session_id = str(row["session_id"])
        gpu_id = int(row["gpu_id"])
        raw_intervals = row.get("gt_attack_intervals_epoch")
        intervals = None if pd.isna(raw_intervals) else parse_intervals(raw_intervals)
        t0_epoch = None
        if roots is not None:
            metadata_path = roots / session_id / "session.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("t0_epoch") is not None:
                    t0_epoch = float(metadata["t0_epoch"])
        host = row.get("host_workload")
        index[(session_id, gpu_id)] = {
            "intervals": intervals,
            "session_gt_label": str(row["gt_label"]),
            "host_workload": None if pd.isna(host) else str(host),
            "hosted": bool(not pd.isna(host) and str(host).strip()),
            "gpu_role": None if pd.isna(row.get("gpu_role")) else str(row.get("gpu_role")),
            "gt_variant": None if pd.isna(row.get("gt_variant")) else str(row.get("gt_variant")),
            "declared_policy": (
                None if pd.isna(row.get("declared_policy")) else str(row.get("declared_policy"))
            ),
            "group_id": None if pd.isna(row.get("group_id")) else str(row.get("group_id")),
            "source": None if pd.isna(row.get("source")) else str(row.get("source")),
            "t0_epoch": t0_epoch,
        }
    return index


def _window_epoch_bounds(subset: pd.DataFrame, truth: dict) -> tuple[float, float]:
    if "t_epoch" in subset.columns:
        epochs = pd.to_numeric(subset["t_epoch"], errors="coerce")
        if epochs.notna().all():
            return float(epochs.iloc[0]), float(epochs.iloc[-1])
    timestamps = pd.to_numeric(subset["timestamp"], errors="raise")
    t0_epoch = truth.get("t0_epoch")
    if t0_epoch is None and "t0_epoch" in subset.columns:
        origins = pd.to_numeric(subset["t0_epoch"], errors="coerce").dropna()
        if len(origins):
            t0_epoch = float(origins.iloc[0])
    if t0_epoch is None and float(timestamps.iloc[0]) > 1_000_000_000:
        return float(timestamps.iloc[0]), float(timestamps.iloc[-1])
    if t0_epoch is None:
        raise ValueError("interval truth requires t_epoch telemetry or session t0_epoch")
    return float(t0_epoch) + float(timestamps.iloc[0]), float(t0_epoch) + float(timestamps.iloc[-1])


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


def empirical_tail_p(value: float, reference: list[float] | np.ndarray) -> float:
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0
    return float((1 + np.sum(values >= float(value))) / (len(values) + 1))


def impact_level(impact_raw: float, quantiles: dict | None) -> str:
    if not quantiles:
        return "high" if impact_raw >= 0.5 else "medium" if impact_raw >= 0.2 else "low"
    if impact_raw >= quantiles.get("p90", np.inf):
        return "high"
    if impact_raw >= quantiles.get("p50", np.inf):
        return "medium"
    return "low"


def impact_components(row: dict, config: dict) -> dict:
    """Absolute band-energy impact; swing/ramp/sync are descriptive only."""
    weights = config.get("grid_weights", {})
    band_power = {
        k.removeprefix("band_power_w2_"): float(v)
        for k, v in row.items()
        if k.startswith("band_power_w2_")
    }
    band_rel = {
        k.removeprefix("band_reliability_"): float(v)
        for k, v in row.items()
        if k.startswith("band_reliability_")
    }
    tdp = float(config.get("tdp_w", 450.0)) or 450.0
    weighted = 0.0
    for key, weight in weights.items():
        weighted += float(band_power.get(key, 0.0)) * float(weight) * float(band_rel.get(key, 1.0))
    impact_raw = float(np.sqrt(max(weighted, 0.0)) / tdp)
    return {
        "impact_raw": impact_raw,
        "swing_frac_tdp": float(row.get("swing_frac_tdp", 0.0)),
        "ramp_frac_tdp_s": float(row.get("ramp_p95_w_per_s", 0.0)) / tdp,
        "multi_gpu_sync_index": float(row.get("multi_gpu_sync_index", 0.0) or 0.0),
        "cross_job_sync_index": float(row.get("cross_job_sync_index", 0.0) or 0.0),
    }


def require_calibration_schema(calibration: dict) -> None:
    version = calibration.get("schema_version")
    if version != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema_version={version!r}; "
            f"expected {CALIBRATION_SCHEMA_VERSION!r}. Refit with fit_stage1_v2.py."
        )


def _select_feature_refs(
    calibration: dict,
    family: str,
    feature: str,
    *,
    min_n: int,
) -> tuple[list[float], str]:
    by_family = (calibration.get("feature_abs_z") or {}).get(family) or {}
    refs = by_family.get(feature) or []
    if len(refs) >= min_n:
        return list(refs), "family"
    global_refs = ((calibration.get("feature_abs_z") or {}).get("global") or {}).get(feature) or []
    return list(global_refs), "global"


def _select_u_scores(
    calibration: dict,
    family: str,
    *,
    profile: str,
    min_n: int,
) -> tuple[list[float], str]:
    profiles = calibration.get("normal_u_scores") or {}
    bucket = profiles.get(profile) or {}
    refs = bucket.get(family) or []
    if len(refs) >= min_n:
        return list(refs), "family"
    return list(bucket.get("global") or []), "global"


def _active_evidence_keys(
    bool_evidence: dict,
    row: dict,
    weights: dict,
    *,
    progress_log_allowed: bool,
) -> list[str]:
    other_active = any(
        bool_evidence.get(k) and float(weights.get(k, 0.0)) > 0
        for k in SUSPICIOUS_EVIDENCE_KEYS
        if k != "progress_log_missing"
    )
    active = []
    for key in SUSPICIOUS_EVIDENCE_KEYS:
        if not bool_evidence.get(key):
            continue
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        if key == "progress_log_missing":
            if not progress_log_allowed:
                continue
            if row.get("declared_job_family") != "training" or not other_active:
                continue
        active.append(key)
    return active


def combine_u_score(
    z_values: dict[str, float],
    bool_evidence: dict,
    row: dict,
    calibration: dict,
    config: dict,
    *,
    include_evidence: bool = True,
    leave_out_feature_z: dict[str, float] | None = None,
) -> dict:
    """Tail-min feature score + optional normal-surprisal evidence."""
    min_n = int(config.get("mondrian_min_n", 30))
    family = str(row.get("declared_job_family", "unknown"))
    feature_ps = {}
    levels = []
    for feature, abs_z in z_values.items():
        refs, level = _select_feature_refs(calibration, family, feature, min_n=min_n)
        if leave_out_feature_z and feature in leave_out_feature_z and refs:
            # Leave-one-out: drop one matching value (closest) for cal self-scoring.
            target = float(leave_out_feature_z[feature])
            arr = list(refs)
            best_i = min(range(len(arr)), key=lambda i: abs(arr[i] - target))
            arr.pop(best_i)
            refs = arr
        feature_ps[feature] = empirical_tail_p(abs_z, refs)
        levels.append(level)
    if feature_ps:
        min_p = min(feature_ps.values())
        feature_score = float(-np.log(max(min_p, 1e-12)))
        driving = min(feature_ps, key=feature_ps.get)
    else:
        min_p = 1.0
        feature_score = 0.0
        driving = None

    weights = dict(calibration.get("evidence_surprisal_weights") or config.get("evidence_weights") or {})
    policy_ok = bool(calibration.get("progress_log_policy_ok", False))
    # Until synth is regenerated under required policy, force weight 0.
    if not policy_ok:
        weights["progress_log_missing"] = 0.0
    evidence_score = 0.0
    active = []
    if include_evidence:
        active = _active_evidence_keys(
            bool_evidence, row, weights, progress_log_allowed=policy_ok
        )
        evidence_score = float(sum(float(weights.get(k, 0.0)) for k in active))
    return {
        "u_score": float(feature_score + evidence_score),
        "feature_score": feature_score,
        "evidence_score": evidence_score,
        "feature_tail_p": feature_ps,
        "min_feature_tail_p": float(min_p),
        "driving_feature": driving,
        "active_evidence": active,
        "feature_tail_level": "family" if levels and all(x == "family" for x in levels) else (
            "mixed" if levels and any(x == "family" for x in levels) else "global"
        ),
    }


def score_window(
    row: dict,
    baseline: CohortBaseline,
    calibration: dict,
    config: dict,
    *,
    profile: str = "full",
    leave_out_feature_z: dict[str, float] | None = None,
) -> dict:
    require_calibration_schema(calibration)
    z = baseline.robust_z(row)
    z_values = {k: abs(v) for k, v in z.items() if k != "baseline_level" and k in FEATURES_V2}
    raw_evidence = dict(row.get("evidence", {}))
    family_stats = baseline.declared_family_mismatch_stats(row)
    raw_evidence.update({k: v for k, v in family_stats.items() if v is not None})
    merged = strip_recompute_bools({**row, **raw_evidence})
    bool_evidence = to_bool_evidence(merged, config)
    include_evidence = profile != "no_evidence"
    combined = combine_u_score(
        z_values,
        bool_evidence,
        row,
        calibration,
        config,
        include_evidence=include_evidence,
        leave_out_feature_z=leave_out_feature_z,
    )
    u_score = combined["u_score"]
    family = str(row.get("declared_job_family", "unknown"))
    min_n = int(config.get("mondrian_min_n", 30))
    cal_scores, cal_level = _select_u_scores(calibration, family, profile=profile, min_n=min_n)
    p_value = conformal_pvalue(u_score, cal_scores)
    comps = impact_components(row, config)
    raw_impact = float(comps.get("impact_raw", 0.0))
    level = impact_level(raw_impact, calibration.get("impact_quantiles"))
    is_candidate = p_value <= config.get("alpha", 0.05) or (
        level == "high" and p_value <= config.get("alpha_high_impact", 0.10)
    )
    max_abs_z = max(z_values.values(), default=0.0)
    return {
        "u_score": float(u_score),
        "p_value": p_value,
        "baseline_level": z.get("baseline_level", "global"),
        "robust_z": z_values,
        "max_abs_z": float(max_abs_z),
        "impact_components": comps,
        "impact_raw": raw_impact,
        "impact_level": level,
        "is_candidate": bool(is_candidate),
        "grid_watch": bool(level == "high" and not is_candidate),
        "candidate_reasons": sorted(z_values, key=z_values.get, reverse=True)[:3],
        "evidence_bool": bool_evidence,
        "declared_family_mean_abs_z": family_stats.get("declared_family_mean_abs_z"),
        "best_other_family_mean_abs_z": family_stats.get("best_other_family_mean_abs_z"),
        "feature_score": combined["feature_score"],
        "evidence_score": combined["evidence_score"],
        "feature_tail_p": combined["feature_tail_p"],
        "driving_feature": combined["driving_feature"],
        "active_evidence": combined["active_evidence"],
        "mondrian_level": cal_level,
        "feature_tail_level": combined["feature_tail_level"],
        "score_profile": profile,
    }


def build_windows(
    df: pd.DataFrame,
    *,
    window_s: float,
    stride_s: float,
    config: dict,
    progress_log_dir: str | Path | None = None,
    sessions_root: str | Path | None = None,
    truth_index: dict[tuple[str, int], dict] | None = None,
) -> pd.DataFrame:
    rows = []
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        hz = float(group["sample_hz"].dropna().iloc[0]) if "sample_hz" in group else 1.0
        window = max(8, int(round(window_s * hz)))
        stride = max(1, int(round(stride_s * hz)))
        if sessions_root is not None:
            events = read_session_progress(Path(sessions_root) / str(session_id))
        else:
            events = read_progress_log(
                Path(progress_log_dir) / f"{session_id}.jsonl" if progress_log_dir else None
            )
        truth = None
        if truth_index is not None:
            key = (str(session_id), int(gpu_id))
            if key not in truth_index:
                raise KeyError(f"missing truth for session/GPU {key!r}")
            truth = truth_index[key]
        for start in range(0, len(group) - window + 1, stride):
            subset = group.iloc[start:start + window]
            t0, t1 = float(subset["timestamp"].iloc[0]), float(subset["timestamp"].iloc[-1])
            feats = compute_window_features_v2(
                subset["power_w"].to_numpy(),
                subset["util_gpu_pct"].to_numpy() if "util_gpu_pct" in subset else None,
                sample_hz=hz,
                tdp_w=float(config.get("tdp_w", 450.0)),
                nvml_avg_window_s=float(config.get("nvml_avg_window_s", 1.0)),
                util_min_range_pct=float(config.get("util_min_range_pct", 5.0)),
            )
            flat = flatten_features(feats)
            cps = cusum_changepoints(subset["power_w"].to_numpy(), sample_hz=hz)
            step_period, n_steps, step_cv = step_period_from_log(events, t0, t1, gpu_id=int(gpu_id))
            matched = period_match(float(flat.get("dominant_freq_hz", 0.0)), step_period)
            explained = explained_changepoints([t0 + cp for cp in cps], events, gpu_id=int(gpu_id))
            raw_evidence = {
                "period_match": matched,
                # period_mismatch deferred: needs strong_peak after correct tau (SCORE_RECOMPUTE).
                "step_period_s": step_period,
                "step_count": n_steps,
                "step_period_cv": step_cv,
                "changepoints": [t0 + cp for cp in cps],
                "unexplained_changepoint_count": len(explained["unexplained"]),
                "progress_log_available": bool(events),
                "progress_log_missing": not bool(events),
                "cross_job_sync_index": flat.get("cross_job_sync_index", 0.0),
                "dominant_peak_prominence": flat.get("dominant_peak_prominence", 0.0),
                "dominant_peak_prominence_log": flat.get(
                    "dominant_peak_prominence_log",
                    float(np.log10(1.0 + max(float(flat.get("dominant_peak_prominence", 0.0) or 0.0), 0.0))),
                ),
                "high_load_fraction": flat.get("high_load_fraction", 0.0),
                "longest_high_seconds": flat.get("longest_high_seconds", 0.0),
                "swing_abs_w": flat.get("swing_abs_w", 0.0),
                "mean_w": flat.get("mean_w", 0.0),
                "ramp_p95_w_per_s": flat.get("ramp_p95_w_per_s", flat.get("ramp_max_w_per_s", 0.0)),
                "util_residual_mad_w": flat.get("util_residual_mad_w", 0.0),
            }
            bool_evidence = to_bool_evidence(raw_evidence, config)
            bool_evidence = {k: v for k, v in bool_evidence.items() if k not in BUILD_DEFERRED_BOOL_KEYS}
            flat.update({
                "session_id": str(session_id),
                "gpu_id": int(gpu_id),
                "sample_hz": hz,
                "start": start,
                "end": start + window,
                "window_id": f"{session_id}:{int(gpu_id)}:{start}:{start + window}",
                "window_start_s": t0,
                "window_end_s": t1,
                "declared_job_type": str(subset.get("declared_job_type", pd.Series(["unknown"])).iloc[0]),
                "declared_job_family": str(subset.get("declared_job_family", pd.Series(["unknown"])).iloc[0]),
                "gpu_model": str(subset.get("gpu_model", pd.Series(["unknown"])).iloc[0]),
                "evidence": {**raw_evidence, **bool_evidence},
                "evidence_bool": bool_evidence,
                "multi_gpu_sync_index": 0.0,
                "cross_job_sync_index": 0.0,
                "sync_applicable": False,
            })
            if truth is not None:
                epoch_start, epoch_end = _window_epoch_bounds(subset, truth)
                gt_label, gt_is_attack, truth_source = label_window(
                    window_start_epoch=epoch_start,
                    window_end_epoch=epoch_end,
                    intervals=truth.get("intervals"),
                    session_gt_label=str(truth["session_gt_label"]),
                    host_workload=truth.get("host_workload"),
                    hosted=bool(truth.get("hosted", False)),
                )
                warmup = False
                if "warmup" in subset.columns:
                    warm = subset["warmup"]
                    if warm.dtype != bool:
                        warm = warm.astype(str).str.lower().isin({"true", "1"})
                    warmup = float(warm.mean()) >= 0.5
                flat.update({
                    "gt_label": gt_label,
                    "gt_is_attack": bool(gt_is_attack),
                    "truth_source": truth_source,
                    "warmup": bool(warmup),
                    "gpu_role": truth.get("gpu_role"),
                    "gt_variant": truth.get("gt_variant"),
                    "declared_policy": truth.get("declared_policy"),
                    "group_id": truth.get("group_id"),
                    "source": truth.get("source"),
                })
            rows.append(flat)
    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows
    return _attach_session_sync(windows, df)


def _attach_session_sync(windows: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Fill multi_gpu / cross_job sync indices for same-session windows only."""
    if "session_id" not in windows.columns or "window_start_s" not in windows.columns:
        return windows
    out = windows.copy()
    for session_id, session_windows in out.groupby("session_id", sort=False):
        session_df = df[df["session_id"].astype(str) == str(session_id)]
        if session_df.empty or session_df["gpu_id"].nunique() < 2:
            continue
        # Unique time windows within the session (shared across GPUs).
        for (t0, t1), _ in session_windows.groupby(["window_start_s", "window_end_s"], sort=False):
            metrics = session_window_sync_metrics(session_df, float(t0), float(t1))
            mask = (
                (out["session_id"].astype(str) == str(session_id))
                & (out["window_start_s"] == float(t0))
                & (out["window_end_s"] == float(t1))
            )
            for key in (
                "multi_gpu_sync_index",
                "cross_job_sync_index",
                "sync_applicable",
                "sync_lag_s",
            ):
                if key in metrics:
                    out.loc[mask, key] = metrics[key]
            # Keep evidence dict in sync for downstream bool conversion.
            for idx in out.index[mask]:
                evidence = dict(out.at[idx, "evidence"] or {})
                evidence["multi_gpu_sync_index"] = float(metrics["multi_gpu_sync_index"])
                evidence["cross_job_sync_index"] = float(metrics["cross_job_sync_index"])
                out.at[idx, "evidence"] = evidence
                out.at[idx, "cross_job_sync_index"] = float(metrics["cross_job_sync_index"])
                out.at[idx, "multi_gpu_sync_index"] = float(metrics["multi_gpu_sync_index"])
    return out
