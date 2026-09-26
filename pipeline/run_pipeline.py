"""
GridPulse 전체 파이프라인 오케스트레이션.

입력: DCGM/NVML로 수집한 텔레메트리 CSV (timestamp, power_w, util_gpu_pct, ... , label)
흐름:
  [1단계] 상시 저비용 스크리닝: 슬라이딩 윈도우별 변동폭 z-score로 의심 후보 선별
  [2단계] 의심 후보만 고해상도 특성 분석 (주기성/규칙성/평균유지)
  [3단계] RAG로 알려진 공격 시그니처와 대조 + LLM 판정·설명

논문 대비 우리 기여:
  - kHz 원신호를 탐지하려 하지 않음 (그건 물리계측 영역, 우선순위에서 제외)
  - 대신 "의심 활동만 선별 → 알려진 공격과 대조" 하는 실용 파이프라인 (검증된 2-tier IDS 패턴)
  - 새 공격은 corpus/에 문서 추가만으로 대응 (재학습 불필요)

사용:
  python run_pipeline.py --telemetry data.csv --baseline-mean 120 \
      --rag-backend tfidf --llm-backend stub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from features import compute_window_features, compute_window_features_v2, features_to_description
from rag_analyzer import SignatureRetriever, analyze_with_llm
from dataset.schema import strip_ground_truth

WINDOW = 200
STRIDE = 100
STAGE1_CONFIG_DEFAULT = ROOT / "config" / "stage1_v2.yaml"


def merge_window_features(
    power,
    util=None,
    *,
    sample_hz: float = 1.0,
    tdp_w: float = 450.0,
    nvml_avg_window_s: float | None = 1.0,
) -> dict:
    """Compute v1 and v2 features; on key collision prefer v2. Flatten nested band dicts."""
    from pipeline.stage1_v2 import flatten_features

    v1 = compute_window_features(power, util, sample_hz=sample_hz) or {}
    v2 = compute_window_features_v2(
        power,
        util,
        sample_hz=sample_hz,
        tdp_w=tdp_w,
        nvml_avg_window_s=nvml_avg_window_s,
    ) or {}
    merged = {**v1, **v2}
    return flatten_features(merged)


def _serialize_features(feats: dict) -> dict:
    out = {}
    for key, value in feats.items():
        if isinstance(value, (bool, str)):
            out[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = round(float(value), 5)
        elif value is None:
            out[key] = None
    return out


def context_to_query(context: dict[str, str]) -> str:
    """선언 컨텍스트를 공격명 없이 사실 문장으로만 바꾼다."""
    parts = []
    if context.get("declared_job_type"):
        parts.append(f"선언 작업 유형: {context['declared_job_type']}.")
    if context.get("declared_job_family"):
        parts.append(f"선언 작업 계열: {context['declared_job_family']}.")
    if context.get("declared_gres"):
        parts.append(f"선언 GPU 요청: {context['declared_gres']}.")
    return " ".join(parts)


def infer_sample_hz(df: pd.DataFrame, fallback: float = 1.0) -> float:
    """명시값 또는 timestamp 간격에서 세션의 샘플링률을 추정한다."""
    if "sample_hz" in df.columns:
        values = pd.to_numeric(df["sample_hz"], errors="coerce").dropna()
        if len(values) and values.iloc[0] > 0:
            return float(values.iloc[0])
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().to_numpy()
        deltas = np.diff(ts)
        deltas = deltas[deltas > 0]
        if len(deltas):
            return float(1.0 / np.median(deltas))
    return fallback


def anomaly_score_components(
    features: dict,
    *,
    z_score: float,
    baseline_mean_w: float | None,
) -> dict[str, float]:
    """Transparent 0-1 Cyber Stage-1 score components (heuristic v1)."""
    mean_w = float(features.get("mean_w", 0.0))
    mean_deviation = 0.0
    if baseline_mean_w and baseline_mean_w > 0:
        mean_deviation = min(1.0, abs(mean_w / baseline_mean_w - 1.0))
    return {
        "z_score": min(1.0, abs(float(z_score)) / 5.0),
        "swing": min(1.0, max(0.0, float(features.get("swing_ratio", 0.0))) / 2.0),
        "mean_deviation": mean_deviation,
        "persistence": min(1.0, max(0.0, float(features.get("high_load_fraction", 0.0)))),
        "mechanical_periodicity": min(
            1.0,
            max(0.0, float(features.get("periodicity_strength", 0.0)))
            * max(0.0, float(features.get("duty_regularity", 0.0))),
        ),
    }


def anomaly_score(components: dict[str, float]) -> float:
    """OR-consistent score: strongest Stage-1 signal, not a calibrated risk."""
    return float(max(components.values(), default=0.0))


def stage1_screen_legacy(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    *,
    baseline_mean_w: float | None = None,
    sample_hz: float = 1.0,
    window: int = WINDOW,
    stride: int = STRIDE,
    tdp_w: float = 450.0,
    nvml_avg_window_s: float | None = 1.0,
):
    """범용 필터들로 한 세션의 의심 윈도우를 선별한다."""
    windows = []
    for start in range(0, len(df) - window + 1, stride):
        w = df.iloc[start:start + window]
        power = w["power_w"].values
        util = w["util_gpu_pct"].values if "util_gpu_pct" in w else None
        feats = merge_window_features(
            power,
            util,
            sample_hz=sample_hz,
            tdp_w=tdp_w,
            nvml_avg_window_s=nvml_avg_window_s,
        )
        if not feats:
            continue
        label = str(w["label"].mode().iloc[0]) if "label" in w and not w["label"].mode().empty else None
        windows.append(
            {
                "start": start,
                "end": start + window,
                "label": label,
                **feats,
            }
        )

    wdf = pd.DataFrame(windows)
    if wdf.empty:
        return wdf.assign(z_score=pd.Series(dtype=float), candidate_reasons="", is_candidate=False)

    mu, sigma = wdf["swing_ratio"].mean(), wdf["swing_ratio"].std()
    if len(wdf) < 3 or not np.isfinite(sigma) or sigma < 1e-9:
        wdf["z_score"] = 0.0
    else:
        wdf["z_score"] = (wdf["swing_ratio"] - mu) / sigma

    def reasons(row) -> str:
        found: list[str] = []
        baseline = baseline_mean_w
        if abs(float(row["z_score"])) > z_threshold:
            found.append("swing_zscore")
        if float(row["swing_ratio"]) > 1.1:
            found.append("large_swing")
        if baseline and baseline > 0:
            ratio = float(row["mean_w"]) / baseline
            if ratio > 1.8:
                found.append("high_mean_power")
            elif ratio < 0.5:
                found.append("low_mean_power")
        persistent = (
            float(row.get("high_load_fraction", 0)) > 0.8
            and float(row.get("cv", 1)) < 0.15
        )
        if persistent:
            found.append("persistent_high_load")
        mechanical = (
            int(row.get("n_transitions", 0)) >= 3
            and float(row.get("swing_ratio", 0)) > 1.1
            and float(row.get("duty_regularity", 0)) > 0.75
            and float(row.get("periodicity_strength", 0)) > 0.45
        )
        if mechanical:
            found.append("mechanical_periodicity")
        return ",".join(found)

    wdf["candidate_reasons"] = wdf.apply(reasons, axis=1)
    wdf["is_candidate"] = wdf["candidate_reasons"].str.len() > 0
    return wdf


stage1_screen = stage1_screen_legacy


def _stage1_window_summary(row: pd.Series) -> dict:
    """Compact Stage-1 screening fields for all windows / grid_watch / candidates."""
    impact = row.get("impact_components") or {}
    if not isinstance(impact, dict):
        impact = {}
    impact_raw = None
    if "impact_raw" in row and pd.notna(row.get("impact_raw")):
        impact_raw = float(row["impact_raw"])
    elif impact.get("impact_raw") is not None:
        impact_raw = float(impact["impact_raw"])
    return {
        "window_id": str(
            row.get("window_id")
            or f"{row.get('session_id')}:{row.get('gpu_id')}:{row.get('start')}:{row.get('end')}"
        ),
        "session_id": str(row.get("session_id")),
        "gpu_id": int(row.get("gpu_id")) if pd.notna(row.get("gpu_id")) else None,
        "window_start_s": float(row["window_start_s"]) if "window_start_s" in row and pd.notna(row.get("window_start_s")) else None,
        "window_end_s": float(row["window_end_s"]) if "window_end_s" in row and pd.notna(row.get("window_end_s")) else None,
        "declared_job_family": str(row.get("declared_job_family")) if row.get("declared_job_family") is not None else None,
        "baseline_level": row.get("baseline_level"),
        "u_score": float(row["u_score"]) if "u_score" in row and pd.notna(row.get("u_score")) else None,
        "p_value": float(row["p_value"]) if "p_value" in row and pd.notna(row.get("p_value")) else None,
        "impact_raw": impact_raw,
        "impact_level": row.get("impact_level"),
        "multi_gpu_sync_index": float(row["multi_gpu_sync_index"]) if "multi_gpu_sync_index" in row and pd.notna(row.get("multi_gpu_sync_index")) else None,
        "cross_job_sync_index": float(row["cross_job_sync_index"]) if "cross_job_sync_index" in row and pd.notna(row.get("cross_job_sync_index")) else None,
        "is_candidate": bool(row.get("is_candidate", False)),
        "grid_watch": bool(row.get("grid_watch", False)),
        "candidate_reasons": str(row.get("candidate_reasons") or ""),
    }


def _write_stage1_envelope(path: Path | None, wdf: pd.DataFrame, *, telemetry: str, stage1_version: str) -> dict:
    if wdf is None or wdf.empty:
        envelope = {
            "stage1_version": stage1_version,
            "telemetry": telemetry,
            "all_windows": [],
            "grid_watch": [],
            "candidates": [],
            "summary": {"n_windows": 0, "n_candidates": 0, "n_grid_watch": 0},
        }
    else:
        all_windows = [_stage1_window_summary(row) for _, row in wdf.iterrows()]
        candidates = [w for w in all_windows if w["is_candidate"]]
        grid_watch = [w for w in all_windows if w["grid_watch"] and not w["is_candidate"]]
        envelope = {
            "stage1_version": stage1_version,
            "telemetry": telemetry,
            "all_windows": all_windows,
            "grid_watch": grid_watch,
            "candidates": candidates,
            "summary": {
                "n_windows": len(all_windows),
                "n_candidates": len(candidates),
                "n_grid_watch": len(grid_watch),
            },
        }
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Stage1 저장: {path} (windows={envelope['summary']['n_windows']}, "
            f"candidates={envelope['summary']['n_candidates']}, "
            f"grid_watch={envelope['summary']['n_grid_watch']})"
        )
    return envelope


def _write_stage2_results(path: Path | None, results: list) -> None:
    """Atomically persist Stage2 rows so a finished session survives a later crash."""
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def _segment_candidate_frame(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    from pipeline.stage2_segments import aggregate_bool_evidence, merge_candidate_segments

    stage2_cfg = config.get("stage2") or {}
    rate_min = float(stage2_cfg.get("evidence_rate_min", 0.6))
    drop = ("progress_log_missing",) if stage2_cfg.get("drop_progress_log_missing") else ()
    rows = []
    for idxs in merge_candidate_segments(candidates):
        part = candidates.loc[idxs]
        first = part.iloc[0].to_dict()
        evs = []
        for _, item in part.iterrows():
            ev = item.get("evidence_bool") or item.get("evidence") or {}
            if isinstance(ev, dict):
                evs.append({key: value for key, value in ev.items() if isinstance(value, bool)})
        bools, rates = aggregate_bool_evidence(evs, rate_min=rate_min, drop_keys=drop)
        first["start"] = int(part["start"].min())
        first["end"] = int(part["end"].max()) if "end" in part else first.get("end")
        first["window_ids"] = [str(x) for x in part["window_id"].tolist()] if "window_id" in part else []
        first["n_windows"] = int(len(part))
        first["evidence_bool"] = bools
        first["evidence_rates"] = rates
        if "u_score" in part:
            first["u_score"] = float(pd.to_numeric(part["u_score"], errors="coerce").median())
        rows.append(first)
    return pd.DataFrame(rows) if rows else candidates.iloc[0:0]


def run(args):
    df = pd.read_csv(args.telemetry)
    print(f"[입력] {args.telemetry}: {len(df)}행")

    if "power_w" not in df:
        raise ValueError("입력 CSV에 필수 컬럼 power_w가 없습니다.")
    if "session_id" not in df:
        df["session_id"] = "legacy-session"
    if "gpu_id" not in df:
        df["gpu_id"] = 0
    df_infer = strip_ground_truth(df)
    if "session_id" not in df_infer:
        df_infer["session_id"] = df["session_id"]
    if "gpu_id" not in df_infer:
        df_infer["gpu_id"] = df["gpu_id"]

    baseline = args.baseline_mean
    stage1_mode = getattr(args, "stage1", "legacy")
    from pipeline.stage1_v2 import load_config

    stage1_config = load_config(getattr(args, "stage1_config", STAGE1_CONFIG_DEFAULT))
    tdp_w = float(stage1_config.get("tdp_w", 450.0))
    nvml_avg_window_s = stage1_config.get("nvml_avg_window_s", 1.0)
    if baseline is None and stage1_mode == "legacy":
        baseline = 120.0
        print("[기준선] legacy 모드 기본값 120W 사용")

    screened: list[pd.DataFrame] = []
    grouped_frames: dict[tuple[str, object], pd.DataFrame] = {}
    original_groups = {
        (str(session_id), gpu_id): group.sort_values("timestamp").reset_index(drop=True)
        if "timestamp" in group else group.reset_index(drop=True)
        for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False)
    }
    if stage1_mode == "v2":
        from pipeline.baseline import CohortBaseline
        from pipeline.stage1_v2 import build_windows, score_window

        config = stage1_config
        baseline_model = CohortBaseline.load(getattr(args, "baseline_model", "dataset/eval/cohort_baseline_v2.json"))
        calibration_path = Path(getattr(args, "calibration", "dataset/eval/stage1_v2_calibration.json"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("evidence_surprisal_weights"):
            config = dict(config)
            config["evidence_weights"] = {
                **config.get("evidence_weights", {}),
                **calibration["evidence_surprisal_weights"],
            }
        if calibration.get("evidence_thresholds"):
            config = dict(config)
            config["evidence_thresholds"] = calibration["evidence_thresholds"]
        current = build_windows(
            df_infer,
            window_s=float(getattr(args, "window_s", 30.0)),
            stride_s=float(getattr(args, "stride_s", 15.0)),
            progress_log_dir=getattr(args, "progress_log_dir", None),
            sessions_root=getattr(args, "sessions_root", None),
            config=config,
            progress_source=getattr(args, "progress_source", "visible"),
        )
        if not current.empty:
            scores = [score_window(row.to_dict(), baseline_model, calibration, config) for _, row in current.iterrows()]
            for key in ("u_score", "p_value", "baseline_level", "impact_level", "grid_watch", "impact_raw", "idle_status"):
                current[key] = [item.get(key) for item in scores]
            current["is_candidate"] = [item["is_candidate"] for item in scores]
            current["candidate_reasons"] = [",".join(item["candidate_reasons"]) for item in scores]
            current["evidence_bool"] = [item.get("evidence_bool") for item in scores]
            current["impact_components"] = [item["impact_components"] for item in scores]
            current["robust_z"] = [item["robust_z"] for item in scores]
            current["stage1_version"] = "v2"
            screened.append(current)
        for (session_id, gpu_id), group in df_infer.groupby(["session_id", "gpu_id"], sort=False):
            grouped_frames[(str(session_id), gpu_id)] = group.sort_values("timestamp").reset_index(drop=True)
    else:
        for (session_id, gpu_id), group in df_infer.groupby(["session_id", "gpu_id"], sort=False):
            group = group.sort_values("timestamp") if "timestamp" in group else group
            group = group.reset_index(drop=True)
            hz = args.sample_hz or infer_sample_hz(group)
            current = stage1_screen_legacy(
                group,
                z_threshold=args.z_threshold,
                baseline_mean_w=baseline,
                sample_hz=hz,
                window=args.window,
                stride=args.stride,
                tdp_w=tdp_w,
                nvml_avg_window_s=nvml_avg_window_s,
            )
            if current.empty:
                continue
            current["session_id"] = str(session_id)
            current["gpu_id"] = gpu_id
            current["sample_hz"] = hz
            current["stage1_version"] = "legacy"
            screened.append(current)
            grouped_frames[(str(session_id), gpu_id)] = group

    wdf = pd.concat(screened, ignore_index=True) if screened else pd.DataFrame()
    candidates = wdf[wdf["is_candidate"]] if not wdf.empty else wdf
    print(f"[1단계] 전체 {len(wdf)}개 윈도우 → 후보 {len(candidates)}개 "
          f"(다중 필터, z>{args.z_threshold})")

    stage1_path = getattr(args, "stage1_out", None)
    _write_stage1_envelope(
        Path(stage1_path) if stage1_path else None,
        wdf,
        telemetry=str(args.telemetry),
        stage1_version=stage1_mode,
    )

    out_path = Path(args.out) if getattr(args, "out", None) else None
    if len(candidates) == 0:
        print("의심 후보 없음. 정상으로 판단.")
        _write_stage2_results(out_path, [])
        if out_path:
            print(f"저장: {out_path}")
        return []

    if getattr(args, "stage2", "legacy") == "none":
        print("Stage2 skipped (--stage2 none)")
        return []

    # RAG 검색기 준비 (1회)
    retriever = SignatureRetriever(backend=args.rag_backend)

    sort_cols = [c for c in ("session_id", "gpu_id", "start") if c in candidates.columns]
    if sort_cols:
        candidates = candidates.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if getattr(args, "stage2", "legacy") == "v2":
        candidates = _segment_candidate_frame(candidates, stage1_config)

    results = []

    for index, cand in candidates.iterrows():
        key = (str(cand["session_id"]), cand["gpu_id"])
        group = grouped_frames[key]
        start = int(cand["start"])
        end = int(cand["end"])
        w = group.iloc[start:end]
        power = w["power_w"].values
        util = w["util_gpu_pct"].values if "util_gpu_pct" in w else None

        # 2단계: 특성 분석 (v1+v2 병합)
        sample_hz = float(cand["sample_hz"]) if "sample_hz" in cand and pd.notna(cand.get("sample_hz")) else float(
            infer_sample_hz(group)
        )
        feats = merge_window_features(
            power,
            util,
            sample_hz=sample_hz,
            tdp_w=tdp_w,
            nvml_avg_window_s=nvml_avg_window_s,
        )
        desc = features_to_description(feats, baseline_mean_w=baseline)
        z_like_score = float(cand.get("z_score", cand.get("u_score", 0.0)))

        context = {}
        for c in (
            "declared_user",
            "declared_job_type",
            "declared_job_family",
            "declared_gres",
            "declared_process_name",
            "pid",
        ):
            if c in w.columns:
                context[c] = str(w[c].iloc[0])
        declared_context = {
            "job_type": context.get("declared_job_type"),
            "job_family": context.get("declared_job_family"),
            "user": context.get("declared_user"),
        }
        # 3단계: RAG + LLM
        context_query = context_to_query(context) if getattr(args, "query_context", "off") == "declared" else ""
        query = f"{desc} {context_query}".strip()
        retrieved = retriever.search(query, top_k=3, features=feats)
        top_meta = getattr(retrieved[0], "meta", {}) if retrieved else {}
        stage2_mode = getattr(args, "stage2", "legacy")
        if stage2_mode == "legacy":
            verdict = analyze_with_llm(desc, retrieved, context=context,
                                        backend=args.llm_backend, model=args.llm_model)
            if top_meta.get("category") == "benign":
                verdict = {
                    **verdict,
                    "risk": "정상",
                    "benign_match": retrieved[0].name,
                    "reason": (
                        f"benign corpus '{retrieved[0].name}' 수치 조건과 문서가 함께 "
                        f"매칭되어 경보를 억제합니다. {verdict.get('reason', '')}"
                    ).strip(),
                }
        else:
            verdict = {
                "risk": "주의",
                "closest_match": retrieved[0].name if retrieved else "unknown",
                "reason": "",
            }
        original = original_groups[key].iloc[start:end]
        raw_attack_id = (
            original["gt_attack_id"].iloc[0] if "gt_attack_id" in original.columns
            else original["attack_id"].iloc[0] if "attack_id" in original.columns
            else None
        )
        raw_label = (
            original["gt_label"].iloc[0] if "gt_label" in original.columns
            else original["label"].iloc[0] if "label" in original.columns
            else None
        )
        attack_id = (
            str(raw_attack_id)
            if raw_attack_id is not None and not pd.isna(raw_attack_id) and str(raw_attack_id)
            else str(cand["session_id"])
        )
        components = anomaly_score_components(
            feats,
            z_score=z_like_score,
            baseline_mean_w=baseline,
        )

        row = {
            "session_id": str(cand["session_id"]),
            "attack_id": attack_id,
            "window_id": (
                str(cand.get("window_id") or f"{cand['session_id']}:{int(cand['gpu_id'])}:{start}:{end}")
            ),
            "gpu_id": int(cand["gpu_id"]),
            "window_start": start,
            "window_end": end,
            "ground_truth": raw_label,
            "swing_ratio": round(float(cand["swing_ratio"]), 3),
            "z_score": round(z_like_score, 3),
            "candidate_reasons": str(cand["candidate_reasons"]).split(","),
            "features": _serialize_features(feats),
            "features_version": "v1+v2",
            "declared_context": declared_context,
            "declared_job_type": declared_context.get("job_type"),
            "declared_job_family": declared_context.get("job_family"),
            "declared_user": declared_context.get("user"),
            "description": desc,
            "rag_top": [(n, round(s, 3)) for n, s, _ in retrieved],
            "rag_filter_excluded": {
                name: reasons
                for name, reasons in retriever.last_filter_report.items()
                if reasons
            },
            "threat_id": verdict.get("closest_match"),
            "anomaly_score": round(anomaly_score(components), 5),
            "score_components": {key: round(value, 5) for key, value in components.items()},
            "score_version": "cyber-stage1-v2" if stage1_mode == "v2" else "cyber-stage1-or-v1",
            "stage1_version": cand.get("stage1_version", stage1_mode),
            "verdict": verdict,
        }
        if stage1_mode == "v2":
            row.update({
                "impact_level": cand.get("impact_level"),
                "impact_components": cand.get("impact_components", {}),
                "u_score": round(float(cand.get("u_score", 0.0)), 6),
                "p_value": round(float(cand.get("p_value", 1.0)), 6),
                "baseline_level": cand.get("baseline_level"),
                "evidence": cand.get("evidence", {}),
                "grid_watch": bool(cand.get("grid_watch", False)),
                "robust_z": cand.get("robust_z", {}),
            })

        # Ops mode: event-triggered physics validation on Stage-1 candidates only.
        if getattr(args, "physics_validate", False):
            from bit2watt_impl.physics.simulation import (
                inject_observed_waveform_with_timeout,
            )

            physics = inject_observed_waveform_with_timeout(
                power,
                float(cand["sample_hz"]),
                timeout_s=float(getattr(args, "physics_timeout_s", 60.0)),
                test_system=str(getattr(args, "physics_test_system", "kundur_ieeest")),
            )
            row["physics_converged"] = bool(physics.get("converged"))
            row["physics_failure_reason"] = physics.get("failure_reason") or ""
            row["physics_test_system"] = physics.get("test_system")
            row["physics_mode"] = physics.get("mode", "observed_waveform_replay")
            for key in ("osc_std", "rocof_hz_s", "dominant_freq_hz", "osc_ptp"):
                value = physics.get(key)
                row[f"physics_{key}"] = (
                    None if value is None else round(float(value), 8)
                )
            print(
                f"  [physics] {row['window_id']} converged={row['physics_converged']} "
                f"osc_std={row['physics_osc_std']} rocof={row['physics_rocof_hz_s']}"
            )

        if getattr(args, "stage2", "legacy") == "v2":
            from pipeline.corpus_schema import parse_corpus_v2
            from pipeline.rag_analyzer import CORPUS_DIR
            from pipeline.stage2_evidence import build_evidence_bundle
            from pipeline.stage2_llm import judge, normal_verdict

            docs = []
            for result in retrieved:
                name = result.name if hasattr(result, "name") else result[0]
                path = CORPUS_DIR / f"{name}.md"
                if path.exists():
                    docs.append(parse_corpus_v2(path))
            top1_name = None
            if retrieved:
                top1_name = retrieved[0].name if hasattr(retrieved[0], "name") else retrieved[0][0]
            drop_missing = bool((stage1_config.get("stage2") or {}).get("drop_progress_log_missing"))
            if cand.get("evidence_bool"):
                row["evidence_bool"] = cand.get("evidence_bool")
            bundle = build_evidence_bundle(row, drop_progress_log_missing=drop_missing)
            if top_meta.get("category") == "benign":
                stage2 = normal_verdict(top1_name)
            else:
                stage2 = judge(
                    bundle,
                    docs,
                    backend=args.llm_backend,
                    model=args.llm_model,
                    fallback="legacy",
                    retriever_top1=top1_name,
                )
            risk = {"known": "의심", "normal": "정상", "unknown": "주의", "partial": "주의"}.get(stage2.verdict, "주의")
            row["stage2_version"] = "v2"
            row["stage2_evidence_bundle"] = bundle
            row["window_ids"] = cand.get("window_ids") or [row["window_id"]]
            row["evidence_rates"] = cand.get("evidence_rates") or {}
            row["verdict"] = {
                "risk": risk,
                "closest_match": stage2.closest_match or "unknown",
                "reason": stage2.explanation,
                "verdict": stage2.verdict,
                "confidence": stage2.confidence,
                "matched_evidence": stage2.matched_evidence,
                "contradicting_evidence": stage2.contradicting_evidence,
                "fallback_used": stage2.fallback_used,
            }
            row["threat_id"] = row["verdict"]["closest_match"]
        else:
            row["stage2_version"] = "legacy"

        results.append(row)
        this_session = str(cand["session_id"])
        is_last = int(index) >= len(candidates) - 1
        next_session = None if is_last else str(candidates.iloc[int(index) + 1]["session_id"])
        if is_last or next_session != this_session:
            _write_stage2_results(out_path, results)
            if out_path:
                n_session = sum(1 for row in results if str(row["session_id"]) == this_session)
                print(
                    f"[2단계] session {this_session}: {n_session}개 저장 "
                    f"(누적 {len(results)}/{len(candidates)})",
                    flush=True,
                )

    print(f"\n[결과] 의심 후보 {len(results)}개 분석 완료:\n")
    for r in results:
        v = r["verdict"]
        print(f"  {r['session_id']}/GPU{r['gpu_id']} 윈도우@{r['window_start']} "
              f"(필터 {r['candidate_reasons']})")
        print(f"    → 위험도: {v.get('risk')} / 최근접: {v.get('closest_match')}")
        print(f"    → 근거: {v.get('reason')}")
        print(f"    → RAG: {r['rag_top']}\n")

    _write_stage2_results(out_path, results)
    if out_path:
        print(f"저장: {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True, help="DCGM/NVML 수집 CSV")
    ap.add_argument("--baseline-mean", type=float, default=None,
                     help="정상 평균 전력(W). 정상 베이스라인 수집에서 산출")
    ap.add_argument("--sample-hz", type=float, default=None,
                     help="텔레메트리 실효 샘플링레이트(미지정 시 CSV/timestamp에서 추정)")
    ap.add_argument("--window", type=int, default=WINDOW, help="윈도우 크기(행)")
    ap.add_argument("--stride", type=int, default=STRIDE, help="윈도우 이동 간격(행)")
    ap.add_argument("--z-threshold", type=float, default=2.5)
    ap.add_argument("--stage1", choices=["legacy", "v2"], default="legacy")
    ap.add_argument("--stage2", choices=["legacy", "v2", "none"], default="legacy")
    ap.add_argument("--baseline-model", default="dataset/eval/cohort_baseline_v2.json")
    ap.add_argument("--stage1-config", default="config/stage1_v2.yaml")
    ap.add_argument("--calibration", default="dataset/eval/stage1_v2_calibration.json")
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--stride-s", type=float, default=15.0)
    ap.add_argument("--progress-log-dir", default="dataset/synthetic/steps")
    ap.add_argument("--sessions-root", default=None)
    ap.add_argument("--progress-source", choices=["visible", "raw"], default="visible")
    ap.add_argument("--query-context", choices=["off", "declared"], default="off")
    ap.add_argument("--rag-backend", choices=["sbert", "tfidf"], default="sbert")
    ap.add_argument("--llm-backend", choices=["ollama", "stub"], default="stub")
    ap.add_argument("--llm-model", default="gemma3:12b",
                     help="Ollama 모델명 (이 서버 검증: gemma3:12b)")
    ap.add_argument(
        "--physics-validate",
        action="store_true",
        default=False,
        help=(
            "이벤트 트리거 Physics 검증: Stage-1 후보 윈도우의 관측 파형을 "
            "공개 테스트계통에 온디맨드 재생(기본 off)"
        ),
    )
    ap.add_argument(
        "--physics-timeout-s",
        type=float,
        default=60.0,
        help="후보당 Physics 시뮬레이션 최대 벽시계 초",
    )
    ap.add_argument(
        "--physics-test-system",
        choices=["kundur_ieeest", "wecc_179_gencls"],
        default="kundur_ieeest",
        help="운영 모드 Physics 검증에 사용할 공개 테스트계통",
    )
    ap.add_argument("--out", default=None, help="Stage2 후보 분석 결과 JSON (기존 계약)")
    ap.add_argument(
        "--stage1-out",
        default=None,
        help="Stage1 envelope JSON: all_windows / grid_watch / candidates / summary",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
