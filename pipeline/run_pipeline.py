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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from features import compute_window_features, features_to_description
from rag_analyzer import SignatureRetriever, analyze_with_llm

WINDOW = 200
STRIDE = 100


def context_to_query(context: dict[str, str]) -> str:
    """스케줄러/실험 하네스 맥락을 RAG 검색 가능한 설명으로 바꾼다."""
    job_type = context.get("job_type", "").lower()
    if job_type.startswith("normal_"):
        return (
            "정상 워크로드 프로파일이며 오탐 방지용 기준에 해당함. "
            "변동 원인이 예정된 작업 종류와 사용자 맥락으로 설명되고 과거 패턴과 일치함."
        )
    if "hash" in job_type:
        return (
            "등록되지 않은 GPU 크립토재킹 의심 해시 연산. "
            "지속적으로 높은 GPU 사용률과 평평한 장시간 부하."
        )
    if "llm" in job_type and "train" in job_type:
        return (
            "LLM Training Modulation Attack LTMA 대조 대상. 실제 LLM 학습 파이프라인 "
            "내부에서 평균을 유지하며 불규칙하게 변조된 부하."
        )
    if "unknown" in job_type:
        return (
            "Synthetic Workload Modulation Attack SWMA 대조 대상. "
            "사용자 작업 맥락으로 설명되지 않는 별도 CUDA 커널의 규칙적 on/off."
        )
    return ""


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


def stage1_screen(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    *,
    baseline_mean_w: float | None = None,
    sample_hz: float = 1.0,
    window: int = WINDOW,
    stride: int = STRIDE,
):
    """범용 필터들로 한 세션의 의심 윈도우를 선별한다."""
    windows = []
    for start in range(0, len(df) - window + 1, stride):
        w = df.iloc[start:start + window]
        power = w["power_w"].values
        util = w["util_gpu_pct"].values if "util_gpu_pct" in w else None
        feats = compute_window_features(power, util, sample_hz=sample_hz)
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


def run(args):
    df = pd.read_csv(args.telemetry)
    print(f"[입력] {args.telemetry}: {len(df)}행")

    if "power_w" not in df:
        raise ValueError("입력 CSV에 필수 컬럼 power_w가 없습니다.")
    if "session_id" not in df:
        df["session_id"] = "legacy-session"
    if "gpu_id" not in df:
        df["gpu_id"] = 0

    baseline = args.baseline_mean
    if baseline is None and "label" in df:
        normal = df[df["label"].astype(str).str.startswith("normal")]
        if len(normal):
            baseline = float(pd.to_numeric(normal["power_w"], errors="coerce").median())
            print(f"[기준선] normal 라벨 중앙값으로 자동 추정: {baseline:.2f}W")

    screened: list[pd.DataFrame] = []
    grouped_frames: dict[tuple[str, object], pd.DataFrame] = {}
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        group = group.sort_values("timestamp") if "timestamp" in group else group
        group = group.reset_index(drop=True)
        hz = args.sample_hz or infer_sample_hz(group)
        current = stage1_screen(
            group,
            z_threshold=args.z_threshold,
            baseline_mean_w=baseline,
            sample_hz=hz,
            window=args.window,
            stride=args.stride,
        )
        if current.empty:
            continue
        current["session_id"] = str(session_id)
        current["gpu_id"] = gpu_id
        current["sample_hz"] = hz
        screened.append(current)
        grouped_frames[(str(session_id), gpu_id)] = group

    wdf = pd.concat(screened, ignore_index=True) if screened else pd.DataFrame()
    candidates = wdf[wdf["is_candidate"]] if not wdf.empty else wdf
    print(f"[1단계] 전체 {len(wdf)}개 윈도우 → 후보 {len(candidates)}개 "
          f"(다중 필터, z>{args.z_threshold})")

    if len(candidates) == 0:
        print("의심 후보 없음. 정상으로 판단.")
        return []

    # RAG 검색기 준비 (1회)
    retriever = SignatureRetriever(backend=args.rag_backend)

    results = []

    for _, cand in candidates.iterrows():
        key = (str(cand["session_id"]), cand["gpu_id"])
        group = grouped_frames[key]
        start = int(cand["start"])
        end = int(cand["end"])
        w = group.iloc[start:end]
        power = w["power_w"].values
        util = w["util_gpu_pct"].values if "util_gpu_pct" in w else None

        # 2단계: 특성 분석
        feats = compute_window_features(power, util, sample_hz=float(cand["sample_hz"]))
        desc = features_to_description(feats, baseline_mean_w=baseline)

        context = {}
        for c in ("id_user", "job_type", "gres_req", "pid", "process_name"):
            if c in w.columns:
                context[c] = str(w[c].iloc[0])
        # 3단계: RAG + LLM
        context_query = context_to_query(context)
        query = f"{desc} {context_query}".strip()
        retrieved = retriever.search(query, top_k=3, features=feats)
        verdict = analyze_with_llm(desc, retrieved, context=context,
                                    backend=args.llm_backend, model=args.llm_model)
        top_meta = getattr(retrieved[0], "meta", {}) if retrieved else {}
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
        raw_attack_id = w["attack_id"].iloc[0] if "attack_id" in w.columns else None
        attack_id = (
            str(raw_attack_id)
            if raw_attack_id is not None and not pd.isna(raw_attack_id) and str(raw_attack_id)
            else str(cand["session_id"])
        )
        components = anomaly_score_components(
            feats,
            z_score=float(cand["z_score"]),
            baseline_mean_w=baseline,
        )

        row = {
            "session_id": str(cand["session_id"]),
            "attack_id": attack_id,
            "window_id": (
                f"{cand['session_id']}:gpu{int(cand['gpu_id'])}:rows{start}-{end}"
            ),
            "gpu_id": int(cand["gpu_id"]),
            "window_start": start,
            "window_end": end,
            "ground_truth": cand.get("label"),
            "swing_ratio": round(float(cand["swing_ratio"]), 3),
            "z_score": round(float(cand["z_score"]), 3),
            "candidate_reasons": str(cand["candidate_reasons"]).split(","),
            "features": {k: round(float(v), 5) for k, v in feats.items()},
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
            "score_version": "cyber-stage1-or-v1",
            "verdict": verdict,
        }

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

        results.append(row)

    print(f"\n[결과] 의심 후보 {len(results)}개 분석 완료:\n")
    for r in results:
        v = r["verdict"]
        print(f"  {r['session_id']}/GPU{r['gpu_id']} 윈도우@{r['window_start']} "
              f"(필터 {r['candidate_reasons']})")
        print(f"    → 위험도: {v.get('risk')} / 최근접: {v.get('closest_match')}")
        print(f"    → 근거: {v.get('reason')}")
        print(f"    → RAG: {r['rag_top']}\n")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"저장: {args.out}")
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
