"""
피처 추출: 텔레메트리 시계열 → 통계 피처 → 정성적 특성(자연어 설명).

이 모듈이 파이프라인의 1·2단계 핵심 로직을 담는다:
  - compute_window_features: 숫자 피처 계산 (변동폭, 주기성, 평균유지 등)
  - features_to_description: 숫자 피처를 RAG 검색용 자연어 설명으로 변환

핵심 설계: 원 신호(kHz)를 직접 보지 않고, 관측 가능한 해상도에서의
통계적 특성만 사용한다. 그리고 RAG 매칭은 숫자가 아니라 "정성적 설명"으로 한다
(정상/공격을 '규칙적인가', '평균이 유지되는가' 같은 특성으로 구분).
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats


def compute_window_features(power: np.ndarray,
                             util: np.ndarray | None = None,
                             sample_hz: float = 1000.0) -> dict:
    """한 시간창의 전력(및 선택적 사용률) 시계열에서 피처를 계산.

    Args:
        power: 전력 시계열 (W)
        util:  GPU 사용률 시계열 (%) — 있으면 추가 피처 사용
        sample_hz: 이 시계열의 실효 샘플링 레이트 (주기 추정에 사용)
    """
    raw_power = np.asarray(power, dtype=float)
    power_mask = np.isfinite(raw_power)
    power = raw_power[power_mask]
    if len(power) < 8:
        return {}

    mean = float(np.mean(power))
    std = float(np.std(power))
    pmin, pmax = float(np.min(power)), float(np.max(power))
    swing_ratio = (pmax - pmin) / mean if mean > 1e-6 else 0.0
    cv = std / mean if mean > 1e-6 else 0.0  # 변동계수
    ramp = np.abs(np.diff(power)) * sample_hz
    ramp_mean_w_per_s = float(np.mean(ramp)) if len(ramp) else 0.0
    ramp_max_w_per_s = float(np.max(ramp)) if len(ramp) else 0.0

    # --- 주기성 / 규칙성 (자기상관 기반) ---
    x = power - mean
    if np.any(x):
        acf = np.correlate(x, x, mode="full")[len(x) - 1:]
        acf = acf / acf[0] if acf[0] != 0 else acf
        # 첫 번째 유의미한 피크(lag>2)를 찾아 주기성 강도 추정
        search = acf[2:min(len(acf), len(acf) // 2)]
        periodicity_strength = float(np.max(search)) if len(search) else 0.0
        peak_lag = int(np.argmax(search) + 2) if len(search) else 0
        dominant_freq_hz = sample_hz / peak_lag if peak_lag > 0 else 0.0
    else:
        periodicity_strength = 0.0
        dominant_freq_hz = 0.0

    # --- duty cycle 규칙성 (high/low 전환의 균일성) ---
    threshold = (pmax + pmin) / 2
    high = power > threshold
    # 전환 지점 간격의 변동계수가 낮을수록 규칙적
    transitions = np.where(np.diff(high.astype(int)) != 0)[0]
    if len(transitions) >= 3:
        intervals = np.diff(transitions)
        duty_regularity = 1.0 - min(1.0, float(np.std(intervals) / (np.mean(intervals) + 1e-6)))
    else:
        duty_regularity = 0.0  # 전환이 거의 없음 = 평평함

    # --- 지속성: 높은 부하가 얼마나 오래 연속되는지 ---
    high_load = power > max(200.0, mean * 0.9)
    high_load_fraction = float(np.mean(high_load))
    longest = current = 0
    for value in high_load:
        current = current + 1 if value else 0
        longest = max(longest, current)
    longest_high_seconds = float(longest / sample_hz) if sample_hz > 0 else 0.0

    feats = {
        "mean_w": mean,
        "std_w": std,
        "swing_ratio": swing_ratio,
        "cv": cv,
        "periodicity_strength": periodicity_strength,
        "dominant_freq_hz": dominant_freq_hz,
        "duty_regularity": duty_regularity,
        "n_transitions": int(len(transitions)),
        "ramp_mean_w_per_s": ramp_mean_w_per_s,
        "ramp_max_w_per_s": ramp_max_w_per_s,
        "high_load_fraction": high_load_fraction,
        "longest_high_seconds": longest_high_seconds,
    }

    if util is not None:
        raw_util = np.asarray(util, dtype=float)
        if len(raw_util) == len(raw_power):
            aligned = raw_util[power_mask]
            aligned = aligned[np.isfinite(aligned)]
            if len(aligned):
                feats["util_mean"] = float(np.mean(aligned))
                feats["util_std"] = float(np.std(aligned))
    return feats


def features_to_description(feats: dict, baseline_mean_w: float | None = None) -> str:
    """숫자 피처를 RAG 검색용 자연어 설명으로 변환.

    핵심: '평균이 유지되는가', '규칙적인가', '지속적으로 높은가' 같은
    정성적 특성으로 표현해야 공격 시그니처 문서와 의미 기반 매칭이 된다.
    """
    if not feats:
        return "분석 불가: 데이터 부족."

    parts = []

    # 평균 전력 수준
    if baseline_mean_w:
        ratio = feats["mean_w"] / baseline_mean_w if baseline_mean_w > 0 else 1.0
        if ratio > 1.3:
            parts.append("평균 전력이 정상 기준선보다 크게 높게 지속됨")
        elif ratio < 0.8:
            parts.append("평균 전력이 정상 기준선보다 낮음")
        else:
            parts.append("평균 전력은 정상 범위 내로 유지됨")
    else:
        parts.append(f"평균 전력 약 {feats['mean_w']:.0f}W")

    # 변동폭
    sr = feats["swing_ratio"]
    if sr > 1.5:
        parts.append(f"전력 변동폭이 매우 큼 (swing ratio {sr:.2f})")
    elif sr > 0.5:
        parts.append(f"전력 변동폭이 다소 큼 (swing ratio {sr:.2f})")
    else:
        parts.append(f"전력이 비교적 평평함 (swing ratio {sr:.2f}, 변동 적음)")

    # 규칙성
    dr = feats["duty_regularity"]
    if feats["n_transitions"] < 3:
        parts.append("on/off 전환이 거의 없어 지속적인 부하 상태")
    elif dr > 0.6:
        parts.append("high/low 전환이 매우 규칙적이고 기계적임")
    else:
        parts.append("high/low 전환이 불규칙하고 예측하기 어려움")

    # 주기성
    ps = feats["periodicity_strength"]
    if ps > 0.5:
        parts.append(f"지속적이고 뚜렷한 주기성 존재 (강도 {ps:.2f})")
    elif ps > 0.2:
        parts.append(f"약한 주기성 존재 (강도 {ps:.2f})")
    else:
        parts.append("뚜렷한 주기성 없음")

    if feats.get("high_load_fraction", 0.0) > 0.8:
        parts.append("높은 부하가 시간창 대부분에서 지속됨")
    if feats.get("ramp_max_w_per_s", 0.0) > 500:
        parts.append("순간 전력 상승·하강 기울기가 큼")

    return ". ".join(parts) + "."


def _band_key(lo: float, hi: float) -> str:
    hi_text = "nyq" if np.isinf(hi) else f"{hi:g}"
    return f"{lo:g}_{hi_text}"


def compute_window_features_v2(
    power: np.ndarray,
    util: np.ndarray | None = None,
    sample_hz: float = 1.0,
    *,
    band_edges_hz: tuple[tuple[float, float], ...] = ((0.05, 0.1), (0.1, 0.7), (0.7, 2.0), (2.0, np.inf)),
    tdp_w: float = 450.0,
    nvml_avg_window_s: float | None = None,
) -> dict:
    """AI datacenter oriented Stage-1 v2 features."""
    base = compute_window_features(power, util, sample_hz=sample_hz)
    if not base:
        return {}
    x = np.asarray(power, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return base
    centered = x - np.nanmean(x)
    nperseg = min(256, len(centered))
    freqs, psd = signal.welch(centered, fs=sample_hz, nperseg=nperseg)
    positive = freqs > 0
    freqs, psd = freqs[positive], psd[positive]
    total_power = float(np.trapz(psd, freqs)) if len(freqs) else 0.0
    band_power = {}
    band_frac = {}
    reliability = {}
    for lo, hi in band_edges_hz:
        upper = min(hi, sample_hz / 2.0) if np.isfinite(hi) else sample_hz / 2.0
        mask = (freqs >= lo) & (freqs < upper)
        key = _band_key(lo, hi)
        value = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        band_power[key] = value
        band_frac[key] = value / total_power if total_power > 0 else 0.0
        center = (lo + upper) / 2.0 if upper > lo else lo
        reliability[key] = 1.0 if not nvml_avg_window_s else float(abs(np.sinc(center * nvml_avg_window_s)))
    if len(psd):
        peak_idx = int(np.argmax(psd))
        dominant_freq_hz = float(freqs[peak_idx])
        median_psd = float(np.median(psd)) if np.median(psd) > 0 else 1e-12
        dominant_peak_prominence = float(psd[peak_idx] / median_psd)
        prob = psd / np.sum(psd) if np.sum(psd) > 0 else np.zeros_like(psd)
        spectral_entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])) / np.log2(len(prob))) if len(prob) > 1 else 0.0
    else:
        dominant_freq_hz = 0.0
        dominant_peak_prominence = 0.0
        spectral_entropy = 0.0

    p5, p95 = np.percentile(x, [5, 95])
    swing_abs_w = float(p95 - p5)
    ramp = np.abs(np.diff(x)) * sample_hz
    out = {
        **base,
        "dominant_freq_hz": dominant_freq_hz,
        "dominant_peak_prominence": dominant_peak_prominence,
        "spectral_entropy": spectral_entropy,
        "band_power_w2": band_power,
        "band_frac": band_frac,
        "band_reliability": reliability,
        "swing_abs_w": swing_abs_w,
        "swing_frac_tdp": swing_abs_w / tdp_w if tdp_w > 0 else 0.0,
        "ramp_p95_w_per_s": float(np.percentile(ramp, 95)) if len(ramp) else 0.0,
        "quality_flag": "ok" if nvml_avg_window_s is not None else "assumed_bandwidth",
    }
    if util is not None:
        u = np.asarray(util, dtype=float)
        mask = np.isfinite(u) & np.isfinite(np.asarray(power, dtype=float))
        if np.count_nonzero(mask) >= 3:
            slope, intercept, _, _ = stats.theilslopes(np.asarray(power, dtype=float)[mask], u[mask])
            residual = np.asarray(power, dtype=float)[mask] - (slope * u[mask] + intercept)
            med = np.median(residual)
            out.update({
                "util_slope_w_per_pct": float(slope),
                "util_intercept_w": float(intercept),
                "util_residual_mad_w": float(np.median(np.abs(residual - med))),
            })
    return out


if __name__ == "__main__":
    # 간단 자체 테스트: 규칙적 사각파(SWMA류) vs 평평한 고부하(크립토재킹류)
    t = np.linspace(0, 1, 1000)
    swma_like = 100 + 100 * (np.sign(np.sin(2 * np.pi * 50 * t)) > 0)
    crypto_like = 200 + np.random.randn(1000) * 3

    print("=== SWMA류 (규칙적 사각파) ===")
    f1 = compute_window_features(swma_like, sample_hz=1000)
    print(features_to_description(f1, baseline_mean_w=120))
    print(f1)
    print()
    print("=== 크립토재킹류 (평평한 고부하) ===")
    f2 = compute_window_features(crypto_like, sample_hz=1000)
    print(features_to_description(f2, baseline_mean_w=120))
    print(f2)
