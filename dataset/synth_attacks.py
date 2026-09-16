"""corpus의 정성적 특성에 대응하는 세 가지 공격 합성기."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.attack_profiles import CYBER_ATTACK_PROFILES
from dataset.declared_context import sample_declared
from dataset.synth_common import (
    BASELINE_POWER_W,
    build_frame,
    rng_for,
    smooth_noise,
    trailing_moving_average,
)


def _with_attrs(frame: pd.DataFrame, events: list[dict] | None = None) -> pd.DataFrame:
    frame.attrs["progress_events"] = events or []
    return frame


def _context(rng: np.random.Generator, family: str = "training", *, disguise: bool = True) -> dict:
    return sample_declared(family, rng, disguise=disguise)


def swma(
    n: int = 1200,
    sample_hz: float = 10,
    seed: int = 200,
    frequency_hz: float = 0.5,
    duty_cycle: float = 0.5,
    amplitude_frac: float = 1.0,
    nvml_avg_window_s: float = 0.0,
    variant: str = "swma",
) -> pd.DataFrame:
    """규칙적인 high/low 부하를 갖는 소프트웨어 관측 가능 SWMA-like 신호."""
    if not 0 < duty_cycle < 1:
        raise ValueError("duty_cycle은 0과 1 사이여야 합니다.")
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["swma"]
    phase = (np.arange(n) * frequency_hz / sample_hz) % 1.0
    active = phase < duty_cycle
    high = 120.0 + amplitude_frac * (330.0 - 120.0)
    low = 120.0 - amplitude_frac * (120.0 - 38.0)
    power_phys = np.where(active, high, low) + rng.normal(0, 3, n)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util_phys = np.where(active, 98.0, 1.0) + rng.normal(0, 1, n)
    util = trailing_moving_average(util_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="swma",
        session_id=f"syn-attack-{variant}-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant=variant,
        gt_params_json=f'{{"frequency_hz": {frequency_hz}, "duty_cycle": {duty_cycle}, "amplitude_frac": {amplitude_frac}}}',
        **_context(rng, "training", disguise=True),
        attack_id=(
            profile.attack_id
            if frequency_hz == profile.frequency_hz and duty_cycle == profile.duty_cycle
            else f"{variant}-f{frequency_hz:.2f}-d{duty_cycle:.2f}-s{seed}"
        ),
        waveform_kind=profile.kind,
        waveform_frequency_hz=frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=duty_cycle,
    ))


def swma_shallow(n: int = 1200, sample_hz: float = 10, seed: int = 210,
                 nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    return swma(
        n=n,
        sample_hz=sample_hz,
        seed=seed,
        frequency_hz=float(rng.uniform(0.5, 4.0)),
        duty_cycle=float(rng.uniform(0.3, 0.7)),
        amplitude_frac=float(rng.uniform(0.15, 0.35)),
        nvml_avg_window_s=nvml_avg_window_s,
        variant="swma_shallow",
    )


def swma_jitter(n: int = 1200, sample_hz: float = 10, seed: int = 211,
                nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    base_period = float(rng.uniform(0.5, 4.0))
    duty = float(rng.uniform(0.3, 0.7))
    sigma = float(rng.choice([0.10, 0.30, 0.50]))
    t = np.arange(n) / sample_hz
    power_phys = np.empty(n)
    active = False
    next_toggle = 0.0
    for idx, time_s in enumerate(t):
        if time_s >= next_toggle:
            active = not active
            frac = duty if active else 1.0 - duty
            next_toggle = time_s + max(0.1, base_period * frac * (1.0 + rng.normal(0, sigma)))
        power_phys[idx] = (330.0 if active else 38.0) + rng.normal(0, 3)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = trailing_moving_average(np.where(power_phys > 160, 98.0, 1.0), sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="swma",
        session_id=f"syn-attack-swma_jitter-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="swma_jitter",
        gt_params_json=f'{{"base_period_s": {base_period}, "jitter_sigma": {sigma}}}',
        attack_id=f"swma-jitter-s{seed}",
        waveform_kind="irregular",
        waveform_frequency_hz=1.0 / base_period,
        waveform_amplitude_frac=CYBER_ATTACK_PROFILES["swma"].cyber_amplitude_frac,
        waveform_duty_cycle=duty,
        **_context(rng, "training", disguise=True),
    ))


def swma_piggyback(n: int = 1200, sample_hz: float = 10, seed: int = 212,
                   nvml_avg_window_s: float = 1.0, mimicry: bool = False) -> pd.DataFrame:
    rng = rng_for(seed)
    t = np.arange(n) / sample_hz
    train_period = float(rng.uniform(1.0, 6.0))
    ratio = float(rng.uniform(0.3, 0.7) if not mimicry else rng.uniform(0.9, 1.1))
    attack_period = train_period * ratio
    train = np.where((t % train_period) < train_period * 0.75, 260.0, 170.0)
    modulation = 35.0 * np.sign(np.sin(2 * np.pi * t / attack_period))
    power_phys = train + modulation + rng.normal(0, 4, n)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = trailing_moving_average(np.clip((power_phys - 80) / 2.5, 0, 100), sample_hz, nvml_avg_window_s)
    events = [
        {"t": float(time_s), "gpu_id": 0, "event": "step_end"}
        for time_s in np.arange(train_period, n / sample_hz, train_period)
    ]
    variant = "swma_mimicry" if mimicry else "swma_piggyback"
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="swma",
        session_id=f"syn-attack-{variant}-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant=variant,
        gt_params_json=f'{{"train_period_s": {train_period}, "attack_period_s": {attack_period}}}',
        attack_id=f"{variant}-s{seed}",
        waveform_kind="square",
        waveform_frequency_hz=1.0 / attack_period,
        waveform_amplitude_frac=0.15,
        waveform_duty_cycle=0.5,
        **_context(rng, "training", disguise=True),
    ), events)


def swma_mimicry(n: int = 1200, sample_hz: float = 10, seed: int = 213,
                 nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    return swma_piggyback(n, sample_hz, seed, nvml_avg_window_s, mimicry=True)


def swma_coordinated(n: int = 1200, sample_hz: float = 10, seed: int = 214,
                     nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    left = swma(n, sample_hz, seed, frequency_hz=0.5, nvml_avg_window_s=nvml_avg_window_s, variant="swma_coordinated")
    right = swma(n, sample_hz, seed + 1, frequency_hz=0.5, nvml_avg_window_s=nvml_avg_window_s, variant="swma_coordinated")
    right["gpu_id"] = 1
    right["session_id"] = left["session_id"].iloc[0]
    right["declared_user"] = "user_049"
    right["id_user"] = "user_049"
    combined = pd.concat([left, right], ignore_index=True)
    combined.attrs["progress_events"] = []
    return combined


def ltma(n: int = 1200, sample_hz: float = 10, seed: int = 201,
         nvml_avg_window_s: float = 0.0, variant: str = "ltma") -> pd.DataFrame:
    """평균은 정상 부근이지만 학습 이벤트에 결합된 불규칙 변조 신호."""
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["ltma"]
    power = BASELINE_POWER_W + smooth_noise(rng, n, 24, width=5)
    cursor = 0
    sign = 1.0
    while cursor < n:
        width = int(rng.integers(7, 75))
        amplitude = float(rng.uniform(28, 85))
        power[cursor : cursor + width] += sign * amplitude
        sign *= -1 if rng.random() > 0.25 else 1
        cursor += width
    power_phys = np.clip(power - (np.mean(power) - BASELINE_POWER_W), 28, 260)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = np.clip(45 + (power_phys - BASELINE_POWER_W) / 1.8 + rng.normal(0, 8, n), 0, 100)
    events = [{"t": float(t), "gpu_id": 0, "event": "step_end"} for t in np.arange(1.5, n / sample_hz, 1.5)]
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="ltma",
        session_id=f"syn-attack-ltma-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant=variant,
        gt_params_json=f'{{"variant": "{variant}"}}',
        **_context(rng, "training", disguise=True),
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    ), events)


def cryptojacking(n: int = 1200, sample_hz: float = 10, seed: int = 202,
                  nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """오랫동안 높고 평평하게 유지되는 무단 채굴성 부하 근사."""
    rng = rng_for(seed)
    profile = CYBER_ATTACK_PROFILES["cryptojacking"]
    power_phys = 285 + smooth_noise(rng, n, 5, width=11)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = np.clip(96 + rng.normal(0, 1.2, n), 90, 100)
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="cryptojacking",
        session_id=f"syn-attack-crypto-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="cryptojacking",
        gt_params_json='{"flat_high_load": true}',
        **_context(rng, "training", disguise=True),
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.cyber_amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    ))


ATTACK_GENERATORS = {
    "swma": swma,
    "swma_shallow": swma_shallow,
    "swma_jitter": swma_jitter,
    "swma_piggyback": swma_piggyback,
    "swma_mimicry": swma_mimicry,
    "swma_coordinated": swma_coordinated,
    "ltma": ltma,
    "cryptojacking": cryptojacking,
}

