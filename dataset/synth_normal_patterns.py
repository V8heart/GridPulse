"""정상인데 전력 변동이 큰 다섯 가지 hard-negative 합성기."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.attack_profiles import PHASE1_PROFILES
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


def _context(rng: np.random.Generator, family: str = "training") -> dict:
    return sample_declared(family, rng, disguise=False)


def _piecewise(
    rng: np.random.Generator,
    n: int,
    levels: tuple[float, ...],
    min_len: int,
    max_len: int,
) -> np.ndarray:
    values: list[float] = []
    previous = -1
    while len(values) < n:
        choices = [i for i in range(len(levels)) if i != previous]
        idx = int(rng.choice(choices))
        previous = idx
        values.extend([levels[idx]] * int(rng.integers(min_len, max_len + 1)))
    return np.asarray(values[:n], dtype=float)


def normal_distributed_training(n: int = 1200, sample_hz: float = 10, seed: int = 101,
                                nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """compute/all-reduce 위상이 불규칙하게 전환되는 분산학습 근사."""
    rng = rng_for(seed)
    power = _piecewise(rng, n, (205, 145, 185), 13, 61)
    power_phys = power + smooth_noise(rng, n, 12, width=7)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = np.clip((power_phys - 85) / 1.35 + rng.normal(0, 5, n), 5, 100)
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="normal_distributed_training",
        session_id=f"syn-normal-ddp-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        declared_gres="gpu:2",
        power_phys_w=power_phys,
        gt_variant="legacy_irregular_ddp",
        **_context(rng, "training"),
    ))


def normal_hpo_search(n: int = 1200, sample_hz: float = 10, seed: int = 102,
                      nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """서로 다른 강도의 짧은 학습 trial과 준비 구간이 반복되는 HPO 근사."""
    rng = rng_for(seed)
    power = np.full(n, 35.0)
    cursor = 0
    while cursor < n:
        idle = int(rng.integers(8, 35))
        run = int(rng.integers(35, 125))
        cursor += idle
        end = min(n, cursor + run)
        power[cursor:end] = float(rng.uniform(135, 260))
        cursor = end
    power_phys = power + smooth_noise(rng, n, 8)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        label="normal_hpo_search",
        session_id=f"syn-normal-hpo-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="hpo_search",
        **_context(rng, "training"),
    ))


def normal_checkpoint(n: int = 1200, sample_hz: float = 10, seed: int = 103,
                      nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """지속 학습 중 체크포인트 저장 때 짧게 전력이 떨어지는 패턴."""
    rng = rng_for(seed)
    power_phys = 190 + smooth_noise(rng, n, 10)
    events = []
    cursor = int(rng.integers(120, 180))
    while cursor < n:
        width = int(rng.integers(8, 24))
        stop = min(n, cursor + width)
        power_phys[cursor:stop] = 55 + rng.normal(0, 5, stop - cursor)
        events.extend([
            {"t": float(cursor / sample_hz), "gpu_id": 0, "event": "checkpoint_start"},
            {"t": float(stop / sample_hz), "gpu_id": 0, "event": "checkpoint_end"},
        ])
        cursor += int(rng.integers(150, 260))
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        label="normal_checkpoint",
        session_id=f"syn-normal-checkpoint-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="checkpoint",
        **_context(rng, "training"),
    ), events)


def normal_dataloader_stall(n: int = 1200, sample_hz: float = 10, seed: int = 104,
                            nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """불규칙한 I/O 대기로 인한 비주기적 전력 하락."""
    rng = rng_for(seed)
    power_phys = 175 + smooth_noise(rng, n, 15, width=5)
    candidates = np.arange(max(5, n // 10), max(6, n - max(5, n // 10)))
    stall_count = min(18, len(candidates))
    for start in rng.choice(candidates, size=stall_count, replace=False):
        width = int(rng.integers(4, 35))
        power_phys[start : start + width] = 45 + rng.normal(0, 7, min(width, n - start))
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        label="normal_dataloader_stall",
        session_id=f"syn-normal-dataloader-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="dataloader_stall",
        **_context(rng, "training"),
    ))


def normal_eval_train_switch(n: int = 1200, sample_hz: float = 10, seed: int = 105,
                             nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """학습과 평가가 서로 다른 길이·전력 수준으로 교차하는 패턴."""
    rng = rng_for(seed)
    power_phys = _piecewise(rng, n, (205, 105), 45, 180)
    power_phys += smooth_noise(rng, n, 9)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    events = [{"t": float(t), "gpu_id": 0, "event": "eval_start"} for t in np.arange(6.0, n / sample_hz, 18.0)]
    return _with_attrs(build_frame(
        power,
        label="normal_eval_train_switch",
        session_id=f"syn-normal-eval-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="eval_train_switch",
        **_context(rng, "training"),
    ), events)


def baseline_idle(n: int = 1200, sample_hz: float = 10, seed: int = 100,
                  nvml_avg_window_s: float = 0.0) -> pd.DataFrame:
    """임계값 캘리브레이션을 위한 비교적 안정적인 정상 기준선."""
    rng = rng_for(seed)
    profile = PHASE1_PROFILES["normal"]
    power_phys = BASELINE_POWER_W + smooth_noise(rng, n, 6)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        label="normal_baseline",
        session_id=f"syn-normal-baseline-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="baseline_idle",
        **_context(rng, "interactive"),
        attack_id=profile.attack_id,
        waveform_kind=profile.kind,
        waveform_frequency_hz=profile.frequency_hz,
        waveform_amplitude_frac=profile.amplitude_frac,
        waveform_duty_cycle=profile.duty_cycle,
    ))


def normal_sync_ddp(n: int = 1200, sample_hz: float = 10, seed: int = 106,
                    nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    period = float(rng.uniform(1.0, 6.0))
    duty = float(rng.uniform(0.6, 0.85))
    high, low = float(rng.uniform(330, 440)), float(rng.uniform(150, 220))
    t = np.arange(n) / sample_hz
    phase = (t % period) / period
    power_phys = np.where(phase < duty, high, low) + rng.normal(0, 4, n)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    util = trailing_moving_average(np.where(phase < duty, 95, 45), sample_hz, nvml_avg_window_s)
    events = [{"t": float(x), "gpu_id": 0, "event": "step_end"} for x in np.arange(period, n / sample_hz, period)]
    return _with_attrs(build_frame(
        power,
        util_gpu_pct=util,
        label="normal_sync_ddp",
        session_id=f"syn-normal-sync-ddp-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        declared_gres="gpu:1",
        power_phys_w=power_phys,
        gt_variant="sync_ddp",
        gt_params_json=f'{{"period_s": {period}, "duty": {duty}}}',
        **_context(rng, "training"),
    ), events)


def normal_fsdp_deep_trough(n: int = 1200, sample_hz: float = 10, seed: int = 107,
                            nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    period = float(rng.uniform(1.5, 4.0))
    low = float(rng.uniform(60, 120))
    t = np.arange(n) / sample_hz
    phase = (t % period) / period
    power_phys = np.where(phase < 0.75, 360.0, low) + rng.normal(0, 5, n)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    events = [{"t": float(x), "gpu_id": 0, "event": "step_end"} for x in np.arange(period, n / sample_hz, period)]
    return _with_attrs(build_frame(
        power,
        label="normal_fsdp_deep_trough",
        session_id=f"syn-normal-fsdp-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="fsdp_deep_trough",
        **_context(rng, "training"),
    ), events)


def normal_flat_pretrain(n: int = 1200, sample_hz: float = 10, seed: int = 108,
                         nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    power_phys = rng.uniform(400, 440) + smooth_noise(rng, n, 4, width=7)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    events = [{"t": float(x), "gpu_id": 0, "event": "step_end"} for x in np.arange(2.0, n / sample_hz, 2.0)]
    return _with_attrs(build_frame(
        power,
        label="normal_flat_pretrain",
        session_id=f"syn-normal-flat-pretrain-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="flat_pretrain",
        **_context(rng, "training"),
    ), events)


def normal_inference_bursty(n: int = 1200, sample_hz: float = 10, seed: int = 109,
                            nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    rng = rng_for(seed)
    t_end = n / sample_hz
    power_phys = np.full(n, 70.0)
    events = []
    now = 0.0
    lam = float(rng.uniform(0.1, 1.0))
    while now < t_end:
        now += float(rng.exponential(1.0 / lam))
        length = float(rng.uniform(0.3, 2.0))
        start, stop = int(now * sample_hz), min(n, int((now + length) * sample_hz))
        if start < n:
            power_phys[start:stop] = float(rng.uniform(220, 420))
            events.extend([
                {"t": float(now), "gpu_id": 0, "event": "request_in"},
                {"t": float(min(t_end, now + length)), "gpu_id": 0, "event": "request_out"},
            ])
    power_phys += smooth_noise(rng, n, 6)
    power = trailing_moving_average(power_phys, sample_hz, nvml_avg_window_s)
    return _with_attrs(build_frame(
        power,
        label="normal_inference_bursty",
        session_id=f"syn-normal-infer-bursty-{seed}",
        sample_hz=sample_hz,
        seed=seed,
        power_phys_w=power_phys,
        gt_variant="inference_bursty",
        **_context(rng, "inference"),
    ), events)


def normal_mixed_tenants(n: int = 1200, sample_hz: float = 10, seed: int = 110,
                         nvml_avg_window_s: float = 1.0) -> pd.DataFrame:
    left = normal_sync_ddp(n, sample_hz, seed, nvml_avg_window_s)
    right = normal_inference_bursty(n, sample_hz, seed + 1, nvml_avg_window_s)
    right["gpu_id"] = 1
    right["session_id"] = left["session_id"].iloc[0].replace("sync-ddp", "mixed")
    left["session_id"] = right["session_id"].iloc[0]
    combined = pd.concat([left, right], ignore_index=True)
    combined.attrs["progress_events"] = left.attrs.get("progress_events", []) + [
        {**event, "gpu_id": 1} for event in right.attrs.get("progress_events", [])
    ]
    combined["label"] = "normal_mixed_tenants"
    combined["gt_label"] = "normal_mixed_tenants"
    combined["gt_variant"] = "mixed_tenants"
    return combined


NORMAL_GENERATORS = {
    "normal_baseline": baseline_idle,
    "normal_distributed_training": normal_distributed_training,
    "normal_hpo_search": normal_hpo_search,
    "normal_checkpoint": normal_checkpoint,
    "normal_dataloader_stall": normal_dataloader_stall,
    "normal_eval_train_switch": normal_eval_train_switch,
    "normal_sync_ddp": normal_sync_ddp,
    "normal_fsdp_deep_trough": normal_fsdp_deep_trough,
    "normal_flat_pretrain": normal_flat_pretrain,
    "normal_inference_bursty": normal_inference_bursty,
    "normal_mixed_tenants": normal_mixed_tenants,
}

