"""정상 학습 루프에 불규칙 보조 연산을 삽입하는 LTMA-like wrapper (gp-telemetry/1.2)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workloads.progress_log import ProgressLog


def _read_gpu_power_w(gpu_id: int) -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
            return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _mlp_training_step(args, device, torch, nn):
    layers = [nn.Linear(args.width, args.width), nn.GELU()]
    for _ in range(max(0, args.depth - 1)):
        layers.extend([nn.Linear(args.width, args.width), nn.GELU()])
    layers.append(nn.Linear(args.width, args.width))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = nn.Sequential(*layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        batch = random.choice([16, 24, 32, 40])
        x = torch.randn(batch, args.width, device=device, dtype=dtype)
        target = torch.randn_like(x)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()

    return step


def _build_host_step(args, device, torch, nn):
    if args.host == "llm":
        try:
            from workloads.llm_workloads import make_host_training_step

            preset = getattr(args, "llm_preset", None) or "small"
            return (
                make_host_training_step(
                    device,
                    preset=preset,
                    batch_size=args.llm_micro_batch,
                    seq_len=args.seq_len,
                    seed=args.seed,
                ),
                "llm",
            )
        except (ImportError, AttributeError, RuntimeError, ValueError):
            # Keep LTMA capturable on installations where the GPT path is unavailable.
            return _mlp_training_step(args, device, torch, nn), "mlp_fallback"
    return _mlp_training_step(args, device, torch, nn), "mlp"


def mean_preserving_repeats(
    *,
    baseline_w: float | None,
    recent_w: float | None,
    phase_high: bool,
    max_aux_repeats: int,
    tolerance_w: float = 8.0,
) -> int:
    """Choose aux repeats so the window mean stays near the host-only baseline."""
    high = max(1, int(max_aux_repeats))
    if baseline_w is None or recent_w is None:
        return high if phase_high else 0
    if recent_w > baseline_w + tolerance_w:
        return 0
    if recent_w < baseline_w - tolerance_w:
        return high
    return high if phase_high else 0


def run(args) -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("CUDA 지원 PyTorch가 필요합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if not 0 < args.duration <= 1800:
        raise ValueError("--duration은 0초 초과 1800초 이하여야 합니다.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    host_step, host_kind = _build_host_step(args, device, torch, nn)
    aux_w = int(getattr(args, "aux_width", None) or args.width)
    if host_kind.startswith("mlp") and getattr(args, "aux_width", None) is None:
        aux_w = int(args.width)
    aux_a = torch.randn(aux_w, aux_w, device=device, dtype=torch.float16)
    aux_b = torch.randn_like(aux_a)
    aux_out = torch.randn_like(aux_a)
    started = time.monotonic()
    deadline = started + args.duration
    baseline_s = min(30.0, max(8.0, args.duration * 0.15))
    baseline_deadline = started + baseline_s
    modulate_s = 30.0
    phase_high = True
    phase_deadline = baseline_deadline + modulate_s
    step = inserted = 0
    attack_intervals: list[list[float]] = []
    baseline_samples: list[float] = []
    window_samples: list[float] = []
    log = ProgressLog(args.progress_log, int(args.gpu_id))
    log.emit("workload_start", host=args.host, ltma_mode="mean_preserving")

    try:
        while time.monotonic() < deadline:
            host_step()
            torch.cuda.synchronize()
            reading = _read_gpu_power_w(args.gpu_id)
            now = time.monotonic()
            if now < baseline_deadline:
                if reading is not None:
                    baseline_samples.append(reading)
                log.emit("step_end", step=step)
                step += 1
                continue
            if now >= phase_deadline:
                phase_high = not phase_high
                phase_deadline = now + modulate_s
            baseline_w = (
                sum(baseline_samples) / len(baseline_samples) if baseline_samples else None
            )
            recent_w = (
                sum(window_samples[-20:]) / len(window_samples[-20:]) if window_samples else None
            )
            repeats = mean_preserving_repeats(
                baseline_w=baseline_w,
                recent_w=recent_w,
                phase_high=phase_high,
                max_aux_repeats=args.max_aux_repeats,
            )
            if repeats:
                inj_start = time.time()
                for _ in range(repeats):
                    torch.mm(aux_a, aux_b, out=aux_out)
                torch.cuda.synchronize()
                attack_intervals.append([inj_start, time.time()])
                inserted += 1
            if reading is not None:
                window_samples.append(reading)
            log.emit("step_end", step=step)
            step += 1
        torch.cuda.synchronize()
    finally:
        log.emit("workload_end", steps=step, injections=inserted)
        log.close()
    baseline_w = sum(baseline_samples) / len(baseline_samples) if baseline_samples else None
    print(
        json.dumps(
            {
                "workload": "ltma-like",
                "host": args.host,
                "resolved_host": host_kind,
                "gpu_id": args.gpu_id,
                "duration_s": args.duration,
                "training_steps": step,
                "injection_events": inserted,
                "ltma_mode": "mean_preserving",
                "baseline_w": baseline_w,
                "aux_width": aux_w,
                "approximation": True,
                "native_progress_available": True,
                "progress_log": args.progress_log,
                "gt_attack_intervals_epoch": {str(args.gpu_id): attack_intervals},
            }
        )
    )


def _duration(value: str) -> float:
    duration = float(value)
    if not 0 < duration <= 1800:
        raise argparse.ArgumentTypeError("duration must be in (0, 1800]")
    return duration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=_duration, default=30.0)
    parser.add_argument("--host", choices=["mlp", "llm"], default="llm")
    parser.add_argument("--llm-preset", default="small")
    parser.add_argument("--llm-micro-batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--aux-width", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-aux-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-log", default=None, help="JSONL progress log path (host steps only)")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
