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
            from workloads.llm_workloads import make_tiny_training_step

            return (
                make_tiny_training_step(
                    device,
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
    aux_w = args.width if host_kind.startswith("mlp") else 512
    aux_a = torch.randn(aux_w, aux_w, device=device, dtype=torch.float16)
    aux_b = torch.randn_like(aux_a)
    deadline = time.monotonic() + args.duration
    step = inserted = 0
    next_injection = random.randint(7, 23)
    attack_intervals: list[list[float]] = []
    log = ProgressLog(args.progress_log, int(args.gpu_id))
    log.emit("workload_start", host=args.host)

    try:
        while time.monotonic() < deadline:
            host_step()
            torch.cuda.synchronize()
            # Host step only — never log injection as progress events.
            log.emit("step_end", step=step)
            step += 1

            if step >= next_injection:
                inj_start = time.time()
                repeats = random.randint(1, args.max_aux_repeats)
                for _ in range(repeats):
                    aux_a = torch.tanh(aux_a @ aux_b)
                torch.cuda.synchronize()
                attack_intervals.append([inj_start, time.time()])
                if random.random() < 0.5:
                    time.sleep(random.uniform(0.01, 0.15))
                inserted += 1
                next_injection = step + random.randint(5, 31)
        torch.cuda.synchronize()
    finally:
        log.emit("workload_end", steps=step, injections=inserted)
        log.close()
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
    parser.add_argument("--llm-preset", default="tiny")
    parser.add_argument("--llm-micro-batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
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
