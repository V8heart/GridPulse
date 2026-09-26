"""PyTorch 연산/sleep 토글로 규칙적 SWMA-like 부하를 재현한다.

논문의 persistent CUDA kernel을 동일 재현하는 코드가 아니라, NVML/DCGM에서
관측 가능한 규칙적 전력 변동을 검증하기 위한 통제된 근사 워크로드다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workloads.progress_log import DecoyStepper, ProgressLog

MIN_SWMA_PERIOD_S = 0.05


def allocate_independent_streams(torch, device, matrix_size: int, n_streams: int, dtype):
    """One independent (A, B, out) triple per stream — no a@b chaining."""
    streams = []
    for _ in range(max(1, int(n_streams))):
        left = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        right = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        out = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        streams.append((left, right, out))
    return streams


def run(args) -> None:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise RuntimeError("CUDA 지원 PyTorch가 필요합니다.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if not 0 < args.duration <= 1800:
        raise ValueError("--duration은 0초 초과 1800초 이하여야 합니다.")
    if not 0.1 <= args.duty_cycle <= 0.9:
        raise ValueError("--duty-cycle은 0.1~0.9 범위여야 합니다.")
    if args.period < MIN_SWMA_PERIOD_S:
        raise ValueError("--period는 최소 50ms입니다. kHz 물리 재현용 도구가 아닙니다.")

    rank = 0
    if args.distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        args.gpu_id = rank
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)

    decoy_period = args.decoy_step_s
    if decoy_period is None:
        decoy_period = random.uniform(1.0, 4.0)

    log = ProgressLog(args.progress_log if rank == 0 or args.distributed else None, int(args.gpu_id))
    # DDP: each rank logs with its gpu_id to the same file
    if args.distributed:
        log = ProgressLog(args.progress_log, int(args.gpu_id))
    decoy = DecoyStepper(log, decoy_period, jitter_frac=0.05, rng=random.Random(args.seed)) if args.progress_log else None

    streams = max(1, int(args.active_streams))
    mats = allocate_independent_streams(
        torch, device, int(args.matrix_size), streams, torch.float16
    )
    attack_intervals: list[list[float]] = []
    deadline = time.monotonic() + args.duration
    cycles = 0
    log.emit("workload_start")
    if args.distributed:
        dist.barrier()
        deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        period = args.period * (1.0 + random.uniform(-args.jitter_frac, args.jitter_frac))
        active_deadline = cycle_start + period * args.duty_cycle
        active_start_epoch = time.time()
        while time.monotonic() < active_deadline:
            for left, right, out in mats:
                torch.mm(left, right, out=out)
            if decoy is not None:
                decoy.tick()
        torch.cuda.synchronize()
        attack_intervals.append([active_start_epoch, time.time()])
        passive_deadline = cycle_start + period
        while time.monotonic() < passive_deadline:
            if decoy is not None:
                decoy.tick()
            time.sleep(0.001)
        cycles += 1
    log.emit("workload_end")
    log.close()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({
            "workload": "swma-like",
            "gpu_id": args.gpu_id,
            "distributed": args.distributed,
            "duration_s": args.duration,
            "period_s": args.period,
            "duty_cycle": args.duty_cycle,
            "jitter_frac": args.jitter_frac,
            "active_streams": streams,
            "matrix_size": args.matrix_size,
            "decoy_step_s": decoy_period,
            "cycles": cycles,
            "approximation": True,
            "gt_attack_intervals_epoch": {str(args.gpu_id): attack_intervals},
            "progress_log": args.progress_log,
            "native_progress_available": True,
        }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--active-streams", type=int, default=2)
    parser.add_argument("--jitter-frac", type=float, default=0.0)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--decoy-step-s", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distributed", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
