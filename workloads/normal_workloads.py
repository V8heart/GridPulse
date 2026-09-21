"""정상 hard-negative 캡처를 위한 작은 PyTorch 학습 워크로드."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def _torch():
    try:
        import torch
        import torch.distributed as dist
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("실측 워크로드에는 CUDA 지원 PyTorch가 필요합니다.") from exc
    return torch, dist, nn


def run(args) -> None:
    torch, dist, nn = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")
    if args.max_seconds <= 0 or args.max_seconds > 300:
        raise ValueError("--max-seconds는 0초 초과 300초 이하여야 합니다.")

    distributed = args.mode == "distributed"
    rank = 0
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    deadline = time.monotonic() + args.max_seconds
    step = 0
    events: list[dict] = []
    t0_wall = time.time()
    gpu_id = int(rank if distributed else args.gpu_id)

    while time.monotonic() < deadline:
        if args.mode == "hpo" and step and step % 25 == 0:
            optimizer.param_groups[0]["lr"] = random.choice([1e-4, 3e-4, 1e-3, 3e-3])
            time.sleep(random.uniform(0.1, 0.7))
        if args.mode == "dataloader_stall" and step % 13 == 0:
            time.sleep(random.uniform(0.05, 0.8))

        batch = random.choice([16, 24, 32, 48]) if args.mode == "hpo" else args.batch_size
        x = torch.randn(batch, args.width, device=device)
        target = torch.randn_like(x)
        training = not (args.mode == "eval_train_switch" and (step // 20) % 3 == 2)
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(x)

        now = time.time()
        rel_t = float(now - t0_wall)
        if args.mode == "eval_train_switch" and not training and rank == 0 and step % 20 == 0:
            events.append({
                "t": rel_t,
                "t_epoch": now,
                "gpu_id": gpu_id,
                "event": "eval_start",
                "step": step,
            })
        if args.mode == "checkpoint" and step and step % 30 == 0 and rank == 0:
            target_path = Path(args.checkpoint_dir) / "gridpulse-checkpoint.pt"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target_path)
            events.append({
                "t": rel_t,
                "t_epoch": now,
                "gpu_id": gpu_id,
                "event": "checkpoint_end",
                "step": step,
            })
        if rank == 0 and step and step % 10 == 0:
            events.append({
                "t": rel_t,
                "t_epoch": now,
                "gpu_id": gpu_id,
                "event": "step_end",
                "step": step,
            })
        step += 1

    torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        if args.progress_log:
            path = Path(args.progress_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + ("\n" if events else ""),
                encoding="utf-8",
            )
        print(json.dumps({
            "mode": args.mode,
            "steps": step,
            "events": events,
            "native_progress_available": True,
            "progress_log": args.progress_log,
        }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "distributed", "hpo", "checkpoint", "dataloader_stall", "eval_train_switch"],
        default="baseline",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="/tmp/gridpulse-checkpoints")
    parser.add_argument("--progress-log", default=None, help="JSONL progress log path (original; never deleted)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
