"""Minimal ResNet-style vision workloads with synthetic images (no downloads)."""
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

from workloads.progress_log import ProgressLog


def build_resnet18(num_classes: int = 10):
    """Lightweight local ResNet-18-ish stack (no torchvision dependency required)."""
    import torch
    from torch import nn

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.short = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.short = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.short(x)
            return torch.relu(out)

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
            self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
            self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    return ResNet()


def run(args) -> None:
    import torch
    import torch.distributed as dist

    if args.max_seconds <= 0 or args.max_seconds > 3600:
        raise ValueError("--max-seconds must be in (0, 3600]")

    distributed = args.mode == "ddp"
    rank = 0
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{rank}")
    else:
        if not torch.cuda.is_available() and not args.allow_cpu:
            raise RuntimeError("CUDA GPU required (or pass --allow-cpu)")
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    model = build_resnet18().to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if device.type == "cuda" else None)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    gpu_id = int(rank if distributed else args.gpu_id)
    log = ProgressLog(args.progress_log, gpu_id)
    log.emit("workload_start", mode=args.mode)
    deadline = time.monotonic() + args.max_seconds
    step = 0
    while time.monotonic() < deadline:
        if args.dataloader == "cpu":
            # Intentionally produce on CPU then copy — stalls like a CPU dataloader.
            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
            labels = torch.randint(0, 10, (args.batch_size,))
            time.sleep(random.uniform(0.01, 0.05))
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
            labels = torch.randint(0, 10, (args.batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        log.emit("step_end", step=step)
        step += 1
    log.emit("workload_end", steps=step)
    log.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print(
            json.dumps(
                {
                    "workload": "vision_resnet",
                    "mode": args.mode,
                    "dataloader": args.dataloader,
                    "steps": step,
                    "image_size": args.image_size,
                    "native_progress_available": True,
                    "progress_log": args.progress_log,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ddp"], default="single")
    parser.add_argument("--dataloader", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
