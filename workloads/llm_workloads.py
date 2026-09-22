"""LLM training workloads (pretrain/finetune/DDP/FSDP/HPO/dataloader-bound).

No network downloads. Probe selects a safe micro-batch before the main run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workloads.gpt_model import PRESETS, build_gpt, count_parameters
from workloads.progress_log import ProgressLog

PROBE_ROOT = ROOT / "dataset" / "real" / "private" / "probes"


def _dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def make_tiny_training_step(
    device,
    *,
    batch_size: int = 2,
    seq_len: int = 64,
    seed: int = 7,
):
    """Build the tiny, download-free GPT step shared by hosted workloads."""
    import torch

    torch.manual_seed(seed)
    model = build_gpt("tiny", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seq_len = min(int(seq_len), model.cfg.block_size)

    def step() -> float:
        tokens = torch.randint(
            0, model.cfg.vocab_size, (int(batch_size), seq_len), device=device
        )
        targets = torch.randint(
            0, model.cfg.vocab_size, (int(batch_size), seq_len), device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    return step


def _probe_key(args, gpu_name: str, world: int) -> str:
    payload = "|".join(
        [
            gpu_name,
            args.preset,
            str(args.seq_len),
            args.dtype,
            str(bool(args.compile)),
            str(world),
            args.mode,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _try_batch(model, batch: int, seq_len: int, vocab: int, device, dtype, steps: int) -> bool:
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        for _ in range(steps):
            x = torch.randint(0, vocab, (batch, seq_len), device=device)
            y = torch.randint(0, vocab, (batch, seq_len), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        return True
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False
        raise


def probe_microbatch(args, device, model, world: int) -> int:
    import torch

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    key = _probe_key(args, gpu_name, world)
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = PROBE_ROOT / f"{key}.json"
    if cache.exists() and not args.force_probe:
        return int(json.loads(cache.read_text(encoding="utf-8"))["micro_batch"])

    dtype = _dtype(args.dtype)
    candidates = [args.batch_size, max(1, args.batch_size // 2), max(1, args.batch_size // 4), 1]
    seen = []
    chosen = None
    for batch in candidates:
        if batch in seen:
            continue
        seen.append(batch)
        ok = _try_batch(model, batch, args.seq_len, model.cfg.vocab_size, device, dtype, args.probe_steps)
        if ok:
            chosen = batch
            break
    if chosen is None:
        raise RuntimeError("probe failed: no safe micro-batch (runtime_oom)")
    payload = {
        "micro_batch": chosen,
        "gpu_name": gpu_name,
        "preset": args.preset,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "compile": bool(args.compile),
        "world": world,
        "nccl_p2p_disabled": os.environ.get("NCCL_P2P_DISABLE"),
    }
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, cache)
    return chosen


def run(args) -> None:
    import torch
    import torch.distributed as dist

    if args.max_seconds <= 0 or args.max_seconds > 3600:
        raise ValueError("--max-seconds must be in (0, 3600]")

    distributed = args.mode in {"ddp", "fsdp"}
    rank = 0
    world = 1
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ.get("WORLD_SIZE", "1"))
        device = torch.device(f"cuda:{rank}")
    else:
        if not torch.cuda.is_available() and args.allow_cpu:
            device = torch.device("cpu")
        elif not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required (or pass --allow-cpu for smoke)")
        else:
            device = torch.device(f"cuda:{args.gpu_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    model = build_gpt(args.preset, device=device)
    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    if args.probe or args.probe_only:
        batch = probe_microbatch(args, device, model, world)
        if args.probe_only:
            if rank == 0:
                print(json.dumps({"probe_only": True, "micro_batch": batch, "params": count_parameters(model)}))
            if distributed:
                dist.barrier()
                dist.destroy_process_group()
            return
    else:
        batch = int(args.batch_size)

    if args.mode == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank] if device.type == "cuda" else None)
    elif args.mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(model)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dtype = _dtype(args.dtype)
    gpu_id = int(rank if distributed else args.gpu_id)
    log = ProgressLog(args.progress_log, gpu_id)
    log.emit("workload_start", mode=args.mode, preset=args.preset, micro_batch=batch)
    deadline = time.monotonic() + args.max_seconds
    step = 0
    oom = False
    try:
        while time.monotonic() < deadline:
            if args.mode == "hpo" and step and step % 20 == 0:
                optimizer.param_groups[0]["lr"] = random.choice([1e-4, 3e-4, 1e-3])
                log.emit("phase_change", step=step, lr=optimizer.param_groups[0]["lr"])
            if args.mode == "dataloader_bound" and step % 11 == 0:
                time.sleep(random.uniform(0.05, 0.6))
            x = torch.randint(0, raw_model.cfg.vocab_size, (batch, args.seq_len), device=device)
            y = torch.randint(0, raw_model.cfg.vocab_size, (batch, args.seq_len), device=device)
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype != torch.float32,
                ):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    )
                loss.backward()
                optimizer.step()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    oom = True
                    raise RuntimeError("runtime_oom") from exc
                raise
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            log.emit("step_end", step=step)
            step += 1
    finally:
        log.emit("workload_end", steps=step, oom=oom)
        log.close()
        if distributed:
            dist.barrier()
            dist.destroy_process_group()

    if rank == 0:
        print(
            json.dumps(
                {
                    "workload": "llm",
                    "mode": args.mode,
                    "preset": args.preset,
                    "params": count_parameters(raw_model),
                    "micro_batch": batch,
                    "seq_len": args.seq_len,
                    "dtype": args.dtype,
                    "steps": step,
                    "duration_s": args.max_seconds,
                    "resolved_config": {
                        "preset": args.preset,
                        "params": count_parameters(raw_model),
                        "micro_batch": batch,
                        "seq_len": args.seq_len,
                        "dtype": args.dtype,
                        "compile": bool(args.compile),
                        "world": world,
                        "device": str(device),
                    },
                    "native_progress_available": True,
                    "progress_log": args.progress_log,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "pretrain",
            "finetune",
            "ddp",
            "fsdp",
            "hpo",
            "dataloader_bound",
            # orchestrator aliases
            "pretrain_ddp",
            "pretrain_fsdp",
            "flat_pretrain",
            "inference_serving",
            "inference_batch",
        ],
        default="finetune",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--duration", type=float, default=None, help="alias for --max-seconds")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-steps", type=int, choices=range(10, 21), default=12)
    parser.add_argument("--force-probe", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.duration is not None:
        args.max_seconds = float(args.duration)
    # Normalize orchestrator aliases.
    alias = {
        "pretrain_ddp": "ddp",
        "pretrain_fsdp": "fsdp",
        "flat_pretrain": "pretrain",
    }
    if args.mode in alias:
        args.mode = alias[args.mode]
    if args.mode in {"inference_serving", "inference_batch"}:
        # Delegate inference modes to the dedicated module.
        from workloads.llm_inference import main as infer_main
        import sys as _sys

        infer_argv = [
            "workloads.llm_inference",
            "--mode",
            "serving" if args.mode == "inference_serving" else "batch",
            "--preset",
            args.preset,
            "--gpu-id",
            str(args.gpu_id),
            "--max-seconds",
            str(args.max_seconds),
        ]
        if args.progress_log:
            infer_argv.extend(["--progress-log", args.progress_log])
        if args.allow_cpu:
            infer_argv.append("--allow-cpu")
        _sys.argv = infer_argv
        infer_main()
        return
    run(args)


if __name__ == "__main__":
    main()
