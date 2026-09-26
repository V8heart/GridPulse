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
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workloads.gpt_model import PRESETS, build_gpt, count_parameters, resolve_preset
from workloads.progress_log import ProgressLog

PROBE_ROOT = ROOT / "dataset" / "real" / "private" / "probes"
NCCL_TIMEOUT = timedelta(minutes=10)
BARRIER_TIMEOUT = timedelta(seconds=30)
PROBE_CANDIDATES = (8, 16, 32, 64, 128)
DEFAULT_PROBE_TARGET_W = 300.0


def _dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def distributed_should_stop(stop_value: float, *, rank: int, deadline: float, now: float) -> float:
    """Return the local stop contribution. Rank 0 raises 1.0 after ``deadline``."""
    if rank == 0 and now >= deadline:
        return 1.0
    return float(stop_value)


def _timed_barrier(dist, timeout: timedelta = BARRIER_TIMEOUT) -> None:
    try:
        if hasattr(dist, "monitored_barrier"):
            dist.monitored_barrier(timeout=timeout)
            return
    except ValueError:
        # NCCL process groups are not CPU-capable; fall back to a plain barrier.
        pass
    dist.barrier()


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


def make_host_training_step(
    device,
    *,
    preset: str = "small",
    batch_size: int = 64,
    seq_len: int = 512,
    seed: int = 7,
):
    """Build a download-free GPT step shared by hosted attacks and LTMA."""
    import torch

    torch.manual_seed(seed)
    model = build_gpt(preset, device=device)
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


def make_tiny_training_step(
    device,
    *,
    batch_size: int = 2,
    seq_len: int = 64,
    seed: int = 7,
):
    """Backward-compatible wrapper. Capture hosts should use make_host_training_step."""
    return make_host_training_step(
        device, preset="tiny", batch_size=batch_size, seq_len=seq_len, seed=seed
    )


def _probe_key(args, gpu_name: str, world: int) -> str:
    preset = resolve_preset(
        args.preset,
        vocab_size=getattr(args, "vocab_size", None),
        block_size=getattr(args, "block_size", None),
    )
    payload = "|".join(
        [
            gpu_name,
            args.preset,
            str(args.seq_len),
            args.dtype,
            str(bool(args.compile)),
            str(world),
            args.mode,
            str(getattr(args, "probe_target_w", DEFAULT_PROBE_TARGET_W)),
            str(max(PROBE_CANDIDATES)),
            str(preset.block_size),
            str(preset.vocab_size),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def probe_candidate_batches(batch_size: int, candidates: tuple[int, ...] = PROBE_CANDIDATES) -> list[int]:
    """Ascending probe sizes, capped by the candidate ceiling."""
    ceiling = max(int(batch_size), max(candidates))
    return [int(c) for c in candidates if int(c) <= ceiling]


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


def probe_microbatch(args, device, model, world: int) -> dict:
    import torch

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    key = _probe_key(args, gpu_name, world)
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = PROBE_ROOT / f"{key}.json"
    if cache.exists() and not args.force_probe:
        return json.loads(cache.read_text(encoding="utf-8"))

    dtype = _dtype(args.dtype)
    target_w = float(getattr(args, "probe_target_w", DEFAULT_PROBE_TARGET_W))
    seq_len = min(int(args.seq_len), int(model.cfg.block_size))
    gpu_id = int(os.environ.get("LOCAL_RANK", args.gpu_id))
    samples: list[dict] = []
    hit_target = None
    last_ok = None
    for batch in probe_candidate_batches(int(args.batch_size)):
        ok = _try_batch(model, batch, seq_len, model.cfg.vocab_size, device, dtype, args.probe_steps)
        if not ok:
            samples.append({"micro_batch": batch, "oom": True})
            continue
        watts: list[float] = []
        for _ in range(4):
            _try_batch(model, batch, seq_len, model.cfg.vocab_size, device, dtype, 1)
            reading = _read_gpu_power_w(gpu_id)
            if reading is not None:
                watts.append(reading)
        mean_w = sum(watts) / len(watts) if watts else None
        used_mb = None
        if device.type == "cuda":
            used_mb = float(torch.cuda.memory_allocated(device)) / (1024 * 1024)
        samples.append({"micro_batch": batch, "mean_w": mean_w, "fb_used_mb": used_mb})
        last_ok = batch
        if mean_w is not None and mean_w >= target_w:
            hit_target = batch
            break
    if hit_target is not None:
        chosen = hit_target
    elif last_ok is not None:
        chosen = last_ok
    else:
        raise RuntimeError("probe failed: no safe micro-batch (runtime_oom)")
    winner = next((s for s in samples if s.get("micro_batch") == chosen and not s.get("oom")), {})
    payload = {
        "micro_batch": chosen,
        "mean_w": winner.get("mean_w"),
        "target_w": target_w,
        "fb_used_mb": winner.get("fb_used_mb"),
        "samples": samples,
        "gpu_name": gpu_name,
        "preset": args.preset,
        "seq_len": seq_len,
        "dtype": args.dtype,
        "compile": bool(args.compile),
        "world": world,
        "nccl_p2p_disabled": os.environ.get("NCCL_P2P_DISABLE"),
    }
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, cache)
    return payload


def run(args) -> None:
    import torch
    import torch.distributed as dist

    if args.max_seconds <= 0 or args.max_seconds > 3600:
        raise ValueError("--max-seconds must be in (0, 3600]")

    distributed = args.mode in {"ddp", "fsdp"}
    rank = 0
    world = 1
    if distributed:
        dist.init_process_group("nccl", timeout=NCCL_TIMEOUT)
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
    model = build_gpt(
        args.preset,
        device=device,
        vocab_size=getattr(args, "vocab_size", None),
        block_size=getattr(args, "block_size", None),
    )
    if args.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    probe_info: dict = {}
    if args.probe or args.probe_only:
        if distributed:
            flag = torch.zeros(2, device=device, dtype=torch.float64)
            if rank == 0:
                try:
                    probe_info = probe_microbatch(args, device, model, world)
                    flag[0] = 1.0
                    flag[1] = float(probe_info["micro_batch"])
                except Exception:
                    dist.broadcast(flag, src=0)
                    raise
            dist.broadcast(flag, src=0)
            if float(flag[0]) < 1.0:
                raise RuntimeError("distributed probe failed on rank 0")
            batch = int(flag[1].item())
            if rank != 0:
                probe_info = {"micro_batch": batch}
        else:
            probe_info = probe_microbatch(args, device, model, world)
            batch = int(probe_info["micro_batch"])
        if args.probe_only:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "probe_only": True,
                            "micro_batch": batch,
                            "mean_w": probe_info.get("mean_w"),
                            "target_w": probe_info.get("target_w"),
                            "fb_used_mb": probe_info.get("fb_used_mb"),
                            "params": count_parameters(model),
                        }
                    )
                )
            if distributed:
                _timed_barrier(dist)
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
    seq_len = min(int(args.seq_len), int(raw_model.cfg.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dtype = _dtype(args.dtype)
    gpu_id = int(rank if distributed else args.gpu_id)
    log = ProgressLog(args.progress_log, gpu_id)
    log.emit("workload_start", mode=args.mode, preset=args.preset, micro_batch=batch)
    deadline = time.monotonic() + args.max_seconds
    stop = torch.zeros(1, device=device)
    step = 0
    oom = False
    try:
        while True:
            local = distributed_should_stop(
                float(stop.item()), rank=rank, deadline=deadline, now=time.monotonic()
            )
            stop.fill_(local)
            if distributed:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if float(stop.item()):
                break
            if args.mode == "hpo" and step and step % 20 == 0:
                optimizer.param_groups[0]["lr"] = random.choice([1e-4, 3e-4, 1e-3])
                log.emit("phase_change", step=step, lr=optimizer.param_groups[0]["lr"])
            if args.mode == "dataloader_bound" and step % 11 == 0:
                time.sleep(random.uniform(0.05, 0.6))
            x = torch.randint(0, raw_model.cfg.vocab_size, (batch, seq_len), device=device)
            y = torch.randint(0, raw_model.cfg.vocab_size, (batch, seq_len), device=device)
            accum = max(1, int(args.grad_accum))
            if step % accum == 0:
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
                (loss / accum).backward()
                if (step + 1) % accum == 0:
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
            try:
                _timed_barrier(dist)
            except Exception:
                pass
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
                    "mean_w": probe_info.get("mean_w"),
                    "target_w": probe_info.get("target_w"),
                    "fb_used_mb": probe_info.get("fb_used_mb"),
                    "seq_len": seq_len,
                    "dtype": args.dtype,
                    "steps": step,
                    "duration_s": args.max_seconds,
                    "resolved_config": {
                        "preset": args.preset,
                        "params": count_parameters(raw_model),
                        "micro_batch": batch,
                        "seq_len": seq_len,
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
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-steps", type=int, choices=range(10, 21), default=12)
    parser.add_argument("--probe-target-w", type=float, default=DEFAULT_PROBE_TARGET_W)
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
