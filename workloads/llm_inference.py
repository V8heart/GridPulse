"""LLM inference workloads (batch / serving / diurnal) with KV-style caching."""
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

from workloads.gpt_model import PRESETS, build_gpt
from workloads.progress_log import ProgressLog


def run(args) -> None:
    import torch

    if args.max_seconds <= 0 or args.max_seconds > 3600:
        raise ValueError("--max-seconds must be in (0, 3600]")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required (or pass --allow-cpu)")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = build_gpt(args.preset, device=device)
    model.eval()
    log = ProgressLog(args.progress_log, int(args.gpu_id))
    log.emit("workload_start", mode=args.mode, preset=args.preset)
    deadline = time.monotonic() + args.max_seconds
    req = 0
    try:
        while time.monotonic() < deadline:
            if args.mode == "diurnal":
                hour_frac = (time.time() % 86400) / 86400.0
                # Night quieter: longer sleeps near midnight.
                quiet = 0.05 + 0.55 * (1.0 - abs(hour_frac - 0.5) * 2.0)
                time.sleep(quiet * random.uniform(0.02, 0.2))
            elif args.mode == "serving":
                time.sleep(1.0 / max(float(args.rps), 0.1))

            n_prompt = random.randint(8, min(args.seq_len, model.cfg.block_size // 2))
            n_gen = random.randint(4, min(32, model.cfg.block_size - n_prompt))
            req_id = f"r{req}"
            prompt_tokens = n_prompt * args.batch_size
            log.emit(
                "request_in",
                request_id=req_id,
                prompt_tokens=prompt_tokens,
                requested_output_tokens=n_gen * args.batch_size,
            )
            prompt = torch.randint(
                0, model.cfg.vocab_size, (args.batch_size, n_prompt), device=device
            )
            # Prefill once, then decode one token at a time from the real per-layer KV cache.
            with torch.no_grad():
                logits, cache = model(prompt, use_cache=True)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                for _ in range(1, n_gen):
                    logits, cache = model(
                        next_id, past_key_values=cache, use_cache=True
                    )
                    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output_tokens = n_gen * args.batch_size
            log.emit(
                "request_out",
                request_id=req_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
            )
            req += 1
    finally:
        log.emit("workload_end", requests=req)
        log.close()
    print(
        json.dumps(
            {
                "workload": "llm_inference",
                "mode": args.mode,
                "preset": args.preset,
                "requests": req,
                "duration_s": args.max_seconds,
                "kv_cache": True,
                "resolved_config": {
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
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
    parser.add_argument("--mode", choices=["batch", "serving", "diurnal"], default="serving")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--duration", type=float, default=None, help="alias for --max-seconds")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.duration is not None:
        args.max_seconds = float(args.duration)
    run(args)


if __name__ == "__main__":
    main()
