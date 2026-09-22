"""Orchestrate observed collection + controlled workloads (gp-telemetry/1.2)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.declared_context import sample_declared
from dataset.finalize_session import finalize_session
from dataset.identifiers import is_session_id, new_run_id, new_session_id, require_session_id

DURATION_POOL = (480.0, 600.0, 720.0)

NORMAL_MODES = {
    "baseline": ("normal_mlp_small", "interactive"),
    "distributed": ("normal_mlp_small", "training"),
    "hpo": ("normal_hpo_search", "training"),
    "checkpoint": ("normal_mlp_small", "training"),
    "dataloader_stall": ("normal_dataloader_bound", "training"),
    "eval_train_switch": ("normal_mlp_small", "training"),
}

ATTACK_WORKLOADS = {
    "swma": ("swma", "swma_basic", False),
    "swma_multi": ("swma", "swma_coordinated", False),
    "ltma": ("ltma", "ltma_basic", True),
    "cryptojacking": ("cryptojacking", "crypto_flat", False),
}

OPTIONAL_LLM_WORKLOADS = {
    "gpt_tiny_finetune": ("normal_llm_finetune", "workloads.gpt_tiny_finetune", "finetune"),
    "llm_pretrain_ddp": ("normal_llm_pretrain_ddp", "workloads.llm_workloads", "pretrain_ddp"),
    "llm_pretrain_fsdp": ("normal_llm_pretrain_fsdp", "workloads.llm_workloads", "pretrain_fsdp"),
    "llm_flat_pretrain": ("normal_llm_flat_pretrain", "workloads.llm_workloads", "flat_pretrain"),
    "llm_finetune": ("normal_llm_finetune", "workloads.llm_workloads", "finetune"),
    "llm_inference_serving": ("normal_llm_inference_serving", "workloads.llm_workloads", "inference_serving"),
    "llm_inference_batch": ("normal_llm_inference_batch", "workloads.llm_workloads", "inference_batch"),
}


def _atomic_write(path: Path, text: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _pick_duration(seed: int, session_id: str) -> float:
    digest = hashlib.sha256(f"duration:{int(seed)}:{session_id}".encode()).digest()
    return float(DURATION_POOL[int.from_bytes(digest[:4], "big") % len(DURATION_POOL)])


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _run_preflight(gpu_ids: list[int], *, sample_s: float = 5.0) -> dict:
    """Return process inventory and an idle-power mean for each target GPU."""
    try:
        from bit2watt_impl.collect_telemetry import NvmlBackend
    except ImportError as exc:
        raise RuntimeError(f"NVML preflight unavailable: {exc}") from exc

    backend = NvmlBackend()
    errors: dict[str, int] = {}
    samples = {str(gpu_id): [] for gpu_id in gpu_ids}
    processes: list[dict] = []
    backend.init()
    try:
        for gpu_id in gpu_ids:
            for proc in backend.read_processes(gpu_id, errors):
                processes.append({"gpu_id": gpu_id, **proc})
        deadline = time.monotonic() + max(0.0, float(sample_s))
        while True:
            for gpu_id in gpu_ids:
                values = backend.read_gpu(
                    gpu_id,
                    errors=errors,
                    power_instant_supported=False,
                    include_processes=False,
                    cached_proc=None,
                )
                power = values.get("power_w")
                if power is not None:
                    samples[str(gpu_id)].append(float(power))
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    finally:
        backend.shutdown()
    return {
        "status": "ok",
        "sample_duration_s": float(sample_s),
        "other_gpu_processes": processes,
        "idle_power_w": {
            gpu_id: (float(np.mean(values)) if values else None)
            for gpu_id, values in samples.items()
        },
        "field_errors": errors,
    }


def build_plan(args) -> dict:
    session_id = args.session_id or new_session_id()
    require_session_id(session_id)
    run_id = args.run_id or new_run_id()
    seed = int(args.seed)
    duration = float(args.duration) if args.duration is not None else _pick_duration(seed, session_id)

    session_dir = ROOT / "dataset" / "real" / "sessions" / session_id
    staging_dir = ROOT / "dataset" / "real" / ".staging" / session_id
    private_stdout = ROOT / "dataset" / "real" / "private" / "stdout" / f"{session_id}.log"
    progress_raw = session_dir / "progress.raw.jsonl"
    telemetry = session_dir / "telemetry.csv"
    stop_file = staging_dir / "collector.stop"
    result_json = staging_dir / "collector_result.json"

    rng = np.random.default_rng(seed)

    if args.workload in NORMAL_MODES:
        gt_label, true_family = NORMAL_MODES[args.workload]
        gt_variant = args.workload
        gt_is_attack = False
        declared_policy = "honest"
        host_workload = None
        hosted = False
        if args.workload == "distributed":
            target_gpus = [0, 1]
            workload = [
                "torchrun",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "workloads.normal_workloads",
                "--mode",
                "distributed",
                "--max-seconds",
                str(duration),
                "--progress-log",
                str(progress_raw),
            ]
        else:
            target_gpus = [args.gpu_id]
            workload = [
                sys.executable,
                "-m",
                "workloads.normal_workloads",
                "--mode",
                args.workload,
                "--gpu-id",
                str(args.gpu_id),
                "--max-seconds",
                str(duration),
                "--progress-log",
                str(progress_raw),
            ]
        module = "workloads.normal_workloads"
    elif args.workload in OPTIONAL_LLM_WORKLOADS:
        gt_label, module, mode = OPTIONAL_LLM_WORKLOADS[args.workload]
        gt_variant = mode
        gt_is_attack = False
        true_family = "training" if "inference" not in mode else "inference"
        declared_policy = "honest"
        host_workload = None
        hosted = False
        target_gpus = [0, 1] if mode in {"pretrain_ddp", "pretrain_fsdp"} else [args.gpu_id]
        available = _module_available(module)
        workload = [
            sys.executable,
            "-m",
            module,
            "--mode",
            mode,
            "--duration",
            str(duration),
            "--progress-log",
            str(progress_raw),
        ]
        if len(target_gpus) == 1:
            workload.extend(["--gpu-id", str(args.gpu_id)])
        if not available and not args.dry_run:
            raise ImportError(
                f"{args.workload} requires optional module {module!r}; "
                "install/add that workload module or use --dry-run to inspect the plan"
            )
    elif args.workload in ATTACK_WORKLOADS:
        gt_label, gt_variant, hosted = ATTACK_WORKLOADS[args.workload]
        gt_is_attack = True
        true_family = "training"
        host_workload = "normal_llm_finetune" if hosted else None
        declared_policy = args.declared_policy if not hosted else "host_inherited"
        target_gpus = [0, 1] if args.workload == "swma_multi" else [args.gpu_id]
        if args.workload.startswith("swma"):
            module = "bit2watt_impl.swma_workload"
            if args.workload == "swma_multi":
                workload = [
                    "torchrun",
                    "--standalone",
                    "--nproc-per-node=2",
                    "-m",
                    module,
                    "--distributed",
                    "--duration",
                    str(duration),
                    "--period",
                    str(args.period),
                    "--duty-cycle",
                    str(args.duty_cycle),
                    "--progress-log",
                    str(progress_raw),
                ]
            else:
                workload = [
                    sys.executable,
                    "-m",
                    module,
                    "--gpu-id",
                    str(args.gpu_id),
                    "--duration",
                    str(duration),
                    "--period",
                    str(args.period),
                    "--duty-cycle",
                    str(args.duty_cycle),
                    "--progress-log",
                    str(progress_raw),
                ]
        elif args.workload == "ltma":
            module = "bit2watt_impl.ltma_inject"
            workload = [
                sys.executable,
                "-m",
                module,
                "--gpu-id",
                str(args.gpu_id),
                "--duration",
                str(duration),
                "--progress-log",
                str(progress_raw),
            ]
        else:
            module = "bit2watt_impl.crypto_workload"
            workload = [
                sys.executable,
                "-m",
                module,
                "--gpu-id",
                str(args.gpu_id),
                "--duration",
                str(duration),
                "--progress-log",
                str(progress_raw),
            ]
    else:
        raise ValueError(f"unknown workload: {args.workload}")

    # GPU roles: all recorded GPUs; non-targets are companion_idle
    record_gpus = list(range(2))  # collector default all on dual-GPU box; dry-run assumes 0..1
    if args.gpu_ids_override:
        record_gpus = [int(x) for x in args.gpu_ids_override.split(",")]
    gpu_roles = {str(g): ("target" if g in target_gpus else "companion_idle") for g in record_gpus}

    declared_by_gpu = {}
    if hosted:
        host_declared = sample_declared(true_family, rng, policy="honest", mismatch_rate=0.0)
        for gid, role in gpu_roles.items():
            if role == "target":
                declared_by_gpu[gid] = sample_declared(
                    true_family, rng, policy="host_inherited", host_declared=host_declared
                )
            else:
                declared_by_gpu[gid] = sample_declared("interactive", rng, policy="honest", mismatch_rate=0.0)
                declared_by_gpu[gid]["declared_job_family"] = "interactive"
    else:
        for gid, role in gpu_roles.items():
            if role == "companion_idle":
                declared_by_gpu[gid] = sample_declared("interactive", rng, policy="honest", mismatch_rate=0.0)
                declared_by_gpu[gid]["declared_job_family"] = "interactive"
            else:
                declared_by_gpu[gid] = sample_declared(
                    true_family,
                    rng,
                    policy=declared_policy if gt_is_attack else "honest",
                )

    collector = [
        sys.executable,
        "-m",
        "bit2watt_impl.collect_telemetry",
        "--output",
        str(telemetry),
        "--session-id",
        session_id,
        "--interval-ms",
        str(args.interval_ms),
        "--gpu-ids",
        "all" if not args.gpu_ids_override else args.gpu_ids_override,
        "--max-duration",
        str(duration + args.pre_capture_s + args.post_capture_s + 300),
        "--stop-file",
        str(stop_file),
        "--result-json",
        str(result_json),
        "--proc-poll-s",
        str(args.proc_poll_s),
    ]

    private = {
        "gt_label": gt_label,
        "gt_is_attack": gt_is_attack,
        "gt_variant": gt_variant,
        "gt_params_json": json.dumps(
            {
                "workload": args.workload,
                "duration_s": duration,
                "period": args.period,
                "duty_cycle": args.duty_cycle,
                "declared_policy": declared_policy,
            }
        ),
        "declared_policy": declared_policy,
        "workload_module": module,
        "workload_cmd": workload,
        "host_workload": host_workload,
        "group_id": args.group_id or f"g-{session_id[2:]}",
        "capture_key": args.capture_key or f"{args.workload}:{duration}:{seed}",
        "run_id": run_id,
        "source": "real",
        "created_epoch": time.time(),
        "progress_mask_seed": int(args.progress_mask_seed),
        "progress_mask_drop_prob": float(args.progress_mask_drop_prob),
        "gpu_roles": gpu_roles,
        "declared_by_gpu": declared_by_gpu,
        "gt_attack_intervals_epoch": {str(g): [] for g in target_gpus},
        "preflight": {},
        "warmup_s": float(args.warmup_s),
        "optional_module_available": _module_available(module),
    }

    return {
        "session_id": session_id,
        "run_id": run_id,
        "duration_s": duration,
        "session_dir": str(session_dir),
        "staging_dir": str(staging_dir),
        "private_stdout": str(private_stdout),
        "progress_raw": str(progress_raw),
        "telemetry": str(telemetry),
        "stop_file": str(stop_file),
        "collector": collector,
        "workload": workload,
        "private": private,
        "gpu_roles": gpu_roles,
        "declared_by_gpu": declared_by_gpu,
        "target_gpus": target_gpus,
        "probe": (
            shlex.split(args.probe_cmd)
            if args.probe_cmd and args.workload in OPTIONAL_LLM_WORKLOADS
            else None
        ),
        "dry_run": bool(args.dry_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=[*NORMAL_MODES, *OPTIONAL_LLM_WORKLOADS, *ATTACK_WORKLOADS],
        required=True,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids-override", default=None, help="optional collector gpu id list")
    parser.add_argument("--duration", type=float, default=None, help="override; default from shared pool")
    parser.add_argument("--interval-ms", type=float, default=100.0)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--capture-key", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pre-capture-s", type=float, default=10.0)
    parser.add_argument("--post-capture-s", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=180.0)
    parser.add_argument("--preflight-s", type=float, default=5.0)
    parser.add_argument(
        "--probe-cmd",
        default=None,
        help="optional pre-collector probe command for an LLM workload",
    )
    parser.add_argument("--proc-poll-s", type=float, default=1.0)
    parser.add_argument("--progress-mask-seed", type=int, default=7)
    parser.add_argument("--progress-mask-drop-prob", type=float, default=0.4)
    parser.add_argument(
        "--declared-policy",
        choices=["pool_random", "host_family_matched"],
        default="pool_random",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--confirm-shared-gpu-safe",
        action="store_true",
        help="다른 사용자 작업이 없음을 확인했을 때만 지정",
    )
    parser.add_argument("--skip-finalize", action="store_true")
    args = parser.parse_args()

    if args.session_id and not is_session_id(args.session_id):
        parser.error("session_id must match ^s-[0-9a-f]{16}$")
    if args.duration is not None and not (0 < args.duration <= 3600):
        parser.error("--duration must be in (0, 3600]")
    if not args.confirm_shared_gpu_safe and not args.dry_run:
        parser.error("실측 부하 실행 전 --confirm-shared-gpu-safe가 필요합니다.")

    plan = build_plan(args)
    if args.dry_run:
        plan["private"]["preflight"] = {
            "status": "skipped_dry_run",
            "sample_duration_s": float(args.preflight_s),
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    preflight = _run_preflight(
        sorted(int(gpu_id) for gpu_id in plan["gpu_roles"]),
        sample_s=args.preflight_s,
    )
    plan["private"]["preflight"] = preflight
    if preflight["other_gpu_processes"] and not args.allow_busy:
        raise RuntimeError(
            "target GPU is busy; pass --allow-busy only after verifying the listed processes: "
            + json.dumps(preflight["other_gpu_processes"], ensure_ascii=False)
        )

    session_dir = Path(plan["session_dir"])
    staging_dir = Path(plan["staging_dir"])
    session_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["private_stdout"]).parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(staging_dir / "capture_private.json", json.dumps(plan["private"], ensure_ascii=False, indent=2), mode=0o600)

    if plan["probe"]:
        probe = subprocess.run(plan["probe"], cwd=ROOT, check=False)
        plan["private"]["preflight"]["probe"] = {
            "command": plan["probe"],
            "returncode": probe.returncode,
        }
        _atomic_write(
            staging_dir / "capture_private.json",
            json.dumps(plan["private"], ensure_ascii=False, indent=2),
            mode=0o600,
        )
        if probe.returncode:
            raise RuntimeError(f"LLM probe failed exit={probe.returncode}")

    collector = subprocess.Popen(plan["collector"], cwd=ROOT)
    try:
        time.sleep(args.pre_capture_s)
        with open(plan["private_stdout"], "w", encoding="utf-8") as stdout_fh:
            os.chmod(plan["private_stdout"], 0o600)
            workload = subprocess.run(
                plan["workload"],
                cwd=ROOT,
                check=False,
                stdout=stdout_fh,
                stderr=subprocess.STDOUT,
            )
        if workload.returncode:
            raise RuntimeError(f"workload failed exit={workload.returncode}")
        # Merge attack intervals / final params from last JSON line of stdout.
        stdout_text = Path(plan["private_stdout"]).read_text(encoding="utf-8")
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "gt_attack_intervals_epoch" in summary:
                plan["private"]["gt_attack_intervals_epoch"] = summary["gt_attack_intervals_epoch"]
            params = json.loads(plan["private"]["gt_params_json"])
            params["workload_summary"] = {
                k: summary[k]
                for k in summary
                if k not in {"gt_attack_intervals_epoch", "events"}
            }
            plan["private"]["gt_params_json"] = json.dumps(params)
            _atomic_write(
                Path(plan["staging_dir"]) / "capture_private.json",
                json.dumps(plan["private"], ensure_ascii=False, indent=2),
                mode=0o600,
            )
            break
        time.sleep(args.post_capture_s)
        Path(plan["stop_file"]).write_text("stop\n", encoding="utf-8")
        return_code = collector.wait(timeout=args.duration or plan["duration_s"] + 60)
        if return_code:
            raise RuntimeError(f"collector failed exit={return_code}")
    finally:
        if collector.poll() is None:
            collector.terminate()
            try:
                collector.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector.kill()

    if not args.skip_finalize:
        report = finalize_session(plan["session_id"], warmup_s=args.warmup_s)
        plan["finalize"] = report
    print(json.dumps({k: plan[k] for k in ("session_id", "run_id", "duration_s", "finalize") if k in plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
