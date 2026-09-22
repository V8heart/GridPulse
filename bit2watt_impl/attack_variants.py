"""Attack variant launchers (SWMA/LTMA/crypto hosted & coordinated) for gp-telemetry/1.2."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWMA_VARIANTS = {
    "basic": {"period": 1.0, "duty_cycle": 0.5, "matrix_size": 2048, "active_streams": 1, "jitter_frac": 0.0},
    "shallow": {"period": 1.0, "duty_cycle": 0.5, "matrix_size": 512, "active_streams": 1, "jitter_frac": 0.0},
    "jitter": {"period": 1.0, "duty_cycle": 0.5, "matrix_size": 2048, "active_streams": 1, "jitter_frac": 0.15},
}


def _read_step_times(progress_path: Path, *, gpu_id: int, limit: int = 30) -> list[float]:
    if not progress_path.exists():
        return []
    times: list[float] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") != "step_end":
            continue
        if int(obj.get("gpu_id", gpu_id)) != int(gpu_id):
            continue
        times.append(float(obj["t_epoch"]))
        if len(times) >= limit:
            break
    return times


def _median_period(times: list[float]) -> float | None:
    if len(times) < 2:
        return None
    diffs = sorted(times[i + 1] - times[i] for i in range(len(times) - 1))
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid]
    return 0.5 * (diffs[mid - 1] + diffs[mid])


def _swma_argv(args, *, gpu_id: int, progress_log: str | None, duration: float, params: dict) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "bit2watt_impl.swma_workload",
        "--gpu-id",
        str(gpu_id),
        "--duration",
        str(duration),
        "--period",
        str(params["period"]),
        "--duty-cycle",
        str(params["duty_cycle"]),
        "--matrix-size",
        str(params["matrix_size"]),
        "--active-streams",
        str(params["active_streams"]),
        "--jitter-frac",
        str(params["jitter_frac"]),
        "--seed",
        str(args.seed),
    ]
    if progress_log:
        cmd.extend(["--progress-log", progress_log])
    return cmd


def _host_argv(args, *, progress_log: str, duration: float) -> list[str]:
    if args.host == "llm":
        return [
            sys.executable,
            "-m",
            "workloads.llm_workloads",
            "--mode",
            "finetune",
            "--preset",
            "tiny",
            "--gpu-id",
            str(args.gpu_id),
            "--max-seconds",
            str(duration),
            "--progress-log",
            progress_log,
            "--batch-size",
            "2",
        ]
    return [
        sys.executable,
        "-m",
        "workloads.normal_workloads",
        "--mode",
        "baseline",
        "--gpu-id",
        str(args.gpu_id),
        "--max-seconds",
        str(duration),
        "--progress-log",
        progress_log,
    ]


def _parse_summary(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def build_variant_plan(args) -> dict:
    """Return argv plan without launching (for tests / dry-run)."""
    variant = args.variant
    params = dict(SWMA_VARIANTS.get(variant.replace("swma_", ""), SWMA_VARIANTS["basic"]))
    if args.period is not None:
        params["period"] = args.period
    if args.duty_cycle is not None:
        params["duty_cycle"] = args.duty_cycle
    hosted = variant in {"piggyback", "mimicry"} or args.variant.startswith("ltma")
    coordinated = variant == "coordinated"
    return {
        "variant": variant,
        "params": params,
        "hosted": hosted,
        "coordinated": coordinated,
        "host_progress_log": bool(hosted or not hosted),  # host always logs when present
        "attack_progress_log": not hosted,  # hosted attack must NOT get progress-log
        "passive_host_steps": 30 if hosted and variant in {"piggyback", "mimicry"} else 0,
    }


def run(args) -> None:
    if not 0 < args.duration <= 1800:
        raise ValueError("--duration must be in (0, 1800]")
    plan = build_variant_plan(args)
    progress = Path(args.progress_log) if args.progress_log else None
    if progress:
        progress.parent.mkdir(parents=True, exist_ok=True)

    children: list[subprocess.Popen] = []
    stdout_chunks: list[str] = []
    intervals: dict[str, list[list[float]]] = {}
    try:
        if args.variant in {"basic", "shallow", "jitter", "swma_basic", "swma_shallow", "swma_jitter"}:
            key = args.variant.replace("swma_", "")
            params = dict(SWMA_VARIANTS[key if key in SWMA_VARIANTS else "basic"])
            if args.period is not None:
                params["period"] = args.period
            argv = _swma_argv(
                args,
                gpu_id=args.gpu_id,
                progress_log=str(progress) if progress else None,
                duration=args.duration,
                params=params,
            )
            plan["argv_attack"] = argv
            proc = subprocess.run(argv, cwd=ROOT, check=False, capture_output=True, text=True)
            stdout_chunks.append(proc.stdout or "")
            if proc.returncode:
                raise RuntimeError(f"swma failed exit={proc.returncode}: {proc.stderr}")
            summary = _parse_summary(proc.stdout or "")
            intervals = summary.get("gt_attack_intervals_epoch") or {str(args.gpu_id): []}
        elif args.variant == "coordinated":
            params = dict(SWMA_VARIANTS["basic"])
            coordinated: list[tuple[int, subprocess.Popen]] = []
            plan["argv_attack"] = []
            for gpu_id in (args.gpu_id, args.gpu_id + 1):
                argv = _swma_argv(
                    args,
                    gpu_id=gpu_id,
                    progress_log=str(progress) if progress else None,
                    duration=args.duration,
                    params=params,
                )
                plan["argv_attack"].append(argv)
                child = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                children.append(child)
                coordinated.append((gpu_id, child))
            intervals = {}
            for gpu_id, child in coordinated:
                output, _ = child.communicate()
                stdout_chunks.append(output or "")
                if child.returncode:
                    raise RuntimeError(
                        f"coordinated GPU {gpu_id} failed exit={child.returncode}"
                    )
                summary = _parse_summary(output or "")
                intervals[str(gpu_id)] = (
                    summary.get("gt_attack_intervals_epoch", {}).get(str(gpu_id), [])
                )
        elif args.variant in {"piggyback", "mimicry"}:
            if progress is None:
                raise ValueError("hosted variants require --progress-log for the host")
            host_cmd = _host_argv(
                args, progress_log=str(progress), duration=args.duration + 120.0
            )
            plan["argv_host"] = host_cmd
            # Host gets progress-log; attack does not.
            host = subprocess.Popen(host_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            children.append(host)
            # Wait until 30 host step_end events exist.
            deadline = time.monotonic() + min(120.0, args.duration)
            times: list[float] = []
            while time.monotonic() < deadline:
                times = _read_step_times(progress, gpu_id=args.gpu_id, limit=30)
                if len(times) >= 30:
                    break
                if host.poll() is not None:
                    break
                time.sleep(0.2)
            if len(times) < 30:
                raise RuntimeError(
                    f"host did not complete 30 warmup steps (observed={len(times)})"
                )
            period = _median_period(times) or 1.0
            # Passive first 30 host steps are NOT attack intervals.
            params = dict(SWMA_VARIANTS["basic"])
            if args.variant == "mimicry":
                params["period"] = float(period)
            attack_cmd = _swma_argv(
                args,
                gpu_id=args.gpu_id,
                progress_log=None,  # hosted attack must not write progress
                duration=args.duration,
                params=params,
            )
            plan["argv_attack"] = attack_cmd
            attack = subprocess.Popen(attack_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            children.append(attack)
            attack_out, _ = attack.communicate()
            stdout_chunks.append(attack_out or "")
            if attack.returncode:
                raise RuntimeError(f"hosted attack failed exit={attack.returncode}")
            if host.poll() not in (None, 0):
                raise RuntimeError(f"host failed exit={host.returncode}")
            _terminate(host)
            summary = _parse_summary(attack_out or "")
            intervals = summary.get("gt_attack_intervals_epoch") or {str(args.gpu_id): []}
            # Ensure passive prefix is not present (attack process started after 30 steps).
            if times:
                prefix_end = times[min(29, len(times) - 1)]
                cleaned = []
                for start, end in intervals.get(str(args.gpu_id), []):
                    if end <= prefix_end:
                        continue
                    cleaned.append([max(float(start), prefix_end), float(end)])
                intervals[str(args.gpu_id)] = cleaned
        elif args.variant in {"ltma", "ltma_basic"}:
            cmd = [
                sys.executable,
                "-m",
                "bit2watt_impl.ltma_inject",
                "--gpu-id",
                str(args.gpu_id),
                "--duration",
                str(args.duration),
                "--host",
                args.host,
            ]
            if progress:
                cmd.extend(["--progress-log", str(progress)])
            plan["argv_attack"] = cmd
            proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
            stdout_chunks.append(proc.stdout or "")
            if proc.returncode:
                raise RuntimeError(f"ltma failed exit={proc.returncode}")
            summary = _parse_summary(proc.stdout or "")
            intervals = summary.get("gt_attack_intervals_epoch") or {str(args.gpu_id): []}
        elif args.variant in {"crypto", "cryptojacking"}:
            cmd = [
                sys.executable,
                "-m",
                "bit2watt_impl.crypto_workload",
                "--gpu-id",
                str(args.gpu_id),
                "--duration",
                str(args.duration),
            ]
            if progress:
                cmd.extend(["--progress-log", str(progress)])
            plan["argv_attack"] = cmd
            proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
            stdout_chunks.append(proc.stdout or "")
            if proc.returncode:
                raise RuntimeError(f"crypto failed exit={proc.returncode}")
            summary = _parse_summary(proc.stdout or "")
            intervals = summary.get("gt_attack_intervals_epoch") or {str(args.gpu_id): []}
        else:
            raise ValueError(f"unknown variant: {args.variant}")
    except Exception as exc:
        for child in children:
            _terminate(child)
        print(json.dumps({"ok": False, "error": str(exc), "plan": plan}, ensure_ascii=False))
        raise
    finally:
        for child in children:
            _terminate(child)

    print(
        json.dumps(
            {
                "ok": True,
                "variant": args.variant,
                "hosted": plan["hosted"],
                "attack_progress_log": plan["attack_progress_log"],
                "gt_attack_intervals_epoch": intervals,
                "plan": {k: v for k, v in plan.items() if k.startswith("argv") or k in {"params", "passive_host_steps"}},
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "basic",
            "shallow",
            "jitter",
            "swma_basic",
            "swma_shallow",
            "swma_jitter",
            "piggyback",
            "mimicry",
            "coordinated",
            "ltma",
            "ltma_basic",
            "crypto",
            "cryptojacking",
        ],
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--duty-cycle", type=float, default=None)
    parser.add_argument("--host", choices=["mlp", "llm"], default="llm")
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-plan", action="store_true", help="print argv plan only")
    args = parser.parse_args()
    if args.dry_plan:
        plan = build_variant_plan(args)
        # Materialize argv snapshots without launching.
        progress = args.progress_log
        if args.variant in {"piggyback", "mimicry"}:
            plan["argv_host"] = _host_argv(args, progress_log=progress or "/tmp/host.jsonl", duration=args.duration)
            plan["argv_attack"] = _swma_argv(
                args,
                gpu_id=args.gpu_id,
                progress_log=None,
                duration=args.duration,
                params=dict(SWMA_VARIANTS["basic"]),
            )
        elif args.variant == "coordinated":
            plan["argv_attack"] = ["torchrun", "--standalone", "--nproc-per-node=2", "-m", "bit2watt_impl.swma_workload"]
        else:
            plan["argv_attack"] = _swma_argv(
                args,
                gpu_id=args.gpu_id,
                progress_log=progress if plan["attack_progress_log"] else None,
                duration=args.duration,
                params=dict(plan["params"]),
            )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
