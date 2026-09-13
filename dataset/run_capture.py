"""수집기와 통제된 워크로드를 함께 실행해 라벨 세션을 캡처한다."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NORMAL_MODES = {
    "baseline": "normal_baseline",
    "distributed": "normal_distributed_training",
    "hpo": "normal_hpo_search",
    "checkpoint": "normal_checkpoint",
    "dataloader_stall": "normal_dataloader_stall",
    "eval_train_switch": "normal_eval_train_switch",
}


def commands(args, session_id: str) -> tuple[list[str], list[str], str, list[int], Path, str]:
    if args.workload in NORMAL_MODES:
        label = NORMAL_MODES[args.workload]
        if args.workload == "distributed":
            gpu_ids = [0, 1]
            workload = [
                "torchrun", "--standalone", "--nproc-per-node=2", "-m",
                "workloads.normal_workloads", "--mode", "distributed",
                "--max-seconds", str(args.duration),
            ]
        else:
            gpu_ids = [args.gpu_id]
            workload = [
                sys.executable, "-m", "workloads.normal_workloads",
                "--mode", args.workload, "--gpu-id", str(args.gpu_id),
                "--max-seconds", str(args.duration),
            ]
    elif args.workload in ("swma", "swma_multi"):
        label = "swma"
        if args.workload == "swma_multi":
            gpu_ids = [0, 1]
            workload = [
                "torchrun", "--standalone", "--nproc-per-node=2", "-m",
                "bit2watt_impl.swma_workload", "--distributed",
                "--duration", str(args.duration), "--period", str(args.period),
                "--duty-cycle", str(args.duty_cycle),
            ]
        else:
            gpu_ids = [args.gpu_id]
            workload = [
                sys.executable, "-m", "bit2watt_impl.swma_workload",
                "--gpu-id", str(args.gpu_id), "--duration", str(args.duration),
                "--period", str(args.period), "--duty-cycle", str(args.duty_cycle),
            ]
    elif args.workload == "ltma":
        label, gpu_ids = "ltma", [args.gpu_id]
        workload = [
            sys.executable, "-m", "bit2watt_impl.ltma_inject",
            "--gpu-id", str(args.gpu_id), "--duration", str(args.duration),
        ]
    elif args.workload == "cryptojacking":
        label, gpu_ids = "cryptojacking", [args.gpu_id]
        workload = [
            sys.executable, "-m", "bit2watt_impl.crypto_workload",
            "--gpu-id", str(args.gpu_id), "--duration", str(args.duration),
        ]
    else:
        raise ValueError(f"알 수 없는 workload: {args.workload}")

    output = args.output or ROOT / "dataset" / "real" / label / f"{session_id}.csv"
    job_type_map = {
        "swma": "unknown_cuda_workload",
        "ltma": "llm_training",
        "cryptojacking": "unregistered_hash_benchmark",
    }
    job_type = args.job_type or job_type_map.get(label, label)
    gres_req = args.gres_req or f"gpu:{len(gpu_ids)}"
    collector = [
        sys.executable, "-m", "bit2watt_impl.collect_telemetry",
        "--output", str(output),
        "--duration", str(args.duration + args.pre_capture_s + args.post_capture_s),
        "--interval-ms", str(args.interval_ms),
        "--gpu-ids", ",".join(map(str, gpu_ids)),
        "--session-id", session_id,
        "--label", label,
        "--id-user", args.id_user,
        "--job-type", job_type,
        "--gres-req", gres_req,
    ]
    return collector, workload, label, gpu_ids, output, job_type


def write_session_metadata(
    output: Path,
    *,
    session_id: str,
    label: str,
    gpu_ids: list[int],
    collector_cmd: list[str],
    workload_cmd: list[str],
    job_type: str,
    dry_run: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "session_id": session_id,
        "label": label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": gpu_ids,
        "job_type": job_type,
        "collector": collector_cmd,
        "workload": workload_cmd,
        "dry_run": dry_run,
    }
    output.with_suffix(".session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=[*NORMAL_MODES, "swma", "swma_multi", "ltma", "cryptojacking"],
        required=True,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--interval-ms", type=float, default=100)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--duty-cycle", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--id-user", default="unknown")
    parser.add_argument("--job-type")
    parser.add_argument("--gres-req")
    parser.add_argument("--pre-capture-s", type=float, default=3.0)
    parser.add_argument("--post-capture-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-shared-gpu-safe",
        action="store_true",
        help="다른 사용자 작업이 없음을 확인했을 때만 지정",
    )
    args = parser.parse_args()
    if not args.confirm_shared_gpu_safe and not args.dry_run:
        parser.error("실측 부하 실행 전 --confirm-shared-gpu-safe가 필요합니다.")
    if not 0 < args.duration <= 300:
        parser.error("--duration은 0초 초과 300초 이하여야 합니다.")
    if args.pre_capture_s < 0 or args.post_capture_s < 0:
        parser.error("--pre-capture-s와 --post-capture-s는 음수일 수 없습니다.")

    session_id = args.session_id or f"real-{args.workload}-{uuid.uuid4().hex[:10]}"
    collector_cmd, workload_cmd, label, gpu_ids, output, job_type = commands(args, session_id)
    summary = {
        "session_id": session_id,
        "label": label,
        "gpu_ids": gpu_ids,
        "output": str(output),
        "collector": collector_cmd,
        "workload": workload_cmd,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    collector = subprocess.Popen(collector_cmd, cwd=ROOT)
    try:
        time.sleep(args.pre_capture_s)
        workload = subprocess.run(workload_cmd, cwd=ROOT, check=False)
        if workload.returncode:
            raise RuntimeError(f"워크로드 실패(exit={workload.returncode})")
        time.sleep(args.post_capture_s)
        return_code = collector.wait(timeout=args.duration + 15)
        if return_code:
            raise RuntimeError(f"수집기 실패(exit={return_code})")
    finally:
        if collector.poll() is None:
            collector.terminate()
            collector.wait(timeout=5)
    write_session_metadata(
        output,
        session_id=session_id,
        label=label,
        gpu_ids=gpu_ids,
        collector_cmd=collector_cmd,
        workload_cmd=workload_cmd,
        job_type=job_type,
        dry_run=False,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

