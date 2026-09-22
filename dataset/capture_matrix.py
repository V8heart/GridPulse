"""Seeded, resumable real-capture matrix runner for gp-telemetry/1.2."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dataset.identifiers import group_id_from_parts

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "capture_matrix.yaml"
DEFAULT_LABELS = ROOT / "dataset" / "real" / "index" / "labels.csv"
ATTACK_WORKLOADS = {
    "swma",
    "swma_basic",
    "swma_shallow",
    "swma_jitter",
    "swma_piggyback",
    "swma_mimicry",
    "swma_multi",
    "swma_coordinated",
    "ltma",
    "cryptojacking",
}
HOSTED_WORKLOADS = {"swma_piggyback", "swma_mimicry", "ltma"}
NORMAL_WORKLOADS = {
    "baseline",
    "distributed",
    "hpo",
    "checkpoint",
    "dataloader_stall",
    "eval_train_switch",
    "gpt_tiny_finetune",
    "llm_pretrain_ddp",
    "llm_pretrain_fsdp",
    "llm_flat_pretrain",
    "llm_finetune",
    "llm_inference_serving",
    "llm_inference_batch",
    "resnet_single",
    "resnet_ddp",
}
FORWARDED_PARAMS = {
    "period",
    "duty_cycle",
    "declared_policy",
    "host",
    "dataloader",
    "preset",
    "seq_len",
    "grad_accum",
    "rps",
    "batch_size",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def capture_seed(config_seed: int, capture_key: str) -> int:
    """Positive int63 seed unique to one capture and stable across dry-runs."""
    digest = hashlib.sha256(f"{int(config_seed)}:{capture_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "kdn-capture-matrix/1":
        raise ValueError("capture matrix schema must be kdn-capture-matrix/1")
    durations = data.get("duration_pool_s")
    if durations != [480, 600, 720]:
        raise ValueError("duration_pool_s must be exactly [480, 600, 720]")
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a non-empty list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("workload"):
            raise ValueError(f"entry {index} requires workload")
        repeats = int(entry.get("repeats", 0))
        if repeats <= 0:
            raise ValueError(f"entry {index} repeats must be positive")
        if entry["workload"] in ATTACK_WORKLOADS and repeats % 2:
            raise ValueError(f"attack entry {index} ({entry['workload']}) has odd repeats")
    cooldown = data.get("cooldown") or {}
    if float(cooldown.get("min_seconds", 0)) < 60:
        raise ValueError("cooldown.min_seconds must be at least 60")
    if float(cooldown.get("max_wait_seconds", 0)) > 900:
        raise ValueError("cooldown.max_wait_seconds must not exceed 900")
    if float(cooldown.get("idle_temp_delta_c", 0)) != 3:
        raise ValueError("cooldown.idle_temp_delta_c must be 3")
    return data


def _grid_assignments(grid: dict[str, Any], repeats: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if not grid:
        return [{} for _ in range(repeats)]
    combos: list[dict[str, Any]] = [{}]
    for key in sorted(grid):
        values = list(grid[key])
        if not values:
            raise ValueError(f"params_grid.{key} is empty")
        combos = [{**base, key: value} for base in combos for value in values]
    order = list(range(len(combos)))
    rng.shuffle(order)
    return [dict(combos[order[index % len(combos)]]) for index in range(repeats)]


def build_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand and seed-shuffle entries; duration assignment does not use labels."""
    seed = int(config.get("seed", 7))
    config_hash = hashlib.sha256(_canonical(config).encode()).hexdigest()[:16]
    grid_rng = np.random.default_rng(seed + 10007)
    pending: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for entry_index, entry in enumerate(config["entries"]):
        repeats = int(entry["repeats"])
        assignments = _grid_assignments(dict(entry.get("params_grid") or {}), repeats, grid_rng)
        policies = list(entry.get("declared_policies") or [])
        hosted = entry["workload"] in HOSTED_WORKLOADS
        for repeat in range(repeats):
            params = dict(entry.get("params") or {})
            params.update(assignments[repeat])
            if hosted:
                params["declared_policy"] = "host_inherited"
            elif policies:
                params["declared_policy"] = policies[repeat % len(policies)]
            pending.append((entry_index, repeat, entry, params))
    rng = np.random.default_rng(seed)
    rng.shuffle(pending)
    durations = list(config["duration_pool_s"])
    result = []
    used_seeds: set[int] = set()
    for shuffled_index, (entry_index, repeat, entry, params) in enumerate(pending):
        duration = int(durations[int(rng.integers(0, len(durations)))])
        key_payload = {
            "config": config_hash,
            "entry": entry_index,
            "repeat": repeat,
            "params": params,
        }
        capture_key = "c-" + hashlib.sha256(_canonical(key_payload).encode()).hexdigest()[:20]
        session_seed = capture_seed(seed, capture_key)
        while session_seed in used_seeds:
            session_seed = (session_seed + 1) & ((1 << 63) - 1) or 1
        used_seeds.add(session_seed)
        result.append(
            {
                "capture_key": capture_key,
                "seed": session_seed,
                "group_id": group_id_from_parts(config_hash, entry["workload"], params),
                "entry_index": entry_index,
                "repeat": repeat,
                "shuffle_index": shuffled_index,
                "workload": str(entry["workload"]),
                "gpu_id": int(entry.get("gpu_id", 0)),
                "duration_s": duration,
                "params": params,
            }
        )
    return result


def completed_capture_keys(labels_path: Path = DEFAULT_LABELS) -> set[str]:
    if not labels_path.exists():
        return set()
    frame = pd.read_csv(labels_path)
    required = {"capture_key", "gpu_role", "valid"}
    if not required.issubset(frame.columns):
        return set()
    valid = frame["valid"].astype(str).str.lower().isin({"true", "1"})
    targets = frame["gpu_role"].astype(str).eq("target")
    return set(frame.loc[valid & targets, "capture_key"].dropna().astype(str))


def shard(items: list[dict[str, Any]], count: int, index: int) -> list[dict[str, Any]]:
    if count not in {1, 2, 3}:
        raise ValueError("night shards must be 1, 2, or 3")
    if not 0 <= index < count:
        raise ValueError("shard index must be in [0, shards)")
    return [item for item in items if item["shuffle_index"] % count == index]


def summarize(items: list[dict[str, Any]], cooldown_s: float) -> dict[str, Any]:
    duration_seconds = sum(item["duration_s"] for item in items)
    gaps = max(0, len(items) - 1)
    estimated = duration_seconds + gaps * cooldown_s
    # Startup plus a typical cooldown tail. The 15 minute cap is not assumed every session.
    planning = duration_seconds + len(items) * 90 + gaps * 180
    normal = [item for item in items if item["workload"] in NORMAL_WORKLOADS]
    attack = [item for item in items if item["workload"] in ATTACK_WORKLOADS]
    return {
        "session_count": len(items),
        "normal_sessions": len(normal),
        "attack_sessions": len(attack),
        "estimated_hours": round(estimated / 3600.0, 2),
        "planning_hours": round(planning / 3600.0, 2),
        "planning_note": "3 night shards; 90s startup and 3 min mean cooldown are in planning_hours. The 15 minute cooldown cap is not fully budgeted.",
        "workloads": dict(sorted(Counter(x["workload"] for x in items).items())),
        "durations_s": dict(sorted(Counter(x["duration_s"] for x in items).items())),
        "declared_policy": dict(
            sorted(Counter(str(x["params"].get("declared_policy", "default")) for x in items).items())
        ),
        "periods": dict(
            sorted(Counter(x["params"].get("period") for x in items if "period" in x["params"]).items())
        ),
        "hosted_vs_standalone": {
            "hosted": sum(item["workload"] in HOSTED_WORKLOADS for item in attack),
            "standalone": sum(item["workload"] not in HOSTED_WORKLOADS for item in attack),
        },
        "attack_policy_counts": dict(
            sorted(Counter(str(item["params"].get("declared_policy", "default")) for item in attack).items())
        ),
        "normal_group_count": len({item["group_id"] for item in normal}),
        "attack_group_count": len({item["group_id"] for item in attack}),
    }


def _gpu_temperature(gpu_id: int) -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def cooldown(gpu_id: int, *, minimum_s: float, max_wait_s: float, idle_temp_c: float | None,
             delta_c: float, poll_s: float) -> None:
    time.sleep(minimum_s)
    if idle_temp_c is None:
        return
    deadline = time.monotonic() + max(0.0, max_wait_s - minimum_s)
    while time.monotonic() < deadline:
        temp = _gpu_temperature(gpu_id)
        if temp is None or temp <= idle_temp_c + delta_c:
            return
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


def command_for(item: dict[str, Any], *, allow_busy: bool, confirm: bool) -> list[str]:
    command = [
        sys.executable, "-m", "dataset.run_capture",
        "--workload", item["workload"],
        "--gpu-id", str(item["gpu_id"]),
        "--duration", str(item["duration_s"]),
        "--capture-key", item["capture_key"],
        "--group-id", item["group_id"],
        "--seed", str(item["seed"]),
    ]
    for key, value in sorted(item["params"].items()):
        if key not in FORWARDED_PARAMS or value is None:
            continue
        if key == "declared_policy" and value == "host_inherited":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    if allow_busy:
        command.append("--allow-busy")
    if confirm:
        command.append("--confirm-shared-gpu-safe")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--confirm-shared-gpu-safe", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    all_items = build_matrix(config)
    selected = shard(all_items, args.shards, args.shard_index)
    done = completed_capture_keys(args.labels)
    remaining = [item for item in selected if item["capture_key"] not in done]
    cooldown_cfg = config["cooldown"]
    summary = summarize(remaining, float(cooldown_cfg["min_seconds"]))
    summary.update({
        "total_matrix_sessions": len(all_items),
        "resumed_sessions": len(selected) - len(remaining),
        "shards": args.shards,
        "shard_index": args.shard_index,
    })
    if args.dry_run:
        print(json.dumps({"summary": summary, "captures": remaining}, indent=2))
        return
    if not args.confirm_shared_gpu_safe:
        parser.error("--confirm-shared-gpu-safe is required for real captures")

    for position, item in enumerate(remaining):
        subprocess.run(
            command_for(item, allow_busy=args.allow_busy, confirm=True),
            cwd=ROOT,
            check=True,
        )
        if position + 1 < len(remaining):
            cooldown(
                item["gpu_id"],
                minimum_s=float(cooldown_cfg["min_seconds"]),
                max_wait_s=float(cooldown_cfg["max_wait_seconds"]),
                idle_temp_c=(
                    None if cooldown_cfg.get("idle_temp_c") is None
                    else float(cooldown_cfg["idle_temp_c"])
                ),
                delta_c=float(cooldown_cfg["idle_temp_delta_c"]),
                poll_s=float(cooldown_cfg.get("poll_seconds", 10)),
            )
    print(json.dumps({"summary": summary, "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
