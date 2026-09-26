"""Rewrite honest-normal declared_job_type without touching progress.raw."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from dataset.declared_context import (
    ATTACK_DECLARED_POLICIES,
    MID_GROUP_FOR_JOB_TYPE,
    honest_declared_for_workload,
    resolve_honest_workload_key,
)
from dataset.finalize_session import (
    DEFAULT_WARMUP_S,
    LABEL_COLUMNS,
    _attach_columns,
    _atomic_write_json,
    _resolve_paths,
    _upsert_labels,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_params(raw) -> dict:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _cmd_flag(cmd, flag: str) -> str | None:
    if cmd is None or (isinstance(cmd, float) and pd.isna(cmd)):
        return None
    parts = cmd
    if isinstance(cmd, str):
        try:
            parsed = json.loads(cmd)
            parts = parsed if isinstance(parsed, list) else cmd.split()
        except json.JSONDecodeError:
            parts = cmd.split()
    if not isinstance(parts, list):
        return None
    tokens = [str(x) for x in parts]
    if flag in tokens:
        index = tokens.index(flag)
        if index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def workload_candidates(private: dict, labels_row: pd.Series | None = None) -> list[str]:
    params = _parse_params(private.get("gt_params_json"))
    cmd = private.get("workload_cmd")
    if labels_row is not None and (cmd is None or (isinstance(cmd, float) and pd.isna(cmd))):
        cmd = labels_row.get("workload_cmd")
    values = [
        params.get("workload"),
        private.get("gt_variant"),
        _cmd_flag(cmd, "--mode"),
        _cmd_flag(cmd, "--variant"),
    ]
    if labels_row is not None:
        values.extend([labels_row.get("gt_variant"), _parse_params(labels_row.get("gt_params_json")).get("workload")])
    return [str(v) for v in values if v is not None and str(v).strip() and str(v).lower() not in {"nan", "none"}]


def should_redeclare(private: dict) -> bool:
    policy = str(private.get("declared_policy") or "")
    if policy in ATTACK_DECLARED_POLICIES:
        return False
    if policy != "honest":
        return False
    if bool(private.get("gt_is_attack")):
        return False
    label = str(private.get("gt_label") or "")
    return label.startswith("normal")


def _raw_bytes(session_dir: Path) -> bytes:
    raw = session_dir / "progress.raw.jsonl"
    return raw.read_bytes() if raw.exists() else b""


def redeclare_session(session_id: str, *, root: Path = ROOT) -> list[dict]:
    paths = _resolve_paths(session_id, root=root)
    session_dir = paths["session_dir"]
    private_path = paths["private_capture"]
    if not private_path.exists():
        raise FileNotFoundError(private_path)
    private = _load_json(private_path)
    session_json_path = session_dir / "session.json"
    session_meta = _load_json(session_json_path) if session_json_path.exists() else {}
    telemetry_path = session_dir / "telemetry.csv"
    telemetry = pd.read_csv(telemetry_path, low_memory=False)
    raw_before = _raw_bytes(session_dir)

    roles = private.get("gpu_roles") or {}
    declared_by_gpu = dict(private.get("declared_by_gpu") or {})
    remap_rows = []
    changed = False
    if should_redeclare(private):
        keys = workload_candidates(private)
        for gid, role in roles.items():
            if role != "target":
                continue
            current = dict(declared_by_gpu.get(str(gid)) or declared_by_gpu.get(int(gid) if str(gid).isdigit() else gid) or {})
            user = str(current.get("declared_user") or "user_000")
            updated = honest_declared_for_workload(*keys, user=user)
            before = current.get("declared_job_type")
            after = updated["declared_job_type"]
            remap_rows.append(
                {
                    "session_id": session_id,
                    "gpu_id": gid,
                    "gpu_role": role,
                    "declared_policy": private.get("declared_policy"),
                    "workload": resolve_honest_workload_key(*keys),
                    "before": before,
                    "after": after,
                    "changed": before != after,
                }
            )
            if before != after:
                changed = True
                declared_by_gpu[str(gid)] = updated
    else:
        for gid, role in roles.items():
            if role != "target":
                continue
            current = declared_by_gpu.get(str(gid)) or {}
            remap_rows.append(
                {
                    "session_id": session_id,
                    "gpu_id": gid,
                    "gpu_role": role,
                    "declared_policy": private.get("declared_policy"),
                    "workload": (_parse_params(private.get("gt_params_json")).get("workload")),
                    "before": current.get("declared_job_type"),
                    "after": current.get("declared_job_type"),
                    "changed": False,
                }
            )

    if changed:
        private["declared_by_gpu"] = declared_by_gpu
        t0_epoch = float(session_meta.get("t0_epoch") or telemetry["t_epoch"].iloc[0] - telemetry["timestamp"].iloc[0])
        warmup_s = float(session_meta.get("warmup_s") or DEFAULT_WARMUP_S)
        enriched = _attach_columns(
            telemetry,
            private,
            t0_epoch=t0_epoch,
            warmup_s=warmup_s,
            workload_start_epoch=None,
        )
        tmp = telemetry_path.with_suffix(".csv.tmp")
        enriched.to_csv(tmp, index=False)
        tmp.replace(telemetry_path)
        if session_json_path.exists():
            session_meta["declared"] = declared_by_gpu
            session_json_path.write_text(json.dumps(session_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _atomic_write_json(private_path, private, mode=0o600)

    raw_after = _raw_bytes(session_dir)
    if raw_before != raw_after:
        raise RuntimeError(f"redeclare must not mutate progress.raw.jsonl ({session_id})")

    return remap_rows


def _label_declared_type(private: dict, gpu_id: int, role: str) -> str | None:
    declared = (private.get("declared_by_gpu") or {}).get(str(gpu_id)) or {}
    if role == "companion_idle":
        return declared.get("declared_job_type", "notebook")
    return declared.get("declared_job_type")


def write_label_declared_types(*, root: Path = ROOT) -> None:
    paths = _resolve_paths("s-placeholder", root=root)
    labels_path = paths["labels"]
    labels = pd.read_csv(labels_path, low_memory=False)
    rows = []
    for _, row in labels.iterrows():
        data = row.to_dict()
        session_id = str(row["session_id"])
        private_path = paths["private_capture"].parent / f"{session_id}.json"
        if private_path.exists():
            private = _load_json(private_path)
            role = str(row.get("gpu_role") or "target")
            data["declared_job_type"] = _label_declared_type(private, int(row["gpu_id"]), role)
        rows.append(data)
    _upsert_labels(labels_path, paths["labels_lock"], rows)


def cohort_inventory(labels: pd.DataFrame) -> dict:
    valid = labels.copy()
    if "valid" in valid.columns:
        valid = valid[valid["valid"].astype(str).str.lower().isin({"true", "1"})]
    if "gpu_role" in valid.columns:
        valid = valid[valid["gpu_role"].astype(str) == "target"]
    rows = []
    for job_type, group in valid.groupby(valid["declared_job_type"].astype(str), dropna=False):
        n_all = int(group["session_id"].nunique())
        grp_attack = group["gt_is_attack"].astype(str).str.lower().isin({"true", "1"})
        n_normal = int(group.loc[~grp_attack, "session_id"].nunique())
        n_attack = int(group.loc[grp_attack, "session_id"].nunique())
        est_train = round(n_normal / 3.0, 1)
        rows.append(
            {
                "declared_job_type": str(job_type),
                "n_all": n_all,
                "n_normal": n_normal,
                "n_attack": n_attack,
                "est_train": est_train,
                "thin": bool(n_normal > 0 and est_train < 3),
            }
        )
    rows.sort(key=lambda item: item["declared_job_type"])
    normal_types = [item for item in rows if item["n_normal"] > 0]
    thin_frac = (sum(1 for item in normal_types if item["thin"]) / len(normal_types)) if normal_types else 0.0
    return {
        "rows": rows,
        "n_normal_job_types": len(normal_types),
        "n_thin_normal_job_types": sum(1 for item in normal_types if item["thin"]),
        "thin_fraction": thin_frac,
        "recommend_decision": thin_frac > 0.5,
    }


def write_inventory_md(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Round2 cohort inventory (target sessions)",
        "",
        "3-fold train estimate is `n_normal / 3`. Split/fit not run yet.",
        "",
        "| job_type | all | normal | attack | est_train | thin (<3) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['declared_job_type']} | {row['n_all']} | {row['n_normal']} | "
            f"{row['n_attack']} | {row['est_train']} | {row['thin']} |"
        )
    lines.extend(
        [
            "",
            f"normal job types: {report['n_normal_job_types']}",
            f"thin normal job types (est_train < 3): {report['n_thin_normal_job_types']} "
            f"({report['thin_fraction']:.2f})",
            "",
        ]
    )
    mid_rows = []
    by_mid = defaultdict(lambda: {"n_normal": 0, "n_attack": 0})
    for row in report["rows"]:
        mid = MID_GROUP_FOR_JOB_TYPE.get(row["declared_job_type"], row["declared_job_type"])
        by_mid[mid]["n_normal"] += row["n_normal"]
        by_mid[mid]["n_attack"] += row["n_attack"]
    for mid in sorted(by_mid):
        n_normal = by_mid[mid]["n_normal"]
        est_train = round(n_normal / 3.0, 1)
        mid_rows.append(
            {
                "mid_group": mid,
                "n_normal": n_normal,
                "n_attack": by_mid[mid]["n_attack"],
                "est_train": est_train,
                "thin": bool(n_normal > 0 and est_train < 3),
            }
        )
    lines.extend(
        [
            "## Mid-group rollup (if option 2)",
            "",
            "| mid_group | normal | attack | est_train | thin (<3) |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in mid_rows:
        lines.append(
            f"| {row['mid_group']} | {row['n_normal']} | {row['n_attack']} | "
            f"{row['est_train']} | {row['thin']} |"
        )
    lines.append("")
    if report["recommend_decision"]:
        lines.extend(
            [
                "Thin normal cohorts exceed half. Choose one before split/fit:",
                "",
                "1. `min_sessions=2`",
                "2. Mid-group cohort keys: `llm_train` / `distributed_train` / "
                "`vision_train` / `inference_online` / `inference_batch` / `interactive`",
                "",
                f"Mid-group map: `{MID_GROUP_FOR_JOB_TYPE}`",
                "",
                "Do not proceed with min_sessions=3 by default.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Thin share is at most half. `min_sessions=3` on declared job_type keys is viable.",
                "Still wait for an explicit choice before split/fit.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out-table",
        type=Path,
        default=ROOT / "dataset/real/pipeline/refit/round2/declared_remap.csv",
    )
    parser.add_argument(
        "--inventory-out",
        type=Path,
        default=ROOT / "dataset/real/pipeline/refit/round2/cohort_inventory.md",
    )
    args = parser.parse_args()
    paths = _resolve_paths("s-placeholder", root=args.root)
    labels = pd.read_csv(paths["labels"], low_memory=False)
    valid = labels
    if "valid" in labels.columns:
        valid = labels[labels["valid"].astype(str).str.lower().isin({"true", "1"})]
    session_ids = sorted(set(valid["session_id"].astype(str)))
    remap = []
    for session_id in session_ids:
        remap.extend(redeclare_session(session_id, root=args.root))
    attack_changed = [row for row in remap if row["declared_policy"] in ATTACK_DECLARED_POLICIES and row["changed"]]
    if attack_changed:
        raise RuntimeError(f"attack declarations were changed: {attack_changed[:3]}")
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(remap).to_csv(args.out_table, index=False)
    write_label_declared_types(root=args.root)
    labels_after = pd.read_csv(paths["labels"], low_memory=False)
    inventory = cohort_inventory(labels_after)
    write_inventory_md(inventory, args.inventory_out)
    print(
        json.dumps(
            {
                "sessions": len(session_ids),
                "remap_rows": len(remap),
                "honest_changed": int(sum(1 for row in remap if row["changed"])),
                "inventory": str(args.inventory_out),
                "thin_fraction": inventory["thin_fraction"],
                "stop": "choose min_sessions vs mid-group before split/fit",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
