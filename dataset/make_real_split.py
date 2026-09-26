"""Create group-stratified outer and inner real-data folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.declared_context import expected_band_for_job_type, family_for_job_type

ROOT = Path(__file__).resolve().parent.parent


def _truth_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def _group_records(labels: pd.DataFrame) -> list[dict]:
    required = {"session_id", "group_id", "gt_label", "gt_is_attack"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels missing required columns: {sorted(missing)}")
    valid = labels.copy()
    if "valid" in valid:
        valid = valid[valid["valid"].astype(str).str.lower().isin({"true", "1"})]
    valid["session_id"] = valid["session_id"].astype(str)
    valid["group_id"] = valid["group_id"].astype(str)
    valid["gt_is_attack"] = _truth_bool(valid["gt_is_attack"])
    records = []
    for group_id, rows in valid.groupby("group_id", sort=True):
        labels_in_group = sorted(set(rows["gt_label"].astype(str)))
        attack_state = "attack" if rows["gt_is_attack"].any() else "normal"
        targets = rows
        if "gpu_role" in rows.columns:
            targets = rows[rows["gpu_role"].astype(str) == "target"]
            if targets.empty:
                targets = rows
        job_type = None
        if "declared_job_type" in targets.columns and targets["declared_job_type"].notna().any():
            job_type = str(targets["declared_job_type"].dropna().astype(str).iloc[0])
        band = expected_band_for_job_type(job_type) if job_type else None
        family = family_for_job_type(job_type) if job_type else None
        cohort = f"{family}|{band}" if family and band else None
        if attack_state == "normal" and job_type:
            stratum = f"normal:{job_type}"
        else:
            stratum = f"{attack_state}:{'+'.join(labels_in_group)}"
        records.append(
            {
                "group_id": str(group_id),
                "stratum": stratum,
                "job_type": job_type,
                "cohort": cohort if attack_state == "normal" else None,
                "sessions": sorted(set(rows["session_id"])),
            }
        )
    return records


def _assign_folds(records: list[dict], n_splits: int, seed: int) -> list[list[dict]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_stratum[record["stratum"]].append(record)
    folds: list[list[dict]] = [[] for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for stratum in sorted(by_stratum):
        items = list(by_stratum[stratum])
        rng.shuffle(items)
        # Rare real-capture strata may have fewer groups than n_splits.
        # Still assign without repeating a group across folds.
        for index, item in enumerate(items):
            folds[index % n_splits].append(item)
    return folds


def make_real_split(
    labels: pd.DataFrame,
    *,
    outer_folds: int = 3,
    seed: int = 7,
) -> dict:
    records = _group_records(labels)
    outer = _assign_folds(records, outer_folds, seed)
    result = {
        "schema_version": "real_nested_cv_v1",
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "folds": [],
    }
    for fold_index, test_records in enumerate(outer):
        test_groups = {record["group_id"] for record in test_records}
        remaining = [record for record in records if record["group_id"] not in test_groups]
        rng = np.random.default_rng(seed + 1000 + fold_index)
        by_job: dict[str, list[dict]] = defaultdict(list)
        for record in remaining:
            if record.get("job_type") and str(record["stratum"]).startswith("normal:"):
                by_job[str(record["job_type"])].append(record)
        seeded = []
        seeded_ids = set()
        for job_type in sorted(by_job):
            items = list(by_job[job_type])
            rng.shuffle(items)
            chosen = items[0]
            seeded.append(chosen)
            seeded_ids.add(chosen["group_id"])
        leftover = [record for record in remaining if record["group_id"] not in seeded_ids]
        by_cohort: dict[str, list[dict]] = defaultdict(list)
        for record in leftover:
            if record.get("cohort") and str(record["stratum"]).startswith("normal:"):
                by_cohort[str(record["cohort"])].append(record)
        cal_seeded = []
        cal_seeded_ids = set()
        for cohort in sorted(by_cohort):
            items = list(by_cohort[cohort])
            rng.shuffle(items)
            chosen = items[0]
            cal_seeded.append(chosen)
            cal_seeded_ids.add(chosen["group_id"])
        leftover = [record for record in leftover if record["group_id"] not in cal_seeded_ids]
        if leftover:
            inner = _assign_folds(leftover, 2, seed + 2000 + fold_index)
            cal_records = cal_seeded + inner[0]
            train_records = seeded + inner[1]
        else:
            cal_records = cal_seeded
            train_records = seeded
        train_job_types = sorted(
            {
                str(item["job_type"])
                for item in train_records
                if item.get("job_type") and str(item["stratum"]).startswith("normal:")
            }
        )
        remaining_normal_types = sorted(by_job)
        remaining_cohorts = sorted(
            {
                str(item["cohort"])
                for item in remaining
                if item.get("cohort") and str(item["stratum"]).startswith("normal:")
            }
        )
        cal_cohorts = {
            str(item["cohort"])
            for item in cal_records
            if item.get("cohort") and str(item["stratum"]).startswith("normal:")
        }
        fold = {
            "fold": fold_index,
            "train": sorted(session for item in train_records for session in item["sessions"]),
            "cal": sorted(session for item in cal_records for session in item["sessions"]),
            "test": sorted(session for item in test_records for session in item["sessions"]),
            "train_groups": sorted(item["group_id"] for item in train_records),
            "cal_groups": sorted(item["group_id"] for item in cal_records),
            "test_groups": sorted(test_groups),
            "train_job_types": train_job_types,
            "missing_train_job_types": [name for name in remaining_normal_types if name not in train_job_types],
            "cal_cohorts": sorted(cal_cohorts),
            "missing_cal_cohorts": [name for name in remaining_cohorts if name not in cal_cohorts],
        }
        group_sets = [set(fold[name]) for name in ("train_groups", "cal_groups", "test_groups")]
        if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
            raise AssertionError("group leaked across nested CV partitions")
        result["folds"].append(fold)
    canonical = json.dumps(result["folds"], sort_keys=True).encode("utf-8")
    result["fold_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def write_real_split(report: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for fold in report["folds"]:
        fold_dir = output_root / f"fold_{int(fold['fold']):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": report["schema_version"],
            "seed": report["seed"],
            **fold,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "dataset/real/index/labels.csv",
    )
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "dataset/eval/real_cv",
    )
    args = parser.parse_args()
    report = make_real_split(
        pd.read_csv(args.labels, low_memory=False),
        outer_folds=args.outer_folds,
        seed=args.seed,
    )
    write_real_split(report, args.out_root)
    print(json.dumps({"folds": len(report["folds"]), "sha256": report["fold_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
