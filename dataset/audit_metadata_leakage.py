"""Audit whether session/GPU metadata alone predicts attack status."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN_FEATURE_TOKENS = (
    "power",
    "util",
    "clock",
    "temp",
    "fan",
    "pstate",
    "throttle",
    "gt",
    "label",
    "variant",
    "role",
    "interval",
    "session_id",
    "attack",
    "waveform",
)
IDENTITY_COLUMNS = {"group_id", "capture_key", "run_id", "observed_pid", "pid"}


def _allowed_metadata_columns(frame: pd.DataFrame) -> list[str]:
    preferred = {
        "sample_hz",
        "requested_interval_ms",
        "gpu_model",
        "declared_job_type",
        "declared_job_family",
        "declared_gres",
        "declared_process_name",
        "source",
    }
    return [
        column
        for column in frame.columns
        if column in preferred
        and column not in IDENTITY_COLUMNS
        and not any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]


def aggregate_metadata(telemetry: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Create one metadata-only row per labelled session/GPU."""
    keys = ["session_id", "gpu_id"]
    missing = [column for column in keys if column not in telemetry or column not in labels]
    if missing:
        raise ValueError(f"missing session/GPU keys: {sorted(set(missing))}")
    allowed = _allowed_metadata_columns(telemetry)
    rows = []
    for key, group in telemetry.groupby(keys, sort=False):
        row = {"session_id": str(key[0]), "gpu_id": int(key[1])}
        # Shape/timing metadata is intentionally audited: it must not encode class.
        row["meta_n_rows"] = int(len(group))
        if "timestamp" in group:
            timestamp = pd.to_numeric(group["timestamp"], errors="coerce").dropna()
            row["meta_duration_s"] = (
                float(timestamp.max() - timestamp.min()) if len(timestamp) > 1 else 0.0
            )
        for column in allowed:
            values = group[column].dropna()
            if not len(values):
                row[column] = np.nan
            elif pd.api.types.is_numeric_dtype(values):
                row[column] = float(pd.to_numeric(values, errors="coerce").median())
            else:
                row[column] = str(values.mode().iloc[0])
        rows.append(row)
    metadata = pd.DataFrame(rows)
    label_columns = keys + ["gt_is_attack"]
    if "group_id" in labels:
        label_columns.append("group_id")
    truth = labels[label_columns].copy()
    truth["session_id"] = truth["session_id"].astype(str)
    truth["gpu_id"] = pd.to_numeric(truth["gpu_id"], errors="raise").astype(int)
    truth["gt_is_attack"] = truth["gt_is_attack"].astype(str).str.lower().isin({"true", "1"})
    merged = metadata.merge(truth, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no telemetry rows matched labels")
    return merged


def audit_metadata_leakage(
    telemetry: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    threshold: float = 0.60,
    folds: int = 5,
    seed: int = 7,
) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frame = aggregate_metadata(telemetry, labels)
    y = frame.pop("gt_is_attack").astype(int).to_numpy()
    groups = frame.pop("group_id").astype(str).to_numpy() if "group_id" in frame else None
    frame = frame.drop(columns=["session_id", "gpu_id"], errors="ignore")
    if len(np.unique(y)) != 2:
        raise ValueError("leakage audit requires both normal and attack rows")
    if any(any(token in c.lower() for token in FORBIDDEN_FEATURE_TOKENS) for c in frame):
        raise AssertionError("forbidden feature reached leakage model")

    class_counts = np.bincount(y)
    n_splits = min(int(folds), int(class_counts.min()))
    if groups is not None:
        per_class_groups = [
            len(set(groups[y == klass])) for klass in (0, 1)
        ]
        n_splits = min(n_splits, *per_class_groups)
    if n_splits < 2:
        raise ValueError("insufficient class/group support for OOF leakage audit")

    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in frame.columns if column not in numeric]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    models = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "tree": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
        ),
    }
    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    aucs = {}
    for name, model in models.items():
        predictions = np.full(len(frame), np.nan)
        split_iter = splitter.split(frame, y, groups) if groups is not None else splitter.split(frame, y)
        for train_idx, test_idx in split_iter:
            pipeline = Pipeline([("metadata", preprocessing), ("model", model)])
            pipeline.fit(frame.iloc[train_idx], y[train_idx])
            predictions[test_idx] = pipeline.predict_proba(frame.iloc[test_idx])[:, 1]
        aucs[name] = float(roc_auc_score(y, predictions))
    maximum = max(aucs.values())
    return {
        "passed": bool(maximum <= threshold),
        "threshold": float(threshold),
        "max_oof_auc": float(maximum),
        "oof_auc": aucs,
        "n_rows": int(len(frame)),
        "n_splits": int(n_splits),
        "group_aware": groups is not None,
        "features": list(frame.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--out", "--output",
        type=Path,
        default=ROOT / "dataset/eval/v2_regen/metadata_leakage_audit.json",
    )
    args = parser.parse_args()
    report = audit_metadata_leakage(
        pd.read_csv(args.telemetry, low_memory=False),
        pd.read_csv(args.labels, low_memory=False),
        threshold=args.threshold,
        folds=args.folds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
