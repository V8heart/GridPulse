"""Cohort robust baselines for Stage 1 v2."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_V2 = [
    "mean_w",
    "swing_abs_w",
    "band_frac_0.1_0.7",
    "band_frac_0.7_2",
    "dominant_freq_hz",
    "util_slope_w_per_pct",
    "util_residual_mad_w",
    "spectral_entropy",
]


@dataclass
class CohortBaseline:
    cohort_keys: tuple[str, ...] = ("declared_job_family", "gpu_model")
    features: list[str] = field(default_factory=lambda: FEATURES_V2.copy())
    min_windows: int = 30
    stats: dict = field(default_factory=dict)

    def fit(
        self,
        windows: pd.DataFrame,
        *,
        cohort_keys=("declared_job_family", "gpu_model"),
        features=FEATURES_V2,
        min_windows: int = 30,
    ) -> "CohortBaseline":
        self.cohort_keys = tuple(cohort_keys)
        self.features = list(features)
        self.min_windows = min_windows
        self.stats = {}
        self._fit_level(windows, self.cohort_keys, "full")
        self._fit_level(windows, self.cohort_keys[:1], "family")
        self._fit_global(windows)
        return self

    def _summarize(self, group: pd.DataFrame) -> dict:
        values = {}
        for feature in self.features:
            if feature not in group:
                continue
            series = pd.to_numeric(group[feature], errors="coerce").dropna()
            if series.empty:
                continue
            med = float(series.median())
            mad = float(np.median(np.abs(series.to_numpy() - med)))
            values[feature] = {"median": med, "mad": max(mad, 1e-6)}
        return values

    def _fit_level(self, windows: pd.DataFrame, keys: tuple[str, ...], level: str) -> None:
        if not keys:
            return
        for key, group in windows.groupby(list(keys), dropna=False):
            if len(group) < self.min_windows:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            self.stats[f"{level}:{'|'.join(map(str, key))}"] = {"n": len(group), "features": self._summarize(group)}

    def _fit_global(self, windows: pd.DataFrame) -> None:
        self.stats["global"] = {"n": len(windows), "features": self._summarize(windows)}

    def robust_z(self, row: pd.Series | dict) -> dict[str, float]:
        stat, level = self._select_stats(row)
        out = {"baseline_level": level}
        for feature, values in stat.get("features", {}).items():
            if feature not in row or pd.isna(row[feature]):
                continue
            denom = max(1.4826 * float(values["mad"]), 1e-6)
            out[feature] = float((float(row[feature]) - float(values["median"])) / denom)
        return out

    def _select_stats(self, row: pd.Series | dict) -> tuple[dict, str]:
        full_key = "full:" + "|".join(str(row.get(k, "")) for k in self.cohort_keys)
        if full_key in self.stats:
            return self.stats[full_key], "full"
        family_key = "family:" + str(row.get(self.cohort_keys[0], ""))
        if family_key in self.stats:
            return self.stats[family_key], "family"
        return self.stats.get("global", {"features": {}}), "global"

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "cohort_keys": list(self.cohort_keys),
            "features": self.features,
            "min_windows": self.min_windows,
            "stats": self.stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CohortBaseline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            cohort_keys=tuple(data["cohort_keys"]),
            features=list(data["features"]),
            min_windows=int(data["min_windows"]),
            stats=data["stats"],
        )
