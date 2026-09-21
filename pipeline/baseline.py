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

# Absolute MAD floors by feature family. Ratio-like features need unitless floors;
# watt features use sensor-scale floors; slope uses W/% floor.
DEFAULT_MAD_FLOORS = {
    "mean_w": 5.0,
    "swing_abs_w": 5.0,
    "util_residual_mad_w": 1.0,
    "util_slope_w_per_pct": 0.05,
    "dominant_freq_hz": 0.02,
    "spectral_entropy": 0.02,
    "band_frac_0.1_0.7": 0.02,
    "band_frac_0.7_2": 0.02,
    "band_frac_0.05_0.1": 0.02,
    "band_frac_2_nyq": 0.02,
}
MEDIAN_FRAC_FLOOR = 0.05


def mad_floor_for(feature: str, median: float, floors: dict[str, float] | None = None) -> float:
    table = floors or DEFAULT_MAD_FLOORS
    absolute = float(table.get(feature, 1e-3))
    proportional = MEDIAN_FRAC_FLOOR * abs(float(median))
    # Ratio / entropy / freq features: absolute floor only (proportional is meaningless).
    if feature.startswith("band_frac_") or feature in {"spectral_entropy", "dominant_freq_hz"}:
        return absolute
    return max(absolute, proportional)


@dataclass
class CohortBaseline:
    cohort_keys: tuple[str, ...] = ("declared_job_family", "gpu_model")
    features: list[str] = field(default_factory=lambda: FEATURES_V2.copy())
    min_windows: int = 30
    stats: dict = field(default_factory=dict)
    mad_floors: dict[str, float] = field(default_factory=lambda: DEFAULT_MAD_FLOORS.copy())
    fallback_counts: dict[str, int] = field(default_factory=lambda: {"full": 0, "family": 0, "global": 0})

    def fit(
        self,
        windows: pd.DataFrame,
        *,
        cohort_keys=("declared_job_family", "gpu_model"),
        features=FEATURES_V2,
        min_windows: int = 30,
        mad_floors: dict[str, float] | None = None,
    ) -> "CohortBaseline":
        self.cohort_keys = tuple(cohort_keys)
        self.features = list(features)
        self.min_windows = min_windows
        if mad_floors is not None:
            self.mad_floors = dict(mad_floors)
        self.stats = {}
        self.fallback_counts = {"full": 0, "family": 0, "global": 0}
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
            floor = mad_floor_for(feature, med, self.mad_floors)
            values[feature] = {"median": med, "mad": max(mad, floor), "mad_raw": mad, "mad_floor": floor}
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
        self.fallback_counts[level] = self.fallback_counts.get(level, 0) + 1
        for feature, values in stat.get("features", {}).items():
            if feature not in row or pd.isna(row[feature]):
                continue
            floor = float(values.get("mad_floor", mad_floor_for(feature, values["median"], self.mad_floors)))
            denom = max(1.4826 * float(values["mad"]), 1.4826 * floor)
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

    def mean_abs_robust_z(self, row: pd.Series | dict, level_key: str) -> float | None:
        stat = self.stats.get(level_key, {}).get("features", {})
        if not stat:
            return None
        values = []
        for feature, summary in stat.items():
            if feature not in row or pd.isna(row[feature]):
                continue
            floor = float(summary.get("mad_floor", mad_floor_for(feature, summary["median"], self.mad_floors)))
            denom = max(1.4826 * float(summary["mad"]), 1.4826 * floor)
            values.append(abs((float(row[feature]) - float(summary["median"])) / denom))
        if not values:
            return None
        return float(np.mean(values))

    def family_distances(self, row: pd.Series | dict) -> dict[str, float]:
        """Mean |robust-z| to each family:* cohort that exists."""
        out = {}
        for key in self.stats:
            if not key.startswith("family:"):
                continue
            family = key.split(":", 1)[1]
            dist = self.mean_abs_robust_z(row, key)
            if dist is not None:
                out[family] = dist
        return out

    def declared_family_mismatch_stats(self, row: pd.Series | dict) -> dict[str, float | None]:
        family_key = self.cohort_keys[0] if self.cohort_keys else "declared_job_family"
        declared = str(row.get(family_key, ""))
        distances = self.family_distances(row)
        declared_z = distances.get(declared)
        others = {k: v for k, v in distances.items() if k != declared}
        best_other = min(others.values()) if others else None
        return {
            "declared_family_mean_abs_z": declared_z,
            "best_other_family_mean_abs_z": best_other,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "cohort_keys": list(self.cohort_keys),
            "features": self.features,
            "min_windows": self.min_windows,
            "mad_floors": self.mad_floors,
            "fallback_counts": self.fallback_counts,
            "stats": self.stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CohortBaseline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            cohort_keys=tuple(data["cohort_keys"]),
            features=list(data["features"]),
            min_windows=int(data.get("min_windows", 30)),
            mad_floors=dict(data.get("mad_floors") or DEFAULT_MAD_FLOORS),
            fallback_counts=dict(data.get("fallback_counts") or {"full": 0, "family": 0, "global": 0}),
            stats=data["stats"],
        )
