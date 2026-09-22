"""Compare Stage 1 feature distributions between synthetic and real windows."""
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

BASE_FEATURES = [
    "mean_w",
    "swing_abs_w",
    "dominant_freq_hz",
    "util_residual_mad_w",
    "spectral_entropy",
]


def _read_windows(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    for key in ("all_windows", "scored_rows", "rows", "windows"):
        if isinstance(payload.get(key), list):
            return pd.DataFrame(payload[key])
    raise ValueError(f"no window rows in {path}")


def _attack(frame: pd.DataFrame) -> pd.Series:
    if "gt_is_attack" in frame:
        return frame["gt_is_attack"].astype(str).str.lower().isin({"true", "1"})
    return ~frame["gt_label"].astype(str).str.startswith("normal")


def compare_synth_vs_real(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    *,
    tdp_w: float = 450.0,
) -> dict:
    features = [
        column
        for column in sorted(set(synthetic.columns) & set(real.columns))
        if column in BASE_FEATURES or column.startswith("band_power_w2_")
    ]
    if not features:
        raise ValueError("inputs share no Stage 1 comparison features")
    rows = []
    for source, frame in (("synthetic", synthetic), ("real", real)):
        if "gt_label" not in frame:
            raise ValueError(f"{source} windows lack gt_label")
        for label, group in frame.groupby(frame["gt_label"].astype(str), sort=True):
            for feature in features:
                values = pd.to_numeric(group[feature], errors="coerce").dropna()
                rows.append(
                    {
                        "source": source,
                        "gt_label": str(label),
                        "feature": feature,
                        "n_windows": int(len(values)),
                        "n_sessions": int(group["session_id"].nunique()) if "session_id" in group else None,
                        "q05": float(values.quantile(0.05)) if len(values) else None,
                        "q50": float(values.quantile(0.50)) if len(values) else None,
                        "q95": float(values.quantile(0.95)) if len(values) else None,
                    }
                )
    direction = {}
    for source, frame in (("synthetic", synthetic), ("real", real)):
        attack = _attack(frame)
        normal_mean = pd.to_numeric(frame.loc[~attack, "mean_w"], errors="coerce").mean()
        attack_mean = pd.to_numeric(frame.loc[attack, "mean_w"], errors="coerce").mean()
        delta = float(attack_mean - normal_mean)
        direction[source] = {
            "normal_mean_w": float(normal_mean),
            "attack_mean_w": float(attack_mean),
            "attack_minus_normal_w": delta,
            "direction": "higher" if delta > 0 else "lower" if delta < 0 else "equal",
        }
    real_normal = real[~_attack(real)]
    if "gt_label" in real_normal:
        labels = real_normal["gt_label"].astype(str)
        llm = labels.str.startswith("normal_llm_") & ~labels.str.contains("inference")
        real_normal = real_normal[llm]
    if real_normal.empty or "mean_w" not in real_normal:
        llm_mean = float("nan")
        llm_status = "pending_real_capture"
    else:
        llm_mean = pd.to_numeric(real_normal["mean_w"], errors="coerce").mean()
        llm_status = "measured" if pd.notna(llm_mean) else "pending_real_capture"
    same_direction = direction["synthetic"]["direction"] == direction["real"]["direction"]
    return {
        "features": features,
        "distributions": rows,
        "normal_llm_training": {
            "status": llm_status,
            "real_mean_w": None if np.isnan(llm_mean) else float(llm_mean),
            "tdp_w": float(tdp_w),
            "mean_fraction_tdp": None if np.isnan(llm_mean) else float(llm_mean / tdp_w),
        },
        "attack_normal_mean_direction": direction,
        "same_direction": bool(same_direction),
        "generator_adjustment": (
            "none implied by this report"
            if same_direction
            else "direction differs; report only, generator correction is separate work"
        ),
    }


def _write_markdown(report: dict, path: Path) -> None:
    normal = report["normal_llm_training"]
    lines = [
        "# Synthetic vs real Stage 1 features",
        "",
        f"- Real normal training mean/TDP: `{normal['mean_fraction_tdp']}`",
        f"- Attack-minus-normal direction matches: `{report['same_direction']}`",
        f"- Follow-up: {report['generator_adjustment']}",
        "",
        "| source | label | feature | windows | sessions | q05 | q50 | q95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["distributions"]:
        lines.append(
            f"| {row['source']} | {row['gt_label']} | {row['feature']} | "
            f"{row['n_windows']} | {row['n_sessions']} | {row['q05']} | "
            f"{row['q50']} | {row['q95']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--tdp-w", type=float, default=450.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dataset/eval/v2_regen/synth_vs_real.json",
    )
    args = parser.parse_args()
    report = compare_synth_vs_real(
        _read_windows(args.synthetic),
        _read_windows(args.real),
        tdp_w=args.tdp_w,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(report, args.out.with_suffix(".md"))
    print(json.dumps({"features": report["features"], "same_direction": report["same_direction"]}, indent=2))


if __name__ == "__main__":
    main()
