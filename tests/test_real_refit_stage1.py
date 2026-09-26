"""Idle gate, real cohort keys, Stage2 rate merge, progress opt-in."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline.attack_visibility import attack_visibility_group, parse_period_s
from pipeline.baseline import CohortBaseline
from pipeline.context_evidence import read_session_progress
from pipeline.eval_stage1_auc_alpha import select_alpha
from pipeline.stage1_v2 import (
    CALIBRATION_SCHEMA_VERSION,
    combine_u_score,
    idle_gate_decision,
    load_config,
    score_window,
    zero_proc_mask,
)
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_llm import judge, normal_verdict
from pipeline.stage2_segments import aggregate_bool_evidence, merge_candidate_segments


def _mini_calibration(**overrides):
    cal = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "feature_abs_z": {
            "training": {"mean_w": [0.5, 1.0, 1.5, 2.0] * 10},
            "global": {"mean_w": [0.5, 1.0, 1.5, 2.0] * 20},
        },
        "normal_u_scores": {
            "full": {"training": [0.5, 1.0, 1.5, 2.0] * 10, "global": [0.5, 1.0, 1.5, 2.0] * 20},
            "no_evidence": {"training": [0.4, 0.8, 1.2, 1.6] * 10, "global": [0.4, 0.8, 1.2, 1.6] * 20},
        },
        "evidence_surprisal_weights": {"progress_log_missing": 1.5, "declared_family_mismatch": 1.0},
        "progress_log_policy_ok": True,
        "impact_quantiles": {"p50": 0.2, "p90": 0.5},
    }
    cal.update(overrides)
    return cal


def _baseline() -> CohortBaseline:
    return CohortBaseline(
        stats={"global": {"features": {"mean_w": {"median": 100, "mad": 10, "mad_floor": 5}}}}
    )


def _score_cfg(**overrides) -> dict:
    cfg = {
        "alpha": 0.05,
        "alpha_high_impact": 0.10,
        "tdp_w": 450,
        "grid_weights": {"0.1_0.7": 1.0},
        "evidence_weights": {},
        "mondrian_min_n": 30,
        "idle_gate": {"enabled": True, "mean_w_max": 40.0, "n_procs_agg": "median"},
    }
    cfg.update(overrides)
    return cfg


def test_idle_gate_zero_procs_low_power_is_idle():
    decision = idle_gate_decision({"observed_n_procs": 0, "mean_w": 24.0}, _score_cfg())
    assert decision == {"is_candidate": False, "idle_status": "idle", "candidate_reasons": []}


def test_idle_gate_zero_procs_high_power_is_undeclared_load():
    decision = idle_gate_decision({"observed_n_procs": 0, "mean_w": 200.0}, _score_cfg())
    assert decision["is_candidate"] is True
    assert decision["idle_status"] == "undeclared_load"
    assert decision["candidate_reasons"] == ["undeclared_load"]


def test_idle_gate_with_procs_passes_even_if_low_or_interactive():
    low = idle_gate_decision(
        {
            "observed_n_procs": 1,
            "mean_w": 24.0,
            "declared_job_family": "interactive",
            "gpu_role": "companion_idle",
            "gt_label": "normal_idle",
        },
        _score_cfg(),
    )
    mlp = idle_gate_decision(
        {
            "observed_n_procs": 2,
            "mean_w": 194.0,
            "declared_job_family": "interactive",
            "gpu_role": "target",
            "gt_label": "normal_mlp",
        },
        _score_cfg(),
    )
    assert low is None
    assert mlp is None


def test_idle_gate_ignores_gpu_role_family_and_gt_label():
    base = {"observed_n_procs": 0, "mean_w": 24.0}
    a = idle_gate_decision({**base, "gpu_role": "companion_idle", "gt_label": "normal_idle"}, _score_cfg())
    b = idle_gate_decision(
        {**base, "gpu_role": "target", "declared_job_family": "training", "gt_label": "swma"},
        _score_cfg(),
    )
    c = idle_gate_decision({**base, "declared_job_family": "interactive"}, _score_cfg())
    assert a == b == c
    high = {"observed_n_procs": 0, "mean_w": 200.0}
    d = idle_gate_decision({**high, "gpu_role": "companion_idle"}, _score_cfg())
    e = idle_gate_decision({**high, "gpu_role": "target", "gt_label": "normal_finetune"}, _score_cfg())
    assert d == e


def test_idle_gate_missing_n_procs_does_not_apply():
    assert idle_gate_decision({"mean_w": 24.0}, _score_cfg()) is None
    assert idle_gate_decision({"observed_n_procs": float("nan"), "mean_w": 24.0}, _score_cfg()) is None


def test_score_window_idle_overrides_conformal():
    baseline = _baseline()
    cal = _mini_calibration()
    idle = score_window(
        {"mean_w": 24.0, "declared_job_family": "training", "observed_n_procs": 0, "evidence": {}},
        baseline,
        cal,
        _score_cfg(),
    )
    assert idle["is_candidate"] is False
    assert idle["idle_status"] == "idle"
    undeclared = score_window(
        {"mean_w": 200.0, "declared_job_family": "training", "observed_n_procs": 0, "evidence": {}},
        baseline,
        cal,
        _score_cfg(alpha=0.0),
    )
    assert undeclared["is_candidate"] is True
    assert undeclared["idle_status"] == "undeclared_load"
    through = score_window(
        {
            "mean_w": 194.0,
            "declared_job_family": "interactive",
            "observed_n_procs": 1,
            "gpu_role": "companion_idle",
            "evidence": {},
        },
        baseline,
        cal,
        _score_cfg(),
    )
    assert through["idle_status"] is None


def test_zero_proc_mask_for_fit_exclusion():
    frame = pd.DataFrame({"observed_n_procs": [0, 1, 0, None], "gt_label": ["normal"] * 4})
    assert zero_proc_mask(frame).tolist() == [True, False, True, False]


def test_real_cohort_keys_are_declared_only_no_power_band():
    rows = pd.DataFrame({
        "declared_job_family": ["training"] * 40 + ["inference"] * 40,
        "declared_job_type": ["llm_finetune"] * 40 + ["online_inference"] * 40,
        "gpu_model": ["RTX4090"] * 80,
        "mean_w": np.concatenate([np.linspace(180, 220, 40), np.linspace(50, 70, 40)]),
        "swing_abs_w": np.linspace(10, 20, 80),
        "band_frac_0.1_0.7": np.linspace(0.1, 0.2, 80),
        "band_frac_0.7_2": np.linspace(0.1, 0.2, 80),
        "dominant_freq_hz": np.ones(80),
        "util_slope_w_per_pct": np.ones(80),
        "util_residual_mad_w": np.ones(80),
        "spectral_entropy": np.ones(80),
    })
    keys = ("declared_job_family", "declared_job_type", "gpu_model")
    baseline = CohortBaseline().fit(rows, cohort_keys=keys, min_windows=30)
    assert baseline.cohort_keys == keys
    assert "power_band" not in baseline.cohort_keys
    assert "full:training|llm_finetune|RTX4090" in baseline.stats
    assert "full:inference|online_inference|RTX4090" in baseline.stats
    real_cfg = yaml.safe_load(Path("config/stage1_v2_real.yaml").read_text(encoding="utf-8"))
    assert real_cfg["cohort_keys"] == list(keys)
    assert "power_band" not in real_cfg["cohort_keys"]


def test_force_zero_progress_log_missing_weight():
    row = {"declared_job_family": "training", "evidence": {}}
    cal = _mini_calibration()
    with_weight = combine_u_score(
        {"mean_w": 1.2},
        {"progress_log_missing": True, "declared_family_mismatch": True},
        row,
        cal,
        {"mondrian_min_n": 30, "evidence_weights": {}, "force_zero_progress_log_missing": False},
        include_evidence=True,
    )
    forced = combine_u_score(
        {"mean_w": 1.2},
        {"progress_log_missing": True, "declared_family_mismatch": True},
        row,
        cal,
        {"mondrian_min_n": 30, "evidence_weights": {}, "force_zero_progress_log_missing": True},
        include_evidence=True,
    )
    assert with_weight["u_score"] > forced["u_score"]
    assert "progress_log_missing" not in forced["active_evidence"]


def test_read_session_progress_raw_is_opt_in(tmp_path: Path):
    session = tmp_path / "s-0123456789abcdef"
    session.mkdir()
    (session / "session.json").write_text('{"t0_epoch": 1000.0}')
    (session / "progress.raw.jsonl").write_text(
        '{"event": "step_end", "t_epoch": 1004.0}\n'
    )
    assert read_session_progress(session) == []
    raw = read_session_progress(session, source="raw")
    assert len(raw) == 1
    assert raw[0]["t"] == 4.0


def test_real_bundle_drops_progress_log_missing():
    bundle = build_evidence_bundle(
        {
            "window_id": "w1",
            "evidence_bool": {"progress_log_missing": True, "flat_power": True},
        },
        drop_progress_log_missing=True,
    )
    evidence = bundle["unexplainedness"]["evidence"]
    assert "progress_log_missing" not in evidence
    assert evidence["flat_power"] is True


def test_segment_evidence_uses_rate_not_or():
    two_of_three = [
        {"flat_power": True, "period_mismatch": False},
        {"flat_power": True, "period_mismatch": False},
        {"flat_power": False, "period_mismatch": False},
    ]
    bools, rates = aggregate_bool_evidence(two_of_three, rate_min=0.6)
    assert rates["flat_power"] == pytest.approx(2 / 3)
    assert bools["flat_power"] is True
    one_of_three = [
        {"flat_power": True},
        {"flat_power": False},
        {"flat_power": False},
    ]
    bools_or_would_pass, rates_low = aggregate_bool_evidence(one_of_three, rate_min=0.6)
    assert rates_low["flat_power"] == pytest.approx(1 / 3)
    assert bools_or_would_pass["flat_power"] is False
    dropped, _ = aggregate_bool_evidence(
        [{"progress_log_missing": True, "flat_power": True}] * 3,
        rate_min=0.6,
        drop_keys=("progress_log_missing",),
    )
    assert "progress_log_missing" not in dropped


def test_merge_overlapping_windows_same_session_gpu():
    frame = pd.DataFrame(
        [
            {"session_id": "s1", "gpu_id": 0, "start": 0, "end": 300, "window_id": "a"},
            {"session_id": "s1", "gpu_id": 0, "start": 150, "end": 450, "window_id": "b"},
            {"session_id": "s1", "gpu_id": 0, "start": 450, "end": 750, "window_id": "c"},
            {"session_id": "s1", "gpu_id": 1, "start": 0, "end": 300, "window_id": "d"},
        ]
    )
    segments = merge_candidate_segments(frame)
    sizes = sorted(len(seg) for seg in segments)
    assert sizes == [1, 3]


def test_benign_top1_normal_verdict_without_judge():
    verdict = normal_verdict("normal_workloads")
    assert verdict.verdict == "normal"
    assert verdict.closest_match == "normal_workloads"
    with patch("pipeline.stage2_llm.judge", side_effect=AssertionError("judge must not run")):
        # Decision path used by run_pipeline: skip judge when category is benign.
        top_meta = {"category": "benign"}
        stage2 = normal_verdict("benign_periodic") if top_meta.get("category") == "benign" else judge({}, [])
        assert stage2.verdict == "normal"


def test_v2_pipeline_benign_skips_judge(tmp_path: Path):
    from pipeline.rag_analyzer import SearchResult
    from pipeline.run_pipeline import run

    n = 220
    frame = pd.DataFrame({
        "timestamp": np.arange(n, dtype=float),
        "session_id": ["s-benign"] * n,
        "gpu_id": [0] * n,
        "sample_hz": [1.0] * n,
        "power_w": np.full(n, 300.0),
        "util_gpu_pct": np.full(n, 80.0),
        "declared_job_family": ["training"] * n,
        "declared_job_type": ["llm_finetune"] * n,
    })
    telemetry = tmp_path / "one.csv"
    frame.to_csv(telemetry, index=False)
    hit = SearchResult(
        name="benign_periodic",
        score=0.9,
        text="benign",
        meta={"category": "benign"},
        filtered_by=[],
    )
    calls = {"judge": 0}

    def boom(*_args, **_kwargs):
        calls["judge"] += 1
        raise AssertionError("judge must not be called")

    with patch("rag_analyzer.SignatureRetriever.search", return_value=[hit]), patch(
        "pipeline.stage2_llm.judge", side_effect=boom
    ):
        rows = run(
            type(
                "Args",
                (),
                {
                    "telemetry": str(telemetry),
                    "baseline_mean": 120.0,
                    "sample_hz": None,
                    "window": 200,
                    "stride": 200,
                    "z_threshold": 0.0,
                    "stage1": "legacy",
                    "stage2": "v2",
                    "baseline_model": "dataset/eval/cohort_baseline_v2.json",
                    "stage1_config": "config/stage1_v2_real.yaml",
                    "calibration": "dataset/eval/stage1_v2_calibration.json",
                    "window_s": 30.0,
                    "stride_s": 15.0,
                    "progress_log_dir": None,
                    "sessions_root": None,
                    "progress_source": "visible",
                    "query_context": "off",
                    "rag_backend": "tfidf",
                    "llm_backend": "stub",
                    "llm_model": "unused",
                    "physics_validate": False,
                    "physics_timeout_s": 60.0,
                    "physics_test_system": "kundur_ieeest",
                    "out": str(tmp_path / "stage2.json"),
                    "stage1_out": str(tmp_path / "stage1.json"),
                },
            )()
        )
    assert rows
    assert rows[0]["verdict"]["verdict"] == "normal"
    assert rows[0]["verdict"]["risk"] == "정상"
    assert calls["judge"] == 0


def test_visibility_groups_and_period_parse():
    assert attack_visibility_group(gt_label="cryptojacking") == "detectable"
    assert attack_visibility_group(gt_label="swma", gt_variant="swma_jitter", period_s=2.0) == "detectable"
    assert attack_visibility_group(gt_label="swma", period_s=0.5) == "unobservable"
    assert attack_visibility_group(gt_label="ltma") == "unobservable"
    assert attack_visibility_group(gt_label="swma", gt_variant="piggyback", period_s=3.0) == "unobservable"
    assert attack_visibility_group(gt_label="normal_finetune") is None
    params = '{"period_actual_s": 2.0, "period": 4.0}'
    assert parse_period_s(params) == 2.0


def test_alpha_requires_detectable_min_recall():
    sweep = [
        {"alpha": 0.05, "cal_normal_candidate_rate": 0.04, "cal_attack_recall": 0.40, "cal_detectable_recall": 0.50},
        {"alpha": 0.10, "cal_normal_candidate_rate": 0.08, "cal_attack_recall": 0.80, "cal_detectable_recall": 0.75},
        {"alpha": 0.20, "cal_normal_candidate_rate": 0.12, "cal_attack_recall": 0.90, "cal_detectable_recall": 0.90},
    ]
    selected, failed, rule = select_alpha(sweep, {"alpha_min_detectable_recall": 0.70})
    assert failed is False
    assert selected["alpha"] == 0.10
    assert "0.70" in rule
    none, failed2, _ = select_alpha(
        [
            {"alpha": 0.05, "cal_normal_candidate_rate": 0.04, "cal_attack_recall": 0.2, "cal_detectable_recall": 0.2},
            {"alpha": 0.10, "cal_normal_candidate_rate": 0.09, "cal_attack_recall": 0.3, "cal_detectable_recall": 0.4},
        ],
        {"alpha_min_detectable_recall": 0.70},
    )
    assert failed2 is True
    assert none is None


def test_real_stage1_config_does_not_replace_synth_defaults():
    synth = load_config("config/stage1_v2.yaml")
    real = load_config("config/stage1_v2_real.yaml")
    assert synth["cohort_keys"] == ["declared_job_family", "gpu_model"]
    assert real["cohort_keys"] == ["declared_job_family", "declared_job_type", "gpu_model"]
    assert real["force_zero_progress_log_missing"] is True
    assert real["stage2"]["drop_progress_log_missing"] is True
    assert real["alpha_min_detectable_recall"] == 0.70
