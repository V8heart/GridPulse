from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.corpus_schema import checklist_satisfied, parse_corpus_v2, validate_all_corpus_docs
from pipeline.evidence_vocab import (
    ALLOWED_EVIDENCE_NAMES,
    FORBIDDEN_EVIDENCE_NAMES,
    assert_allowed_evidence_name,
    to_bool_evidence,
)
from pipeline.rag_analyzer import CORPUS_DIR


def test_forbidden_feature_range_match_not_allowed():
    assert "feature_range_match" in FORBIDDEN_EVIDENCE_NAMES
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_evidence_name("feature_range_match")


def test_new_vocab_names_present():
    assert "util_power_decoupled" in ALLOWED_EVIDENCE_NAMES
    assert "declared_family_mismatch" in ALLOWED_EVIDENCE_NAMES


def test_to_bool_evidence_uses_thresholds():
    raw = {
        "high_load_fraction": 0.95,
        "longest_high_seconds": 20.0,
        "swing_abs_w": 10.0,
        "mean_w": 300.0,
        "ramp_p95_w_per_s": 1200.0,
        "unexplained_changepoint_count": 2,
        "cross_job_sync_index": 0.9,
        "progress_log_missing": True,
        "progress_log_available": False,
        "period_match": False,
        "dominant_peak_prominence": 10.0,
        "util_residual_mad_w": 40.0,
        "declared_family_mean_abs_z": 4.0,
        "best_other_family_mean_abs_z": 1.0,
    }
    out = to_bool_evidence(raw, {"tau_peak": 8.0, "tau_sync": 0.8})
    assert set(out) <= ALLOWED_EVIDENCE_NAMES
    assert out["util_power_decoupled"] is True
    assert out["declared_family_mismatch"] is True


def test_required_any_schema_and_checklist(tmp_path):
    path = tmp_path / "crypto.md"
    path.write_text(
        "\n".join(
            [
                "---",
                "threat_id: cryptojacking",
                "category: attack",
                "grid_relevance:",
                "  mechanism: none",
                "  tier: unsupported",
                "evidence:",
                "  - name: crypto_high_or_flat",
                "    necessity: required_any",
                "    description: high or flat",
                "    any_of:",
                "      - sustained_high_load",
                "      - flat_power",
                "  - name: period_match",
                "    necessity: exclusion",
                "    description: exclude",
                "---",
                "",
                "# crypto",
            ]
        ),
        encoding="utf-8",
    )
    doc = parse_corpus_v2(path)
    ok, matched, _ = checklist_satisfied(doc, {"sustained_high_load": True, "flat_power": False})
    assert ok and "sustained_high_load" in matched
    ok2, _, _ = checklist_satisfied(doc, {"sustained_high_load": False, "flat_power": False})
    assert ok2 is False


def test_invalid_mechanism_rejected(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text(
        "\n".join(
            [
                "---",
                "threat_id: bad",
                "category: attack",
                "grid_relevance:",
                "  mechanism: impedance_resonance",
                "  tier: unsupported",
                "evidence:",
                "  - name: period_mismatch",
                "    necessity: required",
                "    description: x",
                "  - name: period_match",
                "    necessity: exclusion",
                "    description: y",
                "---",
                "",
                "# bad",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid grid_relevance.mechanism"):
        parse_corpus_v2(path)


def test_corpus_docs_validate():
    assert validate_all_corpus_docs(CORPUS_DIR) == []
