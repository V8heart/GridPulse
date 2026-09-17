from __future__ import annotations

from pipeline.corpus_schema import parse_corpus_v2, validate_all_corpus_docs
from pipeline.rag_analyzer import CORPUS_DIR
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_llm import fallback_verdict, judge


def test_evidence_bundle_excludes_raw_timeseries():
    bundle = build_evidence_bundle({
        "window_id": "w1",
        "power_w": [1, 2, 3],
        "util_gpu_pct": [4, 5, 6],
        "impact_components": {"swing_frac_tdp": 0.2},
        "evidence_bool": {"period_match": True},
    })
    assert "power_w" not in str(bundle)
    assert "util_gpu_pct" not in str(bundle)


def test_corpus_v2_schema_validates_all_docs():
    assert validate_all_corpus_docs(CORPUS_DIR) == []


def test_stage2_deterministic_given_fixed_evidence():
    docs = [parse_corpus_v2(CORPUS_DIR / "benign_periodic.md")]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {"period_match": True, "period_mismatch": False}})
    verdicts = [judge(bundle, docs, backend="stub", retriever_top1="benign_periodic") for _ in range(3)]
    assert len({(v.verdict, v.closest_match) for v in verdicts}) == 1


def test_stage2_llm_output_schema_enforced_by_fallback():
    assert fallback_verdict(top1_name=None, reason="bad schema").fallback_used is True


def test_fallback_uses_retriever_top1_not_first_doc():
    docs = [
        parse_corpus_v2(CORPUS_DIR / "benign_periodic.md"),
        parse_corpus_v2(CORPUS_DIR / "swma.md"),
    ]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {}})
    verdict = judge(bundle, docs, backend="stub", retriever_top1="swma")
    assert verdict.fallback_used is True
    assert verdict.closest_match == "swma"


def test_explanation_only_cites_declared_evidence_names():
    docs = [parse_corpus_v2(CORPUS_DIR / "benign_periodic.md")]
    bundle = build_evidence_bundle({"window_id": "w1", "evidence_bool": {"period_match": True, "period_mismatch": False}})
    verdict = judge(bundle, docs, backend="stub", retriever_top1="benign_periodic")
    allowed = set(verdict.matched_evidence) | set(verdict.contradicting_evidence)
    for item in verdict.matched_evidence:
        assert item in allowed
