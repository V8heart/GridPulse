"""Evidence-grounded Stage 2 LLM judge."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass

from pipeline.corpus_schema import CorpusDoc

VALID_VERDICTS = {"known", "partial", "unknown"}


@dataclass(frozen=True)
class Stage2Verdict:
    verdict: str
    closest_match: str | None
    confidence: float
    matched_evidence: list[str]
    contradicting_evidence: list[str]
    explanation: str
    fallback_used: bool = False


def _validate(data: dict) -> Stage2Verdict:
    missing = {"verdict", "closest_match", "confidence", "matched_evidence", "contradicting_evidence", "explanation"} - set(data)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if data["verdict"] not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {data['verdict']}")
    return Stage2Verdict(
        verdict=str(data["verdict"]),
        closest_match=data.get("closest_match"),
        confidence=float(data["confidence"]),
        matched_evidence=list(data["matched_evidence"]),
        contradicting_evidence=list(data["contradicting_evidence"]),
        explanation=str(data["explanation"]),
        fallback_used=bool(data.get("fallback_used", False)),
    )


def fallback_verdict(candidate_docs: list[CorpusDoc], reason: str = "fallback") -> Stage2Verdict:
    doc = candidate_docs[0] if candidate_docs else None
    return Stage2Verdict(
        verdict="partial" if doc else "unknown",
        closest_match=doc.name if doc else None,
        confidence=0.0,
        matched_evidence=[],
        contradicting_evidence=[],
        explanation=f"fallback_used: {reason}",
        fallback_used=True,
    )


def judge(
    evidence_bundle: dict,
    candidate_docs: list[CorpusDoc],
    *,
    backend: str = "stub",
    model: str = "gemma3:12b",
    fallback: str = "legacy",
) -> Stage2Verdict:
    if backend == "stub":
        evidence = evidence_bundle.get("unexplainedness", {}).get("evidence", {})
        for doc in candidate_docs:
            required = [item.name for item in doc.evidence if item.necessity == "required"]
            if required and all(evidence.get(name) for name in required):
                return Stage2Verdict("known", doc.name, 0.8, required, [], f"required evidence matched: {', '.join(required)}")
        return fallback_verdict(candidate_docs, "stub_no_required_match")

    prompt = {
        "instruction": "Judge only from the structured evidence names and corpus checklist.",
        "evidence_bundle": evidence_bundle,
        "candidate_docs": [
            {
                "name": doc.name,
                "threat_id": doc.threat_id,
                "category": doc.category,
                "evidence": [asdict(item) for item in doc.evidence],
                "body": doc.body[:1200],
            }
            for doc in candidate_docs
        ],
        "schema": {
            "verdict": "known|partial|unknown",
            "closest_match": "doc name or null",
            "confidence": "0.0-1.0",
            "matched_evidence": [],
            "contradicting_evidence": [],
            "explanation": "cite only evidence names",
        },
    }
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "prompt": json.dumps(prompt, ensure_ascii=False), "stream": False, "format": "json", "options": {"temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read())
        return _validate(json.loads(out["response"]))
    except Exception as exc:
        if fallback == "legacy":
            return fallback_verdict(candidate_docs, str(exc))
        raise
