"""Corpus v2 schema for evidence-grounded Stage 2."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.evidence_vocab import ALLOWED_EVIDENCE_NAMES, FORBIDDEN_EVIDENCE_NAMES

ALLOWED_MECHANISMS = frozenset(
    {
        "electromechanical_oscillation",
        "frequency_response_stress",
        "converter_resonance",
        "none",
    }
)
ALLOWED_TIERS = frozenset({"A", "B", "C", "unsupported"})
ALLOWED_NECESSITY = frozenset({"required", "required_any", "supporting", "exclusion"})


@dataclass(frozen=True)
class EvidenceRule:
    name: str
    necessity: str
    description: str
    any_of: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CorpusDoc:
    name: str
    threat_id: str
    category: str
    evidence: list[EvidenceRule]
    body: str
    meta: dict[str, Any]


def _load_yaml_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    raw, body = text.split("---\n", 2)[1:]
    try:
        import yaml

        return yaml.safe_load(raw) or {}, body.lstrip()
    except Exception:
        from pipeline.rag_analyzer import parse_front_matter

        return parse_front_matter(text)


def _validate_name(name: str, errors: list[str]) -> None:
    if name in FORBIDDEN_EVIDENCE_NAMES:
        errors.append(f"forbidden evidence name: {name}")
    elif name not in ALLOWED_EVIDENCE_NAMES:
        errors.append(f"unknown evidence name: {name}")


def parse_corpus_v2(path: str | Path, *, enforce_counts: bool = True) -> CorpusDoc:
    target = Path(path)
    meta, body = _load_yaml_front_matter(target.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not meta.get("threat_id"):
        errors.append("missing threat_id")
    if not meta.get("category"):
        errors.append("missing category")

    grid = meta.get("grid_relevance") or {}
    if isinstance(grid, dict):
        mechanism = grid.get("mechanism")
        tier = grid.get("tier")
        if mechanism is not None and mechanism not in ALLOWED_MECHANISMS:
            errors.append(f"invalid grid_relevance.mechanism: {mechanism}")
        if tier is not None and tier not in ALLOWED_TIERS:
            errors.append(f"invalid grid_relevance.tier: {tier}")

    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("missing evidence list")
        raw_evidence = []
    evidence: list[EvidenceRule] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("evidence item must be a mapping")
            continue
        necessity = str(item.get("necessity", ""))
        if necessity not in ALLOWED_NECESSITY:
            errors.append(f"invalid necessity for {item.get('name')}: {necessity}")
        any_of_raw = item.get("any_of") or []
        any_of = tuple(str(x) for x in any_of_raw) if isinstance(any_of_raw, list) else ()
        if necessity == "required_any":
            if len(any_of) < 2:
                errors.append(f"required_any needs any_of with >=2 names: {item.get('name')}")
            for member in any_of:
                _validate_name(member, errors)
            name = str(item.get("name") or ("any__" + "__".join(any_of)))
        else:
            name = str(item.get("name", ""))
            _validate_name(name, errors)
            if any_of:
                errors.append(f"any_of only allowed with required_any: {name}")
        evidence.append(
            EvidenceRule(
                name=name,
                necessity=necessity,
                description=str(item.get("description", "")),
                any_of=any_of,
            )
        )

    if enforce_counts:
        n_required = sum(1 for item in evidence if item.necessity == "required")
        n_required_any = sum(1 for item in evidence if item.necessity == "required_any")
        n_exclusion = sum(1 for item in evidence if item.necessity == "exclusion")
        if n_required + n_required_any < 1:
            errors.append("need required>=1 or required_any>=1")
        if n_exclusion < 1:
            errors.append("need exclusion >= 1")

    if errors:
        raise ValueError(f"{target}: {', '.join(errors)}")
    return CorpusDoc(
        name=target.stem,
        threat_id=str(meta["threat_id"]),
        category=str(meta["category"]),
        evidence=evidence,
        body=body,
        meta=meta,
    )


def checklist_satisfied(doc: CorpusDoc, evidence: dict) -> tuple[bool, list[str], list[str]]:
    """Return (ok, matched_names, contradicting_exclusion_names)."""
    matched: list[str] = []
    contradicting = [item.name for item in doc.evidence if item.necessity == "exclusion" and evidence.get(item.name)]
    if contradicting:
        return False, matched, contradicting

    and_required = [item for item in doc.evidence if item.necessity == "required"]
    any_groups = [item for item in doc.evidence if item.necessity == "required_any"]
    if not and_required and not any_groups:
        return False, matched, contradicting

    for item in and_required:
        if not evidence.get(item.name):
            return False, matched, contradicting
        matched.append(item.name)

    for group in any_groups:
        hits = [name for name in group.any_of if evidence.get(name)]
        if not hits:
            return False, matched, contradicting
        matched.extend(hits)
    return True, matched, contradicting


def validate_all_corpus_docs(corpus_dir: str | Path, *, enforce_counts: bool = True) -> list[str]:
    errors = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        try:
            parse_corpus_v2(path, enforce_counts=enforce_counts)
        except Exception as exc:
            errors.append(str(exc))
    return errors
