"""Corpus v2 schema for evidence-grounded Stage 2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvidenceRule:
    name: str
    necessity: str
    description: str


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


def parse_corpus_v2(path: str | Path) -> CorpusDoc:
    target = Path(path)
    meta, body = _load_yaml_front_matter(target.read_text(encoding="utf-8"))
    errors = []
    if not meta.get("threat_id"):
        errors.append("missing threat_id")
    if not meta.get("category"):
        errors.append("missing category")
    raw_evidence = meta.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("missing evidence list")
        raw_evidence = []
    evidence = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("evidence item must be a mapping")
            continue
        if item.get("necessity") not in {"required", "supporting", "exclusion"}:
            errors.append(f"invalid necessity for {item.get('name')}")
        evidence.append(EvidenceRule(
            name=str(item.get("name", "")),
            necessity=str(item.get("necessity", "")),
            description=str(item.get("description", "")),
        ))
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


def validate_all_corpus_docs(corpus_dir: str | Path) -> list[str]:
    errors = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        try:
            parse_corpus_v2(path)
        except Exception as exc:
            errors.append(str(exc))
    return errors
