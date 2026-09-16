"""Evaluate Stage 2 v2 evidence-grounded judging."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.corpus_schema import parse_corpus_v2
from pipeline.rag_analyzer import CORPUS_DIR
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_llm import judge


def evaluate(input_path: Path, backend: str = "stub") -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", [])
    docs = [parse_corpus_v2(path) for path in sorted(CORPUS_DIR.glob("*.md"))]
    total = correct = fallback = deterministic = 0
    details = []
    for row in rows:
        expected = row.get("ground_truth") or row.get("gt_label")
        bundle = build_evidence_bundle(row)
        verdicts = [judge(bundle, docs, backend=backend) for _ in range(3)]
        first = verdicts[0]
        total += 1
        correct += int(first.closest_match == expected or (str(expected).startswith("normal") and first.closest_match in {"normal_workloads", "benign_periodic"}))
        fallback += int(first.fallback_used)
        deterministic += int(len({(v.verdict, v.closest_match) for v in verdicts}) == 1)
        details.append({
            "window_id": row.get("window_id"),
            "expected": expected,
            "closest_match": first.closest_match,
            "fallback_used": first.fallback_used,
            "matched_evidence": first.matched_evidence,
        })
    return {
        "backend": backend,
        "cases": total,
        "top1_accuracy": correct / total if total else 0.0,
        "determinism_rate": deterministic / total if total else 0.0,
        "fallback_rate": fallback / total if total else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, default=ROOT / "dataset/eval/cyber_pipeline_results_v3.json")
    parser.add_argument("--backend", choices=["stub", "ollama"], default="stub")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage2_v2_report.json")
    args = parser.parse_args()
    report = evaluate(args.input, backend=args.backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
