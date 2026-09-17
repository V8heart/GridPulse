"""Evaluate Stage 2 v2 evidence-grounded judging on the test split only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.corpus_schema import parse_corpus_v2
from pipeline.label_to_doc import expected_document
from pipeline.rag_analyzer import CORPUS_DIR, SignatureRetriever
from pipeline.stage2_evidence import build_evidence_bundle
from pipeline.stage2_llm import judge


def _evidence_pr(matched: list[str], required: list[str]) -> tuple[float | None, float | None]:
    m, r = set(matched), set(required)
    if not m and not r:
        return None, None
    precision = len(m & r) / len(m) if m else None
    recall = len(m & r) / len(r) if r else None
    return precision, recall


def evaluate(
    input_path: Path,
    *,
    backend: str = "stub",
    split_manifest: Path | None = None,
    split: str = "test",
    top_k: int = 5,
) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", [])
    n_before = len(rows)
    sessions_before = {str(r.get("session_id")) for r in rows if r.get("session_id") is not None}

    if split_manifest and split != "all":
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        allowed = set(map(str, manifest[split]))
        rows = [r for r in rows if str(r.get("session_id")) in allowed]
        if not rows:
            raise ValueError(f"no rows left after filtering to split={split}")

    sessions_after = {str(r.get("session_id")) for r in rows if r.get("session_id") is not None}
    holdout_ids = set()
    if split_manifest:
        holdout_ids = set(
            map(str, json.loads(split_manifest.read_text(encoding="utf-8"))
                .get("unseen_param_holdout", {})
                .get("swma_period_2_8_3_2", []))
        )

    all_docs = {doc.name: doc for doc in (parse_corpus_v2(p) for p in sorted(CORPUS_DIR.glob("*.md")))}
    retriever = SignatureRetriever(backend="tfidf")

    total = v2_correct = fb_correct = fb_n = deterministic = 0
    ev_p, ev_r, ev_n = [], [], 0
    ev_p_all, ev_r_all = [], []
    holdout_total = holdout_correct = 0
    details = []

    for row in rows:
        gt = row.get("ground_truth") or row.get("gt_label")
        expected = expected_document(gt)
        desc = row.get("description") or ""
        feats = row.get("features")
        retrieved = retriever.search(desc, top_k=top_k, features=feats)
        top1 = retrieved[0].name if retrieved else None
        docs = [all_docs[item.name] for item in retrieved if item.name in all_docs]
        bundle = build_evidence_bundle(row)
        verdicts = [
            judge(bundle, docs, backend=backend, retriever_top1=top1)
            for _ in range(3)
        ]
        first = verdicts[0]
        total += 1
        ok = first.closest_match == expected
        if first.fallback_used:
            fb_n += 1
            fb_correct += int(ok)
        else:
            v2_correct += int(ok)

        required = []
        if expected and expected in all_docs:
            required = [e.name for e in all_docs[expected].evidence if e.necessity == "required"]
        p, r = _evidence_pr(first.matched_evidence, required)
        if p is not None and r is not None:
            ev_p_all.append(p)
            ev_r_all.append(r)
            if not first.fallback_used and ok:
                ev_p.append(p)
                ev_r.append(r)
                ev_n += 1

        if str(row.get("session_id")) in holdout_ids:
            holdout_total += 1
            holdout_correct += int(ok)

        deterministic += int(len({(v.verdict, v.closest_match) for v in verdicts}) == 1)
        details.append({
            "window_id": row.get("window_id"),
            "session_id": row.get("session_id"),
            "gt_label": gt,
            "expected_doc": expected,
            "closest_match": first.closest_match,
            "fallback_used": first.fallback_used,
            "matched_evidence": first.matched_evidence,
            "retriever_top1": top1,
        })

    non_fb = total - fb_n
    report = {
        "backend": backend,
        "split": split,
        "n_rows_before_filter": n_before,
        "n_sessions_before_filter": len(sessions_before),
        "n_rows_after_filter": total,
        "n_sessions_after_filter": len(sessions_after),
        "cases": total,
        "v2_accuracy": (v2_correct / non_fb) if non_fb else None,
        "fallback_accuracy": (fb_correct / fb_n) if fb_n else None,
        "overall_accuracy": (v2_correct + fb_correct) / total if total else 0.0,
        "determinism_rate": deterministic / total if total else 0.0,
        "fallback_rate": fb_n / total if total else 0.0,
        "evidence_precision_excl_fallback": sum(ev_p) / len(ev_p) if ev_p else None,
        "evidence_recall_excl_fallback": sum(ev_r) / len(ev_r) if ev_r else None,
        "evidence_precision_incl_fallback": sum(ev_p_all) / len(ev_p_all) if ev_p_all else None,
        "evidence_recall_incl_fallback": sum(ev_r_all) / len(ev_r_all) if ev_r_all else None,
        "holdout": {
            "sessions": sorted(holdout_ids),
            "cases": holdout_total,
            "overall_accuracy": holdout_correct / holdout_total if holdout_total else None,
        },
        "details": details,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, default=ROOT / "dataset/eval/cyber_pipeline_results_v3.json")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--split", choices=["train", "cal", "test", "all"], default="test")
    parser.add_argument("--backend", choices=["stub", "ollama"], default="stub")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "dataset/eval/stage2_v2_report.json")
    args = parser.parse_args()
    report = evaluate(
        args.input,
        backend=args.backend,
        split_manifest=args.split_manifest,
        split=args.split,
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
