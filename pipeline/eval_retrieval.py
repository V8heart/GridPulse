"""D7 RAG positive/open-set 평가셋 생성과 retrieval 지표 계산."""
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

from pipeline.features import compute_window_features, features_to_description
from pipeline.label_to_doc import expected_document
from pipeline.rag_analyzer import SignatureRetriever
from pipeline.run_pipeline import context_to_query, infer_sample_hz

LEGACY_DOCS = {"swma", "ltma", "cryptojacking", "normal_workloads"}
DOC_SANITY_CASES = [
    {
        "case_id": "doc-hidden-ml-training",
        "kind": "positive",
        "label": "hidden_ml_training",
        "expected_document": "hidden_ml_training",
        "description": (
            "평균 전력이 승인된 학습 작업처럼 높게 유지되고 약한 주기성이 있으나 "
            "스케줄러에 등록되지 않은 비인가 사용자 작업이다."
        ),
        "features": {
            "mean_w": 175.0,
            "swing_ratio": 1.2,
            "periodicity_strength": 0.88,
            "duty_regularity": 0.25,
            "ramp_max_w_per_s": 1700.0,
            "high_load_fraction": 0.4,
            "longest_high_seconds": 5.0,
        },
    },
    {
        "case_id": "doc-llmjacking",
        "kind": "positive",
        "label": "llmjacking",
        "expected_document": "llmjacking",
        "description": (
            "외부 요청 도착이 불규칙해 전력 변동폭과 순간 램프가 크고 "
            "승인되지 않은 API 자격증명 사용 맥락이 있다."
        ),
        "features": {
            "mean_w": 150.0,
            "swing_ratio": 1.35,
            "periodicity_strength": 0.85,
            "duty_regularity": 0.3,
            "ramp_max_w_per_s": 1650.0,
            "high_load_fraction": 0.2,
            "longest_high_seconds": 1.0,
        },
    },
    {
        "case_id": "doc-sponge-attack",
        "kind": "positive",
        "label": "sponge_attack",
        "expected_document": "sponge_attack",
        "description": (
            "동일 모델 추론인데 입력 때문에 에너지와 지연이 증폭되어 높은 부하가 "
            "평평하게 지속되고 요청 주입 간격은 공격자가 조절한다."
        ),
        "features": {
            "mean_w": 285.0,
            "swing_ratio": 0.025,
            "periodicity_strength": 0.8,
            "duty_regularity": 0.0,
            "ramp_max_w_per_s": 20.0,
            "high_load_fraction": 1.0,
            "longest_high_seconds": 20.0,
        },
    },
    {
        "case_id": "doc-coordinated-multi-gpu",
        "kind": "positive",
        "label": "coordinated_multi_gpu",
        "expected_document": "coordinated_multi_gpu",
        "description": (
            "여러 GPU의 전력 변화가 같은 위상으로 동기화되어 합산 부하 변조가 "
            "커지며 서로 다른 사용자 작업 사이의 위상 일치가 관측된다."
        ),
        "features": {
            "mean_w": 184.0,
            "swing_ratio": 1.65,
            "periodicity_strength": 0.9,
            "duty_regularity": 0.9,
            "ramp_max_w_per_s": 3000.0,
            "high_load_fraction": 0.5,
            "longest_high_seconds": 1.0,
            "multi_gpu_sync_index": 0.95,
        },
    },
    {
        "case_id": "doc-burst-ramp-attack",
        "kind": "positive",
        "label": "burst_ramp_attack",
        "expected_document": "burst_ramp_attack",
        "description": (
            "스케줄러 이벤트와 맞지 않는 급격한 부하 상승과 하강이 짧은 시간에 "
            "발생하고 평균만으로는 놓치기 쉬운 큰 램프가 핵심이다."
        ),
        "features": {
            "mean_w": 160.0,
            "swing_ratio": 1.25,
            "periodicity_strength": 0.86,
            "duty_regularity": 0.2,
            "ramp_max_w_per_s": 1400.0,
            "high_load_fraction": 0.1,
            "longest_high_seconds": 0.5,
        },
    },
    {
        "case_id": "doc-benign-periodic",
        "kind": "positive",
        "label": "benign_periodic",
        "expected_document": "benign_periodic",
        "description": (
            "정상 분산학습 또는 체크포인트 때문에 주기성과 전력 변동폭이 크지만 "
            "동일 job_id와 승인된 사용자 맥락으로 설명된다."
        ),
        "features": {
            "mean_w": 178.0,
            "swing_ratio": 0.8,
            "periodicity_strength": 0.9,
            "duty_regularity": 0.1,
            "ramp_max_w_per_s": 1000.0,
            "high_load_fraction": 0.3,
            "longest_high_seconds": 3.0,
        },
    },
]


def build_cases(
    telemetry: Path,
    session_ids: set[str] | None = None,
    *,
    include_doc_sanity: bool = True,
    context_mode: str = "no_context",
) -> list[dict]:
    df = pd.read_csv(telemetry, low_memory=False)
    if session_ids is not None:
        df = df[df["session_id"].astype(str).isin(session_ids)]
    cases = []
    for (session_id, gpu_id), group in df.groupby(["session_id", "gpu_id"], sort=False):
        hz = infer_sample_hz(group)
        feats = compute_window_features(
            group["power_w"].to_numpy(),
            group["util_gpu_pct"].to_numpy() if "util_gpu_pct" in group else None,
            sample_hz=hz,
        )
        description = features_to_description(feats, baseline_mean_w=120)
        label_col = "gt_label" if "gt_label" in group else "label"
        label = str(group[label_col].iloc[0])
        if context_mode == "declared":
            context = {
                key: str(group[key].iloc[0])
                for key in ("declared_user", "declared_job_type", "declared_job_family", "declared_gres")
                if key in group
            }
            query = f"{description} {context_to_query(context)}".strip()
        elif context_mode == "oracle_label":
            query = f"{description} ground_truth_label={label} attack_class={label}".strip()
        else:
            query = description
        cases.append({
            "case_id": f"{session_id}-gpu{gpu_id}",
            "kind": "positive",
            "label": label,
            "expected_document": expected_document(label),
            "description": description,
            "query": query,
            "features": feats,
            "context_mode": context_mode,
        })

    rng = np.random.default_rng(999)
    unknown_signals = {
        "unknown_single_impulse": np.r_[np.full(499, 120.0), 350.0, np.full(500, 120.0)],
        "unknown_slow_drift": np.linspace(70, 180, 1000),
        "unknown_random_walk": 120 + np.cumsum(rng.normal(0, 1.5, 1000)),
    }
    for name, signal in unknown_signals.items():
        feats = compute_window_features(signal, sample_hz=10)
        description = features_to_description(feats, baseline_mean_w=120)
        cases.append({
            "case_id": name,
            "kind": "unknown",
            "label": "unknown",
            "expected_document": None,
            "description": description,
            "query": description,
            "features": feats,
            "context_mode": context_mode,
        })
    return cases + (DOC_SANITY_CASES if include_doc_sanity else [])


def evaluate(
    cases: list[dict],
    backend: str = "tfidf",
    unknown_threshold: float = 0.28,
    include_docs: set[str] | None = None,
    retriever: SignatureRetriever | None = None,
) -> dict:
    retriever = retriever or SignatureRetriever(backend=backend, include_docs=include_docs)
    positive = correct = 0
    legacy_positive = legacy_correct = 0
    unknown_scores = []
    details = []
    for case in cases:
        query = case.get("query", case["description"])
        result = retriever.search(query, top_k=3, features=case.get("features"))
        if result:
            top_name, top_score, _ = result[0]
        else:
            top_name, top_score = "unknown", 0.0
        if case["kind"] == "positive":
            expected = case["expected_document"]
            official = not str(case["case_id"]).startswith("doc-")
            if official and (include_docs is None or expected in include_docs):
                positive += 1
                correct += int(top_name == expected)
            if official and expected in LEGACY_DOCS:
                legacy_positive += 1
                legacy_correct += int(top_name == expected)
        else:
            unknown_scores.append(top_score)
        details.append({
            "case_id": case["case_id"],
            "expected": case["expected_document"],
            "top1": top_name,
            "top1_score": top_score,
        })
    return {
        "backend": backend,
        "positive_cases": positive,
        "top1_accuracy": correct / positive if positive else 0.0,
        "legacy_positive_cases": legacy_positive,
        "legacy_top1_accuracy": legacy_correct / legacy_positive if legacy_positive else 0.0,
        "unknown_cases": len(unknown_scores),
        "unknown_max_similarity": max(unknown_scores, default=None),
        "unknown_threshold": unknown_threshold,
        "unknown_rejection_rate": (
            sum(score < unknown_threshold for score in unknown_scores) / len(unknown_scores)
            if unknown_scores else 0.0
        ),
        "details": details,
    }


def evaluate_scaling(cases: list[dict], backend: str, unknown_threshold: float) -> dict:
    legacy = evaluate(cases, backend=backend, unknown_threshold=unknown_threshold, include_docs=LEGACY_DOCS)
    expanded = evaluate(cases, backend=backend, unknown_threshold=unknown_threshold)
    return {
        "backend": backend,
        "legacy_4_docs": {k: v for k, v in legacy.items() if k != "details"},
        "expanded_10_docs": {k: v for k, v in expanded.items() if k != "details"},
        "legacy_accuracy_regressed": (
            expanded["legacy_top1_accuracy"] < legacy["legacy_top1_accuracy"]
        ),
    }


def write_token_stats(corpus_dir: Path, output: Path, model_name: str = "all-MiniLM-L6-v2") -> dict:
    stats = {
        "model_name": model_name,
        "max_model_tokens": 256,
        "documents": {},
        "notes": (
            "If documents exceed 256 tokens in SBERT mode, either keep documents shorter "
            "or evaluate a multilingual embedding model. This run does not change models."
        ),
    }
    try:
        from sentence_transformers import SentenceTransformer
        from pipeline.rag_analyzer import parse_front_matter

        model = SentenceTransformer(model_name)
        tokenizer = model.tokenizer
        for path in sorted(corpus_dir.glob("*.md")):
            _, body = parse_front_matter(path.read_text(encoding="utf-8"))
            count = len(tokenizer.encode(body, add_special_tokens=True, truncation=False))
            stats["documents"][path.stem] = {
                "token_count": int(count),
                "exceeds_256": bool(count > 256),
            }
    except Exception as exc:
        stats["error"] = str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", type=Path, default=ROOT / "dataset/synthetic/all_v3.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "dataset/synthetic/split_manifest.json")
    parser.add_argument("--split", choices=["train", "cal", "test", "all"], default="test")
    parser.add_argument("--context-mode", choices=["no_context", "declared", "oracle_label"], default="no_context")
    parser.add_argument("--write-leakage-table", action="store_true")
    parser.add_argument("--leakage-out", type=Path, default=ROOT / "dataset/eval/leakage_before_after.json")
    parser.add_argument("--cases-out", type=Path, default=ROOT / "dataset/eval/retrieval_eval.jsonl")
    parser.add_argument("--report-out", type=Path, default=ROOT / "dataset/eval/retrieval_report.json")
    parser.add_argument("--scaling-out", type=Path, default=ROOT / "dataset/eval/retrieval_scaling.json")
    parser.add_argument("--token-stats-out", type=Path, default=ROOT / "dataset/eval/corpus_token_stats.json")
    parser.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--unknown-threshold", type=float, default=0.28)
    parser.add_argument("--write-scaling", action="store_true")
    args = parser.parse_args()
    session_ids = None
    if args.split_manifest and args.split != "all":
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        session_ids = set(map(str, manifest[args.split]))

    if args.write_leakage_table:
        from pipeline.fit_corpus_ranges import fit_ranges

        modes = {}
        for mode in ("no_context", "declared", "oracle_label"):
            cases = build_cases(args.telemetry, session_ids=session_ids, include_doc_sanity=False, context_mode=mode)
            report = evaluate(cases, args.backend, args.unknown_threshold)
            scaling = evaluate_scaling(cases, args.backend, args.unknown_threshold)
            entry = {
                "top1_accuracy": report["top1_accuracy"],
                "legacy_4_docs": scaling["legacy_4_docs"],
                "expanded_10_docs": scaling["expanded_10_docs"],
            }
            if mode == "oracle_label":
                entry["label"] = "누설 포함 상한"
            modes[mode] = entry

        # legacy_full_leak: oracle + hard ranges fitted on TEST (intentional upper bound)
        leak_ranges = ROOT / "dataset/eval/corpus_feature_ranges_testfit_leak.json"
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        leak_split = {
            "seed": manifest.get("seed"),
            "split_ratios": manifest.get("split_ratios"),
            "data": manifest.get("data"),
            "train": list(manifest["test"]),
            "cal": [],
            "test": list(manifest["test"]),
            "unseen_param_holdout": manifest.get("unseen_param_holdout", {}),
        }
        leak_manifest_path = ROOT / "dataset/eval/_tmp_testfit_split.json"
        leak_manifest_path.write_text(json.dumps(leak_split, ensure_ascii=False, indent=2), encoding="utf-8")
        report_leak_fit = fit_ranges(args.telemetry, leak_manifest_path)
        report_leak_fit["fit_split"] = "test_intentional_leak"
        leak_ranges.write_text(json.dumps(report_leak_fit, ensure_ascii=False, indent=2), encoding="utf-8")

        leak_cases = build_cases(
            args.telemetry, session_ids=session_ids, include_doc_sanity=False, context_mode="oracle_label"
        )
        leak_retriever = SignatureRetriever(
            backend=args.backend,
            range_path=leak_ranges,
            range_filter="hard",
        )
        leak_eval = evaluate(leak_cases, args.backend, args.unknown_threshold, retriever=leak_retriever)
        modes["legacy_full_leak"] = {
            "label": "누설 포함 상한 (oracle_label + test-fit hard range)",
            "top1_accuracy": leak_eval["top1_accuracy"],
            "positive_cases": leak_eval["positive_cases"],
            "note": "Ranges fitted on test split intentionally; not for official claims.",
        }

        table = {
            "split": args.split,
            "backend": args.backend,
            "modes": modes,
            "note": "oracle_label / legacy_full_leak은 공식 수치로 사용하지 않음",
        }
        args.leakage_out.parent.mkdir(parents=True, exist_ok=True)
        args.leakage_out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        # markdown companion
        md_lines = [
            "# Leakage before/after",
            "",
            "| mode | top1 | note |",
            "|------|------|------|",
        ]
        for key, val in modes.items():
            md_lines.append(
                f"| {key} | {val.get('top1_accuracy')} | {val.get('label') or val.get('note') or ''} |"
            )
        args.leakage_out.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(json.dumps(table, ensure_ascii=False, indent=2))
        return

    cases = build_cases(
        args.telemetry,
        session_ids=session_ids,
        include_doc_sanity=True,
        context_mode=args.context_mode,
    )
    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    report = evaluate(cases, args.backend, args.unknown_threshold)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.write_scaling:
        scaling = {}
        for backend in ("tfidf", "sbert"):
            try:
                scaling[backend] = evaluate_scaling(cases, backend, args.unknown_threshold)
            except Exception as exc:
                scaling[backend] = {"error": str(exc)}
        args.scaling_out.write_text(json.dumps(scaling, ensure_ascii=False, indent=2), encoding="utf-8")
        write_token_stats(ROOT / "corpus", args.token_stats_out)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
