"""Migrate corpus markdown files to evidence-checklist front-matter."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.rag_analyzer import parse_front_matter

DEFAULT_EVIDENCE = {
    "swma": [
        ("period_mismatch", "supporting", "전력 지배 주파수가 선언된 학습 진행 로그와 불일치"),
        ("feature_range_match", "supporting", "train split에서 적합된 SWMA feature range에 근접"),
        ("progress_log_present_training", "exclusion", "정상 학습 진행 로그가 전력 주기를 설명하면 배제 방향"),
    ],
    "ltma": [
        ("util_residual_anomaly", "supporting", "전력-사용률 결합 잔차가 코호트 기준에서 이탈"),
        ("period_mismatch", "supporting", "학습 step 구조로 설명되지 않는 변조 주기"),
    ],
    "cryptojacking": [
        ("persistent_high_load", "supporting", "평탄하고 높은 부하가 장시간 지속"),
        ("progress_log_present_training", "exclusion", "정상 학습 로그가 지속 고부하를 설명하면 배제 방향"),
    ],
    "normal_workloads": [
        ("period_match", "supporting", "전력 주기가 정상 진행 로그와 일치"),
        ("explained_changepoint", "supporting", "변화점이 체크포인트/평가/요청 이벤트로 설명됨"),
    ],
    "benign_periodic": [
        ("period_match", "required", "강한 주기성이 정상 job 진행 로그로 설명됨"),
        ("declared_context_consistent", "supporting", "선언 작업과 관측 패턴이 일치"),
    ],
}


def evidence_for(name: str) -> list[tuple[str, str, str]]:
    return DEFAULT_EVIDENCE.get(name, [
        ("feature_range_match", "supporting", "train split에서 적합된 feature range와 근접"),
        ("context_inconsistency", "supporting", "선언 컨텍스트와 관측 evidence가 불일치"),
    ])


def migrate_file(path: Path) -> None:
    old_meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    category = old_meta.get("category", "attack")
    if category == "attack":
        category = "cyber_physical_attack"
    meta_lines = [
        "---",
        f"threat_id: {old_meta.get('threat_id', path.stem)}",
        f"category: {category}",
        "mitre_technique: null",
        "evidence:",
    ]
    for name, necessity, description in evidence_for(path.stem):
        meta_lines.extend([
            f"  - name: {name}",
            f"    necessity: {necessity}",
            f"    description: \"{description}\"",
        ])
    meta_lines.extend([
        "benign_lookalikes: []",
        "grid_relevance:",
        "  mechanism: unsupported",
        "  tier: unsupported",
        "  notes: \"# TODO(review): public test-system evidence only; do not overstate real-grid impact\"",
        "observability:",
        "  min_sample_hz: 1",
        "  notes: \"NVML 1초 평균을 전제로 해석\"",
        "thresholds_provenance: dataset/eval/corpus_feature_ranges.json",
        "---",
        "",
    ])
    path.write_text("\n".join(meta_lines) + body.lstrip(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    for path in sorted(args.corpus_dir.glob("*.md")):
        migrate_file(path)
    print(f"migrated {len(list(args.corpus_dir.glob('*.md')))} corpus docs")


if __name__ == "__main__":
    main()
