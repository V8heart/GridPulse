"""Draft corpus v2 evidence checklists (human approval required).

Mechanism-based drafts only — not tuned on test metrics.
Supports required_any OR-groups.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.rag_analyzer import parse_front_matter

# Each draft:
# required: list of (name, description, why_required)
# required_any: list of (group_name, any_of_names, description, why_required)
# exclusion / supporting: list of (name, description)
# mechanism, tier, rationale

DRAFTS = {
    "swma": {
        "required": [
            (
                "period_mismatch",
                "전력 지배 주기가 진행 로그 step 주기와 불일치",
                "로그로 설명되는 정상 위상이면 SWMA(별도 스케줄 커널)를 주장할 수 없다",
            ),
            (
                "strong_peak",
                "규칙적 on/off가 스펙트럼에 뚜렷한 peak로 남음",
                "peak 없으면 기계적 변조 주장을 스펙트럼으로 뒷받침할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("period_match", "로그 주기와 일치하면 정상 학습 위상으로 배제")],
        "supporting": [("progress_log_missing", "위장 커널은 training progress log가 빈약한 경우가 많음")],
        "mechanism": "electromechanical_oscillation",
        "tier": "B",
        "rationale": "Bit2Watt SWMA: 별도 CUDA 커널의 규칙적 active/passive 전환 → 저주파 전기기계 외란 후보",
    },
    "ltma": {
        "required": [
            (
                "util_power_decoupled",
                "전력-사용률 Theil-Sen 잔차가 코호트 대비 큼",
                "util에 묶인 정상 학습 동력이면 LTMA식 평균유지 변조라고 주장할 수 없다",
            ),
        ],
        "required_any": [
            (
                "ltma_period_or_mismatch",
                ["period_mismatch", "unexplained_changepoint"],
                "step 구조와 어긋난 주기 또는 로그 미설명 변화점",
                "둘 다 없으면 학습 내부 변조의 시간 구조를 주장할 근거가 없다",
            ),
        ],
        "exclusion": [("flat_power", "완전 평탄 고부하는 cryptojacking류에 가깝다")],
        "supporting": [],
        "mechanism": "electromechanical_oscillation",
        "tier": "B",
        "rationale": "Bit2Watt LTMA: 학습 파이프라인 내부의 평균 유지형 부하 변조",
    },
    "cryptojacking": {
        "required": [],
        "required_any": [
            (
                "crypto_high_or_flat",
                ["sustained_high_load", "flat_power"],
                "장시간 고부하 지속 또는 고전력·저변동 평탄 프로파일",
                "고부하 지속/평탄 둘 다 없으면 채굴형 무단 부하를 주장할 수 없다",
            ),
        ],
        "exclusion": [("period_match", "정상 step 주기와 맞으면 채굴형으로 보기 어려움")],
        "supporting": [("progress_log_missing", "등록되지 않은 해시 작업은 학습 progress log가 없음")],
        "mechanism": "none",
        "tier": "unsupported",
        "rationale": "자원 남용형 지속 고부하 — 계통 공진 메커니즘이 본질이 아님",
    },
    "coordinated_multi_gpu": {
        "required": [
            (
                "cross_job_sync",
                "서로 다른 선언 job 사이 GPU 위상 동기화",
                "cross-job 동기가 없으면 협조형 다중 GPU 공격을 주장할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("period_match", "단일 job 내부 step만으로 설명되면 협조 공격이 아님")],
        "supporting": [("period_mismatch", "각 GPU 변조가 로컬 로그와 불일치할 수 있음")],
        "mechanism": "electromechanical_oscillation",
        "tier": "B",
        "rationale": "다중 GPU 협조 변조로 주입 진폭을 키우는 시나리오",
    },
    "burst_ramp_attack": {
        "required": [
            (
                "high_ramp",
                "급격한 전력 램프(p95 ramp)",
                "급램프가 없으면 burst/ramp 외란을 주장할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("flat_power", "평탄 고부하는 burst-ramp와 반대")],
        "supporting": [("unexplained_changepoint", "급변이 로그로 설명되지 않으면 더 의심")],
        "mechanism": "frequency_response_stress",
        "tier": "B",
        "rationale": "급격한 부하 램프가 주파수 응답 스트레스로 이어질 수 있는 가설",
    },
    "hidden_ml_training": {
        "required": [],
        "required_any": [
            (
                "hidden_context_break",
                ["declared_family_mismatch", "progress_log_missing"],
                "선언 코호트와 관측이 어긋나거나 정상 progress log가 없음",
                "둘 다 없으면 '숨은/비인가 학습'을 주장할 컨텍스트 근거가 없다",
            ),
        ],
        "exclusion": [("period_match", "공식 학습 로그와 주기가 맞으면 숨은 학습으로 보기 어려움")],
        "supporting": [("period_mismatch", "숨은 커널이 별도 주기를 만들면 mismatch로 관측")],
        "mechanism": "none",
        "tier": "unsupported",
        "rationale": "스케줄러 선언과 불일치하는 비인가 학습",
    },
    "llmjacking": {
        "required": [
            (
                "high_ramp",
                "요청 버스트성 전력 램프",
                "버스트 램프가 없으면 탈취된 추론 트래픽 주장을 관측으로 못 박기 어렵다",
            ),
        ],
        "required_any": [],
        "exclusion": [("flat_power", "항상 평탄한 고부하는 온라인 추론 버스트와 다름")],
        "supporting": [
            ("declared_family_mismatch", "선언 계열과 다른 관측이면 탈취/위장 정황"),
            ("progress_log_missing", "정상 training step log가 없음"),
        ],
        "mechanism": "none",
        "tier": "unsupported",
        "rationale": "무단 LLM API/엔드포인트 남용 — 추론 버스트 패턴",
    },
    "sponge_attack": {
        "required": [
            (
                "sustained_high_load",
                "비정상적으로 비싼 입력으로 고부하 지속",
                "지속 고부하가 없으면 sponge식 에너지/지연 공격을 주장할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("period_match", "정상 step 주기 설명이면 sponge보다 정상/학습 쪽")],
        "supporting": [("flat_power", "고비용 입력이 변동을 줄인 채 고전력을 유지할 수 있음")],
        "mechanism": "frequency_response_stress",
        "tier": "unsupported",
        "rationale": "에너지/지연을 노린 sponge 입력 — 지속 부하 스트레스",
    },
    "normal_workloads": {
        "required": [
            (
                "period_match",
                "전력 주기가 진행/요청 로그와 대체로 일치",
                "주기 일치가 없으면 '설명 가능한 정상 워크로드' 문서를 주장할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("period_mismatch", "강한 주기 불일치는 정상 문서 배제 근거")],
        "supporting": [("progress_log_available", "정상 워크로드는 progress/request 로그가 있는 경우가 많음")],
        "mechanism": "none",
        "tier": "unsupported",
        "rationale": "오탐 방지용 정상 프로파일 집합",
    },
    "benign_periodic": {
        "required": [
            (
                "period_match",
                "규칙적 전력이 정상 step/동기화 주기로 설명됨",
                "period_match가 없으면 benign 주기성이라고 주장할 수 없다",
            ),
        ],
        "required_any": [],
        "exclusion": [("period_mismatch", "주기 불일치가 있으면 정상이 아니다")],
        "supporting": [("progress_log_available", "DDP/체크포인트 등 정상 주기성의 설명 로그")],
        "mechanism": "none",
        "tier": "unsupported",
        "rationale": "정상 분산학습의 규칙적 출렁임",
    },
}


def _yaml_scalar(name: str, necessity: str, description: str) -> list[str]:
    return [
        f"  - name: {name}",
        f"    necessity: {necessity}",
        f'    description: "{description}"',
    ]


def _yaml_any(group: str, names: list[str], description: str) -> list[str]:
    lines = [
        f"  - name: {group}",
        "    necessity: required_any",
        f'    description: "{description}"',
        "    any_of:",
    ]
    lines.extend(f"      - {n}" for n in names)
    return lines


def apply_drafts(corpus_dir: Path) -> None:
    for path in sorted(corpus_dir.glob("*.md")):
        if path.stem not in DRAFTS:
            continue
        draft = DRAFTS[path.stem]
        old_meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        category = old_meta.get("category", "cyber_physical_attack")
        lines = [
            "---",
            f"threat_id: {old_meta.get('threat_id', path.stem)}",
            f"category: {category}",
            "mitre_technique: null",
            "evidence:",
        ]
        for name, desc, _why in draft["required"]:
            lines.extend(_yaml_scalar(name, "required", desc))
        for group, names, desc, _why in draft["required_any"]:
            lines.extend(_yaml_any(group, names, desc))
        for name, desc in draft["exclusion"]:
            lines.extend(_yaml_scalar(name, "exclusion", desc))
        for name, desc in draft["supporting"]:
            lines.extend(_yaml_scalar(name, "supporting", desc))
        lines.extend(
            [
                "benign_lookalikes: []",
                "grid_relevance:",
                f"  mechanism: {draft['mechanism']}",
                f"  tier: {draft['tier']}",
                f'  notes: "DRAFT pending human approval. {draft["rationale"]}"',
                "observability:",
                "  min_sample_hz: 1",
                '  notes: "NVML 1초 평균을 전제로 해석"',
                "thresholds_provenance: dataset/eval/evidence_thresholds.json",
                "---",
                "",
            ]
        )
        note = f"\n## Evidence 근거 (초안)\n{draft['rationale']}\n"
        if "## Evidence 근거" not in body:
            body = body.rstrip() + "\n" + note
        path.write_text("\n".join(lines) + body.lstrip(), encoding="utf-8")


def write_review_table(out: Path) -> None:
    rows = [
        "| doc | required / required_any | exclusion | supporting | mechanism | tier | 이 증거가 없으면 주장 불가인 이유 |",
        "|-----|-------------------------|-----------|------------|-----------|------|----------------------------------|",
    ]
    for name, draft in DRAFTS.items():
        req_bits = [n for n, _, _ in draft["required"]]
        for group, names, _, why in draft["required_any"]:
            req_bits.append(f"{group}=({'|'.join(names)})")
        whys = [why for *_, why in draft["required"]] + [why for *_, why in draft["required_any"]]
        why_text = " / ".join(whys) if whys else "-"
        rows.append(
            f"| {name} | {', '.join(req_bits) or '-'} | "
            f"{', '.join(n for n, _ in draft['exclusion'])} | "
            f"{', '.join(n for n, _ in draft['supporting']) or '-'} | "
            f"{draft['mechanism']} | {draft['tier']} | {why_text} |"
        )
    out.write_text(
        "# Corpus evidence draft — human review (v2)\n\n"
        "Test metrics were **not** used to choose these rows.\n\n"
        "Mechanisms: `electromechanical_oscillation` | `frequency_response_stress` | "
        "`converter_resonance` | `none`.\n\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    corpus = ROOT / "corpus"
    apply_drafts(corpus)
    write_review_table(ROOT / "dataset/eval/corpus_evidence_draft_review.md")
    from pipeline.corpus_schema import validate_all_corpus_docs

    errors = validate_all_corpus_docs(corpus)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"updated {len(DRAFTS)} corpus docs; review table written")


if __name__ == "__main__":
    main()
