from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dataset.run_capture import main as capture_main
from pipeline.rag_analyzer import SignatureRetriever, analyze_with_llm, parse_front_matter


def _write_doc(path: Path, name: str, *, category: str, swing: list[float]) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f"threat_id: {name}",
                f"category: {category}",
                f"swing_ratio: {json.dumps(swing)}",
                "periodicity_strength: [0.8, 1.0]",
                "duty_regularity: [0.8, 1.0]",
                "mean_power_level: any",
                "ramp_level: any",
                "multi_gpu_sync: optional",
                "min_duration_s: null",
                "---",
                "",
                f"# {name}",
                "high low 전환이 규칙적인 문서",
            ]
        ),
        encoding="utf-8",
    )


def test_parse_front_matter_excludes_metadata_from_body():
    meta, body = parse_front_matter("---\nthreat_id: T1\nswing_ratio: [1.0, 2.0]\n---\n# Body")
    assert meta["threat_id"] == "T1"
    assert meta["swing_ratio"] == [1.0, 2.0]
    assert "threat_id" not in body
    assert body.startswith("# Body")


def test_retriever_filters_by_feature_ranges(tmp_path):
    _write_doc(tmp_path / "matching.md", "matching", category="attack", swing=[1.0, 2.0])
    _write_doc(tmp_path / "excluded.md", "excluded", category="attack", swing=[0.0, 0.2])
    retriever = SignatureRetriever(corpus_dir=tmp_path, backend="tfidf", range_filter="hard")
    results = retriever.search(
        "규칙적인 high low 전환",
        features={"swing_ratio": 1.5, "periodicity_strength": 0.9, "duty_regularity": 0.9},
    )
    assert [item.name for item in results] == ["matching"]
    assert retriever.last_filter_report["excluded"] == ["swing_ratio"]


def test_empty_filtered_retrieval_becomes_unknown(tmp_path):
    _write_doc(tmp_path / "excluded.md", "excluded", category="attack", swing=[0.0, 0.2])
    retriever = SignatureRetriever(corpus_dir=tmp_path, backend="tfidf", range_filter="hard")
    results = retriever.search(
        "규칙적인 high low 전환",
        features={"swing_ratio": 2.0, "periodicity_strength": 0.9, "duty_regularity": 0.9},
    )
    assert results == []
    verdict = analyze_with_llm("설명", results, backend="stub")
    assert verdict["closest_match"] == "unknown"


def test_benign_match_is_normal_in_stub(tmp_path):
    _write_doc(tmp_path / "benign_periodic.md", "benign_periodic", category="benign", swing=[1.0, 2.0])
    retriever = SignatureRetriever(corpus_dir=tmp_path, backend="tfidf")
    results = retriever.search(
        "정상 분산학습의 규칙적인 high low 전환",
        features={"swing_ratio": 1.5, "periodicity_strength": 0.9, "duty_regularity": 0.9},
    )
    verdict = analyze_with_llm("설명", results, backend="stub")
    assert verdict["risk"] == "정상"
    assert verdict["closest_match"] == "benign_periodic"


def test_capture_dry_run_does_not_require_gpu(capsys):
    argv = [
        "run_capture.py",
        "--workload",
        "swma",
        "--duration",
        "10",
        "--dry-run",
    ]
    with patch("sys.argv", argv):
        capture_main()
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["private"]["gt_label"] == "swma"
    assert "/sessions/" in output["session_dir"]
    assert output["session_id"].startswith("s-")
    assert "--progress-log" in output["workload"]
    assert "label" not in output
    assert all("swma" not in str(output["session_dir"]).lower() for _ in [0])
