"""Capture matrix dry-run / group_id / duration tests."""
from __future__ import annotations

from pathlib import Path

from dataset.capture_matrix import build_matrix, load_config
from dataset.identifiers import is_group_id

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_seed_reproducible_and_group_independent_of_repeat():
    config = load_config(ROOT / "config" / "capture_matrix.yaml")
    a = build_matrix(config)
    b = build_matrix(config)
    assert [x["capture_key"] for x in a] == [x["capture_key"] for x in b]
    assert all(is_group_id(x["group_id"]) for x in a)
    # Same entry different repeats share group_id
    by_entry = {}
    for item in a:
        by_entry.setdefault(item["entry_index"], set()).add(item["group_id"])
    assert all(len(v) == 1 for v in by_entry.values())
    assert all(item["duration_s"] in {480, 600, 720} for item in a)


def test_attack_odd_repeats_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
schema: kdn-capture-matrix/1
seed: 1
duration_pool_s: [480, 600, 720]
cooldown: {min_seconds: 60, max_wait_seconds: 900, idle_temp_delta_c: 3}
entries:
  - workload: swma
    repeats: 3
""",
        encoding="utf-8",
    )
    try:
        load_config(bad)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "odd" in str(exc).lower()
