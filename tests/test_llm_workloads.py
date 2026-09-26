from __future__ import annotations

from argparse import Namespace

from workloads.gpt_model import PRESETS
from workloads.llm_workloads import (
    DEFAULT_PROBE_TARGET_W,
    PROBE_CANDIDATES,
    _probe_key,
    distributed_should_stop,
    probe_candidate_batches,
)


def _fake_all_reduce_max(values: list[float]) -> list[float]:
    shared = max(values)
    return [shared for _ in values]


def test_distributed_should_stop_shared_via_fake_all_reduce():
    deadline = 10.0
    after = [
        distributed_should_stop(0.0, rank=0, deadline=deadline, now=11.0),
        distributed_should_stop(0.0, rank=1, deadline=deadline, now=11.0),
    ]
    assert _fake_all_reduce_max(after) == [1.0, 1.0]
    before = [
        distributed_should_stop(0.0, rank=0, deadline=deadline, now=9.0),
        distributed_should_stop(0.0, rank=1, deadline=deadline, now=9.0),
    ]
    assert _fake_all_reduce_max(before) == [0.0, 0.0]
    peer = [
        distributed_should_stop(0.0, rank=0, deadline=deadline, now=9.0),
        1.0,
    ]
    assert _fake_all_reduce_max(peer) == [1.0, 1.0]


def test_probe_candidates_are_ascending_and_key_includes_capacity():
    assert list(PROBE_CANDIDATES) == [8, 16, 32, 64, 128]
    assert probe_candidate_batches(4) == [8, 16, 32, 64, 128]
    args = Namespace(
        preset="small",
        seq_len=256,
        dtype="bf16",
        compile=False,
        mode="finetune",
        probe_target_w=DEFAULT_PROBE_TARGET_W,
    )
    key = _probe_key(args, "NVIDIA GeForce RTX 4090", 1)
    other = Namespace(
        preset="small",
        seq_len=256,
        dtype="bf16",
        compile=False,
        mode="finetune",
        probe_target_w=200.0,
    )
    assert key != _probe_key(other, "NVIDIA GeForce RTX 4090", 1)
    payload = "|".join(
        [
            "NVIDIA GeForce RTX 4090",
            "small",
            "256",
            "bf16",
            "False",
            "1",
            "finetune",
            str(DEFAULT_PROBE_TARGET_W),
            str(max(PROBE_CANDIDATES)),
            str(PRESETS["small"].block_size),
            str(PRESETS["small"].vocab_size),
        ]
    )
    import hashlib

    assert key == hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
