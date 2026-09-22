from __future__ import annotations

import pytest
import torch

from bit2watt_impl import ltma_inject


def test_parser_accepts_duration_1800_and_rejects_larger():
    parser = ltma_inject.build_parser()
    assert parser.parse_args(["--duration", "1800"]).duration == 1800
    with pytest.raises(SystemExit):
        parser.parse_args(["--duration", "1800.1"])


def test_run_uses_streaming_progress_log(monkeypatch, capsys):
    emitted = []

    class FakeLog:
        def __init__(self, path, gpu_id):
            emitted.append(("init", path, gpu_id))

        def emit(self, event, **extra):
            emitted.append((event, extra))

        def close(self):
            emitted.append(("close",))

    fake_tensor = object()
    monkeypatch.setattr(ltma_inject, "ProgressLog", FakeLog)
    monkeypatch.setattr(
        ltma_inject, "_build_host_step", lambda *_args: (lambda: None, "llm")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "randn", lambda *_args, **_kwargs: fake_tensor)
    monkeypatch.setattr(torch, "randn_like", lambda _tensor: fake_tensor)
    monotonic = iter([0.0, 2.0])
    monkeypatch.setattr(ltma_inject.time, "monotonic", lambda: next(monotonic))

    args = ltma_inject.build_parser().parse_args(
        ["--duration", "1", "--progress-log", "progress.jsonl"]
    )
    ltma_inject.run(args)
    capsys.readouterr()
    assert emitted[0] == ("init", "progress.jsonl", 0)
    assert [entry[0] for entry in emitted] == [
        "init",
        "workload_start",
        "workload_end",
        "close",
    ]

