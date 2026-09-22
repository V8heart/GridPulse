"""CPU tests for GPT model and workload CLIs."""
from __future__ import annotations

import subprocess
import sys

import pytest

from workloads.gpt_model import PRESETS, build_gpt, count_parameters


def test_gpt_tiny_cpu_forward_and_param_count():
    import torch

    model = build_gpt("tiny", device="cpu")
    n = count_parameters(model)
    assert n > 1_000
    x = torch.randint(0, model.cfg.vocab_size, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, model.cfg.vocab_size)


@pytest.mark.parametrize(
    "module",
    [
        "workloads.llm_workloads",
        "workloads.llm_inference",
        "workloads.vision_workloads",
        "workloads.gpt_tiny_finetune",
        "workloads.normal_workloads",
        "bit2watt_impl.swma_workload",
        "bit2watt_impl.ltma_inject",
        "bit2watt_impl.crypto_workload",
        "bit2watt_impl.attack_variants",
    ],
)
def test_workload_modules_help(module):
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd="/home/agew1597/KDN",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_presets_defined():
    assert set(PRESETS) >= {"tiny", "small", "medium"}
    small = PRESETS["small"]
    assert (small.n_layer, small.n_head, small.n_embd, small.block_size) == (8, 12, 768, 128)
