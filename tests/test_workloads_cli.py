from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "workloads.llm_workloads",
        "workloads.llm_inference",
        "workloads.vision_workloads",
        "bit2watt_impl.ltma_inject",
        "bit2watt_impl.attack_variants",
    ],
)
def test_workload_help_is_cpu_only_and_download_free(module):
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()

