from __future__ import annotations

import torch

from workloads.gpt_model import PRESETS, build_gpt, count_parameters


def test_tiny_gpt_cpu_forward_and_parameter_count():
    model = build_gpt("tiny").eval()
    tokens = torch.randint(0, PRESETS["tiny"].vocab_size, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, PRESETS["tiny"].vocab_size)
    assert count_parameters(model) > 0


def test_kv_cache_matches_full_decode_last_token():
    torch.manual_seed(1)
    model = build_gpt("tiny").eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.no_grad():
        expected = model(tokens)[:, -1]
        _, cache = model(tokens[:, :-1], use_cache=True)
        actual, cache = model(
            tokens[:, -1:], past_key_values=cache, use_cache=True
        )
    assert len(cache) == model.cfg.n_layer
    torch.testing.assert_close(actual[:, -1], expected, rtol=1e-4, atol=1e-5)

