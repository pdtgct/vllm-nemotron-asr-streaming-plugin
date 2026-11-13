# SPDX-License-Identifier: Apache-2.0
"""Sampling-shaping fields cannot change a file's tokens (FILE-GRD-007).

CPU check: a complete real file step through the model on CPU
(``support.model_harness``) produces logits whose rows each have exactly one
finite entry (the decision). Every shaping operation then keeps that entry
the only candidate: temperature scaling, core's top-k / top-p masking
(``apply_top_k_top_p_pytorch``), min-p, the repetition, frequency and
presence penalties (OpenAI's definitions, as core applies them), a
watermark's additive bias, and seeded sampling. Core's GPU sampler itself is
not run here; the ``gpu`` test in ``test_engine_files.py`` compares served
tokens with each field set and unset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from support import model_harness as h

pytestmark = pytest.mark.engine


@pytest.fixture
def file_logits(monkeypatch, tmp_path):
    h.spec_without_state_name(monkeypatch)
    h.legacy_state_api(monkeypatch)
    card = h.write_runnable_card(tmp_path / "card")
    vllm_config = h.build_vllm_config(card)
    model = h.build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    runner = h.Runner(vllm_config, model, state)
    runner.profile_run(core_embeds=False)
    runner.bind_pool(h.real_bindings(h.model_spec(model)))
    samples = np.random.default_rng(3).standard_normal(20000).astype(np.float32) * 0.1
    request = h.new_request(
        "f1",
        int(model.config.file_carrier_token_id),
        samples=samples,
        header=[3, 0, len(samples)],
    )
    hidden, _ = runner.real_step(h.scheduler_output(new=(request,), handles={"f1": 1}))
    return model.compute_logits(hidden)


def _shape(logits: torch.Tensor, setting: str) -> torch.Tensor:
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    out = logits.clone()
    rows, vocab = out.shape
    seen = torch.randint(0, vocab, (rows, 8))
    if setting == "temperature":
        out = out / 1.7
    elif setting == "top_k":
        out = apply_top_k_top_p_pytorch(out, torch.full((rows,), 3), None)
    elif setting == "top_p":
        out = apply_top_k_top_p_pytorch(out, None, torch.full((rows,), 0.2))
    elif setting == "min_p":
        probs = out.softmax(-1)
        keep = probs >= 0.3 * probs.max(-1, keepdim=True).values
        out = out.masked_fill(~keep, float("-inf"))
    elif setting == "repetition_penalty":
        mask = torch.zeros_like(out, dtype=torch.bool).scatter_(1, seen, True)
        penalized = torch.where(out > 0, out / 1.5, out * 1.5)
        out = torch.where(mask, penalized, out)
    elif setting in ("frequency_penalty", "presence_penalty"):
        counts = torch.zeros_like(out).scatter_add_(
            1, seen, torch.ones_like(seen, dtype=out.dtype)
        )
        term = counts if setting == "frequency_penalty" else (counts > 0).to(out.dtype)
        out = out - 1.2 * term
    elif setting == "watermarking":
        bias = torch.zeros(vocab)
        bias[torch.randperm(vocab)[: vocab // 2]] = 2.0
        out = out + bias
    return out


SETTINGS = [
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "watermarking",
]


# @spec FILE-GRD-007
def test_file_logits_leave_exactly_one_candidate_per_row(file_logits):
    finite = torch.isfinite(file_logits)
    assert file_logits.shape[0] >= 1
    assert torch.all(finite.sum(-1) == 1)


@pytest.mark.parametrize("setting", SETTINGS)
@pytest.mark.parametrize("seed", [0, 7, 12345])
# @spec FILE-GRD-007
def test_shaping_keeps_the_decision_under_seeded_sampling(file_logits, setting, seed):
    decision = file_logits.argmax(-1)
    shaped = _shape(file_logits, setting)
    assert torch.equal(torch.isfinite(shaped), torch.isfinite(file_logits))
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.multinomial(shaped.softmax(-1), 1, generator=generator).squeeze(-1)
    assert torch.equal(sampled, decision)
    assert torch.equal(shaped.argmax(-1), decision)
