# SPDX-License-Identifier: Apache-2.0
"""Installed-engine startup checks on CUDA with the published card
(REG-ENG-032, REG-STATE-019), each engine in a fresh child process
(``support.engine_child``, mode ``"llm"``: the worker runs in the child, so a
patch there sees the model state and the state layer).

``gpu`` and ``card`` tier: they collect and skip without CUDA or
``NEMOTRON_CARD_DIR``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.card, pytest.mark.engine]

#: The hook configuration (REG-ENG-005).
HOOK = {
    "enforce_eager": False,
    "compilation_config": {"mode": "none", "cudagraph_mode": "none"},
}
EAGER = {"enforce_eager": True}


def _settings(**settings):
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    return {
        "model": card,
        "config_format": "nemotron_asr_streaming",
        "tokenizer_mode": "nemotron_asr_streaming",
        "dtype": "float32",
        "async_scheduling": False,
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "gpu_memory_utilization": 0.5,
        **settings,
    }


def _run(tmp_path, settings, patch):
    from support.engine_child import run_engine

    reply = run_engine(tmp_path, {"mode": "llm", "settings": settings, "patch": patch})
    assert reply["ok"], reply.get("error")
    return reply


@pytest.mark.parametrize("runner", ["enforce-eager", "hook"])
# @spec REG-ENG-032
def test_startup_adds_no_request_before_the_first_client_request(tmp_path, runner):
    """Transaction graphs off and the extensible KV cache at its default: the
    model declines synthetic warmup (REG-TASK-004), so startup completes with
    no request reaching the model state's ``add_request``."""
    settings = _settings(**(EAGER if runner == "enforce-eager" else HOOK))
    reply = _run(tmp_path, settings, "record-add-request")
    assert reply["reads"]["add_requests"] == []


#: REG-STATE-019's two arms, both with core's graph-memory estimate running
#: and the extensible KV cache off.
_ARMS = {
    # transaction graphs on under the restricted startup (TXG-CFG-017 to
    # TXG-CFG-019) at 160 ms: core's state-group layout and profiling cells
    "block-count-override": {
        **HOOK,
        "num_gpu_blocks_override": 5,
        "hf_overrides": {
            "num_lookahead_tokens": 1,
            "transaction_graph_populations": [4],
        },
    },
    # transaction graphs off, a profiled budget with the extensible KV cache
    # turned off
    "profiled-budget": {**HOOK, "enable_extensible_kv_cache": False},
}


@pytest.mark.parametrize("arm", list(_ARMS))
# @spec REG-STATE-019, REG-STATE-018
def test_no_plugin_object_keeps_the_profiling_kv_cache(tmp_path, arm):
    """A weak reference to the profiling KV cache's storage, taken when core
    binds it to the state layer, has expired when core binds the serving KV
    cache."""
    reply = _run(tmp_path, _settings(**_ARMS[arm]), "profiling-kv-cache-ref")
    expired = reply["reads"]["profiling_kv_cache_expired"]
    assert expired, "core bound the state layer only once: no profiling KV cache"
    assert expired[0] is True, "an object kept the profiling KV cache"
