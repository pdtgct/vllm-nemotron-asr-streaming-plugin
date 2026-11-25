# SPDX-License-Identifier: Apache-2.0
"""Installed-engine checks on CUDA with the published card (REG-ENG-032,
REG-STATE-019 at startup; REG-MS-005 while serving), each engine in a fresh
child process (``support.engine_child``, mode ``"llm"``: the worker runs in
the child, so a patch there sees the model state and the state layer).

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


def _run(tmp_path, settings, patch, **request):
    from support.engine_child import run_engine

    reply = run_engine(
        tmp_path, {"mode": "llm", "settings": settings, "patch": patch, **request}
    )
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


# @spec REG-MS-005
def test_a_failed_request_ends_alone_and_a_retired_id_ends_nothing(tmp_path):
    """Complete files through the engine, in two phases: ``first`` alone,
    then ``victim`` and ``other`` together. On every real step after
    ``first`` retired, the model state's ``take_failed_requests`` also
    returns ``first``'s ID, which core's scheduler no longer holds; once,
    on the first step that holds ``victim`` beside ``other``, it returns
    ``victim``. ``victim`` alone ends with finish reason ``error``; ``other``,
    which shared that step (the recorded rows show it), and ``first`` end
    normally, each once; no error is raised."""
    files = [
        [{"id": "first", "seconds": 1.0, "seed": 0}],
        [
            {"id": "victim", "seconds": 3.0, "seed": 1},
            {"id": "other", "seconds": 3.0, "seed": 2},
        ],
    ]
    reply = _run(
        tmp_path,
        _settings(**EAGER),
        "inject-failed-requests",
        phases=files,
        inject={"live": "victim"},
    )
    reads = reply["reads"]
    ends: dict[str, list[str]] = {}
    for req_id, reason in reads["finished"]:
        ends.setdefault(req_id, []).append(reason)
    assert ends == {"first": ["stop"], "victim": ["error"], "other": ["stop"]}, ends
    (live,) = reads["injected"]["live"]
    assert live["id"].startswith("victim"), live
    assert any(r.startswith("other") for r in live["step"]), (
        "victim was failed on a step without other",
        live,
    )
    retired = reads["retired"][0]
    assert retired.startswith("first"), reads["retired"]
    assert reads["injected"]["retired"], "no step returned the retired ID"
    assert set(reads["injected"]["retired"]) == {retired}
