# SPDX-License-Identifier: Apache-2.0
"""Tier 5 pass/fail checks that run on an A40 (``gpu``).

Installed-engine checks (``card``: the published checkpoint named by
``NEMOTRON_CARD_DIR``): startup refusals (TXG-QUAL-023), the uncharged
WARNING and the required-but-suppressed profiling capture (TXG-QUAL-022),
and core's graph-memory estimate against ``B_hook`` (TXG-QUAL-021). Each
engine runs in a fresh child process with its engine core and worker in that
process (``support.engine_child``: ``LLMEngine`` with
``VLLM_ENABLE_V1_MULTIPROCESSING=0``; ``AsyncLLM`` always runs the engine core
out of process in the stage 1 build), so a check reads the worker's log
records and figures, patches core, and sees the worker's own startup error.
The engines start, one at a time, before any test of the session runs
(marker ``installed_engine``), so none shares the device with memory the
in-process tests left in the pytest process.

Harness checks (the runnable synthetic card, ``support.txgcap_harness``):
the expected fallbacks of ``K > N`` and ``E > N`` steps with outputs equal to
``B_hook``'s (TXG-QUAL-007) and recovery after a worker-fatal fault
(TXG-QUAL-024). The served halves of TXG-QUAL-007 (a forced signature
mismatch) and TXG-QUAL-024 (``EngineDeadError`` at the client) are
qualification-run obligations.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
from support import model_harness as h
from support import txg_api
from support import txgcap_harness as t
from support.txg_api import Record

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

N = 64


def engine_settings(**overrides: Any) -> dict[str, Any]:
    """The ``AsyncEngineArgs`` keywords of the hook configuration with the
    option at ``N`` (``hf`` entries set to ``...`` are removed)."""
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    hf = {"num_lookahead_tokens": 0, t.OPTION: [N]}
    hf.update(overrides.pop("hf", {}))
    hf = {k: v for k, v in hf.items() if v is not ...}
    settings = dict(
        model=card,
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        enforce_eager=False,
        async_scheduling=False,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=N,
        compilation_config={"cudagraph_mode": "NONE", "mode": 0},
        hf_overrides=hf,
    )
    settings.update(overrides)
    return settings


def engine_args(**overrides: Any) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

    return AsyncEngineArgs(**engine_settings(**overrides))


def create_config(args: Any) -> Any:
    from vllm_nemotron_asr_streaming import register

    register()
    return args.create_engine_config()


# ---------------------------------------------------------------------------
# TXG-QUAL-023: refusals


REFUSALS = {
    "enforce-eager": dict(enforce_eager=True),
    "320-ms": dict(hf={"num_lookahead_tokens": 3}),
    # At lookahead 1, where native burst is otherwise valid: at lookahead 0
    # the model configuration hook refuses native burst first (REG-LA-007),
    # before the configuration hook evaluates the transaction-graph checks
    # (REG-VAL-008); each hook lists its own failed checks (REG-VAL-003).
    "native-burst": dict(
        hf={"experimental_native_burst": True, "num_lookahead_tokens": 1}
    ),
    "not-a-list": dict(hf={t.OPTION: N}),
    "bool": dict(hf={t.OPTION: [True]}),
    "duplicate": dict(hf={t.OPTION: [N, N]}),
    "zero": dict(hf={t.OPTION: [0]}),
    "above-p-max": dict(hf={t.OPTION: [N + 1]}),
    "block-override": dict(num_gpu_blocks_override=8, hf={t.OPTION: [8]}),
    "eager-encoder": dict(hf={"encoder_execution_arm": "eager"}),
    "dense-graphed-encoder": dict(hf={"encoder_execution_arm": "dense-graphed"}),
    "eager-decoder": dict(hf={"decode_dispatch_arm": "compact-eager"}),
    "cudagraph-mode": dict(compilation_config={"cudagraph_mode": "FULL", "mode": 0}),
}
NAMES_OPTION = {
    "enforce-eager",
    "320-ms",
    "native-burst",
    "not-a-list",
    "bool",
    "duplicate",
    "zero",
    "above-p-max",
    "block-override",
    "eager-encoder",
    "dense-graphed-encoder",
    "eager-decoder",
}


@pytest.mark.card
# @spec TXG-QUAL-023
def test_the_valid_hook_configuration_starts_configuration():
    create_config(engine_args())


@pytest.mark.card
@pytest.mark.parametrize("case", list(REFUSALS))
# @spec TXG-QUAL-023
def test_startup_is_refused_for_each_unsupported_condition(case):
    error = h.call_with(create_config, engine_args(**dict(REFUSALS[case])))
    assert error is not None, f"{case} was accepted"
    if case in NAMES_OPTION:
        assert t.OPTION in t.text(error), t.text(error)


@pytest.mark.card
# @spec TXG-QUAL-023
def test_startup_is_refused_on_a_core_without_the_model_state_hooks(monkeypatch):
    from vllm.v1.worker.gpu.model_states import interface

    monkeypatch.delattr(interface.ModelState, "post_warmup")
    error = h.call_with(create_config, engine_args())
    assert error is not None, "a core without post_warmup was accepted"


# ---------------------------------------------------------------------------
# Installed engine, one child process per engine (``support.engine_child``)


def _run(settings: dict[str, Any], *, patch=None, reads=()) -> dict[str, Any]:
    return {"mode": "llm", "settings": settings, "patch": patch, "reads": list(reads)}


#: The engine runs of this module's checks, by name. The conftest starts the
#: selected ones before any test runs (``support.engine_child.prestart``).
ENGINE_RUNS = {
    "fixed-budget": lambda: _run(
        engine_settings(kv_cache_memory_bytes=8 << 30), reads=("record",)
    ),
    "suppressed-profiling": lambda: _run(engine_settings(), patch="suppress-profiling"),
    "b-hook-1": lambda: _run(engine_settings(hf={t.OPTION: ...}), reads=("estimate",)),
    "b-hook-2": lambda: _run(engine_settings(hf={t.OPTION: ...}), reads=("estimate",)),
    "option-on": lambda: _run(engine_settings(), reads=("estimate", "record")),
}


def engine(name: str, tmp_path) -> dict[str, Any]:
    from support.engine_child import reply

    return reply(name, ENGINE_RUNS[name], tmp_path)


def started(reply):
    assert reply["ok"], (reply.get("error") or {}).get("traceback", reply)
    return reply


@pytest.mark.card
@pytest.mark.installed_engine("fixed-budget")
# @spec TXG-QUAL-022
def test_a_fixed_budget_logs_the_uncharged_warning_with_its_figures(tmp_path):
    reply = started(engine("fixed-budget", tmp_path))
    warnings = [
        message
        for level, message in reply["records"]
        if level == "WARNING" and "not charged" in message
    ]
    assert len(warnings) == 1, reply["records"]
    serving = reply["reads"]["record"][Record.SERVING]
    for value in (serving[Record.CALL_TOTAL], serving[Record.TXN_TOTAL]):
        assert t.mentions_bytes(warnings[0], value), (value, warnings[0])


@pytest.mark.card
@pytest.mark.installed_engine("suppressed-profiling")
# @spec TXG-QUAL-022
def test_a_suppressed_required_profiling_capture_fails_startup(tmp_path):
    reply = engine("suppressed-profiling", tmp_path)
    assert not reply["ok"], "startup succeeded without the required estimate"
    error = reply["error"]
    assert t.OPTION in "\n".join(error["messages"]), error["traceback"]


@pytest.mark.card
@pytest.mark.installed_engine("b-hook-1", "b-hook-2", "option-on")
# @spec TXG-QUAL-021
def test_cores_estimate_charges_the_profiling_transaction_total(tmp_path, capsys):
    baseline = [
        started(engine(name, tmp_path))["reads"]["estimate"]
        for name in ("b-hook-1", "b-hook-2")
    ]
    reply = started(engine("option-on", tmp_path))
    estimate, kv = reply["reads"]["estimate"]
    record = reply["reads"]["record"]
    spread = abs(baseline[0][0] - baseline[1][0])
    transaction = record[Record.PROFILING][Record.TXN_TOTAL]
    assert estimate - max(b[0] for b in baseline) >= transaction - spread, (
        estimate,
        baseline,
        transaction,
        spread,
    )
    assert (
        record[Record.SERVING][Record.CALL_TOTAL]
        <= record[Record.PROFILING][Record.CALL_TOTAL]
    )
    with capsys.disabled():
        print(
            f"\nTXG-QUAL-021: estimate {estimate} B_hook {baseline} spread {spread} "
            f"transaction {transaction} KV budget difference "
            f"{kv - min(b[1] for b in baseline)}"
        )


# ---------------------------------------------------------------------------
# Harness checks on the runnable synthetic card


def b_hook(on_cuda, **kwargs):
    """``B_hook``: the same CUDA configuration with the option absent."""
    return on_cuda(**kwargs).option_off()


def uncovered_schedule(env):
    ids = [f"s{i}" for i in range(env.n + 1)]
    handles = {r: i + 1 for i, r in enumerate(ids)}
    return [
        dict(new=tuple(t.stream_unit(env, r) for r in ids), cached=(), handles=handles),
        dict(new=(), cached=tuple(ids), handles=handles),
    ]


def run_steps(env, schedule):
    env.core.startup()
    outputs = []
    for item in schedule(env):
        hidden, _ = env.core.step(
            *item["new"], cached=item["cached"], handles=item["handles"]
        )
        outputs.append(hidden.clone())
    return outputs, h.pool_tensor(env.runner.bindings).clone()


# @spec TXG-QUAL-007
def test_uncovered_steps_fall_back_once_with_option_off_outputs(on_cuda):
    on = on_cuda(n=2, max_num_seqs=3)
    on_out, on_pool = run_steps(on, uncovered_schedule)
    fallbacks = t.counter(on, "fallbacks")
    for reason, kind in (("chunk-uncovered", "chunk"), ("echo-uncovered", "echo")):
        key = txg_api.fallback_key(reason=reason, kind=kind, count=3)
        assert fallbacks.get(key) == 1, fallbacks
    off_out, off_pool = run_steps(
        b_hook(on_cuda, n=2, max_num_seqs=3), uncovered_schedule
    )
    for a, b in zip(on_out, off_out, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(on_pool, off_pool)


# @spec TXG-QUAL-024
def test_a_fresh_process_recovers_after_a_worker_fatal_fault(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    core = env.core
    core.startup()
    original = torch.cuda.CUDAGraph.replay
    launches = []

    def replay(self):
        original(self)
        launches.append(1)
        if len(launches) == 1:
            raise RuntimeError("test-only fault after the first launch")

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    split = (t.stream_unit(env, "s1"), t.file_item(env, "f1", header_samples=1))
    assert h.call_with(core.step, *split) is not None
    later = h.call_with(core.step, t.stream_unit(env, "s2"), handles={"s2": 3})
    assert later is not None and t.FRESH_ENGINE.search(t.text(later))
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", original)
    env.state.shutdown()

    def one_stream(env):
        return [dict(new=(t.stream_unit(env, "r1"),), cached=(), handles={"r1": 1})]

    fresh_out, _ = run_steps(on_cuda(n=2), one_stream)
    off_out, _ = run_steps(b_hook(on_cuda, n=2), one_stream)
    assert torch.equal(fresh_out[0], off_out[0])
