# SPDX-License-Identifier: Apache-2.0
"""Tier 5 pass/fail checks that run on an A40 (``gpu``).

Installed-engine checks (``card``: the published checkpoint named by
``NEMOTRON_CARD_DIR``): startup refusals (TXG-QUAL-023), the uncharged
WARNING and the required-but-suppressed profiling capture (TXG-QUAL-022),
and core's graph-memory estimate against ``B_hook`` (TXG-QUAL-021). They run
the engine core in process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so a test
can read the worker and patch core.

Harness checks (the runnable synthetic card, ``support.txgcap_harness``):
the expected fallbacks of ``K > N`` and ``E > N`` steps with outputs equal to
``B_hook``'s (TXG-QUAL-007) and recovery after a worker-fatal fault
(TXG-QUAL-024). The served halves of TXG-QUAL-007 (a forced signature
mismatch) and TXG-QUAL-024 (``EngineDeadError`` at the client) are
qualification-run obligations.
"""

from __future__ import annotations

import asyncio
import logging
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


def engine_args(**overrides: Any) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

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
    return AsyncEngineArgs(**settings)


def create_config(args: Any) -> Any:
    from vllm_nemotron_asr_streaming import register

    register()
    return args.create_engine_config()


# ---------------------------------------------------------------------------
# TXG-QUAL-023: refusals


REFUSALS = {
    "enforce-eager": dict(enforce_eager=True),
    "320-ms": dict(hf={"num_lookahead_tokens": 3}),
    "native-burst": dict(hf={"experimental_native_burst": True}),
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
# Installed engine in process


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    loop = asyncio.new_event_loop()
    engines = []

    def start(args):
        from vllm.v1.engine.async_llm import AsyncLLM

        from vllm_nemotron_asr_streaming import register

        register()

        async def make():
            return AsyncLLM.from_engine_args(args)

        llm = loop.run_until_complete(make())
        engines.append(llm)
        return llm

    def rpc(llm, fn):
        return loop.run_until_complete(llm.collective_rpc(fn))[0]

    yield start, rpc
    for llm in engines:
        try:
            llm.shutdown()
        except Exception:  # noqa: BLE001 - already shut down by the test
            pass
    loop.close()


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.card
# @spec TXG-QUAL-022
def test_a_fixed_budget_logs_the_uncharged_warning_with_its_figures(in_process):
    start, rpc = in_process
    handler = Capture()
    logger = logging.getLogger("vllm_nemotron_asr_streaming")
    logger.addHandler(handler)
    try:
        llm = start(engine_args(kv_cache_memory_bytes=8 << 30))
    finally:
        logger.removeHandler(handler)
    warnings = [
        r.getMessage()
        for r in handler.records
        if r.levelno == logging.WARNING and "not charged" in r.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in handler.records]
    record = rpc(llm, _record_of_worker)
    serving = record[Record.SERVING]
    for value in (serving[Record.CALL_TOTAL], serving[Record.TXN_TOTAL]):
        assert t.mentions_bytes(warnings[0], value), (value, warnings[0])


@pytest.mark.card
# @spec TXG-QUAL-022
def test_a_suppressed_required_profiling_capture_fails_startup(in_process, monkeypatch):
    from vllm.v1.worker.gpu import model_runner

    start, _ = in_process
    monkeypatch.setattr(
        model_runner.GPUModelRunner, "profile_cudagraph_memory", lambda self: 0
    )
    error = h.call_with(start, engine_args())
    assert error is not None, "startup succeeded without the required estimate"
    assert t.OPTION in t.text(error), t.text(error)


def _record_of_worker(worker):
    from support import txg_api

    return txg_api.counters(worker)["capture_record"]


def _estimate_of_worker(worker):
    return int(getattr(worker, "cudagraph_memory_estimate", 0)), int(
        getattr(worker, "available_kv_cache_memory_bytes", 0)
    )


@pytest.mark.card
# @spec TXG-QUAL-021
def test_cores_estimate_charges_the_profiling_transaction_total(in_process, capsys):
    start, rpc = in_process
    baseline = []
    for _ in range(2):
        llm = start(engine_args(hf={t.OPTION: ...}))
        baseline.append(rpc(llm, _estimate_of_worker))
        llm.shutdown()
    llm = start(engine_args())
    estimate, kv = rpc(llm, _estimate_of_worker)
    record = rpc(llm, _record_of_worker)
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
