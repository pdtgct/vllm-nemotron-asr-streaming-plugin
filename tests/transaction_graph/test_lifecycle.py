# SPDX-License-Identifier: Apache-2.0
"""Capture lifecycle on #67583's hooks (TXG-LIFE-*).

CPU tests (``engine``) hold an option-on model state through the stated
fake-capture seam of ``support.txgcap_harness`` (``FakeCuda`` for core's CUDA
surface, ``FakeGraphRuntime`` for segment graphs) and call the hooks in
core's order (``support.core_order.CoreOrder``); they pin ordering,
inventory, release and refusal behaviour. ``gpu`` tests run the same
lifecycle on CUDA with core's ``graph_capture`` stream and graph pools and
read the allocator.
"""

from __future__ import annotations

import gc
import logging
import weakref

import pytest
import torch
from support import model_harness as h
from support import txgcap_harness as t
from support.fake_cuda import FakeCuda
from support.txg_api import Record

pytestmark = pytest.mark.engine


def names_since(env, mark):
    return [e[0] for e in env.events[mark:]]


def first(names, name, start=0):
    return names.index(name, start)


def cell_begins(names):
    return [i for i, n in enumerate(names) if n == "cuda.graph.capture_begin"]


def segment_captures(names):
    return [i for i, n in enumerate(names) if n == "capture"]


def scatter_warmups(env, mark):
    return [
        i
        for i, e in enumerate(env.events[mark:])
        if e[0] == "scatter" and t.is_scatter_warmup(e[1])
    ]


def assert_option_recognized(env):
    """Positive control for the negative guards: the model state holds an
    option-on transaction-graph binding and asks core to capture."""
    from support import txg_api

    assert txg_api.binding_of(env.model) is not None, "transaction graphs are off"
    assert env.state.needs_cudagraph_capture() is True


def assert_capture_creates_cells(env, core):
    """Positive control: the same fake counters a guard reads as zero become
    nonzero once the profiling capture runs."""
    if core.runner.bindings is None:
        core.bind_profiling()
    core.profiling_capture()
    assert env.fake.capture_attempts == 2 * env.n, env.fake.capture_attempts
    assert env.fake.replay_count >= 2 * env.n, env.fake.replay_count
    assert env.fake.pinned_allocations > 0, "no binding-owned pinned buffer"
    assert [c for c in env.scatter.calls if t.is_scatter_warmup(c)], (
        "no scatter warm-up"
    )


def assert_injected(error, phase):
    """The propagated error carries the injected fault of ``phase``."""
    from support.fake_cuda import InjectedFault

    faults = [
        e
        for e in t.chain(error)
        if isinstance(e, InjectedFault | torch.OutOfMemoryError)
        and "injected fault" in str(e)
    ]
    assert faults, f"the injected {phase} fault is not in the chain: {t.text(error)}"
    assert any(phase in str(e) for e in faults), (phase, [str(e) for e in faults])


def assert_refused_at_entry(env, call, mark):
    error = h.call_with(call)
    assert error is not None, "the capture call was not refused"
    assert t.OPTION in t.text(error), t.text(error)
    assert t.captures_since(env, mark) == [], "the refused call captured a graph"
    return error


# ---------------------------------------------------------------------------
# prepare_execution_memory and the profiling forward


# @spec TXG-LIFE-001
def test_prepare_execution_memory_allocates_nothing_for_transaction_graphs(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    assert_option_recognized(env)
    env.runner.prepare_execution_memory()
    assert env.fake.graphs == [], "a cell graph was created in prepare_execution_memory"
    assert env.fake.margin_allocations == 0
    assert env.fake.pinned_allocations == 0, (
        "the step arena's pinned buffers were allocated"
    )
    # Registration's segment part runs the default encoder's one-row profile
    # transactions here, on model-owned scratch; none is a transaction-graph
    # scatter warm-up.
    assert not [c for c in env.scatter.calls if t.is_scatter_warmup(c)], (
        "a transaction-graph scatter warm-up ran in prepare_execution_memory"
    )
    env.runner.profile_run(core_embeds=False)
    assert_capture_creates_cells(env, run.core)


# @spec TXG-LIFE-002
def test_profiling_forward_captures_and_replays_no_cell(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    assert_option_recognized(env)
    run.core.prepare()
    assert env.fake.capture_attempts == 0
    assert env.fake.replay_count == 0
    assert env.fake.graphs == []
    assert_capture_creates_cells(env, run.core)


# ---------------------------------------------------------------------------
# initialize_state_slots, the captures and clear_cudagraphs


# @spec TXG-LIFE-003
def test_cells_exist_only_from_their_capture_over_the_current_bindings(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    profiling = core.bind_profiling()
    assert t.fake_graph_count(env) == 0, "cells before the profiling capture"
    core.profiling_capture()
    assert t.fake_graph_count(env) == 2 * env.n
    core.clear()
    assert t.fake_graph_count(env) == 0, "profiling cells outlived clear_cudagraphs"
    serving = core.bind_serving()
    assert t.fake_graph_count(env) == 0, "cells before the serving capture"
    mark = len(env.events)
    core.serving_capture()
    assert t.fake_graph_count(env) == 2 * env.n
    layer = h.STATE_LAYER
    serving_pools = {
        v.data_ptr for v in h.core_bindings(serving).pool_layout(layer).values()
    }
    profiling_pools = {
        v.data_ptr for v in h.core_bindings(profiling).pool_layout(layer).values()
    }
    warmed = {env.events[mark + i][1]["pool"] for i in scatter_warmups(env, mark)}
    assert warmed and warmed <= serving_pools, (warmed, serving_pools)
    assert not warmed & profiling_pools, "the serving capture used the profiling pools"


# @spec TXG-LIFE-004
def test_profiling_capture_runs_segments_warm_up_cells_then_the_margin(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    mark = len(env.events)
    core.profiling_capture()
    names = names_since(env, mark)
    segments, cells = segment_captures(names), cell_begins(names)
    warmups = scatter_warmups(env, mark)
    assert segments and cells and warmups
    assert max(segments) < min(warmups) < min(cells), "not segments, warm-up, cells"
    margin = first(names, "cuda.margin.allocate")
    last_end = max(i for i, n in enumerate(names) if n == "cuda.graph.capture_end")
    assert margin > last_end, "the margin was allocated before the last cell"
    assert names.count("cuda.margin.allocate") == 1
    assert env.fake.live_margins() == 1, "the margin was not held after the call"
    pools = {e[1][1] for e in env.events[mark:] if e[0] == "cuda.graph.capture_begin"}
    assert pools == {"profiling-pool"}, pools


# @spec TXG-LIFE-005
def test_clear_releases_the_profiling_capture_and_keeps_its_figures(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    free_before = env.fake.free
    core.profiling_capture()
    core.clear()
    assert t.fake_graph_count(env) == 0
    assert env.fake.live_margins() == 0, "the margin outlived clear_cudagraphs"
    assert env.fake.free == free_before, "profiling-capture memory was not released"
    profiling = t.record(env)[Record.PROFILING]
    for key in (Record.CALL_TOTAL, Record.TXN_TOTAL, Record.PEAK):
        assert type(profiling[key]) is int, (key, profiling[key])
    assert all(type(v) is int for v in profiling[Record.KIND_TOTALS].values())


@pytest.mark.parametrize("before", ["nothing-captured", "capture-raised"])
# @spec TXG-LIFE-005
def test_clear_tolerates_a_profiling_capture_that_raised_or_never_ran(on_cpu, before):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    assert_option_recognized(env)
    core.prepare()
    core.bind_profiling()
    if before == "capture-raised":
        env.fake.fail["capture"] = 1
        error = h.call_with(core.profiling_capture)
        assert error is not None
        assert env.fake.capture_attempts == 2, "the capture was not attempted"
        assert_injected(error, "cell capture")
    core.clear()
    assert t.fake_graph_count(env) == 0
    assert env.fake.live_margins() == 0
    if before == "nothing-captured":
        assert_capture_creates_cells(env, core)


# @spec TXG-LIFE-006
def test_serving_capture_runs_segments_warm_up_cells_without_a_margin(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    mark = len(env.events)
    core.serving_capture()
    names = names_since(env, mark)
    segments, cells = segment_captures(names), cell_begins(names)
    warmups = scatter_warmups(env, mark)
    assert segments and cells and warmups
    assert max(segments) < min(warmups) < min(cells)
    assert "cuda.margin.allocate" not in names, "the serving capture allocated a margin"
    pools = {e[1][1] for e in env.events[mark:] if e[0] == "cuda.graph.capture_begin"}
    assert pools == {"global-pool"}, pools
    # the memory comparison reads the call total after the last cell
    last_end = max(i for i, n in enumerate(names) if n == "cuda.graph.capture_end")
    assert any(n == "cuda.mem_get_info" for n in names[last_end:])
    rec = t.record(env)
    assert Record.REQUIRED in rec and Record.EFFECTIVE in rec


# @spec TXG-LIFE-006, TXG-LIFE-008, TXG-MEM-009
def test_serving_capture_checks_the_second_capture_before_the_profiling_decision(
    on_cpu,
):
    """Entry order: a second serving capture is refused as such even when the
    profiling decision, made after it, would also refuse the call."""
    run = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving()
    core.serving_capture()  # fixed budget: no profiling capture required
    t.set_cache(env.vllm_config, kv_cache_memory_bytes=None)  # now it would be required
    mark = len(env.events)
    error = assert_refused_at_entry(env, core.serving_capture, mark)
    message = t.text(error)
    assert "restart" in message.lower(), message
    assert "estimate" not in message.lower(), "the profiling decision ran first"


# ---------------------------------------------------------------------------
# Ordering and a second serving capture


# @spec TXG-LIFE-007
def test_serving_capture_while_profiling_cells_exist_is_refused_at_entry(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    cells = t.fake_graph_count(env)
    mark = len(env.events)
    assert_refused_at_entry(env, core.serving_capture, mark)
    assert t.fake_graph_count(env) == cells


@pytest.mark.parametrize("serving", ["succeeded", "failed"])
# @spec TXG-LIFE-007
def test_profiling_capture_after_a_serving_capture_attempt_is_refused(on_cpu, serving):
    run = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving()
    if serving == "failed":
        env.fake.fail["capture"] = env.fake.capture_attempts + 1
        assert h.call_with(core.serving_capture) is not None
        env.fake.fail.clear()
    else:
        core.serving_capture()
    mark = len(env.events)
    assert_refused_at_entry(env, core.profiling_capture, mark)


# @spec TXG-LIFE-008
def test_second_serving_capture_is_refused_naming_restart(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup(verify=False)
    cells = t.fake_graph_count(env)
    mark = len(env.events)
    error = assert_refused_at_entry(env, core.serving_capture, mark)
    message = t.text(error).lower()
    assert "recapture" in message and "not supported" in message, message
    assert "restart" in message, message
    assert t.fake_graph_count(env) == cells, "the refusal dropped the serving cells"


@pytest.mark.parametrize("where", ["segment", "cell-capture", "aggregate"])
# @spec TXG-LIFE-007, TXG-LIFE-008, REG-MEM-024
def test_a_retry_after_a_failed_serving_capture_is_refused(on_cpu, where):
    """The refusal is keyed on the attempt, not on cells or segment graphs
    existing: the failed call discarded both. The refusal raises at the
    call's entry and changes nothing: it captures nothing, and the latch and
    the (empty) inventory stay as they were."""
    from support import txg_api

    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    _arm(env, where)
    assert h.call_with(core.serving_capture) is not None, "the capture did not fail"
    env.fake.fail.clear()
    env.graphs.disarm()
    binding = txg_api.binding_of(env.model)
    gc.collect()
    assert t.fake_graph_count(env) == 0 and env.graphs.captured() == [], (
        "the failed call left a graph: the retry would be refused for that reason"
    )
    assert binding.failed is False
    for retry in (core.serving_capture, core.profiling_capture):
        mark = len(env.events)
        error = assert_refused_at_entry(env, retry, mark)
        message = t.text(error).lower()
        assert "restart" in message, message
        assert "scatter" not in names_since(env, mark), "the refused call launched"
        assert binding.failed is False, "a refused capture latched the binding"
        assert t.fake_graph_count(env) == 0 and binding.cells == {}
    message = t.text(h.call_with(core.serving_capture)).lower()
    assert "recapture" in message and "not supported" in message, message
    env.state.shutdown()


# @spec REG-MEM-024
def test_a_retry_after_a_failed_serving_capture_is_refused_with_the_option_off(
    on_cpu,
):
    run = on_cpu(n=2, populations=[])
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    env.graphs.fail_capture_after = env.graphs.captures_completed + 1
    assert h.call_with(core.serving_capture) is not None, "the capture did not fail"
    env.graphs.disarm()
    gc.collect()
    assert env.graphs.captured() == [], "the failed call left a segment graph"
    for retry in (core.serving_capture, core.profiling_capture):
        mark = len(env.events)
        error = h.call_with(retry)
        assert error is not None, "the retry was not refused"
        assert "restart" in t.text(error).lower(), t.text(error)
        assert t.captures_since(env, mark) == [], "the refused call captured a graph"
    message = t.text(h.call_with(core.serving_capture)).lower()
    assert "recapture" in message and "not supported" in message, message
    env.state.shutdown()


# ---------------------------------------------------------------------------
# Repeated bindings


# @spec TXG-LIFE-009
def test_repeated_bindings_with_the_captured_layout_change_nothing(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup(verify=False)
    mark = len(env.events)
    env.state.initialize_state_slots(core.serving_bindings)
    assert t.captures_since(env, mark) == []
    assert t.fake_graph_count(env) == 2 * env.n
    core.post_warmup()  # still verifies


# @spec TXG-LIFE-009
def test_repeated_bindings_with_another_layout_latch_and_raise(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    other = core.bindings()
    error = h.call_with(env.state.initialize_state_slots, other)
    assert error is not None, "a different serving layout was accepted"
    error = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert error is not None and t.FRESH_ENGINE.search(t.text(error)), error


# @spec TXG-LIFE-009
def test_profiling_bindings_take_no_part_in_the_layout_comparison(on_cpu):
    core = on_cpu(n=2).core
    core.prepare()
    core.bind_profiling()
    core.bind_profiling()  # another temporary pool before the estimate
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    core.serving_capture()
    core.post_warmup()


# ---------------------------------------------------------------------------
# shutdown and the option off


# @spec TXG-LIFE-010
def test_shutdown_releases_every_cell_and_logs_the_summary(on_cpu, caplog):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    with caplog.at_level(logging.INFO):
        env.state.shutdown()
    gc.collect()
    assert t.fake_graph_count(env) == 0
    summaries = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO
        and r.name.startswith("vllm_nemotron_asr_streaming")
        and "replay" in r.getMessage().lower()
    ]
    assert len(summaries) == 1, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize(
    "stage", ["never-captured", "serving-capture-failed", "uninitialized"]
)
# @spec TXG-LIFE-010
def test_shutdown_tolerates_incomplete_state(on_cpu, stage):
    run = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    env, core = run.env, run.core
    state = env.state
    assert_option_recognized(env)
    if stage == "uninitialized":
        state = type(env.state).__new__(type(env.state))
    elif stage == "serving-capture-failed":
        core.prepare()
        core.bind_serving()
        env.fake.fail["capture"] = env.fake.capture_attempts
        error = h.call_with(core.serving_capture)
        assert error is not None
        assert env.fake.capture_attempts == 1, "the capture was not attempted"
        assert_injected(error, "cell capture")
    state.shutdown()
    assert t.fake_graph_count(env) == 0
    if stage in ("never-captured", "uninitialized"):
        # positive control on the guarded counters: in this environment (the
        # same model state for never-captured; its initialized model state
        # beside the bare object for uninitialized) a startup captures cells
        # this count sees, and shutdown releases them
        core.startup(profiling=False)
        assert t.fake_graph_count(env) == 2 * env.n
        assert t.counter(env, "captured_cells") == 2 * env.n
        env.state.shutdown()
        assert t.fake_graph_count(env) == 0


@pytest.mark.parametrize("runner", ["enforce-eager", "hook-configuration"])
# @spec TXG-LIFE-011
def test_option_off_hooks_allocate_capture_verify_and_release_nothing(
    off_cpu, monkeypatch, runner
):
    overrides = dict(t.DEFAULT_ARMS) if runner == "hook-configuration" else None
    env = off_cpu(overrides=overrides)
    fake = FakeCuda(env.events).install(
        monkeypatch, claim_cuda=runner != "enforce-eager"
    )
    scatter = t.ScatterSpy(env.events).install(monkeypatch)
    if runner == "hook-configuration":
        t._hook_configuration(env.vllm_config, [])
        env.state = h.build_model_state(env.vllm_config, env.model)
        env.runner = h.Runner(env.vllm_config, env.model, env.state)
        env.capture_stream = fake.new_stream(name="core-capture")
    t.core_of(env).startup()
    env.state.clear_cudagraphs()
    env.state.shutdown()
    assert fake.graphs == [], "a transaction-graph cell was created with the option off"
    assert fake.pinned_allocations == 0, (
        "the step arena was allocated with the option off"
    )
    # A transaction-graph scatter warm-up writes one rejected row to page 1
    # (TXG-LIFE-004); the default encoder's one-row profile
    # transaction writes model-owned scratch and commits its row.
    assert not [c for c in scatter.calls if t.is_scatter_warmup(c)], (
        "a transaction-graph scatter warm-up ran with the option off"
    )


# ---------------------------------------------------------------------------
# Whole-call cleanup


FAILURES = {
    True: ["segment", "cell-capture", "validation-replay", "margin"],
    False: ["segment", "cell-capture", "validation-replay", "aggregate"],
}


def _arm(env, where):
    fake = env.fake
    if where == "segment":
        env.graphs.fail_capture_after = env.graphs.captures_completed + 1
    elif where == "cell-capture":
        fake.fail["capture"] = fake.capture_attempts + 1
    elif where == "validation-replay":
        fake.fail["replay"] = fake.replay_count
    elif where == "margin":
        fake.fail["margin"] = fake.margin_allocations
    elif where == "aggregate":
        # the serving call retains more than the profiling call, margin included
        fake.capture_cost += fake.margin_bytes // (2 * env.n) + (1 << 20)


def _assert_nothing_of_the_call_remains(
    env, error, free_at_start, mark, where, attempts
):
    gc.collect()
    assert t.fake_graph_count(env) == 0, "a cell of the failed call remains"
    assert env.fake.live_margins() == 0, "the failed call's margin remains"
    assert env.graphs.captured() == [], "a segment graph of the failed call remains"
    assert env.fake.free == free_at_start, "device memory of the failed call remains"
    names = names_since(env, mark)
    syncs = [
        i
        for i, n in enumerate(names)
        if n in ("cuda.stream.synchronize", "cuda.synchronize")
    ]
    faults = [i for i, n in enumerate(names) if n == "cuda.fault"] or [0]
    assert syncs and max(syncs) > min(faults), (
        "no capture-stream synchronize after the failure"
    )
    phase = {
        "cell-capture": "cell capture",
        "validation-replay": "replay",
        "margin": "margin",
    }.get(where)
    if phase is not None:
        assert_injected(error, phase)
    elif where == "segment":
        assert env.graphs.failed_capture is not None, "no segment capture was attempted"
    elif where == "aggregate":
        assert env.fake.capture_attempts - attempts == 2 * env.n, (
            "the serving cells were not all captured before the comparison"
        )


@pytest.mark.parametrize("where", FAILURES[True])
# @spec TXG-LIFE-012
def test_failed_profiling_capture_discards_everything_the_call_created(on_cpu, where):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    free_at_start, mark = env.fake.free, len(env.events)
    attempts = env.fake.capture_attempts
    _arm(env, where)
    error = h.call_with(core.profiling_capture)
    assert error is not None
    _assert_nothing_of_the_call_remains(
        env, error, free_at_start, mark, where, attempts
    )
    core.clear()


@pytest.mark.parametrize("where", FAILURES[False])
# @spec TXG-LIFE-012
def test_failed_serving_capture_discards_everything_the_call_created(on_cpu, where):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    free_at_start, mark = env.fake.free, len(env.events)
    attempts = env.fake.capture_attempts
    _arm(env, where)
    error = h.call_with(core.serving_capture)
    assert error is not None
    _assert_nothing_of_the_call_remains(
        env, error, free_at_start, mark, where, attempts
    )
    env.state.shutdown()


# ---------------------------------------------------------------------------
# Replay only on scheduled steps; release order


# @spec TXG-LIFE-013
def test_cells_replay_only_in_their_own_validation_and_on_verified_steps(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    profiling = [g for ref in env.fake.graphs if (g := ref()) is not None]
    assert profiling and all(g.replays == 1 for g in profiling), (
        "each profiling cell replays once, its own validation replay"
    )
    profiling_ids = {g.index for g in profiling}
    core.clear()
    core.bind_serving()
    core.serving_capture()
    replays = env.fake.replay_count
    error = h.call_with(core.step, t.stream_unit(env, "s0"))
    assert error is not None, "a step ran before post-warmup verification"
    assert env.fake.replay_count == replays, "a cell replayed before verification"
    core.post_warmup()
    mark = len(env.events)
    # s0 still holds slot 1: core refuses a held slot at the same generation
    # before the model runs.
    core.step(t.stream_unit(env, "s1"), handles={"s1": 2})
    replayed = [e[1] for e in env.events[mark:] if e[0] == "cuda.graph.replay"]
    assert replayed, "the verified covered step replayed no cell"
    assert not set(replayed) & profiling_ids, "a step replayed a profiling cell"


def _watch_segment_graphs(env):
    for ref in env.graphs.graphs:
        graph = ref()
        if graph is not None:
            weakref.finalize(graph, env.events.append, ("segment.released", None))


def _assert_cells_released_before_segments(env, mark):
    gc.collect()
    names = names_since(env, mark)
    cells = [
        i
        for i, n in enumerate(names)
        if n in ("cuda.graph.released", "cuda.graph.reset", "cuda.margin.released")
    ]
    segments = [i for i, n in enumerate(names) if n == "segment.released"]
    assert cells and segments, (cells, segments)
    assert max(cells) < min(segments), "a segment graph was released before a cell"


@pytest.mark.parametrize("release", ["clear", "shutdown", "failure"])
# @spec TXG-LIFE-014
def test_transaction_graph_part_is_released_before_the_segment_graphs(on_cpu, release):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    if release == "clear":
        core.profiling_capture()
        _watch_segment_graphs(env)
        mark = len(env.events)
        env.state.clear_cudagraphs()
    elif release == "shutdown":
        core.profiling_capture()
        core.clear()
        core.bind_serving()
        core.serving_capture()
        _watch_segment_graphs(env)
        mark = len(env.events)
        env.state.shutdown()
    else:
        env.fake.fail["replay"] = env.fake.replay_count + 2 * env.n - 1  # the last cell
        watched = []

        def watch(graph):
            if not watched:  # the call's segment graphs exist by its first cell
                watched.append(True)
                _watch_segment_graphs(env)

        env.fake.on_capture = watch
        mark = len(env.events)
        assert h.call_with(core.profiling_capture) is not None
    _assert_cells_released_before_segments(env, mark)


# ---------------------------------------------------------------------------
# CUDA


@pytest.mark.gpu
# @spec TXG-LIFE-001
def test_prepare_execution_memory_allocates_no_device_memory_for_the_option_cuda(
    on_cuda, monkeypatch, tmp_path
):
    def delta(env):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        env.runner.prepare_execution_memory()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - before

    on = on_cuda(n=2)
    on_delta = delta(on)
    off = on_cuda(n=2).option_off()
    assert on_delta == delta(off), "the option allocated device memory in preparation"


@pytest.mark.gpu
# @spec TXG-LIFE-004, TXG-LIFE-005, TXG-LIFE-006, TXG-LIFE-013
def test_lifecycle_captures_releases_and_serves_on_cuda(on_cuda):
    env = on_cuda(n=2)
    core = env.core
    core.prepare()
    core.bind_profiling()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    core.profiling_capture()
    assert len(t.record(env)[Record.PROFILING][Record.CELLS]) == 2 * env.n
    core.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert torch.cuda.mem_get_info()[0] >= free_before - (8 << 20), (
        "the profiling capture and its margin were not released"
    )
    core.bind_serving()
    core.serving_capture()
    assert len(t.record(env)[Record.SERVING][Record.CELLS]) == 2 * env.n
    core.post_warmup()
    before = sum(t.counter(env, "replays").values())
    core.step(t.stream_unit(env, "s1"))
    assert sum(t.counter(env, "replays").values()) == before + 1
    assert env.runner.commit_log and not env.runner.commit_log[-1][1]
    env.state.shutdown()


@pytest.mark.gpu
# @spec TXG-LIFE-012
def test_failed_cell_capture_on_cuda_returns_its_memory(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    core = env.core
    core.prepare()
    core.bind_profiling()
    calls = {"n": 0}
    original = torch.cuda.CUDAGraph.capture_end

    def capture_end(self):
        original(self)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected fault in cell capture")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", capture_end)
    error = h.call_with(core.profiling_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", original)
    assert error is not None
    assert any("injected fault" in str(e) for e in t.chain(error))
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert torch.cuda.mem_get_info()[0] >= free_before - (8 << 20), (
        "the failed call's graphs or buffers remain"
    )
    core.clear()
    env.state.shutdown()
