# SPDX-License-Identifier: Apache-2.0
"""Capture lifecycle on #67583's hooks (TXG-LIFE-*).

CPU tests (``engine``) hold an option-on model state through the stated
fake-capture seam of ``support.txgcap_harness`` (``FakeCuda`` for core's CUDA
surface, ``FakeGraphRuntime`` for segment graphs) and call the hooks in
core's order (``support.core_order.CoreOrder``); they pin ordering,
inventory, release and refusal behaviour. A test runs core's
default KV-cache sizing, the extensible KV cache, unless it names another.
The lifecycle includes the profiling capture, which core runs only for its
graph-memory estimate, so a test whose subject includes that capture names
a fixed ``--num-gpu-blocks-override`` with the estimate enabled (the HLD's
choice for the checks that need the estimate; TXG-LIFE-004 names the
override's profiling KV cache), a profiled budget for core's one-block
profiling KV cache, or a fixed budget. ``gpu`` tests run the same
lifecycle on CUDA with core's ``graph_capture`` stream and graph pools and
read the allocator.
"""

from __future__ import annotations

import gc
import inspect
import logging
import re
import weakref

import pytest
import torch
from support import model_harness as h
from support import txgcap_harness as t
from support.fake_cuda import FakeCuda
from support.isolation import ran_in_child
from support.txg_api import Record, Sizing

pytestmark = pytest.mark.engine


#: The KV-cache sizing of a test whose subject includes core's graph-memory
#: estimate and its profiling capture (module docstring).
ESTIMATE = {"sizing": Sizing.BLOCK_COUNT}


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


def assert_capture_creates_cells(env, core, capture="profiling"):
    """Positive control: the same fake counters a guard reads as zero become
    nonzero once ``capture``, the profiling or the serving capture, runs."""
    if capture == "serving":
        core.bind_serving()
        core.serving_capture()
    else:
        # Bind the profiling pool again: after a clear, core's layout is
        # empty, and TXG-CAP-008 refuses a capture without one.
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
    assert_capture_creates_cells(env, run.core, capture="serving")


# @spec TXG-LIFE-002
def test_profiling_forward_captures_and_replays_no_cell(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env = run.env
    assert_option_recognized(env)
    run.core.prepare()
    assert env.fake.capture_attempts == 0
    assert env.fake.replay_count == 0
    assert env.fake.graphs == []
    assert_capture_creates_cells(env, run.core)


# ---------------------------------------------------------------------------
# Core's KV-cache binding, the captures and clear_cudagraphs


def _component_addresses(env):
    """The addresses of the component pools core's state-group layout holds
    now (the state layer's bound views)."""
    return {
        address
        for _layer, layouts in t.component_layouts(env.state)
        for address, *_ in layouts
    }


def _held(binding):
    """What the binding holds, by attribute and object identity."""
    return {k: id(v) for k, v in getattr(binding, "__dict__", {}).items()}


# @spec TXG-LIFE-003
def test_cells_exist_only_from_their_capture_over_the_current_layouts(on_cpu):
    """Core's binding of either KV cache (``bind_kv_cache``, the layout and
    ``initialize_kv_cache``) leaves the binding as it was; each capture
    takes its pools from the layout current at the call: the serving
    capture's scatter warm-up names the serving pools and none of the
    profiling ones."""
    from support import txg_api

    run = on_cpu(n=2, **ESTIMATE)
    env, core = run.env, run.core
    binding = txg_api.binding_of(env.model)
    core.prepare()
    held = _held(binding)
    core.bind_profiling()
    assert _held(binding) == held, "the binding recorded core's profiling binding"
    profiling_pools = _component_addresses(env)
    assert t.fake_graph_count(env) == 0, "cells before the profiling capture"
    core.profiling_capture()
    assert t.fake_graph_count(env) == 2 * env.n
    core.clear()
    assert t.fake_graph_count(env) == 0, "profiling cells outlived clear_cudagraphs"
    held = _held(binding)
    core.bind_serving()
    assert _held(binding) == held, "the binding recorded core's serving binding"
    serving_pools = _component_addresses(env)
    assert t.fake_graph_count(env) == 0, "cells before the serving capture"
    mark = len(env.events)
    core.serving_capture()
    assert t.fake_graph_count(env) == 2 * env.n
    warmed = {env.events[mark + i][1]["pool"] for i in scatter_warmups(env, mark)}
    assert serving_pools and serving_pools <= warmed, (warmed, serving_pools)
    assert not warmed & profiling_pools, "the serving capture used the profiling pools"


# @spec TXG-LIFE-004
def test_profiling_capture_runs_segments_warm_up_cells_then_the_margin(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
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
    # Over block 0 of the profiling pools: core's profiling KV cache, which
    # holds num_gpu_blocks_override blocks when that is set. Every scatter
    # into the profiling pools, the warm-up's and the capture rows', names
    # block 0; the segment graphs' scatters write the model's own scratch,
    # outside those pools.
    blocks = env.vllm_config.cache_config.num_gpu_blocks_override
    assert blocks is not None, "the environment sets no block count"
    assert {
        shape[0]
        for _, layouts in t.component_layouts(env.state)
        for _, shape, *_ in layouts
    } == {blocks}
    profiling_pool = h.pool_tensor(core.profiling_bindings)
    warmed = [env.events[mark + i][1] for i in warmups]
    assert all(t.within(profiling_pool, c["pool"]) for c in warmed), warmed
    scatters = [e[1] for e in env.events[mark:] if e[0] == "scatter"]
    rows = [c for c in scatters if t.within(profiling_pool, c["pool"])]
    assert rows and all(set(c["pages"]) == {0} for c in rows), rows


# @spec TXG-LIFE-004
def test_profiling_capture_over_one_block_under_a_profiled_budget(on_cpu):
    """Under a profiled budget with the extensible KV cache off, core's
    minimal profiling KV cache holds one block under the explicit NONE
    runner mode that TXG-CFG-018 requires: the profiling capture runs the
    segment graphs, the scatter warm-up and every profiling cell over block
    0 of that one-block cache into the pool core passes, then allocates the
    charged margin."""
    run = on_cpu(n=2, sizing=Sizing.PROFILED)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    assert {
        shape[0]
        for _, layouts in t.component_layouts(env.state)
        for _, shape, *_ in layouts
    } == {1}, "core's profiling KV cache under explicit NONE holds one block"
    mark = len(env.events)
    error = h.call_with(core.profiling_capture)
    assert error is None, t.text(error)
    names = names_since(env, mark)
    segments, cells = segment_captures(names), cell_begins(names)
    warmups = scatter_warmups(env, mark)
    assert segments and warmups and len(cells) == 2 * env.n
    assert max(segments) < min(warmups) < min(cells)
    assert first(names, "cuda.margin.allocate") > max(cells)
    pools = {e[1][1] for e in env.events[mark:] if e[0] == "cuda.graph.capture_begin"}
    assert pools == {"profiling-pool"}, pools


# @spec TXG-LIFE-005
def test_clear_releases_the_profiling_capture_and_keeps_its_figures(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, **ESTIMATE)
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

    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, populations=[], **ESTIMATE)
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
def test_a_rebinding_with_the_captured_layout_changes_nothing(on_cpu):
    """Core binds the same serving pages again through ``bind_kv_cache``
    alone (new view objects over the same storage; no
    ``initialize_kv_cache``): the plain-value layout equals the captured
    one, so the binding changes nothing and the worker still verifies and
    serves."""
    from support import txg_api

    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup(verify=False)
    binding = txg_api.binding_of(env.model)
    mark = len(env.events)
    initialized = _count_calls(env.state, "initialize_kv_cache")
    assert h.call_with(core.rebind_views, core.serving_bindings) is None
    assert initialized == [], "the rebinding initialized the KV cache"
    assert t.captures_since(env, mark) == []
    assert t.fake_graph_count(env) == 2 * env.n
    assert binding.failed is False
    core.post_warmup()  # still verifies
    core.step(t.stream_unit(env, "s1"))


def _count_calls(obj, name):
    """Record each later call of ``obj.name`` (an instance attribute wraps
    the method); returns the record."""
    calls = []
    real = getattr(obj, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    setattr(obj, name, counted)
    return calls


@pytest.mark.parametrize("attempt", ["succeeded", "failed"])
# @spec TXG-LIFE-009
def test_a_rebinding_with_another_layout_latches_and_raises(on_cpu, attempt):
    """After a serving capture was attempted, whether or not it succeeded,
    core binding the state layer's views over other pages through
    ``bind_kv_cache`` alone (another plain-value layout; no
    ``initialize_kv_cache``) raises from that call and has latched the
    binding when it returns, before any other lifecycle call; every later
    step then raises the fresh-engine error."""
    from support import txg_api

    run = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving()
    if attempt == "failed":
        env.fake.fail["capture"] = env.fake.capture_attempts + 1
        assert h.call_with(core.serving_capture) is not None
        env.fake.fail.clear()
    else:
        core.serving_capture()
        core.post_warmup()
    binding = txg_api.binding_of(env.model)
    assert binding.failed is False
    initialized = _count_calls(env.state, "initialize_kv_cache")
    error = h.call_with(core.rebind_views)  # a fresh pool: other addresses
    assert error is not None, "a different serving layout was accepted"
    assert initialized == [], "the rebinding initialized the KV cache"
    assert binding.failed is True, "bind_kv_cache returned without the latch"
    later = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


# @spec TXG-LIFE-009
def test_bindings_before_the_serving_capture_take_no_part_in_the_comparison(on_cpu):
    """Core's profiling KV cache, bound twice here, and a serving binding
    replaced before the serving capture are not compared: the serving
    capture records the layout current at its call."""
    core = on_cpu(n=2, **ESTIMATE).core
    core.prepare()
    core.bind_profiling()
    core.bind_profiling()  # another temporary pool before the estimate
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    core.bind_serving()  # other pages, before the serving capture
    core.serving_capture()
    core.post_warmup()


def _served_extensible(on_cpu, *, final, capacity=t.CAPACITY, extra=0, **kwargs):
    """An option-on environment under core's default KV-cache sizing, the
    extensible KV cache, through ``post_warmup``: the serving capture over a
    layout of ``capacity`` pages (the reservation capacity), then core's
    extension binding views of ``final`` pages over the same storage. With
    ``extra``, the serving cache's allocation holds ``extra`` pages beyond
    the bound views, so ``final`` may exceed the captured page count."""
    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE, capacity=capacity, **kwargs)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving(t.narrow_pool(core.bindings(capacity + extra), capacity))
    core.serving_capture()
    core.post_warmup()
    error = h.call_with(core.extend, final)
    return env, core, error


@pytest.mark.parametrize("final", [t.CAPACITY, 5, 3], ids=["all", "5", "3"])
# @spec TXG-LIFE-009
def test_cores_extension_with_no_more_pages_changes_nothing(on_cpu, final):
    """Under the extensible KV cache core binds the state layer's views again
    after post-warmup verification and before the first request, over the
    same pages with the final page count, at most the layout's page count:
    the views fit the layout the serving capture recorded, so the binding
    changes nothing, latches nothing and recaptures nothing, and the worker
    serves."""
    from support import txg_api

    env, core, error = _served_extensible(on_cpu, final=final)
    binding = txg_api.binding_of(env.model)
    assert error is None, ("core's extension latched the binding", t.text(error))
    assert binding.failed is False
    assert t.fake_graph_count(env) == 2 * env.n, "the extension recaptured"
    assert int(t.state_layer_of(env).num_blocks) == final
    core.step(t.stream_unit(env, "s1"))


# @spec TXG-LIFE-009
def test_a_binding_of_more_pages_than_the_serving_capture_read_latches(on_cpu):
    """Views over the same address with more pages than the layout the
    serving capture recorded do not fit: the cells' captured bounds do not
    cover them, so the binding latches and raises (worker-fatal)."""
    from support import txg_api

    env, core, error = _served_extensible(on_cpu, final=t.CAPACITY + 2, extra=2)
    assert error is not None, "a binding of more pages than captured was accepted"
    assert txg_api.binding_of(env.model).failed is True
    later = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


def _no_binding_report(env):
    """The state layer's bindings stop reaching the transaction-graph
    binding, so a test of a later rule sees core's extension without the
    binding's comparison (TXG-LIFE-009 is tested above)."""
    t.state_layer_of(env).bind_observer = None


# @spec TXG-CELL-004, TXG-EXEC-001
def test_steps_after_an_extension_to_fewer_pages_replay_with_no_signature_fallback(
    on_cpu,
):
    """A step's resident-pool views after core's extension have the captured
    address, page shape, strides and dtype and fewer pages: each fits the
    cell's entry, so the step's signature equals the cell's and the cell
    replays, with no fallback of reason ``signature``. The binding's report
    of the extension is cut, so the signature rule alone decides."""
    from support import txg_api

    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    env, core = run.env, run.core
    core.startup(extend=False)
    _no_binding_report(env)
    core.extend(5)
    replays = sum(t.counter(env, "replays").values())
    core.step(t.stream_unit(env, "s1"))
    assert sum(t.counter(env, "replays").values()) == replays + 1, (
        "the covered step after the extension did not replay its cell",
        t.fallback_reasons(env),
    )
    signature = [k for k in t.fallback_reasons(env) if k[0] == "signature"]
    assert signature == [], t.fallback_reasons(env)
    assert txg_api.binding_of(env.model).failed is False


# @spec TXG-CELL-004
def test_a_step_view_with_more_pages_than_the_cell_falls_back_on_signature(on_cpu):
    """A resident-pool view with more pages than the cell captured does not
    fit the cell's entry (the captured bounds do not cover the extra pages):
    the step falls back with reason ``signature``. The binding's report is
    cut, so the signature rule alone decides."""
    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving(t.narrow_pool(core.bindings(t.CAPACITY + 2), t.CAPACITY))
    core.serving_capture()
    core.post_warmup()
    _no_binding_report(env)
    core.extend(t.CAPACITY + 2)
    core.step(t.stream_unit(env, "s1"))
    signature = [k for k in t.fallback_reasons(env) if k[0] == "signature"]
    assert signature, ("a view of more pages matched the cell", t.fallback_reasons(env))


# @spec TXG-EXEC-010, REG-STATE-003
def test_a_slot_at_or_above_the_extended_count_is_refused_on_the_host(on_cpu):
    """After core's extension the rows' slots are bounded on the host by the
    bound views' first dimension, the final count, never by the state-group
    layout's page count (the reservation capacity): a request on slot 5
    after an extension to 5 pages is refused before any page is read, while
    slot 4 serves."""
    env, core, error = _served_extensible(on_cpu, final=5)
    assert error is None, t.text(error)
    core.step(t.stream_unit(env, "s4"), handles={"s4": 4})
    refused = h.call_with(core.step, t.stream_unit(env, "s5"), handles={"s5": 5})
    assert refused is not None, "a slot at the extended count was accepted"
    message = t.text(refused)
    assert "5" in message and re.search(r"(?i)slot|block", message), message


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
    spy = t.AccessorSpy(monkeypatch, env.state)
    core = t.core_of(env)
    core.startup()
    # Core binds the state layer's views again after the serving capture: with
    # the option off the state layer reports nothing to compare.
    assert h.call_with(core.bind_serving) is None, (
        "a rebinding raised with the option off"
    )
    env.state.clear_cudagraphs()
    env.state.shutdown()
    assert spy.txg_reads(inside=True) == spy.txg_reads(inside=False) == [], (
        "the transaction-graph part read the state-group layouts with the option off"
    )
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
    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, **ESTIMATE)
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
    run = on_cpu(n=2, **ESTIMATE)
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
    # s1 takes a block of its own: s0 was added before its step raised.
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
    run = on_cpu(n=2, **ESTIMATE)
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


def _assert_the_profile_transaction_binds_the_real_function() -> None:
    """``profile_execution``'s binding of ``advance_model_rows`` is the
    function ``advance`` defines, not a spy or recorder a test installed
    (``tests/conftest.py`` imports the module before any test runs)."""
    from vllm_nemotron_asr_streaming.nemotron_asr import advance, profile_execution

    bound = profile_execution.advance_model_rows
    assert (
        inspect.isfunction(bound)
        and bound.__module__ == advance.__name__
        and bound.__qualname__ == "advance_model_rows"
    ), ("profile_execution binds a patched row transaction", bound)


def _released() -> tuple[int, int]:
    """Device memory after everything unreferenced is returned: the test
    spies' recorded transaction arguments are dropped (they hold the pools of
    every call the spies saw), then a collection, a synchronize, PyTorch's
    per-stream cuBLAS workspaces (a library cache: the first matmul on a
    stream, such as the capture stream core creates for each capture call,
    allocates one of 8,519,680 bytes on the A40 that PyTorch keeps for the
    process) and the allocator's cache release. The profile transaction must
    call the real row transaction, so that no spy records it. Returns the
    allocator's reserved bytes and the driver's free bytes."""
    _assert_the_profile_transaction_binds_the_real_function()
    h.drop_advance_records()
    gc.collect()
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved(), torch.cuda.mem_get_info()[0]


def _warm_the_process(on_cuda) -> None:
    """One throwaway profiling capture and release: the process's first
    capture loads kernels and creates library handles for core's capture
    stream, device memory outside the allocator that no release returns
    (about 70 MiB on the A40, once per process), so a driver-level reading
    of a later capture measures only that capture. The allocator traces the
    warm-up, and after its clear and shutdown no live block may come from an
    allocation by the transaction-graph part: the warm-up cannot absorb
    allocator memory that only the first capture retains (TXG-LIFE-004,
    TXG-LIFE-012)."""
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(stacks="python", max_entries=200_000)
    try:
        env = on_cuda(n=2, **ESTIMATE)
        env.core.prepare()
        env.core.bind_profiling()
        env.core.profiling_capture()
        env.core.clear()
        env.state.shutdown()
        _released()
        snapshot = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    retained = _live_blocks_from(snapshot, TRANSACTION_GRAPH_SOURCE)
    assert not retained, (
        "the warm-up's clear and shutdown left blocks the transaction-graph "
        "part allocated",
        retained[:3],
    )


#: The transaction-graph part of the model, as an allocation's stack names it.
TRANSACTION_GRAPH_SOURCE = (
    "vllm_nemotron_asr_streaming/nemotron_asr/transaction_graph.py"
)


def _live_blocks_from(snapshot: dict, source: str) -> list[tuple[int, list[str]]]:
    """The blocks of the current device in an allocator ``snapshot``
    (``torch.cuda.memory._snapshot``) that are still in use (allocated or
    awaiting free) and whose allocation stack names ``source``, as
    ``(size, frames)``."""
    device = torch.cuda.current_device()
    found = []
    for segment in snapshot["segments"]:
        if segment.get("device", device) != device:
            continue
        for block in segment["blocks"]:
            if not str(block["state"]).startswith("active"):
                continue
            frames = [
                f"{frame['filename']}:{frame['line']}"
                for frame in block.get("frames", [])
            ]
            if any(source in frame for frame in frames):
                found.append((int(block["size"]), frames))
    return found


def _allocations_during(fn) -> list[list[str]]:
    """The Python stacks of every device allocation the caching allocator
    traces while ``fn`` runs (``torch.cuda.memory._record_memory_history``)."""
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    try:
        fn()
        torch.cuda.synchronize()
        snapshot = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    device = torch.cuda.current_device()
    return [
        [f"{frame['filename']}:{frame['line']}" for frame in entry.get("frames", [])]
        for entry in snapshot["device_traces"][device]
        if entry["action"] in ("alloc", "segment_alloc")
    ]


@pytest.mark.gpu
# @spec TXG-LIFE-001
def test_prepare_execution_memory_allocates_no_device_memory_for_the_option_cuda(
    on_cuda, monkeypatch, tmp_path, request
):
    """The allocator's own trace of the preparation with the option on holds
    no allocation from the transaction-graph part (the positive control: it
    holds the preparation's own allocations), and the device memory the
    preparation leaves allocated is the option-off preparation's. Each arm is
    a fresh model; a throwaway preparation first takes the process's one-time
    allocations, and the retained bytes are read with the spies' records
    dropped, after a collection and with PyTorch's cuBLAS workspaces cleared
    (``_released``). On the A40 each arm then retains 479,232 bytes, the
    encoder's extended positional-encoding buffer. The test runs alone in a
    child pytest process (``support.isolation``)."""
    if ran_in_child(request):
        return

    def delta(env):
        _released()
        before = torch.cuda.memory_allocated()
        env.runner.prepare_execution_memory()
        _released()
        return torch.cuda.memory_allocated() - before

    delta(on_cuda(n=2).option_off())
    on = on_cuda(n=2)
    stacks = _allocations_during(on.runner.prepare_execution_memory)
    assert stacks, "the allocator traced no allocation of the preparation"
    from_option = [s for s in stacks if any(TRANSACTION_GRAPH_SOURCE in f for f in s)]
    assert not from_option, ("the option allocated device memory", from_option[:3])
    off, on_delta = delta(on_cuda(n=2).option_off()), delta(on_cuda(n=2))
    assert on_delta == off, ("the option's preparation retains more", on_delta, off)


@pytest.mark.gpu
# @spec TXG-LIFE-004, TXG-LIFE-005, TXG-LIFE-006, TXG-LIFE-013
def test_lifecycle_captures_releases_and_serves_on_cuda(on_cuda, request):
    """Released: the allocator's reserved bytes (cells, binding-owned buffers,
    margin and the profiling graph pool's segments) return to their level
    before the capture, and the driver's free memory within 8 MiB of it, with
    the process's first-capture costs taken by ``_warm_the_process``. The test
    runs alone in a child pytest process (``support.isolation``)."""
    if ran_in_child(request):
        return
    _warm_the_process(on_cuda)
    env = on_cuda(n=2, **ESTIMATE)
    core = env.core
    core.prepare()
    core.bind_profiling()
    reserved_before, free_before = _released()
    core.profiling_capture()
    assert len(t.record(env)[Record.PROFILING][Record.CELLS]) == 2 * env.n
    core.clear()
    reserved_after, free_after = _released()
    assert reserved_after <= reserved_before, (
        "the profiling capture and its margin were not released to the allocator",
        reserved_before,
        reserved_after,
    )
    assert free_after >= free_before - (8 << 20), (
        "the profiling capture and its margin were not released",
        free_before,
        free_after,
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
def test_failed_cell_capture_on_cuda_returns_its_memory(on_cuda, monkeypatch, request):
    """Measured as in ``test_lifecycle_captures_releases_and_serves_on_cuda``,
    with the propagated error still held (it keeps no reference to the
    call's tensors or graphs)."""
    if ran_in_child(request):
        return
    _warm_the_process(on_cuda)
    env = on_cuda(n=2, **ESTIMATE)
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

    reserved_before, free_before = _released()
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", capture_end)
    error = h.call_with(core.profiling_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_end", original)
    assert error is not None
    assert any("injected fault" in str(e) for e in t.chain(error))
    reserved_after, free_after = _released()
    assert reserved_after <= reserved_before, (
        "the failed call's graphs or buffers remain in the allocator",
        reserved_before,
        reserved_after,
    )
    assert free_after >= free_before - (8 << 20), (
        "the failed call's graphs or buffers remain",
        free_before,
        free_after,
    )
    core.clear()
    env.state.shutdown()
