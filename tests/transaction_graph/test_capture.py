# SPDX-License-Identifier: Apache-2.0
"""Capture procedure (TXG-CAP-001 to TXG-CAP-015).

CPU tests (``engine``) use the fake-capture seam of
``support.txgcap_harness``: a fake cell capture runs the body eagerly inside
the capture block and records nothing, so these tests pin the procedure's
order, inventory, rows, pools, streams and refusals. ``gpu`` tests capture
real cells.
"""

from __future__ import annotations

import gc
import re

import pytest
import torch
from support import model_harness as h
from support import txg_eager as eager
from support import txgcap_harness as t
from support.txg_api import Record

from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
    ROW_STATUS_ENVELOPE,
    ROW_STATUS_ROLE_MISMATCH,
)

pytestmark = pytest.mark.engine

#: The start of the error a validation replay raises when a component pool's
#: null page or capture-row pages differ from the call's snapshot (TXG-CAP-004).
PAGE_MISMATCH = "the null page or the capture rows' pages"


def cell_windows(env, mark):
    """``[(begin, end)]`` event indices of each cell capture block since
    ``mark``, in capture order."""
    begins, ends = [], []
    for i, (name, _) in enumerate(env.events[mark:], start=mark):
        if name == "cuda.graph.capture_begin":
            begins.append(i)
        elif name == "cuda.graph.capture_end":
            ends.append(i)
    return list(zip(begins, ends, strict=True))


def pool_snapshot(bindings):
    """Every byte of the state pool, null page included."""
    return {"pool": h.pool_tensor(bindings).detach().clone()}


def assert_pools_equal(before, after):
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name]), f"pool {name} was written"


def startup_through_profiling(run):
    run.core.prepare()
    run.core.bind_profiling()


# ---------------------------------------------------------------------------
# Flat capture and the nested eager paths


# @spec TXG-CAP-001
def test_no_nested_graph_launch_while_a_cell_captures(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    startup_through_profiling(run)
    mark = len(env.events)
    run.core.profiling_capture()
    for begin, end in cell_windows(env, mark):
        nested = [
            e for e in env.events[begin:end] if e[0] in ("replay", "cuda.graph.replay")
        ]
        assert nested == [], (
            f"a nested graph was launched inside a cell capture: {nested}"
        )


# @spec TXG-CAP-002
def test_host_staging_inside_an_outer_capture_raises(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    startup_through_profiling(run)
    outcomes = []

    def stage(graph):
        staging = env.model._ensure_host_staging()
        try:
            staging.stage(
                "is_chunk", torch.ones(1, dtype=torch.bool), torch.device("cpu")
            )
        except Exception as error:  # noqa: BLE001 - the outcome is the assertion
            outcomes.append(error)
        else:
            outcomes.append(None)

    env.fake.on_capture = stage
    run.core.profiling_capture()
    assert outcomes, "no cell capture began"
    assert all(o is not None for o in outcomes), (
        "staging inside a capture did not raise"
    )


# @spec TXG-CAP-003
def test_segment_bindings_expose_their_uncaptured_paths(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    startup_through_profiling(run)
    run.core.profiling_capture()
    decode = env.model._decode_graph_binding
    assert callable(decode.uncaptured_decode_fn(geometry=env.geometry, tier=1))
    encoder = env.model._encoder_execution
    transition = getattr(encoder, "uncaptured_transition", None)
    assert transition is not None, (
        "the eager-graphed encoder has no uncaptured transition"
    )
    for count in range(1, env.max_num_seqs + 1):
        assert callable(transition(geometry=env.geometry, count=count))


# ---------------------------------------------------------------------------
# The procedure


def windows_before(env, mark, windows):
    """For each cell, the event span from the end of the previous cell's
    procedure (its validation replay; for the first cell, the scatter
    warm-up) to the cell's capture."""
    starts = []
    for index, (begin, _end) in enumerate(windows):
        if index == 0:
            warm = [
                i
                for i in range(mark, begin)
                if env.events[i][0] == "scatter"
                and t.is_scatter_warmup(env.events[i][1])
            ]
            starts.append(max(warm) + 1 if warm else mark)
        else:
            graph = env.events[windows[index - 1][0]][1][0]
            replay = next(
                i
                for i in range(windows[index - 1][1], begin)
                if env.events[i] == ("cuda.graph.replay", graph)
            )
            starts.append(replay + 1)
    return [(start, begin) for start, (begin, _) in zip(starts, windows, strict=True)]


# @spec TXG-CAP-004
def test_each_cell_runs_reference_three_warm_ups_capture_and_one_validation_replay(
    on_cpu, monkeypatch
):
    from support import txg_api

    run = on_cpu(n=2)
    env = run.env
    startup_through_profiling(run)
    t.spy_phases(monkeypatch, env, "model_rows_body")
    published_early = []

    def check_unpublished(graph):
        binding = txg_api.binding_of(env.model)
        if any(getattr(c, "graph", None) is graph for c in binding.cells.values()):
            published_early.append(graph.index)

    env.fake.on_replay = check_unpublished
    mark = len(env.events)
    run.core.profiling_capture()
    windows = cell_windows(env, mark)
    assert len(windows) == 2 * env.n
    for index, ((start, begin), (_, end)) in enumerate(
        zip(windows_before(env, mark, windows), windows, strict=True)
    ):
        bodies = [e for e in env.events[start:begin] if e[0] == "phase.model_rows_body"]
        assert len(bodies) == 4, f"cell {index}: {len(bodies)} eager runs, not 1 + 3"
        inside = [e for e in env.events[begin:end] if e[0] == "phase.model_rows_body"]
        assert len(inside) == 1, (
            f"cell {index}: the capture ran the body {len(inside)}x"
        )
        graph = env.events[begin][1][0]
        following = (
            windows[index + 1][0] if index + 1 < len(windows) else len(env.events)
        )
        replays = [
            e for e in env.events[end:following] if e == ("cuda.graph.replay", graph)
        ]
        assert len(replays) == 1, f"cell {index}: {len(replays)} validation replays"
    # chunk cells: the eager reference launches the nested segment graphs
    for index, (start, begin) in enumerate(windows_before(env, mark, windows)[: env.n]):
        assert any(e[0] == "replay" for e in env.events[start:begin]), (
            f"chunk cell {index}: no eager reference through the nested graphs"
        )
    assert published_early == [], "a cell was published before its validation replay"
    binding = txg_api.binding_of(env.model)
    published = [c.graph for c in binding.cells.values()]
    assert len(published) == 2 * env.n
    assert all(g is not None and g.captured for g in published), "a cell is unpublished"


@pytest.mark.parametrize("page", [1, 0], ids=["capture-row-page", "null-page"])
# @spec TXG-CAP-004, TXG-CAP-008
def test_a_validation_replay_that_differs_from_the_reference_fails_the_capture(
    on_cpu, page
):
    """After each replay, one page differs from the call's snapshot: page 1,
    the first capture row's page, or the null page, which no capture may
    store to and from which the fresh capture rows gather their zero-initial
    state (TXG-MASK-001). The pages are checked before the outputs, so the
    error names the changed page in both cases."""
    run = on_cpu(n=2)
    env = run.env
    startup_through_profiling(run)
    bindings = run.core.profiling_bindings

    def corrupt(graph):
        h.slot_bytes(bindings, page).fill_(0x7F)

    env.fake.on_replay = corrupt
    error = h.call_with(run.core.profiling_capture)
    assert error is not None, "a mismatching validation replay published its cell"
    assert PAGE_MISMATCH in t.text(error), t.text(error)
    assert t.fake_graph_count(env) == 0


# @spec TXG-CAP-005
def test_capture_rows_use_pages_one_to_k_and_are_all_rejected(on_cpu):
    run = on_cpu(n=3, max_num_seqs=3)
    env = run.env
    startup_through_profiling(run)
    before = pool_snapshot(run.core.profiling_bindings)
    mark = len(env.events)
    run.core.profiling_capture()
    order = t.largest_first(env.n)
    windows = cell_windows(env, mark)
    assert len(windows) == len(order)
    for (kind, count), (begin, end) in zip(order, windows, strict=True):
        scatters = [e[1] for e in env.events[begin:end] if e[0] == "scatter"]
        assert scatters, f"{kind} cell {count}: no predicated scatter captured"
        expected = ROW_STATUS_ENVELOPE if kind == "chunk" else ROW_STATUS_ROLE_MISMATCH
        for call in scatters:
            assert call["rows"] == count, (kind, count, call)
            assert sorted(call["pages"]) == list(range(1, count + 1)), (
                kind,
                count,
                call,
            )
            assert all(s & expected for s in call["status"]), (kind, count, call)
    assert_pools_equal(before, pool_snapshot(run.core.profiling_bindings))


# @spec TXG-CAP-005
def test_capture_rows_alternate_their_fresh_and_file_bits(on_cpu, monkeypatch):
    run = on_cpu(n=3, max_num_seqs=3)
    env = run.env
    startup_through_profiling(run)
    t.spy_phases(monkeypatch, env, "prepare_model_rows")
    mark = len(env.events)
    run.core.profiling_capture()
    windows = cell_windows(env, mark)
    spans = windows_before(env, mark, windows)
    for (kind, count), (start, begin) in zip(
        t.largest_first(env.n), spans, strict=True
    ):
        plans = [
            e[1] for e in env.events[start:begin] if e[0] == "phase.prepare_model_rows"
        ]
        assert plans and plans[-1] is not None, f"{kind} {count}: no capture-row plan"
        fresh, file = t.row_bits(plans[-1])
        assert len(fresh) == len(file) == count, (kind, count, fresh, file)
        for bits in (fresh, file):
            assert all(a != b for a, b in zip(bits, bits[1:])), (kind, count, bits)
        if count >= 2:
            assert set(fresh) == set(file) == {True, False}, (kind, count)


# @spec TXG-CAP-006
def test_scatter_warm_up_names_each_pools_first_free_page_before_the_first_cell(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    startup_through_profiling(run)
    for capture in ("profiling", "serving"):
        if capture == "serving":
            core.clear()
            core.bind_serving()
        # The component pools: the state layer's bound views, as core's
        # state-group layout holds them at the call (TXG-CAP-008).
        components = [
            address
            for _layer, layouts in t.component_layouts(env.state)
            for address, *_ in layouts
        ]
        assert components, "core set no state-group layout"
        mark = len(env.events)
        core.capture(profile_only=capture == "profiling")
        first_cell = next(
            i
            for i in range(mark, len(env.events))
            if env.events[i][0] == "cuda.graph.capture_begin"
        )
        # The segment graphs come first (TXG-LIFE-004, TXG-LIFE-006), after
        # their own part's scatters on model-owned scratch; the first cell's
        # eager reference and warm-ups follow the scatter warm-up.
        after_segments = 1 + max(
            (i for i in range(mark, first_cell) if env.events[i][0] == "capture"),
            default=mark - 1,
        )
        launched = [
            e[1] for e in env.events[after_segments:first_cell] if e[0] == "scatter"
        ]
        warm = launched[: len(components)]
        assert sorted(c["pool"] for c in warm) == sorted(components), (
            capture,
            warm,
        )
        for call in warm:
            assert call["rows"] == 1 and call["pages"] == [1], call
            assert all(s != 0 for s in call["status"]), call
        # After the component pools come the projected resident-pool views
        # that are not themselves a component pool (TXG-CAP-013): each a view
        # over the component pools, launched once. The first cell is the
        # count-N chunk cell, whose rows are N > 1, so no other launch here
        # has the warm-up's shape.
        views = [c for c in launched[len(components) :] if t.is_scatter_warmup(c)]
        pool = h.pool_tensor(env.runner.bindings)
        assert views and all(t.within(pool, c["pool"]) for c in views), (
            capture,
            views,
        )
        layouts = [c["layout"] for c in (*warm, *views)]
        assert len(set(layouts)) == len(layouts), (
            capture,
            "an argument was warmed more than once",
        )


# @spec TXG-CAP-007
def test_cells_are_captured_largest_first_chunk_then_echo(on_cpu):
    run = on_cpu(n=3, max_num_seqs=3)
    env, core = run.env, run.core
    core.startup(verify=False)
    for capture in (Record.PROFILING, Record.SERVING):
        assert t.cell_order(env, capture) == t.largest_first(env.n), capture


def _dead(refs) -> bool:
    gc.collect()
    return all(ref() is None for ref in refs)


def _assert_the_plugin_holds_no_accessor_object(env, when):
    """TXG-CAP-008: the plugin holds no layout object, component or tensor
    that the accessor returned (core's own references and the state layer's
    bound views aside), checked at ``when``."""
    held = t.plugin_holds(env, t.accessor_objects(env.state))
    assert held == [], (f"{when}, the plugin holds what the accessor returned", held)


# @spec TXG-CAP-008, TXG-LIFE-003
def test_cells_reach_the_pools_only_through_the_state_group_layouts_in_the_call(
    on_cpu, monkeypatch
):
    """The transaction-graph part reads ``ModelState.state_group_layouts``
    inside each capture call and nowhere else (not at core's binding, not in
    ``post_warmup`` or a step), stores to no page of either pool (the null
    page included), and, as soon as each call returns, holds no layout
    object, component layout or tensor the accessor returned, nor a view of
    one: only plain values. Core still holds them then (it installed them),
    so the check is what the plugin's objects reach, not whether they are
    dead. Once core releases the profiling KV cache and binds the serving
    one, the profiling tensors are gone."""
    run = on_cpu(n=2)
    env, core = run.env, run.core
    spy = t.AccessorSpy(monkeypatch, env.state)
    core.prepare()
    core.bind_profiling()
    before = pool_snapshot(core.profiling_bindings)
    core.profiling_capture()
    _assert_the_plugin_holds_no_accessor_object(env, "when the profiling call returned")
    assert_pools_equal(before, pool_snapshot(core.profiling_bindings))
    # Inside the call any reader counts: the model state may read the
    # accessor and hand the layouts to the binding (implementation layout,
    # ``check_bindings``); outside it, no transaction-graph frame may read it.
    profiling = [r for r in spy.reads if r["inside"]]
    assert profiling, "the profiling capture never read the state-group layouts"
    core.clear()
    core.bind_serving()
    before = pool_snapshot(core.serving_bindings)
    mark = len(spy.reads)
    core.serving_capture()
    _assert_the_plugin_holds_no_accessor_object(env, "when the serving call returned")
    assert_pools_equal(before, pool_snapshot(core.serving_bindings))
    serving = [r for r in spy.reads[mark:] if r["inside"]]
    assert serving, "the serving capture never read the state-group layouts"
    core.post_warmup()
    core.step(t.stream_unit(env, "s1"))
    assert spy.txg_reads(inside=False) == [], (
        "the transaction-graph part read the state-group layouts outside a capture call"
    )
    h.drop_advance_records()
    assert _dead(ref for r in profiling for ref in r["tensors"]), (
        "a profiling component tensor outlived core's release"
    )


def _break_layouts(env, broken):
    """Core's state-group layout with one part of its contract broken."""
    layer = t.state_layer_of(env)
    views = t.bound_views(env)
    other = "nemotron.other_state"
    return {
        "no-layout": (),
        "two-layouts": t.make_layouts({layer.prefix: views}, {layer.prefix: views}),
        "another-layer": t.make_layouts({other: views}),
        "extra-layer": t.make_layouts({layer.prefix: views, other: views}),
        "copied-view": t.make_layouts({layer.prefix: (views[0].clone(), *views[1:])}),
    }[broken]


#: What each broken condition's error names (TXG-CAP-008).
BROKEN = {
    "no-layout": r"(?i)layout",
    "two-layouts": r"(?i)layout",
    "another-layer": r"nemotron\.other_state|nemotron\.state",
    "extra-layer": r"nemotron\.other_state",
    "copied-view": None,  # the component's name, below
}


@pytest.mark.parametrize("broken", list(BROKEN))
# @spec TXG-CAP-008, TXG-MEM-010
def test_a_broken_layout_contract_fails_pool_access_with_no_remedy(on_cpu, broken):
    """A layout count other than one, a layout that does not hold exactly the
    state layer, or a component tensor that is not the layer's bound view
    fails the pool-access phase naming the broken condition, and gives no
    configuration remedy: no setting repairs core's contract. The fixed
    budget needs no profiling capture, so the serving capture is the call."""
    from support import txg_conditions as c

    run = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving()
    env.state._state_group_layouts = _break_layouts(env, broken)
    before = pool_snapshot(core.serving_bindings)
    error = h.call_with(core.serving_capture)
    assert error is not None, f"a {broken} layout passed pool access"
    message = t.text(error)
    assert t.OPTION in str(error) and "pool access" in str(error).lower(), message
    needle = BROKEN[broken]
    if needle is None:
        needle = re.escape(h.model_spec(env.model).names[0])
    assert re.search(needle, message), (broken, message)
    if broken == "two-layouts":
        assert c.shows_number(message, 2), message
    # The error states the budget setting (TXG-MEM-010), which names
    # ``--kv-cache-memory-bytes`` with its current value: a fact, not a remedy.
    settings = c.current_settings(env.vllm_config)
    assert c.configuration_remedies(message, settings) == [], (
        "a contract break gave a configuration remedy",
        c.configuration_remedies(message, settings),
        message,
    )
    assert t.fake_graph_count(env) == 0, "a cell was captured past pool access"
    assert_pools_equal(before, pool_snapshot(core.serving_bindings))


@pytest.mark.parametrize(
    "limit, passes", [(2, True), (None, False)], ids=["active-seqs-2", "max-seqs-3"]
)
# @spec TXG-CAP-008
def test_the_block_count_check_uses_the_admission_limit(on_cpu, limit, passes):
    """Pool access needs the admission limit plus one blocks, not ``N + 1``:
    three blocks hold the two capture rows of ``N = 2`` beside the null page,
    and pass with an admission limit of 2 (``max_num_active_seqs``) and fail
    with 3 (``max_num_seqs``). The block-count failure's message is
    ``test_memory.py``'s (TXG-MEM-017)."""
    run = on_cpu(
        n=2,
        max_num_seqs=3,
        max_num_active_seqs=limit,
        kv_cache_memory_bytes=1 << 30,
        capacity=3,
    )
    run.core.prepare()
    run.core.bind_serving()
    error = h.call_with(run.core.serving_capture)
    assert (error is None) is passes, (limit, error and t.text(error))
    assert t.fake_graph_count(run.env) == (2 * run.env.n if passes else 0)


# @spec TXG-CAP-009
def test_every_capture_uses_cores_stream_and_pool_and_creates_neither(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    startup_through_profiling(run)
    created = env.fake.streams_created
    for capture, pool in (("profiling", "profiling-pool"), ("serving", "global-pool")):
        if capture == "serving":
            core.clear()
            core.bind_serving()
        mark = len(env.events)
        core.capture(profile_only=capture == "profiling")
        contexts = [e[1] for e in env.events[mark:] if e[0] == "cuda.graph.context"]
        begins = [e[1] for e in env.events[mark:] if e[0] == "cuda.graph.capture_begin"]
        assert begins and {p for _, p in begins} == {pool}, (capture, begins)
        assert all(stream is env.capture_stream for _, _, stream in contexts), contexts
        pools = [e for e in env.events[mark:] if e[0] == "cuda.capture_pool.enter"]
        assert pools and all(p[1] == pool for p in pools), pools
        inside = False
        for name, _ in env.events[mark:]:
            inside = {"capture_pool.enter": True, "capture_pool.exit": False}.get(
                name.removeprefix("cuda."), inside
            )
            if name == "cuda.graph.capture_begin":
                assert inside, "a cell was captured outside capture_pool"
    assert env.fake.streams_created == created, "the model created a stream"


@pytest.mark.parametrize("outcome", ["returns", "raises"])
# @spec TXG-CAP-010
def test_the_capture_stream_is_synchronized_before_the_call_ends(on_cpu, outcome):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    startup_through_profiling(run)
    if outcome == "raises":
        env.fake.fail["replay"] = env.fake.replay_count + 1
    mark = len(env.events)
    h.call_with(core.profiling_capture)
    names = [e for e in env.events[mark:]]
    work = [
        i
        for i, (n, _) in enumerate(names)
        if n in ("cuda.graph.capture_end", "cuda.graph.replay", "cuda.fault")
    ]
    syncs = [
        i
        for i, (n, d) in enumerate(names)
        if (n == "cuda.stream.synchronize" and d == env.capture_stream.name)
        or n == "cuda.synchronize"
    ]
    assert work and syncs and max(syncs) > max(work), (
        "no synchronize after the last work"
    )


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
# @spec TXG-CAP-011
def test_a_capture_call_captures_no_cell_off_and_exactly_2n_on(on_cpu, on):
    run = on_cpu(n=3, max_num_seqs=3, populations=[1, 3] if on else [])
    env, core = run.env, run.core
    expected = 2 * 3 if on else 0
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    assert t.fake_graph_count(env) == expected
    core.clear()
    core.bind_serving()
    core.serving_capture()
    assert t.fake_graph_count(env) == expected


def startup_scatter_arguments(calls, serving_pool):
    """The resident-pool arguments, as host-side ``(address, shape, strides,
    dtype)``, that the masked scatter received over the serving pool: at
    startup these are the serving capture's scatter warm-up and its cells'
    eager references, warm-ups and captures, which pass the same views."""
    return {c["layout"] for c in calls if t.within(serving_pool, c["pool"])}


def run_fallback_classes_at_any_n(env, core, *, first_slot: int = 1) -> list[str]:
    """``t.run_fallback_classes`` at any ``N``, with ``max_num_seqs >= N + 1``
    and ``N + 3`` slots from ``first_slot``: one covered step and one step of
    every reachable fallback class (``chunk-uncovered`` and
    ``echo-uncovered`` at ``N + 1`` rows, ``signature`` with the fields
    ``endpoint_policy`` and ``forced_eou``). Fails unless the counters show
    each one."""
    from support import txg_api

    n = env.n
    slots: dict[str, int] = {}

    def step(*rows, cached: tuple[str, ...] = ()) -> None:
        for row in rows:
            slots.setdefault(row.req_id, first_slot + len(slots))
        ids = [r.req_id for r in rows] + list(cached)
        core.step(*rows, cached=cached, handles={r: slots[r] for r in ids})

    streams = tuple(f"cap13-k{i}" for i in range(n + 1))
    before = t.fallback_reasons(env)
    step(t.stream_unit(env, "cap13-covered"))
    assert t.fallback_reasons(env) == before, "the covered step fell back"
    # K = N + 1, then the same requests with no new audio: E = N + 1.
    step(*[t.stream_unit(env, r) for r in streams])
    step(cached=streams)
    step(t.with_endpoint_policy(t.stream_unit(env, "cap13-policy")))
    step(t.eou_request(env, streams[0]))
    after = t.fallback_reasons(env)
    grown = {key for key, value in after.items() if value != before.get(key, 0)}
    expected = {
        ("chunk-uncovered", txg_api.CHUNK, n + 1),
        ("echo-uncovered", txg_api.ECHO, n + 1),
        ("signature", txg_api.CHUNK, 1),
        ("signature", txg_api.ECHO, 1),
    }
    assert grown == expected, (grown, after)
    return ["chunk-uncovered", "echo-uncovered", "endpoint_policy", "forced_eou"]


def assert_scatter_arguments_were_launched_at_startup(
    env, core, spy, *, run_classes=t.run_fallback_classes
):
    """TXG-CAP-013: every resident-pool argument a scheduled step passes to
    the masked scatter, covered or in any reachable fallback class (run by
    ``run_classes``, at ``N = 2`` by default), has the address, shape,
    strides and dtype of an argument launched at startup."""
    serving_pool = h.pool_tensor(core.serving_bindings)
    startup = startup_scatter_arguments(spy.calls, serving_pool)
    assert startup, "the serving capture launched no scatter over the serving pool"
    mark = len(spy.calls)
    classes = run_classes(env, core)
    later = spy.calls[mark:]
    assert later, f"no scatter was launched by {classes}"
    foreign = [c for c in later if not t.within(serving_pool, c["pool"])]
    assert not foreign, "a step scattered outside the serving pool"
    unwarmed = {c["layout"] for c in later} - startup
    assert not unwarmed, (
        "the masked scatter received arguments that startup never launched",
        sorted(map(str, unwarmed)),
    )


# @spec TXG-CAP-013
def test_the_masked_scatter_receives_only_arguments_launched_at_startup(
    on_cpu, monkeypatch
):
    run = on_cpu(n=2, max_num_seqs=3, capacity=12)
    env, core = run.env, run.core
    core.startup()
    # The cells compute, so the covered step's scatters are compared too.
    eager.EagerCells(env.model).install(monkeypatch)
    assert_scatter_arguments_were_launched_at_startup(env, core, env.scatter)


# ---------------------------------------------------------------------------
# CUDA


def _replay_spy(monkeypatch, log):
    original = torch.cuda.CUDAGraph.replay

    def replay(self):
        log.append(("replay", torch.cuda.is_current_stream_capturing()))
        return original(self)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)


@pytest.mark.gpu
# @spec TXG-CAP-001
def test_no_nested_graph_replay_inside_a_cell_capture_cuda(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    log = []
    _replay_spy(monkeypatch, log)
    env.core.startup()
    assert not [e for e in log if e[1]], "a graph replayed while a cell was capturing"
    log.clear()
    env.core.step(t.stream_unit(env, "s1"))
    assert len(log) == 1, f"a covered one-kind step launched {len(log)} graphs"


@pytest.mark.gpu
# @spec TXG-CAP-002
def test_host_staging_inside_an_outer_capture_raises_cuda(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    outcomes = []
    original = torch.cuda.CUDAGraph.capture_begin

    def capture_begin(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not outcomes:
            staging = env.model._ensure_host_staging()
            error = h.call_with(
                staging.stage, "is_chunk", torch.ones(1, dtype=torch.bool), env.device
            )
            outcomes.append(error)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", capture_begin)
    h.call_with(env.core.startup)
    assert outcomes and outcomes[0] is not None, (
        "staging inside a capture did not raise"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("page", [1, 0], ids=["capture-row-page", "null-page"])
# @spec TXG-CAP-004, TXG-CAP-008, TXG-CAP-011
def test_corrupted_validation_replay_fails_and_publishes_nothing_cuda(
    on_cuda, monkeypatch, page
):
    """Every replay in the call (the segment graphs' and the cells') writes a
    value no earlier replay wrote into one page of the profiling pool, so the
    page after each cell's validation replay differs from the page at the
    start of the call's transaction-graph part: page 1, the first capture
    row's page, or the null page, which no capture may store to either and
    from which the fresh capture rows gather their zero-initial state. The
    pool is one flat byte tensor (``model_harness.real_bindings``); a page is
    ``model_harness.slot_bytes``."""
    env = on_cuda(n=2)
    core = env.core
    core.prepare()
    bindings = core.bind_profiling()
    original = torch.cuda.CUDAGraph.replay
    replays = []

    def replay(self):
        original(self)
        replays.append(1)
        h.slot_bytes(bindings, page).fill_(len(replays) % 255 + 1)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    error = h.call_with(core.profiling_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", original)
    assert error is not None, "the mismatching validation replay was accepted"
    assert "validation replay" in t.text(error), t.text(error)
    if page == 0:
        # The pages are checked before the outputs, so the page check reports
        # the change although the fresh capture rows gather the null page.
        assert PAGE_MISMATCH in t.text(error), t.text(error)
    from support import txg_api

    assert not txg_api.binding_of(env.model).cells, "a cell was published"
    core.clear()


@pytest.mark.gpu
# @spec TXG-CAP-005, TXG-CAP-008
def test_capture_stores_no_page_cuda(on_cuda):
    env = on_cuda(n=2)
    core = env.core
    core.prepare()
    bindings = core.bind_profiling()
    before = h.pool_tensor(bindings).clone()
    core.profiling_capture()
    assert torch.equal(before, h.pool_tensor(bindings)), (
        "the profiling capture stored a page"
    )
    core.clear()
    bindings = core.bind_serving()
    before = h.pool_tensor(bindings).clone()
    core.serving_capture()
    assert torch.equal(before, h.pool_tensor(bindings)), (
        "the serving capture stored a page"
    )


@pytest.mark.gpu
# @spec TXG-CAP-009
def test_cells_use_cores_stream_and_pool_cuda(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    pools = []
    original_graph = torch.cuda.graph

    class SpyGraph(original_graph):
        def __init__(self, cuda_graph, pool=None, stream=None, **kwargs):
            pools.append((pool, stream))
            super().__init__(cuda_graph, pool=pool, stream=stream, **kwargs)

    env.core.prepare()
    env.core.bind_profiling()
    env.core.profiling_capture()
    env.core.clear()
    env.core.bind_serving()
    from support import txg_gpu
    from vllm.platforms import current_platform

    created = txg_gpu.spy_stream_creation(monkeypatch)
    monkeypatch.setattr(torch.cuda, "graph", SpyGraph)

    env.core.serving_capture()
    cells = [p for p in pools if p[0] == current_platform.get_global_graph_pool()]
    assert len(cells) >= 2 * env.n, pools
    assert created == [], "the model created a stream"


@pytest.mark.gpu
# @spec TXG-CAP-012
def test_transaction_graphs_allocate_nothing_from_cores_workspace(on_cuda):
    from vllm.v1.worker import workspace

    env = on_cuda(n=2)
    if not workspace.is_workspace_manager_initialized():
        workspace.init_workspace_manager(env.device)
    manager = workspace.current_workspace_manager()
    env.core.prepare()
    size = manager._workspace_size_bytes(getattr(manager, "_workspace", None))
    env.core.startup()
    env.core.step(t.stream_unit(env, "s1"))
    assert manager._workspace_size_bytes(getattr(manager, "_workspace", None)) == size


@pytest.mark.gpu
@pytest.mark.parametrize("n", [2, 3])
# @spec TXG-CAP-013
def test_the_masked_scatter_receives_only_arguments_launched_at_startup_cuda(
    on_cuda, monkeypatch, n
):
    """The spy records host-side layouts only (no device read), so it runs
    inside the real captures. The echo section of the step arena starts
    ``152 * N`` bytes in, 16-byte aligned only for even ``N``, so the odd
    ``N = 3`` also runs the captures and every fallback class with that
    section misaligned."""
    env = on_cuda(n=n, max_num_seqs=n + 1, capacity=12)
    spy = t.ScatterSpy(env.events).install(monkeypatch)
    env.core.startup()
    assert_scatter_arguments_were_launched_at_startup(
        env, env.core, spy, run_classes=run_fallback_classes_at_any_n
    )


@pytest.mark.gpu
# @spec TXG-CAP-015
def test_covered_steps_keep_passing_their_host_checks_after_startup_cuda(on_cuda):
    """Host-side nested counters frozen under replay are read as deltas from a
    baseline taken after startup: consecutive covered steps neither raise nor
    fall back."""
    env = on_cuda(n=2)
    env.core.startup()
    replays = 0
    for sequence in range(4):
        env.core.step(
            t.stream_unit(env, "s1", sequence=sequence),
            handles={"s1": 1},
        ) if sequence == 0 else env.core.step(cached=("s1",), handles={"s1": 1})
        assert not any(t.counter(env, "fallbacks").values())
        replays_now = sum(t.counter(env, "replays").values())
        assert replays_now > replays, "a covered step did not replay"
        replays = replays_now
