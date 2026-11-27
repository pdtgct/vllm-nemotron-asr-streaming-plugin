# SPDX-License-Identifier: Apache-2.0
"""Capture procedure (TXG-CAP-001 to TXG-CAP-018).

CPU tests (``engine``) use the fake-capture seam of
``support.txgcap_harness``: a fake cell capture runs the body eagerly inside
the capture block and records nothing, so these tests pin the procedure's
order, inventory, rows, pools, streams and refusals. They run under core's
default KV-cache sizing, the extensible KV cache, whose one capture is the
serving capture. A test whose subject includes the profiling capture runs a
fixed ``--num-gpu-blocks-override`` with core's estimate enabled (the HLD's
choice for the checks that need the estimate; TXG-LIFE-004 names the
override's profiling KV cache), or a profiled budget with the extensible KV
cache off where core's one-block profiling KV cache is the subject
(``support.txgcap_harness.TxgCoreOrder``). ``gpu`` tests capture real
cells.
"""

from __future__ import annotations

import gc
import re

import pytest
import torch
from support import model_harness as h
from support import txg_eager as eager
from support import txgcap_harness as t
from support.txg_api import CAPTURE_ONLY_KEYWORD, Record, Sizing

from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
    ROW_STATUS_ENVELOPE,
    ROW_STATUS_ROLE_MISMATCH,
)

pytestmark = pytest.mark.engine

#: What the error a validation replay raises names when block 0 of a
#: component pool, the null page, differs from the call's snapshot
#: (TXG-CAP-004).
PAGE_MISMATCH = re.compile(r"(?i)null page|block 0")


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


def ready_to_capture(run):
    """Core's calls before the serving capture under the extensible KV
    cache: no profiling capture runs."""
    run.core.prepare()
    run.core.bind_serving()


# ---------------------------------------------------------------------------
# Flat capture and the nested eager paths


# @spec TXG-CAP-001
def test_no_nested_graph_launch_while_a_cell_captures(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    ready_to_capture(run)
    mark = len(env.events)
    run.core.serving_capture()
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
    ready_to_capture(run)
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
    run.core.serving_capture()
    assert outcomes, "no cell capture began"
    assert all(o is not None for o in outcomes), (
        "staging inside a capture did not raise"
    )


# @spec TXG-CAP-003
def test_segment_bindings_expose_their_uncaptured_paths(on_cpu):
    run = on_cpu(n=2)
    env = run.env
    ready_to_capture(run)
    run.core.serving_capture()
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
    ready_to_capture(run)
    t.spy_phases(monkeypatch, env, "model_rows_body")
    published_early = []

    def check_unpublished(graph):
        binding = txg_api.binding_of(env.model)
        if any(getattr(c, "graph", None) is graph for c in binding.cells.values()):
            published_early.append(graph.index)

    env.fake.on_replay = check_unpublished
    mark = len(env.events)
    run.core.serving_capture()
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


# @spec TXG-CAP-004, TXG-CAP-008
def test_a_validation_replay_that_changes_block_zero_fails_the_capture(on_cpu):
    """After each replay, block 0, the null page, differs from the call's
    snapshot: no capture may store to it, and every capture row gathers it
    (TXG-CAP-005). Block 0 is checked before the outputs, so the error names
    the changed page."""
    run = on_cpu(n=2)
    env = run.env
    ready_to_capture(run)
    bindings = run.core.serving_bindings

    def corrupt(graph):
        h.slot_bytes(bindings, 0).fill_(0x7F)

    env.fake.on_replay = corrupt
    error = h.call_with(run.core.serving_capture)
    assert error is not None, "a mismatching validation replay published its cell"
    assert PAGE_MISMATCH.search(t.text(error)), t.text(error)
    assert t.fake_graph_count(env) == 0


# @spec TXG-CAP-004
def test_the_snapshot_precedes_the_scatter_warm_up(on_cpu, monkeypatch):
    """The call takes its snapshot of block 0 at the start of its
    transaction-graph part, before the scatter warm-up: a warm-up launch
    that changed block 0 makes the first cell's validation replay differ
    from the snapshot, and the capture fails with nothing published."""
    from vllm_nemotron_asr_streaming.nemotron_asr import state_scatter

    run = on_cpu(n=2)
    env = run.env
    ready_to_capture(run)
    original = state_scatter.launch_rejected_row_scatter

    def warm_and_change_block_zero(pool, *args, **kwargs):
        original(pool, *args, **kwargs)
        pool[0].fill_(7)

    monkeypatch.setattr(
        state_scatter, "launch_rejected_row_scatter", warm_and_change_block_zero
    )
    error = h.call_with(run.core.serving_capture)
    assert error is not None, "a block 0 changed by the warm-up passed validation"
    assert PAGE_MISMATCH.search(t.text(error)), t.text(error)
    assert t.fake_graph_count(env) == 0


# @spec TXG-CAP-004, TXG-CAP-018
def test_the_call_compares_and_addresses_no_page_but_block_zero(on_cpu):
    """A page other than block 0 that changes during the call is neither
    read nor compared: under the extensible KV cache it may have no memory
    at the capture, so the snapshot, the validation replay's comparison and
    every capture row name block 0 alone, and the cells publish."""
    run = on_cpu(n=2)
    env = run.env
    ready_to_capture(run)
    bindings = run.core.serving_bindings

    def change_another_page(graph):
        h.slot_bytes(bindings, 1).fill_(0x7F)

    env.fake.on_replay = change_another_page
    error = h.call_with(run.core.serving_capture)
    assert error is None, ("a change outside block 0 failed the capture", error)
    assert t.fake_graph_count(env) == 2 * env.n


# @spec TXG-CAP-005
def test_capture_rows_all_name_block_zero_and_are_all_rejected(on_cpu):
    """Every capture row of a cell at count ``k`` names block 0, the only
    page core guarantees backed at a capture, ``k`` times; every row is
    rejected (a CHUNK row by its zero envelope header, an echo row by the
    stream carrier token), so no page is stored."""
    run = on_cpu(n=3, max_num_seqs=3)
    env = run.env
    ready_to_capture(run)
    before = pool_snapshot(run.core.serving_bindings)
    mark = len(env.events)
    run.core.serving_capture()
    order = t.largest_first(env.n)
    windows = cell_windows(env, mark)
    assert len(windows) == len(order)
    for (kind, count), (begin, end) in zip(order, windows, strict=True):
        scatters = [e[1] for e in env.events[begin:end] if e[0] == "scatter"]
        assert scatters, f"{kind} cell {count}: no predicated scatter captured"
        expected = ROW_STATUS_ENVELOPE if kind == "chunk" else ROW_STATUS_ROLE_MISMATCH
        for call in scatters:
            assert call["rows"] == count, (kind, count, call)
            assert call["pages"] == [0] * count, (kind, count, call)
            assert all(s & expected for s in call["status"]), (kind, count, call)
    assert_pools_equal(before, pool_snapshot(run.core.serving_bindings))


# @spec TXG-CAP-005
def test_capture_rows_alternate_their_fresh_and_file_bits(on_cpu, monkeypatch):
    run = on_cpu(n=3, max_num_seqs=3)
    env = run.env
    ready_to_capture(run)
    t.spy_phases(monkeypatch, env, "prepare_model_rows")
    mark = len(env.events)
    run.core.serving_capture()
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


def fill_block_zero(run, content):
    """Write ``content`` into block 0 of every component pool before the
    capture: ``bookkeeping-<seed>`` sets each integer component to bounded
    values (a seeded draw from 0 to 3; seed 0 writes zeros, seed 1 ones) and
    leaves the floating-point components; ``nan``, ``inf`` and ``-inf`` fill
    every floating-point component and leave the integer ones."""
    layer = h.state_layer_of(run.env.model)
    filled = 0
    for name, view in layer.component_pools.items():
        page = view[0]
        if content.startswith("bookkeeping-"):
            if page.is_floating_point():
                continue
            seed = int(content.rsplit("-", 1)[1])
            if seed < 2:
                page.fill_(seed)
            else:
                gen = torch.Generator().manual_seed(seed)
                page.copy_(torch.randint(0, 4, page.shape, generator=gen))
        else:
            if not page.is_floating_point():
                continue
            page.fill_(float(content))
        filled += 1
    assert filled, f"block 0 holds no component that {content} fills"


#: Block 0's contents for TXG-CAP-005's content independence: bounded
#: bookkeeping values (all zero, all one, seeded draws) and the IEEE specials.
BLOCK_ZERO_CONTENTS = [
    "bookkeeping-0",
    "bookkeeping-1",
    "bookkeeping-7",
    "bookkeeping-11",
    "nan",
    "inf",
    "-inf",
]


@pytest.mark.parametrize("content", BLOCK_ZERO_CONTENTS)
# @spec TXG-CAP-005
def test_each_capture_row_keeps_its_rejection_whatever_block_zero_holds(
    on_cpu, monkeypatch, content
):
    """The rejection predicate does not read the page a capture row gathers:
    with block 0 holding bounded bookkeeping values, NaN or an infinity,
    every capture row of every cell keeps its rejection bit
    (``ROW_STATUS_ENVELOPE`` for a CHUNK row, ``ROW_STATUS_ROLE_MISMATCH``
    for an echo row) and no page changes. Chunk and echo cells are checked
    independently: every cell is captured even when an earlier cell fails
    to publish, and the rows of every cell, at both of the fresh and file
    combinations the capture rows alternate (fresh stream, continuing
    file), are inspected one by one. A cell may fail only at its validation
    replay (a replay may find NaN outputs), so publication may fail; a
    failure at any earlier phase of any cell fails the test."""
    from support import txg_api

    graph = txg_api.graph_module()
    run = on_cpu(n=3, max_num_seqs=3)
    env = run.env
    ready_to_capture(run)
    fill_block_zero(run, content)
    t.spy_phases(monkeypatch, env, "prepare_model_rows")
    binding_type = type(txg_api.binding_of(env.model))
    real_capture_cell = binding_type._capture_cell
    attempted, unpublished = [], []

    def capture_cell(self, work, kind, count):
        attempted.append((kind, count))
        try:
            real_capture_cell(self, work, kind, count)
        except Exception as error:
            if self._phase != graph.PHASE_VALIDATION:
                raise
            unpublished.append((kind, count, repr(error)))

    monkeypatch.setattr(binding_type, "_capture_cell", capture_cell)
    before = pool_snapshot(run.core.serving_bindings)
    mark = len(env.events)
    # Publication after the cells may fail too; every cell has run by then.
    h.call_with(run.core.serving_capture)
    order = t.largest_first(env.n)
    assert attempted == order, (content, attempted, unpublished)
    windows = cell_windows(env, mark)
    assert len(windows) == len(order), (content, len(windows), unpublished)
    spans = windows_before(env, mark, windows)
    inspected: dict[str, set[tuple[bool, bool]]] = {"chunk": set(), "echo": set()}
    cells: dict[str, list[int]] = {"chunk": [], "echo": []}
    for (kind, count), (start, begin), (_, end) in zip(
        order, spans, windows, strict=True
    ):
        plans = [
            e[1] for e in env.events[start:begin] if e[0] == "phase.prepare_model_rows"
        ]
        assert plans and plans[-1] is not None, f"{kind} {count}: no capture-row plan"
        fresh, file = t.row_bits(plans[-1])
        assert len(fresh) == len(file) == count, (content, kind, count, fresh, file)
        scatters = [e[1] for e in env.events[begin:end] if e[0] == "scatter"]
        assert scatters, f"{kind} cell {count}: no predicated scatter captured"
        expected = ROW_STATUS_ENVELOPE if kind == "chunk" else ROW_STATUS_ROLE_MISMATCH
        for call in scatters:
            assert call["rows"] == count, (content, kind, count, call)
            for row, (f, fl, status) in enumerate(
                zip(fresh, file, call["status"], strict=True)
            ):
                assert status & expected, (content, kind, count, row, status)
                inspected[kind].add((f, fl))
        cells[kind].append(count)
    for kind in ("chunk", "echo"):
        assert cells[kind] == list(range(env.n, 0, -1)), (content, kind, cells)
        assert inspected[kind] == {(True, False), (False, True)}, (
            content,
            kind,
            inspected,
        )
    assert_pools_equal(before, pool_snapshot(run.core.serving_bindings))


# @spec TXG-CAP-006
def test_scatter_warm_up_names_block_zero_of_each_pool_before_the_first_cell(on_cpu):
    """Both captures, under a fixed block count with core's estimate
    enabled."""
    run = on_cpu(n=2, sizing=Sizing.BLOCK_COUNT)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
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
            assert call["rows"] == 1 and call["pages"] == [0], call
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
        assert all(c["pages"] == [0] for c in views), (capture, views)
        layouts = [c["layout"] for c in (*warm, *views)]
        assert len(set(layouts)) == len(layouts), (
            capture,
            "an argument was warmed more than once",
        )


# @spec TXG-CAP-007
def test_cells_are_captured_largest_first_chunk_then_echo(on_cpu):
    """Both captures, under a fixed block count with core's estimate
    enabled."""
    run = on_cpu(n=3, max_num_seqs=3, sizing=Sizing.BLOCK_COUNT)
    env, core = run.env, run.core
    core.startup(verify=False)
    for capture in (Record.PROFILING, Record.SERVING):
        assert t.cell_order(env, capture) == t.largest_first(env.n), capture


def _dead(storages) -> bool:
    """Whether every storage weakly referenced in ``storages``
    (``StorageWeakRef``) is freed: a view the plugin kept of a component
    keeps the storage although the component object is gone."""
    gc.collect()
    return all(ref.expired() for ref in storages)


def _assert_the_plugin_holds_no_accessor_object(env, when):
    """TXG-CAP-008: the plugin holds no layout object, component or tensor
    that the accessor returned, nor any tensor sharing a component tensor's
    storage (core's own references and the state layer's bound views
    aside), checked at ``when``."""
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
    object, component layout or tensor the accessor returned, nor any tensor
    sharing one's storage: only plain values. Core still holds them then (it
    installed them), so the check is what the plugin's objects reach, not
    whether they are dead. Once core releases the profiling KV cache and
    binds the serving one, the profiling component tensors' storage is
    freed."""
    run = on_cpu(n=2, sizing=Sizing.BLOCK_COUNT)
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
    core.profiling_bindings = None  # the harness's own reference to core's pool
    if not _dead([ref for r in profiling for ref in r["storages"]]):
        spans = [span for r in profiling for span in r["spans"]]
        pytest.fail(
            "a profiling component tensor's storage outlived core's release; "
            f"the plugin's paths to it: {t.plugin_holds(env, [], spans=spans)}"
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


# @spec TXG-CAP-008, TXG-LIFE-004
def test_the_profiling_capture_makes_no_page_count_check(on_cpu):
    """Under a profiled budget with the extensible KV cache off, core's
    profiling KV cache holds one block under the explicit NONE runner mode,
    fewer than the admission limit plus one: the profiling capture's cells
    need block 0 alone and its page count is not the serving count, so the
    profiling capture captures every cell over it."""
    run = on_cpu(n=2, max_num_seqs=2, sizing=Sizing.PROFILED)
    env, core = run.env, run.core
    assert core.profiling_capacity() == 1
    core.prepare()
    core.bind_profiling()
    error = h.call_with(core.profiling_capture)
    assert error is None, t.text(error)
    assert t.fake_graph_count(env) == 2 * env.n


# @spec TXG-CAP-009
def test_every_capture_uses_cores_stream_and_pool_and_creates_neither(on_cpu):
    """Both captures, under a fixed block count with core's estimate
    enabled."""
    run = on_cpu(n=2, sizing=Sizing.BLOCK_COUNT)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
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
    ready_to_capture(run)
    if outcome == "raises":
        env.fake.fail["replay"] = env.fake.replay_count + 1
    mark = len(env.events)
    h.call_with(core.serving_capture)
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
    run = on_cpu(
        n=3, max_num_seqs=3, populations=[1, 3] if on else [], sizing=Sizing.BLOCK_COUNT
    )
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


def fits(argument, launched):
    """TXG-CAP-013's rule: the same address, page shape (every dimension
    after the first), strides and dtype, and a first dimension at most the
    launched argument's."""
    address, shape, strides, dtype = argument
    return any(
        address == a
        and tuple(shape[1:]) == tuple(s[1:])
        and tuple(strides) == tuple(st)
        and dtype == d
        and shape[0] <= s[0]
        for a, s, st, d in launched
    )


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
    ``run_classes``, at ``N = 2`` by default), fits an argument launched at
    startup (:func:`fits`)."""
    serving_pool = h.pool_tensor(core.serving_bindings)
    startup = startup_scatter_arguments(spy.calls, serving_pool)
    assert startup, "the serving capture launched no scatter over the serving pool"
    mark = len(spy.calls)
    classes = run_classes(env, core)
    later = spy.calls[mark:]
    assert later, f"no scatter was launched by {classes}"
    foreign = [c for c in later if not t.within(serving_pool, c["pool"])]
    assert not foreign, "a step scattered outside the serving pool"
    unwarmed = {c["layout"] for c in later if not fits(c["layout"], startup)}
    assert not unwarmed, (
        "the masked scatter received arguments that startup never launched",
        sorted(map(str, unwarmed)),
    )


@pytest.mark.parametrize("final", [None, 6], ids=["final-capacity", "final-6"])
# @spec TXG-CAP-013
def test_the_masked_scatter_receives_only_arguments_launched_at_startup(
    on_cpu, monkeypatch, final
):
    """Under the extensible KV cache, after core's extension: to the whole
    reservation (views equal to the captured ones), and to 6 of its 12
    pages, whose views of fewer pages over the same addresses fit the
    arguments the serving capture launched."""
    run = on_cpu(n=2, max_num_seqs=3, capacity=12)
    env, core = run.env, run.core
    core.startup(extend=False)
    core.extend(final)
    # The cells compute, so the covered step's scatters are compared too.
    eager.EagerCells(env.model).install(monkeypatch)
    assert_scatter_arguments_were_launched_at_startup(env, core, env.scatter)
    if final is not None:
        later = [c["layout"] for c in env.scatter.calls]
        assert any(shape[0] == final for _, shape, *_ in later), (
            "no step scattered over the extended views"
        )


# ---------------------------------------------------------------------------
# The structural preflight's capture-only mode (TXG-CAP-017)


def null_row_plan(count: int = 3, **fields):
    """A CPU row plan of ``count`` CHUNK capture rows, every one naming the
    null block ID 0, with an empty live set: the shape of a cell's capture
    rows (TXG-CAP-005). ``fields`` replace plan fields."""
    import dataclasses

    from support.txg_fakes import Row, row_plan

    plan = row_plan([Row("chunk", fresh=i % 2 == 0) for i in range(count)])
    plan = dataclasses.replace(
        plan,
        state_indices_p=torch.zeros(count, dtype=torch.int64),
        live_block_ids=torch.zeros(0, dtype=torch.int64),
        num_pool_blocks=8,
        bindings=tuple(dataclasses.replace(b, block_id=0) for b in plan.bindings),
    )
    return dataclasses.replace(plan, **fields)


def preflight(plan, *, rows=None, **kwargs):
    from support import txg_api

    rows = plan.num_prefills if rows is None else rows
    return txg_api.structural_preflight()(plan, rows, rows, **kwargs)


# @spec TXG-CAP-017
def test_the_capture_only_mode_accepts_rows_that_all_name_the_null_block():
    """The mode waives exactly three row checks: the null block ID named by
    a real row, a block ID named by two rows and a block ID outside the live
    set, which an empty live set makes every row's."""
    for count in (1, 3):
        index = preflight(null_row_plan(count), capture_only=True)
        assert index.tolist() == [0] * count


def _off_the_null_block(plan):
    import dataclasses

    bindings = list(plan.bindings)
    bindings[1] = dataclasses.replace(bindings[1], block_id=1)
    return dataclasses.replace(
        plan,
        state_indices_p=torch.tensor([0, 1, 0], dtype=torch.int64),
        bindings=tuple(bindings),
    )


def _with_live(plan):
    import dataclasses

    return dataclasses.replace(plan, live_block_ids=torch.tensor([1]))


@pytest.mark.parametrize(
    "defect", [_off_the_null_block, _with_live], ids=["row-on-block-1", "live-set"]
)
# @spec TXG-CAP-017
def test_the_capture_only_mode_refuses_a_live_page(defect):
    """In the mode, a plan whose row names a block ID other than the null
    block ID, or whose live set is not empty, is refused: the mode never
    lets a plan name a page a request holds."""
    plan = defect(null_row_plan(3))
    with pytest.raises(ValueError):
        preflight(plan, capture_only=True)


def _row_count(plan):
    return plan, {"rows": plan.num_prefills + 1}


def _duplicate_request(plan):
    import dataclasses

    ids = (plan.request_ids[0],) * plan.num_prefills
    bindings = tuple(dataclasses.replace(b, request_id=ids[0]) for b in plan.bindings)
    return dataclasses.replace(plan, request_ids=ids, bindings=bindings), {}


def _binding_mismatch(plan):
    import dataclasses

    bindings = list(plan.bindings)
    bindings[0] = dataclasses.replace(bindings[0], admission_generation=7)
    return dataclasses.replace(plan, bindings=tuple(bindings)), {}


def _null_outside_pool(plan):
    import dataclasses

    return dataclasses.replace(plan, null_block_id=8), {}


def _chunk_deadline(plan):
    import dataclasses

    deadlines = plan.ready_deadline_ns.clone()
    deadlines[0] = 0
    bindings = list(plan.bindings)
    bindings[0] = dataclasses.replace(bindings[0], ready_deadline_ns=0)
    return dataclasses.replace(
        plan, ready_deadline_ns=deadlines, bindings=tuple(bindings)
    ), {}


OTHER_CHECKS = {
    "row-count": _row_count,
    "duplicate-request": _duplicate_request,
    "binding-mismatch": _binding_mismatch,
    "null-outside-pool": _null_outside_pool,
    "chunk-deadline": _chunk_deadline,
}


@pytest.mark.parametrize("defect", list(OTHER_CHECKS))
# @spec TXG-CAP-017
def test_the_capture_only_mode_keeps_every_other_check(defect):
    """Every structural check other than the three waived row checks holds
    in the mode: here the row count, unique request IDs, each row's prepared
    binding, the null block ID's bounds and the CHUNK deadlines."""
    preflight(null_row_plan(3), capture_only=True)  # the positive control
    plan, kwargs = OTHER_CHECKS[defect](null_row_plan(3))
    with pytest.raises(ValueError):
        preflight(plan, capture_only=True, **kwargs)


# @spec TXG-CAP-017
def test_outside_the_capture_only_mode_a_null_block_row_is_refused():
    """A scheduled step's plan never takes the mode: without it, a real row
    that names the null block ID is refused, as is a block ID named by two
    rows."""
    from support import txg_api

    with pytest.raises(ValueError, match="null block"):
        txg_api.advance_module()._structural_preflight(null_row_plan(3), 3, 3)


# @spec TXG-CAP-017, TXG-TXN-007
def test_only_a_capture_calls_cell_programs_pass_the_capture_only_mode(
    on_cpu, monkeypatch
):
    """``prepare_model_rows`` receives the mode only from the capture call's
    cell programs, one call per cell; ``advance_model_rows`` has no such
    parameter, and no scheduled step (the option-off transaction or
    ``execute``) passes it."""
    import inspect

    from support import txg_api

    advance = txg_api.advance_module()
    assert (
        CAPTURE_ONLY_KEYWORD in inspect.signature(advance.prepare_model_rows).parameters
    ), "prepare_model_rows takes no capture-only keyword"
    assert (
        CAPTURE_ONLY_KEYWORD
        not in inspect.signature(advance.advance_model_rows).parameters
    ), "advance_model_rows's interface changed"
    run = on_cpu(n=2, max_num_seqs=3)
    env, core = run.env, run.core
    # Each prepare_model_rows call as (through advance_model_rows, mode).
    calls = []
    depth = [0]
    original_prepare = advance.prepare_model_rows
    original_advance = advance.advance_model_rows

    def spy(*args, **kwargs):
        calls.append((depth[0] > 0, bool(kwargs.get(CAPTURE_ONLY_KEYWORD, False))))
        return original_prepare(*args, **kwargs)

    def advance_spy(*args, **kwargs):
        depth[0] += 1
        try:
            return original_advance(*args, **kwargs)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(advance, "prepare_model_rows", spy)
    monkeypatch.setattr(advance, "advance_model_rows", advance_spy)
    # profile_execution binds the name at import.
    import importlib

    profile_execution = importlib.import_module(
        advance.__name__.rsplit(".", 1)[0] + ".profile_execution"
    )
    monkeypatch.setattr(profile_execution, "advance_model_rows", advance_spy)
    ready_to_capture(run)
    assert not any(mode for _, mode in calls), calls
    calls.clear()
    core.serving_capture()
    # The warm-ups inside the capture call prepare rows through
    # advance_model_rows, without the mode; the cell programs call
    # prepare_model_rows directly, each with it, one call per cell.
    assert not any(mode for through, mode in calls if through), calls
    direct = [mode for through, mode in calls if not through]
    assert direct == [True] * (2 * env.n), calls
    calls.clear()
    core.post_warmup()
    core.extend()
    core.step(t.stream_unit(env, "cap17-s1"))
    run_fallback_classes_at_any_n(env, core, first_slot=2)
    assert calls and not any(mode for _, mode in calls), calls


# ---------------------------------------------------------------------------
# Block 0 alone (TXG-CAP-018)


def assert_only_block_zero(recorder, what):
    assert recorder.accesses, f"{what} addressed no state page (positive control)"
    outside = recorder.outside({0})
    assert outside == [], (f"{what} addressed pages other than block 0", outside)


class _CapturingAccesses(t.BlockAccesses):
    """``BlockAccesses`` with the CUDA capture's unreadable index tensors
    simulated on CPU: while ``capturing`` is set, no index tensor can be
    read."""

    capturing = False

    def _unreadable(self, index):
        return self.capturing


# @spec TXG-CAP-018
def test_the_block_reading_reads_a_captured_index_after_the_capture():
    """A gather index the capture itself computes (a fresh tensor, at
    another address than any warm-up read) is read when the reading's
    context exits, not charged as the whole view: a gather of block 0
    addresses block 0, and a gather of another block still addresses that
    block. A warm-up's values still answer for the same tensor."""
    pool = torch.zeros(6, 1, 1, 8, dtype=torch.uint8)
    view = pool.view(6, 8)
    warm = torch.tensor([0])
    with _CapturingAccesses(pool) as rec:
        view.index_select(0, warm)
        rec.capturing = True
        view.index_select(0, warm)
        view.index_select(0, torch.where(torch.tensor([True]), 0, 3))
        view.index_select(0, torch.where(torch.tensor([False]), 0, 3))
        assert rec.accesses[2:] == [("index_select", ())] * 2
        rec.capturing = False
    assert rec.accesses == [
        ("index_select", (0,)),
        ("index_select", (0,)),
        ("index_select", (0,)),
        ("index_select", (3,)),
    ]
    assert rec.outside({0}) == [("index_select", (3,))]


# @spec TXG-CAP-018, TXG-CAP-004, TXG-CAP-005, TXG-CAP-006
def test_a_capture_call_addresses_only_block_zero_of_the_state_pools(on_cpu):
    """Every operation of the serving capture that reads or writes a
    component pool or a projected resident-pool view (the snapshot, the
    scatter warm-up, each cell's eager reference, warm-ups, capture and
    validation replay, the comparison with the snapshot) addresses block 0
    alone, so the call needs memory behind block 0 only, as under the
    extensible KV cache, which backs only its minimum blocks then. The
    reading is ``support.txgcap_harness.BlockAccesses`` over the pool's
    backing storage."""
    run = on_cpu(n=3, max_num_seqs=3)
    ready_to_capture(run)
    with t.BlockAccesses(h.core_bindings(run.core.serving_bindings).tensor) as rec:
        run.core.serving_capture()
    assert t.fake_graph_count(run.env) == 2 * run.env.n
    assert_only_block_zero(rec, "the serving capture")


# @spec TXG-CAP-018, TXG-LIFE-004
def test_the_profiling_capture_addresses_only_block_zero_of_its_one_block_cache(
    on_cpu,
):
    """The profiling capture over core's one-block profiling KV cache (a
    profiled budget with the extensible KV cache off) addresses block 0
    alone: no operation reaches past the cache's only block. The cache is
    bound as one page over a storage of four, and the reading covers the
    whole storage, so an operation on any page past block 0 is observed."""
    run = on_cpu(n=3, max_num_seqs=3, sizing=Sizing.PROFILED)
    run.core.prepare()
    assert run.core.profiling_capacity() == 1
    storage = run.core.bindings(4)
    bindings = run.core.bind_profiling(t.narrow_pool(storage, 1))
    assert h.core_bindings(bindings).capacity == 1
    with t.BlockAccesses(h.core_bindings(storage).tensor) as rec:
        run.core.profiling_capture()
    assert t.fake_graph_count(run.env) == 2 * run.env.n
    assert_only_block_zero(rec, "the profiling capture")


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
@pytest.mark.parametrize("page", [0, 1], ids=["null-page", "another-page"])
# @spec TXG-CAP-004, TXG-CAP-008, TXG-CAP-011, TXG-CAP-018
def test_corrupted_validation_replay_fails_and_publishes_nothing_cuda(
    on_cuda, monkeypatch, page
):
    """Every replay in the call (the segment graphs' and the cells') writes a
    value no earlier replay wrote into one page of the serving pool, under
    the extensible KV cache. Block 0, the null page, then differs after each
    cell's validation replay from its snapshot at the start of the call's
    transaction-graph part, and the capture fails and publishes nothing;
    page 1 is neither read nor compared (it may have no memory then), so the
    capture publishes every cell. The pool is one flat byte tensor
    (``model_harness.real_bindings``); a page is
    ``model_harness.slot_bytes``."""
    env = on_cuda(n=2, sizing=Sizing.EXTENSIBLE)
    core = env.core
    core.prepare()
    bindings = core.bind_serving()
    original = torch.cuda.CUDAGraph.replay
    replays = []

    def replay(self):
        original(self)
        replays.append(1)
        h.slot_bytes(bindings, page).fill_(len(replays) % 255 + 1)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    error = h.call_with(core.serving_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", original)
    from support import txg_api

    if page == 1:
        assert error is None, t.text(error)
        assert len(txg_api.binding_of(env.model).cells) == 2 * env.n
        return
    assert error is not None, "the mismatching validation replay was accepted"
    assert "validation replay" in t.text(error), t.text(error)
    # Block 0 is checked before the outputs, so the page check reports the
    # change although every capture row gathers the null page.
    assert PAGE_MISMATCH.search(t.text(error)), t.text(error)
    assert not txg_api.binding_of(env.model).cells, "a cell was published"


@pytest.mark.gpu
# @spec TXG-CAP-005, TXG-CAP-008
def test_capture_stores_no_page_cuda(on_cuda):
    """Both captures, under a profiled budget with the extensible KV cache
    off (a one-block profiling KV cache)."""
    env = on_cuda(n=2, sizing=Sizing.PROFILED)
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
    env = on_cuda(n=2, sizing=Sizing.BLOCK_COUNT)
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
    env.core.startup(extend=False)
    env.core.extend(8)  # views of fewer pages over the captured addresses
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


@pytest.mark.gpu
# @spec TXG-CAP-018
def test_a_capture_call_addresses_only_block_zero_cuda(on_cuda):
    """TXG-CAP-018's CUDA leg: under the extensible KV cache, the serving
    capture's operations on the state pools (snapshot, scatter warm-up, each
    cell's eager reference, warm-ups, capture and validation replay, the
    comparison with the snapshot) address block 0 alone, read as the CPU
    test reads them, over the real CUDA pool; the capture then publishes
    every cell. The installed-engine leg, with only core's minimum blocks
    backed, is TXG-QUAL-028's."""
    env = on_cuda(n=2, sizing=Sizing.EXTENSIBLE)
    core = env.core
    core.prepare()
    bindings = core.bind_serving()
    with t.BlockAccesses(h.core_bindings(bindings).tensor) as rec:
        core.serving_capture()
    from support import txg_api

    assert len(txg_api.binding_of(env.model).cells) == 2 * env.n
    assert_only_block_zero(rec, "the serving capture")
