# SPDX-License-Identifier: Apache-2.0
"""The state layer and the model state's request bookkeeping on the target core
(REG-STATE-*, REG-MS-003, REG-MS-004, REG-MS-006, REG-MEM-001, REG-MEM-012,
FILE-BUF-003, FILE-BUF-004), on CPU.

The model state is the real class over a weightless model with a real
``NemotronStateLayer`` bound to a CPU pool laid out as core's ``MambaManager``
pool (``support.state_layer``). Steps that need the model's forward (fresh-page
initialization by the forward, an all-failed step, the envelope-header faults)
are in ``test_execution_memory`` on ``support.model_harness``.

Model-internal seams these tests read: the model state's request records
(``_requests``, ``_pending_removal``), its file buffers (``file_buffers``),
file progress (``_file_progress``, ``_failed_files``), the step's projection
snapshot (``bindings``, ``planning_failed``, ``state_batch``) and the
layer's views (``pages``, ``component_pools``).
"""

from __future__ import annotations

import dataclasses
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support import state_layer as sl

pytestmark = pytest.mark.engine


@pytest.fixture
def served(card_dir):
    return sl.served_config(card_dir)


@pytest.fixture
def layout(served):
    return sl.layout_of(served)


@pytest.fixture
def state(served, layout):
    return sl.weightless_state(served, layout)


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def file_request(state, req_id: str, slot: int, n_samples: int = 4000):
    """A file request at the server's geometry (FILE-PROMPT-001)."""
    from support import model_harness as h

    from vllm_nemotron_asr_streaming.lookahead import geometry_index, served_lookahead

    config = state.model.config
    geometry = geometry_index(served_lookahead(config))
    samples = np.full(n_samples, 0.01, dtype=np.float32)
    request = h.new_request(
        req_id,
        int(config.file_carrier_token_id),
        samples=samples,
        header=[geometry, 0, n_samples],
    )
    return h.with_slot(request, slot)


# ---------------------------------------------------------------------------
# State declaration


# @spec REG-STATE-001
def test_state_layer_declares_one_non_recomputable_mamba_page(layout):
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.kv_cache_interface import MambaSpec

    layer = sl.layer(layout, bind=False)
    config = layer.get_kv_cache_config(sl.forward_context_config(max_model_len=777))
    assert config.manager_cls is MambaManager
    assert config.manager_config.recomputable is False
    spec = config.spec
    assert isinstance(spec, MambaSpec)
    assert spec.block_size == 777
    assert spec.state_alignment == 1
    assert spec.shapes == layout.shapes and spec.dtypes == layout.dtypes
    assert spec.page_size_bytes == layout.page_size_bytes


# @spec REG-STATE-001
def test_layout_page_equals_the_manifest_total(served, layout):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        manifest_page_bytes,
    )

    assert layout.page_size_bytes == manifest_page_bytes(served.model_config.hf_config)


# @spec REG-STATE-001
def test_a_page_size_mismatch_is_refused_naming_both_sizes(layout):
    wrong = dataclasses.replace(layout, page_size_bytes=layout.page_size_bytes + 8)
    layer = sl.layer(wrong, bind=False)
    with pytest.raises(ValueError) as excinfo:
        layer.get_kv_cache_config(sl.forward_context_config())
    message = str(excinfo.value)
    assert str(layout.page_size_bytes) in message, message
    assert str(wrong.page_size_bytes) in message, message


# @spec REG-STATE-009
def test_the_spec_and_the_layer_are_persistent_state_mamba_types(layout):
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    layer = sl.layer(layout, bind=False)
    spec = layer.get_kv_cache_config(sl.forward_context_config()).spec
    assert spec.mamba_type == MambaAttentionBackendEnum.PERSISTENT_STATE
    assert layer.mamba_type == MambaAttentionBackendEnum.PERSISTENT_STATE


def test_layer_registers_under_its_prefix(layout):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import (
        NemotronStateLayer,
    )

    config = sl.forward_context_config()
    layer = NemotronStateLayer(layout, prefix=sl.LAYER, vllm_config=config)
    assert config.compilation_config.static_forward_context == {sl.LAYER: layer}
    with pytest.raises(ValueError, match="duplicate"):
        NemotronStateLayer(layout, prefix=sl.LAYER, vllm_config=config)


# ---------------------------------------------------------------------------
# Views (REG-STATE-018)


# @spec REG-STATE-018
def test_bound_views_are_the_layout_descriptors_at_their_offsets(layout):
    layer = sl.layer(layout, bind=False)
    kv = sl.pool(layout)
    layer.bind_kv_cache(kv)
    assert tuple(layer.component_pools) == layout.names
    assert layer.num_blocks == sl.BLOCKS
    raw = kv.view(torch.uint8).reshape(sl.BLOCKS, layout.page_size_bytes)
    for descriptor in layout.descriptors:
        view = layer.component_pools[descriptor.name]
        assert view.shape == (sl.BLOCKS, *descriptor.shape)
        assert view.dtype == descriptor.dtype
        view[3].fill_(1)
        start = descriptor.offset_bytes
        page = raw[3, start : start + descriptor.size_bytes]
        assert page.view(descriptor.dtype).eq(1).all()
        view[3].zero_()
    assert not raw.any()


# @spec REG-STATE-018
def test_views_are_views_of_the_bound_storage_never_copies(layout):
    layer = sl.layer(layout, bind=False)
    kv = sl.pool(layout)
    layer.bind_kv_cache(kv)
    assert _storage(layer.pages) == _storage(kv)
    for view in layer.component_pools.values():
        assert _storage(view) == _storage(kv)


# @spec REG-STATE-018
def test_a_read_while_the_cache_is_current_does_not_rebuild_it(layout):
    layer = sl.layer(layout)
    pools, pages = layer.component_pools, layer.pages
    assert layer.component_pools is pools
    assert layer.pages is pages


# @spec REG-STATE-018
def test_bind_kv_cache_drops_the_cache_and_the_next_read_rebuilds_it(layout):
    layer = sl.layer(layout)
    first = layer.component_pools
    kv = sl.pool(layout, blocks=sl.BLOCKS + 2)
    layer.bind_kv_cache(kv)
    assert layer.num_blocks == sl.BLOCKS + 2
    assert layer.component_pools is not first
    assert all(_storage(v) == _storage(kv) for v in layer.component_pools.values())


# @spec REG-STATE-018
def test_a_read_that_finds_another_kv_cache_bound_rebuilds_from_it(layout):
    """Core replaces the layer's ``kv_cache`` object itself (``MambaBase``'s
    binding, or its teardown): the cache is keyed on that object's
    identity, so the next read builds the views of the object now bound."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    layer = sl.layer(layout)
    first = layer.component_pools
    kv = sl.pool(layout)
    MambaBase.bind_kv_cache(layer, kv)  # core's binding, not the layer's
    pools = layer.component_pools
    assert pools is not first
    assert all(_storage(v) == _storage(kv) for v in pools.values())
    assert _storage(layer.pages) == _storage(kv)


# @spec REG-STATE-018
def test_a_read_after_a_drop_with_nothing_bound_raises(layout):
    """Core's teardown leaves an empty ``kv_cache`` (``clear_layer_kv_caches``
    sets ``[]`` for the tuple ``MambaBase`` binds)."""
    from vllm.v1.worker.utils import clear_layer_kv_caches

    layer = sl.layer(layout)
    assert layer.component_pools
    clear_layer_kv_caches([layer])
    with pytest.raises(Exception):  # noqa: B017 - any refusal, no stale view
        layer.component_pools  # noqa: B018
    with pytest.raises(Exception):  # noqa: B017
        layer.pages  # noqa: B018


# @spec REG-STATE-018, REG-MEM-021
def test_clear_cudagraphs_drops_the_layers_views(state):
    layer = state.model._persistent_state_layer
    views = [weakref.ref(v) for v in layer.component_pools.values()]
    pages = weakref.ref(layer.pages)
    state.clear_cudagraphs()
    gc.collect()
    assert all(ref() is None for ref in views), "a view survived clear_cudagraphs"
    assert pages() is None, "the page view survived clear_cudagraphs"
    # the next read rebuilds from the still bound kv_cache
    kv = layer.kv_cache
    assert all(_storage(v) == _storage(kv[0]) for v in layer.component_pools.values())


# @spec REG-STATE-018
def test_after_clear_and_cores_teardown_nothing_keeps_the_profiling_kv_cache(
    served, layout
):
    """The CPU analogue of REG-STATE-019: bind a profiling KV cache, read the
    views as a profiling step would, then core's ``clear_cudagraphs`` and its
    teardown (``_state_group_layouts`` emptied, the layer's ``kv_cache``
    cleared, ``gc.collect``): a weak reference to the cache's storage has
    expired."""
    from torch.multiprocessing.reductions import StorageWeakRef
    from vllm.v1.worker.utils import clear_layer_kv_caches

    state = sl.weightless_state(served, layout, bound=False)
    layer = state.model._persistent_state_layer
    kv = sl.pool(layout)
    ref = StorageWeakRef(kv.untyped_storage())
    layer.bind_kv_cache(kv)
    state._state_group_layouts = (SimpleNamespace(kv=kv),)  # core installs it
    del kv
    assert layer.component_pools and layer.pages is not None
    sl.inputs(state, ("req_0_random",), (0,))  # a profiling dummy step
    state.clear_cudagraphs()
    state._state_group_layouts = ()
    clear_layer_kv_caches([layer])
    gc.collect()
    assert ref.expired(), "an object of the plugin kept the profiling KV cache"


# ---------------------------------------------------------------------------
# Page initialization (REG-STATE-010; the forward's part is in
# test_execution_memory)


# @spec REG-STATE-010
def test_page_initialization_zeroes_the_page_and_writes_the_blank_label(served, layout):
    layer = sl.layer(layout)
    pages = layer.pages
    pages.fill_(-1)
    layer.initialize_pages([2, 4])
    blank = int(served.model_config.hf_config.num_asr_labels)
    blank_descriptors = [d for d in layout.descriptors if not d.zero_initialized]
    assert blank_descriptors, "the book's blank-label entries"
    for slot in (2, 4):
        for descriptor in layout.descriptors:
            row = layer.component_pools[descriptor.name][slot]
            expected = 0 if descriptor.zero_initialized else blank
            assert row.eq(expected).all(), descriptor.name
    for slot in (0, 1, 3, 5):
        assert pages[slot].eq(-1).all()


# @spec REG-STATE-010
def test_a_fresh_page_is_initialized_by_the_step_not_by_planning(state):
    layer = state.model._persistent_state_layer
    layer.pages.fill_(-1)
    state.add_request(0, sl.request("a", 3))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a",), (0,))
    assert layer.pages.eq(-1).all(), "planning wrote a page"
    assert snapshot.state_batch.fresh_slots == (3,)
    snapshot.state_batch.initialize_fresh_pages()
    for descriptor in state.model._state_spec.descriptors:
        if descriptor.zero_initialized:
            assert layer.component_pools[descriptor.name][3].eq(0).all()
    assert layer.pages[2].eq(-1).all()


# @spec REG-STATE-010
def test_a_row_is_fresh_exactly_when_its_num_computed_tokens_is_zero(state):
    layer = state.model._persistent_state_layer
    state.add_request(0, sl.request("a", 1))
    sl.attn(state)
    sl.inputs(state, ("a",), (0,))
    layer.pages[1].fill_(5)
    sl.attn(state)
    snapshot = sl.inputs(state, ("a",), (1,))
    assert [b.fresh for b in snapshot.bindings] == [False]
    snapshot.state_batch.initialize_fresh_pages()
    assert layer.pages[1].eq(5).all(), "a continuing row's page was initialized"


# ---------------------------------------------------------------------------
# Step classification (REG-MEM-001, REG-MEM-012)


# @spec REG-MEM-001, REG-MEM-012
def test_a_dummy_step_without_prepare_attn_is_the_profiling_dummy_step(state):
    snapshot = sl.inputs(state, ("req_0_random",), (0,))
    assert snapshot.dummy_run and snapshot.is_profile and snapshot.no_page_io
    assert snapshot.state_batch is None


# @spec REG-MEM-001, REG-MEM-012
def test_a_dummy_step_after_prepare_attn_is_not_profiling(state):
    sl.attn(state)
    snapshot = sl.inputs(state, ("req_0_random", "req_1_random"), (0, 0))
    assert snapshot.dummy_run and not snapshot.is_profile and snapshot.no_page_io


# @spec REG-MEM-001
def test_each_prepare_inputs_replaces_the_earlier_classification(state):
    sl.attn(state)
    assert not sl.inputs(state, ("req_0_a",), (0,)).is_profile
    # the next execute_model's profiling step calls no prepare_attn
    assert sl.inputs(state, ("req_0_b",), (0,)).is_profile
    state.add_request(0, sl.request("a", 1))
    sl.attn(state)
    real = sl.inputs(state, ("a",), (0,))
    assert not real.dummy_run and not real.is_profile


# @spec REG-MEM-001, REG-MEM-012
def test_a_step_with_one_unrecorded_row_is_a_dummy_step(state):
    state.add_request(0, sl.request("a", 1))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a", "req_1_random"), (0, 0))
    assert snapshot.dummy_run and not snapshot.is_profile
    assert snapshot.state_batch is None


# @spec REG-MEM-001, REG-MEM-012
def test_a_cached_row_the_model_state_retired_makes_a_dummy_step(state):
    """A row whose request the model state retired (after deciding a pending
    removal) is a dummy step's row: no refusal, no page read."""
    state.add_request(0, sl.request("a", 1))
    state.remove_request("a")
    snapshot = sl.inputs(state, ("a",), (3,))
    assert snapshot.dummy_run and snapshot.state_batch is None


# @spec REG-MEM-001
def test_a_step_of_recorded_rows_is_a_real_step(state):
    state.add_request(0, sl.request("a", 1))
    state.add_request(1, sl.request("b", 2))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a", "b"), (0, 0))
    assert not snapshot.dummy_run and not snapshot.is_profile
    assert not snapshot.no_page_io
    assert [b.slot_id for b in snapshot.bindings] == [1, 2]
    assert [b.fresh for b in snapshot.bindings] == [True, True]
    assert snapshot.state_batch.slot_indices == (1, 2)


# ---------------------------------------------------------------------------
# Slots and admission generations (REG-STATE-002, REG-STATE-003)


# @spec REG-STATE-002
def test_add_request_takes_the_slot_from_block_ids(state):
    state.add_request(0, sl.request("a", 4))
    assert state._requests["a"].handle.slot == 4


@pytest.mark.parametrize(
    "block_ids",
    [([],), ([2, 3],), ([2], [3]), ()],
    ids=["no-block", "two-blocks", "two-groups", "no-group"],
)
# @spec REG-STATE-002, REG-STATE-015
def test_a_request_without_exactly_one_block_in_one_group_raises(state, block_ids):
    with pytest.raises(Exception, match="'a'|a\\b"):
        state.add_request(0, sl.request("a", None, block_ids=block_ids))


# @spec REG-STATE-002
def test_every_fresh_add_takes_a_new_admission_generation(state):
    state.add_request(0, sl.request("a", 1))
    state.add_request(1, sl.request("b", 2))
    generations = {r: state._requests[r].handle.generation for r in ("a", "b")}
    assert generations["a"] != generations["b"]


@pytest.mark.parametrize("slot", [0, sl.BLOCKS], ids=["null-block", "past-the-pool"])
# @spec REG-STATE-003, REG-STATE-015
def test_a_slot_outside_the_bound_pages_raises_before_any_page_is_touched(state, slot):
    layer = state.model._persistent_state_layer
    layer.pages.fill_(-3)
    before = layer.pages.clone()
    state.add_request(0, sl.request("a", slot))
    sl.attn(state)
    with pytest.raises(Exception, match=str(slot)):
        sl.inputs(state, ("a",), (0,))
    assert torch.equal(layer.pages, before)


# @spec REG-STATE-003
def test_a_step_reads_the_views_as_bound_at_that_step(state, layout):
    """A rebinding between two steps (the extensible KV cache grows): the
    second step's rows are the new binding's pages."""
    layer = state.model._persistent_state_layer
    state.add_request(0, sl.request("a", 2))
    sl.attn(state)
    sl.inputs(state, ("a",), (0,))
    kv = sl.pool(layout, blocks=sl.BLOCKS + 2)
    layer.bind_kv_cache(kv)
    sl.attn(state)
    snapshot = sl.inputs(state, ("a",), (1,))
    for tensor in _tensors(snapshot.state_batch.pools):
        assert _storage(tensor) == _storage(kv), "a step read an earlier binding"


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            yield from _tensors(getattr(value, f.name))


# ---------------------------------------------------------------------------
# Commit check and projection (REG-STATE-004, REG-STATE-005)


class _Sink:
    def __init__(self) -> None:
        self.reserved: list = []

    def reserve(self, plan):
        self.reserved.append(plan)
        return plan


def _plan(*pairs):
    return SimpleNamespace(
        bindings=[SimpleNamespace(request_id=r, block_id=s) for r, s in pairs]
    )


def _two_row_batch(state):
    state.add_request(0, sl.request("a", 1))
    state.add_request(1, sl.request("b", 2))
    sl.attn(state)
    return sl.inputs(state, ("a", "b"), (0, 0)).state_batch


# @spec REG-STATE-004
def test_a_commit_matching_the_steps_bindings_is_reserved(state):
    sink = _Sink()
    _two_row_batch(state).commit_sink(sink).reserve(_plan(("a", 1), ("b", 2)))
    assert len(sink.reserved) == 1


@pytest.mark.parametrize(
    "pairs",
    [(("b", 2), ("a", 1)), (("a", 1), ("b", 3)), (("a", 1),)],
    ids=["order", "slot", "rows"],
)
# @spec REG-STATE-004
def test_a_commit_differing_from_the_steps_bindings_writes_nothing(state, pairs):
    sink = _Sink()
    commit = _two_row_batch(state).commit_sink(sink)
    try:
        commit.reserve(_plan(*pairs))
    except Exception:  # noqa: BLE001 - refusing is allowed; writing is not
        pass
    assert sink.reserved == [], "a mismatched commit reached the state"


# @spec REG-STATE-005
def test_the_projection_takes_the_views_and_the_layout_only(state, layout):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        project_nemotron_state_pools,
    )

    layer = state.model._persistent_state_layer
    views = dict(layer.component_pools)
    projected = project_nemotron_state_pools(
        views, layout, n_layers=int(state.model.config.n_layers)
    )
    tensors = list(_tensors(projected))
    assert tensors, "the projection named no state tensor"
    for tensor in tensors:
        assert _storage(tensor) == _storage(layer.pages), "the projection copied"


# ---------------------------------------------------------------------------
# Request lifetime (REG-STATE-011, REG-STATE-012, REG-STATE-016)


def _compacted(req_id: str, slot: int, *new_input: int):
    """A continuing turn as the target core delivers it: one frontier token
    then the turn's input, ``num_computed_tokens`` 1 (REG-STATE-008)."""
    return sl.request(req_id, slot, computed=1, tokens=(sl.STREAM_TOKEN, *new_input))


# @spec REG-STATE-011, REG-STATE-008
def test_a_continuing_turn_keeps_the_generation_and_replaces_the_prompt(state):
    state.add_request(0, sl.request("a", 3))
    generation = state._requests["a"].handle.generation
    state.remove_request("a")
    turn = _compacted("a", 3, 9, 11)
    state.add_request(0, turn)
    record = state._requests["a"]
    assert record.handle.slot == 3 and record.handle.generation == generation
    assert record.prompt == (sl.STREAM_TOKEN, 9, 11)
    assert record.metadata is turn
    assert state._pending_removal is None
    assert state.model._registry.prunes == []


@pytest.mark.parametrize("call", ["prepare_attn", "prepare_inputs", "remove", "add"])
# @spec REG-STATE-011
def test_a_finished_stream_retires_at_the_next_call(state, call):
    state.add_request(0, sl.request("a", 3))
    state.remove_request("a")
    assert "a" in state._requests and state._pending_removal == "a"
    if call == "prepare_attn":
        sl.attn(state)
    elif call == "prepare_inputs":
        sl.inputs(state, ("req_0_random",), (0,))
    elif call == "remove":
        state.remove_request("never-added")
    else:
        state.add_request(1, sl.request("b", 4))
    assert "a" not in state._requests and state._pending_removal is None
    assert state.model._registry.prunes[:1] == [()]


# @spec REG-STATE-011
def test_a_reused_request_id_starts_fresh(state):
    state.add_request(0, sl.request("a", 3))
    generation = state._requests["a"].handle.generation
    state.remove_request("a")
    state.add_request(0, sl.request("a", 3, computed=0))
    assert state.model._registry.prunes == [()]
    assert state._requests["a"].handle.generation != generation


# @spec REG-STATE-011
def test_a_reused_request_id_on_another_slot_starts_fresh(state):
    state.add_request(0, sl.request("a", 3))
    state.remove_request("a")
    state.add_request(0, sl.request("a", 4))
    assert state._requests["a"].handle.slot == 4
    assert state.model._registry.prunes == [()]


# @spec REG-STATE-011
def test_a_second_removal_retires_the_first(state):
    state.add_request(0, sl.request("a", 3))
    state.add_request(1, sl.request("b", 4))
    state.remove_request("a")
    state.remove_request("b")
    assert "a" not in state._requests and state._pending_removal == "b"


# @spec REG-STATE-011
def test_a_zero_token_step_defers_retirement_to_the_next_steps_first_call(state):
    state.add_request(0, sl.request("a", 3))
    state.add_request(1, sl.request("b", 4))
    # step 1 finishes "a" and schedules nothing: core returns before
    # prepare_attn and prepare_inputs
    state.remove_request("a")
    assert "a" in state._requests
    # step 2's first call is core's removal-before-add of a new request
    state.remove_request("c")
    assert "a" not in state._requests
    state.add_request(0, sl.request("c", 3))
    sl.attn(state)
    snapshot = sl.inputs(state, ("b", "c"), (1, 0))
    assert [b.request_id for b in snapshot.bindings] == ["b", "c"]


@pytest.mark.parametrize("computed", [0, 1], ids=["fresh", "computed"])
# @spec REG-STATE-012, REG-STATE-015
def test_add_request_for_a_live_request_raises_naming_it(state, computed):
    state.add_request(0, sl.request("live", 3))
    with pytest.raises(Exception, match="live"):
        state.add_request(0, sl.request("live", 3, computed=computed))


@pytest.mark.parametrize("pending", [False, True], ids=["unknown", "other-slot"])
# @spec REG-STATE-012, REG-STATE-015
def test_add_request_with_prior_state_that_continues_nothing_raises(state, pending):
    if pending:
        state.add_request(0, sl.request("x", 3))
        state.remove_request("x")
        bad = _compacted("x", 4, 9)
    else:
        bad = _compacted("x", 3, 9)
    with pytest.raises(Exception, match="x"):
        state.add_request(0, bad)


# @spec REG-STATE-016, FILE-BUF-003
def test_remove_request_retires_a_file_request_within_the_call(state):
    state.add_request(0, file_request(state, "f", 2))
    assert "f" in state.file_buffers._entries and "f" in state._file_progress
    state._failed_files.add("f")
    state.remove_request("f")
    assert "f" not in state._requests
    assert state._pending_removal is None, "a file request was held pending"
    assert "f" not in state.file_buffers._entries
    assert "f" not in state._file_progress and "f" not in state._failed_files


# @spec REG-STATE-016
def test_a_file_removal_does_not_decide_or_replace_a_pending_stream(state):
    """A file's removal retires it within the call; a stream held pending
    before it is decided by that call too (any ``remove_request`` decides an
    earlier pending removal)."""
    state.add_request(0, sl.request("s", 1))
    state.add_request(1, file_request(state, "f", 2))
    state.remove_request("s")
    state.remove_request("f")
    assert "s" not in state._requests and "f" not in state._requests
    assert state._pending_removal is None


# @spec FILE-BUF-004
def test_a_new_generation_on_a_slot_drops_an_earlier_generations_buffer(state):
    """A buffer recorded under an earlier admission generation on the same
    slot is dropped when ``add_request`` gives the slot a new generation."""
    state.add_request(0, file_request(state, "f1", 2))
    # a stale record on slot 2 that no removal cleared
    state._requests.pop("f1")
    state.add_request(1, file_request(state, "f2", 2))
    assert "f1" not in state.file_buffers._entries
    assert "f2" in state.file_buffers._entries


# ---------------------------------------------------------------------------
# Recorded prompt and multi-token rows (REG-STATE-008, REG-STATE-014)


# @spec REG-STATE-008
def test_a_row_reads_its_token_from_the_compacted_prompt(state):
    state.add_request(0, sl.request("a", 3))
    sl.attn(state)
    sl.inputs(state, ("a",), (0,))
    state.remove_request("a")
    state.add_request(0, _compacted("a", 3, 9, 11))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a",), (1,))
    assert snapshot.token_ids == (9,)
    sl.attn(state)
    assert sl.inputs(state, ("a",), (2,)).token_ids == (11,)


# @spec REG-STATE-008
def test_a_position_past_the_prompt_is_a_sampled_non_chunk_token(state):
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import SAMPLED_TOKEN

    state.add_request(0, sl.request("a", 3))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a",), (1,))
    assert snapshot.token_ids == (SAMPLED_TOKEN,)
    assert snapshot.token_ids[0] != int(state.model.config.audio_chunk_token_id)


# @spec REG-STATE-008, REG-STATE-014, REG-STATE-015
def test_a_window_crossing_the_end_of_the_prompt_fails_the_row_at_planning(state):
    state.add_request(0, sl.request("a", 3, tokens=(sl.STREAM_TOKEN, 9)))
    state.add_request(1, sl.request("b", 4))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a", "b"), (1, 0), scheduled=(2, 1))
    assert snapshot.planning_failed == {"a"}
    assert [b.request_id for b in snapshot.bindings] == ["a", "b"]


# @spec REG-STATE-014, REG-STATE-015
def test_a_multi_token_row_fails_alone_at_planning(state):
    state.add_request(0, sl.request("a", 3, tokens=(sl.STREAM_TOKEN,) * 4))
    state.add_request(1, sl.request("b", 4))
    sl.attn(state)
    snapshot = sl.inputs(state, ("a", "b"), (0, 0), scheduled=(3, 1))
    assert not snapshot.dummy_run
    assert snapshot.planning_failed == {"a"}
    assert "b" not in snapshot.planning_failed


# ---------------------------------------------------------------------------
# initialize_kv_cache (REG-STATE-013)


def _kv_config(*groups):
    return SimpleNamespace(
        num_blocks=sl.BLOCKS,
        kv_cache_groups=[SimpleNamespace(layer_names=list(g)) for g in groups],
    )


@pytest.mark.parametrize(
    "groups",
    [((sl.LAYER,), ("other",)), ((sl.LAYER, "other"),), (("other",),), ()],
    ids=["two-groups", "extra-layer", "other-layer", "no-group"],
)
# @spec REG-STATE-013
def test_initialize_kv_cache_refuses_any_other_grouping_naming_it(state, groups):
    with pytest.raises(Exception) as excinfo:
        state.initialize_kv_cache(_kv_config(*groups), None)
    for group in groups:
        for name in group:
            assert name in str(excinfo.value), str(excinfo.value)


# @spec REG-STATE-013
def test_initialize_kv_cache_refuses_unbound_views_naming_the_groups(served, layout):
    state = sl.weightless_state(served, layout, bound=False)
    with pytest.raises(Exception) as excinfo:
        state.initialize_kv_cache(_kv_config((sl.LAYER,)), None)
    assert sl.LAYER in str(excinfo.value), str(excinfo.value)


# @spec REG-STATE-013
def test_initialize_kv_cache_accepts_the_one_state_layer_group(state):
    state.initialize_kv_cache(_kv_config((sl.LAYER,)), None)


# ---------------------------------------------------------------------------
# Row failure (REG-MS-003 to REG-MS-006, REG-STATE-017)


def _failing_step(state, *, failed=(), planning=(), raise_in=None):
    """A real step of rows ``a`` and ``b`` whose collection reports
    ``failed`` and whose plan failed ``planning``."""
    state.add_request(0, sl.request("a", 1))
    state.add_request(1, sl.request("b", 2))
    sl.attn(state)
    sl.inputs(state, ("a", "b"), (0, 0))
    state._snapshot = dataclasses.replace(
        state._snapshot, planning_failed=set(planning)
    )

    def collect():
        if raise_in == "collect":
            raise RuntimeError("collection failed")
        return sl.collection(failed=failed, statuses={r: 1 for r in failed})

    def stats():
        if raise_in == "stats":
            raise RuntimeError("statistics failed")

    state.model.collect_commit_status = collect
    state.model.consume_batch_stats = stats


# @spec REG-MS-003, SESS-MODEL-003
def test_postprocess_records_collected_failures_without_raising(state):
    _failing_step(state, failed={"b"})
    sl.postprocess(state, 2)
    assert state.take_failed_requests() == {"b"}


# @spec REG-MS-003, TXG-FAIL-008
def test_postprocess_records_the_planning_failed_set(state):
    _failing_step(state, failed={"b"}, planning={"a"})
    sl.postprocess(state, 2)
    assert state.take_failed_requests() == {"a", "b"}


# @spec REG-MS-003
def test_a_dummy_step_records_no_failed_request(state):
    sl.inputs(state, ("req_0_random",), (0,))
    state.model.collect_commit_status = lambda: sl.collection(failed={"x"})
    state.model.consume_batch_stats = lambda: None
    sl.postprocess(state)
    assert state.take_failed_requests() is None


# @spec REG-MS-004
def test_take_failed_requests_returns_and_clears_the_record(state):
    assert state.take_failed_requests() is None
    _failing_step(state, failed={"a"})
    sl.postprocess(state, 2)
    assert state.take_failed_requests() == {"a"}
    assert state.take_failed_requests() is None


# @spec REG-MS-004
def test_take_failed_requests_returns_every_step_since_the_previous_call(state):
    _failing_step(state, failed={"a"})
    sl.postprocess(state, 2)
    state.remove_request("a")
    state.remove_request("b")
    state.add_request(0, sl.request("c", 3))
    sl.attn(state)
    sl.inputs(state, ("c",), (0,))
    state.model.collect_commit_status = lambda: sl.collection(
        failed={"c"}, statuses={"c": 2}
    )
    sl.postprocess(state)
    assert state.take_failed_requests() == {"a", "c"}


@pytest.mark.parametrize("raise_in", ["collect", "stats"])
# @spec REG-MS-006
def test_a_raising_postprocess_drops_the_plan_and_records_nothing(state, raise_in):
    _failing_step(state, failed={"a"}, planning={"b"}, raise_in=raise_in)
    with pytest.raises(RuntimeError):
        sl.postprocess(state, 2)
    assert state._snapshot is None, "the step's plan survived the raise"
    assert state.take_failed_requests() is None


# @spec REG-STATE-003, REG-STATE-018
def test_a_real_step_refuses_a_layer_core_has_not_bound(served, layout):
    state = sl.weightless_state(served, layout, bound=False)
    state.add_request(0, sl.request("a", 1))
    sl.attn(state)
    with pytest.raises(Exception):  # noqa: B017 - no view to read
        sl.inputs(state, ("a",), (0,))


# @spec REG-ENG-019
def test_the_configuration_hook_refuses_native_burst_naming_its_removal(
    engine_config, platform
):
    """In the hook configuration on CUDA with the default arms, where every
    other condition holds: the refusal states that native burst is not
    supported on the state layer and names its removal. Without it the same
    configuration is accepted."""
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    platform(True)

    def config(hf):
        return engine_config(
            enforce_eager=False,
            compilation_config=SimpleNamespace(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.NONE,
                use_breakable_cudagraph=None,
            ),
            hf=hf,
        )

    refused = config({"experimental_native_burst": True, "num_lookahead_tokens": 1})
    with pytest.raises(ValueError) as excinfo:
        Nemotron3_5AsrStreamingConfig.verify_and_update_config(refused)
    message = str(excinfo.value)
    assert "experimental_native_burst" in message, message
    assert "state layer" in message.lower(), message
    assert "remov" in message.lower(), message
    Nemotron3_5AsrStreamingConfig.verify_and_update_config(
        config({"num_lookahead_tokens": 1})
    )


class _StagedSink:
    """A commit sink holding one staged step's status reports."""

    def __init__(self, reports) -> None:
        self.reports = list(reports)
        self.has_staged = True

    def collect(self):
        self.has_staged = False
        return self.reports, None, True


class _GenerationRegistry:
    """The model's session registry reduced to what collection asks: each
    request's current admission generation; it records published progress."""

    def __init__(self, current: dict[str, int]) -> None:
        self.current = current
        self.published: list[str] = []

    def resolve_status(self, *, request_id, admission_generation, clean):
        return self.current.get(request_id) == admission_generation

    def publish_progress(self, request_id, generation, raw, *, starts_chunk):
        self.published.append(request_id)
        return raw


@pytest.mark.parametrize("current", [{}, {"a": 2}], ids=["retired", "re-admitted"])
# @spec REG-STATE-017, REG-MS-003
def test_a_report_for_a_non_current_incarnation_fails_its_request(state, current):
    """A clean report whose request the registry no longer holds at the
    report's admission generation resolves nonzero: failed, no queue count
    and no published progress; a current report beside it is unaffected."""
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        SessionStatusReport,
    )

    model = state.model
    registry = _GenerationRegistry({**current, "b": 3})
    model._registry = registry
    model._commit_sink = _StagedSink(
        [
            SessionStatusReport("a", 1, 0, progress_key=(1, 0, 1), queue_count=4),
            SessionStatusReport("b", 3, 0, progress_key=(1, 0, 1), queue_count=2),
        ]
    )
    collection = model.collect_commit_status()
    assert collection.statuses["a"] != 0 and "a" in collection.failed
    assert collection.statuses["b"] == 0 and "b" not in collection.failed
    assert [c.request_id for c in collection.queue_counts] == ["b"]
    assert registry.published == ["b"]
