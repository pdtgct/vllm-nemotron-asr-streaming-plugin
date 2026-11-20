# SPDX-License-Identifier: Apache-2.0
"""The PR 6 proof of concept's state layer and model-state bookkeeping, on
CPU: the page-size check, the layer's views and page initialization, step
identity (dummy, profiling dummy, real) and the pending-removal decision of
the request lifetime.

The model state is the real class, built as core builds it, over a model
without weights whose few attributes the hooks read are set directly (as
``tests/transcription/conftest.py`` does); its state layer is the real
``NemotronStateLayer`` bound to a CPU pool laid out as core's ``MambaManager``
pool.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.engine

LAYER = "nemotron.state"
STREAM_TOKEN = 7
BLOCKS = 6


def _forward_context_config(max_model_len: int = 2048) -> Any:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


@pytest.fixture
def served(tmp_path):
    """The served ``VllmConfig`` of the synthetic card."""
    from support.card import write_card
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from vllm_nemotron_asr_streaming import register

    register()
    model_config = ModelConfig(
        model=str(write_card(tmp_path / "card")),
        config_format="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": 6},
        enforce_eager=True,
        dtype="float32",
    )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
    )


@pytest.fixture
def layout(served):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        build_nemotron_state_layout,
    )

    return build_nemotron_state_layout(served.model_config.hf_config)


def _pool(layout: Any, blocks: int = BLOCKS) -> torch.Tensor:
    return torch.zeros((blocks, 1, 1, layout.page_size_bytes), dtype=torch.int8)


def _layer(layout: Any, *, bind: bool = True) -> Any:
    from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import (
        NemotronStateLayer,
    )

    layer = NemotronStateLayer(
        layout, prefix=LAYER, vllm_config=_forward_context_config()
    )
    if bind:
        layer.bind_kv_cache(_pool(layout))
    return layer


class Registry:
    """Records every prune's keep set."""

    def __init__(self) -> None:
        self.prunes: list[tuple[str, ...]] = []

    def prune(self, keep: Any) -> None:
        self.prunes.append(tuple(sorted(keep)))


@pytest.fixture
def state(served, layout):
    """The real model state over a weightless model with a bound state layer."""
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    model = Nemotron3_5AsrStreamingForRNNT.__new__(Nemotron3_5AsrStreamingForRNNT)
    torch.nn.Module.__init__(model)
    model.config = served.model_config.hf_config
    model._state_spec = layout
    model._persistent_state_layer = _layer(layout)
    model._registry = Registry()
    model._native_burst_handoff = None
    cls = Nemotron3_5AsrStreamingForRNNT.get_model_state_cls()
    return cls(served, model, None, torch.device("cpu"))


def request(
    req_id: str, slot: int, *, computed: int = 0, tokens: tuple[int, ...] = ()
) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    prompt = list(tokens or (STREAM_TOKEN,) * (computed + 1))
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_ids=([slot],),
        num_computed_tokens=computed,
        lora_request=None,
        prefill_token_ids=prompt,
    )


def attn(state: Any) -> None:
    from vllm.config import CUDAGraphMode

    assert state.prepare_attn(None, CUDAGraphMode.NONE, (), None, [], None) == {}


def batch(req_ids: tuple[str, ...], computed: tuple[int, ...]) -> Any:
    n = len(req_ids)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_scheduled_tokens=np.ones(n, dtype=np.int32),
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
    )


def inputs(state: Any, req_ids: tuple[str, ...], computed: tuple[int, ...]) -> Any:
    projection = state.prepare_inputs(batch(req_ids, computed), None)
    return projection["persistent_state_projection"]


# ---------------------------------------------------------------------------
# State declaration (algorithm 1)


def test_state_layer_declares_one_non_recomputable_mamba_page(layout):
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.kv_cache_interface import MambaSpec

    layer = _layer(layout, bind=False)
    config = layer.get_kv_cache_config(_forward_context_config(max_model_len=777))
    assert config.manager_cls is MambaManager
    assert config.manager_config.recomputable is False
    spec = config.spec
    assert isinstance(spec, MambaSpec)
    assert spec.block_size == 777
    assert spec.state_alignment == 1
    assert spec.shapes == layout.shapes and spec.dtypes == layout.dtypes
    assert spec.page_size_bytes == layout.page_size_bytes


def test_layout_page_equals_the_manifest_total(served, layout):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        manifest_page_bytes,
    )

    assert layout.page_size_bytes == manifest_page_bytes(served.model_config.hf_config)


def test_page_size_mismatch_is_refused(layout):
    wrong = dataclasses.replace(layout, page_size_bytes=layout.page_size_bytes + 8)
    layer = _layer(wrong, bind=False)
    with pytest.raises(ValueError, match="MambaSpec page"):
        layer.get_kv_cache_config(_forward_context_config())


def test_layer_registers_under_its_prefix(layout):
    from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import (
        NemotronStateLayer,
    )

    config = _forward_context_config()
    layer = NemotronStateLayer(layout, prefix=LAYER, vllm_config=config)
    assert config.compilation_config.static_forward_context == {LAYER: layer}
    with pytest.raises(ValueError, match="duplicate"):
        NemotronStateLayer(layout, prefix=LAYER, vllm_config=config)


def test_bound_views_are_the_layout_descriptors_at_their_offsets(layout):
    layer = _layer(layout, bind=False)
    pool = _pool(layout)
    layer.bind_kv_cache(pool)
    assert tuple(layer.component_pools) == layout.names
    assert layer.num_blocks == BLOCKS
    raw = pool.view(torch.uint8).reshape(BLOCKS, layout.page_size_bytes)
    for descriptor in layout.descriptors:
        view = layer.component_pools[descriptor.name]
        assert view.shape == (BLOCKS, *descriptor.shape)
        assert view.dtype == descriptor.dtype
        view[3].fill_(1)
        start = descriptor.offset_bytes
        page = raw[3, start : start + descriptor.size_bytes]
        assert page.view(descriptor.dtype).eq(1).all()
        view[3].zero_()
    assert not raw.any()


def test_rebinding_replaces_the_views(layout):
    layer = _layer(layout)
    first = layer.component_pools
    layer.bind_kv_cache(_pool(layout, blocks=BLOCKS + 2))
    assert layer.num_blocks == BLOCKS + 2
    assert layer.component_pools is not first


def test_page_initialization_zeroes_the_page_and_writes_the_blank_label(served, layout):
    layer = _layer(layout)
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


# ---------------------------------------------------------------------------
# Step identity (algorithm 3)


def test_dummy_step_without_prepare_attn_is_the_profiling_dummy_step(state):
    snapshot = inputs(state, ("req_0_random",), (0,))
    assert snapshot.dummy_run and snapshot.is_profile and snapshot.no_page_io


def test_dummy_step_after_prepare_attn_is_not_profiling(state):
    attn(state)
    snapshot = inputs(state, ("req_0_random", "req_1_random"), (0, 0))
    assert snapshot.dummy_run and not snapshot.is_profile and snapshot.no_page_io


def test_prepare_inputs_clears_the_prepare_attn_flag(state):
    attn(state)
    assert not inputs(state, ("req_0_a",), (0,)).is_profile
    # the next execute_model's profiling step calls no prepare_attn
    assert inputs(state, ("req_0_b",), (0,)).is_profile


def test_a_step_with_one_unrecorded_row_is_a_dummy_step(state):
    state.add_request(0, request("a", 1))
    attn(state)
    snapshot = inputs(state, ("a", "req_1_random"), (0, 0))
    assert snapshot.dummy_run and not snapshot.is_profile
    assert snapshot.state_batch is None


def test_a_step_of_recorded_rows_is_a_real_step(state):
    state.add_request(0, request("a", 1))
    state.add_request(1, request("b", 2, computed=0))
    attn(state)
    snapshot = inputs(state, ("a", "b"), (0, 0))
    assert not snapshot.dummy_run and not snapshot.is_profile
    assert not snapshot.no_page_io
    assert [b.slot_id for b in snapshot.bindings] == [1, 2]
    assert [b.fresh for b in snapshot.bindings] == [True, True]
    assert snapshot.state_batch.slot_indices == (1, 2)


def test_real_step_freshness_is_zero_computed_tokens(state):
    layer = state.model._persistent_state_layer
    state.add_request(0, request("a", 1))
    attn(state)
    inputs(state, ("a",), (0,))
    layer.pages[1].fill_(5)
    attn(state)
    snapshot = inputs(state, ("a",), (1,))
    assert [b.fresh for b in snapshot.bindings] == [False]
    # a continuing row's page is not initialized
    snapshot.state_batch.initialize_fresh_pages()
    assert layer.pages[1].eq(5).all()


def test_fresh_row_page_is_initialized_by_the_step_not_by_planning(state):
    layer = state.model._persistent_state_layer
    layer.pages.fill_(-1)
    state.add_request(0, request("a", 3))
    attn(state)
    snapshot = inputs(state, ("a",), (0,))
    # planning (prepare_inputs) writes no page
    assert layer.pages.eq(-1).all()
    assert snapshot.state_batch.fresh_slots == (3,)
    # the model's forward initializes the fresh page before its transaction
    snapshot.state_batch.initialize_fresh_pages()
    for descriptor in state.model._state_spec.descriptors:
        if descriptor.zero_initialized:
            assert layer.component_pools[descriptor.name][3].eq(0).all()
    assert layer.pages[2].eq(-1).all()


def test_real_step_refuses_an_unbound_pool(state, layout):
    state.model._persistent_state_layer = _layer(layout, bind=False)
    state.add_request(0, request("a", 1))
    attn(state)
    with pytest.raises(RuntimeError, match="not bound"):
        inputs(state, ("a",), (0,))


# ---------------------------------------------------------------------------
# Request lifetime (algorithm 4)


def test_turn_replacement_keeps_the_session(state):
    state.add_request(0, request("a", 3))
    generation = state._requests["a"].handle.generation
    state.remove_request("a")
    state.add_request(0, request("a", 3, computed=4))
    record = state._requests["a"]
    assert record.handle.slot == 3 and record.handle.generation == generation
    assert record.prompt == (STREAM_TOKEN,) * 5
    assert state._pending_removal is None
    assert state.model._registry.prunes == []


def test_finish_retires_at_the_next_call(state):
    state.add_request(0, request("a", 3))
    state.remove_request("a")
    # still pending: the next call decides it
    assert "a" in state._requests and state._pending_removal == "a"
    attn(state)
    assert "a" not in state._requests and state._pending_removal is None
    assert state.model._registry.prunes == [()]


def test_reused_request_id_starts_fresh(state):
    state.add_request(0, request("a", 3))
    generation = state._requests["a"].handle.generation
    state.remove_request("a")
    # same id, no prior state: the old incarnation retires first
    state.add_request(0, request("a", 3, computed=0))
    assert state.model._registry.prunes == [()]
    assert state._requests["a"].handle.generation == generation + 1


def test_reused_request_id_on_another_slot_starts_fresh(state):
    state.add_request(0, request("a", 3))
    state.remove_request("a")
    state.add_request(0, request("a", 4))
    assert state._requests["a"].handle.slot == 4
    assert state.model._registry.prunes == [()]


def test_prior_state_on_another_slot_is_refused(state):
    state.add_request(0, request("a", 3))
    state.remove_request("a")
    with pytest.raises(RuntimeError, match="prior state"):
        state.add_request(0, request("a", 4, computed=2))


def test_add_for_another_request_retires_the_pending_one(state):
    state.add_request(0, request("a", 3))
    state.remove_request("a")
    state.add_request(1, request("b", 4))
    assert "a" not in state._requests and "b" in state._requests
    # retired before "b" is recorded
    assert state.model._registry.prunes == [()]


def test_a_second_removal_retires_the_first(state):
    state.add_request(0, request("a", 3))
    state.add_request(1, request("b", 4))
    state.remove_request("a")
    state.remove_request("b")
    assert "a" not in state._requests and state._pending_removal == "b"


def test_removing_an_unknown_request_retires_the_pending_one_only(state):
    state.add_request(0, request("a", 3))
    state.remove_request("a")
    state.remove_request("never-added")
    assert "a" not in state._requests and state._pending_removal is None


def test_zero_token_step_defers_retirement_to_the_next_steps_first_call(state):
    state.add_request(0, request("a", 3))
    state.add_request(1, request("b", 4))
    # step 1 finishes "a" and schedules nothing: core returns before
    # prepare_attn and prepare_inputs
    state.remove_request("a")
    assert "a" in state._requests
    # step 2's first call is core's removal-before-add of a new request
    state.remove_request("c")
    assert "a" not in state._requests
    state.add_request(0, request("c", 3))
    attn(state)
    snapshot = inputs(state, ("b", "c"), (1, 0))
    assert [b.request_id for b in snapshot.bindings] == ["b", "c"]


def test_prepare_inputs_retires_before_it_binds_rows(state):
    state.add_request(0, request("a", 3))
    state.add_request(1, request("b", 4))
    state.remove_request("a")
    snapshot = inputs(state, ("b",), (1,))
    assert "a" not in state._requests
    assert not snapshot.dummy_run


def test_retirement_frees_the_file_state(state):
    state.add_request(0, request("a", 3))
    state._file_progress["a"] = object()
    state._failed_files.add("a")
    state.remove_request("a")
    attn(state)
    assert "a" not in state._file_progress and "a" not in state._failed_files


def test_row_failure_raises_naming_the_requests(state):
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        ProjectionSnapshot,
    )

    state.model.collect_commit_status = lambda: SimpleNamespace(
        failed={"b", "a"}, statuses={}, queue_counts=()
    )
    state.model.consume_batch_stats = lambda: None
    state._snapshot = ProjectionSnapshot(
        1, ("a", "b"), {}, None, (), (), None, None, {}
    )
    with pytest.raises(RuntimeError, match="a, b"):
        state.postprocess_state(torch.zeros(2), torch.ones(2), None)
    assert state._snapshot is None


# ---------------------------------------------------------------------------
# Startup checks (plugin load and the engine tier)


def test_plugin_load_accepts_this_build():
    from vllm_nemotron_asr_streaming import plugin

    plugin._require_client_path()


def test_plugin_load_refuses_a_build_without_the_model_state_hooks(monkeypatch):
    from vllm_nemotron_asr_streaming import engine_flags, plugin

    monkeypatch.setattr(
        engine_flags, "missing_graph_hooks", lambda: ["capture_cudagraphs"]
    )
    with pytest.raises(RuntimeError, match=r"ModelState\.capture_cudagraphs"):
        plugin._require_client_path()


@pytest.mark.parametrize(
    "overrides",
    [{"transaction_graph_populations": [1]}, {"experimental_native_burst": True}],
    ids=["transaction-graphs", "native-burst"],
)
def test_the_poc_refuses_transaction_graphs_and_native_burst(overrides):
    from vllm_nemotron_asr_streaming.engine_flags import pr6_poc_refusals

    (message,) = pr6_poc_refusals(SimpleNamespace(**overrides))
    assert "PR 6" in message
    assert pr6_poc_refusals(SimpleNamespace()) == []
