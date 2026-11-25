# SPDX-License-Identifier: Apache-2.0
"""The state layer and the model state's request bookkeeping on CPU, without
weights, as the target core (``pdtgct/vllm`` at ``ddf6db2ab8``) drives them.

The model state is the real class, built as core builds it, over a model
without weights whose few attributes the hooks read are set directly; its
state layer is the real ``NemotronStateLayer`` bound to a CPU pool laid out as
core's ``MambaManager`` pool (``[blocks, 1, 1, page]`` bytes; block 0 is the
null block). Steps that need the model's forward use
``support.model_harness`` instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

LAYER = "nemotron.state"
#: A stream token that is neither a carrier nor a control token.
STREAM_TOKEN = 7
BLOCKS = 6


def forward_context_config(max_model_len: int = 2048) -> Any:
    """What the layer reads of the engine configuration: the static forward
    context it registers in and ``max_model_len`` (its block size)."""
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def served_config(card_dir: Any, *, right_context: int = 6) -> Any:
    """The served ``VllmConfig`` of a card directory under ``--enforce-eager``."""
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
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": right_context},
        enforce_eager=True,
        dtype="float32",
    )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
    )


def layout_of(vllm_config: Any) -> Any:
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        build_nemotron_state_layout,
    )

    return build_nemotron_state_layout(vllm_config.model_config.hf_config)


def pool(layout: Any, blocks: int = BLOCKS) -> torch.Tensor:
    """A ``kv_cache`` for the layer as core's ``MambaManager`` lays it out."""
    return torch.zeros((blocks, 1, 1, layout.page_size_bytes), dtype=torch.int8)


def layer(
    layout: Any,
    *,
    bind: bool = True,
    blocks: int = BLOCKS,
    config: Any = None,
) -> Any:
    from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import (
        NemotronStateLayer,
    )

    built = NemotronStateLayer(
        layout, prefix=LAYER, vllm_config=config or forward_context_config()
    )
    if bind:
        built.bind_kv_cache(pool(layout, blocks))
    return built


class Registry:
    """Records every prune's keep set (the model's session registry)."""

    def __init__(self) -> None:
        self.prunes: list[tuple[str, ...]] = []

    def prune(self, keep: Iterable[str]) -> None:
        self.prunes.append(tuple(sorted(keep)))


def weightless_state(vllm_config: Any, layout: Any, *, bound: bool = True) -> Any:
    """The real model state over a weightless model with a state layer."""
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    model = Nemotron3_5AsrStreamingForRNNT.__new__(Nemotron3_5AsrStreamingForRNNT)
    torch.nn.Module.__init__(model)
    model.config = vllm_config.model_config.hf_config
    model._state_spec = layout
    model._persistent_state_layer = layer(layout, bind=bound)
    model._registry = Registry()
    model._native_burst_handoff = None
    cls = Nemotron3_5AsrStreamingForRNNT.get_model_state_cls()
    return cls(vllm_config, model, None, torch.device("cpu"))


def request(
    req_id: str,
    slot: int | None,
    *,
    computed: int = 0,
    tokens: tuple[int, ...] = (),
    block_ids: tuple[list[int], ...] | None = None,
    mm_features: list[Any] | None = None,
) -> Any:
    """``NewRequestData`` as core delivers it: ``block_ids`` holds the
    request's one block (its slot) unless given; ``tokens`` is the prompt
    (default ``computed + 1`` stream tokens)."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    prompt = list(tokens or (STREAM_TOKEN,) * (computed + 1))
    if block_ids is None:
        block_ids = ([slot],)
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=list(mm_features or []),
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=computed,
        lora_request=None,
        prefill_token_ids=prompt,
    )


def attn(state: Any) -> None:
    """Core's ``prepare_attn`` for a step (no capture, ``cudagraph_mode``
    NONE): the model state builds no metadata."""
    from vllm.config import CUDAGraphMode

    assert state.prepare_attn(None, CUDAGraphMode.NONE, (), None, [], None) == {}


def batch(
    req_ids: tuple[str, ...],
    computed: tuple[int, ...],
    scheduled: tuple[int, ...] | None = None,
) -> Any:
    """The ``InputBatch`` fields the model state reads to plan a step."""
    n = len(req_ids)
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_scheduled_tokens=np.array(scheduled or (1,) * n, dtype=np.int32),
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
    )


def inputs(
    state: Any,
    req_ids: tuple[str, ...],
    computed: tuple[int, ...],
    scheduled: tuple[int, ...] | None = None,
) -> Any:
    """``prepare_inputs``; returns the step's projection snapshot."""
    projection = state.prepare_inputs(batch(req_ids, computed, scheduled), None)
    return projection["persistent_state_projection"]


def collection(
    *,
    failed: Iterable[str] = (),
    statuses: dict[str, int] | None = None,
    queue_counts: tuple[Any, ...] = (),
) -> Any:
    """A ``CommitCollection``-shaped result of ``collect_commit_status``."""
    return SimpleNamespace(
        failed=set(failed), statuses=dict(statuses or {}), queue_counts=queue_counts
    )


def postprocess(state: Any, rows: int = 1) -> None:
    """Core's ``postprocess_state`` call for a step of ``rows`` rows."""
    state.postprocess_state(torch.zeros(rows), torch.ones(rows), None)


class ViewTrap:
    """Stands in for the model's state layer and records every read of its
    views or pages (``component_pools``, ``pages``, ``kv_cache``,
    ``num_blocks``, ``initialize_pages``); once ``armed`` such a read fails.
    Every other attribute (the layout, the prefix) passes through. ``pool``
    is the bound pool the layer's views are views of."""

    VIEWS = frozenset(
        {"component_pools", "pages", "kv_cache", "num_blocks", "initialize_pages"}
    )

    def __init__(self, layer: Any, pool: Any = None) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_target", pool)
        object.__setattr__(self, "armed", False)
        object.__setattr__(self, "reads", [])

    def __getattr__(self, attr: str) -> Any:
        if attr in ViewTrap.VIEWS:
            self.reads.append(attr)
            if self.armed:
                raise AssertionError(f"the state layer's {attr} was read")
        return getattr(self._layer, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "armed":
            object.__setattr__(self, attr, value)
        else:
            setattr(self._layer, attr, value)


def trap_views(model: Any, pool: Any = None) -> ViewTrap:
    """Make the model (and so its model state) reach its state layer only
    through a :class:`ViewTrap`; returns the trap, unarmed."""
    trap = ViewTrap(model._persistent_state_layer, pool)
    # The layer is a registered submodule; the instance attribute takes
    # precedence over ``nn.Module``'s submodule lookup.
    object.__setattr__(model, "_persistent_state_layer", trap)
    return trap
