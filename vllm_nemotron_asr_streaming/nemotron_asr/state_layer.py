# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model's state layer: one ``MambaSpec`` page per request.

Core's ``MambaManager`` allocates the page with ``recomputable=False`` (no
preemption, admission bounded by the pool) and binds it through
``MambaBase.bind_kv_cache``, which yields one view per state, one row per
block, with the page as the row stride. The layer names those views after the
layout's descriptors (the "component pools") and initializes a newly taken
page; it builds no attention metadata.

The page view and the component pools are views of the storage core bound,
never copies. They live in one cache keyed on the identity of the layer's
bound ``kv_cache`` object: a binding, a read that finds another object bound,
and ``drop_views`` (the model state's ``clear_cudagraphs``) drop it, and the
next read rebuilds it from the bound ``kv_cache``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, MambaSpec

from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import StateLayout


# @spec REG-STATE-001, REG-STATE-009
def state_spec(layout: StateLayout, *, block_size: int) -> MambaSpec:
    """The ``MambaSpec`` of ``layout``: its states in manifest order, state
    alignment 1 (the manifest publishes its padding explicitly), of the
    persistent-state Mamba type."""
    return MambaSpec(
        shapes=layout.shapes,
        dtypes=layout.dtypes,
        state_alignment=1,
        block_size=block_size,
        mamba_type=MambaAttentionBackendEnum.PERSISTENT_STATE,
    )


def check_page_size(spec: MambaSpec, layout: StateLayout) -> None:
    """Refuse a ``MambaSpec`` whose page differs from the manifest's page."""
    if spec.page_size_bytes != layout.page_size_bytes:
        raise ValueError(
            f"the state layer's MambaSpec page is {spec.page_size_bytes} bytes, "
            f"but the state manifest's page is {layout.page_size_bytes} bytes"
        )


class _Views:
    """The views of one bound ``kv_cache`` object: the raw ``[blocks, page]``
    byte view and the per-state views, by descriptor name."""

    __slots__ = ("kv_cache", "pages", "component_pools")

    def __init__(self, kv_cache: Any, layout: StateLayout) -> None:
        states = tuple(kv_cache)
        if len(states) != len(layout.descriptors):
            raise RuntimeError(
                f"the state layer's bound kv_cache holds {len(states)} states, "
                f"but its layout has {len(layout.descriptors)}"
            )
        first = states[0]
        element = first.element_size()
        # The page view over the same storage: block ``b`` starts where the
        # first state's row ``b`` starts, less that state's offset in the page.
        raw = torch.empty(0, dtype=torch.int8, device=first.device)
        raw.set_(first.untyped_storage())
        self.pages = raw.as_strided(
            (int(first.shape[0]), layout.page_size_bytes),
            (int(first.stride(0)) * element, 1),
            int(first.storage_offset()) * element - layout.descriptors[0].offset_bytes,
        )
        # The cache's own view objects over the bound storage: dropping the
        # cache drops them, whatever else holds core's ``kv_cache``.
        self.component_pools = {
            name: state.view(state.shape)
            for name, state in zip(layout.names, states, strict=True)
        }
        #: The cache's key: the bound object itself, compared by identity.
        self.kv_cache = kv_cache


class NemotronStateLayer(MambaBase, nn.Module):
    """The one state layer, registered in the static forward context under
    its prefix."""

    mamba_type = MambaAttentionBackendEnum.PERSISTENT_STATE
    state_alignment = 1

    def __init__(self, layout: StateLayout, *, prefix: str, vllm_config: Any) -> None:
        super().__init__()
        self.prefix = prefix
        self.layout = layout
        #: The view cache (REG-STATE-018); ``None`` until a read builds it.
        self._views: _Views | None = None
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return self.layout.shapes

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self.layout.dtypes

    def get_kv_cache_config(self, vllm_config: Any) -> KVCacheLayerConfig:
        spec = state_spec(
            self.layout, block_size=int(vllm_config.model_config.max_model_len)
        )
        check_page_size(spec, self.layout)
        return MambaManager.layer_config(spec, recomputable=False)

    # @spec REG-STATE-018
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # Core binds the capacity-sized pool at initialization and rebinds it
        # when the extensible KV cache grows: the views are rebuilt each time.
        self._views = None
        super().bind_kv_cache(kv_cache)

    # @spec REG-STATE-018
    def drop_views(self) -> None:
        """Drop the view cache: the layer then holds core's storage only
        through the ``kv_cache`` core bound."""
        self._views = None

    # @spec REG-STATE-018
    def _current_views(self) -> _Views:
        """The view cache of the bound ``kv_cache``, rebuilt only when it was
        dropped or another object is bound; raises when none is bound."""
        bound = getattr(self, "kv_cache", None)
        views = self._views
        if views is not None and views.kv_cache is bound:
            return views
        self._views = None
        if bound is None or isinstance(bound, torch.Tensor) or not len(bound):
            raise RuntimeError(
                f"the state layer {self.prefix!r} has no bound kv_cache: core "
                "has not bound the state pages (or has released them)"
            )
        self._views = _Views(bound, self.layout)
        return self._views

    @property
    def pages(self) -> torch.Tensor:
        """The bound pages as one ``[blocks, page]`` byte view."""
        return self._current_views().pages

    @property
    def component_pools(self) -> dict[str, torch.Tensor]:
        """The bound per-state views, by descriptor name."""
        return self._current_views().component_pools

    @property
    def num_blocks(self) -> int:
        return int(self.pages.shape[0])

    # @spec REG-STATE-003, REG-STATE-010
    def initialize_pages(self, slots: Sequence[int]) -> None:
        """Initialize the pages of ``slots`` on the current stream: zero every
        byte, then write each descriptor's non-zero initializer. Block 0 (the
        null page) and a slot outside the bound pages are refused before any
        page is written."""
        if not slots:
            return
        views = self._current_views()
        pages, pools = views.pages, views.component_pools
        num_blocks = int(pages.shape[0])
        outside = [int(s) for s in slots if not 0 < int(s) < num_blocks]
        if outside:
            raise ValueError(
                f"state slots {outside} are outside the bound pages 1 to "
                f"{num_blocks - 1} (block 0 is the null page)"
            )
        index = torch.tensor(list(slots), dtype=torch.int64, device=pages.device)
        pages.index_fill_(0, index, 0)
        for descriptor in self.layout.descriptors:
            if descriptor.zero_initialized:
                continue
            view = pools[descriptor.name]
            value = descriptor.initializer(
                descriptor.shape, dtype=descriptor.dtype, device=view.device
            )
            view.index_copy_(
                0, index, value.unsqueeze(0).expand(len(slots), *descriptor.shape)
            )

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the state layer is not called; the model reads its views")
