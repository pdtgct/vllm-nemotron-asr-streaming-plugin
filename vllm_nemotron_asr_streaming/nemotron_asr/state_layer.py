# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model's state layer: one ``MambaSpec`` page per request.

Core's ``MambaManager`` allocates the page with ``recomputable=False`` (no
preemption, admission bounded by the pool) and binds it through
``MambaBase.bind_kv_cache``, which yields one view per state, one row per
block, with the page as the row stride. The layer names those views after the
layout's descriptors (the "component pools") and initializes a newly taken
page; it builds no attention metadata.
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


def state_spec(layout: StateLayout, *, block_size: int) -> MambaSpec:
    """The ``MambaSpec`` of ``layout``: its states in manifest order, state
    alignment 1 (the manifest publishes its padding explicitly)."""
    return MambaSpec(
        shapes=layout.shapes,
        dtypes=layout.dtypes,
        state_alignment=1,
        block_size=block_size,
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
    )


def check_page_size(spec: MambaSpec, layout: StateLayout) -> None:
    """Refuse a ``MambaSpec`` whose page differs from the manifest's page."""
    if spec.page_size_bytes != layout.page_size_bytes:
        raise ValueError(
            f"the state layer's MambaSpec page is {spec.page_size_bytes} bytes, "
            f"but the state manifest's page is {layout.page_size_bytes} bytes"
        )


class NemotronStateLayer(MambaBase, nn.Module):
    """The one state layer, registered in the static forward context under
    its prefix."""

    mamba_type = MambaAttentionBackendEnum.SHORT_CONV
    state_alignment = 1

    def __init__(self, layout: StateLayout, *, prefix: str, vllm_config: Any) -> None:
        super().__init__()
        self.prefix = prefix
        self.layout = layout
        #: The raw ``[blocks, page]`` byte view and the per-state views of the
        #: last binding; ``None`` until core binds the pool.
        self.pages: torch.Tensor | None = None
        self.component_pools: dict[str, torch.Tensor] | None = None
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

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # Core binds the capacity-sized pool at initialization and rebinds it
        # when the extensible KV cache grows: the views are replaced each time.
        super().bind_kv_cache(kv_cache)
        self.pages = kv_cache.squeeze(dim=(1, 2))
        self.component_pools = dict(zip(self.layout.names, self.kv_cache, strict=True))

    @property
    def num_blocks(self) -> int:
        if self.pages is None:
            raise RuntimeError("the state pool is not bound")
        return int(self.pages.shape[0])

    def initialize_pages(self, slots: Sequence[int]) -> None:
        """Initialize the pages of ``slots`` on the current stream: zero every
        byte, then write each descriptor's non-zero initializer."""
        if not slots:
            return
        if self.pages is None or self.component_pools is None:
            raise RuntimeError("the state pool is not bound")
        index = torch.tensor(list(slots), dtype=torch.int64, device=self.pages.device)
        self.pages.index_fill_(0, index, 0)
        for descriptor in self.layout.descriptors:
            if descriptor.zero_initialized:
                continue
            view = self.component_pools[descriptor.name]
            value = descriptor.initializer(
                descriptor.shape, dtype=descriptor.dtype, device=view.device
            )
            view.index_copy_(
                0, index, value.unsqueeze(0).expand(len(slots), *descriptor.shape)
            )

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the state layer is not called; the model reads its views")
