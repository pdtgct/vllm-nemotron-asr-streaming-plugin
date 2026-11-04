# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` entry point.

``register`` imports the RFC1 tip symbols and registers only the
streaming architecture. It does not import the model module, so a
process that has not initialized CUDA can still load the plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _require_rfc1_tip(
    *,
    persistent_state_layer: Any,
    async_llm: Any,
    admit_stream_async: Any,
    state_completion: Any,
    resolve_model_state_cls: Callable[..., Any],
) -> None:
    """Fail plugin load when the RFC1 tip surface is absent."""
    if not isinstance(persistent_state_layer, type):
        raise RuntimeError(
            "vllm.model_executor.layers.state.PersistentStateLayerBase "
            "is not a class. This plugin targets the RFC1 tip, not a "
            "released vLLM."
        )
    if not isinstance(async_llm, type):
        raise RuntimeError(
            "vllm.v1.engine.async_llm.AsyncLLM is not a class. "
            "This plugin targets the RFC1 tip, not a released vLLM."
        )
    if not callable(admit_stream_async):
        raise RuntimeError(
            "EngineCoreClient.admit_stream_async is missing. "
            "This plugin targets the RFC1 tip, not a released vLLM."
        )
    if not isinstance(state_completion, type):
        raise RuntimeError(
            "vllm.v1.kv_cache_interface.StateCompletion is not a class. "
            "This plugin targets the RFC1 tip, not a released vLLM."
        )
    if not callable(resolve_model_state_cls):
        raise RuntimeError(
            "resolve_model_state_cls is missing. The streaming model "
            "has to be discovered through get_model_state_cls."
        )


def register() -> None:
    """Register the streaming architecture with vLLM.

    Safe to call more than once in one process. Never writes the
    offline architecture key.
    """
    from vllm.model_executor.layers.state import PersistentStateLayerBase
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.kv_cache_interface import StateCompletion
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls

    from .identity import ARCHITECTURE, LAZY_MODEL, OFFLINE_ARCHITECTURE
    from .streaming_config import Nemotron3_5AsrStreamingConfig

    _require_rfc1_tip(
        persistent_state_layer=PersistentStateLayerBase,
        async_llm=AsyncLLM,
        admit_stream_async=getattr(EngineCoreClient, "admit_stream_async", None),
        state_completion=StateCompletion,
        resolve_model_state_cls=resolve_model_state_cls,
    )
    if ARCHITECTURE == OFFLINE_ARCHITECTURE:
        raise RuntimeError(
            "refusing to register the offline architecture "
            f"{OFFLINE_ARCHITECTURE!r}"
        )

    offline_before = ModelRegistry.models.get(OFFLINE_ARCHITECTURE)
    config_before = MODELS_CONFIG_MAP.get(OFFLINE_ARCHITECTURE)
    ModelRegistry.register_model(ARCHITECTURE, LAZY_MODEL)
    MODELS_CONFIG_MAP[ARCHITECTURE] = Nemotron3_5AsrStreamingConfig
    if MODELS_CONFIG_MAP.get(OFFLINE_ARCHITECTURE) != config_before:
        raise RuntimeError(
            "streaming registration changed the offline config hook "
            f"{OFFLINE_ARCHITECTURE!r}"
        )
    if ModelRegistry.models.get(OFFLINE_ARCHITECTURE) != offline_before:
        raise RuntimeError(
            "streaming registration changed the offline architecture "
            f"{OFFLINE_ARCHITECTURE!r}"
        )
    registered = ModelRegistry.models.get(ARCHITECTURE)
    if registered is None:
        raise RuntimeError(f"{ARCHITECTURE!r} was not registered")
