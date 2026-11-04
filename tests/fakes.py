# SPDX-License-Identifier: Apache-2.0
"""In-process stand-ins for the RFC1 tip symbols.

These are not vLLM. Tests that pass against them are not installed-plugin
evidence.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class PersistentStateLayerBase:
    pass


class AsyncLLM:
    def __init__(self, engine_core: Any) -> None:
        self.engine_core = engine_core


class EngineCoreClient:
    async def admit_stream_async(self, request: Any) -> Any:
        return request


class StateCompletion:
    pass


def resolve_model_state_cls(vllm_config: Any, model: Any) -> Any:
    del vllm_config
    return model.get_model_state_cls()


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}

    def register_model(self, model_arch: str, model_cls: Any) -> None:
        self.models[model_arch] = model_cls


def install() -> ModelRegistry:
    """Install the stub modules and return the registry instance."""
    registry = ModelRegistry()

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("vllm")
    module("vllm.model_executor")
    module("vllm.model_executor.models", ModelRegistry=registry)
    module("vllm.model_executor.layers")
    module(
        "vllm.model_executor.layers.state",
        PersistentStateLayerBase=PersistentStateLayerBase,
    )
    module("vllm.v1")
    module("vllm.v1.engine")
    module("vllm.v1.engine.async_llm", AsyncLLM=AsyncLLM)
    module("vllm.v1.engine.core_client", EngineCoreClient=EngineCoreClient)
    module("vllm.v1.kv_cache_interface", StateCompletion=StateCompletion)
    module("vllm.v1.worker")
    module("vllm.v1.worker.gpu")
    module(
        "vllm.v1.worker.gpu.model_states",
        resolve_model_state_cls=resolve_model_state_cls,
    )
    return registry


def drop_cached_plugin_modules() -> None:
    for name in list(sys.modules):
        if name == "vllm_nemotron_asr_streaming" or name.startswith(
            "vllm_nemotron_asr_streaming."
        ):
            del sys.modules[name]
