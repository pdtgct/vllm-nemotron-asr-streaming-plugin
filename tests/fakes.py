# SPDX-License-Identifier: Apache-2.0
"""In-process stand-ins for the RFC12 tip symbols.

These are not vLLM. Tests that pass against them are not installed-plugin
evidence.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class PersistentStateLayerBase:
    pass


class _Namespace:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def safe_vllm_config() -> _Namespace:
    """A config ``persistent_state_config_errors`` accepts."""
    parallel = _Namespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_dbo=False,
    )
    return _Namespace(
        scheduler_config=_Namespace(async_scheduling=False),
        cache_config=_Namespace(enable_prefix_caching=False),
        parallel_config=parallel,
        model_config=_Namespace(
            enable_sleep_mode=False,
            enable_cumem_allocator=False,
        ),
        attention_config=_Namespace(hisparse_config=None),
        speculative_config=None,
        kv_transfer_config=None,
    )


class AsyncLLM:
    def __init__(
        self,
        engine_core: Any,
        vllm_config: Any | None = None,
    ) -> None:
        self.engine_core = engine_core
        self.vllm_config = (
            safe_vllm_config() if vllm_config is None else vllm_config
        )


class StreamAdmissionResult:
    def __init__(
        self,
        accepted: bool,
        request_id: str,
        status: str,
        reason: str | None = None,
    ) -> None:
        self.accepted = accepted
        self.request_id = request_id
        self.status = status
        self.reason = reason


class StreamAdmissionContext:
    def __init__(self, input_closed: bool = False) -> None:
        self.input_closed = input_closed


class EngineCoreRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.n = kwargs.get("n", 1)
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 1)
        self.stop = kwargs.get("stop")
        self.structured_outputs = kwargs.get("structured_outputs")
        self.output_kind = kwargs.get("output_kind", "cumulative")
        self.extra_args = kwargs.get("extra_args")


class EngineCoreClient:
    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.aborts: list[str] = []
        self.admitted: dict[str, Any] = {}
        self.completions: dict[str, StateCompletion] = {}

    async def admit_stream_async(self, request: Any) -> StreamAdmissionResult:
        previous = self.admitted.get(request.request_id)
        if previous is not None and previous is not request:
            return StreamAdmissionResult(
                False, request.request_id, "refused", "request_id_in_use"
            )
        self.admitted[request.request_id] = request
        return StreamAdmissionResult(True, request.request_id, "never_fed")

    async def add_request_async(self, request: Any) -> None:
        self.updates.append(request)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.aborts.extend(request_ids)
        for request_id in request_ids:
            self.completions[request_id] = StateCompletion()

    async def wait_state_completion(self, request_id: str) -> StateCompletion:
        return self.completions[request_id]


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
    config_map: dict[str, Any] = {}

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    module("vllm")
    module("vllm.sampling_params", SamplingParams=SamplingParams)
    module("vllm.model_executor")
    module("vllm.model_executor.models", ModelRegistry=registry)
    module(
        "vllm.model_executor.models.config",
        MODELS_CONFIG_MAP=config_map,
    )
    module("vllm.model_executor.layers")
    module(
        "vllm.model_executor.layers.state",
        PersistentStateLayerBase=PersistentStateLayerBase,
    )
    module(
        "vllm.v1",
    )
    module(
        "vllm.v1.engine",
        EngineCoreRequest=EngineCoreRequest,
        StreamAdmissionContext=StreamAdmissionContext,
        StreamAdmissionResult=StreamAdmissionResult,
    )
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
