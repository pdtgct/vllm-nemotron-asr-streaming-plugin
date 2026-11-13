# SPDX-License-Identifier: Apache-2.0
"""Engine settings a persistent-state serve refuses, and the model adapter's
import-time dependency check.

``persistent_state_config_errors`` reports the refused settings of a
``VllmConfig``; it does not turn anything off. The model-configuration hook
(``streaming_config``) and the model state's construction both raise from that
report, so a setting that core resolves only after the hook ran is still
caught.

Serve with the settings explicit (see the README), for example::

    vllm serve <checkpoint> \\
        --dtype float32 \\
        --enforce-eager

Importing this module imports no vLLM.
"""

from __future__ import annotations

import importlib
from typing import Any

# Parallel sizes that have to be 1. A size the running vLLM build does not
# define counts as 1.
PARALLEL_SIZES_MUST_BE_ONE: tuple[str, ...] = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
)

#: The model adapter's internal vLLM dependencies, beyond the public client
#: surface: module name -> names the adapter imports from it.
ADAPTER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.config.compilation": ("CUDAGraphMode",),
    "vllm.v1.core.sched.output": ("NewRequestData", "SchedulerOutput"),
    "vllm.v1.kv_cache_interface": (
        "KVCacheConfig",
        "StateSlotHandle",
        "PersistentStateSpec",
        "PersistentStateDescriptor",
    ),
    "vllm.v1.worker.gpu.input_batch": ("InputBatch",),
    "vllm.v1.worker.gpu.model_states.interface": ("ModelState",),
    "vllm.v1.worker.gpu.state_slots": ("StateSlotBindings", "StateSlotRows"),
    "vllm.v1.worker.gpu.states": ("RequestState",),
    "vllm.v1.worker.utils": ("AttentionGroup",),
    "vllm.model_executor.layers.state": ("PersistentStateLayerBase",),
    "vllm.multimodal": ("MULTIMODAL_REGISTRY",),
    "vllm.v1.attention.backends.utils": ("NULL_BLOCK_ID",),
    "vllm.v1.worker.gpu.sample.output": ("SamplerOutput",),
    "vllm.utils.torch_utils": ("direct_register_custom_op",),
}


# @spec REG-ENG-001, REG-ENG-002, REG-ENG-003, REG-ENG-004, REG-ENG-005
def persistent_state_config_errors(
    vllm_config: Any, *, resolved: bool = False
) -> list[str]:
    """Return human-readable refusals for a persistent-state engine.

    ``async_scheduling`` of ``True`` is refused. An unset value (``None``) is
    accepted before core resolves the option (``resolved=False``: it resolves
    to ``False`` for a model that declares persistent state); once the engine
    configuration is final (``resolved=True``) only ``False`` is accepted.
    """
    errors: list[str] = []
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    attention = getattr(vllm_config, "attention_config", None)
    async_scheduling = getattr(scheduler, "async_scheduling", None)
    if async_scheduling is True or (resolved and async_scheduling is not False):
        errors.append(
            "async scheduling must be False "
            "(pass async_scheduling=False or --no-async-scheduling)"
        )
    if getattr(cache, "enable_prefix_caching", None):
        errors.append(
            "prefix caching must be False "
            "(pass enable_prefix_caching=False or --no-enable-prefix-caching)"
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        errors.append("speculative decoding must be off")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        errors.append("KV transfer must be off")
    if getattr(parallel, "enable_dbo", False):
        errors.append("dual batch overlap must be off")
    if getattr(model, "enable_sleep_mode", False):
        errors.append("sleep mode must be off")
    if getattr(model, "enable_cumem_allocator", False):
        errors.append("CuMem must be off")
    if getattr(attention, "hisparse_config", None) is not None:
        errors.append("HiSparse (hierarchical sparse attention) must be off")
    if getattr(parallel, "enable_fault_tolerance", False):
        # Core's kernel warm-up then runs synthetic real steps that carry no
        # state-slot handles (pass neither --enable-fault-tolerance nor
        # --fault-tolerance-config).
        errors.append(
            "fault tolerance must be off "
            "(remove --enable-fault-tolerance and --fault-tolerance-config)"
        )
    for name in PARALLEL_SIZES_MUST_BE_ONE:
        if getattr(parallel, name, 1) != 1:
            errors.append(f"{name} must be 1")
    if not getattr(model, "enforce_eager", False):
        errors.append(
            "enforce_eager must be set: the outer runner has to be eager "
            "(pass enforce_eager=True or --enforce-eager); the model runs its "
            "own captured graphs inside the eager step"
        )
    return errors


def engine_config_refusal(
    vllm_config: Any, subject: str, *, resolved: bool = False
) -> str | None:
    """Return one message listing every refused engine setting, or ``None``."""
    errors = persistent_state_config_errors(vllm_config, resolved=resolved)
    if not errors:
        return None
    return f"{subject} refuses this engine configuration: " + "; ".join(errors)


# @spec REG-ENG-006, REG-VAL-003
def check_engine_config(
    vllm_config: Any, subject: str, *, resolved: bool = False
) -> None:
    """Raise one ``ValueError`` listing every refused engine setting."""
    message = engine_config_refusal(vllm_config, subject, resolved=resolved)
    if message is not None:
        raise ValueError(message)


# @spec REG-010
def check_adapter_dependencies() -> None:
    """Import every adapter dependency; raise naming every missing one.

    The model class module calls this before it imports the adapter's vLLM
    symbols, so a vLLM build without them fails with one clear error before
    any weights load.
    """
    missing: list[str] = []
    for module_name, names in ADAPTER_DEPENDENCIES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            missing.extend(f"{module_name}.{name} ({exc})" for name in names)
            continue
        missing.extend(
            f"{module_name}.{name} (name not found)"
            for name in names
            if not hasattr(module, name)
        )
    if missing:
        raise ImportError(
            "the installed vLLM lacks adapter dependencies of "
            "vllm-nemotron-asr-streaming (a vLLM build with RFC 1 stream "
            "admission, vllm#67375 on #67374, is required): " + "; ".join(missing)
        )
