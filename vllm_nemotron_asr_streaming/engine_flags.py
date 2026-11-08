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
        --enforce-eager \\
        --no-async-scheduling \\
        --no-enable-prefix-caching

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
    "vllm.v1.kv_cache_interface": ("KVCacheConfig", "StateSlotHandle"),
    "vllm.v1.worker.gpu.input_batch": ("InputBatch",),
    "vllm.v1.worker.gpu.model_states.interface": ("ModelState",),
    "vllm.v1.worker.gpu.state_slots": ("StateSlotBindings", "StateSlotRows"),
    "vllm.v1.worker.gpu.states": ("RequestState",),
    "vllm.v1.worker.utils": ("AttentionGroup",),
    "vllm.model_executor.layers.state": ("PersistentStateLayerBase",),
    "vllm.multimodal": ("MULTIMODAL_REGISTRY",),
    "vllm.v1.attention.backends.utils": ("NULL_BLOCK_ID",),
}


# @spec REG-ENG-001, REG-ENG-002, REG-ENG-003, REG-ENG-004, REG-ENG-005
def persistent_state_config_errors(vllm_config: Any) -> list[str]:
    """Return human-readable refusals for a persistent-state engine.

    ``async_scheduling`` has to be ``False``: an unset value (``None``) is a
    refusal, because core resolves it to ``True`` when the executor supports
    asynchronous scheduling.
    """
    errors: list[str] = []
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    attention = getattr(vllm_config, "attention_config", None)
    if getattr(scheduler, "async_scheduling", None) is not False:
        errors.append(
            "async scheduling must be False "
            "(an unset value resolves to True; "
            "pass async_scheduling=False or --no-async-scheduling)"
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


def engine_config_refusal(vllm_config: Any, subject: str) -> str | None:
    """Return one message listing every refused engine setting, or ``None``."""
    errors = persistent_state_config_errors(vllm_config)
    if not errors:
        return None
    return f"{subject} refuses this engine configuration: " + "; ".join(errors)


# @spec REG-ENG-006, REG-VAL-003
def check_engine_config(vllm_config: Any, subject: str) -> None:
    """Raise one ``ValueError`` listing every refused engine setting."""
    message = engine_config_refusal(vllm_config, subject)
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
            "vllm-nemotron-asr-streaming (an RFC 1 + RFC 2 build is "
            "required): " + "; ".join(missing)
        )
