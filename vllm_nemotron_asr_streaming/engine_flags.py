# SPDX-License-Identifier: Apache-2.0
"""F7 engine flags for a persistent-state serve.

On ``rfc12-integration`` @ ``4c180969`` (tree ``465f10ad65``),
``CacheConfig.enable_prefix_caching`` defaults to ``True`` and
``SchedulerConfig.async_scheduling`` defaults to ``None``. ``VllmConfig``
resolves ``None`` to ``True`` when the executor supports async scheduling.
``validate_state_config`` refuses a persistent-state engine when either
is on.

``persistent_state_config_errors`` reports those flags. It does not turn
them off. ``Nemotron3_5AsrStreamingConfig`` and
``StreamingSessionFactory`` raise from that report.

Serve with the flags set explicitly, for example:

```
vllm serve <translated-or-overridden-checkpoint> \\
    --dtype float32 \\
    --enforce-eager \\
    --no-async-scheduling \\
    --no-enable-prefix-caching
```

``--no-async-scheduling`` is the flag named by the RFC12 scheduler
config. ``--no-enable-prefix-caching`` is the negative form of
``CacheConfig.enable_prefix_caching``. The Python values are
``async_scheduling=False`` and ``enable_prefix_caching=False``.
"""

from __future__ import annotations

from typing import Any

# Explicit values a serve has to pass. The tip's own defaults are the
# opposite for both of these.
REQUIRED_OVERRIDES: dict[str, bool] = {
    "async_scheduling": False,
    "enable_prefix_caching": False,
}

# Parallel sizes ``validate_state_config`` requires to be 1.
PARALLEL_SIZES_MUST_BE_ONE: tuple[str, ...] = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
)

# Labels ``validate_state_config`` appends when the feature is enabled.
REFUSED_WHEN_ENABLED: tuple[str, ...] = (
    "async scheduling",
    "speculative decoding",
    "prefix caching",
    "KV transfer",
    "dual batch overlap",
    "sleep mode",
    "CuMem",
    "HiSparse",
)


def persistent_state_config_errors(vllm_config: Any) -> list[str]:
    """Return human-readable refusals for a persistent-state engine.

    ``async_scheduling is None`` is a refusal: the RFC12 resolver turns
    that into ``True``.
    """
    errors: list[str] = []
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    attention = vllm_config.attention_config
    if scheduler.async_scheduling is not False:
        errors.append(
            "async scheduling must be False "
            "(RFC12 resolves a missing value to True; "
            "pass async_scheduling=False or --no-async-scheduling)"
        )
    if cache.enable_prefix_caching:
        errors.append(
            "prefix caching must be False "
            "(RFC12 defaults it to True; "
            "pass enable_prefix_caching=False or --no-enable-prefix-caching)"
        )
    if vllm_config.speculative_config is not None:
        errors.append("speculative decoding must be off")
    if vllm_config.kv_transfer_config is not None:
        errors.append("KV transfer must be off")
    if getattr(parallel, "enable_dbo", False):
        errors.append("dual batch overlap must be off")
    if getattr(model, "enable_sleep_mode", False):
        errors.append("sleep mode must be off")
    if getattr(model, "enable_cumem_allocator", False):
        errors.append("CuMem must be off")
    if getattr(attention, "hisparse_config", None) is not None:
        errors.append("HiSparse must be off")
    for name in PARALLEL_SIZES_MUST_BE_ONE:
        if getattr(parallel, name, None) != 1:
            errors.append(f"{name} must be 1")
    return errors
