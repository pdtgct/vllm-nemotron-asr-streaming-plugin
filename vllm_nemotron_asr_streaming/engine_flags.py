# SPDX-License-Identifier: Apache-2.0
"""F7 async-scheduling and prefix-caching defaults. Recorded, not applied.

On RFC1 tip ``8b8327e2ee``, ``CacheConfig.enable_prefix_caching`` defaults
to ``True`` (``vllm/config/cache.py``). ``SchedulerConfig.async_scheduling``
defaults to ``None``, and ``VllmConfig`` resolves ``None`` to ``True`` when
the executor supports async scheduling (``vllm/config/vllm.py``).
``validate_state_config`` (``vllm/v1/core/state_slot_manager.py``) refuses
a persistent-state engine when either flag is on, along with the other
items listed below.

Nothing in this package launches an engine or passes these flags.
"""

from __future__ import annotations

# Explicit values a future engine launch has to pass. The tip's own
# defaults are the opposite for both of these.
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
