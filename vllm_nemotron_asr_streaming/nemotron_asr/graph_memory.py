# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph memory of a capture: the one implementation of the measurement, the
charged margin, the required-capture predicate and the aggregate comparison
(registration design, Segment-graph memory; REG-MEM-027).

The segment-graph part of each capture uses it, and the transaction-graph part
calls it for its own spans (TXG-MEM-001, TXG-MEM-004, TXG-MEM-005,
TXG-MEM-007); neither implements its own.

Readings use ``torch.cuda`` and are looked up at call time. On a host without
the CUDA runtime (CPU conformance runs) a reading is unavailable: it returns
``None`` and the span it bounds is reported as not measured. The measurement
reads free memory only and resets no allocator peak statistic, because under
``--enforce-eager`` the capture runs inside core's memory-profiling window,
which reads them.

The transient peak of a capture call (``begin_transient_peak``,
``transient_peak``) does reset the allocator's peak statistics. Only the
transaction-graph part calls it (TXG-MEM-002), and only in core's capture
hooks, which run after core's memory-profiling window has exited.
"""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from typing import Any

import torch

#: The charged margin ``m``: one flat device buffer per profiling capture call,
#: the transaction-graph design's value (Graph memory, When core's estimate
#: runs) until the first A40 measurement replaces it.
MARGIN_BYTES = 64 << 20

_MIB = 1 << 20


def format_bytes(value: int | None) -> str:
    """``value`` as exact bytes and MiB, or ``"not measured"``."""
    if value is None:
        return "not measured"
    return f"{value} bytes ({value / _MIB:.2f} MiB)"


# @spec REG-MEM-017, TXG-LIFE-012
def release_error_frames(error: BaseException | None) -> None:
    """Drop the locals of every finished frame that ``error`` holds, through
    its cause and its context alike, so that the error a failed capture
    propagates keeps no reference to the tensors or graphs the capture
    created. The messages, the chain and the tracebacks' file and line
    records stay; a frame that is still executing is left alone."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))


# @spec REG-MEM-027
def free_memory(device: torch.device, stream: Any = None) -> int | None:
    """One reading of driver free memory on ``device``, taken after
    synchronizing ``stream`` (the capture stream; the device when ``None``) and
    releasing the allocator's unused cached blocks. ``None`` on a host without
    the CUDA runtime."""
    cuda_device = device if device.type == "cuda" else None
    try:
        if stream is None:
            torch.cuda.synchronize(cuda_device)
        else:
            stream.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free, _total = torch.cuda.mem_get_info(cuda_device)
    except (AssertionError, RuntimeError):
        # torch built without CUDA, or no device: nothing to read. On a CUDA
        # host the failure is real and propagates.
        if torch.cuda.is_available():
            raise
        return None
    return int(free)


def retained(before: int | None, after: int | None) -> int | None:
    """A span's retained bytes: the drop in free memory between its readings
    (``None`` when either reading is unavailable)."""
    if before is None or after is None:
        return None
    return before - after


def _cuda_device(device: torch.device) -> torch.device | None:
    return device if device.type == "cuda" else None


# @spec TXG-MEM-002
def begin_transient_peak(device: torch.device) -> int | None:
    """Reset the allocator's peak statistics on ``device`` and return its
    reserved bytes at that moment, the base of a capture call's transient
    peak. ``None`` on a host without the CUDA runtime."""
    cuda_device = _cuda_device(device)
    try:
        torch.cuda.reset_peak_memory_stats(cuda_device)
        return int(torch.cuda.memory_reserved(cuda_device))
    except Exception:
        if torch.cuda.is_available():
            raise
        return None


# @spec TXG-MEM-002
def transient_peak(device: torch.device, reserved_at_start: int | None) -> int | None:
    """A capture call's transient peak: the allocator's peak reserved bytes
    since :func:`begin_transient_peak`, less the reserved bytes it returned
    (``None`` when that reading is unavailable)."""
    if reserved_at_start is None:
        return None
    cuda_device = _cuda_device(device)
    try:
        peak = int(torch.cuda.max_memory_reserved(cuda_device))
    except Exception:
        if torch.cuda.is_available():
            raise
        return None
    return peak - reserved_at_start


@dataclass
class ChargedMargin:
    """The charged margin of one profiling capture call: its buffer (held
    until ``clear_cudagraphs``), ``m``, the free-memory readings immediately
    around its allocation and its retained charge ``r``."""

    buffer: torch.Tensor | None
    margin_bytes: int
    free_before: int | None
    free_after: int | None

    @property
    def charge(self) -> int | None:
        return retained(self.free_before, self.free_after)

    def release(self) -> None:
        self.buffer = None


# @spec REG-MEM-027
def allocate_margin(
    device: torch.device,
    *,
    stream: Any = None,
    free_before: int | None = None,
    margin_bytes: int = MARGIN_BYTES,
) -> ChargedMargin:
    """Allocate the charged margin, one device buffer of ``margin_bytes``, and
    measure its retained charge ``r`` across the allocation. ``free_before``
    reuses a reading the caller took immediately before (the end of the
    call's last span)."""
    if free_before is None:
        free_before = free_memory(device, stream)
    buffer = torch.empty(margin_bytes, dtype=torch.uint8, device=device)
    free_after = free_memory(device, stream)
    return ChargedMargin(
        buffer=buffer,
        margin_bytes=margin_bytes,
        free_before=free_before,
        free_after=free_after,
    )


@dataclass(frozen=True)
class ProfilingRequirement:
    """Whether a profiling capture was required, from the worker's effective
    cache configuration when the serving capture begins, and why."""

    required: bool
    reason: str
    kv_cache_memory_bytes: Any
    estimate: bool
    extensible_kv_cache: bool


# @spec REG-MEM-027
def profiling_requirement(vllm_config: Any) -> ProfilingRequirement:
    """The required-capture predicate: a profiling capture was required
    exactly when the effective ``kv_cache_memory_bytes`` is unset or 0,
    ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`` is not 0 and the effective
    extensible KV cache is off."""
    import vllm.envs as envs

    cache = getattr(vllm_config, "cache_config", None)
    budget = getattr(cache, "kv_cache_memory_bytes", None)
    estimate = bool(envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS)
    extensible = bool(getattr(cache, "enable_extensible_kv_cache", False))
    if budget:
        reason = (
            f"a fixed budget (kv_cache_memory_bytes={budget}): core runs no "
            "graph-memory estimate"
        )
    elif not estimate:
        reason = (
            "core's graph-memory estimate is disabled "
            "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0)"
        )
    elif extensible:
        reason = (
            "the extensible KV cache is on: core measures the capture after "
            "warm-up instead of estimating it"
        )
    else:
        reason = (
            "a profiled budget (kv_cache_memory_bytes unset, "
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, extensible KV cache "
            "off): core's graph-memory estimate should have run the profiling "
            "capture"
        )
    return ProfilingRequirement(
        required=not budget and estimate and not extensible,
        reason=reason,
        kv_cache_memory_bytes=budget,
        estimate=estimate,
        extensible_kv_cache=extensible,
    )


@dataclass(frozen=True)
class CaptureCall:
    """One ``capture_cudagraphs`` call after its segment-graph part: what the
    transaction-graph part receives to capture its cells and close the call
    (its spans, the margin and the comparison through this module)."""

    #: ``"profiling capture"`` or ``"serving capture"``.
    capture: str
    profile_only: bool
    device: torch.device
    stream: Any
    graph_pool: Any
    #: The readings at the call's start and after its segment graphs.
    call_free_before: int | None
    segment_free_after: int | None
    segment_seconds: float
    #: The required-capture decision, made at a serving call's entry when no
    #: profiling capture ran in the worker (``None`` otherwise).
    requirement: ProfilingRequirement | None
    #: Whether a profiling capture ran in the worker before this call.
    profiling_ran: bool

    @property
    def segment_total(self) -> int | None:
        """The segment-graph part's retained bytes."""
        return retained(self.call_free_before, self.segment_free_after)


# @spec REG-MEM-027
def serving_excess(serving_total: int | None, profiling_total: int | None) -> int:
    """The aggregate comparison: by how many bytes a serving call's total
    exceeds the profiling call's total (which includes the margin's charge
    ``r``); 0 when it does not, or when either total is not measured."""
    if serving_total is None or profiling_total is None:
        return 0
    return max(serving_total - profiling_total, 0)


__all__ = [
    "MARGIN_BYTES",
    "CaptureCall",
    "ChargedMargin",
    "ProfilingRequirement",
    "allocate_margin",
    "begin_transient_peak",
    "format_bytes",
    "free_memory",
    "profiling_requirement",
    "release_error_frames",
    "retained",
    "serving_excess",
    "transient_peak",
]
