# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transaction graphs: the streaming transaction's body and scatters replayed
from CUDA graphs, split by row kind (transaction-graph design).

With ``transaction_graph_populations`` on (largest entry ``N``), the model
holds one :class:`TransactionGraphBinding`. Through core's model-state capture
hook it captures ``2N`` *cells*, each one ``torch.cuda.CUDAGraph`` of
``advance.model_rows_body`` and its predicated scatters on a one-kind
sub-plan: a ``chunk`` cell and an ``echo`` cell at every count from 1 to
``N``. A scheduled step is classified from its CPU row plan by its CHUNK
count ``K`` and its echo count ``E``; a covered step replays the chunk cell
at ``K`` and the echo cell at ``E`` between one commit reservation and its
one stage. Any other step is counted and runs wholly as the option-off
transaction (``advance.advance_model_rows``), which is the reference.

Every cell reads and writes binding-owned buffers at fixed addresses: the
step arena (the per-row ``int64`` and ``bool`` vectors of both kinds, one
upload per step from one of two pinned host buffers), the step static inputs
and the step outputs. The cells are captured into the graph pool and on the
stream core passes; the binding owns no pool, no stream and no memory
admission. Graph memory is measured through ``graph_memory``, the one
implementation of that measurement.

A failure from a step's first launch on latches the binding: the worker is
not reused, and every later step that schedules a row raises.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from vllm_nemotron_asr_streaming.lookahead import (
    cadence_label,
    geometry_index,
    served_lookahead,
)
from vllm_nemotron_asr_streaming.nemotron_asr import (
    advance,
    graph_memory,
    state_scatter,
)
from vllm_nemotron_asr_streaming.transaction_graph_config import (
    KEY,
    largest_population,
    transaction_graphs_on,
)

logger = logging.getLogger(__name__)

CHUNK = "chunk"
ECHO = "echo"
KINDS = (CHUNK, ECHO)

#: Fallback reasons, in the order classification assigns them.
REASONS = (
    "memory-profile",
    "native-burst",
    "empty",
    "multi-geometry",
    "chunk-uncovered",
    "echo-uncovered",
    "uncaptured",
    "signature",
)

#: The model attribute that holds the binding (``None`` while off).
BINDING_ATTRIBUTE = "_transaction_graph_binding"

PROFILING = "profiling"
SERVING = "serving"

LATCHED = (
    "transaction graphs failed on this worker; a fresh engine is required "
    f"({KEY} is on, and a failure after a cell launch, a failed verification "
    "or a changed process constant is not recoverable in this worker)"
)

# The capture record's keys (TXG-OBS-003).
R_CALL_TOTAL = "call_total"
R_TXN_TOTAL = "transaction_total"
R_KIND_TOTALS = "kind_totals"
R_CELLS = "cells"
R_PEAK = "transient_peak"
R_SECONDS = "capture_seconds"
R_IDENTITY = "configuration_identity"
R_MARGIN = "margin_bytes"
R_FREE_BEFORE = "margin_free_before"
R_FREE_AFTER = "margin_free_after"
R_CHARGE = "margin_charge"
R_ABSORBED = "absorbed_rounding"
R_EFFECTIVE = "effective_cache"
R_REQUIRED = "profiling_required"
R_REASON = "profiling_reason"
R_RAN = "profiling_ran"
R_BUDGET = "budget_setting"
R_ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"
R_CUBLAS = "CUBLAS_WORKSPACE_CONFIG"
R_VERIFICATION_SECONDS = "verification_seconds"

# The step arena (TXG-EXEC-005): ``int64``, one row per section. A section
# holds, for ``N`` rows, one vector of ``N`` entries per scalar field, then
# the rows' initial book rows as an ``(N, BOOK_SLOTS)`` row-major block, so
# that a cell's view of each is contiguous. The last three scalar fields are
# the endpoint mode, threshold and residue frames, which only the chunk
# section's body reads (the policy-free endpoint bookkeeping); both sections
# have one layout, and the echo section's three vectors are never read.
F_BLOCK, F_POSITION, F_FRESH, F_FILE, F_REJECTED = range(5)
F_GEOMETRY, F_ADMITTED, F_PRIOR, F_CADENCE = range(5, 9)
F_ENDPOINT_MODE, F_ENDPOINT_THRESHOLD, F_ENDPOINT_RESIDUE = range(9, 12)
SCALAR_FIELDS = 12
BOOK_SLOTS = 7
SECTION = {CHUNK: 0, ECHO: 1}

#: The raw envelope carrier's dtype, which ``advance.prepare_model_rows``
#: requires of ``inputs_embeds``.
CARRIER_DTYPE = torch.float32

# The phases of a capture call's transaction-graph part (TXG-MEM-010).
PHASE_BUFFERS = "binding-buffer allocation"
PHASE_WARMUP = "scatter warm-up"
PHASE_REFERENCE = "eager reference"
PHASE_WARMUPS = "warm-ups"
PHASE_CAPTURE = "capture"
PHASE_VALIDATION = "validation replay"
PHASE_MARGIN = "margin allocation"
CELL_PHASES = (PHASE_REFERENCE, PHASE_WARMUPS, PHASE_CAPTURE, PHASE_VALIDATION)


# ---------------------------------------------------------------------------
# Cells and classification


# @spec TXG-CELL-001
def cell_keys(n: int, geometry_index: int) -> list[tuple[int, str, int]]:
    """The ``2N`` cell keys ``(geometry_index, kind, count)`` of the
    inventory, in capture order: chunk cells from ``n`` down to 1, then echo
    cells from ``n`` down to 1 (TXG-CAP-007)."""
    return [
        (int(geometry_index), kind, count)
        for kind in KINDS
        for count in range(int(n), 0, -1)
    ]


@dataclasses.dataclass(frozen=True)
class Classification:
    """What a step needs: its population ``P``, CHUNK count ``K`` and echo
    count ``E``, the keys of the cells it needs (the chunk cell first), and,
    when it is not covered, the reason with the fallback counter's ``kind``
    and ``count``."""

    population: int
    chunk_count: int
    echo_count: int
    cells: tuple[tuple[int, str, int], ...] = ()
    reason: str | None = None
    kind: str | None = None
    count: int | None = None


# @spec TXG-COV-001, TXG-COV-002, TXG-COV-004, TXG-COV-006, TXG-COV-007
# @spec TXG-COV-009, TXG-COV-010, TXG-CELL-002
def classify(
    plan: Any,
    *,
    n: int,
    geometry_index: int,
    has_graph: Callable[[tuple[int, str, int]], bool],
    memory_profile: bool = False,
    native_burst: bool = False,
) -> Classification:
    """Classify a step from its CPU row plan alone.

    A step with ``K`` CHUNK rows and ``E`` other rows (planning-failed rows
    are bound non-CHUNK, so they count in ``E``) needs the chunk cell at
    ``K`` when ``K > 0`` and the echo cell at ``E`` when ``E > 0``, each at
    exactly its count. It reads no device memory, changes nothing and never
    raises on a row plan; signatures are compared by ``execute``.
    """
    population = int(plan.num_decodes) + int(plan.num_prefills)
    is_chunk = plan.is_chunk
    chunk_count = int(is_chunk.sum())
    echo_count = population - chunk_count

    def declined(
        reason: str, kind: str | None = None, count: int | None = None
    ) -> Classification:
        return Classification(
            population, chunk_count, echo_count, (), reason, kind, count
        )

    if memory_profile:
        return declined("memory-profile")
    if native_burst:
        return declined("native-burst")
    if population < 1:
        return declined("empty")
    if chunk_count and bool(
        (is_chunk & (plan.geometry_id != int(geometry_index))).any()
    ):
        # Defensive: the production planner binds a foreign-geometry row
        # non-CHUNK. The declined step runs option-off at the server's
        # geometry, whose device geometry status rejects the row.
        return declined("multi-geometry")
    if chunk_count > n:
        return declined("chunk-uncovered", CHUNK, chunk_count)
    if echo_count > n:
        return declined("echo-uncovered", ECHO, echo_count)
    needed: list[tuple[int, str, int]] = []
    if chunk_count:
        needed.append((int(geometry_index), CHUNK, chunk_count))
    if echo_count:
        needed.append((int(geometry_index), ECHO, echo_count))
    for key in needed:
        if not has_graph(key):
            return declined("uncaptured", key[1], key[2])
    return Classification(population, chunk_count, echo_count, tuple(needed))


def _section_positions(plan: Any) -> tuple[torch.Tensor, torch.Tensor]:
    is_chunk = plan.is_chunk
    return (
        is_chunk.nonzero(as_tuple=True)[0],
        (~is_chunk).nonzero(as_tuple=True)[0],
    )


# @spec TXG-EXEC-009
def arena_rows(plan: Any) -> tuple[list[int], list[int]]:
    """The step positions the arena's chunk section and echo section list,
    each in ascending order: the CHUNK rows, then every other row."""
    chunk, echo = _section_positions(plan)
    return chunk.tolist(), echo.tolist()


@dataclasses.dataclass
class Cell:
    """One captured cell: its key, its graph (``None`` when it holds none),
    its signature and, for a chunk cell, the dense decoder tier it replays
    (TXG-CELL-006)."""

    key: tuple[int, str, int]
    graph: Any
    signature: tuple[tuple[str, Any], ...]
    decoder_tier: int | None = None

    @property
    def kind(self) -> str:
        return self.key[1]

    @property
    def count(self) -> int:
        return self.key[2]


# ---------------------------------------------------------------------------
# Signatures, layouts and process constants


# @spec TXG-CELL-005
def callable_identity(fn: Any) -> Any:
    """A bound method by ``(id(__self__), id(__func__))``, any other callable
    by ``id``."""
    owner = getattr(fn, "__self__", None)
    func = getattr(fn, "__func__", None)
    if owner is not None and func is not None:
        return (id(owner), id(func))
    return id(fn)


def _layout(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype)


def _first_difference(
    captured: tuple[tuple[str, Any], ...], step: tuple[tuple[str, Any], ...]
) -> str | None:
    """The first field in which a step's signature differs from a cell's."""
    if len(captured) != len(step):
        return "fields"
    for (name, mine), (other_name, theirs) in zip(captured, step, strict=True):
        if name != other_name:
            return name
        if mine != theirs:
            return name
    return None


# @spec TXG-CAP-016
def process_constants(core: Any) -> tuple[tuple[str, Any], ...]:
    """The process settings every graph was captured under: the
    batch-invariant mode, the per-operation FP32 precision settings and
    ``CUBLAS_WORKSPACE_CONFIG``. Reads the per-operation settings only (the
    legacy getters raise once a per-operation value is set)."""
    mode = getattr(core, "batch_invariant_mode", None)
    return (
        (
            "batch_invariant_mode",
            (
                bool(getattr(mode, "enabled", False)),
                os.environ.get("VLLM_BATCH_INVARIANT"),
            ),
        ),
        (
            "torch.backends.cuda.matmul.fp32_precision",
            torch.backends.cuda.matmul.fp32_precision,
        ),
        (
            "torch.backends.cudnn.conv.fp32_precision",
            torch.backends.cudnn.conv.fp32_precision,
        ),
        (
            "torch.backends.cudnn.rnn.fp32_precision",
            torch.backends.cudnn.rnn.fp32_precision,
        ),
        ("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG")),
    )


def _same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bit-for-bit equality of two tensors of one shape and dtype."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        integer = {2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size()]
        left, right = left.contiguous().view(integer), right.contiguous().view(integer)
    return bool(torch.equal(left, right))


def _driver_version() -> str | None:
    """The NVIDIA driver version, when the host exposes it."""
    try:
        with open("/proc/driver/nvidia/version") as handle:
            for line in handle:
                if "Kernel Module" in line:
                    for token in line.split():
                        if token[:1].isdigit() and "." in token:
                            return token
    except OSError:
        return None
    return None


def _tensors_of(label: str, value: Any) -> list[tuple[str, torch.Tensor]]:
    """Every tensor a nested binding's storage object holds."""
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        return [(label, value)]
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            found.extend(_tensors_of(f"{label}[{index}]", item))
        return found
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            found.extend(
                _tensors_of(f"{label}.{item.name}", getattr(value, item.name, None))
            )
        storage = getattr(value, "cache_storage", None)
        if callable(storage):
            found.extend(_tensors_of(f"{label}.cache_storage", tuple(storage())))
    return found


# ---------------------------------------------------------------------------
# The binding


@dataclasses.dataclass(frozen=True)
class _CellProgram:
    """What one cell's body runs: the prepare phase's output for the cell's
    capture row plan (its nested launch sites bound), the encoder transition
    the body receives, the cell's signature and, for a chunk cell, its dense
    decoder tier."""

    prepared: Any
    encoder_transition: Any
    signature: tuple[tuple[str, Any], ...]
    tier: int | None


class _CaptureWork:
    """What one capture call's transaction-graph part holds while it runs:
    the state pools (from core's startup access), their snapshot and the
    readings. Released before the call ends, so nothing of it outlives the
    call."""

    def __init__(self, capture: str, call: Any) -> None:
        self.capture = capture
        self.call = call
        self.components: dict[str, torch.Tensor] = {}
        self.roles: Any = None
        self.pool_kwargs: dict[str, Any] = {}
        self.resident: Any = None
        self.capacity = 0
        #: Pages ``1 … N`` of every component pool, in host memory.
        self.snapshot: dict[str, torch.Tensor] = {}
        self.started = time.perf_counter()
        self.transaction_before: int | None = None
        self.cells: list[dict[str, Any]] = []
        self.last_reading: int | None = None

    def release(self) -> None:
        self.components = {}
        self.roles = None
        self.pool_kwargs = {}
        self.resident = None
        self.snapshot = {}


class TransactionGraphBinding:
    """The model's transaction-graph binding (see the module docstring).

    ``cells`` maps a cell key to its :class:`Cell`: profiling cells from the
    profiling capture until ``clear_cudagraphs``, serving cells from the
    serving capture on. ``failed`` is the worker-fatal latch and ``verified``
    whether post-warmup verification has passed.
    """

    # @spec TXG-OBS-007
    def __init__(self, model: Any, vllm_config: Any) -> None:
        config = model.config
        self._model = model
        self._vllm_config = vllm_config
        self.populations = getattr(config, KEY)
        self.n = largest_population(config)
        self.lookahead = served_lookahead(config)
        self.geometry = geometry_index(self.lookahead)
        self.label = cadence_label(self.lookahead)
        scheduler = vllm_config.scheduler_config
        limits = [
            int(value)
            for value in (
                getattr(scheduler, "max_num_seqs", None),
                getattr(scheduler, "max_num_batched_tokens", None),
            )
            if value is not None
        ]
        #: The largest step population.
        self.p_max = max(min(limits) if limits else self.n, self.n)
        self.hidden = int(config.hidden_size)
        self.cells: dict[tuple[int, str, int], Cell] = {}
        self.failed = False
        self.verified = False
        #: Which capture created ``cells``: ``None``, PROFILING or SERVING.
        self._cells_capture: str | None = None
        self._graphs: list[Any] = []
        self._serving_attempted = False
        self._bindings: Any = None
        self._layer: str | None = None
        self._margin: Any = None
        self._device: torch.device | None = None
        self._reserved_at_start: int | None = None
        self._phase = PHASE_BUFFERS
        self._phase_cell: tuple[int, str, int] | None = None
        self._work: _CaptureWork | None = None
        self._verify: dict[str, Any] | None = None
        self._capture_layout: tuple[Any, Any] | None = None
        self._constants: tuple[tuple[str, Any], ...] | None = None
        self._buffer_layout: tuple[Any, ...] = ()
        #: The encoder's modules, listed once per capture call (the signature
        #: reads their lazily bound buffers on every step).
        self._encoder_modules: tuple[Any, ...] | None = None
        self._step = 0
        self._release_buffers()
        cache = getattr(vllm_config, "cache_config", None)
        #: The budget as configured when the model was built: a value core
        #: sets later comes from a persisted startup plan.
        self._constructed_budget = getattr(cache, "kv_cache_memory_bytes", None)
        self._tokens = (
            int(config.audio_chunk_token_id),
            int(config.eos_token_id),
            int(config.file_carrier_token_id),
            int(config.continuation_token_id),
        )
        # Host-only counters (TXG-OBS-001).
        self._replays: dict[tuple[int, str, int], int] = {}
        self._covered = {"one_kind": 0, "split": 0}
        self._fallbacks: dict[tuple[str, str | None, int | None], int] = {}
        self._histogram: dict[tuple[int, int], int] = {}
        self._prologue_seconds = 0.0
        self._warned: set[tuple[str, str]] = set()
        self._record: dict[str, Any] = {
            R_IDENTITY: self._identity(),
            PROFILING: None,
            SERVING: None,
            R_MARGIN: None,
            R_FREE_BEFORE: None,
            R_FREE_AFTER: None,
            R_CHARGE: None,
            R_ABSORBED: None,
            R_EFFECTIVE: None,
            R_REQUIRED: None,
            R_REASON: None,
            R_RAN: None,
            R_BUDGET: None,
            R_VERIFICATION_SECONDS: None,
        }
        self._record_settings()
        # A second engine in this process starts with clean module-level
        # batch statistics.
        advance.consume_batch_stats()

    # -- configuration and record -------------------------------------------

    def _identity(self) -> dict[str, Any]:
        config = self._model.config
        scheduler = self._vllm_config.scheduler_config
        try:
            from importlib.metadata import version

            plugin = version("vllm-nemotron-asr-streaming")
        except Exception:  # noqa: BLE001 - an uninstalled source tree
            plugin = None
        return {
            KEY: list(self.populations),
            "N": self.n,
            "geometry": self.label,
            "lookahead": self.lookahead,
            "encoder_execution_arm": getattr(config, "encoder_execution_arm", None),
            "decode_dispatch_arm": getattr(config, "decode_dispatch_arm", None),
            "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(
                scheduler, "max_num_batched_tokens", None
            ),
            "async_scheduling": getattr(scheduler, "async_scheduling", None),
            "dtype": str(getattr(self._vllm_config.model_config, "dtype", None)),
            "model": getattr(self._vllm_config.model_config, "model", None),
            "plugin_version": plugin,
            "process_constants": dict(process_constants(self._model.core)),
        }

    def _record_settings(self) -> None:
        """The allocator settings and versions (TXG-MEM-003)."""
        record = self._record
        record[R_ALLOC_CONF] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        record[R_CUBLAS] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        record["torch_version"] = torch.__version__
        record["cuda_version"] = torch.version.cuda
        record["driver_version"] = _driver_version()

    def _allocator_settings(self) -> str:
        record = self._record
        return (
            f"PYTORCH_CUDA_ALLOC_CONF={record[R_ALLOC_CONF]}, "
            f"CUBLAS_WORKSPACE_CONFIG={record[R_CUBLAS]}"
        )

    def _name(self) -> str:
        return f"{KEY}={list(self.populations)!r}, N = {self.n}"

    def _budget_setting(self, requirement: Any) -> tuple[str, bool]:
        """The budget setting as the messages name it, and whether it is a
        fixed budget."""
        budget = requirement.kv_cache_memory_bytes
        if budget:
            plan = (
                "set by a persisted startup plan after the model was built"
                if not self._constructed_budget
                else "set by configuration, not by a startup plan"
            )
            return (
                f"fixed at kv_cache_memory_bytes="
                f"{graph_memory.format_bytes(int(budget))} ({plan})",
                True,
            )
        if requirement.estimate:
            return "profiled with the estimate", False
        return "profiled with the estimate disabled", False

    # -- counters -------------------------------------------------------------

    # @spec TXG-COV-013
    def record_all_failed_step(self, population: int) -> None:
        """An all-failed step: a bypass, not a fallback. One ``(0, P)`` entry
        in the ``(K, E)`` histogram."""
        key = (0, int(population))
        self._histogram[key] = self._histogram.get(key, 0) + 1

    # @spec TXG-OBS-001, TXG-OBS-003
    def counters(self) -> dict[str, Any]:
        """The counters and the capture record, as plain host values."""
        record = dict(self._record)
        for capture in (PROFILING, SERVING):
            if record[capture] is not None:
                figures = dict(record[capture])
                figures[R_CELLS] = [dict(cell) for cell in figures[R_CELLS]]
                figures[R_KIND_TOTALS] = dict(figures[R_KIND_TOTALS])
                record[capture] = figures
        return {
            "counters": {
                "captured_cells": sum(
                    1 for cell in self.cells.values() if cell.graph is not None
                ),
                "replays": dict(self._replays),
                "covered_steps": dict(self._covered),
                "fallbacks": dict(self._fallbacks),
                "histogram": dict(self._histogram),
                "prologue_host_seconds": self._prologue_seconds,
            },
            "capture_record": record,
        }

    def _summary(self) -> str:
        by_reason: dict[str, int] = {}
        for (reason, _kind, _count), value in self._fallbacks.items():
            by_reason[reason] = by_reason.get(reason, 0) + value
        return (
            f"Nemotron transaction graphs ({self._name()}) at shutdown: "
            f"{sum(self._replays.values())} cell replays; "
            f"{self._covered['split']} split steps and "
            f"{self._covered['one_kind']} one-kind steps covered; "
            f"{sum(by_reason.values())} fallbacks by reason {by_reason}"
        )

    # -- bindings -------------------------------------------------------------

    # @spec TXG-LIFE-009
    def check_bindings(self, bindings: Any, layer: str) -> None:
        """Before ``initialize_state_slots`` takes new bindings: once serving
        cells exist, a pool layout other than the one the serving capture
        recorded is worker-fatal, because the cells bake in the captured
        addresses. Profiling bindings take no part."""
        if self._cells_capture != SERVING or self._verify is None:
            return
        live = dict(bindings.pool_layout(layer))
        if live != self._verify["layout"]:
            self.failed = True
            raise RuntimeError(
                f"{KEY}: initialize_state_slots received serving state pools "
                "whose layout differs from the layout the serving capture "
                "recorded; the captured cells bake in those addresses. " + LATCHED
            )

    # @spec TXG-LIFE-003
    def record_bindings(self, bindings: Any, layer: str) -> None:
        """Record the bindings ``initialize_state_slots`` received: the
        profiling bindings, then the serving bindings."""
        self._bindings = bindings
        self._layer = layer

    # -- capture entry ----------------------------------------------------------

    # @spec TXG-LIFE-007, TXG-LIFE-008
    def check_capture_entry(self, *, profile_only: bool) -> None:
        """The ordering checks of a capture call, at its entry, before any
        graph is captured: TXG-LIFE-007, then TXG-LIFE-008. A serving capture
        counts as attempted once it passed its entry checks
        (``begin_capture``), whether or not it then succeeded; a call refused
        at its entry does not count. A refusal changes nothing: it leaves the
        latch and every existing graph as they are."""
        if not profile_only and self._cells_capture == PROFILING:
            raise RuntimeError(
                f"{KEY}: a serving capture while profiling cells exist is a "
                "lifecycle defect: core releases the profiling capture "
                "(clear_cudagraphs) before it captures for serving"
            )
        if profile_only and self._serving_attempted:
            raise RuntimeError(
                f"{KEY}: a profiling capture after a serving capture was "
                "attempted in this worker is a lifecycle defect: core captures "
                "for its graph-memory estimate before it serves; recover with "
                "an engine restart"
            )
        # TXG-LIFE-008, after the ordering checks of TXG-LIFE-007.
        if not profile_only and self._serving_attempted:
            raise RuntimeError(
                f"{KEY}: a serving capture after a serving capture was "
                "attempted in this worker, whether or not that attempt "
                "succeeded. Recapture over live state pools is not supported "
                "(the cells bake in the pools' addresses while streams hold "
                "slots); recover with an engine restart"
            )

    # @spec TXG-MEM-007, TXG-MEM-009
    def check_profiling_requirement(
        self, requirement: Any, *, profiling_ran: bool
    ) -> None:
        """At the serving capture's entry: record whether a profiling capture
        was required, from the worker's effective cache configuration, and
        refuse the call when one was required and none ran."""
        setting, _fixed = self._budget_setting(requirement)
        self._record.update(
            {
                R_REQUIRED: bool(requirement.required),
                R_REASON: requirement.reason,
                R_RAN: bool(profiling_ran),
                R_BUDGET: setting,
                R_EFFECTIVE: {
                    "kv_cache_memory_bytes": requirement.kv_cache_memory_bytes,
                    "enable_extensible_kv_cache": bool(requirement.extensible_kv_cache),
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": bool(
                        requirement.estimate
                    ),
                },
            }
        )
        if requirement.required and not profiling_ran:
            raise RuntimeError(
                f"{KEY}: the serving capture found no profiling capture in this "
                f"worker although one was required: {requirement.reason}. "
                "Core's graph-memory estimate "
                "(capture_cudagraphs(profile_only=True)) did not run, so the "
                "transaction graphs and segment graphs would not be charged to "
                "the budget. Restart the engine with the estimate enabled, or "
                "set a fixed --kv-cache-memory-bytes that leaves room for them"
            )

    # @spec TXG-MEM-002
    def begin_capture(self, *, profile_only: bool, device: torch.device) -> None:
        """A capture call passed its entry checks: start its transient peak
        (peak statistics reset at the call's start) and, for a serving call,
        count it as attempted."""
        if not profile_only:
            self._serving_attempted = True
        self._device = device
        self._reserved_at_start = graph_memory.begin_transient_peak(device)
        self._record_settings()

    # -- buffers ----------------------------------------------------------------

    def _release_buffers(self) -> None:
        self._arena: torch.Tensor | None = None
        self._hosts: list[torch.Tensor] = []
        self._events: list[Any] = [None, None]
        self._pack: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._ids_static: torch.Tensor | None = None
        self._embeds_static: torch.Tensor | None = None
        self._rows_step: torch.Tensor | None = None
        self._status_step: torch.Tensor | None = None
        self._progress_step: torch.Tensor | None = None
        self._remaining_step: torch.Tensor | None = None
        self._all_true: torch.Tensor | None = None
        self._all_false: torch.Tensor | None = None
        self._row_index: torch.Tensor | None = None
        self._buffer_layout = ()

    def _owned_buffers(self) -> list[tuple[str, torch.Tensor]]:
        named = [
            ("step arena", self._arena),
            ("ids_static", self._ids_static),
            ("embeds_static", self._embeds_static),
            ("rows_step", self._rows_step),
            ("status_step", self._status_step),
            ("progress_step", self._progress_step),
            ("remaining_step", self._remaining_step),
            ("chunk mask", self._all_true),
            ("echo mask", self._all_false),
            ("row index", self._row_index),
        ]
        return [(name, tensor) for name, tensor in named if tensor is not None]

    # @spec TXG-EXEC-005, TXG-EXEC-007, TXG-EXEC-008
    def _allocate_buffers(self, device: torch.device) -> None:
        """The binding-owned buffers every cell reads or writes at fixed
        addresses, outside the graph pool: the step arena and its two pinned
        host buffers, the step static inputs and the step outputs."""
        n, p_max, hidden = self.n, self.p_max, self.hidden
        pin = torch.cuda.is_available()
        width = (SCALAR_FIELDS + BOOK_SLOTS) * n
        self._hosts = [
            torch.zeros((2, width), dtype=torch.int64, pin_memory=pin) for _ in range(2)
        ]
        self._events = [None, None]
        self._pack = torch.zeros((SCALAR_FIELDS, p_max), dtype=torch.int64)
        self._positions = torch.arange(p_max, dtype=torch.int64)
        self._arena = torch.zeros((2, width), dtype=torch.int64, device=device)
        self._ids_static = torch.zeros(p_max, dtype=torch.int64, device=device)
        self._embeds_static = torch.zeros(
            (p_max, hidden), dtype=CARRIER_DTYPE, device=device
        )
        self._rows_step = torch.zeros(
            (p_max, hidden), dtype=CARRIER_DTYPE, device=device
        )
        self._status_step = torch.zeros(p_max, dtype=torch.int32, device=device)
        self._progress_step = torch.zeros((p_max, 3), dtype=torch.int64, device=device)
        self._remaining_step = torch.zeros(p_max, dtype=torch.int32, device=device)
        self._all_true = torch.ones(n, dtype=torch.bool, device=device)
        self._all_false = torch.zeros(n, dtype=torch.bool, device=device)
        self._row_index = torch.arange(n, dtype=torch.int64, device=device)
        self._buffer_layout = tuple(
            (name, *_layout(tensor)) for name, tensor in self._owned_buffers()
        )

    def _pack_arena(
        self,
        prepared: Any,
        chunk_positions: torch.Tensor,
        echo_positions: torch.Tensor,
        host: torch.Tensor,
    ) -> None:
        """Write both arena sections of a row plan into a pinned host
        buffer: every per-row ``int64`` and ``bool`` vector the body reads,
        each kind's rows in ascending step position."""
        assert self._pack is not None and self._positions is not None
        rows = int(prepared.rows)
        table = self._pack[:, :rows]
        table[F_BLOCK] = prepared.composed_indices
        table[F_POSITION] = self._positions[:rows]
        table[F_FRESH] = prepared.is_fresh
        table[F_FILE] = prepared.is_file
        table[F_REJECTED] = prepared.planned_rejection
        table[F_GEOMETRY] = prepared.geometry
        table[F_ADMITTED] = prepared.admitted_prompt
        table[F_PRIOR] = prepared.prior_prompt
        table[F_CADENCE] = prepared.cadence_frames
        table[F_ENDPOINT_MODE] = prepared.endpoint_mode
        table[F_ENDPOINT_THRESHOLD] = prepared.endpoint_threshold
        table[F_ENDPOINT_RESIDUE] = prepared.endpoint_residue
        n, scalars = self.n, SCALAR_FIELDS * self.n
        for section, positions in enumerate((chunk_positions, echo_positions)):
            count = int(positions.numel())
            if not count:
                continue
            row = host[section]
            row[:scalars].view(SCALAR_FIELDS, n)[:, :count] = table.index_select(
                1, positions
            )
            row[scalars:].view(n, BOOK_SLOTS)[:count] = (
                prepared.initial_book.index_select(0, positions)
            )

    # @spec TXG-EXEC-006
    def _upload_arena(
        self,
        prepared: Any,
        chunk_positions: torch.Tensor,
        echo_positions: torch.Tensor,
    ) -> None:
        """Pack both sections into one of the two pinned host buffers and
        upload the whole arena with one host-to-device copy. A host buffer is
        rewritten only after the event recorded after its previous upload has
        completed."""
        assert self._arena is not None
        index = self._step % 2
        self._step += 1
        event = self._events[index]
        if event is not None:
            event.synchronize()
        host = self._hosts[index]
        self._pack_arena(prepared, chunk_positions, echo_positions, host)
        self._arena.copy_(host, non_blocking=True)
        if event is None:
            event = torch.cuda.Event()
            self._events[index] = event
        event.record()

    # -- the cell body --------------------------------------------------------

    def _run_cell(
        self,
        kind: str,
        count: int,
        prepared: Any,
        pool_kwargs: Mapping[str, Any],
        encoder_transition: Any,
    ) -> None:
        """One cell's work on the first ``count`` rows of its arena section:
        gather the rows' inputs from the step static inputs, run the body
        with the endpoint arguments (``endpoint_enabled`` true, as every
        transaction of a real step runs), issue its prevalidated predicated
        scatters and write the rows' outputs at their step positions. This is
        what a cell captures, and what its eager reference and warm-ups
        run."""
        model = self._model
        config = model.config
        assert self._arena is not None and self._row_index is not None
        n = self.n
        section = self._arena[SECTION[kind]]

        def field(index: int) -> torch.Tensor:
            return section[index * n : index * n + count]

        positions = field(F_POSITION)
        chunk = kind == CHUNK
        rows = self._row_index[:count]
        vectors = advance.RowVectors(
            composed_indices=field(F_BLOCK),
            is_chunk=(self._all_true if chunk else self._all_false)[:count],
            is_file=field(F_FILE) != 0,
            is_fresh=field(F_FRESH) != 0,
            planned_rejection=field(F_REJECTED) != 0,
            initial_book=section[SCALAR_FIELDS * n :].view(n, BOOK_SLOTS)[:count],
            geometry=field(F_GEOMETRY),
            admitted_prompt=field(F_ADMITTED),
            prior_prompt=field(F_PRIOR),
            cadence_frames=field(F_CADENCE),
            bucket_rows=(rows,) if chunk else (),
            chunk_all=rows if chunk else self._row_index[:0],
            # Only a CHUNK bucket reads the endpoint vectors.
            endpoint_mode=field(F_ENDPOINT_MODE) if chunk else None,
            endpoint_threshold=field(F_ENDPOINT_THRESHOLD) if chunk else None,
            endpoint_residue=field(F_ENDPOINT_RESIDUE) if chunk else None,
        )
        body = advance.model_rows_body(
            model.core,
            self._ids_static.index_select(0, positions),
            self._embeds_static.index_select(0, positions),
            prepared,
            vectors,
            **pool_kwargs,
            eou_token_id=int(config.eou_token_id),
            adapter=model._emission_adapter,
            encoder_transition=encoder_transition,
            placeholder_id=int(config.audio_chunk_token_id),
            park_id=int(config.eos_token_id),
            capture=False,
            native_burst_handoff=None,
            file_carrier_id=int(config.file_carrier_token_id),
            continuation_id=int(config.continuation_token_id),
        )
        for op in body.scatters:
            state_scatter.execute_masked_page_scatter_(
                op.pool, op.scratch, op.blocks, op.row_status
            )
        self._rows_step.index_copy_(0, positions, body.rows)
        self._status_step.index_copy_(0, positions, body.status)
        self._progress_step.index_copy_(0, positions, body.progress)
        self._remaining_step.index_copy_(0, positions, body.queue_counts)

    def _outputs(self, count: int) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("projection rows", self._rows_step[:count]),
            ("row status", self._status_step[:count]),
            ("progress", self._progress_step[:count]),
            ("remaining queue counts", self._remaining_step[:count]),
        )

    def _capture_plan(self, kind: str, count: int, capacity: int) -> Any:
        """The row plan of a cell's capture rows: the non-null free pages
        ``1 … count``, the server's geometry, alternating fresh and file
        bits (TXG-CAP-005)."""
        chunk = kind == CHUNK
        fresh = [index % 2 == 0 for index in range(count)]
        file = [index % 2 == 1 for index in range(count)]
        blocks = torch.arange(1, count + 1, dtype=torch.int64)
        deadlines = [index + 1 if chunk else 0 for index in range(count)]
        request_ids = tuple(f"txg-capture-{index}" for index in range(count))
        bindings = tuple(
            advance.PreparedRowBinding(
                request_id=request_ids[index],
                block_id=index + 1,
                admission_generation=1,
                geometry_id=self.geometry,
                prompt_index=0,
                prior_prompt_index=0,
                allow_prompt_transition=False,
                is_chunk=chunk,
                ready_deadline_ns=deadlines[index],
            )
            for index in range(count)
        )
        tier = self._model._decode_graph_binding.execution_tier(count) if chunk else 0
        return advance.RowPlan(
            state_indices_d=torch.empty((0, 1), dtype=torch.int64),
            num_decodes=0,
            state_indices_p=blocks,
            num_prefills=count,
            has_initial_states_p=~torch.tensor(fresh, dtype=torch.bool),
            null_block_id=0,
            num_pool_blocks=capacity,
            live_block_ids=blocks.clone(),
            geometry_id=torch.full((count,), self.geometry, dtype=torch.int64),
            prompt_index=torch.zeros(count, dtype=torch.int64),
            is_chunk=torch.full((count,), chunk, dtype=torch.bool),
            admission_generation=torch.ones(count, dtype=torch.int64),
            ready_deadline_ns=torch.tensor(deadlines, dtype=torch.int64),
            request_ids=request_ids,
            execution_tier=tier,
            bindings=bindings,
            is_file=torch.tensor(file, dtype=torch.bool),
        )

    def _encoder_buffers(self) -> tuple[Any, ...]:
        """The lazily bound (non-persistent) buffers of the encoder: where
        each lives, and how many there are."""
        modules = self._encoder_modules
        if modules is None:
            modules = tuple(self._model.core.encoder.modules())
        found = []
        for module in modules:
            if not module._non_persistent_buffers_set:
                continue
            for name in sorted(module._non_persistent_buffers_set):
                tensor = module._buffers.get(name)
                if tensor is not None:
                    found.append((name, tensor.data_ptr(), tuple(tensor.shape)))
        return tuple(found)

    # @spec TXG-CELL-004
    def _signature(
        self,
        kind: str,
        *,
        endpoint_enabled: bool,
        endpoint_policy: bool,
        forced_eou: bool,
        capture: bool,
        tokens: tuple[Any, ...],
        carrier: tuple[Any, ...],
        pools: tuple[Any, ...],
        adapter: Any,
        bucket: Any,
        encoder_transition: Any,
    ) -> tuple[tuple[str, Any], ...]:
        """Every structural input a cell's body bakes in, beyond its key
        (geometry, kind, count), and the two row classes the cells exclude:
        whether any of the cell's rows carries an endpoint policy
        (``endpoint_policy``) and whether any is a forced-EOU candidate
        (``forced_eou``), both false at capture."""
        fields: list[tuple[str, Any]] = [
            ("endpoint_enabled", bool(endpoint_enabled)),
            ("endpoint_policy", bool(endpoint_policy)),
            ("forced_eou", bool(forced_eou)),
            ("capture", bool(capture)),
            ("tokens", tokens),
            ("carrier", carrier),
            ("pools", pools),
            ("adapter", callable_identity(adapter)),
        ]
        if kind == CHUNK:
            fields += [
                (
                    "decode",
                    None
                    if bucket is None
                    else callable_identity(bucket.resolved.decode_fn),
                ),
                ("encoder_transition", callable_identity(encoder_transition)),
                (
                    "bucket_transition",
                    None if bucket is None else callable_identity(bucket.transition),
                ),
                ("encoder_buffers", self._encoder_buffers()),
            ]
        fields.append(("buffers", self._buffer_layout))
        return tuple(fields)

    # -- capture --------------------------------------------------------------

    @contextmanager
    def _on_stream(self, stream: Any) -> Iterator[None]:
        """Run on the capture stream core passed (core's capture context has
        already made it current; this makes the dependency explicit)."""
        if stream is None:
            yield
            return
        with torch.cuda.stream(stream):
            yield

    # @spec TXG-CAP-008
    def _startup_pools(self, work: _CaptureWork) -> None:
        """Reach the state pools through core's startup access, valid only
        inside this capture call; no access object is kept."""
        from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
            project_nemotron_state_pools,
        )

        if self._bindings is None or self._layer is None:
            raise RuntimeError(
                "no state-slot bindings were recorded before the capture "
                "(initialize_state_slots did not run)"
            )
        model = self._model
        components = dict(
            self._bindings.startup_pool_access(self._layer).component_pools
        )
        roles = project_nemotron_state_pools(
            components, model._state_spec, n_layers=int(model.config.n_layers)
        )
        work.components = components
        work.roles = roles
        work.capacity = int(roles.replay_queue.shape[0])
        if work.capacity < self.n + 1:
            raise RuntimeError(
                f"the state pool holds {work.capacity - 1} non-null pages, fewer "
                f"than the {self.n} capture rows of the largest cell"
            )
        work.pool_kwargs = {
            "channel_pools": list(roles.channel),
            "time_pools": list(roles.convolution),
            "len_pools": list(roles.valid_length),
            "h_pool": roles.predictor_h,
            "c_pool": roles.predictor_c,
            "queue_pool": roles.replay_queue,
            "book_pool": roles.replay_book,
            "frontend_raw_pool": roles.frontend_raw,
            "frontend_mel_pool": roles.frontend_mel,
            "frontend_counter_pool": roles.frontend_counters,
            "endpoint_history_pool": roles.endpoint_history,
            "endpoint_book_pool": roles.endpoint_book,
        }
        work.resident = advance.resident_layout(**work.pool_kwargs)
        # The null page and the capture rows' pages, to prove that no capture
        # stores a page, the null page included (TXG-CAP-008): copied to host
        # memory once per call, so the snapshot adds no device memory beside
        # the state pages and the KV cache.
        work.snapshot = {
            name: pool[: self.n + 1].to("cpu", copy=True)
            for name, pool in components.items()
        }

    # @spec TXG-CAP-006, TXG-CAP-013
    def _scatter_warmup(self, work: _CaptureWork) -> None:
        """Launch, for each component pool and then for each projected
        resident-pool view a transaction passes to the kernel, one scatter
        row with a rejected status that names the first non-null free page.
        The kernel's status predicate returns before it reads the page ID or
        any resident address. A projected view with the address, shape,
        strides and dtype of an argument already launched is the same
        argument and is not launched again."""
        launched: set[tuple[Any, ...]] = set()
        views = advance.resident_pool_sequence(**work.pool_kwargs)
        for pool in (*work.components.values(), *views):
            layout = _layout(pool)
            if layout in launched:
                continue
            launched.add(layout)
            state_scatter.launch_rejected_row_scatter(pool, page=1)

    def _reading(self, work: _CaptureWork) -> int | None:
        call = work.call
        work.last_reading = graph_memory.free_memory(call.device, call.stream)
        return work.last_reading

    def _cell_program(
        self,
        kind: str,
        count: int,
        *,
        capacity: int,
        resident: Any,
        pools: tuple[Any, ...],
    ) -> _CellProgram:
        """What the cell at ``(kind, count)`` runs: the prepare phase's
        output for its capture row plan over pools of ``capacity`` pages
        (layout ``resident``, signature layouts ``pools``), with each nested
        launch site bound to its dispatcher, and the cell's signature. Host
        only: it reads no device memory and launches nothing."""
        model = self._model
        config = model.config
        chunk = kind == CHUNK
        plan = self._capture_plan(kind, count, capacity)
        served_transition = model._encoder_execution.transition
        prepared = advance.prepare_model_rows(
            model.core,
            self._ids_static[:count],
            self._embeds_static[:count],
            plan,
            resident=resident,
            decode_resolver=model._decode_resolver,
            eou_token_id=int(config.eou_token_id),
            encoder_transition=served_transition,
            bucket_transition=None,
            has_commit_sink=False,
            capture=False,
            graph_covers_decode=model._decode_graph_binding is not None,
            memory_profile=False,
            file_carrier_id=int(config.file_carrier_token_id),
            continuation_id=int(config.continuation_token_id),
            geometry_order=None,
        )
        bucket = prepared.buckets[0] if chunk else None
        signature = self._signature(
            kind,
            endpoint_enabled=prepared.endpoint_enabled,
            endpoint_policy=False,
            forced_eou=False,
            capture=False,
            tokens=self._tokens,
            carrier=(self.hidden, CARRIER_DTYPE),
            pools=pools,
            adapter=model._emission_adapter,
            bucket=bucket,
            encoder_transition=served_transition,
        )
        tier = None
        encoder_transition = served_transition
        if bucket is not None:
            tier = int(plan.execution_tier)
            # Each nested launch site sees the eager mode through the flat
            # capture's context variable and then runs over its binding's own
            # storage and padding, so the cell records the nested kernels and
            # no nested graph launch.
            decode = _nested_launch(
                bucket.resolved.decode_fn,
                model._decode_graph_binding.uncaptured_decode_fn(
                    geometry=self.geometry, tier=tier
                ),
            )
            encoder_transition = _nested_launch(
                served_transition,
                model._encoder_execution.uncaptured_transition(
                    geometry=self.geometry, count=count
                ),
            )
            prepared = dataclasses.replace(
                prepared,
                buckets=(
                    dataclasses.replace(
                        bucket,
                        resolved=dataclasses.replace(bucket.resolved, decode_fn=decode),
                    ),
                ),
            )
        return _CellProgram(prepared, encoder_transition, signature, tier)

    # @spec TXG-CAP-001, TXG-CAP-004, TXG-CAP-005, TXG-CAP-009, TXG-CELL-001
    def _capture_cell(self, work: _CaptureWork, kind: str, count: int) -> None:
        """Capture one cell: an eager reference of the body on its rejected
        capture rows (launching the nested segment graphs), three warm-ups
        and the capture with the nested bindings running eagerly, and one
        validation replay. The replay's outputs must equal the reference's
        bit for bit, and the null page and the capture rows' pages 1 ... k of
        every component pool (k the cell's count) must equal the snapshot
        taken before the call. Only then is the cell published."""
        import vllm.compilation.cudagraph_pool as cudagraph_pool

        config = self._model.config
        call = work.call
        key = (self.geometry, kind, count)
        chunk = kind == CHUNK
        self._phase_cell = key
        self._phase = PHASE_REFERENCE
        # The capture rows' inputs: the stream carrier token and a zero
        # envelope header. A CHUNK row is rejected by its header and any
        # other row by its token, whatever page it gathers.
        self._ids_static[:count].fill_(int(config.audio_chunk_token_id))
        self._embeds_static[:count].zero_()
        program = self._cell_program(
            kind,
            count,
            capacity=work.capacity,
            resident=work.resident,
            pools=tuple(
                _layout(pool)
                for pool in advance.resident_pool_sequence(**work.pool_kwargs)
            ),
        )
        prepared = program.prepared
        positions = torch.arange(count, dtype=torch.int64)
        empty = positions[:0]
        self._upload_arena(
            prepared, positions if chunk else empty, empty if chunk else positions
        )

        def run() -> None:
            self._run_cell(
                kind, count, prepared, work.pool_kwargs, program.encoder_transition
            )

        run()
        expected = [(name, value.clone()) for name, value in self._outputs(count)]
        self._phase = PHASE_WARMUPS
        with advance.flat_capture():
            for _ in range(3):
                run()
        # Until the validation replay writes them, the outputs hold no value
        # an earlier run left.
        self._rows_step[:count].fill_(float("nan"))
        self._status_step[:count].fill_(-1)
        self._progress_step[:count].fill_(-1)
        self._remaining_step[:count].fill_(-1)
        self._phase = PHASE_CAPTURE
        self._graphs.append(torch.cuda.CUDAGraph())
        with (
            advance.flat_capture(),
            cudagraph_pool.capture_pool(call.graph_pool, self._vllm_config) as pool,
            torch.cuda.graph(self._graphs[-1], pool=pool, stream=call.stream),
        ):
            run()
        self._phase = PHASE_VALIDATION
        self._graphs[-1].replay()
        for (name, reference), (_name, actual) in zip(
            expected, self._outputs(count), strict=True
        ):
            if not _same_bits(reference, actual):
                raise RuntimeError(
                    f"the validation replay's {name} differ from the eager "
                    "reference of the same capture rows"
                )
        # The pages are compared in host memory with the call's snapshot: the
        # null page and the cell's capture rows' pages.
        for name, pool in work.components.items():
            if not _same_bits(
                pool[: count + 1].cpu(), work.snapshot[name][: count + 1]
            ):
                raise RuntimeError(
                    f"the null page or the capture rows' pages of state pool "
                    f"{name!r} changed: the reference and the validation "
                    "replay store no page"
                )
        self.cells[key] = Cell(key, self._graphs[-1], program.signature, program.tier)

    # @spec TXG-LIFE-004, TXG-LIFE-006, TXG-CAP-007, TXG-CAP-010, TXG-CAP-011
    # @spec TXG-MEM-001, TXG-MEM-002, TXG-MEM-004, TXG-MEM-005, TXG-MEM-006
    # @spec TXG-MEM-008, TXG-MEM-010, TXG-OBS-004
    def capture(self, call: Any) -> None:
        """The transaction-graph part of one ``capture_cudagraphs`` call,
        after its segment graphs: the binding-owned buffers, the scatter
        warm-up, the ``2N`` cells largest first (chunk cells, then echo
        cells), then the charged margin (profiling call) or the memory
        comparison (serving call). ``call`` is ``graph_memory.CaptureCall``.

        A failure here raises an error chained from the original that names
        the phase and the figures; the caller discards what the call created
        (``discard_capture``). The original error's finished frames are
        cleared first, so it holds none of the call's tensors or graphs.
        """
        capture = PROFILING if call.profile_only else SERVING
        work = _CaptureWork(capture, call)
        self._work = work
        self._phase, self._phase_cell = PHASE_BUFFERS, None
        self._cells_capture = capture
        self.verified = False
        self._encoder_modules = tuple(self._model.core.encoder.modules())
        try:
            with self._on_stream(call.stream):
                self._allocate_buffers(call.device)
                self._startup_pools(work)
                work.transaction_before = self._reading(work)
                self._phase = PHASE_WARMUP
                self._scatter_warmup(work)
                kind_before = self._reading(work)
                kind_totals: dict[str, int | None] = {}
                for kind in KINDS:
                    before = kind_before
                    for count in range(self.n, 0, -1):
                        self._capture_cell(work, kind, count)
                        after = self._reading(work)
                        work.cells.append(
                            {
                                "geometry": self.geometry,
                                "kind": kind,
                                "count": count,
                                "growth": graph_memory.retained(before, after),
                            }
                        )
                        before = after
                    kind_totals[kind] = graph_memory.retained(kind_before, before)
                    kind_before = before
                self._phase_cell = None
                transaction_after = kind_before
                buffers = self._encoder_buffers()
                for cell in self.cells.values():
                    captured = dict(cell.signature).get("encoder_buffers", buffers)
                    if captured != buffers:
                        raise RuntimeError(
                            "a lazily bound encoder buffer was rebound while "
                            "the cells were captured: the "
                            f"{cell.kind} cell at count {cell.count} holds its "
                            "old address"
                        )
                # Plain values for post-warmup verification, read while the
                # pools are in hand; the snapshot and the pools are then
                # dropped, so the call total excludes them.
                self._capture_layout = (
                    next(iter(work.components.values())).device,
                    self._role_table(
                        work.roles,
                        min(pool.data_ptr() for pool in work.components.values()),
                    ),
                )
                work.release()
                if call.profile_only:
                    self._phase = PHASE_MARGIN
                    self._margin = graph_memory.allocate_margin(
                        call.device, stream=call.stream
                    )
                    final = self._margin.free_after
                else:
                    final = self._reading(work)
        except Exception as error:
            message = self._failure_message(work, error)
            work.release()
            graph_memory.release_error_frames(error)
            raise RuntimeError(message) from error
        finally:
            self._work = None
            # Module-level batch statistics of the capture rows are not a
            # step's.
            advance.consume_batch_stats()
        figures = {
            R_CALL_TOTAL: graph_memory.retained(call.call_free_before, final),
            R_TXN_TOTAL: graph_memory.retained(
                work.transaction_before, transaction_after
            ),
            R_KIND_TOTALS: kind_totals,
            R_CELLS: work.cells,
            R_PEAK: graph_memory.transient_peak(call.device, self._reserved_at_start),
            R_SECONDS: call.segment_seconds + (time.perf_counter() - work.started),
        }
        self._record[capture] = figures
        if call.profile_only:
            margin = self._margin
            self._record.update(
                {
                    R_MARGIN: margin.margin_bytes,
                    R_FREE_BEFORE: margin.free_before,
                    R_FREE_AFTER: margin.free_after,
                    R_CHARGE: margin.charge,
                }
            )
        else:
            self._finish_serving(call, figures)
        logger.info(
            "Nemotron transaction graphs (%s): the %s capture captured %d "
            "cells at %s; call total %s, transaction total %s, transient "
            "peak %s, capture time %.3f s",
            self._name(),
            capture,
            len(self.cells),
            self.label,
            graph_memory.format_bytes(figures[R_CALL_TOTAL]),
            graph_memory.format_bytes(figures[R_TXN_TOTAL]),
            graph_memory.format_bytes(figures[R_PEAK]),
            figures[R_SECONDS],
        )

    def _finish_serving(self, call: Any, figures: dict[str, Any]) -> None:
        """End a serving capture: the aggregate comparison with the
        profiling call (or the uncharged-memory warning), and the plain
        values post-warmup verification compares."""
        record = self._record
        serving_total = figures[R_CALL_TOTAL]
        profiling = record[PROFILING] if call.profiling_ran else None
        if profiling is not None:
            profiling_total = profiling[R_CALL_TOTAL]
            excess = graph_memory.serving_excess(serving_total, profiling_total)
            if excess:
                raise RuntimeError(
                    f"{KEY}: the serving capture retains more graph memory "
                    "than the profiling capture core charged to the budget "
                    f"({self._name()}): serving call total "
                    f"{graph_memory.format_bytes(serving_total)}, profiling call "
                    f"total {graph_memory.format_bytes(profiling_total)}, which "
                    "includes the charged margin's retained charge r = "
                    f"{graph_memory.format_bytes(record[R_CHARGE])} of a margin "
                    f"m = {graph_memory.format_bytes(record[R_MARGIN])}; excess "
                    f"{graph_memory.format_bytes(excess)}. Allocator settings: "
                    f"{self._allocator_settings()}. Remedies: a fixed "
                    "--kv-cache-memory-bytes that leaves room for the reported "
                    "serving total (the graphs are then reported as not charged "
                    f"to the budget); a smaller largest population in {KEY}; or "
                    f"transaction graphs off (remove {KEY})"
                )
            if serving_total is not None and profiling_total is not None:
                charge = record[R_CHARGE] or 0
                record[R_ABSORBED] = serving_total - (profiling_total - charge)
        else:
            totals = figures[R_KIND_TOTALS]
            logger.warning(
                "Nemotron transaction graphs and segment graphs were not "
                "charged to the budget (%s): no profiling capture ran in this "
                "worker, and none was required (%s; budget %s). The serving "
                "capture retains call total %s (segment graphs %s; transaction "
                "graphs %s: chunk cells %s, echo cells %s); the budget must "
                "leave room for them.",
                self._name(),
                record[R_REASON],
                record[R_BUDGET],
                graph_memory.format_bytes(serving_total),
                graph_memory.format_bytes(call.segment_total),
                graph_memory.format_bytes(figures[R_TXN_TOTAL]),
                graph_memory.format_bytes(totals[CHUNK]),
                graph_memory.format_bytes(totals[ECHO]),
            )
        self._constants = process_constants(self._model.core)
        self._verify = self._verification_values()

    # @spec TXG-MEM-010, TXG-MEM-011, TXG-MEM-012, TXG-MEM-013
    def _failure_message(self, work: _CaptureWork, error: BaseException) -> str:
        """The message of a failure in a capture call's transaction-graph
        part: the option, the capture, the phase, the failing cell when the
        phase is a cell's, what was retained so far, the transient peak, the
        budget setting, the allocator settings and the remedies."""
        call = work.call
        try:
            now = graph_memory.free_memory(call.device, call.stream)
        except Exception:  # noqa: BLE001 - the original error is reported
            now = None
        try:
            peak = graph_memory.transient_peak(call.device, self._reserved_at_start)
        except Exception:  # noqa: BLE001
            peak = None
        requirement = graph_memory.profiling_requirement(self._vllm_config)
        setting, fixed = self._budget_setting(requirement)
        where = (
            "the profiling capture, before the state pages are planned"
            if call.profile_only
            else "the serving capture"
        )
        phase = self._phase
        cell = ""
        if phase in CELL_PHASES and self._phase_cell is not None:
            _geometry, kind, count = self._phase_cell
            cell = f" of the {kind} cell at count {count} ({self.label})"
        smaller = (
            f"a smaller largest population in {KEY}; a smaller --max-num-seqs; "
            f"or transaction graphs off (remove {KEY})"
        )
        if call.profile_only:
            remedies = smaller
        elif fixed:
            parts, total, unavailable = [], 0, []
            for kind in KINDS:
                growths = [
                    item["growth"]
                    for item in work.cells
                    if item["kind"] == kind and item["growth"] is not None
                ]
                if not growths:
                    unavailable.append(kind)
                    continue
                left = self.n - len(growths)
                estimate = int(sum(growths) / len(growths)) * left
                total += estimate
                parts.append(
                    f"{kind} cells: {left} not yet captured at a mean growth of "
                    f"{graph_memory.format_bytes(int(sum(growths) / len(growths)))}"
                )
            projected = (
                f"a --kv-cache-memory-bytes smaller by at least the projected "
                f"remainder, an estimate of {graph_memory.format_bytes(total)}"
                + (f" ({'; '.join(parts)})" if parts else "")
            )
            if unavailable:
                projected += (
                    "; the estimate is unavailable for the "
                    f"{' and '.join(unavailable)} cells, none of which was "
                    "captured in this call"
                )
            remedies = f"{projected}; or {smaller}"
        else:
            remedies = (
                f"{smaller}; or a lower --gpu-memory-utilization, which helps "
                "only when the allocated state pages and KV cache fill the "
                "budget and leave too little device memory for the capture's "
                "transient peak"
            )
        call_so_far = graph_memory.format_bytes(
            graph_memory.retained(call.call_free_before, now)
        )
        cells_so_far = graph_memory.format_bytes(
            graph_memory.retained(work.transaction_before, now)
        )
        return (
            f"Nemotron transaction-graph capture failed ({self._name()}) in "
            f"{where}: the {phase}{cell} failed "
            f"({type(error).__name__}: {error}). Retained so far: call total "
            f"{call_so_far}, transaction graphs {cells_so_far}"
            f" in {len(work.cells)} captured cells; transient peak "
            f"{graph_memory.format_bytes(peak)}. Budget setting: {setting}. "
            f"Allocator settings: {self._allocator_settings()}. "
            f"Remedies: {remedies}."
        )

    # -- release --------------------------------------------------------------

    def _release(self) -> None:
        """Release every cell, binding-owned buffer, signature, recorded
        layout and the margin."""
        graphs, self._graphs = self._graphs, []
        self.cells = {}
        self._cells_capture = None
        for graph in graphs:
            try:
                graph.reset()
            except Exception:  # noqa: BLE001 - a graph that never captured
                logger.debug("a cell graph could not be reset", exc_info=True)
        del graphs
        self._release_buffers()
        self._encoder_modules = None
        margin, self._margin = self._margin, None
        if margin is not None:
            margin.release()
        self._verify = None
        self._capture_layout = None
        self._constants = None
        self.verified = False

    # @spec TXG-LIFE-012
    def discard_capture(self, error: BaseException | None = None) -> None:
        """A capture call failed after its entry checks with ``error``:
        discard every cell, binding-owned buffer and the margin the call
        created. The finished frames ``error`` holds, through its cause and
        its context, are cleared first, so the propagated error keeps no
        reference to the call's tensors or graphs; its messages and chain
        stay. The capture record keeps its host integers."""
        graph_memory.release_error_frames(error)
        work, self._work = self._work, None
        if work is not None:
            work.release()
        self._release()

    # @spec TXG-LIFE-005, TXG-LIFE-014
    def release_profiling(self) -> None:
        """``clear_cudagraphs``: release what the profiling capture created,
        keeping the capture record. Tolerates a profiling capture that raised
        or never ran, and leaves serving cells alone."""
        if self._cells_capture == SERVING:
            return
        self._release()

    # @spec TXG-LIFE-010, TXG-OBS-006
    def shutdown(self) -> None:
        """``shutdown``: release every cell and binding-owned buffer and log
        the counter summary. Tolerates a binding that never captured."""
        try:
            self._release()
        finally:
            logger.info("%s", self._summary())

    # -- verification ---------------------------------------------------------

    def _role_table(self, roles: Any, base: int) -> dict[str, tuple[Any, ...]]:
        """The projected resident layout per transaction role: byte offset
        in the page, page shape, strides and dtype."""
        table: dict[str, tuple[Any, ...]] = {}
        for item in dataclasses.fields(roles):
            value = getattr(roles, item.name)
            tensors = value if isinstance(value, tuple) else (value,)
            for index, tensor in enumerate(tensors):
                table[f"{item.name}[{index}]"] = (
                    tensor.data_ptr() - base,
                    tuple(tensor.shape[1:]),
                    tensor.stride(),
                    tensor.dtype,
                )
        return table

    def _callable_identities(self) -> dict[str, Any]:
        model = self._model
        identities: dict[str, Any] = {
            "adapter": callable_identity(model._emission_adapter),
            "encoder_transition": callable_identity(
                model._encoder_execution.transition
            ),
        }
        decoder = model._decode_graph_binding
        for cell in self.cells.values():
            if cell.decoder_tier is None:
                continue
            identities[f"decode at count {cell.count}"] = callable_identity(
                decoder.decode_fn(geometry=self.geometry, tier=cell.decoder_tier)
            )
        return identities

    def _external_buffers(self) -> dict[str, tuple[int, int]]:
        """The address and size of every buffer outside the cells that the
        graphs read or write."""
        model = self._model
        named: list[tuple[str, torch.Tensor]] = [
            (f"binding {name}", tensor) for name, tensor in self._owned_buffers()
        ]
        named += [(f"parameter {name}", p) for name, p in model.named_parameters()]
        named += [(f"buffer {name}", b) for name, b in model.named_buffers()]
        decoder = model._decode_graph_binding
        for key, entry in getattr(decoder, "_entries", {}).items():
            named += _tensors_of(f"decoder scratch {key}", entry)
        encoder = model._encoder_execution
        for key, entry in getattr(encoder, "_graph_entries", {}).items():
            named += _tensors_of(f"encoder scratch {key}", entry)
        return {
            name: (tensor.data_ptr(), tensor.numel() * tensor.element_size())
            for name, tensor in named
        }

    def _verification_values(self) -> dict[str, Any]:
        """The plain values post-warmup verification compares, recorded at
        the serving capture."""
        assert self._bindings is not None and self._layer is not None
        assert self._capture_layout is not None
        device, roles = self._capture_layout
        return {
            "layout": dict(self._bindings.pool_layout(self._layer)),
            "device": device,
            "roles": roles,
            "identities": self._callable_identities(),
            "buffers": self._external_buffers(),
            "constants": self._constants,
        }

    def _shadow_role_table(self) -> dict[str, tuple[Any, ...]]:
        """The role table the model's projection gives now, over a one-page
        host shadow of the state layout (no device memory)."""
        from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
            allocate_state_scratch,
            project_nemotron_state_pools,
        )

        model = self._model
        shadow = allocate_state_scratch(model._state_spec, 1, torch.device("cpu"))
        roles = project_nemotron_state_pools(
            shadow.views, shadow.spec, n_layers=int(model.config.n_layers)
        )
        return self._role_table(roles, shadow.raw.data_ptr())

    # @spec TXG-VER-001, TXG-VER-002, TXG-VER-003, TXG-OBS-005
    def verify_post_warmup(self) -> None:
        """``post_warmup``: compare, in order and with plain values and host
        state only, the live pool layout and device, the projected resident
        layout per role, the cells and the callable identities, the address
        and size of every buffer outside the cells, and the process
        constants, with what the serving capture recorded. A mismatch latches
        the binding and raises. Replays nothing and allocates no device
        memory."""
        started = time.perf_counter()
        problem = self._verification_problem()
        if problem is not None:
            self.failed = True
            raise RuntimeError(
                f"{KEY}: post-warmup verification failed: {problem}. " + LATCHED
            )
        self.verified = True
        seconds = time.perf_counter() - started
        self._record[R_VERIFICATION_SECONDS] = seconds
        logger.info(
            "Nemotron transaction graphs (%s): post-warmup verification "
            "passed for %d cells in %.3f s",
            self._name(),
            len(self.cells),
            seconds,
        )

    def _verification_problem(self) -> str | None:
        recorded = self._verify
        if self._cells_capture != SERVING or recorded is None:
            return "no serving capture recorded its cells in this worker"
        assert self._bindings is not None and self._layer is not None
        # 1. the live pool layout, addresses included, and the pools' device
        live = dict(self._bindings.pool_layout(self._layer))
        if live != recorded["layout"]:
            changed = sorted(
                name
                for name in set(live) | set(recorded["layout"])
                if live.get(name) != recorded["layout"].get(name)
            )
            return (
                "the state pool layout differs from the layout the serving "
                f"capture recorded (pools {changed[:4]})"
            )
        parameter = next(self._model.core.encoder.parameters(), None)
        if parameter is not None and parameter.device != recorded["device"]:
            return (
                f"the state pool layout is on {recorded['device']}, the encoder "
                f"parameters on {parameter.device}"
            )
        # 2. the projected resident layout per transaction role
        shadow = self._shadow_role_table()
        if shadow != recorded["roles"]:
            roles = sorted(
                name
                for name in set(shadow) | set(recorded["roles"])
                if shadow.get(name) != recorded["roles"].get(name)
            )
            return (
                "the projected resident layout of the transaction roles "
                f"differs from the captured one (roles {roles[:4]})"
            )
        # 3. all 2N cells hold their graphs, and the callable identities
        for key in cell_keys(self.n, self.geometry):
            cell = self.cells.get(key)
            if cell is None or cell.graph is None:
                return f"the {key[1]} cell at count {key[2]} holds no graph"
        identities = self._callable_identities()
        if identities != recorded["identities"]:
            names = sorted(
                name
                for name in set(identities) | set(recorded["identities"])
                if identities.get(name) != recorded["identities"].get(name)
            )
            return f"a captured callable identity changed ({names[:4]})"
        # 4. every buffer outside the cells that the graphs read or write
        buffers = self._external_buffers()
        if buffers != recorded["buffers"]:
            names = sorted(
                name
                for name in set(buffers) | set(recorded["buffers"])
                if buffers.get(name) != recorded["buffers"].get(name)
            )
            return (
                "the address or size of a buffer the graphs read or write "
                f"changed ({names[:4]})"
            )
        # 5. the process constants
        return self._constants_problem()

    def _constants_problem(self) -> str | None:
        current = process_constants(self._model.core)
        if current == self._constants:
            return None
        changed = [
            f"{name} was {before!r} at the serving capture and is {after!r}"
            for (name, before), (_name, after) in zip(
                self._constants or (), current, strict=False
            )
            if before != after
        ]
        return "a process constant changed: " + "; ".join(changed)

    # -- execution ------------------------------------------------------------

    def _has_graph(self, key: tuple[int, str, int]) -> bool:
        cell = self.cells.get(key)
        return cell is not None and cell.graph is not None

    def _count_fallback(self, reason: str, kind: str | None, count: int | None) -> None:
        key = (reason, kind, count)
        self._fallbacks[key] = self._fallbacks.get(key, 0) + 1

    # @spec TXG-EXEC-001, TXG-EXEC-002, TXG-EXEC-003, TXG-EXEC-004, TXG-EXEC-010
    # @spec TXG-EXEC-013, TXG-EXEC-014, TXG-EXEC-015, TXG-TXN-009, TXG-COV-003
    # @spec TXG-COV-005, TXG-COV-012, TXG-QC-002, TXG-FAIL-002, TXG-FAIL-003
    # @spec TXG-LIFE-013, TXG-MEM-014, TXG-MEM-015
    def execute(
        self,
        core: Any,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        plan: Any,
        **transaction: Any,
    ) -> torch.Tensor | None:
        """Run a scheduled step's transaction from the serving cells.

        Takes ``advance.advance_model_rows``'s arguments. Returns the escaped
        row tensor of a covered step, or ``None`` after counting the step's
        fallback, for the caller to run the option-off transaction.

        The order is fixed: the failed latch; verification; the process
        constants; classification; the prepare phase over the whole row plan;
        each needed cell's signature and graph object; the arena upload and
        the static-input copies; the escaped row tensor; the one commit
        reservation; the chunk cell's launch, then the echo cell's; the
        escaped-row copy, the commit stage and the counters. A failure before
        the first launch propagates with every resident page unchanged and
        the reservation cancelled, without latching, unless that cancel
        itself raises, which latches. From the first launch on a failure
        latches the binding and cancels the reservation, which a stage that
        raised before publishing leaves cancellable. It issues every device
        operation on the current stream and reads no memory statistic.

        A step with a row that carries an endpoint policy, or with a
        forced-EOU candidate (``plan.forced_eou_candidate``, marked on the
        host), is a ``signature`` fallback with field ``endpoint_policy`` or
        ``forced_eou``: the cells run the policy-free endpoint bookkeeping
        only.
        """
        if self.failed:
            raise RuntimeError(LATCHED)
        if not self.verified:
            raise RuntimeError(
                f"{KEY}: a scheduled step reached the transaction graphs before "
                "post-warmup verification passed in this worker (post_warmup "
                "runs after the serving capture and before the worker serves); "
                "this is a lifecycle defect, and no cell was replayed"
            )
        problem = self._constants_problem()
        if problem is not None:
            self.failed = True
            raise RuntimeError(f"{KEY}: {problem}. " + LATCHED)
        started = time.perf_counter()
        step = classify(
            plan,
            n=self.n,
            geometry_index=self.geometry,
            has_graph=self._has_graph,
            memory_profile=bool(transaction.get("memory_profile", False)),
            native_burst=transaction.get("native_burst_handoff") is not None,
        )
        shape = (step.chunk_count, step.echo_count)
        self._histogram[shape] = self._histogram.get(shape, 0) + 1
        if step.reason is not None:
            self._count_fallback(step.reason, step.kind, step.count)
            self._prologue_seconds += time.perf_counter() - started
            return None
        commit_sink = transaction.get("commit_sink")
        pool_kwargs = {
            name: transaction[name]
            for name in (
                "channel_pools",
                "time_pools",
                "len_pools",
                "h_pool",
                "c_pool",
                "queue_pool",
                "book_pool",
                "frontend_raw_pool",
                "frontend_mel_pool",
                "frontend_counter_pool",
            )
        }
        # The endpoint arguments: the forward passes them on every real step
        # (TXG-COV-012); a caller that omits them gets a transaction with
        # ``endpoint_enabled`` false, which no cell's signature matches.
        pool_kwargs["endpoint_history_pool"] = transaction.get("endpoint_history_pool")
        pool_kwargs["endpoint_book_pool"] = transaction.get("endpoint_book_pool")
        prepared = advance.prepare_model_rows(
            core,
            input_ids,
            inputs_embeds,
            plan,
            resident=advance.resident_layout(**pool_kwargs),
            decode_resolver=transaction["decode_resolver"],
            eou_token_id=transaction.get("eou_token_id"),
            encoder_transition=transaction.get("encoder_transition"),
            bucket_transition=transaction.get("bucket_transition"),
            has_commit_sink=commit_sink is not None,
            capture=bool(transaction.get("capture", False)),
            graph_covers_decode=bool(transaction.get("graph_covers_decode", False)),
            memory_profile=False,
            file_carrier_id=transaction.get("file_carrier_id"),
            continuation_id=transaction.get("continuation_id"),
            geometry_order=transaction.get("geometry_order"),
        )
        population = step.population
        if population > self.p_max:
            raise ValueError(
                f"a step of {population} rows exceeds the largest step "
                f"population {self.p_max} the transaction graphs were built for"
            )
        chunk_positions, echo_positions = _section_positions(plan)
        # The signature of each needed cell, before any upload. The two
        # excluded row classes are read per kind from the CPU row plan.
        section_rows = {CHUNK: chunk_positions, ECHO: echo_positions}
        policy_rows = prepared.endpoint_mode != 0
        candidates = plan.forced_eou_candidate
        tokens = (
            transaction.get("placeholder_id"),
            transaction.get("park_id"),
            transaction.get("file_carrier_id"),
            transaction.get("continuation_id"),
        )
        carrier = (int(inputs_embeds.shape[1]), inputs_embeds.dtype)
        pools = tuple(
            _layout(pool) for pool in advance.resident_pool_sequence(**pool_kwargs)
        )
        bucket = prepared.buckets[0] if len(prepared.buckets) == 1 else None
        cells = [self.cells[key] for key in step.cells]
        for cell in cells:
            positions = section_rows[cell.kind]
            signature = self._signature(
                cell.kind,
                endpoint_enabled=prepared.endpoint_enabled,
                endpoint_policy=bool(policy_rows.index_select(0, positions).any()),
                forced_eou=bool(
                    candidates.numel() and candidates.index_select(0, positions).any()
                ),
                capture=bool(transaction.get("capture", False)),
                tokens=tokens,
                carrier=carrier,
                pools=pools,
                adapter=transaction.get("adapter"),
                bucket=bucket,
                encoder_transition=transaction.get("encoder_transition"),
            )
            field = _first_difference(cell.signature, signature)
            if field is not None:
                self._count_fallback("signature", cell.kind, cell.count)
                if (cell.kind, field) not in self._warned:
                    self._warned.add((cell.kind, field))
                    logger.warning(
                        "Nemotron transaction graphs (%s): a step's structural "
                        "input %r differs from the captured signature of the "
                        "%s cells; such steps run as the option-off "
                        "transaction and are counted as signature fallbacks",
                        self._name(),
                        field,
                        cell.kind,
                    )
                self._prologue_seconds += time.perf_counter() - started
                return None
        # Each needed cell's graph object: lost after classification is
        # worker-fatal.
        for cell in cells:
            if cell.graph is None:
                self.failed = True
                raise RuntimeError(
                    f"{KEY}: the {cell.kind} cell at count {cell.count} lost "
                    "its graph after the step was classified as covered. " + LATCHED
                )
        self._upload_arena(prepared, chunk_positions, echo_positions)
        self._ids_static[:population].copy_(input_ids)
        self._embeds_static[:population].copy_(inputs_embeds)
        escaped = torch.empty(
            (population, int(inputs_embeds.shape[1])),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        cancel: Callable[[], None] | None = None
        stage: Callable[[torch.Tensor], None] | None = None
        if commit_sink is not None:
            # One reservation covers all P rows, in row order; the sink's
            # resident wrapper re-reads the rows' component pools first.
            reservation = commit_sink.reserve(
                advance.CommitPlan(
                    bindings=plan.bindings,
                    capture=None,
                    records=(),
                    progress=self._progress_step[:population],
                    queue_counts=self._remaining_step[:population],
                )
            )
            cancel = getattr(reservation, "cancel", None)
            if not callable(cancel):
                raise ValueError("commit sink returned a reservation without cancel()")
            try:
                stage = getattr(reservation, "stage", None)
            except BaseException:
                self._cancel_before_launch(cancel)
                raise
            if not callable(stage):
                self._cancel_before_launch(cancel)
                raise ValueError("commit sink returned a reservation without stage()")
        self._prologue_seconds += time.perf_counter() - started
        try:
            for cell in cells:
                cell.graph.replay()
            escaped.copy_(self._rows_step[:population])
            if stage is not None:
                stage(self._status_step[:population])
            # The counters, after the stage; a failure here latches like any
            # other failure from the first launch on.
            for cell in cells:
                self._replays[cell.key] = self._replays.get(cell.key, 0) + 1
            self._covered["split" if len(cells) == 2 else "one_kind"] += 1
        except BaseException:
            # The first launch may already have scattered: no retry, no
            # fallback of either kind, and this worker is not reused. A stage
            # that raised before it published the reservation left it
            # cancellable; after a successful stage the cancel is a no-op.
            self.failed = True
            if cancel is not None:
                try:
                    cancel()
                except BaseException:  # noqa: BLE001 - the worker is not reused
                    logger.debug("the reservation was not cancelled", exc_info=True)
            raise
        return escaped

    def _cancel_before_launch(self, cancel: Callable[[], None]) -> None:
        """Cancel the held reservation of a step that fails before its first
        launch. No page was written, so the binding does not latch, unless
        the cancel itself raises: the reservation is then unresolved, and the
        binding latches and propagates that failure as worker-fatal."""
        try:
            cancel()
        except BaseException:
            self.failed = True
            raise


def _nested_launch(captured: Any, uncaptured: Any) -> Any:
    """A cell's nested launch site: the binding's captured callable, or,
    while a flat capture is in progress (``advance.flat_capture``), its
    uncaptured callable over the same storage and padding."""

    def launch(*args: Any, **kwargs: Any) -> Any:
        if advance.flat_capture_active():
            return uncaptured(*args, **kwargs)
        return captured(*args, **kwargs)

    return launch


# ---------------------------------------------------------------------------
# The reader


def _reach_model(obj: Any) -> Any:
    """The model behind a model, runner, worker or engine core held in this
    process (a harness convenience over runner and worker attributes, not a
    core contract)."""
    pending, seen = [obj], set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, BINDING_ATTRIBUTE):
            return candidate
        if len(seen) > 16:
            break
        for name in (
            "model",
            "model_runner",
            "worker",
            "driver_worker",
            "model_executor",
        ):
            try:
                pending.append(getattr(candidate, name, None))
            except Exception:  # noqa: BLE001 - a holder that refuses the read
                continue
    raise RuntimeError(
        f"cannot reach the Nemotron model from {type(obj).__name__!r}: pass the "
        "model, its runner, its worker or the engine core, or run this "
        "function on the worker through collective_rpc"
    )


# @spec TXG-OBS-002, TXG-OBS-008, TXG-OBS-009, TXG-OBS-010
def transaction_graph_counters(worker_or_model: Any) -> dict[str, Any]:
    """The transaction-graph counters and capture record of the model behind
    ``worker_or_model`` (a model, runner, worker or engine core in this
    process; under a multiprocess engine, run it on the worker through
    ``AsyncLLM.collective_rpc``). An empty mapping while transaction graphs
    are off. Raises when it cannot reach the model, so that an unreachable
    model is never reported as off."""
    model = _reach_model(worker_or_model)
    binding = getattr(model, BINDING_ATTRIBUTE)
    if binding is None:
        return {}
    return binding.counters()


__all__ = [
    "CHUNK",
    "ECHO",
    "Cell",
    "Classification",
    "TransactionGraphBinding",
    "arena_rows",
    "callable_identity",
    "cell_keys",
    "classify",
    "process_constants",
    "transaction_graph_counters",
    "transaction_graphs_on",
]
