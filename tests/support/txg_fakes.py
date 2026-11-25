# SPDX-License-Identifier: Apache-2.0
"""Doubles for transaction-graph tests: engine configurations, row plans,
cells and commit sinks. Imports no vLLM.

* :func:`engine_config` builds a duck-typed engine configuration in the shape
  ``transaction_graph_config_errors`` reads (see ``support.txg_api``).
* :func:`row_plan` builds a CPU ``advance.RowPlan`` from short row
  descriptions, for classification and planning tests that run no
  transaction.
* :class:`FakeGraph` / :class:`FakeCell` / :func:`fake_inventory` stand for a
  captured cell inventory without CUDA (a fake-capture seam).
* :class:`RecordingSink` is a ``CommitSink`` that records every reservation,
  its plan, the staged status and cancellations.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any

#: The served configuration's control tokens in the synthetic card
#: (``support.card``): park, chunk placeholder, EOU, file carrier,
#: continuation.
PARK, PLACEHOLDER, EOU, FILE_CARRIER, CONTINUATION = 41, 42, 43, 44, 45

_ABSENT = object()


# ---------------------------------------------------------------------------
# Engine configurations


def hf_config(
    populations: Any = _ABSENT,
    *,
    lookahead: Any = 0,
    encoder_arm: Any = "eager-graphed",
    decoder_arm: Any = "dense-graphed",
    native_burst: Any = _ABSENT,
    chunk_populations: Any = _ABSENT,
    **extra: Any,
) -> SimpleNamespace:
    """A served configuration. ``_ABSENT`` leaves the attribute unset; the
    default is a valid option-on configuration once ``populations`` is given.
    ``lookahead=None`` sets no ``num_lookahead_tokens``."""
    fields: dict[str, Any] = {
        "card_default_num_lookahead_tokens": 3,
    }
    if lookahead is not None:
        fields["num_lookahead_tokens"] = lookahead
    if populations is not _ABSENT:
        fields["transaction_graph_populations"] = populations
    if encoder_arm is not _ABSENT:
        fields["encoder_execution_arm"] = encoder_arm
    if decoder_arm is not _ABSENT:
        fields["decode_dispatch_arm"] = decoder_arm
    if native_burst is not _ABSENT:
        fields["experimental_native_burst"] = native_burst
    if chunk_populations is not _ABSENT:
        fields["chunk_bucket_graph_populations"] = chunk_populations
    fields.update(extra)
    return SimpleNamespace(**fields)


class GraphMode(enum.IntEnum):
    """Core's ``CUDAGraphMode`` members as a configuration double carries
    them (name and value; this module imports no vLLM)."""

    NONE = 0
    PIECEWISE = 1
    FULL = 2


class CompilationMode(enum.IntEnum):
    """Core's ``CompilationMode`` members (name and value)."""

    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3


_DEFAULT = object()


def admission_limit(max_num_seqs: Any, max_num_active_seqs: Any) -> Any:
    """The number of requests core's scheduler admits into RUNNING:
    ``max_num_active_seqs`` when set, otherwise ``max_num_seqs``."""
    return max_num_seqs if max_num_active_seqs is None else max_num_active_seqs


def engine_config(
    hf: Any = None,
    *,
    max_num_seqs: int | None = 64,
    max_num_batched_tokens: int | None = 8192,
    max_num_active_seqs: int | None = None,
    num_gpu_blocks_override: Any = _DEFAULT,
    kv_cache_memory_bytes: int | None = None,
    async_scheduling: Any = False,
    enforce_eager: bool = False,
    cudagraph_mode: Any = GraphMode.NONE,
    compilation_mode: Any = CompilationMode.NONE,
    device_type: str = "cuda",
) -> SimpleNamespace:
    """An engine configuration double in core's attribute shape.

    The defaults meet the restricted startup (TXG-CFG-017 to TXG-CFG-019):
    synchronous scheduling, ``cudagraph_mode`` and compilation mode NONE, and
    ``num_gpu_blocks_override`` the admission limit plus one (65 while the
    limit is unresolved). Pass ``num_gpu_blocks_override=None`` for an unset
    override."""
    hf = hf if hf is not None else hf_config([64])
    if num_gpu_blocks_override is _DEFAULT:
        limit = admission_limit(max_num_seqs, max_num_active_seqs)
        num_gpu_blocks_override = (64 if limit is None else limit) + 1
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf,
            enforce_eager=enforce_eager,
            dtype="float32",
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_active_seqs=max_num_active_seqs,
            async_scheduling=async_scheduling,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_gpu_blocks_override,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            enable_prefix_caching=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode, mode=compilation_mode
        ),
        device_config=SimpleNamespace(device_type=device_type),
        parallel_config=SimpleNamespace(),
        speculative_config=None,
        kv_transfer_config=None,
    )


# ---------------------------------------------------------------------------
# Row plans


@dataclasses.dataclass(frozen=True)
class Row:
    """One row of a CPU row plan.

    ``kind`` is ``"chunk"`` or ``"other"``. ``failed`` marks a planning-failed
    row (bound non-CHUNK with its planned-rejection bit; TXG-COV-010).
    """

    kind: str = "chunk"
    fresh: bool = False
    file: bool = False
    geometry: int = 0
    failed: bool = False
    prompt: int = 0


def rows(spec: str, **common: Any) -> list[Row]:
    """Rows from a compact spec: ``c`` CHUNK, ``o`` other, ``x`` planning
    failed; e.g. ``"ccoxc"``."""
    out = []
    for ch in spec:
        if ch == "c":
            out.append(Row("chunk", **common))
        elif ch == "o":
            out.append(Row("other", **common))
        elif ch == "x":
            out.append(Row("other", failed=True, **common))
        else:
            raise ValueError(f"unknown row code {ch!r}")
    return out


def row_plan(
    plan_rows: Sequence[Row],
    *,
    first_block: int = 1,
    num_pool_blocks: int | None = None,
    planned_rejection: bool | None = None,
) -> Any:
    """A CPU ``RowPlan`` with one row per ``plan_rows`` entry, each on its own
    block, all rows prefill rows (as the model builds them).

    ``planned_rejection``: ``True`` sets the ``planned_rejection`` field from
    the rows' ``failed`` bits (TXG-COV-010); ``None`` sets it only when a row
    failed; ``False`` never sets it.
    """
    import torch

    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        PreparedRowBinding,
        RowPlan,
    )

    n = len(plan_rows)
    blocks = list(range(first_block, first_block + n))
    is_chunk = [r.kind == "chunk" and not r.failed for r in plan_rows]
    bindings = tuple(
        PreparedRowBinding(
            request_id=f"r{i}",
            block_id=blocks[i],
            admission_generation=1,
            geometry_id=r.geometry,
            prompt_index=r.prompt,
            prior_prompt_index=r.prompt,
            allow_prompt_transition=False,
            is_chunk=is_chunk[i],
            ready_deadline_ns=1000 + i if is_chunk[i] else 0,
        )
        for i, r in enumerate(plan_rows)
    )
    plan = RowPlan(
        state_indices_d=torch.zeros((0, 1), dtype=torch.int64),
        num_decodes=0,
        state_indices_p=torch.tensor(blocks, dtype=torch.int64),
        num_prefills=n,
        has_initial_states_p=torch.tensor(
            [not r.fresh for r in plan_rows], dtype=torch.bool
        ),
        null_block_id=0,
        num_pool_blocks=num_pool_blocks or first_block + n,
        live_block_ids=torch.tensor(blocks, dtype=torch.int64),
        geometry_id=torch.tensor([r.geometry for r in plan_rows], dtype=torch.int64),
        prompt_index=torch.tensor([r.prompt for r in plan_rows], dtype=torch.int64),
        is_chunk=torch.tensor(is_chunk, dtype=torch.bool),
        admission_generation=torch.ones(n, dtype=torch.int64),
        ready_deadline_ns=torch.tensor(
            [b.ready_deadline_ns for b in bindings], dtype=torch.int64
        ),
        request_ids=tuple(b.request_id for b in bindings),
        execution_tier=0,
        bindings=bindings,
        endpoint_mode=torch.zeros(n, dtype=torch.int64),
        endpoint_threshold_frames=torch.zeros(n, dtype=torch.int64),
        endpoint_residue_frames=torch.zeros(n, dtype=torch.int64),
        is_file=torch.tensor([r.file for r in plan_rows], dtype=torch.bool),
    )
    wants = (
        any(r.failed for r in plan_rows)
        if planned_rejection is None
        else planned_rejection
    )
    if wants:
        plan = with_planned_rejection(plan, [r.failed for r in plan_rows])
    return plan


def with_planned_rejection(plan: Any, bits: Iterable[bool]) -> Any:
    """``plan`` with its planned-rejection vector set (TXG-COV-010); a plan
    type without the field fails the calling test."""
    import torch

    names = {f.name for f in dataclasses.fields(plan)}
    assert "planned_rejection" in names, (
        "RowPlan has no planned_rejection field (TXG-COV-010, TXG-MASK-006)"
    )
    return dataclasses.replace(
        plan, planned_rejection=torch.tensor(list(bits), dtype=torch.bool)
    )


# ---------------------------------------------------------------------------
# Cells without CUDA


class FakeGraph:
    """Stands for a captured ``torch.cuda.CUDAGraph``: ``replay()`` appends
    ``("replay", key)`` to the shared event log."""

    def __init__(self, key: Any, events: list | None = None) -> None:
        self.key = key
        self.events = events if events is not None else []
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.events.append(("replay", self.key))


class AnySignature:
    """A captured signature that equals every step signature."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    __hash__ = None  # type: ignore[assignment]


@dataclasses.dataclass
class FakeCell:
    key: Any
    graph: Any
    signature: Any = dataclasses.field(default_factory=AnySignature)


def fake_inventory(
    n: int,
    geometry_index: int = 0,
    *,
    missing: Iterable[tuple[int, str, int]] = (),
    events: list | None = None,
) -> dict[tuple[int, str, int], FakeCell]:
    """``2N`` fake cells; the keys in ``missing`` hold no graph."""
    missing = set(missing)
    events = events if events is not None else []
    cells = {}
    for kind in ("chunk", "echo"):
        for count in range(1, n + 1):
            key = (geometry_index, kind, count)
            graph = None if key in missing else FakeGraph(key, events)
            cells[key] = FakeCell(key, graph)
    return cells


def has_graph_of(cells: dict) -> Any:
    """The ``has_graph`` callable ``classify`` takes, over a fake inventory."""
    return lambda key: key in cells and cells[key].graph is not None


# ---------------------------------------------------------------------------
# Commit sinks


class RecordingReservation:
    def __init__(self, sink: RecordingSink, plan: Any) -> None:
        self.sink = sink
        self.plan = plan
        self.staged: Any = None
        self.cancelled = 0

    def stage(self, row_status: Any) -> None:
        self.sink.events.append(("stage", self))
        self.staged = row_status.detach().clone()
        progress = getattr(self.plan, "progress", None)
        self.progress = None if progress is None else progress.detach().clone()
        counts = getattr(self.plan, "queue_counts", None)
        self.queue_counts = None if counts is None else counts.detach().clone()

    def cancel(self) -> None:
        self.sink.events.append(("cancel", self))
        self.cancelled += 1


class RecordingSink:
    """A ``CommitSink`` that records reservations in order."""

    def __init__(self, events: list | None = None) -> None:
        self.events = events if events is not None else []
        self.reservations: list[RecordingReservation] = []

    def reserve(self, plan: Any) -> RecordingReservation:
        reservation = RecordingReservation(self, plan)
        self.reservations.append(reservation)
        self.events.append(("reserve", reservation))
        return reservation

    @property
    def last(self) -> RecordingReservation:
        assert self.reservations, "no reservation was made"
        return self.reservations[-1]
