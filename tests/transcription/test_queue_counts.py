# SPDX-License-Identifier: Apache-2.0
"""The queue-count transport (FILE-QC-001 to FILE-QC-005), seam by seam.

``CommitPlan.queue_counts`` → ``BoundedCommitSink.reserve`` (validation) →
the reservation's ``stage`` (one non-blocking copy into the sink's host
buffer) → ``collect`` (``SessionStatusReport.queue_count``) →
``collect_commit_status`` (``CommitCollection`` with ``QueueCount`` values).
The end-to-end path through the model state is in ``test_file_steps.py``.

The sink and its reports import no vLLM (unmarked CPU tests); the registry is a
double with the lease and resolution answers the test sets. The model's
``collect_commit_status`` needs the vLLM tree (``engine``).

Layout additions: ``CommitPlan.queue_counts``; in
``nemotron_asr.commit_sink``, ``SessionStatusReport.queue_count``,
``CommitCollection(statuses, failed, queue_counts)`` and
``QueueCount(request_id, admission_generation, count)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch


class Registry:
    """The registry surface the sink and the status resolution use.

    ``current`` maps each request ID to its current admission generation; a
    binding or report is current when its generation matches."""

    def __init__(self, current: dict[str, int]) -> None:
        self.current = dict(current)
        self.lease_checks = 0

    def validate_lease(self, bindings) -> None:
        self.lease_checks += 1
        for binding in bindings:
            if self.current.get(binding.request_id) != binding.admission_generation:
                raise ValueError(f"lease check: {binding.request_id!r} is not current")

    def lease_is_current(self, bindings) -> bool:
        return all(
            self.current.get(b.request_id) == b.admission_generation for b in bindings
        )

    def resolve_status(self, *, request_id, admission_generation, clean) -> bool:
        return self.current.get(request_id) == admission_generation

    def publish_progress(self, *args, **kwargs):
        return None


def binding(request_id: str, generation: int, block: int, *, chunk: bool = False):
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import PreparedRowBinding

    return PreparedRowBinding(
        request_id=request_id,
        block_id=block,
        admission_generation=generation,
        geometry_id=3,
        prompt_index=0,
        prior_prompt_index=0,
        allow_prompt_transition=False,
        is_chunk=chunk,
        ready_deadline_ns=0,
    )


ROWS = (("a", 7), ("b", 8), ("c", 9))


def plan(queue_counts: Any = None, *, progress: Any = None, rows=ROWS):
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import CommitPlan

    return CommitPlan(
        bindings=tuple(binding(r, g, n + 1) for n, (r, g) in enumerate(rows)),
        capture=None,
        progress=progress,
        queue_counts=queue_counts,
    )


def sink(registry: Registry, *, max_rows: int = 4, device: str = "cpu"):
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        BoundedCommitSink,
    )

    return BoundedCommitSink(registry, max_rows=max_rows, device=torch.device(device))


def staged(registry: Registry, counts, status, *, progress=None, rows=ROWS):
    """Reserve, stage and collect one step; returns the reports by request."""
    s = sink(registry)
    ticket = s.reserve(plan(counts, progress=progress, rows=rows))
    ticket.stage(torch.tensor(status, dtype=torch.int32))
    reports, _records, _lease_ok = s.collect()
    return {r.request_id: r for r in reports}


# ---------------------------------------------------------------------------
# FILE-QC-001: the field


# @spec FILE-QC-001
def test_commit_plan_has_an_optional_queue_counts_field():
    import dataclasses

    from vllm_nemotron_asr_streaming.nemotron_asr.advance import CommitPlan

    fields = {f.name: f for f in dataclasses.fields(CommitPlan)}
    assert "queue_counts" in fields
    assert fields["queue_counts"].default is None


# ---------------------------------------------------------------------------
# FILE-QC-002: reserve validation


@pytest.mark.parametrize(
    "counts",
    [
        torch.zeros(2, dtype=torch.int32),  # too few rows
        torch.zeros(4, dtype=torch.int32),  # too many rows
        torch.zeros(3, 1, dtype=torch.int32),  # not one column
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.bool),
    ],
    ids=["short", "long", "2d", "int64", "float32", "bool"],
)
# @spec FILE-QC-002
def test_reserve_refuses_malformed_counts_before_the_lease_and_holds_nothing(counts):
    registry = Registry(dict(ROWS))
    s = sink(registry)
    with pytest.raises(ValueError):
        s.reserve(plan(counts))
    assert registry.lease_checks == 0  # refused before the lease is validated
    # nothing is held: a valid plan reserves next
    s.reserve(plan(torch.zeros(3, dtype=torch.int32)))
    assert registry.lease_checks == 1


# @spec FILE-QC-002
def test_reserve_refuses_malformed_counts_even_when_the_lease_is_stale():
    registry = Registry({"a": 7, "b": 8})  # "c" is not current
    s = sink(registry)
    with pytest.raises(ValueError, match="queue"):
        s.reserve(plan(torch.zeros(3, dtype=torch.int64)))
    assert registry.lease_checks == 0


# @spec FILE-QC-002
def test_reserve_accepts_a_plan_without_counts():
    registry = Registry(dict(ROWS))
    sink(registry).reserve(plan(None))
    assert registry.lease_checks == 1


# ---------------------------------------------------------------------------
# FILE-QC-003: the host buffer and the stage copy


def _tensors(obj: Any) -> list[torch.Tensor]:
    return [v for v in vars(obj).values() if isinstance(v, torch.Tensor)]


class _OpLog:
    """Every aten operation a block runs, with its arguments."""

    def __init__(self) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        log = self.ops = []

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                log.append((func, args, kwargs or {}))
                return func(*args, **(kwargs or {}))

        self.mode = Mode()


#: Operations that allocate tensor storage.
FACTORIES = (
    "empty",
    "zeros",
    "ones",
    "full",
    "empty_strided",
    "new_",
    "clone",
    "_to_copy",
)


def _non_blocking(args, kwargs) -> bool:
    return bool(kwargs.get("non_blocking", args[2] if len(args) > 2 else False))


def _stage_ops(device: str, counts: torch.Tensor, registry: Registry):
    s = sink(registry, device=device)
    construction = {t.data_ptr(): t for t in _tensors(s)}
    ticket = s.reserve(plan(counts))
    status = torch.zeros(3, dtype=torch.int32, device=device)
    log = _OpLog()
    with log.mode:
        ticket.stage(status)
    return s, construction, log.ops


def _copies_of(ops, source: torch.Tensor):
    return [
        (args, kwargs)
        for func, args, kwargs in ops
        if func.__name__.startswith("copy_")
        and len(args) > 1
        and isinstance(args[1], torch.Tensor)
        and args[1].data_ptr() == source.data_ptr()
    ]


# @spec FILE-QC-003
def test_sink_allocates_a_host_count_buffer_of_max_rows_at_construction():
    s = sink(Registry({}), max_rows=6)
    buffers = [
        t
        for t in _tensors(s)
        if t.dtype == torch.int32 and tuple(t.shape) == (6,) and t.device.type == "cpu"
    ]
    # the status buffer and the queue-count buffer
    assert len(buffers) >= 2
    assert len({t.data_ptr() for t in buffers}) == len(buffers)


# @spec FILE-QC-003
def test_stage_copies_the_counts_once_non_blocking_into_the_construction_buffer():
    counts = torch.tensor([3, 0, 5], dtype=torch.int32)
    s, construction, ops = _stage_ops("cpu", counts, Registry(dict(ROWS)))
    copies = _copies_of(ops, counts)
    assert len(copies) == 1, "the counts are copied exactly once"
    (args, kwargs) = copies[0]
    assert _non_blocking(args, kwargs) is True
    destination = args[0]
    assert destination.dtype == torch.int32 and destination.device.type == "cpu"
    assert destination.untyped_storage().data_ptr() in {
        t.untyped_storage().data_ptr() for t in construction.values()
    }
    # no tensor storage is allocated while staging
    names = [func.__name__ for func, _, _ in ops]
    assert not [n for n in names if n.split(".")[0].startswith(FACTORIES)], names


# @spec FILE-QC-003
def test_stage_without_counts_copies_none():
    s, _, ops = _stage_ops("cpu", None, Registry(dict(ROWS)))  # type: ignore[arg-type]
    copies = [func for func, _, _ in ops if func.__name__.startswith("copy_")]
    # the status copy only (no progress, no counts)
    assert len(copies) == 1


# @spec FILE-QC-003
@pytest.mark.gpu
def test_stage_on_cuda_copies_into_pinned_memory_before_the_event_without_sync(
    monkeypatch,
):
    """The record and synchronization spies are installed for ``stage``
    alone; ``collect`` then waits on the real completion event before it
    reads the copied counts."""
    from torch.utils._python_dispatch import TorchDispatchMode

    order: list[Any] = []
    real_record = torch.cuda.Event.record
    real_sync = torch.cuda.synchronize
    real_event_sync = torch.cuda.Event.synchronize

    def record(self, *args, **kwargs):
        order.append("record")
        return real_record(self, *args, **kwargs)

    def synchronize(*args, **kwargs):
        order.append("synchronize")
        return real_sync(*args, **kwargs)

    def event_synchronize(self):
        order.append("synchronize")
        return real_event_sync(self)

    counts = torch.tensor([3, 0, 5], dtype=torch.int32, device="cuda")
    registry = Registry(dict(ROWS))
    s = sink(registry, device="cuda")
    construction = {t.untyped_storage().data_ptr() for t in _tensors(s)}
    ticket = s.reserve(plan(counts))
    status = torch.zeros(3, dtype=torch.int32, device="cuda")

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if (
                func.__name__.startswith("copy_")
                and len(args) > 1
                and isinstance(args[1], torch.Tensor)
                and args[1].data_ptr() == counts.data_ptr()
            ):
                order.append(("counts", args[0], _non_blocking(args, kwargs or {})))
            return func(*args, **(kwargs or {}))

    with monkeypatch.context() as spies:
        spies.setattr(torch.cuda.Event, "record", record)
        spies.setattr(torch.cuda, "synchronize", synchronize)
        spies.setattr(torch.cuda.Event, "synchronize", event_synchronize)
        with Mode():
            ticket.stage(status)
    # the real record and synchronize are back for collection
    assert torch.cuda.Event.synchronize is real_event_sync
    entries = [e for e in order if isinstance(e, tuple)]
    assert len(entries) == 1
    _, destination, non_blocking = entries[0]
    assert destination.is_pinned() and non_blocking is True
    assert destination.untyped_storage().data_ptr() in construction
    assert order.index(entries[0]) < order.index("record")
    assert "synchronize" not in order
    reports, _, _ = s.collect()
    assert [r.queue_count for r in reports] == [3, 0, 5]


# ---------------------------------------------------------------------------
# FILE-QC-004: the report field


# @spec FILE-QC-004
def test_clean_rows_report_their_count_and_rejected_rows_none():
    reports = staged(
        Registry(dict(ROWS)), torch.tensor([3, 5, 0], dtype=torch.int32), [0, 512, 0]
    )
    assert reports["a"].queue_count == 3
    assert reports["b"].queue_count is None  # nonzero status
    assert reports["c"].queue_count == 0  # zero remaining is a count


# @spec FILE-QC-004
def test_the_count_does_not_depend_on_the_progress_key():
    # a row whose progress publishes nothing (progress_key None) still has
    # its count: a park echo, or a plan without progress at all
    progress = torch.tensor([[1, 1, 2], [1, 0, 0], [2, 2, 1]], dtype=torch.int64)
    with_progress = staged(
        Registry(dict(ROWS)),
        torch.tensor([4, 0, 2], dtype=torch.int32),
        [0, 0, 0],
        progress=progress,
    )
    assert with_progress["b"].progress_key is None
    assert [with_progress[r].queue_count for r in "abc"] == [4, 0, 2]
    without = staged(
        Registry(dict(ROWS)), torch.tensor([4, 0, 2], dtype=torch.int32), [0, 0, 0]
    )
    assert all(without[r].progress_key is None for r in "abc")
    assert [without[r].queue_count for r in "abc"] == [4, 0, 2]


# @spec FILE-QC-004
def test_a_failed_stage_time_lease_revalidation_reports_no_count():
    registry = Registry(dict(ROWS))
    s = sink(registry)
    ticket = s.reserve(plan(torch.tensor([1, 2, 3], dtype=torch.int32)))
    registry.current["b"] = 99  # the lease drifted before staging
    ticket.stage(torch.zeros(3, dtype=torch.int32))
    reports, _, lease_ok = s.collect()
    assert lease_ok is False
    assert [r.queue_count for r in reports] == [None, None, None]


# @spec FILE-QC-004
def test_a_plan_without_counts_reports_none():
    reports = staged(Registry(dict(ROWS)), None, [0, 0, 0])
    assert [reports[r].queue_count for r in "abc"] == [None, None, None]


# @spec FILE-QC-004
def test_the_report_field_defaults_to_none():
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        SessionStatusReport,
    )

    report = SessionStatusReport(request_id="a", admission_generation=1, row_status=0)
    assert report.queue_count is None


# ---------------------------------------------------------------------------
# FILE-QC-005: the collection


def _model_with(s, registry):
    """The model's ``collect_commit_status`` bound to a stand-in holding only
    the sink and registry it reads."""
    from vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    holder = SimpleNamespace(_commit_sink=s, _registry=registry)
    return lambda: NemotronASRForRNNT.collect_commit_status(holder)


# @spec FILE-QC-005
@pytest.mark.engine
def test_collection_holds_statuses_failures_and_counts_of_clean_current_reports():
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        CommitCollection,
        QueueCount,
    )

    rows = (("a", 7), ("b", 8), ("c", 9), ("d", 10))
    registry = Registry(dict(rows))
    s = sink(registry, max_rows=4)
    ticket = s.reserve(plan(torch.tensor([3, 5, 0, 2], dtype=torch.int32), rows=rows))
    # "c" is replaced by a later incarnation after staging: its report is stale
    ticket.stage(torch.tensor([0, 1024, 0, 0], dtype=torch.int32))
    registry.current["c"] = 11
    collect = _model_with(s, registry)
    result = collect()
    assert isinstance(result, CommitCollection)
    assert result.statuses == {"a": 0, "b": 1024, "c": -1, "d": 0}
    assert result.failed == {"b", "c"}
    assert tuple(result.queue_counts) == (
        QueueCount(request_id="a", admission_generation=7, count=3),
        QueueCount(request_id="d", admission_generation=10, count=2),
    )
    # one consumption: the next collection holds nothing
    again = collect()
    assert isinstance(again, CommitCollection)
    assert (again.statuses, again.failed, tuple(again.queue_counts)) == ({}, set(), ())


# @spec FILE-QC-005
@pytest.mark.engine
@pytest.mark.parametrize("has_sink", [False, True])
def test_collection_without_a_staged_commit_is_empty(has_sink):
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        CommitCollection,
    )

    registry = Registry({})
    result = _model_with(sink(registry) if has_sink else None, registry)()
    assert isinstance(result, CommitCollection)
    assert result.statuses == {} and result.failed == set()
    assert tuple(result.queue_counts) == ()


# @spec FILE-QC-005
@pytest.mark.engine
def test_a_clean_report_without_a_count_yields_no_queue_count():
    registry = Registry(dict(ROWS))
    s = sink(registry)
    s.reserve(plan(None)).stage(torch.zeros(3, dtype=torch.int32))
    result = _model_with(s, registry)()
    assert result.statuses == {"a": 0, "b": 0, "c": 0}
    assert tuple(result.queue_counts) == ()
