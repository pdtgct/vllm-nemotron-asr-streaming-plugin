# SPDX-License-Identifier: Apache-2.0
"""Whole-file rows: the FILE row role, its step rule and its host buffer.

A file request is one ordinary request whose audio is already complete. Its
samples live in a host buffer owned by the request; each engine step either
drains one queued label, advances one chunk (or the final tail) through the
streaming transition, or ends the request. This module is pure host logic
(numpy plus one torch copy helper); the transaction itself lives in
``advance`` and the per-step planning in the model state.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


class FileAction(enum.Enum):
    """What a FILE row does in one step."""

    DRAIN = "drain"
    CHUNK = "chunk"
    TAIL = "tail"
    END = "end"


# @spec FILE-ROW-002, FILE-ROW-005, FILE-ROW-008
def next_file_action(
    *, queue_len: int, chunks_remaining: int, tail_done: bool
) -> FileAction:
    """The step rule, checked in order: drain, regular chunk, final tail, end.

    Audio advances only while the label queue is empty, and the end step is
    reached only after the final tail with an empty queue.
    """
    if queue_len > 0:
        return FileAction.DRAIN
    if chunks_remaining > 0:
        return FileAction.CHUNK
    if not tail_done:
        return FileAction.TAIL
    return FileAction.END


@dataclass(frozen=True)
class FileStepPlan:
    """One FILE row's step: the action and the sample range it advances.

    ``start`` and ``length`` are sample offsets into the file; ``length`` is
    zero for DRAIN and END, and may be zero for the final tail.
    """

    action: FileAction
    start: int
    length: int

    @property
    def advances_audio(self) -> bool:
        return self.action in (FileAction.CHUNK, FileAction.TAIL)


# @spec FILE-ROW-002, FILE-ROW-003, FILE-ROW-004, FILE-ROW-006
def plan_file_step(
    *,
    queue_len: int,
    position: int,
    total_samples: int,
    chunk_samples: int,
    tail_done: bool,
) -> FileStepPlan:
    """Plan one step of a FILE row.

    ``position`` is the number of samples already advanced; a full regular
    chunk remains while at least ``chunk_samples`` samples are left. The final
    tail takes whatever is left after the last regular chunk (possibly
    nothing).
    """
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    left = max(int(total_samples) - int(position), 0)
    chunks_remaining = 0 if tail_done else left // chunk_samples
    action = next_file_action(
        queue_len=queue_len, chunks_remaining=chunks_remaining, tail_done=tail_done
    )
    if action is FileAction.CHUNK:
        return FileStepPlan(action, int(position), int(chunk_samples))
    if action is FileAction.TAIL:
        return FileStepPlan(action, int(position), left)
    return FileStepPlan(action, int(position), 0)


@dataclass(frozen=True)
class StepRow:
    """One scheduled row as the step planner sees it.

    A stream CHUNK row carries its ``deadline``; a FILE row carries a
    ``file_plan`` and no deadline. ``samples`` is the device copy of the one
    chunk a FILE row advances this step and ``header`` the file's
    ``(geometry, locale row, sample count)``; both are ``None`` for rows that
    do not advance a file.
    """

    request_id: str
    geometry_index: int
    deadline: int | None
    file_plan: FileStepPlan | None
    samples: Any = None
    header: tuple[int, ...] | None = None
    chunk_sequence: int = 0
    #: A file request whose item was malformed: the row is failed alone.
    failed: bool = False

    @property
    def is_file(self) -> bool:
        return self.file_plan is not None

    @property
    def runs_chunk(self) -> bool:
        """Whether the row contributes an encoder chunk to this step."""
        if self.file_plan is not None:
            return self.file_plan.advances_audio
        return self.deadline is not None


@dataclass(frozen=True)
class ChunkGroup:
    """The rows of one geometry that run an encoder chunk this step."""

    geometry_index: int
    deadlines: tuple[int, ...] = ()
    request_ids: tuple[str, ...] = ()

    @property
    def unit_deadlines(self) -> tuple[int, ...]:
        return self.deadlines


# @spec FILE-ENG-004
def order_chunk_groups(groups: Iterable[Any]) -> list[Any]:
    """Order chunk groups by urgency.

    A group's urgency is the earliest unit deadline among its stream rows;
    groups holding only FILE rows (no deadlines) run after every group with a
    stream row, and ties break by geometry index.
    """

    def key(group: Any) -> tuple[int, int, int]:
        deadlines = tuple(group.deadlines)
        if deadlines:
            return (0, min(deadlines), group.geometry_index)
        return (1, 0, group.geometry_index)

    return sorted(groups, key=key)


# @spec FILE-ENG-001, FILE-ENG-003, FILE-ENG-004
def plan_chunk_groups(rows: Sequence[StepRow]) -> list[ChunkGroup]:
    """Group the step's chunk rows by geometry, in running order.

    A FILE row advancing a chunk or its tail joins the group of its geometry
    beside stream rows of the same geometry and adds no deadline; a draining
    or ending FILE row, and stream rows without a chunk, join no group.
    """
    ids: dict[int, list[str]] = {}
    deadlines: dict[int, list[int]] = {}
    for row in rows:
        if not row.runs_chunk:
            continue
        ids.setdefault(row.geometry_index, []).append(row.request_id)
        if row.file_plan is None:
            assert row.deadline is not None
            deadlines.setdefault(row.geometry_index, []).append(row.deadline)
    groups = [
        ChunkGroup(
            geometry_index=geometry,
            deadlines=tuple(deadlines.get(geometry, ())),
            request_ids=tuple(request_ids),
        )
        for geometry, request_ids in ids.items()
    ]
    return order_chunk_groups(groups)


@dataclass
class StepPlan:
    """The step planner's result: the step's rows and its chunk groups in
    running order."""

    rows: list[StepRow]
    groups: list[ChunkGroup]

    @property
    def geometry_order(self) -> tuple[int, ...]:
        return tuple(group.geometry_index for group in self.groups)


@dataclass
class _Buffer:
    samples: np.ndarray
    header: np.ndarray
    slot: int
    generation: int


@dataclass
class FileHostBuffers:
    """Request-owned host buffers of the files in flight.

    A buffer is created at a file request's first schedule (``take``), is
    independent of core's encoder cache, and is freed when the request is
    reported finished for any reason (``free_finished``) or when its state
    slot comes back with a new generation. ``stage`` is the only way samples
    leave a buffer.
    """

    _entries: dict[str, _Buffer] = field(default_factory=dict)
    _by_slot: dict[int, str] = field(default_factory=dict)

    # @spec FILE-BUF-001, FILE-BUF-004
    def take(
        self,
        request_id: str,
        slot: int,
        generation: int,
        samples: Any,
        header: Any,
    ) -> tuple[str, ...]:
        """Copy a file's samples and header into a buffer owned by the request.

        Returns the IDs of buffers dropped because they belonged to an older
        generation of the same slot.
        """
        dropped: list[str] = []
        owner = self._by_slot.get(slot)
        if owner is not None and owner != request_id:
            old = self._entries.get(owner)
            if old is not None and old.generation != generation:
                self._entries.pop(owner)
                self._by_slot.pop(slot)
                dropped.append(owner)
        previous = self._entries.pop(request_id, None)
        if previous is not None and self._by_slot.get(previous.slot) == request_id:
            self._by_slot.pop(previous.slot)
        buffer = _Buffer(
            samples=np.array(_as_numpy(samples), dtype=np.float32, copy=True).reshape(
                -1
            ),
            header=np.array(_as_numpy(header), dtype=np.int64, copy=True).reshape(-1),
            slot=int(slot),
            generation=int(generation),
        )
        buffer.samples.setflags(write=False)
        buffer.header.setflags(write=False)
        self._entries[request_id] = buffer
        self._by_slot[buffer.slot] = request_id
        return tuple(dropped)

    # @spec FILE-BUF-002
    def stage(self, request_id: str, start: int, length: int) -> np.ndarray:
        """A view of ``length`` samples of the request's buffer from ``start``."""
        buffer = self._entries.get(request_id)
        if buffer is None:
            raise ValueError(f"no file buffer for request {request_id!r}")
        if start < 0 or length < 0 or start + length > buffer.samples.shape[0]:
            raise ValueError(
                f"range [{start}, {start + length}) is outside the "
                f"{buffer.samples.shape[0]}-sample file buffer of {request_id!r}"
            )
        return buffer.samples[start : start + length]

    def header(self, request_id: str) -> np.ndarray:
        """The buffer's own copy of the file header (int64)."""
        buffer = self._entries.get(request_id)
        if buffer is None:
            raise ValueError(f"no file buffer for request {request_id!r}")
        return buffer.header

    def num_samples(self, request_id: str) -> int:
        """The number of samples in the request's buffer."""
        buffer = self._entries.get(request_id)
        if buffer is None:
            raise ValueError(f"no file buffer for request {request_id!r}")
        return int(buffer.samples.shape[0])

    # @spec FILE-BUF-003, FILE-BUF-005
    def free_finished(self, finished_request_ids: Iterable[str]) -> tuple[str, ...]:
        """Free the buffers of finished requests; unknown IDs are ignored."""
        freed: list[str] = []
        for request_id in finished_request_ids:
            buffer = self._entries.pop(request_id, None)
            if buffer is None:
                continue
            if self._by_slot.get(buffer.slot) == request_id:
                self._by_slot.pop(buffer.slot)
            freed.append(request_id)
        return tuple(freed)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# @spec FILE-BUF-002
def copy_chunk_to_device(view: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """The only host-to-device copy of FILE samples: one staged chunk."""
    # the buffer is read-only; one chunk is copied out of it
    chunk = torch.from_numpy(np.array(view, dtype=np.float32, copy=True))
    return chunk.to(device)
