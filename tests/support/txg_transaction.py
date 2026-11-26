# SPDX-License-Identifier: Apache-2.0
"""The streaming transaction at the transaction level, on CPU or CUDA.

``build_env`` builds the real model and model state on the runnable synthetic
card (``support.model_harness``), drives a stream and a whole-file request
through real steps, and records every call of
``advance.advance_model_rows`` with a snapshot of the resident pools taken
before the call. From those calls it extracts **row fixtures**: one row's
resident pages (every pool at its block), its scheduled token, its carrier
row of ``inputs_embeds`` and its binding fields, together with the status the
row's real step reported.

A test composes a **step** from fixtures (each row on its own block of a
fresh pool set), and runs the option-off transaction over the whole row plan
or over a sub-plan (:meth:`TransactionEnv.run`), on cloned pools, with a
recording commit sink (``support.txg_fakes.RecordingSink``). The
transaction's own arguments (core, adapter, decode resolver, encoder
transition, token IDs, geometry order) are the model's, as recorded.

Row configurations are ``(kind, fresh, file)`` with ``kind`` ``"chunk"`` or
``"other"``. The production planner produces six of the eight; the two fresh
non-CHUNK configurations are derived from the fresh CHUNK fixtures (the park
token in a non-CHUNK binding), as TXG-EQ-002 enumerates every assignment.

Rejection fixtures (:meth:`TransactionEnv.rejection_fixtures`) mutate an
accepted fixture to produce one status class and are kept only when a
one-row step of the fixture reports exactly that class; every combination
not kept is recorded in :attr:`TransactionEnv.excluded` with its reason.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from support import model_harness as h
from support.txg_fakes import RecordingSink
from support.txgcap_harness import TxgCoreOrder

POOL_TENSORS = (
    "h_pool",
    "c_pool",
    "queue_pool",
    "book_pool",
    "frontend_raw_pool",
    "frontend_mel_pool",
    "frontend_counter_pool",
    "endpoint_history_pool",
    "endpoint_book_pool",
)
POOL_LISTS = ("channel_pools", "time_pools", "len_pools")
#: Transaction arguments a composed step supplies itself.
STEP_OWNED = (
    set(POOL_TENSORS)
    | set(POOL_LISTS)
    | {
        "commit_sink",
        "staging",
        "capture",
        "native_burst_handoff",
        "memory_profile",
    }
)

CONFIGS: tuple[tuple[str, bool, bool], ...] = tuple(
    itertools.product(("chunk", "other"), (False, True), (False, True))
)

#: The seed of the fixture model's weights (``build_env``).
FIXTURE_SEED = 0

HEADER_SLOTS = 7
#: Envelope header slots (``advance.ENV_*``).
H_VERSION, H_SAMPLES, H_GEOMETRY, H_FINAL, H_PROMPT, H_SEQUENCE = 0, 1, 2, 3, 4, 5


def config_name(config: tuple[str, bool, bool]) -> str:
    kind, fresh, file = config
    return f"{kind}-{'fresh' if fresh else 'cont'}-{'file' if file else 'stream'}"


# ---------------------------------------------------------------------------
# Pools


#: The allocator alignment a layout-preserving copy keeps for each storage's
#: base address, so every view keeps its address modulo this value.
_BASE_ALIGNMENT = 512


def _view_end(tensor: torch.Tensor) -> int:
    """The byte just past the last element ``tensor`` views, counted from
    the start of its storage."""
    size = tensor.element_size()
    if tensor.numel() == 0:
        return tensor.storage_offset() * size
    last = sum((n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride()))
    return (tensor.storage_offset() + last) * size + size


def layout_clone(tensors: Sequence[torch.Tensor], *, zero: bool = False) -> list:
    """Copies of ``tensors`` that keep the resident layout: each distinct
    storage is copied once, with its base address at the same offset modulo
    :data:`_BASE_ALIGNMENT`, and every tensor is re-viewed on its storage's
    copy at its own storage offset, shape and strides. Views that shared a
    storage share its copy. A copy therefore has the shape, strides, dtype
    and pointer alignment of the original (a new address), so the masked
    scatter receives a specialization startup warmed (TXG-CAP-013); a plain
    ``Tensor.clone`` of a strided resident view is contiguous and does not.
    ``zero`` zeroes the copies instead of copying the bytes.

    Each storage's copy holds only the bytes up to the end of the last view
    over it, not the whole storage: under the extensible KV cache the bound
    views hold the final committed block count while their storage may span
    the reservation, so a copy of the views is limited to the pages below
    the final committed count (TXG-QUAL-003, TXG-QUAL-029)."""
    copies: dict[tuple[Any, int], tuple[torch.Tensor, int]] = {}
    spans: dict[tuple[Any, int], int] = {}
    for tensor in tensors:
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        spans[key] = max(spans.get(key, 0), _view_end(tensor))
    out = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key not in copies:
            nbytes = spans[key]
            raw = torch.empty(
                nbytes + _BASE_ALIGNMENT, dtype=torch.uint8, device=tensor.device
            )
            shift = (storage.data_ptr() - raw.data_ptr()) % _BASE_ALIGNMENT
            body = raw[shift : shift + nbytes]
            if zero:
                body.zero_()
            else:
                source = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                source.set_(storage, 0, (nbytes,), (1,))
                body.copy_(source)
            copies[key] = (raw, shift)
        raw, shift = copies[key]
        size = tensor.element_size()
        assert shift % size == 0, "the copy cannot keep this view's alignment"
        view = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        view.set_(
            raw.untyped_storage(),
            shift // size + tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.stride(),
        )
        assert view.data_ptr() % _BASE_ALIGNMENT == (
            tensor.data_ptr() % _BASE_ALIGNMENT
        )
        out.append(view)
    return out


@dataclasses.dataclass
class Pools:
    tensors: dict[str, Any]

    def _rebuilt(self, *, zero: bool) -> Pools:
        """A copy over new storage that keeps every view's layout
        (:func:`layout_clone`)."""
        flat = [t for _, t in self.flat()]
        copies = iter(layout_clone(flat, zero=zero))
        return Pools(
            {
                k: [next(copies) for _ in v] if isinstance(v, list) else next(copies)
                for k, v in self.tensors.items()
            }
        )

    def clone(self) -> Pools:
        """A copy of the pages over new storage with the same resident
        layout (:func:`layout_clone`)."""
        return self._rebuilt(zero=False)

    def zeros(self) -> Pools:
        """Zeroed pools over new storage with the same resident layout."""
        return self._rebuilt(zero=True)

    def storage_bytes(self) -> int:
        """The bytes of the distinct storages the pools view (for a copy
        :meth:`clone` made, the copy's own bytes, alignment slack
        included)."""
        seen: dict[tuple[Any, int], int] = {}
        for _, t in self.flat():
            storage = t.untyped_storage()
            seen[(t.device, storage.data_ptr())] = storage.nbytes()
        return sum(seen.values())

    def narrowed(self, pages: int | None) -> Pools:
        """The same views limited to their first ``pages`` pages (every page
        when ``pages`` is ``None``): views over the same storage, at the same
        addresses and strides, that cover no page at or above ``pages``.
        Neither reads nor allocates."""
        if pages is None:
            return self
        if pages < 1:
            raise ValueError(f"a pool extent of {pages} pages holds no page")

        def narrow(t: torch.Tensor) -> torch.Tensor:
            return t[: min(pages, int(t.shape[0]))]

        return Pools(
            {
                k: [narrow(t) for t in v] if isinstance(v, list) else narrow(v)
                for k, v in self.tensors.items()
            }
        )

    def first_dimensions(self) -> set[int]:
        """The first dimension (the page count) of every pool view."""
        return {int(t.shape[0]) for _, t in self.flat()}

    def flat(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, value in self.tensors.items():
            if isinstance(value, list):
                for i, t in enumerate(value):
                    yield f"{name}[{i}]", t
            else:
                yield name, value

    def page(self, block: int) -> dict[str, torch.Tensor]:
        return {name: t[block].clone() for name, t in self.flat()}

    def write_page(self, block: int, page: dict[str, torch.Tensor]) -> None:
        for name, t in self.flat():
            t[block].copy_(page[name])

    def fill_page(self, block: int, pattern: str, seed: int = 0) -> None:
        """Fill ``block`` of every pool with a sentinel pattern: ``nan``,
        ``inf``, ``-inf``, ``random`` (random bits) or ``zero``."""
        gen = torch.Generator().manual_seed(seed)
        for _, t in self.flat():
            row = t[block]
            if pattern == "zero":
                row.zero_()
            elif pattern == "random" or not row.is_floating_point():
                bits = torch.randint(
                    0,
                    256,
                    (row.numel() * row.element_size(),),
                    generator=gen,
                    dtype=torch.uint8,
                )
                row.view(-1).view(torch.uint8).copy_(bits.to(row.device))
            else:
                row.fill_(float(pattern))

    def kwargs(self) -> dict[str, Any]:
        return dict(self.tensors)

    def differences(
        self, other: Pools, blocks: Sequence[int] | None = None
    ) -> list[str]:
        out = []
        theirs = dict(other.flat())
        for name, t in self.flat():
            a, b = t, theirs[name]
            if blocks is not None:
                index = torch.tensor(list(blocks), dtype=torch.long, device=a.device)
                a, b = a.index_select(0, index), b.index_select(0, index)
            if not same_bits(a, b):
                out.append(name)
        return out


def same_bits(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    """Bit-for-bit equality (NaN payloads included)."""
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.device != b.device:
        a, b = a.cpu(), b.cpu()
    # Compared where they live: one comparison and one synchronization per
    # tensor on CUDA, no device-to-host copy of every pool.
    a = a.detach().contiguous().reshape(-1).view(torch.uint8)
    b = b.detach().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(a, b))


def pools_of(kwargs: dict[str, Any]) -> Pools:
    tensors: dict[str, Any] = {}
    for name in POOL_LISTS:
        tensors[name] = list(kwargs[name])
    for name in POOL_TENSORS:
        if kwargs.get(name) is not None:
            tensors[name] = kwargs[name]
    return Pools(tensors)


# ---------------------------------------------------------------------------
# Plans


def restrict(plan: Any, positions: Sequence[int]) -> Any:
    """The sub-plan of ``positions`` in ascending order (the same blocks and
    liveness authority; TXG-EQ-002's ``R|S``). Every per-row field is
    restricted, including fields this module does not name."""
    positions = sorted(positions)
    assert plan.num_decodes == 0, "the model builds prefill rows only"
    n = plan.num_decodes + plan.num_prefills
    index = torch.tensor(positions, dtype=torch.long)
    changes: dict[str, Any] = {"num_prefills": len(positions)}
    for f in dataclasses.fields(plan):
        value = getattr(plan, f.name)
        if f.name in ("live_block_ids", "state_indices_d"):
            continue
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n:
            if value.numel():
                changes[f.name] = value.index_select(0, index)
        elif isinstance(value, tuple) and len(value) == n:
            changes[f.name] = tuple(value[i] for i in positions)
    return dataclasses.replace(plan, **changes)


def chunk_positions(plan: Any) -> list[int]:
    return [i for i, c in enumerate(plan.is_chunk.tolist()) if c]


def other_positions(plan: Any) -> list[int]:
    return [i for i, c in enumerate(plan.is_chunk.tolist()) if not c]


# ---------------------------------------------------------------------------
# Fixtures and steps


@dataclasses.dataclass(frozen=True)
class RowFixture:
    name: str
    kind: str
    fresh: bool
    file: bool
    token: int
    embed: torch.Tensor
    page: dict[str, torch.Tensor]
    geometry: int
    prompt: int
    prior_prompt: int
    generation: int
    deadline: int
    #: The status the row's own step reported (``None``: not yet observed).
    status: int | None = None
    #: The status class this fixture is meant to produce (0: accepted).
    intended: int = 0

    @property
    def config(self) -> tuple[str, bool, bool]:
        return (self.kind, self.fresh, self.file)

    def but(self, **changes: Any) -> RowFixture:
        return dataclasses.replace(self, **changes)

    def with_header(self, **slots: float) -> RowFixture:
        embed = self.embed.clone()
        for name, value in slots.items():
            embed[
                {
                    "version": H_VERSION,
                    "samples": H_SAMPLES,
                    "geometry": H_GEOMETRY,
                    "prompt": H_PROMPT,
                    "sequence": H_SEQUENCE,
                    "final": H_FINAL,
                }[name]
            ] = value
        return self.but(embed=embed)


def stream_envelope(
    *,
    geometry: int,
    lookahead: int,
    sequence: int,
    samples: int | None = None,
    final: bool = False,
    prompt: int = 0,
    silence: bool = False,
    seed: int = 0,
    oversize: bool = False,
) -> torch.Tensor:
    """One stream unit's carrier row, its header in ``advance.ENV_*`` slot
    order (``final`` in ``ENV_FINAL_TAIL``). A regular unit carries a full
    chunk; a valid final tail carries fewer samples than a full chunk
    (``oversize=True`` builds the refused oversized tail on purpose)."""
    full = chunk_samples(lookahead)
    count = full if samples is None else samples
    if final and not oversize:
        assert samples is not None and 0 <= samples < full, (
            f"a final tail carries fewer than {full} samples, not {samples}"
        )
    if not final:
        assert count == full, "a regular unit carries a full chunk"
    row = torch.zeros(HEADER_SLOTS + full, dtype=torch.float32)
    row[H_VERSION] = 2
    row[H_SAMPLES] = count
    row[H_GEOMETRY] = geometry
    row[H_FINAL] = int(final)
    row[H_PROMPT] = prompt
    row[H_SEQUENCE] = sequence
    if count and not silence:
        rng = np.random.default_rng(seed * 1000 + sequence)
        row[HEADER_SLOTS : HEADER_SLOTS + count] = torch.from_numpy(
            (rng.standard_normal(count) * 0.1).astype(np.float32)
        )
    return row


def final_tail(fixture: RowFixture, samples: int, lookahead: int) -> RowFixture:
    """``fixture`` (a CHUNK row) as a final tail of ``samples`` samples (zero:
    an empty final tail); a valid tail is shorter than a full chunk."""
    assert 0 <= samples < chunk_samples(lookahead), samples
    embed = fixture.embed.clone()
    embed[H_FINAL] = 1
    embed[H_SAMPLES] = samples
    embed[HEADER_SLOTS + samples :] = 0
    return fixture.but(embed=embed, name=f"{fixture.name}:final-{samples}")


def sequence_of(fixture: RowFixture) -> int:
    return int(fixture.embed[H_SEQUENCE])


@dataclasses.dataclass
class Step:
    plan: Any
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    pools: Pools
    fixtures: tuple[RowFixture, ...]

    @property
    def population(self) -> int:
        return len(self.fixtures)


@dataclasses.dataclass
class Outcome:
    rows: torch.Tensor | None
    status: torch.Tensor | None
    progress: torch.Tensor | None
    queue_counts: torch.Tensor | None
    pools: Pools
    sink: RecordingSink
    error: BaseException | None = None


class TransactionEnv:
    """See the module docstring."""

    def __init__(
        self,
        *,
        core,
        kwargs,
        template: Pools,
        geometry: int,
        lookahead: int,
        config,
        device,
        fixtures,
        model,
        state,
    ) -> None:
        self.core = core
        self.kwargs = kwargs
        self.template = template
        self.geometry = geometry
        self.lookahead = lookahead
        self.config = config
        self.device = device
        self.fixtures: dict[tuple[str, bool, bool], list[RowFixture]] = fixtures
        self.model = model
        self.state = state
        self.excluded: list[tuple[str, str, str]] = []
        self._rejections: dict | None = None
        #: The live serving pool views (``env_from_runner`` only).
        self.serving: Pools | None = None
        #: The null page of every pool (``Pools.page(0)``) before each driven
        #: step and after the drive's last step.
        self.drive_null_pages: list[dict[str, torch.Tensor]] = []

    # -- tokens -------------------------------------------------------------

    @property
    def placeholder(self) -> int:
        return int(self.config.audio_chunk_token_id)

    @property
    def park(self) -> int:
        return int(self.config.eos_token_id)

    @property
    def capacity(self) -> int:
        return int(self.template.tensors["queue_pool"].shape[0])

    def accepted(self, config: tuple[str, bool, bool]) -> RowFixture:
        return self.fixtures[config][0]

    # -- composing ------------------------------------------------------------

    def compose(
        self,
        fixtures: Sequence[RowFixture],
        *,
        endpoint: bool = False,
        blocks: Sequence[int] | None = None,
        pools: Pools | None = None,
    ) -> Step:
        from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
            PreparedRowBinding,
            RowPlan,
        )

        n = len(fixtures)
        blocks = list(blocks) if blocks is not None else list(range(1, n + 1))
        assert len(set(blocks)) == n and 0 not in blocks
        assert max(blocks) < self.capacity, "the pool set has too few blocks"
        pools = pools.clone() if pools is not None else self.template.clone()
        for fixture, block in zip(fixtures, blocks):
            pools.write_page(block, fixture.page)
        is_chunk = [f.kind == "chunk" for f in fixtures]
        bindings = tuple(
            PreparedRowBinding(
                request_id=f"r{i}",
                block_id=blocks[i],
                admission_generation=f.generation,
                geometry_id=f.geometry,
                prompt_index=f.prompt,
                prior_prompt_index=f.prior_prompt,
                allow_prompt_transition=False,
                is_chunk=is_chunk[i],
                ready_deadline_ns=(f.deadline or 1) if is_chunk[i] else 0,
            )
            for i, f in enumerate(fixtures)
        )
        endpoint_on = [endpoint and not f.file for f in fixtures]
        plan = RowPlan(
            state_indices_d=torch.zeros((0, 1), dtype=torch.int64),
            num_decodes=0,
            state_indices_p=torch.tensor(blocks, dtype=torch.int64),
            num_prefills=n,
            has_initial_states_p=torch.tensor(
                [not f.fresh for f in fixtures], dtype=torch.bool
            ),
            null_block_id=0,
            num_pool_blocks=self.capacity,
            live_block_ids=torch.tensor(sorted(blocks), dtype=torch.int64),
            geometry_id=torch.tensor([f.geometry for f in fixtures], dtype=torch.int64),
            prompt_index=torch.tensor([f.prompt for f in fixtures], dtype=torch.int64),
            is_chunk=torch.tensor(is_chunk, dtype=torch.bool),
            admission_generation=torch.tensor(
                [f.generation for f in fixtures], dtype=torch.int64
            ),
            ready_deadline_ns=torch.tensor(
                [b.ready_deadline_ns for b in bindings], dtype=torch.int64
            ),
            request_ids=tuple(b.request_id for b in bindings),
            execution_tier=0,
            bindings=bindings,
            endpoint_mode=torch.tensor(
                [int(e) for e in endpoint_on], dtype=torch.int64
            ),
            endpoint_threshold_frames=torch.tensor(
                [2 if e else 0 for e in endpoint_on], dtype=torch.int64
            ),
            endpoint_residue_frames=torch.zeros(n, dtype=torch.int64),
            is_file=torch.tensor([f.file for f in fixtures], dtype=torch.bool),
        )
        return Step(
            plan=plan,
            input_ids=torch.tensor([f.token for f in fixtures], dtype=torch.int64).to(
                self.device
            ),
            inputs_embeds=torch.stack([f.embed for f in fixtures]).to(self.device),
            pools=pools,
            fixtures=tuple(fixtures),
        )

    # -- running ----------------------------------------------------------------

    def run(
        self,
        step: Step,
        *,
        rows: Sequence[int] | None = None,
        pools: Pools | None = None,
        plan: Any = None,
        raises: bool = True,
        **overrides: Any,
    ) -> Outcome:
        """The option-off transaction over ``step`` (or its sub-plan ``rows``)
        on ``pools`` (default: a clone of the step's), in place."""
        from vllm_nemotron_asr_streaming.nemotron_asr import advance

        pools = pools if pools is not None else step.pools.clone()
        plan = plan if plan is not None else step.plan
        input_ids, embeds = step.input_ids, step.inputs_embeds
        if rows is not None:
            rows = sorted(rows)
            plan = restrict(plan, rows)
            index = torch.tensor(rows, dtype=torch.long, device=input_ids.device)
            input_ids = input_ids.index_select(0, index)
            embeds = embeds.index_select(0, index)
        sink = overrides.pop("commit_sink", None) or RecordingSink()
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        if kwargs.get("graph_covers_decode") and bool(plan.is_chunk.any()):
            # The model's dense decoder graphs cover CHUNK decode: the plan
            # carries the decoder tier for its CHUNK count, as the model sets it.
            graphs = self.model._decode_graph_binding
            plan = dataclasses.replace(
                plan, execution_tier=graphs.execution_tier(int(plan.is_chunk.sum()))
            )
        error = None
        out = None
        with torch.inference_mode():
            try:
                out = advance.advance_model_rows(
                    self.core,
                    input_ids.clone(),
                    embeds.clone(),
                    plan,
                    **pools.kwargs(),
                    **kwargs,
                    commit_sink=sink,
                    staging=None,
                )
            except Exception as exc:  # noqa: BLE001 - the caller decides
                if raises:
                    raise
                error = exc
        last = sink.reservations[-1] if sink.reservations else None
        staged = last is not None and last.staged is not None
        return Outcome(
            rows=None if out is None else out.detach().clone(),
            status=last.staged if staged else None,
            progress=last.progress if staged else None,
            queue_counts=last.queue_counts if staged else None,
            pools=pools,
            sink=sink,
            error=error,
        )

    def status_of(self, fixture: RowFixture) -> int:
        """The status a one-row step of ``fixture`` reports."""
        outcome = self.run(self.compose([fixture]))
        assert outcome.status is not None, "the one-row step staged no status"
        return int(outcome.status[0])

    # -- the split identity -----------------------------------------------------

    def split_identity(self, step: Step) -> list[str]:
        """Compare ``T(R)`` with the merge of ``T(R|C)`` and ``T(R|O)``
        (TXG-EQ-002); return every difference."""
        whole = self.run(step)
        merged_pools = step.pools.clone()
        chunk, other = chunk_positions(step.plan), other_positions(step.plan)
        parts = []
        for positions in (chunk, other):
            if positions:
                parts.append(
                    (positions, self.run(step, rows=positions, pools=merged_pools))
                )
        return compare_merge(whole, parts, merged_pools, step.population)


#: The sentinel patterns TXG-EQ-003's fresh rows' recycled pages hold.
SENTINELS = ("nan", "inf", "-inf", "random")


def holds_live_state(pools: Pools, block: int) -> bool:
    """Whether ``block`` holds nonzero state that differs from the null page
    (block 0) in at least one pool."""
    page, null = pools.page(block), pools.page(0)
    differs = any(not same_bits(page[name], null[name]) for name in page)
    nonzero = any(
        bool(t.detach().contiguous().view(torch.uint8).any()) for t in page.values()
    )
    return differs and nonzero


def replay_set(
    env: TransactionEnv, kind: str, count: int, *, seed: int = 0
) -> list[tuple[str, Step]]:
    """TXG-EQ-003's one-kind sub-plans for the cell of ``kind`` (``"chunk"``
    or ``"echo"``) at ``count``: one of ``count`` fresh rows whose recycled
    pages hold sentinel patterns (:data:`SENTINELS`), one of ``count``
    continuing rows whose pages hold nonzero state that differs from the
    null page, and, at ``count`` of 2 or more, one that mixes continuing and
    fresh rows. The rows alternate stream and file fixtures and take blocks
    in descending order from the top of the pool, so their block indices,
    row order and masks differ from the cell's capture rows (every one on
    block 0, alternating fresh and file bits). A continuing fixture whose
    page holds no such state fails the call."""
    row_kind = "chunk" if kind == "chunk" else "other"
    blocks = list(range(env.capacity - 1, env.capacity - 1 - count, -1))
    assert count >= 1 and min(blocks) > 0, "the pool set has too few blocks"
    shapes = [("fresh", [True] * count), ("continuing", [False] * count)]
    if count >= 2:
        shapes.append(("mixed", [i % 2 == 1 for i in range(count)]))
    out = []
    for label, freshness in shapes:
        fixtures = [
            env.accepted((row_kind, fresh, i % 2 == 1))
            for i, fresh in enumerate(freshness)
        ]
        step = env.compose(fixtures, blocks=blocks)
        for i, (fresh, block) in enumerate(zip(freshness, blocks, strict=True)):
            if fresh:
                pattern = SENTINELS[(seed + i) % len(SENTINELS)]
                step.pools.fill_page(block, pattern, seed=seed + i)
            elif not holds_live_state(step.pools, block):
                raise AssertionError(
                    f"the continuing {fixtures[i].name} fixture's page holds no "
                    "nonzero state distinct from the null page"
                )
        out.append((f"{kind}-{count}-{label}", step))
    return out


def compare_merge(
    whole: Outcome, parts, merged_pools: Pools, population: int
) -> list[str]:
    problems = []
    for field in ("rows", "status", "progress", "queue_counts"):
        full = getattr(whole, field)
        for positions, part in parts:
            piece = getattr(part, field)
            if full is None and piece is None:
                continue
            if full is None or piece is None:
                problems.append(f"{field}: present in one run only")
                continue
            index = torch.tensor(positions, dtype=torch.long, device=full.device)
            if not same_bits(full.index_select(0, index), piece):
                problems.append(f"{field} differs at rows {positions}")
    differing = whole.pools.differences(merged_pools)
    if differing:
        problems.append(f"resident pages differ in {differing}")
    return problems


# ---------------------------------------------------------------------------
# Building an environment


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._real = None

    def __call__(self, core, input_ids, inputs_embeds, plan, **kwargs):
        self.calls.append(
            {
                "core": core,
                "input_ids": input_ids.detach().clone(),
                "inputs_embeds": inputs_embeds.detach().clone(),
                "plan": plan,
                "kwargs": kwargs,
                "pools": pools_of(kwargs).clone(),
            }
        )
        return self._real(core, input_ids, inputs_embeds, plan, **kwargs)


@contextlib.contextmanager
def recording(monkeypatch) -> Iterator[_Recorder]:
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    recorder = _Recorder()
    recorder._real = advance.advance_model_rows
    monkeypatch.setattr(advance, "advance_model_rows", recorder)
    yield recorder


def _stream(
    config,
    req_id: str,
    geometry: int,
    sequence: int,
    final: bool,
    samples: int | None = None,
):
    lookahead = {0: 0, 1: 1, 2: 3, 3: 6, 4: 13}[geometry]
    audio = stream_envelope(
        geometry=geometry,
        lookahead=lookahead,
        sequence=sequence,
        samples=samples,
        final=final,
    )
    return h.new_request(
        req_id,
        int(config.audio_chunk_token_id),
        audio=audio,
        identifier=f"uuid-{req_id}-{sequence}",
    )


def _file(config, req_id: str, geometry: int, n_samples: int):
    samples = (np.random.default_rng(7).standard_normal(n_samples) * 0.1).astype(
        np.float32
    )
    return h.new_request(
        req_id,
        int(config.file_carrier_token_id),
        samples=samples,
        header=[geometry, 0, n_samples],
    )


def chunk_samples(lookahead: int) -> int:
    return 1280 * (lookahead + 1)


def build_env(
    tmp_path: Path, lookahead: int, *, device: str = "cpu", capacity: int = 18
) -> TransactionEnv:
    """See the module docstring. ``capacity`` is the composed pools' block
    count (block 0 is the null page)."""
    geometry = {0: 0, 1: 1, 3: 2, 6: 3, 13: 4}[lookahead]
    with pytest.MonkeyPatch.context() as mp:
        with recording(mp) as recorder:
            card = h.write_runnable_card(tmp_path / f"txg-card-{lookahead}")
            vllm_config = h.build_vllm_config(
                card,
                right_context=lookahead,
                # The graphed decoder serves at most max_num_seqs rows, so the
                # model is sized for the largest population the composed pools
                # hold (block 0 is the null page).
                max_num_seqs=capacity - 1,
                device=device,
            )
            # Seeded weights: the drive must reach every configuration the
            # planner produces whatever ran earlier in the process.
            devices = [torch.cuda.current_device()] if device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(FIXTURE_SEED)
                model = h.build_model(vllm_config)
            state = h.build_model_state(vllm_config, model)
            runner = h.Runner(vllm_config, model, state)
            # The eager runner: no capture hook; real steps before post_warmup.
            TxgCoreOrder(runner, device=device, capacity=capacity, spy=False).startup(
                verify=False
            )
            calls = _drive(runner, model.config, geometry, lookahead)
            del recorder
    return _env_from_calls(
        calls, model, state, model.config, geometry, lookahead, device, capacity
    )


def env_from_runner(
    runner,
    model,
    state,
    *,
    lookahead: int,
    capacity: int,
    monkeypatch,
    device: str = "cpu",
) -> TransactionEnv:
    """A :class:`TransactionEnv` over an already started ``runner`` (its
    serving pool bound, no request scheduled yet): the same drive and fixture
    extraction as :func:`build_env`. Every driven step must run the option-off
    transaction, so a caller whose model has transaction graphs on makes the
    binding decline while this runs. ``capacity`` is the serving pool's block
    count, which the composed pools then share. The returned environment's
    ``serving`` holds the live serving pool views of the first driven step
    (not copies)."""
    geometry = {0: 0, 1: 1, 3: 2, 6: 3, 13: 4}[lookahead]
    with recording(monkeypatch):
        calls = _drive(runner, model.config, geometry, lookahead)
    env = _env_from_calls(
        calls, model, state, model.config, geometry, lookahead, device, capacity
    )
    env.serving = pools_of(calls[0]["kwargs"])
    for (name, live), (_, composed) in zip(
        env.serving.flat(), env.template.flat(), strict=True
    ):
        assert live.shape == composed.shape, (name, live.shape, composed.shape)
    return env


def _drive(runner, config, geometry: int, lookahead: int) -> list[dict[str, Any]]:
    """Real steps of one stream and one file until every configuration the
    planner produces has been seen; returns the recorded calls with the
    step's statuses."""
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    recorder = advance.advance_model_rows
    park = int(config.eos_token_id)
    handles = {"s1": 1, "f1": 2}
    positions: dict[str, int] = {}
    seen: set[tuple[str, bool, bool]] = set()
    statuses: list[dict[str, int]] = []
    sequence = 0

    def step(output) -> None:
        before = len(recorder.calls)
        runner.real_step(output)
        assert len(recorder.calls) == before + 1, "the step ran no transaction"
        statuses.append(runner.commit_log[-1][0])
        call = recorder.calls[-1]
        call["statuses"] = runner.commit_log[-1][0]
        plan = call["plan"]
        for i in range(len(plan.request_ids)):
            seen.add(_config_at(plan, i))

    step(
        h.scheduler_output(
            new=(
                _stream(config, "s1", geometry, 0, False),
                _file(config, "f1", geometry, int(3.5 * chunk_samples(lookahead))),
            ),
            handles=handles,
        )
    )
    positions = {"s1": 1, "f1": 1}
    file_live = True
    needed = {c for c in CONFIGS if c[0] == "chunk" or not c[1]}
    for _ in range(400):
        if needed <= seen:
            break
        cached, new = [], []
        if runner._sampled.get("s1") == park:
            sequence += 1
            new.append(_stream(config, "s1", geometry, sequence, False))
            positions["s1"] = 0
        else:
            cached.append("s1")
        if file_live:
            cached.append("f1")
        step(
            h.scheduler_output(
                new=tuple(new),
                cached=tuple(cached),
                handles=handles,
                computed={r: positions[r] for r in cached},
            )
        )
        for r in cached:
            positions[r] += 1
        for r in new:
            positions[r.req_id] = 1
        if (
            runner._sampled.get("f1") == park
            and file_live
            and (("chunk", False, True) in seen and ("other", False, True) in seen)
        ):
            file_live = False
    missing = needed - seen
    assert not missing, f"the driver never produced {sorted(missing)}"
    return list(recorder.calls)


def _config_at(plan, i: int) -> tuple[str, bool, bool]:
    is_file = bool(plan.is_file[i]) if plan.is_file.numel() else False
    fresh = not bool(plan.has_initial_states_p[i])
    return ("chunk" if bool(plan.is_chunk[i]) else "other", fresh, is_file)


def _env_from_calls(
    calls, model, state, config, geometry, lookahead, device, capacity
) -> TransactionEnv:
    calls = [c for c in calls if "statuses" in c]
    first = calls[0]
    kwargs = {k: v for k, v in first["kwargs"].items() if k not in STEP_OWNED}
    # Zeroed pools with the live pools' resident layout (the composed pools'
    # block count is the live pool's), so the option-off transaction over a
    # composed step launches the masked scatter at a layout startup warmed.
    live = pools_of(first["kwargs"])
    for name, tensor in live.flat():
        assert tensor.shape[0] == capacity, (name, tuple(tensor.shape), capacity)
    template = live.zeros()
    fixtures: dict[tuple[str, bool, bool], list[RowFixture]] = {c: [] for c in CONFIGS}
    for call in calls:
        plan = call["plan"]
        for i, request_id in enumerate(plan.request_ids):
            config_key = _config_at(plan, i)
            binding = plan.bindings[i]
            status = int(call["statuses"].get(request_id, 0))
            fixtures[config_key].append(
                RowFixture(
                    name=f"{config_name(config_key)}@{request_id}",
                    kind=config_key[0],
                    fresh=config_key[1],
                    file=config_key[2],
                    token=int(call["input_ids"][i]),
                    embed=call["inputs_embeds"][i].clone(),
                    page=call["pools"].page(binding.block_id),
                    geometry=binding.geometry_id,
                    prompt=binding.prompt_index,
                    prior_prompt=binding.prior_prompt_index,
                    generation=binding.admission_generation,
                    deadline=binding.ready_deadline_ns,
                    status=status,
                )
            )
    # Accepted fixtures first; then derive the fresh non-CHUNK rows.
    for config_key, items in fixtures.items():
        items.sort(key=lambda f: f.status != 0)
    park = int(config.eos_token_id)
    for file in (False, True):
        base = fixtures[("chunk", True, file)][0]
        fixtures[("other", True, file)].insert(
            0,
            base.but(
                name=f"other-fresh-{'file' if file else 'stream'}@park",
                kind="other",
                token=park,
                deadline=0,
                status=None,
            ),
        )
    env = TransactionEnv(
        core=first["core"],
        kwargs=kwargs,
        template=template,
        geometry=geometry,
        lookahead=lookahead,
        config=config,
        device=device,
        fixtures=fixtures,
        model=model,
        state=state,
    )
    env.drive_null_pages = [call["pools"].page(0) for call in calls]
    env.drive_null_pages.append(live.page(0))
    for config_key in CONFIGS:
        head = fixtures[config_key][0]
        if head.status is None:
            fixtures[config_key][0] = head.but(status=env.status_of(head))
    for fresh in (False, True):
        # Regular stream units: version 2, the server's geometry, a full chunk,
        # not a final tail; and accepted by their own step.
        head = fixtures[("chunk", fresh, False)][0]
        header = head.embed[:HEADER_SLOTS]
        assert int(header[H_VERSION]) == 2, head.name
        assert int(header[H_GEOMETRY]) == geometry, head.name
        assert int(header[H_FINAL]) == 0, head.name
        assert int(header[H_SAMPLES]) == chunk_samples(lookahead), head.name
        assert head.status == 0, (head.name, head.status)
    return env


# ---------------------------------------------------------------------------
# Rejection fixtures


#: Status classes (``advance`` / ``frontend`` bit values).
def status_classes() -> dict[str, int]:
    from vllm_nemotron_asr_streaming.nemotron_asr import advance, frontend

    return {
        "ROW_STATUS_ROLE_MISMATCH": advance.ROW_STATUS_ROLE_MISMATCH,
        "ROW_STATUS_ENVELOPE": advance.ROW_STATUS_ENVELOPE,
        "ROW_STATUS_SEQUENCE": frontend.ROW_STATUS_SEQUENCE,
        "ROW_STATUS_ECHO_MISMATCH": advance.ROW_STATUS_ECHO_MISMATCH,
        "ROW_STATUS_QUEUE_NOT_DRAINED": advance.ROW_STATUS_QUEUE_NOT_DRAINED,
        "ROW_STATUS_FINAL_OVERSIZE": frontend.ROW_STATUS_FINAL_OVERSIZE,
        "ROW_STATUS_PROMPT_MISMATCH": advance.ROW_STATUS_PROMPT_MISMATCH,
    }


#: Combinations no fixture can produce, by construction (TXG-EQ-002).
STRUCTURAL_EXCLUSIONS = (
    (
        ("chunk", True, False),
        "ROW_STATUS_QUEUE_NOT_DRAINED",
        "a fresh row's initial book has an empty queue",
    ),
    (
        ("chunk", True, True),
        "ROW_STATUS_QUEUE_NOT_DRAINED",
        "a fresh row's initial book has an empty queue",
    ),
    (
        ("other", True, False),
        "ROW_STATUS_ECHO_MISMATCH",
        "a fresh row has no armed echo",
    ),
    (
        ("other", True, True),
        "ROW_STATUS_ECHO_MISMATCH",
        "a fresh row has no armed echo",
    ),
)


def _candidates(env: TransactionEnv, config, name: str) -> list[RowFixture]:
    """Mutations of ``config``'s accepted fixture meant to produce ``name``."""
    kind, fresh, file = config
    base = env.accepted(config)
    out: list[RowFixture] = []
    if name == "ROW_STATUS_ROLE_MISMATCH":
        out.append(
            base.but(token=env.placeholder if (kind == "other" or file) else env.park)
        )
    elif name == "ROW_STATUS_ENVELOPE" and kind == "chunk":
        out.append(base.with_header(version=float(base.embed[H_VERSION]) + 1))
        embed = base.embed.clone()
        embed[:HEADER_SLOTS] = 0
        out.append(base.but(embed=embed))
    elif name == "ROW_STATUS_SEQUENCE" and kind == "chunk":
        out.append(base.with_header(sequence=float(base.embed[H_SEQUENCE]) + 3))
    elif name == "ROW_STATUS_ECHO_MISMATCH" and kind == "other" and not fresh:
        for delta in (1, 2, 3):
            out.append(base.but(token=(base.token + delta) % 40))
    elif name == "ROW_STATUS_QUEUE_NOT_DRAINED" and kind == "chunk" and not fresh:
        for drain in env.fixtures[("other", False, file)]:
            out.append(base.but(page=drain.page))
    elif name == "ROW_STATUS_FINAL_OVERSIZE" and kind == "chunk":
        full = chunk_samples(env.lookahead)
        embed = base.embed.clone()
        embed[H_FINAL] = 1
        embed[H_SAMPLES] = full
        if not bool(embed[HEADER_SLOTS : HEADER_SLOTS + full].any()):
            embed[HEADER_SLOTS : HEADER_SLOTS + full] = 0.05
        out.append(base.but(embed=embed))
    elif name == "ROW_STATUS_PROMPT_MISMATCH" and kind == "chunk":
        out.append(base.with_header(prompt=float(base.prompt) + 1))
    return out


def rejection_fixtures(
    env: TransactionEnv,
) -> dict[tuple, list[tuple[str, RowFixture]]]:
    """Per configuration, ``(class name, fixture)`` for every class a fixture
    reproduces exactly; every other combination goes to ``env.excluded``."""
    if env._rejections is not None:
        return env._rejections
    classes = status_classes()
    structural = {(c, n) for c, n, _ in STRUCTURAL_EXCLUSIONS}
    for c, n, why in STRUCTURAL_EXCLUSIONS:
        env.excluded.append((config_name(c), n, why))
    out: dict[tuple, list[tuple[str, RowFixture]]] = {c: [] for c in CONFIGS}
    for config_key in CONFIGS:
        for name, bit in classes.items():
            if (config_key, name) in structural:
                continue
            kept = None
            observed = []
            for candidate in _candidates(env, config_key, name):
                status = env.status_of(candidate)
                observed.append(status)
                if status == bit:
                    kept = candidate.but(
                        status=status,
                        intended=bit,
                        name=f"{config_name(config_key)}:{name}",
                    )
                    break
            if kept is not None:
                out[config_key].append((name, kept))
            else:
                env.excluded.append(
                    (
                        config_name(config_key),
                        name,
                        "no mutation reports exactly this class "
                        f"(observed {observed or 'no candidate'})",
                    )
                )
    env._rejections = out
    return out


# ---------------------------------------------------------------------------
# Tracing


class OpTrace(torch.utils._python_dispatch.TorchDispatchMode):
    """Records every aten op with its argument shapes and dtypes."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[tuple] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def describe(x):
            if isinstance(x, torch.Tensor):
                return ("T", tuple(x.shape), str(x.dtype))
            if isinstance(x, (list, tuple)):
                return tuple(describe(y) for y in x)
            return type(x).__name__

        self.ops.append((str(func), describe(args)))
        return func(*args, **kwargs)


def trace_phase(
    env: TransactionEnv, step: Step, phase: str, **overrides
) -> list[tuple]:
    """The op sequence of one phase function of ``advance_model_rows`` while
    the option-off transaction runs ``step``."""
    from support import txg_api as api
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    real = api.phase_function(phase)
    traces: list[OpTrace] = []

    def traced(*args, **kwargs):
        trace = OpTrace()
        traces.append(trace)
        with trace:
            return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(advance, phase, traced)
        env.run(step, **overrides)
    assert len(traces) == 1, f"{phase} ran {len(traces)} times"
    return traces[0].ops


def spy_calls(
    monkeypatch, module: Any, name: str, log: list, tag: str | None = None
) -> None:
    """Wrap ``module.name``: append ``(tag, args, kwargs)`` then call through."""
    real = getattr(module, name)

    def spy(*args, **kwargs):
        log.append((tag or name, args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)


def wrap_decode_resolver(env: TransactionEnv, record: Callable[[Any, tuple], None]):
    """A decode resolver that calls ``record(request, decode_args)`` for each
    decode the transaction runs, then the model's decoder."""
    real = env.kwargs["decode_resolver"]

    def resolver(request):
        resolved = real(request)
        decode_fn = resolved.decode_fn

        def recording_decode(*args, **kwargs):
            record(request, args)
            return decode_fn(*args, **kwargs)

        return dataclasses.replace(resolved, decode_fn=recording_decode)

    return resolver


def tensors_in(value: Any, depth: int = 0) -> Iterator[torch.Tensor]:
    """Tensors in ``value`` and in lists, tuples, dicts and dataclass fields
    nested up to three levels."""
    if isinstance(value, torch.Tensor):
        yield value
    elif depth >= 3:
        return
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors_in(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from tensors_in(getattr(value, f.name), depth + 1)
