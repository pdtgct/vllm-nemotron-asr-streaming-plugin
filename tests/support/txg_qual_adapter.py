# SPDX-License-Identifier: Apache-2.0
"""The test-only qualification adapter of TXG-QUAL-026 and the transaction
monitor of TXG-QUAL-003 and TXG-QUAL-013.

``QualAdapter`` drives the worker of a hook-captured process at the
transaction level: the model, the model state and the state layer's views
that core bound and captured over. Per step, as core's
``GPUModelRunner.execute_model`` does (``vllm/v1/worker/gpu/model_runner.py``:
``finish_requests``, ``add_requests``, then the forward and
``sample_tokens``), it:

1. calls ``remove_request`` for each request that finished, in ascending
   request-ID order;
2. for each request the step delivers as ``NewRequestData`` in
   ``scheduled_new_reqs`` (a new request, or a stream's next turn with its
   compacted prompt, one frontier token followed by the turn's input, and
   ``num_computed_tokens`` 1, as core's scheduler compacts a state session),
   in that list's order, calls ``remove_request`` for that request and then,
   before any other lifecycle call, its ``add_request``, so removal and
   addition interleave per request; a cached request (a file's later step, a
   label drain within a stream's turn) gets neither call;
3. calls ``prepare_inputs``, runs the forward on the runner stream (the
   current stream), and calls ``postprocess_state``, which waits for the
   step's commit event, then ``take_failed_requests``.

Each request holds one block below the final committed block count
(:attr:`QualAdapter.final_committed_blocks`), the lowest free block above
block 0, which the driver assigns at the request's first step (``Blocks``),
keeps until its terminal finish and reuses only after that; the block
reaches the model state as the request's ``block_ids``. The lifecycle calls
run outside any timer: the monitor times only the transaction call.

``before_forward(blocks, views)`` runs inside the step, after
``prepare_inputs`` and before the forward, with the blocks of the step's rows
in row order and the state layer's bound views (core's ``kv_cache``); the
step rule restores its snapshot there, by an in-place copy into the same pool
memory on the runner stream, only at those blocks (TXG-QUAL-011). No replay,
restore or store runs outside a step the adapter runs.

``TransactionMonitor`` wraps the binding's ``execute`` and
``advance.advance_model_rows`` (both looked up at call time by the model's
forward). It times each transaction call from the call to
``torch.cuda.synchronize()`` after it, after a synchronization that drains
earlier work (the clock of TXG-QUAL-013), captures each call's staged status,
progress and remaining queue counts through a proxy commit sink, and, with
``compare``, runs the option-off transaction of every covered or declined
step on a detached copy of the pools taken before the call: new storage with
the same resident layout, test-only storage outside REG-STATE-003 and
TXG-CAP-013. The copy holds pages ``0 … copy_extent - 1`` and no more
(:attr:`TransactionMonitor.copy_extent`): block 0 and one page per request
the run's mode can hold (:func:`exactness_extent`; for ``qm`` pages 0 to the
admission limit, for ``fallback`` pages 0 to ``N + 4``), the only pages the
driver assigns its requests; without a request count, the final committed
block count. Each view is narrowed to that extent before anything is read or
allocated, and ``txg_transaction.layout_clone`` copies only the bytes the
narrowed views cover, so no page at or above the extent is read or cloned.
The committed pages from the extent to the final committed block count are
digested in place before the run's first step (:meth:`watch_untouched`) and
after its last (:meth:`untouched_problems`), in bounded chunks and with no
copy of them; a changed digest is a problem of the run. The reference
receives no view of the serving pools, so it can neither read nor write
them; the monitor checks that, and compares rows, status, progress, queue
counts and every copied page bit for bit (TXG-QUAL-003). With ``memory``,
it records ``torch.cuda.memory_reserved`` and ``max_memory_reserved``
before and after each copy, the bytes the copy copies (``copy_bytes``) and
its allocation, alignment slack included (``copy_allocation_bytes``;
TXG-QUAL-029). The pool copy, the option-off reference, the checks and the
comparison are outside the timer.
``force_off`` makes ``execute`` decline without counting, so the forward
runs the option-off transaction (the step rule's OFF legs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from support import txg_transaction as tx

ADVANCE_MODULE = "vllm_nemotron_asr_streaming.nemotron_asr.advance"
BINDING_ATTRIBUTE = "_transaction_graph_binding"


def adapter_identity() -> str:
    """The adapter's identity, frozen with the run (TXG-QUAL-018): the
    SHA-256 of this module's source."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Worker access


def runner_of(obj: Any) -> Any:
    """The GPU model runner of a worker, a worker wrapper or a runner."""
    for name in ("model_runner", "worker"):
        inner = getattr(obj, name, None)
        if inner is not None and inner is not obj:
            return runner_of(inner)
    return obj


@dataclasses.dataclass
class StepResult:
    req_ids: tuple[str, ...]
    #: The token each scheduled request emitted (the argmax of the model's
    #: forced logits, which the sampler draws).
    sampled: dict[str, int]
    #: Requests the step failed (``take_failed_requests``).
    failed: frozenset[str]
    hidden: torch.Tensor | None


class QualAdapter:
    """See the module docstring. ``model`` and ``state`` are the worker's
    own objects; the state layer's views are read where core bound them."""

    def __init__(
        self, *, vllm_config: Any, model: Any, state: Any, runner: Any = None
    ) -> None:
        self.vllm_config = vllm_config
        self.model = model
        self.state = state
        #: Core's model runner, when the adapter has one: its live
        #: ``kv_cache_config`` holds the final committed block count after
        #: core's extension.
        self.runner = runner
        self.layer = model._persistent_state_layer
        if getattr(self.layer, "kv_cache", None) is None or not len(
            self.layer.kv_cache
        ):
            raise RuntimeError("core has bound no state pool to the state layer")
        self.device = torch.device(state.device)
        self.cuda = self.device.type == "cuda"
        from support import model_harness as h

        self._h = h
        self.req_states = h.StepRequestStates()
        #: Each live request's recorded prompt, the position of its current
        #: turn's input in it, and the last token fed (the frontier of its
        #: next turn).
        self._prompts: dict[str, list[int]] = {}
        self._base: dict[str, int] = {}
        self._fed: dict[str, int] = {}
        #: Each live request's block (``block_ids``).
        self.blocks: dict[str, int] = {}
        self.sampled: dict[str, int] = {}

    @classmethod
    def from_worker(cls, worker: Any) -> QualAdapter:
        runner = runner_of(worker)
        state = runner.model_state
        return cls(
            vllm_config=runner.vllm_config,
            model=state.model,
            state=state,
            runner=runner,
        )

    # -- the state layer --------------------------------------------------------

    @property
    def views(self) -> tuple[torch.Tensor, ...]:
        """The state layer's per-state views as core bound them."""
        return tuple(self.layer.kv_cache)

    @property
    def capacity(self) -> int:
        """The bound block count, the null block included: the first
        dimension of the state layer's bound views."""
        return int(self.views[0].shape[0])

    @property
    def final_committed_blocks(self) -> int:
        """The final committed block count, the null block included, read
        now (after core's extension under the extensible KV cache): the
        smaller of the bound views' first dimension and the block count of
        core's live ``kv_cache_config``, which the extension updates in
        place. Every block the driver assigns, and every page its detached
        copies hold, is below it (TXG-QUAL-003, TXG-QUAL-026)."""
        bound = self.capacity
        config = getattr(self.runner, "kv_cache_config", None)
        committed = getattr(config, "num_blocks", None)
        if isinstance(committed, int) and committed > 0:
            return min(bound, committed)
        return bound

    @property
    def page_bytes(self) -> int:
        return int(self.layer.layout.page_size_bytes)

    def last_fed(self, req_id: str) -> int:
        """The last token ``req_id`` fed: the frontier token of its next
        turn (core keeps the last computed token and drops the final sampled
        one)."""
        return self._fed[req_id]

    # -- scheduler output -------------------------------------------------------

    def next_turn(self, request: Any) -> Any:
        """``request`` (a turn's input as a fresh ``NewRequestData``) as core
        delivers a live stream's next turn: the compacted prompt, one
        frontier token (the last token the request fed) followed by the
        input, ``num_computed_tokens`` 1, and the input's features moved
        past the frontier token."""
        from vllm.multimodal.inputs import PlaceholderRange

        prompt = [self.last_fed(request.req_id), *request.prompt_token_ids]
        features = [
            dataclasses.replace(
                f,
                mm_position=PlaceholderRange(
                    offset=int(f.mm_position.offset) + 1,
                    length=int(f.mm_position.length),
                ),
            )
            for f in request.mm_features or ()
        ]
        return dataclasses.replace(
            request,
            prompt_token_ids=prompt,
            prefill_token_ids=prompt,
            mm_features=features,
            num_computed_tokens=1,
        )

    def scheduler_output(
        self,
        *,
        new: Sequence[Any],
        cached: Sequence[str],
        finished: Sequence[str],
        blocks: dict[str, int],
        computed: dict[str, int],
        tokens: dict[str, int] | None = None,
        free_mm_hashes: Sequence[str] = (),
    ) -> Any:
        """The step's ``SchedulerOutput``. A new request of a live stream
        becomes its next turn (:meth:`next_turn`); every new request carries
        its block. ``computed`` gives each cached request's position within
        its turn; ``tokens`` a row's scheduled token count (default 1)."""
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        tokens = dict(tokens or {})
        finished = set(finished)
        delivered = []
        for request in new:
            if request.req_id in self._prompts and request.req_id not in finished:
                request = self.next_turn(request)
            delivered.append(
                dataclasses.replace(request, block_ids=([int(blocks[request.req_id])],))
            )
        scheduled = {r.req_id: tokens.get(r.req_id, 1) for r in delivered}
        scheduled |= {r: tokens.get(r, 1) for r in cached}
        return SchedulerOutput(
            scheduled_new_reqs=delivered,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=list(cached),
                new_token_ids=[],
                new_block_ids=[None for _ in cached],
                num_computed_tokens=[
                    self._base.get(r, 0) + int(computed[r]) for r in cached
                ],
                num_output_tokens=[1 for _ in cached],
            ),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={
                r.req_id: [len(r.mm_features) - 1] for r in delivered if r.mm_features
            },
            finished_req_ids=finished,
            free_encoder_mm_hashes=list(free_mm_hashes),
        )

    # -- one step ----------------------------------------------------------------

    def _forget(self, req_id: str) -> None:
        for table in (self._prompts, self._base, self._fed, self.blocks):
            table.pop(req_id, None)
        self.sampled.pop(req_id, None)

    def _remove_request(self, req_id: str) -> None:
        """Core's ``_remove_request``: the model state first; the runner's
        own request state and the encoder cache only for a request it
        knows."""
        state = self.state
        state.remove_request(req_id)
        if req_id in self.req_states.req_id_to_index:
            self.req_states.remove(req_id)
            state.encoder_cache.remove_request(req_id)

    @torch.inference_mode()
    def step(
        self,
        *,
        new: Sequence[Any] = (),
        cached: Sequence[str] = (),
        finished: Sequence[str] = (),
        blocks: dict[str, int],
        computed: dict[str, int] | None = None,
        tokens: dict[str, int] | None = None,
        order: Sequence[str] | None = None,
        free_mm_hashes: Sequence[str] = (),
        before_forward: Callable[[tuple[int, ...], tuple[torch.Tensor, ...]], None]
        | None = None,
    ) -> StepResult:
        """One real step. ``order`` is the row order (default: new requests,
        then cached ones); ``computed`` gives each cached request's position
        within its current turn; ``tokens`` each row's scheduled token count
        (default 1; core's dummy batch layout puts a row's extra tokens on
        the last rows, so a multi-token row goes last)."""
        computed = dict(computed or {})
        output = self.scheduler_output(
            new=new,
            cached=cached,
            finished=finished,
            blocks=blocks,
            computed=computed,
            tokens=tokens,
            free_mm_hashes=free_mm_hashes,
        )
        state = self.state
        for req_id in sorted(output.finished_req_ids):
            self._remove_request(req_id)
            self._forget(req_id)
        for mm_hash in output.free_encoder_mm_hashes:
            state.encoder_cache.free_encoder_cache(mm_hash)
        for request in output.scheduled_new_reqs:
            self._remove_request(request.req_id)
            index = self.req_states.add(request.req_id)
            state.encoder_cache.add_request(request.req_id, request.mm_features)
            state.add_request(index, request)
            self._prompts[request.req_id] = list(request.prefill_token_ids)
            self._base[request.req_id] = int(request.num_computed_tokens)
            self.blocks[request.req_id] = int(request.block_ids[0][0])
            self.sampled.pop(request.req_id, None)
        if output.scheduled_new_reqs:
            state.apply_staged_writes()
        req_ids = [r.req_id for r in output.scheduled_new_reqs] + list(cached)
        if order is not None:
            if sorted(order) != sorted(req_ids):
                raise ValueError("the row order must name every scheduled request")
            req_ids = list(order)
        hidden = None
        failed: frozenset[str] = frozenset()
        if req_ids:
            batch, fed = self._batch(output, req_ids)
            state.preprocess_state(batch, (), None, None)
            attn = self._attn(batch)
            embeds = state.prepare_inputs_embeds(
                output.scheduled_encoder_inputs, batch, self.req_states
            )
            kwargs = state.prepare_inputs(batch, self.req_states)
            if before_forward is not None:
                before_forward(tuple(self.blocks[r] for r in req_ids), self.views)
            hidden = self._forward(batch, embeds, kwargs, attn)
            logits = state.compute_sample_logits(hidden[batch.logits_indices])
            sampled = logits.argmax(dim=-1).tolist()
            for req_id, token in zip(batch.req_ids, sampled, strict=True):
                self.sampled[req_id] = int(token)
            state.postprocess_state(
                batch.idx_mapping,
                torch.ones(batch.num_reqs, dtype=torch.int32, device=self.device),
                None,
            )
            self._fed.update(fed)
        taken = state.take_failed_requests()
        failed = frozenset(taken or ())
        return StepResult(
            req_ids=tuple(req_ids),
            sampled={r: self.sampled[r] for r in req_ids},
            failed=failed,
            hidden=hidden,
        )

    def _batch(self, output: Any, req_ids: list[str]) -> tuple[Any, dict[str, int]]:
        """Core's input batch for the rows in ``req_ids`` order, each at its
        scheduled position, and the last token each row feeds."""
        counts = [int(output.num_scheduled_tokens[r]) for r in req_ids]
        n, total = len(req_ids), sum(counts)
        batch = self._h.dummy_batch(self.state, n, total, is_padding=False)
        if [int(c) for c in batch.num_scheduled_tokens] != counts:
            raise ValueError(
                f"token counts {counts} do not match core's batch layout "
                f"{list(batch.num_scheduled_tokens)}: put a multi-token row last"
            )
        cached = output.scheduled_cached_reqs
        positions = {
            r.req_id: int(r.num_computed_tokens) for r in output.scheduled_new_reqs
        }
        positions |= {
            r: int(p)
            for r, p in zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        }
        tokens: list[int] = []
        fed: dict[str, int] = {}
        for req_id, count in zip(req_ids, counts, strict=True):
            prompt = self._prompts[req_id]
            for position in range(positions[req_id], positions[req_id] + count):
                tokens.append(
                    prompt[position] if position < len(prompt) else self.sampled[req_id]
                )
            fed[req_id] = tokens[-1]
        batch.input_ids[:total] = torch.tensor(tokens, device=batch.input_ids.device)
        index = np.array(
            [self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp
        )
        return (
            dataclasses.replace(
                batch,
                req_ids=list(req_ids),
                idx_mapping=torch.from_numpy(index).to(self.device),
                idx_mapping_np=index,
                num_computed_tokens_np=np.array(
                    [positions[r] for r in req_ids], dtype=np.int32
                ),
                prefill_len_np=np.array(
                    [len(self._prompts[r]) for r in req_ids], dtype=np.int32
                ),
            ),
            fed,
        )

    def _attn(self, batch: Any) -> Any:
        from vllm.config import CUDAGraphMode

        return self.state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (),
            torch.zeros(batch.num_tokens, dtype=torch.int64, device=self.device),
            [],
            None,
        )

    def _forward(self, batch: Any, embeds: Any, kwargs: dict, attn: Any) -> Any:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.model_executor.models.interfaces import requires_raw_input_tokens

        num_tokens = batch.num_tokens_after_padding
        input_ids = batch.input_ids[:num_tokens]
        if embeds is not None and not requires_raw_input_tokens(self.model):
            input_ids = None
        with set_forward_context(
            attn,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            is_padding=batch.is_padding,
        ):
            return self.model(
                input_ids=input_ids,
                positions=batch.positions[:num_tokens],
                inputs_embeds=embeds,
                intermediate_tensors=None,
                **kwargs,
            )


# ---------------------------------------------------------------------------
# Snapshots of the step's blocks (TXG-QUAL-011)


class BlockSnapshot:
    """The step rule's snapshot: taken inside the first plan step, after
    ``prepare_inputs`` and before its transaction, from the pages the prefix
    left at the blocks of the step's rows; restored inside every later step
    by an in-place copy into the same pool memory on the runner stream, at
    those blocks only (the adapter's ``before_forward``)."""

    def __init__(self) -> None:
        self.pages: dict[int, list[torch.Tensor]] | None = None
        self.restores = 0
        #: The blocks each restore wrote, in order.
        self.restored: list[tuple[int, ...]] = []

    def __call__(
        self, blocks: tuple[int, ...], views: tuple[torch.Tensor, ...]
    ) -> None:
        if self.pages is None:
            self.pages = {b: [view[b].clone() for view in views] for b in blocks}
            return
        if set(blocks) != set(self.pages):
            raise RuntimeError("a plan step holds other blocks than the snapshot")
        for block in blocks:
            for view, saved in zip(views, self.pages[block], strict=True):
                view[block].copy_(saved)
        self.restores += 1
        self.restored.append(tuple(blocks))


# ---------------------------------------------------------------------------
# The transaction monitor


@dataclasses.dataclass
class Call:
    """One transaction call of a step."""

    kind: str  # "execute" (the binding) or "off" (advance_model_rows)
    seconds: float = 0.0
    covered: bool | None = None
    population: int = 0
    chunk_count: int = 0
    plan: Any = None
    input_ids: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    status: torch.Tensor | None = None
    progress: torch.Tensor | None = None
    queue_counts: torch.Tensor | None = None
    #: The option-off reference's rows (``compare``).
    reference_rows: torch.Tensor | None = None
    compared: bool = False
    problems: list[str] = dataclasses.field(default_factory=list)


class _Reservation:
    def __init__(self, real: Any, plan: Any) -> None:
        self._real = real
        self.plan = plan
        self.staged = None
        self.progress = None
        self.queue_counts = None

    def stage(self, row_status: torch.Tensor) -> None:
        self._real.stage(row_status)
        self.staged = row_status.detach().clone()
        progress = getattr(self.plan, "progress", None)
        counts = getattr(self.plan, "queue_counts", None)
        self.progress = None if progress is None else progress.detach().clone()
        self.queue_counts = None if counts is None else counts.detach().clone()

    def cancel(self) -> None:
        self._real.cancel()


class _Sink:
    """Wraps the step's commit sink and keeps what each reservation staged."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.reservations: list[_Reservation] = []

    def reserve(self, plan: Any) -> _Reservation:
        reservation = _Reservation(self._real.reserve(plan), plan)
        self.reservations.append(reservation)
        return reservation

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ReferenceSink:
    """The option-off reference's own sink: stages nothing anywhere."""

    def __init__(self) -> None:
        self.last: _Reservation | None = None

    def reserve(self, plan: Any) -> _Reservation:
        self.last = _Reservation(_NoTicket(), plan)
        return self.last


class _NoTicket:
    def stage(self, row_status: torch.Tensor) -> None:
        pass

    def cancel(self) -> None:
        pass


def _storage_span(t: torch.Tensor) -> tuple[torch.device, int, int]:
    storage = t.untyped_storage()
    return t.device, storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def detached_problems(
    copy: tx.Pools, live: tx.Pools, other: dict[str, Any] | None = None
) -> list[str]:
    """Every way a reference run over ``copy`` with the arguments ``other``
    could read or write the serving pools ``live`` (TXG-QUAL-003): a copy
    tensor, or a tensor argument, whose storage overlaps a serving pool's."""
    spans = [(name, _storage_span(t)) for name, t in live.flat()]

    def overlaps(t: torch.Tensor) -> list[str]:
        device, start, end = _storage_span(t)
        return [
            name
            for name, (d, s, e) in spans
            if d == device and start < e and s < end and end > start
        ]

    problems = []
    for name, tensor in copy.flat():
        hit = overlaps(tensor)
        if hit:
            problems.append(f"the reference copy {name} shares storage with {hit}")
    for key, value in (other or {}).items():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, torch.Tensor) and (hit := overlaps(item)):
                problems.append(
                    f"the reference argument {key} shares storage with {hit}"
                )
    return problems


def exactness_extent(final_blocks: int | None, max_requests: int | None) -> int | None:
    """The pages a detached copy holds (TXG-QUAL-003, TXG-QUAL-007):
    ``txgcap_qual.copy_extent`` of the mode's most requests when
    ``max_requests`` is given (block 0 and one page per request), else
    ``final_blocks`` (``None``: every page of the views). The precondition
    is a final committed block count of at least that extent: a smaller
    ``final_blocks`` raises, naming both counts, before the run's first
    step; the copy never grows to the whole cache instead. (For ``qm``,
    core's scheduler already refuses a final count below the admission limit
    plus one at one block per request.)"""
    from support import txgcap_qual as q

    if max_requests is None:
        return final_blocks
    extent = q.copy_extent(max_requests)
    if final_blocks is not None and int(final_blocks) < extent:
        raise ValueError(
            f"the final committed block count {final_blocks} is below the copy "
            f"extent {extent} (block 0 and {max_requests} requests); the run "
            "cannot copy every page its requests can write"
        )
    return extent


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


class TransactionMonitor:
    """See the module docstring. ``install`` patches; ``uninstall``
    restores. ``calls`` holds the current step's transaction calls."""

    def __init__(
        self,
        model: Any,
        *,
        compare: bool = False,
        check_off: bool = False,
        final_blocks: int | None = None,
        max_requests: int | None = None,
        memory: bool = False,
    ) -> None:
        import importlib

        self.model = model
        #: The final committed block count (TXG-QUAL-003).
        self.final_blocks = final_blocks
        #: The most requests the run's mode holds (``txgcap_qual``'s
        #: ``mode_max_requests``), or ``None``.
        self.max_requests = max_requests
        #: The pages each detached copy holds, ``0 … copy_extent - 1`` (every
        #: page of the views when ``None``): the one value that sets the
        #: copy's extent (:func:`exactness_extent`).
        self.copy_extent = exactness_extent(final_blocks, max_requests)
        #: The committed pages above the copy's extent and their digests at
        #: the run's start (:meth:`watch_untouched`).
        self._untouched: tuple[tx.Pools, dict[str, str]] | None = None
        #: Record the allocator around each copy (TXG-QUAL-029).
        self.memory = memory
        #: ``{"copy_bytes", "memory_reserved_before", ...}`` per copy.
        self.copy_readings: list[dict[str, int]] = []
        self.advance = importlib.import_module(ADVANCE_MODULE)
        self.binding = getattr(model, BINDING_ATTRIBUTE, None)
        self.compare = compare
        #: Also compare every option-off call the forward makes with the
        #: option-off reference on a pool copy (a check of the oracle).
        self.check_off = check_off
        self.force_off = False
        self.cuda = next(model.parameters()).device.type == "cuda"
        self.calls: list[Call] = []
        self._real_advance = self.advance.advance_model_rows
        self._real_execute = None if self.binding is None else self.binding.execute
        self._declined: tuple[tx.Pools, Call] | None = None
        self._installed = False

    # -- installation ------------------------------------------------------------

    def install(self) -> TransactionMonitor:
        self.advance.advance_model_rows = self._advance
        if self.binding is not None:
            self.binding.execute = self._execute
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.advance.advance_model_rows = self._real_advance
        if self.binding is not None:
            with contextlib.suppress(AttributeError):
                del self.binding.execute
        self._installed = False

    def begin_step(self) -> None:
        self.calls = []
        self._declined = None

    def _untouched_pages(self) -> tuple[int, int] | None:
        """The committed pages above the copy's extent, ``(start, stop)``,
        or ``None`` when the copy holds every committed page."""
        if self.copy_extent is None or self.final_blocks is None:
            return None
        if self.copy_extent >= self.final_blocks:
            return None
        return self.copy_extent, self.final_blocks

    def watch_untouched(self, pools: tx.Pools) -> None:
        """Digest, in place, the committed pages of ``pools`` above the
        copy's extent (TXG-QUAL-003), before the run's first step."""
        pages = self._untouched_pages()
        if pages is None:
            return
        rest = pools.pages(*pages)
        self._untouched = (rest, rest.digests())

    def untouched_problems(self) -> list[str]:
        """After the run's last step: each view whose committed pages above
        the copy's extent changed since :meth:`watch_untouched`, or the
        missing digest when there are such pages and none was taken."""
        pages = self._untouched_pages()
        if pages is None:
            return []
        if self._untouched is None:
            return [f"pages {pages[0]} to {pages[1] - 1} were not digested"]
        rest, before = self._untouched
        after = rest.digests()
        return [
            f"pages {pages[0]} to {pages[1] - 1} of {name} changed during the run"
            for name in before
            if after.get(name) != before[name]
        ]

    def _live(self, kwargs: dict[str, Any]) -> tx.Pools:
        """The step's pool views narrowed to the copy's extent
        (:attr:`copy_extent`): the pages a detached copy holds and a
        comparison reads."""
        return tx.pools_of(kwargs).narrowed(self.copy_extent)

    def _copy(self, kwargs: dict[str, Any], call: Call) -> tx.Pools:
        """The detached copy of the step's pools (TXG-QUAL-003): the views
        are narrowed to :attr:`copy_extent` before any byte is read or
        allocated, so the copy holds no page at or above it. The allocator
        is read around it when ``memory`` is set (TXG-QUAL-029)."""
        live = self._live(kwargs)
        reading: dict[str, int] = {}
        if self.memory and self.cuda:
            torch.cuda.synchronize()
            reading["memory_reserved_before"] = int(torch.cuda.memory_reserved())
            reading["max_memory_reserved_before"] = int(
                torch.cuda.max_memory_reserved()
            )
        copy = live.clone()
        if self.memory:
            if self.cuda:
                torch.cuda.synchronize()
                reading["memory_reserved_after"] = int(torch.cuda.memory_reserved())
                reading["max_memory_reserved_after"] = int(
                    torch.cuda.max_memory_reserved()
                )
            reading["copy_bytes"] = live.span_bytes()
            reading["copy_allocation_bytes"] = copy.storage_bytes()
            self.copy_readings.append(reading)
        return copy

    def memory_summary(self) -> dict[str, Any]:
        """The first copy's allocator readings and the largest copy and
        reservation over the run (TXG-QUAL-029)."""
        if not self.copy_readings:
            return {}
        first = dict(self.copy_readings[0])
        first["copy_bytes"] = max(r["copy_bytes"] for r in self.copy_readings)
        for name in ("max_memory_reserved_before", "max_memory_reserved_after"):
            values = [r[name] for r in self.copy_readings if name in r]
            if values:
                first[name] = max(values)
        first["copies"] = len(self.copy_readings)
        return first

    # -- the reference -------------------------------------------------------------

    def _reference(self, core, input_ids, inputs_embeds, plan, pools, kwargs):
        """The option-off transaction on ``pools`` (a detached copy), outside
        the timer; returns ``(rows, reservation, pools, problems)``, the
        problems being every way the reference could reach the serving pools
        (:func:`detached_problems`). The reference's row plan is the step's
        with the copy's page count as its resident pool extent
        (``num_pool_blocks``, the bound the transaction checks every pool and
        block index against), so a row naming a block at or above the copy's
        extent fails the reference."""
        sink = _ReferenceSink()
        own = {k: v for k, v in kwargs.items() if k not in tx.STEP_OWNED}
        problems = detached_problems(pools, tx.pools_of(kwargs), own)
        pages = pools.first_dimensions()
        if len(pages) == 1 and (extent := next(iter(pages))) != plan.num_pool_blocks:
            plan = dataclasses.replace(plan, num_pool_blocks=extent)
        with torch.inference_mode():
            rows = self._real_advance(
                core,
                input_ids.clone(),
                inputs_embeds.clone(),
                plan,
                **pools.kwargs(),
                **own,
                commit_sink=sink,
                staging=None,
            )
        return rows, sink.last, pools, problems

    def _compare(self, call: Call, reference: Any, live: tx.Pools) -> None:
        rows, reservation, pools, problems = reference
        call.reference_rows = None if rows is None else rows.detach().clone()
        call.compared = True
        call.problems.extend(problems)
        for name, mine, theirs in (
            ("rows", call.rows, rows),
            ("status", call.status, reservation.staged),
            ("progress", call.progress, reservation.progress),
            ("queue_counts", call.queue_counts, reservation.queue_counts),
        ):
            if not tx.same_bits(mine, theirs):
                call.problems.append(f"{name} differs from the option-off transaction")
        differing = live.differences(pools)
        if differing:
            call.problems.append(f"pages differ in {differing}")

    # -- the wrappers --------------------------------------------------------------

    def _timed(self, function, core, input_ids, inputs_embeds, plan, kwargs, call):
        sink = _Sink(kwargs["commit_sink"])
        _sync(self.cuda)
        start = time.perf_counter()
        out = function(
            core, input_ids, inputs_embeds, plan, **{**kwargs, "commit_sink": sink}
        )
        _sync(self.cuda)
        call.seconds = time.perf_counter() - start
        call.population = int(input_ids.shape[0])
        call.chunk_count = int(plan.is_chunk.sum().item())
        if sink.reservations:
            ticket = sink.reservations[-1]
            call.status, call.progress = ticket.staged, ticket.progress
            call.queue_counts = ticket.queue_counts
        return out, sink

    def _execute(self, core, input_ids, inputs_embeds, plan, **kwargs):
        if self.force_off:
            return None
        call = Call(kind="execute")
        self.calls.append(call)
        before = self._copy(kwargs, call) if self.compare else None
        if self.compare:
            call.plan = plan
            call.input_ids = input_ids.clone()
            call.inputs_embeds = inputs_embeds.clone()
        out, sink = self._timed(
            self._real_execute, core, input_ids, inputs_embeds, plan, kwargs, call
        )
        call.covered = out is not None
        if out is None:
            if sink.reservations:
                call.problems.append("a declined step made a reservation")
            if before is not None:
                if before.differences(self._live(kwargs)):
                    call.problems.append("a declined step changed resident pages")
                self._declined = (before, call)
            return None
        call.rows = out.detach().clone()
        if self.compare:
            reference = self._reference(
                core, input_ids, inputs_embeds, plan, before, kwargs
            )
            self._compare(call, reference, self._live(kwargs))
        return out

    def _advance(self, core, input_ids, inputs_embeds, plan, **kwargs):
        call = Call(kind="off")
        self.calls.append(call)
        declined = self._declined
        self._declined = None
        if declined is None and self.check_off and not self.force_off:
            declined = (self._copy(kwargs, call), call)
        if declined is not None:
            call.plan = plan
            call.input_ids = input_ids.clone()
            call.inputs_embeds = inputs_embeds.clone()
        out, _ = self._timed(
            self._real_advance, core, input_ids, inputs_embeds, plan, kwargs, call
        )
        call.covered = False
        call.rows = None if out is None else out.detach().clone()
        if declined is not None:
            # A fallback: its outputs must equal the option-off transaction
            # run on a copy of the pools taken before ``execute`` declined.
            before, _ = declined
            reference = self._reference(
                core, call.input_ids, call.inputs_embeds, plan, before, kwargs
            )
            self._compare(call, reference, self._live(kwargs))
        return out


def fallback_total(model: Any) -> int:
    """The binding's fallback counter total (0 while transaction graphs are
    off)."""
    from support import txg_api

    reading = txg_api.counters(model)
    if not reading:
        return 0
    return int(sum(dict(reading["counters"]["fallbacks"]).values()))
