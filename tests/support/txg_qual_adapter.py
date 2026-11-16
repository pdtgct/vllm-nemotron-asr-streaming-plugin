# SPDX-License-Identifier: Apache-2.0
"""The test-only qualification adapter of TXG-QUAL-026 and the transaction
monitor of TXG-QUAL-003 and TXG-QUAL-013.

``QualAdapter`` drives the worker of a hook-captured process at the
transaction level: the model, the model state and the serving
``StateSlotBindings`` that core bound and captured over. Per step it builds
the scheduled handle mapping, calls ``begin_step`` on the runner stream,
runs the step's request cleanup (finished requests, freed encoder outputs),
calls ``activate_requests``, adds the step's new requests, prepares the
inputs (the model state calls ``bind_rows`` there), runs the forward on the
runner stream, calls ``finish_step`` and consumes the step's fenced
completion (``StateCompletionOutput.get_output``) before it returns, so no
slot is reused before the step's writes completed. It follows core's
``GPUModelRunner.execute_model`` order (``vllm/v1/worker/gpu/model_runner.py``)
and ``support.model_harness.Runner``, without that runner's recording
wrappers, so it also drives a plugin commit older than the current one.

``before_forward(projection)`` runs inside the step, after ``bind_rows`` and
before the forward; the step rule restores its snapshot there, through the
step's ``StateSlotRows.component_pools`` and only at the slots that step
authorized (TXG-QUAL-011). No replay, restore or store runs outside a step
the adapter opened.

``TransactionMonitor`` wraps the binding's ``execute`` and
``advance.advance_model_rows`` (both looked up at call time by the model's
forward). It times each transaction call from the call to
``torch.cuda.synchronize()`` after it, after a synchronization that drains
earlier work (the clock of TXG-QUAL-013), captures each call's staged status,
progress and remaining queue counts through a proxy commit sink, and, with
``compare``, runs the option-off transaction of every covered or declined
step on a layout-preserving copy of the pools taken before the call and
compares rows, status, progress, queue counts and every page bit for bit
(TXG-QUAL-003). The pool copy, the option-off reference and the comparison
are outside the timer. ``force_off`` makes ``execute`` decline without
counting, so the forward runs the option-off transaction (the step rule's
OFF legs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock

import numpy as np
import torch

from support import txg_transaction as tx

ADVANCE_MODULE = "vllm_nemotron_asr_streaming.nemotron_asr.advance"
BINDING_ATTRIBUTE = "_transaction_graph_binding"
EPOCH = "txg-qual"


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
    #: Requests the step failed (``StateCompletion.failed_req_ids``).
    failed: frozenset[str]
    hidden: torch.Tensor | None


class QualAdapter:
    """See the module docstring. ``model``, ``state`` and ``bindings`` are
    the worker's own objects."""

    def __init__(
        self, *, vllm_config: Any, model: Any, state: Any, bindings: Any
    ) -> None:
        if bindings is None:
            raise RuntimeError("the worker has no serving state bindings")
        self.vllm_config = vllm_config
        self.model = model
        self.state = state
        self.bindings = bindings
        self.device = torch.device(state.device)
        self.cuda = self.device.type == "cuda"
        self.layer = model._persistent_state_layer.prefix
        from support import model_harness as h

        self._h = h
        self.req_states = h.StepRequestStates()
        self._prompts: dict[str, list[int]] = {}
        self.sampled: dict[str, int] = {}
        self.step_seq = 0

    @classmethod
    def from_worker(cls, worker: Any) -> QualAdapter:
        runner = runner_of(worker)
        state = runner.model_state
        bindings = getattr(runner, "state_slots", None) or state.state_slots
        return cls(
            vllm_config=runner.vllm_config,
            model=state.model,
            state=state,
            bindings=bindings,
        )

    # -- scheduler output ------------------------------------------------------

    def handle(self, slot: int, generation: int) -> Any:
        from vllm.v1.kv_cache_interface import StateSlotHandle

        return StateSlotHandle(EPOCH, int(slot), int(generation))

    def scheduler_output(
        self,
        *,
        new: Sequence[Any],
        cached: Sequence[str],
        finished: Sequence[str],
        handles: dict[str, tuple[int, int]],
        computed: dict[str, int],
        free_mm_hashes: Sequence[str] = (),
    ) -> Any:
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        scheduled = {r.req_id: 1 for r in new} | {r: 1 for r in cached}
        output = SchedulerOutput(
            scheduled_new_reqs=list(new),
            scheduled_cached_reqs=CachedRequestData(
                req_ids=list(cached),
                new_token_ids=[],
                new_block_ids=[None for _ in cached],
                num_computed_tokens=[int(computed[r]) for r in cached],
                num_output_tokens=[1 for _ in cached],
            ),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={r.req_id: [0] for r in new if r.mm_features},
            finished_req_ids=set(finished),
            free_encoder_mm_hashes=list(free_mm_hashes),
            state_slot_handles={
                r: {self.layer: self.handle(*handles[r])} for r in scheduled
            },
        )
        output.state_cleanup_handles = {}
        output.state_step_seq = self.step_seq
        return output

    # -- one step ----------------------------------------------------------------

    def _main_stream(self) -> Any:
        return torch.cuda.current_stream() if self.cuda else None

    def _remove_request(self, req_id: str) -> None:
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
        handles: dict[str, tuple[int, int]],
        computed: dict[str, int] | None = None,
        order: Sequence[str] | None = None,
        free_mm_hashes: Sequence[str] = (),
        before_forward: Callable[[Any], None] | None = None,
    ) -> StepResult:
        """One real step. ``order`` is the row order (default: new requests,
        then cached ones); ``computed`` gives each cached request's
        ``num_computed_tokens``."""
        computed = dict(computed or {})
        self.step_seq += 1
        output = self.scheduler_output(
            new=new,
            cached=cached,
            finished=finished,
            handles=handles,
            computed=computed,
            free_mm_hashes=free_mm_hashes,
        )
        state, bindings, stream = self.state, self.bindings, self._main_stream()
        state.prepare_step(output, dummy_run=False, is_profile=False)
        bindings.begin_step(output, stream)
        for req_id in sorted(set(finished)):
            self._remove_request(req_id)
        for mm_hash in free_mm_hashes:
            state.encoder_cache.free_encoder_cache(mm_hash)
        bindings.activate_requests()
        for request in new:
            self._remove_request(request.req_id)
            index = self.req_states.add(request.req_id)
            state.encoder_cache.add_request(request.req_id, request.mm_features)
            state.add_request(index, request)
        if new:
            state.apply_staged_writes()
        req_ids = [r.req_id for r in new] + list(cached)
        if order is not None:
            if sorted(order) != sorted(req_ids):
                raise ValueError("the row order must name every scheduled request")
            req_ids = list(order)
        hidden = None
        if req_ids:
            batch = self._batch(output, req_ids, computed)
            state.preprocess_state(batch, (), None, None)
            attn = self._attn(batch)
            embeds = state.prepare_inputs_embeds(
                output.scheduled_encoder_inputs, batch, self.req_states
            )
            kwargs = state.prepare_inputs(batch, self.req_states)
            if before_forward is not None:
                before_forward(kwargs.get("persistent_state_projection"))
            hidden = self._forward(batch, embeds, kwargs, attn)
            logits = state.compute_sample_logits(hidden[batch.logits_indices])
            tokens = logits.argmax(dim=-1).tolist()
            for req_id, token in zip(batch.req_ids, tokens, strict=True):
                self.sampled[req_id] = int(token)
            state.postprocess_state(
                batch.idx_mapping,
                torch.ones(batch.num_reqs, dtype=torch.int32, device=self.device),
                None,
            )
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        event = (
            contextlib.nullcontext()
            if self.cuda
            else mock.patch.object(torch.cuda, "Event", _HostEvent)
        )
        with event:
            completion = bindings.finish_step(EMPTY_MODEL_RUNNER_OUTPUT, stream)
            result = completion.get_output()
        failed = frozenset(result.state_completion.failed_req_ids)
        return StepResult(
            req_ids=tuple(req_ids),
            sampled={r: self.sampled[r] for r in req_ids},
            failed=failed,
            hidden=hidden,
        )

    def _batch(self, output: Any, req_ids: list[str], computed: dict[str, int]):
        n = len(req_ids)
        batch = self._h.dummy_batch(self.state, n, n, is_padding=False)
        new_ids = {r.req_id for r in output.scheduled_new_reqs}
        for r in output.scheduled_new_reqs:
            self._prompts[r.req_id] = list(r.prefill_token_ids)
            self.sampled.pop(r.req_id, None)
        tokens = []
        positions = {}
        for req_id in req_ids:
            if req_id in new_ids:
                tokens.append(self._prompts[req_id][0])
                positions[req_id] = 0
                continue
            position = int(computed[req_id])
            prompt = self._prompts[req_id]
            tokens.append(
                prompt[position] if position < len(prompt) else self.sampled[req_id]
            )
            positions[req_id] = position
        batch.input_ids[:n] = torch.tensor(tokens, device=batch.input_ids.device)
        index = np.array(
            [self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp
        )
        return dataclasses.replace(
            batch,
            req_ids=list(req_ids),
            idx_mapping=torch.from_numpy(index).to(self.device),
            idx_mapping_np=index,
            num_scheduled_tokens=np.ones(n, dtype=np.int32),
            num_computed_tokens_np=np.array(
                [positions[r] for r in req_ids], dtype=np.int32
            ),
            prefill_len_np=np.ones(n, dtype=np.int32),
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


class _HostEvent:
    """``torch.cuda.Event`` on a host without CUDA (CPU tests only)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, *args: Any, **kwargs: Any) -> None:
        pass

    def synchronize(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Snapshots of the authorized slots (TXG-QUAL-011)


class SlotSnapshot:
    """The step rule's snapshot: taken inside the first plan step, after
    ``bind_rows`` and before its transaction, from the pages the prefix
    left; restored inside every later step by an in-place copy into the same
    pool memory on the runner stream, at the slots that step authorized."""

    def __init__(self) -> None:
        self.pages: dict[int, dict[str, torch.Tensor]] | None = None
        self.restores = 0

    def __call__(self, projection: Any) -> None:
        access = projection.state_batch.access
        pools = access.component_pools
        slots = tuple(int(s) for s in access.slot_indices_cpu)
        if self.pages is None:
            self.pages = {
                slot: {name: pool[slot].clone() for name, pool in pools.items()}
                for slot in slots
            }
            return
        if set(slots) != set(self.pages):
            raise RuntimeError("a plan step authorized other slots than the snapshot")
        for slot in slots:
            saved = self.pages[slot]
            for name, pool in pools.items():
                pool[slot].copy_(saved[name])
        self.restores += 1


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


def _sync(cuda: bool) -> None:
    if cuda:
        torch.cuda.synchronize()


class TransactionMonitor:
    """See the module docstring. ``install`` patches; ``uninstall``
    restores. ``calls`` holds the current step's transaction calls."""

    def __init__(
        self, model: Any, *, compare: bool = False, check_off: bool = False
    ) -> None:
        import importlib

        self.model = model
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

    # -- the reference -------------------------------------------------------------

    def _reference(self, core, input_ids, inputs_embeds, plan, pools, kwargs):
        """The option-off transaction on ``pools`` (a copy), outside the
        timer; returns ``(rows, reservation, pools)``."""
        sink = _ReferenceSink()
        own = {k: v for k, v in kwargs.items() if k not in tx.STEP_OWNED}
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
        return rows, sink.last, pools

    def _compare(self, call: Call, reference: Any, live: tx.Pools) -> None:
        rows, reservation, pools = reference
        call.reference_rows = None if rows is None else rows.detach().clone()
        call.compared = True
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
        before = tx.pools_of(kwargs).clone() if self.compare else None
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
                if before.differences(tx.pools_of(kwargs)):
                    call.problems.append("a declined step changed resident pages")
                self._declined = (before, call)
            return None
        call.rows = out.detach().clone()
        if self.compare:
            reference = self._reference(
                core, input_ids, inputs_embeds, plan, before, kwargs
            )
            self._compare(call, reference, tx.pools_of(kwargs))
        return out

    def _advance(self, core, input_ids, inputs_embeds, plan, **kwargs):
        call = Call(kind="off")
        self.calls.append(call)
        declined = self._declined
        self._declined = None
        if declined is None and self.check_off and not self.force_off:
            declined = (tx.pools_of(kwargs).clone(), call)
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
            self._compare(call, reference, tx.pools_of(kwargs))
        return out


def fallback_total(model: Any) -> int:
    """The binding's fallback counter total (0 while transaction graphs are
    off)."""
    from support import txg_api

    reading = txg_api.counters(model)
    if not reading:
        return 0
    return int(sum(dict(reading["counters"]["fallbacks"]).values()))
