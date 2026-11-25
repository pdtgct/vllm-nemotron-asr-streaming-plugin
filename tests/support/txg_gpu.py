# SPDX-License-Identifier: Apache-2.0
"""Transaction graphs on CUDA: the model and model state with
``transaction_graph_populations`` on, taken through core's capture lifecycle
in-process, and driven by scheduled real steps (``gpu`` tests).

``build_option_on`` builds the runnable synthetic card's model on CUDA under
the hook configuration (no ``--enforce-eager``, ``cudagraph_mode`` NONE,
compilation off, the default segment arms), with the option set from
configuration on, so core's and the plugin's configuration hooks see it. The
environment's ``core`` (``support.txgcap_harness.TxgCoreOrder``, a
``support.core_order.CoreOrder`` whose bindings set core's state-group layout
accessor) takes it through core's startup order with ``graph_capture``'s
stream and core's graph pools; ``build_option_on`` runs that startup unless
``startup=False``. The configuration meets the restricted startup
(TXG-CFG-017 to TXG-CFG-019): synchronous scheduling, ``cudagraph_mode`` and
compilation mode NONE, and ``num_gpu_blocks_override`` the state pools' block
count, at least the admission limit plus one.

``GpuEnv.compare`` installs a spy on the binding's ``execute`` (see
``support.txg_api``): every covered step is also run as the option-off
transaction on a copy of the pools taken before ``execute``, and the escaped
rows, staged status, progress, queue counts and every page are compared
bit for bit (TXG-EQ-001, TXG-QUAL-003's procedure at the test tier).

``Driver`` schedules streams and files explicitly, step by step, as core's
scheduler would: a parked stream receives its next unit as a new request,
a stream or file with queued work is scheduled as a cached request, and a
finished request is reported finished.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from support import model_harness as h
from support import txg_api as api
from support import txg_transaction as tx
from support.txg_fakes import RecordingSink
from support.txgcap_harness import TxgCoreOrder

KEY = api.KEY


# ---------------------------------------------------------------------------
# Comparison of covered steps with the option-off transaction


class ProxyReservation:
    def __init__(self, sink: ProxySink, real: Any, plan: Any) -> None:
        self._sink = sink
        self._real = real
        self.plan = plan
        self.staged = None
        self.progress = None
        self.queue_counts = None
        self.cancelled = 0

    def stage(self, row_status: torch.Tensor) -> None:
        self._sink.events.append(("stage", torch.cuda.current_stream()))
        self._real.stage(row_status)
        self.staged = row_status.detach().clone()
        progress = getattr(self.plan, "progress", None)
        counts = getattr(self.plan, "queue_counts", None)
        self.progress = None if progress is None else progress.detach().clone()
        self.queue_counts = None if counts is None else counts.detach().clone()

    def cancel(self) -> None:
        self._sink.events.append(("cancel", None))
        self.cancelled += 1
        self._real.cancel()


class ProxySink:
    """Wraps the model's commit sink; records reservations and stages."""

    def __init__(self, real: Any, events: list | None = None) -> None:
        self._real = real
        self.events = events if events is not None else []
        self.reservations: list[ProxyReservation] = []
        self.fail_reserve: BaseException | None = None

    def reserve(self, plan: Any) -> ProxyReservation:
        self.events.append(("reserve", torch.cuda.current_stream()))
        if self.fail_reserve is not None:
            raise self.fail_reserve
        reservation = ProxyReservation(self, self._real.reserve(plan), plan)
        self.reservations.append(reservation)
        return reservation


@dataclasses.dataclass
class StepRecord:
    plan: Any
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    covered: bool
    rows: torch.Tensor | None = None
    status: torch.Tensor | None = None
    progress: torch.Tensor | None = None
    problems: list[str] = dataclasses.field(default_factory=list)
    sink: ProxySink | None = None
    error: BaseException | None = None


def option_off(
    core, input_ids, inputs_embeds, plan, pools: tx.Pools, kwargs
) -> tx.Outcome:
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    sink = RecordingSink()
    own = {k: v for k, v in kwargs.items() if k not in tx.STEP_OWNED}
    with torch.inference_mode():
        rows = advance.advance_model_rows(
            core,
            input_ids.clone(),
            inputs_embeds.clone(),
            plan,
            **pools.kwargs(),
            **own,
            commit_sink=sink,
            staging=None,
        )
    last = sink.last
    return tx.Outcome(
        rows=rows,
        status=last.staged,
        progress=last.progress,
        queue_counts=last.queue_counts,
        pools=pools,
        sink=sink,
    )


# ---------------------------------------------------------------------------
# The environment


class GpuEnv:
    """An option-on (or, from ``option_off``, option-off) model and model
    state on CUDA, its runner and its ``CoreOrder`` (``core``)."""

    device = torch.device("cuda")

    def __init__(
        self,
        *,
        vllm_config,
        model,
        state,
        runner,
        lookahead: int,
        n: int,
        capacity: int,
        max_num_seqs: int | None = None,
        events: h.Events | None = None,
        advance: h.AdvanceSpy | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.model = model
        self.state = state
        self.runner = runner
        self.lookahead = lookahead
        self.geometry = api.GEOMETRY_BY_LOOKAHEAD[lookahead]
        self.n = n
        self.capacity = capacity
        self.max_num_seqs = max_num_seqs
        self.events = events if events is not None else h.Events()
        self.advance = advance
        self.config = model.config
        # A harness-structure check builds an environment without a runner.
        self.core = (
            None
            if runner is None
            else TxgCoreOrder(runner, device="cuda", capacity=capacity, spy=False)
        )
        self.records: list[StepRecord] = []
        self._patches = pytest.MonkeyPatch()

    @property
    def bindings(self) -> Any:
        """The serving pool (``None`` before startup binds one)."""
        return self.core.serving_bindings

    def option_off(self) -> GpuEnv:
        """``B_hook``: the same configuration and model with the option absent,
        a new model state, runner and ``CoreOrder`` (not started)."""
        setattr(self.vllm_config.model_config.hf_config, KEY, [])
        state = h.build_model_state(self.vllm_config, self.model)
        return GpuEnv(
            vllm_config=self.vllm_config,
            model=self.model,
            state=state,
            runner=h.Runner(self.vllm_config, self.model, state),
            lookahead=self.lookahead,
            n=self.n,
            capacity=self.capacity,
            max_num_seqs=self.max_num_seqs,
            events=self.events,
            advance=self.advance,
        )

    def shutdown(self) -> None:
        self._patches.undo()
        with contextlib.suppress(Exception):
            self.state.shutdown()

    # -- the binding ---------------------------------------------------------

    @property
    def binding(self) -> Any:
        binding = api.binding_of(self.model)
        assert binding is not None, "transaction graphs are on but no binding exists"
        return binding

    def counters(self) -> dict:
        reading = api.counters(self.model)
        return dict(reading.get("counters", {}))

    def pools(self) -> tx.Pools:
        return tx.pools_of(self._last_kwargs)

    def compare(self, *, mutate=None, events: list | None = None) -> list[StepRecord]:
        """Spy on ``execute``: record every call; compare each covered step
        with the option-off transaction on a pool copy. ``mutate(kwargs,
        plan)`` may return replacement ``(kwargs, plan)`` for ``execute``;
        ``events`` is the log the proxy sink appends ``reserve`` and ``stage``
        to."""
        binding = self.binding
        real = binding.execute

        def spy(core, input_ids, inputs_embeds, plan, **kwargs):
            if mutate is not None:
                kwargs, plan = mutate(dict(kwargs), plan)
            self._last_kwargs = kwargs
            before = tx.pools_of(kwargs).clone()
            sink = ProxySink(kwargs["commit_sink"], events)
            record = StepRecord(
                plan=plan,
                input_ids=input_ids.clone(),
                inputs_embeds=inputs_embeds.clone(),
                covered=False,
                sink=sink,
            )
            self.records.append(record)
            try:
                out = real(
                    core,
                    input_ids,
                    inputs_embeds,
                    plan,
                    **{**kwargs, "commit_sink": sink},
                )
            except BaseException as exc:
                record.error = exc
                raise
            record.covered = out is not None
            if out is None:
                if before.differences(tx.pools_of(kwargs)):
                    record.problems.append("a declined step changed resident pages")
                if sink.reservations:
                    record.problems.append("a declined step made a reservation")
                return out
            ticket = sink.reservations[-1]
            off = option_off(core, input_ids, inputs_embeds, plan, before, kwargs)
            record.rows, record.status = out.detach().clone(), ticket.staged
            record.progress = ticket.progress
            for name, mine, theirs in (
                ("rows", out, off.rows),
                ("status", ticket.staged, off.status),
                ("progress", ticket.progress, off.progress),
                ("queue_counts", ticket.queue_counts, off.queue_counts),
            ):
                if not tx.same_bits(mine, theirs):
                    record.problems.append(
                        f"{name} differs from the option-off transaction"
                    )
            differing = tx.pools_of(kwargs).differences(off.pools)
            if differing:
                record.problems.append(f"pages differ in {differing}")
            return out

        self._patches.setattr(binding, "execute", spy)
        return self.records

    def spy_replays(self, events: list) -> None:
        """Log ``("replay", key, stream)`` for every cell replay."""
        for key, cell in self.binding.cells.items():
            graph = cell.graph
            if graph is None:
                continue
            real = graph.replay

            def replay(__real=real, __key=key):
                events.append(("replay", __key, torch.cuda.current_stream()))
                return __real()

            self._patches.setattr(graph, "replay", replay)

    # -- weights ---------------------------------------------------------------

    @contextlib.contextmanager
    def silent(self) -> Iterator[None]:
        """Make blank dominate every joint decision (in place, so every
        captured graph keeps its addresses): every chunk is silent."""
        final = self.model.core.joint.joint_net[-1]
        blank = int(self.model.core.blank_id)
        original = final.bias[blank].detach().clone()
        with torch.no_grad():
            final.bias[blank] += 1.0e4
        try:
            yield
        finally:
            with torch.no_grad():
                final.bias[blank].copy_(original)


def set_block_count(vllm_config: Any, blocks: int) -> None:
    """``--num-gpu-blocks-override blocks``, the restricted startup's block
    count (TXG-CFG-005, TXG-CFG-019), set on the built configuration without
    re-running its validators."""
    cache = vllm_config.cache_config
    try:
        cache.num_gpu_blocks_override = blocks
    except Exception:  # noqa: BLE001 - validated dataclass
        object.__setattr__(cache, "num_gpu_blocks_override", blocks)


def build_option_on(
    tmp_path: Path,
    *,
    lookahead: int = 0,
    n: int = 4,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int = 64,
    max_num_active_seqs: int | None = None,
    populations: Sequence[int] | None = None,
    capacity: int | None = None,
    startup: bool = True,
    profile: bool = True,
    verify: bool = True,
    overrides: dict[str, Any] | None = None,
    monkeypatch: Any = None,
) -> GpuEnv:
    """See the module docstring. ``max_num_seqs`` defaults to ``n``
    (``N = P_max``); ``max_num_active_seqs`` sets the admission limit
    (``--max-num-active-seqs``; unset, it is ``max_num_seqs``). The weights
    are seeded, so every environment of a test has the same model.
    ``startup`` runs ``core.startup(profiling=profile, verify=verify)``."""
    patches = monkeypatch or pytest.MonkeyPatch()
    events = h.Events()
    advance = h.AdvanceSpy(events).install(patches)
    max_num_seqs = max_num_seqs or n
    card = h.write_runnable_card(tmp_path / "card")
    vllm_config = h.build_vllm_config(
        card,
        right_context=lookahead,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        overrides={
            "encoder_execution_arm": "eager-graphed",
            "decode_dispatch_arm": "dense-graphed",
            KEY: list(populations or [n]),
            **(overrides or {}),
        },
        device="cuda",
        enforce_eager=False,
    )
    if max_num_active_seqs is not None:
        scheduler = vllm_config.scheduler_config
        try:
            scheduler.max_num_active_seqs = max_num_active_seqs
        except Exception:  # noqa: BLE001 - validated dataclass
            object.__setattr__(scheduler, "max_num_active_seqs", max_num_active_seqs)
    capacity = capacity or max_num_seqs + 4
    assert capacity >= max_num_seqs + 1, "the pools hold the admission limit plus one"
    set_block_count(vllm_config, capacity)
    torch.manual_seed(0)
    model = h.build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    env = GpuEnv(
        vllm_config=vllm_config,
        model=model,
        state=state,
        runner=h.Runner(vllm_config, model, state),
        lookahead=lookahead,
        n=n,
        capacity=capacity,
        max_num_seqs=max_num_seqs,
        events=events,
        advance=advance,
    )
    if monkeypatch is None:
        env._patches = patches
    if startup:
        env.core.startup(profiling=profile, verify=verify)
    return env


#: The plugin package's directory, as a frame's file name contains it.
PLUGIN_PACKAGE = os.sep + "vllm_nemotron_asr_streaming" + os.sep


def spy_stream_creation(monkeypatch) -> list[str]:
    """Replace ``torch.cuda.Stream`` so that each new device stream the model
    creates is recorded (by the creating plugin file) in the returned list.

    Not recorded: a ``Stream(stream_id=…)`` wrapper of an existing stream,
    which ``torch.cuda.current_stream`` builds on every call (and
    ``torch.cuda.graph`` on every capture), and a stream created with no frame
    of the plugin's package on the stack (core's capture context creates the
    capture stream it passes to the hook; a test creates its own). The
    replacement returns plain ``torch.cuda.Stream`` objects."""
    created: list[str] = []
    real = torch.cuda.Stream

    class Recording(real):
        def __new__(cls, *args, **kwargs):
            if "stream_id" not in kwargs:
                frame = sys._getframe(1)
                while frame is not None:
                    if PLUGIN_PACKAGE in frame.f_code.co_filename:
                        created.append(frame.f_code.co_filename)
                        break
                    frame = frame.f_back
            return real.__new__(real, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Stream", Recording)
    return created


def memcpy_counts(profile) -> dict[str, int]:
    """Memcpy events of a ``torch.profiler`` run by direction."""
    counts = {"HtoD": 0, "DtoH": 0, "DtoD": 0}
    for event in profile.events():
        name = event.name
        for direction in counts:
            if "Memcpy" in name and direction in name:
                counts[direction] += 1
    return counts


# ---------------------------------------------------------------------------
# Scheduling


def envelope(config, **fields: Any) -> torch.Tensor:
    """One stream unit's carrier row (``txg_transaction.stream_envelope``)."""
    del config
    return tx.stream_envelope(**fields)


@dataclasses.dataclass
class Stream:
    req_id: str
    units: list[dict]
    prompt: int = 0
    delivered: int = 0
    position: int = 0
    done: bool = False


@dataclasses.dataclass
class File:
    req_id: str
    samples: int
    position: int = 0
    started: bool = False
    done: bool = False


class Driver:
    """Explicit scheduling of streams and files over a ``GpuEnv``."""

    def __init__(self, env: GpuEnv) -> None:
        self.env = env
        self.streams: dict[str, Stream] = {}
        self.files: dict[str, File] = {}
        self.handles: dict[str, tuple[int, int]] = {}
        self._generation: dict[int, int] = {}
        self._free = list(range(env.capacity - 1, 0, -1))
        self.step_seq = 0
        self.finished: list[str] = []
        self.zero_sample_files: set[str] = set()
        self._transient: list[str] = []

    @property
    def park(self) -> int:
        return int(self.env.config.eos_token_id)

    def _slot(self, req_id: str) -> tuple[int, int]:
        if req_id not in self.handles:
            slot = self._free.pop()
            generation = self._generation.get(slot, 0) + 1
            self._generation[slot] = generation
            self.handles[req_id] = (slot, generation)
        return self.handles[req_id]

    def add_stream(self, req_id: str, units: list[dict], prompt: int = 0) -> Stream:
        self.streams[req_id] = Stream(req_id, list(units), prompt)
        return self.streams[req_id]

    def add_file(self, req_id: str, samples: int) -> File:
        self.files[req_id] = File(req_id, samples)
        if samples == 0:
            self.zero_sample_files.add(req_id)
        return self.files[req_id]

    def parked(self, req_id: str) -> bool:
        return self.env.runner._sampled.get(req_id) == self.park

    def _new_unit(self, s: Stream):
        unit = s.units[s.delivered]
        row = envelope(
            self.env.config,
            geometry=self.env.geometry,
            lookahead=self.env.lookahead,
            sequence=s.delivered,
            prompt=s.prompt,
            **unit,
        )
        s.delivered += 1
        s.position = 1
        return h.new_request(
            s.req_id,
            int(self.env.config.audio_chunk_token_id),
            audio=row,
            identifier=f"uuid-{s.req_id}-{s.delivered}",
        )

    def _new_file(self, f: File):
        rng = np.random.default_rng(len(f.req_id))
        samples = (rng.standard_normal(f.samples) * 0.1).astype(np.float32)
        f.started = True
        f.position = 1
        return h.new_request(
            f.req_id,
            int(self.env.config.file_carrier_token_id),
            samples=samples,
            header=[self.env.geometry, 0, f.samples],
        )

    def step(self, only: Sequence[str] | None = None, extra_new: Sequence[Any] = ()):
        """One step over every live request (or only those in ``only``)."""
        new, cached, finished = list(extra_new), [], list(self._transient)
        self._transient = [r.req_id for r in extra_new]
        for s in self.streams.values():
            if s.done or (only is not None and s.req_id not in only):
                continue
            if s.delivered == 0 or self.parked(s.req_id):
                if s.delivered < len(s.units):
                    new.append(self._new_unit(s))
                else:
                    s.done = True
                    finished.append(s.req_id)
            else:
                cached.append(s.req_id)
        for f in self.files.values():
            if f.done or (only is not None and f.req_id not in only):
                continue
            if not f.started:
                new.append(self._new_file(f))
            elif self.parked(f.req_id):
                f.done = True
                finished.append(f.req_id)
            else:
                cached.append(f.req_id)
        for r in new:
            self._slot(r.req_id)
        for r in self._transient:
            self._slot(r)
        handles = {r: self.handles[r] for r in [*(x.req_id for x in new), *cached]}
        computed = {}
        for r in cached:
            item = self.streams.get(r) or self.files.get(r)
            computed[r] = item.position
            item.position += 1
        self.step_seq += 1
        output = h.scheduler_output(
            new=tuple(new),
            cached=tuple(cached),
            finished=tuple(finished),
            handles=handles,
            computed=computed,
            step_seq=self.step_seq,
        )
        for r in finished:
            slot = self.handles.pop(r, None)
            if slot is not None:
                self._free.append(slot[0])
            self.finished.append(r)
        if not new and not cached and not finished:
            return None
        # A step that schedules nothing but reports finished requests still
        # runs, as core's cleanup-only step: the model state releases the
        # requests whose slots the next step may reissue.
        before = len(self.env.runner.commit_log)
        result = self.env.runner.real_step(output)
        if len(self.env.runner.commit_log) > before:
            # A request the step failed is finished with an error by core: it
            # is not scheduled again and its slot is released next step.
            for r in self.env.runner.commit_log[-1][1]:
                item = self.streams.get(r) or self.files.get(r)
                if item is not None and not item.done:
                    item.done = True
                    self._transient.append(r)
        return result

    def until_parked(self, ids: Sequence[str], limit: int = 400) -> None:
        """Step only ``ids`` until every one of them is parked (or done)."""
        for _ in range(limit):
            waiting = [
                r
                for r in ids
                if not (self.streams.get(r) or self.files.get(r)).done
                and not self.parked(r)
            ]
            if not waiting:
                return
            self.step(only=waiting)
        raise AssertionError(f"{ids} never parked")

    def lockstep(self, ids: Sequence[str]) -> None:
        """Bring ``ids`` to the park, then deliver each one's next unit in
        one step."""
        self.until_parked(ids)
        self.step(only=ids)

    def special(self, req_id: str, token: int, *, fresh: bool = False) -> Any:
        """A new request with no audio, whose scheduled token is ``token``,
        for the parked stream ``req_id``: a non-CHUNK row, as a direct engine
        stream client sends one (the planner refuses a non-CHUNK row of a
        request it never saw a CHUNK of). With ``fresh`` the request's slot is
        reissued at its next generation, so the row is a fresh non-CHUNK row.
        The stream ends with this row (finished on the next step)."""
        stream = self.streams[req_id]
        assert not stream.done and self.parked(req_id), f"{req_id} is not parked"
        stream.done = True
        if fresh:
            slot, generation = self.handles[req_id]
            self._generation[slot] = generation + 1
            self.handles[req_id] = (slot, generation + 1)
        return h.new_request(req_id, token)

    def run(self, steps: int, **kwargs) -> None:
        for _ in range(steps):
            live = [s for s in self.streams.values() if not s.done] + [
                f for f in self.files.values() if not f.done
            ]
            if not live:
                return
            self.step(**kwargs)


# ---------------------------------------------------------------------------
# Step cases (TXG-EQ-004, TXG-EQ-005)


def decisions(rows: torch.Tensor) -> list[int]:
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    return [int(t) for t in read_decision_carrier(rows).tolist()]


def step_cases(record: StepRecord, env: GpuEnv, driver: Driver) -> set[str]:
    """The exactness cases a compared covered step exercises."""
    if not record.covered or record.rows is None:
        return set()
    plan = record.plan
    config = env.config
    chunk = plan.is_chunk.tolist()
    fresh = [not b for b in plan.has_initial_states_p.tolist()]
    file = plan.is_file.tolist() if plan.is_file.numel() else [False] * len(chunk)
    ids = record.input_ids.tolist()
    header = record.inputs_embeds[:, : tx.HEADER_SLOTS].cpu()
    final = [bool(header[i, tx.H_FINAL]) for i in range(len(chunk))]
    samples = [int(header[i, tx.H_SAMPLES]) for i in range(len(chunk))]
    out = decisions(record.rows)
    status = record.status.tolist()
    park, cont = int(config.eos_token_id), int(config.continuation_token_id)
    carrier = int(config.file_carrier_token_id)
    controls = {
        park,
        cont,
        carrier,
        int(config.audio_chunk_token_id),
        int(config.eou_token_id),
    }
    label = [t not in controls for t in ids]
    rows = range(len(chunk))
    cases = set()

    def any_(pred):
        return any(pred(i) for i in rows)

    if any_(lambda i: chunk[i] and fresh[i] and not file[i]) and any_(
        lambda i: chunk[i] and not fresh[i]
    ):
        cases.add("first chunk among continuing rows")
    if any_(lambda i: chunk[i] and fresh[i] and file[i]):
        cases.add("file first step")
    if any_(lambda i: chunk[i] and not fresh[i] and not final[i] and not file[i]):
        cases.add("regular chunk")
    if any_(lambda i: chunk[i] and file[i] and out[i] == cont):
        cases.add("silent chunk")
    if any_(lambda i: chunk[i] and not file[i] and final[i] and samples[i] > 0):
        cases.add("final tail with samples")
    if any_(lambda i: chunk[i] and not file[i] and final[i] and samples[i] == 0):
        cases.add("empty final tail")
    if any(r in driver.zero_sample_files for r in plan.request_ids):
        cases.add("zero-sample file")
    if any_(lambda i: not chunk[i] and label[i]):
        cases.add("drain step")
    phase = (
        record.progress[:, 2].tolist()
        if record.progress is not None
        else [None] * len(chunk)
    )
    flush = [not chunk[i] and phase[i] == 3 for i in rows]
    if any_(lambda i: not chunk[i] and file[i] and not flush[i] and out[i] == park):
        cases.add("end step as REPLAY")
    if any_(lambda i: file[i] and flush[i] and out[i] == park):
        cases.add("end step as FLUSH")
    if any_(lambda i: chunk[i] and file[i] and label[i]):
        cases.add("armed echo on a file CHUNK row")
    if (
        any_(lambda i: chunk[i] and file[i])
        and any_(lambda i: chunk[i] and not file[i])
        and any_(lambda i: not chunk[i] and file[i])
        and any_(lambda i: not chunk[i] and not file[i])
    ):
        cases.add("split step mixing file and stream rows in both kinds")
    if any_(lambda i: not chunk[i] and fresh[i] and status[i] != 0):
        cases.add("fresh non-CHUNK row rejected")
    if any_(lambda i: not chunk[i] and fresh[i] and status[i] == 0):
        cases.add("fresh non-CHUNK row accepted as a park echo")
    # 160 ms cases (TXG-EQ-005)
    if any_(lambda i: chunk[i] and fresh[i] and final[i] and not file[i]):
        cases.add("one-chunk session")
    if any_(lambda i: chunk[i] and not fresh[i] and final[i]) and any_(
        lambda i: chunk[i] and not fresh[i] and not final[i]
    ):
        cases.add("final tail among regular continuing rows")
    if any_(lambda i: not file[i] and flush[i]):
        cases.add("stream FLUSH")
    if any_(
        lambda i: (
            chunk[i] and final[i] and samples[i] >= tx.chunk_samples(env.lookahead)
        )
    ):
        cases.add("oversized final tail")
    return cases
