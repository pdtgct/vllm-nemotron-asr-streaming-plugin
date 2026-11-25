# SPDX-License-Identifier: Apache-2.0
"""Whole-file rows through the real model and model state, step by step.

The model, its ``ModelState``, its state layer and the row transaction
(scatter, commit sink and collection included) run as core's model runner
drives them (``support.model_harness``): on CPU with the option-off
transaction (``engine`` tier), and on CUDA with transaction graphs on under
the restricted startup through core's capture hooks in core's order
(``support.core_order``; ``gpu`` tier, the same test bodies).

Observation points (all call-through wrappers; nothing in the step is
replaced unless a test says so):

* ``advance.advance_model_rows``: the row plan (``is_file``, ``is_chunk``,
  freshness), the keyword arguments and each row's carrier header;
* the model's ``_emission_adapter``: each row's role, its CHUNK burst and
  decision, and the *oracle count* ``QUEUE_LEN - QUEUE_HEAD`` of the
  proposed label book the transaction commits (FILE-QC preamble). Under
  transaction graphs a replayed step calls no Python adapter, and the oracle
  is absent;
* ``BoundedCommitSink.reserve``: the commit plan's ``queue_counts``;
* ``NemotronASRModelState.plan_step``: each file row's planned action;
* the state's ``file_buffers.stage``: the samples staged per step;
* ``Runner.queue_count_log`` / ``commit_log``: the model's collection, and
  ``Runner.failed_log``: what the model state's ``take_failed_requests``
  returned after the step (REG-MS-003, REG-MS-004, FILE-QC-012).

Layout addition: the file progress ``state._file_progress[req]``
holds ``queue_count`` (int), ``queue_count_unknown`` (bool) and
``admission_generation`` (int or ``None``) (transcription design,
Implementation notes: "the file progress there holds the count, an unknown
marker and the recorded admission generation"). ``progress`` below is the one
place that reads them.

Fault injection uses only the model's inputs (the token the runner feeds back,
``Runner._sampled``), its collection (``Runner.collect_hook``) and, for a
chunk that queues no label, the CHUNK bucket's burst (``force_empty_bursts``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch
from support import model_harness as h
from support.core_order import CoreOrder

#: CPU, option-off transaction, 560 ms (lookahead 6).
CPU = {"right_context": 6, "device": "cpu", "txg": False}
#: CUDA, transaction graphs on, 160 ms (lookahead 1, TXG-CFG-006).
TXG_ON = {"right_context": 1, "device": "cuda", "txg": True}

ARMS = [
    pytest.param(CPU, id="graphs-off", marks=pytest.mark.engine),
    pytest.param(TXG_ON, id="graphs-on", marks=pytest.mark.gpu),
]


def progress(state: Any, req_id: str) -> tuple[int, bool, int | None] | None:
    """``(queue_count, queue_count_unknown, admission_generation)`` of a
    file's progress (see the layout addition above), ``None`` without one."""
    entry = state._file_progress.get(req_id)
    if entry is None:
        return None
    return (
        int(entry.queue_count),
        bool(entry.queue_count_unknown),
        entry.admission_generation,
    )


@dataclass
class Step:
    """What one real step showed."""

    scheduled: tuple[str, ...]
    emitted: dict[str, int] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    collected: tuple = ()
    plan: Any = None  # the RowPlan
    advance_kwargs: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, list[float]] = field(default_factory=dict)
    roles: dict[str, int] = field(default_factory=dict)
    bursts: dict[str, int] = field(default_factory=dict)  # CHUNK rows: burst length
    first_label: dict[str, int] = field(default_factory=dict)
    oracle: dict[str, int] | None = None  # proposed-book count per row
    reserved_counts: list[Any] = field(default_factory=list)
    actions: dict[str, Any] = field(default_factory=dict)
    deadlines: dict[str, Any] = field(default_factory=dict)  # planner's StepRow
    planned_progress: dict[str, Any] = field(default_factory=dict)
    staged: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    bucket_rows: list[int] = field(default_factory=list)
    progress_after: dict[str, Any] = field(default_factory=dict)

    def fresh(self, req_id: str) -> bool:
        index = self.plan.request_ids.index(req_id)
        return not bool(self.plan.has_initial_states_p[index])

    def in_plan(self, req_id: str) -> int:
        return self.plan.request_ids.index(req_id)


class Env:
    """The harness runner plus the observation wrappers of the module
    docstring; ``step`` runs one real step and returns its ``Step``."""

    def __init__(
        self,
        monkeypatch: Any,
        tmp_path: Any,
        *,
        right_context: int,
        device: str,
        txg: bool,
        max_num_seqs: int = 4,
    ) -> None:
        from vllm_nemotron_asr_streaming.lookahead import geometry_index
        from vllm_nemotron_asr_streaming.nemotron_asr import advance
        from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
            BoundedCommitSink,
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
            NemotronASRModelState,
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
            QUEUE_HEAD,
            QUEUE_LEN,
        )

        self.monkeypatch = monkeypatch
        self.right_context = right_context
        self.geometry = geometry_index(right_context)
        self.chunk = 1280 * (right_context + 1)
        overrides: dict[str, Any] = {}
        if txg:
            overrides = {
                "encoder_execution_arm": "eager-graphed",
                "decode_dispatch_arm": "dense-graphed",
                "transaction_graph_populations": [max_num_seqs],
            }
        card = h.write_runnable_card(tmp_path / "card")
        vllm_config = h.build_vllm_config(
            card,
            right_context=right_context,
            max_num_seqs=max_num_seqs,
            overrides=overrides,
            device=device,
            enforce_eager=not txg,
        )
        torch.manual_seed(0)
        self.model = h.build_model(vllm_config)
        self.state = h.build_model_state(vllm_config, self.model)
        self.runner = h.Runner(vllm_config, self.model, self.state)
        core = CoreOrder(self.runner, device=device, spy=False)
        if txg:
            core.startup(warmup=True)
        else:
            # The eager runner: no capture hook; real steps before post_warmup.
            core.startup(verify=False)
        config = self.model.config
        self.blank = int(config.num_asr_labels)
        self.eos = int(config.eos_token_id)
        self.stream_carrier = int(config.audio_chunk_token_id)
        self.file_carrier = int(config.file_carrier_token_id)
        self.continuation = int(config.continuation_token_id)
        self.slots: dict[str, int] = {}
        self._next_slot = 1
        self.positions: dict[str, int] = {}
        self.steps: list[Step] = []
        self._current: Step | None = None
        self._units: dict[str, int] = {}

        env = self
        real_advance = advance.advance_model_rows

        def advance_spy(core, input_ids, inputs_embeds, plan, **kwargs):
            step = env._current
            if step is not None:
                step.plan = plan
                step.advance_kwargs = dict(kwargs)
                for index, req_id in enumerate(plan.request_ids):
                    step.headers[req_id] = inputs_embeds[index, :7].tolist()
            return real_advance(core, input_ids, inputs_embeds, plan, **kwargs)

        monkeypatch.setattr(advance, "advance_model_rows", advance_spy)

        real_bucket = advance.advance_chunk_bucket

        def bucket_spy(core, env_rows, state, **kwargs):
            if env._current is not None:
                env._current.bucket_rows.append(int(env_rows.shape[0]))
            result = real_bucket(core, env_rows, state, **kwargs)
            rows = env._empty_burst(env) if env._empty_burst is not None else None
            if rows:
                result = _without_burst(result, rows)
            return result

        self._empty_burst = None
        monkeypatch.setattr(advance, "advance_chunk_bucket", bucket_spy)

        real_adapter = self.model._emission_adapter

        def adapter_spy(result, context):
            projection = real_adapter(result, context)
            step = env._current
            if step is not None and step.plan is not None:
                ids = step.plan.request_ids
                roles = context.roles.tolist()
                book = projection.book
                counts = (book[:, QUEUE_LEN] - book[:, QUEUE_HEAD]).long().tolist()
                step.oracle = {}
                for index, req_id in enumerate(ids):
                    step.roles[req_id] = int(roles[index])
                    step.oracle[req_id] = int(counts[index])
                for j, row in enumerate(context.chunk_rows.tolist()):
                    req_id = ids[row]
                    step.bursts[req_id] = int(result.token_lengths[j])
                    if int(result.token_ids.shape[1]):
                        step.first_label[req_id] = int(result.token_ids[j, 0])
            return projection

        self.model._emission_adapter = adapter_spy

        real_reserve = BoundedCommitSink.reserve

        def reserve_spy(sink, plan):
            if env._current is not None:
                counts = getattr(plan, "queue_counts", None)
                env._current.reserved_counts.append(
                    (
                        plan.request_ids,
                        None if counts is None else counts.detach().clone().cpu(),
                        None if counts is None else (counts.dtype, counts.device),
                    )
                )
            return real_reserve(sink, plan)

        monkeypatch.setattr(BoundedCommitSink, "reserve", reserve_spy)

        real_plan_step = NemotronASRModelState.plan_step

        def plan_step_spy(state, *args, **kwargs):
            step = env._current
            if step is not None:
                for req_id in list(state._file_progress):
                    try:
                        step.planned_progress[req_id] = progress(state, req_id)
                    except AttributeError:
                        step.planned_progress[req_id] = "missing"
            plan = real_plan_step(state, *args, **kwargs)
            if step is not None:
                for row in plan.rows:
                    if row.file_plan is not None:
                        step.actions[row.request_id] = (
                            "FAILED" if row.failed else row.file_plan.action.name
                        )
                        step.deadlines[row.request_id] = row.deadline
            return plan

        monkeypatch.setattr(NemotronASRModelState, "plan_step", plan_step_spy)

        buffers = self.state.file_buffers
        real_stage = buffers.stage

        def stage_spy(request_id, start, length):
            if env._current is not None:
                env._current.staged.setdefault(request_id, []).append(
                    (int(start), int(length))
                )
            return real_stage(request_id, start, length)

        buffers.stage = stage_spy

    # -- requests -------------------------------------------------------------

    def file(
        self,
        req_id: str,
        n_samples: int,
        *,
        seed: int = 0,
        geometry: int | None = None,
        silent: bool = False,
    ) -> Any:
        samples = (
            np.zeros(n_samples, np.float32)
            if silent
            else np.random.default_rng(seed)
            .standard_normal(n_samples)
            .astype(np.float32)
            * 0.1
        )
        return h.new_request(
            req_id,
            self.file_carrier,
            samples=samples,
            header=[self.geometry if geometry is None else geometry, 0, n_samples],
        )

    def stream(self, req_id: str, *, seed: int = 1) -> Any:
        """A stream's next unit at the server's geometry (never final: the
        session parks once the unit has drained, and its next unit arrives
        as a new request delivery on the same slot)."""
        sequence = self._units.get(req_id, -1) + 1
        self._units[req_id] = sequence
        return h.new_request(
            req_id,
            self.stream_carrier,
            audio=h.stream_envelope(
                geometry=self.geometry, sequence=sequence, seed=seed
            ),
            identifier=f"uuid-{req_id}-{sequence}",
        )

    def is_stream(self, req_id: str) -> bool:
        return req_id in self._units

    # -- steps ----------------------------------------------------------------

    def step(
        self,
        *,
        new: tuple = (),
        cached: tuple[str, ...] = (),
        finished: tuple[str, ...] = (),
        inject: dict[str, int] | None = None,
    ) -> Step:
        for req_id in finished:
            self.slots.pop(req_id, None)
            self.positions.pop(req_id, None)
        for request in new:
            if request.req_id not in self.slots:
                self.slots[request.req_id] = self._next_slot
                self._next_slot += 1
            self.positions[request.req_id] = 0
        for req_id, token in (inject or {}).items():
            self.runner._sampled[req_id] = int(token)
        scheduled = tuple(r.req_id for r in new) + tuple(cached)
        output = h.scheduler_output(
            new=new,
            cached=cached,
            finished=finished,
            handles={r: self.slots[r] for r in scheduled},
            computed={r: self.positions[r] for r in cached},
        )
        step = Step(scheduled=scheduled)
        self._current = step
        asked = len(self.runner.failed_log)
        try:
            self.runner.real_step(output)
        finally:
            self._current = None
        step.emitted = {r: int(self.runner._sampled[r]) for r in scheduled}
        if self.runner.commit_log:
            step.statuses = dict(self.runner.commit_log[-1][0])
        if self.runner.queue_count_log:
            step.collected = tuple(self.runner.queue_count_log[-1])
        if len(self.runner.failed_log) > asked:
            step.failed = set(self.runner.failed_log[-1] or ())
        for req_id in scheduled:
            self.positions[req_id] += 1
            if req_id in self.state._file_progress:
                try:
                    step.progress_after[req_id] = progress(self.state, req_id)
                except AttributeError:
                    step.progress_after[req_id] = "missing"
        self.steps.append(step)
        return step

    def count_of(self, step: Step, req_id: str) -> int | None:
        """The file's count after ``step``: the oracle, else the collected
        count (graphs on)."""
        if step.oracle is not None and req_id in step.oracle:
            return step.oracle[req_id]
        for count in step.collected:
            if count.request_id == req_id:
                return int(count.count)
        return None

    def next(
        self,
        live: list[str],
        *,
        inject: dict[str, int] | None = None,
        finished: tuple[str, ...] = (),
    ) -> Step:
        """The next step of the ``live`` requests: a file that emitted EOS
        finishes (and leaves ``live``), a stream that parked receives its
        next unit, every other request runs a cached step."""
        last = self.steps[-1]
        done = tuple(
            r for r in live if not self.is_stream(r) and last.emitted.get(r) == self.eos
        )
        live[:] = [r for r in live if r not in done]
        turns = tuple(
            self.stream(r)
            for r in live
            if self.is_stream(r) and last.emitted.get(r) == self.eos
        )
        turned = {t.req_id for t in turns}
        cached = tuple(r for r in live if r not in turned)
        return self.step(
            new=turns, cached=cached, finished=done + tuple(finished), inject=inject
        )

    def run(self, first: tuple, *, limit: int = 400) -> list[Step]:
        """Run ``first`` as new requests, then ``next`` steps until every
        file among them emitted EOS; returns the steps from the first."""
        mark = len(self.steps)
        self.step(new=first)
        live = [r.req_id for r in first]
        files = {r for r in live if not self.is_stream(r)}
        ended: set[str] = set()
        for _ in range(limit):
            last = self.steps[-1]
            ended |= {r for r in files if last.emitted.get(r) == self.eos}
            if ended == files:
                return self.steps[mark:]
            self.next(live)
        raise AssertionError(f"files {sorted(files - ended)} never emitted EOS")


def _without_burst(bucket: Any, rows: set[int]) -> Any:
    """A CHUNK bucket result whose ``rows`` (bucket positions) queued no
    label: a chunk that decoded only blanks."""
    result = bucket.result
    keep = torch.ones(result.token_lengths.shape[0], dtype=torch.bool)
    keep[sorted(rows)] = False
    keep = keep.to(result.token_lengths.device)
    empty = dataclasses.replace(
        result,
        token_ids=result.token_ids * keep.unsqueeze(1).to(result.token_ids.dtype),
        token_lengths=result.token_lengths * keep.to(result.token_lengths.dtype),
    )
    return dataclasses.replace(bucket, result=empty)


@pytest.fixture
def build(monkeypatch, tmp_path, request):
    """``build(arm)`` → ``Env``; the CUDA arm needs the distributed groups."""

    def make(arm: dict[str, Any], **kwargs: Any) -> Env:
        if arm["device"] == "cuda":
            request.getfixturevalue("single_process_dist")
        return Env(monkeypatch, tmp_path, **arm, **kwargs)

    return make


def _roles():
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        ROLE_CHUNK,
        ROLE_FLUSH,
        ROLE_REPLAY,
    )

    return ROLE_CHUNK, ROLE_REPLAY, ROLE_FLUSH


def _status_bits():
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        ROW_STATUS_ECHO_MISMATCH,
        ROW_STATUS_QUEUE_NOT_DRAINED,
        ROW_STATUS_ROLE_MISMATCH,
    )

    return (
        ROW_STATUS_ECHO_MISMATCH,
        ROW_STATUS_QUEUE_NOT_DRAINED,
        ROW_STATUS_ROLE_MISMATCH,
    )


# ===========================================================================
# FILE-ROW: file rows on the stream roles, under the file mask


# @spec FILE-ROW-001, FILE-ROW-002, FILE-ROW-005, FILE-ROW-012, FILE-ENG-001
@pytest.mark.engine
@pytest.mark.parametrize("tail", [1000, 0])
def test_file_steps_run_on_the_stream_roles_with_the_file_bit_set(build, tail):
    env = build(CPU)
    chunk_r, replay_r, flush_r = _roles()
    _, not_drained, _ = _status_bits()
    n = 2 * env.chunk + tail
    steps = env.run((env.file("f1", n),))
    actions = [s.actions["f1"] for s in steps]
    assert actions.count("CHUNK") == 2 and actions.count("TAIL") == 1
    assert actions[-1] == "END"
    tokens = [s.emitted["f1"] for s in steps]
    assert tokens[-1] == env.eos and tokens.count(env.eos) == 1  # FILE-ROW-005
    for index, step in enumerate(steps):
        row = step.in_plan("f1")
        # FILE-ROW-001: the file bit is set on every step the request runs,
        # and its rows carry no unit deadline
        assert step.plan.is_file.tolist()[row] is True
        assert step.deadlines["f1"] is None
        assert step.statuses["f1"] == 0
        action, role = step.actions["f1"], step.roles["f1"]
        staged = step.staged.get("f1", [])
        if action == "DRAIN":
            assert role == replay_r and not bool(step.plan.is_chunk[row])
            assert staged == []
        elif action in ("CHUNK", "TAIL"):
            assert role == chunk_r and bool(step.plan.is_chunk[row])
            assert len(staged) <= 1 and sum(n for _, n in staged) <= env.chunk
            # FILE-ROW-012: an armed echo of the last label does not make a
            # drained file CHUNK row undrained
            assert not step.statuses["f1"] & not_drained
        else:
            previous = steps[index - 1].emitted["f1"]
            assert role == (replay_r if previous < env.blank else flush_r)
            assert step.emitted["f1"] == env.eos
        # FILE-ROW-002: the drain condition is the count the previous step left
        if index:
            count = env.count_of(steps[index - 1], "f1")
            assert (action == "DRAIN") == (count > 0), (index, action, count)


# @spec FILE-ROW-001, FILE-ROW-002, FILE-ROW-004, FILE-ROW-006
@pytest.mark.engine
def test_an_empty_file_runs_a_fresh_final_tail_then_a_flush_end(build):
    env = build(CPU)
    chunk_r, _, flush_r = _roles()
    steps = env.run((env.file("f1", 0),))
    assert [s.actions["f1"] for s in steps] == ["TAIL", "END"]
    assert [s.emitted["f1"] for s in steps] == [env.continuation, env.eos]
    assert [s.roles["f1"] for s in steps] == [chunk_r, flush_r]
    assert steps[0].fresh("f1")
    assert all(s.plan.is_file.tolist() == [True] for s in steps)


# @spec FILE-ROW-003, FILE-ROW-004
@pytest.mark.engine
@pytest.mark.parametrize("n_chunks, tail", [(1, 500), (0, 500)])
def test_advancing_rows_are_fresh_first_and_flag_only_the_final_tail(
    build, n_chunks, tail
):
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        ENV_FINAL_TAIL,
        ENV_VALID_SAMPLES,
    )

    env = build(CPU)
    steps = env.run((env.file("f1", n_chunks * env.chunk + tail),))
    advancing = [s for s in steps if s.actions["f1"] in ("CHUNK", "TAIL")]
    assert [s.actions["f1"] for s in advancing] == ["CHUNK"] * n_chunks + ["TAIL"]
    assert advancing[0].fresh("f1")
    assert not any(s.fresh("f1") for s in steps[1:])
    for step in advancing:
        header = step.headers["f1"]
        final = step.actions["f1"] == "TAIL"
        assert header[ENV_FINAL_TAIL] == (1.0 if final else 0.0)
        assert header[ENV_VALID_SAMPLES] == (tail if final else env.chunk)
        # the first queued token, or the continuation token if none
        burst = step.bursts["f1"]
        expected = step.first_label["f1"] if burst else env.continuation
        assert step.emitted["f1"] == expected


# @spec FILE-ROW-003, FILE-ROW-014
@pytest.mark.engine
def test_a_chunk_that_queues_nothing_emits_the_continuation_token(build):
    env = build(CPU)
    # the first regular chunk queues no label (its bucket's burst is emptied)
    env._empty_burst = lambda e: {0} if len(e.steps) == 0 else None
    steps = env.run((env.file("f1", env.chunk + 700),))
    assert steps[0].actions["f1"] == "CHUNK"
    assert steps[0].emitted["f1"] == env.continuation
    assert env.count_of(steps[0], "f1") == 0
    # the next row is the tail: a CHUNK row whose scheduled token is the
    # continuation token with no echo pending, which is no role mismatch
    assert steps[1].actions["f1"] == "TAIL"
    assert steps[1].statuses["f1"] == 0


# @spec FILE-ROW-009, FILE-ROW-010
@pytest.mark.engine
@pytest.mark.parametrize("kind", ["stream-only", "mixed"])
def test_the_transaction_gets_the_file_mask_and_token_ids_on_every_step(
    build, monkeypatch, kind
):
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    env = build(CPU)
    staged: list[tuple[str, list]] = []
    real_stage = advance.HostStaging.stage

    def stage(self, slot_name, vector, device):
        staged.append((slot_name, vector.tolist()))
        return real_stage(self, slot_name, vector, device)

    monkeypatch.setattr(advance.HostStaging, "stage", stage)
    checked: list[Any] = []
    real_check = advance._mrv1_projection_invariant_rows

    def check(*args, **kwargs):
        checked.append(kwargs.get("continuation_id"))
        return real_check(*args, **kwargs)

    monkeypatch.setattr(advance, "_mrv1_projection_invariant_rows", check)
    first = (env.stream("s1"),) + ((env.file("f1", 3000),) if kind == "mixed" else ())
    step = env.step(new=first)
    expected_mask = [r == "f1" for r in step.plan.request_ids]
    assert step.plan.is_file.tolist() == expected_mask
    assert step.advance_kwargs["file_carrier_id"] == env.file_carrier
    assert step.advance_kwargs["continuation_id"] == env.continuation
    # the mask reaches the device on every step, all false without a file row
    masks = [vector for name, vector in staged if name == "is_file"]
    assert masks == [expected_mask]
    # the emission check receives the continuation ID on every step
    assert checked == [env.continuation]


# @spec FILE-ROW-001, FILE-ENG-003
@pytest.mark.engine
def test_file_and_stream_rows_share_the_one_chunk_bucket_and_the_echo_roles(build):
    env = build(CPU)
    chunk_r, replay_r, _ = _roles()
    first = env.step(new=(env.file("f1", 3 * env.chunk), env.stream("s1")))
    # one CHUNK bucket, both rows, in step position order
    assert first.bucket_rows == [2]
    assert first.plan.request_ids == ("f1", "s1")
    assert first.plan.is_chunk.tolist() == [True, True]
    assert first.plan.is_file.tolist() == [True, False]
    assert first.roles == {"f1": chunk_r, "s1": chunk_r}
    second = env.step(cached=("f1", "s1"))
    if env.count_of(first, "f1") and env.count_of(first, "s1"):
        # both drain: REPLAY rows of the same transaction
        assert second.roles == {"f1": replay_r, "s1": replay_r}
        assert second.bucket_rows == []
    assert second.statuses == {"f1": 0, "s1": 0}


# ===========================================================================
# FILE-QC: the queue-count transport and its application


# @spec FILE-QC-001, FILE-QC-004, FILE-QC-005, FILE-QC-006, FILE-QC-007
# @spec FILE-QC-013
@pytest.mark.parametrize("arm", ARMS)
def test_queue_counts_travel_from_the_transaction_to_the_file_progress(build, arm):
    env = build(arm)
    steps = env.run((env.file("f1", 3 * env.chunk + 900), env.stream("s1")))
    # FILE-QC-007: taken with count 0 (and no generation yet); the first step
    # plans with it
    assert steps[0].planned_progress["f1"] == (0, False, None)
    assert steps[0].actions["f1"] == "CHUNK"
    generation = None
    for step in steps:
        ((ids, counts, kind),) = step.reserved_counts
        # FILE-QC-001: every row of every reserved step has a count
        assert counts is not None, "the commit plan carries no queue_counts"
        assert kind[0] == torch.int32 and kind[1].type == arm["device"]
        assert tuple(counts.shape) == (len(ids),)
        if step.oracle is not None:
            assert dict(zip(ids, counts.tolist())) == step.oracle
        # FILE-QC-005: one QueueCount per clean report
        collected = {c.request_id: c for c in step.collected}
        assert set(collected) == {r for r in ids if step.statuses[r] == 0}
        for req_id, count in collected.items():
            assert count.count == int(counts[ids.index(req_id)])
        if "f1" not in ids:
            continue
        count = collected["f1"]
        # FILE-QC-013: the first clean report's generation is recorded, and
        # every later count carries it
        generation = count.admission_generation if generation is None else generation
        assert count.admission_generation == generation
        # FILE-QC-006: the accepted count is the file's count
        if step.emitted["f1"] != env.eos:
            assert step.progress_after["f1"] == (count.count, False, generation)


# @spec FILE-QC-011
@pytest.mark.engine
def test_planning_reads_no_label_book_and_changes_no_count(build, monkeypatch):
    """No operation that reads the label book's or label queue's memory runs
    while the model state plans a step (``prepare_inputs``), and no file's
    count changes there. View operations (which read nothing) are not
    counted."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    from vllm_nemotron_asr_streaming.nemotron_asr import model_state_v2

    env = build(CPU)
    spans: list[tuple[int, int]] = []
    real_project = model_state_v2.project_nemotron_state_pools

    def span(t: torch.Tensor) -> tuple[int, int]:
        extent = sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n)
        start = t.data_ptr()
        return start, start + (extent + 1) * t.element_size()

    def project(*args, **kwargs):
        pools = real_project(*args, **kwargs)
        spans[:] = [span(pools.replay_book), span(pools.replay_queue)]
        return pools

    monkeypatch.setattr(model_state_v2, "project_nemotron_state_pools", project)

    class BookReads(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.hits: list[str] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if not func.is_view:
                for leaf in tree_leaves((args, kwargs or {})):
                    if isinstance(leaf, torch.Tensor) and leaf.numel():
                        lo, hi = span(leaf)
                        if any(lo < b and a < hi for a, b in spans):
                            self.hits.append(str(func))
                            break
            return func(*args, **(kwargs or {}))

    state = env.state
    real_prepare = state.prepare_inputs
    around: list[tuple[Any, Any]] = []
    hits: list[str] = []

    def prepare_inputs(*args, **kwargs):
        before = progress(state, "f1") if "f1" in state._file_progress else None
        mode = BookReads()
        try:
            with mode:
                return real_prepare(*args, **kwargs)
        finally:
            hits.extend(mode.hits)
            after = progress(state, "f1") if "f1" in state._file_progress else None
            around.append((before, after))

    state.prepare_inputs = prepare_inputs
    env.run((env.file("f1", 2 * env.chunk + 300),))
    assert spans, "the step bound no state pools"
    assert hits == []
    assert around and all(before == after for before, after in around)


def _upcoming(env: Env, req_id: str, chunks: int) -> str:
    """The row a file runs next, by the step rule, from what the steps
    showed: its count, the chunks it advanced and whether its tail ran."""
    count = env.count_of(env.steps[-1], req_id)
    actions = [s.actions.get(req_id) for s in env.steps]
    if count:
        return "drain"
    if actions.count("CHUNK") < chunks:
        return "chunk"
    if "TAIL" not in actions:
        return "tail"
    return "end"


def _advance_to(env: Env, live: list[str], want: str, chunks: int) -> None:
    """Run clean steps until ``f1``'s next row is ``want``; for ``chunk``,
    ``tail``, ``drain`` and ``end`` its last token is a label (the echo is
    armed), for ``chunk-after-continuation`` the continuation token."""
    for _ in range(300):
        label_last = env.steps[-1].emitted["f1"] < env.blank
        kind = want.removesuffix("-after-continuation")
        if _upcoming(env, "f1", chunks) == kind and label_last == (kind == want):
            return
        assert env.steps[-1].emitted["f1"] != env.eos, f"f1 ended before {want}"
        step = env.next(live)
        assert step.failed == set() and all(v == 0 for v in step.statuses.values())
    raise AssertionError(f"no {want} row for f1")


REJECTIONS = [
    # (row f1 runs, token fed back to it, status bit it must get)
    ("drain", "other-label", "echo"),  # a rejected DRAIN row
    ("chunk", "other-label", "echo"),  # FILE-ROW-011, a rejected regular chunk
    ("tail", "other-label", "echo"),  # a rejected final tail
    ("drain", "stream-carrier", "role"),  # FILE-ROW-014, any role
    ("chunk", "stream-carrier", "role"),
    ("end", "stream-carrier", "role"),
    ("chunk-after-continuation", "other-label", "role"),  # FILE-ROW-014, CHUNK
]


# @spec FILE-QC-008, FILE-QC-012, FILE-ROW-011, FILE-ROW-014, FILE-ENG-003
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("row, token, bit", REJECTIONS)
def test_a_rejected_file_row_fails_alone_and_keeps_its_count(
    build, arm, row, token, bit
):
    """f1's row is rejected beside a clean file row (f2) and a clean stream
    row (s1, a session whose units keep arriving); f2 then completes token
    for token as it does alone."""
    if row == "chunk-after-continuation" and arm["txg"]:
        pytest.skip(
            "forcing an empty burst wraps the eager CHUNK bucket, which a "
            "replayed cell does not call; the graphs-off arm covers the case"
        )
    echo, _, role = _status_bits()
    n_other = 4 * 8960 + 77
    reference = build(arm)
    other = reference.file("f2", n_other, seed=5)
    alone = [s.emitted["f2"] for s in reference.run((other,))]

    env = build(arm)
    if row == "chunk-after-continuation":
        # f1's first chunk (bucket row 0 of the first step) queues no label
        env._empty_burst = lambda e: {0} if len(e.steps) == 0 else None
    env.step(
        new=(
            env.file("f1", 2 * env.chunk + 300, seed=3),
            env.file("f2", n_other, seed=5),
            env.stream("s1"),
        )
    )
    live = ["f1", "f2", "s1"]
    _advance_to(env, live, row, chunks=2)
    before = env.steps[-1].progress_after["f1"]
    assert isinstance(before, tuple), f"no file progress count ({before!r})"
    last = env.steps[-1].emitted["f1"]
    fed = env.stream_carrier if token == "stream-carrier" else (last + 1) % env.blank
    hit = env.next(live, inject={"f1": fed})
    assert set(hit.scheduled) >= {"f1", "f2", "s1"}
    status = hit.statuses["f1"]
    assert status & (echo if bit == "echo" else role), status
    # FILE-QC-012: that file request alone fails, from postprocessing
    assert hit.failed == {"f1"}
    assert hit.statuses["f2"] == 0 and hit.statuses["s1"] == 0
    # FILE-QC-008: its count is the one before the rejected step
    assert hit.progress_after["f1"] == before
    # the other file and the stream are unaffected
    live.remove("f1")
    finished: tuple[str, ...] = ("f1",)
    while env.steps[-1].emitted.get("f2") != env.eos:
        step = env.next(live, finished=finished)
        finished = ()
        assert step.failed == set() and all(v == 0 for v in step.statuses.values())
    assert [s.emitted["f2"] for s in env.steps if "f2" in s.scheduled] == alone


NO_OUTCOME = ["dropped", "stale-generation", "unscheduled-request"]


# @spec FILE-QC-006, FILE-QC-009, FILE-QC-010, FILE-QC-012
@pytest.mark.engine
@pytest.mark.parametrize("fault", NO_OUTCOME)
def test_a_file_row_with_no_accepted_count_fails_and_its_next_row_is_failed_unplanned(
    build, fault
):
    env = build(CPU)
    files = (
        env.file("f1", 3 * env.chunk, seed=3),
        env.file("f2", 2 * env.chunk, seed=5),
    )
    env.step(new=files)
    env.step(cached=("f1", "f2"))
    before = env.steps[-1].progress_after["f1"]
    assert isinstance(before, tuple), f"no file progress count ({before!r})"

    def hook(collection):
        env.runner.collect_hook = None  # one step only
        counts = []
        for count in collection.queue_counts:
            if count.request_id != "f1":
                counts.append(count)
            elif fault == "stale-generation":
                counts.append(
                    dataclasses.replace(
                        count, admission_generation=count.admission_generation + 1000
                    )
                )
            elif fault == "unscheduled-request":
                counts.append(dataclasses.replace(count, request_id="not-scheduled"))
        return h.replace_collection(collection, queue_counts=tuple(counts))

    env.runner.collect_hook = hook
    hit = env.step(cached=("f1", "f2"))
    assert hit.statuses == {"f1": 0, "f2": 0}  # a clean report, no usable count
    # FILE-QC-012: no count accepted → the file fails alone
    assert hit.failed == {"f1"}
    # FILE-QC-009: the retained number is kept, its validity unknown
    count, unknown, generation = hit.progress_after["f1"]
    assert unknown is True
    assert generation == before[2]
    assert count == before[0]
    # FILE-QC-010 (backstop): scheduled again, the row fails before its file
    # action is planned; nothing is staged and the step does not fail
    again = env.step(cached=("f1", "f2"))
    assert again.failed == {"f1"}
    assert "f1" not in again.staged
    assert again.actions.get("f1") in (None, "FAILED")
    assert again.statuses.get("f2") == 0


# @spec FILE-QC-006, FILE-QC-013
@pytest.mark.engine
@pytest.mark.parametrize("when", ["first-step", "later-step"])
def test_a_count_from_an_earlier_incarnation_of_the_request_id_is_not_applied(
    build, when
):
    env = build(CPU)
    first = env.run((env.file("f1", env.chunk + 100, seed=3),))
    old = {c.admission_generation for s in first for c in s.collected}
    assert len(old) == 1
    (old_generation,) = old
    reused = env.file("f1", 2 * env.chunk + 100, seed=4)
    forged: list[Any] = []

    def hook(collection):
        real = [c for c in collection.queue_counts if c.request_id == "f1"]
        stale = dataclasses.replace(
            real[0], admission_generation=old_generation, count=77
        )
        forged.append(stale)
        env.runner.collect_hook = None
        # the stale count comes first, beside the request's own
        return h.replace_collection(
            collection, queue_counts=(stale, *collection.queue_counts)
        )

    if when == "first-step":
        env.runner.collect_hook = hook
    step = env.step(new=(reused,), finished=("f1",))
    if when == "later-step":
        env.runner.collect_hook = hook
        step = env.step(cached=("f1",))
    (own,) = [c for c in step.collected if c.request_id == "f1"]
    assert own.admission_generation != old_generation
    assert forged
    assert step.failed == set()
    assert step.progress_after["f1"] == (own.count, False, own.admission_generation)


# @spec FILE-QC-013
@pytest.mark.engine
def test_no_generation_is_recorded_from_another_report_at_the_first_step(build):
    env = build(CPU)

    def hook(collection):
        env.runner.collect_hook = None
        counts = tuple(
            dataclasses.replace(c, admission_generation=c.admission_generation + 1000)
            if c.request_id == "f1"
            else c
            for c in collection.queue_counts
        )
        return h.replace_collection(collection, queue_counts=counts)

    env.runner.collect_hook = hook
    step = env.step(new=(env.file("f1", 2 * env.chunk, seed=3),))
    count, unknown, generation = step.progress_after["f1"]
    assert generation is None and unknown is True and count == 0
    assert step.failed == {"f1"}


# ===========================================================================
# FILE-PROMPT-007: a foreign-geometry header that reaches the model state


# @spec FILE-PROMPT-007, FILE-ENG-003
@pytest.mark.engine
def test_a_foreign_geometry_file_row_fails_at_planning_outside_the_chunk_bucket(
    build,
):
    env = build(CPU)
    foreign = (env.geometry + 1) % 5
    step = env.step(
        new=(
            env.file("bad", 2 * env.chunk, geometry=foreign),
            env.file("good", 2 * env.chunk, seed=2),
            env.stream("s1"),
        )
    )
    assert step.failed == {"bad"}
    assert step.statuses.get("good") == 0 and step.statuses.get("s1") == 0
    row = step.in_plan("bad")
    assert not bool(step.plan.is_chunk[row])
    assert "bad" not in step.staged
    # the CHUNK bucket formed without it
    assert step.bucket_rows == [2]
