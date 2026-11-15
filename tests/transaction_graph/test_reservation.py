# SPDX-License-Identifier: Apache-2.0
"""Reservation disposition under faults (TXG-EXEC-003, TXG-EXEC-010,
TXG-FAIL-002, TXG-MODEL-012's rule at the code level).

Every commit reservation, a covered step's and the option-off transaction's,
is resolved by exactly one successful stage or one cancel:

* a stage that raises before it publishes the reservation to the sink's
  collection leaves it cancellable, and the step cancels it;
* a stage that raises after it published is not a successful stage either,
  and the cancel still releases what it published;
* a cancel after a successful stage does nothing;
* before a covered step's first launch, a failure cancels the held
  reservation without latching, unless that cancel raises, which latches;
* from the first launch on, a failure latches and attempts the cancel.

Covered steps run on CPU through the fake-capture seam (a fake cell's replay
computes nothing, so these tests read whether the step raised, latched and
resolved its reservation, never its outputs). The option-off transaction runs
on the ``--enforce-eager`` harness with the option off. Faults are injected
in the model's own sink (``BoundedCommitSink._stage``) or through a proxy
around it.
"""

from __future__ import annotations

import pytest
import torch
from support import model_harness as h
from support import txg_api
from support import txgcap_harness as t
from support.fake_cuda import InjectedFault

from vllm_nemotron_asr_streaming.nemotron_asr import commit_sink

pytestmark = pytest.mark.engine

ARMS = ["covered", "option-off"]


# ---------------------------------------------------------------------------
# Environments


def started(arm, on_cpu, off_cpu):
    """``(env, step, binding)``: a started environment of ``arm`` and a
    function that runs one real step; ``binding`` is ``None`` with the option
    off."""
    if arm == "covered":
        run = on_cpu(n=2)
        env, core = run.env, run.core
        core.startup()
        binding = txg_api.binding_of(env.model)
        assert binding is not None
    else:
        env = off_cpu()
        env.runner.profile_run(core_embeds=False)
        env.runner.bind_pool(h.real_bindings(h.model_spec(env.model), capacity=8))
        binding = None

    def step(req_id: str, slot: int):
        return h.call_with(
            env.runner.real_step,
            h.scheduler_output(
                new=(t.stream_unit(env, req_id),), handles={req_id: slot}
            ),
        )

    return env, step, binding


def sink_of(env):
    sink = env.model._commit_sink
    assert isinstance(sink, commit_sink.BoundedCommitSink)
    return sink


def assert_resolved(env) -> None:
    """The sink holds neither a reservation nor a staged step."""
    sink = sink_of(env)
    assert sink._reserved is None, "the reservation is still held"
    assert not sink.has_staged, "a staged step was left in the sink"


def assert_ran_as(arm, env, binding, *, replays=0, advances=0) -> None:
    """The step took the arm's path: cells replayed, or the option-off
    transaction ran."""
    if arm == "covered":
        assert env.fake.replay_count > replays, "the covered step replayed no cell"
        assert len(env.advance.calls) == advances, "the covered step fell back"
    else:
        assert len(env.advance.calls) > advances


class FaultReservation:
    """A reservation around the sink's own, with injected faults: ``stage``
    as an attribute that raises on retrieval, a ``cancel`` that raises before
    it reaches the sink."""

    def __init__(self, real, faults, log) -> None:
        self._real, self._faults, self._log = real, faults, log

    @property
    def stage(self):
        if "stage-attribute" in self._faults:
            raise InjectedFault("injected fault retrieving the stage")
        return self._stage

    def _stage(self, row_status: torch.Tensor) -> None:
        self._log.append("stage")
        self._real.stage(row_status)

    def cancel(self) -> None:
        self._log.append("cancel")
        if "cancel" in self._faults:
            raise InjectedFault("injected fault in cancel")
        self._real.cancel()


class FaultSink:
    def __init__(self, real, faults, log) -> None:
        self._real, self._faults, self._log = real, faults, log

    def reserve(self, plan):
        self._log.append("reserve")
        return FaultReservation(self._real.reserve(plan), self._faults, self._log)


def inject_sink(arm, env, binding, monkeypatch, faults) -> list[str]:
    """Put a :class:`FaultSink` around the step's commit sink."""
    log: list[str] = []
    if arm == "covered":
        execute = binding.execute

        def with_faults(core, input_ids, inputs_embeds, plan, **transaction):
            transaction["commit_sink"] = FaultSink(
                transaction["commit_sink"], faults, log
            )
            return execute(core, input_ids, inputs_embeds, plan, **transaction)

        monkeypatch.setattr(binding, "execute", with_faults)
    else:

        def before(kwargs):
            kwargs["commit_sink"] = FaultSink(kwargs["commit_sink"], faults, log)

        env.advance.before = before
    return log


def fail_stage(monkeypatch, when: str) -> list[str]:
    """Make the sink's stage raise ``before`` it writes anything or ``after``
    it published the step to the sink's collection."""
    calls: list[str] = []
    original = commit_sink.BoundedCommitSink._stage

    def stage(self, plan, row_status):
        calls.append(when)
        if when == "after":
            original(self, plan, row_status)
            assert self.has_staged, "the stage did not publish"
        raise InjectedFault(f"injected fault {when} the stage published")

    monkeypatch.setattr(commit_sink.BoundedCommitSink, "_stage", stage)
    return calls


def injected(error) -> bool:
    return any(isinstance(e, InjectedFault) for e in t.chain(error))


# ---------------------------------------------------------------------------
# The ticket


# @spec TXG-EXEC-010
def test_a_ticket_is_done_only_after_its_stage_returned():
    """``CommitTicket`` alone, over a stub sink: a stage that raises leaves
    the ticket cancellable (one release); after a successful stage a cancel
    releases nothing; a second stage or cancel does nothing."""

    class Sink:
        def __init__(self):
            self.staged, self.released, self.fail = 0, 0, False

        def _stage(self, plan, row_status):
            if self.fail:
                raise InjectedFault("stage")
            self.staged += 1

        def _release(self):
            self.released += 1

    sink = Sink()
    sink.fail = True
    ticket = commit_sink.CommitTicket(sink, plan=None)
    with pytest.raises(InjectedFault):
        ticket.stage(torch.zeros(1, dtype=torch.int32))
    ticket.cancel()
    ticket.cancel()
    assert (sink.staged, sink.released) == (0, 1), "a failed stage was not cancellable"
    ticket.stage(torch.zeros(1, dtype=torch.int32))
    assert sink.staged == 0, "a cancelled ticket staged"

    sink = Sink()
    ticket = commit_sink.CommitTicket(sink, plan=None)
    ticket.stage(torch.zeros(1, dtype=torch.int32))
    ticket.cancel()
    ticket.stage(torch.zeros(1, dtype=torch.int32))
    assert (sink.staged, sink.released) == (1, 0), (
        "a cancel after a successful stage released the staged step"
    )

    # A stage that raised, then succeeds on a retry, is one successful stage.
    sink = Sink()
    sink.fail = True
    ticket = commit_sink.CommitTicket(sink, plan=None)
    with pytest.raises(InjectedFault):
        ticket.stage(torch.zeros(1, dtype=torch.int32))
    sink.fail = False
    ticket.stage(torch.zeros(1, dtype=torch.int32))
    ticket.cancel()
    assert (sink.staged, sink.released) == (1, 0)


# ---------------------------------------------------------------------------
# The stage raises


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("arm", ARMS)
# @spec TXG-EXEC-010, TXG-FAIL-002
def test_a_stage_that_raises_is_cancelled_and_propagates(
    on_cpu, off_cpu, monkeypatch, arm, when
):
    env, step, binding = started(arm, on_cpu, off_cpu)
    replays = env.fake.replay_count if arm == "covered" else 0
    advances = len(env.advance.calls)
    spy = t.SinkSpy().install(monkeypatch)
    calls = fail_stage(monkeypatch, when)
    error = step("s1", 1)
    assert error is not None and injected(error), error
    assert calls == [when], "the stage did not run exactly once"
    assert_ran_as(arm, env, binding, replays=replays, advances=advances)
    assert spy.log.count("reserve") == 1 and spy.log.count("stage") == 1, spy.log
    assert spy.log.count("cancel") == 1, f"the reservation was not cancelled: {spy.log}"
    assert spy.log.index("stage") < spy.log.index("cancel"), spy.log
    assert_resolved(env)
    if arm == "covered":
        assert binding.failed is True, "a failure in the commit stage did not latch"
        later = step("s2", 2)
        assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


@pytest.mark.parametrize("arm", ARMS)
# @spec TXG-EXEC-010
def test_a_cancel_after_a_successful_stage_is_a_no_op(
    on_cpu, off_cpu, monkeypatch, arm
):
    env, step, binding = started(arm, on_cpu, off_cpu)
    observed = []
    original = commit_sink.CommitTicket.stage

    def stage_then_cancel(self, row_status):
        original(self, row_status)
        self.cancel()
        observed.append((self._sink.has_staged, self._sink._reserved is None))

    monkeypatch.setattr(commit_sink.CommitTicket, "stage", stage_then_cancel)
    commits = len(env.runner.commit_log)
    assert step("s1", 1) is None
    assert observed == [(True, True)], (
        "the cancel after the stage released the staged step"
    )
    assert len(env.runner.commit_log) == commits + 1, "the step was not collected"
    assert_resolved(env)
    if arm == "covered":
        assert binding.failed is False


# ---------------------------------------------------------------------------
# Before the first launch (covered) and before the scatters (option off)


@pytest.mark.parametrize("arm", ARMS)
# @spec TXG-EXEC-003, TXG-EXEC-010
def test_a_failure_holding_the_reservation_before_any_write_cancels_it(
    on_cpu, off_cpu, monkeypatch, arm
):
    """Retrieving the reservation's stage raises: nothing was launched or
    scattered, the reservation is cancelled once, the failure propagates and
    the binding is not latched."""
    env, step, binding = started(arm, on_cpu, off_cpu)
    replays = env.fake.replay_count if arm == "covered" else 0
    log = inject_sink(arm, env, binding, monkeypatch, {"stage-attribute"})
    before = t.slots_at_step_start(env.runner.bindings, (1,))
    error = step("s1", 1)
    assert error is not None and injected(error), error
    assert log == ["reserve", "cancel"], log
    assert_resolved(env)
    assert torch.equal(h.slot_bytes(env.runner.bindings, 1), before[1]), (
        "a page changed before the failure"
    )
    if arm == "covered":
        assert env.fake.replay_count == replays, "a cell was launched"
        assert binding.failed is False, "a failure before the first launch latched"


@pytest.mark.parametrize("arm", ARMS)
# @spec TXG-EXEC-003
def test_a_cancel_that_raises_before_any_write_propagates_and_latches_the_binding(
    on_cpu, off_cpu, monkeypatch, arm
):
    """The cancel of the held reservation itself raises: the reservation is
    unresolved, the cancel's failure propagates, and a covered step's binding
    latches (every later step then needs a fresh engine)."""
    env, step, binding = started(arm, on_cpu, off_cpu)
    replays = env.fake.replay_count if arm == "covered" else 0
    log = inject_sink(arm, env, binding, monkeypatch, {"stage-attribute", "cancel"})
    error = step("s1", 1)
    assert error is not None, "the failing cancel was swallowed"
    assert "injected fault in cancel" in t.text(error), t.text(error)
    assert log == ["reserve", "cancel"], log
    assert sink_of(env)._reserved is not None, "the positive control: unresolved"
    if arm == "covered":
        assert env.fake.replay_count == replays, "a cell was launched"
        assert binding.failed is True, "a failed cancel before the launch did not latch"
        later = step("s2", 2)
        assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


# @spec TXG-EXEC-010
def test_an_option_off_cancel_that_raises_after_a_failed_stage_propagates_chained(
    on_cpu, off_cpu, monkeypatch
):
    """The option-off stage raises before it publishes, and the cancel that
    follows raises too: the cancel's failure propagates (worker-fatal; there
    is no binding to latch), with the stage's failure chained to it."""
    env, step, _binding = started("option-off", on_cpu, off_cpu)
    log = inject_sink("option-off", env, None, monkeypatch, {"cancel"})
    fail_stage(monkeypatch, "before")
    error = step("s1", 1)
    assert error is not None, "the failing cancel was swallowed"
    cancel_fault = next(
        e for e in t.chain(error) if "injected fault in cancel" in str(e)
    )
    chained = [str(e) for e in t.chain(cancel_fault)[1:]]
    assert any("before the stage published" in m for m in chained), chained
    assert log == ["reserve", "stage", "cancel"], log
    assert sink_of(env)._reserved is not None, "the positive control: unresolved"


# ---------------------------------------------------------------------------
# From the first launch on (covered)


# @spec TXG-FAIL-002
def test_a_cancel_that_raises_after_a_failed_launch_keeps_the_original_failure(
    on_cpu, off_cpu, monkeypatch
):
    env, step, binding = started("covered", on_cpu, off_cpu)
    log = inject_sink("covered", env, binding, monkeypatch, {"cancel"})
    env.fake.fail["replay"] = env.fake.replay_count
    advances = len(env.advance.calls)
    error = step("s1", 1)
    assert error is not None and "injected fault in replay" in t.text(error), error
    assert log == ["reserve", "cancel"], log
    assert binding.failed is True
    assert len(env.advance.calls) == advances, "a latched step ran option-off"


# @spec TXG-FAIL-002, TXG-EXEC-001
def test_a_failed_launch_cancels_once_and_counts_no_replay(
    on_cpu, off_cpu, monkeypatch
):
    """The replay counters are advanced after the stage: a step whose launch
    failed counts no replay and no covered step."""
    env, step, binding = started("covered", on_cpu, off_cpu)
    spy = t.SinkSpy().install(monkeypatch)
    before = (dict(t.counter(env, "replays")), dict(t.counter(env, "covered_steps")))
    env.fake.fail["replay"] = env.fake.replay_count
    error = step("s1", 1)
    assert error is not None and injected(error), error
    assert spy.log.count("reserve") == 1 and spy.log.count("cancel") == 1, spy.log
    assert "stage" not in spy.log, spy.log
    assert_resolved(env)
    after = (dict(t.counter(env, "replays")), dict(t.counter(env, "covered_steps")))
    assert after == before, "a failed step was counted"
    assert binding.failed is True


# @spec TXG-EXEC-001
def test_a_covered_step_counts_its_replays_after_the_stage(
    on_cpu, off_cpu, monkeypatch
):
    env, step, binding = started("covered", on_cpu, off_cpu)
    at_stage = []
    original = commit_sink.CommitTicket.stage

    def stage(self, row_status):
        at_stage.append(sum(t.counter(env, "replays").values()))
        original(self, row_status)

    monkeypatch.setattr(commit_sink.CommitTicket, "stage", stage)
    before = sum(t.counter(env, "replays").values())
    assert step("s1", 1) is None
    assert at_stage == [before], "the replay counter moved before the stage"
    assert sum(t.counter(env, "replays").values()) == before + 1
