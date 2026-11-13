# SPDX-License-Identifier: Apache-2.0
"""Lease behaviour (SESS-IN, SESS-LEASE, SESS-LOC, SESS-FLUSH, SESS-FIN,
SESS-ABORT, SESS-REL, SESS-TEAR, SESS-EXP, SESS-ERR, SESS-TXT, SESS-EP).

Every test here drives a lease through ``support.fake_engine.FakeEngine``,
which imports vLLM, so all of them are ``engine`` tier.
"""

from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest
from support.card import LOCALES
from support.fake_engine import (
    Die,
    Fail,
    FakeEngine,
    Finish,
    Park,
    Reject,
    Silent,
    Text,
    contains_key,
    refusal,
    run,
    unit_of,
    wait_until,
)

pytestmark = pytest.mark.engine

CADENCE = "80ms"
LOCALE = "en-US"
UNIT = 1280


def audio(n: int, start: int = 0) -> np.ndarray:
    return ((np.arange(start, start + n, dtype=np.float32) + 1.0) / 1e6).astype(
        np.float32
    )


async def opened(build, engine=None, *, cadence=CADENCE, locale=LOCALE, **kw):
    engine, factory = build(engine, **kw)
    lease = await factory.open(cadence=cadence, locale=locale)
    return engine, factory, lease


async def feed(lease, samples):
    accepted: list[int] = []
    result = await lease.feed(samples, on_accepted=accepted.append)
    return result, accepted


async def raised(awaitable):
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001
        return error
    return None


def failure(error, reason):
    assert error is not None, f"expected SessionFailed({reason!r})"
    assert getattr(error, "session_error", None) == "failed", repr(error)
    assert error.reason == reason, repr(error)


CALLS = {
    "feed": lambda lease: lease.feed(audio(0), on_accepted=lambda n: None),
    "update_locale": lambda lease: lease.update_locale(LOCALE),
    "flush": lambda lease: lease.flush(),
    "finish": lambda lease: lease.finish(),
    "abort": lambda lease: lease.abort(),
    "release": lambda lease: lease.release(),
}


# ------------------------------------------------------------- feed input

BAD_AUDIO = [
    pytest.param(np.zeros((2, 3), np.float32), id="2d"),
    pytest.param(np.zeros((), np.float32), id="0d"),
    pytest.param(np.zeros(10, np.int16), id="int16"),
    pytest.param(np.zeros(10, np.float64), id="float64"),
    pytest.param(np.array([0.1, np.nan], np.float32), id="nan"),
    pytest.param(np.array([np.inf, 0.1], np.float32), id="inf"),
    pytest.param(np.array([0.1, -np.inf], np.float32), id="neg-inf"),
    pytest.param([0.1, 0.2], id="list"),
]


# @spec SESS-IN-001
@pytest.mark.parametrize("bad", BAD_AUDIO)
def test_invalid_audio_raises_value_error_and_changes_nothing(build, bad):
    async def scenario():
        engine, _, lease = await opened(build)
        first, _ = await feed(lease, audio(500))
        accepted: list[int] = []
        error = await raised(lease.feed(bad, on_accepted=accepted.append))
        # State and buffered audio unchanged: 500 + 780 completes one unit
        # of exactly the good samples, in order.
        result, _ = await feed(lease, audio(780, 500))
        return engine, first, error, accepted, result

    engine, first, error, accepted, result = run(scenario())
    assert isinstance(error, ValueError)
    assert accepted == [] and first == []
    assert len(engine.inputs) == 1 and result == []
    unit = unit_of(engine.inputs[0])
    assert unit.chunk_sequence == 0 and unit.valid_samples == UNIT
    np.testing.assert_array_equal(unit.samples, audio(UNIT))


# @spec SESS-IN-001
def test_invalid_audio_error_leaves_lease_open(build):
    async def scenario():
        engine, _, lease = await opened(build)
        error = await raised(
            lease.feed(np.zeros((2, 2), np.float32), on_accepted=lambda n: None)
        )
        # Still open: flush is accepted.
        transcript = await lease.flush()
        return error, transcript

    error, transcript = run(scenario())
    assert isinstance(error, ValueError) and transcript == ""


# @spec SESS-IN-002
def test_empty_feed_is_accepted_and_sends_nothing(build):
    async def scenario():
        engine, _, lease = await opened(build)
        result, accepted = await feed(lease, np.zeros(0, np.float32))
        return engine, result, accepted

    engine, result, accepted = run(scenario())
    assert result == [] and accepted == [0]
    assert engine.inputs == []


# @spec SESS-IN-003
def test_feed_accepts_then_sends_every_completed_unit_in_order(build):
    async def scenario():
        engine = FakeEngine(default=Park(delay=0.01))
        engine, _, lease = await opened(build, engine)
        accepted: list[int] = []

        def on_accepted(count):
            engine.events.append(("accepted", count))
            accepted.append(count)

        samples = audio(UNIT * 2 + 10)
        await lease.feed(samples, on_accepted=on_accepted)
        return engine, accepted, samples

    engine, accepted, samples = run(scenario())
    assert accepted == [UNIT * 2 + 10], "exactly once, with the sample count"
    kinds = [e[0] for e in engine.events]
    assert kinds.index("accepted") < kinds.index("read")
    assert len(engine.inputs) == 2
    units = [unit_of(i) for i in engine.inputs]
    assert [u.chunk_sequence for u in units] == [0, 1]
    np.testing.assert_array_equal(
        np.concatenate([u.samples for u in units]), samples[: UNIT * 2]
    )
    assert sum(1 for e in engine.events if e[0] == "emit") == 2, (
        "feed returned only after both outcomes"
    )


# @spec SESS-IN-003
def test_remaining_samples_stay_buffered_across_feeds(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT - 1))
        assert engine.inputs == []
        await feed(lease, audio(1, UNIT - 1))
        return engine

    engine = run(scenario())
    assert len(engine.inputs) == 1


# @spec SESS-IN-004
def test_feed_returns_cumulative_text_after_each_changing_unit(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[
                Park((5,), "hel"),
                Park((), ""),
                Park((6,), "lo wor"),
                Park((7,), "ld"),
            ]
        )
        _, _, lease = await opened(build, engine)
        result, _ = await feed(lease, audio(UNIT * 4))
        return result

    assert run(scenario()) == ["hel", "hello wor", "hello world"]


# @spec SESS-IN-004
def test_feed_with_no_complete_unit_returns_at_once(build):
    async def scenario():
        engine = FakeEngine(default=Silent())
        _, _, lease = await opened(build, engine)
        return await asyncio.wait_for(feed(lease, audio(UNIT - 1)), 0.5)

    result, accepted = run(scenario())
    assert result == [] and accepted == [UNIT - 1]


# --------------------------------------------------- state table, ordering


async def lease_in_state(build, state):
    if state == "failed":
        engine = FakeEngine(outcomes=[Finish("error")])
        engine, factory, lease = await opened(build, engine)
        error = await raised(feed(lease, audio(UNIT)))
        failure(error, "abnormal_end")
        return engine, factory, lease
    engine, factory, lease = await opened(build)
    if state == "flushed":
        await lease.flush()
    elif state == "aborted":
        await lease.abort()
    elif state == "released":
        await lease.flush()
        await lease.release()
    return engine, factory, lease


ALLOWED = {
    "open": {"feed", "update_locale", "flush", "abort"},
    "flushed": {"finish", "release"},
    "failed": {"abort", "finish", "release"},
    "aborted": {"finish", "release"},
    "released": set(),
}
#: Calls that return without effect rather than raising (SESS-LEASE-002).
NO_EFFECT = {("released", "finish"), ("released", "release")}
STATE_CASES = [
    (state, call)
    for state in ALLOWED
    for call in CALLS
    if (state, call) not in NO_EFFECT
]


# @spec SESS-LEASE-001, SESS-LEASE-002
@pytest.mark.parametrize("state,call", STATE_CASES)
def test_call_allowed_exactly_in_its_states(build, state, call):
    async def scenario():
        _, _, lease = await lease_in_state(build, state)
        return await raised(CALLS[call](lease))

    error = run(scenario())
    if call in ALLOWED[state]:
        assert error is None, repr(error)
    else:
        expected = "abnormal_end" if state == "failed" else "state"
        failure(error, expected)


# @spec SESS-LEASE-001
@pytest.mark.parametrize("state", list(ALLOWED))
def test_teardown_is_allowed_in_every_state(build, state):
    async def scenario():
        _, _, lease = await lease_in_state(build, state)
        assert not inspect.iscoroutinefunction(lease.teardown)
        return lease.teardown()

    assert run(scenario()) is None


# @spec SESS-LEASE-002
@pytest.mark.parametrize("state", ["aborted", "released"])
def test_finish_after_the_task_ended_returns_without_effect(build, state):
    async def scenario():
        _, _, lease = await lease_in_state(build, state)
        first = await lease.finish()
        second = await lease.finish()
        return first, second

    assert run(scenario()) == (None, None)


# @spec SESS-LEASE-002
def test_release_after_release_returns_without_effect(build):
    async def scenario():
        engine, _, lease = await lease_in_state(build, "released")
        before = list(engine.wait_calls)
        assert await lease.release() is None
        return before, engine.wait_calls

    before, after = run(scenario())
    assert before == after, "no second retirement wait"


# @spec SESS-LEASE-003
@pytest.mark.parametrize("second", ["feed", "update_locale", "flush", "abort"])
def test_concurrent_call_raises_runtime_error_without_effect(build, second):
    async def scenario():
        engine = FakeEngine(outcomes=[Park((), "ab", delay=0.15)])
        _, _, lease = await opened(build, engine)
        first = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await wait_until(lambda: engine.inputs)
        error = await raised(CALLS[second](lease))
        result, _ = await first
        # Unchanged state: the first call completed normally and the lease
        # is still open.
        transcript = await lease.flush()
        return error, result, transcript

    error, result, transcript = run(scenario())
    assert isinstance(error, RuntimeError)
    assert not hasattr(error, "session_error")
    assert result == ["ab"] and transcript == "ab"


# @spec SESS-LEASE-003
def test_teardown_is_accepted_during_another_call(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Silent()])
        _, _, lease = await opened(build, engine)
        first = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await wait_until(lambda: engine.inputs)
        lease.teardown()  # must not raise
        outcome = await asyncio.wait_for(
            asyncio.gather(first, return_exceptions=True), 2
        )
        return outcome[0]

    outcome = run(scenario())
    assert isinstance(outcome, BaseException), "the cancelled stream fails the feed"


# @spec SESS-LEASE-004
@pytest.mark.parametrize("which", ["feed", "flush"])
def test_cancelled_call_breaks_the_lease(build, which):
    async def scenario():
        engine = FakeEngine(outcomes=[Silent()])
        _, _, lease = await opened(build, engine)
        if which == "feed":
            call = asyncio.ensure_future(feed(lease, audio(UNIT)))
        else:
            call = asyncio.ensure_future(lease.flush())
        await wait_until(lambda: engine.inputs)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        errors = {
            name: await raised(CALLS[name](lease))
            for name in ("feed", "update_locale", "flush")
        }
        # End the stream so the cleanup calls can complete.
        engine.end_request(Finish("abort"))
        cleanup = [await raised(lease.abort()), await raised(lease.finish())]
        cleanup.append(await raised(lease.release()))
        lease.teardown()
        return errors, cleanup

    errors, cleanup = run(scenario())
    for error in errors.values():
        failure(error, "cancelled")
    assert cleanup == [None, None, None]


# @spec SESS-LEASE-004
def test_cancelled_call_cleanup_calls_do_not_raise_cancelled(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Silent()])
        _, _, lease = await opened(build, engine)
        call = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await wait_until(lambda: engine.inputs)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        engine.end_request(Finish("abort"))
        return await raised(lease.finish())

    assert run(scenario()) is None


async def trigger_input_capacity(build, monkeypatch):
    from vllm_nemotron_asr_streaming.session import PackerCapacityError
    from vllm_nemotron_asr_streaming.session.packer import Packer

    def exhausted(self, samples):
        raise PackerCapacityError("sequence space exhausted")

    monkeypatch.setattr(Packer, "append", exhausted)
    _, _, lease = await opened(build)
    return lease, await raised(feed(lease, audio(UNIT)))


FAILURES = {
    "rejected_input": (FakeEngine, dict(outcomes=[Reject("unsupported_update")])),
    "abnormal_end": (FakeEngine, dict(outcomes=[Finish("error")])),
    "engine_dead": (FakeEngine, dict(outcomes=[Die()])),
    "output_capacity": (FakeEngine, dict(outcomes=[Park((), "a" * ((1 << 20) + 1))])),
}


# @spec SESS-LEASE-005
@pytest.mark.parametrize("reason", list(FAILURES))
def test_failure_fails_the_lease_with_its_reason(build, reason):
    async def scenario():
        cls, kw = FAILURES[reason]
        _, _, lease = await opened(build, cls(**kw))
        observed = await raised(feed(lease, audio(UNIT)))
        again = await raised(CALLS["feed"](lease))
        flush = await raised(lease.flush())
        update = await raised(lease.update_locale(LOCALE))
        return observed, again, flush, update

    for error in run(scenario()):
        failure(error, reason)


# @spec SESS-LEASE-005
def test_input_capacity_fails_the_lease(build, monkeypatch):
    async def scenario():
        lease, observed = await trigger_input_capacity(build, monkeypatch)
        return observed, await raised(CALLS["feed"](lease))

    for error in run(scenario()):
        failure(error, "input_capacity")


# @spec SESS-LEASE-005
def test_idle_failure_is_raised_by_the_next_call(build):
    async def scenario():
        engine, _, lease = await opened(build)
        engine.abort_idle()
        await asyncio.sleep(0.01)
        return [await raised(CALLS[name](lease)) for name in ("feed", "flush")]

    for error in run(scenario()):
        failure(error, "expired")


# @spec SESS-LEASE-006
def test_serialization_is_checked_before_the_closed_factory(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Silent()])
        _, factory, lease = await opened(build, engine)
        first = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await wait_until(lambda: engine.inputs)
        factory.teardown_all()
        # No await since the close: the first call is still running.
        error = await raised(lease.update_locale(LOCALE))
        await asyncio.gather(first, return_exceptions=True)
        return error

    error = run(scenario())
    assert isinstance(error, RuntimeError) and not hasattr(error, "session_error")


# @spec SESS-LEASE-006
def test_closed_factory_is_checked_before_state(build):
    async def scenario():
        _, factory, lease = await opened(build)  # state open: release not allowed
        factory.teardown_all()
        return await raised(lease.release())

    failure(run(scenario()), "cancelled")


# @spec SESS-LEASE-007
@pytest.mark.parametrize("reason", list(FAILURES))
def test_cleanup_calls_do_not_raise_the_reported_failure(build, reason):
    async def scenario():
        cls, kw = FAILURES[reason]
        _, _, lease = await opened(build, cls(**kw))
        reported = await raised(feed(lease, audio(UNIT)))
        results = [
            await raised(lease.abort()),
            await raised(lease.finish()),
            await raised(lease.release()),
        ]
        lease.teardown()
        return reported, results

    reported, results = run(scenario())
    failure(reported, reason)
    assert results == [None, None, None]


# @spec SESS-LEASE-007
@pytest.mark.parametrize("reason", ["abnormal_end", "engine_dead"])
def test_finish_and_release_alone_do_not_raise_a_ended_task_failure(build, reason):
    async def scenario():
        cls, kw = FAILURES[reason]
        _, _, lease = await opened(build, cls(**kw))
        reported = await raised(feed(lease, audio(UNIT)))
        return reported, await raised(lease.finish()), await raised(lease.release())

    reported, finish, release = run(scenario())
    failure(reported, reason)
    assert finish is None and release is None


# ----------------------------------------------------------------- locale


# @spec SESS-LOC-001
def test_update_locale_refuses_an_unknown_locale_and_keeps_the_old(build):
    async def scenario():
        engine, _, lease = await opened(build, locale="en-GB")
        error = await raised(lease.update_locale("xx-XX"))
        await feed(lease, audio(UNIT))
        return engine, error

    engine, error = run(scenario())
    assert (error.session_error, error.reason) == ("refused", "locale")
    assert unit_of(engine.inputs[0]).prompt_index == LOCALES["en-GB"]


# @spec SESS-LOC-002
def test_new_locale_applies_to_buffered_samples_later_units_and_the_tail(build):
    async def scenario():
        engine, _, lease = await opened(build, locale="en-US")
        await feed(lease, audio(700))
        await lease.update_locale("es-ES")
        await feed(lease, audio(580, 700))  # completes the buffered unit
        await lease.update_locale("de-DE")
        await feed(lease, audio(UNIT + 40, 1280))
        await lease.update_locale("en-GB")
        await lease.flush()
        return engine

    engine = run(scenario())
    rows = [unit_of(i).prompt_index for i in engine.inputs]
    assert rows == [
        LOCALES["es-ES"],
        LOCALES["de-DE"],
        LOCALES["en-GB"],
    ], "buffered unit and tail use the locale in force when packed"
    assert unit_of(engine.inputs[-1]).final_tail == 1


# @spec SESS-LOC-003
def test_update_locale_returns_none(build):
    async def scenario():
        _, _, lease = await opened(build)
        return await lease.update_locale("de-DE")

    assert run(scenario()) is None


# ------------------------------------------------------ flush and the rest


# @spec SESS-FLUSH-001
def test_flush_sends_the_final_tail_then_closes_the_input(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT + 100))
        transcript = await lease.flush()
        return engine, transcript

    engine, _ = run(scenario())
    units = [unit_of(i) for i in engine.inputs]
    assert [u.final_tail for u in units] == [0, 1]
    assert units[1].valid_samples == 100
    emit_tail = next(
        i for i, e in enumerate(engine.events) if e[0] == "emit" and e[2] == 1
    )
    close = [e[0] for e in engine.events].index("close")
    assert emit_tail < close, "the tail's outcome is awaited before closing"
    assert engine.only().input_exhausted and engine.only().ended


# @spec SESS-FLUSH-001
def test_flush_with_no_remainder_still_sends_an_empty_final_tail(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT))
        await lease.flush()
        return engine

    engine = run(scenario())
    tail = unit_of(engine.inputs[-1])
    assert tail.final_tail == 1 and tail.valid_samples == 0


# @spec SESS-FLUSH-002
def test_flush_returns_the_final_transcript_and_flushes_the_lease(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[Park((1,), "good "), Park((2,), "day")],
        )
        _, _, lease = await opened(build, engine)
        await feed(lease, audio(UNIT))
        transcript = await lease.flush()
        after = await raised(CALLS["feed"](lease))
        return transcript, after, await lease.finish(), await lease.release()

    transcript, after, finish, release = run(scenario())
    assert transcript == "good day"
    failure(after, "state")
    assert finish is None and release is None


FLUSH_FAILURES = [
    pytest.param(dict(close_outcome=Finish("error")), "abnormal_end", id="error"),
    pytest.param(dict(close_outcome=Finish("abort")), "abnormal_end", id="abort"),
    pytest.param(dict(close_outcome=Finish("length")), "abnormal_end", id="other"),
    pytest.param(
        dict(close_outcome=Reject("input_closed")), "abnormal_end", id="rejected-close"
    ),
    pytest.param(dict(close_outcome=Die()), "engine_dead", id="dead"),
    pytest.param(
        dict(close_outcome=Fail(RuntimeError("x"))), "abnormal_end", id="exception"
    ),
    pytest.param(
        dict(outcomes=[Reject("unsupported_update")]), "rejected_input", id="rejected"
    ),
    pytest.param(dict(outcomes=[Finish("abort")]), "abnormal_end", id="tail-abort"),
]


# @spec SESS-FLUSH-003
@pytest.mark.parametrize("kw,reason", FLUSH_FAILURES)
def test_flush_fails_unless_the_close_is_a_normal_stop(build, kw, reason):
    async def scenario():
        _, _, lease = await opened(build, FakeEngine(**kw))
        flush = await raised(lease.flush())
        later = await raised(CALLS["feed"](lease))
        return flush, later

    flush, later = run(scenario())
    failure(flush, reason)
    failure(later, reason)


# @spec SESS-FLUSH-003
def test_flush_resends_a_paused_final_tail(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Reject("engine_paused"), Park((), "ok")])
        _, _, lease = await opened(build, engine)
        await feed(lease, audio(30))
        return engine, await lease.flush()

    engine, transcript = run(scenario())
    assert transcript == "ok" and len(engine.inputs) == 2
    assert all(unit_of(i).final_tail == 1 for i in engine.inputs)


# @spec SESS-FIN-001
def test_finish_awaits_the_task_and_returns_none(build):
    async def failed_but_running():
        engine = FakeEngine(outcomes=[Reject("unsupported_update")])
        _, _, lease = await opened(build, engine)
        failure(await raised(feed(lease, audio(UNIT))), "rejected_input")
        task = asyncio.ensure_future(lease.finish())
        await asyncio.sleep(0.03)
        waiting = not task.done()
        engine.end_request(Fail(RuntimeError("late failure")))
        return waiting, await asyncio.wait_for(task, 1)

    waiting, result = run(failed_but_running())
    assert waiting is True and result is None


# @spec SESS-FIN-001
def test_finish_does_not_reraise_the_tasks_exception(build):
    async def scenario():
        engine = FakeEngine(close_outcome=Fail(RuntimeError("task failed")))
        _, _, lease = await opened(build, engine)
        flush = await raised(lease.flush())
        return flush, await lease.finish()

    flush, finish = run(scenario())
    failure(flush, "abnormal_end")
    assert finish is None


# @spec SESS-ABORT-001
def test_abort_cancels_the_task_awaits_it_and_aborts_the_lease(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await asyncio.wait_for(lease.abort(), 1)
        after = await raised(CALLS["feed"](lease))
        release = await raised(lease.release())
        return engine, after, release

    engine, after, release = run(scenario())
    assert engine.aborted == [engine.generate_calls[0].request_id]
    assert engine.only().ended
    failure(after, "state")
    assert release is None


# @spec SESS-ABORT-002
def test_cancelled_abort_still_aborts_the_lease_and_allows_release(build):
    async def scenario():
        engine, factory, lease = await opened(build)
        call = asyncio.ensure_future(lease.abort())
        await asyncio.sleep(0)  # abort has cancelled the task and awaits its end
        assert not call.done()
        call.cancel()
        try:
            await call
        except asyncio.CancelledError as error:
            cancelled = error
        else:
            cancelled = None
        state = lease.state
        release = await raised(asyncio.wait_for(lease.release(), 1))
        return engine, factory, lease, cancelled, state, release

    engine, factory, lease, cancelled, state, release = run(scenario())
    assert isinstance(cancelled, asyncio.CancelledError)
    assert state == "aborted"
    assert release is None
    assert lease.state == "released"
    assert engine.aborted == [engine.generate_calls[0].request_id]


def slow_cancel(engine) -> asyncio.Event:
    """Make the session's generate() task end only after the returned event is
    set once it has been cancelled."""
    gate = asyncio.Event()
    original = engine.generate

    async def generate(*args, **kwargs):
        stream = original(*args, **kwargs)
        try:
            async for item in stream:
                yield item
        except asyncio.CancelledError:
            await gate.wait()
            raise
        finally:
            await stream.aclose()

    engine.generate = generate
    return gate


async def cancel_abort(lease) -> None:
    call = asyncio.ensure_future(lease.abort())
    await asyncio.sleep(0)
    assert not call.done()
    call.cancel()
    try:
        await call
    except asyncio.CancelledError:
        pass


# @spec SESS-ABORT-002, SESS-FIN-001
def test_finish_after_a_cancelled_abort_awaits_the_task(build):
    async def scenario():
        engine = FakeEngine()
        gate = slow_cancel(engine)
        engine, _, lease = await opened(build, engine)
        await cancel_abort(lease)
        finish = asyncio.ensure_future(lease.finish())
        await asyncio.sleep(0.05)
        waited = not finish.done()
        gate.set()
        await asyncio.wait_for(finish, 1)
        return waited, lease

    waited, lease = run(scenario())
    assert waited
    assert lease.task.done()
    assert lease.state == "aborted"


# @spec SESS-ABORT-002, SESS-REL-002
@pytest.mark.parametrize("error", ["engine_dead", "other"])
def test_failed_release_after_a_cancelled_abort_keeps_the_task_owned(build, error):
    from vllm.v1.engine.exceptions import EngineDeadError

    async def scenario():
        raised_by_wait = (
            EngineDeadError() if error == "engine_dead" else RuntimeError("x")
        )
        engine = FakeEngine(retire=[raised_by_wait])
        gate = slow_cancel(engine)
        engine, factory, lease = await opened(build, engine)
        await cancel_abort(lease)
        release = await raised(lease.release())
        # The record stays while the task runs, so teardown_all() still finds it.
        kept = lease.caller_id in factory.sessions and not lease.task.done()
        gate.set()
        await wait_until(lambda: lease.caller_id not in factory.sessions)
        return release, kept, lease.task.done()

    release, kept, ended = run(scenario())
    if error == "engine_dead":
        failure(release, "engine_dead")
    else:
        assert isinstance(release, RuntimeError)
    assert kept
    assert ended


# @spec SESS-ABORT-002
def test_repeated_failed_releases_keep_one_holder(build):
    async def scenario():
        engine = FakeEngine(retire=[RuntimeError("x") for _ in range(3)])
        gate = slow_cancel(engine)
        engine, factory, lease = await opened(build, engine)
        await cancel_abort(lease)
        for _ in range(3):
            assert isinstance(await raised(lease.release()), RuntimeError)
        holders = len(factory._cleanup_tasks)
        gate.set()
        await wait_until(lambda: lease.caller_id not in factory.sessions)
        return holders

    assert run(scenario()) == 1


# @spec SESS-REL-001
def test_release_waits_for_retirement_of_its_caller_id(build):
    async def scenario():
        engine, factory, lease = await opened(build)
        await lease.flush()
        assert await lease.release() is None
        after = await raised(CALLS["feed"](lease))
        return engine, factory, after

    engine, factory, after = run(scenario())
    caller_id = engine.generate_calls[0].request_id
    assert engine.wait_calls == [caller_id]
    assert caller_id not in factory.sessions
    failure(after, "state")


# @spec SESS-REL-001
@pytest.mark.parametrize("kind", ["value-error", "admission-error"])
def test_release_resolves_terminal_wait_results_like_cleanup(build, kind):
    from vllm.v1.engine import StreamAdmissionError

    result = (
        ValueError("no longer tracked")
        if kind == "value-error"
        else StreamAdmissionError(refusal("x", "capacity"))
    )

    async def scenario():
        engine, factory, lease = await opened(build, FakeEngine(retire=[result]))
        await lease.flush()
        outcome = await raised(lease.release())
        return engine, factory, outcome

    engine, factory, outcome = run(scenario())
    assert outcome is None
    assert engine.generate_calls[0].request_id not in factory.sessions


# @spec SESS-REL-002
def test_release_engine_dead_fails_release_and_removes_the_record(build):
    from vllm.v1.engine.exceptions import EngineDeadError

    async def scenario():
        engine, factory, lease = await opened(
            build, FakeEngine(retire=[EngineDeadError()])
        )
        await lease.flush()
        error = await raised(lease.release())
        return engine, factory, error

    engine, factory, error = run(scenario())
    failure(error, "engine_dead")
    assert engine.generate_calls[0].request_id not in factory.sessions


# @spec SESS-REL-002
def test_release_after_an_engine_dead_failure_removes_the_record_and_returns(build):
    async def scenario():
        engine, factory, lease = await opened(build, FakeEngine(outcomes=[Die()]))
        reported = await raised(feed(lease, audio(UNIT)))
        return engine, factory, reported, await raised(lease.release())

    engine, factory, reported, release = run(scenario())
    failure(reported, "engine_dead")
    assert release is None
    assert engine.generate_calls[0].request_id not in factory.sessions


# @spec SESS-REL-003
def test_cancelled_release_leaves_the_session_with_the_factory(build):
    async def scenario():
        engine, factory, lease = await opened(build, FakeEngine(retire=["hang", None]))
        await lease.flush()
        release = asyncio.ensure_future(lease.release())
        await wait_until(lambda: engine.wait_calls)
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        await wait_until(lambda: len(engine.wait_calls) == 2)
        return engine

    engine = run(scenario())
    assert len(set(engine.wait_calls)) == 1


# @spec SESS-TEAR-001
def test_teardown_cancels_the_task_without_awaiting_or_engine_calls(build):
    async def scenario():
        engine, _, lease = await opened(build)
        assert not inspect.iscoroutinefunction(lease.teardown)
        events_before = list(engine.events)
        waits_before = list(engine.wait_calls)
        result = lease.teardown()
        # No engine call happened synchronously.
        assert engine.events == events_before
        assert engine.wait_calls == waits_before
        await wait_until(lambda: engine.only().ended)
        return engine, result

    engine, result = run(scenario())
    assert result is None
    assert engine.wait_calls == []
    assert engine.unexpected == []


# @spec SESS-EXP-001
def test_idle_abort_expires_the_lease(build):
    async def scenario():
        engine, _, lease = await opened(build)
        engine.abort_idle()
        await asyncio.sleep(0.01)
        return await raised(CALLS["feed"](lease)), await raised(lease.finish())

    expired, finish = run(scenario())
    failure(expired, "expired")
    assert finish is None


# @spec SESS-EXP-001
def test_abort_while_parked_idle_expires_the_lease(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT))
        engine.abort_idle()
        await asyncio.sleep(0.01)
        return await raised(CALLS["update_locale"](lease))

    failure(run(scenario()), "expired")


IDLE_ENDS = [
    pytest.param("kill", "engine_dead", id="engine-dead"),
    pytest.param(Finish("error"), "abnormal_end", id="error-finish"),
    pytest.param(Finish("stop"), "abnormal_end", id="stop-finish"),
    pytest.param(Fail(RuntimeError("x")), "abnormal_end", id="exception"),
]


# @spec SESS-EXP-001
@pytest.mark.parametrize("ending,reason", IDLE_ENDS)
def test_other_idle_ends_are_not_expired(build, ending, reason):
    async def scenario():
        engine, _, lease = await opened(build)
        if ending == "kill":
            engine.kill()
        else:
            engine.end_request(ending)
        await asyncio.sleep(0.01)
        return await raised(CALLS["feed"](lease))

    failure(run(scenario()), reason)


# @spec SESS-EXP-001
def test_end_after_the_input_closed_is_not_an_idle_failure(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await lease.flush()
        await asyncio.sleep(0.01)
        return await raised(lease.finish()), await raised(lease.release())

    assert run(scenario()) == (None, None)


# --------------------------------------------------------------- errors


async def refusals(build):
    from vllm_nemotron_asr_streaming.session import SessionRefused  # noqa: F401

    got = {}
    for reason in ("capacity", "active_limit", "engine_paused", "request_id_in_use"):
        _, factory = build(FakeEngine(admission=reason))
        got[reason] = await raised(factory.open(cadence=CADENCE, locale=LOCALE))
    _, factory = build()
    got["cadence"] = await raised(factory.open(cadence="90ms", locale=LOCALE))
    got["locale"] = await raised(factory.open(cadence=CADENCE, locale="xx-XX"))
    return got


# @spec SESS-ERR-001
def test_session_refused_carries_attribute_and_reason(build):
    from vllm_nemotron_asr_streaming.session import SessionRefused

    got = run(refusals(build))
    assert set(got) == {
        "capacity",
        "active_limit",
        "engine_paused",
        "request_id_in_use",
        "cadence",
        "locale",
    }
    for reason, error in got.items():
        assert isinstance(error, SessionRefused)
        assert getattr(error, "session_error", None) == "refused"
        assert getattr(error, "reason", None) == reason


# @spec SESS-ERR-002
def test_session_failed_carries_attribute_and_each_reason(build, monkeypatch):
    from vllm_nemotron_asr_streaming.session import SessionFailed

    async def scenario():
        got = {}
        for reason, (cls, kw) in FAILURES.items():
            _, _, lease = await opened(build, cls(**kw))
            got[reason] = await raised(feed(lease, audio(UNIT)))
        engine, _, lease = await opened(build)
        engine.abort_idle()
        await asyncio.sleep(0.01)
        got["expired"] = await raised(CALLS["feed"](lease))
        _, _, lease = await opened(build)
        await lease.abort()
        got["state"] = await raised(CALLS["feed"](lease))
        engine, factory, lease = await opened(build)
        factory.teardown_all()
        got["cancelled"] = await raised(CALLS["feed"](lease))
        return got

    got = run(scenario())

    async def capacity():
        lease, error = await trigger_input_capacity(build, monkeypatch)
        return error

    got["input_capacity"] = run(capacity())
    assert set(got) == {
        "rejected_input",
        "abnormal_end",
        "expired",
        "engine_dead",
        "cancelled",
        "output_capacity",
        "input_capacity",
        "state",
    }
    for reason, error in got.items():
        assert isinstance(error, SessionFailed), (reason, error)
        assert getattr(error, "session_error", None) == "failed"
        assert getattr(error, "reason", None) == reason


# @spec SESS-ERR-003
def test_caller_bug_exceptions_carry_neither_attribute(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Park((), "x", delay=0.1)])
        _, _, lease = await opened(build, engine)
        bad = await raised(
            lease.feed(np.zeros(3, np.int16), on_accepted=lambda n: None)
        )
        running = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await wait_until(lambda: engine.inputs)
        concurrent = await raised(CALLS["flush"](lease))
        await running
        return bad, concurrent

    bad, concurrent = run(scenario())
    assert isinstance(bad, ValueError) and isinstance(concurrent, RuntimeError)
    for error in (bad, concurrent):
        assert not hasattr(error, "session_error")
        assert not hasattr(error, "reason")


# ------------------------------------------------------------ transcript


# @spec SESS-TXT-001
def test_transcript_appends_each_outputs_delta_text_verbatim(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[
                Park((1, 2), "hel", split=True),
                [Text((3,), "lo "), Park((4,), "wor")],
                Park((5,), "ld"),
            ]
        )
        _, _, lease = await opened(build, engine)
        result, _ = await feed(lease, audio(UNIT * 3))
        return result, await lease.flush()

    result, final = run(scenario())
    assert result[-1] == "hello world" and final == "hello world"
    assert result[0] == "hel"


# @spec SESS-TXT-003
def test_park_token_ends_a_unit_and_adds_no_text(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Park((1,), "ab"), Park((), ""), Park((2,), "cd")])
        _, _, lease = await opened(build, engine)
        result, _ = await feed(lease, audio(UNIT * 3))
        return engine, result

    engine, result = run(scenario())
    assert result == ["ab", "abcd"], "a park-only unit adds nothing"
    assert len(engine.inputs) == 3, "each park ended its unit"


# @spec SESS-TXT-003
def test_unit_text_split_across_outputs_completes_at_the_park(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[Park((1,), "ab", split=True), Park((2,), "cd", split=True)],
            default=Park(),
        )
        _, _, lease = await opened(build, engine)
        result, _ = await feed(lease, audio(UNIT * 2))
        return result

    assert run(scenario())[-1] == "abcd"


# @spec SESS-TXT-004
def test_transcript_over_one_mebibyte_of_utf8_fails_the_lease(build):
    async def scenario():
        # Two bytes per character: fewer than 1 MiB characters, over 1 MiB bytes.
        engine = FakeEngine(outcomes=[Park((), "é" * (512 * 1024 + 1))])
        _, _, lease = await opened(build, engine)
        error = await raised(feed(lease, audio(UNIT)))
        return error, await raised(CALLS["feed"](lease))

    first, second = run(scenario())
    failure(first, "output_capacity")
    failure(second, "output_capacity")


# @spec SESS-TXT-004
def test_transcript_exactly_one_mebibyte_is_accepted(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Park((), "é" * (512 * 1024))])
        _, _, lease = await opened(build, engine)
        result, _ = await feed(lease, audio(UNIT))
        return result

    (text,) = run(scenario())
    assert len(text.encode()) == 1 << 20


# @spec SESS-TXT-004
def test_transcript_limit_counts_the_whole_cumulative_transcript(build):
    async def scenario():
        half = "a" * (512 * 1024)
        engine = FakeEngine(outcomes=[Park((), half), Park((), half), Park((), "b")])
        _, _, lease = await opened(build, engine)
        return await raised(feed(lease, audio(UNIT * 3)))

    failure(run(scenario()), "output_capacity")


# -------------------------------------------------------------- endpointing


# @spec SESS-EP-001
def test_session_requests_set_no_endpoint_policy(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT + 5))
        await lease.flush()
        return engine

    engine = run(scenario())
    (call,) = engine.generate_calls
    assert not getattr(call.sampling_params, "extra_args", None)
    assert not [k for k in call.kwargs if "endpoint" in k.lower()]
    for streaming_input in engine.inputs:
        assert not contains_key(streaming_input.prompt, "endpoint")
        params = getattr(streaming_input, "sampling_params", None)
        assert params is None or not getattr(params, "extra_args", None)


# ------------------------------------------- retirement is its own event


# @spec SESS-OWN-002, SESS-OWN-003
def test_cancelled_open_with_a_late_acceptance_needs_abort_and_retirement(build):
    async def scenario():
        engine = FakeEngine(admission_delay=0.15)
        engine, factory = build(engine)
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        caller_id = engine.generate_calls[0].request_id
        await asyncio.sleep(0.05)
        early = (list(engine.aborted), caller_id in factory.sessions)
        import time

        await factory.drain(time.monotonic() + 2)
        return engine, factory, caller_id, early

    engine, factory, caller_id, early = run(scenario())
    assert early == ([caller_id], True), "aborted, but the slot has not retired"
    assert engine.wait_calls == [caller_id]
    assert caller_id not in factory.sessions


# @spec SESS-OWN-002, SESS-REL-001
def test_successful_flush_leaves_cleanup_pending_until_retirement(build):
    async def scenario():
        import time

        engine = FakeEngine(retire_mode="manual")
        engine, factory, lease = await opened(build, engine)
        await lease.flush()
        release = asyncio.ensure_future(lease.release())
        await asyncio.sleep(0.05)
        pending = not release.done()
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        await factory.drain(time.monotonic() + 0.05)
        still_owned = engine.generate_calls[0].request_id in factory.sessions
        engine.retire_now()
        await factory.drain(time.monotonic() + 2)
        return pending, still_owned, factory, engine

    pending, still_owned, factory, engine = run(scenario())
    assert pending and still_owned
    assert len(factory.sessions) == 0


# @spec SESS-REL-001
def test_release_waits_for_an_explicit_retirement(build):
    async def scenario():
        engine = FakeEngine(retire_mode="manual")
        engine, _, lease = await opened(build, engine)
        await lease.flush()
        release = asyncio.ensure_future(lease.release())
        await asyncio.sleep(0.05)
        waiting = not release.done()
        engine.retire_now()
        await asyncio.wait_for(release, 1)
        return waiting

    assert run(scenario()) is True


# @spec SESS-REL-002
def test_engine_death_between_completion_and_retirement_fails_release(build):
    async def scenario():
        engine = FakeEngine(retire_mode="manual")
        engine, factory, lease = await opened(build, engine)
        await lease.flush()
        engine.kill()
        error = await raised(lease.release())
        return engine, factory, error

    engine, factory, error = run(scenario())
    failure(error, "engine_dead")
    assert engine.generate_calls[0].request_id not in factory.sessions


# @spec SESS-OWN-002, SESS-OWN-008
def test_engine_death_before_retirement_is_a_cleanup_fault(build):
    from vllm.v1.engine.exceptions import EngineDeadError

    async def scenario():
        import time

        engine = FakeEngine(retire_mode="manual")
        engine, factory, lease = await opened(build, engine)
        await lease.flush()
        release = asyncio.ensure_future(lease.release())
        await wait_until(lambda: engine.wait_calls)
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        engine.kill()
        faults = await factory.drain(time.monotonic() + 2)
        return factory, faults

    factory, faults = run(scenario())
    assert [type(f) for f in faults] == [EngineDeadError]
    assert len(factory.sessions) == 0


# @spec SESS-REL-002
@pytest.mark.parametrize("order", ["death-then-retire", "retire-then-death"])
def test_death_is_checked_before_and_after_the_retirement_wait(build, order):
    async def scenario():
        engine = FakeEngine(retire_mode="manual")
        engine, factory, lease = await opened(build, engine)
        await lease.flush()
        if order == "death-then-retire":
            engine.kill()
            engine.retire_now()
        else:
            engine.retire_now()
            engine.kill()
        return engine, factory, await raised(lease.release())

    engine, factory, error = run(scenario())
    failure(error, "engine_dead")
    assert engine.generate_calls[0].request_id not in factory.sessions
