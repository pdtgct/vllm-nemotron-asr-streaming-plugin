# SPDX-License-Identifier: Apache-2.0
"""Cleanup faults (SESS-OWN-008 to SESS-OWN-011): what the factory records,
logs and returns through ``drain``, and what it must not record.

Each fault is checked by identity: recorded exactly once, logged once,
retained after the session record is removed until one ``drain`` returns it,
and absent from the next ``drain``. The engine double imports vLLM, so these
are ``engine`` tier.

Scope: the session factory on the stream-admission client path, which the
session specs' scope separates from the target core's obligations (its port to
core's admission is the session design's Future roadmap 1). The target core
provides no such path (``create_session_factory`` refuses there, SESS-FAC-003),
so these tests run the factory against ``support.fake_engine`` and a pass makes
no claim for the target core.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

import numpy as np
import pytest
from support.fake_engine import (
    Fail,
    FakeEngine,
    Gate,
    SeamEngine,
    Silent,
    run,
    wait_until,
)

import vllm_nemotron_asr_streaming

pytestmark = pytest.mark.engine

LOCALE = "en-US"
UNIT = 1280


class Escaping(BaseException):
    """Not an ``Exception``: a handler for ``Exception`` lets it end the task."""


#: Code of this directory is the factory's.
PACKAGE_DIR = str(Path(vllm_nemotron_asr_streaming.__file__).resolve().parent)


class PluginTasks:
    """Every task the running loop creates from here on whose coroutine is
    code of this plugin's package. ``settled`` is the completion signal for
    work the factory hands to its own tasks: it returns once each such task
    has ended and its done callbacks have run, without reading the factory's
    state."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task] = []
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()

        def factory(loop, coro, **kwargs):
            if previous is not None:
                task = previous(loop, coro, **kwargs)
            else:
                task = asyncio.Task(coro, loop=loop, **kwargs)
            code = getattr(coro, "cr_code", None)
            if code is not None and str(Path(code.co_filename).resolve()).startswith(
                PACKAGE_DIR
            ):
                self.tasks.append(task)
            return task

        loop.set_task_factory(factory)

    async def settled(self, timeout: float = 2.0) -> None:
        pending = [t for t in self.tasks if not t.done()]
        if pending:
            _, running = await asyncio.wait(pending, timeout=timeout)
            assert not running, f"tasks of the factory did not end: {running}"
        # A task's done callbacks run in the order they were added, so the
        # factory's (added when it created the task) have run by now.


def audio(n: int) -> np.ndarray:
    return np.full(n, 0.01, dtype=np.float32)


def logged(caplog, error: BaseException) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and (
            (record.exc_info is not None and record.exc_info[1] is error)
            or str(error) in record.getMessage()
        )
    ]


async def drains(factory) -> tuple[list, list]:
    first = await factory.drain(time.monotonic() + 1)
    second = await factory.drain(time.monotonic() + 1)
    return first, second


async def orphan(factory, engine) -> str:
    """Cancel an ``open`` after its generate() task started: the session is
    handed to the factory's cleanup. Returns its caller ID."""
    task = asyncio.ensure_future(factory.open(locale=LOCALE))
    await wait_until(lambda: engine.generate_calls)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return engine.generate_calls[-1].request_id


async def suppressed(task: asyncio.Future) -> None:
    with contextlib.suppress(BaseException):
        await task


def assert_one_fault(caplog, error, first, second) -> None:
    assert len(first) == 1 and first[0] is error, first
    assert second == [], "a drained fault was returned again"
    assert len(logged(caplog, error)) == 1, [r.getMessage() for r in caplog.records]


# ------------------------------------------------------------ recorded


# @spec SESS-OWN-008, SESS-OWN-010
def test_a_failed_cleanup_retirement_wait_is_one_logged_fault(build, caplog):
    error = RuntimeError("cleanup retirement wait failed")

    async def scenario():
        engine, factory = build(FakeEngine(admission="hang", retire=[error]))
        caller_id = await orphan(factory, engine)
        await wait_until(lambda: caller_id not in factory.sessions)
        return await drains(factory)

    with caplog.at_level(logging.DEBUG):
        first, second = run(scenario())
    assert_one_fault(caplog, error, first, second)


# @spec SESS-OWN-010
def test_a_cleanup_task_ending_with_an_exception_is_one_retained_fault(build, caplog):
    error = Escaping("cleanup task ended with an exception")

    async def scenario():
        tasks = PluginTasks()
        engine, factory = build(FakeEngine(admission="hang", retire=[error]))
        caller_id = await orphan(factory, engine)
        await wait_until(lambda: caller_id not in factory.sessions)
        await tasks.settled()
        return await drains(factory)

    with caplog.at_level(logging.DEBUG):
        first, second = run(scenario())
    assert_one_fault(caplog, error, first, second)


# @spec SESS-OWN-009, SESS-OWN-010
def test_a_generation_exception_after_the_hand_over_is_one_retained_fault(
    build, caplog
):
    """A broken lease whose ``release`` was cancelled: the cleanup task owns
    the session, then its generate() task fails. Nobody surfaced the
    exception; the factory records it once and keeps it past the record's
    removal until one ``drain``."""
    error = RuntimeError("generation failed after the hand-over")

    async def scenario():
        tasks = PluginTasks()
        engine, factory = build(SeamEngine(default=Silent()))
        lease = await factory.open(locale=LOCALE)
        caller_id = engine.generate_calls[0].request_id
        feed = asyncio.ensure_future(
            lease.feed(audio(UNIT), on_accepted=lambda n: None)
        )
        await wait_until(lambda: engine.inputs)
        feed.cancel()
        await suppressed(feed)
        gate = Gate()
        engine.release_gates[caller_id] = gate
        release = asyncio.ensure_future(lease.release())
        await wait_until(gate.entered.is_set)
        release.cancel()
        await suppressed(release)
        engine.fail_request(caller_id, error)
        await wait_until(lambda: caller_id not in factory.sessions)
        await tasks.settled()
        return await drains(factory)

    with caplog.at_level(logging.DEBUG):
        first, second = run(scenario())
    assert_one_fault(caplog, error, first, second)


# @spec SESS-OWN-009, SESS-OWN-010
def test_a_generation_exception_under_a_task_holding_the_session_is_one_fault(
    build, caplog
):
    """``abort`` cancelled while the generate() task's cancellation handler
    awaits the engine, then ``release``: a factory task holds the session
    until the task ends, and the handler then raises."""
    error = RuntimeError("abort handler failed while the session was held")

    async def scenario():
        tasks = PluginTasks()
        engine, factory = build(SeamEngine())
        lease = await factory.open(locale=LOCALE)
        caller_id = engine.generate_calls[0].request_id
        gate = Gate(exc=error)
        engine.abort_gates[caller_id] = gate
        abort = asyncio.ensure_future(lease.abort())
        await wait_until(gate.entered.is_set)
        abort.cancel()
        await suppressed(abort)
        await lease.release()
        held = caller_id in factory.sessions
        gate.release()
        await wait_until(lambda: caller_id not in factory.sessions)
        await tasks.settled()
        return held, *await drains(factory)

    with caplog.at_level(logging.DEBUG):
        held, first, second = run(scenario())
    assert held, "the session was not held while its task was ending"
    assert_one_fault(caplog, error, first, second)


# @spec SESS-OWN-009
def test_an_abort_handler_exception_under_an_uncancelled_abort_is_one_fault(
    build, caplog
):
    error = RuntimeError("abort handler failed under abort")

    async def scenario():
        engine, factory = build(SeamEngine())
        lease = await factory.open(locale=LOCALE)
        caller_id = engine.generate_calls[0].request_id
        gate = Gate(exc=error)
        engine.abort_gates[caller_id] = gate
        abort = asyncio.ensure_future(lease.abort())
        await wait_until(gate.entered.is_set)
        gate.release()
        try:
            returned = await asyncio.wait_for(abort, 1)
            raised = None
        except BaseException as caught:  # noqa: BLE001
            returned, raised = None, caught
        await lease.release()
        await wait_until(lambda: caller_id not in factory.sessions)
        return returned, raised, *await drains(factory)

    with caplog.at_level(logging.DEBUG):
        returned, raised, first, second = run(scenario())
    assert raised is None and returned is None, "abort did not return normally"
    assert_one_fault(caplog, error, first, second)


# -------------------------------------------------------- not recorded


# @spec SESS-OWN-011
def test_a_refusal_open_raises_is_no_fault(build):
    async def scenario():
        tasks = PluginTasks()
        engine, factory = build(SeamEngine())
        engine.refusals.append("capacity")
        with pytest.raises(Exception) as caught:
            await factory.open(locale=LOCALE)
        await tasks.settled()
        return caught.value, await factory.drain(time.monotonic() + 0.5)

    error, faults = run(scenario())
    assert (error.session_error, error.reason) == ("refused", "capacity")
    assert faults == []


# @spec SESS-OWN-011
def test_a_unit_failure_a_lease_call_raises_is_no_fault(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Fail(RuntimeError("unit failed"))])
        engine, factory = build(engine)
        lease = await factory.open(locale=LOCALE)
        with pytest.raises(Exception) as caught:
            await lease.feed(audio(UNIT), on_accepted=lambda n: None)
        await lease.abort()
        await lease.release()
        return caught.value, await factory.drain(time.monotonic() + 0.5)

    error, faults = run(scenario())
    assert (error.session_error, error.reason) == ("failed", "abnormal_end")
    assert faults == []


# @spec SESS-OWN-011
@pytest.mark.parametrize("kind", ["engine-dead", "other"])
def test_an_idle_generation_failure_the_lease_reports_is_no_fault(build, kind):
    from vllm.v1.engine.exceptions import EngineDeadError

    failure = EngineDeadError() if kind == "engine-dead" else RuntimeError("idle")
    reason = "engine_dead" if kind == "engine-dead" else "abnormal_end"

    async def scenario():
        tasks = PluginTasks()
        engine, factory = build(SeamEngine())
        lease = await factory.open(locale=LOCALE)
        engine.fail_request(engine.generate_calls[0].request_id, failure)
        # the session's generate() task has ended with the failure
        await tasks.settled()
        with pytest.raises(Exception) as caught:
            await lease.feed(audio(UNIT), on_accepted=lambda n: None)
        await lease.abort()
        await lease.release()
        return caught.value, await factory.drain(time.monotonic() + 0.5)

    error, faults = run(scenario())
    assert (error.session_error, error.reason) == ("failed", reason)
    assert faults == []


# @spec SESS-OWN-011
def test_tasks_teardown_all_cancels_are_no_fault(build):
    async def scenario():
        engine, factory = build(FakeEngine(admission="hang", retire=["hang"]))
        await orphan(factory, engine)
        await wait_until(lambda: engine.wait_calls)
        engine.admission = None
        live = await factory.open(locale=LOCALE)
        returned = factory.teardown_all()
        await asyncio.wait(returned, timeout=2)
        return live, returned, await factory.drain(time.monotonic() + 0.5)

    _, returned, faults = run(scenario())
    assert len(returned) >= 2 and all(t.done() for t in returned)
    assert faults == []
