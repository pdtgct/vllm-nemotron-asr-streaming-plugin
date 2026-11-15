# SPDX-License-Identifier: Apache-2.0
"""Regressions from the code gate: terminal output before the close, and
ownership of the generate task through orphan cleanup."""

from __future__ import annotations

import asyncio
import time

import pytest
from support.fake_engine import FakeEngine, Finish, Park, run, wait_until

pytestmark = pytest.mark.engine


# @spec SESS-FLUSH-002, SESS-FLUSH-003
def test_terminal_stop_before_the_input_closed_is_an_abnormal_end(build):
    async def scenario():
        engine = FakeEngine(outcomes=[[Park((), "x"), Finish("stop")]])
        _, factory = build(engine)
        lease = await factory.open(locale="en-US")
        try:
            await lease.flush()
        except Exception as error:  # noqa: BLE001
            return error
        return None

    error = run(scenario())
    assert error is not None and error.reason == "abnormal_end"


# @spec SESS-FLUSH-002
def test_ordinary_close_still_flushes(build):
    async def scenario():
        _, factory = build()
        lease = await factory.open(locale="en-US")
        return await lease.flush()

    assert run(scenario()) == ""


class SlowCancel(FakeEngine):
    """A generate whose cancellation stays suspended until ``gate`` is set."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.gate = asyncio.Event()

    async def generate(self, *args, **kwargs):
        try:
            async for out in super().generate(*args, **kwargs):
                yield out
        except asyncio.CancelledError:
            await self.gate.wait()
            raise


# @spec SESS-OWN-002, SESS-OWN-004, SESS-OWN-005
def test_cleanup_keeps_the_generate_task_owned_until_it_ends(build):
    async def scenario():
        engine, factory = build(SlowCancel(admission="hang"))
        task = asyncio.ensure_future(factory.open(locale="en-US"))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: engine.wait_calls)
        await factory.drain(time.monotonic() + 0.1)
        owned = list(factory.sessions)
        gen_task = next(iter(factory.sessions.values())).task
        returned = factory.teardown_all()
        still_pending = gen_task in returned and not gen_task.done()
        engine.gate.set()
        await asyncio.gather(*returned, return_exceptions=True)
        return owned, still_pending

    owned, still_pending = run(scenario())
    assert len(owned) == 1, "record kept while the generate task is suspended"
    assert still_pending


def _retirement(kind):
    from support.fake_engine import refusal
    from vllm.v1.engine import StreamAdmissionError
    from vllm.v1.engine.exceptions import EngineDeadError

    return {
        "returns": None,
        "admission-error": StreamAdmissionError(refusal("x", "capacity")),
        "value-error": ValueError("not tracked"),
        "engine-dead": EngineDeadError(),
    }[kind]


# @spec SESS-OWN-002, SESS-OWN-005
@pytest.mark.parametrize(
    "kind", ["returns", "admission-error", "value-error", "engine-dead"]
)
def test_a_terminal_retirement_result_while_the_task_still_ends_keeps_ownership(
    build, kind
):
    """The retirement wait reaches its terminal result while the cancelled
    generate() task is still ending: the record and one tracked task stay
    with the factory, and teardown_all returns that task, until the task
    ends; then the record is removed."""

    async def scenario():
        engine, factory = build(
            SlowCancel(admission_delay=0.05, retire=[_retirement(kind)])
        )
        task = asyncio.ensure_future(factory.open(locale="en-US"))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: engine.wait_calls)
        await asyncio.sleep(0.02)  # the wait has returned or raised
        kept = list(factory.sessions)
        record = next(iter(factory.sessions.values()), None)
        gen_task = record.task if record is not None else None
        owned_tasks = [
            t for t in factory._cleanup_tasks if not t.done() and t is not gen_task
        ]
        engine.gate.set()
        await factory.drain(time.monotonic() + 1)
        return kept, owned_tasks, list(factory.sessions)

    kept, owned_tasks, after = run(scenario())
    assert len(kept) == 1, "record kept while the generate task is ending"
    assert len(owned_tasks) == 1, "one tracked task keeps the session"
    assert after == []


# @spec SESS-OWN-002, SESS-OWN-005
def test_teardown_all_cancels_the_task_that_keeps_an_ending_session(build):
    async def scenario():
        engine, factory = build(SlowCancel(admission="hang", retire=[None]))
        task = asyncio.ensure_future(factory.open(locale="en-US"))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: engine.wait_calls)
        await asyncio.sleep(0.02)
        keepers = {t for t in factory._cleanup_tasks if not t.done()}
        returned = factory.teardown_all()
        engine.gate.set()
        await asyncio.gather(*returned, return_exceptions=True)
        return keepers, returned

    keepers, returned = run(scenario())
    assert len(keepers) == 1 and keepers <= returned
    assert all(t.done() for t in returned)
