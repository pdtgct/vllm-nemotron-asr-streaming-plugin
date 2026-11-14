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
        lease = await factory.open(cadence="80ms", locale="en-US")
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
        lease = await factory.open(cadence="80ms", locale="en-US")
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
        task = asyncio.ensure_future(factory.open(cadence="80ms", locale="en-US"))
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
