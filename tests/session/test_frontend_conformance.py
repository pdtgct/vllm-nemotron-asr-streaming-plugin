# SPDX-License-Identifier: Apache-2.0
"""Tier 1 seam conformance (SESS-FAC-007): the frontend's seam conformance
suite, run against this plugin's real session factory.

The suite's tests come from ``vllm_riva_frontend_dev.seam_conformance`` (the
``vllm-riva-frontend-dev`` distribution, installed with ``vllm-riva-frontend``
from the two direct references this repository's test dependencies pin) and
are star-imported below. ``tests/session/conftest.py`` registers the suite
for assertion rewriting before this import. The ``seam`` fixture is the
adapter the suite drives: ``PluginSeam`` over the factory that
``create_session_factory`` returns for ``Nemotron3_5AsrStreamingForRNNT`` on
``support.fake_engine.SeamEngine``.

The adapter arranges each case at the engine boundary (a parked admission, a
refusal, a parked unit outcome, a parked retirement wait, a cancellation
handler that awaits the engine abort, a request that fails), never inside the
factory. It observes the factory through the tasks the factory creates: a
task factory on the running loop records every task whose coroutine is code
of this plugin's package, the session each belongs to, and every
cancellation of it. Two reads use the factory's own state: ``sessions`` (the
session table, SESS-OWN-001) to name a new session's record, and the cleanup
fault list for ``inject_fault``, which has no engine-boundary equivalent once
the factory is closed (ING-CORE-034).

The suite needs a vLLM tree to run, so its tests are ``engine`` tier. Tier 1
is met only by an acceptance run of these tests together with
``test_conformance_pin.py``, against non-editable installs of the pinned
frontend distributions (README "Install and test"), in which no test fails
or is skipped; the record names the pinned reference and the command::

    NEMOTRON_CONFORMANCE_ACCEPTANCE=1 PYTHONPATH=<stage 1 vLLM tree> \\
        python -m pytest -q -p no:cacheprovider \\
        tests/session/test_frontend_conformance.py \\
        tests/session/test_conformance_pin.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from support.fake_engine import SeamEngine, make_hf_config, make_model_config
from vllm_riva_frontend_dev import Block, Raise, SeamError
from vllm_riva_frontend_dev.seam_conformance import *  # noqa: F403

import vllm_nemotron_asr_streaming

pytestmark = pytest.mark.engine

#: Code of this directory is the factory's.
PACKAGE_DIR = str(Path(vllm_nemotron_asr_streaming.__file__).resolve().parent)

#: Cancellations that are unconditional by contract (ING-CORE-027).
UNCONDITIONAL = frozenset({"teardown_all", "teardown"})


#: asyncio's own code, skipped when looking for the code that created a task.
ASYNCIO_DIR = str(Path(asyncio.__file__).resolve().parent)


def _in_package(code: types.CodeType | None) -> bool:
    return code is not None and str(Path(code.co_filename).resolve()).startswith(
        PACKAGE_DIR
    )


def _creator(frame: Any) -> Any:
    """The first frame outside asyncio at or above ``frame``: the code that
    asked for a task."""
    while frame is not None and str(
        Path(frame.f_code.co_filename).resolve()
    ).startswith(ASYNCIO_DIR):
        frame = frame.f_back
    return frame


class _FactoryTask(asyncio.Task):
    """A task the factory created; reports each cancellation to the seam."""

    _seam: PluginSeam | None = None

    def cancel(self, msg: Any = None) -> bool:
        seam = self._seam
        if seam is not None and not self.done():
            seam._cancelled_by(self, sys._getframe(1))
        return super().cancel(msg)


async def _held(coro: Any, block: Block) -> Any:
    """A generation task held before its first step until ``block`` is
    released; cancelled while held, its coroutine never runs."""
    try:
        await block.park()
    except BaseException:
        coro.close()
        raise
    exc = getattr(block.then, "exc", None)
    if exc is not None:
        coro.close()
        raise exc
    return await coro


class PluginSeam:
    """The suite's adapter over the plugin's factory (see the module doc and
    the hook contract in the ``vllm_riva_frontend_dev`` package docstring)."""

    def __init__(self, factory: Any, engine: SeamEngine, clock: Any) -> None:
        self.factory = factory
        self.engine = engine
        self.clock = clock
        self._leases: list[Any] = []
        self._caller: dict[int, str] = {}
        self._opener: dict[int, asyncio.Task | None] = {}
        self._generation: dict[int, asyncio.Task] = {}
        self._covers: dict[int, list[asyncio.Task]] = {}
        self._factory_tasks: set[asyncio.Task] = set()
        self._late: list[str] = []
        self._foreign: list[str] = []
        self._hold_next_generation: Block | None = None
        self._previous_task_factory: Any = None
        engine.is_factory_task = lambda: asyncio.current_task() in self._factory_tasks

    # ------------------------------------------------------ task tracking

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        self._previous_task_factory = loop.get_task_factory()
        loop.set_task_factory(self._task_factory)

    def uninstall(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.set_task_factory(self._previous_task_factory)

    def _register_new_sessions(self) -> None:
        for caller_id, lease in list(self.factory.sessions.items()):
            if id(lease) not in self._caller:
                self._leases.append(lease)
                self._caller[id(lease)] = caller_id
                self._opener[id(lease)] = asyncio.current_task()

    def _session_of(self, coro: Any) -> Any:
        """The tracked session a new factory task serves: found among the
        coroutine's arguments, else among the locals of the plugin frames
        that are creating it."""
        frames: list[Any] = []
        if getattr(coro, "cr_frame", None) is not None:
            frames.append(coro.cr_frame)
        frame = sys._getframe(2)
        while frame is not None:
            if _in_package(frame.f_code):
                frames.append(frame)
            frame = frame.f_back
        for candidate in frames:
            for value in list(candidate.f_locals.values()):
                # ``_leases`` keeps every tracked session alive, so an ``id``
                # names one session.
                if id(value) in self._caller:
                    return value
        return None

    def _task_factory(self, loop: Any, coro: Any, **kwargs: Any) -> asyncio.Task:
        # A factory task is one the plugin's code created, whatever coroutine
        # it runs (a caller's ``create_task(lease.release())`` is not one).
        creator = _creator(sys._getframe(1))
        if creator is None or not _in_package(creator.f_code):
            if self._previous_task_factory is not None:
                return self._previous_task_factory(loop, coro, **kwargs)
            return asyncio.Task(coro, loop=loop, **kwargs)
        self._register_new_sessions()
        session = self._session_of(coro)
        is_generation = session is not None and id(session) not in self._generation
        if is_generation and self._hold_next_generation is not None:
            block, self._hold_next_generation = self._hold_next_generation, None
            coro = _held(coro, block)
        task = _FactoryTask(coro, loop=loop, **kwargs)
        task._seam = self
        self._factory_tasks.add(task)
        if session is not None:
            if is_generation:
                self._generation[id(session)] = task
            else:
                self._covers.setdefault(id(session), []).append(task)
        return task

    def _cancelled_by(self, task: asyncio.Task, frame: Any) -> None:
        code = frame.f_code
        origin = f"{code.co_name} ({Path(code.co_filename).name}:{frame.f_lineno})"
        if task in self.engine.in_handler and code.co_name not in UNCONDITIONAL:
            self._late.append(origin)
        if not _in_package(code):
            self._foreign.append(origin)

    def _caller_of(self, lease: Any) -> str:
        self._register_new_sessions()
        return self._caller.get(id(lease)) or lease.caller_id

    # -------------------------------------------------------- arrangement

    def inject_fault(self, fault: BaseException) -> None:
        self.factory._faults.append(fault)

    def slow(self, lease: Any, name: str) -> Block:
        block = Block()
        caller_id = self._caller_of(lease)
        if name in ("feed", "flush"):
            # The unit the call sends waits for its outcome.
            self.engine.outcome_gates[caller_id] = block
        elif name == "release":
            self.engine.release_gates[caller_id] = block
        elif name == "abort":
            self.engine.abort_gates[caller_id] = block
        else:
            raise NotImplementedError(f"slow({name!r}) has no arrangement")
        return block

    def slow_open(
        self, *, before_generation: bool = False, on_enter: Any = None
    ) -> Block:
        block = Block(on_enter=on_enter)
        if before_generation:
            self._hold_next_generation = block
        else:
            self.engine.open_gates.append(block)
        return block

    def refuse_next_open(self, reason: str = "capacity") -> None:
        self.engine.refusals.append(reason)

    def slow_abandoned_engine_abort(self) -> Block:
        block = Block()
        self.engine.next_abort_gates.append(block)
        return block

    def abandoned(self) -> Any:
        for lease in reversed(self._leases):
            opener = self._opener.get(id(lease))
            if opener is not None and opener.done() and opener.cancelled():
                return lease
        return None

    def slow_cleanup(self) -> Block:
        block = Block()
        self.engine.cleanup_gates.append(block)
        return block

    def slow_engine_abort(self, lease: Any) -> Block:
        block = Block()
        self.engine.abort_gates[self._caller_of(lease)] = block
        return block

    def fail_engine_abort(self, lease: Any, exc: BaseException) -> Block:
        block = Block(then=Raise(exc))
        self.engine.abort_gates[self._caller_of(lease)] = block
        return block

    def fail_generation(self, lease: Any, exc: BaseException) -> None:
        if isinstance(exc, SeamError) and (exc.session_error, exc.reason) == (
            "failed",
            "engine_dead",
        ):
            from vllm.v1.engine.exceptions import EngineDeadError

            exc = EngineDeadError()
        self.engine.fail_request(self._caller_of(lease), exc)

    # -------------------------------------------------------- observation

    def generation(self, lease: Any) -> asyncio.Task | None:
        return self._generation.get(id(lease))

    def engine_calls(self, lease: Any) -> list[str]:
        return list(self.engine.engine_calls.get(self._caller_of(lease), []))

    def handed_over(self, lease: Any) -> bool:
        return bool(self._covers.get(id(lease)))

    def owns(self, lease: Any) -> bool:
        return any(not task.done() for task in self._covers.get(id(lease), []))

    def running_tasks(self) -> set[asyncio.Task]:
        return {task for task in self._factory_tasks if not task.done()}

    def late_cancellations(self) -> list[str]:
        return list(self._late)

    def foreign_cancellations(self) -> list[str]:
        return list(self._foreign)


@pytest_asyncio.fixture(loop_scope="function")
async def seam():
    """The adapter over the factory ``create_session_factory`` returns for
    ``Nemotron3_5AsrStreamingForRNNT`` on the engine double (80 ms server
    geometry, so the suite's 1,600-sample pieces complete one unit)."""
    from vllm_riva_frontend.clock import RealClock

    from vllm_nemotron_asr_streaming.session import create_session_factory

    engine = SeamEngine()
    factory = create_session_factory(
        engine, make_model_config(make_hf_config(lookahead=0))
    )
    assert factory is not None
    adapter = PluginSeam(factory, engine, RealClock())
    loop = asyncio.get_running_loop()
    adapter.install(loop)
    try:
        yield adapter
    finally:
        adapter.uninstall(loop)
        tasks = factory.teardown_all()
        if tasks:
            await asyncio.wait(tasks, timeout=5)


# @spec SESS-FAC-007
def test_the_suite_contributes_its_tests_and_the_seam_fixture_is_defined():
    import vllm_riva_frontend_dev.seam_conformance as suite

    names = [name for name in vars(suite) if name.startswith("test_")]
    assert names, "the suite contributed no tests"
    here = globals()
    assert all(here.get(name) is getattr(suite, name) for name in names)
    assert callable(here.get("seam"))
