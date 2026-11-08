# SPDX-License-Identifier: Apache-2.0
"""Factory construction, opening and ownership (SESS-FAC, SESS-OPEN, SESS-OWN)."""

from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import re
import sys
import time
import tomllib
import uuid
from pathlib import Path

import numpy as np
import pytest
from support.card import LOCALES
from support.fake_engine import (
    Delay,
    FakeEngine,
    make_hf_config,
    make_model_config,
    run,
    wait_until,
)

PKG = Path(__file__).resolve().parents[2]
CADENCE = "80ms"
LOCALE = "en-US"
UNIT = 1280


def audio(n: int) -> np.ndarray:
    return np.full(n, 0.01, dtype=np.float32)


# ------------------------------------------------------------ discovery


# @spec SESS-FAC-001
def test_entry_point_declared():
    meta = tomllib.loads((PKG / "pyproject.toml").read_text())
    group = meta["project"]["entry-points"]["vllm_riva_frontend.session_factories"]
    assert group == {
        "nemotron_asr_streaming": (
            "vllm_nemotron_asr_streaming.session:create_session_factory"
        )
    }


# @spec SESS-FAC-001
@pytest.mark.engine
def test_entry_point_target_resolves():
    import importlib

    module, _, attr = (
        "vllm_nemotron_asr_streaming.session:create_session_factory".partition(":")
    )
    assert callable(getattr(importlib.import_module(module), attr))


# @spec SESS-FAC-002
@pytest.mark.engine
def test_other_architecture_gets_no_factory():
    from vllm_nemotron_asr_streaming.session import create_session_factory

    for arch in ("Nemotron3_5AsrForRNNT", "LlamaForCausalLM"):
        model_config = make_model_config(make_hf_config(architecture=arch))
        assert create_session_factory(FakeEngine(), model_config) is None


# @spec SESS-FAC-002
@pytest.mark.engine
def test_served_architecture_gets_a_factory():
    from vllm_nemotron_asr_streaming.session import create_session_factory

    factory = create_session_factory(FakeEngine(), make_model_config())
    assert factory is not None
    assert hasattr(factory, "open") and hasattr(factory, "teardown_all")


# @spec SESS-FAC-003
@pytest.mark.engine
def test_engine_without_wait_stream_retired_is_refused():
    from vllm_nemotron_asr_streaming.session import create_session_factory

    class NoWait:
        async def generate(
            self, prompt, sampling_params, request_id, *, stream_admission=False
        ):
            yield None

    with pytest.raises(Exception, match="wait_stream_retired"):
        create_session_factory(NoWait(), make_model_config())


# @spec SESS-FAC-003
@pytest.mark.engine
def test_engine_whose_generate_lacks_stream_admission_is_refused():
    from vllm_nemotron_asr_streaming.session import create_session_factory

    class NoAdmission:
        async def generate(self, prompt, sampling_params, request_id):
            yield None

        async def wait_stream_retired(self, request_id):
            return None

    with pytest.raises(Exception, match="stream_admission"):
        create_session_factory(NoAdmission(), make_model_config())


# @spec SESS-FAC-003
@pytest.mark.engine
def test_engine_without_generate_is_refused():
    from vllm_nemotron_asr_streaming.session import create_session_factory

    class NoGenerate:
        async def wait_stream_retired(self, request_id):
            return None

    with pytest.raises(Exception, match="generate"):
        create_session_factory(NoGenerate(), make_model_config())


# @spec SESS-FAC-004
@pytest.mark.engine
@pytest.mark.parametrize(
    "card_set,supported,expected",
    [
        ([3, 0, 6, 13], None, {"320ms", "80ms", "560ms", "1120ms"}),
        ([0, 1, 3, 6, 13], [1, 13], {"160ms", "1120ms"}),
        ([3], None, {"320ms"}),
    ],
)
def test_served_cadences_follow_served_lookahead_set(
    build, card_set, supported, expected
):
    hf = make_hf_config(lookaheads=card_set, supported=supported)
    _, factory = build(hf=hf)
    assert isinstance(factory.served_cadences, frozenset)
    assert factory.served_cadences == frozenset(expected)


# @spec SESS-FAC-004
@pytest.mark.engine
def test_locales_are_the_table_keys(build):
    _, factory = build()
    assert isinstance(factory.locales, frozenset)
    assert factory.locales == frozenset(LOCALES)


# @spec SESS-FAC-004
@pytest.mark.engine
@pytest.mark.parametrize(
    "table,expected",
    [
        ({"auto": 9, "en-US": 0, "de-DE": 4}, "auto"),
        ({"en-US": 0}, "en-US"),
        ({"en-US": 0, "en": 0, "de-DE": 4}, None),
    ],
    ids=["auto", "only-locale", "must-name-one"],
)
def test_default_locale_rule(build, table, expected):
    _, factory = build(hf=make_hf_config(locales=table))
    assert factory.default_locale == expected


# @spec SESS-FAC-005
def test_plugin_does_not_import_the_frontend_package():
    pattern = re.compile(r"^\s*(import|from)\s+vllm_riva_frontend\b", re.M)
    offenders = [
        str(path)
        for path in (PKG / "vllm_nemotron_asr_streaming").rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert offenders == []


# @spec SESS-FAC-005
@pytest.mark.engine
def test_importing_the_session_does_not_load_the_frontend():
    import vllm_nemotron_asr_streaming.session  # noqa: F401

    assert "vllm_riva_frontend" not in sys.modules


def _params(fn):
    return [
        (p.name, p.kind, p.default is inspect.Parameter.empty)
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


# @spec SESS-FAC-005
@pytest.mark.engine
def test_factory_satisfies_the_protocol_structurally(build):
    _, factory = build()
    K = inspect.Parameter
    assert isinstance(factory.served_cadences, frozenset)
    assert isinstance(factory.locales, frozenset)
    assert factory.default_locale is None or isinstance(factory.default_locale, str)
    assert inspect.iscoroutinefunction(factory.open)
    assert _params(factory.open) == [
        ("cadence", K.KEYWORD_ONLY, True),
        ("locale", K.KEYWORD_ONLY, True),
    ]
    assert inspect.iscoroutinefunction(factory.drain)
    assert [p[0] for p in _params(factory.drain)] == ["deadline"]
    assert not inspect.iscoroutinefunction(factory.teardown_all)
    assert _params(factory.teardown_all) == []


# @spec SESS-FAC-005
@pytest.mark.engine
def test_lease_satisfies_the_protocol_structurally(build):
    async def scenario():
        _, factory = build()
        return await factory.open(cadence=CADENCE, locale=LOCALE)

    lease = run(scenario())
    K = inspect.Parameter
    for name in ("feed", "update_locale", "flush", "finish", "abort", "release"):
        assert inspect.iscoroutinefunction(getattr(lease, name)), name
    assert not inspect.iscoroutinefunction(lease.teardown)
    assert _params(lease.teardown) == []
    assert _params(lease.feed) == [
        ("samples", K.POSITIONAL_OR_KEYWORD, True),
        ("on_accepted", K.KEYWORD_ONLY, True),
    ]
    assert [p[0] for p in _params(lease.update_locale)] == ["locale"]
    for name in ("flush", "finish", "abort", "release"):
        assert _params(getattr(lease, name)) == [], name


# ---------------------------------------------------------------- open


# @spec SESS-OPEN-001
@pytest.mark.engine
@pytest.mark.parametrize(
    "cadence,locale,reason",
    [("90ms", LOCALE, "cadence"), (CADENCE, "xx-XX", "locale")],
)
def test_open_refuses_unserved_cadence_or_locale_before_the_engine(
    build, cadence, locale, reason
):
    async def scenario():
        engine, factory = build()
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=cadence, locale=locale)
        assert engine.generate_calls == [] and engine.wait_calls == []
        assert engine.events == [] and engine.unexpected == []
        return caught.value

    error = run(scenario())
    assert error.session_error == "refused" and error.reason == reason


# @spec SESS-OPEN-001
@pytest.mark.engine
def test_open_refuses_a_cadence_the_card_does_not_serve(build):
    async def scenario():
        engine, factory = build(hf=make_hf_config(lookaheads=[3]))
        with pytest.raises(Exception) as caught:
            await factory.open(cadence="80ms", locale=LOCALE)
        assert engine.generate_calls == []
        return caught.value

    assert run(scenario()).reason == "cadence"


# @spec SESS-OPEN-002
@pytest.mark.engine
def test_open_mints_a_fresh_uuid4_and_records_before_generate(build):
    async def scenario():
        engine, factory = build()
        seen = []

        def started(call):
            # SESS-OWN-001 view of the table: the record exists already.
            seen.append(call.request_id in factory.sessions)

        engine.on_generate_started = started
        first = await factory.open(cadence=CADENCE, locale=LOCALE)
        second = await factory.open(cadence=CADENCE, locale=LOCALE)
        return engine, seen, (first, second)

    engine, seen, _ = run(scenario())
    ids = [c.request_id for c in engine.generate_calls]
    assert len(ids) == 2 and ids[0] != ids[1]
    for caller_id in ids:
        assert uuid.UUID(caller_id).version == 4
        assert str(uuid.UUID(caller_id)) == caller_id
    assert seen == [True, True]
    for call in engine.generate_calls:
        assert call.stream_admission is True
        assert inspect.isasyncgen(call.prompt)


# @spec SESS-OPEN-003
@pytest.mark.engine
def test_open_returns_on_the_input_generators_first_read(build):
    async def scenario():
        engine = FakeEngine(admission_delay=0.05, first_read_delay=0.05)
        _, factory = build(engine)
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await asyncio.sleep(0.03)
        assert not task.done(), "admission has not happened yet"
        await asyncio.sleep(0.05)
        assert ("admitted", engine.generate_calls[0].request_id) in engine.events
        assert not task.done(), "admitted, but the input generator is unread"
        lease = await task
        # Returned in state open: an empty feed is accepted.
        accepted = []
        assert await lease.feed(audio(0), on_accepted=accepted.append) == []
        assert accepted == [0]
        return engine

    engine = run(scenario())
    assert engine.inputs == [], "open sends no unit"


# @spec SESS-OPEN-004
@pytest.mark.engine
@pytest.mark.parametrize(
    "reason", ["capacity", "active_limit", "engine_paused", "request_id_in_use"]
)
def test_open_maps_admission_refusal_to_session_refused(build, reason):
    async def scenario():
        engine, factory = build(FakeEngine(admission=reason))
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=CADENCE, locale=LOCALE)
        return caught.value

    error = run(scenario())
    assert error.session_error == "refused" and error.reason == reason


# @spec SESS-OPEN-004
@pytest.mark.engine
def test_second_open_with_a_reserved_id_is_refused_by_the_engine(build, monkeypatch):
    """A caller ID in use is refused by core (async_llm.py:786-790); the
    session reports it as request_id_in_use."""
    fixed = uuid.UUID("12345678-1234-4234-8234-123456789abc")
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed)

    async def scenario():
        _, factory = build()
        first = await factory.open(cadence=CADENCE, locale=LOCALE)
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=CADENCE, locale=LOCALE)
        return first, caught.value

    _, error = run(scenario())
    assert error.session_error == "refused" and error.reason == "request_id_in_use"


# @spec SESS-OPEN-005
@pytest.mark.engine
def test_open_maps_engine_death_before_first_read(build):
    from vllm.v1.engine.exceptions import EngineDeadError

    async def scenario():
        _, factory = build(FakeEngine(admission=EngineDeadError()))
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=CADENCE, locale=LOCALE)
        return caught.value

    error = run(scenario())
    assert error.session_error == "failed" and error.reason == "engine_dead"


# @spec SESS-OPEN-005
@pytest.mark.engine
@pytest.mark.parametrize("admission", [RuntimeError("boom"), "end"])
def test_open_maps_other_ends_before_first_read(build, admission):
    async def scenario():
        _, factory = build(FakeEngine(admission=admission))
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=CADENCE, locale=LOCALE)
        return caught.value

    error = run(scenario())
    assert error.session_error == "failed" and error.reason == "abnormal_end"


# @spec SESS-OPEN-006
@pytest.mark.engine
def test_cancelled_open_hands_a_started_task_to_one_cleanup(build):
    async def scenario():
        engine, factory = build(FakeEngine(admission="hang"))
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await factory.drain(time.monotonic() + 1)
        return engine, factory

    engine, factory = run(scenario())
    caller_id = engine.generate_calls[0].request_id
    assert engine.aborted == [caller_id], "the generate() task was cancelled"
    assert engine.wait_calls == [caller_id], "exactly one cleanup retirement wait"
    assert caller_id not in factory.sessions


# @spec SESS-OPEN-006
@pytest.mark.engine
def test_cancelled_open_never_leaves_an_unowned_record(build):
    """Cancelled as early as an outside caller can: the record is removed at
    once if the task never started, else exactly one cleanup owns it. (A
    caller cannot force the first branch deterministically.)"""

    async def scenario():
        engine, factory = build(FakeEngine(admission="hang"))
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await factory.drain(time.monotonic() + 1)
        return engine, factory

    engine, factory = run(scenario())
    started = len(engine.generate_calls)
    assert len(engine.wait_calls) == started, "cleanup iff the task started"
    assert len(factory.sessions) == 0


# ----------------------------------------------------------- ownership


# @spec SESS-OWN-001
@pytest.mark.engine
def test_record_is_accepted_after_the_first_read(build):
    async def scenario():
        engine, factory = build()
        lease = await factory.open(cadence=CADENCE, locale=LOCALE)
        caller_id = engine.generate_calls[0].request_id
        return factory.sessions[caller_id].admission, lease

    state, _ = run(scenario())
    assert state == "accepted"


# @spec SESS-OWN-001
@pytest.mark.engine
def test_record_stays_pending_when_cancelled_before_either_signal(build):
    async def scenario():
        engine, factory = build(
            FakeEngine(admission="hang", retire=["hang"]),
        )
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: engine.wait_calls)
        caller_id = engine.generate_calls[0].request_id
        state = factory.sessions[caller_id].admission
        factory.teardown_all()
        return state

    assert run(scenario()) == "pending"


# @spec SESS-OWN-001
@pytest.mark.engine
def test_record_stays_pending_when_the_task_fails_before_either_signal(build):
    async def scenario():
        engine, factory = build(
            FakeEngine(admission=RuntimeError("boom"), retire=["hang"])
        )
        with pytest.raises(Exception):
            await factory.open(cadence=CADENCE, locale=LOCALE)
        await wait_until(lambda: engine.wait_calls)
        caller_id = engine.generate_calls[0].request_id
        state = factory.sessions[caller_id].admission
        factory.teardown_all()
        return state

    assert run(scenario()) == "pending"


# @spec SESS-OWN-001, SESS-OWN-002
@pytest.mark.engine
def test_refused_record_is_removed_at_once_without_a_wait(build):
    async def scenario():
        engine, factory = build(FakeEngine(admission="capacity"))
        with pytest.raises(Exception):
            await factory.open(cadence=CADENCE, locale=LOCALE)
        await asyncio.sleep(0.02)
        return engine, factory

    engine, factory = run(scenario())
    assert engine.wait_calls == []
    assert len(factory.sessions) == 0


RETIREMENT_RESULTS = [
    pytest.param(None, False, id="returns"),
    pytest.param("admission-error", False, id="stream-admission-error"),
    pytest.param(ValueError("not tracked"), False, id="value-error"),
    pytest.param("engine-dead", True, id="engine-dead"),
]


def _result(value):
    from vllm.v1.engine import StreamAdmissionError, StreamAdmissionResult
    from vllm.v1.engine.exceptions import EngineDeadError

    if value == "admission-error":
        return StreamAdmissionError(
            StreamAdmissionResult(False, "x", "refused", "capacity")
        )
    if value == "engine-dead":
        return EngineDeadError()
    return value


async def _orphan(build, retire):
    """An orphaned session: ``open`` cancelled after the task started."""
    engine, factory = build(FakeEngine(admission="hang", retire=[retire]))
    task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
    await wait_until(lambda: engine.generate_calls)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    faults = await factory.drain(time.monotonic() + 1)
    return engine, factory, faults


# @spec SESS-OWN-002, SESS-OWN-008
@pytest.mark.engine
@pytest.mark.parametrize("value,is_fault", RETIREMENT_RESULTS)
def test_every_terminal_retirement_result_removes_the_record(build, value, is_fault):
    async def scenario():
        return await _orphan(build, _result(value))

    engine, factory, faults = run(scenario())
    assert engine.wait_calls == [engine.generate_calls[0].request_id]
    assert len(factory.sessions) == 0
    assert (len(faults) == 1) is is_fault, faults


# @spec SESS-OWN-008
@pytest.mark.engine
def test_unexpected_cleanup_exception_is_logged_and_recorded(build, caplog):
    async def scenario():
        return await _orphan(build, RuntimeError("surprise"))

    with caplog.at_level(logging.DEBUG):
        _, factory, faults = run(scenario())
    assert [type(f) for f in faults] == [RuntimeError]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert len(factory.sessions) == 0


# @spec SESS-OWN-008
@pytest.mark.engine
def test_engine_dead_cleanup_fault_is_logged(build, caplog):
    async def scenario():
        return await _orphan(build, _result("engine-dead"))

    with caplog.at_level(logging.DEBUG):
        _, _, faults = run(scenario())
    assert len(faults) == 1
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# @spec SESS-OWN-003
@pytest.mark.engine
def test_cleanup_has_no_deadline_of_its_own(build):
    async def scenario():
        engine, factory = build(
            FakeEngine(admission="hang", retire=[Delay(0.2)]),
        )
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        still_owned = len(factory.sessions)
        faults = await factory.drain(time.monotonic() + 2)
        return still_owned, faults, len(factory.sessions)

    still_owned, faults, after = run(scenario())
    assert still_owned == 1, "the wait outlives any short timeout"
    assert faults == [] and after == 0


# @spec SESS-OWN-003, SESS-REL-003
@pytest.mark.engine
def test_cancelled_release_starts_exactly_one_cleanup(build):
    async def scenario():
        engine, factory = build(FakeEngine(retire=["hang", Delay(0.01)]))
        lease = await factory.open(cadence=CADENCE, locale=LOCALE)
        await lease.flush()
        release = asyncio.ensure_future(lease.release())
        await wait_until(lambda: len(engine.wait_calls) == 1)
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
        await factory.drain(time.monotonic() + 1)
        return engine, factory

    engine, factory = run(scenario())
    caller_id = engine.generate_calls[0].request_id
    assert engine.wait_calls == [caller_id, caller_id]
    assert len(factory.sessions) == 0


# @spec SESS-OWN-004
@pytest.mark.engine
def test_drain_waits_to_the_deadline_keeps_tasks_and_clears_faults(build):
    from vllm.v1.engine.exceptions import EngineDeadError

    async def scenario():
        engine, factory = build(
            FakeEngine(admission="hang", retire=[Delay(0.3)]),
        )
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        started = time.monotonic()
        early = await factory.drain(started + 0.05)
        waited = time.monotonic() - started
        owned_after_early = len(factory.sessions)
        # A second orphan fails with a cleanup fault.
        engine.retire.append(EngineDeadError())
        engine.admission = "hang"
        task2 = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: len(engine.generate_calls) == 2)
        task2.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task2
        first = await factory.drain(time.monotonic() + 2)
        second = await factory.drain(time.monotonic() + 2)
        return early, waited, owned_after_early, first, second, len(factory.sessions)

    early, waited, owned, first, second, remaining = run(scenario())
    assert early == [] and 0.04 <= waited < 0.25, "returns at the deadline"
    assert owned == 1, "still-running cleanup stays owned"
    assert len(first) == 1 and isinstance(first[0], BaseException)
    assert second == [], "faults are cleared by drain"
    assert remaining == 0


# @spec SESS-OWN-005
@pytest.mark.engine
def test_teardown_all_closes_synchronously_and_returns_cancelled_tasks(build):
    async def scenario():
        engine, factory = build()
        leases = [await factory.open(cadence=CADENCE, locale=LOCALE) for _ in range(2)]
        tasks = factory.teardown_all()
        # Closed before returning: no await has happened since the call.
        with pytest.raises(Exception) as caught:
            await factory.open(cadence=CADENCE, locale=LOCALE)
        assert isinstance(tasks, set) and len(tasks) >= 2
        assert all(isinstance(t, asyncio.Task) for t in tasks)
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(t.done() for t in tasks)
        return caught.value, leases

    error, _ = run(scenario())
    assert error.session_error == "failed" and error.reason == "cancelled"


# @spec SESS-OWN-005
@pytest.mark.engine
def test_teardown_all_returns_cleanup_tasks_too(build):
    async def scenario():
        engine, factory = build(FakeEngine(admission="hang", retire=["hang"]))
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await wait_until(lambda: engine.wait_calls)
        tasks = factory.teardown_all()
        pending = [t for t in tasks if not t.done()]
        await asyncio.gather(*tasks, return_exceptions=True)
        return tasks, pending

    tasks, pending = run(scenario())
    assert len(pending) >= 1 and all(t.done() for t in tasks)


# @spec SESS-OWN-006
@pytest.mark.engine
def test_closed_factory_refuses_open_and_lease_calls(build):
    async def scenario():
        _, factory = build()
        lease = await factory.open(cadence=CADENCE, locale=LOCALE)
        factory.teardown_all()
        outcomes = {}
        for name, call in {
            "open": lambda: factory.open(cadence=CADENCE, locale=LOCALE),
            "feed": lambda: lease.feed(audio(10), on_accepted=lambda n: None),
            "update_locale": lambda: lease.update_locale(LOCALE),
            "flush": lambda: lease.flush(),
            "release": lambda: lease.release(),
        }.items():
            with pytest.raises(Exception) as caught:
                await call()
            outcomes[name] = caught.value
        lease.teardown()  # exempt
        return outcomes

    for name, error in run(scenario()).items():
        assert (error.session_error, error.reason) == ("failed", "cancelled"), name


# @spec SESS-OWN-006
@pytest.mark.engine
def test_cancelled_pending_open_after_close_starts_no_cleanup(build):
    async def scenario():
        engine, factory = build(FakeEngine(admission="hang"))
        task = asyncio.ensure_future(factory.open(cadence=CADENCE, locale=LOCALE))
        await wait_until(lambda: engine.generate_calls)
        factory.teardown_all()
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - cancelled or SessionFailed
            pass
        await asyncio.sleep(0.05)
        return engine

    engine = run(scenario())
    assert engine.wait_calls == []


# @spec SESS-OWN-007
@pytest.mark.engine
def test_cancelled_tasks_have_their_exceptions_retrieved(build):
    async def scenario():
        engine, factory = build(FakeEngine(cancel_raises=RuntimeError("late")))
        await factory.open(cadence=CADENCE, locale=LOCALE)
        tasks = factory.teardown_all()
        await asyncio.wait(tasks, timeout=1)
        assert all(t.done() for t in tasks)
        # Failed with the RuntimeError instead of cancelling: if nothing
        # retrieved it, the loop reports "never retrieved" when it is freed.
        del tasks
        gc.collect()
        await asyncio.sleep(0.01)
        gc.collect()

    reports = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, ctx: reports.append(ctx.get("message", ""))
        )
        await scenario()

    run(main())
    assert not [r for r in reports if "never retrieved" in r], reports
