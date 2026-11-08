# SPDX-License-Identifier: Apache-2.0
"""Engine binding (SESS-ENG-*): how a session drives ``generate`` with stream
admission, matches outcomes by input position and retries rejections."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from support.fake_engine import (
    Die,
    Fail,
    FakeEngine,
    Finish,
    Park,
    Reject,
    Silent,
    fingerprint,
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


async def opened(build, engine=None, **kw):
    engine, factory = build(engine, **kw)
    return engine, factory, await factory.open(cadence=CADENCE, locale=LOCALE)


async def feed(lease, samples):
    return await lease.feed(samples, on_accepted=lambda n: None)


async def raised(awaitable):
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001
        return error
    return None


def reads(engine):
    return [e for e in engine.events if e[0] == "read"]


# @spec SESS-ENG-001, REG-CARD-016
def test_sampling_parameters(build):
    from vllm.sampling_params import RequestOutputKind

    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT))
        return engine

    engine = run(scenario())
    (params,) = engine.sampling_params
    assert params.n == 1
    assert params.temperature == 0
    assert params.max_tokens == REALTIME_MAX_TOKENS == 142
    assert params.output_kind == RequestOutputKind.DELTA
    assert params.skip_special_tokens is True


# @spec SESS-ENG-002
def test_next_unit_is_yielded_only_after_the_previous_outcome(build):
    async def scenario():
        engine = FakeEngine(default=Park(delay=0.03))
        engine, _, lease = await opened(build, engine)
        await feed(lease, audio(UNIT * 3))
        return engine

    engine = run(scenario())
    assert len(engine.inputs) == 3
    emits = {e[2]: i for i, e in enumerate(engine.events) if e[0] == "emit"}
    read_at = {e[2]: i for i, e in enumerate(engine.events) if e[0] == "read"}
    for position in (1, 2):
        assert emits[position - 1] < read_at[position]
    from vllm.engine.protocol import StreamingInput

    assert all(isinstance(i, StreamingInput) for i in engine.inputs)


# @spec SESS-ENG-002
def test_nothing_is_yielded_while_no_unit_is_complete(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT - 1))
        await asyncio.sleep(0.02)
        return engine

    assert run(scenario()).inputs == []


# @spec SESS-ENG-003
def test_outcome_is_matched_by_position_not_chunk_sequence(build):
    async def scenario():
        # Position 0 is paused; its resend is position 1 (chunk_sequence 0).
        # The second unit has chunk_sequence 1 but is at position 2: a
        # rejection of position 2 must fail it. A chunk_sequence match would
        # look for index 1 and wait forever.
        engine = FakeEngine(
            outcomes=[
                Reject("engine_paused"),
                Park((1,), "a"),
                Reject("unsupported_update"),
            ]
        )
        _, _, lease = await opened(build, engine)
        return engine, await asyncio.wait_for(raised(feed(lease, audio(UNIT * 2))), 2)

    engine, error = run(scenario())
    assert error.session_error == "failed" and error.reason == "rejected_input"
    units = [unit_of(i) for i in engine.inputs]
    assert [u.chunk_sequence for u in units] == [0, 0, 1]


# @spec SESS-ENG-003
def test_outcome_is_the_output_whose_new_token_ids_end_with_the_park(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Park((11, 12), "xy", split=True)], default=Park())
        _, _, lease = await opened(build, engine)
        result = await asyncio.wait_for(feed(lease, audio(UNIT * 2)), 2)
        return engine, result

    engine, result = run(scenario())
    assert result == ["xy"]
    assert len(engine.inputs) == 2, "the unit ended at the park, not at the labels"


# @spec SESS-ENG-004
def test_engine_paused_is_resent_with_doubling_backoff(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Reject("engine_paused")] * 3 + [Park((), "ok")])
        engine, _, lease = await opened(
            build, engine, backoff_initial_s=0.02, backoff_max_s=0.05
        )
        return engine, await feed(lease, audio(UNIT))

    engine, result = run(scenario())
    assert result == ["ok"] and len(engine.inputs) == 4
    times = [e[3] for e in reads(engine)]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    # 20 ms, 40 ms, then capped at 50 ms (lower bounds only).
    assert gaps[0] >= 0.018 and gaps[1] >= 0.036 and gaps[2] >= 0.045


# @spec SESS-ENG-004
def test_engine_paused_is_retried_until_the_caller_cancels(build):
    async def scenario():
        engine = FakeEngine(default=Reject("engine_paused"))
        engine, _, lease = await opened(build, engine)
        call = asyncio.ensure_future(feed(lease, audio(UNIT)))
        await asyncio.sleep(0.15)
        still_trying = not call.done()
        count = len(engine.inputs)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        error = await raised(feed(lease, audio(0)))
        return still_trying, count, error

    still_trying, count, error = run(scenario())
    assert still_trying and count >= 5
    assert error.reason == "cancelled"


# @spec SESS-ENG-005
def test_mm_cache_miss_is_resent_once(build):
    async def scenario():
        engine = FakeEngine(outcomes=[Reject("mm_cache_miss"), Park((), "ok")])
        engine, _, lease = await opened(build, engine)
        return engine, await feed(lease, audio(UNIT))

    engine, result = run(scenario())
    assert result == ["ok"] and len(engine.inputs) == 2


# @spec SESS-ENG-005
def test_second_mm_cache_miss_fails_the_lease(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[Reject("mm_cache_miss"), Reject("mm_cache_miss"), Park()]
        )
        engine, _, lease = await opened(build, engine)
        return engine, await raised(feed(lease, audio(UNIT)))

    engine, error = run(scenario())
    assert error.session_error == "failed" and error.reason == "rejected_input"
    assert len(engine.inputs) == 2, "no third attempt"


# @spec SESS-ENG-005
def test_each_unit_gets_its_own_single_mm_cache_miss_resend(build):
    async def scenario():
        engine = FakeEngine(
            outcomes=[
                Reject("mm_cache_miss"),
                Park((), "a"),
                Reject("mm_cache_miss"),
                Park((), "b"),
            ]
        )
        engine, _, lease = await opened(build, engine)
        return await feed(lease, audio(UNIT * 2))

    assert run(scenario()) == ["a", "ab"]


OTHER_REASONS = [
    "unknown_admission",
    "identity_mismatch",
    "input_closed",
    "empty_update",
    "unsupported_update",
    "preprocessing_error",
    "shutdown",
    "validation_error",
]


# @spec SESS-ENG-006
@pytest.mark.parametrize("reason", OTHER_REASONS)
def test_any_other_rejection_fails_the_lease_without_resend(build, reason):
    async def scenario():
        engine = FakeEngine(outcomes=[Reject(reason)])
        engine, _, lease = await opened(build, engine)
        error = await raised(feed(lease, audio(UNIT)))
        return engine, error, await raised(feed(lease, audio(0)))

    engine, error, later = run(scenario())
    assert (error.session_error, error.reason) == ("failed", "rejected_input")
    assert later.reason == "rejected_input"
    assert len(engine.inputs) == 1


# @spec SESS-ENG-007
@pytest.mark.parametrize(
    "rejections", [["engine_paused", "engine_paused"], ["mm_cache_miss"]]
)
def test_resent_unit_is_identical_including_sequence_and_stamp(build, rejections):
    async def scenario():
        engine = FakeEngine(outcomes=[Reject(r) for r in rejections] + [Park((), "ok")])
        engine, _, lease = await opened(
            build, engine, backoff_initial_s=0.02, backoff_max_s=0.04
        )
        await feed(lease, audio(UNIT * 2 + 10))
        return engine

    engine = run(scenario())
    n = len(rejections) + 1
    first = engine.inputs[:n]
    assert len({fingerprint(i.prompt) for i in first}) == 1
    heads = [unit_of(i) for i in first]
    assert {h.chunk_sequence for h in heads} == {0}
    assert len({h.admission_ms_mod for h in heads}) == 1, "the stamp is kept"
    nxt = unit_of(engine.inputs[n])
    assert nxt.chunk_sequence == 1, "numbering resumes after the resent unit"


# @spec SESS-ENG-008
def test_session_never_builds_admission_contexts_or_numbers_inputs(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT + 7))
        await lease.flush()
        return engine

    engine = run(scenario())
    from vllm.engine.protocol import StreamingInput

    (call,) = engine.generate_calls
    assert call.stream_admission is True
    for item in engine.inputs:
        assert type(item) is StreamingInput
        flat = repr(vars(item))
        assert "StreamAdmissionContext" not in flat and "update_seq" not in flat
    assert engine.only().input_exhausted, "exhausting the generator is the close"
    assert engine.unexpected == [], "no other engine entry point was used"
    assert engine.aborted == []


# @spec SESS-ENG-008
def test_there_is_no_flush_control_token_only_a_final_tail_unit(build):
    async def scenario():
        engine, _, lease = await opened(build)
        await lease.flush()
        return engine

    engine = run(scenario())
    assert len(engine.inputs) == 1
    assert unit_of(engine.inputs[0]).final_tail == 1


END_WHILE_OUTSTANDING = [
    pytest.param(Finish("error"), "abnormal_end", id="error"),
    pytest.param(Finish("abort"), "abnormal_end", id="abort"),
    pytest.param(Finish("stop"), "abnormal_end", id="stop-without-park"),
    pytest.param(Die(), "engine_dead", id="engine-dead"),
    pytest.param(Fail(RuntimeError("x")), "abnormal_end", id="exception"),
]


# @spec SESS-ENG-009
@pytest.mark.parametrize("ending,reason", END_WHILE_OUTSTANDING)
def test_stream_end_while_feed_awaits_a_unit(build, ending, reason):
    async def scenario():
        engine = FakeEngine(outcomes=[ending])
        _, _, lease = await opened(build, engine)
        return await asyncio.wait_for(raised(feed(lease, audio(UNIT))), 2)

    error = run(scenario())
    assert (error.session_error, error.reason) == ("failed", reason)


# @spec SESS-ENG-009
@pytest.mark.parametrize("ending,reason", END_WHILE_OUTSTANDING)
def test_stream_end_while_flush_awaits_the_tail(build, ending, reason):
    async def scenario():
        engine = FakeEngine(outcomes=[ending])
        _, _, lease = await opened(build, engine)
        return await asyncio.wait_for(raised(lease.flush()), 2)

    error = run(scenario())
    assert (error.session_error, error.reason) == ("failed", reason)


# @spec SESS-ENG-009
def test_external_end_while_a_unit_is_outstanding(build):
    async def scenario():
        engine = FakeEngine(default=Silent())
        _, _, lease = await opened(build, engine)
        call = asyncio.ensure_future(raised(feed(lease, audio(UNIT))))
        await wait_until(lambda: engine.inputs)
        engine.abort_idle()  # an abort, but a unit is outstanding
        return await asyncio.wait_for(call, 2)

    error = run(scenario())
    assert error.reason == "abnormal_end", "not expired: a unit was outstanding"


# @spec SESS-ENG-002, SESS-PACK-001
def test_units_use_the_carrier_and_park_ids_of_the_served_configuration(build, layout):
    async def scenario():
        engine, _, lease = await opened(build)
        await feed(lease, audio(UNIT))
        return engine

    engine = run(scenario())
    assert engine.blank == layout
    assert unit_of(engine.inputs[0]).token_ids == [layout + 2]
