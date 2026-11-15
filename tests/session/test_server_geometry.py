# SPDX-License-Identifier: Apache-2.0
"""The server's geometry (SESS-GEOM-001, SESS-GEOM-002, SESS-OPEN-002,
SESS-RT-001): one lookahead, resolved by ``served_lookahead`` (REG-LA-001),
carried by every session unit, every core realtime unit and every whole-file
prompt header, for the life of the server.

The session factory and the realtime hook import vLLM, so those tests are
``engine`` tier; the file prompt imports no vLLM.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from support.card import LOCALES
from support.fake_engine import (
    make_hf_config,
    make_model_config,
    run,
    unit_of,
)

#: Right context -> geometry index, the implemented order (SESS-PACK-005).
GEOMETRY_INDEX = {0: 0, 1: 1, 3: 2, 6: 3, 13: 4}
ALL_LOOKAHEADS = list(GEOMETRY_INDEX)


def samples(n: int) -> np.ndarray:
    return np.full(n, 0.01, dtype=np.float32)


def seg(r: int) -> int:
    return 1280 * (r + 1)


async def session_units(factory, engine, n_samples, *, locale="en-US"):
    """Open one session, feed ``n_samples``, flush; return its sent units."""
    before = len(engine.inputs)
    lease = await factory.open(locale=locale)
    await lease.feed(samples(n_samples), on_accepted=lambda n: None)
    await lease.flush()
    return [unit_of(i) for i in engine.inputs[before:]]


def realtime_units(hf, n_samples):
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    async def stream():
        yield samples(n_samples)

    async def collect():
        hook = Nemotron3_5AsrStreamingForRNNT.buffer_realtime_audio
        return [p async for p in hook(stream(), asyncio.Queue(), make_model_config(hf))]

    return [unit_of(SimpleNamespace(prompt=p)) for p in run(collect())]


def file_header(hf, n_samples=4_000):
    from vllm_nemotron_asr_streaming.transcription import build_file_prompt

    params = SimpleNamespace(
        task_type="transcribe",
        request_prompt="",
        hotwords=None,
        to_language=None,
        use_beam_search=False,
        max_completion_tokens=None,
        extra_args=None,
        language=None,
        audio=samples(n_samples),
        model_config=make_model_config(hf),
    )
    return build_file_prompt(params)["mm_processor_kwargs"]["file_header"]


# ------------------------------------------------------------ one source


# @spec SESS-GEOM-001, SESS-OPEN-002
@pytest.mark.engine
@pytest.mark.parametrize("r", ALL_LOOKAHEADS)
def test_session_units_carry_the_servers_geometry(build, r):
    async def scenario():
        engine, factory = build(hf=make_hf_config(lookahead=r))
        return await session_units(factory, engine, seg(r) * 2 + 33)

    units = run(scenario())
    assert [u.valid_samples for u in units] == [seg(r), seg(r), 33]
    assert {u.geometry_id for u in units} == {GEOMETRY_INDEX[r]}


# @spec SESS-GEOM-001, SESS-RT-001
@pytest.mark.engine
@pytest.mark.parametrize("r", ALL_LOOKAHEADS)
def test_realtime_units_carry_the_servers_geometry(r):
    units = realtime_units(make_hf_config(lookahead=r), seg(r) + 5)
    assert [u.valid_samples for u in units] == [seg(r), 5]
    assert {u.geometry_id for u in units} == {GEOMETRY_INDEX[r]}


# @spec SESS-GEOM-001
@pytest.mark.parametrize("r", ALL_LOOKAHEADS)
def test_file_header_carries_the_servers_geometry_index(r):
    assert file_header(make_hf_config(lookahead=r))[0] == GEOMETRY_INDEX[r]


# @spec SESS-GEOM-001, SESS-OPEN-002, SESS-RT-001
@pytest.mark.engine
@pytest.mark.parametrize("r", ALL_LOOKAHEADS)
def test_the_card_default_is_the_servers_geometry_without_an_override(build, r):
    """``num_lookahead_tokens`` absent: every reader takes the card's default
    lookahead, as ``served_lookahead`` does (REG-LA-001)."""
    hf = make_hf_config(lookahead=None, card_default=r)
    assert not hasattr(hf, "num_lookahead_tokens")

    async def scenario():
        engine, factory = build(hf=hf)
        return await session_units(factory, engine, seg(r) + 1)

    expected = GEOMETRY_INDEX[r]
    assert {u.geometry_id for u in run(scenario())} == {expected}
    assert {u.geometry_id for u in realtime_units(hf, seg(r) + 1)} == {expected}
    assert file_header(hf)[0] == expected


# @spec SESS-GEOM-001, SESS-RT-001
@pytest.mark.engine
def test_the_override_wins_over_the_card_default(build):
    hf = make_hf_config(lookahead=6, card_default=3)

    async def scenario():
        engine, factory = build(hf=hf)
        return await session_units(factory, engine, seg(6) + 1)

    assert {u.geometry_id for u in run(scenario())} == {GEOMETRY_INDEX[6]}
    assert {u.geometry_id for u in realtime_units(hf, seg(6) + 1)} == {
        GEOMETRY_INDEX[6]
    }
    assert file_header(hf)[0] == GEOMETRY_INDEX[6]


def _patch_served_lookahead(monkeypatch, value: int) -> None:
    """Replace ``lookahead.served_lookahead`` wherever the plugin bound it."""
    import vllm_nemotron_asr_streaming.model  # noqa: F401
    import vllm_nemotron_asr_streaming.session.factory  # noqa: F401
    import vllm_nemotron_asr_streaming.session.packer  # noqa: F401
    import vllm_nemotron_asr_streaming.transcription  # noqa: F401
    from vllm_nemotron_asr_streaming import lookahead

    original = lookahead.served_lookahead

    def fixed(hf_config):
        return value

    for name, module in list(sys.modules.items()):
        if not name.startswith("vllm_nemotron_asr_streaming") or module is None:
            continue
        for attr, bound in list(vars(module).items()):
            if bound is original:
                monkeypatch.setattr(module, attr, fixed)


# @spec SESS-GEOM-001, REG-LA-009
@pytest.mark.engine
def test_every_reader_obtains_the_geometry_from_served_lookahead(build, monkeypatch):
    """The configuration says 3 (320 ms); ``served_lookahead`` says 6: the
    session, the realtime hook and the file prompt all follow the function."""
    _patch_served_lookahead(monkeypatch, 6)
    hf = make_hf_config(lookahead=3, card_default=3)

    async def scenario():
        engine, factory = build(hf=hf)
        return await session_units(factory, engine, seg(6) + 1)

    units = run(scenario())
    assert [u.valid_samples for u in units] == [seg(6), 1]
    assert {u.geometry_id for u in units} == {GEOMETRY_INDEX[6]}
    hook_units = realtime_units(hf, seg(6) + 1)
    assert [u.valid_samples for u in hook_units] == [seg(6), 1]
    assert {u.geometry_id for u in hook_units} == {GEOMETRY_INDEX[6]}
    assert file_header(hf)[0] == GEOMETRY_INDEX[6]


# ------------------------------------------------------- one for a lifetime


# @spec SESS-GEOM-002
@pytest.mark.engine
def test_every_session_runs_at_the_servers_geometry_whatever_the_locale(build):
    """Sessions opened one after another and side by side, with every locale
    and with ``update_locale`` mid-stream, all carry one geometry; the
    realtime hook and the file prompt agree with them."""
    r = 6
    hf = make_hf_config(lookahead=r)

    async def scenario():
        engine, factory = build(hf=hf)
        units = []
        for locale in sorted(LOCALES):
            units += await session_units(factory, engine, seg(r) + 7, locale=locale)
        leases = [await factory.open(locale=loc) for loc in ("en-US", "de-DE")]
        for lease in leases:
            await lease.feed(samples(seg(r) // 2), on_accepted=lambda n: None)
        await leases[0].update_locale("es-ES")
        await leases[1].update_locale("auto")
        for lease in leases:
            await lease.feed(samples(seg(r)), on_accepted=lambda n: None)
            await lease.flush()
        return units, engine

    units, engine = run(scenario())
    every = units + [unit_of(i) for i in engine.inputs]
    assert len(every) > 2 * len(LOCALES)
    assert {u.geometry_id for u in every} == {GEOMETRY_INDEX[r]}
    assert {u.valid_samples for u in every if not u.final_tail} == {seg(r)}
    assert {u.geometry_id for u in realtime_units(hf, seg(r))} == {GEOMETRY_INDEX[r]}
    assert file_header(hf)[0] == GEOMETRY_INDEX[r]


# @spec SESS-GEOM-002
@pytest.mark.engine
def test_the_factory_fixes_the_geometry_when_it_is_built(build):
    """The factory reads the server's geometry once, when it is built; a later
    change to the configuration object selects nothing for it."""
    r = 1
    hf = make_hf_config(lookahead=r)

    async def scenario():
        engine, factory = build(hf=hf)
        first = await session_units(factory, engine, seg(r) + 1)
        hf.num_lookahead_tokens = 13
        hf.att_context_right = 13
        second = await session_units(factory, engine, seg(r) + 1)
        return first + second

    units = run(scenario())
    assert {u.geometry_id for u in units} == {GEOMETRY_INDEX[r]}
    assert {u.valid_samples for u in units if not u.final_tail} == {seg(r)}


# @spec SESS-GEOM-002, SESS-FAC-006
@pytest.mark.engine
def test_no_factory_or_lease_call_takes_a_geometry(build):
    async def scenario():
        _, factory = build()
        return factory, await factory.open(locale="en-US")

    factory, lease = run(scenario())
    words = ("cadence", "geometry", "lookahead", "right_context", "chunk_ms")
    calls = [factory.open, factory.drain, factory.teardown_all]
    calls += [
        getattr(lease, name)
        for name in (
            "feed",
            "update_locale",
            "flush",
            "finish",
            "abort",
            "release",
            "teardown",
        )
    ]
    for call in calls:
        for name in inspect.signature(call).parameters:
            assert not any(w in name.lower() for w in words), (call, name)
    for obj in (factory, lease):
        public = [a for a in dir(obj) if not a.startswith("_")]
        assert not [a for a in public if "cadence" in a.lower()], public


# @spec SESS-GEOM-002
@pytest.mark.parametrize(
    "extra",
    [
        {"num_lookahead_tokens": 13},
        {"cadence": "1120ms"},
        {"geometry_id": 4, "locale": "en-US"},
    ],
    ids=["lookahead", "cadence", "geometry"],
)
def test_no_file_request_field_selects_the_geometry(extra):
    from vllm_nemotron_asr_streaming.transcription import build_file_prompt

    hf = make_hf_config(lookahead=1)
    params = SimpleNamespace(
        task_type="transcribe",
        request_prompt="",
        hotwords=None,
        to_language=None,
        use_beam_search=False,
        max_completion_tokens=None,
        extra_args=dict(extra),
        language=None,
        audio=samples(2_000),
        model_config=make_model_config(hf),
    )
    try:
        header = build_file_prompt(params)["mm_processor_kwargs"]["file_header"]
    except ValueError:
        return  # refusing the field also leaves the geometry unselected
    assert header[0] == GEOMETRY_INDEX[1]
