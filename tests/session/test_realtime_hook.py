# SPDX-License-Identifier: Apache-2.0
"""Core ``/v1/realtime`` hook (SESS-RT-*), the one-definition token bound, and
the packer's capacity error on that route (SESS-PACK-008).

The hook is a classmethod that touches no weights, but the model module
imports the engine, so these are ``engine`` tier.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from support.card import LOCALES
from support.fake_engine import (
    make_hf_config,
    make_model_config,
    run,
)
from support.fake_engine import unit_of as _unit_of

pytestmark = pytest.mark.engine

ALL_LOOKAHEADS = [0, 1, 3, 6, 13]
GEOMETRY_INDEX = {0: 0, 1: 1, 3: 2, 6: 3, 13: 4}


def model_class():
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    return Nemotron3_5AsrStreamingForRNNT


def unit(prompt):
    return _unit_of(SimpleNamespace(prompt=prompt))


async def stream(chunks):
    for chunk in chunks:
        yield chunk


async def collect(agen):
    return [item async for item in agen]


def ramp(n, start=0):
    return ((np.arange(start, start + n, dtype=np.float32) + 1.0) / 1e6).astype(
        np.float32
    )


def run_hook(chunks, hf=None, queue=None):
    cls = model_class()
    queue = queue if queue is not None else asyncio.Queue()
    return collect(
        cls.buffer_realtime_audio(stream(chunks), queue, make_model_config(hf))
    )


# @spec SESS-RT-001
@pytest.mark.parametrize("default", ALL_LOOKAHEADS)
def test_hook_packs_at_the_default_lookahead_with_a_final_tail(default):
    s = 1280 * (default + 1)
    total = s * 2 + 300
    audio = ramp(total)
    chunks = [audio[i : i + 777] for i in range(0, total, 777)]
    hf = make_hf_config(lookaheads=ALL_LOOKAHEADS, default_lookahead=default)
    prompts = run(run_hook(chunks, hf))
    units = [unit(p) for p in prompts]
    assert len(units) == 3
    assert [u.chunk_sequence for u in units] == [0, 1, 2]
    assert [u.final_tail for u in units] == [0, 0, 1]
    assert [u.valid_samples for u in units] == [s, s, 300]
    assert {u.geometry_id for u in units} == {GEOMETRY_INDEX[default]}
    assert {u.token_ids and tuple(u.token_ids) for u in units} == {
        (hf.blank_token_id + 2,)
    }
    np.testing.assert_array_equal(np.concatenate([u.samples for u in units]), audio)


# @spec SESS-RT-001
def test_hook_with_exact_multiple_ends_with_an_empty_final_tail():
    hf = make_hf_config(default_lookahead=0)
    units = [unit(p) for p in run(run_hook([ramp(1280 * 2)], hf))]
    assert [u.valid_samples for u in units] == [1280, 1280, 0]
    assert units[-1].final_tail == 1


# @spec SESS-RT-001
def test_hook_yields_each_unit_as_soon_as_it_is_complete():
    async def scenario():
        hf = make_hf_config(default_lookahead=0)
        first_seen = asyncio.Event()

        async def audio_stream():
            yield ramp(1280)
            # The first prompt has to be consumable before the stream ends.
            await asyncio.wait_for(first_seen.wait(), 1)
            yield ramp(100)

        out = []
        agen = model_class().buffer_realtime_audio(
            audio_stream(), asyncio.Queue(), make_model_config(hf)
        )
        async for prompt in agen:
            out.append(prompt)
            first_seen.set()
        return out

    assert len(run(scenario())) == 2


# @spec SESS-RT-001
def test_hook_prompts_match_the_session_packer_segmentation():
    from vllm_nemotron_asr_streaming.session.packer import Packer

    hf = make_hf_config(default_lookahead=3)
    audio = ramp(5120 * 2 + 9)
    units = [unit(p) for p in run(run_hook([audio], hf))]
    packer = Packer(
        right_context=3,
        geometry_index=2,
        locale_row=LOCALES["auto"],
        carrier_token_id=hf.blank_token_id + 2,
        clock_ms=lambda: 0,
    )
    reference = packer.append(audio) + [packer.close()]
    assert [u.valid_samples for u in units] == [int(r.row[1]) for r in reference]
    assert [u.final_tail for u in units] == [int(r.row[3]) for r in reference]


LOCALE_TABLES = [
    pytest.param({"auto": 9, "en-US": 0, "de-DE": 4}, 9, id="auto"),
    pytest.param({"en-US": 0}, 0, id="only-locale"),
    pytest.param({"es-ES": 2}, 2, id="only-locale-other-row"),
]


# @spec SESS-RT-002
@pytest.mark.parametrize("table,row", LOCALE_TABLES)
def test_hook_uses_the_session_default_locale(table, row, build):
    hf = make_hf_config(locales=table)
    units = [unit(p) for p in run(run_hook([ramp(10)], hf))]
    assert [u.prompt_index for u in units] == [row]
    _, factory = build(hf=hf)
    assert factory.default_locale is not None
    assert table[factory.default_locale] == row


# @spec SESS-RT-002
def test_hook_raises_value_error_naming_the_locales_when_none_is_default(build):
    table = {"en-US": 0, "de-DE": 4}
    hf = make_hf_config(locales=table)
    _, factory = build(hf=hf)
    assert factory.default_locale is None
    with pytest.raises(ValueError) as caught:
        run(run_hook([ramp(10)], hf))
    message = str(caught.value)
    assert "en-US" in message and "de-DE" in message


# @spec SESS-RT-003
def test_hook_drains_the_input_queue_without_using_it():
    async def scenario():
        hf = make_hf_config(default_lookahead=0)
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(50):
            queue.put_nowait([i, i + 1])

        async def audio_stream():
            yield ramp(1280)
            for i in range(50):
                queue.put_nowait([1000 + i])
            await asyncio.sleep(0.02)
            queue_size_mid_stream = queue.qsize()
            yield ramp(1280)
            yield ramp(10)
            assert queue_size_mid_stream == 0, "drained for the life of the connection"

        prompts = await collect(
            model_class().buffer_realtime_audio(
                audio_stream(), queue, make_model_config(hf)
            )
        )
        await asyncio.sleep(0.02)
        return prompts, queue.qsize()

    prompts, left = run(scenario())
    assert len(prompts) == 3 and left == 0


# @spec SESS-RT-003
def test_hook_output_does_not_depend_on_the_queue_contents():
    async def scenario(garbage):
        hf = make_hf_config(default_lookahead=0)
        queue: asyncio.Queue = asyncio.Queue()
        if garbage:
            for i in range(20):
                queue.put_nowait([5, 6, 7, i])
        return await collect(
            model_class().buffer_realtime_audio(
                stream([ramp(1280 * 2 + 3)]), queue, make_model_config(hf)
            )
        )

    plain = [unit(p) for p in run(scenario(False))]
    noisy = [unit(p) for p in run(scenario(True))]
    assert [u.header[:6] for u in plain] == [u.header[:6] for u in noisy]
    for a, b in zip(plain, noisy, strict=True):
        np.testing.assert_array_equal(a.samples, b.samples)


# @spec SESS-PACK-008
def test_hook_lets_the_packer_capacity_error_propagate(monkeypatch):
    # Never a normal close: the capacity error escapes the prompt stream.
    from vllm_nemotron_asr_streaming.session.packer import Packer

    from vllm_nemotron_asr_streaming.session import PackerCapacityError

    def exhausted(self, samples):
        raise PackerCapacityError("sequence space exhausted")

    monkeypatch.setattr(Packer, "append", exhausted)
    with pytest.raises(PackerCapacityError):
        run(run_hook([ramp(1280 * 8)], make_hf_config(default_lookahead=0)))


# @spec SESS-EP-001
def test_hook_prompts_carry_no_endpoint_policy():
    from support.fake_engine import contains_key

    hf = make_hf_config(default_lookahead=0)
    prompts = run(run_hook([ramp(1280 * 2 + 5)], hf))
    assert prompts and not any(contains_key(p, "endpoint") for p in prompts)


# @spec SESS-ENG-001, REG-CARD-016
def test_model_class_bound_is_the_manifest_definition():
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    assert REALTIME_MAX_TOKENS == 142
    assert model_class().realtime_max_tokens == REALTIME_MAX_TOKENS
