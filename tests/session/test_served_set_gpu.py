# SPDX-License-Identifier: Apache-2.0
"""Served-set enforcement in the model (SESS-MODEL-*): installed engine on the
published card, a unit sent straight to ``AsyncLLM`` with a geometry the
stream did not start with, or the engine does not serve.

The A40 round installs the plugin with ``pip install -e .`` so the engine-core
process loads the ``vllm.general_plugins`` entry point; the fixture also calls
``register()`` (idempotent) for the front-end process. Units for another
geometry are built with the session packer: correct ``valid_samples`` for the
destination geometry and the stream's next ``chunk_sequence``, so the model's
served-set check, not the envelope check, is what fails the row."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.card, pytest.mark.engine]

#: Served set of the engine under test; 1120 ms (geometry 4) is not served.
SERVED = [3, 6]
DEFAULT = 3
SERVED_GEOMETRY = 3  # right context 6
UNSERVED_GEOMETRY = 4  # right context 13
SERVED_RIGHT_CONTEXT = 6
UNSERVED_RIGHT_CONTEXT = 13


@pytest.fixture(scope="module")
def served_engine():
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_nemotron_asr_streaming import register

    register()
    loop = asyncio.new_event_loop()
    args = AsyncEngineArgs(
        model=card,
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        max_model_len=4096,
        hf_overrides={
            "num_lookahead_tokens": DEFAULT,
            "supported_num_lookahead_tokens": SERVED,
        },
    )

    async def start():
        return AsyncLLM.from_engine_args(args)

    llm = loop.run_until_complete(start())
    yield SimpleNamespace(llm=llm, loop=loop)
    llm.shutdown()
    loop.close()


def default_prompts(llm, units: int) -> list[Any]:
    """Units at the default lookahead, from the model's own realtime hook."""
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    s = 1280 * (DEFAULT + 1)
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(s * units) * 0.01).astype(np.float32)

    async def stream():
        yield audio

    async def collect():
        hook = Nemotron3_5AsrStreamingForRNNT.buffer_realtime_audio
        return [p async for p in hook(stream(), asyncio.Queue(), llm.model_config)]

    return asyncio.run(collect())


def with_geometry(prompt: Any, right_context: int, geometry: int, sequence: int):
    """A copy of ``prompt`` whose audio row is a regular unit of the
    destination geometry (``1280 * (right_context + 1)`` samples), packed by
    the session packer with the given ``chunk_sequence`` and the prompt's own
    locale row and carrier token."""
    from support.fake_engine import unit_of

    from vllm_nemotron_asr_streaming.session.packer import Packer

    template = unit_of(SimpleNamespace(prompt=prompt))
    packer = Packer(
        right_context=right_context,
        geometry_index=geometry,
        locale_row=template.prompt_index,
        carrier_token_id=template.token_ids[0],
        clock_ms=lambda: template.admission_ms_mod,
        first_chunk_sequence=sequence,
    )
    samples = np.full(1280 * (right_context + 1), 0.01, dtype=np.float32)
    (unit,) = packer.append(samples)
    return {
        "prompt_token_ids": list(unit.prompt_token_ids),
        "multi_modal_data": {"audio": unit.row},
    }


async def drive(served, prompts: list[Any]):
    """Send ``prompts`` as one admitted stream, one at a time, each after the
    previous unit's park. Return the outputs received."""
    import uuid

    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    llm = served.llm
    park = llm.model_config.hf_config.eos_token_id
    parked = asyncio.Event()
    ended = asyncio.Event()

    async def inputs():
        for prompt in prompts:
            parked.clear()
            yield StreamingInput(prompt=prompt)
            await asyncio.wait_for(_first(parked, ended), timeout=120)
            if ended.is_set():
                return

    params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=REALTIME_MAX_TOKENS,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=True,
    )
    outputs = []
    async for out in llm.generate(
        inputs(), params, str(uuid.uuid4()), stream_admission=True
    ):
        outputs.append(out)
        completion = out.outputs[0]
        if list(completion.token_ids)[-1:] == [park]:
            parked.set()
        if out.finished:
            ended.set()
    return outputs


async def _first(a: asyncio.Event, b: asyncio.Event) -> None:
    tasks = [asyncio.ensure_future(a.wait()), asyncio.ensure_future(b.wait())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()


def finish_reason(outputs) -> str | None:
    return outputs[-1].outputs[0].finish_reason


# @spec SESS-MODEL-001
def test_a_later_unit_with_a_different_geometry_fails_its_row(served_engine):
    prompts = default_prompts(served_engine.llm, 2)
    # Served, valid for its own geometry, but not the stream's.
    prompts[1] = with_geometry(prompts[1], SERVED_RIGHT_CONTEXT, SERVED_GEOMETRY, 1)
    park = served_engine.llm.model_config.hf_config.eos_token_id
    outputs = served_engine.loop.run_until_complete(drive(served_engine, prompts))
    assert outputs[-1].finished and finish_reason(outputs) == "error"
    before = [o for o in outputs[:-1] if o.outputs[0].token_ids]
    assert list(before[-1].outputs[0].token_ids)[-1] == park, (
        "the first unit ended with the park token before the second failed"
    )


# @spec SESS-MODEL-001
def test_a_stream_of_one_geometry_still_runs_to_a_normal_close(served_engine):
    prompts = default_prompts(served_engine.llm, 2)
    outputs = served_engine.loop.run_until_complete(drive(served_engine, prompts))
    assert outputs[-1].finished and finish_reason(outputs) == "stop"


# @spec SESS-MODEL-002
def test_a_first_unit_with_an_unserved_geometry_fails_its_row(served_engine):
    prompts = default_prompts(served_engine.llm, 1)
    prompts[0] = with_geometry(prompts[0], UNSERVED_RIGHT_CONTEXT, UNSERVED_GEOMETRY, 0)
    outputs = served_engine.loop.run_until_complete(drive(served_engine, prompts))
    assert outputs[-1].finished and finish_reason(outputs) == "error"
