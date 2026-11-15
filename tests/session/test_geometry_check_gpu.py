# SPDX-License-Identifier: Apache-2.0
"""The model's geometry check and row-status failures (SESS-MODEL-001 to
SESS-MODEL-003) on an installed engine with the published card: units sent
straight to ``AsyncLLM`` (a direct engine client, the only kind that can send
another geometry) at the server's geometry and at others.

The server runs at 160 ms (``num_lookahead_tokens`` 1), with transaction
graphs off (``--enforce-eager``) and on (the hook configuration with
``transaction_graph_populations``; TXG-CFG-006 serves them only at 80 and
160 ms). Units for another geometry are built with the session packer:
correct ``valid_samples`` for that geometry and the stream's next
``chunk_sequence``, so the geometry check, not the envelope check, is what
fails the row.

The A40 round installs the plugin with ``pip install -e .`` so the engine-core
process loads the ``vllm.general_plugins`` entry point; the fixture also calls
``register()`` (idempotent) for the front-end process.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.card, pytest.mark.engine]

#: The server's geometry: right context 1 (160 ms), geometry index 1.
SERVER_RIGHT_CONTEXT = 1
#: Another implemented geometry for a later unit: 560 ms (index 3).
LATER_RIGHT_CONTEXT, LATER_GEOMETRY = 6, 3
#: Another implemented geometry for a first unit: 1120 ms (index 4).
FIRST_RIGHT_CONTEXT, FIRST_GEOMETRY = 13, 4

ARMS = {
    "graphs-off": dict(
        enforce_eager=True,
        hf_overrides={"num_lookahead_tokens": SERVER_RIGHT_CONTEXT},
    ),
    "graphs-on": dict(
        enforce_eager=False,
        compilation_config={"mode": "none", "cudagraph_mode": "none"},
        max_num_seqs=4,
        hf_overrides={
            "num_lookahead_tokens": SERVER_RIGHT_CONTEXT,
            "transaction_graph_populations": [4],
        },
    ),
}


@pytest.fixture(scope="module", params=list(ARMS))
def served_engine(request):
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
        async_scheduling=False,
        enable_prefix_caching=False,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        **ARMS[request.param],
    )

    async def start():
        return AsyncLLM.from_engine_args(args)

    llm = loop.run_until_complete(start())
    yield SimpleNamespace(llm=llm, loop=loop, arm=request.param)
    llm.shutdown()
    loop.close()


def server_prompts(served, units: int, seed: int = 0) -> list[Any]:
    """Units at the server's geometry, from the model's own realtime hook."""
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    s = 1280 * (SERVER_RIGHT_CONTEXT + 1)
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(s * units) * 0.01).astype(np.float32)

    async def stream():
        yield audio

    async def collect():
        hook = Nemotron3_5AsrStreamingForRNNT.buffer_realtime_audio
        return [
            p async for p in hook(stream(), asyncio.Queue(), served.llm.model_config)
        ]

    return served.loop.run_until_complete(collect())


def repacked(prompt: Any, right_context: int, geometry: int, sequence: int):
    """A copy of ``prompt`` whose audio row is a regular unit of another
    geometry (``1280 * (right_context + 1)`` samples), packed by the session
    packer with the given ``chunk_sequence`` and the prompt's own locale row,
    carrier token and stamp."""
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


def parked_units(served, outputs) -> int:
    park = served.llm.model_config.hf_config.eos_token_id
    return sum(1 for o in outputs if list(o.outputs[0].token_ids)[-1:] == [park])


def run_streams(served, *streams):
    async def both():
        return await asyncio.gather(*(drive(served, p) for p in streams))

    return served.loop.run_until_complete(both())


# @spec SESS-MODEL-001, SESS-GEOM-001
def test_a_stream_at_the_servers_geometry_runs_to_a_normal_close(served_engine):
    (outputs,) = run_streams(served_engine, server_prompts(served_engine, 2))
    assert outputs[-1].finished and finish_reason(outputs) == "stop"
    assert parked_units(served_engine, outputs) == 2


# @spec SESS-MODEL-001, SESS-MODEL-003
def test_a_later_unit_at_another_geometry_fails_its_row_alone(served_engine):
    faulty = server_prompts(served_engine, 2, seed=1)
    faulty[1] = repacked(faulty[1], LATER_RIGHT_CONTEXT, LATER_GEOMETRY, 1)
    clean = server_prompts(served_engine, 3, seed=2)
    bad, good = run_streams(served_engine, faulty, clean)
    assert bad[-1].finished and finish_reason(bad) == "error"
    assert parked_units(served_engine, bad) == 1, (
        "the first unit parked before the second failed"
    )
    # The other stream, in the same steps, is unaffected.
    assert good[-1].finished and finish_reason(good) == "stop"
    assert parked_units(served_engine, good) == 3


# @spec SESS-MODEL-002
@pytest.mark.parametrize(
    "right_context,geometry",
    [(FIRST_RIGHT_CONTEXT, FIRST_GEOMETRY), (0, 0)],
    ids=["1120ms", "80ms"],
)
def test_a_first_unit_at_another_geometry_fails_its_row(
    served_engine, right_context, geometry
):
    faulty = server_prompts(served_engine, 1, seed=3)
    faulty[0] = repacked(faulty[0], right_context, geometry, 0)
    clean = server_prompts(served_engine, 2, seed=4)
    bad, good = run_streams(served_engine, faulty, clean)
    assert bad[-1].finished and finish_reason(bad) == "error"
    assert parked_units(served_engine, bad) == 0
    assert good[-1].finished and finish_reason(good) == "stop"


# @spec SESS-MODEL-003
def test_a_row_status_of_another_class_fails_that_request_alone(served_engine):
    """A later unit at the server's geometry whose ``chunk_sequence`` skips
    ahead: the transaction's sequence status (a collected row status of
    another class than geometry) fails that request with ``error``; the
    step's other row continues to a normal close."""
    faulty = server_prompts(served_engine, 2, seed=5)
    faulty[1] = repacked(
        faulty[1], SERVER_RIGHT_CONTEXT, SERVER_RIGHT_CONTEXT, sequence=7
    )
    clean = server_prompts(served_engine, 3, seed=6)
    bad, good = run_streams(served_engine, faulty, clean)
    assert bad[-1].finished and finish_reason(bad) == "error"
    assert parked_units(served_engine, bad) == 1
    assert good[-1].finished and finish_reason(good) == "stop"
    assert parked_units(served_engine, good) == 3
