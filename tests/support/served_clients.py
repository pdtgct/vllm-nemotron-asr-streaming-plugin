# SPDX-License-Identifier: Apache-2.0
"""Requests through an ``AsyncLLM`` engine as core's served routes send
them: a complete file as core's transcription serving submits it, and a
stream as core's ``/v1/realtime`` serving feeds it. Each returns the
request's text, token IDs and finish reason, and raises nothing a served
request would not raise; the caller decides what the outcome must be."""

from __future__ import annotations

import asyncio
import io
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FileResult:
    text: str
    token_ids: list[int]
    finish_reason: str | None


async def transcribe_file(
    engine: Any, samples: np.ndarray, *, extra_args: dict | None = None
) -> FileResult:
    """Submit a file the way core's transcription serving does."""
    from fastapi import UploadFile
    from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    stt = SpeechToTextParams(
        audio=samples,
        stt_config=SpeechToTextConfig(
            sample_rate=16000, max_audio_clip_s=None, min_energy_split_window_size=None
        ),
        model_config=engine.model_config,
        extra_args=extra_args,
    )
    prompt = Nemotron3_5AsrStreamingForRNNT.get_generation_prompt(stt)
    request = TranscriptionRequest(
        file=UploadFile(io.BytesIO(b"x"), filename="a.wav"), model="m"
    )
    # core: max_tokens = max_model_len minus the prompt length
    # (speech_to_text/base/serving.py:517, entrypoints/serve/utils/api_utils.py
    # get_max_tokens); the prompt is the one carrier token
    max_tokens = engine.model_config.max_model_len - len(prompt["prompt_token_ids"])
    params = request.to_sampling_params(max_tokens)
    final = None
    async for out in engine.generate(prompt, params, f"file-{uuid.uuid4().hex}"):
        final = out
    assert final is not None
    completion = final.outputs[0]
    return FileResult(
        completion.text, list(completion.token_ids), completion.finish_reason
    )


@dataclass
class StreamResult:
    text: str
    token_ids: list[int]
    finish_reason: str | None = None


async def transcribe_stream(
    engine: Any,
    samples: np.ndarray,
    piece: int = 4000,
    opened: asyncio.Event | None = None,
) -> StreamResult:
    """Feed the same samples as one stream on core's ``/v1/realtime`` path.

    The steps are those of core's realtime serving: the model's
    ``buffer_realtime_audio`` packs the audio, fed in pieces of ``piece``
    samples, into units at the server's geometry and default locale
    (SESS-GEOM-001) and closes with the final tail; core's renderer renders
    each prompt; ``AsyncLLM.generate`` receives them as ``StreamingInput``
    under the realtime route's sampling parameters, and each output's token
    IDs go back on the hook's input queue. These are the units a session
    sends for the same samples, with no ``update_locale``. A stream that
    flushes normally ends with ``stop`` after its final tail. ``opened`` is
    set once core has admitted the stream: core reads the input generator
    only after admission.
    """
    from vllm.engine.protocol import StreamingInput
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    model_config = engine.model_config
    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

    async def audio():
        for start in range(0, len(samples), piece):
            yield samples[start : start + piece]

    async def inputs():
        if opened is not None:
            opened.set()
        hook = Nemotron3_5AsrStreamingForRNNT.buffer_realtime_audio
        async for prompt in hook(audio(), input_stream, model_config):
            parsed = parse_model_prompt(model_config, prompt)
            (engine_input,) = await engine.renderer.render_cmpl_async([parsed])
            yield StreamingInput(prompt=engine_input)

    params = SamplingParams.from_optional(
        temperature=0.0,
        max_tokens=Nemotron3_5AsrStreamingForRNNT.realtime_max_tokens,
        output_kind=RequestOutputKind.DELTA,
        skip_clone=True,
    )
    text, token_ids, finish_reason = "", [], None
    async for out in engine.generate(
        prompt=inputs(), sampling_params=params, request_id=f"rt-{uuid.uuid4()}"
    ):
        if out.outputs:
            completion = out.outputs[0]
            text += completion.text
            token_ids.extend(completion.token_ids)  # delta output
            input_stream.put_nowait(list(completion.token_ids))
        if out.finished:
            finish_reason = out.outputs[0].finish_reason if out.outputs else None
    return StreamResult(text, token_ids, finish_reason)
