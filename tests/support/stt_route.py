# SPDX-License-Identifier: Apache-2.0
"""Core's speech-to-text routes in-process, with a scripted engine client.

The routes are core's own: the FastAPI routers
(``vllm/entrypoints/speech_to_text/{transcription,translation}/api_router.py``),
the serving classes and the exception handlers (``init_exception_handler``),
with the plugin's model class resolved from the served configuration as at
startup. Only the engine client is a double: it records every submission and
replies with the scripted outputs.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MODEL = "nemotron"


def wav_bytes(seconds: float = 0.5) -> bytes:
    samples = np.random.default_rng(1).standard_normal(int(16000 * seconds)) * 0.1
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


def output(text: str, token_ids: list[int], finish_reason: str | None) -> Any:
    from vllm.outputs import CompletionOutput, RequestOutput

    return RequestOutput(
        request_id="r",
        prompt=None,
        prompt_token_ids=[0],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=finish_reason is not None,
    )


class ScriptedEngine:
    """The engine-client surface core's speech-to-text serving uses."""

    errored = False
    dead_error = RuntimeError("dead")
    vllm_config = None
    input_processor = None

    def __init__(self, model_config: Any, script: list) -> None:
        self.model_config = model_config
        self.script = script
        self.aborted: list = []
        self.prompts: list = []
        self.rendered: list = []

        async def render(prompts):
            self.rendered.extend(prompts)
            return list(prompts)

        self.renderer = SimpleNamespace(render_cmpl_async=render)

    def check_admission(self, n: int = 1) -> None:
        return None

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.prompts.append(prompt)
        for item in self.script:
            yield item

    async def abort(self, request_ids, *args, **kwargs):
        self.aborted.append(request_ids)

    async def beam_search(self, *args, **kwargs):  # pragma: no cover - guard
        self.prompts.append(("beam_search", args, kwargs))
        if False:
            yield None


def route_factory(card_dir: Path):
    """``make(script)`` → ``(engine, TestClient)`` serving both routes."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.config import ModelConfig
    from vllm.entrypoints.serve.exception_handling.register import (
        init_exception_handler,
    )
    from vllm.entrypoints.speech_to_text.transcription.api_router import (
        router as transcription_router,
    )
    from vllm.entrypoints.speech_to_text.transcription.serving import (
        OpenAIServingTranscription,
    )
    from vllm.entrypoints.speech_to_text.translation.api_router import (
        router as translation_router,
    )
    from vllm.entrypoints.speech_to_text.translation.serving import (
        OpenAIServingTranslation,
    )

    from vllm_nemotron_asr_streaming import register

    register()
    model_config = ModelConfig(
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": 6},
        enforce_eager=True,
        dtype="float32",
        max_model_len=2048,
    )
    models = SimpleNamespace(
        model_name=lambda: MODEL,
        is_base_model=lambda name: name == MODEL,
        lora_requests={},
    )

    def make(script: list):
        engine = ScriptedEngine(model_config, script)
        app = FastAPI()
        app.include_router(transcription_router)
        app.include_router(translation_router)
        init_exception_handler(app)
        # core's handlers read the server arguments (``log_error_stack``)
        app.state.args = SimpleNamespace(log_error_stack=False)
        app.state.openai_serving_transcription = OpenAIServingTranscription(
            engine, models, request_logger=None
        )
        app.state.openai_serving_translation = OpenAIServingTranslation(
            engine, models, request_logger=None
        )
        return engine, TestClient(app, raise_server_exceptions=False)

    return make


def post(
    test_client: Any,
    *,
    stream: bool,
    include_usage: bool = False,
    route: str = "transcriptions",
    **fields: Any,
) -> Any:
    data = {"model": MODEL, "language": "en", "stream": str(stream).lower()}
    if include_usage:
        data["stream_include_usage"] = "true"
    data.update(
        {
            k: str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in fields.items()
        }
    )
    return test_client.post(
        f"/v1/audio/{route}",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data=data,
    )


def events(response: Any) -> list:
    """The SSE events of a streaming response: parsed JSON, or ``"[DONE]"``."""
    out = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out
