# SPDX-License-Identifier: Apache-2.0
"""Terminal forms of a file request on the stage 1 core's transcription route
(FILE-BUD-004).

The route is core's own: its FastAPI router
(``vllm/entrypoints/speech_to_text/transcription/api_router.py``), its
serving class (``OpenAIServingTranscription``) and its exception handlers
(``init_exception_handler``), with the plugin's model class resolved from the
served configuration as at startup. Only the engine client is a fake: it
scripts the request's outputs, ending with the finish reason under test.

The translation route shares the base serving, but the model refuses every
translation request before it reaches the engine (FILE-IF-004), so no
terminal form is reachable there; it is not exercised.
"""

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.engine

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


def _output(text: str, token_ids: list[int], finish_reason: str | None):
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
    """The engine-client surface core's transcription serving uses."""

    errored = False
    dead_error = RuntimeError("dead")
    vllm_config = None
    input_processor = None

    def __init__(self, model_config, script):
        self.model_config = model_config
        self.script = script
        self.aborted: list = []
        self.prompts: list = []

        async def render(prompts):
            return list(prompts)

        self.renderer = SimpleNamespace(render_cmpl_async=render)

    def check_admission(self, n: int = 1) -> None:
        return None

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.prompts.append(prompt)
        for output in self.script:
            yield output

    async def abort(self, request_ids, *args, **kwargs):
        self.aborted.append(request_ids)


@pytest.fixture
def client(card_dir):
    """``client(script)`` → a test client of core's transcription route."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.config import ModelConfig
    from vllm.entrypoints.serve.exception_handling.register import (
        init_exception_handler,
    )
    from vllm.entrypoints.speech_to_text.transcription.api_router import router
    from vllm.entrypoints.speech_to_text.transcription.serving import (
        OpenAIServingTranscription,
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

    def make(script):
        engine = ScriptedEngine(model_config, script)
        app = FastAPI()
        app.include_router(router)
        init_exception_handler(app)
        # core's handlers read the server arguments (``log_error_stack``)
        app.state.args = SimpleNamespace(log_error_stack=False)
        app.state.openai_serving_transcription = OpenAIServingTranscription(
            engine, models, request_logger=None
        )
        return engine, TestClient(app, raise_server_exceptions=False)

    return make


def post(test_client, *, stream: bool, include_usage: bool = False):
    data = {"model": MODEL, "language": "en", "stream": str(stream).lower()}
    if include_usage:
        data["stream_include_usage"] = "true"
    return test_client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data=data,
    )


def events(response) -> list:
    """The SSE events of a streaming response: parsed JSON, or ``"[DONE]"``."""
    out = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


PARTIAL = [_output("hello", [5], None)]


def ending(reason: str):
    """The file's last output: a server-caused ``length`` (core's
    ``max_model_len`` stops the request; the client set no completion limit,
    which the model refuses anyway), an ``error`` or an ``abort``."""
    return _output(" world", [6], reason)


# --------------------------------------------------------------------------
# Non-streaming
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason,status", [("length", 422), ("error", 500), ("abort", 503)]
)
# @spec FILE-BUD-004
def test_non_streaming_terminal_form_is_an_error_without_partial_text(
    client, reason, status
):
    _, test_client = client([*PARTIAL, ending(reason)])
    response = post(test_client, stream=False)
    assert response.status_code == status, response.text
    assert "error" in response.json()
    assert "hello" not in response.text and "world" not in response.text


# @spec FILE-BUD-004
def test_non_streaming_eos_returns_the_transcript(client):
    """Control: an EOS ending (``stop``) is a normal response, including one
    at the ``max_model_len`` boundary, since core checks EOS first."""
    engine, test_client = client([_output("hello world", [5, 6], "stop")])
    response = post(test_client, stream=False)
    assert len(engine.prompts) == 1, "the file did not reach the engine"
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hello world"


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["length", "error"])
# @spec FILE-BUD-004
def test_streaming_length_or_error_ends_with_an_error_event_then_done(client, reason):
    _, test_client = client([*PARTIAL, ending(reason)])
    stream = events(post(test_client, stream=True))
    assert stream[-1] == "[DONE]"
    error = stream[-2]
    assert isinstance(error, dict) and "error" in error, stream
    deltas = stream[:-2]
    assert all("choices" in e for e in deltas), stream
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in deltas) == (
        "hello"
    )
    assert all(e["choices"][0].get("finish_reason") is None for e in deltas)


@pytest.mark.parametrize("include_usage", [False, True], ids=["no-usage", "usage"])
# @spec FILE-BUD-004
def test_streaming_abort_marks_the_last_chunk_then_usage_then_done(
    client, include_usage
):
    _, test_client = client([*PARTIAL, ending("abort")])
    stream = events(post(test_client, stream=True, include_usage=include_usage))
    assert stream[-1] == "[DONE]"
    body = stream[:-1]
    assert not any("error" in e for e in body), stream
    if include_usage:
        usage = body.pop()
        assert usage["choices"] == [] and "usage" in usage, stream
    last = body[-1]
    assert last["choices"][0]["finish_reason"] == "abort", stream
    assert all(e["choices"][0].get("finish_reason") is None for e in body[:-1])
