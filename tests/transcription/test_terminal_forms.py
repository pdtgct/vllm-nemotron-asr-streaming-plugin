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
terminal form is reachable there; the last test checks that refusal.
"""

from __future__ import annotations

import pytest
from support.stt_route import events, output, post, route_factory

pytestmark = pytest.mark.engine


@pytest.fixture
def client(card_dir):
    """``client(script)`` → ``(engine, TestClient)`` of core's routes."""
    return route_factory(card_dir)


PARTIAL = [output("hello", [5], None)]


def ending(reason: str):
    """The file's last output: a server-caused ``length`` (core's
    ``max_model_len`` stops the request; the client set no completion limit,
    which the model refuses anyway), an ``error`` or an ``abort``."""
    return output(" world", [6], reason)


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
    engine, test_client = client([output("hello world", [5, 6], "stop")])
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


# --------------------------------------------------------------------------
# Translation route
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
# @spec FILE-BUD-004
def test_translation_route_refuses_before_the_engine(client, stream):
    engine, test_client = client([*PARTIAL, ending("length")])
    response = post(test_client, stream=stream, route="translations")
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "error" in response.json()
    assert engine.prompts == [], "the translation request reached the engine"
