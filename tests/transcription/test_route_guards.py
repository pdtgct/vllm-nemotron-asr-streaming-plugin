# SPDX-License-Identifier: Apache-2.0
"""Request-guard refusals through core's transcription route (FILE-GRD-008).

Core's router, serving class and exception handlers, with a recording engine
client (``support.stt_route``). A refused request must get HTTP 400 with a
JSON error body, never an event stream, and must never reach the engine, so
no state slot or host buffer can be attributed to it.
"""

from __future__ import annotations

import pytest
from support.stt_route import output, post, route_factory

pytestmark = pytest.mark.engine

#: Fields the model refuses (FILE-GRD-001, FILE-GRD-002), as form values.
REFUSED = {
    "beam": {"use_beam_search": True},
    "limit": {"max_completion_tokens": 64},
    "limit-zero": {"max_completion_tokens": 0},
}


@pytest.fixture
def client(card_dir):
    return route_factory(card_dir)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("field", list(REFUSED))
# @spec FILE-GRD-008
def test_refused_request_gets_a_400_json_error_and_never_reaches_the_engine(
    client, field, stream
):
    engine, test_client = client([output("hello", [5, 6], "stop")])
    response = post(test_client, stream=stream, **REFUSED[field])
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    assert "data:" not in response.text, "an event stream started"
    body = response.json()
    assert "error" in body
    if not (stream and field == "beam"):
        # core refuses beam search with stream=true itself, before the model
        name = next(iter(REFUSED[field]))
        assert f"{name} is not supported for this model" in body["error"]["message"]
    assert engine.prompts == [], "the refused request reached the engine"
    assert engine.rendered == [], "the refused request was rendered for the engine"


# @spec FILE-GRD-008
def test_an_accepted_request_does_reach_the_engine(client):
    """Control for the spy: the same route without a refused field."""
    engine, test_client = client([output("hello", [5, 6], "stop")])
    response = post(test_client, stream=False)
    assert response.status_code == 200, response.text
    assert len(engine.prompts) == 1
