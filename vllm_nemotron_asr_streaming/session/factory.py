# SPDX-License-Identifier: Apache-2.0
"""Session factory bound to core ``AsyncLLM``.

Admission on the RFC1 tip is ``EngineCoreClient.admit_stream_async``.
``AsyncLLM`` holds that client on ``engine_core`` and does not define
``admit_stream_async`` itself. This factory accepts only an ``AsyncLLM``
and stores that method. It never calls it: ``open`` / ``feed`` /
``flush`` / ``finish`` / ``release`` / ``abort`` are still unimplemented.

This module imports vLLM core only.
"""

from __future__ import annotations

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import StateCompletion


def _engine_core_admit(llm: AsyncLLM):
    if not isinstance(llm, AsyncLLM):
        raise TypeError(
            "streaming sessions bind only to "
            "vllm.v1.engine.async_llm.AsyncLLM"
        )
    engine_core = getattr(llm, "engine_core", None)
    admit = getattr(engine_core, "admit_stream_async", None)
    if not callable(admit):
        raise TypeError(
            "AsyncLLM.engine_core.admit_stream_async is missing. "
            "This package targets pdtgct/vllm persistent-state-primitive-v2 "
            "at 8b8327e2ee, not a released vLLM."
        )
    return admit


class StreamingSession:
    """One admitted stream. Construction does not admit a request."""

    def __init__(self, llm: AsyncLLM, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._admit_stream_async = _engine_core_admit(llm)
        self._llm = llm
        self.request_id = request_id

    async def feed(self, payload: object) -> None:
        """Push the next audio input. Not implemented."""
        del payload
        raise NotImplementedError(
            "TODO: feed() sends a resumable update whose stream_admission "
            "context names this request id. See TODO.md."
        )

    async def flush(self) -> None:
        """Drain the current chunk without closing the slot. Not implemented."""
        raise NotImplementedError(
            "TODO: flush() is the source flush control, not session close. "
            "See TODO.md."
        )

    async def finish(self) -> None:
        """Close input and let the stream drain. Not implemented."""
        raise NotImplementedError(
            "TODO: finish() closes input (input_closed or a non-resumable "
            "update) and waits for drain. See TODO.md."
        )

    async def release(self) -> StateCompletion:
        """Wait for the release fence. Not implemented.

        The scheduler keeps the slot until a ``StateCompletion`` covers
        the request's last scheduled step and its cleanup handle.
        """
        raise NotImplementedError(
            "TODO: release() waits for a "
            f"{StateCompletion.__name__} that covers the last scheduled "
            "step and the cleanup handle. See TODO.md."
        )

    async def abort(self) -> None:
        """Abort the request id and wait for the same fence. Not implemented."""
        raise NotImplementedError(
            "TODO: abort() calls abort_requests_async for this request id, "
            "then waits for the StateCompletion fence. The id is not reused. "
            "See TODO.md."
        )


class StreamingSessionFactory:
    """Binds a core ``AsyncLLM`` and will admit sessions through it."""

    def __init__(self, llm: AsyncLLM) -> None:
        self._admit_stream_async = _engine_core_admit(llm)
        self._llm = llm

    def open(self, request_id: str) -> StreamingSession:
        """Admit an empty stream under ``request_id``. Not implemented.

        The id is caller-supplied. ``admit_stream`` does not go through
        the input processor, so the engine will not add a random suffix.
        An id that is in use, including one still retiring, is refused.
        """
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        raise NotImplementedError(
            "TODO: open() builds an empty resumable EngineCoreRequest and "
            "calls AsyncLLM.engine_core.admit_stream_async. See TODO.md."
        )
