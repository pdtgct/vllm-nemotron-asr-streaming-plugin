# SPDX-License-Identifier: Apache-2.0
"""Session factory bound to core ``AsyncLLM``.

Admission is ``EngineCoreClient.admit_stream_async``. ``AsyncLLM`` holds
that client on ``engine_core``. Updates go through
``add_request_async`` with ``StreamAdmissionContext``. Abort goes
through ``abort_requests_async``.

``StateCompletion`` is applied inside ``StateScheduler`` and is not a
field on the ``AsyncLLM`` output path in ``rfc12-integration``.
``release`` returns one only when the engine core exposes
``wait_state_completion``. It does not construct a fence object.

This module imports vLLM core only.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_cache_interface import StateCompletion

from vllm_nemotron_asr_streaming.engine_flags import persistent_state_config_errors


class StreamAdmissionError(RuntimeError):
    """``admit_stream_async`` refused the request."""

    def __init__(self, result: Any) -> None:
        self.result = result
        reason = getattr(result, "reason", None)
        request_id = getattr(result, "request_id", None)
        super().__init__(
            f"stream admission refused for {request_id!r}: {reason!r}"
        )


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
            "This package targets pdtgct/vllm rfc12-integration at "
            "4c180969 (tree 465f10ad65), not a released vLLM."
        )
    return admit


def _require_persistent_state_engine(llm: AsyncLLM) -> None:
    vllm_config = getattr(llm, "vllm_config", None)
    if vllm_config is None:
        raise TypeError(
            "AsyncLLM.vllm_config is required so the session can refuse "
            "RFC12's async-scheduling and prefix-caching defaults"
        )
    errors = persistent_state_config_errors(vllm_config)
    if errors:
        raise ValueError(
            "Nemotron3_5AsrStreamingForRNNT persistent state refuses: "
            + "; ".join(errors)
        )


def _sampling_params() -> Any:
    from vllm.sampling_params import SamplingParams

    # n=1, no stop strings, cumulative output. Temperature 0 keeps the
    # request on the greedy path the RNN-T replay expects.
    return SamplingParams(n=1, temperature=0.0, max_tokens=1, stop=None)


def _engine_request(
    request_id: str,
    *,
    arrival_time: float,
    prompt_token_ids: list[int],
    mm_features: list[Any] | None,
    resumable: bool,
    input_closed: bool,
    stream_admission: bool,
) -> Any:
    from vllm.v1.engine import EngineCoreRequest, StreamAdmissionContext

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        mm_features=mm_features,
        sampling_params=_sampling_params(),
        pooling_params=None,
        arrival_time=arrival_time,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
        stream_admission=(
            StreamAdmissionContext(input_closed=input_closed)
            if stream_admission
            else None
        ),
    )


class StreamingSession:
    """One admitted stream. Construction does not admit a request."""

    def __init__(
        self,
        llm: AsyncLLM,
        request_id: str,
        *,
        admission_payload: Any,
        flush_token_id: int | None = None,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._admit_stream_async = _engine_core_admit(llm)
        self._llm = llm
        self.request_id = request_id
        self._admission_payload = admission_payload
        self._flush_token_id = flush_token_id
        self._input_closed = False
        self._aborted = False

    async def feed(
        self,
        token_ids: list[int],
        *,
        mm_features: list[Any] | None = None,
        input_closed: bool = False,
    ) -> None:
        """Push the next input. Positions are relative to ``token_ids``.

        A resumable update with no tokens is an ``empty_update`` on the
        RFC12 scheduler. Closing input with an empty token list uses
        ``finish`` (a non-resumable update) instead.
        """
        if self._input_closed or self._aborted:
            raise RuntimeError(f"session {self.request_id!r} is closed")
        if not token_ids and not input_closed:
            raise ValueError("feed() requires at least one token id")
        request = _engine_request(
            self.request_id,
            arrival_time=time.time(),
            prompt_token_ids=list(token_ids),
            mm_features=mm_features,
            resumable=True,
            input_closed=input_closed,
            stream_admission=True,
        )
        await self._llm.engine_core.add_request_async(request)
        if input_closed:
            self._input_closed = True

    async def flush(self) -> None:
        """Send the served ``flush_token_id`` without closing the slot.

        The source config mints that id as ``blank + 4``. The forward
        path does not branch on the id; the session still sends it as
        the flush control token.
        """
        if self._flush_token_id is None:
            raise ValueError(
                "flush() requires flush_token_id from the served config "
                "(blank_token_id + 4)"
            )
        await self.feed([self._flush_token_id])

    async def finish(self) -> None:
        """Close input with a non-resumable update and leave the slot draining.

        A resumable empty update is rejected as ``empty_update`` before
        ``input_closed`` is applied. ``resumable=False`` closes input.
        """
        if self._aborted:
            raise RuntimeError(f"session {self.request_id!r} is aborted")
        if self._input_closed:
            return
        request = _engine_request(
            self.request_id,
            arrival_time=time.time(),
            prompt_token_ids=[],
            mm_features=None,
            resumable=False,
            input_closed=True,
            stream_admission=True,
        )
        await self._llm.engine_core.add_request_async(request)
        self._input_closed = True

    async def release(self) -> StateCompletion:
        """Return the fence object that lets the slot be reused.

        RFC12 applies ``StateCompletion`` inside the scheduler and does
        not put it on ``AsyncLLM`` outputs. This waits on
        ``engine_core.wait_state_completion`` when that method exists.
        """
        if not self._input_closed and not self._aborted:
            raise RuntimeError("release() follows finish() or abort()")
        waiter = getattr(self._llm.engine_core, "wait_state_completion", None)
        if not callable(waiter):
            raise RuntimeError(
                "StateCompletion is consumed by StateScheduler on "
                "rfc12-integration@4c180969 and is not forwarded to "
                "AsyncLLM. release() will not invent a fence object."
            )
        completion = await waiter(self.request_id)
        if not isinstance(completion, StateCompletion):
            raise TypeError(
                "wait_state_completion must return StateCompletion, "
                f"got {type(completion).__name__}"
            )
        return completion

    async def abort(self) -> None:
        """Abort the request id. The id is not reused for a new session."""
        if self._aborted:
            return
        await self._llm.engine_core.abort_requests_async([self.request_id])
        self._aborted = True
        self._input_closed = True


class StreamingSessionFactory:
    """Binds a core ``AsyncLLM`` and admits sessions through it."""

    def __init__(self, llm: AsyncLLM) -> None:
        self._admit_stream_async = _engine_core_admit(llm)
        _require_persistent_state_engine(llm)
        self._llm = llm
        self._admissions: dict[str, Any] = {}

    async def open(
        self,
        request_id: str,
        *,
        flush_token_id: int | None = None,
        arrival_time: float | None = None,
    ) -> StreamingSession:
        """Admit an empty resumable request under ``request_id``.

        The id is caller-supplied. ``admit_stream`` does not go through
        the input processor, so the engine adds no random suffix. A
        second ``open`` of the same id resends the original payload,
        which is the retry rule: same bytes, deadline unchanged.
        """
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        payload = self._admissions.get(request_id)
        if payload is None:
            payload = _engine_request(
                request_id,
                arrival_time=time.time() if arrival_time is None else arrival_time,
                prompt_token_ids=[],
                mm_features=None,
                resumable=True,
                input_closed=False,
                stream_admission=False,
            )
            self._admissions[request_id] = payload
        result = await self._admit_stream_async(payload)
        if not getattr(result, "accepted", False):
            raise StreamAdmissionError(result)
        return StreamingSession(
            self._llm,
            request_id,
            admission_payload=payload,
            flush_token_id=flush_token_id,
        )
