# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from tests.fakes import drop_cached_plugin_modules, install


def _factory():
    install()
    drop_cached_plugin_modules()
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import EngineCoreClient

    from vllm_nemotron_asr_streaming.session import (
        StreamingSession,
        StreamingSessionFactory,
    )

    llm = AsyncLLM(EngineCoreClient())
    return llm, StreamingSession, StreamingSessionFactory


def test_factory_binds_engine_core_admit_stream_async() -> None:
    llm, _, factory_cls = _factory()
    factory = factory_cls(llm)
    assert factory._admit_stream_async == llm.engine_core.admit_stream_async


def test_factory_rejects_a_lookalike() -> None:
    _, _, factory_cls = _factory()

    class Lookalike:
        def __init__(self) -> None:
            self.engine_core = self

        async def admit_stream_async(self, request):
            return request

    with pytest.raises(TypeError, match="AsyncLLM"):
        factory_cls(Lookalike())


def test_factory_rejects_async_llm_without_admit() -> None:
    llm, _, factory_cls = _factory()

    class BareCore:
        pass

    llm.engine_core = BareCore()
    with pytest.raises(TypeError, match="admit_stream_async"):
        factory_cls(llm)


def test_open_and_session_methods_are_unimplemented() -> None:
    llm, session_cls, factory_cls = _factory()
    factory = factory_cls(llm)
    with pytest.raises(ValueError):
        factory.open("")
    with pytest.raises(NotImplementedError, match="admit_stream_async"):
        factory.open("session-1")

    session = session_cls(llm, "session-1")
    assert session._admit_stream_async == llm.engine_core.admit_stream_async

    async def _exercise() -> None:
        with pytest.raises(NotImplementedError):
            await session.feed(b"")
        with pytest.raises(NotImplementedError, match="flush"):
            await session.flush()
        with pytest.raises(NotImplementedError, match="finish"):
            await session.finish()
        with pytest.raises(NotImplementedError, match="StateCompletion"):
            await session.release()
        with pytest.raises(NotImplementedError, match="abort"):
            await session.abort()

    import asyncio

    asyncio.run(_exercise())
