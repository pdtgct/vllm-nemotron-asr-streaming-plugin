# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest

from tests.fakes import EngineCoreClient, drop_cached_plugin_modules, install


def _factory():
    install()
    drop_cached_plugin_modules()
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import EngineCoreClient as Core

    from vllm_nemotron_asr_streaming.session import StreamingSessionFactory

    core = Core()
    llm = AsyncLLM(core)
    return llm, core, StreamingSessionFactory


def test_factory_binds_engine_core_admit_stream_async() -> None:
    llm, _, factory_cls = _factory()
    factory = factory_cls(llm)
    assert factory._admit_stream_async == llm.engine_core.admit_stream_async


def test_factory_rejects_a_lookalike() -> None:
    _, _, factory_cls = _factory()

    class Lookalike:
        def __init__(self) -> None:
            self.engine_core = self
            self.vllm_config = object()

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


def test_factory_refuses_rfc12_defaults() -> None:
    install()
    drop_cached_plugin_modules()
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core_client import EngineCoreClient as Core

    from tests.fakes import _Namespace, safe_vllm_config
    from vllm_nemotron_asr_streaming.session import StreamingSessionFactory

    config = safe_vllm_config()
    config.scheduler_config = _Namespace(async_scheduling=None)
    config.cache_config = _Namespace(enable_prefix_caching=True)
    llm = AsyncLLM(Core(), vllm_config=config)
    with pytest.raises(ValueError, match="async scheduling"):
        StreamingSessionFactory(llm)


def test_open_feed_flush_finish_abort_release() -> None:
    llm, core, factory_cls = _factory()
    factory = factory_cls(llm)

    async def _exercise() -> None:
        session = await factory.open("session-1", flush_token_id=13091)
        assert core.admitted["session-1"].prompt_token_ids == []
        assert core.admitted["session-1"].resumable is True
        assert core.admitted["session-1"].stream_admission is None
        assert core.admitted["session-1"].sampling_params.n == 1

        again = await factory.open("session-1", flush_token_id=13091)
        assert again.request_id == "session-1"
        assert core.admitted["session-1"] is session._admission_payload

        await session.feed([7, 8])
        update = core.updates[-1]
        assert update.prompt_token_ids == [7, 8]
        assert update.resumable is True
        assert update.stream_admission.input_closed is False

        await session.flush()
        flushed = core.updates[-1]
        assert flushed.prompt_token_ids == [13091]
        assert flushed.stream_admission.input_closed is False

        await session.finish()
        closed = core.updates[-1]
        assert closed.resumable is False
        assert closed.stream_admission.input_closed is True

        await session.abort()
        assert core.aborts == ["session-1"]
        completion = await session.release()
        assert type(completion).__name__ == "StateCompletion"

    asyncio.run(_exercise())


def test_release_without_a_forwarded_fence_does_not_invent_one() -> None:
    llm, core, factory_cls = _factory()
    core.wait_state_completion = None
    factory = factory_cls(llm)

    async def _exercise() -> None:
        session = await factory.open("session-2")
        await session.abort()
        with pytest.raises(RuntimeError, match="will not invent"):
            await session.release()

    asyncio.run(_exercise())


def test_open_surfaces_a_refusal() -> None:
    llm, core, factory_cls = _factory()

    async def refuse(request):
        from tests.fakes import StreamAdmissionResult

        return StreamAdmissionResult(
            False, request.request_id, "refused", "capacity"
        )

    core.admit_stream_async = refuse
    factory = factory_cls(llm)

    async def _exercise() -> None:
        from vllm_nemotron_asr_streaming.session.factory import (
            StreamAdmissionError,
        )

        with pytest.raises(StreamAdmissionError, match="capacity"):
            await factory.open("session-3")

    asyncio.run(_exercise())


def test_engine_core_client_symbol_is_the_fake() -> None:
    assert EngineCoreClient.__name__ == "EngineCoreClient"
