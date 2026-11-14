# SPDX-License-Identifier: Apache-2.0
"""The prompts the session packer and the realtime hook produce go through
core's real input processing (``InputProcessor.process_inputs`` with
``resumable=True`` and the multimodal processor), built on CPU from the
synthetic card. Exercises SESS-PACK-001 (one carrier token plus the audio
row) end to end instead of against the fake engine's own prompt check."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from support.fake_engine import run

pytestmark = pytest.mark.engine


def _input_processor(card_dir):
    from vllm.v1.engine.input_processor import InputProcessor

    from vllm_nemotron_asr_streaming import register

    register()
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    # EngineArgs.create_engine_config needs a platform device; build the
    # configuration directly so it constructs on CPU.
    model_config = ModelConfig(
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        enforce_eager=True,
        max_model_len=4096,
        hf_overrides={"num_lookahead_tokens": 3},
    )
    # The plugin's verify-and-update hook refuses prefix caching and
    # asynchronous scheduling (REG-ENG-001/002), so a conforming build needs
    # them off before VllmConfig post-init runs.
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
    )
    return InputProcessor(config), config


def _params():
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    return SamplingParams(
        n=1,
        temperature=0,
        max_tokens=REALTIME_MAX_TOKENS,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=True,
    )


def _field(feature):
    """The processed ``audio`` field as a flat float32 array."""
    item = feature.data["audio"]
    value = getattr(item, "data", item)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _check(request, hf, prompt):
    from support.fake_engine import unit_of

    sent = unit_of(SimpleNamespace(prompt=prompt))
    processed = _field(request.mm_features[0])
    # Serialized contents: header values and samples survive processing.
    assert processed.shape == (17_927,)
    np.testing.assert_array_equal(processed, np.asarray(sent.row).reshape(-1))
    assert [int(v) for v in processed[:7]] == sent.header
    np.testing.assert_array_equal(processed[7 : 7 + sent.valid_samples], sent.samples)

    assert list(request.prompt_token_ids) == [hf.audio_chunk_token_id]
    (feature,) = request.mm_features
    assert feature.modality == "audio"
    assert "audio" in feature.data
    assert request.resumable is True


# @spec SESS-PACK-001
def test_hook_prompts_pass_core_input_processing(card_dir):
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    processor, config = _input_processor(card_dir)
    hf = config.model_config.hf_config

    async def stream():
        yield (np.arange(5120 * 2 + 7, dtype=np.float32) % 7) * 1e-3

    async def collect():
        hook = Nemotron3_5AsrStreamingForRNNT.buffer_realtime_audio
        return [p async for p in hook(stream(), asyncio.Queue(), config.model_config)]

    prompts = run(collect())
    assert len(prompts) == 3
    for prompt in prompts:
        request = processor.process_inputs(
            request_id="r",
            prompt=prompt,
            params=_params(),
            supported_tasks=("generate", "realtime", "transcription"),
            resumable=True,
        )
        _check(request, hf, prompt)


# @spec SESS-PACK-001
def test_session_packer_prompts_pass_core_input_processing(card_dir, build):
    """The prompt a session sends to the engine, captured from the fake."""
    from support.fake_engine import FakeEngine

    processor, config = _input_processor(card_dir)
    hf = config.model_config.hf_config

    async def scenario():
        engine, factory = build(FakeEngine(blank=hf.blank_token_id), hf=hf)
        lease = await factory.open(cadence="320ms", locale=sorted(factory.locales)[0])
        await lease.feed(
            np.full(5120 + 11, 0.01, np.float32), on_accepted=lambda n: None
        )
        await lease.flush()
        return engine

    engine = run(scenario())
    assert len(engine.inputs) == 2
    for item in engine.inputs:
        request = processor.process_inputs(
            request_id="r",
            prompt=item.prompt,
            params=_params(),
            supported_tasks=("generate", "realtime", "transcription"),
            resumable=True,
        )
        _check(request, hf, item.prompt)
