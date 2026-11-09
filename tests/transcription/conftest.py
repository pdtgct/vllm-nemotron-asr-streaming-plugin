# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the transcription (FILE) tests.

``served_config`` builds the served ``hf_config`` the way the plugin does at
startup (the card parser's translation, then the two values
``--hf-overrides`` and ``verify_and_update_model_config`` add), from the
synthetic card of ``support.card``. ``stt_params`` builds a real core
``SpeechToTextParams`` around it.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "inprocess_engine: builds an in-process EngineCore (enable_envs_cache() "
        "freezes vllm.envs for the session); run in its own pytest invocation "
        "(-m inprocess_engine), apart from the other gpu tests",
    )


@pytest.fixture
def served_config() -> Callable[..., Any]:
    def make(
        *,
        right_context: int = 6,
        locales: dict[str, int] | None = None,
        language_locales: dict[str, str] | None = None,
        native_burst: bool = False,
    ) -> Any:
        from support.card import card_config, processor_config

        from vllm_nemotron_asr_streaming.card_parser import translate_card
        from vllm_nemotron_asr_streaming.nemotron_asr.configuration_nemotron_asr import (  # noqa: E501
            NemotronASRConfig,
        )

        processor = processor_config()
        if locales is not None:
            processor = processor_config(prompt_dictionary=dict(locales))
        translated = dict(translate_card(card_config(), processor))
        translated["num_lookahead_tokens"] = right_context
        translated["att_context_right"] = right_context
        # the synthetic card serves [3, 0, 6, 13]; serve all five geometries
        # (as --hf-overrides would) so a default of 1 passes REG-LA-005
        translated["supported_num_lookahead_tokens"] = [0, 1, 3, 6, 13]
        translated["experimental_native_burst"] = native_burst
        if language_locales is not None:
            translated["language_locales"] = dict(language_locales)
        return NemotronASRConfig(**translated)

    return make


@pytest.fixture
def stt_params(served_config) -> Callable[..., Any]:
    def make(
        *,
        n_samples: int = 16_000,
        samples: np.ndarray | None = None,
        language: str | None = None,
        extra_args: dict | None = None,
        max_model_len: int = 100_000,
        task_type: str = "transcribe",
        request_prompt: str = "",
        hotwords: str | None = None,
        to_language: str | None = None,
        config: Any = None,
        **config_kwargs: Any,
    ) -> Any:
        from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams

        hf_config = config if config is not None else served_config(**config_kwargs)
        if samples is None:
            samples = (
                np.random.default_rng(7).standard_normal(n_samples).astype(np.float32)
            )
        # core reads only ``hf_config`` and ``max_model_len`` from the
        # model configuration in ``get_generation_prompt``
        model_config = SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len)
        return SpeechToTextParams(
            audio=samples,
            stt_config=SpeechToTextConfig(
                sample_rate=16000,
                max_audio_clip_s=None,
                min_energy_split_window_size=None,
            ),
            model_config=model_config,  # type: ignore[arg-type]
            language=language,
            hotwords=hotwords,
            task_type=task_type,
            request_prompt=request_prompt,
            to_language=to_language,
            extra_args=extra_args,
        )

    return make


@pytest.fixture
def model_cls():
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    return Nemotron3_5AsrStreamingForRNNT


@pytest.fixture
def prompt_fields() -> Callable[[Any], tuple[np.ndarray, list[int]]]:
    """Check the prompt shape of FILE-PROMPT-001; return ``(samples, file_header)``.

    ``prompt_token_ids`` is one token, ``multi_modal_data`` holds one audio item
    (the samples), ``mm_processor_kwargs`` carries ``file_header`` and
    ``multi_modal_uuids`` carries one UUID for the audio item.
    """

    def get(prompt: Any) -> tuple[np.ndarray, list[int]]:
        assert len(prompt["prompt_token_ids"]) == 1
        audio = prompt["multi_modal_data"]["audio"]
        assert isinstance(audio, np.ndarray) and audio.ndim == 1
        header = [int(v) for v in prompt["mm_processor_kwargs"]["file_header"]]
        assert len(prompt["multi_modal_uuids"]["audio"]) == 1
        return audio, header

    return get


# ---------------------------------------------------------------------------
# The real model state on CPU (FILE-BUF, FILE-ROW-001)

STATE_LAYER = "nemotron.state"


def _set_inert_model_attributes(model: Any, hf_config: Any) -> None:
    """The private model attributes ``prepare_step`` reads in today's code.

    NOTE: these names come from the current source; Phase 6 may rename them,
    and this helper is the one place to update.
    """
    model.config = hf_config
    model._persistent_state_layer = SimpleNamespace(prefix=STATE_LAYER)
    model._registry = SimpleNamespace(prune=lambda resident: None)
    model._native_burst_handoff = None


@pytest.fixture
def file_model_state(tmp_path):
    """The model's real ``ModelState`` on CPU, built as core builds it.

    The state class comes from the model's ``get_model_state_cls``; the model is
    the real class without weights (``nn.Module.__init__`` only), with the few
    attributes ``prepare_step`` reads set to inert values. Layout addition used
    by the tests: ``state.file_buffers`` is the state's ``FileHostBuffers``.
    """

    def build(right_context: int = 6):
        import torch
        from support.card import write_card
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            ModelConfig,
            SchedulerConfig,
            VllmConfig,
        )

        from vllm_nemotron_asr_streaming import register
        from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

        register()
        card = write_card(tmp_path / "card")
        model_config = ModelConfig(
            model=str(card),
            config_format="nemotron_asr_streaming",
            hf_overrides={
                "num_lookahead_tokens": right_context,
                "supported_num_lookahead_tokens": [0, 1, 3, 6, 13],
            },
            enforce_eager=True,
            dtype="float32",
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(enable_prefix_caching=False),
            scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
        )
        model = Nemotron3_5AsrStreamingForRNNT.__new__(Nemotron3_5AsrStreamingForRNNT)
        torch.nn.Module.__init__(model)
        _set_inert_model_attributes(model, model_config.hf_config)
        state_cls = Nemotron3_5AsrStreamingForRNNT.get_model_state_cls()
        return state_cls(vllm_config, model, None, torch.device("cpu"))

    return build


@pytest.fixture
def make_step() -> Callable[..., Any]:
    """Build real ``NewRequestData`` / ``SchedulerOutput`` structures."""

    def new_request(req_id: str, token: int, samples=None, header=None):
        import torch
        from vllm.multimodal.inputs import (
            MultiModalBatchedField,
            MultiModalFeatureSpec,
            MultiModalFieldElem,
            MultiModalKwargsItem,
            PlaceholderRange,
        )
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData

        features = []
        if samples is not None:
            field = MultiModalBatchedField(keep_on_cpu=True)
            item = MultiModalKwargsItem(
                {
                    "samples": MultiModalFieldElem(torch.from_numpy(samples), field),
                    "file_header": MultiModalFieldElem(
                        torch.tensor(header, dtype=torch.int64), field
                    ),
                }
            )
            features.append(
                MultiModalFeatureSpec(
                    data=item,
                    modality="audio",
                    identifier=f"uuid-{req_id}",
                    mm_position=PlaceholderRange(offset=0, length=1),
                )
            )
        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=[token],
            mm_features=features,
            sampling_params=SamplingParams(max_tokens=100),
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=[token],
        )

    def step(
        *, new=(), cached=(), finished=(), handles=None, free_hashes=(), updates=None
    ):
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        scheduled = {r.req_id: 1 for r in new} | {r: 1 for r in cached}
        return SchedulerOutput(
            scheduled_new_reqs=list(new),
            scheduled_cached_reqs=CachedRequestData(
                req_ids=list(cached),
                new_token_ids=[[0] for _ in cached],
                new_block_ids=[None for _ in cached],
                num_computed_tokens=[1 for _ in cached],
                num_output_tokens=[1 for _ in cached],
            ),
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(finished),
            free_encoder_mm_hashes=list(free_hashes),
            encoder_input_updates=dict(updates or {}),
            state_slot_handles=handles,
        )

    def handle(slot: int, generation: int):
        from vllm.v1.kv_cache_interface import StateSlotHandle

        return {STATE_LAYER: StateSlotHandle("epoch", slot, generation)}

    return SimpleNamespace(new_request=new_request, step=step, handle=handle)
