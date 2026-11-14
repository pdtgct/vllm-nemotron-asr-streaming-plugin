# SPDX-License-Identifier: Apache-2.0
"""FILE-IF, FILE-TOK: the model class's transcription interface."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from support.card import BLANK

pytestmark = pytest.mark.engine

ASSERT_NOT_SERVED = ("zu", "nb", "nah", "haw")


def _locale_language(name: str) -> str | None:
    """Independent reading of the rule in the design's Language section."""
    letters = ""
    for ch in name:
        if ch == "-" or ch.isupper():
            break
        letters += ch
    if not letters.isalpha() or not letters.islower() or len(letters) not in (2, 3):
        return None
    return letters


# @spec FILE-IF-001
def test_class_declares_supports_transcription(model_cls):
    from vllm.model_executor.models.interfaces import supports_transcription

    assert supports_transcription(model_cls) is True
    assert model_cls.supports_transcription_only is False
    assert model_cls.supports_segment_timestamp is False
    assert model_cls.supports_diarized_transcription is False


# @spec FILE-IF-002
def test_supported_languages_is_static_and_inside_core_map(model_cls):
    from vllm.model_executor.models.interfaces import LANGUAGES

    table = model_cls.supported_languages  # a class attribute, no instance needed
    assert len(table) > 0
    assert set(table) <= set(LANGUAGES)
    assert all(len(code) == 2 for code in table)
    for code in ASSERT_NOT_SERVED:
        assert code not in table
    assert all(isinstance(name, str) and name for name in table.values())


# @spec FILE-IF-002
def test_core_accepts_the_class_definition(model_cls):
    # vllm/model_executor/models/interfaces.py:1376 (__init_subclass__) raises
    # ValueError for any code outside Whisper's LANGUAGES; importing the model
    # class already ran it, and a subclass runs it again.
    class Again(model_cls):  # noqa: D401 - definition is the test
        pass

    assert dict(Again.supported_languages) == dict(model_cls.supported_languages)


# @spec FILE-IF-002
@pytest.mark.card
def test_supported_languages_equal_card_languages_in_core_map(model_cls, real_card_dir):
    from vllm.model_executor.models.interfaces import LANGUAGES

    processor = json.loads((real_card_dir / "processor_config.json").read_text())
    card_languages = {
        lang
        for name in processor["prompt_dictionary"]
        if (lang := _locale_language(name)) is not None and len(lang) == 2
    }
    assert set(model_cls.supported_languages) == card_languages & set(LANGUAGES)


# @spec FILE-IF-003
@pytest.mark.parametrize("task_type", ["transcribe"])
def test_speech_to_text_config(model_cls, served_config, task_type):
    model_config = SimpleNamespace(hf_config=served_config())
    stt = model_cls.get_speech_to_text_config(model_config, task_type)
    assert stt.sample_rate == 16000
    assert stt.max_audio_clip_s is None
    assert stt.min_energy_split_window_size is None


# @spec FILE-TOK-001
def test_continuation_token_is_special_and_not_a_stop_token(card_dir):
    import io

    from fastapi import UploadFile
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    from vllm_nemotron_asr_streaming.tokenizer_mode import (
        NemotronAsrStreamingTokenizer,
    )

    tokenizer = NemotronAsrStreamingTokenizer.from_pretrained(card_dir)
    continuation = BLANK + 5
    assert continuation in tokenizer.all_special_ids
    assert tokenizer.eos_token_id == BLANK + 1
    assert tokenizer.decode([continuation], skip_special_tokens=True) == ""

    # The sampling parameters core builds for a transcription request, then
    # completed the way the engine's input processor does with the card's
    # generation config (which names no EOS):
    # vllm/entrypoints/speech_to_text/transcription/protocol.py:238 and
    # vllm/sampling_params.py:781.
    generation = json.loads((card_dir / "generation_config.json").read_text())
    request = TranscriptionRequest(
        file=UploadFile(io.BytesIO(b"x"), filename="a.wav"), model="m"
    )
    params = request.to_sampling_params(1000)
    params.update_from_generation_config(
        generation, eos_token_id=tokenizer.eos_token_id
    )
    assert continuation != params.eos_token_id
    assert continuation not in (params.stop_token_ids or [])
    assert params._all_stop_token_ids == {BLANK + 1}


# @spec FILE-TOK-001
@pytest.mark.card
def test_continuation_token_special_on_the_published_card(real_card_dir):
    from vllm_nemotron_asr_streaming.tokenizer_mode import (
        NemotronAsrStreamingTokenizer,
    )

    blank = json.loads((real_card_dir / "config.json").read_text())["blank_token_id"]
    tokenizer = NemotronAsrStreamingTokenizer.from_pretrained(real_card_dir)
    assert blank + 5 in tokenizer.all_special_ids
    assert tokenizer.eos_token_id == blank + 1
