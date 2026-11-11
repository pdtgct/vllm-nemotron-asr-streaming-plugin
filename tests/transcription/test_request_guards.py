# SPDX-License-Identifier: Apache-2.0
"""FILE-GRD-001..008: beam search and completion limits do not apply to audio
transcription with this model, and the model imposes no length limit.

The two request fields, ``use_beam_search`` and ``max_completion_tokens``,
come from the pending core change to ``SpeechToTextParams``; the ``guarded``
fixture builds a real core ``SpeechToTextParams`` that carries them (a
subclass adding exactly those two fields with the change's defaults, while the
installed core lacks them). The plain ``stt_params`` fixture, whose class has
no such fields, exercises FILE-GRD-005.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("vllm.config.speech_to_text")

BEAM = "use_beam_search does not apply to audio transcription with this model"
LIMIT = "max_completion_tokens does not apply to audio transcription with this model"


@pytest.fixture
def build():
    from vllm_nemotron_asr_streaming.transcription import build_file_prompt

    return build_file_prompt


class Untouched:
    """Audio that fails the test if the model reads it."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("the guard ran after the audio was read")


# @spec FILE-GRD-001
def test_beam_search_is_refused(build, guarded):
    # `n` is not a SpeechToTextParams field: the refusal cannot depend on it,
    # so beam search with n = 1 is refused like any other.
    with pytest.raises(ValueError) as err:
        build(guarded(use_beam_search=True))
    assert str(err.value).startswith(BEAM)


# @spec FILE-GRD-002
@pytest.mark.parametrize("limit", [0, -1, 1, 141, 4096, 10**9])
def test_any_present_completion_limit_is_refused(build, guarded, limit):
    with pytest.raises(ValueError) as err:
        build(guarded(max_completion_tokens=limit))
    assert str(err.value).startswith(LIMIT)


# @spec FILE-GRD-003
def test_absent_fields_are_not_refused(build, guarded):
    prompt = build(guarded(use_beam_search=False, max_completion_tokens=None))
    assert len(prompt["prompt_token_ids"]) == 1


# @spec FILE-GRD-003, FILE-GRD-005
def test_a_params_class_without_the_fields_is_not_refused(build, stt_params):
    params = stt_params()
    assert not hasattr(params, "use_beam_search")
    assert not hasattr(params, "max_completion_tokens")
    assert len(build(params)["prompt_token_ids"]) == 1


# @spec FILE-GRD-004
def test_beam_search_is_reported_before_the_completion_limit(build, guarded):
    with pytest.raises(ValueError) as err:
        build(guarded(use_beam_search=True, max_completion_tokens=7))
    assert str(err.value).startswith(BEAM)


# @spec FILE-GRD-004
def test_a_translation_request_with_beam_search_is_a_wrong_task(build, guarded):
    with pytest.raises(ValueError, match="(?i)task") as err:
        build(guarded(use_beam_search=True, task_type="translate"))
    assert BEAM not in str(err.value)


# @spec FILE-GRD-004
@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"request_prompt": "context"}, "prompt"),
        ({"hotwords": "nvidia"}, "hotwords"),
        ({"to_language": "fr"}, "to_language|target"),
    ],
)
def test_unsupported_fields_are_reported_before_the_guards(
    build, guarded, kwargs, pattern
):
    with pytest.raises(ValueError, match=f"(?i){pattern}"):
        build(guarded(use_beam_search=True, max_completion_tokens=3, **kwargs))


# @spec FILE-GRD-004
def test_native_burst_is_reported_before_the_guards(build, guarded):
    with pytest.raises(ValueError, match="(?i)native.burst"):
        build(guarded(use_beam_search=True, right_context=1, native_burst=True))


# @spec FILE-GRD-004
def test_the_guards_are_reported_before_the_language(build, guarded):
    with pytest.raises(ValueError) as err:
        build(guarded(max_completion_tokens=5, language="zh"))
    assert str(err.value).startswith(LIMIT)


# @spec FILE-GRD-004, FILE-GRD-008
@pytest.mark.parametrize(
    "fields", [{"use_beam_search": True}, {"max_completion_tokens": 3}]
)
def test_a_refusal_comes_before_the_audio_is_read(build, guarded, fields):
    params = guarded(samples=np.zeros(1, dtype=np.float32), **fields)
    params.audio = Untouched()
    with pytest.raises(ValueError):
        build(params)


# @spec FILE-GRD-008
def test_a_refusal_builds_no_prompt_and_stages_nothing(build, guarded):
    # the refusal is an exception: there is no prompt dict, so no multimodal
    # UUID, buffer or state slot can exist for the request
    import uuid

    calls = []
    real = uuid.uuid4
    uuid.uuid4 = lambda: calls.append(1) or real()
    try:
        with pytest.raises(ValueError):
            build(guarded(use_beam_search=True))
    finally:
        uuid.uuid4 = real
    assert calls == []


# @spec FILE-GRD-006
def test_a_very_long_file_is_not_refused_for_its_length(build, guarded):
    n = 3_600 * 16_000
    prompt = build(guarded(n_samples=n, max_model_len=100))
    assert prompt["mm_processor_kwargs"]["file_header"][2] == n


# @spec FILE-GRD-001, FILE-GRD-002
@pytest.mark.engine
@pytest.mark.parametrize(
    "fields", [{"use_beam_search": True}, {"max_completion_tokens": 9}]
)
def test_the_model_class_applies_the_guards(model_cls, guarded, fields):
    with pytest.raises(ValueError, match="does not apply to audio transcription"):
        model_cls.get_generation_prompt(guarded(**fields))
