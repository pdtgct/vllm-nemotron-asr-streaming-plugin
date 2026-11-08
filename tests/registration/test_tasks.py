# SPDX-License-Identifier: Apache-2.0
"""Supported tasks and the model class's declared interfaces (REG-TASK-*)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.engine


# @spec REG-TASK-001
def test_model_class_declares_realtime_persistent_state_and_transcription(registered):
    from vllm.model_executor.models.interfaces import (
        supports_realtime,
        supports_transcription,
    )

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    cls = Nemotron3_5AsrStreamingForRNNT
    assert supports_realtime(cls)
    assert getattr(cls, "supports_persistent_state", False) is True
    assert supports_transcription(cls)
    assert cls.supports_transcription_only is False
    assert cls.supports_segment_timestamp is False
    assert cls.supports_diarized_transcription is False


# @spec REG-TASK-002
def test_model_state_supports_exactly_transcription_and_realtime(
    make_model_config, make_vllm_config, build_model_state, card_dir
):
    vllm_config = make_vllm_config(make_model_config(card_dir))
    state = build_model_state(vllm_config)
    assert state.get_supported_generation_tasks() == ("transcription", "realtime")


# @spec REG-TASK-002
def test_generate_is_not_advertised(
    make_model_config, make_vllm_config, build_model_state, card_dir
):
    vllm_config = make_vllm_config(make_model_config(card_dir))
    tasks = build_model_state(vllm_config).get_supported_generation_tasks()
    assert "generate" not in tasks
