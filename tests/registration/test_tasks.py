# SPDX-License-Identifier: Apache-2.0
"""Supported tasks and the model class's declared interfaces (REG-TASK-*)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.engine


# @spec REG-TASK-001
def test_model_class_declares_realtime_persistent_state_and_transcription(registered):
    from vllm.model_executor.models.interfaces import (
        SupportsPersistentState,
        supports_persistent_state,
        supports_realtime,
        supports_transcription,
    )

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    cls = Nemotron3_5AsrStreamingForRNNT
    assert supports_realtime(cls)
    assert SupportsPersistentState in cls.__mro__
    assert supports_persistent_state(cls) is True
    assert supports_transcription(cls)
    assert cls.supports_transcription_only is False
    assert cls.supports_segment_timestamp is False
    assert cls.supports_diarized_transcription is False


def _set_on_the_class(cls, name: str):
    """The value of ``name`` as the model class itself sets it, in its own
    namespace (``vars(cls)``): a value inherited from a base class, the
    protocol ``SupportsPersistentState``'s default included, is not set on the
    class (REG-TASK-001)."""
    assert name in vars(cls), f"{name} is not set on {cls.__name__} itself" + (
        f" (it is inherited: {getattr(cls, name)!r})" if hasattr(cls, name) else ""
    )
    return vars(cls)[name]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("persistent_state_reads_positions", False),
        ("persistent_state_accepts_synthetic_warmup", False),
        ("persistent_state_resumable_stops", frozenset({"eos"})),
    ],
    ids=["REG-TASK-003", "REG-TASK-004", "REG-TASK-005"],
)
# @spec REG-TASK-001, REG-TASK-003, REG-TASK-004, REG-TASK-005
def test_persistent_state_settings_are_set_on_the_model_class(
    registered, name, expected
):
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    value = _set_on_the_class(Nemotron3_5AsrStreamingForRNNT, name)
    assert value == expected and type(value) is type(expected), value


class _Unreadable:
    """A ``positions`` argument whose every use fails."""

    def __getattr__(self, name):
        raise AssertionError(f"the forward read positions.{name}")

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"the forward used positions in {func}")

    def __bool__(self):
        raise AssertionError("the forward read positions")

    def __len__(self):
        raise AssertionError("the forward read positions")

    def __iter__(self):
        raise AssertionError("the forward read positions")

    def __getitem__(self, key):
        raise AssertionError("the forward read positions")


# @spec REG-TASK-003
def test_the_forward_reads_no_value_of_positions(registered, tmp_path, monkeypatch):
    """A real step and a profiling dummy step whose forward receives a
    ``positions`` that fails on any use."""
    from support import model_harness as h

    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(
        card,
        overrides={
            "encoder_execution_arm": "eager",
            "decode_dispatch_arm": "dense-eager",
        },
    )
    model = h.build_model(config)
    forward = type(model).forward

    def guarded(self, *args, **kwargs):
        if "positions" in kwargs:
            kwargs["positions"] = _Unreadable()
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "forward", guarded)
    runner = h.Runner(config, model, h.build_model_state(config, model))
    runner.profile_run(core_embeds=False)
    runner.bind_pool(h.real_bindings(h.model_spec(model)))
    from vllm_nemotron_asr_streaming.lookahead import geometry_index, served_lookahead

    geometry = geometry_index(served_lookahead(model.config))
    unit = h.new_request(
        "s1",
        int(model.config.audio_chunk_token_id),
        audio=h.stream_envelope(model.config, geometry=geometry),
    )
    runner.real_step(h.scheduler_output(new=(unit,), handles={"s1": 1}))


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
