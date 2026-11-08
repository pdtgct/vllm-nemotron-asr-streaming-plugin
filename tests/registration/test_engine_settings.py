# SPDX-License-Identifier: Apache-2.0
"""Engine settings the model refuses (REG-ENG-001 to REG-ENG-007).

``verify_and_update_config`` is called on ``VllmConfig``-shaped namespaces whose
attribute paths are the ones core's own persistent-state validation reads
(``validate_state_config``, ``vllm/v1/core/state_slot_manager.py``) plus
``model_config.enforce_eager``. A real ``VllmConfig`` on CPU is used where it
constructs (the hook runs inside ``VllmConfig.__post_init__`` via
``try_verify_and_update_config``, ``vllm/config/vllm.py``) and for the
model-state re-check, which needs configurations the hook would have refused.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.engine

#: Words any of which may name a refused setting in the error message.
NAMES = {
    "async_scheduling": ("async",),
    "enable_prefix_caching": ("prefix",),
    "speculative_config": ("speculative",),
    "kv_transfer_config": ("kv transfer", "kv_transfer"),
    "enable_dbo": ("dual batch", "dbo", "dual-batch"),
    "enable_sleep_mode": ("sleep",),
    "enable_cumem_allocator": ("cumem",),
    "hisparse_config": ("hisparse", "hierarchical sparse"),
    "enforce_eager": ("eager",),
    "tensor_parallel_size": ("tensor_parallel_size", "tensor parallel"),
    "pipeline_parallel_size": ("pipeline_parallel_size", "pipeline parallel"),
    "data_parallel_size": ("data_parallel_size", "data parallel"),
    "decode_context_parallel_size": (
        "decode_context_parallel_size",
        "decode context parallel",
    ),
    "prefill_context_parallel_size": (
        "prefill_context_parallel_size",
        "prefill context parallel",
    ),
}

PARALLEL = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
)


def _hook():
    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    return Nemotron3_5AsrStreamingConfig


def _names(message: str, setting: str) -> bool:
    return any(word in message.lower() for word in NAMES[setting])


def _refuse(config, *settings: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        _hook().verify_and_update_config(config)
    message = str(excinfo.value)
    for setting in settings:
        assert _names(message, setting), f"{setting} not named in: {message}"
    return message


# @spec REG-ENG-001
def test_accepted_engine_configuration_passes(engine_config):
    _hook().verify_and_update_config(engine_config())


@pytest.mark.parametrize("value", [True, None])
# @spec REG-ENG-001
def test_async_scheduling_enabled_or_unset_is_refused(engine_config, value):
    _refuse(engine_config(async_scheduling=value), "async_scheduling")


# @spec REG-ENG-002
def test_prefix_caching_is_refused(engine_config):
    _refuse(engine_config(enable_prefix_caching=True), "enable_prefix_caching")


@pytest.mark.parametrize(
    "setting",
    [
        "speculative_config",
        "kv_transfer_config",
        "enable_dbo",
        "enable_sleep_mode",
        "enable_cumem_allocator",
        "hisparse_config",
    ],
)
# @spec REG-ENG-003
def test_each_unsupported_feature_is_refused_and_named(engine_config, setting):
    enabled = object() if setting.endswith("_config") else True
    _refuse(engine_config(**{setting: enabled}), setting)


# @spec REG-ENG-003
def test_every_enabled_unsupported_feature_is_named(engine_config):
    settings = (
        "speculative_config",
        "kv_transfer_config",
        "enable_dbo",
        "enable_sleep_mode",
        "enable_cumem_allocator",
        "hisparse_config",
    )
    config = engine_config(
        **{s: (object() if s.endswith("_config") else True) for s in settings}
    )
    _refuse(config, *settings)


@pytest.mark.parametrize("setting", PARALLEL)
# @spec REG-ENG-004
def test_each_parallel_size_other_than_one_is_refused(engine_config, setting):
    _refuse(engine_config(**{setting: 2}), setting)


@pytest.mark.parametrize("setting", PARALLEL)
# @spec REG-ENG-004
def test_a_parallel_setting_the_build_lacks_counts_as_one(engine_config, setting):
    _hook().verify_and_update_config(engine_config(drop=(setting,)))


# @spec REG-ENG-004
def test_a_missing_parallel_setting_does_not_hide_another_refusal(engine_config):
    config = engine_config(
        drop=("decode_context_parallel_size", "prefill_context_parallel_size"),
        tensor_parallel_size=2,
    )
    _refuse(config, "tensor_parallel_size")


@pytest.mark.parametrize("value", [False, None])
# @spec REG-ENG-005
def test_an_outer_runner_that_is_not_eager_is_refused(engine_config, value):
    _refuse(engine_config(enforce_eager=value), "enforce_eager")


# @spec REG-VAL-003
def test_every_refused_setting_is_listed_in_one_error(engine_config):
    config = engine_config(
        async_scheduling=None,
        enable_prefix_caching=True,
        speculative_config=object(),
        kv_transfer_config=object(),
        enable_dbo=True,
        enable_sleep_mode=True,
        enable_cumem_allocator=True,
        hisparse_config=object(),
        enforce_eager=False,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        data_parallel_size=2,
        decode_context_parallel_size=2,
        prefill_context_parallel_size=2,
    )
    _refuse(config, *NAMES)


# --------------------------------------------------------------------------
# Real configurations
# --------------------------------------------------------------------------


# @spec REG-ENG-001, REG-ENG-002
def test_real_vllm_config_with_accepted_settings_constructs(
    make_model_config, make_vllm_config, card_dir
):
    vllm_config = make_vllm_config(make_model_config(card_dir))
    assert vllm_config.scheduler_config.async_scheduling is False
    assert vllm_config.cache_config.enable_prefix_caching is False


# @spec REG-ENG-001
def test_real_vllm_config_with_async_scheduling_left_unset_is_refused(
    make_model_config, make_vllm_config, card_dir
):
    with pytest.raises(ValueError, match="(?i)async"):
        make_vllm_config(make_model_config(card_dir), async_scheduling=None)


# @spec REG-ENG-002
def test_real_vllm_config_with_prefix_caching_is_refused(
    make_model_config, make_vllm_config, card_dir
):
    with pytest.raises(ValueError, match="(?i)prefix"):
        make_vllm_config(make_model_config(card_dir), prefix_caching=True)


# --------------------------------------------------------------------------
# Re-check when the model state is constructed
# --------------------------------------------------------------------------


@pytest.fixture
def good_vllm_config(make_model_config, make_vllm_config, card_dir):
    """Factory: the real configuration, built inside the test."""
    return lambda: make_vllm_config(make_model_config(card_dir))


# @spec REG-ENG-006
def test_model_state_accepts_the_accepted_configuration(
    good_vllm_config, build_model_state
):
    build_model_state(good_vllm_config())


_MUTATIONS = {
    "async_scheduling": lambda c: setattr(c.scheduler_config, "async_scheduling", True),
    "enable_prefix_caching": lambda c: setattr(
        c.cache_config, "enable_prefix_caching", True
    ),
    "enforce_eager": lambda c: setattr(c.model_config, "enforce_eager", False),
    "tensor_parallel_size": lambda c: setattr(
        c.parallel_config, "tensor_parallel_size", 2
    ),
    "speculative_config": lambda c: setattr(c, "speculative_config", object()),
    "enable_dbo": lambda c: setattr(c.parallel_config, "enable_dbo", True),
}


@pytest.mark.parametrize("setting", list(_MUTATIONS))
# @spec REG-ENG-006
def test_model_state_refuses_a_configuration_resolved_after_the_hook(
    good_vllm_config, build_model_state, setting
):
    config = good_vllm_config()
    _MUTATIONS[setting](config)
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    assert _names(str(excinfo.value), setting)


# @spec REG-ENG-006, REG-VAL-003
def test_model_state_lists_every_failed_setting_in_one_error(
    good_vllm_config, build_model_state
):
    config = good_vllm_config()
    for mutate in _MUTATIONS.values():
        mutate(config)
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    for setting in _MUTATIONS:
        assert _names(str(excinfo.value), setting), setting


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------


def _serve_command(readme: str) -> str:
    blocks = re.findall(r"```[a-z]*\n(.*?)```", readme, flags=re.S)
    commands = [b for b in blocks if "vllm serve" in b]
    assert commands, "README has no `vllm serve` command"
    return re.sub(r"\\\s*\n\s*", " ", commands[0])


# @spec REG-ENG-007
def test_readme_serve_command_shows_every_required_flag(repo_root):
    command = _serve_command((repo_root / "README.md").read_text())
    for flag in (
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--dtype float32",
        "--enforce-eager",
        "--config-format nemotron_asr_streaming",
        "--tokenizer-mode nemotron_asr_streaming",
    ):
        assert flag in command, f"{flag} missing from: {command}"
    assert re.search(r"--hf-overrides[^\n]*num_lookahead_tokens", command), command
    assert "--async-scheduling" not in command
