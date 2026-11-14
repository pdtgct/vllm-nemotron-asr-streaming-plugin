# SPDX-License-Identifier: Apache-2.0
"""Engine settings the model refuses (REG-ENG-001 to REG-ENG-008).

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
    "enable_fault_tolerance": ("fault tolerance", "fault_tolerance", "fault-tolerance"),
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


# @spec REG-ENG-001
def test_async_scheduling_enabled_is_refused(engine_config):
    _refuse(engine_config(async_scheduling=True), "async_scheduling")


# @spec REG-ENG-001
def test_async_scheduling_left_unset_is_not_refused_by_the_hook(engine_config):
    """The hook runs before core resolves the option, so it sees ``None``;
    core then resolves it to ``False`` for a persistent-state model."""
    _hook().verify_and_update_config(engine_config(async_scheduling=None))


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
        "enable_fault_tolerance",
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
        "enable_fault_tolerance",
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
        async_scheduling=True,
        enable_prefix_caching=True,
        speculative_config=object(),
        kv_transfer_config=object(),
        enable_dbo=True,
        enable_sleep_mode=True,
        enable_cumem_allocator=True,
        hisparse_config=object(),
        enable_fault_tolerance=True,
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
def test_real_vllm_config_with_async_scheduling_left_unset_constructs(
    make_model_config, make_vllm_config, card_dir
):
    """An unset option is accepted by the hook and core resolves it off for
    a model that declares persistent state."""
    vllm_config = make_vllm_config(make_model_config(card_dir), async_scheduling=None)
    assert vllm_config.scheduler_config.async_scheduling is False


# @spec REG-ENG-003
def test_real_vllm_config_with_fault_tolerance_is_refused(
    make_model_config, make_vllm_config, card_dir
):
    """``enable_fault_tolerance`` keeps core's JIT kernel warm-up on under
    ``enforce_eager``; that warm-up runs synthetic real steps without
    state-slot handles."""
    from vllm.config import ParallelConfig

    model_config = make_model_config(card_dir)
    with pytest.raises(ValueError) as excinfo:
        _hook().verify_and_update_config(
            _with_parallel(
                make_vllm_config(model_config),
                ParallelConfig(enable_fault_tolerance=True),
            )
        )
    assert _names(str(excinfo.value), "enable_fault_tolerance")


def _with_parallel(vllm_config, parallel_config):
    vllm_config.parallel_config = parallel_config
    return vllm_config


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
    "enable_sleep_mode": lambda c: setattr(c.model_config, "enable_sleep_mode", True),
    "enable_cumem_allocator": lambda c: setattr(
        c.model_config, "enable_cumem_allocator", True
    ),
    "enable_fault_tolerance": lambda c: setattr(
        c.parallel_config, "enable_fault_tolerance", True
    ),
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


# @spec REG-ENG-006
def test_model_state_refuses_a_resolved_async_scheduling_of_none(
    good_vllm_config, build_model_state
):
    """The hook accepts ``None``; at model-state construction the value is
    resolved, and anything but ``False`` (including ``None``) is refused."""
    config = good_vllm_config()
    config.scheduler_config.async_scheduling = None
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    assert _names(str(excinfo.value), "async_scheduling")


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
    """``--no-async-scheduling`` and ``--no-enable-prefix-caching`` may be
    omitted (core resolves both off for this model); every other flag the
    engine-setting checks require must be shown."""
    command = _serve_command((repo_root / "README.md").read_text())
    for flag in (
        "--dtype float32",
        "--enforce-eager",
        "--config-format nemotron_asr_streaming",
        "--tokenizer-mode nemotron_asr_streaming",
    ):
        assert flag in command, f"{flag} missing from: {command}"
    assert re.search(r"--hf-overrides[^\n]*num_lookahead_tokens", command), command
    # Nothing in the command turns on a setting REG-ENG-001 to REG-ENG-005
    # refuses.
    for refused in (
        " --async-scheduling",
        " --enable-prefix-caching",
        "--speculative-config",
        "--kv-transfer-config",
        "--enable-dbo",
        "--enable-sleep-mode",
        "--fault-tolerance-config",
        "--enable-fault-tolerance",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
    ):
        assert refused not in command, f"{refused} in: {command}"


# --------------------------------------------------------------------------
# Native burst's configuration check
# --------------------------------------------------------------------------


def _native_burst_config(*, drop=(), **parallel):
    from types import SimpleNamespace

    values = {
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    values.update(parallel)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            **{k: v for k, v in values.items() if k not in drop}
        ),
        model_config=SimpleNamespace(logits_processors=None),
    )


def _validate_native_burst(config):
    from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
        validate_native_burst_config,
    )

    validate_native_burst_config(config)


# @spec REG-ENG-008
def test_native_burst_accepts_the_single_device_configuration():
    _validate_native_burst(_native_burst_config())


@pytest.mark.parametrize(
    "setting",
    [*PARALLEL, "enable_expert_parallel"],
)
# @spec REG-ENG-008
def test_native_burst_counts_an_undefined_parallel_setting_as_one_or_off(setting):
    """A parallel size the build does not define counts as 1, and an
    undefined expert-parallel setting as off (as REG-ENG-004 does)."""
    _validate_native_burst(_native_burst_config(drop=(setting,)))


# @spec REG-ENG-008
def test_native_burst_with_every_parallel_setting_undefined_is_accepted():
    _validate_native_burst(
        _native_burst_config(drop=(*PARALLEL, "enable_expert_parallel"))
    )


@pytest.mark.parametrize("setting", PARALLEL)
# @spec REG-ENG-008
def test_native_burst_still_refuses_a_defined_parallel_size_other_than_one(setting):
    with pytest.raises(ValueError, match="native burst"):
        _validate_native_burst(_native_burst_config(**{setting: 2}))


# @spec REG-ENG-008
def test_native_burst_still_refuses_expert_parallelism():
    with pytest.raises(ValueError, match="native burst"):
        _validate_native_burst(_native_burst_config(enable_expert_parallel=True))
