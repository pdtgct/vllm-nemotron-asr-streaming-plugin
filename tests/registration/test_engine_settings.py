# SPDX-License-Identifier: Apache-2.0
"""Engine settings the model refuses (REG-ENG-001 to REG-ENG-009).

``verify_and_update_config`` is called on ``VllmConfig``-shaped namespaces whose
attribute paths are the ones core's own persistent-state validation reads
(``validate_state_config``, ``vllm/v1/core/state_slot_manager.py``) plus
``model_config.enforce_eager``. A real ``VllmConfig`` on CPU is used where it
constructs (the hook runs inside ``VllmConfig.__post_init__`` via
``try_verify_and_update_config``, ``vllm/config/vllm.py``) and for the
model-state re-check, which needs configurations the hook would have refused.
"""

from __future__ import annotations

from types import SimpleNamespace

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
    "enable_elastic_ep": ("elastic",),
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
        "enable_elastic_ep",
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
        "enable_elastic_ep",
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
    """No compilation settings at all: neither runner configuration."""
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
        enable_elastic_ep=True,
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
    "enable_elastic_ep": lambda c: setattr(
        c.parallel_config, "enable_elastic_ep", True
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


# --------------------------------------------------------------------------
# The hook configuration (REG-ENG-005, REG-ENG-006)
# --------------------------------------------------------------------------

#: The default segment arms (REG-CARD-008).
DEFAULT_ARMS = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
}


def _hook_config(
    engine_config,
    platform,
    *,
    mode="none",
    cudagraph="none",
    level=None,
    cuda=True,
    **hf,
):
    """``enforce_eager`` off with the given compilation settings, platform and
    served-configuration fields (the default arms unless ``hf`` says
    otherwise)."""
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    modes = {"none": CompilationMode.NONE, "compile": CompilationMode.VLLM_COMPILE}
    graphs = {"none": CUDAGraphMode.NONE, "piecewise": CUDAGraphMode.PIECEWISE}
    platform(cuda)
    return engine_config(
        enforce_eager=False,
        compilation_config=SimpleNamespace(
            mode=modes.get(mode), cudagraph_mode=graphs.get(cudagraph)
        ),
        optimization_level=level,
        device="cuda" if cuda else "cpu",
        hf={**DEFAULT_ARMS, **hf},
    )


def _resolved_check(config):
    """The model-state re-check of REG-ENG-006 on a namespace: the engine
    checks with unset values no longer accepted. Seam: ``check_engine_config``
    with ``resolved=True`` (what model-state construction calls)."""
    from vllm_nemotron_asr_streaming.engine_flags import check_engine_config

    check_engine_config(config, "test", resolved=True)


_HOOK_ACCEPTED = {
    "default arms": {},
    "graphed encoder only": {"decode_dispatch_arm": "dense-eager"},
    "graphed decoder only": {"encoder_execution_arm": "eager"},
    "transaction graphs at 80 ms": {
        "num_lookahead_tokens": 0,
        "transaction_graph_populations": [2],
    },
}


@pytest.mark.parametrize("case", list(_HOOK_ACCEPTED))
@pytest.mark.parametrize("resolved", [False, True], ids=["hook", "model-state"])
# @spec REG-ENG-005, REG-ENG-006
def test_the_hook_configuration_is_accepted(engine_config, platform, case, resolved):
    config = _hook_config(engine_config, platform, **_HOOK_ACCEPTED[case])
    if resolved:
        _resolved_check(config)
    else:
        _hook().verify_and_update_config(config)


# @spec REG-ENG-005, REG-ENG-006
def test_unset_graph_mode_and_O0_are_accepted_only_before_core_resolves_them(
    engine_config, platform
):
    config = _hook_config(engine_config, platform, mode=None, cudagraph=None, level=0)
    _hook().verify_and_update_config(config)
    with pytest.raises(ValueError) as excinfo:
        _resolved_check(config)
    message = str(excinfo.value)
    assert "cudagraph_mode" in message and "-cc.mode none" in message, message


#: case -> (settings, words the refusal must contain: the condition and its
#: remedy).
_HOOK_REFUSED = {
    "core graphs": (
        {"cudagraph": "piecewise"},
        ("cudagraph_mode", "-cc.cudagraph_mode none"),
    ),
    "torch.compile": ({"mode": "compile"}, ("-cc.mode none",)),
    "default compile level": ({"mode": None, "level": 2}, ("-cc.mode none",)),
    "not CUDA": ({"cuda": False}, ("cuda", "--enforce-eager")),
    "no hook-captured graph": (
        {"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"},
        ("eager-graphed", "dense-graphed", "--enforce-eager"),
    ),
    "compiled encoder": (
        {"encoder_execution_arm": "dense-graphed"},
        ("encoder_execution_arm", "dense-graphed", "--enforce-eager"),
    ),
    "chunk buckets": (
        {"chunk_bucket_graph_populations": [1]},
        ("chunk_bucket_graph_populations", "--enforce-eager"),
    ),
}


@pytest.mark.parametrize("case", list(_HOOK_REFUSED))
# @spec REG-ENG-005
def test_each_unmet_hook_condition_is_named_with_its_remedy(
    engine_config, platform, case
):
    settings, words = _HOOK_REFUSED[case]
    config = _hook_config(engine_config, platform, **settings)
    message = _refuse(config)
    for word in words:
        assert word.lower() in message.lower(), (word, message)


# @spec REG-ENG-005, REG-VAL-003
def test_every_unmet_hook_condition_is_named_in_one_error(engine_config, platform):
    config = _hook_config(
        engine_config,
        platform,
        mode="compile",
        cudagraph="piecewise",
        cuda=False,
        encoder_execution_arm="dense-graphed",
        chunk_bucket_graph_populations=[1],
    )
    message = _refuse(config)
    for word in (
        "-cc.cudagraph_mode none",
        "-cc.mode none",
        "cuda",
        "dense-graphed",
        "chunk_bucket_graph_populations",
        "--enforce-eager",
    ):
        assert word.lower() in message.lower(), (word, message)


@pytest.mark.parametrize(
    "settings",
    [
        {"cuda": False},
        {"encoder_execution_arm": "dense-graphed"},
        {"chunk_bucket_graph_populations": [1]},
    ],
    ids=["not CUDA", "compiled encoder", "chunk buckets"],
)
# @spec REG-ENG-005
def test_an_enforce_eager_remedy_names_removing_transaction_graphs(
    engine_config, platform, settings
):
    """At 80 ms with the default arms the transaction-graph checks of the
    hook-runner pass, so ``transaction_graph_populations`` reaches the message
    only through the ``--enforce-eager`` remedy (the option is refused under
    ``--enforce-eager``)."""
    config = _hook_config(
        engine_config,
        platform,
        num_lookahead_tokens=0,
        transaction_graph_populations=[2],
        **settings,
    )
    message = _refuse(config)
    assert "--enforce-eager" in message
    assert "transaction_graph_populations" in message, message


@pytest.mark.parametrize("value", [{}, None, False, "", 0], ids=repr)
# @spec REG-ENG-005
def test_a_malformed_transaction_graph_value_counts_as_on(
    engine_config, platform, value
):
    """TXG-CFG-014: one on/off predicate; ``{}`` and the like are on, so the
    ``--enforce-eager`` remedy names the option."""
    config = _hook_config(
        engine_config,
        platform,
        cuda=False,
        num_lookahead_tokens=0,
        transaction_graph_populations=value,
    )
    message = _refuse(config)
    assert "transaction_graph_populations" in message, message


# --------------------------------------------------------------------------
# Core hooks (REG-ENG-009)
# --------------------------------------------------------------------------

HOOKS = (
    "prepare_execution_memory",
    "needs_cudagraph_capture",
    "capture_cudagraphs",
    "clear_cudagraphs",
    "post_warmup",
    "shutdown",
)


# @spec REG-ENG-009
def test_the_hook_list_is_the_six_model_state_hooks():
    from vllm_nemotron_asr_streaming.engine_flags import MODEL_STATE_GRAPH_HOOKS

    assert set(MODEL_STATE_GRAPH_HOOKS) == set(HOOKS)
    assert len(MODEL_STATE_GRAPH_HOOKS) == len(HOOKS)


# @spec REG-ENG-009
def test_the_installed_core_defines_every_hook():
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    assert all(callable(getattr(ModelState, n, None)) for n in HOOKS)


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "hook"])
@pytest.mark.parametrize("hook", HOOKS)
# @spec REG-ENG-009
def test_a_core_without_a_hook_is_refused_in_every_runner_configuration(
    engine_config, platform, monkeypatch, hook, enforce_eager
):
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    config = engine_config() if enforce_eager else _hook_config(engine_config, platform)
    monkeypatch.delattr(ModelState, hook)
    message = _refuse(config)
    assert hook in message, message
    assert "67583" in message, message


# @spec REG-ENG-009, REG-VAL-003
def test_every_missing_hook_is_named(engine_config, monkeypatch):
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    for hook in HOOKS:
        monkeypatch.delattr(ModelState, hook)
    message = _refuse(engine_config(async_scheduling=True), "async_scheduling")
    for hook in HOOKS:
        assert hook in message, (hook, message)
    assert "67583" in message


# --------------------------------------------------------------------------
# Real configurations and the model-state re-check
# --------------------------------------------------------------------------


# @spec REG-ENG-005
def test_real_hook_configuration_on_cpu_is_refused_for_its_platform(
    make_model_config, card_dir
):
    """A CPU device is not a CUDA platform: the remedy is ``--enforce-eager``."""
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = make_model_config(card_dir, enforce_eager=False)
    with pytest.raises(ValueError) as excinfo:
        VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(enable_prefix_caching=False),
            scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
            compilation_config=CompilationConfig(mode=0),
        )
    assert "--enforce-eager" in str(excinfo.value)


@pytest.mark.gpu
# @spec REG-ENG-005, REG-ENG-006
def test_real_hook_configuration_constructs_on_cuda(
    make_model_config, card_dir, build_model_state
):
    """The serve flags of the hook configuration (``-cc.mode none``; core
    resolves ``cudagraph_mode`` to NONE for persistent state) with the default
    arms construct, and the model state's re-check accepts the result."""
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CUDAGraphMode,
        DeviceConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = make_model_config(card_dir, enforce_eager=False)
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
        compilation_config=CompilationConfig(mode=0),
    )
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    build_model_state(config)


# @spec REG-ENG-006
def test_model_state_refuses_an_unset_graph_mode(good_vllm_config, build_model_state):
    """At construction an unset ``cudagraph_mode`` is not accepted."""
    from vllm.config.compilation import CompilationMode

    config = good_vllm_config()
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE, cudagraph_mode=None
    )
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    assert "cudagraph_mode" in str(excinfo.value)


# @spec REG-ENG-006, REG-ENG-009
def test_model_state_refuses_a_core_without_the_hooks(
    good_vllm_config, build_model_state, monkeypatch
):
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    config = good_vllm_config()
    monkeypatch.delattr(ModelState, "post_warmup")
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    assert "post_warmup" in str(excinfo.value)


# @spec REG-ENG-006, REG-VAL-008
def test_model_state_reruns_the_transaction_graph_checks(
    good_vllm_config, build_model_state
):
    """The option turned on after the hook ran (under ``--enforce-eager``)
    is refused at construction, naming the option."""
    config = good_vllm_config()
    config.model_config.hf_config.transaction_graph_populations = [2]
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    assert "transaction_graph_populations" in str(excinfo.value)
