# SPDX-License-Identifier: Apache-2.0
"""Engine settings the model refuses or accepts (REG-ENG-001 to REG-ENG-026).

Core's persistent-state check runs first while core builds a configuration and
names the first setting it refuses; the tests of REG-ENG-002 to REG-ENG-004
therefore call ``verify_and_update_config`` directly, on ``VllmConfig``-shaped
namespaces whose attribute paths are the ones core's check reads
(``VllmConfig._verify_persistent_state_config``, ``vllm/config/vllm.py``) plus
``model_config.enforce_eager``. A real ``VllmConfig`` on CPU is used where it
constructs (the hook runs inside ``VllmConfig.__post_init__`` via
``try_verify_and_update_config``, ``vllm/config/vllm.py``) and for the
model-state re-check, which needs configurations the hook would have refused.

The runner configurations (REG-ENG-005, REG-ENG-011, REG-ENG-021 to
REG-ENG-026) are judged at two stages here: the configuration hook
(``verify_and_update_config``, requested modes, breakable graphs predicted
from ``VLLM_USE_BREAKABLE_CUDAGRAPH``) and the construction re-check
(``_resolved_check``: both modes set, breakable graphs read from
``compilation_config.use_breakable_cudagraph``). Model and model-state
construction themselves, the shared ``engine_flags`` function and core's mode
resolution are in ``test_runner_modes.py``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from support import runner_conditions as rc

pytestmark = pytest.mark.engine

#: The environment variables core reads while it builds the engine
#: configuration that the hook predicts from (REG-ENG-022, REG-ENG-023).
BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"
COMPILE_DISABLE_ENV = "TORCH_COMPILE_DISABLE"


@pytest.fixture(autouse=True)
def _no_runner_env(monkeypatch):
    """Neither variable is inherited from the shell running the tests."""
    monkeypatch.delenv(BREAKABLE_ENV, raising=False)
    monkeypatch.delenv(COMPILE_DISABLE_ENV, raising=False)


@pytest.fixture
def runner_env(monkeypatch):
    """``runner_env(breakable=..., compile_disabled=...)``: set the variables
    the configuration hook predicts from. ``vllm.envs`` reads
    ``VLLM_USE_BREAKABLE_CUDAGRAPH`` lazily from ``os.environ`` unless its
    cache is on (``enable_envs_cache``); the cache is bypassed for the test
    if it is."""
    import vllm.envs as envs

    if envs._is_envs_cache_enabled():
        monkeypatch.setattr(envs, "__getattr__", envs.__getattr__.__wrapped__)

    def set_env(*, breakable: bool = False, compile_disabled: bool = False):
        if breakable:
            monkeypatch.setenv(BREAKABLE_ENV, "1")
        else:
            monkeypatch.delenv(BREAKABLE_ENV, raising=False)
        if compile_disabled:
            monkeypatch.setenv(COMPILE_DISABLE_ENV, "1")
        else:
            monkeypatch.delenv(COMPILE_DISABLE_ENV, raising=False)

    return set_env


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


@pytest.mark.parametrize("value", [True, False, None], ids=repr)
@pytest.mark.parametrize("runner", ["eager", "hook"])
# @spec REG-ENG-001
def test_no_async_scheduling_value_is_refused_by_the_hook(
    engine_config, platform, runner, value
):
    """``True``, ``False`` and unset (the hook runs before core resolves the
    option) are all accepted, in either runner configuration."""
    if runner == "eager":
        config = engine_config(async_scheduling=value)
    else:
        config = _hook_config(engine_config, platform)
        config.scheduler_config.async_scheduling = value
    _hook().verify_and_update_config(config)


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
# @spec REG-ENG-005, REG-ENG-011, REG-ENG-024
def test_a_runner_that_is_not_eager_needs_the_hook_configuration(
    engine_config, platform, value
):
    """No compilation settings at all (both modes unset): the hook accepts the
    unset modes on CUDA with the default arms, and off CUDA refuses with the
    ``--enforce-eager`` remedy."""
    platform(True)
    _hook().verify_and_update_config(engine_config(enforce_eager=value))
    platform(False)
    _refuse(engine_config(enforce_eager=value), "enforce_eager")


# @spec REG-VAL-003
def test_every_refused_setting_is_listed_in_one_error(engine_config, platform):
    platform(False)
    config = engine_config(
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
    _refuse(config, *(name for name in NAMES if name != "async_scheduling"))


# @spec REG-VAL-003, REG-ENG-019, REG-ENG-009
def test_native_burst_and_missing_hooks_are_listed_with_the_other_failures(
    engine_config, platform, monkeypatch
):
    """``verify_and_update_config`` called directly (core's own check would
    name prefix caching first): one ``ValueError`` lists the refused
    setting, native burst on the state layer and a missing core hook."""
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    platform(True)
    monkeypatch.delattr(ModelState, "post_warmup")
    config = engine_config(
        enable_prefix_caching=True,
        hf={"experimental_native_burst": True, "num_lookahead_tokens": 1},
    )
    message = _refuse(config, "enable_prefix_caching")
    assert re.search(r"(?i)experimental_native_burst|native burst", message), message
    assert "post_warmup" in message, message


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


@pytest.mark.parametrize("value", [True, False, None], ids=repr)
# @spec REG-ENG-001
def test_real_vllm_config_with_any_async_scheduling_value_constructs(
    make_model_config, make_vllm_config, card_dir, value
):
    """The hook refuses no value; the configuration constructs. Only
    acceptance is asserted: core resolves an unset value, and on CPU the
    platform's own hook forces asynchronous scheduling off
    (``CpuPlatform.check_and_update_config``), so a requested ``True`` need
    not survive."""
    vllm_config = make_vllm_config(make_model_config(card_dir), async_scheduling=value)
    assert isinstance(vllm_config.scheduler_config.async_scheduling, bool)


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


@pytest.mark.parametrize("value", [True, False, None], ids=repr)
# @spec REG-ENG-001, REG-ENG-006
def test_model_state_refuses_no_async_scheduling_value(
    good_vllm_config, build_model_state, value
):
    config = good_vllm_config()
    config.scheduler_config.async_scheduling = value
    build_model_state(config)


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
#
# REG-ENG-008 describes cores that support native burst; this build's core
# (the state layer) refuses native burst outright (REG-ENG-019, in
# test_runner_modes.py). These are unit tests of the check function only.
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
# The hook configuration (REG-ENG-005, REG-ENG-006, REG-ENG-011,
# REG-ENG-021 to REG-ENG-026)
# --------------------------------------------------------------------------

#: The default segment arms (REG-CARD-008).
DEFAULT_ARMS = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
}

#: Compilation modes by short name (``None``: unset).
MODES = ("none", "stock", "trace", "compile")
#: ``cudagraph_mode`` values by short name (``None``: unset).
GRAPHS = ("none", "piecewise", "full_decode_only", "full_and_piecewise", "full")

#: REG-ENG-011: the graph modes and compilation modes accepted with
#: breakable graphs off.
ACCEPTED_GRAPHS = ("none", "piecewise", "full_decode_only", "full_and_piecewise")
ACCEPTED_MODES = ("none", "trace", "compile")
#: REG-ENG-026: the graph modes accepted with breakable graphs on.
BREAKABLE_GRAPHS = ("none", "full_decode_only", "full_and_piecewise")


def _compilation_mode(name):
    from vllm.config.compilation import CompilationMode

    return {
        None: None,
        "none": CompilationMode.NONE,
        "stock": CompilationMode.STOCK_TORCH_COMPILE,
        "trace": CompilationMode.DYNAMO_TRACE_ONCE,
        "compile": CompilationMode.VLLM_COMPILE,
    }[name]


def _graph_mode(name):
    from vllm.config import CUDAGraphMode

    return None if name is None else CUDAGraphMode[name.upper()]


def _hook_config(
    engine_config,
    platform,
    *,
    mode="none",
    cudagraph="none",
    breakable=None,
    level=None,
    cuda=True,
    **hf,
):
    """``enforce_eager`` off with the given compilation settings, platform and
    served-configuration fields (the default arms unless ``hf`` says
    otherwise). ``mode`` / ``cudagraph`` name the modes (``None``: unset);
    ``breakable`` is ``compilation_config.use_breakable_cudagraph`` (``None``,
    as core leaves it before it resolves breakable graphs, by default)."""
    platform(cuda)
    return engine_config(
        enforce_eager=False,
        compilation_config=SimpleNamespace(
            mode=_compilation_mode(mode),
            cudagraph_mode=_graph_mode(cudagraph),
            use_breakable_cudagraph=breakable,
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


def _refuse_resolved(config) -> str:
    with pytest.raises(ValueError) as excinfo:
        _resolved_check(config)
    return str(excinfo.value)


def _check(config, stage):
    if stage == "hook":
        _hook().verify_and_update_config(config)
    else:
        _resolved_check(config)


def _refused_at(config, stage) -> str:
    return _refuse(config) if stage == "hook" else _refuse_resolved(config)


def _has(message: str, *words: str) -> None:
    for word in words:
        assert word.lower() in message.lower(), (word, message)


STAGES = pytest.mark.parametrize("stage", ["hook", "constructor"])

_HOOK_ACCEPTED = {
    "default arms": {},
    "graphed encoder only": {"decode_dispatch_arm": "dense-eager"},
    "graphed decoder only": {"encoder_execution_arm": "eager"},
}


@pytest.mark.parametrize("case", list(_HOOK_ACCEPTED))
@STAGES
# @spec REG-ENG-005, REG-ENG-006
def test_the_hook_configuration_is_accepted(engine_config, platform, case, stage):
    config = _hook_config(
        engine_config,
        platform,
        breakable=None if stage == "hook" else False,
        **_HOOK_ACCEPTED[case],
    )
    _check(config, stage)


def _no_state_layer_refusal(message: str) -> None:
    """The target core defines ``ModelState.state_group_layouts``: no clause
    of the refusal refuses transaction graphs on account of the state layer
    or core's layout accessor (TXG-CFG-016)."""
    from support import txg_conditions as c

    assert c.state_layer_refusals(message) == [], message


@STAGES
# @spec REG-ENG-005, REG-ENG-006, REG-ENG-025, TXG-CFG-016
def test_transaction_graphs_at_80_ms_are_accepted_under_the_restricted_startup(
    engine_config, platform, stage
):
    """80 ms, the default arms, NONE/NONE on CUDA, synchronous scheduling and
    a block count of the admission limit plus one: accepted on the target
    core."""
    config = _hook_config(
        engine_config,
        platform,
        breakable=None if stage == "hook" else False,
        num_lookahead_tokens=0,
        transaction_graph_populations=[2],
    )
    config.cache_config.num_gpu_blocks_override = (
        config.scheduler_config.max_num_seqs + 1
    )
    _check(config, stage)


@STAGES
# @spec REG-ENG-005, REG-ENG-006, TXG-CFG-016, TXG-CFG-019, TXG-CFG-011
def test_transaction_graphs_without_a_block_count_are_refused_for_that_alone(
    engine_config, platform, stage
):
    """Every other condition holds; without an explicit block count or KV
    budget the refusal names the pool condition (TXG-CFG-019) and nothing
    refuses transaction graphs for the state layer itself."""
    config = _hook_config(
        engine_config,
        platform,
        breakable=None if stage == "hook" else False,
        num_lookahead_tokens=0,
        transaction_graph_populations=[2],
    )
    message = _refused_at(config, stage)
    assert re.search(r"num[_-]gpu[_-]blocks[_-]override", message), message
    assert rc.TRANSACTION_GRAPH_KEY in message, message
    _no_state_layer_refusal(message)


def _restricted(config):
    """The block count of the restricted startup (TXG-CFG-019): the
    admission limit plus one."""
    config.cache_config.num_gpu_blocks_override = (
        config.scheduler_config.max_num_seqs + 1
    )
    return config


@pytest.mark.parametrize("value", [True, None], ids=repr)
@STAGES
# @spec REG-ENG-001, TXG-CFG-017
def test_transaction_graphs_refuse_any_async_scheduling_but_false(
    engine_config, platform, stage, value
):
    """REG-ENG-001's exception: while transaction graphs are on, an
    ``async_scheduling`` that is not ``False`` is refused, naming the value
    and ``--no-async-scheduling``."""
    config = _restricted(
        _hook_config(
            engine_config,
            platform,
            breakable=None if stage == "hook" else False,
            num_lookahead_tokens=0,
            transaction_graph_populations=[2],
        )
    )
    config.scheduler_config.async_scheduling = value
    message = _refused_at(config, stage)
    assert "--no-async-scheduling" in message, message
    assert rc.TRANSACTION_GRAPH_KEY in message, message


#: REG-ENG-005 while transaction graphs are on: each condition's one remedy,
#: and the alternatives it must not offer (TXG-CFG-018 accepts only an
#: explicit NONE).
_TXG_ONLY_REMEDY = {
    "graph mode": (
        {"cudagraph": "full"},
        r"(?i)-cc\.cudagraph_mode[ =]none",
        (r"(?i)\bunset\b",),
    ),
    "breakable piecewise": (
        {"cudagraph": "piecewise", "breakable": True},
        r"(?i)-cc\.cudagraph_mode[ =]none",
        (r"VLLM_USE_BREAKABLE_CUDAGRAPH=0",),
    ),
    "compilation mode": (
        {"mode": "stock"},
        r"(?i)-cc\.mode[ =]none",
        (r"(?i)\bunset\b",),
    ),
}


@pytest.mark.parametrize("condition", list(_TXG_ONLY_REMEDY))
# @spec REG-ENG-005, REG-ENG-021, REG-ENG-022, REG-ENG-023
def test_with_transaction_graphs_on_each_mode_remedy_is_the_explicit_none_alone(
    engine_config, platform, condition
):
    """At model-state construction (the modes as core's build left them):
    the clause naming the condition gives ``-cc.cudagraph_mode none`` or
    ``-cc.mode none`` and no other alternative."""
    modes, remedy, forbidden = _TXG_ONLY_REMEDY[condition]
    settings = {"breakable": False, **modes}
    config = _restricted(
        _hook_config(
            engine_config,
            platform,
            num_lookahead_tokens=0,
            transaction_graph_populations=[2],
            **settings,
        )
    )
    message = _refuse_resolved(config)
    found = [c for c in rc.clauses(message) if rc.names_condition(c, condition)]
    assert found, (condition, message)
    for clause in found:
        assert re.search(remedy, clause), (condition, clause)
        for alternative in forbidden:
            assert not re.search(alternative, clause), (condition, clause)


@pytest.mark.parametrize("cudagraph", ACCEPTED_GRAPHS)
@pytest.mark.parametrize("mode", ACCEPTED_MODES)
@STAGES
# @spec REG-ENG-011, REG-ENG-005, REG-ENG-006
def test_graph_and_compilation_modes_with_breakable_graphs_off_are_accepted(
    engine_config, platform, runner_env, stage, mode, cudagraph
):
    runner_env(breakable=False)
    config = _hook_config(
        engine_config,
        platform,
        mode=mode,
        cudagraph=cudagraph,
        breakable=None if stage == "hook" else False,
    )
    _check(config, stage)


@pytest.mark.parametrize("level", [None, 0, 1, 2, 3])
@pytest.mark.parametrize(
    "mode, cudagraph",
    [
        (None, None),
        *((None, g) for g in ACCEPTED_GRAPHS),
        *((m, None) for m in ACCEPTED_MODES),
    ],
    ids=str,
)
# @spec REG-ENG-011
def test_unset_modes_are_accepted_by_the_hook(
    engine_config, platform, mode, cudagraph, level
):
    """The hook runs before core fills the optimization level's defaults, so
    it does not duplicate core's table."""
    config = _hook_config(
        engine_config, platform, mode=mode, cudagraph=cudagraph, level=level
    )
    _hook().verify_and_update_config(config)


@pytest.mark.parametrize(
    "unset, mode, cudagraph, words",
    [
        ("cudagraph_mode", "none", None, ("cudagraph_mode",)),
        ("compilation mode", None, "none", ("mode",)),
        ("both", None, None, ("cudagraph_mode",)),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
# @spec REG-ENG-006
def test_unset_modes_are_refused_at_construction(
    engine_config, platform, unset, mode, cudagraph, words
):
    config = _hook_config(
        engine_config, platform, mode=mode, cudagraph=cudagraph, breakable=False
    )
    message = _refuse_resolved(config)
    _has(message, *words)
    if unset != "cudagraph_mode":
        assert re.search(
            r"(?i)compilation mode|-cc\.mode|compilation_config\.mode", message
        ), message


@pytest.mark.parametrize("cudagraph", [None, *ACCEPTED_GRAPHS])
# @spec REG-ENG-011, REG-ENG-023
def test_requested_stock_compile_is_accepted_by_the_hook_while_compile_is_disabled(
    engine_config, platform, runner_env, cudagraph
):
    """``TORCH_COMPILE_DISABLE=1``: core sets the compilation mode to NONE."""
    runner_env(compile_disabled=True)
    config = _hook_config(engine_config, platform, mode="stock", cudagraph=cudagraph)
    _hook().verify_and_update_config(config)


@pytest.mark.parametrize("cudagraph", [None, *BREAKABLE_GRAPHS])
@pytest.mark.parametrize("mode", [None, *MODES])
# @spec REG-ENG-026, REG-ENG-022, REG-ENG-023
def test_breakable_graphs_on_are_accepted_by_the_hook_with_any_requested_mode(
    engine_config, platform, runner_env, mode, cudagraph
):
    """``VLLM_USE_BREAKABLE_CUDAGRAPH=1``: core turns breakable graphs on and
    sets the compilation mode to NONE, so even a requested
    ``STOCK_TORCH_COMPILE`` is accepted."""
    runner_env(breakable=True)
    config = _hook_config(engine_config, platform, mode=mode, cudagraph=cudagraph)
    _hook().verify_and_update_config(config)


@pytest.mark.parametrize("cudagraph", BREAKABLE_GRAPHS)
# @spec REG-ENG-026, REG-ENG-022, REG-ENG-006
def test_breakable_graphs_on_are_accepted_at_construction_with_compilation_mode_none(
    engine_config, platform, cudagraph
):
    config = _hook_config(
        engine_config, platform, mode="none", cudagraph=cudagraph, breakable=True
    )
    _resolved_check(config)


# @spec REG-ENG-022, REG-ENG-006
def test_construction_reads_breakable_graphs_from_the_resolved_setting(
    engine_config, platform, runner_env
):
    """At construction the resolved ``use_breakable_cudagraph`` decides, not
    the environment variable."""
    runner_env(breakable=True)
    off = _hook_config(
        engine_config, platform, mode="compile", cudagraph="piecewise", breakable=False
    )
    _resolved_check(off)
    runner_env(breakable=False)
    on = _hook_config(
        engine_config, platform, mode="none", cudagraph="piecewise", breakable=True
    )
    rc.assert_named_with_remedy(_refuse_resolved(on), "breakable piecewise")


#: case -> (stage, environment, settings, the unmet condition, named with a
#: remedy its spec allows: ``support.runner_conditions.CONDITIONS``).
#: ``stage`` is ``"hook"``, ``"constructor"`` or ``"both"``.
_HOOK_REFUSED = {
    "full graphs": ("both", {}, {"cudagraph": "full"}, "graph mode"),
    "full graphs under breakable graphs": (
        "both",
        {"breakable": True},
        {"cudagraph": "full", "breakable": True},
        "graph mode",
    ),
    "breakable piecewise": (
        "both",
        {"breakable": True},
        {"cudagraph": "piecewise", "breakable": True},
        "breakable piecewise",
    ),
    "stock torch.compile": ("both", {}, {"mode": "stock"}, "compilation mode"),
    "stock torch.compile surviving construction under breakable graphs": (
        "constructor",
        {"breakable": True},
        {"mode": "stock", "breakable": True},
        "compilation mode",
    ),
    "stock torch.compile surviving construction under TORCH_COMPILE_DISABLE": (
        "constructor",
        {"compile_disabled": True},
        {"mode": "stock"},
        "compilation mode",
    ),
    "not CUDA": ("both", {}, {"cuda": False}, "platform"),
    "no hook-captured graph": (
        "both",
        {},
        {"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"},
        "hook-captured graph",
    ),
    "compiled encoder": (
        "both",
        {},
        {"encoder_execution_arm": "dense-graphed"},
        "compiled encoder",
    ),
    "chunk buckets": (
        "both",
        {},
        {"chunk_bucket_graph_populations": [1]},
        "chunk buckets",
    ),
}

_REFUSED_CASES = [
    pytest.param(case, stage, id=f"{stage}-{case}")
    for case, (stages, *_rest) in _HOOK_REFUSED.items()
    for stage in (("hook", "constructor") if stages == "both" else (stages,))
]


@pytest.mark.parametrize("case, stage", _REFUSED_CASES)
# @spec REG-ENG-005, REG-ENG-006, REG-ENG-021, REG-ENG-022, REG-ENG-023
# @spec REG-ENG-024, REG-ENG-025
def test_each_unmet_hook_condition_is_named_with_its_remedy(
    engine_config, platform, runner_env, case, stage
):
    """The condition by its own phrase, and one of the remedies its spec
    allows (REG-ENG-025's graph-selection remedy: either graphed arm or
    ``--enforce-eager``)."""
    _stages, env, settings, condition = _HOOK_REFUSED[case]
    runner_env(**env)
    settings = dict(settings)
    if stage == "hook":
        # Core has not resolved breakable graphs yet: the hook predicts them
        # from the environment only.
        settings.pop("breakable", None)
    else:
        settings.setdefault("breakable", False)
    config = _hook_config(engine_config, platform, **settings)
    rc.assert_named_with_remedy(_refused_at(config, stage), condition)


@STAGES
# @spec REG-ENG-005, REG-ENG-006, REG-VAL-003
def test_every_unmet_hook_condition_is_named_in_one_error(
    engine_config, platform, stage
):
    config = _hook_config(
        engine_config,
        platform,
        mode="stock",
        cudagraph="full",
        breakable=None if stage == "hook" else False,
        cuda=False,
        encoder_execution_arm="dense-graphed",
        decode_dispatch_arm="dense-eager",
        chunk_bucket_graph_populations=[1],
    )
    message = _refused_at(config, stage)
    for condition in (
        "graph mode",
        "compilation mode",
        "platform",
        "hook-captured graph",
        "compiled encoder",
        "chunk buckets",
    ):
        rc.assert_condition(message, condition)


@STAGES
# @spec REG-ENG-005, REG-ENG-006, REG-VAL-003
def test_breakable_piecewise_is_named_with_the_other_unmet_conditions(
    engine_config, platform, runner_env, stage
):
    runner_env(breakable=stage == "hook")
    config = _hook_config(
        engine_config,
        platform,
        mode="none",
        cudagraph="piecewise",
        breakable=None if stage == "hook" else True,
        cuda=False,
        chunk_bucket_graph_populations=[1],
    )
    message = _refused_at(config, stage)
    for condition in ("breakable piecewise", "platform", "chunk buckets"):
        rc.assert_condition(message, condition)


def _eager_remedies_name_removing_transaction_graphs(message: str) -> bool:
    """REG-ENG-005: every clause of ``message`` that names
    ``--enforce-eager`` also names removing ``transaction_graph_populations``
    (``support.runner_conditions.clauses``)."""
    remedies = rc.eager_remedy_clauses(message)
    assert remedies, message
    return all(rc.names_removal_of(c, rc.TRANSACTION_GRAPH_KEY) for c in remedies)


_EAGER_REMEDY_CONDITIONS = pytest.mark.parametrize(
    "settings",
    [
        {"cuda": False},
        {"encoder_execution_arm": "dense-graphed"},
        {"chunk_bucket_graph_populations": [1]},
    ],
    ids=["not CUDA", "compiled encoder", "chunk buckets"],
)


@_EAGER_REMEDY_CONDITIONS
# @spec REG-ENG-005
def test_an_enforce_eager_remedy_names_removing_transaction_graphs(
    engine_config, platform, settings
):
    """At 80 ms with the default arms, transaction graphs on: the clause
    carrying the ``--enforce-eager`` remedy itself names removing
    ``transaction_graph_populations`` (the transaction-graph conditions,
    which name the option too, are separate clauses)."""
    config = _hook_config(
        engine_config,
        platform,
        num_lookahead_tokens=0,
        transaction_graph_populations=[2],
        **settings,
    )
    assert _eager_remedies_name_removing_transaction_graphs(_refuse(config))


@pytest.mark.parametrize("value", [{}, None, False, "", 0], ids=repr)
# @spec REG-ENG-005
def test_a_malformed_transaction_graph_value_counts_as_on(
    engine_config, platform, value
):
    """TXG-CFG-014: one on/off predicate; ``{}`` and the like are on, so the
    ``--enforce-eager`` remedy's clause names removing the option."""
    config = _hook_config(
        engine_config,
        platform,
        cuda=False,
        num_lookahead_tokens=0,
        transaction_graph_populations=value,
    )
    assert _eager_remedies_name_removing_transaction_graphs(_refuse(config))


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
    message = _refuse(
        engine_config(enable_prefix_caching=True), "enable_prefix_caching"
    )
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
    """The serve flags of the hook configuration (``-cc.mode none``, graph mode
    left unset) with the default arms construct, and the model state's re-check
    accepts the result. Core resolves an unset ``cudagraph_mode`` to a mode
    other than FULL at construction (REG-ENG-021) and reduces it to NONE only
    when the runner reads the model state's graph support."""
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
    assert config.compilation_config.cudagraph_mode not in (
        None,
        CUDAGraphMode.FULL,
    )
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


# @spec REG-ENG-006, REG-VAL-008, TXG-CFG-011, TXG-CFG-016
def test_model_state_reruns_the_transaction_graph_checks(
    good_vllm_config, build_model_state
):
    """The option turned on after the hook ran (under ``--enforce-eager``,
    with no block count) is refused at construction, naming the option and
    the conditions it violates (the eager runner, TXG-CFG-009; the pool,
    TXG-CFG-019), and nothing refuses it for the state layer itself."""
    config = good_vllm_config()
    config.model_config.hf_config.transaction_graph_populations = [2]
    with pytest.raises(ValueError) as excinfo:
        build_model_state(config)
    message = str(excinfo.value)
    assert "transaction_graph_populations" in message
    assert "--enforce-eager" in message, message
    _no_state_layer_refusal(message)
