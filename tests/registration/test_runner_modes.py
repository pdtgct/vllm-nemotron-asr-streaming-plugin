# SPDX-License-Identifier: Apache-2.0
"""The runner configurations at model and model-state construction, core's
mode resolution, and the model state's runner-graph guards (REG-ENG-001,
REG-ENG-006, REG-ENG-011, REG-ENG-012, REG-ENG-016 to REG-ENG-020,
REG-ENG-026, REG-ENG-027, REG-VAL-003, TXG-CFG-016: transaction graphs are
refused only on a core without ``ModelState.state_group_layouts``; the target
core has it).

The configuration-hook and namespace re-check matrices are in
``test_engine_settings.py``. Here the model and its model state are the real
classes on CPU (``support.model_harness``): a real ``VllmConfig`` is built
under ``--enforce-eager`` (so the configuration hook inside its construction
accepts it), then given the settings under test as core's configuration build
would leave them, and vLLM's current platform reports CUDA
(``support.fake_cuda.set_cuda_platform``) unless a test says otherwise.
Graphed arms run on ``support.model_harness.FakeGraphRuntime``.

Core's mode resolution (REG-ENG-012) is core's own
``CompilationConfig.resolve_cudagraph_mode_and_sizes``
(``vllm/config/compilation.py``); the configuration it receives is built by
``_core_build``, which restates core's configuration build at the pinned core
(``vllm/config/vllm.py``: ``TORCH_COMPILE_DISABLE``,
``_maybe_enable_breakable_cudagraph``, the optimization level's defaults
``OPTIMIZATION_LEVEL_TO_CONFIG`` and ``_normalize_piecewise_cudagraph_mode``).
"""

from __future__ import annotations

import itertools
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from support import model_harness as h
from support import runner_conditions as rc

pytestmark = pytest.mark.engine

BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"
COMPILE_DISABLE_ENV = "TORCH_COMPILE_DISABLE"

#: The card's default segment arms (REG-CARD-008), both hook-captured.
DEFAULT_ARMS = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
}

#: REG-ENG-011 (breakable graphs off) and REG-ENG-026 (on, compilation mode
#: NONE): what model and model-state construction accept.
ACCEPTED_GRAPHS = ("none", "piecewise", "full_decode_only", "full_and_piecewise")
ACCEPTED_MODES = ("none", "trace", "compile")
BREAKABLE_GRAPHS = ("none", "full_decode_only", "full_and_piecewise")

#: The statement of the model state's runner-graph refusals (REG-ENG-017,
#: REG-ENG-029).
NO_RUNNER_GRAPH = "supports no runner graph"


def _compilation_mode(name):
    """A compilation mode by short name (``None``: unset); a mode passes
    through."""
    from vllm.config.compilation import CompilationMode

    if isinstance(name, CompilationMode):
        return name
    return {
        None: None,
        "none": CompilationMode.NONE,
        "stock": CompilationMode.STOCK_TORCH_COMPILE,
        "trace": CompilationMode.DYNAMO_TRACE_ONCE,
        "compile": CompilationMode.VLLM_COMPILE,
    }[name]


def _graph_mode(name):
    """A ``cudagraph_mode`` by short name (``None``: unset); a mode passes
    through."""
    from vllm.config import CUDAGraphMode

    if name is None or isinstance(name, CUDAGraphMode):
        return name
    return CUDAGraphMode[name.upper()]


def _has(message: str, *words: str) -> None:
    for word in words:
        assert word.lower() in message.lower(), (word, message)


def _names_compilation_mode(message: str) -> bool:
    return bool(
        re.search(r"(?i)compilation mode|-cc\.mode|compilation_config\.mode", message)
    )


@pytest.fixture(autouse=True)
def _no_runner_env(monkeypatch):
    monkeypatch.delenv(BREAKABLE_ENV, raising=False)
    monkeypatch.delenv(COMPILE_DISABLE_ENV, raising=False)


@pytest.fixture
def stages(tmp_path, monkeypatch, platform):
    """Build the configuration, the model and the model state one stage at a
    time (see the module docstring)."""

    h.FakeGraphRuntime().install(monkeypatch)
    card = h.write_runnable_card(tmp_path / "card")

    class Stages:
        def config(self, *, right_context: int = 3, overrides=None, **settings):
            config = h.build_vllm_config(
                card,
                right_context=right_context,
                max_num_seqs=2,
                overrides={**DEFAULT_ARMS, **(overrides or {})},
            )
            return self.apply(config, **settings)

        def apply(
            self,
            config,
            *,
            cudagraph="none",
            mode="none",
            breakable=False,
            eager=False,
            cuda=True,
            async_scheduling=False,
            hf=None,
            prefix_caching=False,
        ):
            """Settings the configuration hook inside ``VllmConfig`` did not
            see, as core's build would leave them."""
            for key, value in (hf or {}).items():
                setattr(config.model_config.hf_config, key, value)
            config.model_config.enforce_eager = eager
            compilation = config.compilation_config
            compilation.mode = _compilation_mode(mode)
            compilation.cudagraph_mode = _graph_mode(cudagraph)
            compilation.use_breakable_cudagraph = breakable
            config.scheduler_config.async_scheduling = async_scheduling
            config.cache_config.enable_prefix_caching = prefix_caching
            platform(cuda)
            return config

        def model(self, config):
            return h.build_model(config)

        def state(self, config, model):
            return h.build_model_state(config, model)

    return Stages()


def _refusal(fn, *args) -> str:
    with pytest.raises(ValueError) as excinfo:
        fn(*args)
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# REG-ENG-011, REG-ENG-026: accepted at model and model-state construction


@pytest.mark.parametrize("cudagraph", ACCEPTED_GRAPHS)
@pytest.mark.parametrize("mode", ACCEPTED_MODES)
# @spec REG-ENG-011, REG-ENG-006
def test_construction_accepts_every_mode_pair_with_breakable_graphs_off(
    stages, mode, cudagraph
):
    config = stages.config(mode=mode, cudagraph=cudagraph, breakable=False)
    stages.state(config, stages.model(config))


@pytest.mark.parametrize("cudagraph", BREAKABLE_GRAPHS)
# @spec REG-ENG-026, REG-ENG-006
def test_construction_accepts_breakable_graphs_with_compilation_mode_none(
    stages, cudagraph
):
    config = stages.config(mode="none", cudagraph=cudagraph, breakable=True)
    stages.state(config, stages.model(config))


@pytest.mark.parametrize("value", [True, False, None], ids=repr)
@pytest.mark.parametrize("runner", ["eager", "hook"])
# @spec REG-ENG-001
def test_construction_refuses_no_async_scheduling_value(stages, runner, value):
    config = stages.config(eager=runner == "eager", async_scheduling=value)
    stages.state(config, stages.model(config))


# ---------------------------------------------------------------------------
# REG-ENG-006: model and model-state construction re-check the shared rule

#: case -> (settings, the hook condition named with its remedy
#: (``support.runner_conditions``), or ``None`` and words the refusal must
#: contain).
_REFUSED_AT_CONSTRUCTION = {
    "unset cudagraph_mode": ({"cudagraph": None}, None, ("cudagraph_mode",)),
    "unset compilation mode": ({"mode": None}, None, ()),
    "surviving STOCK_TORCH_COMPILE": ({"mode": "stock"}, "compilation mode", ()),
    "FULL": ({"cudagraph": "full"}, "graph mode", ()),
    "PIECEWISE under resolved breakable graphs": (
        {"cudagraph": "piecewise", "breakable": True},
        "breakable piecewise",
        (),
    ),
    "not CUDA": ({"cuda": False}, "platform", ()),
    "prefix caching": ({"prefix_caching": True}, None, ("prefix",)),
}


@pytest.mark.parametrize("case", list(_REFUSED_AT_CONSTRUCTION))
@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-ENG-021, REG-ENG-022, REG-ENG-023, REG-ENG-024
def test_construction_refuses_what_the_shared_rule_refuses(stages, stage, case):
    settings, condition, words = _REFUSED_AT_CONSTRUCTION[case]
    if stage == "model":
        config = stages.config(**settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config()
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
    if condition is not None:
        rc.assert_named_with_remedy(message, condition)
    _has(message, *words)
    if case == "unset compilation mode":
        assert _names_compilation_mode(message), message


@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-VAL-003
def test_construction_lists_every_failed_condition_in_one_error(stages, stage):
    settings = {
        "cudagraph": "full",
        "mode": "stock",
        "cuda": False,
        "prefix_caching": True,
        "hf": {"chunk_bucket_graph_populations": [1]},
    }
    if stage == "model":
        config = stages.config(**settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config()
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
    for condition in ("graph mode", "compilation mode", "platform", "chunk buckets"):
        rc.assert_condition(message, condition)
    assert re.search(r"(?i)prefix caching|enable_prefix_caching", message), message


def _no_state_layer_refusal(message: str) -> None:
    """The target core defines ``ModelState.state_group_layouts``: no clause
    of the refusal refuses transaction graphs on account of the state layer
    or core's layout accessor (TXG-CFG-016)."""
    from support import txg_conditions as c

    assert c.state_layer_refusals(message) == [], message


#: TXG-CFG-019's pool condition: an explicit block count or KV budget.
_POOL_CONDITION = re.compile(r"num_gpu_blocks_override|num-gpu-blocks-override")


@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-VAL-008, REG-018, TXG-CFG-016, TXG-CFG-019
def test_construction_accepts_transaction_graphs_under_the_restricted_startup(
    stages, stage
):
    """On the target core (``ModelState.state_group_layouts`` defined), with
    synchronous scheduling, explicit ``cudagraph_mode`` and compilation mode
    NONE, a block count of the admission limit plus one and a population
    within the limit, construction accepts transaction graphs."""
    settings = {"hf": {"transaction_graph_populations": [2]}}

    def restricted(config):
        config.cache_config.num_gpu_blocks_override = 3  # max_num_seqs 2, + 1
        return config

    if stage == "model":
        config = restricted(stages.config(right_context=0, **settings))
        stages.model(config)
    else:
        config = restricted(stages.config(right_context=0))
        model = stages.model(config)
        stages.apply(config, **settings)
        stages.state(config, model)


@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-VAL-008, TXG-CFG-016, TXG-CFG-019, TXG-CFG-011
def test_construction_refuses_transaction_graphs_outside_the_restricted_startup(
    stages, stage
):
    """Without an explicit block count or KV budget (TXG-CFG-019) the
    construction refuses, naming that condition, and refuses nothing for the
    state layer itself."""
    settings = {"hf": {"transaction_graph_populations": [2]}}
    if stage == "model":
        config = stages.config(right_context=0, **settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config(right_context=0)
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
    assert _POOL_CONDITION.search(message), message
    assert rc.TRANSACTION_GRAPH_KEY in message, message
    _no_state_layer_refusal(message)


# ---------------------------------------------------------------------------
# REG-ENG-019: native burst is refused on the state layer


def _assert_native_burst_refusal(message: str) -> None:
    assert re.search(r"(?i)experimental_native_burst|native burst", message), message
    _has(message, "state layer", "remov")


@pytest.mark.parametrize("runner", ["eager", "hook"])
# @spec REG-ENG-019
def test_native_burst_is_refused_by_the_configuration_hook(
    engine_config, platform, runner
):
    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    platform(True)
    settings = {}
    if runner == "hook":
        settings = {
            "enforce_eager": False,
            "compilation_config": SimpleNamespace(
                mode=_compilation_mode("none"),
                cudagraph_mode=_graph_mode("none"),
                use_breakable_cudagraph=None,
            ),
        }
    config = engine_config(
        hf={"experimental_native_burst": True, "num_lookahead_tokens": 1}, **settings
    )
    _assert_native_burst_refusal(
        _refusal(Nemotron3_5AsrStreamingConfig.verify_and_update_config, config)
    )


@pytest.mark.parametrize("runner", ["eager", "hook"])
@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-019, REG-ENG-006
def test_native_burst_is_refused_at_construction(stages, stage, runner):
    """At 160 ms (lookahead 1), where native burst's own checks pass."""
    settings = {"eager": runner == "eager", "hf": {"experimental_native_burst": True}}
    if stage == "model":
        config = stages.config(right_context=1, **settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config(right_context=1, eager=runner == "eager")
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
    _assert_native_burst_refusal(message)


# ---------------------------------------------------------------------------
# REG-ENG-027: one engine_flags function, applied at all three stages


def _install_runner_mode_refusals(monkeypatch, fn) -> None:
    """Make ``fn`` ``engine_flags.runner_mode_refusals`` and replace every
    package module's imported name for the real one."""
    import vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2  # noqa: F401
    import vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr  # noqa: F401
    import vllm_nemotron_asr_streaming.streaming_config  # noqa: F401
    from vllm_nemotron_asr_streaming import engine_flags

    real = getattr(engine_flags, "runner_mode_refusals", None)
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("vllm_nemotron_asr_streaming")
            and real is not None
            and getattr(module, "runner_mode_refusals", None) is real
        ):
            monkeypatch.setattr(module, "runner_mode_refusals", fn)
    monkeypatch.setattr(engine_flags, "runner_mode_refusals", fn, raising=False)


# @spec REG-ENG-027
def test_the_runner_mode_function_exists_in_engine_flags():
    import inspect

    from vllm_nemotron_asr_streaming.engine_flags import runner_mode_refusals

    parameters = inspect.signature(runner_mode_refusals).parameters
    assert parameters["stage"].kind is inspect.Parameter.KEYWORD_ONLY


# @spec REG-ENG-027
def test_every_stage_evaluates_the_conditions_through_the_one_function(
    stages, engine_config, platform, monkeypatch
):
    """A condition only the function can produce reaches the refusal of the
    configuration hook (stage ``"hook"``), model construction and model-state
    construction (stage ``"constructor"``)."""
    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    calls: list[str] = []
    sentinel = "sentinel runner-mode condition"

    def accept(vllm_config, *, stage):
        calls.append(stage)
        return []

    def refuse(vllm_config, *, stage):
        calls.append(stage)
        return [sentinel]

    hook_config = engine_config(
        enforce_eager=False,
        compilation_config=SimpleNamespace(
            mode=_compilation_mode("none"),
            cudagraph_mode=_graph_mode("none"),
            use_breakable_cudagraph=None,
        ),
    )

    platform(True)
    _install_runner_mode_refusals(monkeypatch, refuse)
    with pytest.raises(ValueError, match=sentinel):
        Nemotron3_5AsrStreamingConfig.verify_and_update_config(hook_config)
    assert calls == ["hook"], calls

    _install_runner_mode_refusals(monkeypatch, accept)
    config = stages.config()
    _install_runner_mode_refusals(monkeypatch, refuse)
    calls.clear()
    with pytest.raises(ValueError, match=sentinel):
        stages.model(config)
    assert calls == ["constructor"], calls

    _install_runner_mode_refusals(monkeypatch, accept)
    config = stages.config()
    model = stages.model(config)
    _install_runner_mode_refusals(monkeypatch, refuse)
    calls.clear()
    with pytest.raises(ValueError, match=sentinel):
        stages.state(config, model)
    assert calls == ["constructor"], calls


# ---------------------------------------------------------------------------
# REG-ENG-016: the model state reports no runner-graph support


@pytest.fixture
def eager_config(make_model_config, make_vllm_config, card_dir):
    """Factory: the accepted ``--enforce-eager`` configuration on CPU."""
    return lambda: make_vllm_config(make_model_config(card_dir))


# @spec REG-ENG-016
def test_model_state_reports_no_runner_graph_support(eager_config, build_model_state):
    from vllm.v1.attention.backend import AttentionCGSupport

    support, name = build_model_state(eager_config()).get_additional_cg_support()
    assert support == AttentionCGSupport.NEVER
    assert isinstance(name, str) and "nemotron" in name.lower(), name


# ---------------------------------------------------------------------------
# REG-ENG-012: core's resolution yields NONE; the inventory ignores the modes


def _core_build(graph, mode, *, breakable, compile_disabled, level):
    """``(cudagraph_mode, compilation mode)`` as core's configuration build
    leaves a request on CUDA: ``TORCH_COMPILE_DISABLE=1`` and breakable graphs
    set the mode to NONE; an unset mode becomes ``VLLM_COMPILE`` above O0,
    NONE at O0; an unset ``cudagraph_mode`` takes the level's default; a
    piecewise mode loses its piecewise part when the mode is not
    ``VLLM_COMPILE`` and breakable graphs are off."""
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode
    from vllm.config.vllm import OPTIMIZATION_LEVEL_TO_CONFIG, OptimizationLevel

    if compile_disabled or breakable:
        mode = CompilationMode.NONE
    if mode is None:
        mode = (
            CompilationMode.VLLM_COMPILE
            if level > OptimizationLevel.O0
            else CompilationMode.NONE
        )
    if graph is None:
        graph = OPTIMIZATION_LEVEL_TO_CONFIG[level]["compilation_config"][
            "cudagraph_mode"
        ]
    if (
        graph.requires_piecewise_compilation()
        and mode != CompilationMode.VLLM_COMPILE
        and not breakable
    ):
        graph = graph.without_piecewise()
    assert isinstance(graph, CUDAGraphMode)
    return graph, mode


def _accepted_at_construction(graph, mode, breakable) -> bool:
    """REG-ENG-011 and REG-ENG-026, the model state's acceptance."""
    if breakable:
        return graph in tuple(map(_graph_mode, BREAKABLE_GRAPHS)) and mode == (
            _compilation_mode("none")
        )
    return graph in tuple(map(_graph_mode, ACCEPTED_GRAPHS)) and mode in tuple(
        map(_compilation_mode, ACCEPTED_MODES)
    )


def _accepted_built_configurations():
    """Every distinct ``(cudagraph_mode, mode, breakable)`` core's build
    leaves from any request (each graph mode, unset included, each compilation
    mode, unset included, breakable graphs and ``TORCH_COMPILE_DISABLE`` on or
    off, each optimization level) that construction accepts."""
    from vllm.config.vllm import OptimizationLevel

    built = set()
    for graph, mode, breakable, disabled, level in itertools.product(
        (None, *map(_graph_mode, (*ACCEPTED_GRAPHS, "full"))),
        (None, *map(_compilation_mode, ("none", "stock", "trace", "compile"))),
        (False, True),
        (False, True),
        tuple(OptimizationLevel),
    ):
        graph_built, mode_built = _core_build(
            graph, mode, breakable=breakable, compile_disabled=disabled, level=level
        )
        if _accepted_at_construction(graph_built, mode_built, breakable):
            built.add((graph_built, mode_built, breakable))
    return sorted(built, key=lambda c: (c[0].name, int(c[1]), c[2]))


def _configuration_id(configuration) -> str:
    graph, mode, breakable = configuration
    return f"{graph.name}-{mode.name}-{'breakable' if breakable else 'unbreakable'}"


def _accepted_built_params():
    """``_accepted_built_configurations()`` as test parameters (none where
    vLLM is not importable; every test here needs it)."""
    try:
        configurations = _accepted_built_configurations()
    except ImportError:
        return []
    return [pytest.param(c, id=_configuration_id(c)) for c in configurations]


#: Every configuration model-state construction accepts, as core's build
#: leaves it (REG-ENG-011, REG-ENG-026).
ACCEPTED_BUILT = _accepted_built_params()


# @spec REG-ENG-012
def test_the_accepted_built_configurations_reach_every_setting():
    """The enumeration is not vacuous: breakable graphs on and off, and the
    compilation modes under which ``@support_torch_compile`` compiles."""
    from vllm.config.compilation import CompilationMode

    configurations = _accepted_built_configurations()
    assert configurations, "no accepted configuration was enumerated"
    assert {c[2] for c in configurations} == {False, True}
    modes = {c[1] for c in configurations}
    assert {CompilationMode.VLLM_COMPILE, CompilationMode.DYNAMO_TRACE_ONCE} <= modes


def _runner_support(state):
    """``(min_cg_support, backend name)`` as core's runner resolves the mode
    with them (``GPUModelRunner.initialize_kv_cache``): the attention
    backends' support narrowed by the model state's own
    ``get_additional_cg_support()``. The attention side is taken at its most
    permissive (``ALWAYS``), so the model state's report alone decides."""
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo

    info = AttentionCGSupportInfo().narrow(*state.get_additional_cg_support())
    return info.min_cg_support, info.min_cg_attn_backend


# @spec REG-ENG-012, REG-ENG-016
def test_core_resolves_every_accepted_built_configuration_to_none(
    eager_config, build_model_state
):
    from vllm.config import CompilationConfig, CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    state = build_model_state(eager_config())
    support, backend = _runner_support(state)
    configurations = _accepted_built_configurations()
    assert configurations, "no accepted configuration was enumerated"
    resolved = {}
    for graph, mode, breakable in configurations:
        # Splitting ops matter only under VLLM_COMPILE; both possibilities.
        splitting = (
            (None, list(CompilationConfig._attention_ops))
            if mode == CompilationMode.VLLM_COMPILE
            else (None,)
        )
        for ops in splitting:
            for profiling in (False, True):
                compilation = CompilationConfig(
                    mode=mode,
                    cudagraph_mode=graph,
                    use_breakable_cudagraph=breakable,
                )
                compilation.splitting_ops = ops
                try:
                    result = compilation.resolve_cudagraph_mode_and_sizes(
                        support,
                        backend,
                        state.num_new_sampled_tokens_per_step,
                        kv_cache_config=None,
                        max_num_reqs=64,
                        is_profiling=profiling,
                        piecewise_capture_available=breakable,
                    )
                except ValueError as error:
                    result = error
                resolved[(graph.name, mode.name, breakable, ops is None, profiling)] = (
                    result
                )
    wrong = {k: v for k, v in resolved.items() if v != CUDAGraphMode.NONE}
    assert not wrong, (support, wrong)


def _decorated_submodules(model) -> list[str]:
    """Every submodule of a ``@support_torch_compile`` class, compiling or
    not (the decorator adds ``TorchCompileWithNoGuardsWrapper`` to the
    class's bases)."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, TorchCompileWithNoGuardsWrapper)
    ]


@pytest.mark.parametrize("configuration", ACCEPTED_BUILT)
# @spec REG-ENG-012
def test_the_model_has_no_compiled_submodule(stages, configuration):
    """Built under each accepted configuration, as core leaves it, where a
    decorated submodule would compile (``do_not_compile`` is false for every
    mode but NONE and ``STOCK_TORCH_COMPILE``,
    ``vllm/compilation/decorators.py``): core's
    ``piecewise_capture_available`` is then the resolved breakable setting
    alone."""
    from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

    graph, mode, breakable = configuration
    config = stages.config(cudagraph=graph, mode=mode, breakable=breakable)
    model = stages.model(config)
    assert not has_compiled_submodule(model)
    assert not _decorated_submodules(model)


# @spec REG-ENG-012
def test_the_compiled_submodule_checks_find_a_decorated_submodule(stages):
    """Negative control: a ``@support_torch_compile`` stand-in in the model
    is found by core's check under ``VLLM_COMPILE`` and by the class check
    under every mode (core's check no longer sees it under NONE, which is
    why the model is built under the compiling modes above)."""
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config.vllm import set_current_vllm_config
    from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

    # Dimensions given: this module's annotations are strings.
    @support_torch_compile(dynamic_arg_dims={"x": 0})
    class CompiledStandIn(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    for mode, compiles in (("compile", True), ("none", False)):
        # vLLM's CPU platform: with the platform reporting CUDA, importing the
        # compilation backend reaches CUDA custom ops a CPU host lacks.
        config = stages.config(eager=True, cuda=False)
        model = stages.model(config)
        config.compilation_config.mode = _compilation_mode(mode)
        with set_current_vllm_config(config):
            model.add_module("stand_in", CompiledStandIn(vllm_config=config))
        assert has_compiled_submodule(model) is compiles, mode
        assert _decorated_submodules(model) == ["stand_in"], mode


def _core_capture_sizes(config) -> None:
    """The capture sizes core's configuration build gives the requested mode
    with ``enforce_eager`` off (``VllmConfig._set_cudagraph_sizes``: none for
    an explicit NONE)."""
    compilation = config.compilation_config
    compilation.cudagraph_num_of_warmups = 1
    compilation.cudagraph_capture_sizes = None
    compilation.max_cudagraph_capture_size = None
    config._set_cudagraph_sizes()


def _core_resolution(config, model, state, *, is_profiling: bool):
    """Core's mode resolution before each KV-cache initialization
    (``GPUModelRunner.initialize_kv_cache``), with the model state's support
    and core's own ``piecewise_capture_available``."""
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

    compilation = config.compilation_config
    support, backend = _runner_support(state)
    return compilation.resolve_cudagraph_mode_and_sizes(
        support,
        backend,
        state.num_new_sampled_tokens_per_step,
        kv_cache_config=None,
        max_num_reqs=config.scheduler_config.max_num_seqs,
        is_profiling=is_profiling,
        piecewise_capture_available=bool(
            is_breakable_cudagraph_enabled(compilation) or has_compiled_submodule(model)
        ),
    )


def _hook_inventory(tmp_path, monkeypatch, platform, *, cudagraph, mode, breakable):
    """The model's captured segment-graph entry keys after core's startup
    under the hook configuration, construction seeing ``cudagraph`` /
    ``mode`` / ``breakable`` as core's build leaves them, with core's capture
    sizes for that mode, core's resolution before each KV-cache
    initialization, and the profiling pool core sizes from the capture sizes
    (``min(max_num_seqs, largest capture size) or 1`` blocks,
    ``_init_minimal_kv_cache_for_profiling``). A key is the entry's runnable
    (encoder CHUNK entries or decoder tier entries) and its batch
    descriptor."""
    from support.core_order import CoreOrder
    from support.fake_cuda import FakeCuda, FakeStream
    from vllm.config import CUDAGraphMode

    graphs = h.FakeGraphRuntime().install(monkeypatch)
    FakeCuda(
        total=1 << 40, free=1 << 40, segment_graphs=graphs, per_segment_graph=3 << 20
    ).install_memory(monkeypatch)
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    name = _configuration_id((cudagraph, mode, breakable))
    card = h.write_runnable_card(tmp_path / f"card-{name}")
    config = h.build_vllm_config(
        card, right_context=3, max_num_seqs=2, overrides=DEFAULT_ARMS
    )
    config.model_config.enforce_eager = False
    compilation = config.compilation_config
    compilation.mode = mode
    compilation.cudagraph_mode = cudagraph
    compilation.use_breakable_cudagraph = breakable
    platform(True)
    _core_capture_sizes(config)
    profiling_blocks = (
        min(
            config.scheduler_config.max_num_seqs, compilation.max_cudagraph_capture_size
        )
        or 1
    )
    model = h.build_model(config)
    state = h.build_model_state(config, model)
    config.cache_config.kv_cache_memory_bytes = None
    config.cache_config.enable_extensible_kv_cache = False
    core = CoreOrder(h.Runner(config, model, state), capture_stream=FakeStream())
    core.prepare()
    captures = core.needs_capture()
    resolved = [_core_resolution(config, model, state, is_profiling=True)]
    if captures:
        spec = h.model_spec(model)
        core.bind_profiling(
            h.BindingsSpy(h.real_bindings(spec, capacity=profiling_blocks))
        )
        core.profiling_capture()
        core.clear()
    resolved.append(_core_resolution(config, model, state, is_profiling=False))
    core.bind_serving()
    if captures:
        core.serving_capture()
    core.post_warmup()
    assert resolved == [CUDAGraphMode.NONE, CUDAGraphMode.NONE], resolved
    keys = {
        (getattr(w.runnable, "__qualname__", type(w.runnable).__qualname__), d)
        for w, d in graphs.captured()
    }
    core.shutdown()
    return keys, profiling_blocks


@pytest.mark.parametrize("configuration", ACCEPTED_BUILT)
# @spec REG-ENG-012
def test_the_segment_graph_inventory_does_not_depend_on_the_modes(
    tmp_path, monkeypatch, platform, configuration
):
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    explicit_none, none_blocks = _hook_inventory(
        tmp_path,
        monkeypatch,
        platform,
        cudagraph=CUDAGraphMode.NONE,
        mode=CompilationMode.NONE,
        breakable=False,
    )
    graph, mode, breakable = configuration
    other, _blocks = _hook_inventory(
        tmp_path, monkeypatch, platform, cudagraph=graph, mode=mode, breakable=breakable
    )
    # An explicit NONE has no capture sizes, so a one-block profiling pool.
    assert none_blocks == 1
    # Encoder CHUNK entries and decoder tier entries, each at 1 and 2 rows.
    assert len(explicit_none) == 4, explicit_none
    assert other == explicit_none


# ---------------------------------------------------------------------------
# REG-ENG-017, REG-ENG-029: the model state refuses every runner-graph path


def _assert_no_runner_graph(message: str) -> None:
    _has(message, "own graphs", NO_RUNNER_GRAPH)


#: The class of the model state's refusals of a runner-graph path
#: (REG-ENG-017, REG-ENG-029): the plugin raises ``RuntimeError`` for them;
#: the specs name no class.
RUNNER_GRAPH_REFUSAL = RuntimeError


@pytest.mark.parametrize("pool", ["profiling", "serving"])
@pytest.mark.parametrize(
    "resolved", ["PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL"]
)
# @spec REG-ENG-017
def test_initialize_kv_cache_refuses_a_resolved_runner_graph_mode_before_capture(
    stages, monkeypatch, resolved, pool
):
    """Core's resolution left a mode other than NONE: the model state's
    ``initialize_kv_cache``, for the profiling pool or (a fixed budget, no
    profiling capture) the serving pool, raises before any graph is captured,
    naming the resolved mode whole."""
    from support.core_order import CoreOrder
    from support.fake_cuda import FakeCuda, FakeStream
    from vllm.config import CUDAGraphMode

    graphs = h.FakeGraphRuntime().install(monkeypatch)
    FakeCuda(
        total=1 << 40, free=1 << 40, segment_graphs=graphs, per_segment_graph=3 << 20
    ).install_memory(monkeypatch)
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    config = stages.config()
    model = stages.model(config)
    core = CoreOrder(
        h.Runner(config, model, stages.state(config, model)),
        capture_stream=FakeStream(),
    )
    core.prepare()
    captured = len(graphs.captures)
    config.compilation_config.cudagraph_mode = CUDAGraphMode[resolved]
    bind = core.bind_profiling if pool == "profiling" else core.bind_serving
    with pytest.raises(RUNNER_GRAPH_REFUSAL) as excinfo:
        bind()
    message = str(excinfo.value)
    assert rc.whole(resolved).search(message), message
    _assert_no_runner_graph(message)
    assert len(graphs.captures) == captured, "a graph was captured"


# @spec REG-ENG-029
def test_prepare_dummy_inputs_refuses_a_runner_capture(eager_config, build_model_state):
    state = build_model_state(eager_config())
    with pytest.raises(RUNNER_GRAPH_REFUSAL) as excinfo:
        state.prepare_dummy_inputs(1, 1)
    _assert_no_runner_graph(str(excinfo.value))


@pytest.mark.parametrize(
    "cudagraph, for_capture",
    [("FULL", False), ("NONE", True), ("PIECEWISE", True), ("FULL", True)],
)
# @spec REG-ENG-029
def test_prepare_attn_refuses_a_capture_or_a_full_step(
    eager_config, build_model_state, cudagraph, for_capture
):
    from vllm.config import CUDAGraphMode

    state = build_model_state(eager_config())
    with pytest.raises(RUNNER_GRAPH_REFUSAL) as excinfo:
        state.prepare_attn(
            None,
            CUDAGraphMode[cudagraph],
            (),
            None,
            [],
            None,
            for_capture=for_capture,
        )
    _assert_no_runner_graph(str(excinfo.value))


# @spec REG-ENG-029
def test_prepare_attn_for_an_uncaptured_step_is_accepted(
    eager_config, build_model_state
):
    from vllm.config import CUDAGraphMode

    state = build_model_state(eager_config())
    assert state.prepare_attn(None, CUDAGraphMode.NONE, (), None, [], None) == {}


# ---------------------------------------------------------------------------
# REG-ENG-018: one log line on the first initialize_kv_cache in a process
#
# The cases run in a fresh interpreter: the line is logged once per process,
# so a test in this process could not observe a first call. Two model states
# (two models, constructed under different modes) are initialized there:
# the first one twice, then the second.

_LOG_SNIPPET = """
import logging, sys, tempfile
from pathlib import Path
sys.path.insert(0, {tests!r})
import pytest
from support import model_harness as h
from support.fake_cuda import set_cuda_platform
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode

records = []

class Keep(logging.Handler):
    def emit(self, record):
        if not any(r is record for r in records):
            records.append(record)

keep = Keep(level=1)
for name in ("", "vllm", "vllm_nemotron_asr_streaming"):
    logging.getLogger(name).addHandler(keep)
logging.getLogger("vllm_nemotron_asr_streaming").setLevel(logging.DEBUG)

mp = pytest.MonkeyPatch()
h.FakeGraphRuntime().install(mp)

def built(graph, mode):
    # Built on the CPU platform, then CUDA (as the stages fixture does).
    set_cuda_platform(mp, False)
    card = h.write_runnable_card(Path(tempfile.mkdtemp()) / "card")
    config = h.build_vllm_config(
        card,
        right_context=3,
        max_num_seqs=2,
        overrides={{"encoder_execution_arm": "eager-graphed",
                    "decode_dispatch_arm": "dense-graphed"}},
    )
    set_cuda_platform(mp, True)
    config.model_config.enforce_eager = False
    config.compilation_config.mode = mode
    config.compilation_config.cudagraph_mode = graph
    config.compilation_config.use_breakable_cudagraph = False
    model = h.build_model(config)
    state = h.build_model_state(config, model)
    # Core's resolution (REG-ENG-012) before the KV cache is initialized.
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    return h.Runner(config, model, state), model

first = built(CUDAGraphMode.FULL_AND_PIECEWISE, CompilationMode.VLLM_COMPILE)
second = built(CUDAGraphMode.FULL_DECODE_ONLY, CompilationMode.DYNAMO_TRACE_ONCE)
calls = []
for runner, model in (first, first, second):
    start = len(records)
    runner.bind_pool(h.real_bindings(h.model_spec(model)))
    calls.append([r.getMessage() for r in records[start:]])
emit(calls=calls)
"""

#: The line's statements (REG-ENG-018): core's runner captures no graph over
#: the forward; the model captures its own graphs through the capture hook.
_NO_RUNNER_CAPTURE = re.compile(
    r"(?i)\brunner\b[^.;\n]{0,80}(?:\bno\b[^.;\n]{0,20}graph|captures? nothing"
    r"|does not capture|doesn't capture|never captures)"
    r"|\bno\b[^.;\n]{0,30}graph[^.;\n]{0,60}\brunner\b"
)
_OWN_GRAPHS_THROUGH_THE_HOOK = re.compile(
    r"(?i)own graphs[^.;\n]{0,80}(?:capture hook|capture_cudagraphs)"
    r"|(?:capture hook|capture_cudagraphs)[^.;\n]{0,80}own graphs"
)


def _initialize_kv_cache_logs(run_fresh, repo_root):
    code = _LOG_SNIPPET.format(tests=str(repo_root / "tests"))
    return run_fresh(code)["calls"]


def _names_modes(message: str, *modes: str) -> bool:
    return all(rc.whole(mode).search(message) for mode in modes)


# @spec REG-ENG-018
def test_first_initialize_kv_cache_in_the_process_logs_the_modes_once(
    run_fresh, repo_root
):
    first, again, other_state = _initialize_kv_cache_logs(run_fresh, repo_root)
    lines = [m for m in first if _names_modes(m, "FULL_AND_PIECEWISE", "VLLM_COMPILE")]
    assert len(lines) == 1, first
    (line,) = lines
    # The resolved mode, besides the two construction saw.
    assert _names_modes(line, "NONE"), line
    assert _NO_RUNNER_CAPTURE.search(line), line
    assert _OWN_GRAPHS_THROUGH_THE_HOOK.search(line), line
    # Once in the process: not again for the same model state, nor for
    # another one constructed under other modes. A repeat is any line with
    # either statement of the log; other lines naming a mode are not.
    for messages in (again, other_state):
        assert not [
            m
            for m in messages
            if _NO_RUNNER_CAPTURE.search(m) or _OWN_GRAPHS_THROUGH_THE_HOOK.search(m)
        ], messages


# ---------------------------------------------------------------------------
# REG-ENG-020: the commit event follows the step's last read and is waited on
#
# A real stream step on CPU through the harness runner. Every op that reads a
# request's staged inputs or state is logged by a dispatch mode watching
# their storages: the state pool, the model's host staging buffers (on CPU
# the staged device tensors are those buffers) and the forward's input IDs
# and embeddings. The commit sink's event (on CPU it has none) is a recording
# stand-in, and its host status buffer logs its reads.


class _RecordingEvent:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def record(self, stream=None) -> None:
        self.order.append("event.record")

    def synchronize(self) -> None:
        self.order.append("event.synchronize")

    def query(self) -> bool:
        return True

    def wait(self, stream=None) -> None:
        self.order.append("event.wait")


class _RecordingHost:
    """A host status buffer whose reads (``tolist``) are recorded."""

    def __init__(self, tensor: torch.Tensor, order: list[str]) -> None:
        self.tensor = tensor
        self.order = order

    def __getitem__(self, index):
        view = self.tensor[index]
        order = self.order

        class View:
            def tolist(self):
                order.append("status.read")
                return view.tolist()

            def __getattr__(self, name):
                return getattr(view, name)

        return View()


def _storage(tensor: torch.Tensor) -> int | None:
    try:
        return tensor.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return None


def _tensors(obj) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    values = vars(obj).values() if hasattr(obj, "__dict__") else ()
    return [v for v in values if isinstance(v, torch.Tensor)]


def _read_log(order: list[str], watched: dict[int, str]):
    """A dispatch mode that appends ``read:<label>`` to ``order`` for every
    op with an argument in a watched storage."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class ReadLog(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            for leaf in tree_leaves((args, kwargs)):
                if isinstance(leaf, torch.Tensor):
                    label = watched.get(_storage(leaf))
                    if label is not None:
                        order.append(f"read:{label}")
                        break
            return func(*args, **kwargs)

    return ReadLog()


# @spec REG-ENG-020
def test_the_commit_event_follows_the_steps_last_read_and_is_waited_on(stages):
    config = stages.config(eager=True)
    model = stages.model(config)
    state = stages.state(config, model)
    runner = h.Runner(config, model, state)
    runner.profile_run(core_embeds=False)
    bindings = runner.bind_pool(h.real_bindings(h.model_spec(model)))

    order: list[str] = []
    watched = {_storage(h.pool_tensor(bindings)): "state"}
    for tensor in _tensors(model._ensure_host_staging()):
        watched[_storage(tensor)] = "staged"
    sink = model._ensure_commit_sink(torch.device("cpu"))
    sink._event = _RecordingEvent(order)
    sink._status_host = _RecordingHost(sink._status_host, order)

    forward = model.forward

    def watching_forward(*args, **kwargs):
        for name in ("input_ids", "inputs_embeds"):
            if isinstance(kwargs.get(name), torch.Tensor):
                watched[_storage(kwargs[name])] = "staged"
        return forward(*args, **kwargs)

    model.forward = watching_forward
    postprocess = state.postprocess_state

    def recording_postprocess(*args, **kwargs):
        order.append("postprocess")
        postprocess(*args, **kwargs)
        order.append("postprocess.return")

    state.postprocess_state = recording_postprocess
    unit = h.new_request(
        "s1",
        int(model.config.audio_chunk_token_id),
        audio=h.stream_envelope(model.config, geometry=2),
        identifier="uuid-s1-0",
    )
    with _read_log(order, watched):
        runner.real_step(h.scheduler_output(new=(unit,), handles={"s1": 1}))

    assert runner.commit_log and not runner.commit_log[-1][1], runner.commit_log
    reads = [i for i, e in enumerate(order) if e.startswith("read:")]
    assert {order[i] for i in reads} >= {"read:state", "read:staged"}, order
    assert order.count("event.record") == 1, order
    record = order.index("event.record")
    assert reads[-1] < record, ("a read after the commit event", order[record:])
    wait = order.index("event.synchronize")
    status = order.index("status.read")
    assert (
        order.index("postprocess") < wait < status < order.index("postprocess.return")
    ), order
    assert record < wait, order


# ---------------------------------------------------------------------------
# TXG-CFG-016: transaction graphs are refused only on a core without the
# state-group layout accessor


def _state_layer_conditions(errors):
    """Every returned condition that refuses transaction graphs on account of
    the state layer or core's layout accessor (``support.txg_conditions``,
    the one matcher of TXG-CFG-016)."""
    from support import txg_conditions as c

    return [e for e in errors if c.state_layer_refusals(e)]


def _assert_the_state_layer_condition(errors, value) -> str:
    """Exactly one condition of TXG-CFG-016's kind, in full: the option, the
    rejected value (TXG-CFG-003), the missing state-group layout accessor, and
    removing the option."""
    from support import txg_conditions as c

    found = _state_layer_conditions(errors)
    assert len(found) == 1, errors
    (condition,) = found
    assert c.is_state_layer_condition(condition), condition
    assert rc.shows_value(condition, value), (value, condition)
    return condition


VALUES = [[4], [1, 31, 63], 63, None, {}]


@pytest.mark.parametrize("value", VALUES, ids=repr)
# @spec TXG-CFG-016, TXG-CFG-003
def test_a_core_without_the_accessor_returns_the_state_layer_condition(value):
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    errors = api.config_errors(
        engine_config(hf_config(value)), state_group_layouts=False
    )
    _assert_the_state_layer_condition(errors, value)


@pytest.mark.parametrize("value", VALUES, ids=repr)
# @spec TXG-CFG-016, TXG-CFG-003
def test_the_target_core_returns_no_state_layer_condition(value):
    """``ModelState.state_group_layouts`` is defined: no condition for that
    reason, whatever else the value violates."""
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    errors = api.config_errors(
        engine_config(hf_config(value)), state_group_layouts=True
    )
    assert _state_layer_conditions(errors) == [], errors


# @spec TXG-CFG-016, TXG-CFG-003
def test_the_target_core_accepts_a_valid_option_under_the_restricted_startup():
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    assert (
        api.config_errors(engine_config(hf_config([4])), state_group_layouts=True) == []
    )


# @spec TXG-CFG-016, TXG-CFG-003
def test_the_state_layer_condition_is_returned_with_the_other_violations():
    """Each other violated condition is its own entry, not folded into the
    state-layer one."""
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    errors = api.config_errors(
        engine_config(hf_config([4], lookahead=3), enforce_eager=True),
        state_group_layouts=False,
    )
    condition = _assert_the_state_layer_condition(errors, [4])
    others = [e for e in errors if e != condition]
    assert [e for e in others if "lookahead" in e.lower()], errors
    assert [e for e in others if "--enforce-eager" in e], errors


@pytest.mark.parametrize("accessor", [True, False], ids=["target-core", "no-accessor"])
# @spec TXG-CFG-016, TXG-CFG-003
def test_transaction_graphs_off_return_no_condition(accessor):
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    for hf in (hf_config(), hf_config([])):
        assert api.config_errors(engine_config(hf), state_group_layouts=accessor) == []
