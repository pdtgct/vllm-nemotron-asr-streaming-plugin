# SPDX-License-Identifier: Apache-2.0
"""The runner configurations at model and model-state construction, core's
mode resolution, and the model state's runner-graph guards (REG-ENG-001,
REG-ENG-006, REG-ENG-011, REG-ENG-012, REG-ENG-016 to REG-ENG-020,
REG-ENG-026, REG-ENG-027, REG-VAL-003, TXG-CFG-016).

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

#: The name the model state reports with its graph support (REG-ENG-016).
NO_RUNNER_GRAPH = "supports no runner graph"


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

#: case -> (settings, words the refusal must contain).
_REFUSED_AT_CONSTRUCTION = {
    "unset cudagraph_mode": ({"cudagraph": None}, ("cudagraph_mode",)),
    "unset compilation mode": ({"mode": None}, ()),
    "surviving STOCK_TORCH_COMPILE": ({"mode": "stock"}, ("stock", "-cc.mode none")),
    "FULL": ({"cudagraph": "full"}, ("FULL", "-cc.cudagraph_mode none")),
    "PIECEWISE under resolved breakable graphs": (
        {"cudagraph": "piecewise", "breakable": True},
        ("PIECEWISE", "VLLM_USE_BREAKABLE_CUDAGRAPH=0", "-cc.cudagraph_mode none"),
    ),
    "not CUDA": ({"cuda": False}, ("cuda", "--enforce-eager")),
    "prefix caching": ({"prefix_caching": True}, ("prefix",)),
}


@pytest.mark.parametrize("case", list(_REFUSED_AT_CONSTRUCTION))
@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-ENG-021, REG-ENG-022, REG-ENG-023, REG-ENG-024
def test_construction_refuses_what_the_shared_rule_refuses(stages, stage, case):
    settings, words = _REFUSED_AT_CONSTRUCTION[case]
    if stage == "model":
        config = stages.config(**settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config()
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
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
    _has(
        message,
        "-cc.cudagraph_mode none",
        "-cc.mode none",
        "cuda",
        "chunk_bucket_graph_populations",
        "--enforce-eager",
        "prefix",
    )


@pytest.mark.parametrize("stage", ["model", "model-state"])
# @spec REG-ENG-006, REG-VAL-008, TXG-CFG-016
def test_construction_refuses_transaction_graphs_on_the_state_layer(stages, stage):
    settings = {"hf": {"transaction_graph_populations": [2]}}
    if stage == "model":
        config = stages.config(right_context=0, **settings)
        message = _refusal(stages.model, config)
    else:
        config = stages.config(right_context=0)
        model = stages.model(config)
        stages.apply(config, **settings)
        message = _refusal(stages.state, config, model)
    _has(message, "transaction_graph_populations", "state layer")


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


# @spec REG-ENG-012
def test_core_resolves_every_accepted_built_configuration_to_none():
    from vllm.config import CompilationConfig, CUDAGraphMode
    from vllm.config.compilation import CompilationMode
    from vllm.v1.attention.backend import AttentionCGSupport

    configurations = _accepted_built_configurations()
    assert configurations, "no accepted configuration was enumerated"
    # Breakable graphs on and off, compiled and not, are all reached.
    assert {c[2] for c in configurations} == {False, True}
    assert any(c[1] == CompilationMode.VLLM_COMPILE for c in configurations)
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
                result = compilation.resolve_cudagraph_mode_and_sizes(
                    AttentionCGSupport.NEVER,
                    "nemotron_asr_streaming model state (captures its own graphs)",
                    1,
                    kv_cache_config=None,
                    max_num_reqs=64,
                    is_profiling=profiling,
                    piecewise_capture_available=breakable,
                )
                resolved[(graph.name, mode.name, breakable, ops is None, profiling)] = (
                    result
                )
    wrong = {k: v for k, v in resolved.items() if v != CUDAGraphMode.NONE}
    assert not wrong, wrong


# @spec REG-ENG-012
def test_the_model_has_no_compiled_submodule(stages):
    """Core's ``piecewise_capture_available`` is the resolved breakable
    setting alone for this model."""
    from vllm.v1.worker.gpu.cudagraph_utils import has_compiled_submodule

    config = stages.config(eager=True)
    assert not has_compiled_submodule(stages.model(config))


def _hook_inventory(tmp_path, monkeypatch, platform, *, cudagraph, mode, breakable):
    """The model's captured segment-graph entry keys after core's startup
    under the hook configuration, with construction seeing ``cudagraph`` /
    ``mode`` / ``breakable`` and core resolving the mode to NONE before the
    KV cache is initialized (REG-ENG-012). A key is the entry's runnable
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
    card = h.write_runnable_card(tmp_path / f"card-{cudagraph}-{mode}-{breakable}")
    config = h.build_vllm_config(
        card, right_context=3, max_num_seqs=2, overrides=DEFAULT_ARMS
    )
    config.model_config.enforce_eager = False
    compilation = config.compilation_config
    compilation.mode = _compilation_mode(mode)
    compilation.cudagraph_mode = _graph_mode(cudagraph)
    compilation.use_breakable_cudagraph = breakable
    platform(True)
    model = h.build_model(config)
    state = h.build_model_state(config, model)
    compilation.cudagraph_mode = CUDAGraphMode.NONE
    config.cache_config.kv_cache_memory_bytes = None
    config.cache_config.enable_extensible_kv_cache = False
    core = CoreOrder(h.Runner(config, model, state), capture_stream=FakeStream())
    core.startup()
    keys = {
        (getattr(w.runnable, "__qualname__", type(w.runnable).__qualname__), d)
        for w, d in graphs.captured()
    }
    core.shutdown()
    return keys


@pytest.mark.parametrize(
    "cudagraph, mode, breakable",
    [
        ("full_and_piecewise", "compile", False),
        ("full_and_piecewise", "none", True),
    ],
    ids=["serve-defaults", "breakable"],
)
# @spec REG-ENG-012
def test_the_segment_graph_inventory_does_not_depend_on_the_modes(
    tmp_path, monkeypatch, platform, cudagraph, mode, breakable
):
    explicit_none = _hook_inventory(
        tmp_path, monkeypatch, platform, cudagraph="none", mode="none", breakable=False
    )
    other = _hook_inventory(
        tmp_path,
        monkeypatch,
        platform,
        cudagraph=cudagraph,
        mode=mode,
        breakable=breakable,
    )
    # Encoder CHUNK entries and decoder tier entries, each at 1 and 2 rows.
    assert len(explicit_none) == 4, explicit_none
    assert other == explicit_none


# ---------------------------------------------------------------------------
# REG-ENG-017, REG-ENG-029: the model state refuses every runner-graph path


def _assert_no_runner_graph(message: str) -> None:
    _has(message, "own graphs", NO_RUNNER_GRAPH)


@pytest.mark.parametrize(
    "resolved", ["PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL"]
)
# @spec REG-ENG-017
def test_initialize_kv_cache_refuses_a_resolved_runner_graph_mode_first(
    eager_config, build_model_state, monkeypatch, resolved
):
    """Raises before anything else, including the base hook."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    state = build_model_state(eager_config())
    calls = []
    monkeypatch.setattr(
        ModelState,
        "initialize_kv_cache",
        lambda self, *args, **kwargs: calls.append("base"),
    )
    state.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode[resolved]
    with pytest.raises(Exception) as excinfo:
        state.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[]), None)
    message = str(excinfo.value)
    assert resolved in message, message
    _assert_no_runner_graph(message)
    assert calls == []


# @spec REG-ENG-029
def test_prepare_dummy_inputs_refuses_a_runner_capture(eager_config, build_model_state):
    state = build_model_state(eager_config())
    with pytest.raises(Exception) as excinfo:
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
    with pytest.raises(Exception) as excinfo:
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
# Each case runs in a fresh interpreter: the line is logged once per process,
# so a test in this process could not observe a first call.

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
card = h.write_runnable_card(Path(tempfile.mkdtemp()) / "card")
config = h.build_vllm_config(
    card,
    right_context=3,
    max_num_seqs=2,
    overrides={{"encoder_execution_arm": "eager-graphed",
                "decode_dispatch_arm": "dense-graphed"}},
)
if {hook!r}:
    set_cuda_platform(mp, True)
    config.model_config.enforce_eager = False
    config.compilation_config.mode = CompilationMode.VLLM_COMPILE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    config.compilation_config.use_breakable_cudagraph = False
model = h.build_model(config)
state = h.build_model_state(config, model)
# Core's resolution (REG-ENG-012) before the KV cache is initialized.
config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
runner = h.Runner(config, model, state)
calls = []
for _ in range(2):
    start = len(records)
    runner.bind_pool(h.real_bindings(h.model_spec(model)))
    calls.append([r.getMessage() for r in records[start:]])
emit(calls=calls)
"""


def _initialize_kv_cache_logs(run_fresh, repo_root, *, hook: bool):
    code = _LOG_SNIPPET.format(tests=str(repo_root / "tests"), hook=hook)
    return run_fresh(code)["calls"]


def _mode_line(message: str) -> bool:
    return "FULL_AND_PIECEWISE" in message and "VLLM_COMPILE" in message


# @spec REG-ENG-018
def test_first_initialize_kv_cache_logs_the_modes_once(run_fresh, repo_root):
    first, second = _initialize_kv_cache_logs(run_fresh, repo_root, hook=True)
    lines = [m for m in first if _mode_line(m)]
    assert len(lines) == 1, first
    (line,) = lines
    # The resolved mode, besides the two construction saw.
    assert "NONE" in line, line
    assert re.search(r"(?i)runner", line) and re.search(r"(?i)own graphs", line), line
    assert not [m for m in second if _mode_line(m) or "own graphs" in m.lower()], second


# @spec REG-ENG-018
def test_initialize_kv_cache_under_enforce_eager_logs_no_runner_mode_line(
    run_fresh, repo_root
):
    calls = _initialize_kv_cache_logs(run_fresh, repo_root, hook=False)
    for messages in calls:
        assert not [m for m in messages if "own graphs" in m.lower()], messages
        assert not [m for m in messages if re.search(r"(?i)runner.*graph", m)], messages


# ---------------------------------------------------------------------------
# REG-ENG-020: postprocess_state waits for the commit event


class _RecordingEvent:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def record(self, stream=None) -> None:
        self.order.append("event.record")

    def synchronize(self) -> None:
        self.order.append("event.synchronize")


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


# @spec REG-ENG-020
def test_postprocess_state_waits_for_the_commit_event_before_reading_status(
    eager_config, build_model_state
):
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        BoundedCommitSink,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import SessionRegistry

    order: list[str] = []
    state = build_model_state(eager_config())
    model = state.model
    registry = SessionRegistry()
    sink = BoundedCommitSink(registry, max_rows=4)
    sink._event = _RecordingEvent(order)
    sink._status_host = _RecordingHost(sink._status_host, order)
    model._registry = registry
    model._commit_sink = sink
    model.consume_batch_stats = lambda: order.append("batch_stats")
    # The step's staged commit: the event is recorded after the step's device
    # work (here, none).
    sink._stage(
        SimpleNamespace(bindings=(), progress=None, queue_counts=None, records=()),
        torch.zeros(0, dtype=torch.int32),
    )
    state._snapshot = SimpleNamespace(
        dummy_run=False,
        planning_failed=set(),
        planned_files=(),
        admission_generations={},
    )
    state.postprocess_state(torch.zeros(0, dtype=torch.int32), torch.zeros(0))
    order.append("returned")
    assert "event.synchronize" in order and "status.read" in order, order
    assert (
        order.index("event.record")
        < order.index("event.synchronize")
        < order.index("status.read")
        < order.index("returned")
    ), order


# ---------------------------------------------------------------------------
# TXG-CFG-016: transaction graphs are refused on the state layer


def _state_layer_conditions(errors):
    from support import txg_api as api

    return [
        e
        for e in errors
        if api.KEY in e and "state layer" in e.lower() and "remov" in e.lower()
    ]


@pytest.mark.parametrize("value", [[4], [1, 31, 63], 63, None, {}], ids=repr)
# @spec TXG-CFG-016, TXG-CFG-003
def test_transaction_graphs_on_always_return_the_state_layer_condition(value):
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    errors = api.config_errors(engine_config(hf_config(value)))
    assert len(_state_layer_conditions(errors)) == 1, errors


# @spec TXG-CFG-016, TXG-CFG-003
def test_the_state_layer_condition_is_returned_with_the_other_violations():
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    errors = api.config_errors(
        engine_config(hf_config([4], lookahead=3), enforce_eager=True)
    )
    assert _state_layer_conditions(errors), errors
    assert [e for e in errors if "lookahead" in e.lower()], errors
    assert [e for e in errors if "--enforce-eager" in e], errors


# @spec TXG-CFG-016, TXG-CFG-003
def test_transaction_graphs_off_return_no_state_layer_condition():
    from support import txg_api as api
    from support.txg_fakes import engine_config, hf_config

    assert api.config_errors(engine_config(hf_config())) == []
    assert api.config_errors(engine_config(hf_config([]))) == []
