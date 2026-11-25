# SPDX-License-Identifier: Apache-2.0
"""Engine settings a persistent-state serve refuses, the runner
configurations, and the model adapter's dependency checks.

``persistent_state_config_errors`` reports the refused settings of a
``VllmConfig``: the engine settings, the hook configuration's conditions
(``runner_mode_refusals``), the missing model-state hooks, native burst on
the state layer and the transaction-graph conditions
(``transaction_graph_config.transaction_graph_config_errors``); it does not
turn anything off. The model-configuration hook (``streaming_config``), model
construction and the model state's construction all raise from that report,
so a setting that core resolves only after the hook ran is still caught.

Serve with the settings explicit (see the README), for example::

    vllm serve <checkpoint> \\
        --dtype float32 \\
        --enforce-eager

Without ``--enforce-eager`` the hook configuration applies: core captures
the model's graphs through its model-state hooks and no runner graph over the
model's forward (``runner_mode_refusals``).

Importing this module imports no vLLM.
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any

# The transaction-graph checks and predicate are looked up on their module at
# call time, so they stay the one implementation (TXG-CFG-011, TXG-CFG-014).
from . import transaction_graph_config as _txg
from .transaction_graph_config import KEY as TRANSACTION_GRAPH_KEY

# Parallel sizes that have to be 1. A size the running vLLM build does not
# define counts as 1.
PARALLEL_SIZES_MUST_BE_ONE: tuple[str, ...] = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
)

#: The model adapter's unconditional internal vLLM dependencies, beyond the
#: public client surface: module name -> names the adapter uses from it. A
#: dotted name is a member of the name before the dot: a ``ModelState``
#: hook or method, an enum member
#: (``MambaAttentionBackendEnum.PERSISTENT_STATE``), or a call argument
#: (``MambaManager.layer_config``'s ``recomputable``).
ADAPTER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.config.compilation": ("CUDAGraphMode", "CompilationMode"),
    "vllm.compilation.breakable_cudagraph": ("is_breakable_cudagraph_enabled",),
    "vllm.v1.attention.backend": ("AttentionCGSupport",),
    "vllm.v1.core.sched.output": ("NewRequestData", "SchedulerOutput"),
    "vllm.v1.kv_cache_interface": (
        "KVCacheConfig",
        "KVCacheLayerConfig",
        "MambaSpec",
    ),
    "vllm.model_executor.layers.mamba.abstract": ("MambaBase",),
    "vllm.v1.core.single_type_kv_cache_manager": (
        "MambaManager",
        "MambaManager.layer_config.recomputable",
    ),
    "vllm.v1.attention.backends.registry": (
        "MambaAttentionBackendEnum",
        "MambaAttentionBackendEnum.PERSISTENT_STATE",
    ),
    "vllm.model_executor.models.interfaces": ("SupportsPersistentState",),
    "vllm.v1.worker.gpu.input_batch": ("InputBatch",),
    "vllm.v1.worker.gpu.model_states.interface": (
        "ModelState",
        "ModelState.prepare_execution_memory",
        "ModelState.needs_cudagraph_capture",
        "ModelState.capture_cudagraphs",
        "ModelState.clear_cudagraphs",
        "ModelState.post_warmup",
        "ModelState.shutdown",
        "ModelState.get_additional_cg_support",
        "ModelState.take_failed_requests",
    ),
    "vllm.v1.worker.gpu.states": ("RequestState",),
    "vllm.v1.worker.utils": ("AttentionGroup",),
    "vllm.multimodal": ("MULTIMODAL_REGISTRY",),
    "vllm.v1.attention.backends.utils": ("NULL_BLOCK_ID",),
    "vllm.v1.worker.gpu.sample.output": ("SamplerOutput",),
    "vllm.utils.torch_utils": ("direct_register_custom_op",),
}

#: The graph-runtime dependencies, needed only when graphed execution is
#: configured (``graphed_execution_configured``).
GRAPH_RUNTIME_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.distributed.parallel_state": ("graph_capture",),
    "vllm.forward_context": ("BatchDescriptor", "set_forward_context"),
    "vllm.platforms": ("current_platform",),
    "vllm.utils.import_utils": ("resolve_obj_by_qualname",),
    "vllm.compilation.cudagraph_pool": ("capture_pool",),
    "vllm.envs": ("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",),
}

#: The transaction-graph dependency, needed only while transaction graphs
#: are on (``transaction_graph_config.transaction_graphs_on``): core's
#: state-group layout accessor, a property. The attributes of the layouts it
#: returns are read without importing their classes, at capture.
TRANSACTION_GRAPH_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.v1.worker.gpu.model_states.interface": ("ModelState.state_group_layouts",),
}

#: The model-state hooks vllm#67583 adds; the plugin needs every one in every
#: runner configuration.
MODEL_STATE_GRAPH_HOOKS: tuple[str, ...] = (
    "prepare_execution_memory",
    "needs_cudagraph_capture",
    "capture_cudagraphs",
    "clear_cudagraphs",
    "post_warmup",
    "shutdown",
)

#: The state-layer surface ``register()`` checks before it changes any
#: registry (REG-002): module name -> dotted names, written as in
#: ``ADAPTER_DEPENDENCIES``.
CORE_CHECK_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.v1.worker.gpu.model_states.interface": (
        "ModelState.prepare_execution_memory",
        "ModelState.needs_cudagraph_capture",
        "ModelState.capture_cudagraphs",
        "ModelState.clear_cudagraphs",
        "ModelState.post_warmup",
        "ModelState.shutdown",
        "ModelState.take_failed_requests",
    ),
    "vllm.v1.core.single_type_kv_cache_manager": (
        "MambaManager.layer_config.recomputable",
    ),
    "vllm.v1.attention.backends.registry": (
        "MambaAttentionBackendEnum.PERSISTENT_STATE",
    ),
    "vllm.model_executor.models.interfaces": ("SupportsPersistentState",),
}

#: The environment variables core's configuration build reads for breakable
#: graphs and for turning compilation off (``VllmConfig.__post_init__``).
BREAKABLE_CUDAGRAPH_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"
TORCH_COMPILE_DISABLE_ENV = "TORCH_COMPILE_DISABLE"

#: The runner stages ``runner_mode_refusals`` judges (REG-ENG-027).
RUNNER_STAGES: tuple[str, ...] = ("hook", "constructor")

#: Mode names by value, for a configuration that carries a plain integer.
_GRAPH_MODE_NAMES = {0: "NONE", 1: "PIECEWISE", 2: "FULL"}
_COMPILATION_MODE_NAMES = {
    0: "NONE",
    1: "STOCK_TORCH_COMPILE",
    2: "DYNAMO_TRACE_ONCE",
    3: "VLLM_COMPILE",
}


def graphed_execution_configured(hf_config: Any) -> bool:
    """Whether the configuration selects graphed execution: the
    ``dense-graphed`` decoder, a graphed encoder arm (``dense-graphed`` or
    ``eager-graphed``) or CHUNK-bucket graphs."""
    return bool(
        getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed"
        or getattr(hf_config, "encoder_execution_arm", None)
        in ("dense-graphed", "eager-graphed")
        or getattr(hf_config, "chunk_bucket_graph_populations", ())
    )


# @spec REG-ENG-025, REG-MEM-019
def selects_hook_captured_graph(hf_config: Any) -> bool:
    """Whether the configuration selects a graph core's capture hook
    captures: the ``eager-graphed`` encoder, the ``dense-graphed`` decoder,
    or transaction graphs on. The one predicate of hook-captured graphs, for
    the runner check and the model state's ``needs_cudagraph_capture``."""
    return (
        getattr(hf_config, "encoder_execution_arm", None) == "eager-graphed"
        or getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed"
        or _txg.transaction_graphs_on(hf_config)
    )


def missing_graph_hooks() -> list[str]:
    """The model-state hooks (``MODEL_STATE_GRAPH_HOOKS``) the installed
    core's ``ModelState`` does not define (all of them without the class)."""
    try:
        from vllm.v1.worker.gpu.model_states.interface import ModelState
    except ImportError:
        return list(MODEL_STATE_GRAPH_HOOKS)
    return [
        hook
        for hook in MODEL_STATE_GRAPH_HOOKS
        if not callable(getattr(ModelState, hook, None))
    ]


# @spec TXG-CFG-016, REG-018
def model_state_has_state_group_layouts() -> bool:
    """Whether the running core's ``ModelState`` defines
    ``state_group_layouts`` (a property counts). Both callers of
    ``transaction_graph_config_errors`` pass this as
    ``has_state_group_layouts``."""
    try:
        from vllm.v1.worker.gpu.model_states.interface import ModelState
    except ImportError:
        return False
    return hasattr(ModelState, "state_group_layouts")


def _cuda_platform() -> bool:
    """Whether vLLM's current platform is CUDA. vLLM derives the device
    configuration's type from that platform, so the platform alone decides
    (REG-ENG-024, REG-MEM-019)."""
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return False
    return bool(current_platform.is_cuda())


def _mode_name(mode: Any, names: dict[int, str]) -> str | None:
    """A ``cudagraph_mode`` or compilation mode by its name (``None`` while
    unset): an enum member's name, or the name of a plain integer value."""
    if mode is None:
        return None
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(mode, int) and not isinstance(mode, bool):
        return names.get(mode, str(mode))
    return str(mode).upper()


def _enforce_eager_remedy(hf_config: Any) -> str:
    """The ``--enforce-eager`` remedy; while transaction graphs are on it
    also names removing their option, which is refused under
    ``--enforce-eager`` (REG-ENG-005)."""
    if _txg.transaction_graphs_on(hf_config):
        return (
            f"--enforce-eager with {TRANSACTION_GRAPH_KEY} removed, since "
            "transaction graphs are refused under --enforce-eager"
        )
    return "--enforce-eager"


# @spec REG-ENG-005, REG-ENG-006, REG-ENG-011, REG-ENG-021, REG-ENG-022
# @spec REG-ENG-023, REG-ENG-024, REG-ENG-025, REG-ENG-026, REG-ENG-027
def runner_mode_refusals(vllm_config: Any, *, stage: str) -> list[str]:
    """The conditions of the hook configuration that ``vllm_config`` misses,
    each with its remedy (empty under ``--enforce-eager`` or when every one
    holds). The one function of the runner configurations: the
    configuration hook calls it with ``stage="hook"``, model and model-state
    construction with ``stage="constructor"``.

    At ``"hook"`` the modes are the requested ones and an unset mode passes;
    breakable graphs are predicted from ``VLLM_USE_BREAKABLE_CUDAGRAPH``, and
    a requested ``STOCK_TORCH_COMPILE`` passes when core will set the mode to
    NONE (breakable graphs on, or ``TORCH_COMPILE_DISABLE=1``). At
    ``"constructor"`` both modes are as core's configuration build left
    them: an unset mode is refused, breakable graphs are the resolved
    ``compilation_config.use_breakable_cudagraph``, and a surviving
    ``STOCK_TORCH_COMPILE`` is refused.

    While transaction graphs are on, the remedies of the mode conditions are
    the explicit NONE alone, the only value their own runner-mode condition
    accepts (TXG-CFG-018), and every ``--enforce-eager`` remedy names
    removing their option.
    """
    if stage not in RUNNER_STAGES:
        raise ValueError(f"stage must be one of {RUNNER_STAGES}, not {stage!r}")
    model = getattr(vllm_config, "model_config", None)
    if getattr(model, "enforce_eager", False):
        return []
    hf_config = getattr(model, "hf_config", None)
    transaction_graphs = _txg.transaction_graphs_on(hf_config)
    eager = _enforce_eager_remedy(hf_config)
    compilation = getattr(vllm_config, "compilation_config", None)
    graph = _mode_name(getattr(compilation, "cudagraph_mode", None), _GRAPH_MODE_NAMES)
    mode = _mode_name(getattr(compilation, "mode", None), _COMPILATION_MODE_NAMES)
    constructor = stage == "constructor"
    if constructor:
        breakable = bool(getattr(compilation, "use_breakable_cudagraph", False))
        compile_disabled = False
    else:
        breakable = os.environ.get(BREAKABLE_CUDAGRAPH_ENV) == "1"
        compile_disabled = os.environ.get(TORCH_COMPILE_DISABLE_ENV) == "1"

    refusals: list[str] = []
    if constructor and graph is None:
        refusals.append(
            "cudagraph_mode is unset after core's configuration build "
            "(pass -cc.cudagraph_mode none)"
        )
    if constructor and mode is None:
        refusals.append(
            "the compilation mode is unset after core's configuration build "
            "(pass -cc.mode none)"
        )
    if graph == "FULL":
        remedy = "-cc.cudagraph_mode none"
        if not transaction_graphs:
            remedy += ", or leave cudagraph_mode unset"
        refusals.append(
            "cudagraph_mode FULL is not supported: core cannot reduce its "
            f"full graphs for a model that captures its own (pass {remedy})"
        )
    if graph == "PIECEWISE" and breakable:
        remedy = "-cc.cudagraph_mode none"
        if not transaction_graphs:
            remedy = f"{BREAKABLE_CUDAGRAPH_ENV}=0, or {remedy}"
        refusals.append(
            "cudagraph_mode PIECEWISE is not supported while breakable CUDA "
            "graphs are on: core would capture the forward as one breakable "
            f"graph (pass {remedy})"
        )
    if mode == "STOCK_TORCH_COMPILE" and (
        constructor or not (breakable or compile_disabled)
    ):
        remedy = "-cc.mode none"
        if not transaction_graphs:
            remedy += ", or leave the compilation mode unset"
        refusals.append(
            "compilation mode STOCK_TORCH_COMPILE is not supported: it would "
            "compile the whole model, whose forward plans each step on the "
            f"host (pass {remedy})"
        )
    if not _cuda_platform():
        refusals.append(f"the hook configuration needs a CUDA platform (pass {eager})")
    if not selects_hook_captured_graph(hf_config):
        refusals.append(
            "no graph that core's capture hook captures is selected (select "
            "encoder_execution_arm eager-graphed or decode_dispatch_arm "
            f"dense-graphed in --hf-overrides, or pass {eager})"
        )
    if getattr(hf_config, "encoder_execution_arm", None) == "dense-graphed":
        refusals.append(
            "encoder_execution_arm dense-graphed (the compiled encoder) is not "
            f"captured through core's capture hook (pass {eager})"
        )
    chunk_populations = getattr(hf_config, "chunk_bucket_graph_populations", ())
    if chunk_populations:
        refusals.append(
            f"chunk_bucket_graph_populations {chunk_populations!r} are not "
            f"captured through core's capture hook (pass {eager})"
        )
    return refusals


# @spec REG-ENG-019
def native_burst_refusal(hf_config: Any) -> str | None:
    """The state layer's refusal of native burst, or ``None``."""
    if not getattr(hf_config, "experimental_native_burst", False):
        return None
    return (
        "experimental_native_burst (native burst) is not supported on the "
        "state layer: remove experimental_native_burst"
    )


# @spec REG-ENG-001, REG-ENG-002, REG-ENG-003, REG-ENG-004, REG-ENG-005
# @spec REG-ENG-006, REG-ENG-009, REG-ENG-019, REG-VAL-008
def persistent_state_config_errors(
    vllm_config: Any, *, resolved: bool = False
) -> list[str]:
    """Return human-readable refusals for a persistent-state engine.

    ``resolved=False`` is the configuration hook (``runner_mode_refusals``
    stage ``"hook"``); ``resolved=True`` is model and model-state
    construction against the final engine configuration (stage
    ``"constructor"``). ``async_scheduling`` is refused only while
    transaction graphs are on (their scheduling condition). The result also
    lists every model-state hook the core lacks, native burst on the state
    layer and every condition ``transaction_graph_config_errors`` returns.
    """
    errors: list[str] = []
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    attention = getattr(vllm_config, "attention_config", None)
    if getattr(cache, "enable_prefix_caching", None):
        errors.append(
            "prefix caching must be False "
            "(pass enable_prefix_caching=False or --no-enable-prefix-caching)"
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        errors.append("speculative decoding must be off")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        errors.append("KV transfer must be off")
    if getattr(parallel, "enable_dbo", False):
        errors.append("dual batch overlap must be off")
    if getattr(model, "enable_sleep_mode", False):
        errors.append("sleep mode must be off")
    if getattr(model, "enable_cumem_allocator", False):
        errors.append("CuMem must be off")
    if getattr(attention, "hisparse_config", None) is not None:
        errors.append("HiSparse (hierarchical sparse attention) must be off")
    if getattr(parallel, "enable_fault_tolerance", False):
        # Core's kernel warm-up then runs synthetic real steps through the
        # model state, which the model declines.
        errors.append(
            "fault tolerance must be off "
            "(remove --enable-fault-tolerance and --fault-tolerance-config)"
        )
    if getattr(parallel, "enable_elastic_ep", False):
        # Its reconfiguration reruns the worker's warmup, which would call the
        # serving capture again over live state pages.
        errors.append(
            "elastic expert parallelism must be off (remove --enable-elastic-ep)"
        )
    for name in PARALLEL_SIZES_MUST_BE_ONE:
        if getattr(parallel, name, 1) != 1:
            errors.append(f"{name} must be 1")
    errors.extend(
        runner_mode_refusals(vllm_config, stage="constructor" if resolved else "hook")
    )
    hooks = missing_graph_hooks()
    if hooks:
        errors.append(
            "the installed vLLM's ModelState lacks the model-state hooks "
            + ", ".join(hooks)
            + " (the plugin requires vllm#67583's model-state hooks)"
        )
    hf_config = getattr(model, "hf_config", None)
    native_burst = native_burst_refusal(hf_config)
    if native_burst is not None:
        errors.append(native_burst)
    errors.extend(
        _txg.transaction_graph_config_errors(
            vllm_config,
            has_state_group_layouts=model_state_has_state_group_layouts(),
        )
    )
    return errors


def engine_config_refusal(
    vllm_config: Any, subject: str, *, resolved: bool = False
) -> str | None:
    """Return one message listing every refused engine setting, or ``None``."""
    errors = persistent_state_config_errors(vllm_config, resolved=resolved)
    if not errors:
        return None
    return f"{subject} refuses this engine configuration: " + "; ".join(errors)


# @spec REG-ENG-006, REG-VAL-003
def check_engine_config(
    vllm_config: Any, subject: str, *, resolved: bool = False
) -> None:
    """Raise one ``ValueError`` listing every refused engine setting.

    The configuration hook calls this with ``resolved=False``; model and
    model-state construction with ``resolved=True``, so each applies
    ``runner_mode_refusals`` once, at its stage."""
    message = engine_config_refusal(vllm_config, subject, resolved=resolved)
    if message is not None:
        raise ValueError(message)


def _absent(module: Any, name: str) -> bool:
    """Whether ``module`` lacks ``name``. A dotted name is a member of the
    name before the last dot: any attribute (a hook, a property, an enum
    member) or a parameter of a callable owner."""
    *owners, member = name.split(".")
    if not owners:
        return not hasattr(module, member)
    owner = module
    for part in owners:
        try:
            owner = getattr(owner, part)
        except AttributeError:
            return True
    try:
        if member in inspect.signature(owner).parameters:
            return False
    except (TypeError, ValueError):
        pass
    return not hasattr(owner, member)


def _missing_in(
    module_name: str, names: tuple[str, ...], note: str = "not defined"
) -> list[str]:
    """The ``names`` of the dependency table row ``module_name`` that cannot
    be imported or are not defined, each named with its module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return [f"{module_name}.{name} ({exc})" for name in names]
    return [f"{module_name}.{name} ({note})" for name in names if _absent(module, name)]


# @spec REG-002
def missing_core_items() -> list[str]:
    """The items of ``register()``'s core check (``CORE_CHECK_DEPENDENCIES``)
    that the installed vLLM lacks, each named with its module."""
    missing: list[str] = []
    for module_name, names in CORE_CHECK_DEPENDENCIES.items():
        missing += _missing_in(module_name, names)
    return missing


def _dependency_error(missing: list[str], purpose: str) -> ImportError:
    return ImportError(
        f"the installed vLLM lacks {purpose} of vllm-nemotron-asr-streaming: "
        + "; ".join(missing)
    )


# @spec REG-010
def check_adapter_dependencies() -> None:
    """Import every unconditional adapter dependency; raise naming every
    missing one.

    The model class module calls this before it imports the adapter's vLLM
    symbols, so a vLLM build without them fails with one clear error before
    any weights load.
    """
    missing: list[str] = []
    for module_name, names in ADAPTER_DEPENDENCIES.items():
        missing += _missing_in(module_name, names)
    if missing:
        raise _dependency_error(
            missing,
            "adapter dependencies (a vLLM build with RFC 1's state layer and "
            "vllm#67583's model-state hooks is required)",
        )


# @spec REG-014, REG-018
def check_conditional_dependencies(hf_config: Any) -> None:
    """Raise naming every missing conditional dependency the configuration
    needs: the graph-runtime rows when graphed execution is configured, the
    state-group layout accessor while transaction graphs are on. Model
    construction calls this before any weights load."""
    missing: list[str] = []
    if graphed_execution_configured(hf_config):
        for module_name, names in GRAPH_RUNTIME_DEPENDENCIES.items():
            missing += _missing_in(
                module_name, names, "not defined; graphed execution is configured"
            )
    if _txg.transaction_graphs_on(hf_config):
        for module_name, names in TRANSACTION_GRAPH_DEPENDENCIES.items():
            missing += _missing_in(
                module_name, names, f"not defined; {TRANSACTION_GRAPH_KEY} is set"
            )
    if missing:
        raise _dependency_error(missing, "dependencies the configuration needs")
