# SPDX-License-Identifier: Apache-2.0
"""Engine settings a persistent-state serve refuses, and the model adapter's
dependency checks.

``persistent_state_config_errors`` reports the refused settings of a
``VllmConfig``, the transaction-graph conditions
(``transaction_graph_config.transaction_graph_config_errors``) included; it
does not turn anything off. The model-configuration hook
(``streaming_config``) and the model state's construction both raise from that
report, so a setting that core resolves only after the hook ran is still
caught.

Serve with the settings explicit (see the README), for example::

    vllm serve <checkpoint> \\
        --dtype float32 \\
        --enforce-eager

``enforce_eager`` off is accepted only in the hook configuration, where core
captures the model's graphs through its model-state hooks
(``hook_capture_refusals``).

Importing this module imports no vLLM.
"""

from __future__ import annotations

import importlib
import inspect
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
#: dotted name is a member of the name before the dot: a ``ModelState`` hook,
#: or a constructor argument (``MambaManager.layer_config``'s
#: ``recomputable``).
ADAPTER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.config.compilation": ("CUDAGraphMode",),
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
    "vllm.v1.attention.backends.registry": ("MambaAttentionBackendEnum",),
    "vllm.v1.worker.gpu.input_batch": ("InputBatch",),
    "vllm.v1.worker.gpu.model_states.interface": (
        "ModelState",
        "ModelState.prepare_execution_memory",
        "ModelState.needs_cudagraph_capture",
        "ModelState.capture_cudagraphs",
        "ModelState.clear_cudagraphs",
        "ModelState.post_warmup",
        "ModelState.shutdown",
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

#: The transaction-graph dependencies, needed only while transaction graphs
#: are on (``transaction_graph_config.transaction_graphs_on``). The PR 6
#: proof of concept refuses transaction graphs (``pr6_poc_refusals``); no
#: build it runs on has these members.
TRANSACTION_GRAPH_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "vllm.v1.worker.gpu.state_slots": (
        "StateSlotBindings.startup_pool_access",
        "StateSlotBindings.pool_layout",
    ),
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


# @spec REG-ENG-005, REG-MEM-019
def selects_hook_captured_graph(hf_config: Any) -> bool:
    """Whether the configuration selects a graph core's capture hook
    captures: the ``eager-graphed`` encoder, the ``dense-graphed`` decoder,
    or transaction graphs on. The one predicate of hook-captured graphs, for
    the engine check and the model state's ``needs_cudagraph_capture``."""
    return (
        getattr(hf_config, "encoder_execution_arm", None) == "eager-graphed"
        or getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed"
        or _txg.transaction_graphs_on(hf_config)
    )


#: Earlier name of :func:`selects_hook_captured_graph`, kept for the model
#: state's import.
hook_captured_arm = selects_hook_captured_graph


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


def _cuda_platform(vllm_config: Any) -> bool:
    """Whether the engine runs on a CUDA platform: vLLM's current platform is
    CUDA. vLLM derives the device configuration's type from that platform, so
    the platform alone decides (REG-ENG-005, REG-MEM-019)."""
    del vllm_config
    try:
        from vllm.platforms import current_platform
    except ImportError:
        return False
    return bool(current_platform.is_cuda())


# @spec REG-ENG-001
def pr6_poc_refusals(hf_config: Any) -> list[str]:
    """The features the PR 6 proof of concept does not support: transaction
    graphs (they need a state pool with fixed addresses and its layout) and
    native burst (its history check refuses every turn after a session's
    first, since PR 6 sends each turn with its cumulative prompt)."""
    errors: list[str] = []
    if _txg.transaction_graphs_on(hf_config):
        errors.append(
            f"{TRANSACTION_GRAPH_KEY} (transaction graphs) is not supported on "
            "the PR 6 state layer in this proof of concept; remove it"
        )
    if getattr(hf_config, "experimental_native_burst", False):
        errors.append(
            "experimental_native_burst is not supported on the PR 6 state "
            "layer in this proof of concept; remove it"
        )
    return errors


def _mode_value(mode: Any) -> Any:
    return getattr(mode, "value", mode)


def _enforce_eager_remedy(hf_config: Any) -> str:
    if _txg.transaction_graphs_on(hf_config):
        return (
            f"--enforce-eager, with {TRANSACTION_GRAPH_KEY} removed (transaction "
            "graphs are refused under --enforce-eager)"
        )
    return "--enforce-eager"


# @spec REG-ENG-005
def hook_capture_refusals(vllm_config: Any, *, resolved: bool = False) -> list[str]:
    """The conditions of the hook configuration that ``vllm_config`` misses,
    each with its remedy (empty when every one holds).

    ``enforce_eager`` off lets core capture the model's graphs through the
    model-state hooks. It requires core's own CUDA graphs off
    (``cudagraph_mode`` NONE; unset is accepted before core resolves it.
    Stale on PR 6: its core no longer sets NONE for persistent state, so an
    unset mode is refused only once resolved; pass ``-cc.cudagraph_mode
    none``); torch.compile off
    (compilation mode NONE: ``-cc.mode none``, or ``-O0`` before core
    resolves the mode); a CUDA platform; at least one graph the hook
    captures; and no graph it does not (the compiled ``dense-graphed``
    encoder, CHUNK-bucket graphs).
    """
    missing: list[str] = []
    model = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model, "hf_config", None)
    eager_remedy = _enforce_eager_remedy(hf_config)
    compilation = getattr(vllm_config, "compilation_config", None)
    cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
    graph_mode = _mode_value(cudagraph_mode)
    if compilation is None or not (
        graph_mode == 0 or (graph_mode is None and not resolved)
    ):
        missing.append("cudagraph_mode must be NONE (pass -cc.cudagraph_mode none)")
    mode = _mode_value(getattr(compilation, "mode", None))
    if mode is None and not resolved:
        level = _mode_value(getattr(vllm_config, "optimization_level", None))
        mode = 0 if level == 0 else None
    if compilation is None or mode != 0:
        missing.append("torch.compile must be off (pass -cc.mode none)")
    if not _cuda_platform(vllm_config):
        missing.append(
            f"the hook configuration needs a CUDA platform (pass {eager_remedy})"
        )
    if not selects_hook_captured_graph(hf_config):
        missing.append(
            "no graph core's capture hook captures is selected: select "
            "encoder_execution_arm eager-graphed, decode_dispatch_arm "
            f"dense-graphed or {TRANSACTION_GRAPH_KEY} in --hf-overrides, or "
            "pass --enforce-eager"
        )
    if getattr(hf_config, "encoder_execution_arm", None) == "dense-graphed":
        missing.append(
            "encoder_execution_arm dense-graphed (compiled) is not captured "
            f"through core's capture hook (pass {eager_remedy})"
        )
    chunk_populations = getattr(hf_config, "chunk_bucket_graph_populations", ())
    if chunk_populations:
        missing.append(
            f"chunk_bucket_graph_populations {chunk_populations!r} are not "
            f"captured through core's capture hook (pass {eager_remedy})"
        )
    return missing


# @spec REG-ENG-001, REG-ENG-002, REG-ENG-003, REG-ENG-004, REG-ENG-005
# @spec REG-ENG-009, REG-VAL-008
def persistent_state_config_errors(
    vllm_config: Any, *, resolved: bool = False
) -> list[str]:
    """Return human-readable refusals for a persistent-state engine.

    ``async_scheduling`` of ``True`` is refused. An unset value (``None``) is
    accepted before core resolves the option (``resolved=False``: it resolves
    to ``False`` for a model that declares persistent state; stale on PR 6,
    whose core resolves an unset value to ``True``, so the refusal then
    fires only at model-state construction: pass ``--no-async-scheduling``);
    once the engine configuration is final (``resolved=True``) only
    ``False`` is accepted. The result also lists every model-state hook the
    core lacks, the features the PR 6 proof of concept refuses
    (``pr6_poc_refusals``) and every condition
    ``transaction_graph_config_errors`` returns.
    """
    errors: list[str] = []
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    attention = getattr(vllm_config, "attention_config", None)
    async_scheduling = getattr(scheduler, "async_scheduling", None)
    if async_scheduling is True or (resolved and async_scheduling is not False):
        errors.append(
            "async scheduling must be False "
            "(pass async_scheduling=False or --no-async-scheduling)"
        )
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
        # Core's kernel warm-up then runs synthetic real steps that carry no
        # state-slot handles (pass neither --enable-fault-tolerance nor
        # --fault-tolerance-config).
        errors.append(
            "fault tolerance must be off "
            "(remove --enable-fault-tolerance and --fault-tolerance-config)"
        )
    if getattr(parallel, "enable_elastic_ep", False):
        # Its reconfiguration reruns the worker's warmup, which would call the
        # serving capture again over live state pools.
        errors.append(
            "elastic expert parallelism must be off (remove --enable-elastic-ep)"
        )
    for name in PARALLEL_SIZES_MUST_BE_ONE:
        if getattr(parallel, name, 1) != 1:
            errors.append(f"{name} must be 1")
    if not getattr(model, "enforce_eager", False):
        missing = hook_capture_refusals(vllm_config, resolved=resolved)
        if missing:
            errors.append(
                "enforce_eager is off, so the hook configuration is required; "
                "unmet: " + "; ".join(missing)
            )
    hooks = missing_graph_hooks()
    if hooks:
        errors.append(
            "the installed vLLM's ModelState lacks the model-state hooks "
            + ", ".join(hooks)
            + "; the plugin requires vllm#67583's model-state hooks"
        )
    errors.extend(pr6_poc_refusals(getattr(model, "hf_config", None)))
    errors.extend(_txg.transaction_graph_config_errors(vllm_config))
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
    """Raise one ``ValueError`` listing every refused engine setting."""
    message = engine_config_refusal(vllm_config, subject, resolved=resolved)
    if message is not None:
        raise ValueError(message)


def _absent(module: Any, name: str) -> bool:
    """Whether ``module`` lacks ``name``. A dotted name is a member of the
    name before the last dot: a callable attribute (a hook) or a constructor
    or call argument."""
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
    return not callable(getattr(owner, member, None))


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
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            missing.extend(f"{module_name}.{name} ({exc})" for name in names)
            continue
        missing.extend(
            f"{module_name}.{name} (not defined)"
            for name in names
            if _absent(module, name)
        )
    if missing:
        raise _dependency_error(
            missing,
            "adapter dependencies (a vLLM build with pdtgct/vllm#6's state "
            "layer, MambaManager.layer_config(recomputable=...), and "
            "vllm#67583's model-state hooks is required)",
        )


# @spec REG-014, REG-018
def check_conditional_dependencies(hf_config: Any) -> None:
    """Raise naming every missing conditional dependency the configuration
    needs: the graph-runtime rows when graphed execution is configured, the
    transaction-graph rows while transaction graphs are on. Model
    construction calls this before any weights load."""
    missing: list[str] = []
    if graphed_execution_configured(hf_config):
        for module_name, names in GRAPH_RUNTIME_DEPENDENCIES.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                missing.extend(f"{module_name}.{name} ({exc})" for name in names)
                continue
            missing.extend(
                f"{module_name}.{name} (not defined; graphed execution is configured)"
                for name in names
                if _absent(module, name)
            )
    if _txg.transaction_graphs_on(hf_config):
        for module_name, names in TRANSACTION_GRAPH_DEPENDENCIES.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                missing.extend(f"{module_name}.{name} ({exc})" for name in names)
                continue
            missing.extend(
                f"{module_name}.{name} (not defined; {TRANSACTION_GRAPH_KEY} is set)"
                for name in names
                if _absent(module, name)
            )
    if missing:
        raise _dependency_error(missing, "dependencies the configuration needs")
