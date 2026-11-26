# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph interface the tests are written against.

The specs (``docs/intent/transaction-graph/transaction-graph-specs.md``) name
the public functions; the design's Implementation notes name the model
module. Neither names every internal seam a test needs, so this module is the
one place that does. The code implements these names, or a change renames
them here (and in ``docs/implementation-layout.md``); no test imports a
transaction-graph module or reads a transaction-graph attribute except
through this module.

Every accessor imports lazily, so a test module imports cleanly while the
code does not exist yet and the test itself fails with the missing name.

Configuration module (imports no vLLM; TXG-CFG-012)
---------------------------------------------------

``vllm_nemotron_asr_streaming.transaction_graph_config``:

* ``transaction_graphs_on(hf_config) -> bool`` (TXG-CFG-001, TXG-CFG-014):
  off exactly when ``hf_config`` has no attribute
  ``transaction_graph_populations`` or its value is ``[]``.
* ``largest_population(hf_config) -> int``: ``N``; raises ``ValueError`` for
  a value TXG-CFG-004 refuses and when transaction graphs are off.
* ``transaction_graph_config_errors(vllm_config, *, has_state_group_layouts)
  -> list[str]`` (TXG-CFG-003). ``has_state_group_layouts`` is whether the
  running core's ``ModelState`` defines ``state_group_layouts`` (a property
  counts), which the module cannot import and its callers determine through
  ``engine_flags`` and pass in (TXG-CFG-016). It reads, duck-typed:
  ``vllm_config.model_config.hf_config``,
  ``vllm_config.model_config.enforce_eager``,
  ``vllm_config.scheduler_config.max_num_seqs``,
  ``.max_num_batched_tokens`` (``None`` while unresolved, TXG-CFG-015),
  ``.max_num_active_seqs`` (``None`` when unset: the admission limit is then
  ``max_num_seqs``, TXG-CFG-005, TXG-CFG-020) and ``.async_scheduling``
  (TXG-CFG-017); ``vllm_config.cache_config.num_gpu_blocks_override``
  (TXG-CFG-005);
  ``vllm_config.compilation_config.cudagraph_mode`` and ``.mode`` (``None``
  while unset; NONE is a mode whose value is 0, TXG-CFG-018); the geometry
  through ``lookahead.served_lookahead(hf_config)`` (REG-LA-001).
  :func:`config_errors` passes the capability as that keyword.

Model module
------------

``vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph``:

* ``CHUNK = "chunk"``, ``ECHO = "echo"``; a cell key is the tuple
  ``(geometry_index, kind, count)`` (TXG-CELL-001).
* ``cell_keys(n, geometry_index) -> list[tuple[int, str, int]]``: the ``2N``
  keys of the inventory.
* ``classify(plan, *, n, geometry_index, has_graph, memory_profile=False,
  native_burst=False) -> Classification`` (TXG-COV-001): ``plan`` is a CPU
  ``advance.RowPlan``; ``has_graph(key) -> bool`` says whether the serving
  cell at ``key`` holds a graph. ``Classification`` has ``population`` (P),
  ``chunk_count`` (K), ``echo_count`` (E), ``cells`` (the needed keys, the
  chunk cell first), ``reason`` (``None`` when every needed cell holds a
  graph, else one of :data:`REASONS`), and ``kind`` / ``count`` (the
  fallback counter's key fields, ``None`` when the reason has no kind).
  Classification does not compare signatures: ``execute`` does (TXG-EXEC-001).
* ``arena_rows(plan) -> tuple[list[int], list[int]]`` (TXG-EXEC-009): the
  step positions the chunk section and the echo section list, in that
  order.
* A chunk cell has ``decoder_tier``, the dense decoder tier it replays
  (TXG-CELL-006).
* ``transaction_graph_counters(obj) -> Mapping`` (TXG-OBS-002): ``{}`` while
  transaction graphs are off (TXG-OBS-009); otherwise a mapping with keys
  ``"counters"`` (see :data:`COUNTER_KEYS` and :data:`FALLBACK_KEY_FIELDS`)
  and ``"capture_record"`` (see :class:`Record`).
* The model's binding is ``model._transaction_graph_binding`` (``None`` while
  off). The binding has ``cells`` (key -> cell; a cell has ``graph``, an
  object with ``replay()``, or ``None``, and ``signature``), ``failed`` (the
  latch), ``verified`` (TXG-EXEC-015) and
  ``execute(core, input_ids, inputs_embeds, plan, **transaction_kwargs)``,
  which takes ``advance_model_rows``'s parameters and returns the escaped row
  tensor of a covered step, or ``None`` when it declines the step (the caller
  then runs the option-off transaction).
* The binding's ``static_inputs`` (TXG-EXEC-007): the pair ``(ids_static,
  embeds_static)`` of binding-owned step static inputs, ``ids_static`` of
  shape ``(P_max,)`` and ``embeds_static`` of shape ``(P_max, H)`` (``H``
  the carrier width), the tensors whose first ``P`` rows a covered step's
  two device copies write and from which each cell gathers its rows by the
  arena's step positions; ``None`` while no capture call has allocated them
  (and after their release).

Transaction module (``nemotron_asr.advance``)
---------------------------------------------

* ``RowPlan.planned_rejection``: a per-row ``bool`` vector, empty when no row
  failed at planning (TXG-COV-010, TXG-MASK-006). It has the default of
  ``is_file``.
* ``RowPlan.is_fresh``: the per-row ``bool`` freshness mask (the designs'
  name; TXG-MASK-001, TXG-EXEC-005). A fresh row gathers its zero-initial
  state from the null page and its book row through
  ``where(is_fresh, initial, gathered)``. It equals ``~has_initial_states_p``
  for the plan's prefill rows, which is today's plan authority for freshness.
* The phase functions of ``advance_model_rows`` (TXG-TXN-006, TXG-TXN-007):
  :data:`PHASES`, each a module-level function of ``advance`` looked up by
  name at call time (so a spy installed on the module sees the call).
* The structural preflight's capture-only mode (TXG-CAP-017): the keyword
  :data:`CAPTURE_ONLY_KEYWORD` of ``prepare_model_rows`` and of the
  preflight itself, ``advance._structural_preflight(plan, n_ids, n_embeds,
  *, capture_only=False)``, which returns the composed CPU index tensor and
  raises ``ValueError`` naming the first structural defect. Only a capture
  call's cell programs pass it, through ``prepare_model_rows``;
  ``advance_model_rows`` has no such parameter.
* ``CommitPlan.queue_counts``: the device ``int32`` ``(P,)`` remaining-count
  view (FILE-QC-001, TXG-QC-001, TXG-QC-002).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any

CONFIG_MODULE = "vllm_nemotron_asr_streaming.transaction_graph_config"
GRAPH_MODULE = "vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph"
ADVANCE_MODULE = "vllm_nemotron_asr_streaming.nemotron_asr.advance"

KEY = "transaction_graph_populations"
CHUNK = "chunk"
ECHO = "echo"

#: Fallback reasons (TXG-COV-003 to TXG-COV-007, TXG-COV-012, TXG-COV-014).
REASONS = (
    "multi-token",
    "chunk-uncovered",
    "echo-uncovered",
    "signature",
    "multi-geometry",
    "uncaptured",
    "empty",
    "memory-profile",
    "native-burst",
)

#: The counters of TXG-OBS-001, as the reader names them.
COUNTER_KEYS = (
    "captured_cells",
    "replays",
    "covered_steps",
    "fallbacks",
    "histogram",
    "prologue_host_seconds",
)

#: The key shape of the ``fallbacks`` counter: ``(reason, kind, count)``, with
#: ``reason`` one of :data:`REASONS`, ``kind`` :data:`CHUNK`, :data:`ECHO` or
#: ``None``, and ``count`` an ``int`` (TXG-COV-003, TXG-OBS-001). The
#: ``replays`` counter is keyed by cell key ``(geometry_index, kind, count)``;
#: the ``histogram`` by ``(K, E)``; ``covered_steps`` by ``"one_kind"`` and
#: ``"split"``.
FALLBACK_KEY_FIELDS = ("reason", "kind", "count")
COVERED_SHAPES = ("one_kind", "split")


class Record:
    """Key names of the capture record, ``counters(model)["capture_record"]``
    (TXG-OBS-003, TXG-MEM-002 to TXG-MEM-008).

    Per capture (``PROFILING``, ``SERVING``): ``CALL_TOTAL``, ``TXN_TOTAL``,
    ``KIND_TOTALS`` (``{"chunk": int, "echo": int}``), ``CELLS`` (capture
    order, each ``{"geometry", "kind", "count", "growth"}``), ``PEAK``,
    ``SECONDS`` and ``PAGE_COUNT`` (the page count of the capture's
    state-group layout, the first dimension of its component tensors;
    TXG-OBS-003). At the top level: ``IDENTITY`` (the configuration identity,
    including the value of ``transaction_graph_populations``), the margin
    figures (``MARGIN``, ``FREE_BEFORE``, ``FREE_AFTER``, ``CHARGE``),
    ``ABSORBED``, ``EFFECTIVE``, ``REQUIRED``, ``REASON``, ``RAN``,
    ``SIZING`` (the KV-cache sizing, named as TXG-MEM-010 names it, with its
    value: see :data:`SIZING_NAMES`), ``FINAL_BOUND_COUNT`` (under the
    extensible KV cache, the first dimension of the views the state layer
    reported at core's extension, TXG-LIFE-009; a field separate from the
    page counts), the allocator settings (``ALLOC_CONF``, ``CUBLAS``),
    ``VERSIONS`` and ``VERIFICATION_SECONDS``.

    A field that only a profiling capture produces (``PROFILING``'s figures,
    ``MARGIN``, ``FREE_BEFORE``, ``FREE_AFTER``, ``CHARGE``, ``ABSORBED``)
    holds :data:`NOT_APPLICABLE` when no profiling capture ran, as
    ``FINAL_BOUND_COUNT`` does under every other KV-cache sizing
    (TXG-MEM-006, TXG-OBS-003): never zero, never ``None``.
    """

    RECORD = "capture_record"
    PROFILING = "profiling"
    SERVING = "serving"
    CALL_TOTAL = "call_total"
    TXN_TOTAL = "transaction_total"
    KIND_TOTALS = "kind_totals"
    CELLS = "cells"
    PEAK = "transient_peak"
    SECONDS = "capture_seconds"
    PAGE_COUNT = "page_count"
    IDENTITY = "configuration_identity"
    MARGIN = "margin_bytes"
    FREE_BEFORE = "margin_free_before"
    FREE_AFTER = "margin_free_after"
    CHARGE = "margin_charge"
    ABSORBED = "absorbed_rounding"
    EFFECTIVE = "effective_cache"
    REQUIRED = "profiling_required"
    REASON = "profiling_reason"
    RAN = "profiling_ran"
    SIZING = "kv_cache_sizing"
    FINAL_BOUND_COUNT = "final_bound_count"
    ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"
    CUBLAS = "CUBLAS_WORKSPACE_CONFIG"
    VERSIONS = ("torch_version", "cuda_version", "driver_version")
    VERIFICATION_SECONDS = "verification_seconds"


#: The marking of a capture-record field that does not apply to this
#: worker's startup (TXG-MEM-006, TXG-OBS-003): a value no reader can take
#: for a measurement.
NOT_APPLICABLE = "not applicable"


class Sizing:
    """Core's KV-cache sizings (the specs' preamble), as the tests choose
    them. The extensible KV cache is core's default on CUDA: neither fixed
    setting, and ``enable_extensible_kv_cache`` on. A profiled budget is
    ``--no-enable-extensible-kv-cache`` with neither fixed setting. Either
    fixed setting (``--num-gpu-blocks-override``,
    ``--kv-cache-memory-bytes``) turns the extensible KV cache off."""

    EXTENSIBLE = "extensible"
    PROFILED = "profiled"
    BLOCK_COUNT = "block_count"
    BUDGET = "budget"
    ALL = (EXTENSIBLE, PROFILED, BLOCK_COUNT, BUDGET)


def sizing_settings(
    sizing: str, *, blocks: int | None = None, budget: int | None = None
) -> dict[str, Any]:
    """The cache settings of ``sizing``: ``num_gpu_blocks_override``,
    ``kv_cache_memory_bytes`` and ``enable_extensible_kv_cache``.
    ``blocks`` is required for :attr:`Sizing.BLOCK_COUNT` and ``budget``
    for :attr:`Sizing.BUDGET`; neither is accepted for another sizing, so a
    test sets an override only where its spec names one."""
    if sizing not in Sizing.ALL:
        raise ValueError(f"unknown KV-cache sizing {sizing!r}")
    if (blocks is not None) != (sizing == Sizing.BLOCK_COUNT):
        raise ValueError(f"blocks={blocks!r} does not fit the sizing {sizing!r}")
    if (budget is not None) != (sizing == Sizing.BUDGET):
        raise ValueError(f"budget={budget!r} does not fit the sizing {sizing!r}")
    return {
        "num_gpu_blocks_override": blocks,
        "kv_cache_memory_bytes": budget,
        "enable_extensible_kv_cache": sizing == Sizing.EXTENSIBLE,
    }


def sizing_arguments(
    sizing: str, *, blocks: int | None = None, budget: int | None = None
) -> dict[str, Any]:
    """The engine arguments (``AsyncEngineArgs`` or ``CacheConfig``
    keywords) a serve command of ``sizing`` sets, which the configuration's
    validators and the plugin's configuration hook see: none for core's
    default, the extensible KV cache, which a command without KV-cache
    options selects; ``enable_extensible_kv_cache`` false for a profiled
    budget; ``num_gpu_blocks_override`` or ``kv_cache_memory_bytes`` for a
    fixed block count or budget, each of which turns the extensible KV cache
    off. ``blocks`` and ``budget`` follow :func:`sizing_settings`."""
    settings = sizing_settings(sizing, blocks=blocks, budget=budget)
    if sizing == Sizing.EXTENSIBLE:
        return {}
    if sizing == Sizing.PROFILED:
        return {"enable_extensible_kv_cache": False}
    if sizing == Sizing.BLOCK_COUNT:
        return {"num_gpu_blocks_override": int(settings["num_gpu_blocks_override"])}
    return {"kv_cache_memory_bytes": int(settings["kv_cache_memory_bytes"])}


#: How a message or a record names each KV-cache sizing (TXG-MEM-010,
#: TXG-OBS-003): the information, not the wording.
SIZING_NAMES = {
    Sizing.EXTENSIBLE: r"(?i)extensible",
    Sizing.PROFILED: r"(?i)profiled",
    Sizing.BLOCK_COUNT: r"num_gpu_blocks_override|--num-gpu-blocks-override",
    Sizing.BUDGET: r"kv_cache_memory_bytes|--kv-cache-memory-bytes",
}


#: ``advance_model_rows``'s phase functions, in order (TXG-TXN-006).
PHASES = ("prepare_model_rows", "model_rows_body", "commit_model_rows")

#: The keyword of the structural preflight's capture-only mode (TXG-CAP-017).
CAPTURE_ONLY_KEYWORD = "capture_only"

#: The model attribute holding the transaction-graph binding.
BINDING_ATTRIBUTE = "_transaction_graph_binding"

#: Geometry index by served lookahead (TXG-CELL-001).
GEOMETRY_BY_LOOKAHEAD = {0: 0, 1: 1}


def config_module() -> ModuleType:
    return importlib.import_module(CONFIG_MODULE)


def graph_module() -> ModuleType:
    return importlib.import_module(GRAPH_MODULE)


def advance_module() -> ModuleType:
    return importlib.import_module(ADVANCE_MODULE)


def transaction_graphs_on(hf_config: Any) -> bool:
    return config_module().transaction_graphs_on(hf_config)


#: The keyword through which callers pass the state-layer capability.
STATE_LAYER_KEYWORD = "has_state_group_layouts"


def takes_state_layer_keyword() -> bool:
    """Whether ``transaction_graph_config_errors`` takes
    :data:`STATE_LAYER_KEYWORD` (TXG-CFG-016)."""
    import inspect

    fn = config_module().transaction_graph_config_errors
    return STATE_LAYER_KEYWORD in inspect.signature(fn).parameters


def config_errors(vllm_config: Any, *, state_group_layouts: bool = True) -> list[str]:
    """``transaction_graph_config_errors`` on ``vllm_config`` for a core whose
    ``ModelState`` defines ``state_group_layouts`` (the target core) or, with
    ``state_group_layouts=False``, lacks it (TXG-CFG-016).

    A function that does not take the keyword yet is called without it for
    the target core's capability, so the tests of the other conditions keep
    their own signal; the tests of TXG-CFG-016 fail on it
    (``test_config.py``), as does every call for a core without the
    accessor."""
    fn = config_module().transaction_graph_config_errors
    if takes_state_layer_keyword():
        return fn(vllm_config, **{STATE_LAYER_KEYWORD: state_group_layouts})
    if not state_group_layouts:
        raise AssertionError(
            "transaction_graph_config_errors takes no "
            f"{STATE_LAYER_KEYWORD!r} keyword: its callers cannot pass whether "
            "the running core's ModelState defines state_group_layouts "
            "(TXG-CFG-016)"
        )
    return fn(vllm_config)


def largest_population(hf_config: Any) -> int:
    return config_module().largest_population(hf_config)


def cell_keys(n: int, geometry_index: int) -> list[tuple[int, str, int]]:
    return list(graph_module().cell_keys(n, geometry_index))


def classify(plan: Any, **kwargs: Any) -> Any:
    return graph_module().classify(plan, **kwargs)


def arena_rows(plan: Any) -> tuple[list[int], list[int]]:
    chunk, echo = graph_module().arena_rows(plan)
    return list(chunk), list(echo)


def counters(obj: Any) -> Mapping[str, Any]:
    return graph_module().transaction_graph_counters(obj)


def fallback_key(**fields: Any) -> tuple[Any, ...]:
    """A ``fallbacks`` counter key from its named fields, in
    :data:`FALLBACK_KEY_FIELDS` order."""
    if set(fields) != set(FALLBACK_KEY_FIELDS):
        raise TypeError(f"a fallback key has the fields {FALLBACK_KEY_FIELDS}")
    return tuple(fields[name] for name in FALLBACK_KEY_FIELDS)


def binding_of(model: Any) -> Any:
    """The model's transaction-graph binding (``None`` while off). Fails, not
    errors, when the model has no such attribute."""
    if not hasattr(model, BINDING_ATTRIBUTE):
        raise AssertionError(
            f"the model has no {BINDING_ATTRIBUTE!r} attribute "
            "(transaction-graph binding not implemented)"
        )
    return getattr(model, BINDING_ATTRIBUTE)


#: The binding attribute holding the step static inputs (TXG-EXEC-007).
STATIC_INPUTS_ATTRIBUTE = "static_inputs"


def static_inputs(binding: Any) -> Any:
    """The binding's ``(ids_static, embeds_static)`` (see the module
    docstring). Fails, not errors, when the binding has no such attribute."""
    if not hasattr(binding, STATIC_INPUTS_ATTRIBUTE):
        raise AssertionError(
            f"the transaction-graph binding has no {STATIC_INPUTS_ATTRIBUTE!r} "
            "(TXG-EXEC-007's step static inputs are not observable)"
        )
    return getattr(binding, STATIC_INPUTS_ATTRIBUTE)


def phase_function(name: str) -> Any:
    advance = advance_module()
    if not callable(getattr(advance, name, None)):
        raise AssertionError(
            f"advance has no phase function {name!r} (TXG-TXN-006: "
            "advance_model_rows is not split into prepare, body and commit)"
        )
    return getattr(advance, name)


def structural_preflight() -> Any:
    """``advance._structural_preflight`` (TXG-CAP-017). Fails, not errors,
    when the preflight takes no capture-only keyword."""
    import inspect

    fn = advance_module()._structural_preflight
    if CAPTURE_ONLY_KEYWORD not in inspect.signature(fn).parameters:
        raise AssertionError(
            f"the structural preflight takes no {CAPTURE_ONLY_KEYWORD!r} "
            "keyword (TXG-CAP-017's capture-only mode is not implemented)"
        )
    return fn
