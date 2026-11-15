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
* ``transaction_graph_config_errors(vllm_config) -> list[str]`` (TXG-CFG-003).
  It reads, duck-typed: ``vllm_config.model_config.hf_config``,
  ``vllm_config.model_config.enforce_eager``,
  ``vllm_config.scheduler_config.max_num_seqs`` and
  ``.max_num_batched_tokens`` (``None`` while unresolved, TXG-CFG-015) and
  ``vllm_config.cache_config.num_gpu_blocks_override``; the geometry through
  ``lookahead.served_lookahead(hf_config)`` (REG-LA-001).

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

Transaction module (``nemotron_asr.advance``)
---------------------------------------------

* ``RowPlan.planned_rejection``: a per-row ``bool`` vector, empty when no row
  failed at planning (TXG-COV-010, TXG-MASK-006). It has the default of
  ``is_file``.
* ``RowPlan.is_fresh``: the per-row ``bool`` freshness mask the body selects
  on, ``where(is_fresh, initial, gathered)`` (the designs' name; TXG-MASK-001,
  TXG-EXEC-005). It equals ``~has_initial_states_p`` for the plan's prefill
  rows, which is today's plan authority for freshness.
* The phase functions of ``advance_model_rows`` (TXG-TXN-006, TXG-TXN-007):
  :data:`PHASES`, each a module-level function of ``advance`` looked up by
  name at call time (so a spy installed on the module sees the call).
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

#: Fallback reasons (TXG-COV-003 to TXG-COV-007, TXG-COV-012).
REASONS = (
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
    order, each ``{"geometry", "kind", "count", "growth"}``), ``PEAK`` and
    ``SECONDS``. At the top level: ``IDENTITY`` (the configuration identity,
    including the value of ``transaction_graph_populations``), the margin
    figures (``MARGIN``, ``FREE_BEFORE``, ``FREE_AFTER``, ``CHARGE``),
    ``ABSORBED``, ``EFFECTIVE``, ``REQUIRED``, ``REASON``, ``RAN``,
    ``BUDGET``, the allocator settings (``ALLOC_CONF``, ``CUBLAS``),
    ``VERSIONS`` and ``VERIFICATION_SECONDS``.
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
    BUDGET = "budget_setting"
    ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"
    CUBLAS = "CUBLAS_WORKSPACE_CONFIG"
    VERSIONS = ("torch_version", "cuda_version", "driver_version")
    VERIFICATION_SECONDS = "verification_seconds"


#: ``advance_model_rows``'s phase functions, in order (TXG-TXN-006).
PHASES = ("prepare_model_rows", "model_rows_body", "commit_model_rows")

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


def config_errors(vllm_config: Any) -> list[str]:
    return config_module().transaction_graph_config_errors(vllm_config)


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


def phase_function(name: str) -> Any:
    advance = advance_module()
    if not callable(getattr(advance, name, None)):
        raise AssertionError(
            f"advance has no phase function {name!r} (TXG-TXN-006: "
            "advance_model_rows is not split into prepare, body and commit)"
        )
    return getattr(advance, name)
