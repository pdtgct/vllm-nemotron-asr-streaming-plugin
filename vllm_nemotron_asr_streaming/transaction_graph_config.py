# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph option and its configuration checks.

``transaction_graph_populations`` is a key of the served configuration (a
flat ``--hf-overrides`` key, or stored by an already-translated
configuration). An absent key or exactly an empty list is off; every other
value is on and is validated before ``N``, its largest entry, is computed.

``transaction_graphs_on`` is the one implementation of that predicate, and
``transaction_graph_config_errors`` the one implementation of the option's
own conditions: populations, admission limit, block count, geometry, native
burst, segment arms, ``--enforce-eager``, the state-group layout accessor and
the restricted startup (scheduling, runner modes, pool). Startup validation
(``streaming_config``, through ``engine_flags``) and model and model-state
construction call it. The hook configuration's own runner and platform
conditions are ``engine_flags.runner_mode_refusals``', not repeated here.

Importing this module imports no vLLM.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .lookahead import served_lookahead

#: The served-configuration key of the option.
KEY = "transaction_graph_populations"

#: The lookaheads (right contexts) whose cells this release specifies: 80 and
#: 160 ms.
TRANSACTION_GRAPH_LOOKAHEADS: tuple[int, ...] = (0, 1)

#: The segment arms the option runs with (the card's default arms).
REQUIRED_ENCODER_ARM = "eager-graphed"
REQUIRED_DECODER_ARM = "dense-graphed"

#: The arm the model runs when a key is unset.
UNSET_ENCODER_ARM = "eager"
UNSET_DECODER_ARM = "compact-eager"

_ABSENT = object()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# @spec TXG-CFG-001, TXG-CFG-014
def transaction_graphs_on(hf_config: Any) -> bool:
    """Whether transaction graphs are on: the key is present and its value
    is anything but an empty list (a malformed value is on, and refused)."""
    value = getattr(hf_config, KEY, _ABSENT)
    if value is _ABSENT:
        return False
    return not (isinstance(value, list) and not value)


def _population_problems(value: object, bound: int | None) -> list[str]:
    """What is wrong with a populations value, without the page bound."""
    if not isinstance(value, list):
        return [f"it is of type {type(value).__name__}, not a list"]
    problems: list[str] = []
    not_int = [entry for entry in value if not _is_int(entry)]
    if not_int:
        problems.append(
            f"{', '.join(repr(e) for e in not_int)} not an int (a bool is not accepted)"
        )
    ints = [entry for entry in value if _is_int(entry)]
    duplicates = sorted(entry for entry, n in Counter(ints).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate entries {duplicates}")
    upper = "" if bound is None else f" … {bound}"
    outside = [
        entry for entry in ints if entry < 1 or (bound is not None and entry > bound)
    ]
    if outside:
        problems.append(f"entries {outside} outside 1{upper}")
    return problems


# @spec TXG-CFG-002, TXG-CFG-004
def largest_population(hf_config: Any) -> int:
    """Return ``N``, the largest entry of an accepted populations value.

    Raises ``ValueError`` while transaction graphs are off and for a value
    that the populations condition refuses on its own (not a list, an entry
    that is not an ``int``, a duplicate, an entry below 1). The batching and
    page bounds depend on the engine configuration and are checked by
    ``transaction_graph_config_errors``.
    """
    if not transaction_graphs_on(hf_config):
        raise ValueError(f"transaction graphs are off ({KEY} is absent or empty)")
    value = getattr(hf_config, KEY)
    problems = _population_problems(value, None)
    if problems:
        raise ValueError(f"{KEY} {value!r} is refused: " + "; ".join(problems))
    return max(value)


def _resolved_limit(owner: Any, name: str) -> int | None:
    value = getattr(owner, name, None)
    return value if _is_int(value) else None


def _admission_limit(vllm_config: Any) -> tuple[int | None, str]:
    """The number of requests core's scheduler admits into RUNNING and the
    setting it comes from: ``max_num_active_seqs`` when set, otherwise
    ``max_num_seqs`` (``None`` while neither is known)."""
    scheduler = getattr(vllm_config, "scheduler_config", None)
    active = _resolved_limit(scheduler, "max_num_active_seqs")
    if active is not None:
        return active, "max_num_active_seqs"
    return _resolved_limit(scheduler, "max_num_seqs"), "max_num_seqs"


def _int_entries(value: object) -> list[int]:
    return [e for e in value if _is_int(e)] if isinstance(value, list) else []


def _populations_condition(vllm_config: Any, value: object) -> str | None:
    scheduler = getattr(vllm_config, "scheduler_config", None)
    limits = [
        limit
        for limit in (
            _resolved_limit(scheduler, "max_num_seqs"),
            _resolved_limit(scheduler, "max_num_batched_tokens"),
        )
        if limit is not None
    ]
    # TXG-CFG-015: a limit core has not resolved yet bounds nothing here;
    # model-state construction runs this again on the final configuration.
    bound = min(limits) if limits else None
    problems = _population_problems(value, bound)
    if not problems:
        return None
    rule = "1 … " + (
        "min(max_num_seqs, max_num_batched_tokens)" if bound is None else str(bound)
    )
    return (
        f"{KEY} {value!r} is refused: " + ", ".join(problems) + " (give a list "
        f"of distinct int populations, each in {rule})"
    )


def _admission_condition(vllm_config: Any, value: object) -> str | None:
    limit, setting = _admission_limit(vllm_config)
    if limit is None:
        # TXG-CFG-020: before the limit is known this bounds nothing.
        return None
    over = [entry for entry in _int_entries(value) if entry > limit]
    if not over:
        return None
    return (
        f"{KEY} {value!r} is refused: entries {over} exceed the admission "
        f"limit {limit} ({setting}), so no step can hold that many rows "
        f"(give a largest population of at most {limit}, or a larger "
        "--max-num-active-seqs, at most max_num_seqs, or none)"
    )


def _block_count_condition(vllm_config: Any, value: object) -> str | None:
    blocks = _resolved_limit(
        getattr(vllm_config, "cache_config", None), "num_gpu_blocks_override"
    )
    if blocks is None:
        return None
    pages = blocks - 1
    limit, setting = _admission_limit(vllm_config)
    entries = _int_entries(value)
    over = [entry for entry in entries if entry > pages]
    if limit is not None and pages < limit:
        found = (
            f"its {pages} non-null pages are fewer than the admission limit "
            f"{limit} ({setting})"
        )
        least = f"{limit + 1}, the admission limit plus one"
    elif limit is None and over:
        # TXG-CFG-005: before the admission limit is known, the override is
        # compared with the entries only.
        found = f"entries {over} exceed its {pages} non-null pages"
        least = f"{max(entries) + 1} and at least the admission limit plus one"
    else:
        return None
    return (
        f"{KEY} {value!r} is refused under num_gpu_blocks_override {blocks}: "
        f"{found}, and core's scheduler refuses fewer pages only after both "
        f"captures (give a --num-gpu-blocks-override of at least {least}, or a "
        "lower admission limit, --max-num-active-seqs or --max-num-seqs, with "
        "the largest population at most it)"
    )


def _geometry_condition(hf_config: Any) -> str | None:
    remedy = "set num_lookahead_tokens 0 or 1 (80 or 160 ms) in --hf-overrides"
    try:
        lookahead = served_lookahead(hf_config)
    except ValueError as exc:
        return (
            f"{KEY} needs the server's lookahead to be 0 or 1, and it cannot "
            f"be resolved ({exc}): {remedy}"
        )
    if lookahead in TRANSACTION_GRAPH_LOOKAHEADS:
        return None
    return (
        f"{KEY} is refused at the server's lookahead {lookahead}: this release "
        f"captures transaction graphs only at lookahead 0 or 1 ({remedy})"
    )


def _arm_condition(hf_config: Any) -> str | None:
    encoder = getattr(hf_config, "encoder_execution_arm", None)
    decoder = getattr(hf_config, "decode_dispatch_arm", None)
    encoder = UNSET_ENCODER_ARM if encoder is None else encoder
    decoder = UNSET_DECODER_ARM if decoder is None else decoder
    wrong = []
    if encoder != REQUIRED_ENCODER_ARM:
        wrong.append(f"encoder_execution_arm resolves to {encoder!r}")
    if decoder != REQUIRED_DECODER_ARM:
        wrong.append(f"decode_dispatch_arm resolves to {decoder!r}")
    if not wrong:
        return None
    return (
        f"{KEY} needs the default segment arms, here " + " and ".join(wrong) + " "
        f"(set encoder_execution_arm {REQUIRED_ENCODER_ARM} and "
        f"decode_dispatch_arm {REQUIRED_DECODER_ARM}, or leave both to the card)"
    )


def _mode_name(mode: Any) -> str | None:
    """A mode by its name (``None`` while unset)."""
    if mode is None:
        return None
    name = getattr(mode, "name", None)
    return name if isinstance(name, str) else repr(mode)


def _is_none_mode(mode: Any) -> bool:
    """NONE: a mode whose value is 0 (an unset mode is not NONE)."""
    if mode is None:
        return False
    value = getattr(mode, "value", mode)
    return _is_int(value) and value == 0


def _runner_mode_condition(vllm_config: Any) -> str | None:
    compilation = getattr(vllm_config, "compilation_config", None)
    shown = []
    for label, mode in (
        ("cudagraph_mode", getattr(compilation, "cudagraph_mode", None)),
        ("the compilation mode", getattr(compilation, "mode", None)),
    ):
        if not _is_none_mode(mode):
            name = _mode_name(mode)
            shown.append(f"{label} is {'unset' if name is None else name}")
    if not shown:
        return None
    return (
        f"{KEY} needs cudagraph_mode and the compilation mode NONE, and here "
        + " and ".join(shown)
        + " (pass -cc.cudagraph_mode none -cc.mode none)"
    )


def _pool_condition(vllm_config: Any) -> str | None:
    cache = getattr(vllm_config, "cache_config", None)
    if getattr(cache, "num_gpu_blocks_override", None) is not None:
        return None
    budget = getattr(cache, "kv_cache_memory_bytes", None)
    if _is_int(budget) and budget > 0:
        return None
    limit, _ = _admission_limit(vllm_config)
    least = "the admission limit plus one" + (
        "" if limit is None else f" ({limit + 1})"
    )
    return (
        f"{KEY} needs an explicit block count or KV budget on this core, which "
        "turns core's extensible KV cache off, and neither "
        "num_gpu_blocks_override nor a positive kv_cache_memory_bytes "
        f"({'unset' if budget is None else budget!r}) is set (give a "
        "--num-gpu-blocks-override of at least "
        f"{least}, or a positive --kv-cache-memory-bytes that holds at least "
        "that many state pages)"
    )


# @spec TXG-CFG-003, TXG-CFG-004, TXG-CFG-005, TXG-CFG-006, TXG-CFG-007
# @spec TXG-CFG-008, TXG-CFG-009, TXG-CFG-010, TXG-CFG-011, TXG-CFG-012
# @spec TXG-CFG-013, TXG-CFG-015, TXG-CFG-016, TXG-CFG-017, TXG-CFG-018
# @spec TXG-CFG-019, TXG-CFG-020
def transaction_graph_config_errors(
    vllm_config: Any, *, has_state_group_layouts: bool
) -> list[str]:
    """Return every violated transaction-graph condition of ``vllm_config``.

    Empty while transaction graphs are off. ``has_state_group_layouts`` is
    whether the running core's ``ModelState`` defines
    ``state_group_layouts``, which this module (it imports no vLLM) cannot
    determine; its callers do, through
    ``engine_flags.model_state_has_state_group_layouts``. Reads the
    configuration duck-typed and changes nothing. The hook configuration's
    own conditions (graph mode, breakable piecewise, compilation mode,
    platform, graph selection) are ``engine_flags.runner_mode_refusals``'s and
    are not repeated here; the runner-mode condition below is this option's
    narrower one.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if not transaction_graphs_on(hf_config):
        return []
    value = getattr(hf_config, KEY)
    errors: list[str] = []
    if not has_state_group_layouts:
        errors.append(
            f"{KEY} {value!r} needs core's state-group layout accessor "
            "ModelState.state_group_layouts, through which the cells reach the "
            f"state pages, and the running core's ModelState lacks it (remove {KEY})"
        )
    errors.append(_populations_condition(vllm_config, value))
    errors.append(_admission_condition(vllm_config, value))
    errors.append(_block_count_condition(vllm_config, value))
    errors.append(_geometry_condition(hf_config))
    native_burst = getattr(hf_config, "experimental_native_burst", False)
    if native_burst:
        errors.append(
            f"{KEY} cannot be combined with experimental_native_burst "
            f"{native_burst!r} (remove experimental_native_burst)"
        )
    errors.append(_arm_condition(hf_config))
    if getattr(model_config, "enforce_eager", False):
        errors.append(
            f"{KEY} is refused under --enforce-eager: core runs no capture hook "
            "under --enforce-eager, so no cell would be captured (remove "
            f"--enforce-eager to run the hook configuration, or remove {KEY})"
        )
    scheduler = getattr(vllm_config, "scheduler_config", None)
    scheduling = getattr(scheduler, "async_scheduling", None)
    if scheduling is not False:
        shown = (
            "unset, which the target core resolves to asynchronous scheduling"
            if scheduling is None
            else repr(scheduling)
        )
        errors.append(
            f"{KEY} needs synchronous scheduling, and async_scheduling is "
            f"{shown} (pass --no-async-scheduling)"
        )
    errors.append(_runner_mode_condition(vllm_config))
    errors.append(_pool_condition(vllm_config))
    return [error for error in errors if error is not None]
