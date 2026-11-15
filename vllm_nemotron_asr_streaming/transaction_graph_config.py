# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph option and its configuration checks.

``transaction_graph_populations`` is a key of the served configuration (a
flat ``--hf-overrides`` key, or stored by an already-translated
configuration). An absent key or exactly an empty list is off; every other
value is on and is validated before ``N``, its largest entry, is computed.

``transaction_graphs_on`` is the one implementation of that predicate, and
``transaction_graph_config_errors`` the one implementation of the option's
own conditions (populations, geometry, native burst, segment arms and
``--enforce-eager``). Startup validation (``streaming_config``, through
``engine_flags``) and model-state construction both call it. The shared
runner and platform conditions of the hook configuration are
``engine_flags``'s, not repeated here.

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
    blocks = _resolved_limit(
        getattr(vllm_config, "cache_config", None), "num_gpu_blocks_override"
    )
    if blocks is not None and isinstance(value, list):
        pages = blocks - 1
        over = [entry for entry in value if _is_int(entry) and entry > pages]
        if over:
            problems.append(
                f"entries {over} exceed the {pages} non-null pages of "
                f"num_gpu_blocks_override {blocks}"
            )
    if not problems:
        return None
    rule = "a list of distinct int populations, each in 1 … " + (
        "min(max_num_seqs, max_num_batched_tokens)" if bound is None else str(bound)
    )
    return f"{KEY} {value!r} is refused: " + "; ".join(problems) + f" (needs {rule})"


def _geometry_condition(hf_config: Any) -> str | None:
    remedy = (
        "set num_lookahead_tokens 0 or 1 (80 or 160 ms) in --hf-overrides, "
        f"or remove {KEY}"
    )
    try:
        lookahead = served_lookahead(hf_config)
    except ValueError as exc:
        return (
            f"{KEY} needs the server's lookahead to be 0 or 1, and it cannot "
            f"be resolved: {exc}; {remedy}"
        )
    if lookahead in TRANSACTION_GRAPH_LOOKAHEADS:
        return None
    return (
        f"{KEY} is refused at the server's lookahead {lookahead}: this release "
        f"captures transaction graphs only at lookahead 0 or 1; {remedy}"
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
        f"{KEY} needs the default segment arms, encoder {REQUIRED_ENCODER_ARM} "
        f"and decoder {REQUIRED_DECODER_ARM}; here " + " and ".join(wrong)
    )


# @spec TXG-CFG-003, TXG-CFG-004, TXG-CFG-005, TXG-CFG-006, TXG-CFG-007
# @spec TXG-CFG-008, TXG-CFG-009, TXG-CFG-010, TXG-CFG-011, TXG-CFG-012
# @spec TXG-CFG-013, TXG-CFG-015
def transaction_graph_config_errors(vllm_config: Any) -> list[str]:
    """Return every violated transaction-graph condition of ``vllm_config``.

    Empty while transaction graphs are off. Reads the configuration
    duck-typed and changes nothing. The runner and platform conditions of
    the hook configuration (``cudagraph_mode`` NONE, compilation off, a CUDA
    platform) are ``engine_flags``'s and are not repeated here.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if not transaction_graphs_on(hf_config):
        return []
    errors: list[str] = []
    populations = _populations_condition(vllm_config, getattr(hf_config, KEY))
    if populations is not None:
        errors.append(populations)
    geometry = _geometry_condition(hf_config)
    if geometry is not None:
        errors.append(geometry)
    native_burst = getattr(hf_config, "experimental_native_burst", False)
    if native_burst:
        errors.append(
            f"{KEY} cannot be combined with experimental_native_burst "
            f"{native_burst!r}: remove one of them"
        )
    arms = _arm_condition(hf_config)
    if arms is not None:
        errors.append(arms)
    if getattr(model_config, "enforce_eager", False):
        errors.append(
            f"{KEY} is refused under --enforce-eager: core runs no capture hook "
            "under --enforce-eager, so no cell would be captured; remove "
            f"--enforce-eager (the hook configuration) or remove {KEY}"
        )
    return errors
