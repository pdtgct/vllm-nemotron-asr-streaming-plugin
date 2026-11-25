# SPDX-License-Identifier: Apache-2.0
"""How a refusal of transaction graphs names its condition and remedy
(TXG-CFG-003 to TXG-CFG-009, TXG-CFG-016 to TXG-CFG-020) and how a capture
failure routes its remedies (TXG-CAP-008, TXG-MEM-010 to TXG-MEM-013,
TXG-MEM-017), for the tests that read those messages. Imports no vLLM.

Tests assert the information a spec requires, not its wording
(``docs/implementation-layout.md``): each remedy below is a pattern for an
option or value the spec names, and each condition is recognized by the
names only it carries.
"""

from __future__ import annotations

import contextlib
import json
import re

KEY = "transaction_graph_populations"

#: Remedies, each a pattern for what its spec names.
NO_ASYNC_SCHEDULING = re.compile(r"--no-async-scheduling")
RUNNER_MODES_NONE = re.compile(r"(?i)-cc\.cudagraph_mode[ =]none\s+-cc\.mode[ =]none")
BLOCKS_OVERRIDE = re.compile(r"--num-gpu-blocks-override")
KV_BYTES = re.compile(r"--kv-cache-memory-bytes")
ACTIVE_SEQS = re.compile(r"--max-num-active-seqs")
MAX_SEQS = re.compile(r"--max-num-seqs")
LOOKAHEAD = re.compile(r"num_lookahead_tokens")
ACCESSOR = re.compile(r"state_group_layouts")
REMOVAL = re.compile(r"(?i)\bremov")
EXTENSIBLE = re.compile(r"(?i)extensible")
#: A remedy that lowers the budget or the block count (TXG-MEM-012,
#: TXG-MEM-013), which TXG-MEM-017 forbids.
LOWERS_BUDGET = re.compile(
    r"(?i)(smaller|lower|reduce|decrease)\w*[^;\n]{0,40}?"
    r"(--kv-cache-memory-bytes|--num-gpu-blocks-override|budget|block count)"
)
#: A remedy that raises the budget or the block count (TXG-MEM-017).
RAISES_BUDGET = re.compile(
    r"(?i)(larger|greater|higher|increase|at least|raise)\w*[^;\n]{0,60}?"
    r"(--kv-cache-memory-bytes|--num-gpu-blocks-override|budget|block count)"
    r"|(--kv-cache-memory-bytes|--num-gpu-blocks-override)[^;\n]{0,40}?"
    r"(larger|greater|higher|at least)"
)
#: Remedies of TXG-MEM-011: a smaller largest population, a smaller
#: ``max_num_seqs``, or transaction graphs off.
SMALLER_POPULATION = re.compile(r"(?i)(smaller|lower|fewer)[^;\n]{0,40}?population")
SMALLER_MAX_SEQS = re.compile(
    r"(?i)(smaller|lower)[^;\n]{0,40}?(max_num_seqs|--max-num-seqs)"
)
OPTION_OFF = re.compile(
    r"(?i)(transaction graphs off|remov\w*[^;\n]{0,20}?transaction_graph_populations"
    r"|transaction_graph_populations[^;\n]{0,40}?(absent|remov|off))"
)


def found(pattern: re.Pattern[str], message: str) -> bool:
    return pattern.search(message) is not None


def shows_value(error: str, value: object) -> bool:
    """Whether ``error`` shows ``value``: its ``repr``, ``str`` or JSON
    rendering, or a string bare or in any common quotes. An empty rendering
    shows nothing, so ``""`` must appear quoted."""
    forms = {repr(value), str(value)}
    with contextlib.suppress(TypeError, ValueError):
        forms.add(json.dumps(value))
    if isinstance(value, str):
        forms |= {f"'{value}'", f'"{value}"', f"`{value}`"}
    return any(form and form.lower() in error.lower() for form in forms)


def shows_number(message: str, value: int) -> bool:
    """Whether ``message`` states the integer ``value`` on its own."""
    return re.search(rf"(?<![\d.]){value}(?![\d.])", message) is not None


def is_state_layer_condition(error: str) -> bool:
    """TXG-CFG-016's condition: transaction graphs need core's state-group
    layout accessor ``ModelState.state_group_layouts``, which the running
    core lacks; remedy: remove ``transaction_graph_populations``. This is the
    one matcher of that condition (the registration tests use it too)."""
    return (
        KEY in error
        and found(ACCESSOR, error)
        and "ModelState" in error
        and found(REMOVAL, error)
    )


#: A clause that refuses transaction graphs on account of the state layer in
#: other words than TXG-CFG-016's (the proof of concept's refusal said
#: "not supported on the ... state layer").
_ON_THE_STATE_LAYER = re.compile(r"(?i)state layer")


def state_layer_refusals(message: str) -> list[str]:
    """The clauses of ``message`` (its parts between ``;`` and line breaks)
    that refuse ``transaction_graph_populations`` on account of the state
    layer: TXG-CFG-016's condition (it names core's accessor,
    :data:`ACCESSOR`) or any other clause naming the option with the state
    layer. On a core whose ``ModelState`` defines ``state_group_layouts``
    there is none (TXG-CFG-016)."""
    return [
        clause
        for clause in re.split(r"[;\n]", message)
        if KEY in clause
        and (found(ACCESSOR, clause) or found(_ON_THE_STATE_LAYER, clause))
    ]


def is_admission_condition(error: str) -> bool:
    """TXG-CFG-020's condition, recognized by its remedy
    ``--max-num-active-seqs`` without the block-count option of
    TXG-CFG-005 and TXG-CFG-019."""
    return (
        KEY in error
        and found(ACTIVE_SEQS, error)
        and not found(BLOCKS_OVERRIDE, error)
        and not found(KV_BYTES, error)
    )


def is_pool_condition(error: str) -> bool:
    """TXG-CFG-019's condition: both remedies, a block count and a KV
    budget."""
    return KEY in error and found(BLOCKS_OVERRIDE, error) and found(KV_BYTES, error)


def is_block_count_condition(error: str) -> bool:
    """TXG-CFG-005's condition: the block-count remedy without the KV-budget
    one (TXG-CFG-019's)."""
    return KEY in error and found(BLOCKS_OVERRIDE, error) and not found(KV_BYTES, error)


#: The configuration remedies a capture failure can give (TXG-MEM-011 to
#: TXG-MEM-013, TXG-MEM-017), each by the option or change it names.
CONFIGURATION_REMEDIES: dict[str, re.Pattern[str]] = {
    "--num-gpu-blocks-override": BLOCKS_OVERRIDE,
    "--kv-cache-memory-bytes": KV_BYTES,
    "--max-num-active-seqs": ACTIVE_SEQS,
    "--max-num-seqs": MAX_SEQS,
    "smaller max_num_seqs": SMALLER_MAX_SEQS,
    "smaller largest population": SMALLER_POPULATION,
    "transaction graphs off": OPTION_OFF,
}


#: The configuration options a capture failure's diagnostic can name with
#: their current values (TXG-MEM-010's budget setting; ``N`` and the
#: admission limit): CLI flag -> the option's names.
OPTION_NAMES: dict[str, tuple[str, ...]] = {
    "--num-gpu-blocks-override": (
        "--num-gpu-blocks-override",
        "num_gpu_blocks_override",
    ),
    "--kv-cache-memory-bytes": ("--kv-cache-memory-bytes", "kv_cache_memory_bytes"),
    "--max-num-active-seqs": ("--max-num-active-seqs", "max_num_active_seqs"),
    "--max-num-seqs": ("--max-num-seqs", "max_num_seqs"),
}


def current_settings(vllm_config: object) -> dict[str, object]:
    """The current value of each option of :data:`OPTION_NAMES` that
    ``vllm_config`` sets (duck-typed; ``None`` values are left out)."""
    cache = getattr(vllm_config, "cache_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    values = {
        "--num-gpu-blocks-override": getattr(cache, "num_gpu_blocks_override", None),
        "--kv-cache-memory-bytes": getattr(cache, "kv_cache_memory_bytes", None),
        "--max-num-active-seqs": getattr(scheduler, "max_num_active_seqs", None),
        "--max-num-seqs": getattr(scheduler, "max_num_seqs", None),
    }
    return {flag: value for flag, value in values.items() if value is not None}


#: What may stand between an option and its current value in a statement of
#: the configuration: punctuation (``--max-num-seqs (3)``,
#: ``num_gpu_blocks_override=8``) or a stating phrase (``--kv-cache-memory-bytes
#: fixed at 1073741824``, ``--max-num-seqs is 3``).
_STATED_AS = r"(?:[\s=:(\"'`]|\b(?:fixed|set|at|to|is|of)\b)*"


def diagnostic_facts(message: str, settings: dict[str, object]) -> list[str]:
    """The places where ``message`` names an option of ``settings`` together
    with its current value (``--kv-cache-memory-bytes 1073741824``,
    ``--kv-cache-memory-bytes fixed at 1073741824``,
    ``num_gpu_blocks_override=8``, ``--max-num-seqs (3)``): facts about the
    configuration, not suggested changes."""
    facts = []
    for flag, value in settings.items():
        names = "|".join(re.escape(name) for name in OPTION_NAMES[flag])
        pattern = rf"(?:{names}){_STATED_AS}{re.escape(str(value))}(?![\d.])"
        facts += [m.group(0) for m in re.finditer(pattern, message)]
    return facts


#: The phrasing of a suggested change (a remedy): an imperative (``set``,
#: ``raise``, ``lower``, ``use``, ``remove``, ``turn ... off``), a comparative
#: target (``a smaller --max-num-seqs``, ``a larger --kv-cache-memory-bytes``,
#: ``of at least 9``), or the word remedy itself (``remedies: ...``). A
#: clause without one states facts, whatever options it names.
SUGGESTION = re.compile(
    r"(?i)\b(?:set|use|pass|raise|increase|lower|reduce|decrease|remove|drop"
    r"|disable|turn|switch|leave|add|choose|give|try|consider|instead"
    r"|larger|smaller|greater|higher|fewer|at least|at most|remed\w*)\b"
)


def _clauses(message: str) -> list[str]:
    """``message``'s clauses: its parts between ``;``, line breaks and
    sentence ends."""
    return [c for c in re.split(r"[;\n]|\.(?=\s|$)", message) if c.strip()]


def suggested_changes(message: str, settings: dict[str, object] | None = None) -> str:
    """The clauses of ``message`` that suggest a change (:data:`SUGGESTION`),
    joined by ``;``, each without its diagnostic facts
    (:func:`diagnostic_facts`): what remains names only what to change."""
    changes = []
    for clause in _clauses(message):
        for fact in diagnostic_facts(clause, settings or {}):
            clause = clause.replace(fact, " ")
        if found(SUGGESTION, clause):
            changes.append(clause)
    return ";".join(changes)


def configuration_remedies(
    message: str, settings: dict[str, object] | None = None
) -> list[str]:
    """The configuration remedies ``message`` gives (empty for a break of
    core's state-group layout contract, TXG-CAP-008, TXG-MEM-010): the
    changes of :data:`CONFIGURATION_REMEDIES` it suggests
    (:func:`suggested_changes`). An option named in a clause that suggests no
    change (``budget setting: --kv-cache-memory-bytes fixed at 1073741824``),
    or named with its current value in ``settings``
    (:func:`current_settings`), states the configuration, as TXG-MEM-010's
    budget setting does, and is not a remedy."""
    changes = suggested_changes(message, settings)
    return [
        name
        for name, pattern in CONFIGURATION_REMEDIES.items()
        if found(pattern, changes)
    ]


def lower_admission_limit(message: str) -> bool:
    """TXG-MEM-017's and TXG-CFG-005's remedy: a lower admission limit
    (``--max-num-active-seqs``, or ``--max-num-seqs``)."""
    return found(ACTIVE_SEQS, message) or found(MAX_SEQS, message)
