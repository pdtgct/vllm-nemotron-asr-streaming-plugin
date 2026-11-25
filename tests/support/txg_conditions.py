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
    core lacks; remedy: remove ``transaction_graph_populations``."""
    return (
        KEY in error
        and found(ACCESSOR, error)
        and "ModelState" in error
        and found(REMOVAL, error)
    )


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


def configuration_remedies(message: str) -> list[str]:
    """The configuration remedies ``message`` gives (empty for a break of
    core's state-group layout contract, TXG-CAP-008, TXG-MEM-010)."""
    return [
        name
        for name, pattern in CONFIGURATION_REMEDIES.items()
        if found(pattern, message)
    ]


def lower_admission_limit(message: str) -> bool:
    """TXG-MEM-017's and TXG-CFG-005's remedy: a lower admission limit
    (``--max-num-active-seqs``, or ``--max-num-seqs``)."""
    return found(ACTIVE_SEQS, message) or found(MAX_SEQS, message)
