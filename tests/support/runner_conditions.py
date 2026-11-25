# SPDX-License-Identifier: Apache-2.0
"""How a refusal names the hook configuration's conditions and remedies
(REG-ENG-005, REG-ENG-021 to REG-ENG-025) and the state layer's refusal of
native burst (REG-ENG-019), for the tests that read refusal messages. The
transaction-graph conditions, TXG-CFG-016's among them, are
``support.txg_conditions``'s.

Each condition is recognized by a phrase no other condition's text or remedy
contains, so a message that lists several conditions (REG-VAL-003) shows each
one separately: a bare ``cuda`` also occurs in ``-cc.cudagraph_mode`` and
``VLLM_USE_BREAKABLE_CUDAGRAPH``, and ``FULL`` inside ``FULL_AND_PIECEWISE``.
Mode names are matched whole. Each remedy is any one of the alternatives its
spec allows.
"""

from __future__ import annotations

import re

#: A mode name, whole: not part of a longer name such as FULL_AND_PIECEWISE.
_WHOLE = r"(?<![A-Za-z_]){}(?![A-Za-z_])"
#: ``cudagraph_mode`` as an option, an attribute or ``CUDAGraphMode``.
_GRAPH_MODE = r"(?i:cudagraph_?mode)"
_HOOK = r"(?:hook[- ]captured|capture hook)"


def whole(name: str) -> re.Pattern[str]:
    """``name`` as a whole mode name."""
    return re.compile(_WHOLE.format(re.escape(name)))


def _graph_mode_is(name: str) -> str:
    return _GRAPH_MODE + r"[^;\n]{0,30}?" + _WHOLE.format(name)


#: condition -> (the condition's phrase, the remedies its spec allows).
CONDITIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    # REG-ENG-021: cudagraph_mode is not FULL.
    "graph mode": (
        _graph_mode_is("FULL"),
        (r"(?i)-cc\.cudagraph_mode[ =]none", r"(?i)\bunset\b"),
    ),
    # REG-ENG-022: cudagraph_mode is not PIECEWISE while breakable graphs are on.
    "breakable piecewise": (
        _graph_mode_is("PIECEWISE")
        + r"|(?i:breakable)[^;\n]{0,60}?"
        + _WHOLE.format("PIECEWISE"),
        (r"VLLM_USE_BREAKABLE_CUDAGRAPH=0", r"(?i)-cc\.cudagraph_mode[ =]none"),
    ),
    # REG-ENG-023: the compilation mode is not STOCK_TORCH_COMPILE.
    "compilation mode": (
        r"STOCK_TORCH_COMPILE|(?i:stock torch\.compile)",
        (r"(?i)-cc\.mode[ =]none", r"(?i)\bunset\b"),
    ),
    # REG-ENG-024: a CUDA platform.
    "platform": (
        r"(?i)\bcuda\b[^;\n]{0,30}\bplatform\b|\bplatform\b[^;\n]{0,30}\bcuda\b",
        (r"--enforce-eager",),
    ),
    # REG-ENG-025: at least one hook-captured graph is selected.
    "hook-captured graph": (
        r"(?i)\bno\b[^;\n]{0,40}"
        + _HOOK
        + r"|"
        + _HOOK
        + r"[^;\n]{0,40}\bselected\b"
        + r"|\bselected\b[^;\n]{0,60}"
        + _HOOK,
        (r"\beager-graphed\b", r"\bdense-graphed\b", r"--enforce-eager"),
    ),
    # REG-ENG-025: encoder_execution_arm is not dense-graphed.
    "compiled encoder": (
        r"encoder_execution_arm\W{1,4}(?:is\W+|of\W+)?dense-graphed",
        (r"--enforce-eager",),
    ),
    # REG-ENG-025: chunk_bucket_graph_populations is empty.
    "chunk buckets": (r"chunk_bucket_graph_populations", (r"--enforce-eager",)),
}


def names_condition(message: str, condition: str) -> bool:
    return re.search(CONDITIONS[condition][0], message) is not None


def assert_condition(message: str, condition: str) -> None:
    assert names_condition(message, condition), (condition, message)


def assert_remedy(message: str, condition: str) -> None:
    """``message`` names one of the remedies ``condition``'s spec allows."""
    remedies = CONDITIONS[condition][1]
    assert any(re.search(r, message) for r in remedies), (condition, message)


def assert_named_with_remedy(message: str, condition: str) -> None:
    assert_condition(message, condition)
    assert_remedy(message, condition)


def clauses(message: str) -> list[str]:
    """The message's clauses: its parts between ``;`` and line breaks, the
    separators the plugin's refusals list their conditions with."""
    return [c for c in re.split(r"[;\n]", message) if c.strip()]


def eager_remedy_clauses(message: str) -> list[str]:
    """The clauses that name ``--enforce-eager`` (REG-ENG-005)."""
    return [c for c in clauses(message) if "--enforce-eager" in c]


def names_removal_of(text: str, option: str) -> bool:
    """Whether ``text`` names ``option`` and removing it."""
    return option in text and re.search(r"(?i)\bremov", text) is not None


TRANSACTION_GRAPH_KEY = "transaction_graph_populations"


def state_layer_mentions(message: str) -> int:
    """How often ``message`` mentions the state layer (REG-ENG-019's refusal
    of native burst names it). TXG-CFG-016's condition is matched by
    ``support.txg_conditions`` alone."""
    return len(re.findall(r"(?i)state layer", message))


def shows_value(error: str, value) -> bool:
    """Whether ``error`` shows ``value``: its ``repr``, ``str`` or JSON
    rendering."""
    import contextlib
    import json

    forms = {repr(value), str(value)}
    with contextlib.suppress(TypeError, ValueError):
        forms.add(json.dumps(value))
    return any(form and form in error for form in forms)
