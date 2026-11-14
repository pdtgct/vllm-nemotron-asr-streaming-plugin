# SPDX-License-Identifier: Apache-2.0
"""Encoder geometries and lookahead selection.

The model implements five encoder geometries, identified by their right
context (the lookahead in encoder frames). This module holds the one table
relating right context, cadence label and geometry index, and reads the
served lookahead set and the default lookahead from a served
configuration. The checks that refuse a bad selection run in the
model-configuration hook (``streaming_config``), after every
``--hf-overrides`` key has been applied.

Importing this module imports no vLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: (right context, cadence label, geometry index). The geometry index is the
#: position in this table and the value the chunk envelope carries.
GEOMETRIES: tuple[tuple[int, str, int], ...] = (
    (0, "80ms", 0),
    (1, "160ms", 1),
    (3, "320ms", 2),
    (6, "560ms", 3),
    (13, "1120ms", 4),
)

#: The implemented right contexts, in table order.
IMPLEMENTED_RIGHT_CONTEXTS: tuple[int, ...] = tuple(g[0] for g in GEOMETRIES)

#: The right context of the 160 ms native-burst profile.
NATIVE_BURST_LOOKAHEAD = 1

_LABEL_BY_RIGHT = {right: label for right, label, _ in GEOMETRIES}
_INDEX_BY_RIGHT = {right: index for right, _, index in GEOMETRIES}
_RIGHT_BY_LABEL = {label: right for right, label, _ in GEOMETRIES}


def cadence_label(right_context: int) -> str:
    """Return the cadence label of an implemented right context."""
    try:
        return _LABEL_BY_RIGHT[right_context]
    except (KeyError, TypeError):
        raise ValueError(
            f"lookahead {right_context!r} is not an implemented right context "
            f"{IMPLEMENTED_RIGHT_CONTEXTS}"
        ) from None


def geometry_index(right_context: int) -> int:
    """Return the geometry index of an implemented right context."""
    try:
        return _INDEX_BY_RIGHT[right_context]
    except (KeyError, TypeError):
        raise ValueError(
            f"lookahead {right_context!r} is not an implemented right context "
            f"{IMPLEMENTED_RIGHT_CONTEXTS}"
        ) from None


def right_context_for_label(label: str) -> int:
    """Return the right context of a cadence label such as ``"320ms"``."""
    try:
        return _RIGHT_BY_LABEL[label]
    except (KeyError, TypeError):
        raise ValueError(
            f"cadence {label!r} is not one of {tuple(_RIGHT_BY_LABEL)}"
        ) from None


def served_lookahead_source(hf_config: Any) -> str:
    """Name the configuration field the served set is read from."""
    if getattr(hf_config, "supported_num_lookahead_tokens", None) is not None:
        return "supported_num_lookahead_tokens"
    return "card_num_lookahead_tokens"


# @spec REG-LA-001
def served_lookaheads(hf_config: Any) -> list[int]:
    """Return the served lookahead set.

    It is ``supported_num_lookahead_tokens`` when the configuration has it
    (set by ``--hf-overrides``, or stored by an already-translated
    configuration), and ``card_num_lookahead_tokens`` otherwise. The result
    is not validated here; an absent set is empty.
    """
    value = getattr(hf_config, "supported_num_lookahead_tokens", None)
    if value is None:
        value = getattr(hf_config, "card_num_lookahead_tokens", None)
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    raise ValueError(
        f"{served_lookahead_source(hf_config)} must be a list of integers, "
        f"got {value!r}"
    )


def default_lookahead(hf_config: Any) -> int | None:
    """Return the configuration's default lookahead, or ``None`` when unset."""
    return getattr(hf_config, "num_lookahead_tokens", None)


def validate_selected_lookahead(
    *,
    selected: object,
    supported: object,
    require_160ms_profile: bool = False,
) -> None:
    """Check a configuration's selected right context.

    ``selected`` has to be an implemented right context and, when
    ``supported`` is given, one of its members. The native-burst 160 ms
    profile needs right context 1; the served set is not required to be
    exactly ``(1,)``.
    """
    if (
        isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected not in IMPLEMENTED_RIGHT_CONTEXTS
    ):
        raise ValueError(
            f"lookahead {selected!r} is not an implemented right context "
            f"{IMPLEMENTED_RIGHT_CONTEXTS}"
        )
    if supported is not None and selected not in list(supported):  # type: ignore[call-overload]
        raise ValueError(
            f"lookahead {selected} is not in the served set {sorted(supported)}"  # type: ignore[call-overload]
        )
    if require_160ms_profile and selected != NATIVE_BURST_LOOKAHEAD:
        raise ValueError(
            "native burst's 160ms profile needs att_context_right=1, and "
            "1 must be in the served lookahead set. The served set is not "
            "required to be exactly (1,)."
        )
