# SPDX-License-Identifier: Apache-2.0
"""Encoder geometries and lookahead selection.

The model implements five encoder geometries, identified by their right
context (the lookahead in encoder frames). This module holds the one table
relating right context, cadence label and geometry index, and
``served_lookahead``, the one reader of the server's lookahead: a server
serves exactly one geometry. The checks that refuse a bad selection run in
the model-configuration hook (``streaming_config``), after every
``--hf-overrides`` key has been applied.

Importing this module imports no vLLM.
"""

from __future__ import annotations

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


_ABSENT = object()


def _implemented_message(field: str, value: object) -> str:
    return (
        f"{field} {value!r} is not an implemented right context "
        f"(one of {', '.join(str(r) for r in IMPLEMENTED_RIGHT_CONTEXTS)})"
    )


# @spec REG-LA-001
def served_lookahead(hf_config: Any) -> int:
    """Return the server's one lookahead (right context).

    It is ``num_lookahead_tokens`` when the configuration has it (set by
    ``--hf-overrides``, or stored by an already-translated configuration),
    and ``card_default_num_lookahead_tokens`` otherwise. A present value is
    used as it stands: ``None``, a ``bool``, a non-integer or a value that is
    not an implemented right context raises ``ValueError`` naming the field,
    as does a configuration that has neither field.
    """
    for field in ("num_lookahead_tokens", "card_default_num_lookahead_tokens"):
        value = getattr(hf_config, field, _ABSENT)
        if value is _ABSENT:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in IMPLEMENTED_RIGHT_CONTEXTS
        ):
            raise ValueError(_implemented_message(field, value))
        return int(value)
    raise ValueError(
        "the server's lookahead is not set: the configuration has neither "
        "num_lookahead_tokens nor card_default_num_lookahead_tokens. Set it "
        "with --hf-overrides '{\"num_lookahead_tokens\": <N>}', N one of "
        f"{', '.join(str(r) for r in IMPLEMENTED_RIGHT_CONTEXTS)}"
    )
