# SPDX-License-Identifier: Apache-2.0
"""F6 lookahead selection.

The Hub encoder config's ``default_num_lookahead_tokens`` is the card's
default arm. The source port pinned 13, and its native-burst check
required the supported set to be exactly ``(1,)``. Neither substitution
is applied here. A caller passes the arm, and it has to be in the
card's supported set.
"""

from __future__ import annotations

from .identity import OFFLINE_ARCHITECTURE

# Snapshot of nvidia/nemotron-3.5-asr-streaming-0.6b config.json.
CARD_ARCHITECTURE = OFFLINE_ARCHITECTURE
CARD_DEFAULT_NUM_LOOKAHEAD_TOKENS = 3
CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS = (3, 0, 6, 13)

# Source ``manifests.CADENCES`` right-context values. Labels are the
# source's cadence names. This scaffold does not pick one.
SOURCE_CADENCE_BY_LOOKAHEAD: dict[int, str] = {
    0: "80ms",
    1: "160ms",
    3: "320ms",
    6: "560ms",
    13: "1120ms",
}

# Historical source pin (``_PINNED_LOOKAHEAD``). Not applied.
SOURCE_TRANSLATE_PINNED_LOOKAHEAD = 13

# Historical native-burst requirement (supported set exactly this tuple).
# Not applied. The 160 ms profile is lookahead 1, and only when requested
# and listed by the card.
SOURCE_NATIVE_BURST_REQUIRED_LOOKAHEADS = (1,)
NATIVE_BURST_LOOKAHEAD = 1


def resolve_served_lookahead(
    supported: object,
    requested: object,
    *,
    card_default: object = None,
) -> int:
    """Return ``requested`` when the card lists it.

    ``card_default`` is reported in the error and is never substituted.
    """
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError(
            "num_lookahead_tokens must be passed explicitly. "
            f"The card default {card_default!r} is not applied, and "
            "this plugin does not substitute lookahead 13 or 1. "
            f"Card supported set: {supported!r}."
        )
    if supported is None:
        raise ValueError(
            "card is missing supported_num_lookahead_tokens; "
            f"refusing to accept lookahead {requested}"
        )
    try:
        allowed = {int(value) for value in supported}  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(
            "supported_num_lookahead_tokens must be a sequence of integers"
        ) from exc
    if requested not in allowed:
        raise ValueError(
            f"lookahead {requested} is not in the card supported set "
            f"{sorted(allowed)}"
        )
    return requested


def validate_selected_lookahead(
    *,
    selected: object,
    supported: object,
    require_160ms_profile: bool = False,
) -> None:
    """Check a config's already-selected right context against the card."""
    resolve_served_lookahead(supported, selected)
    if require_160ms_profile and selected != NATIVE_BURST_LOOKAHEAD:
        raise ValueError(
            "native burst's 160ms profile needs att_context_right=1, and "
            "1 must be listed in supported_num_lookahead_tokens. The "
            "supported set is not required to be exactly (1,)."
        )
