# SPDX-License-Identifier: Apache-2.0
"""F6 lookahead pin trap. Recorded, not resolved.

The Hub encoder config and the source port disagree about which
right-context arm is in force. This module selects none of them.
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

# ``configuration_nemotron_asr._PINNED_LOOKAHEAD``, written into
# ``att_context_right`` by ``translate_card_config``.
SOURCE_TRANSLATE_PINNED_LOOKAHEAD = 13

# ``validate_native_burst_config`` requires the declared supported set
# to be exactly this tuple. The Hub card's supported set is not that.
SOURCE_NATIVE_BURST_REQUIRED_LOOKAHEADS = (1,)
