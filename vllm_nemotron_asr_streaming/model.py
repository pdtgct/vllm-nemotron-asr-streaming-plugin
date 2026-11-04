# SPDX-License-Identifier: Apache-2.0
"""Engine-facing streaming model.

The implementation lives in ``nemotron_asr``. This module is the lazy
registry target, so importing it loads the ported model.
"""

from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
    NemotronASRModelState,
)
from vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr import NemotronASRForRNNT

# Registry class. The implementation name stays ``NemotronASRForRNNT``;
# vLLM selects it by the architecture string, not by this alias.
Nemotron3_5AsrStreamingForRNNT = NemotronASRForRNNT

__all__ = [
    "Nemotron3_5AsrStreamingForRNNT",
    "NemotronASRModelState",
]
