# SPDX-License-Identifier: Apache-2.0
"""Engine-facing streaming model.

The implementation lives in ``nemotron_asr``. This module is the lazy
registry target, so importing it loads the ported model.
"""

from vllm_nemotron_asr_streaming.engine_flags import check_adapter_dependencies

# @spec REG-010, REG-013
# REG-010: name every missing adapter dependency before the adapter imports
# them one by one.
check_adapter_dependencies()

from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (  # noqa: E402
    NemotronASRModelState,
)
from vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr import (  # noqa: E402
    NemotronASRForRNNT,
)

# Registry class. The implementation name stays ``NemotronASRForRNNT``;
# vLLM selects it by the architecture string, not by this alias.
Nemotron3_5AsrStreamingForRNNT = NemotronASRForRNNT

__all__ = [
    "Nemotron3_5AsrStreamingForRNNT",
    "NemotronASRModelState",
]
