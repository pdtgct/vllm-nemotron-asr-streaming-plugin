# SPDX-License-Identifier: Apache-2.0
"""Architecture identity.

``nvidia/nemotron-3.5-asr-streaming-0.6b`` ``config.json`` lists
``architectures: ["Nemotron3_5AsrForRNNT"]``. That string is also the
model-registry key of the offline plugin
(``Sohaib-Ahmed21/vllm-nemotron-asr-plugin``) and of the source port's
``vllm/models/nemotron_asr/identity.py``. ``ModelRegistry.register_model``
overwrites an existing key. This package registers a different name and
leaves the offline key untouched.
"""

from __future__ import annotations

# Hub ``architectures[0]`` and the offline plugin registry key.
OFFLINE_ARCHITECTURE = "Nemotron3_5AsrForRNNT"

# Hub ``model_type``. The offline plugin also uses this string as its
# tokenizer mode and renderer key. This package does not register it.
OFFLINE_MODEL_TYPE = "nemotron3_5_asr"

ARCHITECTURE = "Nemotron3_5AsrStreamingForRNNT"
LAZY_MODEL = "vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT"

__all__ = [
    "ARCHITECTURE",
    "LAZY_MODEL",
    "OFFLINE_ARCHITECTURE",
    "OFFLINE_MODEL_TYPE",
]
