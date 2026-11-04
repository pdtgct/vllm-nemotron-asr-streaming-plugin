# SPDX-License-Identifier: Apache-2.0
"""Registration stub for the streaming RNN-T model.

``ModelRegistry`` loads this class by string. Constructing it does not
build weights or a ``PersistentStateLayerBase``. The port of
``vllm/models/nemotron_asr/`` is still ahead. See ``TODO.md``.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from vllm.model_executor.layers.state import PersistentStateLayerBase
from vllm.v1.worker.gpu.model_states.interface import ModelState

_NOT_PORTED = (
    "Nemotron ASR streaming model is not ported. "
    "Source is pdtgct/vllm rfc1-5212-vllm @ 43c0bed5 "
    "under vllm/models/nemotron_asr/. See TODO.md."
)


class NemotronAsrStreamingModelState(ModelState):
    """Stand-in for the source ``NemotronASRModelState``.

    ``resolve_model_state_cls`` returns this type because the model
    defines ``get_model_state_cls``. Every hook raises until the port
    lands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_NOT_PORTED)

    def prepare_inputs_embeds(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_NOT_PORTED)

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_NOT_PORTED)

    def prepare_dummy_inputs(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_NOT_PORTED)

    def prepare_attn(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_NOT_PORTED)


class Nemotron3_5AsrStreamingForRNNT(nn.Module):
    """Streaming architecture. Distinct from the offline class name."""

    supports_persistent_state = True
    # Kept on the class so loading the model imports the RFC1 layer base.
    persistent_state_layer_base = PersistentStateLayerBase

    @staticmethod
    def get_model_state_cls() -> type[NemotronAsrStreamingModelState]:
        return NemotronAsrStreamingModelState

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        del vllm_config, prefix
        raise NotImplementedError(_NOT_PORTED)
