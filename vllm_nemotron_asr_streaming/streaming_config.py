# SPDX-License-Identifier: Apache-2.0
"""Per-architecture config check for the streaming registry key.

Registered only under ``Nemotron3_5AsrStreamingForRNNT``.
"""

from __future__ import annotations

from typing import Any

from .engine_flags import persistent_state_config_errors


class Nemotron3_5AsrStreamingConfig:
    """Refuse a persistent-state engine left on RFC12 defaults."""

    @staticmethod
    def verify_and_update_model_config(model_config: Any) -> None:
        del model_config

    @staticmethod
    def verify_and_update_config(vllm_config: Any) -> None:
        errors = persistent_state_config_errors(vllm_config)
        if errors:
            raise ValueError(
                "Nemotron3_5AsrStreamingForRNNT persistent state refuses: "
                + "; ".join(errors)
            )
