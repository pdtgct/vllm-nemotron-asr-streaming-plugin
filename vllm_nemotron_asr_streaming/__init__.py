# SPDX-License-Identifier: Apache-2.0
"""Streaming Nemotron ASR plugin for vLLM core.

``register`` is the ``vllm.general_plugins`` entry point. Importing this
package does not import vLLM.
"""

from __future__ import annotations

from .identity import ARCHITECTURE, OFFLINE_ARCHITECTURE
from .plugin import register

__all__ = [
    "ARCHITECTURE",
    "OFFLINE_ARCHITECTURE",
    "register",
]
