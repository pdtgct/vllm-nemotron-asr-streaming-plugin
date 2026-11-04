# SPDX-License-Identifier: Apache-2.0
"""Streaming Nemotron ASR plugin for vLLM core.

Staging package. The model is not ported and no session is admitted.
See ``TODO.md``. Importing this package does not import vLLM.
"""

from __future__ import annotations

from .identity import ARCHITECTURE, OFFLINE_ARCHITECTURE
from .plugin import register

__all__ = [
    "ARCHITECTURE",
    "OFFLINE_ARCHITECTURE",
    "register",
]
