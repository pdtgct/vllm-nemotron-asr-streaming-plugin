# SPDX-License-Identifier: Apache-2.0
"""Streaming sessions for the Nemotron streaming ASR plugin.

Importing this package imports neither vLLM nor the model class modules; the
factory module imports vLLM when a factory is created. Frontends discover the
factory through the entry point ``create_session_factory``.
"""

from __future__ import annotations

from typing import Any

from .errors import SessionFailed, SessionRefused
from .packer import Packer, PackerCapacityError, Unit

__all__ = [
    "Packer",
    "PackerCapacityError",
    "SessionFailed",
    "SessionRefused",
    "Unit",
    "create_session_factory",
]


# @spec REG-012
# @spec SESS-FAC-001, SESS-FAC-002, SESS-FAC-003
def create_session_factory(engine_client: Any, model_config: Any):
    """Return the session factory for the served model, or ``None``."""
    from .factory import create_session_factory as create

    return create(engine_client, model_config)


def __getattr__(name: str) -> Any:
    if name == "StreamingSessionFactory":
        from .factory import StreamingSessionFactory

        return StreamingSessionFactory
    raise AttributeError(name)
