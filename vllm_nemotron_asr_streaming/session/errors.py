# SPDX-License-Identifier: Apache-2.0
"""Errors a session factory and its leases raise to protocol frontends.

A frontend reads ``session_error`` and ``reason`` with ``getattr``; it does not
import this module. Exceptions that carry neither attribute (``ValueError`` for
bad audio, ``RuntimeError`` for concurrent calls) are caller bugs.
"""

from __future__ import annotations

REFUSED_REASONS = (
    "capacity",
    "active_limit",
    "engine_paused",
    "request_id_in_use",
    "cadence",
    "locale",
)
FAILED_REASONS = (
    "rejected_input",
    "abnormal_end",
    "expired",
    "engine_dead",
    "cancelled",
    "output_capacity",
    "input_capacity",
    "state",
)


# @spec SESS-ERR-003
# @spec SESS-ERR-001
class SessionRefused(Exception):
    """The session was not started or a request was refused; state unchanged."""

    session_error = "refused"

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or f"session refused: {reason}")


# @spec SESS-ERR-002
class SessionFailed(Exception):
    """The session ended or can no longer continue."""

    session_error = "failed"

    def __init__(self, reason: str, message: str | None = None) -> None:
        self.reason = reason
        super().__init__(message or f"session failed: {reason}")
