# SPDX-License-Identifier: Apache-2.0
"""Row status in the model (SESS-MODEL-003): no row status is recoverable.

Every nonzero collected stream-row status, of any class, fails that request
alone, and the step's other rows go on. The status resolution imports no
vLLM (CPU). The served behaviour, ``postprocess_state`` failing that request
through ``state_slots.fail_requests`` with transaction graphs on and off, is
``test_geometry_check_gpu``.
"""

from __future__ import annotations

import pytest


def _status_classes() -> dict[str, int]:
    from vllm_nemotron_asr_streaming.nemotron_asr import advance, frontend

    classes = {}
    for module in (frontend, advance):
        for name, value in vars(module).items():
            if name.startswith("ROW_STATUS_") and isinstance(value, int) and value:
                classes[name] = value
    return classes


class _Registry:
    """Every report is current (a registered session, its generation)."""

    def resolve_status(self, *, request_id, admission_generation, clean):
        return True


def test_the_status_classes_are_found():
    classes = _status_classes()
    assert "ROW_STATUS_GEOMETRY" in classes and "ROW_STATUS_SEQUENCE" in classes
    assert len(classes) >= 10


# @spec SESS-MODEL-003
@pytest.mark.parametrize("name", sorted(_status_classes()))
def test_every_nonzero_status_class_fails_that_request_alone(name):
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        SessionStatusReport,
        resolve_status_reports,
    )

    status = _status_classes()[name]
    reports = [
        SessionStatusReport("bad", admission_generation=1, row_status=status),
        SessionStatusReport("good", admission_generation=2, row_status=0),
        SessionStatusReport("also-good", admission_generation=3, row_status=0),
    ]
    statuses, failed = resolve_status_reports(_Registry(), reports, lease_ok=True)
    assert failed == {"bad"}, f"{name} left the request alive"
    assert statuses["good"] == 0 and statuses["also-good"] == 0


# @spec SESS-MODEL-003
def test_a_composite_status_including_geometry_is_terminal():
    from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
        SessionStatusReport,
        resolve_status_reports,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.frontend import (
        ROW_STATUS_GEOMETRY,
        ROW_STATUS_SEQUENCE,
    )

    reports = [
        SessionStatusReport("geometry", 1, ROW_STATUS_GEOMETRY),
        SessionStatusReport("both", 2, ROW_STATUS_GEOMETRY | ROW_STATUS_SEQUENCE),
    ]
    _, failed = resolve_status_reports(_Registry(), reports, lease_ok=True)
    assert failed == {"geometry", "both"}
