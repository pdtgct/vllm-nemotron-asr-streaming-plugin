# SPDX-License-Identifier: Apache-2.0
"""The frozen calibration record (TXG-CAL-003).

The calibration run (TXG-CAL-001, -002) is a preregistered A40 measurement;
it writes one JSON record, read here from ``TXG_CALIBRATION_RECORD`` (the
check is skipped when it is unset). The CPU test pins the check itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from support import txgcap_qual as q


def complete_record():
    return {
        "audio": {"utterance_id": "1089-134686-0000", "chunk_index": 3},
        "snapshot_rule": "after the prefix",
        "plan": "one continuing CHUNK row",
        "procedure": "2 warm-up and 21 counted steps per arm",
        "u1_on": 4.1,
        "u1_off": 9.7,
        "ladder": {"on": {"1": 4.1}, "off": {"1": 9.7}},
        "served_replay_fraction": 0.995,
        "device": "NVIDIA A40",
        "date": "2026-11-20",
        "identity": {k: "x" for k in q.CALIBRATION_IDENTITY},
    }


# @spec TXG-CAL-003
def test_the_calibration_record_holds_every_required_field():
    path = os.environ.get("TXG_CALIBRATION_RECORD")
    if not path:
        pytest.skip("TXG_CALIBRATION_RECORD not set")
    missing = q.missing_calibration_fields(json.loads(Path(path).read_text()))
    assert missing == [], missing


# @spec TXG-CAL-003
def test_the_calibration_check_names_each_missing_field():
    record = complete_record()
    assert q.missing_calibration_fields(record) == []
    del record["served_replay_fraction"]
    del record["identity"]["process_constants"]
    del record["audio"]["chunk_index"]
    assert sorted(q.missing_calibration_fields(record)) == sorted(
        ["served_replay_fraction", "identity.process_constants", "audio.chunk_index"]
    )
