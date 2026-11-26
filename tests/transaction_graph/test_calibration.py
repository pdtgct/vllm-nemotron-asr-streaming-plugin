# SPDX-License-Identifier: Apache-2.0
"""The frozen calibration records (TXG-CAL-001 to TXG-CAL-003).

The calibration run (TXG-CAL-001, -002) is a preregistered A40 measurement
run once under core's default KV-cache sizing and once under the step rule's
fixed ``--num-gpu-blocks-override``, each sizing giving its own frozen
record, which names the sizing in its configuration identity. The records
are read here from ``TXG_CALIBRATION_RECORD``, a JSON list of the records (a
single record is read as a list of one); the check is skipped when it is
unset. The CPU tests pin the checks themselves.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from support import txgcap_qual as q
from support.txg_api import Sizing


def complete_record(sizing=Sizing.EXTENSIBLE):
    identity = {k: "x" for k in q.CALIBRATION_IDENTITY}
    identity["kv_cache_sizing"] = sizing
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
        "identity": identity,
    }


@pytest.fixture
def calibration_records():
    path = os.environ.get("TXG_CALIBRATION_RECORD")
    if not path:
        pytest.skip("TXG_CALIBRATION_RECORD not set")
    records = json.loads(Path(path).read_text())
    return records if isinstance(records, list) else [records]


# @spec TXG-CAL-003
def test_the_calibration_record_holds_every_required_field(calibration_records):
    for record in calibration_records:
        missing = q.missing_calibration_fields(record)
        assert missing == [], missing


# @spec TXG-CAL-001, TXG-CAL-002
def test_the_calibration_ran_under_each_kv_cache_sizing(calibration_records):
    missing = q.missing_calibration_records(calibration_records)
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


# @spec TXG-CAL-003
def test_the_configuration_identity_holds_the_kv_cache_sizing():
    record = complete_record()
    del record["identity"]["kv_cache_sizing"]
    assert q.missing_calibration_fields(record) == ["identity.kv_cache_sizing"]


# @spec TXG-CAL-001, TXG-CAL-002
def test_one_record_per_sizing_with_u1_and_the_ladder():
    """U(1) and the ladder are measured at core's default sizing and under
    the step rule's fixed override, each in its own record; a missing sizing,
    a second record of one sizing, a record of another sizing and a ladder
    without one arm are each reported."""
    assert q.CALIBRATION_SIZINGS == (Sizing.EXTENSIBLE, Sizing.BLOCK_COUNT)
    both = [complete_record(Sizing.EXTENSIBLE), complete_record(Sizing.BLOCK_COUNT)]
    assert q.missing_calibration_records(both) == []
    assert q.missing_calibration_records(both[:1]), "a missing sizing passed"
    assert q.missing_calibration_records([*both, complete_record()]), (
        "two records of one sizing passed"
    )
    assert q.missing_calibration_records([*both, complete_record(Sizing.BUDGET)]), (
        "a record of another sizing passed"
    )
    one_arm = complete_record(Sizing.BLOCK_COUNT)
    del one_arm["ladder"]["off"]
    assert q.missing_calibration_records([both[0], one_arm]) == [
        f"{Sizing.BLOCK_COUNT}: ladder.off"
    ]


# @spec TXG-CAL-001
def test_the_calibration_runs_the_step_rule_procedure_under_both_sizings():
    """The calibration's driver spec: the step rule's mode and procedure (2
    warm-up and 21 counted steps per arm) at population 1, under the step
    rule's fixed override unless it names core's default sizing."""
    from support import txg_qual_driver as d

    spec = {
        "mode": "steprule",
        "card": "/card",
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "hf_overrides": {
            "num_lookahead_tokens": 0,
            "transaction_graph_populations": [1],
        },
    }
    assert d.run_sizing(spec) == Sizing.BLOCK_COUNT
    assert d.engine_settings(spec)["num_gpu_blocks_override"] == 65
    default = spec | {"kv_cache_sizing": Sizing.EXTENSIBLE}
    assert d.run_sizing(default) == Sizing.EXTENSIBLE
    settings = d.engine_settings(default)
    assert "num_gpu_blocks_override" not in settings
    assert "kv_cache_memory_bytes" not in settings
    assert (q.WARMUP, q.COUNTED) == (2, 21)
