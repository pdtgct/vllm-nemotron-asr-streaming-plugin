# SPDX-License-Identifier: Apache-2.0
"""Contents of the tier 5 qualification record (TXG-QUAL-019, -020).

The preregistered A40 run writes its record as JSON; these checks read it
from ``TXG_QUALIFICATION_RECORD`` and are skipped when it is unset. The
record holds, under ``"capture_record"``, the tier 5 run's capture record
(the reader's ``capture_record``) and, under ``"first_run"``, the first A40
run's records keyed ``"<lookahead>,<N>"``, each naming its KV-cache sizing,
a fixed ``--num-gpu-blocks-override``, and that core's graph-memory estimate
was enabled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from support import txgcap_qual as q
from support.txg_api import Sizing


@pytest.fixture
def qualification_record():
    path = os.environ.get("TXG_QUALIFICATION_RECORD")
    if not path:
        pytest.skip("TXG_QUALIFICATION_RECORD not set")
    return json.loads(Path(path).read_text())


# @spec TXG-QUAL-019
def test_the_tier5_run_reports_every_capture_figure(qualification_record):
    missing = q.missing_capture_figures(qualification_record["capture_record"])
    assert missing == [], missing


# @spec TXG-QUAL-020
def test_the_first_a40_run_records_both_captures_at_each_n_and_geometry(
    qualification_record,
):
    runs = {
        tuple(int(v) for v in key.split(",")): value
        for key, value in qualification_record["first_run"].items()
    }
    missing = q.missing_first_run_figures(runs)
    assert missing == [], missing


# @spec TXG-QUAL-019, TXG-QUAL-020
def test_the_record_checks_name_what_is_missing():
    """The checks themselves, on a hand-made record (CPU)."""
    figures = {
        "call_total": 1,
        "transaction_total": 1,
        "kind_totals": {"chunk": 1, "echo": 1},
        "cells": [{"growth": 1}],
        "transient_peak": 1,
        "capture_seconds": 1.0,
    }
    assert q.missing_capture_figures({"profiling": figures, "serving": figures}) == []
    assert q.missing_capture_figures({"profiling": figures}) != []
    run = {
        "max_num_seqs": 64,
        "kv_cache_sizing": Sizing.BLOCK_COUNT,
        "estimate_enabled": True,
        "profiling": figures,
        "serving": figures,
        "margin_bytes": 1,
        "margin_charge": 1,
        "absorbed_rounding": 0,
    }
    runs = {(g, n): run for g in (0, 1) for n in (16, 32, 64)}
    assert q.missing_first_run_figures(runs) == []
    del runs[(1, 64)]
    assert q.missing_first_run_figures(runs) == ["run lookahead=1 N=64"]


# @spec TXG-QUAL-020
def test_the_first_a40_run_is_under_a_fixed_override_with_the_estimate():
    """A run at core's default sizing, or with the estimate disabled, has no
    profiling capture to record and is reported."""
    figures = {
        "call_total": 1,
        "transaction_total": 1,
        "kind_totals": {"chunk": 1, "echo": 1},
        "transient_peak": 1,
        "capture_seconds": 1.0,
    }
    run = {
        "max_num_seqs": 64,
        "kv_cache_sizing": Sizing.BLOCK_COUNT,
        "estimate_enabled": True,
        "profiling": figures,
        "serving": figures,
        "margin_bytes": 1,
        "margin_charge": 1,
        "absorbed_rounding": 0,
    }
    runs = {(g, n): dict(run) for g in (0, 1) for n in (16, 32, 64)}
    runs[(0, 16)]["kv_cache_sizing"] = Sizing.EXTENSIBLE
    runs[(1, 32)]["estimate_enabled"] = False
    missing = q.missing_first_run_figures(runs)
    assert missing == [
        "run lookahead=0 N=16: a fixed --num-gpu-blocks-override",
        "run lookahead=1 N=32: core's graph-memory estimate",
    ], missing
