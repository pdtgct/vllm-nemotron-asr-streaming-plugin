# SPDX-License-Identifier: Apache-2.0
"""Counters, capture record and log lines with transaction graphs on, on
CUDA (TXG-OBS-001 to TXG-OBS-008, TXG-CFG-002).

The environments run core's default KV-cache sizing, the extensible KV
cache, which runs no profiling capture, unless a test names a fixed block
count, under which core's estimate runs both captures."""

from __future__ import annotations

import logging
import re

import pytest
from support import txg_api as api
from support import txg_gpu as g
from support.txg_api import NOT_APPLICABLE, SIZING_NAMES, Sizing

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

N = 4


def info_lines(caplog, *needles):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
        and all(n.lower() in r.getMessage().lower() for n in needles)
    ]


# @spec TXG-OBS-001, TXG-OBS-002
def test_counters_count_cells_replays_shapes_fallbacks_and_the_histogram(
    on_cuda,
):
    env = on_cuda(startup=True, lookahead=0, n=2, max_num_seqs=4)
    records = env.compare()
    reading = api.counters(env.model)
    assert set(reading) >= {"counters", "capture_record"}
    counters = reading["counters"]
    assert set(api.COUNTER_KEYS) <= set(counters)
    assert counters["captured_cells"] == 4
    d = g.Driver(env)
    for r in ("a", "b", "c"):
        d.add_stream(r, [{}, {}])
    d.step(only=["a"])  # one kind, K = 1
    d.step(only=["b", "c"])
    d.run(30)
    counters = api.counters(env.model)["counters"]
    expected_hist: dict = {}
    for record in records:
        k = int(record.plan.is_chunk.sum())
        key = (k, len(record.plan.request_ids) - k)
        expected_hist[key] = expected_hist.get(key, 0) + 1
    assert dict(counters["histogram"]) == expected_hist
    covered = [r for r in records if r.covered]
    split = [
        r for r in covered if 0 < int(r.plan.is_chunk.sum()) < len(r.plan.request_ids)
    ]
    shapes = counters["covered_steps"]
    assert sum(shapes.values()) == len(covered)
    assert any(v == len(split) for v in shapes.values())
    assert sum(counters["replays"].values()) == sum(
        (int(r.plan.is_chunk.sum()) > 0) + (int((~r.plan.is_chunk).sum()) > 0)
        for r in covered
    )
    assert sum(counters["fallbacks"].values()) == len(records) - len(covered)
    assert counters["prologue_host_seconds"] > 0


@pytest.mark.parametrize("holder", ["runner", "worker", "engine_core"])
# @spec TXG-OBS-002
def test_reader_returns_the_same_reading_through_each_holder(on_cuda, holder):
    from types import SimpleNamespace

    env = on_cuda(startup=True, lookahead=0, n=N)
    runner = SimpleNamespace(model=env.model)
    worker = SimpleNamespace(model_runner=runner)
    core = SimpleNamespace(
        model_executor=SimpleNamespace(driver_worker=SimpleNamespace(worker=worker))
    )
    obj = {"runner": runner, "worker": worker, "engine_core": core}[holder]
    assert api.counters(obj) == api.counters(env.model)


FIGURES = (
    api.Record.CALL_TOTAL,
    api.Record.TXN_TOTAL,
    api.Record.KIND_TOTALS,
    api.Record.CELLS,
    api.Record.PEAK,
    api.Record.SECONDS,
    api.Record.PAGE_COUNT,
)
#: The fields only a profiling capture produces (TXG-MEM-006).
PROFILING_ONLY = (
    api.Record.MARGIN,
    api.Record.FREE_BEFORE,
    api.Record.FREE_AFTER,
    api.Record.CHARGE,
    api.Record.ABSORBED,
)


@pytest.mark.parametrize("sizing", [Sizing.EXTENSIBLE, Sizing.BLOCK_COUNT])
# @spec TXG-OBS-003, TXG-CFG-002, TXG-MEM-006
def test_capture_record_holds_the_identity_and_figures(on_cuda, sizing):
    """For each capture that ran, its figures and its state-group layout's
    page count; the KV-cache sizing; under the extensible KV cache the final
    bound count after core's extension, separate from the page counts, and
    the profiling-only fields marked not applicable; under a fixed block
    count the margin's figures and no final bound count."""
    R = api.Record
    env = on_cuda(startup=True, lookahead=0, n=N, populations=[1, 3, N], sizing=sizing)
    record = api.counters(env.model)[R.RECORD]
    assert "[1, 3, 4]" in repr(record[R.IDENTITY]), record[R.IDENTITY]
    ran = (R.SERVING,) if sizing == Sizing.EXTENSIBLE else (R.PROFILING, R.SERVING)
    for capture in ran:
        figures = record[capture]
        for key in FIGURES:
            assert key in figures, (capture, key)
        assert figures[R.PAGE_COUNT] == env.capacity, (capture, figures)
    for key in (
        R.EFFECTIVE,
        R.RAN,
        R.SIZING,
        R.FINAL_BOUND_COUNT,
        R.ALLOC_CONF,
        R.CUBLAS,
        R.VERIFICATION_SECONDS,
        *R.VERSIONS,
        *PROFILING_ONLY,
    ):
        assert key in record, key
    assert re.search(SIZING_NAMES[sizing], str(record[R.SIZING])), record[R.SIZING]
    if sizing == Sizing.EXTENSIBLE:
        assert record[R.RAN] is False
        assert record[R.PROFILING] == NOT_APPLICABLE or all(
            record[R.PROFILING][k] == NOT_APPLICABLE
            for k in (R.CALL_TOTAL, R.TXN_TOTAL, R.KIND_TOTALS)
        ), record[R.PROFILING]
        for key in PROFILING_ONLY:
            assert record[key] == NOT_APPLICABLE, (key, record[key])
        # The views core bound at its extension, as the state layer reported
        # them: a field of its own.
        assert record[R.FINAL_BOUND_COUNT] == env.core.final_count
    else:
        assert record[R.RAN] is True
        for key in (R.MARGIN, R.CHARGE):
            assert isinstance(record[key], int), (key, record[key])
        assert record[R.FINAL_BOUND_COUNT] == NOT_APPLICABLE
    same_cells = on_cuda(startup=True, lookahead=0, n=N, populations=[N])
    assert set(same_cells.binding.cells) == set(env.binding.cells)


@pytest.mark.parametrize("sizing", [Sizing.EXTENSIBLE, Sizing.BLOCK_COUNT])
# @spec TXG-OBS-004, TXG-OBS-005
def test_capture_and_verification_log_one_info_line_each(
    tmp_path, single_process_dist, caplog, monkeypatch, sizing
):
    """One INFO line per capture that ran: the serving capture alone under
    the extensible KV cache, both under a fixed block count."""
    caplog.set_level(logging.INFO)
    env = g.build_option_on(
        tmp_path, lookahead=0, n=N, sizing=sizing, monkeypatch=monkeypatch
    )
    ran = ("serving",) if sizing == Sizing.EXTENSIBLE else ("profiling", "serving")
    try:
        for name in ("profiling", "serving"):
            lines = info_lines(caplog, name, str(2 * N), api.KEY)
            expected = 1 if name in ran else 0
            assert len(lines) == expected, (
                name,
                [r.getMessage() for r in caplog.records],
            )
        assert len(info_lines(caplog, "verification", "pass")) == 1
    finally:
        env.shutdown()


# @spec TXG-OBS-006
def test_shutdown_logs_the_counter_summary(
    tmp_path, single_process_dist, caplog, monkeypatch
):
    env = g.build_option_on(
        tmp_path, lookahead=0, n=2, max_num_seqs=4, monkeypatch=monkeypatch
    )
    env.compare()
    d = g.Driver(env)
    for r in ("a", "b", "c"):
        d.add_stream(r, [{}])
    d.step()
    caplog.set_level(logging.INFO)
    env.state.shutdown()
    lines = info_lines(caplog, "replay", "split", "fallback")
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    assert "chunk-uncovered" in lines[0]


# @spec TXG-OBS-007
def test_a_second_engine_starts_with_clean_counters(on_cuda):
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    first = on_cuda(startup=True, lookahead=0, n=N)
    first.compare()
    d = g.Driver(first)
    d.add_stream("s1", [{}])
    d.step()
    advance._stage_batch_stats([("leftover", 1)])
    second = on_cuda(startup=True, lookahead=0, n=N)
    assert not advance.consume_batch_stats()
    counters = api.counters(second.model)["counters"]
    assert not counters["histogram"] and not counters["fallbacks"]
    assert sum(counters["replays"].values()) == 0


def _multiprocess_reader_run() -> dict:
    import tempfile
    from pathlib import Path

    from support import model_harness as h
    from support import txgcap_qual as q

    card = h.write_runnable_card(Path(tempfile.mkdtemp(prefix="txg-obs-")) / "card")
    settings = dict(
        model=str(card),
        config_format="nemotron_asr_streaming",
        load_format="dummy",
        dtype="float32",
        max_model_len=2048,
        max_num_seqs=N,
        hf_overrides={
            "num_lookahead_tokens": 0,
            "encoder_execution_arm": "eager-graphed",
            "decode_dispatch_arm": "dense-graphed",
            api.KEY: [N],
        },
        gpu_memory_utilization=0.3,
        # the restricted startup (TXG-CFG-017 and TXG-CFG-018): synchronous
        # scheduling and runner modes NONE, at core's default KV-cache sizing
        **q.restricted_startup(),
    )
    # A callable sent through collective_rpc is pickled, which core allows
    # only with insecure serialization on.
    return {
        "mode": "async",
        "settings": settings,
        "reads": ["counters"],
        "env": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
    }


#: This module's engine run (``support.engine_child``).
ENGINE_RUNS = {"multiprocess-reader": _multiprocess_reader_run}


@pytest.mark.installed_engine("multiprocess-reader")
# @spec TXG-OBS-008
def test_reader_runs_on_the_worker_under_a_multiprocess_engine(tmp_path):
    """``AsyncLLM`` (its engine core in another process) in a child process
    started before the session's tests (``support.engine_child``); the
    reader is sent to the worker through ``AsyncLLM.collective_rpc``."""
    from support.engine_child import reply

    result = reply("multiprocess-reader", _multiprocess_reader_run, tmp_path)
    assert result["ok"], (result.get("error") or {}).get("traceback", result)
    readings = result["reads"]["counters"]
    assert len(readings) == 1
    assert readings[0]["counters"]["captured_cells"] == 2 * N
