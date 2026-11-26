# SPDX-License-Identifier: Apache-2.0
"""Graph memory (TXG-MEM-001 to TXG-MEM-013, TXG-MEM-017 to TXG-MEM-019).

CPU tests (``engine``) read memory through ``FakeCuda``: a completed cell
capture retains ``capture_cost`` bytes, the charged margin ``m +
margin_rounding``, and each free-memory reading is logged, so the record's
figures, the margin's retained charge from its two raw readings, the
aggregate serving-over-profiling guard, the required-profiling decision and
every failure message are checked against hand-computed values. Each test
names its KV-cache sizing under the restricted startup: a fixed
``--num-gpu-blocks-override`` (``support.txgcap_harness.CAPACITY``) with
core's estimate enabled where its subject includes the estimate or the
profiling capture (the HLD's choice for the checks that need the
estimate), core's default, the extensible KV cache (measured after the
captures), a profiled budget, or a fixed ``kv_cache_memory_bytes``.
``gpu`` tests read the real allocator.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import torch
from support import model_harness as h
from support import txg_conditions as c
from support import txgcap_harness as t
from support.fake_cuda import MARGIN_BYTES
from support.txg_api import NOT_APPLICABLE, SIZING_NAMES, Record, Sizing

pytestmark = pytest.mark.engine

MiB = 1 << 20
GEOMETRY_LABEL = re.compile(r"80\s*ms|geometry\s*(index\s*)?0\b", re.IGNORECASE)
ALLOC_CONF = "expandable_segments:True"


#: The KV-cache sizing of a test whose subject includes core's graph-memory
#: estimate or its profiling capture (module docstring).
ESTIMATE = {"sizing": Sizing.BLOCK_COUNT}


@pytest.fixture
def on_cpu(on_cpu, monkeypatch):
    """The shared ``on_cpu`` under a stated allocator configuration."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)
    return on_cpu


def profiled(run, *, capture_cost=MiB, rounding=0):
    env, core = run.env, run.core
    env.fake.capture_cost = capture_cost
    env.fake.margin_rounding = rounding
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    core.clear()
    core.bind_serving()
    return env, core


# ---------------------------------------------------------------------------
# Measurement


# @spec TXG-MEM-001
def test_every_reading_follows_a_synchronize_and_a_cache_release(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    mark = len(env.events)
    core.profiling_capture()
    names = [e[0] for e in env.events[mark:]]
    readings = [i for i, n in enumerate(names) if n == "cuda.mem_get_info"]
    assert readings, "the capture took no free-memory reading"
    previous = 0
    for index in readings:
        window = names[previous:index]
        syncs = [
            i
            for i, n in enumerate(window)
            if n in ("cuda.stream.synchronize", "cuda.synchronize")
        ]
        releases = [i for i, n in enumerate(window) if n == "cuda.empty_cache"]
        assert syncs and releases and max(releases) > max(syncs), (
            f"reading {index}: not after a synchronize then a cache release"
        )
        previous = index + 1


# @spec TXG-MEM-001
def test_transaction_graph_module_has_no_measurement_of_its_own():
    import vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph as txg

    source = Path(txg.__file__).read_text()
    for name in ("mem_get_info", "empty_cache"):
        assert name not in source, f"transaction_graph.py calls {name} itself"


@pytest.mark.parametrize("capture", [Record.PROFILING, Record.SERVING])
# @spec TXG-MEM-002
def test_record_holds_call_transaction_kind_cell_and_peak_figures(on_cpu, capture):
    run = on_cpu(n=3, max_num_seqs=3, **ESTIMATE)
    env, core = profiled(run, capture_cost=2 * MiB, rounding=MiB)
    if capture == Record.SERVING:
        core.serving_capture()
    rec = t.record(env)[capture]
    cells = rec[Record.CELLS]
    assert [(c["kind"], c["count"]) for c in cells] == t.largest_first(3)
    assert all(c["growth"] == 2 * MiB for c in cells), cells
    assert all(c["geometry"] == env.geometry for c in cells)
    assert rec[Record.KIND_TOTALS] == {"chunk": 3 * 2 * MiB, "echo": 3 * 2 * MiB}
    assert rec[Record.TXN_TOTAL] == 6 * 2 * MiB
    margin = MARGIN_BYTES + MiB if capture == Record.PROFILING else 0
    assert rec[Record.CALL_TOTAL] == rec[Record.TXN_TOTAL] + margin
    assert (
        isinstance(rec[Record.PEAK], int) and rec[Record.PEAK] > rec[Record.TXN_TOTAL]
    )


# @spec TXG-MEM-002
def test_transient_peak_resets_at_the_call_start(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = profiled(run)
    mark = len(env.events)
    core.serving_capture()
    names = [e[0] for e in env.events[mark:]]
    reset = names.index("cuda.reset_peak_memory_stats")
    work = [
        i for i, n in enumerate(names) if n in ("capture", "cuda.graph.capture_begin")
    ]
    assert reset < min(work), (
        "peak statistics were reset after the call's first capture"
    )
    assert (
        t.record(env)[Record.SERVING][Record.PEAK]
        == env.fake.transient_peak_since_reset()
    )


# @spec TXG-MEM-003
def test_record_holds_allocator_settings_and_versions(on_cpu, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run = on_cpu(n=2, **ESTIMATE)
    env, core = profiled(run)
    core.serving_capture()
    rec = t.record(env)
    assert rec[Record.ALLOC_CONF] == ALLOC_CONF
    assert rec[Record.CUBLAS] == ":4096:8"
    assert rec["torch_version"] == torch.__version__
    for key in Record.VERSIONS:
        assert key in rec, key


# ---------------------------------------------------------------------------
# The charged margin and the aggregate guard


@pytest.mark.parametrize("n", [1, 3])
# @spec TXG-MEM-004
def test_profiling_capture_ends_with_one_flat_margin_and_its_raw_charge(on_cpu, n):
    run = on_cpu(n=n, max_num_seqs=3, **ESTIMATE)
    env, core = run.env, run.core
    env.fake.jitter = 4096  # every reading is a distinct value
    env.fake.margin_rounding = 2 * MiB
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    assert env.fake.margin_allocations == 1
    assert env.fake.live_margins() == 1
    rec = t.record(env)
    assert rec[Record.MARGIN] == MARGIN_BYTES, "m depends on N or is not 64 MiB"
    before, after = rec[Record.FREE_BEFORE], rec[Record.FREE_AFTER]
    readings = env.fake.readings
    assert before in readings and after in readings
    assert readings.index(after) == readings.index(before) + 1, (
        "the margin's readings are not taken immediately around it"
    )
    assert rec[Record.CHARGE] == before - after == MARGIN_BYTES + 2 * MiB + 4096
    core.clear()
    assert env.fake.live_margins() == 0


# @spec TXG-MEM-005
def test_serving_over_profiling_fails_startup_with_both_totals(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = profiled(run, capture_cost=MiB, rounding=MiB)
    profiling = t.record(env)[Record.PROFILING][Record.CALL_TOTAL]
    env.fake.capture_cost = MiB + MARGIN_BYTES // 4 + MiB
    serving = 4 * env.fake.capture_cost
    error = h.call_with(core.serving_capture)
    assert error is not None, "a serving total above the profiling total was accepted"
    message = t.text(error)
    charge = MARGIN_BYTES + MiB
    for value in (serving, profiling, MARGIN_BYTES, charge, serving - profiling):
        assert t.mentions_bytes(message, value), (value, message)
    assert ALLOC_CONF in message
    assert "--kv-cache-memory-bytes" in message and t.OPTION in message


# @spec TXG-MEM-005
def test_serving_within_profiling_passes_and_no_guard_without_a_profiling_capture(
    on_cpu,
):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = profiled(run, rounding=MiB)
    core.serving_capture()  # equal cells: serving ≤ profiling − nothing
    fixed = on_cpu(n=2, kv_cache_memory_bytes=1 << 30)
    fixed.env.fake.capture_cost = 64 * MiB
    fixed.core.prepare()
    fixed.core.bind_serving()
    fixed.core.serving_capture()  # no profiling capture ran: nothing to compare


# @spec TXG-MEM-006
def test_record_keeps_profiling_figures_and_absorbed_rounding(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = run.env, run.core
    env.fake.margin_rounding = 3 * MiB
    core.prepare()
    core.bind_profiling()
    core.profiling_capture()
    kept = dict(t.record(env))
    core.clear()
    rec = t.record(env)
    for key in (Record.MARGIN, Record.FREE_BEFORE, Record.FREE_AFTER, Record.CHARGE):
        assert rec[key] == kept[key], key
    assert (
        rec[Record.PROFILING][Record.CALL_TOTAL]
        == kept[Record.PROFILING][Record.CALL_TOTAL]
    )
    core.bind_serving()
    core.serving_capture()
    rec = t.record(env)
    serving = rec[Record.SERVING][Record.CALL_TOTAL]
    profiling = rec[Record.PROFILING][Record.CALL_TOTAL]
    assert rec[Record.ABSORBED] == serving - (profiling - rec[Record.CHARGE])


def profiling_figures_not_applicable(rec):
    """The profiling-only fields of TXG-MEM-006, each marked not applicable
    (``support.txg_api.NOT_APPLICABLE``), never zero."""
    profiling = rec[Record.PROFILING]
    marked = {
        key: rec[key]
        for key in (
            Record.MARGIN,
            Record.FREE_BEFORE,
            Record.FREE_AFTER,
            Record.CHARGE,
            Record.ABSORBED,
        )
    }
    if profiling == NOT_APPLICABLE:
        marked[Record.PROFILING] = profiling
    else:
        for key in (Record.CALL_TOTAL, Record.TXN_TOTAL, Record.KIND_TOTALS):
            marked[f"{Record.PROFILING}.{key}"] = (profiling or {}).get(key)
    return {key: value for key, value in marked.items() if value != NOT_APPLICABLE}


@pytest.mark.parametrize("sizing", [Sizing.EXTENSIBLE, Sizing.BUDGET])
# @spec TXG-MEM-006, TXG-OBS-003
def test_with_no_profiling_capture_the_profiling_figures_are_not_applicable(
    on_cpu, sizing
):
    """Under the extensible KV cache and under a fixed budget no profiling
    capture runs: the profiling call's totals, ``m``, both margin readings,
    ``r`` and the absorbed rounding are marked not applicable, by a value no
    reader can take for a measurement, never zero."""
    budget = 1 << 30 if sizing == Sizing.BUDGET else None
    run = on_cpu(n=2, sizing=sizing, kv_cache_memory_bytes=budget)
    env, core = run.env, run.core
    core.startup()
    rec = t.record(env)
    assert rec[Record.RAN] is False
    assert profiling_figures_not_applicable(rec) == {}, rec


@pytest.mark.parametrize("sizing", [Sizing.BLOCK_COUNT, Sizing.EXTENSIBLE])
# @spec TXG-OBS-003
def test_the_record_holds_each_capture_layouts_page_count_and_the_final_count(
    on_cpu, sizing
):
    """The record holds the page count of each capture's state-group layout
    (the profiling KV cache's and the serving KV cache's, the reservation
    capacity under the extensible KV cache), the KV-cache sizing, and, under
    the extensible KV cache, the final bound count (the first dimension of
    the views reported at core's extension) as a separate field."""
    run = on_cpu(n=2, sizing=sizing)
    env, core = run.env, run.core
    core.startup(extend=False)
    if sizing == Sizing.EXTENSIBLE:
        core.extend(5)
    rec = t.record(env)
    assert rec[Record.SERVING][Record.PAGE_COUNT] == t.CAPACITY, rec[Record.SERVING]
    assert re.search(SIZING_NAMES[sizing], str(rec[Record.SIZING])), rec[Record.SIZING]
    if sizing == Sizing.BLOCK_COUNT:
        assert rec[Record.PROFILING][Record.PAGE_COUNT] == t.CAPACITY
        assert c.shows_number(str(rec[Record.SIZING]), t.CAPACITY), rec[Record.SIZING]
        assert rec[Record.FINAL_BOUND_COUNT] == NOT_APPLICABLE
    else:
        assert rec[Record.FINAL_BOUND_COUNT] == 5, rec[Record.FINAL_BOUND_COUNT]
        assert rec[Record.PROFILING] == NOT_APPLICABLE or (
            rec[Record.PROFILING].get(Record.PAGE_COUNT) == NOT_APPLICABLE
        ), rec[Record.PROFILING]


# ---------------------------------------------------------------------------
# Whether a profiling capture was required


# @spec TXG-MEM-007, TXG-MEM-008
def test_a_startup_plan_budget_is_an_expected_absence(on_cpu, caplog):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = run.env, run.core
    budget = 7 << 30
    core.prepare()
    t.set_cache(env.vllm_config, kv_cache_memory_bytes=budget)  # a persisted plan
    core.bind_serving()
    with caplog.at_level(logging.WARNING):
        core.serving_capture()
    rec = t.record(env)
    assert rec[Record.REQUIRED] is False
    assert rec[Record.EFFECTIVE]["kv_cache_memory_bytes"] == budget
    assert rec[Record.RAN] is False and rec[Record.REASON]
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name.startswith("vllm_nemotron_asr_streaming")
        and "not charged" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    message = warnings[0]
    # required information: the fixed value and that a startup plan set it
    assert "plan" in message.lower() and t.mentions_bytes(message, budget), message
    serving = rec[Record.SERVING]
    for value in (
        serving[Record.CALL_TOTAL],
        serving[Record.TXN_TOTAL],
        *serving[Record.KIND_TOTALS].values(),
    ):
        assert t.mentions_bytes(message, value), (value, message)
    assert "segment" in message.lower(), "the segment graphs are not named"


#: What TXG-MEM-018's INFO line states: core sizes the KV cache from the
#: device memory left after the captures.
MEASURED = re.compile(r"(?i)after the captures")


def _txg_records(caplog, level):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == level and r.name.startswith("vllm_nemotron_asr_streaming")
    ]


@pytest.mark.parametrize(
    "case", ["default", "startup-plan-budget", "estimate-disabled"]
)
# @spec TXG-MEM-018, TXG-MEM-007, TXG-MEM-008
def test_under_the_extensible_kv_cache_the_graph_memory_is_logged_as_measured(
    on_cpu, monkeypatch, caplog, case
):
    """With the effective extensible KV cache on, whatever the budget or
    estimate setting (a startup plan's ``kv_cache_memory_bytes`` set with the
    extensible KV cache left on, or the estimate disabled), no profiling
    capture ran and none was required: the serving capture records that
    reason, logs no uncharged-memory WARNING, and logs one INFO line that
    core sizes the KV cache from the memory left after the captures, which
    includes the transaction graphs and segment graphs, with the call,
    transaction and per-kind totals."""
    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    env, core = run.env, run.core
    if case == "estimate-disabled":
        t.set_estimate(monkeypatch, False)
    core.prepare()
    if case == "startup-plan-budget":
        t.set_cache(env.vllm_config, kv_cache_memory_bytes=7 << 30)
    core.bind_serving()
    with caplog.at_level(logging.INFO):
        core.serving_capture()
    rec = t.record(env)
    assert rec[Record.REQUIRED] is False and rec[Record.RAN] is False
    assert re.search(SIZING_NAMES[Sizing.EXTENSIBLE], str(rec[Record.REASON])), rec[
        Record.REASON
    ]
    warnings = [m for m in _txg_records(caplog, logging.WARNING) if "charged" in m]
    assert warnings == [], ("the measured case was logged as uncharged", warnings)
    measured = [
        m
        for m in _txg_records(caplog, logging.INFO)
        if MEASURED.search(m) and "segment" in m.lower()
    ]
    assert len(measured) == 1, _txg_records(caplog, logging.INFO)
    serving = rec[Record.SERVING]
    for value in (
        serving[Record.CALL_TOTAL],
        serving[Record.TXN_TOTAL],
        *serving[Record.KIND_TOTALS].values(),
    ):
        assert t.mentions_bytes(measured[0], value), (value, measured[0])


@pytest.mark.parametrize("sizing", [Sizing.BLOCK_COUNT, Sizing.PROFILED])
# @spec TXG-MEM-008, TXG-MEM-007
def test_with_the_extensible_kv_cache_off_a_disabled_estimate_is_uncharged(
    on_cpu, monkeypatch, caplog, sizing
):
    """With the extensible KV cache off and the estimate disabled, under a
    fixed block count or a profiled budget, no profiling capture was
    required, and the serving capture logs the uncharged-memory WARNING with
    the estimate's setting, and no measured INFO line."""
    run = on_cpu(n=2, sizing=sizing)
    env, core = run.env, run.core
    t.set_estimate(monkeypatch, False)
    core.prepare()
    core.bind_serving()
    with caplog.at_level(logging.INFO):
        core.serving_capture()
    assert t.record(env)[Record.REQUIRED] is False
    warnings = [
        m
        for m in _txg_records(caplog, logging.WARNING)
        if "not charged" in m and "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in m
    ]
    assert len(warnings) == 1, _txg_records(caplog, logging.WARNING)
    measured = [m for m in _txg_records(caplog, logging.INFO) if MEASURED.search(m)]
    assert measured == [], measured


# @spec TXG-MEM-007, TXG-MEM-009
def test_an_extensible_cache_disabled_after_construction_requires_profiling(on_cpu):
    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    env, core = run.env, run.core
    core.prepare()
    t.set_cache(env.vllm_config, enable_extensible_kv_cache=False)
    core.bind_serving()
    mark = len(env.events)
    error = h.call_with(core.serving_capture)
    assert error is not None, "a required profiling capture was absent and accepted"
    assert t.captures_since(env, mark) == [], "the refused call captured a graph"
    message = t.text(error)
    assert t.OPTION in message and "estimate" in message.lower(), message


@pytest.mark.parametrize("estimate", [True, False])
# @spec TXG-MEM-007, TXG-MEM-008
def test_the_estimate_setting_decides_the_requirement(
    on_cpu, monkeypatch, caplog, estimate
):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = run.env, run.core
    t.set_estimate(monkeypatch, estimate)
    core.prepare()
    core.bind_serving()
    mark = len(env.events)
    with caplog.at_level(logging.WARNING):
        error = h.call_with(core.serving_capture)
    if estimate:
        # required (estimate on, budget profiled, extensible cache off) and
        # absent: TXG-MEM-009's refusal at entry, nothing captured
        assert error is not None, "the estimate was enabled and did not run"
        message = t.text(error)
        assert t.OPTION in message and "estimate" in message.lower(), message
        assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in message or (
            "kv_cache_memory_bytes" in message or "kv-cache-memory-bytes" in message
        ), "the error names no configuration that should have run the estimate"
        assert t.captures_since(env, mark) == [], "the refused call captured a graph"
        # positive control: this environment captures once the required
        # profiling capture has run, so the refusal above was the requirement
        core.bind_profiling()
        core.profiling_capture()
        core.clear()
        core.bind_serving(core.serving_bindings)
        mark = len(env.events)
        core.serving_capture()
        assert t.captures_since(env, mark), "the continued serving call captured none"
        rec = t.record(env)
        assert rec[Record.REQUIRED] is True and rec[Record.RAN] is True
        assert t.counter(env, "captured_cells") == 2 * env.n
        assert t.fake_graph_count(env) == 2 * env.n
        return
    assert error is None, error
    rec = t.record(env)
    assert rec[Record.REQUIRED] is False
    assert any(
        "not charged" in r.getMessage()
        and "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# @spec TXG-MEM-007
def test_a_profiled_startup_records_the_requirement_and_its_values(on_cpu):
    run = on_cpu(n=2, **ESTIMATE)
    env, core = profiled(run)
    core.serving_capture()
    rec = t.record(env)
    assert rec[Record.REQUIRED] is True and rec[Record.RAN] is True
    effective = rec[Record.EFFECTIVE]
    assert effective["kv_cache_memory_bytes"] in (None, 0)
    assert effective["enable_extensible_kv_cache"] is False
    assert effective["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] in (True, 1, "1")


# ---------------------------------------------------------------------------
# Failure messages


PHASES = {
    "binding-buffer allocation": ("pinned", None, False),
    "scatter warm-up": ("warmup", None, False),
    "capture": ("capture", 1, True),
    "validation replay": ("replay", 1, True),
    "margin allocation": ("margin", None, False),
}


def _inject(env, phase):
    what, offset, _ = PHASES[phase]
    fake = env.fake
    if what == "pinned":
        fake.fail["pinned"] = fake.pinned_requests
    elif what == "warmup":
        env.scatter.fail_warmup = env.scatter.warmups
    elif what == "capture":
        fake.fail["capture"] = fake.capture_attempts + offset
    elif what == "replay":
        fake.fail["replay"] = fake.replay_count + offset
    elif what == "margin":
        fake.fail["margin"] = fake.margin_allocations


def _failure(run, *, capture, phase, budget=None):
    env, core = run.env, run.core
    core.prepare()
    if capture == "profiling":
        core.bind_profiling()
    else:
        if core.estimate_runs():
            core.bind_profiling()
            core.profiling_capture()
            core.clear()
        core.bind_serving()
    _inject(env, phase)
    peak_holder = {}
    error = h.call_with(core.capture, profile_only=capture == "profiling")
    assert error is not None, f"no failure from the injected {phase} fault"
    assert env.fake.peak_resets, "peak statistics were not reset at the call's start"
    peak_holder["peak"] = env.fake.peak_reserved - env.fake.peak_resets[-1]
    return error, peak_holder["peak"]


@pytest.mark.parametrize("phase", list(PHASES))
@pytest.mark.parametrize("capture", ["profiling", "serving"])
# @spec TXG-MEM-010
def test_a_transaction_graph_part_failure_names_its_phase_and_figures(
    on_cpu, capture, phase
):
    if phase == "margin allocation" and capture == "serving":
        pytest.skip("the serving capture allocates no margin")
    run = on_cpu(n=2, **ESTIMATE)
    error, peak = _failure(run, capture=capture, phase=phase)
    assert any("injected fault" in str(e) for e in t.chain(error)[1:]), (
        "the error is not chained from the original"
    )
    message = str(error)
    assert t.OPTION in message and "[2]" in message, message
    assert re.search(r"\bN\b\D{0,4}2\b", message), message
    assert capture in message.lower(), message
    assert phase in message.lower(), message
    cell_phase = PHASES[phase][2]
    if cell_phase:
        # the second cell captured at N = 2 is the chunk cell at 1
        assert GEOMETRY_LABEL.search(message), message
        assert re.search(r"chunk\D{0,24}\b1\b", message), message
    else:
        assert not re.search(r"\b(chunk|echo) cell\b", message), message
    assert t.mentions_bytes(message, peak), (peak, message)
    assert ALLOC_CONF in message
    # The budget setting: a block count fixed at the given
    # num_gpu_blocks_override, with core's estimate enabled.
    assert re.search(r"num_gpu_blocks_override|--num-gpu-blocks-override", message)
    assert c.shows_number(message, t.CAPACITY), message
    assert re.search(
        r"(?i)estimate[^;\n]{0,40}enabled|enabled[^;\n]{0,40}estimate", message
    ), message


@pytest.mark.parametrize("sizing", [Sizing.EXTENSIBLE, Sizing.PROFILED, Sizing.BUDGET])
# @spec TXG-MEM-010
def test_a_serving_capture_failure_names_the_kv_cache_sizing(on_cpu, sizing):
    """The error names the KV-cache sizing: the extensible KV cache with its
    page count; a profiled budget with the extensible KV cache off, with the
    estimate enabled; or a budget fixed at the given
    ``kv_cache_memory_bytes``."""
    budget = 9 << 30 if sizing == Sizing.BUDGET else None
    run = on_cpu(n=2, sizing=sizing, kv_cache_memory_bytes=budget)
    error, _ = _failure(run, capture="serving", phase="capture")
    message = str(error)
    assert re.search(SIZING_NAMES[sizing], message), message
    if sizing == Sizing.EXTENSIBLE:
        assert c.shows_number(message, t.CAPACITY), message
    elif sizing == Sizing.PROFILED:
        assert re.search(r"(?i)estimate[^;\n]{0,40}enabled", message), message
    else:
        assert t.mentions_bytes(message, budget), message


def assert_profiling_remedies(message):
    """TXG-MEM-011: a smaller largest population, a smaller ``max_num_seqs``,
    or transaction graphs off."""
    assert c.found(c.SMALLER_POPULATION, message), message
    assert re.search(r"--max-num-seqs|max_num_seqs", message), message
    assert c.found(c.OPTION_OFF, message) or "off" in message.lower(), message


# @spec TXG-MEM-011, TXG-MEM-010
def test_profiling_failure_remedies(on_cpu):
    """A profiling capture that fails after pool access gives TXG-MEM-011's
    remedies, and none of TXG-MEM-017's larger budget or block count."""
    error, _ = _failure(on_cpu(n=2, **ESTIMATE), capture="profiling", phase="capture")
    message = str(error)
    assert t.OPTION in message
    assert_profiling_remedies(message)
    assert not c.found(c.RAISES_BUDGET, message), message


# @spec TXG-MEM-012, TXG-MEM-010
def test_serving_failure_under_a_fixed_block_count_remedies(on_cpu):
    """Under a fixed ``--num-gpu-blocks-override``, a serving capture that
    fails after pool access gives TXG-MEM-011's remedies or a smaller
    override that still holds the admission limit plus one blocks (here
    ``max_num_seqs`` 2, so 3), stating that it helps only when the state
    pages leave too little memory for the transient peak; no
    ``--gpu-memory-utilization`` (the budget is not profiled) and no
    ``--kv-cache-memory-bytes`` remedy (TXG-MEM-013's case)."""
    run = on_cpu(n=2, **ESTIMATE)
    error, _ = _failure(run, capture="serving", phase="capture")
    message = str(error)
    assert t.OPTION in message
    assert_profiling_remedies(message)
    assert re.search(
        r"(?i)(smaller|lower)[^;\n]{0,40}?--num-gpu-blocks-override", message
    ), message
    assert c.shows_number(message, run.env.admission_limit + 1), message
    assert "transient peak" in message.lower(), message
    assert "--gpu-memory-utilization" not in message, message
    assert not c.found(c.KV_BYTES, message), message


#: A lower ``--gpu-memory-utilization`` remedy (TXG-MEM-019).
LOWER_UTILIZATION = re.compile(
    r"(?i)(lower|smaller|reduce)[^;\n]{0,40}--gpu-memory-utilization"
)


@pytest.mark.parametrize("sizing", [Sizing.PROFILED, Sizing.EXTENSIBLE])
# @spec TXG-MEM-019, TXG-MEM-010
def test_serving_failure_under_a_profiled_budget_or_the_extensible_cache_remedies(
    on_cpu, sizing
):
    """A serving capture that fails after pool access under a profiled
    budget or the extensible KV cache gives TXG-MEM-011's remedies; under a
    profiled budget also a lower ``--gpu-memory-utilization``, stating that
    it helps only when the allocated state pages leave too little device
    memory for the capture's transient peak. Under the extensible KV cache
    the capture precedes core's sizing, so no such remedy; neither sizing
    gives a block-count or budget remedy."""
    run = on_cpu(n=2, sizing=sizing)
    error, _ = _failure(run, capture="serving", phase="capture")
    message = str(error)
    assert t.OPTION in message
    assert_profiling_remedies(message)
    if sizing == Sizing.PROFILED:
        assert LOWER_UTILIZATION.search(message), message
        assert "transient peak" in message.lower(), message
    else:
        assert not LOWER_UTILIZATION.search(message), message
    assert not c.found(c.BLOCKS_OVERRIDE, message), message
    assert not c.found(c.KV_BYTES, message), message


@pytest.mark.parametrize(
    "fail_at,remainder,unavailable",
    [
        # N = 3: chunk 3 captured, fails at chunk 2; chunk mean × 2 left; echo none
        (1, 2, True),
        # chunk 3, 2, 1 and echo 3 captured, fails at echo 2; echo mean × 2 left
        (4, 2, False),
    ],
    ids=["chunk-cell", "echo-cell"],
)
# @spec TXG-MEM-013
def test_serving_failure_under_a_fixed_budget_projects_the_remainder(
    on_cpu, fail_at, remainder, unavailable
):
    budget = 9 << 30
    run = on_cpu(n=3, max_num_seqs=3, kv_cache_memory_bytes=budget)
    env, core = run.env, run.core
    growth = 5 * MiB
    env.fake.capture_cost = growth
    core.prepare()
    core.bind_serving()
    env.fake.fail["capture"] = env.fake.capture_attempts + fail_at
    error = h.call_with(core.serving_capture)
    assert error is not None
    message = str(error)
    assert "--kv-cache-memory-bytes" in message and "estimate" in message.lower()
    assert t.mentions_bytes(message, remainder * growth), message
    if unavailable:
        assert "unavailable" in message.lower(), message
    assert "fixed" in message.lower() and t.mentions_bytes(message, budget), message
    assert re.search(r"--max-num-seqs|max_num_seqs", message)
    # TXG-MEM-010's routing: not TXG-MEM-012's block-count remedy
    assert not c.found(c.BLOCKS_OVERRIDE, message), message


@pytest.mark.parametrize("sizing", [Sizing.BUDGET, Sizing.PROFILED, Sizing.EXTENSIBLE])
# @spec TXG-MEM-017, TXG-MEM-010, TXG-CAP-008
def test_too_few_pages_fail_the_serving_pool_access_with_a_larger_budget_or_lower_limit(
    on_cpu, sizing
):
    """The admission limit is ``max_num_seqs`` 4, so the serving capture's
    pool access needs five pages; the serving layout holds three. Under a
    fixed ``--kv-cache-memory-bytes``, a profiled budget or the extensible
    KV cache (whose layout's page count is the reservation capacity) no
    configuration check knows the count. The pool-access failure names the
    page count and the layouts it received, and its remedies are a larger
    budget or block count (under a profiled budget or the extensible KV
    cache, a larger ``--gpu-memory-utilization``), or a lower admission
    limit, never a smaller budget or block count. The profiling capture
    makes no page-count check (``test_capture.py``)."""
    budget = 1 << 30 if sizing == Sizing.BUDGET else None
    run = on_cpu(n=2, max_num_seqs=4, sizing=sizing, kv_cache_memory_bytes=budget)
    env, core = run.env, run.core
    small = h.real_bindings(h.model_spec(env.model), capacity=3)
    core.prepare()
    if core.estimate_runs():
        core.bind_profiling()
        core.profiling_capture()
        core.clear()
    core.bind_serving(small)
    error = h.call_with(core.serving_capture)
    assert error is not None, "a pool of too few pages passed pool access"
    message, full = str(error), t.text(error)
    assert t.OPTION in message and "serving" in message.lower(), message
    assert "pool access" in message.lower(), message
    assert not re.search(r"\b(chunk|echo) cell\b", message), message
    assert "layout" in full.lower() and c.shows_number(full, 3), full
    assert c.shows_number(full, 4) or c.shows_number(full, 5), full
    assert c.found(c.RAISES_BUDGET, message), message
    assert c.lower_admission_limit(message), message
    assert not c.found(c.LOWERS_BUDGET, message), (
        "a pool-access block-count failure gave a lowering remedy",
        message,
    )
    if sizing != Sizing.BUDGET:
        assert re.search(
            r"(?i)(larger|higher|raise)[^;\n]{0,40}--gpu-memory-utilization", message
        ), message
    assert t.fake_graph_count(env) == 0, "a cell was captured past pool access"


# ---------------------------------------------------------------------------
# CUDA


@pytest.mark.gpu
# @spec TXG-MEM-002, TXG-MEM-004, TXG-MEM-006
def test_capture_record_figures_are_consistent_on_cuda(on_cuda):
    env = on_cuda(n=2, sizing=Sizing.BLOCK_COUNT)
    core = env.core
    core.startup()
    rec = t.record(env)
    for capture in (Record.PROFILING, Record.SERVING):
        figures = rec[capture]
        assert len(figures[Record.CELLS]) == 2 * env.n
        assert figures[Record.TXN_TOTAL] <= figures[Record.CALL_TOTAL]
        assert sum(figures[Record.KIND_TOTALS].values()) <= figures[Record.TXN_TOTAL]
        assert figures[Record.PEAK] >= 0
    assert rec[Record.MARGIN] == MARGIN_BYTES
    assert rec[Record.CHARGE] == rec[Record.FREE_BEFORE] - rec[Record.FREE_AFTER] > 0
    assert (
        rec[Record.SERVING][Record.CALL_TOTAL]
        <= rec[Record.PROFILING][Record.CALL_TOTAL]
    )
    assert rec[Record.ABSORBED] == rec[Record.SERVING][Record.CALL_TOTAL] - (
        rec[Record.PROFILING][Record.CALL_TOTAL] - rec[Record.CHARGE]
    )


@pytest.mark.gpu
# @spec TXG-MEM-001
def test_measurement_readings_follow_synchronize_and_cache_release_cuda(
    on_cuda, monkeypatch
):
    log = []
    for name in ("mem_get_info", "empty_cache", "synchronize"):
        original = getattr(torch.cuda, name)

        def spy(*args, __original=original, __name=name, **kwargs):
            log.append(__name)
            return __original(*args, **kwargs)

        monkeypatch.setattr(torch.cuda, name, spy)
    original_sync = torch.cuda.Stream.synchronize

    def stream_sync(self):
        log.append("synchronize")
        return original_sync(self)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", stream_sync)
    env = on_cuda(n=2, sizing=Sizing.BLOCK_COUNT)
    env.core.prepare()
    env.core.bind_profiling()
    log.clear()
    env.core.profiling_capture()
    previous = 0
    for index, name in enumerate(log):
        if name != "mem_get_info":
            continue
        window = log[previous:index]
        assert "synchronize" in window and "empty_cache" in window, window
        previous = index + 1
