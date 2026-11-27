# SPDX-License-Identifier: Apache-2.0
"""The counter reader with transaction graphs off, and its reach
(TXG-OBS-002, TXG-OBS-009, TXG-OBS-010), on CPU with the real model.

The counters themselves, the capture record, the log lines, the per-engine
reset and the multiprocess reader (TXG-OBS-001, TXG-OBS-003 to TXG-OBS-008)
need transaction graphs on and are in ``test_observability_gpu.py``; one CPU
check here, through the fake-capture seam, keeps the capture call's INFO line
(TXG-OBS-004) distinguishable from the serving call's measured INFO line
(TXG-MEM-018) by the needles the GPU test counts.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
from support import txg_api as api
from support.txg_api import Sizing

pytestmark = pytest.mark.engine


@pytest.fixture
def off_model(tmp_path):
    from support import model_harness as h

    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0)
    return h.build_model(config)


def holders(model):
    """The model as core's in-process objects hold it: runner, worker and
    engine core (``EngineCore.model_executor.driver_worker.worker``)."""
    runner = SimpleNamespace(model=model)
    worker = SimpleNamespace(model_runner=runner)
    wrapper = SimpleNamespace(worker=worker)
    engine_core = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=wrapper))
    return {
        "model": model,
        "runner": runner,
        "worker": worker,
        "engine_core": engine_core,
    }


# @spec TXG-OBS-009
def test_counters_are_empty_while_off(off_model):
    assert not api.transaction_graphs_on(off_model.config)
    assert dict(api.counters(off_model)) == {}


# @spec TXG-OBS-002, TXG-OBS-009
@pytest.mark.parametrize("holder", ["model", "runner", "worker", "engine_core"])
def test_reader_reaches_the_model_through_each_holder(off_model, holder):
    assert dict(api.counters(holders(off_model)[holder])) == {}


# @spec TXG-OBS-010
@pytest.mark.parametrize(
    "unreachable",
    [
        object(),
        SimpleNamespace(),
        SimpleNamespace(model_runner=SimpleNamespace()),
        SimpleNamespace(model_executor=SimpleNamespace(driver_worker=None)),
        SimpleNamespace(model=object()),
    ],
    ids=[
        "object",
        "empty",
        "runner-without-model",
        "executor-without-worker",
        "foreign-model",
    ],
)
def test_an_unreachable_model_raises(unreachable):
    with pytest.raises(Exception) as caught:
        api.counters(unreachable)
    assert not isinstance(caught.value, (AssertionError, ImportError))


def _info_lines(caplog, *needles):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
        and r.name.startswith("vllm_nemotron_asr_streaming")
        and all(n.lower() in r.getMessage().lower() for n in needles)
    ]


# @spec TXG-OBS-004, TXG-MEM-018
def test_the_capture_line_is_distinguishable_from_the_measured_line(on_cpu, caplog):
    """Under the extensible KV cache the serving call logs two INFO lines,
    both naming the serving capture and the setting: the capture call's
    (TXG-OBS-004) and the measured one (TXG-MEM-018). The OBS-004 needles of
    ``test_observability_gpu.py`` (the cell count, the transient peak and the
    capture time) match the first alone."""
    n = 2
    run = on_cpu(n=n, sizing=Sizing.EXTENSIBLE)
    run.core.prepare()
    run.core.bind_serving()
    with caplog.at_level(logging.INFO):
        run.core.serving_capture()
    serving = _info_lines(caplog, "serving", api.KEY)
    measured = [m for m in serving if re.search(r"(?i)after the captures", m)]
    assert len(measured) == 1, serving
    needles = ("serving", f"{2 * n} cells", "transient peak", "capture time")
    captured = _info_lines(caplog, *needles, api.KEY)
    assert len(captured) == 1, serving
    assert captured[0] not in measured, captured


# @spec TXG-OBS-003
def test_without_core_rebinding_the_final_count_is_the_bound_views_page_count(
    on_cpu,
):
    """Under the extensible KV cache with a layout that is not block-compact
    core's extension binds nothing, so the views the serving capture read
    stay bound: the final bound count is their page count, the reservation
    capacity, not an unset field."""
    from support import txgcap_harness as t

    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    run.core.startup(extend=False)
    rec = t.record(run.env)
    assert rec[api.Record.FINAL_BOUND_COUNT] == t.CAPACITY, rec


# @spec TXG-OBS-003
def test_the_capture_record_crosses_core_s_message_serializer(on_cpu):
    """The capture record holds plain values only, so a worker can return it
    through core's ``collective_rpc``: core's msgpack encoder, which refuses
    a ``str`` subclass such as ``torch.__version__``'s type, encodes it."""
    from support import txgcap_harness as t
    from vllm.v1.serial_utils import MsgpackEncoder

    run = on_cpu(n=2, sizing=Sizing.EXTENSIBLE)
    run.core.startup()
    rec = t.record(run.env)
    MsgpackEncoder().encode(dict(rec))
    plain = (type(None), bool, int, float, str, list, dict)
    stack = [rec]
    while stack:
        value = stack.pop()
        assert type(value) in plain or isinstance(value, tuple), (type(value), value)
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
