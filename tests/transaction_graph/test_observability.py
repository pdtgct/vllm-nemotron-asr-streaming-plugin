# SPDX-License-Identifier: Apache-2.0
"""The counter reader with transaction graphs off, and its reach
(TXG-OBS-002, TXG-OBS-009, TXG-OBS-010), on CPU with the real model.

The counters themselves, the capture record, the log lines, the per-engine
reset and the multiprocess reader (TXG-OBS-001, TXG-OBS-003 to TXG-OBS-008)
need transaction graphs on and are in ``test_observability_gpu.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from support import txg_api as api

pytestmark = pytest.mark.engine


@pytest.fixture
def off_model(tmp_path, monkeypatch):
    from support import model_harness as h

    h.spec_without_state_name(monkeypatch)
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
