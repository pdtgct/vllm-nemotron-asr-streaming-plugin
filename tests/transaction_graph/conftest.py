# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the transaction-graph tests (``TXG`` specs).

The interface these tests assume is documented once, in ``support.txg_api``.

Generic fixtures, usable by every transaction-graph test module:

* ``repo_root``: the repository root.
* ``hf_config`` / ``engine_config``: factories of served and engine
  configuration doubles (``support.txg_fakes``), in the attribute shape
  ``transaction_graph_config_errors`` reads.
* ``plan_builder``: ``plan_builder("ccox", geometry=0, fresh=False,
  file=False)`` or ``plan_builder([Row(...), ...])`` builds a CPU
  ``RowPlan`` (``c`` CHUNK, ``o`` other, ``x`` planning failed).
* ``cell_inventory``: ``cell_inventory(n, geometry_index, missing=())``
  builds ``2N`` fake cells (``FakeCell`` with a ``FakeGraph`` whose
  ``replay()`` logs to ``cell_events``); ``has_graph_of(cells)`` gives the
  callable ``classify`` takes.
* ``recording_sink``: a ``RecordingSink`` commit sink.
* ``txn_env``: ``txn_env(lookahead, device="cpu")`` -> a
  ``support.txg_transaction.TransactionEnv`` (``engine`` tests only): row
  fixtures from real steps of the runnable synthetic card, composed steps,
  and the option-off transaction on cloned pools. Built once per
  ``(lookahead, device)`` per session.
* ``on_cpu``: ``on_cpu(n=2, ...)`` -> ``SimpleNamespace(env, core)``: an
  option-on model state on CPU through the fake-capture seam
  (``support.txgcap_harness.on_cpu``) and its ``support.core_order.CoreOrder``,
  not started.
* ``off_cpu``: ``off_cpu(...)`` -> the option-off environment
  (``support.txgcap_harness.build_off``).
* ``on_cuda``: ``on_cuda(lookahead=0, n=..., startup=False, ...)`` -> a
  ``support.txg_gpu.GpuEnv`` (``gpu`` tests only): the model and model state
  on CUDA with transaction graphs on; its ``core`` takes it through core's
  startup order (``startup=True`` runs it). Every environment built is shut
  down after the test.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: an exhaustive case the tier record requires but a developer loop "
        'may deselect (-m "not slow")',
    )
    config.addinivalue_line(
        "markers",
        "installed_engine(name, ...): the test reads the named engine runs of its "
        "module's ENGINE_RUNS, started in child processes before any test runs "
        "(support.engine_child)",
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Start the installed-engine runs the selected, unskipped tests name,
    one at a time, before any test runs (``support.engine_child``): the
    in-process CUDA tests leave device memory allocated in this process, and
    each engine needs its fraction of the device free."""
    if session.config.option.collectonly:
        return
    runs = []
    for item in session.items:
        marker = item.get_closest_marker("installed_engine")
        if marker is None or item.get_closest_marker("skip") is not None:
            continue
        table = getattr(item.module, "ENGINE_RUNS", {})
        runs.extend((name, table[name]) for name in marker.args)
    if runs:
        from support.engine_child import prestart

        prestart(runs)


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def hf_config():
    from support.txg_fakes import hf_config as make

    return make


@pytest.fixture
def engine_config():
    from support.txg_fakes import engine_config as make

    return make


@pytest.fixture
def plan_builder():
    from support.txg_fakes import Row, row_plan, rows

    def build(spec, **common):
        planned = common.pop("planned_rejection", None)
        items = rows(spec, **common) if isinstance(spec, str) else list(spec)
        assert all(isinstance(r, Row) for r in items)
        return row_plan(items, planned_rejection=planned)

    return build


@pytest.fixture
def cell_events() -> list:
    return []


@pytest.fixture
def cell_inventory(cell_events):
    from support.txg_fakes import fake_inventory

    def build(n, geometry_index=0, missing=()):
        return fake_inventory(n, geometry_index, missing=missing, events=cell_events)

    return build


@pytest.fixture
def recording_sink():
    from support.txg_fakes import RecordingSink

    return RecordingSink()


_ENVS: dict = {}


@pytest.fixture(scope="session")
def _txn_env_cache(tmp_path_factory):
    def get(lookahead: int, device: str = "cpu"):
        key = (lookahead, device)
        if key not in _ENVS:
            from support.txg_transaction import build_env

            _ENVS[key] = build_env(
                tmp_path_factory.mktemp(f"txg-env-{lookahead}-{device}"),
                lookahead,
                device=device,
            )
        return _ENVS[key]

    return get


@pytest.fixture
def txn_env(_txn_env_cache):
    return _txn_env_cache


@pytest.fixture
def on_cpu(monkeypatch, tmp_path):
    from support import txgcap_harness as t

    return functools.partial(t.on_cpu, monkeypatch, tmp_path)


@pytest.fixture
def off_cpu(monkeypatch, tmp_path):
    from support import txgcap_harness as t

    return functools.partial(t.build_off, monkeypatch, tmp_path)


@pytest.fixture
def on_cuda(monkeypatch, tmp_path, single_process_dist):
    from support.txg_gpu import build_option_on

    built = []

    def build(*, startup: bool = False, **kwargs):
        env = build_option_on(
            tmp_path / f"engine-{len(built)}",
            startup=startup,
            monkeypatch=monkeypatch,
            **kwargs,
        )
        built.append(env)
        return env

    yield build
    for env in built:
        env.shutdown()
