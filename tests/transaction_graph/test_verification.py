# SPDX-License-Identifier: Apache-2.0
"""Post-warmup verification (TXG-VER-001 to TXG-VER-003).

CPU tests (``engine``) perturb one recorded quantity at a time after the
serving capture, through the fake-capture seam, and require ``post_warmup``
to fail, latch and report the first failing check in TXG-VER-001's order.
Perturbations use public objects (the state layer's currently bound views,
``component_pools``, the cells' graph objects, a model parameter's storage,
the per-operation precision setting)
and one model-internal seam, the dense decode binding's per-key callable
cache (``_decode_fns``), for the callable-identity check. The projected
resident layout per role (check 2) is exercised only through its effect in
``gpu`` tests: no public operation swaps two same-shaped roles. The tests
run under core's default KV-cache sizing, the extensible KV cache, whose
serving capture is the only capture.
"""

from __future__ import annotations

import pytest
import torch
from support import model_harness as h
from support import txgcap_harness as t

pytestmark = pytest.mark.engine


def captured(run):
    run.core.startup(verify=False)
    return run.env, run.core


def perturb_layout(monkeypatch, env):
    """The state layer's currently bound views (``component_pools``) no
    longer have the plain-value layout the serving capture recorded: one
    component is a copy at another address, as a rebinding the binding never
    saw would leave it."""
    layer = t.state_layer_of(env)
    pools = dict(layer.component_pools)
    name = next(iter(pools))
    pools[name] = pools[name].clone()
    if isinstance(getattr(type(layer), "component_pools", None), property):
        monkeypatch.setattr(
            type(layer), "component_pools", property(lambda self: pools)
        )
    else:
        monkeypatch.setattr(layer, "component_pools", pools)


def _set_component_pools(monkeypatch, layer, pools):
    if isinstance(getattr(type(layer), "component_pools", None), property):
        monkeypatch.setattr(
            type(layer), "component_pools", property(lambda self: pools)
        )
    else:
        monkeypatch.setattr(layer, "component_pools", pools)


def views_of(env, pages):
    """The state layer's bound views narrowed, or widened over the same
    storage, to ``pages`` pages: the same address, page shape, strides and
    dtype, another first dimension."""
    layer = t.state_layer_of(env)
    views = {}
    for name, view in layer.component_pools.items():
        views[name] = view.as_strided(
            (pages, *view.shape[1:]), view.stride(), view.storage_offset()
        )
    return layer, views


def perturb_cells(monkeypatch, env):
    """A cell that lost its graph: the cell's ``graph`` is ``None``."""
    from support import txg_api

    cells = txg_api.binding_of(env.model).cells
    monkeypatch.setattr(cells[next(iter(cells))], "graph", None)


def perturb_callable(monkeypatch, env):
    cache = env.model._decode_graph_binding._decode_fns
    key = next(iter(cache))
    original = cache[key]
    monkeypatch.setitem(cache, key, lambda *a, **k: original(*a, **k))


def perturb_buffer(monkeypatch, env):
    parameter = next(env.model.parameters())
    parameter.data = parameter.data.clone()


def perturb_constant(monkeypatch, env):
    matmul = torch.backends.cuda.matmul
    current = matmul.fp32_precision
    # A plain assignment, restored by ``_restore_matmul_precision``:
    # monkeypatch's undo deletes an attribute that torch serves through
    # ``__getattr__``, and torch refuses the deletion.
    matmul.fp32_precision = "tf32" if current != "tf32" else "ieee"


@pytest.fixture(autouse=True)
def _restore_matmul_precision():
    matmul = torch.backends.cuda.matmul
    saved = matmul.fp32_precision
    yield
    matmul.fp32_precision = saved


PERTURBATIONS = {
    "pool-layout": (perturb_layout, ("layout",)),
    "cell-graph": (perturb_cells, ("graph",)),
    "callable-identity": (perturb_callable, ("callable",)),
    "buffer-address": (perturb_buffer, ("address", "buffer")),
    "process-constant": (perturb_constant, ("fp32_precision",)),
}


@pytest.mark.parametrize("which", list(PERTURBATIONS))
# @spec TXG-VER-001, TXG-VER-002
def test_each_recorded_quantity_is_verified_and_a_mismatch_latches(
    on_cpu, monkeypatch, which
):
    env, core = captured(on_cpu(n=2))
    perturb, _ = PERTURBATIONS[which]
    perturb(monkeypatch, env)
    error = h.call_with(core.post_warmup)
    assert error is not None, f"post_warmup accepted a perturbed {which}"
    later = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


@pytest.mark.parametrize("first", ["pool-layout", "cell-graph", "buffer-address"])
# @spec TXG-VER-001
def test_checks_run_in_order_and_report_the_first_mismatch(on_cpu, monkeypatch, first):
    env, core = captured(on_cpu(n=2))
    PERTURBATIONS[first][0](monkeypatch, env)
    perturb_constant(monkeypatch, env)  # the last check also fails
    error = h.call_with(core.post_warmup)
    assert error is not None
    # The order of TXG-VER-001 is observable only through which comparison
    # the error reports, so this test needs the error to name the quantity
    # that failed (any wording containing one of the listed terms).
    message = t.text(error).lower()
    assert any(w in message for w in PERTURBATIONS[first][1]), message
    assert "fp32_precision" not in message, "a later check reported first"


# @spec TXG-VER-001
def test_bound_views_of_fewer_pages_over_the_captured_layout_verify(
    on_cpu, monkeypatch
):
    """Views with the recorded address, page shape, strides and dtype and a
    first dimension below the recorded page count fit the layout the serving
    capture recorded, as the views core binds when its extensible KV cache
    grows do: verification passes and the worker serves."""
    env, core = captured(on_cpu(n=2))
    layer, views = views_of(env, t.CAPACITY - 3)
    _set_component_pools(monkeypatch, layer, views)
    error = h.call_with(core.post_warmup)
    assert error is None, ("views that fit failed verification", t.text(error))


# @spec TXG-VER-001, TXG-VER-002
def test_bound_views_of_more_pages_than_recorded_fail_verification(on_cpu, monkeypatch):
    """Views with more pages than the recorded page count do not fit:
    verification fails, names the pool layout and latches."""
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.prepare()
    core.bind_serving(t.narrow_pool(core.bindings(t.CAPACITY + 2), t.CAPACITY))
    core.serving_capture()
    layer, views = views_of(env, t.CAPACITY + 2)
    _set_component_pools(monkeypatch, layer, views)
    error = h.call_with(core.post_warmup)
    assert error is not None, "views of more pages than recorded verified"
    assert "layout" in t.text(error).lower(), t.text(error)
    later = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


# @spec TXG-VER-001
def test_an_unperturbed_startup_verifies(on_cpu):
    env, core = captured(on_cpu(n=2))
    core.post_warmup()
    core.step(t.stream_unit(env, "s1"))


# @spec TXG-VER-003
def test_verification_replays_and_allocates_nothing(on_cpu):
    env, core = captured(on_cpu(n=2))
    replays, graphs = env.fake.replay_count, len(env.fake.graphs)
    pinned, margins = env.fake.pinned_allocations, env.fake.margin_allocations
    mark = len(env.events)
    core.post_warmup()
    assert env.fake.replay_count == replays, "post_warmup replayed a graph"
    assert len(env.fake.graphs) == graphs and env.fake.pinned_allocations == pinned
    assert env.fake.margin_allocations == margins
    assert t.captures_since(env, mark) == [] and not [
        e for e in env.events[mark:] if e[0] == "replay"
    ], "post_warmup launched a segment graph"


@pytest.mark.gpu
# @spec TXG-VER-003
def test_verification_replays_and_allocates_nothing_cuda(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    env.core.startup(verify=False)
    replays = []
    original = torch.cuda.CUDAGraph.replay

    def replay(self):
        replays.append(self)
        return original(self)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    env.core.post_warmup()
    torch.cuda.synchronize()
    assert replays == []
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.memory_reserved() == reserved


@pytest.mark.gpu
# @spec TXG-VER-001, TXG-VER-002
def test_a_moved_parameter_fails_verification_cuda(on_cuda):
    env = on_cuda(n=2)
    env.core.startup(verify=False)
    parameter = next(env.model.parameters())
    parameter.data = parameter.data.clone()
    assert h.call_with(env.core.post_warmup) is not None
