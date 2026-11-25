# SPDX-License-Identifier: Apache-2.0
"""Process constants (TXG-CAP-016): a change after the serving capture is
worker-fatal on every ``execute`` call, covered or falling back, and the
check reads no legacy precision getter."""

from __future__ import annotations

import pytest
import torch
from support import model_harness as h
from support import txgcap_harness as t

pytestmark = pytest.mark.engine


CONSTANTS = {
    "matmul": (torch.backends.cuda.matmul, "fp32_precision", "tf32", "ieee"),
    "conv": (torch.backends.cudnn.conv, "fp32_precision", "ieee", "tf32"),
    "rnn": (torch.backends.cudnn.rnn, "fp32_precision", "ieee", "tf32"),
}


def change(monkeypatch, which):
    if which == "cublas":
        import os

        value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        monkeypatch.setenv(
            "CUBLAS_WORKSPACE_CONFIG", ":16:8" if value != ":16:8" else ":4096:8"
        )
        return
    target, name, first, second = CONSTANTS[which]
    current = getattr(target, name)
    # A plain assignment, restored by ``_restore_constants``: monkeypatch's
    # undo deletes an attribute that torch serves through ``__getattr__``,
    # and torch refuses the deletion.
    setattr(target, name, first if current != first else second)


@pytest.fixture(autouse=True)
def _restore_constants():
    saved = [
        (target, name, getattr(target, name)) for target, name, *_ in CONSTANTS.values()
    ]
    yield
    for target, name, value in saved:
        setattr(target, name, value)


def steps(env, kind):
    """A covered step (one CHUNK row) or one that falls back with
    ``chunk-uncovered`` (``K = N + 1``)."""
    count = 1 if kind == "covered" else env.n + 1
    return [t.stream_unit(env, f"s{i}") for i in range(count)]


@pytest.mark.parametrize("kind", ["covered", "fallback"])
@pytest.mark.parametrize("which", ["matmul", "conv", "rnn", "cublas"])
# @spec TXG-CAP-016
def test_a_changed_process_constant_latches_and_raises(
    on_cpu, monkeypatch, which, kind
):
    run = on_cpu(n=2, max_num_seqs=3)
    env, core = run.env, run.core
    core.startup()
    change(monkeypatch, which)
    replays, advances = env.fake.replay_count, len(env.advance.calls)
    rows = steps(env, kind)
    error = h.call_with(core.step, *rows)
    assert error is not None, f"a changed {which} setting did not raise"
    assert env.fake.replay_count == replays, "a cell replayed under a changed setting"
    assert len(env.advance.calls) == advances, (
        "the step fell back under a changed setting"
    )
    # The later request takes a block no earlier request holds (one block per
    # request, kept until its terminal finish).
    later = h.call_with(
        core.step, t.stream_unit(env, "later"), handles={"later": len(rows) + 1}
    )
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


# @spec TXG-CAP-016
def test_the_check_reads_no_legacy_precision_getter(on_cpu, monkeypatch):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()

    def legacy(*args, **kwargs):
        raise AssertionError("a legacy precision getter was read")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", legacy)
    core.step(t.stream_unit(env, "s1"))


@pytest.mark.gpu
@pytest.mark.parametrize("which", ["matmul", "cublas"])
# @spec TXG-CAP-016
def test_a_changed_process_constant_latches_and_raises_cuda(
    on_cuda, monkeypatch, which
):
    env = on_cuda(n=2)
    core = env.core
    core.startup()
    change(monkeypatch, which)
    advances = len(env.advance.calls)
    error = h.call_with(core.step, t.stream_unit(env, "s1"))
    assert error is not None
    assert len(env.advance.calls) == advances
    # As on CPU: the later request takes a block no earlier request holds.
    later = h.call_with(core.step, t.stream_unit(env, "s2"), handles={"s2": 2})
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later
