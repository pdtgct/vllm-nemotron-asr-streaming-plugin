# SPDX-License-Identifier: Apache-2.0
"""Counted fallbacks, signatures and the all-failed bypass on CUDA
(TXG-COV-001, TXG-COV-003 to TXG-COV-005, TXG-COV-008, TXG-COV-012,
TXG-COV-013, TXG-CELL-004, TXG-CELL-005)."""

from __future__ import annotations

import dataclasses
import logging

import pytest
import torch
from support import model_harness as h
from support import txg_api as api
from support import txg_gpu as g

pytestmark = [pytest.mark.gpu, pytest.mark.engine]


def fallbacks(env):
    return dict(env.counters().get("fallbacks", {}))


def replays(env):
    return sum(env.counters().get("replays", {}).values())


def histogram(env):
    return dict(env.counters().get("histogram", {}))


# @spec TXG-COV-001
def test_classification_reads_no_device_memory(on_cuda, monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    env = on_cuda(startup=True, lookahead=0, n=4)
    env.compare()
    reads = []

    class DeviceReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            for x in g.tx.tensors_in([list(args), kwargs or {}]):
                if x.device.type == "cuda":
                    reads.append(str(func))
            return func(*args, **(kwargs or {}))

    module = api.graph_module()
    real = module.classify

    def classify(plan, **kwargs):
        with DeviceReads():
            return real(plan, **kwargs)

    monkeypatch.setattr(module, "classify", classify)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_file("f1", 3000)
    d.run(6)
    assert not reads, sorted(set(reads))


@pytest.mark.parametrize(
    "streams, drains, reason, kind",
    [(3, 0, "chunk-uncovered", api.CHUNK), (0, 3, "echo-uncovered", api.ECHO)],
)
# @spec TXG-COV-003, TXG-COV-004
def test_an_uncovered_step_runs_option_off_and_is_counted_once(
    on_cuda, streams, drains, reason, kind
):
    env = on_cuda(startup=True, lookahead=0, n=2, max_num_seqs=4)
    records = env.compare()
    d = g.Driver(env)
    ids = [f"s{i}" for i in range(3)]
    for r in ids:
        d.add_stream(r, [{}, {}])
    if drains:
        d.step(only=ids[:2])  # covered: two CHUNK rows
        d.step(only=[ids[2]])
        waiting = [r for r in ids if not d.parked(r)]
        assert len(waiting) == 3, "a stream queued no label"
    before_fallbacks, before_replays = fallbacks(env), replays(env)
    if drains:
        d.step(only=ids)  # three REPLAY rows, E > N
    else:
        d.step(only=ids)  # three CHUNK rows, K > N
    record = records[-1]
    assert not record.covered and not record.problems
    after = fallbacks(env)
    key = (reason, kind, 3)
    assert after.get(key, 0) - before_fallbacks.get(key, 0) == 1, after
    assert sum(after.values()) - sum(before_fallbacks.values()) == 1
    assert replays(env) == before_replays
    statuses, _ = env.runner.commit_log[-1]
    assert all(s == 0 for s in statuses.values()), statuses


def wrapped_transition(kwargs, plan):
    real = kwargs["encoder_transition"]

    def transition(*args, **kw):
        return real(*args, **kw)

    kwargs["encoder_transition"] = transition
    return kwargs, plan


def endpoint_policy(kwargs, plan):
    """An endpoint policy on every row of a CHUNK-only step: the cells run
    the policy-free endpoint bookkeeping only, so the step falls back."""
    if not bool(plan.is_chunk.all()):
        return kwargs, plan
    n = len(plan.request_ids)
    plan = dataclasses.replace(
        plan,
        endpoint_mode=torch.ones(n, dtype=torch.int64),
        endpoint_threshold_frames=torch.full((n,), 2, dtype=torch.int64),
        endpoint_residue_frames=torch.zeros(n, dtype=torch.int64),
    )
    return kwargs, plan


@pytest.mark.parametrize(
    "mutate, field",
    [(wrapped_transition, "encoder"), (endpoint_policy, "endpoint_policy")],
    ids=["callable", "endpoint"],
)
# @spec TXG-COV-005, TXG-COV-012, TXG-CELL-004
def test_a_signature_mismatch_falls_back_with_one_warning(
    on_cuda, caplog, mutate, field
):
    env = on_cuda(startup=True, lookahead=0, n=4)
    records = env.compare(mutate=mutate)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}, {}])
    caplog.set_level(logging.WARNING)
    before = fallbacks(env)
    d.step()
    d.until_parked(["s1"])
    d.step()
    chunk_steps = [r for r in records if bool(r.plan.is_chunk.any())]
    assert chunk_steps and not any(r.covered for r in chunk_steps)
    after = fallbacks(env)
    keys = [k for k in after if k[0] == "signature" and after[k] != before.get(k, 0)]
    assert keys and all(k[1] == api.CHUNK for k in keys), after
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and field in r.getMessage()
        and "signature" in r.getMessage().lower()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]


# @spec TXG-COV-012, TXG-CELL-002, TXG-CELL-004
def test_a_forced_eou_candidate_falls_back_with_one_warning(on_cuda, caplog):
    """A row that schedules the EOU token without an endpoint policy (a
    direct engine stream client's) is a forced-EOU candidate: its step runs
    as the option-off transaction, which forces EOU as ``B_hook`` does."""
    env = on_cuda(startup=True, lookahead=0, n=4)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}, {}])
    for name in ("q1", "q2"):
        d.add_stream(name, [{}, {}])
    d.step()
    d.until_parked(["q1", "q2"])
    caplog.set_level(logging.WARNING)
    before, before_replays = fallbacks(env), replays(env)
    for name in ("q1", "q2"):
        d.step(only=[], extra_new=[d.special(name, int(env.config.eou_token_id))])
        record = records[-1]
        assert not record.covered and not record.problems
        assert bool(record.plan.forced_eou_candidate.all())
    after = fallbacks(env)
    key = ("signature", api.ECHO, 1)
    assert after.get(key, 0) - before.get(key, 0) == 2, after
    assert sum(after.values()) - sum(before.values()) == 2
    assert replays(env) == before_replays
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "forced_eou" in r.getMessage()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    # a later step without a candidate is covered again
    d.step(only=["s1"])
    assert records[-1].covered and not records[-1].problems


# @spec TXG-CELL-005
def test_a_new_bound_method_of_the_same_callable_matches(on_cuda):
    """The model hands ``execute`` a fresh bound method each step; the
    signature identifies it by ``(id(__self__), id(__func__))``."""
    env = on_cuda(startup=True, lookahead=0, n=4)

    def rebound(kwargs, plan):
        transition = kwargs["encoder_transition"]
        if hasattr(transition, "__func__"):
            kwargs["encoder_transition"] = transition.__func__.__get__(
                transition.__self__
            )
        return kwargs, plan

    records = env.compare(mutate=rebound)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.run(10)
    assert records and all(r.covered for r in records)


# @spec TXG-COV-008
def test_no_fallback_at_n_equal_p_max(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=6)
    records = env.compare()
    d = g.Driver(env)
    for i in range(4):
        d.add_stream(f"s{i}", [{}, {}, {}])
    d.add_file("f1", 6000)
    d.add_file("f2", 1000)
    d.run(80)
    assert records and all(r.covered for r in records)
    assert not fallbacks(env)


def foreign(env, req_id: str):
    other = 1 - env.geometry
    row = g.envelope(env.config, geometry=other, lookahead=env.lookahead, sequence=0)
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=row,
        identifier=f"uuid-{req_id}",
    )


# @spec TXG-COV-013
def test_an_all_failed_step_is_a_bypass(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=4)
    records = env.compare()
    d = g.Driver(env)
    before_hist, before_fallbacks, before_replays = (
        histogram(env),
        fallbacks(env),
        replays(env),
    )
    rows = [foreign(env, "x1"), foreign(env, "x2")]
    for r in rows:
        d._slot(r.req_id)
    d.step(extra_new=rows)
    assert records == [], "an all-failed step reached execute"
    assert fallbacks(env) == before_fallbacks
    assert replays(env) == before_replays
    after = histogram(env)
    assert after.get((0, 2), 0) - before_hist.get((0, 2), 0) == 1
    assert sum(after.values()) - sum(before_hist.values()) == 1
    assert len(env.binding.cells) == 2 * 4


# @spec TXG-COV-013, TXG-COV-010
def test_a_mixed_step_counts_its_failed_rows_in_e(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=4)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}])
    row = foreign(env, "x1")
    d._slot("x1")
    d.step(extra_new=[row])
    record = records[-1]
    assert record.covered and not record.problems
    assert int(record.plan.is_chunk.sum()) == 1 and len(record.plan.request_ids) == 2
    assert histogram(env).get((1, 1), 0) >= 1
