# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the tier 5 qualification driver and its adapter
(``support.txg_qual_driver``, ``support.txg_qual_adapter``; TXG-QUAL-002,
-004, -011, -026) on the runnable synthetic card with transaction graphs
off. They check the driver's own mechanics (the adapter's step lifecycle,
pacing, prefix targets, snapshot and restore), not any tier 5 figure; the
qualifying runs are A40 runs of the same code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from support import model_harness as h
from support import txg_qual_driver as d
from support.txg_qual_adapter import QualAdapter, TransactionMonitor

pytestmark = pytest.mark.engine


def context(
    env,
    *,
    capacity=12,
    lengths=(9000, 4000, 12000, 7000, 5000, 11000),
    check_off=False,
):
    env.runner.profile_run(core_embeds=False)
    bindings = env.runner.bind_pool(
        h.real_bindings(h.model_spec(env.model), capacity=capacity)
    )
    adapter = QualAdapter(
        vllm_config=env.vllm_config, model=env.model, state=env.state, bindings=bindings
    )
    monitor = TransactionMonitor(env.model, check_off=check_off).install()
    return SimpleNamespace(
        adapter=adapter,
        model=env.model,
        monitor=monitor,
        requests=d.Requests(env.model, lookahead=0),
        corpus=d.SyntheticCorpus(0, lengths),
        capacity=capacity,
        slots=d.Slots(capacity),
        free_next=[],
        counter=0,
    )


def test_drain_positions_are_evenly_spaced_and_distinct():
    assert d.drain_positions(31, 4) == [3, 11, 19, 27]
    for p, e in ((31, 4), (31, 15), (63, 8), (63, 31), (1, 1), (31, 31)):
        positions = d.drain_positions(p, e)
        assert len(set(positions)) == e and max(positions) < p


def test_slots_reissue_a_released_slot_at_the_next_generation():
    slots = d.Slots(4)
    assert slots.take("a") == (1, 1) and slots.take("b") == (2, 1)
    slots.release("a")
    assert slots.take("c") == (1, 2)


# @spec TXG-QUAL-002, TXG-QUAL-026
def test_paced_streams_run_through_the_adapter(off_cpu):
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env)
    try:
        run = d.Workload(ctx, name="Q2", n_streams=2, n_files=0, steps=14, seed=3)
        run.run()
        run.close()
        summary = run.summary()
    finally:
        ctx.monitor.uninstall()
    assert summary["steps"] == 14
    assert summary["failed_rows"] == 0, summary["problems"]
    assert all(r["P"] >= 1 for r in run.records)
    # every step timed one option-off transaction call
    assert len(summary["durations"]) == 14 and all(s > 0 for s in summary["durations"])
    assert not ctx.slots.handles, "close() left live requests"


# @spec TXG-QUAL-011, TXG-QUAL-026
def test_a_plan_restores_its_snapshot_on_every_step(off_cpu):
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env)
    try:
        result = d.run_composition(
            ctx, 1, 1, legs=["OFF1", "OFF2"], warmup=1, counted=2
        )
    finally:
        ctx.monitor.uninstall()
    assert not result["void"], result
    assert result["restores"] == 2 * 3 - 1
    assert [a["role"] for a in result["audio"]] == ["chunk", "drain"] or [
        a["role"] for a in result["audio"]
    ] == ["drain", "chunk"]
    assert all(len(leg["ms"]) == 3 for leg in result["legs"])


# @spec TXG-QUAL-003, TXG-QUAL-004
def test_mixed_files_run_in_order_and_match_the_reference(off_cpu):
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env, lengths=(3000, 2600, 4100, 3300, 2900), check_off=True)
    clips = ctx.corpus.clips(5)
    try:
        run = d.Workload(
            ctx,
            name="M",
            n_streams=1,
            n_files=1,
            steps=700,
            seed=5,
            files=d.special_files(ctx.requests, clips),
            clips=clips,
        )
        run.run()
        run.close()
        summary = run.summary()
    finally:
        ctx.monitor.uninstall()
    names = [f.name.split(":")[0] for f in run.finished_files]
    assert names[:4] == ["silence-padded", "whole-chunks", "zero-samples", "short"]
    # the option-off reference on a pool copy equals the live option-off call
    assert summary["mismatched_steps"] == 0, summary["problems"]
    assert all(r.get("ref") for r in run.records)
    for case in ("file-first-step", "empty-final-tail", "zero-sample-file"):
        assert summary["cases"].get(case), (case, summary["cases"])


# @spec TXG-QUAL-002, TXG-QUAL-008
def test_a_run_uses_the_served_q_convention_for_clips_and_offsets(off_cpu):
    """One generator per run: the permutation first, then each stream's
    offset as it is taken, slots in order (the served Q run's convention)."""
    import numpy as np

    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env)
    try:
        run = d.Workload(ctx, name="Q3", n_streams=3, n_files=0, steps=1, seed=11)
    finally:
        ctx.monitor.uninstall()
    rng = np.random.default_rng(11)
    perm = rng.permutation(ctx.corpus.count)
    c = ctx.requests.chunk_seconds
    for slot, stream in enumerate(run.streams):
        assert stream.utt == ctx.corpus.ids[perm[slot]]
        assert abs((stream.arrivals[0] - c) - rng.uniform(0.0, c)) < 1e-12
