# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the tier 5 qualification driver and its adapter
(``support.txg_qual_driver``, ``support.txg_qual_adapter``; TXG-QUAL-002,
-003, -004, -007, -011, -026, -028) on the runnable synthetic card with
transaction graphs off. They check the driver's own mechanics (the adapter's
step lifecycle on the state layer, block assignment below the final
committed block count, pacing, prefix targets, snapshot and restore, the
detached reference and its bound, the requests run alone), not any tier 5
figure; the qualifying runs are A40 runs of the same code.

A state pool bound with fewer pages than its storage holds
(``txgcap_harness.narrow_pool``) stands for the views core binds after its
extensible KV cache grows: the final committed block count's pages over a
reservation that spans more.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from support import model_harness as h
from support import txg_qual_driver as d
from support import txg_transaction as tx
from support import txgcap_harness as t
from support.txg_qual_adapter import (
    Call,
    QualAdapter,
    TransactionMonitor,
    detached_problems,
)

pytestmark = pytest.mark.engine


def context(
    env,
    *,
    capacity=12,
    lengths=(9000, 4000, 12000, 7000, 5000, 11000),
    check_off=False,
):
    """Core's profiling forward, then a serving pool bound to the state layer
    (``bind_kv_cache``), then the adapter over the worker's objects."""
    env.runner.profile_run(core_embeds=False)
    env.runner.bind_pool(h.real_bindings(h.model_spec(env.model), capacity=capacity))
    adapter = QualAdapter(vllm_config=env.vllm_config, model=env.model, state=env.state)
    assert adapter.capacity == capacity
    monitor = TransactionMonitor(env.model, check_off=check_off).install()
    return SimpleNamespace(
        adapter=adapter,
        model=env.model,
        monitor=monitor,
        requests=d.Requests(env.model, lookahead=0),
        corpus=d.SyntheticCorpus(0, lengths),
        capacity=capacity,
        blocks=d.Blocks(capacity),
        free_next=[],
        counter=0,
    )


# @spec TXG-QUAL-011
def test_drain_positions_are_evenly_spaced_and_distinct():
    assert d.drain_positions(31, 4) == [3, 11, 19, 27]
    for p, e in ((31, 4), (31, 15), (63, 8), (63, 31), (1, 1), (31, 31)):
        positions = d.drain_positions(p, e)
        assert len(set(positions)) == e and max(positions) < p


# @spec TXG-QUAL-026
def test_blocks_are_kept_until_terminal_finish_and_reused_only_after():
    blocks = d.Blocks(4)
    assert blocks.take("a") == 1 and blocks.take("b") == 2
    # a request keeps its block on every later step, across turns
    assert blocks.take("a") == 1 and blocks.of(["a", "b"]) == {"a": 1, "b": 2}
    assert blocks.take("c") == 3
    with pytest.raises(RuntimeError):
        blocks.take("d")
    blocks.release("a")
    assert blocks.take("d") == 1, "a finished request's block is reused"
    assert blocks.held == {"b": 2, "c": 3, "d": 1}


class LifecycleLog:
    """Records the model state's lifecycle calls in order, as
    ``(name, request_id)`` or ``(name,)``, calling through."""

    def __init__(self, state) -> None:
        self.calls: list[tuple] = []
        self.added: dict[str, list] = {}
        for name in (
            "remove_request",
            "add_request",
            "prepare_inputs",
            "postprocess_state",
            "take_failed_requests",
        ):
            original = getattr(state, name)

            def wrapper(*args, __original=original, __name=name, **kwargs):
                if __name == "remove_request":
                    self.calls.append((__name, args[0]))
                elif __name == "add_request":
                    request = args[1]
                    self.calls.append((__name, request.req_id))
                    self.added.setdefault(request.req_id, []).append(request)
                else:
                    self.calls.append((__name,))
                return __original(*args, **kwargs)

            setattr(state, name, wrapper)

    def take(self) -> list[tuple]:
        calls, self.calls = self.calls, []
        return calls


STEP_END = [("prepare_inputs",), ("postprocess_state",), ("take_failed_requests",)]


# @spec TXG-QUAL-026
def test_the_adapter_drives_cores_step_lifecycle(off_cpu):
    """Finished requests are removed in ascending ID order; each delivered
    ``NewRequestData`` is removed then added before the next one (a stream's
    next turn with its compacted prompt and ``num_computed_tokens`` 1, on
    the block it kept); a cached request (a file's later step, a label drain
    within a turn) gets neither call; then ``prepare_inputs``, the forward,
    ``postprocess_state`` and ``take_failed_requests``."""
    import numpy as np

    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env)
    log = LifecycleLog(env.state)
    req, adapter = ctx.requests, ctx.adapter
    units = {r: req.units(ctx.corpus.take()[1]) for r in ("zz-s", "aa-s")}
    samples = (np.random.default_rng(7).standard_normal(20000) * 0.1).astype(np.float32)
    try:
        first = [req.unit("zz-s", units["zz-s"][0], "0"), req.file("mm-f", samples)]
        adapter.step(new=first, blocks=ctx.blocks.of(["zz-s", "mm-f"]))
        assert log.take() == [
            ("remove_request", "zz-s"),
            ("add_request", "zz-s"),
            ("remove_request", "mm-f"),
            ("add_request", "mm-f"),
            *STEP_END,
        ]
        assert log.added["zz-s"][0].num_computed_tokens == 0
        assert log.added["zz-s"][0].block_ids == ([1],)
        # Cached steps: the file's later steps and the stream's label drains
        # until it parks.
        position = {"zz-s": 1, "mm-f": 1}
        for _ in range(40):
            if adapter.sampled["zz-s"] == req.park:
                break
            adapter.step(
                cached=["zz-s", "mm-f"],
                blocks=ctx.blocks.of(["zz-s", "mm-f"]),
                computed=dict(position),
            )
            assert log.take() == STEP_END, "a cached request got a lifecycle call"
            position = {r: p + 1 for r, p in position.items()}
        assert adapter.sampled["zz-s"] == req.park, "the stream never parked"
        assert adapter.sampled["mm-f"] != req.park, "the file ended too soon"
        # The stream's next turn, a new stream and the file's later step.
        frontier = adapter.last_fed("zz-s")
        turn = req.unit("zz-s", units["zz-s"][1], "1")
        fresh = req.unit("aa-s", units["aa-s"][0], "0")
        adapter.step(
            new=[turn, fresh],
            cached=["mm-f"],
            blocks=ctx.blocks.of(["zz-s", "aa-s", "mm-f"]),
            computed={"mm-f": position["mm-f"]},
        )
        assert log.take() == [
            ("remove_request", "zz-s"),
            ("add_request", "zz-s"),
            ("remove_request", "aa-s"),
            ("add_request", "aa-s"),
            *STEP_END,
        ], "a cached request got a lifecycle call"
        continued = log.added["zz-s"][1]
        assert continued.num_computed_tokens == 1
        assert continued.block_ids == ([1],), "the stream's block changed"
        assert len(continued.prompt_token_ids) == 2
        # core keeps the last computed token, not the final sampled one
        assert continued.prompt_token_ids[0] == frontier != req.park
        assert continued.prompt_token_ids[1] == turn.prompt_token_ids[0]
        assert [int(f.mm_position.offset) for f in continued.mm_features] == [1]
        # Terminal finishes, in ascending request-ID order, on a cleanup-only
        # step that runs no forward work.
        adapter.step(finished=["zz-s", "mm-f", "aa-s"], blocks={})
        assert log.take() == [
            ("remove_request", "aa-s"),
            ("remove_request", "mm-f"),
            ("remove_request", "zz-s"),
            ("take_failed_requests",),
        ]
    finally:
        ctx.monitor.uninstall()


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
    assert not ctx.blocks.held, "close() left live requests"


# @spec TXG-QUAL-011, TXG-QUAL-026
def test_a_plan_restores_its_snapshot_on_every_step(off_cpu):
    """Every plan step after the first restores the snapshot in place, at
    the blocks the adapter assigned to the plan's rows and nowhere else."""
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
    assert len(set(result["blocks"])) == result["P"] == 2
    assert all(sorted(b) == sorted(result["blocks"]) for b in result["restored"])
    assert sorted(a["role"] for a in result["audio"]) == ["chunk", "drain"]
    assert all(len(leg["ms"]) == 3 for leg in result["legs"])
    assert not ctx.blocks.held, "the plan's requests were not finished"


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
    # the option-off reference on a detached pool copy equals the live
    # option-off call, and never reached the serving pools
    assert summary["mismatched_steps"] == 0, summary["problems"]
    assert all(r.get("ref") for r in run.records)
    for case in ("file-first-step", "empty-final-tail", "zero-sample-file"):
        assert summary["cases"].get(case), (case, summary["cases"])


# @spec TXG-QUAL-003
def test_the_reference_runs_on_a_detached_copy():
    """The reference's pools are new storage, and no argument it receives is
    a view of the serving pools, so it can neither read nor write them."""
    storage = torch.zeros(4, 64, dtype=torch.int8)
    live = tx.Pools(
        {
            "h_pool": storage[:, :32].view(torch.float32),
            "channel_pools": [storage[:, 32:]],
        }
    )
    copy = live.clone()
    assert detached_problems(copy, live) == []
    assert detached_problems(copy, live, {"adapter": None, "eou_token_id": 3}) == []
    aliased = tx.Pools({"h_pool": storage[:, :32].view(torch.float32)})
    assert detached_problems(aliased, live), "a view of the serving pool passed"
    leaked = {"extra": storage[1:, 40:]}
    assert detached_problems(copy, live, leaked), "a serving-pool argument passed"


# @spec TXG-QUAL-007
def test_a_multi_token_row_fails_alone_among_one_token_rows(off_cpu):
    """The driver's multi-token step (a direct client's two-token request
    beside a stream's chunk): planning fails the multi-token row alone, the
    other row commits, and the option-off transaction equals its reference
    on a detached copy."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env, check_off=True)
    req = ctx.requests
    try:
        one = req.unit("one-token", req.units(ctx.corpus.take()[1])[0], "0")
        multi = req.multi_token("multi-token", req.label_tokens(2))
        ctx.monitor.begin_step()
        result = ctx.adapter.step(
            new=[one, multi],
            blocks=ctx.blocks.of(["one-token", "multi-token"]),
            tokens={"multi-token": 2},
        )
    finally:
        ctx.monitor.uninstall()
    assert result.failed == {"multi-token"}, result.failed
    off = [c for c in ctx.monitor.calls if c.kind == "off"]
    assert len(off) == 1 and off[0].compared and not off[0].problems, off


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


# ---------------------------------------------------------------------------
# The final committed block count (TXG-QUAL-003, TXG-QUAL-026)

PAGE = 4096


def narrowed_pools(pages: int = 8, storage_pages: int = 12) -> tx.Pools:
    """Pool views of ``pages`` pages over a storage of ``storage_pages``,
    laid out as a strided resident pool (two views per page)."""
    storage = torch.zeros(storage_pages, PAGE, dtype=torch.int8)
    half = PAGE // 2
    return tx.Pools(
        {
            "h_pool": storage[:pages, :half].view(torch.float32),
            "channel_pools": [storage[:pages, half:]],
            "time_pools": [],
            "len_pools": [],
        }
    )


# @spec TXG-QUAL-003
def test_the_detached_copy_holds_only_the_pages_below_the_final_committed_count():
    """The copy of views of the final committed count's pages, over a
    storage that spans more (the reservation), holds those pages' bytes and
    no more, keeps each view's layout and alignment, and equals the live
    views bit for bit."""
    live = narrowed_pools(pages=8, storage_pages=12)
    for _, view in live.flat():
        view.copy_(torch.randint(-100, 100, view.shape).to(view.dtype))
    copy = live.clone()
    assert copy.storage_bytes() <= 8 * PAGE + 512, copy.storage_bytes()
    assert copy.storage_bytes() < 12 * PAGE
    assert copy.first_dimensions() == {8}
    assert live.differences(copy) == []
    assert detached_problems(copy, live) == []
    for (_, a), (_, b) in zip(live.flat(), copy.flat(), strict=True):
        assert (a.shape, a.stride(), a.dtype) == (b.shape, b.stride(), b.dtype)
        assert a.data_ptr() % 512 == b.data_ptr() % 512


# @spec TXG-QUAL-003
def test_a_copy_of_views_above_the_final_committed_count_is_a_problem(off_cpu):
    """Views that keep more pages than the final committed block count (a
    layout core's extension leaves at the reservation's page count) are
    reported: the copy would not be limited to the committed pages."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    pools = narrowed_pools(pages=8)
    monitor = TransactionMonitor(env.model, compare=True, final_blocks=6)
    call = Call(kind="execute")
    copy = monitor._copy(pools.kwargs(), call)
    assert any("final committed" in p for p in call.problems), call.problems
    within = TransactionMonitor(env.model, compare=True, final_blocks=8)
    call = Call(kind="execute")
    within._copy(pools.kwargs(), call)
    assert call.problems == [], call.problems
    assert copy.first_dimensions() == {8}


# @spec TXG-QUAL-026, TXG-QUAL-003
def test_the_adapter_reads_the_final_committed_block_count(off_cpu):
    """After a binding of fewer pages than the storage holds, the adapter's
    final committed block count is the bound views' first dimension, or
    core's live ``kv_cache_config`` count when that is smaller; every block
    the driver assigns is below it."""
    from types import SimpleNamespace

    env = off_cpu(right_context=0, max_num_seqs=4)
    env.runner.profile_run(core_embeds=False)
    reservation = h.real_bindings(h.model_spec(env.model), capacity=12)
    env.runner.bind_pool(t.narrow_pool(reservation, 8))
    adapter = QualAdapter(vllm_config=env.vllm_config, model=env.model, state=env.state)
    assert adapter.capacity == 8 and adapter.final_committed_blocks == 8
    live_config = SimpleNamespace(kv_cache_config=SimpleNamespace(num_blocks=6))
    core = QualAdapter(
        vllm_config=env.vllm_config,
        model=env.model,
        state=env.state,
        runner=live_config,
    )
    assert core.final_committed_blocks == 6
    blocks = d.Blocks(core.final_committed_blocks)
    taken = [blocks.take(f"r{i}") for i in range(5)]
    assert sorted(taken) == [1, 2, 3, 4, 5] and max(taken) < 6
    with pytest.raises(RuntimeError):
        blocks.take("r5")


# ---------------------------------------------------------------------------
# Requests alone (TXG-QUAL-028)


# @spec TXG-QUAL-028
def test_the_alone_mode_runs_one_stream_and_each_file_to_its_end(off_cpu):
    """One stream served alone, then each of TXG-QUAL-004's files alone (the
    four special files and one utterance), each its own request on an
    otherwise idle server; each item's transcript is recorded for the
    comparison with ``B_hook``."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env, lengths=(3000, 2600, 4100, 3300, 2900, 3100, 2700))
    try:
        result = d.run_alone(ctx, {"seed": 5, "alone_utterances": 1})
    finally:
        ctx.monitor.uninstall()
    names = [item["name"] for item in result["items"]]
    assert names == ["alone-stream", *(f"alone-file{i}" for i in range(5))]
    kinds = [
        next(iter(item["transcripts"].values()))["utt"].split(":")[0]
        for item in result["items"][1:]
    ]
    assert kinds == [
        "silence-padded",
        "whole-chunks",
        "zero-samples",
        "short",
        "utterance",
    ]
    for item in result["items"]:
        assert len(item["transcripts"]) == 1, item
        assert item["steps"] >= 1 and item["failed_rows"] == 0, item
    assert result["fallbacks"] == {} and result["replays"] == 0
    assert not ctx.blocks.held, "a request was left live"
