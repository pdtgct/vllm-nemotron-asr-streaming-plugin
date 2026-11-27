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
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

pytestmark = pytest.mark.engine


def context(
    env,
    *,
    capacity=12,
    lengths=(9000, 4000, 12000, 7000, 5000, 11000),
    check_off=False,
    max_requests=None,
):
    """Core's profiling forward, then a serving pool bound to the state layer
    (``bind_kv_cache``), then the adapter over the worker's objects. With
    ``max_requests`` the monitor copies as the ``qm`` and ``fallback`` modes
    do (block 0 and one page per request, the rest digested; TXG-QUAL-003)."""
    env.runner.profile_run(core_embeds=False)
    env.runner.bind_pool(h.real_bindings(h.model_spec(env.model), capacity=capacity))
    adapter = QualAdapter(vllm_config=env.vllm_config, model=env.model, state=env.state)
    assert adapter.capacity == capacity
    monitor = TransactionMonitor(
        env.model,
        check_off=check_off,
        final_blocks=capacity if max_requests is not None else None,
        max_requests=max_requests,
    ).install()
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
def test_a_copy_of_views_above_the_final_committed_count_holds_only_its_pages(
    off_cpu,
):
    """Views that keep more pages than the final committed block count (a
    layout core's extension leaves at the reservation's page count) are
    copied up to that count only: the copy holds pages ``0 … 5`` of six
    committed, with the views' layout, and equals the live views there."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    pools = narrowed_pools(pages=8, storage_pages=12)
    for _, view in pools.flat():
        view.copy_(torch.randint(-100, 100, view.shape).to(view.dtype))
    monitor = TransactionMonitor(env.model, compare=True, final_blocks=6)
    assert monitor.copy_extent == 6
    call = Call(kind="execute")
    copy = monitor._copy(pools.kwargs(), call)
    assert call.problems == [], call.problems
    assert copy.first_dimensions() == {6}
    assert copy.storage_bytes() <= 6 * PAGE + 512, copy.storage_bytes()
    assert pools.narrowed(6).differences(copy) == []
    copied = dict(copy.flat())
    for name, a in pools.narrowed(6).flat():
        b = copied[name]
        assert (a.shape, a.stride(), a.dtype) == (b.shape, b.stride(), b.dtype)
    within = TransactionMonitor(env.model, compare=True, final_blocks=8)
    call = Call(kind="execute")
    assert within._copy(pools.kwargs(), call).first_dimensions() == {8}
    assert call.problems == [], call.problems


# @spec TXG-QUAL-003
def test_the_copy_reads_no_page_at_or_above_the_final_committed_count(off_cpu):
    """Taking the detached copy addresses only pages below the final
    committed block count (``txgcap_harness.BlockAccesses`` over the pool's
    whole storage), and leaves every page at or above it untouched."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    pools = narrowed_pools(pages=8, storage_pages=12)
    storage = pools.tensors["channel_pools"][0]._base
    assert storage is not None and storage.shape[0] == 12
    storage.copy_(torch.randint(-100, 100, storage.shape, dtype=torch.int8))
    above = storage[6:].clone()
    monitor = TransactionMonitor(env.model, compare=True, final_blocks=6)
    with t.BlockAccesses(storage.view(12, 1, 1, PAGE)) as rec:
        copy = monitor._copy(pools.kwargs(), Call(kind="execute"))
    assert rec.accesses, "the copy addressed no page (positive control)"
    assert rec.outside(set(range(6))) == [], rec.outside(set(range(6)))
    assert torch.equal(storage[6:], above), "a page at or above the count changed"
    assert copy.first_dimensions() == {6}


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
# The exactness copy extent (TXG-QUAL-003, TXG-QUAL-029)


class Allocations(TorchDispatchMode):
    """Records the bytes of every new storage an op creates: an output whose
    storage is not an input's (a view, ``set_``) and not one of ``known``
    (the pool's)."""

    def __init__(self, known: set[int]) -> None:
        super().__init__()
        self.known = known
        self.sizes: list[tuple[str, int]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        inputs = {
            a.untyped_storage().data_ptr()
            for a in tree_flatten((args, kwargs))[0]
            if isinstance(a, torch.Tensor)
        }
        for o in tree_flatten(out)[0]:
            if isinstance(o, torch.Tensor):
                ptr = o.untyped_storage().data_ptr()
                if ptr and ptr not in inputs | self.known:
                    self.sizes.append((str(func), o.untyped_storage().nbytes()))
        return out


def _tiny_model() -> torch.nn.Module:
    """A model for the monitor's copy mechanics alone: CPU parameters, no
    binding."""
    return torch.nn.Linear(1, 1)


def _random(pools: tx.Pools, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _, view in pools.flat():
        bits = torch.randint(-100, 100, view.shape, generator=gen)
        view.copy_(bits.to(view.dtype))


# @spec TXG-QUAL-003
def test_an_exactness_copy_holds_pages_0_to_the_admission_limit():
    """With the mode's request count given (``qm``: the admission limit),
    the detached copy holds pages ``0 … limit``, whatever the final
    committed count; a smaller count raises rather than choosing another
    extent. Without a request count the copy keeps the final committed
    count (the modes with no detached copy)."""
    from support.txg_qual_adapter import exactness_extent

    assert exactness_extent(8, 3) == 4 and exactness_extent(None, 3) == 4
    assert exactness_extent(4, 3) == 4
    assert exactness_extent(8, None) == 8 and exactness_extent(None, None) is None
    with pytest.raises(ValueError, match="copy extent 4"):
        exactness_extent(3, 3)
    pools = narrowed_pools(pages=8, storage_pages=12)
    _random(pools)
    monitor = TransactionMonitor(
        _tiny_model(), compare=True, final_blocks=8, max_requests=3, memory=True
    )
    assert monitor.copy_extent == 4
    copy = monitor._copy(pools.kwargs(), Call(kind="execute"))
    assert copy.first_dimensions() == {4}
    assert pools.narrowed(4).differences(copy) == []
    reading = monitor.copy_readings[-1]
    assert reading["copy_bytes"] == 4 * PAGE, reading
    assert reading["copy_bytes"] <= reading["copy_allocation_bytes"] <= 4 * PAGE + 512
    with pytest.raises(ValueError):
        TransactionMonitor(_tiny_model(), final_blocks=3, max_requests=3)


# @spec TXG-QUAL-003
def test_a_write_above_the_admission_limit_fails_the_run_through_the_digest():
    """The committed pages from the admission limit plus one to the final
    committed count are digested in place before the run and after it: a
    write to one of them (a changed byte, one flipped bit, two elements
    swapped within a page) is a problem of the run, which counts as a
    mismatch and fails it; a write inside the copy's extent or above the
    committed count is not this check's, and a run whose pages were never
    digested reports that."""
    from support import txgcap_qual as q

    def fresh():
        pools = narrowed_pools(pages=8, storage_pages=12)
        _random(pools)
        monitor = TransactionMonitor(_tiny_model(), final_blocks=8, max_requests=3)
        monitor.watch_untouched(pools)
        return pools, monitor

    pools, monitor = fresh()
    assert monitor.untouched_problems() == []
    pools.tensors["channel_pools"][0][2, 0] += 1  # inside the copy's extent
    pools.tensors["channel_pools"][0]._base[10, -1] += 1  # above the count
    assert monitor.untouched_problems() == []

    writes = {
        "byte": lambda p: p.tensors["channel_pools"][0][6].__setitem__(
            5, p.tensors["channel_pools"][0][6, 5] + 1
        ),
        "bit": lambda p: p.tensors["h_pool"][7].view(torch.int32)[3].__ixor__(1),
        "swap": lambda p: p.tensors["channel_pools"][0][4, :2].copy_(
            p.tensors["channel_pools"][0][4, :2].flip(0).clone()
        ),
    }
    for name, write in writes.items():
        pools, monitor = fresh()
        if name == "swap":
            row = pools.tensors["channel_pools"][0][4]
            row[0], row[1] = 7, 9
            monitor.watch_untouched(pools)
        write(pools)
        problems = monitor.untouched_problems()
        assert len(problems) == 1 and "changed during the run" in problems[0], (
            name,
            problems,
        )
        clean = {
            "name": "Q4",
            "counts": {"split": 0, "fresh_among_continuing": 0, "split_with_file": 0},
            "cases": {},
            "mismatched_steps": 0,
            "fallback_steps": 0,
            "failed_rows": 0,
        }
        result = d.qm_result([clean, clean | {"name": "M"}], 4, problems)
        assert result["mismatches"] == 1 and result["untouched_problems"] == problems
        assert q.qualification_outcome([result], [1]) == ("fail", [1])
    assert d.qm_result([clean, clean | {"name": "M"}], 4)["mismatches"] == 0
    unwatched = TransactionMonitor(_tiny_model(), final_blocks=8, max_requests=3)
    assert unwatched.untouched_problems() == ["pages 4 to 7 were not digested"]
    whole = TransactionMonitor(_tiny_model(), final_blocks=8)
    assert whole.untouched_problems() == []


# @spec TXG-QUAL-003, TXG-QUAL-029
def test_the_pages_above_the_admission_limit_are_never_copied(monkeypatch):
    """Taking the copy addresses only pages ``0 … limit``, and neither the
    copy nor the digests allocate storage the size of the pages above the
    limit: the copy's one allocation is the extent's pages plus its
    alignment slack, and every digest allocation is bounded by the digest's
    chunk, not by the pages it covers."""
    from support import txgcap_qual as q

    chunk = 256
    monkeypatch.setattr(tx, "DIGEST_CHUNK_ELEMENTS", chunk)
    pools = narrowed_pools(pages=12, storage_pages=16)
    _random(pools)
    storage = pools.tensors["channel_pools"][0]._base
    known = {storage.untyped_storage().data_ptr()}
    above = 8 * PAGE  # pages 4 to 11
    monitor = TransactionMonitor(
        _tiny_model(), compare=True, final_blocks=12, max_requests=3
    )
    with t.BlockAccesses(storage.view(16, 1, 1, PAGE)) as rec, Allocations(known) as a:
        copy = monitor._copy(pools.kwargs(), Call(kind="execute"))
    assert rec.accesses and rec.outside(set(range(4))) == [], rec.accesses
    assert copy.first_dimensions() == {4}
    copied = [size for _, size in a.sizes]
    assert max(copied) <= 4 * PAGE + 512 < above, a.sizes
    with Allocations(known) as a:
        monitor.watch_untouched(pools)
        assert monitor.untouched_problems() == []
    assert a.sizes, "the digests allocated no workspace (positive control)"
    assert max(size for _, size in a.sizes) <= q.digest_workspace_bytes(chunk), a.sizes
    assert q.digest_workspace_bytes(chunk) < above


# @spec TXG-QUAL-003
def test_the_digest_tells_apart_every_change_within_a_chunk():
    """The digest hashes each chunk's bytes, not a reduction of them: chunks
    with equal sums and equal position-weighted sums (``[4, 4, 4]`` and
    ``[5, 2, 5]``) digest differently, and so does one flipped bit or two
    values swapped within one chunk, for every element size, on a strided
    view and across chunk boundaries; equal bits digest equally, and the
    view's identity and each chunk's offset are part of the digest."""
    counter = (
        torch.tensor([4, 4, 4], dtype=torch.int32),
        torch.tensor([5, 2, 5], dtype=torch.int32),
    )
    assert counter[0].sum() == counter[1].sum()
    weights = torch.arange(1, 4, dtype=torch.int32)
    assert (counter[0] * weights).sum() == (counter[1] * weights).sum()
    assert tx.digest(counter[0]) != tx.digest(counter[1])
    assert tx.digest(counter[0]) == tx.digest(counter[0].clone())
    gen = torch.Generator().manual_seed(5)
    for dtype in (torch.int8, torch.float16, torch.float32, torch.int64):
        storage = torch.randint(-50, 50, (16, 40), generator=gen).to(dtype)
        view = storage[:, 3:35:2]  # strided, as a page-strided state slice
        for chunk in (None, 7, 64):
            reference = tx.digest(view, chunk_elements=chunk)
            assert tx.digest(view.clone(), chunk_elements=chunk) == reference
            flipped = view.clone()
            flipped.view(tx._SAME_SIZE_INTEGERS[flipped.element_size()])[9, 4] ^= 1
            assert tx.digest(flipped, chunk_elements=chunk) != reference, (dtype, chunk)
            swapped = view.clone()
            swapped[2, 1], swapped[2, 2] = 11, 13
            before = tx.digest(swapped, chunk_elements=chunk)
            swapped[2, 1], swapped[2, 2] = 13, 11
            assert tx.digest(swapped, chunk_elements=chunk) != before, (dtype, chunk)
    flat = torch.arange(8, dtype=torch.int32)
    assert tx.digest(flat, name="a") != tx.digest(flat, name="b")
    assert tx.digest(flat) != tx.digest(flat.view(2, 4))
    halves = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
    assert tx.digest(halves, chunk_elements=2) != tx.digest(
        torch.tensor([1, 2, 2, 1], dtype=torch.int32), chunk_elements=2
    )


# @spec TXG-QUAL-003, TXG-QUAL-029
def test_the_comparison_reads_in_place_within_its_stated_workspace(monkeypatch):
    """Comparing a page-strided copy with the live views allocates no
    contiguous copy of either side: every allocation the comparison makes is
    at most ``txgcap_qual.comparison_workspace_bytes`` of its chunk, far
    below the views' bytes, and the comparison still finds one flipped bit,
    a swap and a NaN payload change, and equates NaNs with equal bits."""
    from support import txgcap_qual as q

    chunk = 256
    monkeypatch.setattr(tx, "DIGEST_CHUNK_ELEMENTS", chunk)
    pools = narrowed_pools(pages=12, storage_pages=16)
    _random(pools)
    copy = pools.clone()
    known = {t.untyped_storage().data_ptr() for p in (pools, copy) for _, t in p.flat()}
    view_bytes = min(t.numel() * t.element_size() for _, t in pools.flat())
    with Allocations(known) as a:
        assert pools.differences(copy) == []
        assert pools.differences(copy, blocks=[0, 5, 11]) == []
    assert a.sizes, "the comparison allocated no workspace (positive control)"
    assert max(size for _, size in a.sizes) <= q.comparison_workspace_bytes(chunk)
    assert q.comparison_workspace_bytes(chunk) < view_bytes, view_bytes
    h_pool = copy.tensors["h_pool"]
    h_pool.view(torch.int32)[7, 3] ^= 1
    assert pools.differences(copy) == ["h_pool"]
    assert pools.differences(copy, blocks=[6, 8]) == []
    h_pool.view(torch.int32)[7, 3] ^= 1
    assert pools.differences(copy) == []
    live, copied = (
        pools.tensors["channel_pools"][0][4],
        copy.tensors["channel_pools"][0][4],
    )
    live[0], live[1] = 7, 9
    copied[0], copied[1] = 9, 7
    assert pools.differences(copy) == ["channel_pools[0]"]
    nan = torch.tensor([float("nan")] * 3)
    assert tx.same_bits(nan, nan.clone(), chunk_elements=2)
    payload = nan.clone()
    payload.view(torch.int32)[1] ^= 1
    assert not tx.same_bits(nan, payload, chunk_elements=2)
    assert not tx.same_bits(torch.tensor(0.0), torch.tensor(-0.0))


# @spec TXG-QUAL-003, TXG-QUAL-026
def test_an_exactness_run_assigns_and_writes_only_pages_up_to_the_limit(off_cpu):
    """A paced ``Q`` run at concurrency 2, admission limit 2, over 12
    committed pages: the adapter's lowest-free assignment gives no request a
    block above 2, every option-off call equals its reference on a copy of
    pages 0 to 2, and the digest of pages 3 to 11 is unchanged after the
    run."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env, check_off=True, max_requests=2)
    taken: list[int] = []
    real_take = ctx.blocks.take

    def take(req_id):
        block = real_take(req_id)
        taken.append(block)
        return block

    ctx.blocks.take = take
    try:
        assert ctx.monitor.copy_extent == 3
        ctx.monitor.watch_untouched(tx.Pools({"views": list(ctx.adapter.views)}))
        run = d.Workload(ctx, name="Q2", n_streams=2, n_files=0, steps=14, seed=3)
        run.run()
        run.close()
        summary = run.summary()
        untouched = ctx.monitor.untouched_problems()
    finally:
        ctx.monitor.uninstall()
    assert taken and max(taken) <= 2, sorted(set(taken))
    assert summary["mismatched_steps"] == 0, summary["problems"]
    assert all(r.get("ref") for r in run.records)
    assert untouched == []


# @spec TXG-QUAL-007, TXG-QUAL-003
def test_a_final_count_below_the_modes_extent_fails_before_the_first_step():
    """The precondition: a final committed block count below the mode's copy
    extent (block 0 and one page per request it holds) fails at the
    monitor's construction, before any step, naming both counts; the copy
    never falls back to the whole cache."""
    from support import txgcap_qual as q

    held = q.mode_max_requests("fallback", admission_limit=4, n=2)
    assert held == 6 and q.copy_extent(held) == 7
    with pytest.raises(ValueError, match=r"count 6 is below the copy extent 7"):
        TransactionMonitor(_tiny_model(), final_blocks=6, max_requests=held)
    with pytest.raises(ValueError, match=r"count 4 is below the copy extent 5"):
        TransactionMonitor(_tiny_model(), final_blocks=4, max_requests=4)
    fits = TransactionMonitor(_tiny_model(), final_blocks=7, max_requests=held)
    assert fits.copy_extent == 7 and fits.untouched_problems() == []


# @spec TXG-QUAL-007, TXG-QUAL-026
def test_the_fallback_mode_writes_only_its_copy_extent(off_cpu):
    """The fallback mode at ``N`` = 2 over 12 committed pages holds exactly
    ``mode_max_requests`` requests (``N + 4``), on blocks 1 to 6: its copy
    of pages 0 to 6 holds every page it writes, every option-off call
    equals its reference on that copy, and the digest of pages 7 to 11 is
    unchanged after the run."""
    from support import txgcap_qual as q

    held = q.mode_max_requests("fallback", admission_limit=4, n=2)
    env = off_cpu(right_context=0, max_num_seqs=4)
    ctx = context(env, check_off=True, max_requests=held)
    taken: set[int] = set()
    real_take = ctx.blocks.take

    def take(req_id):
        block = real_take(req_id)
        taken.add(block)
        return block

    ctx.blocks.take = take
    try:
        assert ctx.monitor.copy_extent == q.copy_extent(held) == 7
        out = d.run_fallbacks(ctx, 2)
    finally:
        ctx.monitor.uninstall()
    assert taken == set(range(1, held + 1)), sorted(taken)
    assert out["problems"] == [] and out["uncompared"] == [], out
    assert out["untouched_problems"] == []
    assert not ctx.blocks.held


# @spec TXG-QUAL-003
def test_the_driver_reads_the_engines_admission_limit():
    """``max_num_active_seqs`` when set, otherwise ``max_num_seqs``."""
    for active, expected in ((None, 8), (3, 3)):
        config = SimpleNamespace(max_num_seqs=8, max_num_active_seqs=active)
        adapter = SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=config))
        assert d.admission_limit(adapter) == expected


# ---------------------------------------------------------------------------
# Requests alone (TXG-QUAL-028)


# @spec TXG-QUAL-028
def test_the_alone_items_are_one_stream_then_each_file_in_order(off_cpu):
    """One stream (the run's first clip), then TXG-QUAL-004's files in order
    from the same permutation, then ``alone_utterances`` utterances; each is
    served alone, its own request, on the client path."""
    env = off_cpu(right_context=0, max_num_seqs=4)
    requests = d.Requests(env.model, lookahead=0)
    corpus = d.SyntheticCorpus(0, (3000, 2600, 4100, 3300, 2900, 3100, 2700))
    items = d.alone_items(requests, corpus, {"seed": 5, "alone_utterances": 2})
    names = [name for name, *_ in items]
    assert names == ["alone-stream", *(f"alone-file{i}" for i in range(6))]
    kinds = [kind for _, kind, *_ in items]
    assert kinds == ["stream"] + ["file"] * 6
    utterances = [utt.split(":")[0] for _, _, utt, _ in items[1:]]
    assert utterances == [
        "silence-padded",
        "whole-chunks",
        "zero-samples",
        "short",
        "utterance",
        "utterance",
    ]
    assert items[3][3].shape[0] == 0, "the third file has zero samples"
    assert 0 < items[4][3].shape[0] < requests.unit_samples
    assert items[2][3].shape[0] % requests.unit_samples == 0
    first = corpus.clips(5).take()[0]
    assert items[0][2] == first, "the stream is the run's first clip"


# @spec TXG-QUAL-028
def test_each_alone_item_reads_its_own_replays_and_fallbacks():
    """The counters are read around each item, so an item's replays and its
    fallbacks by reason are its own, not the mode's aggregate."""
    before = {"replays": 10, "fallbacks": {"signature": 1, "chunk-uncovered": 2}}
    after = {"replays": 14, "fallbacks": {"signature": 1, "chunk-uncovered": 3}}
    assert d.alone_deltas(before, after) == {
        "replays": 4,
        "fallbacks": {"chunk-uncovered": 1},
    }
    empty = {"replays": 0, "fallbacks": {}}
    assert d.alone_deltas(empty, empty) == {"replays": 0, "fallbacks": {}}


# @spec TXG-QUAL-028
def test_the_alone_mode_serves_through_a_multiprocess_engine_with_its_probes():
    """The ``alone`` mode's worker extension resolves as core resolves it in
    the engine-core process, takes no name of core's worker, and its import
    installs the startup probes there; it runs in a child process, so this
    process keeps core's own classes."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parents[1]
    code = (
        "from vllm.utils.import_utils import resolve_obj_by_qualname\n"
        "from vllm.v1.worker import gpu_worker\n"
        "from vllm.v1.worker.gpu import model_runner\n"
        "from support import txg_qual_driver as d\n"
        "cls = resolve_obj_by_qualname(d.ALONE_READER)\n"
        "import support.txg_alone_worker as w\n"
        "assert isinstance(w.PROBES, d.StartupProbes)\n"
        "assert 'runner_init' in model_runner.GPUModelRunner.__init__.__qualname__\n"
        "assert 'extend_kv_cache' in gpu_worker.Worker.extend_kv_cache.__qualname__\n"
        "names = [a for a in dir(cls) if not a.startswith('__')]\n"
        "assert names == ['alone_counters', 'alone_startup'], names\n"
        "assert not [a for a in names if hasattr(gpu_worker.Worker, a)]\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tests), env.get("PYTHONPATH", "")) if p
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tests,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert d.ALONE_READER == "support.txg_alone_worker.AloneReader"
