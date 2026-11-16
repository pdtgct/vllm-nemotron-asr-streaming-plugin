# SPDX-License-Identifier: Apache-2.0
"""Covered steps equal the option-off transaction, on CUDA (TXG-EQ-001,
TXG-EQ-003 to TXG-EQ-005, TXG-MASK-002, TXG-MASK-005, TXG-COV-008,
TXG-CELL-005, TXG-CELL-006).

Each test builds the runnable synthetic card's model with transaction graphs
on (``on_cuda``, ``support.txg_gpu``), drives scheduled steps, and
compares every covered step with the option-off transaction of the same row
plan on a copy of the pools (``GpuEnv.compare``). ``N = P_max`` unless a test
says otherwise, so every step is covered.
"""

from __future__ import annotations

import pytest
from support import txg_api as api
from support import txg_gpu as g
from support import txg_transaction as tx

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

N = 8

#: The scheduled token of the stream FLUSH row: a label of the runnable
#: synthetic card (a base-vocabulary token, below ``blank``), so neither a
#: carrier, the park (EOS) nor the continuation token.
FLUSH_TOKEN = 7


def assert_all_equal(records, *, covered=True):
    assert records, "no step reached execute"
    bad = [(i, r.problems) for i, r in enumerate(records) if r.problems]
    assert not bad, bad[:5]
    if covered:
        declined = [i for i, r in enumerate(records) if not r.covered]
        assert not declined, f"steps {declined} fell back"


def mixed_scenario(env, driver, *, silent_phase: bool = True):
    """Streams, files and fresh non-CHUNK rows that exercise every case of
    TXG-EQ-004 (see ``txg_gpu.step_cases``)."""
    chunk = tx.chunk_samples(env.lookahead)
    park, label = int(env.config.eos_token_id), 7
    d = driver
    d.add_stream("s1", [{}, {}, {}, {"samples": 600, "final": True}])
    d.add_file("f1", int(2.5 * chunk))
    # Two streams whose slots are reissued below for the fresh non-CHUNK rows.
    d.add_stream("p1", [{}, {}, {}, {}])
    d.add_stream("p2", [{}, {}, {}, {}])
    d.step()
    d.until_parked(["s1"])
    d.add_stream("s2", [{}, {"samples": 0, "final": True}], prompt=1)
    d.add_file("f2", 0)
    d.step()  # s1's next unit (continuing CHUNK) beside s2's first (fresh CHUNK)
    d.add_stream("s3", [{}, {}, {}])
    d.add_file("f3", 700)
    d.until_parked(["p1", "p2"])
    d.step(
        extra_new=[
            d.special("p1", park, fresh=True),
            d.special("p2", label, fresh=True),
        ]
    )
    d.run(120)
    if silent_phase:
        with env.silent():
            d.add_stream("s4", [{}, {}, {"samples": 300, "final": True}])
            d.add_file("f4", int(1.5 * chunk))
            d.add_stream("s5", [{}, {}])
            d.run(80)


# @spec TXG-EQ-001, TXG-EXEC-008, TXG-COV-008, TXG-CELL-005
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
def test_every_covered_step_equals_the_option_off_transaction(on_cuda, lookahead):
    env = on_cuda(startup=True, lookahead=lookahead, n=N)
    records = env.compare()
    driver = g.Driver(env)
    mixed_scenario(env, driver)
    assert_all_equal(records)
    splits = [
        r for r in records if 0 < int(r.plan.is_chunk.sum()) < len(r.plan.request_ids)
    ]
    assert splits, "no split step ran"
    interleaved = [
        r
        for r in splits
        if r.plan.is_chunk.tolist() != sorted(r.plan.is_chunk.tolist(), reverse=True)
    ]
    assert interleaved, "no split step interleaved CHUNK and echo rows"


# @spec TXG-EQ-004, TXG-MASK-005
def test_each_masked_row_case_is_exact(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    driver = g.Driver(env)
    mixed_scenario(env, driver)
    assert_all_equal(records)
    seen = set()
    for record in records:
        seen |= g.step_cases(record, env, driver)
    required = {
        "first chunk among continuing rows",
        "file first step",
        "regular chunk",
        "silent chunk",
        "final tail with samples",
        "empty final tail",
        "zero-sample file",
        "drain step",
        "end step as REPLAY",
        "end step as FLUSH",
        "armed echo on a file CHUNK row",
        "split step mixing file and stream rows in both kinds",
        "fresh non-CHUNK row rejected",
        "fresh non-CHUNK row accepted as a park echo",
    }
    missing = required - seen
    assert not missing, f"the scenario never covered {sorted(missing)}"


# @spec TXG-EQ-005
def test_160_ms_rows_are_exact(on_cuda):
    env = on_cuda(startup=True, lookahead=1, n=N)
    records = env.compare()
    d = g.Driver(env)
    oversize = tx.chunk_samples(1)
    assert oversize == 2560
    d.add_stream("one", [{"samples": 1500, "final": True}])
    d.add_stream("a", [{}, {}, {"samples": 800, "final": True}])
    d.add_stream("b", [{}, {}, {}, {}])
    d.step()
    d.lockstep(["a", "b"])
    d.lockstep(["a", "b"])  # a's final tail beside b's regular chunk
    d.until_parked(["a"])  # a's drains: the last REPLAY drain emits park
    # a's FLUSH: a non-CHUNK row of the drained, finalized stream with no
    # pending echo, as a direct engine stream client sends it. After REPLAY
    # drains, plugin scheduling sends no FLUSH row: the last drain already
    # parked the stream. (Plugin scheduling ends with a FLUSH row only when
    # the final tail queued nothing; transcription design, role derivation.)
    d.step(only=["a"], extra_new=[d.special("a", FLUSH_TOKEN)])
    d.add_stream("big", [{}, {"samples": oversize, "final": True, "oversize": True}])
    d.add_stream("c", [{}, {}])
    d.step(only=["big", "c"])
    d.lockstep(["big", "c"])
    d.run(60)
    assert_all_equal(records)
    seen = set()
    for record in records:
        seen |= g.step_cases(record, env, d)
    required = {
        "one-chunk session",
        "final tail among regular continuing rows",
        "drain step",
        "stream FLUSH",
        "oversized final tail",
    }
    assert not required - seen, sorted(required - seen)
    from vllm_nemotron_asr_streaming.nemotron_asr.frontend import (
        ROW_STATUS_FINAL_OVERSIZE,
    )

    oversized = [
        (r, i)
        for r in records
        if r.covered
        for i, rid in enumerate(r.plan.request_ids)
        if rid == "big"
        and bool(r.plan.is_chunk[i])
        and int(r.inputs_embeds[i, tx.H_FINAL])
    ]
    assert oversized
    record, i = oversized[0]
    assert int(record.status[i]) & ROW_STATUS_FINAL_OVERSIZE
    others = [j for j in range(len(record.plan.request_ids)) if j != i]
    assert all(int(record.status[j]) == 0 for j in others)
    assert any("big" in failed for _, failed in env.runner.commit_log), (
        "the oversized request did not fail"
    )


def one_kind_steps(env, driver, kind: str, n: int):
    """For each count 1..n, one step of exactly that many rows of ``kind``:
    CHUNK rows are units delivered to parked streams; echo rows are drains of
    streams with queued labels. Streams take slots in descending order and
    alternate prompts, so block indices, row orders and prompts differ from
    the capture rows (pages 1..k). The counts rise from 1 to ``n`` and fall
    back to 1, so a cell at ``k`` runs after larger steps left rows beyond
    ``k`` in its arena section."""
    d = driver
    ids = [f"{kind[0]}{i}" for i in range(n)]
    for i, r in enumerate(ids):
        d.add_stream(r, [{} for _ in range(2 * n + 2)], prompt=i % 2)
    d.step(only=ids)  # every first chunk
    for k in [*range(1, n + 1), *range(n - 1, 0, -1)]:
        if kind == "chunk":
            d.until_parked(ids)
            d.step(only=ids[:k])
        else:
            d.until_parked(ids)
            d.step(only=ids)  # next units: every stream queues labels
            waiting = [r for r in ids if not d.parked(r)]
            if len(waiting) < k:
                continue
            d.step(only=waiting[:k])


# @spec TXG-EQ-003, TXG-CELL-003, TXG-EXEC-005, TXG-MASK-002
@pytest.mark.parametrize("kind", [api.CHUNK, api.ECHO])
def test_each_cell_equals_its_sub_plan_at_every_count(on_cuda, kind):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    driver = g.Driver(env)
    one_kind_steps(env, driver, kind, N)
    assert_all_equal(records)
    counts = set()
    for record in records:
        k = int(record.plan.is_chunk.sum())
        e = len(record.plan.request_ids) - k
        if kind == api.CHUNK and e == 0:
            counts.add(k)
        if kind == api.ECHO and k == 0:
            counts.add(e)
    assert set(range(1, N + 1)) <= counts, sorted(counts)


# @spec TXG-MASK-002
def test_fresh_and_continuing_chunk_rows_replay_one_chunk_cell(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.step()
    d.until_parked(["s1"])
    d.add_stream("s2", [{}])
    before = env.counters().get("replays", {})
    d.step(only=["s1", "s2"])
    after = env.counters().get("replays", {})
    key = (env.geometry, api.CHUNK, 2)
    assert after.get(key, 0) - before.get(key, 0) == 1
    assert sum(after.values()) - sum(before.values()) == 1
    assert_all_equal(records)


# @spec TXG-CELL-006
def test_a_chunk_cell_replays_the_option_off_decoder_tier(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    graphs = env.model._decode_graph_binding
    for count in range(1, N + 1):
        cell = env.binding.cells[(env.geometry, api.CHUNK, count)]
        assert cell.decoder_tier == graphs.execution_tier(count), count


# @spec TXG-CELL-005
def test_bindings_keep_the_callables_they_hand_out(on_cuda):
    """The dense decode binding returns the same callable per graph key, and
    the encoder binding the same transition per count, on every call."""
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import DecodeRequest

    env = on_cuda(startup=True, lookahead=0, n=N)
    resolver = env.model._decode_resolver
    for count in range(1, N + 1):
        request = DecodeRequest(
            geometry=env.geometry,
            execution_batch_size=count,
            graph_covers_decode=True,
            ready_decode_buckets=1,
            execution_tier_limit=env.model._decode_graph_binding.execution_tier(count),
        )
        assert resolver(request).decode_fn is resolver(request).decode_fn, count
    encoder = env.model._encoder_execution
    for count in range(1, N + 1):
        first = encoder.uncaptured_transition(geometry=env.geometry, count=count)
        again = encoder.uncaptured_transition(geometry=env.geometry, count=count)
        assert first is again, count
