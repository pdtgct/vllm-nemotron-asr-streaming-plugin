# SPDX-License-Identifier: Apache-2.0
"""FILE-BUF and the FILE-ROW step rule (pure-CPU tests of file_rows), and the
transaction's decision legality for file and stream rows (FILE-ROW-013)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def make_header(n: int, geometry: int = 3, row: int = 0) -> np.ndarray:
    return np.array([geometry, row, n], dtype=np.int64)


def samples(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def new_buffers():
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileHostBuffers

    return FileHostBuffers()


# ---------------------------------------------------------------------------
# FILE-BUF


# @spec FILE-BUF-001
def test_take_copies_the_samples_into_a_request_owned_buffer():
    buffers = new_buffers()
    source = samples(5000)
    original = source.copy()
    buffers.take("req-1", 4, 1, source, make_header(5000))
    assert "req-1" in buffers
    source[:] = 0.0  # core may drop or reuse its copy after the prompt is consumed
    np.testing.assert_array_equal(buffers.stage("req-1", 0, 5000), original)
    assert not np.shares_memory(buffers.stage("req-1", 0, 5000), source)
    # layout addition: header(request_id) reads the buffer's own copy
    header = make_header(5000)
    buffers.take("req-2", 5, 1, samples(10), header)
    header[:] = 0
    assert buffers.header("req-2").tolist() == [3, 0, 5000]


# @spec FILE-BUF-001
def test_buffers_are_independent_per_request():
    buffers = new_buffers()
    a, b = samples(300, 1), samples(300, 2)
    buffers.take("a", 1, 1, a, make_header(300))
    buffers.take("b", 2, 1, b, make_header(300))
    np.testing.assert_array_equal(buffers.stage("a", 0, 300), a)
    np.testing.assert_array_equal(buffers.stage("b", 0, 300), b)


# @spec FILE-BUF-002
@pytest.mark.parametrize("start, length", [(0, 8960), (8960, 8960), (17920, 1234)])
def test_stage_returns_exactly_one_chunk_as_a_view(start, length):
    buffers = new_buffers()
    whole = samples(17920 + 1234)
    buffers.take("req", 0, 1, whole, make_header(len(whole)))
    chunk = buffers.stage("req", start, length)
    assert chunk.shape == (length,)
    assert chunk.dtype == np.float32
    np.testing.assert_array_equal(chunk, whole[start : start + length])
    # a view of the request's buffer, not a copy of the file and not a new one
    again = buffers.stage("req", start, length)
    assert np.shares_memory(chunk, again)


# @spec FILE-BUF-002
def test_stage_of_an_empty_tail_is_empty():
    buffers = new_buffers()
    whole = samples(8960)
    buffers.take("req", 0, 1, whole, make_header(8960))
    assert buffers.stage("req", 8960, 0).shape == (0,)


# @spec FILE-BUF-003
def test_finished_request_frees_its_buffer():
    buffers = new_buffers()
    buffers.take("keep", 1, 1, samples(100), make_header(100))
    buffers.take("done", 2, 1, samples(100), make_header(100))
    buffers.free_finished(["done"])
    assert "done" not in buffers
    assert "keep" in buffers


# @spec FILE-BUF-003
def test_free_finished_ignores_unknown_ids_and_empty_input():
    buffers = new_buffers()
    buffers.take("keep", 1, 1, samples(100), make_header(100))
    buffers.free_finished([])
    buffers.free_finished(["never-seen", "keep"])
    buffers.free_finished(["keep"])  # twice is harmless
    assert "keep" not in buffers


# @spec FILE-BUF-003
@pytest.mark.parametrize("reason", ["finish", "abort", "preemption", "error"])
def test_free_depends_only_on_the_id_not_the_reason(reason):
    buffers = new_buffers()
    # The model state only learns the finished IDs, never why (design,
    # request-owned host buffer); any ID listed is freed.
    buffers.take(reason, 3, 1, samples(64), make_header(64))
    buffers.free_finished([reason])
    assert reason not in buffers


# @spec FILE-BUF-004
def test_new_generation_of_a_slot_drops_the_old_buffer():
    buffers = new_buffers()
    buffers.take("old", 7, 1, samples(100), make_header(100))
    buffers.take("new", 7, 2, samples(100, 5), make_header(100))
    assert "old" not in buffers
    assert "new" in buffers


# @spec FILE-BUF-004
def test_same_generation_or_other_slot_keeps_buffers():
    buffers = new_buffers()
    buffers.take("a", 7, 1, samples(100), make_header(100))
    buffers.take("b", 8, 9, samples(100), make_header(100))
    assert "a" in buffers and "b" in buffers


# @spec FILE-BUF-005
def test_no_buffer_exists_for_a_request_never_scheduled():
    buffers = new_buffers()
    # a buffer is created only by take(), at the first schedule; an abort
    # before that delivers a finished ID the buffers never saw
    buffers.free_finished(["aborted-before-first-schedule"])
    assert "aborted-before-first-schedule" not in buffers


# ---------------------------------------------------------------------------
# FILE-ROW step rule


def actions():
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileAction

    return FileAction


def reference(queue_len: int, chunks_remaining: int, tail_done: bool, A):
    if queue_len > 0:
        return A.DRAIN
    if chunks_remaining > 0:
        return A.CHUNK
    if not tail_done:
        return A.TAIL
    return A.END


# @spec FILE-ROW-002
@pytest.mark.parametrize("queue_len", [0, 1, 70, 141])
@pytest.mark.parametrize("chunks_remaining", [0, 1, 6428])
@pytest.mark.parametrize("tail_done", [False, True])
def test_step_rule_checks_in_order(queue_len, chunks_remaining, tail_done):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    assert next_file_action(
        queue_len=queue_len, chunks_remaining=chunks_remaining, tail_done=tail_done
    ) == reference(queue_len, chunks_remaining, tail_done, A)


# @spec FILE-ROW-002
def test_action_members_are_the_four_steps():
    assert {m.name for m in actions()} == {"DRAIN", "CHUNK", "TAIL", "END"}


# @spec FILE-ROW-003
def test_regular_chunk_follows_a_drained_queue():
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    # chunks remaining and an empty queue: a chunk, even when the tail is
    # still to do (the tail comes only after the last regular chunk)
    assert next_file_action(queue_len=0, chunks_remaining=3, tail_done=False) == A.CHUNK
    assert next_file_action(queue_len=2, chunks_remaining=3, tail_done=False) == A.DRAIN


# @spec FILE-ROW-004
def test_tail_follows_the_last_chunk_and_runs_once():
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    assert next_file_action(queue_len=0, chunks_remaining=0, tail_done=False) == A.TAIL
    # queued labels of the last chunk drain before the tail starts
    assert next_file_action(queue_len=4, chunks_remaining=0, tail_done=False) == A.DRAIN
    assert next_file_action(queue_len=0, chunks_remaining=0, tail_done=True) == A.END


# @spec FILE-ROW-005, FILE-ROW-008
@pytest.mark.parametrize("chunks_remaining", [0, 5])
@pytest.mark.parametrize("tail_done", [False, True])
def test_end_only_after_tail_and_empty_queue_and_audio_only_when_queue_empty(
    chunks_remaining, tail_done
):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    for queue_len in range(0, 142):
        action = next_file_action(
            queue_len=queue_len, chunks_remaining=chunks_remaining, tail_done=tail_done
        )
        if action == A.END:
            assert tail_done and queue_len == 0 and chunks_remaining == 0
        if action in (A.CHUNK, A.TAIL):
            assert queue_len == 0


# @spec FILE-ROW-005
@pytest.mark.parametrize("chunks", [0, 1, 3])
@pytest.mark.parametrize("queued_per_advance", [0, 1, 141])
def test_simulated_row_reaches_end_exactly_once_after_all_audio(
    chunks, queued_per_advance
):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    queue, remaining, tail_done = 0, chunks, False
    trace = []
    for _ in range(10_000):
        action = next_file_action(
            queue_len=queue, chunks_remaining=remaining, tail_done=tail_done
        )
        trace.append(action)
        if action == A.END:
            break
        if action == A.DRAIN:
            queue -= 1
        elif action == A.CHUNK:
            remaining -= 1
            queue += queued_per_advance
            queue = max(queue - 1, 0) if queued_per_advance else queue  # emits first
        else:
            tail_done = True
            queue += queued_per_advance
            queue = max(queue - 1, 0) if queued_per_advance else queue
    assert trace[-1] == A.END
    assert trace.count(A.END) == 1
    assert trace.count(A.CHUNK) == chunks
    assert trace.count(A.TAIL) == 1
    assert trace.index(A.TAIL) > max(
        [i for i, a in enumerate(trace) if a == A.CHUNK], default=-1
    )


# @spec FILE-ROW-006
def test_empty_file_runs_only_tail_and_end():
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import next_file_action

    A = actions()
    # N = 0: C = 0, and the tail queues nothing
    first = next_file_action(queue_len=0, chunks_remaining=0, tail_done=False)
    second = next_file_action(queue_len=0, chunks_remaining=0, tail_done=True)
    assert [first, second] == [A.TAIL, A.END]


# ---------------------------------------------------------------------------
# FILE-ENG-001, FILE-BUF-002, FILE-ROW-002: the production step logic,
# instrumented. Layout addition: file_rows.plan_file_step returning a
# FileStepPlan(action, start, length) in samples.

CHUNK = 8960  # 560 ms


def plan_all(total: int, queued_per_advance: int, buffers, request_id: str = "f"):
    """Drive plan_file_step for one request to its end; record each step."""
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import (
        FileAction,
        plan_file_step,
    )

    position, queue, tail_done = 0, 0, False
    steps = []
    for _ in range(100_000):
        plan = plan_file_step(
            queue_len=queue,
            position=position,
            total_samples=total,
            chunk_samples=CHUNK,
            tail_done=tail_done,
        )
        staged = 0
        if plan.length:
            staged = buffers.stage(request_id, plan.start, plan.length).shape[0]
        steps.append((plan, staged))
        if plan.action == FileAction.END:
            break
        if plan.action == FileAction.DRAIN:
            queue -= 1
        else:
            position = plan.start + plan.length
            tail_done = plan.action == FileAction.TAIL
            queue += queued_per_advance
            queue = max(queue - 1, 0)  # the first queued token is emitted now
    return steps


# @spec FILE-ENG-001, FILE-BUF-002, FILE-ROW-002
@pytest.mark.parametrize("queued", [0, 1, 40, 141])
@pytest.mark.parametrize("total", [0, 1, CHUNK - 1, CHUNK, CHUNK * 5, CHUNK * 5 + 4321])
def test_each_step_advances_and_stages_at_most_one_chunk(total, queued):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileAction

    buffers = new_buffers()
    buffers.take("f", 1, 1, samples(total), make_header(total))
    steps = plan_all(total, queued, buffers)
    c, t = divmod(total, CHUNK)

    for plan, staged in steps:
        assert staged == plan.length <= CHUNK  # never more than one chunk
        if plan.action in (FileAction.DRAIN, FileAction.END):
            assert plan.length == 0 and staged == 0  # no audio while draining
    actions = [p.action for p, _ in steps]
    assert actions.count(FileAction.CHUNK) == c
    assert actions.count(FileAction.TAIL) == 1
    assert actions[-1] == FileAction.END
    # chunks then tail are contiguous and cover the file exactly once
    advancing = [p for p, _ in steps if p.length or p.action == FileAction.TAIL]
    assert [p.start for p in advancing] == [i * CHUNK for i in range(c)] + [c * CHUNK]
    assert sum(s for _, s in steps) == total
    assert [p.length for p in advancing][-1] == t


# ---------------------------------------------------------------------------
# FILE-ROW-013: decision legality in the transaction's emission check


def _emission_case(continuation: int, eos: int, blank: int):
    """Two clean rows of one step: row 0 a stream row in the EOU role (whose
    decision only the legality rule constrains), row 1 a file CHUNK row whose
    chunk queued nothing; both decide the continuation token."""
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        ROLE_CHUNK,
        ROLE_EOU,
        AdvanceResult,
        EmissionContext,
        EmissionProjection,
    )

    cap, n = 8, 2
    context = EmissionContext(
        roles=torch.tensor([ROLE_EOU, ROLE_CHUNK]),
        input_ids=torch.tensor([eos, blank + 4]),
        chunk_rows=torch.tensor([1]),
        queue=torch.zeros(n, cap, dtype=torch.int32),
        book=torch.zeros(n, 7, dtype=torch.int32),
        prompt_index=torch.zeros(n, dtype=torch.int64),
        row_status=torch.zeros(n, dtype=torch.int32),
        file_rows=torch.tensor([False, True]),
    )
    result = AdvanceResult(
        token_ids=torch.zeros(1, 4, dtype=torch.int32),
        token_lengths=torch.zeros(1, dtype=torch.int32),
        row_status=torch.zeros(1, dtype=torch.int32),
    )
    rows = torch.zeros(n, 4)
    rows[:, 0] = float(continuation)
    projection = EmissionProjection(
        rows=rows,
        queue=context.queue.clone(),
        book=context.book.clone(),
        row_status=context.row_status.clone(),
    )
    return result, context, projection


# @spec FILE-ROW-013
def test_continuation_is_illegal_on_a_stream_row_even_with_its_id_supplied():
    """Layout constraint: ``advance._mrv1_projection_invariant_rows`` is the
    transaction's emission check; its result marks the rows whose decision or
    scratch the check rejects."""
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
        _mrv1_projection_invariant_rows,
    )

    blank = 40
    continuation, eos = blank + 5, blank + 1
    result, context, projection = _emission_case(continuation, eos, blank)
    bad = _mrv1_projection_invariant_rows(
        result,
        context,
        projection,
        torch.zeros(2, dtype=torch.int32),
        park_id=eos,
        blank_id=blank,
        eou_token_id=blank + 3,
        continuation_id=continuation,
    )
    # the stream row (file bit clear) may not decide the continuation token;
    # the file CHUNK row that queued nothing must
    assert bad.tolist() == [True, False]
