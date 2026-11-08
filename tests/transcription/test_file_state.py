# SPDX-License-Identifier: Apache-2.0
"""FILE-BUF and FILE-ROW-001 through the model's real ModelState hooks.

The state is built on CPU the way core builds it (``file_model_state``) and
fed real ``SchedulerOutput`` structures to ``prepare_step``, the hook core
calls first in every step, including a step with no scheduled tokens
(``vllm/v1/worker/gpu/model_runner.py`` ``execute_model``). The state's
``file_buffers`` attribute (``FileHostBuffers``) is a layout addition.
"""

from __future__ import annotations

import numpy as np
import pytest
from support.card import BLANK, LOCALES

pytestmark = pytest.mark.engine

FILE_TOKEN = BLANK + 4
STREAM_TOKEN = BLANK + 2


def audio(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def file_request(make_step, req_id, samples, geometry=3, row=0):
    return make_step.new_request(
        req_id, FILE_TOKEN, samples=samples, header=[geometry, row, len(samples)]
    )


def run_step(state, scheduler_output):
    state.prepare_step(scheduler_output, dummy_run=False, is_profile=False)


# @spec FILE-BUF-001, FILE-ROW-001
def test_first_schedule_takes_samples_into_a_request_owned_buffer(
    file_model_state, make_step
):
    state = file_model_state()
    samples = audio(20_000)
    request = file_request(make_step, "f1", samples, row=LOCALES["en-GB"])
    original = samples.copy()
    run_step(
        state, make_step.step(new=[request], handles={"f1": make_step.handle(2, 1)})
    )
    assert "f1" in state.file_buffers
    np.testing.assert_array_equal(state.file_buffers.stage("f1", 0, 20_000), original)

    assert state.file_buffers.header("f1").dtype == np.int64
    assert state.file_buffers.header("f1").tolist() == [3, LOCALES["en-GB"], 20_000]

    # independent of core's copy: mutate the feature's tensors (samples and
    # header), then let core release the item and its encoder-cache entry
    samples[:] = 0.0
    request.mm_features[0].data["samples"].data.zero_()
    request.mm_features[0].data["file_header"].data.zero_()
    run_step(
        state,
        make_step.step(
            cached=["f1"],
            handles={"f1": make_step.handle(2, 1)},
            free_hashes=["uuid-f1"],
            updates={"f1": {0: None}},
        ),
    )
    np.testing.assert_array_equal(state.file_buffers.stage("f1", 0, 20_000), original)
    # the header (geometry, locale row, sample count) is the buffer's own too
    assert state.file_buffers.header("f1").tolist() == [3, LOCALES["en-GB"], 20_000]


# @spec FILE-ROW-001
def test_stream_carrier_request_gets_no_file_buffer(file_model_state, make_step):
    state = file_model_state()
    request = make_step.new_request("s1", STREAM_TOKEN)
    run_step(
        state,
        make_step.step(new=[request], handles={"s1": make_step.handle(1, 1)}),
    )
    assert "s1" not in state.file_buffers


# @spec FILE-BUF-003
@pytest.mark.parametrize("with_other_work", [False, True])
def test_finished_request_frees_buffer_even_without_forward_work(
    file_model_state, make_step, with_other_work
):
    state = file_model_state()
    run_step(
        state,
        make_step.step(
            new=[
                file_request(make_step, "f1", audio(500)),
                file_request(make_step, "f2", audio(500, 1)),
            ],
            handles={"f1": make_step.handle(1, 1), "f2": make_step.handle(2, 1)},
        ),
    )
    assert "f1" in state.file_buffers and "f2" in state.file_buffers

    if with_other_work:
        out = make_step.step(
            cached=["f2"], finished=["f1"], handles={"f2": make_step.handle(2, 1)}
        )
    else:
        # a step whose only content is the finished IDs: nothing scheduled
        out = make_step.step(finished=["f1"])
        assert out.total_num_scheduled_tokens == 0 and not out.num_scheduled_tokens
    run_step(state, out)
    assert "f1" not in state.file_buffers
    assert "f2" in state.file_buffers


# @spec FILE-BUF-003
def test_preempted_or_aborted_ids_are_just_finished_ids(file_model_state, make_step):
    state = file_model_state()
    run_step(
        state,
        make_step.step(
            new=[file_request(make_step, "f1", audio(500))],
            handles={"f1": make_step.handle(1, 1)},
        ),
    )
    run_step(state, make_step.step(finished=["f1"]))
    assert "f1" not in state.file_buffers
    run_step(state, make_step.step(finished=["f1"]))  # a repeat is harmless


# @spec FILE-BUF-004
def test_new_generation_of_a_slot_drops_the_old_buffer(file_model_state, make_step):
    state = file_model_state()
    run_step(
        state,
        make_step.step(
            new=[file_request(make_step, "old", audio(500))],
            handles={"old": make_step.handle(5, 1)},
        ),
    )
    assert "old" in state.file_buffers
    # the engine did not report "old" finished; the slot comes back with a new
    # generation for another request
    run_step(
        state,
        make_step.step(
            new=[file_request(make_step, "new", audio(500, 1))],
            handles={"new": make_step.handle(5, 2)},
        ),
    )
    assert "old" not in state.file_buffers
    assert "new" in state.file_buffers


# @spec FILE-BUF-005
def test_request_aborted_before_first_schedule_creates_no_buffer(
    file_model_state, make_step
):
    state = file_model_state()
    # core reports an abort of a request the worker never saw as a finished ID
    run_step(state, make_step.step(finished=["never-scheduled"]))
    assert "never-scheduled" not in state.file_buffers
    # and it does not disturb a request scheduled in the same step
    run_step(
        state,
        make_step.step(
            new=[file_request(make_step, "f1", audio(500))],
            finished=["never-scheduled"],
            handles={"f1": make_step.handle(1, 1)},
        ),
    )
    assert "f1" in state.file_buffers and "never-scheduled" not in state.file_buffers


# ---------------------------------------------------------------------------
# FILE-ENG-001, FILE-BUF-002, FILE-ROW-002, FILE-ENG-003 on the production path.
#
# Layout addition: ``state.plan_step(scheduler_output, *, queue_lens=None,
# stream_rows=None) -> StepPlan`` with ``.rows`` (list of ``file_rows.StepRow``)
# and ``.groups`` (``plan_chunk_groups(rows)``). It is the step planning the
# model runner makes before the forward: it resolves FILE rows from the
# buffers, stages each advancing row's samples through
# ``state.file_buffers.stage`` and advances the row's position.
# ``queue_lens`` maps request IDs to the label queue length the device book
# reports (default 0); ``stream_rows`` maps stream request IDs to the
# ``(geometry_index, deadline)`` the existing header resolution produced.

GEOMETRIES = ((0, 0), (1, 1), (3, 2), (6, 3), (13, 4))  # (right context, index)


class StageProbe:
    """Layout constraint: FileHostBuffers.stage is the single host-to-device
    route for FILE samples, so wrapping it sees every staged range. Wraps the
    real ``FileHostBuffers.stage`` and records every call."""

    def __init__(self, buffers, monkeypatch=None):
        self.calls: list[tuple[str, int, int]] = []
        self.copies: list[int] = []  # samples copied to the device, per call
        if monkeypatch is not None:
            # layout: file_rows.copy_chunk_to_device(view, *, device) is the
            # only host-to-device copy of FILE samples (looked up at call time)
            from vllm_nemotron_asr_streaming.nemotron_asr import file_rows

            real_copy = file_rows.copy_chunk_to_device

            def copy(view, *, device):
                self.copies.append(int(view.shape[0]))
                return real_copy(view, device=device)

            monkeypatch.setattr(file_rows, "copy_chunk_to_device", copy)
        real = buffers.stage

        def stage(request_id, start, length):
            self.calls.append((request_id, int(start), int(length)))
            return real(request_id, start, length)

        buffers.stage = stage

    def take(self) -> list[tuple[str, int, int]]:
        calls, self.calls = self.calls, []
        return calls

    def take_copies(self) -> list[int]:
        copies, self.copies = self.copies, []
        return copies


# @spec FILE-ENG-001, FILE-BUF-002, FILE-ROW-002
@pytest.mark.parametrize("right_context, geometry", GEOMETRIES)
def test_production_step_plan_gives_a_file_row_at_most_one_chunk(
    file_model_state, make_step, monkeypatch, right_context, geometry
):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileAction

    chunk = 1280 * (right_context + 1)
    total = chunk * 4 + 321
    state = file_model_state(right_context)
    probe = StageProbe(state.file_buffers, monkeypatch)
    request = make_step.new_request(
        "f1", FILE_TOKEN, samples=audio(total), header=[geometry, 0, total]
    )
    handles = {"f1": make_step.handle(1, 1)}
    # the label queue holds 3 tokens after the second advance, then drains
    queue_after_advance = {2: 3}
    queue_len, advances, staged_total, actions = 0, 0, 0, []
    # Layout constraint: FileHostBuffers.stage is the single host-to-device
    # route for FILE samples, so the probe sees every staged range.
    for step_no in range(40):
        out = make_step.step(
            new=[request] if step_no == 0 else [],
            cached=[] if step_no == 0 else ["f1"],
            handles=handles,
        )
        state.prepare_step(out, dummy_run=False, is_profile=False)
        plan = state.plan_step(out, queue_lens={"f1": queue_len})
        (row,) = [r for r in plan.rows if r.request_id == "f1"]
        calls = probe.take()
        copies = probe.take_copies()
        # the device copy is the staged view, once, and nothing while draining
        assert copies == [length for _, _, length in calls]
        for _, start, _ in calls:
            assert (
                start == staged_total
            )  # contiguous: each call starts where the last ended
        actions.append(row.file_plan.action)
        assert len(calls) <= 1  # at most one chunk or tail staged per step
        staged = sum(length for _, _, length in calls)
        assert staged == row.file_plan.length <= chunk
        if row.file_plan.action in (FileAction.DRAIN, FileAction.END):
            assert calls == []  # zero while draining the label queue
        if row.file_plan.action == FileAction.END:
            break
        if row.file_plan.action == FileAction.DRAIN:
            queue_len -= 1
        else:
            advances += 1
            staged_total += staged
            queue_len = max(queue_after_advance.get(advances, 0) - 1, 0)
    assert actions[-1] == FileAction.END
    assert advances == 5  # four regular chunks and the tail
    assert staged_total == total
    assert FileAction.DRAIN in actions


# The encoder dispatch (advance.advance_chunk_bucket) runs kernels and is not
# reachable on CPU: it is probed in the in-process GPU test of
# test_engine_files.py. Here the planned groups are what execution receives.


# @spec FILE-ENG-003, FILE-ENG-001
@pytest.mark.parametrize("right_context, geometry", GEOMETRIES)
def test_production_mixed_step_plans_file_and_stream_rows_together(
    file_model_state, make_step, right_context, geometry
):
    state = file_model_state(right_context)
    probe = StageProbe(state.file_buffers)
    chunk = 1280 * (right_context + 1)
    file_req = make_step.new_request(
        "f1", FILE_TOKEN, samples=audio(chunk * 3), header=[geometry, 0, chunk * 3]
    )
    stream_req = make_step.new_request("s1", STREAM_TOKEN)
    out = make_step.step(
        new=[file_req, stream_req],
        handles={"f1": make_step.handle(1, 1), "s1": make_step.handle(2, 1)},
    )
    state.prepare_step(out, dummy_run=False, is_profile=False)
    plan = state.plan_step(out, stream_rows={"s1": (geometry, 700)})
    assert {r.request_id for r in plan.rows} == {"f1", "s1"}
    (group,) = plan.groups  # one geometry, both rows in it
    assert group.geometry_index == geometry
    assert set(group.request_ids) == {"f1", "s1"}
    assert tuple(group.deadlines) == (700,)  # the file row adds no deadline
    assert probe.take() == [("f1", 0, chunk)]
