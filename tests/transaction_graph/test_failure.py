# SPDX-License-Identifier: Apache-2.0
"""Failure containment (TXG-FAIL-001 to TXG-FAIL-010).

Specs that hold with transaction graphs on or off run on CPU with the option
off (``build_off``, the ``--enforce-eager`` harness) and, where replay
matters, on CUDA with it on. Latch and delivery behaviour with the option on
runs on CPU through the fake-capture seam: a fake cell's replay computes
nothing, so those tests read only whether the step raised, latched,
cancelled, launched or fell back, never its outputs.

A transaction-validation rejection is made by a stream's first unit that
carries a later chunk sequence (the device protocol check rejects it). A
planning-failed row is a whole file whose header names another geometry
(FILE-PROMPT-007) or disagrees with its samples (a malformed item), or a
stream's first unit at another geometry (SESS-MODEL-002).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from support import model_harness as h
from support import txgcap_harness as t

from vllm_nemotron_asr_streaming.nemotron_asr.frontend import ROW_STATUS_SEQUENCE

pytestmark = pytest.mark.engine

TESTS = Path(__file__).resolve().parents[1]


def ready_off(env, capacity=8):
    env.runner.profile_run(core_embeds=False)
    bindings = env.runner.bind_pool(
        h.real_bindings(h.model_spec(env.model), capacity=capacity)
    )
    return bindings


def step(env, *new, cached=(), handles=None, **kw):
    if handles is None:
        handles = {r.req_id: i + 1 for i, r in enumerate(new)}
    return env.runner.real_step(
        h.scheduler_output(new=tuple(new), cached=cached, handles=handles, **kw)
    )


def planning_failed(env, kind, req_id):
    if kind == "file-geometry":
        return t.file_item(env, req_id, geometry=t.foreign_geometry(env))
    if kind == "file-malformed":
        return t.file_item(env, req_id, n_samples=4000, header_samples=3999)
    if kind == "stream-geometry":
        return t.stream_unit(env, req_id, geometry=t.foreign_geometry(env))
    raise AssertionError(kind)


PLANNING_FAILURES = ["file-geometry", "file-malformed", "stream-geometry"]


# ---------------------------------------------------------------------------
# Per-row rejections (on or off)


# @spec TXG-FAIL-001
def test_a_rejected_row_keeps_its_pages_while_the_others_commit(off_cpu):
    env = off_cpu()
    bindings = ready_off(env)
    fails = t.FailSpy(bindings)
    before = t.slots_at_step_start(bindings, (1, 2))
    step(env, t.stream_unit(env, "good"), t.stream_unit(env, "bad", sequence=3))
    statuses, failed = env.runner.commit_log[-1]
    # the intended outcome first: good accepted, bad rejected by the
    # device protocol check for its chunk sequence
    assert statuses.get("good", 0) == 0, statuses
    assert statuses.get("bad", 0) & ROW_STATUS_SEQUENCE, statuses
    assert torch.equal(h.slot_bytes(bindings, 2), before[2]), "the rejected row stored"
    assert not torch.equal(h.slot_bytes(bindings, 1), before[1]), (
        "the good row did not commit"
    )
    assert fails.failed == {"bad"}, fails.calls
    step(env, t.stream_unit(env, "next"), handles={"next": 3})  # no latch


# @spec TXG-FAIL-007
def test_step_outputs_do_not_depend_on_scratch_left_by_an_earlier_step(off_cpu):
    first = off_cpu()
    ready_off(first)
    step(first, t.stream_unit(first, "a"), t.stream_unit(first, "b"))
    hidden_after, _ = step(first, t.stream_unit(first, "y"), handles={"y": 3})
    fresh = off_cpu()
    ready_off(fresh)
    hidden_alone, _ = step(fresh, t.stream_unit(fresh, "y"), handles={"y": 3})
    assert torch.equal(hidden_after, hidden_alone), "an earlier step's scratch leaked"
    assert torch.equal(
        h.slot_bytes(first.runner.bindings, 3), h.slot_bytes(fresh.runner.bindings, 3)
    )


# ---------------------------------------------------------------------------
# Planning-failed rows and the all-failed bypass (on or off)


@pytest.mark.parametrize("kind", PLANNING_FAILURES)
# @spec TXG-FAIL-008
def test_planning_failed_rows_fail_from_the_planning_failed_set(off_cpu, kind):
    env = off_cpu()
    bindings = ready_off(env)
    fails = t.FailSpy(bindings)
    t.silence_collection(env, {"bad"})  # no status reaches collection
    before = t.slots_at_step_start(bindings, (2,))
    step(env, t.stream_unit(env, "good"), planning_failed(env, kind, "bad"))
    assert fails.failed == {"bad"}, fails.calls
    assert torch.equal(h.slot_bytes(bindings, 2), before[2]), (
        "the failed row's pages changed"
    )


@pytest.mark.parametrize("kind", PLANNING_FAILURES)
# @spec TXG-FAIL-008
def test_planning_failed_rows_fail_once_when_collection_also_reports_them(
    off_cpu, kind
):
    env = off_cpu()
    bindings = ready_off(env)
    fails = t.FailSpy(bindings)
    step(env, t.stream_unit(env, "good"), planning_failed(env, kind, "bad"))
    assert fails.failed == {"bad"}, fails.calls


@pytest.mark.parametrize("kind", PLANNING_FAILURES)
# @spec TXG-FAIL-009
def test_an_all_failed_step_runs_no_transaction_and_returns_rejected_rows(
    off_cpu, kind
):
    env = off_cpu()
    bindings = ready_off(env)
    # the rejected-row output, from a mixed step that reaches the transaction
    mixed, _ = step(env, t.stream_unit(env, "good"), planning_failed(env, kind, "bad"))
    reference = mixed[1].clone()
    fails = t.FailSpy(bindings)
    advances = len(env.advance.calls)
    hidden, _ = step(
        env,
        planning_failed(env, kind, "b1"),
        planning_failed(env, kind, "b2"),
        handles={"b1": 3, "b2": 4},
    )
    assert len(env.advance.calls) == advances, "an all-failed step ran the transaction"
    assert hidden.shape[0] == 2
    for row in hidden:
        assert torch.equal(row, reference), "a row is not the rejected-row output"
    assert fails.failed == {"b1", "b2"}, fails.calls


# @spec TXG-FAIL-009
def test_an_all_failed_step_launches_nothing_with_the_option_on(on_cpu, monkeypatch):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    calls = []
    t.wrap_execute(monkeypatch, env.model, lambda call: calls.append(1) or call())
    fails = t.FailSpy(core.serving_bindings)
    replays, advances = env.fake.replay_count, len(env.advance.calls)
    core.step(planning_failed(env, "file-malformed", "b1"))
    assert calls == [], "an all-failed step called execute"
    assert env.fake.replay_count == replays and len(env.advance.calls) == advances
    assert fails.failed == {"b1"}


# ---------------------------------------------------------------------------
# Worker-fatal failures and the latch (option on)


def latch_by_fault(env, core, where):
    """Run a split step (a CHUNK row and a planning-failed echo row) whose
    ``where`` fails; return the error."""
    fake = env.fake
    if where == "first-launch":
        fake.fail["replay"] = fake.replay_count
    elif where == "between-launches":
        fake.fail["replay"] = fake.replay_count + 1
    elif where == "commit-stage":
        from vllm_nemotron_asr_streaming.nemotron_asr import commit_sink

        original = commit_sink.CommitTicket.stage

        def stage(self, *args, **kwargs):
            commit_sink.CommitTicket.stage = original
            raise RuntimeError("injected fault in the commit stage")

        commit_sink.CommitTicket.stage = stage
    return h.call_with(
        core.step,
        t.stream_unit(env, "s1"),
        planning_failed(env, "file-malformed", "f1"),
    )


@pytest.mark.parametrize("where", ["first-launch", "between-launches", "commit-stage"])
# @spec TXG-FAIL-002
def test_a_failure_from_the_first_launch_on_latches_and_cancels(
    on_cpu, monkeypatch, where
):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    sink = t.SinkSpy().install(monkeypatch)
    replays, advances = env.fake.replay_count, len(env.advance.calls)
    if where == "commit-stage":
        from vllm_nemotron_asr_streaming.nemotron_asr import commit_sink

        monkeypatch.setattr(
            commit_sink.CommitTicket, "stage", commit_sink.CommitTicket.stage
        )
    error = latch_by_fault(env, core, where)
    assert error is not None, "the failure did not propagate"
    assert (
        "reserve" in sink.log and "cancel" in sink.log[sink.log.index("reserve") :]
    ), sink.log
    launched = env.fake.replay_count - replays
    assert (
        launched == {"first-launch": 1, "between-launches": 2, "commit-stage": 2}[where]
    ), f"{launched} launches: a launch was retried or skipped"
    assert len(env.advance.calls) == advances, "a half of the step fell back"
    later = h.call_with(core.step, t.stream_unit(env, "s2"), handles={"s2": 3})
    assert later is not None and t.FRESH_ENGINE.search(t.text(later)), later


# @spec TXG-FAIL-003
def test_a_latched_binding_refuses_every_later_step(on_cpu):
    run = on_cpu(n=2, max_num_seqs=3)
    env, core = run.env, run.core
    core.startup()
    assert latch_by_fault(env, core, "first-launch") is not None
    replays, advances = env.fake.replay_count, len(env.advance.calls)
    shapes = [
        [t.stream_unit(env, "a")],  # would be covered
        [t.stream_unit(env, f"k{i}") for i in range(3)],  # would fall back (K > N)
    ]
    # Each request takes a slot no earlier request holds (the latching step
    # holds 1 and 2): core refuses a held slot at the same generation before
    # the model runs.
    slots = iter(range(3, 3 + sum(len(rows) for rows in shapes)))
    for rows in shapes:
        handles = {r.req_id: next(slots) for r in rows}
        error = h.call_with(core.step, *rows, handles=handles)
        assert error is not None and t.FRESH_ENGINE.search(t.text(error)), error
        assert "transaction graphs failed" in t.text(error).lower(), t.text(error)
    assert env.fake.replay_count == replays, "a latched binding launched a cell"
    assert len(env.advance.calls) == advances, (
        "a latched binding ran the option-off path"
    )


# @spec TXG-FAIL-005
def test_a_launched_step_completes_its_stage_and_collection(on_cpu, monkeypatch):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    sink = t.SinkSpy().install(monkeypatch)
    core.step(t.stream_unit(env, "s1"))
    assert (
        sink.log.count("reserve") == 1 and "stage" in sink.log and "collect" in sink.log
    )
    assert "cancel" not in sink.log, "a launched step's reservation was cancelled"
    # the request is aborted: core finishes it and fences its slot next step
    env.runner.real_step(
        h.scheduler_output(finished=("s1",), cleanup={"s1": 1}, handles={})
    )
    assert "cancel" not in sink.log


# @spec TXG-FAIL-010
def test_the_latch_is_checked_before_the_all_failed_bypass(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    assert latch_by_fault(env, core, "first-launch") is not None
    error = h.call_with(
        core.step, planning_failed(env, "file-malformed", "b1"), handles={"b1": 4}
    )
    assert error is not None and t.FRESH_ENGINE.search(t.text(error)), (
        "an all-failed step completed on a latched worker"
    )
    # a cleanup-only step still completes
    env.runner.real_step(
        h.scheduler_output(finished=("s1",), cleanup={"s1": 1}, handles={})
    )


# ---------------------------------------------------------------------------
# CUDA


def off_engine(on_cuda):
    """``B_hook``: the same CUDA configuration with the option absent."""
    return on_cuda(n=2).option_off()


def schedule(env):
    """Covered steps of both kinds and a split step, interleaved."""
    return [
        dict(new=(t.stream_unit(env, "s1"),), handles={"s1": 1}),
        dict(new=(), cached=("s1",), handles={"s1": 1}),
        dict(
            new=(t.stream_unit(env, "s2"),), cached=("s1",), handles={"s1": 1, "s2": 2}
        ),
        dict(new=(), cached=("s1", "s2"), handles={"s1": 1, "s2": 2}),
    ]


def run_schedule(env):
    env.core.startup()
    outputs = []
    for item in schedule(env):
        hidden, _ = env.core.step(
            *item["new"], cached=item.get("cached", ()), handles=item["handles"]
        )
        outputs.append(hidden.clone())
    return (
        outputs,
        [dict(s) for s, _ in env.runner.commit_log],
        h.pool_tensor(env.runner.bindings).clone(),
    )


@pytest.mark.gpu
# @spec TXG-FAIL-006, TXG-FAIL-007
def test_interleaved_replays_equal_the_option_off_engine_cuda(on_cuda):
    on_outputs, on_status, on_pool = run_schedule(on_cuda(n=2))
    off_outputs, off_status, off_pool = run_schedule(off_engine(on_cuda))
    for a, b in zip(on_outputs, off_outputs, strict=True):
        assert torch.equal(a, b), "a replayed step read an intermediate across replays"
    assert on_status == off_status
    assert torch.equal(on_pool, off_pool)


@pytest.mark.gpu
# @spec TXG-FAIL-001
def test_a_rejected_row_under_replay_keeps_its_pages_cuda(on_cuda):
    env = on_cuda(n=2)
    core = env.core
    core.startup()
    bindings = core.serving_bindings
    fails = t.FailSpy(bindings)
    before = t.slots_at_step_start(bindings, (1, 2))
    core.step(t.stream_unit(env, "good"), t.stream_unit(env, "bad", sequence=3))
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("good", 0) == 0 and statuses.get("bad", 0) != 0
    assert torch.equal(h.slot_bytes(bindings, 2), before[2])
    assert not torch.equal(h.slot_bytes(bindings, 1), before[1])
    assert fails.failed == {"bad"}
    core.step(t.stream_unit(env, "next"), handles={"next": 3})


FAULT_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import pytest
    import torch
    from support import txg_gpu as g
    from support import txgcap_harness as t
    from support.distributed import single_process_distributed

    tmp = Path(sys.argv[1])
    with pytest.MonkeyPatch.context() as mp, single_process_distributed(
        tmp / "dist", "nccl"
    ):
        env = g.build_option_on(tmp, n=2, startup=False, monkeypatch=mp)
        core = env.core
        core.startup()
        launches = []
        original = torch.cuda.CUDAGraph.replay

        def replay(self):
            launches.append(1)
            original(self)
            index = torch.tensor([1 << 40], device="cuda")
            torch.zeros(1, device="cuda")[index]  # asynchronous device fault

        mp.setattr(torch.cuda.CUDAGraph, "replay", replay)
        advances = len(env.advance.calls)
        try:
            core.step(t.stream_unit(env, "s1"))
        except Exception as error:  # noqa: BLE001
            print("PROPAGATED", type(error).__name__)
        else:
            print("SWALLOWED")
        print("LAUNCHES", len(launches))
        print("FALLBACKS", len(env.advance.calls) - advances)
    """
)


@pytest.mark.gpu
# @spec TXG-FAIL-004
def test_an_asynchronous_device_fault_propagates_without_retry_or_fallback(tmp_path):
    script = tmp_path / "fault_scenario.py"
    script.write_text(FAULT_SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = result.stdout
    assert "PROPAGATED" in out, out + result.stderr[-4000:]
    assert "LAUNCHES 1" in out, out
    assert "FALLBACKS 0" in out, out
