# SPDX-License-Identifier: Apache-2.0
"""Execution of covered steps on CUDA (TXG-EXEC-001 to TXG-EXEC-004,
TXG-EXEC-006, TXG-EXEC-007, TXG-EXEC-010, TXG-EXEC-012 to TXG-EXEC-015,
TXG-TXN-001, TXG-TXN-002, TXG-TXN-009, TXG-QC-002, TXG-CELL-001).

Spies observe the binding's ``execute`` (``GpuEnv.compare``), the commit sink
(``ProxySink``), the cells' replays, the classification and the phase
functions (``support.txg_api``), and ``torch.profiler`` counts the copies.
"""

from __future__ import annotations

import pytest
import torch
from support import txg_api as api
from support import txg_gpu as g

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

N = 4
KEY = api.KEY


def two_kind_step(env, driver):
    """Run steps until one covered split step ran; return its record."""
    d = driver
    d.add_stream("s1", [{}, {}, {}])
    d.add_stream("s2", [{}, {}, {}])
    d.step()
    for _ in range(40):
        d.until_parked(["s1"])
        d.step(only=["s1", "s2"])
        record = d.env.records[-1]
        k = int(record.plan.is_chunk.sum())
        if record.covered and 0 < k < len(record.plan.request_ids):
            return record
    raise AssertionError("no covered split step ran")


def spy_classify(monkeypatch, events, after=None):
    module = api.graph_module()
    real = module.classify

    def classify(plan, **kwargs):
        result = real(plan, **kwargs)
        events.append(("classify", result))
        if after is not None:
            after(result)
        return result

    monkeypatch.setattr(module, "classify", classify)


def spy_phase(monkeypatch, events, name, fail=None):
    advance = api.advance_module()
    real = api.phase_function(name)

    def phase(*args, **kwargs):
        events.append((name,))
        if fail is not None:
            raise fail
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, name, phase)


def spy_option_off(monkeypatch, calls):
    advance = api.advance_module()
    real = advance.advance_model_rows

    def off(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", off)


# @spec TXG-CELL-001
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
def test_the_serving_capture_holds_every_cell(on_cuda, lookahead):
    env = on_cuda(startup=True, lookahead=lookahead, n=N)
    cells = env.binding.cells
    assert set(cells) == set(api.cell_keys(N, lookahead))
    assert all(cell.graph is not None for cell in cells.values())


# @spec TXG-EXEC-001, TXG-TXN-009, TXG-EXEC-004
def test_execute_runs_its_steps_in_order(on_cuda, monkeypatch):
    env = on_cuda(startup=True, lookahead=0, n=N)
    events = []
    env.compare(events=events)
    env.spy_replays(events)
    spy_classify(monkeypatch, events)
    spy_phase(monkeypatch, events, api.PHASES[0])
    real_empty = torch.empty

    def empty(*args, **kwargs):
        events.append(("empty", args))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    record = two_kind_step(env, g.Driver(env))
    monkeypatch.setattr(torch, "empty", real_empty)
    assert not record.problems, record.problems
    population = len(record.plan.request_ids)
    width = record.inputs_embeds.shape[1]
    start = max(i for i, e in enumerate(events) if e[0] == "classify")
    tail = events[start:]
    kinds = [e[0] for e in tail]
    first_replay = kinds.index("replay")
    replays = [e[1] for e in tail if e[0] == "replay"]
    assert [key[1] for key in replays] == [api.CHUNK, api.ECHO]
    assert kinds.index(api.PHASES[0]) < kinds.index("reserve") < first_replay
    escaped = [
        i
        for i, e in enumerate(tail)
        if e[0] == "empty"
        and tuple(e[1][0] if len(e[1]) == 1 else e[1]) == (population, width)
    ]
    assert escaped and escaped[0] < kinds.index("reserve"), (
        "escaped rows not allocated first"
    )
    assert kinds.index("stage") > max(i for i, k in enumerate(kinds) if k == "replay")
    assert kinds.count("reserve") == 1 and kinds.count("stage") == 1


# @spec TXG-EXEC-002
def test_a_declined_step_reserves_nothing_and_changes_no_page(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=2, max_num_seqs=4)
    records = env.compare()
    d = g.Driver(env)
    for r in ("a", "b", "c"):
        d.add_stream(r, [{}])
    d.step()  # three CHUNK rows, K > N
    declined = [r for r in records if not r.covered]
    assert declined, "the step was covered"
    for record in declined:
        assert not record.problems, record.problems
        assert not record.sink.reservations


@pytest.mark.parametrize("where", ["prepare", "reservation"])
# @spec TXG-EXEC-003
def test_a_pre_launch_failure_propagates_without_latching(on_cuda, monkeypatch, where):
    from support import model_harness as h
    from support import txgcap_harness as t

    env = on_cuda(startup=True, lookahead=0, n=N)
    events = []
    injected = RuntimeError(f"injected at {where}")

    class FailingSink:
        def __init__(self, real):
            self.real = real

        def reserve(self, plan):
            raise injected

    def mutate(kwargs, plan):
        if where == "reservation":
            kwargs["commit_sink"] = FailingSink(kwargs["commit_sink"])
        return kwargs, plan

    records = env.compare(mutate=mutate, events=events)
    env.spy_replays(events)
    d = g.Driver(env)
    d.add_stream("s1", [{}])
    if where == "prepare":
        spy_phase(monkeypatch, events, api.PHASES[0], fail=injected)
    # The pages as the step's transaction finds them: the fresh-page
    # initialization of the new request's slot (REG-STATE-010) precedes it.
    start = t.pool_at_step_start(env)
    with pytest.raises(RuntimeError, match="injected"):
        d.step()
    assert torch.equal(start["pool"], h.pool_tensor(env.bindings)), "a page changed"
    assert env.binding.failed is False
    assert not [e for e in events if e[0] == "replay"]
    record = records[-1]
    assert all(t.cancelled for t in record.sink.reservations)


# @spec TXG-EXEC-004
def test_escaped_rows_belong_to_the_caller_and_survive_later_steps(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}, {}])
    d.add_stream("s2", [{}, {}])
    d.step()
    first = next(r for r in records if r.covered)
    kept = first.rows.clone()
    d.run(20)
    assert sum(r.covered for r in records) > 1
    assert g.tx.same_bits(first.rows, kept), (
        "a later step overwrote earlier escaped rows"
    )


# @spec TXG-EXEC-006, TXG-EXEC-007, TXG-EXEC-011
def test_one_host_to_device_copy_per_covered_step(on_cuda, monkeypatch):
    env = on_cuda(startup=True, lookahead=0, n=N)
    profiles = []
    binding = env.binding
    real = binding.execute

    def profiled(*args, **kwargs):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            out = real(*args, **kwargs)
            torch.cuda.synchronize()
        profiles.append((out is not None, prof))
        return out

    monkeypatch.setattr(binding, "execute", profiled)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_file("f1", 5000)
    d.run(10)
    covered = [prof for ok, prof in profiles if ok]
    assert covered
    for prof in covered:
        counts = g.memcpy_counts(prof)
        assert counts["HtoD"] == 1, counts
        assert counts["DtoD"] >= 2, counts


# @spec TXG-EXEC-007, TXG-COV-002
def test_covered_steps_are_one_token_steps_read_from_the_static_inputs(on_cuda):
    """Every covered step schedules one token per row, so its ``input_ids``
    and ``inputs_embeds`` hold ``P`` entries, row ``i`` at step position
    ``i``, which the two device copies place in the first ``P`` rows of the
    static inputs; the cells' gathers from them by step position reproduce
    the option-off transaction (``GpuEnv.compare``)."""
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}, {}])
    d.add_stream("s2", [{}, {}])
    d.add_file("f1", 5000)
    d.run(15)
    covered = [r for r in records if r.covered]
    assert covered, "no step was covered"
    for record in covered:
        population = len(record.plan.request_ids)
        assert record.inputs_embeds.shape[0] == population
        assert record.input_ids is None or record.input_ids.shape[0] == population
        assert not record.problems, record.problems


# @spec TXG-EXEC-010
def test_the_reservation_reads_the_views_bound_at_the_step(on_cuda):
    """Every resident pool a covered step reserves and commits through is a
    view of the state layer's per-state views as core bound them at that step
    (``component_pools``)."""
    from support import txgcap_harness as t

    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    seen = []
    real = env.binding.execute

    def execute(core, input_ids, inputs_embeds, plan, **kwargs):
        layer = t.state_layer_of(env)
        views = list(layer.component_pools.values())
        seen.append((views, g.tx.pools_of(kwargs)))
        return real(core, input_ids, inputs_embeds, plan, **kwargs)

    env._patches.setattr(env.binding, "execute", execute)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_stream("s2", [{}])
    d.run(8)
    assert any(r.covered for r in records)
    for views, pools in seen:
        for name, pool in pools.flat():
            assert any(t.within(view, pool) for view in views), (
                f"{name} is not a view of the state layer's bound views"
            )


# @spec TXG-EXEC-010, TXG-EXEC-003
def test_a_commit_whose_rows_disagree_with_the_step_bindings_writes_nothing(on_cuda):
    """A covered step whose row order no longer equals the step's row
    bindings (REG-STATE-004) is refused before its first launch: it
    propagates, launches no cell, changes no page and holds no reservation."""
    import dataclasses

    from support import model_harness as h
    from support import txgcap_harness as t

    env = on_cuda(startup=True, lookahead=0, n=N)
    events = []

    def swap(kwargs, plan):
        if len(plan.bindings) < 2:
            return kwargs, plan
        order = [1, 0, *range(2, len(plan.bindings))]
        index = torch.tensor(order)
        moved = dataclasses.replace(
            plan,
            bindings=tuple(plan.bindings[i] for i in order),
            request_ids=tuple(plan.request_ids[i] for i in order),
            state_indices_p=plan.state_indices_p[index],
        )
        return kwargs, moved

    records = env.compare(mutate=swap, events=events)
    env.spy_replays(events)
    d = g.Driver(env)
    d.add_stream("s1", [{}])
    d.add_stream("s2", [{}])
    # the pages as the transaction finds them, after the step's fresh-page
    # initialization (REG-STATE-010)
    start = t.pool_at_step_start(env)
    with pytest.raises(Exception):
        d.step()
    assert not [e for e in events if e[0] == "replay"], "a cell launched"
    assert records and all(r.cancelled for r in records[-1].sink.reservations)
    assert torch.equal(start["pool"], h.pool_tensor(env.bindings)), "a page changed"
    assert env.binding.failed is False


# @spec TXG-EXEC-010, TXG-QC-002
def test_one_composite_reservation_per_covered_step(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_stream("s2", [{}])
    d.add_file("f1", 4000)
    d.run(15)
    pointers = set()
    for record in [r for r in records if r.covered]:
        population = len(record.plan.request_ids)
        assert len(record.sink.reservations) == 1
        ticket = record.sink.reservations[0]
        plan = ticket.plan
        assert tuple(plan.bindings) == tuple(record.plan.bindings)
        assert plan.records == () and plan.capture is None
        assert tuple(plan.progress.shape) == (population, 3)
        counts = plan.queue_counts
        assert counts.dtype == torch.int32 and tuple(counts.shape) == (population,)
        assert counts.device.type == "cuda"
        pointers.add((plan.progress.data_ptr(), counts.data_ptr()))
        assert tuple(ticket.staged.shape) == (population,)
        events = [e[0] for e in record.sink.events]
        assert events.count("reserve") == 1 and events.count("stage") == 1
    assert len(pointers) == 1, "the reserved views moved between steps"


# @spec TXG-EXEC-012, TXG-COV-011
def test_dummy_steps_never_call_execute(on_cuda):
    env = on_cuda(startup=True, lookahead=0, n=N)
    records = env.compare()
    before = env.counters()
    env.runner.profile_run(core_embeds=False)
    env.runner.profile_run(randomize_inputs=True, attn=True, core_embeds=False)
    env.runner.dummy_step(is_profile=False)
    assert records == []
    after = env.counters()
    assert after.get("fallbacks") == before.get("fallbacks")
    assert after.get("replays") == before.get("replays")


# @spec TXG-EXEC-013
def test_a_graph_lost_after_classification_is_worker_fatal(on_cuda, monkeypatch):
    env = on_cuda(startup=True, lookahead=0, n=N)
    events, calls = [], []
    env.compare(events=events)
    env.spy_replays(events)
    binding = env.binding

    def lose(result):
        if result.reason is None:
            for key in result.cells:
                binding.cells[key].graph = None

    spy_classify(monkeypatch, events, after=lose)
    spy_option_off(monkeypatch, calls)
    d = g.Driver(env)
    d.add_stream("s1", [{}])
    with pytest.raises(Exception):
        d.step()
    assert binding.failed is True
    assert not [e for e in events if e[0] == "replay"]
    assert not calls, "the option-off transaction ran"


# @spec TXG-EXEC-014
def test_execute_uses_the_runners_current_stream(on_cuda, monkeypatch):
    env = on_cuda(startup=True, lookahead=0, n=N)
    events = []
    env.compare(events=events)
    env.spy_replays(events)
    side = torch.cuda.Stream()
    created = g.spy_stream_creation(monkeypatch)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_file("f1", 3000)
    with torch.cuda.stream(side):
        d.run(8)
    torch.cuda.synchronize()
    assert not created, "execute created a stream"
    streams = {e[-1] for e in events if e[0] in ("reserve", "stage", "replay")}
    assert streams == {side}, streams


# @spec TXG-EXEC-015
def test_execute_before_verification_raises(tmp_path, single_process_dist, monkeypatch):
    env = g.build_option_on(
        tmp_path, lookahead=0, n=N, verify=False, monkeypatch=monkeypatch
    )
    try:
        events, calls = [], []
        env.compare(events=events)
        env.spy_replays(events)
        spy_option_off(monkeypatch, calls)
        d = g.Driver(env)
        d.add_stream("s1", [{}])
        with pytest.raises(Exception) as caught:
            d.step()
        message = str(caught.value)
        assert KEY in message and ("verif" in message or "post_warmup" in message), (
            message
        )
        assert env.binding.failed is False
        assert not [e for e in events if e[0] == "replay"]
        assert not calls
    finally:
        env.shutdown()


# @spec TXG-TXN-001
def test_prepare_reads_no_device_memory(on_cuda, monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    env = on_cuda(startup=True, lookahead=0, n=N)
    env.compare()
    reads = []

    class DeviceReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            for x in g.tx.tensors_in([list(args), kwargs or {}]):
                if x.device.type == "cuda":
                    reads.append(str(func))
            return func(*args, **(kwargs or {}))

    advance = api.advance_module()
    real = api.phase_function(api.PHASES[0])

    def prepare(*args, **kwargs):
        with DeviceReads():
            return real(*args, **kwargs)

    monkeypatch.setattr(advance, api.PHASES[0], prepare)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_file("f1", 4000)
    d.run(6)
    assert not reads, f"the prepare phase read device memory: {sorted(set(reads))}"


# @spec TXG-TXN-002
def test_body_issues_no_host_to_device_copy_on_cuda(on_cuda, monkeypatch):
    env = on_cuda(startup=True, lookahead=0, n=N)
    env.compare()  # covered steps also run the option-off transaction
    advance = api.advance_module()
    real = api.phase_function(api.PHASES[1])
    counts = []

    def body(*args, **kwargs):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            out = real(*args, **kwargs)
            torch.cuda.synchronize()
        counts.append(g.memcpy_counts(prof)["HtoD"])
        return out

    monkeypatch.setattr(advance, api.PHASES[1], body)
    d = g.Driver(env)
    d.add_stream("s1", [{}, {}])
    d.add_file("f1", 4000)
    d.run(6)
    assert counts and all(c == 0 for c in counts), counts
