# SPDX-License-Identifier: Apache-2.0
"""Cells and classification on CPU (TXG-CELL-001, TXG-CELL-002, TXG-COV-001,
TXG-COV-002, TXG-COV-004, TXG-COV-006 to TXG-COV-010, TXG-COV-013,
TXG-COV-014).

``classify`` is called on CPU row plans (``plan_builder``) with a fake cell
inventory (``cell_inventory``), as ``support.txg_api`` documents. Signature
comparison, counting and execution are ``execute``'s and are tested on CUDA
(``test_coverage_gpu.py``, ``test_exec_gpu.py``). A step with a multi-token
row (TXG-COV-014) runs through the model state on CPU with transaction graphs
on, through the fake-capture seam (``on_cpu``; ``support.txg_steps`` lays
out its token axis): a fake cell computes nothing, so those tests read the
counters, launches and transactions, and the output's shape.
"""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
from support import txg_api as api
from support.txg_fakes import Row, has_graph_of

CHUNK, ECHO = api.CHUNK, api.ECHO


def classify(plan, cells, n, geometry=0, **kwargs):
    return api.classify(
        plan, n=n, geometry_index=geometry, has_graph=has_graph_of(cells), **kwargs
    )


def spec_of(k: int, e: int) -> str:
    return "c" * k + "o" * e


def interleaved(k: int, e: int, seed: int = 0) -> str:
    rng = torch.Generator().manual_seed(seed)
    codes = list(spec_of(k, e))
    order = torch.randperm(len(codes), generator=rng).tolist()
    return "".join(codes[i] for i in order)


# ---------------------------------------------------------------------------
# Cells


# @spec TXG-CELL-001
@pytest.mark.parametrize("n", [1, 4, 63])
@pytest.mark.parametrize("geometry", [0, 1])
def test_inventory_is_one_chunk_and_one_echo_cell_per_count(n, geometry):
    keys = api.cell_keys(n, geometry)
    assert len(keys) == 2 * n == len(set(keys))
    assert set(keys) == {
        (geometry, kind, count) for kind in (CHUNK, ECHO) for count in range(1, n + 1)
    }


# @spec TXG-CELL-001
def test_geometry_index_is_the_lookahead_at_80_and_160_ms():
    assert api.GEOMETRY_BY_LOOKAHEAD == {0: 0, 1: 1}
    from vllm_nemotron_asr_streaming.lookahead import geometry_index

    assert geometry_index(0) == 0 and geometry_index(1) == 1


# @spec TXG-CELL-002
@pytest.mark.parametrize(
    "row, kind",
    [
        (Row("chunk"), CHUNK),
        (Row("chunk", fresh=True), CHUNK),
        (Row("chunk", file=True), CHUNK),
        (Row("chunk", fresh=True, file=True), CHUNK),
        (Row("other"), ECHO),
        (Row("other", file=True), ECHO),
        (Row("other", fresh=True), ECHO),
    ],
)
def test_chunk_rows_need_the_chunk_cell_and_other_rows_the_echo_cell(
    plan_builder, cell_inventory, row, kind
):
    cells = cell_inventory(4)
    result = classify(plan_builder([row]), cells, 4)
    assert result.reason is None
    assert tuple(result.cells) == ((0, kind, 1),)


# ---------------------------------------------------------------------------
# Classification


# @spec TXG-COV-001
def test_classification_reads_the_cpu_plan_only(plan_builder, cell_inventory):
    """No tensor operation touches anything but the CPU plan, and the plan is
    left unchanged."""
    from torch.utils._python_dispatch import TorchDispatchMode

    devices: list[str] = []

    class DeviceWatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            for x in itertools.chain(args, (kwargs or {}).values()):
                if isinstance(x, torch.Tensor):
                    devices.append(x.device.type)
            return func(*args, **(kwargs or {}))

    plan = plan_builder("ccoco")
    before = copy.deepcopy(plan)
    with DeviceWatch():
        result = classify(plan, cell_inventory(4), 4)
    assert set(devices) <= {"cpu"}
    assert (result.chunk_count, result.echo_count) == (3, 2)
    for name in ("is_chunk", "geometry_id", "is_file", "has_initial_states_p"):
        assert torch.equal(getattr(plan, name), getattr(before, name))


# @spec TXG-COV-002, TXG-COV-004
@pytest.mark.parametrize("n", [1, 3, 4])
def test_covered_exactly_when_each_present_kind_has_its_cell(
    plan_builder, cell_inventory, n
):
    cells = cell_inventory(n)
    for k, e in itertools.product(range(n + 3), repeat=2):
        if k + e == 0:
            continue
        result = classify(plan_builder(interleaved(k, e, k * 7 + e)), cells, n)
        assert (result.population, result.chunk_count, result.echo_count) == (
            k + e,
            k,
            e,
        )
        if k > n:
            assert (result.reason, result.kind, result.count) == (
                "chunk-uncovered",
                CHUNK,
                k,
            ), (k, e)
        elif e > n:
            assert (result.reason, result.kind, result.count) == (
                "echo-uncovered",
                ECHO,
                e,
            ), (k, e)
        else:
            needed = [(0, CHUNK, k)] if k else []
            needed += [(0, ECHO, e)] if e else []
            assert result.reason is None, (k, e, result.reason)
            assert list(result.cells) == needed, (k, e)


# @spec TXG-COV-002
def test_population_need_not_be_a_declared_entry(plan_builder, cell_inventory):
    """A 33-row step split 30/3 is covered when N >= 30."""
    cells = cell_inventory(30)
    result = classify(plan_builder(spec_of(30, 3)), cells, 30)
    assert result.reason is None
    assert list(result.cells) == [(0, CHUNK, 30), (0, ECHO, 3)]


# @spec TXG-COV-006
def test_a_needed_cell_without_a_graph_is_uncaptured(plan_builder, cell_inventory):
    cells = cell_inventory(4, missing=[(0, ECHO, 2)])
    result = classify(plan_builder("cocoo"), cells, 4)
    assert result.reason is None
    result = classify(plan_builder("ccoo"), cells, 4)
    assert (result.reason, result.kind, result.count) == ("uncaptured", ECHO, 2)
    cells = cell_inventory(4, missing=[(0, CHUNK, 3)])
    result = classify(plan_builder("ccco"), cells, 4)
    assert (result.reason, result.kind, result.count) == ("uncaptured", CHUNK, 3)


# @spec TXG-COV-006
def test_a_chunk_row_of_another_geometry_is_multi_geometry(
    plan_builder, cell_inventory
):
    cells = cell_inventory(4)
    plan = plan_builder([Row("chunk"), Row("chunk", geometry=1), Row("other")])
    result = classify(plan, cells, 4, geometry=0)
    assert result.reason == "multi-geometry"
    assert not result.cells


# @spec TXG-COV-006
def test_a_step_without_rows_is_empty(plan_builder, cell_inventory):
    result = classify(plan_builder([]), cell_inventory(4), 4)
    assert result.reason == "empty"
    assert result.population == 0 and not result.cells


# @spec TXG-COV-007
@pytest.mark.parametrize(
    "mode, reason",
    [
        ({"memory_profile": True}, "memory-profile"),
        ({"native_burst": True}, "native-burst"),
    ],
)
def test_ineligible_modes_are_never_covered(plan_builder, cell_inventory, mode, reason):
    result = classify(plan_builder("cco"), cell_inventory(4), 4, **mode)
    assert result.reason == reason
    assert not result.cells


# @spec TXG-COV-008
@pytest.mark.parametrize(
    "seqs, tokens, active",
    [(6, 8192, None), (6, 4, None), (6, 8192, 3), (6, 4, 5), (6, 8192, 6)],
    ids=["seqs", "tokens", "active", "tokens-below-active", "equal"],
)
def test_n_is_the_smaller_of_p_max_and_the_admission_limit(
    plan_builder, cell_inventory, seqs, tokens, active
):
    """``N = min(P_max, admission limit)`` is the largest population the
    configuration checks accept (TXG-CFG-004, TXG-CFG-020), and at that ``N``
    every one-token step of ``1 <= P <= N`` rows is covered."""
    from support.txg_fakes import admission_limit, engine_config, hf_config

    limits = dict(
        max_num_seqs=seqs, max_num_batched_tokens=tokens, max_num_active_seqs=active
    )
    n = min(seqs, tokens, admission_limit(seqs, active))
    assert api.config_errors(engine_config(hf_config([n]), **limits)) == []
    assert api.config_errors(engine_config(hf_config([n + 1]), **limits)), (
        f"N = {n + 1} above min(P_max, admission limit) was accepted"
    )
    cells = cell_inventory(n)
    for p in range(1, n + 1):
        for k in range(p + 1):
            result = classify(plan_builder(interleaved(k, p - k, p)), cells, n)
            assert result.reason is None, (p, k, result.reason)


# @spec TXG-COV-008
@pytest.mark.parametrize("p_max", [1, 6])
def test_every_population_is_covered_at_n_equal_p_max(
    plan_builder, cell_inventory, p_max
):
    cells = cell_inventory(p_max)
    for p in range(1, p_max + 1):
        for k in range(p + 1):
            for seed in range(3):
                result = classify(
                    plan_builder(interleaved(k, p - k, seed)), cells, p_max
                )
                assert result.reason is None, (p, k, result.reason)


# @spec TXG-COV-009
def test_rows_are_never_padded_into_a_larger_cell(plan_builder, cell_inventory):
    cells = cell_inventory(6, missing=[(0, CHUNK, 3), (0, ECHO, 1)])
    result = classify(plan_builder("ccc"), cells, 6)
    assert (result.reason, result.kind, result.count) == ("uncaptured", CHUNK, 3)
    result = classify(plan_builder("cco"), cells, 6)
    assert (result.reason, result.kind, result.count) == ("uncaptured", ECHO, 1)
    result = classify(plan_builder("cc"), cells, 6)
    assert list(result.cells) == [(0, CHUNK, 2)]


# @spec TXG-COV-010
def test_planning_failed_rows_count_in_p_and_e_never_k(plan_builder, cell_inventory):
    plan = plan_builder("cxcxo")
    assert plan.planned_rejection.tolist() == [False, True, False, True, False]
    assert plan.is_chunk.tolist() == [True, False, True, False, False]
    result = classify(plan, cell_inventory(4), 4)
    assert (result.population, result.chunk_count, result.echo_count) == (5, 2, 3)
    assert list(result.cells) == [(0, CHUNK, 2), (0, ECHO, 3)]


# @spec TXG-COV-010, TXG-COV-013
def test_a_mixed_step_with_failed_rows_needs_the_echo_cell(
    plan_builder, cell_inventory
):
    """A step that holds planning-failed and other rows reaches classification
    and counts its failed rows in E; the inventory stays 2N."""
    cells = cell_inventory(2)
    result = classify(plan_builder("cxx"), cells, 2)
    assert result.reason is None
    assert list(result.cells) == [(0, CHUNK, 1), (0, ECHO, 2)]
    result = classify(plan_builder("cxxo"), cells, 2)
    assert (result.reason, result.kind, result.count) == ("echo-uncovered", ECHO, 3)
    assert len(api.cell_keys(2, 0)) == 4


# @spec TXG-COV-002, TXG-COV-006
def test_classification_never_raises_on_an_unusual_plan(plan_builder, cell_inventory):
    cells = cell_inventory(2, missing=[(0, CHUNK, 1)])
    for spec in ("", "c", "o", "ccc", "ooo", "cccooo"):
        result = classify(plan_builder(spec), cells, 2)
        assert result.reason in (None, *api.REASONS)


# ---------------------------------------------------------------------------
# Multi-token rows (CPU, the model state with transaction graphs on)


def _fallback_growth(before, after):
    return {
        key: after[key] - before.get(key, 0)
        for key in after
        if after[key] != before.get(key, 0)
    }


@pytest.mark.engine
# @spec TXG-COV-014, TXG-COV-002, TXG-COV-003, TXG-MASK-002
@pytest.mark.parametrize("chunk_rows", [1, 3], ids=["covered-shape", "k-above-n"])
def test_a_step_with_a_multi_token_row_falls_back_by_multi_token(on_cpu, chunk_rows):
    """A step that holds a multi-token row among one-token rows is not
    covered even when cells exist for its shape (only a one-token step is,
    TXG-COV-002), and falls back with reason ``multi-token`` before any
    reason of TXG-COV-004 (``K = 3 > N = 2`` here), without raising: one
    fallback count, no cell launched, and one option-off transaction over the
    whole row plan, whose output has one row per scheduled token."""
    from support import txg_steps as steps
    from support import txgcap_harness as t

    run = on_cpu(n=2, max_num_seqs=4)
    env, core = run.env, run.core
    core.startup()
    before = dict(t.counter(env, "fallbacks"))
    replays, advances = env.fake.replay_count, len(env.advance.calls)
    rows = [t.stream_unit(env, f"a{i}") for i in range(chunk_rows)]
    wide = steps.with_tokens(
        t.stream_unit(env, "wide"), int(env.config.audio_chunk_token_id)
    )
    hidden, _ = steps.token_step(
        env.runner,
        *rows,
        wide,
        counts={"wide": 2},
        handles={r.req_id: i + 1 for i, r in enumerate([*rows, wide])},
    )
    grown = _fallback_growth(before, dict(t.counter(env, "fallbacks")))
    assert len(grown) == 1 and list(grown.values()) == [1], grown
    ((reason, _kind, _count),) = grown
    assert reason == "multi-token", grown
    assert env.fake.replay_count == replays, "a step with a multi-token row launched"
    assert len(env.advance.calls) == advances + 1, (
        "the option-off transaction did not run once over the whole row plan"
    )
    assert hidden.shape[0] == chunk_rows + 2, "not one output row per scheduled token"


@pytest.mark.engine
# @spec TXG-COV-002, TXG-COV-008
def test_a_one_token_step_within_n_is_covered(on_cpu):
    """The same rows, each scheduling one token, are covered: no fallback,
    one launch per kind present."""
    from support import txg_steps as steps
    from support import txgcap_harness as t

    run = on_cpu(n=2, max_num_seqs=4)
    env, core = run.env, run.core
    core.startup()
    before = dict(t.counter(env, "fallbacks"))
    replays = env.fake.replay_count
    hidden, _ = steps.token_step(
        env.runner, t.stream_unit(env, "a"), t.stream_unit(env, "b")
    )
    assert _fallback_growth(before, dict(t.counter(env, "fallbacks"))) == {}
    assert env.fake.replay_count == replays + 1, "one chunk cell launch expected"
    assert hidden.shape[0] == 2


@pytest.mark.engine
# @spec TXG-COV-014, TXG-COV-012
def test_multi_token_takes_precedence_over_an_endpoint_signature(on_cpu):
    """A step with a multi-token row and an endpoint-policy row falls back
    once, by ``multi-token``: the reason classification assigns first takes
    precedence over the signature comparison's ``endpoint_policy``."""
    from support import txg_steps as steps
    from support import txgcap_harness as t

    run = on_cpu(n=2, max_num_seqs=3)
    env, core = run.env, run.core
    core.startup()
    before = dict(t.counter(env, "fallbacks"))
    policy = t.with_endpoint_policy(t.stream_unit(env, "p"))
    wide = steps.with_tokens(
        t.stream_unit(env, "wide"), int(env.config.audio_chunk_token_id)
    )
    steps.token_step(
        env.runner, policy, wide, counts={"wide": 2}, handles={"p": 1, "wide": 2}
    )
    grown = _fallback_growth(before, dict(t.counter(env, "fallbacks")))
    assert [key[0] for key in grown] == ["multi-token"], grown
    assert list(grown.values()) == [1], grown


@pytest.mark.engine
# @spec TXG-EXEC-012, TXG-COV-011
def test_dummy_steps_never_call_execute_on_cpu(on_cpu, monkeypatch):
    """The dummy steps core runs after startup, as the model state's
    ``prepare_inputs`` classifies them (REG-MEM-001: the profiling dummy step
    and the warm-up dummy steps), call no ``execute``, replay no cell and
    count no fallback."""
    from support import txgcap_harness as t

    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    calls = []
    t.wrap_execute(monkeypatch, env.model, lambda call: calls.append(1) or call())
    before = dict(t.counter(env, "fallbacks"))
    replays = env.fake.replay_count
    env.runner.profile_run(core_embeds=False)
    core.warmup_step()
    env.runner.dummy_step(is_profile=False, randomize_inputs=True, core_embeds=False)
    assert calls == [], "a dummy step called execute"
    assert env.fake.replay_count == replays, "a dummy step replayed a cell"
    assert dict(t.counter(env, "fallbacks")) == before, "a dummy step was counted"
