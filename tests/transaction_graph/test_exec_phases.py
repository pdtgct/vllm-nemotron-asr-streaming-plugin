# SPDX-License-Identifier: Apache-2.0
"""The streaming transaction's phase split (TXG-TXN-001 to TXG-TXN-008,
TXG-TXN-010), with transaction graphs off, on CPU.

Rows by token position (TXG-TXN-010) run on composed one-token steps and on
real steps of the model with a multi-token row (``support.txg_steps``), which
REG-STATE-014 fails at planning while the step's other rows execute.

The option-off transaction runs on composed steps (``txn_env``,
``support.txg_transaction``); the phase functions are named in
``support.txg_api.PHASES``. The covered-step order (TXG-TXN-009) and the
device checks of TXG-TXN-001 and TXG-TXN-002 are in ``test_exec_gpu.py``.
"""

from __future__ import annotations

import inspect
from collections import Counter

import pytest
import torch
from support import txg_api as api
from support import txg_transaction as tx
from support.txgcap_harness import TxgCoreOrder

PREPARE, BODY, COMMIT = api.PHASES

CC = ("chunk", False, False)
CF = ("chunk", True, False)
OC = ("other", False, False)
OF = ("other", True, False)
CCF = ("chunk", False, True)
CFF = ("chunk", True, True)
OCF = ("other", False, True)
OFF = ("other", True, True)


def fixtures(env, *configs):
    return [env.accepted(c) for c in configs]


def spy_phases(monkeypatch, log):
    advance = api.advance_module()
    for name in api.PHASES:
        real = api.phase_function(name)

        def spy(*args, __real=real, __name=name, **kwargs):
            log.append((__name, args, kwargs))
            return __real(*args, **kwargs)

        monkeypatch.setattr(advance, name, spy)


def tensors_in(value, depth=0):
    if isinstance(value, torch.Tensor):
        yield value
    elif depth < 3 and isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors_in(item, depth + 1)
    elif depth < 3 and isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item, depth + 1)


# ---------------------------------------------------------------------------
# The interface (CPU, no vLLM)


# @spec TXG-TXN-007
def test_advance_model_rows_keeps_its_interface():
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    signature = inspect.signature(advance.advance_model_rows)
    params = list(signature.parameters.values())
    positional = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == ["core", "input_ids", "inputs_embeds", "plan"]
    keyword = {p.name: p for p in params if p.kind is p.KEYWORD_ONLY}
    required = {
        "channel_pools",
        "time_pools",
        "len_pools",
        "h_pool",
        "c_pool",
        "queue_pool",
        "book_pool",
        "frontend_raw_pool",
        "frontend_mel_pool",
        "frontend_counter_pool",
        "adapter",
        "decode_resolver",
        "placeholder_id",
        "park_id",
    }
    false_default = {"capture", "graph_covers_decode", "memory_profile"}
    none_default = {
        "endpoint_history_pool",
        "endpoint_book_pool",
        "eou_token_id",
        "encoder_transition",
        "bucket_transition",
        "commit_sink",
        "staging",
        "native_burst_handoff",
        "file_carrier_id",
        "continuation_id",
        "geometry_order",
    }
    assert set(keyword) == required | false_default | none_default
    assert len(params) == 4 + len(keyword)
    for name in required:
        assert keyword[name].default is inspect.Parameter.empty, name
    for name in false_default:
        assert keyword[name].default is False, name
    for name in none_default:
        assert keyword[name].default is None, name
    assert signature.return_annotation in (torch.Tensor, "torch.Tensor")


# @spec TXG-TXN-006, TXG-TXN-007
def test_advance_model_rows_is_composed_of_the_three_phases():
    for name in api.PHASES:
        assert callable(api.phase_function(name)), name
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    source = inspect.getsource(advance.advance_model_rows)
    for name in api.PHASES:
        assert f"{name}(" in source, f"advance_model_rows does not call {name}"


# ---------------------------------------------------------------------------
# Phases on the option-off transaction


@pytest.mark.engine
# @spec TXG-TXN-006
@pytest.mark.parametrize(
    "configs", [(CC, OC, CFF), (OC, OCF), (CF,)], ids=["split", "echo", "chunk"]
)
def test_option_off_runs_prepare_body_commit_once_in_order(
    txn_env, monkeypatch, configs
):
    env = txn_env(0)
    log = []
    spy_phases(monkeypatch, log)
    env.run(env.compose(fixtures(env, *configs)))
    assert [name for name, _, _ in log] == list(api.PHASES)


@pytest.mark.engine
# @spec TXG-TXN-006
def test_the_model_runs_the_phases_through_its_host_staging(tmp_path, monkeypatch):
    """A real step of the model runs the phases over its whole row plan with
    the model's ``HostStaging``."""
    from support import model_harness as h

    advance = api.advance_module()
    seen = {}
    real = advance.advance_model_rows

    def spy(core, ids, embeds, plan, **kwargs):
        seen["staging"] = kwargs.get("staging")
        return real(core, ids, embeds, plan, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", spy)
    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0)
    model = h.build_model(config)
    state = h.build_model_state(config, model)
    runner = h.Runner(config, model, state)
    TxgCoreOrder(runner, spy=False).startup(verify=False)
    log = []
    spy_phases(monkeypatch, log)
    stream = tx._stream(model.config, "p1", 0, 0, False)
    file = tx._file(model.config, "f1", 0, 3000)
    runner.real_step(h.scheduler_output(new=(stream, file), handles={"p1": 1, "f1": 2}))
    assert isinstance(seen["staging"], advance.HostStaging)
    assert [name for name, _, _ in log] == list(api.PHASES)


@pytest.mark.engine
# @spec TXG-TXN-001
def test_prepare_receives_no_resident_pool(txn_env, monkeypatch):
    env = txn_env(0)
    log = []
    spy_phases(monkeypatch, log)
    step = env.compose(fixtures(env, CC, OC, CFF, OCF))
    pools = step.pools.clone()
    env.run(step, pools=pools)
    _, args, kwargs = next(entry for entry in log if entry[0] == PREPARE)
    pointers = {t.untyped_storage().data_ptr() for _, t in pools.flat()}
    for t in tensors_in([list(args), kwargs]):
        assert t.untyped_storage().data_ptr() not in pointers, (
            "the prepare phase was given a resident pool"
        )


@pytest.mark.engine
# @spec TXG-TXN-001
def test_prepare_produces_the_body_vectors_on_the_host(txn_env, monkeypatch):
    """Every ``int64`` and ``bool`` vector the prepare phase returns is a host
    tensor built from plan authority."""
    env = txn_env(0)
    results = []
    advance = api.advance_module()
    real = api.phase_function(PREPARE)

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        results.append(out)
        return out

    monkeypatch.setattr(advance, PREPARE, spy)
    env.run(env.compose(fixtures(env, CC, CF, OC, CFF, OCF)))
    vectors = [
        t
        for t in tensors_in(
            list(vars(results[0]).values())
            if hasattr(results[0], "__dict__")
            else results[0]
        )
        if t.dtype in (torch.int64, torch.bool)
    ]
    assert vectors, "the prepare phase returned no int64 or bool vector"
    assert all(t.device.type == "cpu" for t in vectors)


@pytest.mark.engine
# @spec TXG-TXN-002
def test_body_issues_no_host_to_device_copy(txn_env, monkeypatch):
    env = txn_env(0)
    advance = api.advance_module()
    active = {"body": False}
    copies = []
    real_body = api.phase_function(BODY)

    def body(*args, **kwargs):
        active["body"] = True
        try:
            return real_body(*args, **kwargs)
        finally:
            active["body"] = False

    def watch(name, real):
        def wrapped(*args, **kwargs):
            if active["body"]:
                copies.append(name)
            return real(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(advance, BODY, body)
    monkeypatch.setattr(advance, "_h2d", watch("_h2d", advance._h2d))
    for method in ("stage", "stage_bucket"):
        if hasattr(advance.HostStaging, method):
            monkeypatch.setattr(
                advance.HostStaging,
                method,
                watch(method, getattr(advance.HostStaging, method)),
            )
    env.run(env.compose(fixtures(env, CC, CF, OC, OF, CCF, CFF, OCF)))
    assert copies == [], f"the body staged host data itself: {copies}"


def _body_trace(env, configs):
    return tx.trace_phase(env, env.compose(fixtures(env, *configs)), BODY)


@pytest.mark.engine
# @spec TXG-TXN-003, TXG-MASK-003
@pytest.mark.parametrize(
    "variants",
    [
        [(CC, CC, OC), (CFF, CCF, OCF), (CF, CC, OF), (CCF, CF, OFF)],
        [(OC, OC), (OCF, OC), (OF, OFF)],
        [(CC,), (CF,), (CCF,), (CFF,)],
    ],
    ids=["split", "echo", "chunk"],
)
def test_body_control_flow_ignores_file_and_fresh_bits(txn_env, variants):
    """At a fixed sequence of row kinds, the body runs the same operations on
    the same shapes whatever the rows' file and fresh bits."""
    env = txn_env(0)
    reference = _body_trace(env, variants[0])
    for configs in variants[1:]:
        trace = _body_trace(env, configs)
        assert trace == reference, (
            "the body's operations differ between "
            f"{[tx.config_name(c) for c in variants[0]]} "
            f"and {[tx.config_name(c) for c in configs]}"
        )


@pytest.mark.engine
# @spec TXG-TXN-004
def test_body_shapes_do_not_depend_on_the_fresh_count(txn_env):
    env = txn_env(0)
    layouts = [(CC, CC, CC), (CF, CC, CC), (CF, CF, CC), (CF, CF, CF)]
    traces = [_body_trace(env, layout) for layout in layouts]
    for layout, trace in zip(layouts[1:], traces[1:]):
        assert trace == traces[0], f"fresh count changed the body: {layout}"


@pytest.mark.engine
# @spec TXG-TXN-004
def test_no_per_call_staging_of_fresh_positions(txn_env, monkeypatch):
    """The host vectors the transaction uploads have the same shapes whatever
    the number of fresh rows."""
    env = txn_env(0)
    advance = api.advance_module()
    staged: list[tuple] = []
    real = advance._h2d

    def watch(t, device):
        staged.append((tuple(t.shape), str(t.dtype)))
        return real(t, device)

    monkeypatch.setattr(advance, "_h2d", watch)
    shapes = []
    for layout in [(CC, CC, OC), (CF, CC, OC), (CF, CF, OF), (CF, CF, OF)]:
        staged.clear()
        env.run(env.compose(fixtures(env, *layout)))
        shapes.append(Counter(staged))
    for layout_shapes in shapes[1:]:
        assert layout_shapes == shapes[0], (shapes[0], layout_shapes)


def _strided_copy(t: torch.Tensor) -> torch.Tensor:
    """Same shape and values, non-canonical page interior."""
    wide = torch.zeros((*t.shape[:-1], 2 * t.shape[-1]), dtype=t.dtype, device=t.device)
    view = wide[..., ::2]
    view.copy_(t)
    return view


@pytest.mark.engine
# @spec TXG-TXN-005
def test_scatter_plan_is_validated_before_the_first_scatter(txn_env, monkeypatch):
    """Each resident pool in turn is given an interior the scatter refuses;
    when the transaction refuses it, no pool has been written."""
    env = txn_env(0)
    advance = api.advance_module()
    scatters = []
    real = advance.execute_masked_page_scatter_

    def watch(*args, **kwargs):
        scatters.append(args[0].data_ptr())
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, "execute_masked_page_scatter_", watch)
    step = env.compose(fixtures(env, CC, OC, CCF))
    refused = 0
    for name, _ in step.pools.flat():
        pools = step.pools.clone()
        before = pools.clone()
        target = name.split("[")[0]
        if "[" in name:
            i = int(name.split("[")[1].rstrip("]"))
            pools.tensors[target][i] = _strided_copy(pools.tensors[target][i])
        else:
            pools.tensors[target] = _strided_copy(pools.tensors[target])
        scatters.clear()
        outcome = env.run(step, pools=pools, raises=False)
        if outcome.error is None:
            continue
        refused += 1
        assert scatters == [], f"{name}: a scatter ran before the refusal"
        assert not before.differences(pools), (
            f"{name}: a pool changed before the refusal"
        )
        assert not outcome.sink.reservations or outcome.sink.last.cancelled, name
    assert refused, "no invalid scatter descriptor was refused"


@pytest.mark.engine
# @spec TXG-TXN-008
def test_option_off_commit_reserves_then_scatters_then_stages(txn_env, monkeypatch):
    env = txn_env(0)
    advance = api.advance_module()
    from support.txg_fakes import RecordingSink

    sink = RecordingSink()
    real = advance.execute_masked_page_scatter_

    def watch(*args, **kwargs):
        sink.events.append(("scatter", None))
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, "execute_masked_page_scatter_", watch)
    env.run(env.compose(fixtures(env, CC, OC, CFF, OCF)), commit_sink=sink)
    kinds = [kind for kind, _ in sink.events]
    assert kinds.count("reserve") == 1 and kinds.count("stage") == 1, kinds
    assert kinds[0] == "reserve" and kinds[-1] == "stage", kinds
    assert set(kinds[1:-1]) == {"scatter"}, kinds


# ---------------------------------------------------------------------------
# Rows by scheduled position


@pytest.mark.engine
# @spec TXG-TXN-010
def test_outputs_follow_the_runner_order(txn_env):
    """Moving the non-CHUNK rows among the CHUNK rows (whose relative order is
    kept) moves each row's output with it."""
    env = txn_env(0)
    rows = fixtures(env, CC, OC, CFF, OCF, OF)
    blocks = [1, 2, 3, 4, 5]
    base = env.compose(rows, blocks=blocks)
    order = [1, 0, 4, 3, 2]  # CHUNK rows 0 and 2 keep their order
    moved = env.compose([rows[i] for i in order], blocks=[blocks[i] for i in order])
    a, b = env.run(base), env.run(moved)
    for new, old in enumerate(order):
        assert tx.same_bits(a.rows[old], b.rows[new]), (old, new)
        assert int(a.status[old]) == int(b.status[new]), (old, new)
        assert tx.same_bits(a.progress[old], b.progress[new]), (old, new)
    assert not a.pools.differences(b.pools)


@pytest.mark.engine
# @spec TXG-TXN-010
def test_a_row_reads_only_its_own_inputs(txn_env):
    """Changing one row's scheduled token and carrier row changes no other
    row's output, status, progress or pages."""
    env = txn_env(0)
    rows = fixtures(env, CC, OC, CF, OCF)
    step = env.compose(rows)
    base = env.run(step)
    for j in range(len(rows)):
        changed = env.compose(rows)
        changed.input_ids[j] = (
            env.park if rows[j].token != env.park else env.placeholder
        )
        changed.inputs_embeds[j, tx.HEADER_SLOTS :] += 0.25
        out = env.run(changed)
        others = [i for i in range(len(rows)) if i != j]
        for i in others:
            assert tx.same_bits(base.rows[i], out.rows[i]), (j, i)
            assert int(base.status[i]) == int(out.status[i]), (j, i)
            assert tx.same_bits(base.progress[i], out.progress[i]), (j, i)
        blocks = [i + 1 for i in others]
        assert not base.pools.differences(out.pools, blocks=blocks), j


@pytest.mark.engine
# @spec TXG-TXN-010
def test_output_has_one_row_per_scheduled_row(txn_env):
    env = txn_env(0)
    for configs in [(CC,), (OC, OF), (CC, OC, CFF, OCF, OF, CF)]:
        step = env.compose(fixtures(env, *configs))
        out = env.run(step)
        assert out.rows.shape[0] == step.population
        assert out.status.shape == (step.population,)


@pytest.mark.engine
# @spec TXG-TXN-010
def test_a_planning_failed_row_gets_the_rejected_row_output(txn_env):
    from support.txg_fakes import with_planned_rejection

    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    env = txn_env(0)
    rows = fixtures(env, CC, OC, OCF)
    failed = rows[1].but(kind="other")
    step = env.compose([rows[0], failed, rows[2]])
    step.plan = with_planned_rejection(step.plan, [False, True, False])
    out = env.run(step)
    assert out.rows.shape[0] == 3
    assert int(out.status[1]) != 0
    decisions = read_decision_carrier(out.rows)
    assert int(decisions[1]) == env.park


def _token_env(off_cpu):
    from support import model_harness as h

    env = off_cpu(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    env.runner.bind_pool(h.real_bindings(h.model_spec(env.model), capacity=8))
    return env


@pytest.mark.engine
# @spec TXG-TXN-010
def test_rows_read_and_write_at_their_own_token_positions(off_cpu):
    """A step of rows scheduling one, two and one tokens: the runner-facing
    output has one row per scheduled token; each one-token row's output, at
    its last token position (``query_start_loc[i + 1] - 1``), and its pages
    equal those of the same rows in a one-token step, so the row after the
    multi-token row read its own token position and not its step position;
    both token positions of the multi-token row, which fails at planning
    (REG-STATE-014), hold the rejected-row output, and its pages stay as the
    fresh-page initialization left them."""
    from support import model_harness as h
    from support import txg_steps as steps
    from support import txgcap_harness as t

    env = _token_env(off_cpu)
    fails = t.FailSpy(env.state)
    before = t.slots_at_step_start(env, (2,))
    wide = steps.with_tokens(
        t.stream_unit(env, "b"), int(env.config.audio_chunk_token_id)
    )
    counts = [1, 2, 1]
    hidden, _ = steps.token_step(
        env.runner,
        t.stream_unit(env, "a"),
        wide,
        t.stream_unit(env, "c"),
        counts={"b": 2},
        handles={"a": 1, "b": 2, "c": 3},
    )
    assert hidden.shape[0] == sum(counts), "not one output row per scheduled token"
    positions = steps.token_positions(counts)
    assert fails.failed == {"b"}, fails.calls
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("a", 0) == 0 and statuses.get("c", 0) == 0, statuses
    assert torch.equal(h.slot_bytes(env.runner.bindings, 2), before[2]), (
        "the multi-token row's page changed"
    )

    reference = _token_env(off_cpu)
    alone, _ = steps.token_step(
        reference.runner,
        t.stream_unit(reference, "a"),
        t.stream_unit(reference, "c"),
        handles={"a": 1, "c": 3},
    )
    assert torch.equal(hidden[positions[0][-1]], alone[0]), "row 0's output moved"
    assert torch.equal(hidden[positions[2][-1]], alone[1]), (
        "the row after the multi-token row did not read its own token position"
    )
    for slot in (1, 3):
        assert torch.equal(
            h.slot_bytes(env.runner.bindings, slot),
            h.slot_bytes(reference.runner.bindings, slot),
        ), f"slot {slot}'s pages differ"
    # the rejected-row output, from a step with a planning-failed file row
    mixed, _ = steps.token_step(
        reference.runner,
        t.stream_unit(reference, "g"),
        t.file_item(reference, "f", n_samples=4000, header_samples=3999),
        handles={"g": 4, "f": 5},
    )
    for position in positions[1]:
        assert torch.equal(hidden[position], mixed[1]), (
            f"token position {position} of the multi-token row is not the "
            "rejected-row output"
        )


@pytest.mark.engine
# @spec TXG-TXN-010
def test_a_one_token_step_reads_and_writes_row_i_at_position_i(off_cpu, monkeypatch):
    """In a one-token step, row ``i``'s token position is ``i``: the
    transaction receives the runner's ``input_ids`` and ``inputs_embeds`` with
    one entry per row and returns one output row per row."""
    from support import txg_steps as steps
    from support import txgcap_harness as t

    env = _token_env(off_cpu)
    seen = []
    advance = api.advance_module()
    real = advance.advance_model_rows

    def spy(core, input_ids, inputs_embeds, plan, **kwargs):
        seen.append((input_ids, inputs_embeds, plan))
        return real(core, input_ids, inputs_embeds, plan, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", spy)
    hidden, _ = steps.token_step(
        env.runner,
        t.stream_unit(env, "a"),
        t.stream_unit(env, "b"),
        t.stream_unit(env, "c"),
    )
    (input_ids, inputs_embeds, plan), *_ = seen
    population = len(plan.bindings)
    assert population == 3
    assert input_ids is None or input_ids.shape[0] == population
    assert inputs_embeds.shape[0] == population
    assert hidden.shape[0] == population
    assert [b.request_id for b in plan.bindings] == ["a", "b", "c"]
