# SPDX-License-Identifier: Apache-2.0
"""Freshness, file and planned-rejection masks of the transaction body with
transaction graphs off, on CPU (TXG-MASK-001 to TXG-MASK-004, TXG-MASK-006).

TXG-MASK-001's sentinel check is ``test_equivalence.py``'s
``test_recycled_pages_of_fresh_rows_do_not_matter``; TXG-MASK-003's
control-flow check is ``test_exec_phases.py``'s
``test_body_control_flow_ignores_file_and_fresh_bits``. The cell checks
(TXG-MASK-002's one launch, TXG-MASK-005) are in ``test_exec_gpu.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from support import txg_transaction as tx
from support.core_order import CoreOrder
from support.txg_fakes import with_planned_rejection

pytestmark = pytest.mark.engine

CC = ("chunk", False, False)
CF = ("chunk", True, False)
OC = ("other", False, False)
OF = ("other", True, False)
CCF = ("chunk", False, True)
CFF = ("chunk", True, True)
OCF = ("other", False, True)
OFF = ("other", True, True)


def book_slots():
    from vllm_nemotron_asr_streaming.nemotron_asr import rnnt

    return rnnt


# @spec TXG-MASK-001
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
def test_a_fresh_row_starts_from_the_initial_book(txn_env, lookahead):
    """A fresh CHUNK row's stored book holds its admitted prompt and geometry
    whatever its recycled page held."""
    rnnt = book_slots()
    env = txn_env(lookahead)
    cont_page = env.accepted(OC).page
    for config in (CF, CFF):
        fresh = env.accepted(config).but(page=cont_page)
        out = env.run(env.compose([fresh]))
        assert int(out.status[0]) == 0
        book = out.pools.tensors["book_pool"][1]
        assert int(book[rnnt.QUEUE_PROMPT]) == fresh.prompt
        assert int(book[rnnt.BOOK_GEOMETRY]) == fresh.geometry
        clean = env.run(env.compose([env.accepted(config)]))
        assert tx.same_bits(out.rows, clean.rows)
        assert not out.pools.differences(clean.pools, blocks=[1])


# @spec TXG-MASK-002
def test_fresh_and_continuing_chunk_rows_share_one_bucket(txn_env):
    env = txn_env(0)
    requests = []
    resolver = tx.wrap_decode_resolver(
        env, lambda request, args: requests.append(request)
    )
    transitions = []
    real = env.kwargs["encoder_transition"]

    def transition(*args, **kwargs):
        transitions.append(int(args[0].shape[0]))
        return real(*args, **kwargs)

    rows = [env.accepted(c) for c in (CF, CC, OC, CFF, CCF, OF)]
    env.run(env.compose(rows), decode_resolver=resolver, encoder_transition=transition)
    assert len(requests) == 1, requests
    assert requests[0].execution_batch_size == 4
    assert transitions == [4], transitions


@pytest.fixture
def stream_only_kwargs(tmp_path, monkeypatch):
    """The transaction arguments of a real stream-only step of the model."""
    from support import model_harness as h

    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    h.spec_without_state_name(monkeypatch)
    h.legacy_state_api(monkeypatch)
    seen = []
    real = advance.advance_model_rows

    def spy(core, ids, embeds, plan, **kwargs):
        seen.append((plan, kwargs))
        return real(core, ids, embeds, plan, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", spy)
    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0)
    model = h.build_model(config)
    state = h.build_model_state(config, model)
    runner = h.Runner(config, model, state)
    CoreOrder(runner, spy=False).startup(verify=False)
    request = tx._stream(model.config, "s1", 0, 0, False)
    runner.real_step(h.scheduler_output(new=(request,), handles={"s1": 1}))
    plan, kwargs = seen[-1]
    return model.config, plan, kwargs


# @spec TXG-MASK-003
def test_file_token_ids_reach_every_step(stream_only_kwargs):
    config, plan, kwargs = stream_only_kwargs
    assert not bool(plan.is_file.any())
    assert kwargs["file_carrier_id"] == int(config.file_carrier_token_id)
    assert kwargs["continuation_id"] == int(config.continuation_token_id)


# @spec TXG-MASK-003
def test_the_body_receives_the_file_bit_of_every_row(txn_env, monkeypatch):
    """A stream-only step's plan carries a file bit per row (all false), so
    the file selects are evaluated on it."""
    from support import txg_api as api

    env = txn_env(0)
    log = []
    advance = api.advance_module()
    real = api.phase_function(api.PHASES[1])

    def spy(*args, **kwargs):
        log.append(
            [
                t
                for t in tx.tensors_in([list(args), kwargs])
                if t.dtype == torch.bool and t.dim() == 1
            ]
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, api.PHASES[1], spy)
    step = env.compose([env.accepted(c) for c in (CC, OC)])
    step.plan = dataclasses.replace(step.plan, is_file=torch.zeros(0, dtype=torch.bool))
    env.run(step)
    assert log and any(t.numel() == 2 for t in log[0]), (
        "the body received no per-row bool vector for a stream-only step"
    )


# @spec TXG-MASK-004
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
@pytest.mark.parametrize(
    "configs", [(CC, OC, CC), (OC, OC), (CC,)], ids=["split", "echo", "chunk"]
)
def test_all_false_masks_equal_the_stream_only_transaction(txn_env, lookahead, configs):
    env = txn_env(lookahead)
    step = env.compose([env.accepted(c) for c in configs])
    assert not bool(step.plan.is_file.any())
    assert bool(step.plan.has_initial_states_p.all())
    masked = env.run(step)
    stream_only = dataclasses.replace(
        step.plan, is_file=torch.zeros(0, dtype=torch.bool)
    )
    plain = env.run(step, plan=stream_only)
    for field in ("rows", "status", "progress", "queue_counts"):
        assert tx.same_bits(getattr(masked, field), getattr(plain, field)), field
    assert not masked.pools.differences(plain.pools)


TOKENS = ("placeholder", "park", "file_carrier", "continuation", "label")


def token_value(env, name, fixture):
    return (
        {
            "placeholder": env.placeholder,
            "park": env.park,
            "file_carrier": int(env.config.file_carrier_token_id),
            "continuation": int(env.config.continuation_token_id),
            "label": 7,
        }[name]
        if name != "own"
        else fixture.token
    )


# @spec TXG-MASK-006, TXG-COV-010
@pytest.mark.parametrize("config", tx.CONFIGS, ids=tx.config_name)
@pytest.mark.parametrize("bound_as", ["other", "own"])
def test_the_planned_rejection_bit_rejects_any_row(txn_env, config, bound_as):
    """Whatever the row's kind, fresh bit, file bit or scheduled token, its
    planned-rejection bit gives it a nonzero status, its pages stay as they
    were, and the other rows of the step are unaffected."""
    env = txn_env(0)
    neighbours = [env.accepted(CC), env.accepted(OC)]
    for token in TOKENS + ("own",):
        fixture = env.accepted(config)
        fixture = fixture.but(token=token_value(env, token, fixture))
        if bound_as == "other":
            fixture = fixture.but(kind="other", deadline=0)
        rows = [neighbours[0], fixture, neighbours[1]]
        step = env.compose(rows)
        before = step.pools.clone()
        clean = env.run(step)
        step.plan = with_planned_rejection(step.plan, [False, True, False])
        out = env.run(step)
        assert int(out.status[1]) != 0, (config, bound_as, token)
        assert not out.pools.differences(before, blocks=[2]), (config, bound_as, token)
        for i in (0, 2):
            assert int(out.status[i]) == int(clean.status[i]), (config, token, i)
            assert tx.same_bits(out.rows[i], clean.rows[i]), (config, token, i)
        assert not out.pools.differences(clean.pools, blocks=[1, 3])


# @spec TXG-MASK-006
def test_the_planned_rejection_bit_is_a_plan_vector(plan_builder):
    plan = plan_builder("cxo")
    assert plan.planned_rejection.dtype == torch.bool
    assert plan.planned_rejection.tolist() == [False, True, False]
    assert plan.is_chunk.tolist() == [True, False, False]
