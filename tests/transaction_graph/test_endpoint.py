# SPDX-License-Identifier: Apache-2.0
"""Endpoint arguments, endpoint-policy rows and forced-EOU candidates
(TXG-COV-012, TXG-CELL-001, TXG-CELL-002, TXG-CELL-004, TXG-EXEC-005).

The forward gives every transaction of a real step the endpoint pools and the
EOU token, with transaction graphs on or off, so ``B_hook`` is unchanged: its
transaction always runs with ``endpoint_enabled`` true, and a row without an
endpoint policy that schedules the EOU token takes the forced-EOU path. The
cells run the same policy-free endpoint bookkeeping, and a step with an
endpoint-policy row or a forced-EOU candidate falls back by ``signature``.

* The host predicate (``advance.forced_eou_candidates``) is checked against
  the option-off transaction itself: over an enumeration of small row plans
  that includes every way a row can reach the forced-EOU path, every row the
  transaction forces carries an endpoint policy or was marked a candidate.
* Full-forward comparison: one schedule runs through the model's forward on
  an option-on engine whose cells compute (``support.txg_eager``) and on
  ``B_hook``; every step's outputs, statuses and every byte of the state pool,
  the endpoint pages included, are equal. The schedule holds a policy-free
  EOU row in a step without a policy row and in a step with one.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import random

import pytest
import torch
from support import model_harness as h
from support import txg_api as api
from support import txg_eager as eager
from support import txg_transaction as tx
from support import txgcap_harness as t

pytestmark = pytest.mark.engine

SEED = 20261115
#: The host's mark for a row whose scheduled token it does not record (a
#: position past the request's recorded prompt holds a sampled token).
SAMPLED = -1

CC = ("chunk", False, False)
CF = ("chunk", True, False)
OC = ("other", False, False)
OF = ("other", True, False)
OFF = ("other", True, True)
OCF = ("other", False, True)


# ---------------------------------------------------------------------------
# The host predicate


# @spec TXG-COV-012
def test_candidates_are_the_non_chunk_rows_whose_host_token_is_the_eou_token():
    advance = api.advance_module()
    eou, park, placeholder = 43, 41, 42
    tokens = [eou, eou, park, placeholder, SAMPLED, 7]
    chunk = [False, True, False, True, False, False]
    assert advance.forced_eou_candidates(tokens, chunk, eou) == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert advance.forced_eou_candidates(tokens, chunk, None) == [False] * 6
    assert advance.forced_eou_candidates([], [], eou) == []


def book_of(env, page):
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
        BOOK_EXPECTED_LABEL,
        BOOK_PENDING_ECHO,
    )

    book = page["book_pool"]
    return int(book[BOOK_PENDING_ECHO]), int(book[BOOK_EXPECTED_LABEL])


def page_after(env, fixture):
    """The row's page after a one-row step of ``fixture``, and its status."""
    outcome = env.run(env.compose([fixture]))
    return outcome.pools.page(1), int(outcome.status[0])


def drained(env, fixture, limit=64):
    """Run ``fixture`` (a CHUNK row) and then its label drains until no echo
    is pending: the row's page on a drained session."""
    page, status = page_after(env, fixture)
    assert status == 0, (fixture.name, status)
    for _ in range(limit):
        pending, expected = book_of(env, page)
        if not pending:
            return page
        drain = fixture.but(
            kind="other", fresh=False, token=expected, page=page, deadline=0
        )
        page, status = page_after(env, drain)
        assert status == 0, (drain.name, status)
    raise AssertionError("the session never drained")


def eou_menu(env):
    """``[(fixture, host token)]``: the accepted fixture of every
    configuration, and rows that schedule the EOU token on every kind of
    page. ``host token`` is the scheduled token the host records for the row:
    a prompt token, or :data:`SAMPLED` when the row's token was sampled on the
    device."""
    config = env.config
    eou = int(config.eou_token_id)
    prompts = {env.placeholder, int(config.file_carrier_token_id)}
    menu = []
    for key in tx.CONFIGS:
        fixture = env.accepted(key)
        menu.append((fixture, fixture.token if fixture.token in prompts else SAMPLED))
    cont, mid_drain = env.accepted(CC), env.accepted(OC)
    live = drained(env, cont)
    finished = drained(env, tx.final_tail(cont, 300, env.lookahead))

    def eou_row(name, page, **changes):
        base = cont.but(kind="other", fresh=False, token=eou, deadline=0, page=page)
        return base.but(name=name, **changes)

    forced = eou_row("eou-prompt@drained-live", live)
    after_forced, status = page_after(env, forced)
    assert status == 0, status
    menu += [
        (forced, eou),
        (eou_row("eou-prompt@mid-drain", mid_drain.page), eou),
        (eou_row("eou-prompt@finished", finished), eou),
        (eou_row("eou-prompt@fresh", live, fresh=True), eou),
        (eou_row("eou-prompt@fresh-file", live, fresh=True, file=True), eou),
        (eou_row("eou-prompt@file", env.accepted(OCF).page, file=True), eou),
        (cont.but(name="eou-token-on-a-chunk-row", token=eou), eou),
    ]
    # The token a forced EOU emitted comes back as a sampled token, on the
    # page that step left (its echo armed). A sampled EOU token on any other
    # page does not occur: see the premise test below.
    assert book_of(env, after_forced) == (1, eou), "the forced EOU emitted no EOU"
    menu.append((eou_row("eou-sampled@after-forced", after_forced), SAMPLED))
    return menu, forced, after_forced


def forced_rows(outcome) -> list[bool]:
    """The rows the transaction took down the forced-EOU path: the progress
    phase of an EOU-role row is its phase plus three."""
    return [phase >= 4 for phase in outcome.progress[:, 2].tolist()]


@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "endpoint-on"])
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-COV-012, TXG-CELL-002
def test_the_host_predicate_never_misses_a_forced_eou_row(txn_env, lookahead, endpoint):
    """Every plan of one or two rows over the menu and a seeded sample of
    three- and four-row plans: each row the option-off transaction forces
    carries an endpoint policy or is a candidate. The enumeration holds
    forced rows, candidates that are not forced, and the sampled EOU echo,
    which is neither."""
    advance = api.advance_module()
    env = txn_env(lookahead)
    eou = int(env.config.eou_token_id)
    menu, forced_fixture, after_forced = eou_menu(env)
    rng = random.Random(SEED)
    plans = [list(p) for rows in (1, 2) for p in itertools.product(menu, repeat=rows)]
    plans += [
        [menu[rng.randrange(len(menu))] for _ in range(rng.choice((3, 4)))]
        for _ in range(150)
    ]
    missed, forced_seen, spare_candidates = [], 0, 0
    for plan in plans:
        fixtures = [fixture for fixture, _ in plan]
        step = env.compose(fixtures, endpoint=endpoint)
        outcome = env.run(step)
        forced = forced_rows(outcome)
        candidates = advance.forced_eou_candidates(
            [token for _, token in plan], step.plan.is_chunk.tolist(), eou
        )
        policy = (step.plan.endpoint_mode != 0).tolist()
        for row, is_forced in enumerate(forced):
            forced_seen += is_forced
            spare_candidates += candidates[row] and not is_forced
            if is_forced and not (candidates[row] or policy[row]):
                missed.append(([f.name for f in fixtures], row))
            if fixtures[row].name.startswith("eou-sampled@after-forced"):
                # The pending echo makes it a REPLAY row.
                assert not is_forced and not candidates[row], fixtures[row].name
    assert not missed, missed[:10]
    assert forced_seen, "no row of the enumeration took the forced-EOU path"
    assert spare_candidates, "no candidate of the enumeration stayed unforced"
    # The rows are what they claim to be.
    alone = env.run(env.compose([forced_fixture], endpoint=False))
    assert forced_rows(alone) == [True] and int(alone.status[0]) == 0
    echo = env.run(env.compose([forced_fixture.but(page=after_forced)], endpoint=False))
    assert forced_rows(echo) == [False], "the echo of a forced EOU was forced again"


# @spec TXG-COV-012
def test_a_policy_free_row_emits_a_forced_eou_only_through_an_armed_echo(txn_env):
    """The premise of the host predicate. A policy-free row's decision is the
    EOU token only on the forced-EOU path, and that step arms the row's echo:
    the token then returns on a REPLAY row. So a sampled EOU token is never on
    the forced-EOU path, and the host need not read sampled tokens."""
    env = txn_env(0)
    eou = int(env.config.eou_token_id)
    menu, forced_fixture, after_forced = eou_menu(env)
    for fixture, _host in menu:
        outcome = env.run(env.compose([fixture], endpoint=False))
        decision = int(tx_decision(outcome.rows)[0])
        if decision == eou:
            assert forced_rows(outcome) == [True], fixture.name
            pending, expected = book_of(env, outcome.pools.page(1))
            assert (pending, expected) == (1, eou), fixture.name
    pending, expected = book_of(env, after_forced)
    outcome = env.run(env.compose([forced_fixture], endpoint=False))
    emitted = int(tx_decision(outcome.rows)[0]) == eou
    assert (pending, expected) == ((1, eou) if emitted else (0, 0))


def tx_decision(rows):
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    return read_decision_carrier(rows)


# ---------------------------------------------------------------------------
# The arena and the signature


# @spec TXG-EXEC-005
def test_the_arena_holds_the_endpoint_vectors_of_its_rows(on_cpu, monkeypatch):
    """Each section holds, with the other per-row vectors, the endpoint mode,
    threshold and residue frames of its rows in ascending step position."""
    graph = api.graph_module()
    advance = api.advance_module()
    run = on_cpu(n=3, max_num_seqs=3)
    steps = eager.build_covered(monkeypatch, run)
    env, binding = steps.env, steps.cells.binding
    fixtures = [env.accepted(c) for c in (OC, CC, CF, OC)][:3]
    step = env.compose(fixtures, endpoint=True)
    plan = dataclasses.replace(
        step.plan,
        endpoint_threshold_frames=torch.tensor([5, 6, 7]),
        endpoint_residue_frames=torch.tensor([1, 2, 3]),
        execution_tier=env.model._decode_graph_binding.execution_tier(2),
    )
    kwargs = env.kwargs
    prepared = advance.prepare_model_rows(
        env.core,
        step.input_ids,
        step.inputs_embeds,
        plan,
        resident=advance.resident_layout(**step.pools.kwargs()),
        decode_resolver=kwargs["decode_resolver"],
        eou_token_id=kwargs["eou_token_id"],
        encoder_transition=kwargs["encoder_transition"],
        bucket_transition=kwargs.get("bucket_transition"),
        graph_covers_decode=kwargs["graph_covers_decode"],
        file_carrier_id=kwargs["file_carrier_id"],
        continuation_id=kwargs["continuation_id"],
    )
    chunk, echo = (torch.tensor(p, dtype=torch.int64) for p in api.arena_rows(plan))
    assert (chunk.tolist(), echo.tolist()) == ([1, 2], [0])
    host = torch.full_like(binding._hosts[0], -9)
    with torch.inference_mode():
        binding._pack_arena(prepared, chunk, echo, host)
    n = binding.n
    for section, positions in ((0, chunk), (1, echo)):
        for field, source in (
            (graph.F_ENDPOINT_MODE, plan.endpoint_mode),
            (graph.F_ENDPOINT_THRESHOLD, plan.endpoint_threshold_frames),
            (graph.F_ENDPOINT_RESIDUE, plan.endpoint_residue_frames),
            (graph.F_POSITION, torch.arange(3)),
        ):
            stored = host[section, field * n : field * n + len(positions)]
            assert stored.tolist() == source.index_select(0, positions).tolist(), (
                section,
                field,
            )


def signature_fields(cell) -> dict:
    return dict(cell.signature)


# @spec TXG-CELL-001, TXG-CELL-004
def test_every_cell_is_captured_with_the_endpoint_arguments(on_cpu):
    run = on_cpu(n=2)
    env, core = run.env, run.core
    core.startup()
    binding = api.binding_of(env.model)
    assert len(binding.cells) == 2 * env.n
    for key, cell in binding.cells.items():
        fields = signature_fields(cell)
        assert fields["endpoint_enabled"] is True, key
        assert fields["endpoint_policy"] is False and fields["forced_eou"] is False
        # The emission adapter, the evidence-capture flag and, for a chunk
        # cell, the bucket transition are part of the signature.
        assert "adapter" in fields and fields["capture"] is False, key
        assert ("bucket_transition" in fields) == (key[1] == api.CHUNK), key
    # The capture's transactions received both endpoint pools: the cells'
    # signatures list them with the other projected resident-pool views.
    pools = signature_fields(next(iter(binding.cells.values())))["pools"]
    history = t.state_layer_of(env).component_pools["endpoint.history"]
    assert history.data_ptr() in {entry[0] for entry in pools}, "no endpoint pool"


def one_warning(caplog, field):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and field in r.getMessage()
        and "signature" in r.getMessage().lower()
    ]


# @spec TXG-COV-012, TXG-COV-005, TXG-CELL-002, TXG-CELL-004
def test_policy_rows_and_candidates_fall_back_by_their_own_field(
    on_cpu, monkeypatch, caplog
):
    """A covered step's plan, changed only in its endpoint policy or its
    candidate marks, falls back by ``signature`` with the field and the kind
    of the cell whose rows hold the row; a mark on the other kind's rows
    leaves that cell's signature equal."""
    run = on_cpu(n=3, max_num_seqs=3)
    steps = eager.build_covered(monkeypatch, run)
    env = steps.env
    caplog.set_level(logging.WARNING)
    step = env.compose([env.accepted(c) for c in (CC, OC, CF)])
    assert steps.run(step) is not None, "the unchanged plan is covered"

    def fallbacks():
        return dict(t.counter(run.env, "fallbacks"))

    def marked(**fields):
        return dataclasses.replace(step.plan, **fields)

    cases = [
        ("endpoint_policy", api.CHUNK, 2, {"endpoint_mode": [1, 0, 0]}),
        ("endpoint_policy", api.CHUNK, 2, {"endpoint_mode": [0, 0, 1]}),
        ("endpoint_policy", api.ECHO, 1, {"endpoint_mode": [0, 1, 0]}),
        ("forced_eou", api.ECHO, 1, {"forced_eou_candidate": [False, True, False]}),
        # the chunk cell is compared first
        ("endpoint_policy", api.CHUNK, 2, {"endpoint_mode": [1, 1, 1]}),
    ]
    for field, kind, count, change in cases:
        change = {
            name: torch.tensor(
                value, dtype=torch.bool if name.startswith("forced") else torch.int64
            )
            for name, value in change.items()
        }
        before = fallbacks()
        assert steps.run(step, plan=marked(**change)) is None, (field, change)
        after = fallbacks()
        key = ("signature", kind, count)
        assert after.get(key, 0) - before.get(key, 0) == 1, (field, after)
        assert sum(after.values()) - sum(before.values()) == 1
        assert not env.serving.differences(step.pools), "a declined step wrote a page"
    for field in ("endpoint_policy", "forced_eou"):
        assert one_warning(caplog, field), f"no WARNING names {field}"
    # one WARNING for each (kind, field), however many steps fell back
    assert len(one_warning(caplog, "endpoint_policy")) == 2
    assert len(one_warning(caplog, "forced_eou")) == 1
    # A candidate mark cannot sit on a CHUNK row; an empty vector marks none.
    assert steps.run(step) is not None


# ---------------------------------------------------------------------------
# Full forward: option on against B_hook


class Schedule:
    """One schedule of streams over a started environment, step by step, as
    core's scheduler would: a new unit or an EOU request as a new request, a
    request with queued work as a cached one. ``log`` holds each step's
    hidden rows, statuses, failed requests, sampled tokens, state pool and the
    rows the transaction took down the forced-EOU path."""

    def __init__(self, env, core, monkeypatch) -> None:
        self.env, self.core = env, core
        self.park = int(env.config.eos_token_id)
        self.eou = int(env.config.eou_token_id)
        self.slots: dict[str, int] = {}
        self.positions: dict[str, int] = {}
        self.sequence: dict[str, int] = {}
        self.log: list[dict] = []
        self.forced: list[dict[str, bool]] = []
        advance = api.advance_module()
        commit = advance.commit_model_rows

        def watching(plan, body, **kwargs):
            phases = body.progress[:, 2].tolist()
            self.forced.append(
                {r: phase >= 4 for r, phase in zip(plan.request_ids, phases)}
            )
            return commit(plan, body, **kwargs)

        monkeypatch.setattr(advance, "commit_model_rows", watching)

    def unit(self, req_id: str, *, policy: bool = False):
        sequence = self.sequence.get(req_id, -1) + 1
        self.sequence[req_id] = sequence
        request = t.stream_unit(self.env, req_id, sequence=sequence)
        return t.with_endpoint_policy(request) if policy else request

    def eou_request(self, req_id: str):
        return h.new_request(req_id, self.eou)

    def step(self, *new, cached=()) -> dict:
        for request in new:
            self.slots.setdefault(request.req_id, len(self.slots) + 1)
        ids = [r.req_id for r in new] + list(cached)
        forced = len(self.forced)
        hidden, _ = self.core.step(
            *new,
            cached=tuple(cached),
            handles={r: self.slots[r] for r in ids},
            computed={r: self.positions[r] for r in cached},
        )
        for r in cached:
            self.positions[r] += 1
        for request in new:
            self.positions[request.req_id] = 1
        statuses, failed = self.env.runner.commit_log[-1]
        record = {
            "rows": ids,
            "hidden": hidden.clone(),
            "statuses": dict(statuses),
            "failed": set(failed),
            "sampled": {r: self.env.runner._sampled[r] for r in ids},
            "pool": h.pool_tensor(self.core.serving_bindings).clone(),
            "forced": self.forced[forced] if len(self.forced) > forced else None,
            "plan": self.env.advance.calls[-1] if self.env.advance.calls else None,
        }
        self.log.append(record)
        return record

    def drain(self, *ids: str) -> None:
        for _ in range(80):
            waiting = [r for r in ids if self.env.runner._sampled.get(r) != self.park]
            if not waiting:
                return
            self.step(cached=waiting)
        raise AssertionError(f"{ids} never parked")


def run_endpoint_schedule(schedule: Schedule) -> dict[str, int]:
    """The schedule of the full-forward comparison; returns the index of each
    named step in ``schedule.log``."""
    s = schedule
    marks = {}

    def mark(name):
        marks[name] = len(s.log) - 1

    s.step(s.unit("a"), s.unit("b"))
    mark("two first chunks")
    s.drain("a", "b")
    # A policy-free EOU row in a step without a policy row.
    s.step(s.unit("a"), s.eou_request("b"))
    mark("eou without a policy row")
    s.drain("a", "b")
    mark("drained after the first eou")
    # A policy-free EOU row in a step with a policy row.
    s.step(s.unit("c", policy=True), s.eou_request("a"), s.unit("b"))
    mark("eou with a policy row")
    s.drain("a", "b", "c")
    s.step(s.unit("a"))
    mark("a covered chunk afterwards")
    s.drain("a")
    return marks


def start(on_cpu, monkeypatch, *, on: bool):
    run = on_cpu(n=3, max_num_seqs=3, populations=None if on else [])
    env, core = run.env, run.core
    core.startup()
    cells = None
    if on:
        cells = eager.EagerCells(env.model).install(monkeypatch)
    else:
        assert api.binding_of(env.model) is None, "B_hook holds a binding"
    return env, core, cells


def counters_of(env, name):
    return dict(t.counter(env, name))


# @spec TXG-COV-012, TXG-CELL-001, TXG-EQ-001
def test_the_forward_with_the_option_on_equals_b_hook_endpoint_pages_included(
    on_cpu, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING)
    env, core, cells = start(on_cpu, monkeypatch, on=True)
    on = Schedule(env, core, monkeypatch)
    fallbacks = [counters_of(env, "fallbacks")]
    replays = [sum(counters_of(env, "replays").values())]
    step = on.step

    def counted(*new, cached=()):
        record = step(*new, cached=cached)
        fallbacks.append(counters_of(env, "fallbacks"))
        replays.append(sum(counters_of(env, "replays").values()))
        return record

    on.step = counted
    marks = run_endpoint_schedule(on)
    eou = on.eou

    def grew(index):
        return {
            key: value - fallbacks[index].get(key, 0)
            for key, value in fallbacks[index + 1].items()
            if value != fallbacks[index].get(key, 0)
        }

    # The option-on arm is what the comparison is meant to compare.
    first = marks["two first chunks"]
    assert not grew(first) and replays[first + 1] - replays[first] == 1
    without = marks["eou without a policy row"]
    assert grew(without) == {("signature", api.ECHO, 1): 1}, grew(without)
    assert on.log[without]["plan"].forced_eou_candidate.tolist() == [False, True]
    assert not (on.log[without]["plan"].endpoint_mode != 0).any()
    with_policy = marks["eou with a policy row"]
    assert grew(with_policy) == {("signature", api.CHUNK, 2): 1}, grew(with_policy)
    plan = on.log[with_policy]["plan"]
    assert plan.forced_eou_candidate.tolist() == [False, True, False]
    assert plan.endpoint_mode.tolist() == [1, 0, 0]
    for field in ("forced_eou", "endpoint_policy"):
        assert one_warning(caplog, field), f"no WARNING names {field}"
    last = marks["a covered chunk afterwards"]
    assert not grew(last) and replays[last + 1] - replays[last] == 1
    # The EOU rows took the forced-EOU path, and the first one emitted the
    # EOU token, which then came back as a sampled token on a covered step.
    assert on.log[without]["forced"] == {"a": False, "b": True}
    assert on.log[with_policy]["forced"]["a"] is True
    assert on.log[without]["sampled"]["b"] == eou, on.log[without]["sampled"]
    echo = on.log[without + 1]
    assert "b" in echo["rows"] and echo["plan"] is on.log[without]["plan"], (
        "the step after the forced EOU was not covered"
    )
    assert echo["statuses"]["b"] == 0
    # Policy-free rows store endpoint pages in covered steps.
    book = cells.pools["endpoint_book_pool"]
    assert bool(book[on.slots["a"]].any()) and bool(book[on.slots["b"]].any())

    off_env, off_core, _ = start(on_cpu, monkeypatch, on=False)
    off = Schedule(off_env, off_core, monkeypatch)
    off_marks = run_endpoint_schedule(off)
    assert off_marks == marks, "the two arms scheduled different steps"
    assert len(on.log) == len(off.log)
    for index, (mine, theirs) in enumerate(zip(on.log, off.log, strict=True)):
        assert mine["rows"] == theirs["rows"], index
        assert torch.equal(mine["hidden"], theirs["hidden"]), (
            f"step {index} {mine['rows']}: the hidden rows differ from B_hook"
        )
        assert mine["statuses"] == theirs["statuses"], (index, mine, theirs)
        assert mine["failed"] == theirs["failed"], index
        assert mine["sampled"] == theirs["sampled"], index
        assert torch.equal(mine["pool"], theirs["pool"]), (
            f"step {index} {mine['rows']}: the state pool differs from B_hook"
        )
        if mine["forced"] is not None:
            assert mine["forced"] == theirs["forced"], index
    # In B_hook the same rows took the forced-EOU path.
    assert off.log[without]["forced"] == {"a": False, "b": True}
    assert off.log[with_policy]["forced"]["a"] is True
    assert not any(record["failed"] for record in off.log), "a request failed"


@pytest.mark.parametrize("on", [True, False], ids=["option-on", "b-hook"])
# @spec TXG-COV-012
def test_every_transaction_of_a_real_step_receives_the_endpoint_arguments(
    on_cpu, monkeypatch, on
):
    env, core, _cells = start(on_cpu, monkeypatch, on=on)
    received = []
    binding = api.binding_of(env.model)
    if on:
        execute = binding.execute

        def recording(core_, input_ids, inputs_embeds, plan, **transaction):
            received.append(("execute", transaction))
            return execute(core_, input_ids, inputs_embeds, plan, **transaction)

        monkeypatch.setattr(binding, "execute", recording)
    schedule = Schedule(env, core, monkeypatch)
    schedule.step(schedule.unit("a"), schedule.unit("b"))
    schedule.drain("a", "b")
    schedule.step(schedule.unit("a"), schedule.eou_request("b"))
    received += [("advance", kwargs) for kwargs in env.advance.kwargs]
    assert received, "no transaction ran"
    assert {name for name, _ in received} == (
        {"execute", "advance"} if on else {"advance"}
    )
    views = t.state_layer_of(env).component_pools
    for name, transaction in received:
        assert transaction["eou_token_id"] == int(env.config.eou_token_id), name
        history = transaction["endpoint_history_pool"]
        book = transaction["endpoint_book_pool"]
        assert history is not None and book is not None, name
        assert history.data_ptr() == views["endpoint.history"].data_ptr(), name
        assert book.shape[1] == 6, name
