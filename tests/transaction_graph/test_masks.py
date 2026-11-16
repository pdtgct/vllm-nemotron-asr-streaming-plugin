# SPDX-License-Identifier: Apache-2.0
"""Freshness, file and planned-rejection masks of the transaction body with
transaction graphs off, on CPU (TXG-MASK-001 to TXG-MASK-004, TXG-MASK-006,
TXG-MASK-007), and the null page the cells bake in (TXG-CELL-004).

TXG-MASK-001's sentinel check is ``test_equivalence.py``'s
``test_recycled_pages_of_fresh_rows_do_not_matter``; TXG-MASK-003's
control-flow check is ``test_exec_phases.py``'s
``test_body_control_flow_ignores_file_and_fresh_bits``. The cell checks
(TXG-MASK-002's one launch, TXG-MASK-005) are in ``test_exec_gpu.py``; the
capture's null-page checks (TXG-CAP-004, TXG-CAP-008) are in
``test_capture.py``.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest
import torch
from support import txg_api as api
from support import txg_transaction as tx
from support.core_order import CoreOrder
from support.txg_fakes import with_planned_rejection
from torch.utils._python_dispatch import TorchDispatchMode

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


class GatherWatch(TorchDispatchMode):
    """Records each ``index_select`` on a named resident pool (the pool's
    name and the index's values) and each ``masked_fill`` whose input is such
    a gather's output or a view of it."""

    def __init__(self, pools: tx.Pools) -> None:
        super().__init__()
        self.names = {
            (t.data_ptr(), tuple(t.shape), tuple(t.stride())): name
            for name, t in pools.flat()
        }
        self.gathers: list[tuple[str, list[int]]] = []
        self.fills: list[str] = []
        # Gather outputs, kept alive so that no address is reused.
        self._outputs: list[tuple[torch.Tensor, str]] = []

    def _gathered(self, t: object) -> str | None:
        if not isinstance(t, torch.Tensor):
            return None
        ptr = t.untyped_storage().data_ptr()
        for out, name in self._outputs:
            if out.untyped_storage().data_ptr() == ptr:
                return name
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func.overloadpacket.__name__.startswith("masked_fill"):
            name = self._gathered(args[0])
            if name is not None:
                self.fills.append(name)
        out = func(*args, **kwargs)
        if func is torch.ops.aten.index_select.default:
            source = args[0]
            name = self.names.get(
                (source.data_ptr(), tuple(source.shape), tuple(source.stride()))
            )
            if name is not None:
                self.gathers.append((name, args[2].tolist()))
                self._outputs.append((out, name))
        return out


def watch_body(env, step, pools):
    """Run the option-off transaction of ``step`` on ``pools`` with its body
    under a :class:`GatherWatch`."""
    advance = api.advance_module()
    real = api.phase_function(api.PHASES[1])
    watches = []

    def watched(*args, **kwargs):
        watch = GatherWatch(pools)
        watches.append(watch)
        with watch:
            return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(advance, api.PHASES[1], watched)
        out = env.run(step, pools=pools)
    assert len(watches) == 1
    return watches[0], out


# @spec TXG-MASK-001, TXG-MASK-007
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "on"])
def test_fresh_rows_gather_zero_initial_state_from_the_null_page(txn_env, endpoint):
    """Every gather of a zero-initial state tensor reads the null page for
    each fresh row and the row's own page for each continuing row; the book
    is gathered by the rows' own pages; and no gathered state is overwritten
    by a fill (one select on the gather index, not one per state tensor)."""
    env = txn_env(0)
    configs = [CF, CC, OF, OC, CFF, CCF, OFF, OCF]
    rows = [env.accepted(c) for c in configs]
    step = env.compose(rows, endpoint=endpoint)
    null = int(step.plan.null_block_id)
    blocks = step.plan.state_indices_p.tolist()
    fresh = [f.fresh for f in rows]
    chunk = [f.kind == "chunk" for f in rows]
    assert null not in blocks
    initial = [null if fresh[i] else blocks[i] for i in range(len(rows))]
    expected = {
        # step-level gathers (queue, counters, endpoint pages): every row
        len(rows): initial,
        # the CHUNK bucket's gathers (encoder, frontend, predictor state)
        sum(chunk): [v for v, c in zip(initial, chunk) if c],
    }
    assert len(expected) == 2, "the step must tell the two gather kinds apart"
    watch, _ = watch_body(env, step, step.pools.clone())
    zero_initial = {name for name, _ in step.pools.flat()} - {"book_pool"}
    seen = {name for name, _ in watch.gathers}
    assert zero_initial <= seen, sorted(zero_initial - seen)
    for name, index in watch.gathers:
        if name == "book_pool":
            assert list(index) == blocks, name
        else:
            assert list(index) == expected[len(index)], (name, index)
    assert watch.fills == [], f"fills of gathered state: {sorted(set(watch.fills))}"


def nonzero(pages: dict[str, torch.Tensor]) -> list[str]:
    """The names of the pages in ``pages`` that hold a nonzero byte."""
    return [
        name
        for name, t in pages.items()
        if bool(t.contiguous().view(torch.uint8).ne(0).any())
    ]


# @spec TXG-MASK-007, TXG-MASK-006
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "on"])
def test_no_transaction_stores_to_the_null_page(txn_env, lookahead, endpoint):
    """A step of every row configuration, rejected fresh and continuing rows,
    a planning-failed row and CUDA-graph padding entries leaves the null page
    zero, while its accepted fresh rows store their own pages."""
    env = txn_env(lookahead)
    rejections = tx.rejection_fixtures(env)
    rejected = [
        rejections[config][0][1] for config in (CF, CCF, CFF, OCF) if rejections[config]
    ]
    assert any(f.fresh for f in rejected) and any(not f.fresh for f in rejected)
    accepted = [env.accepted(c) for c in tx.CONFIGS]
    rows = accepted + rejected
    step = env.compose(rows, endpoint=endpoint)
    planned = [False] * len(rows)
    planned[tx.CONFIGS.index(CF)] = True
    step.plan = with_planned_rejection(step.plan, planned)
    null = int(step.plan.null_block_id)
    # Two padding entries in the decode indices, as a padded CUDA-graph batch
    # carries them: they name the null block and never enter the body.
    step.plan = dataclasses.replace(
        step.plan, state_indices_d=torch.full((2, 1), null, dtype=torch.int64)
    )
    assert not nonzero(step.pools.page(null))
    before = step.pools.clone()
    out = env.run(step)
    status = [int(s) for s in out.status.tolist()]
    assert all(s != 0 for s in status[len(accepted) :]), status
    assert status[tx.CONFIGS.index(CF)] != 0, status
    assert not nonzero(out.pools.page(null))
    assert not out.pools.differences(before, blocks=[null])
    stored = [
        i
        for i, f in enumerate(rows)
        if f.fresh and status[i] == 0 and out.pools.differences(before, blocks=[i + 1])
    ]
    assert stored, "no accepted fresh row stored its own page"


# @spec TXG-MASK-007
def test_no_driven_step_stores_to_the_null_page(txn_env):
    """The real steps of a stream and a file (fresh, continuing, file, drain
    and end rows) leave the serving pools' null page zero, before every step
    and after the last."""
    for lookahead in (0, 1):
        env = txn_env(lookahead)
        assert len(env.drive_null_pages) > 2
        for i, page in enumerate(env.drive_null_pages):
            assert not nonzero(page), (lookahead, i, nonzero(page))


# @spec TXG-CELL-004, TXG-COV-005, TXG-MASK-001
def test_cells_bake_in_the_null_page(on_cpu, monkeypatch, caplog):
    """Each cell's signature holds the null block ID of its capture rows' plan;
    a covered step whose plan names another null block falls back by
    ``signature`` with field ``null_page``, so no cell gathers a fresh row's
    initial state from a page other than the one it was captured with."""
    from support import txg_eager as eager
    from support import txgcap_harness as t

    run = on_cpu(n=3, max_num_seqs=3)
    steps = eager.build_covered(monkeypatch, run)
    env = steps.env
    binding = api.binding_of(env.model)
    for key, cell in binding.cells.items():
        assert dict(cell.signature)["null_page"] == 0, key
    caplog.set_level(logging.WARNING)
    step = env.compose([env.accepted(c) for c in (CF, OC)])
    assert steps.run(step) is not None, "the unchanged plan is covered"
    free = env.capacity - 1
    assert free not in step.plan.state_indices_p.tolist(), env.capacity
    before = dict(t.counter(run.env, "fallbacks"))
    moved = dataclasses.replace(step.plan, null_block_id=free)
    assert steps.run(step, plan=moved) is None
    after = dict(t.counter(run.env, "fallbacks"))
    key = ("signature", api.CHUNK, 1)
    assert after.get(key, 0) - before.get(key, 0) == 1, after
    assert any(
        "null_page" in r.getMessage() and "signature" in r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), "no WARNING names null_page"


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
