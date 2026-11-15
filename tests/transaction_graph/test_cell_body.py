# SPDX-License-Identifier: Apache-2.0
"""The cell body on CPU: accepted-row equivalence with the option-off
transaction, and no host read where a capture would run (TXG-EQ-001,
TXG-EQ-004, TXG-EQ-005, TXG-CELL-001, TXG-EXEC-005, TXG-EXEC-008,
TXG-EXEC-009, TXG-TXN-002, TXG-CAP-001, TXG-CAP-004).

The fake-capture seam captures cells whose replay runs nothing, and every
capture row is rejected, so neither says what a cell computes for an accepted
row. These tests run the function a cell captures
(``TransactionGraphBinding._run_cell``) eagerly on CPU
(``support.txg_eager``):

* Equivalence. A composed step (``support.txg_transaction``: row fixtures
  from real steps, each row on its own page) goes through the binding's own
  ``execute`` over the serving pools, each replay running its cell's body.
  Its escaped rows, staged status, progress, remaining queue counts and every
  resident page, the endpoint pages included, equal the option-off
  transaction of the same row plan on a copy of the pools, bit for bit. The
  plans are TXG-EQ-002's accepted plans of 1 to 4 rows (every assignment of
  kind, fresh bit and file bit), its one-rejected-row plans, and TXG-EQ-004's
  and TXG-EQ-005's cases. This covers the arena packing, the ``int64``
  vectors, the gathers by step position, the body with the endpoint
  arguments, the predicated scatters and the ``index_copy_`` of the outputs.
  What stays for CUDA is the graph itself.
* Host reads. Every cell's warm-ups and capture, at every count and with the
  nested segment bindings on their uncaptured paths, and covered steps with
  accepted rows, run under a dispatch mode that raises on an operation a
  graph capture cannot hold (``txg_eager.SyncDetector``).

The 3- and 4-row enumeration (4,608 plans a geometry) is marked ``slow``; a
seeded sample of it stays in the developer loop.
"""

from __future__ import annotations

import collections
import itertools
import random

import pytest
import torch
from support import txg_api as api
from support import txg_eager as eager
from support import txg_transaction as tx

pytestmark = pytest.mark.engine

CONFIGS = tx.CONFIGS
N = 4
SEED = 20261115
SAMPLE = 200

CC = ("chunk", False, False)
CF = ("chunk", True, False)
CCF = ("chunk", False, True)
CFF = ("chunk", True, True)
OC = ("other", False, False)
OF = ("other", True, False)
OCF = ("other", False, True)
OFF = ("other", True, True)


@pytest.fixture
def covered(on_cpu, monkeypatch):
    """``covered(lookahead)``: an option-on CPU environment at ``N = 4 =
    max_num_seqs`` whose cells compute (``txg_eager.build_covered``)."""

    def build(lookahead: int = 0):
        run = on_cpu(n=N, max_num_seqs=N, right_context=lookahead)
        return eager.build_covered(monkeypatch, run, lookahead=lookahead)

    return build


def check(covered_steps, fixtures, failures, label):
    env = covered_steps.env
    problems = covered_steps.compare(env.compose(fixtures))
    if problems:
        failures.append((label, [f.name for f in fixtures], problems))


def report(failures, limit=10):
    lines = [f"{len(failures)} plans differ from the option-off transaction"]
    for label, names, problems in failures[:limit]:
        lines.append(f"  {label}: {names}: {problems}")
    return "\n".join(lines)


def assert_only_covered(covered_steps, plans: int) -> None:
    """Every plan replayed its cells: nothing fell back."""
    counters = api.counters(covered_steps.env.model)["counters"]
    assert not counters["fallbacks"], counters["fallbacks"]
    assert sum(counters["covered_steps"].values()) >= plans, counters


# ---------------------------------------------------------------------------
# Equivalence on accepted and rejected plans


@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-001, TXG-CELL-001, TXG-EXEC-005, TXG-EXEC-008, TXG-EXEC-009
def test_every_plan_of_one_or_two_rows_equals_the_option_off_transaction(
    covered, lookahead
):
    steps = covered(lookahead)
    env = steps.env
    failures, plans = [], 0
    for rows in (1, 2):
        for configs in itertools.product(CONFIGS, repeat=rows):
            check(
                steps,
                [env.accepted(c) for c in configs],
                failures,
                [tx.config_name(c) for c in configs],
            )
            plans += 1
    assert plans == 8 + 64
    assert not failures, report(failures)
    assert_only_covered(steps, plans)
    # Both kinds ran at both counts, and split steps ran both.
    ran = collections.Counter((kind, count) for _g, kind, count in steps.cells.runs)
    for kind in (api.CHUNK, api.ECHO):
        assert ran[(kind, 1)] and ran[(kind, 2)], ran


# @spec TXG-EQ-001
def test_a_seeded_sample_of_three_and_four_row_plans_equals_the_option_off_transaction(
    covered,
):
    steps = covered(0)
    env = steps.env
    rng = random.Random(SEED)
    failures = []
    for index in range(SAMPLE):
        configs = [
            CONFIGS[rng.randrange(len(CONFIGS))] for _ in range(rng.choice((3, 4)))
        ]
        check(
            steps,
            [env.accepted(c) for c in configs],
            failures,
            f"seed {SEED} plan {index}",
        )
    assert not failures, report(failures)
    assert_only_covered(steps, SAMPLE)
    ran = {(kind, count) for _g, kind, count in steps.cells.runs}
    assert {(k, c) for k in (api.CHUNK, api.ECHO) for c in range(1, N + 1)} <= ran, (
        "the sample did not replay every cell"
    )


@pytest.mark.slow
@pytest.mark.parametrize("rows", [3, 4])
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-001
def test_every_plan_of_three_or_four_rows_equals_the_option_off_transaction(
    covered, lookahead, rows
):
    steps = covered(lookahead)
    env = steps.env
    failures, plans = [], 0
    for configs in itertools.product(CONFIGS, repeat=rows):
        check(
            steps,
            [env.accepted(c) for c in configs],
            failures,
            [tx.config_name(c) for c in configs],
        )
        plans += 1
    assert plans == len(CONFIGS) ** rows
    assert not failures, report(failures)
    assert_only_covered(steps, plans)


@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-001, TXG-MASK-005, TXG-MASK-006
def test_plans_with_one_rejected_row_equal_the_option_off_transaction(
    covered, lookahead
):
    """Each rejection class a fixture reproduces, alone and beside an
    accepted row of every configuration, in both orders: a rejected row's
    pages stay, its output is the rejected-row output, and the accepted row
    beside it is unchanged."""
    steps = covered(lookahead)
    env = steps.env
    rejections = tx.rejection_fixtures(env)
    failures, plans = [], 0
    for items in rejections.values():
        for name, rejected in items:
            assert env.status_of(rejected) == rejected.intended, rejected.name
            check(steps, [rejected], failures, name)
            plans += 1
            for other in CONFIGS:
                accepted = env.accepted(other)
                check(steps, [rejected, accepted], failures, name)
                check(steps, [accepted, rejected], failures, name)
                plans += 2
    assert plans, "no rejection fixture was built"
    assert not failures, report(failures)
    assert_only_covered(steps, plans)


# ---------------------------------------------------------------------------
# The cases of TXG-EQ-004 and TXG-EQ-005


def masked_row_cases(env, lookahead):
    """``{case: fixtures}``: one composed step per case of TXG-EQ-004, and
    the 160 ms rows of TXG-EQ-005, from the accepted fixtures."""
    full = tx.chunk_samples(lookahead)
    cont, first = env.accepted(CC), env.accepted(CF)
    file_first, file_cont = env.accepted(CFF), env.accepted(CCF)
    drain, file_drain = env.accepted(OC), env.accepted(OCF)
    park, placeholder = env.park, env.placeholder
    silent = cont.but(
        embed=torch.cat(
            [cont.embed[: tx.HEADER_SLOTS], cont.embed[tx.HEADER_SLOTS :] * 0]
        ),
        name="silent-chunk",
    )
    # A file's rows by what its real step did: an end step (a non-CHUNK file
    # row whose token is the park's twin of a drained queue) and the armed
    # echo on a file CHUNK row (a CHUNK row whose token is a label).
    controls = {
        park,
        placeholder,
        int(env.config.file_carrier_token_id),
        int(env.config.continuation_token_id),
        int(env.config.eou_token_id),
    }
    file_chunks = env.fixtures[CCF]
    armed = [f for f in file_chunks if f.token not in controls and f.status == 0]
    cases = {
        "first chunk among continuing rows": [first, cont, cont.but(name="cont-2")],
        "file first step": [file_first],
        "regular chunk": [cont],
        "silent chunk": [silent, file_cont],
        "final tail with samples": [tx.final_tail(cont, 600, lookahead)],
        "empty final tail": [tx.final_tail(cont, 0, lookahead)],
        "final tail among regular continuing rows": [
            tx.final_tail(cont, 900, lookahead),
            cont,
        ],
        "one-chunk session": [tx.final_tail(first, 500, lookahead)],
        "drain step": [drain],
        "file drain and end steps": [file_drain, *env.fixtures[OCF][-2:]],
        "split step mixing file and stream rows in both kinds": [
            file_cont,
            drain,
            cont,
            file_drain,
        ],
        "interleaved split step": [drain, cont, file_drain, first],
        "fresh non-CHUNK row accepted as a park echo": [
            env.accepted(OF).but(token=park),
            cont,
        ],
        "fresh non-CHUNK row rejected": [
            env.accepted(OF).but(token=7, name="fresh-other-label"),
            env.accepted(OFF).but(token=placeholder, name="fresh-file-placeholder"),
        ],
        "oversized final tail": [
            cont.with_header(final=1.0, samples=float(full)).but(name="oversized-tail"),
            cont,
        ],
    }
    if armed:
        cases["armed echo on a file CHUNK row"] = [armed[0], cont]
    return cases


@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-004, TXG-EQ-005, TXG-MASK-005
def test_each_masked_row_case_equals_the_option_off_transaction(covered, lookahead):
    steps = covered(lookahead)
    env = steps.env
    cases = masked_row_cases(env, lookahead)
    failures = []
    statuses = {}
    for case, fixtures in cases.items():
        step = env.compose(fixtures)
        problems = steps.compare(step)
        if problems:
            failures.append((case, [f.name for f in fixtures], problems))
        statuses[case] = [int(s) for s in env.run(step).status.tolist()]
    assert not failures, report(failures)
    assert_only_covered(steps, len(cases))
    # The cases are what they claim to be.
    for case in (
        "first chunk among continuing rows",
        "file first step",
        "regular chunk",
        "final tail with samples",
        "empty final tail",
        "drain step",
        "split step mixing file and stream rows in both kinds",
    ):
        assert all(s == 0 for s in statuses[case]), (case, statuses[case])
    assert statuses["fresh non-CHUNK row accepted as a park echo"][0] == 0
    assert all(s != 0 for s in statuses["fresh non-CHUNK row rejected"])
    assert statuses["oversized final tail"][0] != 0
    assert statuses["oversized final tail"][1] == 0


# @spec TXG-CELL-001, TXG-COV-012
def test_a_covered_step_stores_its_endpoint_pages(covered):
    """The cells run with the endpoint arguments: a fresh CHUNK row's
    endpoint pages are initialized and stored, and a CHUNK row's endpoint
    book follows its output, exactly as in the option-off transaction (the
    comparison above). Here: those pages do change, so that comparison is
    not one of untouched pages."""
    steps = covered(0)
    env = steps.env
    fresh, cont = env.accepted(CF), env.accepted(CC)
    step = env.compose([fresh, cont])
    for block in (1, 2):
        step.pools.fill_page(block, "random", seed=block)
    # The continuing row keeps a valid page; only its endpoint pages are
    # scrambled, so a stored endpoint page is visible as a change.
    step.pools.write_page(2, cont.page)
    for name in ("endpoint_history_pool", "endpoint_book_pool"):
        step.pools.tensors[name][2].fill_(0)
    before = step.pools.clone()
    assert not steps.compare(step)
    after = steps.run(step)
    assert [int(s) for s in after.status.tolist()] == [0, 0]
    changed = set(after.pools.differences(before, blocks=[1]))
    assert {"endpoint_history_pool", "endpoint_book_pool"} <= changed, changed
    off = env.run(step)
    for name in ("endpoint_history_pool", "endpoint_book_pool"):
        assert tx.same_bits(after.pools.tensors[name], off.pools.tensors[name]), name


# ---------------------------------------------------------------------------
# No host read where a capture runs


# @spec TXG-TXN-002, TXG-CAP-001
def test_the_detector_refuses_each_host_read_and_host_built_tensor():
    """Positive control: each operation a capture cannot hold raises."""
    x = torch.arange(6.0)
    refused = {
        "int": lambda: int(x[0]),
        "bool": lambda: bool(x[0] > 1),
        "item": lambda: x[1].item(),
        "tolist": lambda: x.tolist(),
        "numpy": lambda: x.numpy(),
        "nonzero": lambda: (x > 1).nonzero(),
        "where": lambda: torch.where(x > 1),
        "masked_select": lambda: torch.masked_select(x, x > 1),
        "mask index": lambda: x[x > 1],
        "tensor": lambda: torch.tensor([1, 2]),
        "as_tensor": lambda: torch.as_tensor([1.0, 2.0]),
        "equal": lambda: torch.equal(x, x),
    }
    for name, operation in refused.items():
        with pytest.raises(eager.SyncError), eager.detecting():
            operation()
        operation()  # and it is restored outside the detector
        assert not eager._ACTIVE, name
    allowed = {
        "select": lambda: torch.where(x > 1, 2.0, 1.0),
        "gather": lambda: x.index_select(0, torch.zeros(2, dtype=torch.int64)),
        "index_copy": lambda: x.clone().index_copy_(0, x[:2].long(), x[:2]),
        "full": lambda: torch.full_like(x, 3.0),
        "arange": lambda: torch.arange(4),
    }
    for name, operation in allowed.items():
        with eager.detecting() as detector:
            operation()
        assert detector.seen, name


def nested_calls(monkeypatch, env, log):
    """Log each call of a cell's nested uncaptured path (the dense decoder's
    and the encoder's) with whether a detector was active."""
    graph = api.graph_module()
    real = graph._nested_launch

    def nested_launch(captured, uncaptured):
        def watched(*args, **kwargs):
            log.append(bool(eager._ACTIVE))
            return uncaptured(*args, **kwargs)

        return real(captured, watched)

    monkeypatch.setattr(graph, "_nested_launch", nested_launch)


@pytest.mark.parametrize("capture", ["profiling", "serving"])
# @spec TXG-TXN-002, TXG-CAP-001, TXG-CAP-004
def test_every_cells_warm_ups_and_capture_read_nothing_on_the_host(
    on_cpu, monkeypatch, capture
):
    """Each of the ``2N`` cells runs its body four times inside the flat
    capture scope (three warm-ups and the capture). All of them, at every
    count, with the nested encoder and decoder on their uncaptured paths,
    pass the detector."""
    from vllm_nemotron_asr_streaming.nemotron_asr import state_scatter

    advance = api.advance_module()
    run = on_cpu(n=3, max_num_seqs=3)
    env, core = run.env, run.core
    core.prepare()
    core.bind_profiling()
    if capture == "serving":
        core.profiling_capture()
        core.clear()
        core.bind_serving()
    # The CPU scatter executor reads each row's status on the host; the CUDA
    # executor is one kernel launch.
    monkeypatch.setattr(
        state_scatter, "execute_masked_page_scatter_", eager.sync_free_scatter
    )
    eager.pause_in_host_only_functions(monkeypatch)
    nested: list[bool] = []
    nested_calls(monkeypatch, env, nested)
    binding = api.binding_of(env.model)
    real = binding._run_cell
    guarded: list[tuple[str, int, int]] = []
    outside: list[tuple[str, int]] = []

    def run_cell(kind, count, *args):
        if not advance.flat_capture_active():
            outside.append((kind, count))  # the eager reference
            return real(kind, count, *args)
        with eager.detecting() as detector:
            real(kind, count, *args)
        guarded.append((kind, count, detector.seen))

    monkeypatch.setattr(binding, "_run_cell", run_cell)
    core.capture(profile_only=capture == "profiling")
    cells = [(kind, count) for _g, kind, count in api.cell_keys(env.n, env.geometry)]
    assert collections.Counter((k, c) for k, c, _ in guarded) == dict.fromkeys(cells, 4)
    assert collections.Counter(outside) == dict.fromkeys(cells, 1)
    assert all(seen > 0 for _k, _c, seen in guarded), "a body ran no operation"
    # Each chunk cell's four guarded runs took both nested uncaptured paths.
    assert all(nested), "a nested path ran outside the detector"
    assert len(nested) >= 2 * 4 * env.n, len(nested)
    assert len(binding.cells) == 2 * env.n


@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-TXN-002, TXG-TXN-003
def test_covered_steps_with_accepted_rows_read_nothing_on_the_host(
    covered, monkeypatch, lookahead
):
    """The same bodies on accepted rows: every case of TXG-EQ-004 and
    TXG-EQ-005 and a split step at every pair of counts, each replay under
    the detector, and still equal to the option-off transaction."""
    from vllm_nemotron_asr_streaming.nemotron_asr import state_scatter

    steps = covered(lookahead)
    env = steps.env
    eager.pause_in_host_only_functions(monkeypatch)
    nested: list[bool] = []
    nested_calls(monkeypatch, env, nested)
    steps.cells._programs.clear()  # rebuild the programs over the watched paths
    monkeypatch.setattr(
        state_scatter, "execute_masked_page_scatter_", eager.sync_free_scatter
    )
    seen = []

    def inside(key):
        detector = eager.SyncDetector()
        seen.append(detector)
        return detector

    steps.cells.inside = inside
    plans = list(masked_row_cases(env, lookahead).values())
    chunk, other = env.accepted(CC), env.accepted(OC)
    for k in range(0, N + 1):
        for e in range(0, N + 1 - k):
            if k + e:
                plans.append([chunk] * k + [other] * e)
    failures = []
    for index, fixtures in enumerate(plans):
        check(steps, fixtures, failures, f"plan {index}")
    assert not failures, report(failures)
    assert_only_covered(steps, len(plans))
    assert seen and all(d.seen > 0 for d in seen)
    assert nested and all(nested), "a nested path ran outside the detector"
