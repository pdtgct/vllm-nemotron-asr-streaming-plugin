# SPDX-License-Identifier: Apache-2.0
"""Split equivalence of the option-off transaction and its exactness
obligations (TXG-EQ-002, TXG-EQ-006 to TXG-EQ-010).

These run the option-off transaction on composed steps
(``support.txg_transaction``): row fixtures from real steps of the runnable
synthetic card, each row on its own block. TXG-EQ-002 runs on CPU
(``engine``) and on CUDA (``gpu``). The covered-step equivalences
(TXG-EQ-001, TXG-EQ-003 to TXG-EQ-005) are in ``test_equivalence_gpu.py``.

Cost: the bounded enumeration of TXG-EQ-002 runs about 14,000 option-off
transactions per geometry and endpointing setting (about 50 minutes on CPU in
all). The exhaustive CPU cases (3- and 4-row plans, the random plans) are
marked ``slow``: a developer loop deselects them with ``-m "not slow"``; the
tier record requires them. The 1- and 2-row plans, the rejection plans and a
small random smoke run stay unmarked.

Seed policy (TXG-EQ-002's random plans): every run draws its plans from
``random.Random(seed)`` with ``seed`` taken unchanged from the preregistered
seed list ``SEED_LIST``, the same listed seed for each geometry and
endpointing setting. A run uses ``SEED_LIST[0]``; ``TXG_EQ002_SEED_INDEX``
selects the next listed seed for a rerun the preregistered record calls for.
A failing plan is reported with its seed and index.
"""

from __future__ import annotations

import itertools
import os
import random

import pytest
import torch
from support import txg_transaction as tx

CONFIGS = tx.CONFIGS

#: The preregistered seed list of TXG-EQ-002's random plans (see the module
#: docstring for the seed policy).
SEED_LIST = (20261114, 20261115, 20261116)
RANDOM_PLANS = 500
SMOKE_PLANS = 25
RANDOM_MAX_ROWS = 16


def run_seed() -> int:
    return SEED_LIST[int(os.environ.get("TXG_EQ002_SEED_INDEX", "0"))]


#: The exhaustive cases: slow on CPU (``engine``), ``gpu`` on CUDA.
EXHAUSTIVE = [
    pytest.param("cpu", marks=[pytest.mark.engine, pytest.mark.slow], id="cpu"),
    pytest.param("cuda", marks=[pytest.mark.engine, pytest.mark.gpu], id="cuda"),
]

DEVICES = [
    pytest.param("cpu", marks=pytest.mark.engine, id="cpu"),
    pytest.param("cuda", marks=[pytest.mark.engine, pytest.mark.gpu], id="cuda"),
]


@pytest.fixture
def env_for(txn_env, request):
    """``env_for(lookahead, device)``: the cached CPU environment, or a CUDA
    environment built for the test (its segment graphs need the test's
    distributed environment)."""
    built = {}

    def get(lookahead: int, device: str):
        if device == "cpu":
            return txn_env(lookahead)
        if (lookahead, device) not in built:
            request.getfixturevalue("single_process_dist")
            tmp = request.getfixturevalue("tmp_path")
            built[(lookahead, device)] = tx.build_env(
                tmp / f"cuda-{lookahead}", lookahead, device="cuda"
            )
        return built[(lookahead, device)]

    return get


def check(env, fixtures, endpoint, failures, label):
    problems = env.split_identity(env.compose(fixtures, endpoint=endpoint))
    if problems:
        failures.append((label, [f.name for f in fixtures], problems))


def report(failures, limit=10):
    lines = [f"{len(failures)} plans failed the split identity"]
    for label, names, problems in failures[:limit]:
        lines.append(f"  {label}: {names}: {problems}")
    return "\n".join(lines)


# @spec TXG-EQ-002
def test_the_enumeration_bound_is_4680_plans():
    assert len(CONFIGS) == 8
    assert sum(len(CONFIGS) ** p for p in range(1, 5)) == 4680


@pytest.mark.parametrize("device", DEVICES)
# @spec TXG-EQ-002
def test_accepted_fixtures_are_accepted(env_for, device):
    """Every configuration the planner produces has an accepted fixture; the
    two derived fresh non-CHUNK rows are reported with their status."""
    for lookahead in (0, 1):
        env = env_for(lookahead, device)
        for config in CONFIGS:
            fixture = env.accepted(config)
            if config[0] == "chunk" or not config[1]:
                assert fixture.status == 0, (lookahead, fixture.name, fixture.status)
            else:
                assert fixture.status is not None


@pytest.mark.parametrize(
    "rows, device",
    [
        *[
            pytest.param(r, d.values[0], marks=d.marks, id=f"rows{r}-{d.id}")
            for r in (1, 2)
            for d in DEVICES
        ],
        *[
            pytest.param(r, d.values[0], marks=d.marks, id=f"rows{r}-{d.id}")
            for r in (3, 4)
            for d in EXHAUSTIVE
        ],
    ],
)
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "endpoint-on"])
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-002
def test_split_identity_on_every_accepted_plan(
    env_for, device, endpoint, lookahead, rows
):
    """``T(R)`` equals the merge of ``T(R|C)`` and ``T(R|O)`` for every plan
    of ``rows`` rows over every assignment of kind, fresh bit and file bit."""
    env = env_for(lookahead, device)
    failures = []
    for configs in itertools.product(CONFIGS, repeat=rows):
        check(
            env,
            [env.accepted(c) for c in configs],
            endpoint,
            failures,
            [tx.config_name(c) for c in configs],
        )
    assert not failures, report(failures)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "endpoint-on"])
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-002
def test_split_identity_with_one_rejected_row(
    env_for, device, endpoint, lookahead, record_property
):
    """Every plan of one or two rows in which one row carries each rejection
    class reachable for its configuration; the excluded combinations are
    recorded."""
    env = env_for(lookahead, device)
    rejections = tx.rejection_fixtures(env)
    failures = []
    plans = 0
    for config, items in rejections.items():
        for name, rejected in items:
            assert env.status_of(rejected) == rejected.intended, rejected.name
            check(env, [rejected], endpoint, failures, name)
            plans += 1
            for other in CONFIGS:
                accepted = env.accepted(other)
                check(env, [rejected, accepted], endpoint, failures, name)
                check(env, [accepted, rejected], endpoint, failures, name)
                plans += 2
    excluded = sorted(set(env.excluded))
    record_property("txg_eq_002_rejection_plans", plans)
    record_property("txg_eq_002_excluded", excluded)
    for config, name, _ in tx.STRUCTURAL_EXCLUSIONS:
        assert (tx.config_name(config), name) in {(c, n) for c, n, _ in excluded}
    for config in CONFIGS:
        if config[0] == "chunk" or not config[1]:
            assert rejections[config], (
                f"no rejection class for {tx.config_name(config)}"
            )
    assert not failures, report(failures)


def random_plans(env, endpoint, plans):
    rejections = tx.rejection_fixtures(env)
    seed = run_seed()
    rng = random.Random(seed)
    failures = []
    for index in range(plans):
        population = rng.randint(1, RANDOM_MAX_ROWS)
        fixtures = []
        for _ in range(population):
            config = CONFIGS[rng.randrange(len(CONFIGS))]
            if rng.random() < 0.75 or not rejections[config]:
                fixtures.append(env.accepted(config))
            else:
                choices = rejections[config]
                fixtures.append(choices[rng.randrange(len(choices))][1])
        check(env, fixtures, endpoint, failures, f"seed {seed} plan {index}")
    return failures


@pytest.mark.parametrize("device", EXHAUSTIVE)
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "endpoint-on"])
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-EQ-002
def test_split_identity_on_random_plans(env_for, device, endpoint, lookahead):
    env = env_for(lookahead, device)
    failures = random_plans(env, endpoint, RANDOM_PLANS)
    assert not failures, report(failures)


@pytest.mark.engine
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "endpoint-on"])
# @spec TXG-EQ-002
def test_split_identity_on_a_random_smoke_run(env_for, endpoint):
    """The first plans of the seeded run (a quick check; the full run is
    ``test_split_identity_on_random_plans``)."""
    failures = random_plans(env_for(0, "cpu"), endpoint, SMOKE_PLANS)
    assert not failures, report(failures)


# ---------------------------------------------------------------------------
# Recycled pages of fresh rows


SENTINELS = ["nan", "inf", "-inf", "random"]


@pytest.mark.engine
# @spec TXG-EQ-006, TXG-MASK-001
@pytest.mark.parametrize("pattern", SENTINELS)
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
@pytest.mark.parametrize("endpoint", [False, True], ids=["endpoint-off", "on"])
def test_recycled_pages_of_fresh_rows_do_not_matter(
    txn_env, pattern, lookahead, endpoint
):
    env = txn_env(lookahead)
    rejected = next(
        f
        for name, f in tx.rejection_fixtures(env)[("chunk", True, False)]
        if name == "ROW_STATUS_SEQUENCE"
    )
    rows = [
        env.accepted(("chunk", True, False)),
        env.accepted(("chunk", False, False)),
        env.accepted(("chunk", True, True)),
        env.accepted(("other", True, False)),
        env.accepted(("other", False, True)),
        env.accepted(("other", True, True)),
        rejected,
    ]
    fresh_blocks = [i + 1 for i, f in enumerate(rows) if f.fresh]
    zeroed = env.compose(rows, endpoint=endpoint)
    sentinel = env.compose(rows, endpoint=endpoint)
    for block in fresh_blocks:
        zeroed.pools.fill_page(block, "zero")
        sentinel.pools.fill_page(block, pattern, seed=block)
    a_before, b_before = zeroed.pools.clone(), sentinel.pools.clone()
    a, b = env.run(zeroed), env.run(sentinel)
    for field in ("rows", "status", "progress", "queue_counts"):
        assert tx.same_bits(getattr(a, field), getattr(b, field)), field
    for i, fixture in enumerate(rows):
        block = [i + 1]
        if int(a.status[i]) == 0:
            # A page the row stored equals across the runs; a page it did not
            # store (a non-CHUNK row stores no encoder state) is unchanged.
            for name in a.pools.differences(b.pools, blocks=block):
                unchanged = name not in a.pools.differences(a_before, blocks=block)
                unchanged &= name not in b.pools.differences(b_before, blocks=block)
                assert unchanged, (fixture.name, name)
        else:
            assert not a.pools.differences(a_before, blocks=block), fixture.name
            assert not b.pools.differences(b_before, blocks=block), fixture.name
    assert int(a.status[-1]) != 0, "the rejected fresh row was accepted"


# ---------------------------------------------------------------------------
# 160 ms rows

L = 1
REGULAR_FRAMES = 16
REGULAR_SAMPLES = 2560


def frontend_calls(monkeypatch, perturb_final_targets: int = 0):
    """Spy on the frontend: each call's targets, finals and committed counts.
    ``perturb_final_targets`` adds that many frames to final rows' targets."""
    from vllm_nemotron_asr_streaming.nemotron_asr import frontend

    calls = []
    real = frontend.advance_frontend

    def spy(featurizer, samples, valid, final, targets, **kwargs):
        if perturb_final_targets:
            targets = torch.where(final, targets + perturb_final_targets, targets)
        out = real(featurizer, samples, valid, final, targets, **kwargs)
        calls.append(
            {
                "targets": targets.detach().cpu().clone(),
                "final": final.detach().cpu().clone(),
                "counts": out[1].detach().cpu().clone(),
            }
        )
        return out

    monkeypatch.setattr(frontend, "advance_frontend", spy)
    return calls


def encoder_inputs(monkeypatch):
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    calls = []
    real = advance.prepare_encoder_transition_inputs

    def spy(mel, caches, drop, enc_lengths, out_width, prompt_index, mel_len):
        calls.append(
            {
                "drop": drop.detach().cpu().clone(),
                "enc_lengths": enc_lengths.detach().cpu().clone(),
                "mel_len": mel_len.detach().cpu().clone(),
            }
        )
        return real(mel, caches, drop, enc_lengths, out_width, prompt_index, mel_len)

    monkeypatch.setattr(advance, "prepare_encoder_transition_inputs", spy)
    return calls


def chunk_rows_160(env):
    cont = env.accepted(("chunk", False, False))
    rows = [
        env.accepted(("chunk", True, False)),
        cont,
        env.accepted(("chunk", True, True)),
        env.accepted(("chunk", False, True)),
        tx.final_tail(cont, 900, L),
        tx.final_tail(cont, 0, L),
    ]
    for fixture in rows:
        # Each row is accepted on its own before any frame arithmetic is read.
        assert env.status_of(fixture) == 0, fixture.name
    for fixture in rows[-2:]:
        assert int(fixture.embed[tx.H_FINAL]) == 1
        assert int(fixture.embed[tx.H_SAMPLES]) < tx.chunk_samples(L)
    return rows


@pytest.mark.engine
# @spec TXG-EQ-007
def test_160_ms_chunk_targets(txn_env, monkeypatch):
    env = txn_env(L)
    assert tx.chunk_samples(L) == REGULAR_SAMPLES
    rows = chunk_rows_160(env)
    calls = frontend_calls(monkeypatch)
    env.run(env.compose(rows + [env.accepted(("other", False, False))]))
    assert len(calls) == 1
    targets, final = calls[0]["targets"], calls[0]["final"]
    for j, fixture in enumerate(rows):
        if not bool(final[j]):
            seq = tx.sequence_of(fixture)
            assert int(targets[j]) == (8 * L + 1) + REGULAR_FRAMES * seq == 9 + 16 * seq


@pytest.mark.engine
# @spec TXG-EQ-007
def test_160_ms_final_tail_ignores_its_target(txn_env, monkeypatch):
    env = txn_env(L)
    rows = chunk_rows_160(env)
    base = env.run(env.compose(rows))
    frontend_calls(monkeypatch, perturb_final_targets=5)
    moved = env.run(env.compose(rows))
    for field in ("rows", "status", "progress"):
        assert tx.same_bits(getattr(base, field), getattr(moved, field)), field
    assert not base.pools.differences(moved.pools)


@pytest.mark.engine
# @spec TXG-EQ-008
def test_160_ms_mel_prefix_and_pre_encode_drop(txn_env, monkeypatch):
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import PRE_ENCODE_DROP
    from vllm_nemotron_asr_streaming.nemotron_asr.frontend import MEL_TAIL_FRAMES

    env = txn_env(L)
    rows = chunk_rows_160(env)
    fronts = frontend_calls(monkeypatch)
    encoders = encoder_inputs(monkeypatch)
    env.run(env.compose(rows))
    counts = fronts[0]["counts"]
    drop, mel_len = encoders[0]["drop"], encoders[0]["mel_len"]
    assert MEL_TAIL_FRAMES == 9
    for j, fixture in enumerate(rows):
        first = tx.sequence_of(fixture) == 0
        if first:
            assert int(mel_len[j]) == int(counts[j]), fixture.name
            assert int(drop[j]) == 0, fixture.name
        else:
            assert int(mel_len[j]) == int(counts[j]) + 9, fixture.name
            assert int(drop[j]) == PRE_ENCODE_DROP, fixture.name


@pytest.mark.engine
# @spec TXG-EQ-009
def test_160_ms_encoder_lengths_are_per_row_and_clamped(txn_env, monkeypatch):
    env = txn_env(L)
    rows = chunk_rows_160(env)
    fronts = frontend_calls(monkeypatch)
    encoders = encoder_inputs(monkeypatch)
    env.run(env.compose(rows))
    together = encoders[-1]["enc_lengths"]
    counts = fronts[-1]["counts"]
    assert bool((together >= 0).all())
    for j, fixture in enumerate(rows):
        if int(counts[j]) == 0:
            assert int(together[j]) == 0, fixture.name
        env.run(env.compose([fixture]))
        assert int(encoders[-1]["enc_lengths"][0]) == int(together[j]), fixture.name
    assert any(int(c) == 0 for c in counts), "no row produced zero new frames"


@pytest.mark.engine
# @spec TXG-EQ-010
def test_160_ms_decode_reads_at_most_two_encoder_frames(txn_env):
    env = txn_env(L)
    widths = []
    resolver = tx.wrap_decode_resolver(
        env, lambda request, args: widths.append(int(args[0].shape[1]))
    )
    env.run(env.compose(chunk_rows_160(env)), decode_resolver=resolver)
    assert widths and all(w <= L + 1 == 2 for w in widths), widths
