# SPDX-License-Identifier: Apache-2.0
"""The tier 5 decision rules (``support.txgcap_qual``) on hand-made figures:
each biconditional at and across its boundary, the void rule and seed list,
the preregistration freeze, and the choice of ``N`` (TXG-QUAL-005, -013,
-014, -015, -017, -018, -025, -027). CPU only; no vLLM."""

from __future__ import annotations

import pytest
from support import txgcap_qual as q


def leg(median: float, spread: float = 0.0) -> list[float]:
    """21 counted samples with the given median."""
    return [median - spread] * 10 + [median] + [median + spread] * 10


def block(off1: float, on: float, off2: float) -> q.Block:
    return q.Block(leg(off1, 0.3), leg(on, 0.3), leg(off2, 0.3))


# ---------------------------------------------------------------------------
# Step rule


@pytest.mark.parametrize(
    "off1,on,off2,expected",
    [
        (20.0, 15.0, 20.0, True),  # Δ = 5 = max(5, 0): material at the boundary
        (20.0, 15.1, 20.0, False),  # Δ just below 5
        (30.0, 15.0, 22.0, False),  # Δ = 7 < drift 8
        (30.0, 15.0, 23.0, True),  # Δ = 8 = drift 7 → max(5, 7) = 7 ≤ 8
        (20.0, 21.0, 20.0, False),  # slower with the option
    ],
)
# @spec TXG-QUAL-013
def test_step_rule_block_is_material_exactly_at_its_threshold(off1, on, off2, expected):
    assert q.block_material(block(off1, on, off2)) is expected


# @spec TXG-QUAL-013
def test_step_rule_requires_both_blocks_and_counts_only_21_samples():
    good, bad = block(20, 10, 20), block(20, 19, 20)
    assert q.material([good, good]) is True
    assert q.material([good, bad]) is False and q.material([bad, good]) is False
    with pytest.raises(ValueError):
        q.material([good])
    warm = q.Block(
        [999.0, 999.0] + leg(20), [0.0, 0.0] + leg(10), [999.0, 999.0] + leg(20)
    )
    assert q.block_material(warm) is True, "warm-up samples were counted"
    with pytest.raises(ValueError):
        q.block_material(q.Block(leg(20)[:20], leg(10), leg(20)))


# @spec TXG-QUAL-014
def test_groups_pass_only_when_every_member_is_material_without_pooling():
    assert q.one_kind_group({1: True, 31: True, 63: True}) is True
    assert q.one_kind_group({1: True, 31: False, 63: True}) is False
    splits = {c: True for c in q.SPLITS}
    assert q.split_group(splits) is True
    splits[(63, 32, 31)] = False
    assert q.split_group(splits) is False
    with pytest.raises(ValueError):
        q.one_kind_group({1: True, 31: True})
    with pytest.raises(ValueError):
        q.split_group({(31, 27, 4): True})


# ---------------------------------------------------------------------------
# Body non-regression


@pytest.mark.parametrize(
    "old1,new,old2,expected",
    [
        (10.0, 11.0, 10.0, True),  # +1 ms = max(1, 0)
        (10.0, 11.2, 10.0, False),
        (10.0, 12.0, 12.5, True),  # +2 ≤ drift 2.5
        (10.0, 12.6, 12.5, False),
        (10.0, 8.0, 10.0, True),  # faster
    ],
)
# @spec TXG-QUAL-015
def test_body_check_block_passes_exactly_at_its_threshold(old1, new, old2, expected):
    assert q.body_block_passes(block(old1, new, old2)) is expected


# @spec TXG-QUAL-015
def test_body_check_requires_both_blocks():
    good, bad = block(10, 10, 10), block(10, 12, 10)
    assert q.body_check([good, good]) is True
    assert q.body_check([good, bad]) is False


# ---------------------------------------------------------------------------
# U_api


def bracket(off1, on, off2, *, discard=1000.0):
    def legs(value):
        return [discard, value - 0.5, value, value + 0.5]

    return q.Bracket(legs(off1), legs(on), legs(off2))


# @spec TXG-QUAL-017
def test_u_api_gain_holds_with_every_condition_in_all_five_brackets():
    brackets = [bracket(40, 30, 40) for _ in range(5)]
    assert q.u_api_gain(brackets) is True
    weak = brackets[:4] + [bracket(40, 36, 40)]
    assert q.u_api_gain(weak) is False, "one bracket below 5 ms passed"
    with pytest.raises(ValueError):
        q.u_api_gain(brackets[:4])


# @spec TXG-QUAL-017
def test_u_api_repeat_zero_is_discarded_and_the_pooled_maximum_is_checked():
    brackets = [bracket(40, 30, 40, discard=1e6) for _ in range(5)]
    assert q.u_api_gain(brackets) is True, "repeat 0 was counted"
    # every bracket passes on its own; pooled over 15 samples, the ON maximum
    # of one bracket meets the OFF2 maximum of another and the drift grows
    wide, close = bracket(100, 50, 80), bracket(79, 73.9, 79)
    mixed = [wide, close, wide, wide, wide]
    assert all(q._gain(*b.u()) for b in mixed)
    assert q.u_api_gain(mixed) is False


# @spec TXG-QUAL-017
def test_u_api_median_condition_is_strict():
    # medians: OFF 40, ON 35 → 40 − 35 − 5 = 0, not > 0; maxima pass
    exact = [
        q.Bracket([0, 30.0, 40, 50.0], [0, 20.0, 35, 35.0], [0, 30.0, 40, 50.0])
        for _ in range(5)
    ]
    assert all(q._gain(*b.u()) for b in exact)
    assert q.u_api_gain(exact) is False


# ---------------------------------------------------------------------------
# Void runs, seeds and preregistration


def counts(**overrides):
    base = {
        "q_split_at_n": 200,
        "q_fresh_among_continuing_at_n": 50,
        "m_split_with_file": 100,
        "m_cases": list(q.EQ004_CASES),
    }
    return base | overrides


@pytest.mark.parametrize(
    "overrides,void",
    [
        ({}, False),
        ({"q_split_at_n": 199}, True),
        ({"q_fresh_among_continuing_at_n": 49}, True),
        ({"m_split_with_file": 99}, True),
        ({"m_cases": list(q.EQ004_CASES[:-1])}, True),
    ],
)
# @spec TXG-QUAL-005
def test_void_rule(overrides, void):
    assert q.run_void(counts(**overrides)) is void


# @spec TXG-QUAL-005
def test_seed_list_outcomes():
    clean = {"mismatches": 0, "fallbacks": 0}
    void = clean | {"counts": counts(q_split_at_n=0)}
    ok = clean | {"counts": counts()}
    assert q.qualification_outcome([void, ok], [7, 8]) == ("pass", [7, 8])
    assert q.qualification_outcome([void, void], [7, 8]) == ("incomplete", [7, 8])
    failed_void = void | {"fallbacks": 1}
    assert q.qualification_outcome([failed_void, ok], [7, 8]) == ("fail", [7])


# @spec TXG-QUAL-018
def test_a_run_counts_only_against_a_matching_record_frozen_before_it():
    params = {"seeds": [1, 2], "concurrencies": [1, 32, 64]}
    record = {"frozen_at": 100.0, "parameters": dict(params)}
    run = {"first_counted_at": 200.0, "parameters": dict(params)}
    assert q.frozen_before_counting(record, run) is True
    assert q.frozen_before_counting(record | {"frozen_at": 300.0}, run) is False
    changed = run | {"parameters": params | {"seeds": [1, 3]}}
    assert q.frozen_before_counting(record, changed) is False
    extra = run | {"parameters": params | {"offset_rule": "new"}}
    assert q.frozen_before_counting(record, extra) is False
    assert q.frozen_before_counting({"parameters": params}, run) is False


# ---------------------------------------------------------------------------
# Choosing N


def histogram(at_63=990, above=10):
    """Served ``(K, E)`` steps: most within 63, a tail above."""
    return {(40, 23): at_63, (70, 1): above}


# @spec TXG-QUAL-025
def test_tier5_inventory_is_the_smallest_fitting_n_at_least_63_with_99_percent():
    fits = {63: True, 64: True, 96: True}
    assert q.tier5_inventory(histogram(990, 10), fits) == 63
    assert q.tier5_inventory(histogram(980, 20), fits) == 96
    assert q.tier5_inventory(histogram(980, 20), {63: True, 64: True}) is None
    assert q.tier5_inventory(histogram(), {63: False, 64: True}) == 64
    assert q.tier5_inventory({(10, 5): 100}, {32: True}) is None, (
        "N below 63 was chosen"
    )


# @spec TXG-QUAL-027
def test_a_deployment_n_below_63_carries_no_tier5_claim():
    choice = q.deployment_choice(
        {(10, 5): 995, (40, 30): 5}, {16: True, 32: True, 64: True}
    )
    assert choice == {"n": 16, "histogram": {(10, 5): 995, (40, 30): 5}, "tier5": False}
    assert q.deployment_choice({(40, 30): 100}, {16: True, 32: True}) is None
