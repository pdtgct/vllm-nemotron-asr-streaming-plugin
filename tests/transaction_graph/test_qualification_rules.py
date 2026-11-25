# SPDX-License-Identifier: Apache-2.0
"""The tier 5 decision rules (``support.txgcap_qual``) on hand-made figures:
each biconditional at and across its boundary, the void rule and seed list,
the preregistration freeze, the ``U_api`` procedure, the restricted startup
of every qualifying engine, and the choice of ``N`` (TXG-QUAL-001, -005,
-010, -013, -014, -016 to -018, -025, -027). CPU only; no vLLM."""

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


# @spec TXG-QUAL-016
def test_u_api_runs_five_brackets_each_leg_after_a_washout_of_its_arm():
    schedule = q.u_api_schedule()
    counted = [r for r in schedule if not r["washout"]]
    assert [(r["bracket"], r["leg"]) for r in counted] == [
        (b, leg) for b in range(5) for leg in ("OFF1", "ON", "OFF2")
    ]
    for washout, leg in zip(schedule[::2], schedule[1::2], strict=True):
        assert washout["washout"] and not leg["washout"]
        assert (washout["bracket"], washout["leg"], washout["arm"]) == (
            leg["bracket"],
            leg["leg"],
            leg["arm"],
        )
        assert leg["arm"] == ("on" if leg["leg"] == "ON" else "off")


def u_leg(arm, *, tokens=3, parks=1, errors=0, deltas=None, repeats=4):
    on = {"chunk1": 1, "echo1": tokens - 1, "other_replays": 0, "fallbacks": 0}
    repeat = {
        "ms": 30.0,
        "parks": parks,
        "errors": errors,
        "tokens": tokens,
        "deltas": on if deltas is None else deltas,
    }
    return {"arm": arm, "leg": arm.upper(), "repeats": [dict(repeat)] * repeats}


# @spec TXG-QUAL-016
def test_each_u_api_repeat_parks_once_and_an_on_leg_replays_the_cells_at_one():
    assert q.u_api_leg_problems(u_leg("on")) == []
    assert q.u_api_leg_problems(u_leg("off", deltas={})) == []
    assert q.u_api_leg_problems(u_leg("on", repeats=3)), "three repeats passed"
    assert q.u_api_leg_problems(u_leg("off", parks=2)), "two parks passed"
    assert q.u_api_leg_problems(u_leg("off", parks=0)), "no park passed"
    assert q.u_api_leg_problems(u_leg("off", errors=1)), "an error output passed"
    for name, value in (
        ("chunk1", 2),
        ("echo1", 1),
        ("other_replays", 1),
        ("fallbacks", 1),
    ):
        deltas = {"chunk1": 1, "echo1": 2, "other_replays": 0, "fallbacks": 0}
        deltas[name] = value
        assert q.u_api_leg_problems(u_leg("on", deltas=deltas)), name


# @spec TXG-QUAL-016, TXG-QUAL-017
def test_a_u_api_bracket_takes_its_legs_in_order():
    legs = [
        {"leg": leg, "repeats": [{"ms": ms}] * 4}
        for leg, ms in (("OFF1", 40.0), ("ON", 30.0), ("OFF2", 40.0))
    ]
    bracket = q.u_api_bracket(legs)
    assert bracket.u() == (40.0, 30.0, 40.0)
    with pytest.raises(ValueError):
        q.u_api_bracket(legs[::-1])


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
def test_the_frozen_record_names_the_adapter_and_every_procedure():
    """The step rule's adapter identity is frozen with the prefix, snapshot
    rule, plans, compositions and aggregation; there is no body check."""
    complete = {"parameters": {k: "x" for k in q.FROZEN_PARAMETERS}}
    assert q.missing_frozen_parameters(complete) == []
    for name in ("adapter_identity", "snapshot_rule", "u_api_input_generator"):
        record = {"parameters": dict(complete["parameters"])}
        del record["parameters"][name]
        assert q.missing_frozen_parameters(record) == [name]
    assert not [k for k in q.FROZEN_PARAMETERS if "body" in k]


# @spec TXG-QUAL-018
def test_the_adapter_identity_is_its_source_digest():
    from support import txg_qual_adapter as adapter

    identity = adapter.adapter_identity()
    assert len(identity) == 64 and identity == adapter.adapter_identity()


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
def test_the_served_run_uses_the_largest_accepted_n_when_it_fits():
    """``N = min(P_max, admission limit)``, else the largest ``N`` below it
    that fits."""
    fits = {48: True, 56: True, 64: True}
    assert q.served_run_n(64, 64, fits) == 64
    assert q.served_run_n(64, 56, fits) == 56, "the admission limit bounds N"
    assert q.served_run_n(48, 64, fits) == 48, "P_max bounds N"
    assert q.served_run_n(64, 64, {48: True, 56: True, 64: False}) == 56
    assert q.served_run_n(64, 64, {64: False}) is None


# @spec TXG-QUAL-025
def test_fits_means_both_captures_under_max_num_seqs_plus_one_with_the_estimate():
    settings = q.fit_startup(64)
    assert settings["num_gpu_blocks_override"] == 65
    assert q.estimate_enabled() == {q.ESTIMATE_ENV: "1"}
    assert "kv_cache_memory_bytes" not in settings


# @spec TXG-QUAL-001, TXG-QUAL-010
def test_every_qualifying_engine_runs_the_restricted_startup_with_a_fixed_block_count():
    settings = q.restricted_startup(64)
    assert settings["async_scheduling"] is False
    assert settings["compilation_config"] == {"cudagraph_mode": "NONE", "mode": 0}
    assert settings["num_gpu_blocks_override"] == 65
    assert "kv_cache_memory_bytes" not in settings
    # the admission limit, not max_num_seqs, fixes the block count
    limited = q.restricted_startup(64, max_num_active_seqs=32)
    assert limited["num_gpu_blocks_override"] == 33


# @spec TXG-QUAL-001, TXG-QUAL-010
def test_the_driver_runs_b_hook_with_the_same_settings_and_the_estimate():
    """``B_hook`` differs from the option-on run only by the absent key: the
    block count included (TXG-QUAL-001); every run enables core's
    graph-memory estimate (TXG-QUAL-010)."""
    from support import txg_qual_driver as d

    key = "transaction_graph_populations"
    on = {
        "card": "/card",
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "hf_overrides": {"num_lookahead_tokens": 0, key: [64]},
    }
    off = on | {"hf_overrides": {"num_lookahead_tokens": 0}}
    a, b = d.engine_settings(on), d.engine_settings(off)
    assert {k: v for k, v in a.items() if k != "hf_overrides"} == {
        k: v for k, v in b.items() if k != "hf_overrides"
    }
    assert a["num_gpu_blocks_override"] == 65 and a["async_scheduling"] is False
    assert d.engine_env(on)[q.ESTIMATE_ENV] == "1"


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
