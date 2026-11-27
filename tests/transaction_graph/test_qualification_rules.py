# SPDX-License-Identifier: Apache-2.0
"""The tier 5 decision rules (``support.txgcap_qual``) on hand-made figures:
each biconditional at and across its boundary, the void rule and seed list,
the preregistration freeze, the ``U_api`` procedure, the restricted startup
of every qualifying engine and its KV-cache sizing, the record of every run,
the exactness runs' headroom, and the choice of ``N`` (TXG-QUAL-001, -005,
-010, -013, -014, -016 to -018, -025, -027, -029). CPU only; no vLLM."""

from __future__ import annotations

import pytest
from support import txgcap_qual as q
from support.txg_api import NOT_APPLICABLE, Sizing


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


# @spec TXG-QUAL-018, TXG-QUAL-029
def test_the_frozen_record_holds_the_exactness_headroom_and_its_margin():
    """The exactness runs' test-only headroom and its margin are frozen with
    the rest of the run, so a change after the first counted sample voids
    the run."""
    complete = {"parameters": {k: "x" for k in q.FROZEN_PARAMETERS}}
    for name in ("exactness_headroom", "exactness_headroom_margin"):
        assert name in q.FROZEN_PARAMETERS, name
        record = {"parameters": dict(complete["parameters"])}
        del record["parameters"][name]
        assert q.missing_frozen_parameters(record) == [name]
    params = {"exactness_headroom": 1 << 30, "exactness_headroom_margin": 1 << 28}
    record = {"frozen_at": 1.0, "parameters": dict(params)}
    run = {"first_counted_at": 2.0, "parameters": dict(params)}
    assert q.frozen_before_counting(record, run) is True
    changed = run | {"parameters": params | {"exactness_headroom_margin": 0}}
    assert q.frozen_before_counting(record, changed) is False


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


#: The KV-cache settings of core's ``AsyncEngineArgs``.
KV_SETTINGS = ("num_gpu_blocks_override", "kv_cache_memory_bytes")


# @spec TXG-QUAL-025
def test_fits_means_both_captures_under_max_num_seqs_plus_one_with_the_estimate():
    settings = q.fit_startup(64)
    assert settings["num_gpu_blocks_override"] == 65
    assert q.estimate_enabled() == {q.ESTIMATE_ENV: "1"}
    assert "kv_cache_memory_bytes" not in settings
    assert {k: v for k, v in settings.items() if k not in KV_SETTINGS} == (
        q.restricted_startup()
    )


# @spec TXG-QUAL-001
def test_the_restricted_startup_names_no_kv_cache_setting():
    """The restricted startup is synchronous scheduling and an explicit NONE
    runner and compilation mode (TXG-CFG-017, TXG-CFG-018) and nothing else:
    every qualifying engine runs at core's default KV-cache sizing, the one a
    serve command without KV-cache options selects, unless a spec names a
    fixed block count or budget."""
    settings = q.restricted_startup()
    assert settings["async_scheduling"] is False
    assert settings["compilation_config"] == {"cudagraph_mode": "NONE", "mode": 0}
    for name in (*KV_SETTINGS, "enable_extensible_kv_cache"):
        assert name not in settings, name
    assert q.kv_cache_sizing() == {}, "core's default sizing sets an option"


@pytest.mark.parametrize(
    "sizing, kwargs, expected",
    [
        (Sizing.EXTENSIBLE, {}, {}),
        (Sizing.PROFILED, {}, {"enable_extensible_kv_cache": False}),
        (Sizing.BLOCK_COUNT, {"blocks": 65}, {"num_gpu_blocks_override": 65}),
        (Sizing.BUDGET, {"budget": 8 << 30}, {"kv_cache_memory_bytes": 8 << 30}),
    ],
)
# @spec TXG-QUAL-001
def test_each_kv_cache_sizing_has_its_own_settings(sizing, kwargs, expected):
    assert q.kv_cache_sizing(sizing, **kwargs) == expected
    with pytest.raises(ValueError):
        q.kv_cache_sizing(sizing, blocks=65 if "blocks" not in kwargs else None)


# @spec TXG-QUAL-010, TXG-QUAL-020, TXG-QUAL-021
def test_the_checks_that_need_the_estimate_run_a_fixed_block_count():
    """The step rule, the first A40 run and the estimate check keep a fixed
    ``--num-gpu-blocks-override`` of the admission limit plus one with core's
    estimate enabled."""
    settings = q.estimate_startup(64)
    assert settings["num_gpu_blocks_override"] == 65
    assert "kv_cache_memory_bytes" not in settings
    # the admission limit, not max_num_seqs, fixes the block count
    limited = q.estimate_startup(64, max_num_active_seqs=32)
    assert limited["num_gpu_blocks_override"] == 33


def _spec(**overrides):
    key = "transaction_graph_populations"
    return {
        "card": "/card",
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "hf_overrides": {"num_lookahead_tokens": 0, key: [64]},
    } | overrides


# @spec TXG-QUAL-001, TXG-QUAL-010
def test_the_driver_runs_b_hook_with_the_same_settings_and_the_estimate():
    """``B_hook`` differs from the option-on run only by the absent key: the
    KV-cache sizing included (TXG-QUAL-001); every run enables core's
    graph-memory estimate (TXG-QUAL-010). The workload runs at core's default
    sizing; the step rule under a fixed block count."""
    from support import txg_qual_driver as d

    for mode, sizing in (("qm", Sizing.EXTENSIBLE), ("steprule", Sizing.BLOCK_COUNT)):
        on = _spec(mode=mode)
        off = on | {"hf_overrides": {"num_lookahead_tokens": 0}}
        a, b = d.engine_settings(on), d.engine_settings(off)
        assert {k: v for k, v in a.items() if k != "hf_overrides"} == {
            k: v for k, v in b.items() if k != "hf_overrides"
        }, mode
        assert d.run_sizing(on) == d.run_sizing(off) == sizing, mode
        assert a["async_scheduling"] is False
        assert d.engine_env(on)[q.ESTIMATE_ENV] == "1"
        if sizing == Sizing.EXTENSIBLE:
            assert not set(KV_SETTINGS) & set(a), a
        else:
            assert a["num_gpu_blocks_override"] == 65
    named = d.engine_settings(_spec(mode="qm", num_gpu_blocks_override=80))
    assert named["num_gpu_blocks_override"] == 80
    assert d.run_sizing(_spec(kv_cache_sizing=Sizing.PROFILED)) == Sizing.PROFILED


# @spec TXG-QUAL-001
def test_the_record_of_every_run_holds_its_sizing_page_counts_and_final_count():
    """Option on and ``B_hook`` alike: the KV-cache sizing, each capture
    layout's page count and, under the extensible KV cache, the final bound
    count read after core's extension."""
    extensible = {
        "kv_cache_sizing": Sizing.EXTENSIBLE,
        "capture_page_counts": {"serving": 4096},
        "final_bound_count": 1800,
    }
    assert q.missing_run_record_fields(extensible) == []
    assert q.missing_run_record_fields(extensible | {"final_bound_count": None}) == [
        "final_bound_count"
    ]
    fixed = {
        "kv_cache_sizing": Sizing.BLOCK_COUNT,
        "capture_page_counts": {"profiling": 65, "serving": 65},
        "final_bound_count": None,
    }
    assert q.missing_run_record_fields(fixed) == []
    assert "kv_cache_sizing" in q.missing_run_record_fields(
        {k: v for k, v in fixed.items() if k != "kv_cache_sizing"}
    )
    assert "capture_page_counts.serving" in q.missing_run_record_fields(
        fixed | {"capture_page_counts": {"profiling": 65}}
    )


# ---------------------------------------------------------------------------
# The exactness runs' headroom (TXG-QUAL-029)


#: The published checkpoint's state page (the manifest's ``total_page_bytes``).
PAGE_BYTES = 6_314_944


def _headroom_reply(**changes):
    limit, total = 64, 48 << 30
    headroom = q.preregistered_headroom(limit, PAGE_BYTES)
    copy, margin = headroom["copy_bytes"], headroom["margin_bytes"]
    reply = {
        "spec": {"mode": "qm", "max_num_seqs": limit, "headroom": headroom},
        "identity": {
            "kv_cache_sizing": Sizing.EXTENSIBLE,
            "state_page_bytes": PAGE_BYTES,
            "settings": {
                "gpu_memory_utilization": q.headroom_utilization(copy, margin, total)
            },
        },
        "exactness_memory": {
            "torch_memory_limit": 20 << 30,
            "memory_reserved_before": 4 << 30,
            "max_memory_reserved_before": 5 << 30,
            "memory_reserved_after": 4 << 30,
            "max_memory_reserved_after": 5 << 30,
            "copy_bytes": copy,
            "copy_allocation_bytes": copy + 512,
            "context_before": {"mem_get_info": [1, 2], "nvidia_smi": "1, 2"},
        },
    }
    for path, value in changes.items():
        node = reply
        *parents, leaf = path.split(".")
        for name in parents:
            node = node[name]
        if value is ...:
            del node[leaf]
        else:
            node[leaf] = value
    return reply


# @spec TXG-QUAL-029
def test_the_headroom_is_left_through_gpu_memory_utilization_below_one():
    utilization = q.headroom_utilization(10 << 30, 1 << 30, 48 << 30)
    assert utilization == (48 - 11) / 48
    for copy, margin in ((48 << 30, 0), (0, 0)):
        with pytest.raises(ValueError):
            q.headroom_utilization(copy, margin, 48 << 30)


# @spec TXG-QUAL-029
def test_the_driver_refuses_a_headroom_outside_the_extensible_kv_cache():
    from support import txg_qual_driver as d

    headroom = q.preregistered_headroom(64, PAGE_BYTES)
    spec = _spec(mode="qm", headroom=headroom, device_total_bytes=48 << 30)
    settings = d.engine_settings(spec)
    assert 0.0 < settings["gpu_memory_utilization"] < 1.0
    assert not set(KV_SETTINGS) & set(settings), settings
    for named in (
        {"kv_cache_memory_bytes": 8 << 30},
        {"num_gpu_blocks_override": 65},
        {"kv_cache_sizing": Sizing.PROFILED},
    ):
        with pytest.raises(ValueError):
            d.engine_settings(spec | named)


# @spec TXG-QUAL-029
def test_the_driver_refuses_a_comparing_run_without_a_preregistered_headroom():
    """A ``qm`` or ``fallback`` run that compares with a detached copy under
    the extensible KV cache is refused before the engine starts unless its
    spec holds the preregistered headroom, and the refusal names the
    remedy; the same run with its headroom, a run that does not compare, a
    mode with no detached copy and a comparing run under a fixed block count
    start."""
    from support import txg_qual_driver as d

    total = {"device_total_bytes": 48 << 30}
    for mode, extra in (("qm", {}), ("fallback", {"n": 4})):
        spec = _spec(mode=mode, compare=True, **extra, **total)
        assert d.run_sizing(spec) == Sizing.EXTENSIBLE
        with pytest.raises(ValueError, match="preregistered_headroom") as refused:
            d.engine_settings(spec)
        assert "TXG-QUAL-029" in str(refused.value), refused.value
        headroom = q.preregistered_headroom(q.spec_max_requests(spec), PAGE_BYTES)
        settings = d.engine_settings(spec | {"headroom": headroom})
        assert 0.0 < settings["gpu_memory_utilization"] < 1.0, (mode, settings)
        assert "gpu_memory_utilization" not in d.engine_settings(
            spec | {"compare": False}
        )
        assert d.engine_settings(spec | {"num_gpu_blocks_override": 80})
    assert d.engine_settings(_spec(mode="steprule", compare=True))


# @spec TXG-QUAL-007
def test_the_driver_refuses_fallback_steps_above_the_sink_row_bound():
    """The fallback steps' ``K = N + 1`` step holds ``N + 1`` rows, and the
    commit sink holds at most ``max_num_seqs``: a ``fallback`` spec with
    ``max_num_seqs`` below ``N + 1`` is refused before the engine starts,
    naming the remedy; at ``N + 1`` it starts."""
    from support import txg_qual_driver as d

    spec = _spec(mode="fallback", n=4, max_num_seqs=4)
    with pytest.raises(ValueError, match=r"at least N \+ 1") as refused:
        d.engine_settings(spec)
    assert "TXG-QUAL-007" in str(refused.value), refused.value
    assert d.engine_settings(spec | {"max_num_seqs": 5})


# @spec TXG-QUAL-029
def test_the_margin_counts_what_the_caching_allocator_reserves():
    """Each margin term is the bytes the CUDA caching allocator reserves for
    its allocations: 2 MiB for at most 1 MiB, a 20 MiB segment below 10 MiB,
    and the size rounded up to 2 MiB above; the digest's and the
    comparison's one-chunk workspaces, and the copy's slack per allocation
    (its 512-byte shift plus that rounding), for the copy allocations the
    margin states."""
    mib = 1 << 20
    assert q.allocator_reserved_bound(512) == 2 * mib
    assert q.allocator_reserved_bound(mib) == 2 * mib
    assert q.allocator_reserved_bound(mib + 1) == 20 * mib
    assert q.allocator_reserved_bound(16 * mib) == 20 * mib
    assert q.allocator_reserved_bound(21 * mib + 1) == 22 * mib
    assert q.digest_workspace_bytes() == 16 * mib
    assert q.DIGEST_WORKSPACE_BYTES == q.allocator_reserved_bound(16 * mib)
    assert q.COMPARISON_WORKSPACE_BYTES == (
        q.allocator_reserved_bound(q.comparison_workspace_bytes()) + 2 * mib
    )
    assert q.COPY_ALLOCATIONS == 1
    assert q.COPY_ALIGNMENT_ALLOWANCE_BYTES == q.COPY_ALLOCATIONS * (512 + 20 * mib)
    for copy in (64 * PAGE_BYTES, 4 * mib, 12 * mib + 3):
        slack = q.allocator_reserved_bound(copy + 512) - copy
        assert 0 < slack <= q.COPY_ALIGNMENT_ALLOWANCE_BYTES, (copy, slack)


# @spec TXG-QUAL-029
def test_a_copy_with_more_allocations_than_the_margin_states_fails():
    """Under the allocator reading, a detached copy that makes more
    allocations (distinct storages) than ``COPY_ALLOCATIONS`` fails the run
    and names the remedy; one allocation is recorded and passes."""
    import torch
    from support import txg_transaction as tx
    from support.txg_qual_adapter import Call, TransactionMonitor

    def pools(*views):
        return tx.Pools(
            {"channel_pools": list(views), "time_pools": [], "len_pools": []}
        )

    one = torch.zeros(8, 64, dtype=torch.int8)
    monitor = TransactionMonitor(torch.nn.Linear(1, 1), compare=True, memory=True)
    monitor._copy(pools(one[:, :32], one[:, 32:]).kwargs(), Call(kind="execute"))
    assert monitor.copy_readings[-1]["copy_allocations"] == 1
    two = pools(
        torch.zeros(8, 32, dtype=torch.int8), torch.zeros(8, 32, dtype=torch.int8)
    )
    with pytest.raises(RuntimeError, match="COPY_ALLOCATIONS"):
        monitor._copy(two.kwargs(), Call(kind="execute"))


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({}, None),
        ({"identity.kv_cache_sizing": Sizing.BLOCK_COUNT}, "extensible"),
        ({"identity.settings.kv_cache_memory_bytes": 8 << 30}, "kv-cache-memory"),
        ({"identity.settings.gpu_memory_utilization": 1.0}, "below 1.0"),
        ({"exactness_memory.torch_memory_limit": ...}, "torch_memory_limit"),
        ({"exactness_memory.max_memory_reserved_after": ...}, "max_memory_reserved"),
        ({"exactness_memory.copy_bytes": 65 * PAGE_BYTES + 1}, "more than the"),
        ({"spec.headroom.margin_bytes": ...}, "margin_bytes"),
        (
            {"spec.headroom.copy_bytes": 22 << 30},
            "copy extent times the bytes per page",
        ),
        ({"spec.headroom.page_bytes": PAGE_BYTES + 64}, "copy extent times the bytes"),
        ({"identity.state_page_bytes": PAGE_BYTES + 64}, "not the bound views'"),
        ({"spec.max_num_active_seqs": 32}, "is not the mode's 32"),
        ({"spec.headroom.margin_bytes": 1 << 30}, "below the stated"),
        ({"spec.headroom.max_requests": ...}, "max_requests"),
        ({"exactness_memory.memory_reserved_before": None}, "memory_reserved_before"),
        ({"exactness_memory.memory_reserved_after": 1.5e9}, "memory_reserved_after"),
        ({"exactness_memory.torch_memory_limit": "20 GiB"}, "torch_memory_limit"),
        ({"exactness_memory.max_memory_reserved_before": True}, "max_memory_reserved"),
        ({"exactness_memory.copy_bytes": None}, "copy_bytes"),
    ],
)
# @spec TXG-QUAL-029
def test_an_exactness_run_records_the_allocator_around_the_copy(changes, problem):
    problems = q.headroom_problems(_headroom_reply(**changes))
    if problem is None:
        assert problems == [], problems
    else:
        assert any(problem in p for p in problems), (problem, problems)


# @spec TXG-QUAL-029
def test_a_record_whose_allocator_figures_are_all_none_fails():
    """A record that holds every allocator figure and the copy's bytes as
    ``None`` (an exactness run on a device the allocator did not read)
    records nothing TXG-QUAL-029 requires."""
    figures = {f"exactness_memory.{name}": None for name in q.HEADROOM_FIGURES}
    problems = q.headroom_problems(
        _headroom_reply(**figures, **{"exactness_memory.copy_bytes": None})
    )
    for name in (*q.HEADROOM_FIGURES, "copy_bytes"):
        assert any(name in p for p in problems), (name, problems)


# @spec TXG-QUAL-029
def test_the_device_level_figures_are_read_by_no_check():
    """``nvidia-smi`` and ``torch.cuda.mem_get_info`` are context only: the
    checks give the same verdict without them or with absurd values."""
    reply = _headroom_reply()
    assert q.headroom_problems(reply) == []
    reply["exactness_memory"]["context_before"] = {"mem_get_info": [0, 0]}
    assert q.headroom_problems(reply) == []
    del reply["exactness_memory"]["context_before"]
    assert q.headroom_problems(reply) == []


# @spec TXG-QUAL-029, TXG-QUAL-003
def test_the_headroom_fails_when_the_copy_exceeds_its_preregistered_figure():
    """The copy figure is the admission limit plus one pages: a recorded copy
    of one byte more (a page beyond the limit's, or the whole committed
    cache) fails the run; a copy of exactly the figure, or less, passes."""
    figure = 65 * PAGE_BYTES
    for copied, ok in (
        (figure, True),
        (figure - PAGE_BYTES, True),
        (figure + 1, False),
        (66 * PAGE_BYTES, False),
        (22 << 30, False),
    ):
        problems = q.headroom_problems(
            _headroom_reply(**{"exactness_memory.copy_bytes": copied})
        )
        assert (problems == []) is ok, (copied, problems)


# @spec TXG-QUAL-003, TXG-QUAL-007, TXG-QUAL-029
def test_both_modes_take_their_copy_extent_from_one_function():
    """``qm`` holds at most the admission limit's requests, ``fallback``
    ``N + 4``; both copy extents and both preregistered copy figures are
    ``copy_extent`` of that count, so they cannot diverge. A mode with no
    detached copy takes no headroom."""
    from support import txg_qual_driver as d

    assert q.mode_max_requests("qm", admission_limit=8, n=4) == 8
    assert q.mode_max_requests("fallback", admission_limit=8, n=4) == 8
    assert q.mode_max_requests("fallback", admission_limit=64, n=64) == 68
    assert q.mode_max_requests("steprule", admission_limit=8, n=4) is None
    assert q.copy_extent(68) == 69
    for bad in (0, -1, None, True):
        with pytest.raises(ValueError):
            q.copy_extent(bad)
    # The fallback run admits the N + 4 requests it holds (TXG-QUAL-007).
    spec = _spec(mode="fallback", n=64, max_num_seqs=68, device_total_bytes=48 << 30)
    assert q.spec_max_requests(spec) == 68
    headroom = q.preregistered_headroom(68, PAGE_BYTES)
    assert headroom["copy_bytes"] == q.copy_extent(68) * PAGE_BYTES == 69 * PAGE_BYTES
    assert (
        0.0
        < d.engine_settings(spec | {"headroom": headroom})["gpu_memory_utilization"]
        < 1.0
    )
    with pytest.raises(ValueError, match="preregistered"):
        d.engine_settings(spec | {"headroom": q.preregistered_headroom(64, PAGE_BYTES)})
    reply = _headroom_reply(
        **{
            "spec.mode": "fallback",
            "spec.n": 64,
            "spec.headroom": headroom,
            "exactness_memory.copy_bytes": headroom["copy_bytes"],
        }
    )
    assert q.headroom_problems(reply) == []
    reply["exactness_memory"]["copy_bytes"] += 1
    assert any("more than the" in p for p in q.headroom_problems(reply))
    steprule = _spec(mode="steprule", headroom=q.preregistered_headroom(64, PAGE_BYTES))
    assert q.preregistration_problems(steprule["headroom"], None)


# @spec TXG-QUAL-029, TXG-QUAL-018
def test_the_headroom_is_computed_before_the_run_from_limit_page_and_margin():
    """The preregistered figure is a function of the admission limit, the
    bound views' bytes per page and the stated margin alone, so it exists
    before any run sample: ``preregistered_headroom`` takes no reading, the
    driver derives ``--gpu-memory-utilization`` from the spec alone before
    the engine starts, and refuses a headroom that is not that figure (a
    whole-cache copy figure, another admission limit, a margin below the
    stated one)."""
    import inspect

    from support import txg_qual_driver as d

    assert list(inspect.signature(q.preregistered_headroom).parameters) == [
        "max_requests",
        "page_bytes",
        "margin_bytes",
    ]
    headroom = q.preregistered_headroom(64, PAGE_BYTES)
    assert headroom == {
        "max_requests": 64,
        "page_bytes": PAGE_BYTES,
        "copy_bytes": 65 * PAGE_BYTES,
        "margin_bytes": q.EXACTNESS_MARGIN_BYTES,
    }
    assert q.EXACTNESS_MARGIN_BYTES == (
        q.CORE_NON_TORCH_RESERVE_BYTES
        + q.STARTUP_ALLOWANCE_BYTES
        + q.DIGEST_WORKSPACE_BYTES
        + q.COMPARISON_WORKSPACE_BYTES
        + q.COPY_ALIGNMENT_ALLOWANCE_BYTES
    )
    assert q.CORE_NON_TORCH_RESERVE_BYTES == 1 << 29
    limited = q.preregistered_headroom(8, PAGE_BYTES, 2 << 30)
    assert limited["copy_bytes"] == 9 * PAGE_BYTES
    for bad in ((0, PAGE_BYTES), (64, 0), (64, PAGE_BYTES, 0), (True, PAGE_BYTES)):
        with pytest.raises(ValueError):
            q.preregistered_headroom(*bad)
    spec = _spec(mode="qm", headroom=headroom, device_total_bytes=48 << 30)
    expected = q.headroom_utilization(
        headroom["copy_bytes"], headroom["margin_bytes"], 48 << 30
    )
    assert d.engine_settings(spec)["gpu_memory_utilization"] == expected
    for changed in (
        headroom | {"copy_bytes": 22 << 30},
        headroom | {"max_requests": 32},
        headroom | {"margin_bytes": 1 << 30},
        {k: v for k, v in headroom.items() if k != "page_bytes"},
    ):
        with pytest.raises(ValueError, match="preregistered"):
            d.engine_settings(spec | {"headroom": changed})
    active = spec | {
        "max_num_active_seqs": 32,
        "headroom": q.preregistered_headroom(32, PAGE_BYTES),
    }
    assert 0.0 < d.engine_settings(active)["gpu_memory_utilization"] < 1.0


# @spec TXG-QUAL-029
def test_no_served_figure_comes_from_a_run_with_the_headroom():
    served = {"spec": {"mode": "qm"}}
    exactness = _headroom_reply()
    assert q.served_figure_runs([served, exactness]) == [served]


# @spec TXG-QUAL-019, TXG-QUAL-001
def test_the_capture_figures_check_accepts_a_capture_that_did_not_run():
    """At core's default sizing no profiling capture runs and the record
    marks it not applicable; the serving capture's figures are still
    required."""
    figures = {
        "call_total": 1,
        "transaction_total": 1,
        "kind_totals": {"chunk": 1, "echo": 1},
        "cells": [{"growth": 1}],
        "transient_peak": 1,
    }
    record = {"profiling": NOT_APPLICABLE, "serving": figures}
    assert q.missing_capture_figures(record) == []
    assert q.missing_capture_figures({"profiling": NOT_APPLICABLE}) != []


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


# ---------------------------------------------------------------------------
# Requests served alone (TXG-QUAL-028)


def _alone(option: bool, **changes):
    """An ``alone`` result of three items; ``changes`` maps ``"<item>.<key>"``
    to a value (``...`` deletes the key)."""
    names = ["alone-stream", "alone-file0", "alone-file1"]
    items = [
        {
            "name": name,
            "kind": "stream" if i == 0 else "file",
            "finish_reason": "stop",
            "token_ids": [i + 1, i + 2, 7],
            "failed_rows": 0,
            "error": None,
            "replays": 3 if option else 0,
            "fallbacks": {},
        }
        for i, name in enumerate(names)
    ]
    for path, value in changes.items():
        index, key = path.split(".")
        if value is ...:
            del items[int(index)][key]
        else:
            items[int(index)][key] = value
    return {"items": items}


@pytest.mark.parametrize(
    "on, off, problem",
    [
        ({}, {}, None),
        ({"1.finish_reason": "error", "1.failed_rows": 1}, {}, "failed rows"),
        ({}, {"2.finish_reason": "length"}, "not 'stop'"),
        ({"0.error": "EngineDeadError: dead"}, {}, "EngineDeadError"),
        ({"0.failed_rows": ...}, {}, "failed rows"),
        ({"2.replays": 0}, {}, "replayed no cell"),
        ({"1.fallbacks": {"signature": 1}}, {}, "'signature'"),
        ({}, {"1.replays": 2}, "B_hook alone-file0 replayed"),
        ({"2.token_ids": [3, 4, 8]}, {}, "differ from B_hook"),
        ({"2.name": "alone-file9"}, {}, "different items"),
    ],
)
# @spec TXG-QUAL-028
def test_each_item_served_alone_completes_replays_and_equals_b_hook(on, off, problem):
    """Every check is per item: each item of both runs completes with no
    failed row and no client error, each option-on item replays a cell with
    no ``signature`` fallback, no ``B_hook`` item replays, and each item's
    tokens equal ``B_hook``'s."""
    problems = q.alone_problems(_alone(True, **on), _alone(False, **off))
    if problem is None:
        assert problems == [], problems
    else:
        assert any(problem in p for p in problems), (problem, problems)


# @spec TXG-QUAL-028
def test_replays_of_other_items_do_not_cover_an_item_that_replayed_none():
    """An item with no replay fails even when the run's total is positive,
    and a ``signature`` fallback in one item fails though another item's
    fallbacks of other reasons are allowed."""
    on = _alone(
        True,
        **{"0.replays": 40, "1.replays": 0, "2.fallbacks": {"chunk-uncovered": 1}},
    )
    problems = q.alone_problems(on, _alone(False))
    assert problems == ["option-on alone-file0 replayed no cell (0)"], problems
    assert q.alone_problems({"items": []}, {"items": []}) == [
        "the option-on run served no item"
    ]
