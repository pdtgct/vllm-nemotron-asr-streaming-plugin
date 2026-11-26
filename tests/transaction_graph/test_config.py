# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph option and its configuration checks (TXG-CFG-001 to
TXG-CFG-020).

The target core's ``ModelState`` defines ``state_group_layouts``, so the
checks return TXG-CFG-016's state-layer condition only for a core without
it; ``api.config_errors`` passes that capability (``state_group_layouts``,
default the target core's). The configuration doubles
(``support.txg_fakes.engine_config``) meet the restricted startup by default
(TXG-CFG-017 and TXG-CFG-018): synchronous scheduling, ``cudagraph_mode``
and compilation mode NONE, under core's default KV-cache sizing, the
extensible KV cache (neither ``num_gpu_blocks_override`` nor
``kv_cache_memory_bytes``). A test that violates one condition therefore
sees exactly that condition; a test sets a fixed block count or budget only
where its spec names one.

CPU tests on configuration doubles (``support.txg_fakes``), except the
callers of TXG-CFG-011 and TXG-CFG-016, which use the stage 1 build
(``engine``). The checks' interface is documented in ``support.txg_api``;
the remedy patterns are ``support.txg_conditions``'.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from support import txg_api as api
from support import txg_conditions as cond
from support.txg_fakes import CompilationMode, GraphMode, engine_config, hf_config

KEY = api.KEY
VALID = [4]
shows_value = cond.shows_value


def errors_for(populations=VALID, *, state_group_layouts=True, **kwargs):
    """The conditions for a configuration with transaction graphs on."""
    hf_kwargs = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k
        in (
            "lookahead",
            "encoder_arm",
            "decoder_arm",
            "native_burst",
            "chunk_populations",
        )
    }
    return api.config_errors(
        engine_config(hf_config(populations, **hf_kwargs), **kwargs),
        state_group_layouts=state_group_layouts,
    )


def mentioning(errors, *needles):
    return [e for e in errors if all(n.lower() in e.lower() for n in needles)]


def only(errors):
    """The one condition of a configuration that violates exactly one."""
    assert len(errors) == 1, errors
    (error,) = errors
    assert KEY in error, error
    return error


def patch_served_lookahead(monkeypatch, fn):
    """Replace ``lookahead.served_lookahead`` (REG-LA-001) wherever the
    configuration module can reach it."""
    from vllm_nemotron_asr_streaming import lookahead

    monkeypatch.setattr(lookahead, "served_lookahead", fn, raising=False)
    module = api.config_module()
    if hasattr(module, "served_lookahead"):
        monkeypatch.setattr(module, "served_lookahead", fn)


# ---------------------------------------------------------------------------
# The on/off predicate

OFF_VALUES = [pytest.param("absent", id="absent"), pytest.param([], id="empty-list")]
ON_VALUES = [
    pytest.param([63], id="list"),
    pytest.param(63, id="int"),
    pytest.param("63", id="string"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param({}, id="empty-dict"),
    pytest.param(None, id="null"),
    pytest.param("", id="empty-string"),
    pytest.param(0, id="zero"),
    pytest.param((), id="empty-tuple"),
    pytest.param([0], id="list-of-zero"),
]


def served(value):
    return hf_config() if value == "absent" else hf_config(value)


# @spec TXG-CFG-001, TXG-CFG-014
@pytest.mark.parametrize("value", OFF_VALUES)
def test_absent_key_or_empty_list_is_off(value):
    assert api.transaction_graphs_on(served(value)) is False


# @spec TXG-CFG-001, TXG-CFG-014
@pytest.mark.parametrize("value", ON_VALUES)
def test_every_other_value_is_on(value):
    assert api.transaction_graphs_on(served(value)) is True


# @spec TXG-CFG-001
def test_option_is_read_from_a_translated_configuration():
    """A configuration that stored the key (an already-translated
    configuration, not a flat override) is read the same way."""
    transformers = pytest.importorskip("transformers")
    PretrainedConfig = transformers.PretrainedConfig

    stored = PretrainedConfig.from_dict({KEY: [8], "num_lookahead_tokens": 0})
    assert api.transaction_graphs_on(stored) is True
    assert api.transaction_graphs_on(PretrainedConfig.from_dict({})) is False
    assert api.largest_population(stored) == 8


# @spec TXG-CFG-001
def test_option_is_read_only_from_its_key():
    """The value is read from ``transaction_graph_populations`` only: neither
    a nested override mapping nor a similarly named key turns it on."""
    nested = hf_config(hf_overrides={KEY: [4]}, transaction_graph_population=[4])
    assert api.transaction_graphs_on(nested) is False
    assert api.config_errors(engine_config(nested, enforce_eager=True)) == []


# @spec TXG-CFG-001, TXG-CFG-004
@pytest.mark.parametrize("value", ON_VALUES[1:])
def test_malformed_values_are_on_and_refused(value):
    errors = api.config_errors(engine_config(hf_config(value)))
    # required information: the option's name and the offending value, in
    # whatever quoting the diagnostic uses
    assert [e for e in mentioning(errors, KEY) if shows_value(e, value)], errors
    with pytest.raises(ValueError):
        api.largest_population(hf_config(value))


# @spec TXG-CFG-003
@pytest.mark.parametrize("value", OFF_VALUES)
@pytest.mark.parametrize("state_group_layouts", [True, False], ids=["accessor", "none"])
def test_no_condition_while_off(value, state_group_layouts):
    """With the option off, nothing this function checks is reported, even
    for settings that the option on would refuse, on a core with or without
    the state-group layout accessor."""
    hf = served(value)
    hf.num_lookahead_tokens = 3
    hf.experimental_native_burst = True
    hf.encoder_execution_arm = "eager"
    config = engine_config(
        hf,
        enforce_eager=True,
        async_scheduling=None,
        cudagraph_mode=None,
        compilation_mode=None,
        max_num_active_seqs=1,
    )
    assert api.config_errors(config, state_group_layouts=state_group_layouts) == []


# ---------------------------------------------------------------------------
# Populations


# @spec TXG-CFG-002
def test_only_the_largest_entry_fixes_the_inventory():
    single, several = hf_config([63]), hf_config([1, 31, 63])
    assert api.largest_population(single) == api.largest_population(several) == 63
    for g in (0, 1):
        assert api.cell_keys(api.largest_population(single), g) == api.cell_keys(
            api.largest_population(several), g
        )
    assert errors_for([63]) == errors_for([1, 31, 63]) == []


# @spec TXG-CFG-002
def test_entries_other_than_the_largest_are_validated():
    errors = errors_for([0, 31, 63])
    assert mentioning(errors, KEY, "0"), errors


# @spec TXG-CFG-004
@pytest.mark.parametrize(
    "value",
    [
        pytest.param([True], id="bool-entry"),
        pytest.param([1, False], id="false-entry"),
        pytest.param([4.0], id="float-entry"),
        pytest.param(["4"], id="string-entry"),
        pytest.param([None], id="null-entry"),
        pytest.param([4, 4], id="duplicate"),
        pytest.param([0], id="zero"),
        pytest.param([-1], id="negative"),
        pytest.param((4,), id="tuple"),
        pytest.param({4}, id="set"),
    ],
)
def test_populations_condition(value):
    error = only(errors_for(value, max_num_seqs=64, max_num_batched_tokens=8192))
    assert repr(value) in error or str(value) in error, error
    with pytest.raises(ValueError):
        api.largest_population(hf_config(value))


# @spec TXG-CFG-004, TXG-CFG-020
def test_an_entry_above_max_num_seqs_is_refused_by_both_bounds():
    """``65`` exceeds ``P_max`` (TXG-CFG-004) and the admission limit
    ``max_num_seqs`` (TXG-CFG-020): both conditions are returned. The
    batching limit is engine configuration that ``largest_population`` does
    not see, so only the configuration check refuses this value."""
    errors = errors_for([65], max_num_seqs=64, max_num_batched_tokens=8192)
    assert len(errors) == 2, errors
    admission = [e for e in errors if cond.is_admission_condition(e)]
    assert len(admission) == 1, errors
    others = [e for e in errors if e not in admission and "65" in e]
    assert len(others) == 1, errors
    assert api.largest_population(hf_config([65])) == 65


# @spec TXG-CFG-004
def test_populations_are_bounded_by_the_smaller_batching_limit():
    """``P_max = min(max_num_seqs, max_num_batched_tokens)``."""
    assert errors_for([32], max_num_seqs=64, max_num_batched_tokens=32) == []
    assert mentioning(
        errors_for([33], max_num_seqs=64, max_num_batched_tokens=32), KEY, "33"
    )
    assert errors_for([64], max_num_seqs=64, max_num_batched_tokens=8192) == []
    assert mentioning(errors_for([64], max_num_seqs=63), KEY, "64")


# @spec TXG-CFG-004
def test_every_accepted_value_yields_its_largest_entry():
    assert api.largest_population(hf_config([1])) == 1
    assert api.largest_population(hf_config([31, 1, 63])) == 63
    with pytest.raises(ValueError):
        api.largest_population(hf_config([]))
    with pytest.raises(ValueError):
        api.largest_population(hf_config())


# @spec TXG-CFG-015
@pytest.mark.parametrize(
    "limits, value, refused",
    [
        ((None, None), [4096], False),
        ((None, 32), [32], False),
        ((None, 32), [33], True),
        ((32, None), [32], False),
        ((32, None), [33], True),
        ((None, None), [0], True),
        ((None, None), [4, 4], True),
    ],
)
def test_unresolved_limits_bound_nothing(limits, value, refused):
    """At the configuration hook an unresolved limit bounds nothing; the
    other conditions still apply. Under core's default KV-cache sizing no
    block count is fixed, so the block-count condition (TXG-CFG-005) takes
    no part."""
    seqs, tokens = limits
    errors = errors_for(value, max_num_seqs=seqs, max_num_batched_tokens=tokens)
    assert bool(mentioning(errors, KEY)) is refused, errors


# ---------------------------------------------------------------------------
# Admission limit (TXG-CFG-020) and block count (TXG-CFG-005)


# @spec TXG-CFG-020, TXG-CFG-003
def test_an_entry_above_max_num_active_seqs_is_refused_with_its_remedies():
    """The admission limit is ``max_num_active_seqs`` when set: an entry above
    it can never run. The condition names the entry, the limit, the setting
    it came from and both remedies."""
    error = only(
        errors_for(
            [4, 16], max_num_seqs=64, max_num_active_seqs=8, num_gpu_blocks_override=17
        )
    )
    assert cond.is_admission_condition(error), error
    assert cond.shows_number(error, 16), error
    assert cond.shows_number(error, 8), error
    assert "max_num_active_seqs" in error, error
    # remedies: a largest population of at most the limit, or a larger
    # --max-num-active-seqs (at most max_num_seqs) or none
    assert cond.found(cond.ACTIVE_SEQS, error), error
    assert mentioning([error], "population"), error


# @spec TXG-CFG-020
def test_an_entry_at_the_admission_limit_is_accepted():
    assert errors_for([8], max_num_seqs=64, max_num_active_seqs=8) == []
    assert errors_for([1, 8], max_num_seqs=8) == []


# @spec TXG-CFG-020
def test_the_admission_limit_from_max_num_seqs_is_named():
    """Without ``max_num_active_seqs`` the limit is ``max_num_seqs``; an
    entry above it is also above ``P_max``, so both conditions are
    returned, the admission one naming ``max_num_seqs``."""
    errors = errors_for([16], max_num_seqs=8)
    admission = [e for e in errors if cond.is_admission_condition(e)]
    assert len(admission) == 1, errors
    assert "max_num_seqs" in admission[0], admission
    assert cond.shows_number(admission[0], 16), admission
    assert cond.shows_number(admission[0], 8), admission


# @spec TXG-CFG-020
def test_an_unknown_admission_limit_bounds_nothing():
    """Before ``max_num_seqs`` is resolved and with ``max_num_active_seqs``
    unset, the configuration hook does not know the limit."""
    errors = errors_for([512], max_num_seqs=None, max_num_batched_tokens=None)
    assert errors == [], errors


# @spec TXG-CFG-020
def test_a_set_max_num_active_seqs_bounds_before_max_num_seqs_is_resolved():
    errors = errors_for(
        [16],
        max_num_seqs=None,
        max_num_batched_tokens=None,
        max_num_active_seqs=8,
    )
    admission = [e for e in errors if cond.is_admission_condition(e)]
    assert len(admission) == 1, errors


@pytest.mark.parametrize(
    "limits, setting",
    [
        pytest.param({"max_num_seqs": 8}, "max_num_seqs", id="max-num-seqs"),
        pytest.param(
            {"max_num_seqs": 64, "max_num_active_seqs": 8},
            "max_num_active_seqs",
            id="max-num-active-seqs",
        ),
    ],
)
# @spec TXG-CFG-005, TXG-CFG-003
def test_a_block_count_below_the_admission_limit_plus_one_is_refused(limits, setting):
    """``override − 1`` non-null pages fewer than the admission limit: core's
    scheduler, built after both captures, would refuse to start. The
    condition names the override, the limit and its setting, and the
    remedies: an override of at least the limit plus one, or a lower
    admission limit with the largest population at most it."""
    error = only(errors_for([4], num_gpu_blocks_override=8, **limits))
    assert cond.is_block_count_condition(error), error
    assert cond.shows_number(error, 8), error
    assert setting in error, error
    assert cond.found(cond.BLOCKS_OVERRIDE, error), error
    assert cond.shows_number(error, 9), error  # the limit plus one
    assert cond.lower_admission_limit(error), error


# @spec TXG-CFG-005
def test_a_block_count_of_the_admission_limit_plus_one_is_accepted():
    assert errors_for([4], max_num_seqs=8, num_gpu_blocks_override=9) == []
    assert (
        errors_for(
            [4], max_num_seqs=64, max_num_active_seqs=8, num_gpu_blocks_override=9
        )
        == []
    )


# @spec TXG-CFG-005, TXG-CFG-015
def test_an_unknown_admission_limit_compares_the_override_with_the_entries():
    """Before the admission limit is known, the configuration hook compares
    the override with the entries only; model-state construction applies the
    full condition."""
    unresolved = dict(max_num_seqs=None, max_num_batched_tokens=None)
    assert errors_for([32], num_gpu_blocks_override=33, **unresolved) == []
    error = only(errors_for([33], num_gpu_blocks_override=33, **unresolved))
    assert cond.is_block_count_condition(error), error
    assert cond.shows_number(error, 33), error
    assert errors_for([4, 33], num_gpu_blocks_override=33, **unresolved), (
        "every entry counts"
    )


# @spec TXG-CFG-005
def test_a_set_max_num_active_seqs_is_a_known_admission_limit():
    errors = errors_for(
        [4],
        max_num_seqs=None,
        max_num_batched_tokens=None,
        max_num_active_seqs=40,
        num_gpu_blocks_override=33,
    )
    error = only(errors)
    assert cond.is_block_count_condition(error), error
    assert "max_num_active_seqs" in error, error


# ---------------------------------------------------------------------------
# Geometry, native burst, segment arms, enforce_eager


# @spec TXG-CFG-006
@pytest.mark.parametrize("lookahead", [0, 1])
def test_80_and_160_ms_are_accepted(lookahead):
    assert errors_for(lookahead=lookahead) == []


# @spec TXG-CFG-006, TXG-CFG-003
@pytest.mark.parametrize(
    "lookahead", [3, 6, 13, None], ids=["3", "6", "13", "card-default-3"]
)
def test_other_geometries_are_refused_with_the_remedy(lookahead):
    error = only(errors_for(lookahead=lookahead))
    assert mentioning([error], KEY, "num_lookahead_tokens", "--hf-overrides"), error
    shown = 3 if lookahead is None else lookahead
    assert str(shown) in error, error
    assert "0" in error and "1" in error, error


# @spec TXG-CFG-006
def test_a_raising_served_lookahead_becomes_the_geometry_condition(monkeypatch):
    message = "num_lookahead_tokens must be one of 0, 1, 3, 6, 13 (sentinel)"

    def raising(hf):
        raise ValueError(message)

    patch_served_lookahead(monkeypatch, raising)
    errors = errors_for()
    assert mentioning(errors, KEY, "sentinel"), errors


# @spec TXG-CFG-006
@pytest.mark.parametrize("lookahead", [2, True, "0"], ids=["2", "true", "string"])
def test_an_invalid_lookahead_does_not_raise(lookahead):
    """``served_lookahead`` (REG-LA-001) raises for these values; the check
    returns the geometry condition instead."""
    errors = errors_for(lookahead=lookahead)
    assert mentioning(errors, KEY, "num_lookahead_tokens"), errors


# @spec TXG-CFG-006
def test_geometry_comes_from_served_lookahead(monkeypatch):
    """The served lookahead decides, not the configuration key the check
    could read itself."""
    patch_served_lookahead(monkeypatch, lambda hf: 3)
    assert mentioning(errors_for(lookahead=0), KEY, "num_lookahead_tokens")
    patch_served_lookahead(monkeypatch, lambda hf: 1)
    assert errors_for(lookahead=6) == []


# @spec TXG-CFG-007, TXG-CFG-003
def test_native_burst_is_refused():
    error = only(errors_for(native_burst=True))
    assert "experimental_native_burst" in error, error
    assert shows_value(error, True), error
    assert cond.found(cond.REMOVAL, error), error
    assert errors_for(native_burst=False) == []


# @spec TXG-CFG-008, TXG-CFG-003
@pytest.mark.parametrize(
    "encoder, decoder, names",
    [
        (
            "unset",
            "dense-graphed",
            [("encoder_execution_arm", "eager")],
        ),
        (
            "eager-graphed",
            "unset",
            [("decode_dispatch_arm", "compact-eager")],
        ),
        ("eager", "dense-graphed", [("encoder_execution_arm", "eager")]),
        (
            "dense-graphed",
            "dense-graphed",
            [("encoder_execution_arm", "dense-graphed")],
        ),
        ("eager-graphed", "dense-eager", [("decode_dispatch_arm", "dense-eager")]),
        ("eager-graphed", "compact-eager", [("decode_dispatch_arm", "compact-eager")]),
        (
            "eager",
            "compact-eager",
            [
                ("encoder_execution_arm", "eager"),
                ("decode_dispatch_arm", "compact-eager"),
            ],
        ),
    ],
)
def test_segment_arms_other_than_the_defaults_are_refused(encoder, decoder, names):
    from support.txg_fakes import _ABSENT

    error = only(
        errors_for(
            encoder_arm=_ABSENT if encoder == "unset" else encoder,
            decoder_arm=_ABSENT if decoder == "unset" else decoder,
        )
    )
    for arm, value in names:
        assert arm in error and value in error, (arm, value, error)
    # the remedy: the default arms
    assert "eager-graphed" in error and "dense-graphed" in error, error


# @spec TXG-CFG-008
def test_default_arms_are_accepted_and_chunk_graphs_are_left_to_registration():
    assert errors_for() == []
    errors = errors_for(chunk_populations=[1, 2])
    assert not mentioning(errors, "chunk_bucket_graph_populations"), errors


# @spec TXG-CFG-009, TXG-CFG-003
def test_enforce_eager_is_refused_naming_both():
    error = only(errors_for(enforce_eager=True))
    assert "--enforce-eager" in error, error
    assert "hook" in error.lower(), error
    assert cond.found(cond.REMOVAL, error), error


# @spec TXG-CFG-010
def test_the_platform_condition_is_not_repeated():
    assert errors_for(device_type="cpu") == []


# @spec TXG-CFG-010, TXG-CFG-018
@pytest.mark.parametrize(
    "setting, shown",
    [
        ({"cudagraph_mode": GraphMode.FULL}, "FULL"),
        ({"cudagraph_mode": GraphMode.PIECEWISE}, "PIECEWISE"),
        ({"compilation_mode": CompilationMode.STOCK_TORCH_COMPILE}, "STOCK_TORCH"),
    ],
    ids=["full", "piecewise", "stock-torch-compile"],
)
def test_the_hook_conditions_are_not_repeated_beside_the_runner_mode_one(
    setting, shown
):
    """A mode the hook configuration refuses (REG-ENG-021 to REG-ENG-023)
    gives this function's one runner-mode condition only: the hook
    configuration's own condition is ``registration``'s and is not
    repeated."""
    error = only(errors_for(**setting))
    assert cond.found(cond.RUNNER_MODES_NONE, error), error
    assert shown in error, error


# ---------------------------------------------------------------------------
# The restricted startup (TXG-CFG-017 and TXG-CFG-018)


# @spec TXG-CFG-017, TXG-CFG-003
@pytest.mark.parametrize("value", [True, None], ids=["async", "unset"])
def test_scheduling_other_than_synchronous_is_refused(value):
    """``True``, or unset, which the target core resolves to asynchronous
    scheduling: the condition names the value and ``--no-async-scheduling``."""
    error = only(errors_for(async_scheduling=value))
    assert "async_scheduling" in error, error
    assert shows_value(error, value) or (value is None and "unset" in error), error
    assert cond.found(cond.NO_ASYNC_SCHEDULING, error), error
    assert errors_for(async_scheduling=False) == []


# @spec TXG-CFG-018, TXG-CFG-003
@pytest.mark.parametrize(
    "setting, shown",
    [
        ({"cudagraph_mode": GraphMode.FULL}, ["FULL"]),
        ({"cudagraph_mode": None}, [None]),
        ({"compilation_mode": CompilationMode.VLLM_COMPILE}, ["VLLM_COMPILE"]),
        ({"compilation_mode": None}, [None]),
        (
            {
                "cudagraph_mode": GraphMode.PIECEWISE,
                "compilation_mode": CompilationMode.VLLM_COMPILE,
            },
            ["PIECEWISE", "VLLM_COMPILE"],
        ),
    ],
    ids=["graph-full", "graph-unset", "compile", "compile-unset", "both"],
)
def test_runner_modes_other_than_none_are_refused(setting, shown):
    """Each mode that is not NONE (an unset mode included) is named, in one
    runner-mode condition, with the remedy ``-cc.cudagraph_mode none
    -cc.mode none``."""
    error = only(errors_for(**setting))
    for value in shown:
        if value is None:
            assert "unset" in error.lower() or shows_value(error, None), error
        else:
            assert value in error, (value, error)
    assert cond.found(cond.RUNNER_MODES_NONE, error), error


# @spec TXG-CFG-018
@pytest.mark.engine
def test_core_none_modes_are_accepted_and_others_refused():
    """Core's own enum members, as model construction sees them."""
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode as CoreCompilationMode

    assert (
        errors_for(
            cudagraph_mode=CUDAGraphMode.NONE,
            compilation_mode=CoreCompilationMode.NONE,
        )
        == []
    )
    error = only(
        errors_for(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            compilation_mode=CoreCompilationMode.NONE,
        )
    )
    assert "FULL_AND_PIECEWISE" in error, error


#: Every KV-cache sizing core offers, as the configuration checks see it:
#: core's default, the extensible KV cache; a profiled budget with it off; a
#: zero budget, which core reads as unset; a fixed block count of the
#: admission limit plus one (TXG-CFG-005); a fixed budget; and a startup
#: plan's budget, which core sets with the extensible KV cache left on.
SIZINGS = {
    "extensible": api.sizing_settings(api.Sizing.EXTENSIBLE),
    "profiled": api.sizing_settings(api.Sizing.PROFILED),
    "zero-budget": dict(kv_cache_memory_bytes=0, enable_extensible_kv_cache=False),
    "block-count": api.sizing_settings(api.Sizing.BLOCK_COUNT, blocks=65),
    "budget": api.sizing_settings(api.Sizing.BUDGET, budget=1 << 30),
    "startup-plan": dict(
        kv_cache_memory_bytes=1 << 30, enable_extensible_kv_cache=True
    ),
}


@pytest.mark.parametrize("sizing", list(SIZINGS))
# @spec TXG-CFG-003
def test_every_kv_cache_sizing_meets_the_restricted_startup(sizing):
    """The restricted startup is synchronous scheduling and an explicit NONE
    runner and compilation mode, under every KV-cache sizing core offers:
    each capture addresses only block 0, which core backs at every capture,
    so neither an explicit block count nor a KV budget is a condition."""
    assert errors_for(max_num_seqs=64, **SIZINGS[sizing]) == []


@pytest.mark.parametrize("sizing", list(SIZINGS))
@pytest.mark.parametrize(
    "unmet, check",
    [
        ({"async_scheduling": None}, cond.NO_ASYNC_SCHEDULING),
        ({"async_scheduling": True}, cond.NO_ASYNC_SCHEDULING),
        ({"cudagraph_mode": GraphMode.FULL}, cond.RUNNER_MODES_NONE),
        ({"compilation_mode": None}, cond.RUNNER_MODES_NONE),
    ],
    ids=["scheduling-unset", "async", "graph-full", "compile-unset"],
)
# @spec TXG-CFG-017, TXG-CFG-018, TXG-CFG-003
def test_the_restricted_startup_refuses_under_every_kv_cache_sizing(
    sizing, unmet, check
):
    """Under each sizing, an unmet scheduling or runner-mode setting is
    refused with its own remedy, and the refusal is the only condition: no
    KV-cache sizing is one."""
    error = only(errors_for(max_num_seqs=64, **SIZINGS[sizing], **unmet))
    assert cond.found(check, error), error
    assert not cond.is_sizing_condition(error), error


# ---------------------------------------------------------------------------
# The state layer (TXG-CFG-016)


# @spec TXG-CFG-016
def test_the_checks_take_the_state_layer_capability():
    """Callers determine whether the running core's ``ModelState`` defines
    ``state_group_layouts`` and pass it in (the module imports no vLLM)."""
    assert api.takes_state_layer_keyword(), (
        f"transaction_graph_config_errors takes no {api.STATE_LAYER_KEYWORD!r}"
    )


# @spec TXG-CFG-016, TXG-CFG-003
@pytest.mark.parametrize("value", [[4], [1, 31, 63], 63, None, {}], ids=repr)
def test_a_core_without_the_accessor_is_refused(value):
    errors = errors_for(value, state_group_layouts=False)
    found = [e for e in errors if cond.is_state_layer_condition(e)]
    assert len(found) == 1, errors
    assert found[0] in errors


# @spec TXG-CFG-016
@pytest.mark.parametrize("value", [[4], [1, 31, 63], 63, None, {}], ids=repr)
def test_a_core_with_the_accessor_gets_no_state_layer_condition(value):
    errors = errors_for(value)
    assert not [e for e in errors if cond.found(cond.ACCESSOR, e)], errors


# @spec TXG-CFG-016
def test_the_state_layer_condition_is_its_own_entry():
    errors = errors_for(state_group_layouts=False)
    error = only(errors)
    assert cond.is_state_layer_condition(error), error


# ---------------------------------------------------------------------------
# Every condition


# @spec TXG-CFG-003, TXG-CFG-011
def test_every_violated_condition_is_returned():
    """Each violated condition is its own entry, naming the option: here
    every condition but the block count's (no block count is fixed), nine in
    all, none of them a KV-cache sizing."""
    from support.txg_fakes import _ABSENT

    errors = errors_for(
        [0, 70],
        state_group_layouts=False,
        lookahead=3,
        native_burst=True,
        encoder_arm=_ABSENT,
        enforce_eager=True,
        max_num_seqs=64,
        async_scheduling=None,
        cudagraph_mode=None,
        compilation_mode=None,
    )
    assert len(errors) == 9, errors
    for error in errors:
        assert KEY in error, error
    for check in (cond.is_state_layer_condition, cond.is_admission_condition):
        assert len([e for e in errors if check(e)]) == 1, (check.__name__, errors)
    assert not [e for e in errors if cond.is_sizing_condition(e)], errors
    for needle in (
        "[0, 70]",
        "num_lookahead_tokens",
        "experimental_native_burst",
        "encoder_execution_arm",
        "--enforce-eager",
        "--no-async-scheduling",
    ):
        assert mentioning(errors, needle), (needle, errors)
    assert [e for e in errors if cond.found(cond.RUNNER_MODES_NONE, e)], errors


# @spec TXG-CFG-003, TXG-CFG-005
def test_the_block_count_condition_is_returned_with_the_others():
    errors = errors_for(
        [4], max_num_seqs=8, num_gpu_blocks_override=4, async_scheduling=True
    )
    assert len(errors) == 2, errors
    assert len([e for e in errors if cond.is_block_count_condition(e)]) == 1, errors


# ---------------------------------------------------------------------------
# The module


# @spec TXG-CFG-012
def test_configuration_module_imports_no_vllm(repo_root):
    code = (
        "import sys\n"
        f"import {api.CONFIG_MODULE} as m\n"
        "assert callable(m.transaction_graph_config_errors)\n"
        "assert callable(m.transaction_graphs_on)\n"
        "loaded = [k for k in sys.modules if k == 'vllm' or k.startswith('vllm.')]\n"
        "print(sorted(loaded))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout


# @spec TXG-CFG-013
@pytest.mark.parametrize(
    "populations, extra",
    [
        ([4, 2], {}),
        ([0, 0], {"enforce_eager": True, "num_gpu_blocks_override": 2}),
        ({}, {"max_num_seqs": None}),
        (
            [70],
            {
                "max_num_active_seqs": 8,
                "async_scheduling": None,
                "cudagraph_mode": None,
                "compilation_mode": None,
                "num_gpu_blocks_override": None,
                "kv_cache_memory_bytes": 0,
            },
        ),
    ],
)
@pytest.mark.parametrize("state_group_layouts", [True, False], ids=["accessor", "none"])
def test_checks_modify_no_configuration(populations, extra, state_group_layouts):
    config = engine_config(
        hf_config(populations, lookahead=3, native_burst=True), **extra
    )
    before = copy.deepcopy(config)
    api.config_errors(config, state_group_layouts=state_group_layouts)
    api.transaction_graphs_on(config.model_config.hf_config)
    assert config == before


# @spec TXG-CFG-011
def test_the_checks_are_defined_once():
    module = api.config_module()
    assert callable(module.transaction_graph_config_errors)
    assert callable(module.transaction_graphs_on)


# ---------------------------------------------------------------------------
# Callers


def _patch_predicate(monkeypatch, value: bool) -> None:
    """Make every loaded package module that holds ``transaction_graphs_on``
    see a predicate that answers ``value``."""
    module = api.config_module()
    real = module.transaction_graphs_on
    fake = lambda hf_config: value  # noqa: E731
    for name, loaded in list(sys.modules.items()):
        if (
            name.startswith("vllm_nemotron_asr_streaming")
            and getattr(loaded, "transaction_graphs_on", None) is real
        ):
            monkeypatch.setattr(loaded, "transaction_graphs_on", fake)


def _hook_refusals(hf):
    """The hook configuration's unmet conditions (REG-ENG-021 to REG-ENG-025)
    at construction, through their one function (REG-ENG-027)."""
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode as CoreCompilationMode

    from vllm_nemotron_asr_streaming import engine_flags

    config = engine_config(hf)
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE,
        mode=CoreCompilationMode.NONE,
        use_breakable_cudagraph=False,
    )
    return engine_flags.runner_mode_refusals(config, stage="constructor")


@pytest.mark.engine
# @spec TXG-CFG-014
@pytest.mark.parametrize("value", [{}, None, False, "", 0], ids=repr)
def test_registration_reads_malformed_values_as_on(value):
    """REG-ENG-005 counts transaction graphs as a hook-captured graph exactly
    when they are on: a malformed value is on there too."""
    eager = dict(encoder_arm="eager", decoder_arm="compact-eager")
    off = _hook_refusals(hf_config(**eager))
    on = _hook_refusals(hf_config(value, **eager))
    assert len(on) == len(off) - 1, (off, on)


@pytest.mark.engine
# @spec TXG-CFG-014
def test_registration_decides_on_off_through_the_predicate(monkeypatch):
    api.config_module()
    import vllm_nemotron_asr_streaming.engine_flags  # noqa: F401 - loaded for patching

    eager = dict(encoder_arm="eager", decoder_arm="compact-eager")
    off = _hook_refusals(hf_config(**eager))
    _patch_predicate(monkeypatch, True)
    assert len(_hook_refusals(hf_config(**eager))) == len(off) - 1
    _patch_predicate(monkeypatch, False)
    assert len(_hook_refusals(hf_config([4], **eager))) == len(off)


def _verify(config) -> None:
    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    Nemotron3_5AsrStreamingConfig.verify_and_update_config(config)


@pytest.mark.engine
# @spec TXG-CFG-011
def test_configuration_hook_reports_the_transaction_graph_conditions(tmp_path):
    from support import model_harness as h

    card = h.write_runnable_card(tmp_path / "card")
    with pytest.raises(ValueError) as caught:
        _verify(h.build_vllm_config(card, right_context=0, overrides={KEY: [0, 0]}))
    message = str(caught.value)
    assert KEY in message and "[0, 0]" in message, message
    assert "--enforce-eager" in message, message


@pytest.mark.engine
# @spec TXG-CFG-011, TXG-CFG-015, TXG-CFG-020
def test_model_state_construction_reruns_the_checks(tmp_path):
    """A value the configuration hook did not see is refused when the model
    state is built against the final engine configuration: ``[3]`` exceeds
    ``P_max`` and the admission limit ``max_num_seqs`` 2."""
    from support import model_harness as h

    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0, max_num_seqs=2)
    model = h.build_model(config)
    config.model_config.hf_config.transaction_graph_populations = [3]
    with pytest.raises(ValueError) as caught:
        h.build_model_state(config, model)
    message = str(caught.value)
    assert KEY in message and "[3]" in message, message
    assert cond.found(cond.ACTIVE_SEQS, message), message


def _install_checks(monkeypatch, fn) -> None:
    """Make ``fn`` the checks function wherever a package module holds the
    real one (by module attribute or by an imported name)."""
    module = api.config_module()
    real = getattr(module, "_txg_real_checks", module.transaction_graph_config_errors)
    monkeypatch.setattr(module, "_txg_real_checks", real, raising=False)
    for name, loaded in list(sys.modules.items()):
        if not name.startswith("vllm_nemotron_asr_streaming"):
            continue
        current = getattr(loaded, "transaction_graph_config_errors", None)
        if current is real or getattr(current, "_txg_installed", False):
            monkeypatch.setattr(loaded, "transaction_graph_config_errors", fn)
    fn._txg_installed = True


@pytest.mark.engine
# @spec TXG-CFG-011
def test_both_callers_use_the_one_function(tmp_path, monkeypatch):
    """A condition that only ``transaction_graph_config_errors`` can produce
    reaches both callers' errors."""
    from support import model_harness as h

    import vllm_nemotron_asr_streaming.streaming_config  # noqa: F401
    from vllm_nemotron_asr_streaming.nemotron_asr import model_state_v2  # noqa: F401

    module = api.config_module()
    real = module.transaction_graph_config_errors
    sentinel = f"{KEY}: sentinel condition from the one function"

    def with_sentinel(vllm_config, **kwargs):
        return [*real(vllm_config, **kwargs), sentinel]

    def plain(vllm_config, **kwargs):
        return real(vllm_config, **kwargs)

    card = h.write_runnable_card(tmp_path / "card")
    _install_checks(monkeypatch, with_sentinel)
    with pytest.raises(ValueError, match="sentinel condition"):
        _verify(h.build_vllm_config(card, right_context=0))
    _install_checks(monkeypatch, plain)
    config = h.build_vllm_config(card, right_context=0)
    model = h.build_model(config)
    _install_checks(monkeypatch, with_sentinel)
    with pytest.raises(ValueError, match="sentinel condition"):
        h.build_model_state(config, model)


def _record_capability(monkeypatch) -> list:
    """Wrap the one function so each call records the state-layer capability
    its caller passed (``None`` when the caller passed none)."""
    module = api.config_module()
    real = module.transaction_graph_config_errors
    seen: list = []

    def recording(vllm_config, **kwargs):
        seen.append(kwargs.get(api.STATE_LAYER_KEYWORD))
        return real(vllm_config, **kwargs)

    _install_checks(monkeypatch, recording)
    return seen


def _without_the_accessor(monkeypatch) -> None:
    """A running core whose ``ModelState`` lacks ``state_group_layouts``."""
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    monkeypatch.delattr(ModelState, "state_group_layouts")


@pytest.mark.engine
# @spec TXG-CFG-016, TXG-CFG-011
@pytest.mark.parametrize("accessor", [True, False], ids=["accessor", "none"])
def test_the_configuration_hook_passes_the_running_cores_capability(
    tmp_path, monkeypatch, accessor
):
    """``verify_and_update_config`` determines whether the running core's
    ``ModelState`` defines ``state_group_layouts`` and passes it; the
    state-layer condition appears exactly when it does not."""
    from support import model_harness as h

    import vllm_nemotron_asr_streaming.streaming_config  # noqa: F401

    card = h.write_runnable_card(tmp_path / "card")
    seen = _record_capability(monkeypatch)
    if not accessor:
        _without_the_accessor(monkeypatch)
    with pytest.raises(ValueError) as caught:
        # core runs the configuration hook while it builds the configuration
        _verify(h.build_vllm_config(card, right_context=0, overrides={KEY: [0, 0]}))
    assert seen and all(value is accessor for value in seen), seen
    message = str(caught.value)
    assert cond.found(cond.ACCESSOR, message) is not accessor, message


@pytest.mark.engine
# @spec TXG-CFG-016, TXG-CFG-011
@pytest.mark.parametrize("accessor", [True, False], ids=["accessor", "none"])
def test_model_state_construction_passes_the_running_cores_capability(
    tmp_path, monkeypatch, accessor
):
    """Model-state construction (REG-ENG-006) passes the same capability."""
    from support import model_harness as h

    from vllm_nemotron_asr_streaming.nemotron_asr import model_state_v2  # noqa: F401

    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0, max_num_seqs=2)
    model = h.build_model(config)
    config.model_config.hf_config.transaction_graph_populations = [0, 0]
    seen = _record_capability(monkeypatch)
    if not accessor:
        _without_the_accessor(monkeypatch)
    with pytest.raises(ValueError) as caught:
        h.build_model_state(config, model)
    assert seen and all(value is accessor for value in seen), seen
    message = str(caught.value)
    assert cond.found(cond.ACCESSOR, message) is not accessor, message
