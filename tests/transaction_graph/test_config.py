# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph option and its configuration checks (TXG-CFG-001 to
TXG-CFG-016).

Every core this build supports has a state layer without a state-group layout
accessor, so whenever transaction graphs are on the checks also return
TXG-CFG-016's state-layer condition (``errors_for`` asserts that exactly one
condition mentions the state layer, that it is TXG-CFG-016's in full, and
returns the other conditions).

CPU tests on configuration doubles (``support.txg_fakes``), except the
callers of TXG-CFG-011, which use the stage 1 build (``engine``). The checks'
interface is documented in ``support.txg_api``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from support import runner_conditions as rc
from support import txg_api as api
from support.txg_fakes import engine_config, hf_config

KEY = api.KEY
VALID = [4]


def without_state_layer(errors, value):
    """The conditions other than TXG-CFG-016's, after asserting that exactly
    one condition mentions the state layer and that it is TXG-CFG-016's in
    full: the option, the rejected ``value`` (TXG-CFG-003), the missing
    state-group layout accessor and the option's removal. A condition that
    folded another violation into the state-layer one would be the one
    mentioning it, so it is not silently dropped: the other conditions are
    still asserted by the callers."""
    found = [e for e in errors if rc.state_layer_mentions(e)]
    assert len(found) == 1, errors
    (condition,) = found
    assert rc.is_state_layer_condition(condition), condition
    assert shows_value(condition, value), (value, condition)
    return [e for e in errors if e != condition]


def errors_for(populations=VALID, **kwargs):
    """The conditions other than the state-layer one (TXG-CFG-016) for a
    configuration with transaction graphs on; asserts that one is there."""
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
    return without_state_layer(
        api.config_errors(engine_config(hf_config(populations, **hf_kwargs), **kwargs)),
        populations,
    )


def mentioning(errors, *needles):
    return [e for e in errors if all(n.lower() in e.lower() for n in needles)]


def shows_value(error, value):
    """Whether ``error`` shows ``value``: its ``repr``, ``str`` or JSON
    rendering, or a string bare or in any common quotes. An empty rendering
    shows nothing, so ``""`` must appear quoted."""
    forms = {repr(value), str(value)}
    with contextlib.suppress(TypeError, ValueError):
        forms.add(json.dumps(value))
    if isinstance(value, str):
        forms |= {f"'{value}'", f'"{value}"', f"`{value}`"}
    return any(form and form.lower() in error.lower() for form in forms)


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
    errors = without_state_layer(
        api.config_errors(engine_config(hf_config(value))), value
    )
    # required information: the option's name and the offending value, in
    # whatever quoting the diagnostic uses
    assert [e for e in mentioning(errors, KEY) if shows_value(e, value)], errors
    with pytest.raises(ValueError):
        api.largest_population(hf_config(value))


# @spec TXG-CFG-003
@pytest.mark.parametrize("value", OFF_VALUES)
def test_no_condition_while_off(value):
    """With the option off, nothing this function checks is reported, even
    for settings that the option on would refuse."""
    hf = served(value)
    hf.num_lookahead_tokens = 3
    hf.experimental_native_burst = True
    hf.encoder_execution_arm = "eager"
    config = engine_config(hf, enforce_eager=True, num_gpu_blocks_override=2)
    assert api.config_errors(config) == []


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
        pytest.param([65], id="above-max-num-seqs"),
        pytest.param((4,), id="tuple"),
        pytest.param({4}, id="set"),
    ],
)
def test_populations_condition(value):
    errors = errors_for(value, max_num_seqs=64, max_num_batched_tokens=8192)
    found = mentioning(errors, KEY)
    assert len(found) == 1, errors
    assert repr(value) in found[0] or str(value) in found[0], found
    if value == [65]:
        # The batching limit is engine configuration that ``largest_population``
        # does not see; only the configuration check can refuse this value.
        return
    with pytest.raises(ValueError):
        api.largest_population(hf_config(value))


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


# @spec TXG-CFG-005
def test_populations_are_bounded_by_the_non_null_pages():
    assert errors_for([32], num_gpu_blocks_override=33) == []
    found = mentioning(errors_for([33], num_gpu_blocks_override=33), KEY, "33")
    assert found, errors_for([33], num_gpu_blocks_override=33)
    assert mentioning(errors_for([4, 33], num_gpu_blocks_override=33), KEY)
    assert errors_for([64], num_gpu_blocks_override=None) == []


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
    other conditions still apply."""
    seqs, tokens = limits
    errors = errors_for(value, max_num_seqs=seqs, max_num_batched_tokens=tokens)
    assert bool(mentioning(errors, KEY)) is refused, errors


# ---------------------------------------------------------------------------
# Geometry, native burst, segment arms, enforce_eager


# @spec TXG-CFG-006
@pytest.mark.parametrize("lookahead", [0, 1])
def test_80_and_160_ms_are_accepted(lookahead):
    assert errors_for(lookahead=lookahead) == []


# @spec TXG-CFG-006
@pytest.mark.parametrize(
    "lookahead", [3, 6, 13, None], ids=["3", "6", "13", "card-default-3"]
)
def test_other_geometries_are_refused_with_the_remedy(lookahead):
    errors = errors_for(lookahead=lookahead)
    found = mentioning(errors, KEY, "num_lookahead_tokens", "--hf-overrides")
    assert len(found) == 1, errors
    shown = 3 if lookahead is None else lookahead
    assert str(shown) in found[0], found
    assert "0" in found[0] and "1" in found[0], found


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


# @spec TXG-CFG-007
def test_native_burst_is_refused():
    errors = errors_for(native_burst=True)
    found = mentioning(errors, KEY, "experimental_native_burst")
    assert len(found) == 1, errors
    assert errors_for(native_burst=False) == []


# @spec TXG-CFG-008
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

    errors = errors_for(
        encoder_arm=_ABSENT if encoder == "unset" else encoder,
        decoder_arm=_ABSENT if decoder == "unset" else decoder,
    )
    found = mentioning(errors, KEY, *[n for n, _ in names])
    assert len(found) == 1, errors
    for arm, value in names:
        assert value in found[0], (arm, value, found)


# @spec TXG-CFG-008
def test_default_arms_are_accepted_and_chunk_graphs_are_left_to_registration():
    assert errors_for() == []
    errors = errors_for(chunk_populations=[1, 2])
    assert not mentioning(errors, "chunk_bucket_graph_populations"), errors


# @spec TXG-CFG-009
def test_enforce_eager_is_refused_naming_both():
    errors = errors_for(enforce_eager=True)
    found = mentioning(errors, KEY, "--enforce-eager")
    assert len(found) == 1, errors
    assert "hook" in found[0].lower(), found


# @spec TXG-CFG-010
@pytest.mark.parametrize(
    "setting",
    [
        {"cudagraph_mode": 2},
        {"compilation_mode": 3},
        {"device_type": "cpu"},
        {"cudagraph_mode": None, "compilation_mode": None},
    ],
)
def test_runner_and_platform_conditions_are_not_repeated(setting):
    assert errors_for(**setting) == []


# @spec TXG-CFG-003, TXG-CFG-016
def test_every_violated_condition_is_returned():
    from support.txg_fakes import _ABSENT

    # The whole list: the state-layer condition is one entry among six, and
    # each other violation is its own entry beside it.
    errors = api.config_errors(
        engine_config(
            hf_config([0], lookahead=3, native_burst=True, encoder_arm=_ABSENT),
            enforce_eager=True,
        )
    )
    assert len(errors) == 6, errors
    for error in errors:
        assert KEY in error, error
    others = without_state_layer(errors, [0])
    assert len(others) == 5, errors
    for needle in (
        "0",
        "num_lookahead_tokens",
        "experimental_native_burst",
        "encoder_execution_arm",
        "--enforce-eager",
    ):
        assert mentioning(others, needle), (needle, errors)


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
    ],
)
def test_checks_modify_no_configuration(populations, extra):
    config = engine_config(
        hf_config(populations, lookahead=3, native_burst=True), **extra
    )
    before = copy.deepcopy(config)
    api.config_errors(config)
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
    from vllm.config.compilation import CompilationMode

    from vllm_nemotron_asr_streaming import engine_flags

    config = engine_config(hf)
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE,
        mode=CompilationMode.NONE,
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


@pytest.mark.engine
# @spec TXG-CFG-011
def test_configuration_hook_reports_the_transaction_graph_conditions(tmp_path):
    from support import model_harness as h

    card = h.write_runnable_card(tmp_path / "card")
    with pytest.raises(ValueError) as caught:
        config = h.build_vllm_config(card, right_context=0, overrides={KEY: [0, 0]})
        from vllm_nemotron_asr_streaming.streaming_config import (
            Nemotron3_5AsrStreamingConfig,
        )

        Nemotron3_5AsrStreamingConfig.verify_and_update_config(config)
    message = str(caught.value)
    assert KEY in message and "[0, 0]" in message, message
    assert "--enforce-eager" in message, message


@pytest.mark.engine
# @spec TXG-CFG-011, TXG-CFG-015
def test_model_state_construction_reruns_the_checks(tmp_path, monkeypatch):
    """A value the configuration hook did not see is refused when the model
    state is built against the final engine configuration."""
    from support import model_harness as h

    h.spec_without_state_name(monkeypatch)
    card = h.write_runnable_card(tmp_path / "card")
    config = h.build_vllm_config(card, right_context=0, max_num_seqs=2)
    model = h.build_model(config)
    config.model_config.hf_config.transaction_graph_populations = [3]
    with pytest.raises(ValueError) as caught:
        h.build_model_state(config, model)
    message = str(caught.value)
    assert KEY in message and "[3]" in message, message


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

    def with_sentinel(vllm_config):
        return [*real(vllm_config), sentinel]

    def plain(vllm_config):
        return real(vllm_config)

    h.spec_without_state_name(monkeypatch)
    card = h.write_runnable_card(tmp_path / "card")
    _install_checks(monkeypatch, with_sentinel)
    with pytest.raises(ValueError, match="sentinel condition"):
        config = h.build_vllm_config(card, right_context=0)
        from vllm_nemotron_asr_streaming.streaming_config import (
            Nemotron3_5AsrStreamingConfig,
        )

        Nemotron3_5AsrStreamingConfig.verify_and_update_config(config)
    _install_checks(monkeypatch, plain)
    config = h.build_vllm_config(card, right_context=0)
    model = h.build_model(config)
    _install_checks(monkeypatch, with_sentinel)
    with pytest.raises(ValueError, match="sentinel condition"):
        h.build_model_state(config, model)
