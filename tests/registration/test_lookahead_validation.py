# SPDX-License-Identifier: Apache-2.0
"""Lookahead selection and post-override validation (REG-LA-*, REG-VAL-*).

These run through a real ``ModelConfig`` so that core's own order is
exercised: flat ``hf_overrides`` reach the parser and are re-applied to its
result, dictionary-valued ones (``language_locales``, ``prompt_dictionary``)
are applied afterwards (``ModelConfig.__post_init__``, ``_apply_dict_overrides``
in ``vllm/config/model.py``), and the architecture's
``verify_and_update_model_config`` hook runs last
(``_try_verify_and_update_model_config``).
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

import pytest
from support import runner_conditions as rc
from support.card import (
    BLANK,
    CARD_DEFAULT_LOOKAHEAD,
    CARD_LOOKAHEADS,
    NUM_PROMPTS,
    generation_config,
    write_card,
)

pytestmark = pytest.mark.engine

IMPLEMENTED = (0, 1, 3, 6, 13)
#: The synthetic card's default lookahead (its ``encoder_config``).
CARD_DEFAULT = CARD_DEFAULT_LOOKAHEAD
#: The implemented right contexts, in order, as an error names them.
IMPLEMENTED_PATTERN = r"0\D+1\D+3\D+6\D+13"


def _served(model_config):
    from vllm_nemotron_asr_streaming.lookahead import served_lookahead

    return served_lookahead(model_config.hf_config)


def _without_card_default(translated):
    translated.pop("card_default_num_lookahead_tokens", None)
    translated.pop("num_lookahead_tokens", None)


# --------------------------------------------------------------------------
# The server's lookahead
# --------------------------------------------------------------------------


# @spec REG-LA-001, REG-LA-006
def test_without_an_override_the_card_default_is_served(make_model_config, card_dir):
    model_config = make_model_config(card_dir, lookahead=None)
    assert _served(model_config) == CARD_DEFAULT
    assert model_config.hf_config.att_context_right == CARD_DEFAULT
    assert model_config.hf_config.card_num_lookahead_tokens == CARD_LOOKAHEADS


@pytest.mark.parametrize("right_context", IMPLEMENTED)
# @spec REG-LA-001, REG-LA-006
def test_the_override_selects_the_servers_lookahead(
    make_model_config, card_dir, right_context
):
    model_config = make_model_config(card_dir, lookahead=right_context)
    assert _served(model_config) == right_context
    assert model_config.hf_config.att_context_right == right_context


# @spec REG-LA-001
def test_top_level_card_lookahead_keys_do_not_serve(make_model_config, tmp_path):
    """Translation drops a card's top-level lookahead keys; only its encoder
    configuration and ``--hf-overrides`` choose the geometry."""
    from support.card import card_config

    path = write_card(
        tmp_path / "c",
        config=card_config(
            num_lookahead_tokens=13,
            default_num_lookahead_tokens=13,
            supported_num_lookahead_tokens=[13],
        ),
    )
    model_config = make_model_config(path, lookahead=None)
    assert _served(model_config) == CARD_DEFAULT


# @spec REG-LA-001
def test_translated_configuration_may_store_its_lookahead(
    make_model_config, write_translated, tmp_path
):
    path = write_translated(tmp_path / "t", lambda t: t.update(num_lookahead_tokens=6))
    assert _served(make_model_config(path, lookahead=None)) == 6
    stored_default = write_translated(tmp_path / "d")
    assert _served(make_model_config(stored_default, lookahead=None)) == CARD_DEFAULT


@pytest.mark.parametrize("value", [2, 4, 14, -1, True, 3.0, "3"], ids=repr)
# @spec REG-LA-002
def test_a_bad_override_fails_startup_naming_field_value_and_contexts(
    make_model_config, card_dir, value
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(card_dir, lookahead=value)
    message = str(excinfo.value)
    assert "num_lookahead_tokens" in message
    assert repr(value) in message or str(value) in message
    assert re.search(IMPLEMENTED_PATTERN, message), message


# @spec REG-LA-002
def test_a_bad_stored_card_default_fails_startup_naming_its_field(
    make_model_config, write_translated, tmp_path
):
    def bad(translated):
        _without_card_default(translated)
        translated["card_default_num_lookahead_tokens"] = True

    path = write_translated(tmp_path / "t", bad)
    with pytest.raises(ValueError) as excinfo:
        make_model_config(path, lookahead=None)
    message = str(excinfo.value)
    assert "card_default_num_lookahead_tokens" in message
    assert "True" in message
    assert re.search(IMPLEMENTED_PATTERN, message), message


# @spec REG-LA-003
def test_a_geometry_the_card_does_not_declare_is_logged_and_served(
    make_model_config, card_dir, captured_logs
):
    assert 1 not in CARD_LOOKAHEADS
    model_config = make_model_config(card_dir, lookahead=1)
    assert _served(model_config) == 1
    warnings = [
        r
        for r in captured_logs.records
        if r.levelno >= logging.WARNING and "declare" in r.getMessage().lower()
    ]
    assert warnings, "no warning for the undeclared geometry"
    assert any(re.search(r"\b1\b", r.getMessage()) for r in warnings)


# @spec REG-LA-004
def test_a_configuration_without_any_lookahead_is_refused_with_the_remedy(
    make_model_config, write_translated, tmp_path
):
    path = write_translated(tmp_path / "t", _without_card_default)
    with pytest.raises(ValueError) as excinfo:
        make_model_config(path, lookahead=None)
    message = str(excinfo.value)
    assert "--hf-overrides" in message and "num_lookahead_tokens" in message, message
    assert re.search(IMPLEMENTED_PATTERN, message), message


# @spec REG-LA-004
def test_the_override_alone_serves_a_configuration_without_a_card_default(
    make_model_config, write_translated, tmp_path
):
    path = write_translated(tmp_path / "t", _without_card_default)
    assert _served(make_model_config(path, lookahead=0)) == 0


# @spec REG-LA-006
def test_stale_stored_right_context_is_replaced(
    make_model_config, write_translated, tmp_path
):
    def store(translated):
        translated["att_context_right"] = 13
        translated["num_lookahead_tokens"] = 3

    path = write_translated(tmp_path / "t", store)
    model_config = make_model_config(path, lookahead=None)
    assert model_config.hf_config.att_context_right == 3


# --------------------------------------------------------------------------
# Native burst's lookahead (REG-LA-007, REG-ENG-019)
#
# REG-LA-007 holds where native burst is supported on the running core. This
# build's cores have the state layer, which refuses native burst outright
# (REG-ENG-019), so REG-LA-007's lookahead condition is unit-tested on native
# burst's own check of it (``validate_native_burst_config`` with the 160 ms
# profile), and startup on this core is tested after it.
# --------------------------------------------------------------------------


def _native_burst_lookahead_check(lookahead: int) -> None:
    from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
        validate_native_burst_config,
    )

    single_device = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(logits_processors=None),
    )
    validate_native_burst_config(
        single_device,
        hf_config=SimpleNamespace(num_lookahead_tokens=lookahead),
        require_160ms_profile=True,
    )


@pytest.mark.parametrize("lookahead", [0, 3, 6, 13])
# @spec REG-LA-007
def test_native_bursts_lookahead_check_refuses_another_lookahead_naming_both(
    lookahead,
):
    with pytest.raises(ValueError) as excinfo:
        _native_burst_lookahead_check(lookahead)
    message = str(excinfo.value)
    assert re.search(rf"\b{lookahead}\b", message), message
    assert re.search(r"\b1\b", message), message


# @spec REG-LA-007
def test_native_bursts_lookahead_check_accepts_lookahead_one():
    _native_burst_lookahead_check(1)


#: Advice to serve lookahead 1 (160 ms), REG-LA-007's remedy.
_LOOKAHEAD_ONE_ADVICE = re.compile(
    r"(?i)(?:\blookahead\b|num_lookahead_tokens)[^;]{0,60}?(?<![\d.])1(?![\d])"
    r"|\b160\s?ms\b"
)


@pytest.mark.parametrize("lookahead", [3, 0])
# @spec REG-ENG-019, REG-LA-007
def test_native_burst_at_another_lookahead_is_refused_once_for_the_state_layer(
    make_model_config, make_vllm_config, card_dir, lookahead
):
    """Startup on this core refuses native burst once, for the state layer,
    naming its removal. It does not ask for lookahead 1 first (REG-LA-007's
    remedy, whose premise does not hold here): a user who followed that
    advice would only meet the same refusal."""
    with pytest.raises(ValueError) as excinfo:
        make_vllm_config(
            make_model_config(
                card_dir,
                lookahead=lookahead,
                overrides={"experimental_native_burst": True},
            )
        )
    message = str(excinfo.value)
    assert re.search(r"(?i)experimental_native_burst|native burst", message), message
    assert rc.state_layer_mentions(message) == 1, message
    assert re.search(r"(?i)\bremov", message), message
    assert not _LOOKAHEAD_ONE_ADVICE.search(message), message


# @spec REG-LA-007
def test_native_burst_false_adds_no_requirement(make_model_config, card_dir):
    make_model_config(card_dir, overrides={"experimental_native_burst": False})


def _assert_served_set_refused(message: str) -> None:
    assert "supported_num_lookahead_tokens" in message
    assert "num_lookahead_tokens" in message.replace(
        "supported_num_lookahead_tokens", ""
    ), "the error does not name num_lookahead_tokens as the selection"
    assert re.search(r"remov", message, re.I), message
    assert re.search(r"\b(one|single)\b[^.;]*geometr", message, re.I), message


@pytest.mark.parametrize("served", [[3], [0, 3, 6, 13], [], [1]], ids=str)
# @spec REG-LA-008
def test_a_served_set_override_is_refused(make_model_config, card_dir, served):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            card_dir, overrides={"supported_num_lookahead_tokens": served}
        )
    _assert_served_set_refused(str(excinfo.value))


# @spec REG-LA-008
def test_a_stored_served_set_is_refused(make_model_config, write_translated, tmp_path):
    path = write_translated(
        tmp_path / "t", lambda t: t.update(supported_num_lookahead_tokens=[3, 6])
    )
    with pytest.raises(ValueError) as excinfo:
        make_model_config(path, lookahead=None)
    _assert_served_set_refused(str(excinfo.value))


# @spec REG-LA-008
def test_a_hub_cards_top_level_served_set_is_removed_not_refused(
    make_model_config, tmp_path
):
    from support.card import card_config

    path = write_card(
        tmp_path / "c", config=card_config(supported_num_lookahead_tokens=[0, 3])
    )
    model_config = make_model_config(path, lookahead=None)
    assert (
        getattr(model_config.hf_config, "supported_num_lookahead_tokens", None) is None
    )


# --------------------------------------------------------------------------
# Final-configuration checks
# --------------------------------------------------------------------------

_BAD_FINAL = {
    "eos_token_id": {"eos_token_id": BLANK + 2},
    "audio_chunk_token_id": {"audio_chunk_token_id": BLANK + 9},
    "eou_token_id": {"eou_token_id": BLANK + 9},
    "file_carrier_token_id": {"file_carrier_token_id": BLANK + 9},
    "continuation_token_id": {"continuation_token_id": BLANK + 9},
    "vocab_size": {"vocab_size": BLANK + 5},
    "num_prompts": {"num_prompts": 0},
    "card_num_lookahead_tokens": {"card_num_lookahead_tokens": [2]},
    # implemented, but outside the card's set [3, 0, 6, 13]
    "card_default_num_lookahead_tokens": {"card_default_num_lookahead_tokens": 1},
    "prompt_dictionary": {"prompt_dictionary": {"en": NUM_PROMPTS}},
}


@pytest.mark.parametrize("field", list(_BAD_FINAL))
# @spec REG-VAL-001
def test_overrides_that_break_the_layout_or_schema_are_refused(
    make_model_config, card_dir, field
):
    with pytest.raises(ValueError, match=field):
        make_model_config(card_dir, overrides=_BAD_FINAL[field])


# @spec REG-VAL-001
def test_empty_locale_table_override_is_refused(make_model_config, card_dir):
    with pytest.raises(ValueError, match="prompt_dictionary"):
        make_model_config(card_dir, overrides={"prompt_dictionary": {}})


@pytest.mark.parametrize("extra", [-1, 0])
# @spec REG-VAL-001
def test_max_model_len_must_fit_one_stream_unit(make_model_config, card_dir, extra):
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    # A compacted stream holds core's frontier token and the unit's input
    # token before the unit's emitted tokens.
    smallest = 2 + REALTIME_MAX_TOKENS
    assert smallest == 144
    if extra < 0:
        with pytest.raises(ValueError, match="max_model_len"):
            make_model_config(card_dir, max_model_len=smallest + extra)
    else:
        model_config = make_model_config(card_dir, max_model_len=smallest)
        assert model_config.max_model_len == smallest


# @spec REG-VAL-001
def test_unmodified_final_configuration_passes(make_model_config, card_dir):
    make_model_config(card_dir)


_GOOD_LANGUAGE_LOCALES = [
    {"en": "en-GB"},
    {"en": "enGB"},
    {"es": "es-ES"},
    {"zh": "zh-TW"},
    {"haw": "haw-US"},
    {"nah": "nah-MX"},
    {"zu": "zu-ZA"},
    {"zh": "zh-CN", "en": "en-US"},
]

_BAD_LANGUAGE_LOCALES = [
    {"en": "en-AU"},  # value is not a key of the table
    {"en": "es-US"},  # key is not that locale's language
    {"zh": "zh"},  # not a key
    {"zh": "auto"},  # `auto` has no language
    {"auto": "auto"},
    {"EN": "en-US"},  # the language is lowercase
]


@pytest.mark.parametrize("mapping", _GOOD_LANGUAGE_LOCALES, ids=str)
# @spec REG-VAL-002
def test_language_locales_naming_a_locale_of_the_language_pass(
    make_model_config, card_dir, mapping
):
    model_config = make_model_config(card_dir, overrides={"language_locales": mapping})
    assert model_config.hf_config.language_locales == mapping


@pytest.mark.parametrize("mapping", _BAD_LANGUAGE_LOCALES, ids=str)
# @spec REG-VAL-002
def test_language_locales_entry_that_does_not_match_is_named(
    make_model_config, card_dir, mapping
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(card_dir, overrides={"language_locales": mapping})
    message = str(excinfo.value)
    for key, value in mapping.items():
        assert key in message and value in message


# @spec REG-VAL-003
def test_every_failed_check_is_listed_in_one_error(make_model_config, tmp_path):
    path = write_card(tmp_path / "c", generation=generation_config(eos_token_id=1))
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            path,
            lookahead=12,
            overrides={
                "supported_num_lookahead_tokens": [11],
                "language_locales": {"en": "fr-FR"},
                "vocab_size": BLANK + 5,
            },
            max_model_len=77,
            override_generation_config={"max_new_tokens": 9},
        )
    message = str(excinfo.value)
    for needle in (
        "12",  # REG-LA-002
        "supported_num_lookahead_tokens",  # REG-LA-008
        "fr-FR",  # REG-VAL-002
        "vocab_size",  # REG-VAL-001
        "max_model_len",  # REG-VAL-001
        "--generation-config <dir>",  # REG-VAL-005
        "--generation-config vllm",  # REG-VAL-006
    ):
        assert needle in message, f"{needle!r} missing from: {message}"


# @spec REG-VAL-003, REG-LA-004
def test_a_missing_lookahead_is_listed_with_the_other_failures(
    make_model_config, write_translated, tmp_path
):
    path = write_translated(tmp_path / "t", _without_card_default)
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            path,
            overrides={"language_locales": {"en": "fr-FR"}},
            lookahead=None,
            max_model_len=77,
        )
    message = str(excinfo.value)
    for needle in ("num_lookahead_tokens", "fr-FR", "max_model_len"):
        assert needle in message


# @spec REG-VAL-004
def test_languages_without_a_file_mapping_are_logged(
    make_model_config, card_dir, captured_logs
):
    # `zh` has two rows (zh-CN, zh-TW), no locale named `zh`.
    make_model_config(card_dir)
    assert any(
        r.name.startswith("vllm_nemotron_asr_streaming")
        and re.search(r"\bzh\b", r.getMessage())
        for r in captured_logs.records
    )


# @spec REG-VAL-004
def test_a_language_locales_entry_silences_the_log(
    make_model_config, card_dir, captured_logs
):
    make_model_config(card_dir, overrides={"language_locales": {"zh": "zh-CN"}})
    assert not any(
        r.name.startswith("vllm_nemotron_asr_streaming")
        and re.search(r"\bzh\b", r.getMessage())
        for r in captured_logs.records
    )


def _card_with_generation(tmp_path, name, **fields):
    return write_card(tmp_path / name, generation=generation_config(**fields))


@pytest.mark.parametrize("mode", ["auto", "vllm"])
@pytest.mark.parametrize("eos", [1, BLANK + 2, [BLANK + 1, 1], [1, BLANK + 1]], ids=str)
# @spec REG-VAL-005
def test_a_foreign_checkpoint_eos_is_refused(make_model_config, tmp_path, mode, eos):
    path = _card_with_generation(tmp_path, "c", eos_token_id=eos)
    with pytest.raises(ValueError) as excinfo:
        make_model_config(path, generation_config=mode)
    assert "--generation-config <dir>" in str(excinfo.value)


@pytest.mark.parametrize("eos", [BLANK + 1, [BLANK + 1]], ids=str)
# @spec REG-VAL-005
def test_the_park_token_alone_is_accepted(make_model_config, tmp_path, eos):
    path = _card_with_generation(tmp_path, "c", eos_token_id=eos)
    make_model_config(path)


# @spec REG-VAL-005
def test_a_clean_generation_config_directory_is_the_remedy(make_model_config, tmp_path):
    path = _card_with_generation(tmp_path, "card", eos_token_id=1)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "generation_config.json").write_text(json.dumps(generation_config()))
    model_config = make_model_config(path, generation_config=str(clean))
    assert "eos_token_id" not in model_config.try_get_generation_config()


# @spec REG-VAL-006
def test_a_token_limit_override_is_refused(make_model_config, card_dir):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(card_dir, override_generation_config={"max_new_tokens": 64})
    assert "--generation-config vllm" in str(excinfo.value)


# @spec REG-VAL-006
def test_a_token_limit_override_is_refused_under_the_vllm_defaults_too(
    make_model_config, card_dir
):
    with pytest.raises(ValueError, match="--generation-config vllm"):
        make_model_config(
            card_dir,
            generation_config="vllm",
            override_generation_config={"max_new_tokens": 64},
        )


# @spec REG-VAL-006
def test_a_checkpoint_token_limit_is_refused_and_vllm_defaults_remove_it(
    make_model_config, tmp_path
):
    path = _card_with_generation(tmp_path, "c", max_new_tokens=33)
    with pytest.raises(ValueError, match="--generation-config vllm"):
        make_model_config(path)
    model_config = make_model_config(path, generation_config="vllm")
    assert "max_tokens" not in model_config.get_diff_sampling_param()


# @spec REG-VAL-006
def test_other_sampling_overrides_are_accepted(make_model_config, card_dir):
    model_config = make_model_config(
        card_dir, override_generation_config={"temperature": 0.0}
    )
    assert "max_tokens" not in model_config.get_diff_sampling_param()


# --------------------------------------------------------------------------
# Segment arms are kept (REG-VAL-007)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("encoder", "decoder"),
    [
        ("eager", "dense-eager"),
        ("compiled-static", "compact-eager"),
        ("dense-graphed", "dense-eager"),
        ("eager-graphed", "dense-graphed"),
    ],
)
# @spec REG-VAL-007
def test_arm_overrides_are_the_served_arms(
    make_model_config, card_dir, encoder, decoder
):
    model_config = make_model_config(
        card_dir,
        overrides={"encoder_execution_arm": encoder, "decode_dispatch_arm": decoder},
    )
    assert model_config.hf_config.encoder_execution_arm == encoder
    assert model_config.hf_config.decode_dispatch_arm == decoder


# @spec REG-VAL-007
def test_validation_keeps_the_translated_default_arms(make_model_config, card_dir):
    hf_config = make_model_config(card_dir).hf_config
    assert hf_config.encoder_execution_arm == "eager-graphed"
    assert hf_config.decode_dispatch_arm == "dense-graphed"


# @spec REG-VAL-007
def test_validation_leaves_an_absent_arm_as_the_parser_gave_it(
    make_model_config, write_translated, parse_card, tmp_path
):
    """An already-translated configuration is used as it stands; validation
    writes no arm it lacks."""

    def drop(translated):
        translated.pop("encoder_execution_arm", None)
        translated.pop("decode_dispatch_arm", None)

    path = write_translated(tmp_path / "t", drop)
    _, parsed = parse_card(path)
    hf_config = make_model_config(path).hf_config
    for key in ("encoder_execution_arm", "decode_dispatch_arm"):
        assert getattr(hf_config, key, None) == getattr(parsed, key, None), key


# --------------------------------------------------------------------------
# Transaction-graph configuration checks in startup validation (REG-VAL-008)
# --------------------------------------------------------------------------


def _engine_refusal(make_model_config, make_vllm_config, card_dir, **kwargs) -> str:
    """The message of the engine hook's refusal of a real ``VllmConfig``
    (``verify_and_update_config`` runs inside ``VllmConfig.__post_init__``,
    after ``verify_and_update_model_config`` resolved the geometry)."""
    model_config = make_model_config(card_dir, **kwargs)
    with pytest.raises(ValueError) as excinfo:
        make_vllm_config(model_config)
    return str(excinfo.value)


# @spec REG-VAL-008
def test_transaction_graph_conditions_are_included_at_startup(
    make_model_config, make_vllm_config, card_dir
):
    """At the card default (3, 320 ms) under ``--enforce-eager``: the geometry
    condition (resolved by model validation) and the ``--enforce-eager``
    condition, both naming the option, in one error."""
    message = _engine_refusal(
        make_model_config,
        make_vllm_config,
        card_dir,
        lookahead=None,
        overrides={"transaction_graph_populations": [2]},
    )
    assert "transaction_graph_populations" in message
    assert "--enforce-eager" in message
    assert re.search(r"\b3\b", message), "the resolved lookahead is not named"
    assert "num_lookahead_tokens" in message, "no geometry remedy"


# @spec REG-VAL-008
def test_native_burst_is_refused_while_transaction_graphs_are_on(
    make_model_config, make_vllm_config, card_dir
):
    """At lookahead 1 native burst passes REG-LA-007; the refusal is the
    transaction-graph native-burst condition."""
    message = _engine_refusal(
        make_model_config,
        make_vllm_config,
        card_dir,
        lookahead=1,
        overrides={
            "transaction_graph_populations": [2],
            "experimental_native_burst": True,
        },
    )
    assert "transaction_graph_populations" in message
    assert "experimental_native_burst" in message or "native burst" in message.lower()


# @spec REG-VAL-008
def test_transaction_graphs_off_add_no_condition(
    make_model_config, make_vllm_config, card_dir
):
    for overrides in ({}, {"transaction_graph_populations": []}):
        make_vllm_config(make_model_config(card_dir, lookahead=0, overrides=overrides))


def _transaction_graph_check_owner(repo_root):
    """The module that defines ``transaction_graph_config_errors``."""
    import importlib

    root = repo_root / "vllm_nemotron_asr_streaming"
    for path in sorted(root.rglob("*.py")):
        if "def transaction_graph_config_errors" in path.read_text():
            name = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            return importlib.import_module(name)
    return None


# @spec REG-VAL-008
def test_the_engine_hook_includes_every_condition_the_check_returns(
    make_model_config, make_vllm_config, card_dir, monkeypatch, repo_root
):
    """A sentinel condition added to the check's result reaches the startup
    error, so the hook calls the one check and drops nothing it returns."""
    import sys

    model_config = make_model_config(
        card_dir, lookahead=0, overrides={"transaction_graph_populations": [2]}
    )
    owner = _transaction_graph_check_owner(repo_root)
    assert owner is not None, "no module defines transaction_graph_config_errors"
    original = owner.transaction_graph_config_errors
    sentinel = "transaction_graph_populations sentinel condition 4242"

    def with_sentinel(*args, **kwargs):
        return [*original(*args, **kwargs), sentinel]

    for module in [owner, *sys.modules.values()]:
        if (
            module is not None
            and getattr(module, "__name__", "").startswith(
                "vllm_nemotron_asr_streaming"
            )
            and getattr(module, "transaction_graph_config_errors", None) is original
        ):
            monkeypatch.setattr(
                module, "transaction_graph_config_errors", with_sentinel
            )
    with pytest.raises(ValueError) as excinfo:
        make_vllm_config(model_config)
    assert sentinel in str(excinfo.value)
