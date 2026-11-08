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

import pytest
from support.card import (
    BLANK,
    CARD_LOOKAHEADS,
    NUM_PROMPTS,
    generation_config,
    write_card,
)

pytestmark = pytest.mark.engine

IMPLEMENTED = (0, 1, 3, 6, 13)


def _served(model_config):
    from vllm_nemotron_asr_streaming.lookahead import served_lookaheads

    return served_lookaheads(model_config.hf_config)


# --------------------------------------------------------------------------
# Lookahead selection
# --------------------------------------------------------------------------


# @spec REG-LA-001
def test_served_set_defaults_to_the_cards_set(make_model_config, card_dir):
    model_config = make_model_config(card_dir)
    served = _served(model_config)
    assert sorted(served) == sorted(CARD_LOOKAHEADS) and len(set(served)) == 4
    assert model_config.hf_config.card_num_lookahead_tokens == CARD_LOOKAHEADS


# @spec REG-LA-001
def test_served_set_override_replaces_the_cards_set(make_model_config, card_dir):
    model_config = make_model_config(
        card_dir, overrides={"supported_num_lookahead_tokens": [6, 3]}
    )
    assert sorted(_served(model_config)) == [3, 6]


# @spec REG-LA-001
def test_a_top_level_card_set_does_not_serve(make_model_config, tmp_path):
    from support.card import card_config

    path = write_card(
        tmp_path / "c", config=card_config(supported_num_lookahead_tokens=[1])
    )
    model_config = make_model_config(path)
    assert sorted(_served(model_config)) == sorted(CARD_LOOKAHEADS)


# @spec REG-LA-001, REG-LA-004
def test_translated_configuration_may_store_set_and_default(
    make_model_config, write_translated, tmp_path
):
    def store(translated):
        translated["supported_num_lookahead_tokens"] = [3, 6]
        translated["num_lookahead_tokens"] = 6

    path = write_translated(tmp_path / "t", store)
    model_config = make_model_config(path, lookahead=None)
    assert sorted(_served(model_config)) == [3, 6]
    from vllm_nemotron_asr_streaming.lookahead import default_lookahead

    assert default_lookahead(model_config.hf_config) == 6


@pytest.mark.parametrize("served", [[], [2], [3, 14], [-1], [3, 6, 7]])
# @spec REG-LA-002
def test_served_values_must_be_implemented_right_contexts(
    make_model_config, card_dir, served
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            card_dir, overrides={"supported_num_lookahead_tokens": served}
        )
    for value in served:
        if value not in IMPLEMENTED:
            assert str(value) in str(excinfo.value)


@pytest.mark.parametrize("right_context", IMPLEMENTED)
# @spec REG-LA-002
def test_every_implemented_right_context_is_servable(
    make_model_config, card_dir, right_context
):
    model_config = make_model_config(
        card_dir,
        overrides={"supported_num_lookahead_tokens": list(IMPLEMENTED)},
        lookahead=right_context,
    )
    assert sorted(_served(model_config)) == list(IMPLEMENTED)


# @spec REG-LA-003
def test_a_geometry_the_card_does_not_declare_is_logged(
    make_model_config, card_dir, captured_logs
):
    make_model_config(card_dir, overrides={"supported_num_lookahead_tokens": [1, 3]})
    warnings = [
        r
        for r in captured_logs.records
        if r.levelno >= logging.WARNING and "declare" in r.getMessage().lower()
    ]
    assert warnings, "no warning for the undeclared geometry"
    assert any("1" in r.getMessage() for r in warnings)


# @spec REG-LA-003
def test_declared_geometries_log_no_such_warning(
    make_model_config, card_dir, captured_logs
):
    make_model_config(card_dir, overrides={"supported_num_lookahead_tokens": [0, 3]})
    assert not [r for r in captured_logs.records if "declare" in r.getMessage().lower()]


# @spec REG-LA-004
def test_hub_card_needs_the_default_lookahead_from_hf_overrides(
    make_model_config, card_dir
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(card_dir, lookahead=None)
    message = str(excinfo.value)
    assert "num_lookahead_tokens" in message
    assert "--hf-overrides" in message


# @spec REG-LA-004
def test_translated_configuration_without_a_default_is_refused(
    make_model_config, write_translated, tmp_path
):
    path = write_translated(tmp_path / "t")
    with pytest.raises(ValueError, match="num_lookahead_tokens"):
        make_model_config(path, lookahead=None)


@pytest.mark.parametrize(
    ("served", "default"),
    [([0, 6], 3), ([3], 13), ([0, 3, 6, 13], 1)],
)
# @spec REG-LA-005
def test_default_must_be_in_the_served_set(
    make_model_config, card_dir, served, default
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            card_dir,
            overrides={"supported_num_lookahead_tokens": served},
            lookahead=default,
        )
    message = str(excinfo.value)
    assert "num_lookahead_tokens" in message
    assert str(default) in message


@pytest.mark.parametrize("right_context", IMPLEMENTED)
# @spec REG-LA-006
def test_nominal_right_context_is_set_to_the_default(
    make_model_config, card_dir, right_context
):
    model_config = make_model_config(
        card_dir,
        overrides={"supported_num_lookahead_tokens": list(IMPLEMENTED)},
        lookahead=right_context,
    )
    assert model_config.hf_config.att_context_right == right_context


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


# @spec REG-LA-007
def test_native_burst_requires_default_lookahead_one(make_model_config, card_dir):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(card_dir, overrides={"experimental_native_burst": True})
    message = str(excinfo.value)
    assert "experimental_native_burst" in message
    assert "3" in message and "1" in message


# @spec REG-LA-007
def test_native_burst_with_default_one_serves_any_set(make_model_config, card_dir):
    model_config = make_model_config(
        card_dir,
        overrides={
            "experimental_native_burst": True,
            "supported_num_lookahead_tokens": [0, 1, 3],
        },
        lookahead=1,
    )
    assert sorted(_served(model_config)) == [0, 1, 3]


# @spec REG-LA-007
def test_native_burst_false_adds_no_requirement(make_model_config, card_dir):
    make_model_config(card_dir, overrides={"experimental_native_burst": False})


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

    smallest = 1 + REALTIME_MAX_TOKENS
    assert smallest == 143
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
                "experimental_native_burst": True,
                "language_locales": {"en": "fr-FR"},
                "vocab_size": BLANK + 5,
            },
            max_model_len=77,
            override_generation_config={"max_new_tokens": 9},
        )
    message = str(excinfo.value)
    for needle in (
        "11",  # REG-LA-002
        "12",  # REG-LA-005
        "experimental_native_burst",  # REG-LA-007
        "fr-FR",  # REG-VAL-002
        "vocab_size",  # REG-VAL-001
        "max_model_len",  # REG-VAL-001
        "--generation-config <dir>",  # REG-VAL-005
        "--generation-config vllm",  # REG-VAL-006
    ):
        assert needle in message, f"{needle!r} missing from: {message}"


# @spec REG-VAL-003
def test_a_missing_default_is_listed_with_the_other_failures(
    make_model_config, card_dir
):
    with pytest.raises(ValueError) as excinfo:
        make_model_config(
            card_dir,
            lookahead=None,
            overrides={"language_locales": {"en": "fr-FR"}},
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
