# SPDX-License-Identifier: Apache-2.0
"""FILE-IF-004/005 refusals and FILE-LANG: locale binding in get_generation_prompt."""

from __future__ import annotations

import pytest
from support.card import LOCALES

pytestmark = pytest.mark.engine


def _row(prompt_fields, prompt) -> int:
    return int(prompt_fields(prompt)[1][1])


# @spec FILE-IF-004
@pytest.mark.parametrize(
    "field, kwargs, pattern",
    [
        ("task", {"task_type": "translate"}, "translat|task"),
        ("prompt", {"request_prompt": "context"}, "prompt"),
        ("hotwords", {"hotwords": "nvidia"}, "hotwords"),
        ("target language", {"to_language": "fr"}, "to_language|target"),
    ],
)
def test_unsupported_request_fields_are_refused(
    model_cls, stt_params, field, kwargs, pattern
):
    with pytest.raises(ValueError, match=f"(?i){pattern}"):
        model_cls.get_generation_prompt(stt_params(**kwargs))


# @spec FILE-IF-005
def test_native_burst_refuses_complete_files(model_cls, stt_params):
    # native burst requires the default lookahead 1 (REG-LA-007)
    with pytest.raises(ValueError, match="(?i)native.burst"):
        model_cls.get_generation_prompt(stt_params(right_context=1, native_burst=True))


# @spec FILE-LANG-001
@pytest.mark.parametrize("language", ["en", "es", "de"])
def test_language_that_is_a_locale_key_uses_it(
    model_cls, stt_params, prompt_fields, language
):
    prompt = model_cls.get_generation_prompt(stt_params(language=language))
    assert _row(prompt_fields, prompt) == LOCALES[language]


# @spec FILE-LANG-001
def test_bare_language_beats_regional_rows(model_cls, stt_params, prompt_fields):
    # "es" is its own key (row 3, the es-US row), not es-ES (row 2)
    assert LOCALES["es"] != LOCALES["es-ES"]
    prompt = model_cls.get_generation_prompt(stt_params(language="es"))
    assert _row(prompt_fields, prompt) == LOCALES["es"]


# @spec FILE-LANG-002
@pytest.mark.parametrize(
    "language, locale",
    [("af", "af-ZA"), ("haw", "haw-US"), ("nah", "nah-MX"), ("zu", "zu-ZA")],
)
def test_language_with_one_row_maps_to_that_row(
    model_cls, stt_params, prompt_fields, language, locale
):
    prompt = model_cls.get_generation_prompt(stt_params(language=language))
    assert _row(prompt_fields, prompt) == LOCALES[locale]


# @spec FILE-LANG-002
def test_language_with_several_rows_and_no_mapping_is_refused(model_cls, stt_params):
    with pytest.raises(ValueError) as err:
        model_cls.get_generation_prompt(stt_params(language="zh"))
    assert "zh" in str(err.value)
    assert "language_locales" in str(err.value)


# @spec FILE-LANG-002
@pytest.mark.parametrize("locale", ["zh-CN", "zh-TW"])
def test_language_with_several_rows_uses_language_locales(
    model_cls, stt_params, prompt_fields, locale
):
    prompt = model_cls.get_generation_prompt(
        stt_params(language="zh", language_locales={"zh": locale})
    )
    assert _row(prompt_fields, prompt) == LOCALES[locale]


# @spec FILE-LANG-002
def test_language_locales_does_not_override_a_single_row(
    model_cls, stt_params, prompt_fields
):
    # "af" has one row; an entry for it must agree with the table, and the
    # one row is used either way
    prompt = model_cls.get_generation_prompt(
        stt_params(language="af", language_locales={"af": "af-ZA"})
    )
    assert _row(prompt_fields, prompt) == LOCALES["af-ZA"]


# @spec FILE-LANG-003
@pytest.mark.parametrize("language", ["fr", "ja", "qq"])
def test_language_without_a_locale_is_refused(model_cls, stt_params, language):
    with pytest.raises(ValueError, match=language):
        model_cls.get_generation_prompt(stt_params(language=language))


# @spec FILE-LANG-004
def test_no_language_uses_auto(model_cls, stt_params, prompt_fields):
    prompt = model_cls.get_generation_prompt(stt_params())
    assert _row(prompt_fields, prompt) == LOCALES["auto"]


# @spec FILE-LANG-004
def test_no_language_uses_the_only_locale_without_auto(
    model_cls, stt_params, prompt_fields
):
    prompt = model_cls.get_generation_prompt(stt_params(locales={"en-US": 3}))
    assert _row(prompt_fields, prompt) == 3


# @spec FILE-LANG-004
def test_no_language_with_several_locales_and_no_auto_is_refused(model_cls, stt_params):
    with pytest.raises(ValueError):
        model_cls.get_generation_prompt(stt_params(locales={"en-US": 0, "de-DE": 1}))


# @spec FILE-LANG-005
def test_exact_locale_beats_language_mapping(model_cls, stt_params, prompt_fields):
    # language "en" alone gives the "en" row (FILE-LANG-001); the exact locale
    # wins and may name an alias row
    assert LOCALES["en-GB"] != LOCALES["en"]
    prompt = model_cls.get_generation_prompt(
        stt_params(language="en", extra_args={"locale": "en-GB"})
    )
    assert _row(prompt_fields, prompt) == LOCALES["en-GB"]


# @spec FILE-LANG-005
@pytest.mark.parametrize("locale", ["en-GB", "enGB", "zh-TW", "zu-ZA", "nah-MX"])
def test_exact_locale_without_language_is_used(
    model_cls, stt_params, prompt_fields, locale
):
    # no language: FILE-LANG-004 would give "auto"
    prompt = model_cls.get_generation_prompt(stt_params(extra_args={"locale": locale}))
    assert _row(prompt_fields, prompt) == LOCALES[locale]


# @spec FILE-LANG-006
def test_unknown_exact_locale_is_refused(model_cls, stt_params):
    with pytest.raises(ValueError, match="xx-XX"):
        model_cls.get_generation_prompt(stt_params(extra_args={"locale": "xx-XX"}))


# @spec FILE-LANG-006
def test_language_that_is_not_the_locales_language_is_refused(model_cls, stt_params):
    with pytest.raises(ValueError) as err:
        model_cls.get_generation_prompt(
            stt_params(language="de", extra_args={"locale": "en-GB"})
        )
    assert "en-GB" in str(err.value)
    assert "de" in str(err.value)


# @spec FILE-LANG-006
def test_language_matching_the_locales_language_is_accepted(
    model_cls, stt_params, prompt_fields
):
    # "enGB" is language "en" by the lowercase-letters rule
    prompt = model_cls.get_generation_prompt(
        stt_params(language="en", extra_args={"locale": "enGB"})
    )
    assert _row(prompt_fields, prompt) == LOCALES["enGB"]
