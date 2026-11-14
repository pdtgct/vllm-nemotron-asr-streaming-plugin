# SPDX-License-Identifier: Apache-2.0
"""Locale-table helpers: language of a locale, default locale, file-request
locale resolution and the static supported-language table.

The locale table (``prompt_dictionary`` in the served configuration) maps
locale names such as ``en``, ``en-US`` or ``auto`` to rows of the model's
prompt embedding table. Several names may share a row.

Importing this module imports no vLLM.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Core's speech-to-text language map (Whisper's ``LANGUAGES``) restricted to
#: the two-letter languages of the published card's locale table. Static, so
#: the model class can declare it without a configuration.
_SUPPORTED_LANGUAGES: dict[str, str] = {
    "af": "afrikaans",
    "am": "amharic",
    "ar": "arabic",
    "az": "azerbaijani",
    "bg": "bulgarian",
    "bn": "bengali",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "gu": "gujarati",
    "ha": "hausa",
    "he": "hebrew",
    "hi": "hindi",
    "hr": "croatian",
    "hu": "hungarian",
    "hy": "armenian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ka": "georgian",
    "km": "khmer",
    "kn": "kannada",
    "ko": "korean",
    "ln": "lingala",
    "lt": "lithuanian",
    "lv": "latvian",
    "mi": "maori",
    "ml": "malayalam",
    "mr": "marathi",
    "ms": "malay",
    "mt": "maltese",
    "ne": "nepali",
    "nl": "dutch",
    "nn": "nynorsk",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovenian",
    "so": "somali",
    "sv": "swedish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "tg": "tajik",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "uz": "uzbek",
    "vi": "vietnamese",
    "yo": "yoruba",
    "zh": "chinese",
}


def supported_languages() -> dict[str, str]:
    """Return a copy of the static supported-language table."""
    return dict(_SUPPORTED_LANGUAGES)


def locale_language(name: str) -> str | None:
    """Return the language of a locale name, or ``None`` when it has none.

    The language is the lowercase letters before the first ``-`` or
    uppercase letter, when there are two or three of them. ``auto`` has none.
    """
    letters = ""
    for char in name:
        if char == "-" or char.isupper():
            break
        letters += char
    if not letters.isascii() or not letters.isalpha() or not letters.islower():
        return None
    if len(letters) not in (2, 3):
        return None
    return letters


def default_locale(table: Mapping[str, int]) -> str | None:
    """Return ``auto`` when the table has it, else its only locale, else ``None``."""
    if "auto" in table:
        return "auto"
    if len(table) == 1:
        return next(iter(table))
    return None


def locales_of_language(table: Mapping[str, int], language: str) -> list[str]:
    """Return the table's locale names whose language is ``language``, sorted."""
    return sorted(name for name in table if locale_language(name) == language)


def unmapped_languages(
    table: Mapping[str, int],
    language_locales: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the languages that a file request cannot map to a row.

    These are languages whose locales use several distinct rows, that have no
    locale named for the language itself and no ``language_locales`` entry.
    """
    entries = language_locales or {}
    found: dict[str, set[int]] = {}
    for name, row in table.items():
        language = locale_language(name)
        if language is not None:
            found.setdefault(language, set()).add(row)
    return sorted(
        language
        for language, rows in found.items()
        if len(rows) > 1 and language not in table and language not in entries
    )


# @spec FILE-LANG-001, FILE-LANG-002, FILE-LANG-003, FILE-LANG-004
# @spec FILE-LANG-005, FILE-LANG-006
def resolve_file_locale(
    table: Mapping[str, int],
    *,
    language: str | None,
    exact_locale: str | None,
    language_locales: Mapping[str, str] | None = None,
) -> str:
    """Return the locale name (a key of ``table``) for a complete-file request.

    ``exact_locale`` (the request's ``locale`` extra argument) wins and must
    agree with ``language`` when both are given. A ``language`` that is itself
    a key is used as is; otherwise the locales of that language decide: one row
    among them selects it, several rows need a ``language_locales`` entry.
    With neither, the default locale applies.
    """
    if exact_locale is not None:
        if exact_locale not in table:
            named = f" (request language {language!r})" if language else ""
            raise ValueError(
                f"locale {exact_locale!r}{named} is not in the served locale "
                f"table {sorted(table)}"
            )
        if language is not None and locale_language(exact_locale) != language:
            raise ValueError(
                f"request language {language!r} is not the language of locale "
                f"{exact_locale!r}"
            )
        return exact_locale
    if language is not None:
        if language in table:
            return language
        candidates = locales_of_language(table, language)
        if not candidates:
            raise ValueError(
                f"language {language!r} has no locale in the served locale table"
            )
        if len({table[name] for name in candidates}) == 1:
            return candidates[0]
        chosen = (language_locales or {}).get(language)
        if chosen is None or chosen not in table:
            raise ValueError(
                f"language {language!r} has locales {candidates} on several "
                "prompt rows and no locale named for it; set the "
                f"'language_locales' override for {language!r}"
            )
        return chosen
    default = default_locale(table)
    if default is None:
        raise ValueError(
            "the request names no language or locale and the served locale "
            f"table has no 'auto' locale: {sorted(table)}"
        )
    return default
