# SPDX-License-Identifier: Apache-2.0
"""Post-override validation for the streaming architecture key.

Registered in ``MODELS_CONFIG_MAP`` under ``Nemotron3_5AsrStreamingForRNNT``
only. ``verify_and_update_model_config`` runs at the end of model
configuration, after core has applied every ``--hf-overrides`` key, and
resolves the launch options (served lookahead set, default lookahead);
``verify_and_update_config`` runs once while the engine configuration is
resolved and refuses the engine settings the model cannot run under.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm.model_executor.models.config import VerifyAndUpdateConfig

from .card_parser import SCHEMA_FIELDS, schema_errors
from .engine_flags import check_engine_config
from .identity import ARCHITECTURE
from .locales import locale_language, unmapped_languages
from .lookahead import (
    IMPLEMENTED_RIGHT_CONTEXTS,
    default_lookahead,
    served_lookahead_source,
)

logger = logging.getLogger(__name__)


def _lookahead_errors(hf_config: Any) -> list[str]:
    """Check the served set and the default lookahead; set the nominal context."""
    errors: list[str] = []
    source = served_lookahead_source(hf_config)
    raw = getattr(hf_config, source, None)
    served: list[Any] = []
    if raw is None:
        errors.append(f"{source} is missing")
    elif isinstance(raw, (list, tuple)):
        served = list(raw)
        if not served:
            errors.append(f"{source} is empty; it needs at least one right context")
    else:
        errors.append(f"{source} must be a list of integers, got {raw!r}")
    for value in served:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in IMPLEMENTED_RIGHT_CONTEXTS
        ):
            errors.append(
                f"{source} contains {value!r}, which is not an implemented "
                f"right context {IMPLEMENTED_RIGHT_CONTEXTS}"
            )
    card = getattr(hf_config, "card_num_lookahead_tokens", None)
    if isinstance(card, (list, tuple)):
        for value in served:
            if value in IMPLEMENTED_RIGHT_CONTEXTS and value not in card:
                logger.warning(
                    "Serving lookahead %s, a geometry the checkpoint does not "
                    "declare (it declares %s).",
                    value,
                    sorted(card),
                )

    default = default_lookahead(hf_config)
    if default is None:
        errors.append(
            "num_lookahead_tokens is not set: set the default lookahead with "
            "--hf-overrides '{\"num_lookahead_tokens\": <N>}' "
            f"(one of {sorted(v for v in served if isinstance(v, int))})"
        )
    elif default not in served:
        errors.append(
            f"num_lookahead_tokens {default!r} is not in the served lookahead "
            f"set {served!r} ({source})"
        )
    else:
        hf_config.att_context_right = default
    if getattr(hf_config, "experimental_native_burst", False) and default != 1:
        errors.append(
            "experimental_native_burst requires num_lookahead_tokens 1, "
            f"got {default!r}"
        )
    return errors


def _locale_errors(hf_config: Any) -> list[str]:
    errors: list[str] = []
    table = getattr(hf_config, "prompt_dictionary", None)
    language_locales = getattr(hf_config, "language_locales", None)
    if language_locales is None:
        return errors
    if not isinstance(language_locales, dict):
        return [
            f"language_locales must map languages to locales, got {language_locales!r}"
        ]
    if not isinstance(table, dict):
        return errors
    for language, locale in language_locales.items():
        if locale not in table:
            errors.append(
                f"language_locales entry {language!r}: {locale!r} is not a "
                "locale of the served locale table"
            )
        elif locale_language(locale) != language:
            errors.append(
                f"language_locales entry {language!r}: {locale!r} is not a "
                f"locale of language {language!r}"
            )
    return errors


def _generation_errors(model_config: Any, park: int | None) -> list[str]:
    errors: list[str] = []
    if park is not None:
        eos = model_config.try_get_generation_config().get("eos_token_id")
        if eos is not None:
            ids = [eos] if isinstance(eos, int) else list(eos)
            if ids != [park]:
                errors.append(
                    f"the generation configuration has eos_token_id {eos!r}; "
                    f"only the park token {park} may be an EOS. Pass "
                    "--generation-config <dir> naming a directory whose "
                    "generation_config.json has no other EOS"
                )
    if "max_tokens" in model_config.get_diff_sampling_param():
        errors.append(
            "the default sampling parameters set max_tokens (a token limit "
            "from the checkpoint's generation configuration or "
            "--override-generation-config). Pass --generation-config vllm "
            "and remove any token-limit --override-generation-config"
        )
    return errors


class Nemotron3_5AsrStreamingConfig(VerifyAndUpdateConfig):
    """Refuse a served configuration or engine the streaming model cannot run."""

    @staticmethod
    # @spec REG-LA-001, REG-LA-002, REG-LA-003, REG-LA-004, REG-LA-005
    # @spec REG-LA-006, REG-LA-007, REG-VAL-001, REG-VAL-002, REG-VAL-003
    # @spec REG-VAL-004, REG-VAL-005, REG-VAL-006
    def verify_and_update_model_config(model_config: Any) -> None:
        from .nemotron_asr.manifests import REALTIME_MAX_TOKENS

        hf_config = model_config.hf_config
        view = {name: getattr(hf_config, name, None) for name in SCHEMA_FIELDS}
        errors = schema_errors(view, None)
        errors += _lookahead_errors(hf_config)

        smallest = 1 + REALTIME_MAX_TOKENS
        max_model_len = getattr(model_config, "max_model_len", None)
        if isinstance(max_model_len, int) and max_model_len < smallest:
            errors.append(
                f"max_model_len {max_model_len} is below {smallest}, the smallest "
                f"context that holds one stream unit (1 + {REALTIME_MAX_TOKENS} "
                "emitted tokens)"
            )
        errors += _locale_errors(hf_config)
        blank = view.get("blank_token_id")
        park = (
            blank + 1
            if isinstance(blank, int) and not isinstance(blank, bool)
            else None
        )
        errors += _generation_errors(model_config, park)
        table = getattr(hf_config, "prompt_dictionary", None)
        language_locales = getattr(hf_config, "language_locales", None)
        if (
            isinstance(table, dict)
            and all(isinstance(row, int) for row in table.values())
            and (language_locales is None or isinstance(language_locales, dict))
        ):
            for language in unmapped_languages(table, language_locales):
                logger.warning(
                    "Complete-file requests for language %r cannot be mapped to "
                    "a prompt row: its locales use several rows. Add a "
                    "language_locales entry to serve it.",
                    language,
                )
        if errors:
            raise ValueError(
                f"{ARCHITECTURE} configuration refused: " + "; ".join(errors)
            )

    @staticmethod
    def verify_and_update_config(vllm_config: Any) -> None:
        check_engine_config(vllm_config, ARCHITECTURE)
