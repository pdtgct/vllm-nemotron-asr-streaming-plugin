# SPDX-License-Identifier: Apache-2.0
"""Post-override validation for the streaming architecture key.

Registered in ``MODELS_CONFIG_MAP`` under ``Nemotron3_5AsrStreamingForRNNT``
only. ``verify_and_update_model_config`` runs at the end of model
configuration, after core has applied every ``--hf-overrides`` key, and
resolves the server's one geometry (``lookahead.served_lookahead``);
``verify_and_update_config`` runs once while the engine configuration is
resolved and refuses the engine settings the model cannot run under, the
transaction-graph conditions included.
"""

from __future__ import annotations

import logging
from typing import Any

from vllm.model_executor.models.config import VerifyAndUpdateConfig

from .card_parser import SCHEMA_FIELDS, schema_errors
from .engine_flags import check_engine_config
from .identity import ARCHITECTURE
from .locales import locale_language, unmapped_languages
from .lookahead import cadence_label, served_lookahead

logger = logging.getLogger(__name__)


_ABSENT = object()


def _lookahead_errors(hf_config: Any) -> list[str]:
    """Check the server's geometry; set the nominal context."""
    errors: list[str] = []
    served_set = getattr(hf_config, "supported_num_lookahead_tokens", _ABSENT)
    if served_set is not _ABSENT:
        errors.append(
            f"supported_num_lookahead_tokens {served_set!r} is refused: a "
            "server serves one geometry, selected with num_lookahead_tokens "
            "in --hf-overrides (or the checkpoint's default lookahead); remove "
            "supported_num_lookahead_tokens"
        )
    try:
        lookahead: int | None = served_lookahead(hf_config)
    except ValueError as exc:
        errors.append(str(exc))
        lookahead = None
    if lookahead is not None:
        hf_config.att_context_right = lookahead
        card = getattr(hf_config, "card_num_lookahead_tokens", None)
        if isinstance(card, (list, tuple)) and lookahead not in card:
            logger.warning(
                "Serving lookahead %s (%s), a geometry the checkpoint does not "
                "declare (it declares %s).",
                lookahead,
                cadence_label(lookahead),
                sorted(card),
            )
    # Native burst's lookahead rule (REG-LA-007) holds only on a core that
    # runs native burst; on the state layer native burst itself is refused
    # once, in the engine hook (REG-ENG-019), so no lookahead advice is given.
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
    # @spec REG-LA-002, REG-LA-003, REG-LA-004, REG-LA-006, REG-LA-008
    # @spec REG-VAL-001, REG-VAL-002, REG-VAL-003, REG-VAL-004, REG-VAL-005
    # @spec REG-VAL-006, REG-VAL-007
    def verify_and_update_model_config(model_config: Any) -> None:
        from .nemotron_asr.manifests import REALTIME_MAX_TOKENS

        hf_config = model_config.hf_config
        view = {name: getattr(hf_config, name, None) for name in SCHEMA_FIELDS}
        errors = schema_errors(view, None)
        errors += _lookahead_errors(hf_config)

        # A compacted stream holds core's frontier token and the unit's input
        # token before the unit's emitted tokens.
        smallest = 2 + REALTIME_MAX_TOKENS
        max_model_len = getattr(model_config, "max_model_len", None)
        if isinstance(max_model_len, int) and max_model_len < smallest:
            errors.append(
                f"max_model_len {max_model_len} is below {smallest}, the smallest "
                "context that holds one stream unit (the frontier token, the "
                f"input token and {REALTIME_MAX_TOKENS} emitted tokens)"
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
    # @spec REG-VAL-003, REG-VAL-008, REG-ENG-005, REG-ENG-027
    def verify_and_update_config(vllm_config: Any) -> None:
        # One ValueError lists every failed engine check: the engine settings,
        # the hook configuration's conditions at the hook stage
        # (``engine_flags.runner_mode_refusals``), native burst on the state
        # layer and the transaction-graph conditions
        # (``transaction_graph_config_errors``), evaluated on the geometry
        # ``verify_and_update_model_config`` resolved. Settings core's own
        # persistent-state check refuses first reach this hook only when it is
        # called directly.
        check_engine_config(vllm_config, ARCHITECTURE)
