# SPDX-License-Identifier: Apache-2.0
"""Config parser for the published Nemotron ASR streaming card.

Selected with ``--config-format nemotron_asr_streaming``. The card's
``model_type`` has no vLLM configuration class and its ``architectures``
names the offline plugin's key, so this parser translates the card into the
served configuration (``NemotronASRConfig``) and registers the streaming
architecture key.

The parser is a function of the checkpoint files alone: it never reads
``hf_overrides`` (core applies the flat ones to the returned configuration
and the dictionary-valued ones later still). Every rule that depends on
launch options runs in post-override validation (``streaming_config``).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from .identity import ARCHITECTURE
from .lookahead import IMPLEMENTED_RIGHT_CONTEXTS
from .nemotron_asr.configuration_nemotron_asr import NemotronASRConfig

#: Control-token configuration fields at ``blank + 1`` to ``blank + 5``.
CONTROL_TOKEN_FIELDS: tuple[str, ...] = (
    "eos_token_id",
    "audio_chunk_token_id",
    "eou_token_id",
    "file_carrier_token_id",
    "continuation_token_id",
)

#: Configuration fields the schema check reads.
SCHEMA_FIELDS: tuple[str, ...] = (
    "blank_token_id",
    "num_asr_labels",
    *CONTROL_TOKEN_FIELDS,
    "vocab_size",
    "num_prompts",
    "prompt_dictionary",
    "card_num_lookahead_tokens",
)

#: The model has no positional limit; this lifts only core's derived context
#: ceiling, so the deployment's ``max_model_len`` is the effective bound.
MAX_POSITION_EMBEDDINGS = 2**24

_ENCODER_INT_FIELDS: tuple[tuple[str, str], ...] = (
    ("hidden_size", "d_model"),
    ("num_hidden_layers", "n_layers"),
    ("conv_kernel_size", "conv_kernel"),
    ("sliding_window", "att_context_left"),
    ("num_mel_bins", "n_mels"),
)

#: Card keys that the translation reinterprets or drops.
_DROPPED_CARD_KEYS: tuple[str, ...] = (
    "encoder_config",
    "model_type",
    "is_encoder_decoder",
    "default_prompt_id",
    "sliding_window",
    "supported_num_lookahead_tokens",
    "default_num_lookahead_tokens",
    "num_lookahead_tokens",
    "hidden_size",
    "vocab_size",
    "dtype",
    "torch_dtype",
    *CONTROL_TOKEN_FIELDS,
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(mapping: Mapping[str, Any], key: str, source: str) -> int:
    value = mapping.get(key)
    if not _is_int(value):
        raise ValueError(f"{source} requires an integer {key!r}, got {value!r}")
    return value


def _table_errors(table: object, num_prompts: object, source: str) -> list[str]:
    """Check a locale table: non-empty, integer rows in ``0 … num_prompts − 1``."""
    if not isinstance(table, dict) or not table:
        return [f"prompt_dictionary ({source}) must be a non-empty locale table"]
    errors: list[str] = []
    for locale, row in table.items():
        if not isinstance(locale, str) or not locale:
            errors.append(
                f"prompt_dictionary ({source}) has a locale name {locale!r} "
                "that is not a non-empty string"
            )
        elif not _is_int(row):
            errors.append(
                f"prompt_dictionary ({source}) row of {locale!r} must be an "
                f"integer, got {row!r}"
            )
        elif row < 0 or (_is_int(num_prompts) and row >= num_prompts):
            errors.append(
                f"prompt_dictionary ({source}) row {row} of {locale!r} is "
                f"outside 0 … num_prompts − 1 (num_prompts {num_prompts!r})"
            )
    return errors


# @spec REG-CARD-011
def schema_errors(
    config: Mapping[str, Any], processor: Mapping[str, Any] | None
) -> list[str]:
    """Return every schema violation of a translated configuration.

    ``processor`` is the card's ``processor_config.json``; ``None`` when only
    the configuration is at hand (post-override validation), in which case the
    configuration's stored locale table is the one checked.
    """
    errors: list[str] = []
    blank = config.get("blank_token_id")
    if not _is_int(blank):
        errors.append(f"blank_token_id must be an integer, got {blank!r}")
    else:
        labels = config.get("num_asr_labels")
        if labels != blank:
            errors.append(
                f"num_asr_labels must equal blank_token_id {blank}, got {labels!r}"
            )
        for offset, name in enumerate(CONTROL_TOKEN_FIELDS, start=1):
            if config.get(name) != blank + offset:
                errors.append(
                    f"{name} must be blank + {offset} = {blank + offset}, "
                    f"got {config.get(name)!r}"
                )
        if config.get("vocab_size") != blank + len(CONTROL_TOKEN_FIELDS) + 1:
            errors.append(
                f"vocab_size must be blank + {len(CONTROL_TOKEN_FIELDS) + 1} = "
                f"{blank + len(CONTROL_TOKEN_FIELDS) + 1}, "
                f"got {config.get('vocab_size')!r}"
            )

    num_prompts = config.get("num_prompts")
    if not _is_int(num_prompts) or num_prompts <= 0:
        errors.append(f"num_prompts must be a positive integer, got {num_prompts!r}")
    elif processor is not None and processor.get("num_prompts") != num_prompts:
        errors.append(
            f"num_prompts {num_prompts} differs from processor_config.json's "
            f"{processor.get('num_prompts')!r}"
        )

    stored = config.get("prompt_dictionary")
    if processor is not None:
        errors += _table_errors(
            processor.get("prompt_dictionary"), num_prompts, "processor_config.json"
        )
        if stored is not None and stored != processor.get("prompt_dictionary"):
            errors.append(
                "prompt_dictionary stored in the configuration differs from "
                "processor_config.json's"
            )
    else:
        errors += _table_errors(stored, num_prompts, "configuration")

    card = config.get("card_num_lookahead_tokens")
    if not isinstance(card, (list, tuple)) or not card:
        errors.append(
            "card_num_lookahead_tokens must be a non-empty list of right "
            f"contexts, got {card!r}"
        )
    else:
        for value in card:
            if not _is_int(value) or value not in IMPLEMENTED_RIGHT_CONTEXTS:
                errors.append(
                    f"card_num_lookahead_tokens contains {value!r}, which is "
                    f"not an implemented right context {IMPLEMENTED_RIGHT_CONTEXTS}"
                )
    return errors


# @spec REG-CARD-005, REG-CARD-006, REG-CARD-007, REG-CARD-008, REG-CARD-009
# @spec REG-CARD-010
def translate_card(
    config: Mapping[str, Any], processor: Mapping[str, Any]
) -> dict[str, Any]:
    """Translate a Hub card's ``config.json`` into the served configuration."""
    card = copy.deepcopy(dict(config))
    encoder = card.get("encoder_config")
    if not isinstance(encoder, dict) or not encoder:
        raise ValueError("public card requires encoder_config")
    blank = _require_int(card, "blank_token_id", "public card")
    joint_width = card.get("vocab_size")
    if joint_width != blank + 1:
        raise ValueError(
            "public card identity mismatch: vocab_size must be "
            f"blank_token_id + 1 (blank last), got vocab_size={joint_width!r} "
            f"blank_token_id={blank}"
        )
    supported = encoder.get("supported_num_lookahead_tokens")
    if supported is None:
        raise ValueError(
            "public card encoder_config requires supported_num_lookahead_tokens"
        )
    mapped: dict[str, int] = {}
    for source_key, target_key in _ENCODER_INT_FIELDS:
        value = _require_int(encoder, source_key, "public card encoder_config")
        mapped[target_key] = value - 1 if source_key == "sliding_window" else value
    d_model = mapped["d_model"]
    prompt_intermediate = card.get("prompt_intermediate_size")
    if prompt_intermediate is not None and prompt_intermediate != 2 * d_model:
        raise ValueError(
            "prompt_intermediate_size mismatch: the model's prompt kernel is "
            "Linear(hidden + prompts -> 2 * hidden); the card declares "
            f"{prompt_intermediate!r} for hidden_size {d_model}"
        )

    translated: dict[str, Any] = {
        key: value for key, value in card.items() if key not in _DROPPED_CARD_KEYS
    }
    if "decoder_hidden_size" in translated:
        translated["pred_hidden"] = translated.pop("decoder_hidden_size")
    if "num_decoder_layers" in translated:
        translated["pred_rnn_layers"] = translated.pop("num_decoder_layers")
    translated.update(mapped)
    for offset, name in enumerate(CONTROL_TOKEN_FIELDS, start=1):
        translated[name] = blank + offset
    translated.update(
        {
            "architectures": [ARCHITECTURE],
            "num_asr_labels": blank,
            "vocab_size": blank + len(CONTROL_TOKEN_FIELDS) + 1,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "decode_dispatch_arm": "dense-eager",
            "card_num_lookahead_tokens": list(supported)
            if isinstance(supported, (list, tuple))
            else supported,
            "torch_dtype": str(card.get("dtype", card.get("torch_dtype", "float32"))),
        }
    )
    if "prompt_dictionary" in processor:
        translated["prompt_dictionary"] = copy.deepcopy(processor["prompt_dictionary"])
    return translated


def _load_json(name: str, model: str | Path, revision: str | None) -> dict | None:
    loaded = get_hf_file_to_dict(name, model, revision or "main")
    if loaded is not None and not isinstance(loaded, dict):
        raise ValueError(f"{name} of {model} must hold a JSON object")
    return loaded


class NemotronAsrStreamingConfigParser(ConfigParserBase):
    """Parse a Hub card or an already-translated streaming configuration."""

    # @spec REG-CARD-001, REG-CARD-002, REG-CARD-003, REG-CARD-004, REG-CARD-009
    # @spec REG-CARD-011, REG-CARD-012, REG-CARD-013, REG-CARD-014
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, NemotronASRConfig]:
        if callable(kwargs.get("hf_overrides")):
            raise ValueError(
                "--config-format nemotron_asr_streaming supports only the "
                "dictionary form of --hf-overrides; a callable cannot be used"
            )
        config = _load_json("config.json", model, revision)
        if config is None:
            raise OSError(f"config.json not found for {model!r}")
        processor = _load_json("processor_config.json", model, revision)

        if "encoder_config" in config:
            translated = translate_card(config, processor or {})
        elif config.get("architectures") == [ARCHITECTURE]:
            translated = copy.deepcopy(config)
            if processor is not None:
                table = processor.get("prompt_dictionary")
                if "prompt_dictionary" not in translated:
                    if table is not None:
                        translated["prompt_dictionary"] = copy.deepcopy(table)
                elif translated["prompt_dictionary"] != table:
                    raise ValueError(
                        "prompt_dictionary stored in the configuration differs "
                        "from processor_config.json's"
                    )
        else:
            raise ValueError(
                f"{model!r} is neither a Hub card (config.json with "
                f"encoder_config) nor an already-translated configuration "
                f"(architectures ['{ARCHITECTURE}'] and no encoder_config); "
                f"its architectures are {config.get('architectures')!r}"
            )

        errors = schema_errors(translated, processor)
        if errors:
            raise ValueError(
                f"{model!r} is not a servable Nemotron ASR streaming "
                "checkpoint: " + "; ".join(errors)
            )
        return translated, NemotronASRConfig(**copy.deepcopy(translated))
