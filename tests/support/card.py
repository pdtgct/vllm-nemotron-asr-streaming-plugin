# SPDX-License-Identifier: Apache-2.0
"""Synthetic checkpoint cards shaped like the published Hub card.

The published card (``nvidia/nemotron-3.5-asr-streaming-0.6b``) has a locale
table with aliases (several names share a prompt row), ``num_prompts`` larger
than the number of rows the table uses, bare-language keys such as ``en``, and
a tokenizer whose base vocabulary ends at ``blank`` followed by the added
special tokens ``<pad>`` (at ``blank``) and ``<blank>`` (at ``blank + 1``).

These builders reproduce that shape with a small ``blank`` so that no test can
depend on the published card's ID values. Nothing here is copied from the
card's files beyond field names.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

#: Label count of the synthetic card (the published card has 13087).
BLANK = 40

#: A locale table with the published card's shapes: bare keys, aliases sharing
#: a row, a language whose locales use several rows (``zh``), a three-letter
#: language (``haw``, ``nah``), a two-letter language outside Whisper's map
#: (``zu``), and ``auto``.
LOCALES: dict[str, int] = {
    "auto": 9,
    "en": 0,
    "en-US": 0,
    "en-GB": 1,
    "enGB": 1,
    "es": 3,
    "es-US": 3,
    "es-ES": 2,
    "de": 4,
    "de-DE": 4,
    "af-ZA": 5,
    "zh-CN": 6,
    "zh-TW": 7,
    "haw-US": 8,
    "nah-MX": 10,
    "zu-ZA": 11,
}

#: Prompt embedding rows; larger than the rows the table uses, as on the card.
NUM_PROMPTS = 16

#: Lookaheads the synthetic card declares, matching the published card.
CARD_LOOKAHEADS = [3, 0, 6, 13]


def card_config(**overrides: Any) -> dict[str, Any]:
    """Return a Hub-card ``config.json`` dictionary."""
    config: dict[str, Any] = {
        "architectures": ["Nemotron3_5AsrForRNNT"],
        "model_type": "nemotron3_5_asr",
        "is_encoder_decoder": True,
        "blank_token_id": BLANK,
        "vocab_size": BLANK + 1,
        "pad_token_id": 0,
        "num_prompts": NUM_PROMPTS,
        "default_prompt_id": LOCALES["auto"],
        "decoder_hidden_size": 64,
        "num_decoder_layers": 1,
        "prompt_intermediate_size": 64,
        "max_symbols_per_step": 10,
        "durations": [],
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "dtype": "float32",
        "encoder_config": {
            "model_type": "parakeet_encoder",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "conv_kernel_size": 9,
            "sliding_window": 57,
            "num_mel_bins": 128,
            "max_position_embeddings": 5000,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 16,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "default_num_lookahead_tokens": 3,
            "supported_num_lookahead_tokens": list(CARD_LOOKAHEADS),
        },
    }
    config.update(copy.deepcopy(overrides))
    return config


def processor_config(**overrides: Any) -> dict[str, Any]:
    """Return a Hub-card ``processor_config.json`` dictionary."""
    config: dict[str, Any] = {
        "processor_class": "Nemotron3_5AsrProcessor",
        "blank_token": "<blank>",
        "num_prompts": NUM_PROMPTS,
        "prompt_dictionary": dict(LOCALES),
        "default_num_lookahead_tokens": 3,
        "supported_num_lookahead_tokens": list(CARD_LOOKAHEADS),
        "feature_extractor": {"sampling_rate": 16000, "feature_size": 128},
    }
    config.update(copy.deepcopy(overrides))
    return config


def generation_config(**overrides: Any) -> dict[str, Any]:
    """Return a ``generation_config.json`` like the card's: no EOS, no limit."""
    config: dict[str, Any] = {
        "_from_model_config": True,
        "decoder_start_token_id": BLANK,
        "pad_token_id": 0,
    }
    config.update(overrides)
    return config


def _tokenizer_files(path: Path, blank: int, added: list[str]) -> None:
    """Write a tiny BPE tokenizer with ``blank`` base tokens.

    ``added`` are special tokens appended after the base vocabulary, so the
    first lands at ``blank`` (the published card adds ``<pad>``, ``<blank>``).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<en-US>": 1}
    letters = "abcdefghijklmnopqrstuvwxyz"
    for ch in letters:
        if len(vocab) >= blank:
            break
        vocab["▁" + ch] = len(vocab)
    while len(vocab) < blank:
        vocab[f"x{len(vocab)}"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.add_special_tokens(["<unk>", "<en-US>"])
    tokenizer.add_special_tokens(added)
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "backend": "tokenizers",
                "unk_token": "<unk>",
                "pad_token": added[0] if added else "<unk>",
                "extra_special_tokens": ["<en-US>"],
                "model_max_length": 10**30,
                "clean_up_tokenization_spaces": False,
            }
        )
    )


def write_card(
    path: Path,
    *,
    config: dict[str, Any] | None = None,
    processor: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    write_generation: bool = True,
    tokenizer_added: list[str] | None = None,
    blank: int = BLANK,
) -> Path:
    """Write a synthetic card directory and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config or card_config()))
    (path / "processor_config.json").write_text(
        json.dumps(processor or processor_config())
    )
    if write_generation:
        (path / "generation_config.json").write_text(
            json.dumps(generation or generation_config())
        )
    _tokenizer_files(
        path,
        blank,
        ["<pad>", "<blank>"] if tokenizer_added is None else tokenizer_added,
    )
    return path
