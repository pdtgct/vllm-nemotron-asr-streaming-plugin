# SPDX-License-Identifier: Apache-2.0
"""Tokenizer mode ``nemotron_asr_streaming``.

The card's tokenizer has a base vocabulary of the label pieces (IDs
``0 … blank − 1``) and may already carry added special tokens at and after
``blank``. Core detokenizes every emitted token, and ``skip_special_tokens``
drops only tokens the tokenizer marks special, so the model's control tokens
(IDs ``blank … blank + 5``: blank placeholder, park, stream chunk carrier,
end of utterance, file carrier, continuation) must all be special in both
senses core's detokenizers use: flagged special in the tokenizer backend and
listed in ``all_special_ids``. The park token (``blank + 1``) is also the
tokenizer's EOS, because a persistent-state stream parks only on the request's
primary EOS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm.tokenizers.hf import CachedHfTokenizer, get_cached_tokenizer
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

#: Names for control tokens the card's tokenizer does not already have, in ID
#: order from ``blank``.
_CONTROL_NAMES: tuple[str, ...] = (
    "<|asr_blank|>",
    "<|asr_park|>",
    "<|asr_stream_chunk|>",
    "<|asr_eou|>",
    "<|asr_file_chunk|>",
    "<|asr_continuation|>",
)

#: Index of the park token among the control tokens (``blank + 1``).
_PARK_OFFSET = 1


def _blank_token_id(path_or_repo: str | Path, revision: str | None) -> int:
    config = get_hf_file_to_dict("config.json", path_or_repo, revision or "main")
    if config is None:
        raise ValueError(
            f"config.json beside the tokenizer of {str(path_or_repo)!r} not found; "
            "it supplies blank_token_id (--tokenizer must name the checkpoint)"
        )
    blank = config.get("blank_token_id")
    if isinstance(blank, bool) or not isinstance(blank, int):
        raise ValueError(
            f"config.json beside the tokenizer of {str(path_or_repo)!r} has no "
            f"integer blank_token_id (got {blank!r})"
        )
    return blank


def _base_vocabulary_size(tokenizer: Any, blank: int) -> int:
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None and hasattr(backend, "get_vocab_size"):
        return backend.get_vocab_size(with_added_tokens=False)
    return sum(1 for token_id in tokenizer.get_vocab().values() if token_id < blank)


def _fresh_name(taken: set[str], name: str) -> str:
    candidate, serial = name, 0
    while candidate in taken:
        serial += 1
        candidate = f"{name[:-2]}_{serial}|>"
    return candidate


# @spec FILE-TOK-001
# @spec REG-TOK-001, REG-TOK-002, REG-TOK-003, REG-TOK-004
def _install_control_tokens(tokenizer: Any, blank: int) -> None:
    """Make IDs ``blank … blank + 5`` special; set the park token as the EOS."""
    last = blank + len(_CONTROL_NAMES) - 1
    base = _base_vocabulary_size(tokenizer, blank)
    if base != blank:
        raise ValueError(
            f"the card tokenizer's base vocabulary has {base} entries; "
            f"blank_token_id {blank} requires exactly {blank}"
        )
    added = tokenizer.added_tokens_decoder
    vocab = tokenizer.get_vocab()
    present: dict[int, str] = {}
    for token, token_id in vocab.items():
        if token_id < blank:
            continue
        info = added.get(token_id)
        if info is None or not getattr(info, "special", False):
            raise ValueError(
                f"the card tokenizer has token {token!r} at ID {token_id} "
                f"(blank_token_id {blank}) that is not a special token; "
                "every token at or above blank must be special"
            )
        if token_id > last:
            raise ValueError(
                f"the card tokenizer has token {token!r} at ID {token_id}, "
                f"above blank + {len(_CONTROL_NAMES) - 1} = {last}"
            )
        present[token_id] = token

    taken = set(vocab)
    names: dict[int, str] = dict(present)
    appended: list[tuple[int, str]] = []
    for offset, name in enumerate(_CONTROL_NAMES):
        token_id = blank + offset
        if token_id not in names:
            fresh = _fresh_name(taken, name)
            taken.add(fresh)
            names[token_id] = fresh
            appended.append((token_id, fresh))

    extras = list(getattr(tokenizer, "extra_special_tokens", None) or [])
    # Append in ID order, so each new token lands at the next ID.
    tokenizer.add_special_tokens(
        {"extra_special_tokens": [*extras, *(name for _, name in appended)]}
    )

    for token_id, name in appended:
        got = tokenizer.convert_tokens_to_ids(name)
        if got != token_id:
            raise ValueError(
                f"appended control token {name!r} got unexpected ID {got}; "
                f"expected {token_id}. The card tokenizer's IDs from "
                f"{blank} to {last} leave a gap that appending cannot fill"
            )
    # List every control token (and the card's previous EOS) in
    # all_special_ids; the park token is the EOS.
    park_name = names[blank + _PARK_OFFSET]
    previous_eos = tokenizer.eos_token
    listed = list(extras)
    for token in [*(names[i] for i in range(blank, last + 1)), previous_eos]:
        if token is not None and token != park_name and token not in listed:
            listed.append(token)
    tokenizer.add_special_tokens(
        {"eos_token": park_name, "extra_special_tokens": listed}
    )
    if tokenizer.eos_token_id != blank + _PARK_OFFSET:
        raise ValueError(
            f"tokenizer EOS is {tokenizer.eos_token_id}; expected the park "
            f"token {blank + _PARK_OFFSET}"
        )


class NemotronAsrStreamingTokenizer(CachedHfTokenizer):
    """The card's tokenizer with the control tokens special and park as EOS."""

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        path_or_repo_id: str | Path,
        *args: Any,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs: Any,
    ) -> Any:
        blank = _blank_token_id(path_or_repo_id, revision)
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
            **kwargs,
        )
        _install_control_tokens(tokenizer, blank)
        return get_cached_tokenizer(tokenizer)
