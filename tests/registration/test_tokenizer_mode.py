# SPDX-License-Identifier: Apache-2.0
"""The ``nemotron_asr_streaming`` tokenizer mode (REG-TOK-001 to REG-TOK-004).

Special has two senses in core's detokenizers
(``vllm/tokenizers/detokenizer_utils.py``, ``vllm/v1/engine/detokenizer.py``):
the flag in the tokenizer backend (the fast incremental path, observed here as
``decode(..., skip_special_tokens=True)`` dropping the token) and membership in
``all_special_ids`` / ``all_special_tokens`` (the slow path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.card import BLANK, write_card

pytestmark = pytest.mark.engine

MODE = "nemotron_asr_streaming"

#: Card tokenizer variants: tokens the card already has at ``blank`` onwards.
VARIANTS = {
    "published-shape": ["<pad>", "<blank>"],
    "no-added-tokens": [],
    "pad-only": ["<pad>"],
    "all-six-present": [f"<card{i}>" for i in range(6)],
}


def _load(path: Path, **extra):
    from vllm.tokenizers.registry import TokenizerRegistry, resolve_tokenizer_args

    if MODE not in TokenizerRegistry.tokenizers:
        pytest.fail(f"tokenizer mode {MODE!r} is not registered")

    mode, name, args, kwargs = resolve_tokenizer_args(
        str(path), tokenizer_mode=MODE, **extra
    )
    assert mode == MODE
    return TokenizerRegistry.load_tokenizer(mode, name, *args, **kwargs)


def _assert_special_both_senses(tok, ids):
    for token_id in ids:
        token = tok.convert_ids_to_tokens(token_id)
        assert tok.decode([token_id], skip_special_tokens=True) == "", (
            f"id {token_id} ({token!r}) not flagged special in the backend"
        )
        assert token_id in tok.all_special_ids, f"id {token_id} not in all_special_ids"
        assert token in tok.all_special_tokens


@pytest.fixture
def registered_tokenizer_mode(registered):
    return registered


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
# @spec REG-TOK-001
def test_ids_blank_to_blank_plus_five_are_special_in_both_senses(
    registered_tokenizer_mode, tmp_path, variant
):
    path = write_card(tmp_path / "c", tokenizer_added=VARIANTS[variant])
    tok = _load(path)
    _assert_special_both_senses(tok, range(BLANK, BLANK + 6))
    assert len(tok) == BLANK + 6
    names = [tok.convert_ids_to_tokens(i) for i in range(BLANK, BLANK + 6)]
    assert len(set(names)) == 6


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
# @spec REG-TOK-001
def test_card_tokens_are_kept_and_base_vocabulary_is_unchanged(
    registered_tokenizer_mode, tmp_path, variant
):
    from transformers import AutoTokenizer

    path = write_card(tmp_path / "c", tokenizer_added=VARIANTS[variant])
    raw = AutoTokenizer.from_pretrained(path)
    tok = _load(path)
    for token_id in range(len(raw)):
        assert tok.convert_ids_to_tokens(token_id) == raw.convert_ids_to_tokens(
            token_id
        )
    # Ordinary pieces are not special and still decode.
    piece = tok.convert_tokens_to_ids("▁a")
    assert piece < BLANK and piece not in tok.all_special_ids
    assert tok.decode([piece]) == raw.decode([piece]) != ""
    # Control tokens vanish from text and do not disturb the surrounding pieces.
    other = tok.convert_tokens_to_ids("▁b")
    plain = tok.decode([piece, other], skip_special_tokens=True)
    with_control = tok.decode(
        [piece, BLANK + 2, BLANK + 3, other], skip_special_tokens=True
    )
    assert with_control == plain


# @spec REG-TOK-001
def test_published_shaped_card_tokens_keep_their_names(
    registered_tokenizer_mode, card_dir
):
    tok = _load(card_dir)
    assert tok.convert_ids_to_tokens(BLANK) == "<pad>"
    assert tok.convert_ids_to_tokens(BLANK + 1) == "<blank>"
    # The card flags <blank> special in the backend but does not list it.
    assert BLANK + 1 in tok.all_special_ids


@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
# @spec REG-TOK-002
def test_park_token_is_the_only_eos(registered_tokenizer_mode, tmp_path, variant):
    path = write_card(tmp_path / "c", tokenizer_added=VARIANTS[variant])
    tok = _load(path)
    assert tok.eos_token_id == BLANK + 1
    assert tok.convert_ids_to_tokens(tok.eos_token_id) == tok.convert_ids_to_tokens(
        BLANK + 1
    )


@pytest.mark.parametrize("foreign", ["<en-US>", "<unk>", "<pad>"])
# @spec REG-TOK-002
def test_a_foreign_eos_in_the_card_is_replaced(
    registered_tokenizer_mode, tmp_path, foreign
):
    path = write_card(tmp_path / "c")
    config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["eos_token"] = foreign
    config_path.write_text(json.dumps(config))
    tok = _load(path)
    assert tok.eos_token_id == BLANK + 1
    assert tok.eos_token_id != tok.convert_tokens_to_ids(foreign)


# @spec REG-TOK-002
def test_model_config_tokenizer_mode_loads_through_core(make_model_config, card_dir):
    from vllm.tokenizers import cached_tokenizer_from_config

    model_config = make_model_config(card_dir, tokenizer_mode=MODE)
    tok = cached_tokenizer_from_config(model_config)
    assert tok.eos_token_id == BLANK + 1
    _assert_special_both_senses(tok, range(BLANK, BLANK + 6))
    assert tok.eos_token_id == model_config.hf_config.eos_token_id


# @spec REG-TOK-003
def test_an_appended_token_at_an_unexpected_id_fails_the_load(
    registered_tokenizer_mode, tmp_path
):
    # Everything REG-TOK-004 checks holds: `<en-US>` stays at id 1, the base
    # vocabulary has `blank` entries (the last piece is swapped for a new
    # special `<gap>` at blank + 3, so vocab_size == blank), and every token at
    # or above `blank` is special and at most `blank + 5` (`<pad>` at blank,
    # `<blank>` at blank + 1, `<gap>` at blank + 3). ID blank + 2 is a gap, so
    # appending the missing controls puts the first at a new maximum id, not
    # at blank + 2. The error must contain the fragment "unexpected ID".
    path = write_card(tmp_path / "c")
    document = json.loads((path / "tokenizer.json").read_text())
    vocab = document["model"]["vocab"]
    del vocab[max(vocab, key=vocab.get)]
    vocab["<gap>"] = BLANK + 3
    document["added_tokens"].append(
        {
            "id": BLANK + 3,
            "content": "<gap>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    (path / "tokenizer.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unexpected ID"):
        _load(path)


# @spec REG-TOK-004
def test_blank_is_taken_from_the_config_beside_the_tokenizer(
    registered_tokenizer_mode, tmp_path
):
    path = write_card(tmp_path / "c")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    del config["blank_token_id"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="blank_token_id"):
        _load(path)
    bare = write_card(tmp_path / "bare" / "c")
    (bare / "config.json").unlink()
    with pytest.raises(Exception):
        _load(bare)


# @spec REG-TOK-004
def test_base_vocabulary_must_equal_blank(registered_tokenizer_mode, tmp_path):
    for wrong in (BLANK - 1, BLANK + 3):
        path = write_card(tmp_path / f"c{wrong}")
        config_path = path / "config.json"
        config = json.loads(config_path.read_text())
        config["blank_token_id"] = wrong
        config["vocab_size"] = wrong + 1
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError):
            _load(path)


@pytest.mark.parametrize("added", [[], ["<pad>"], ["<pad>", "<blank>"]])
# @spec REG-TOK-004
def test_a_card_token_at_blank_or_more_must_be_special(
    registered_tokenizer_mode, tmp_path, added
):
    from tokenizers import AddedToken, Tokenizer

    path = write_card(tmp_path / "c", tokenizer_added=added)
    backend = Tokenizer.from_file(str(path / "tokenizer.json"))
    backend.add_tokens([AddedToken("plain-piece", special=False)])
    backend.save(str(path / "tokenizer.json"))
    with pytest.raises(ValueError):
        _load(path)


# @spec REG-TOK-004
def test_a_card_token_above_blank_plus_five_fails_the_load(
    registered_tokenizer_mode, tmp_path
):
    path = write_card(tmp_path / "c", tokenizer_added=[f"<card{i}>" for i in range(7)])
    with pytest.raises(ValueError):
        _load(path)


@pytest.mark.card
@pytest.mark.parametrize("check", ["special-both-senses", "eos", "base-vocabulary"])
# @spec REG-TOK-001, REG-TOK-002, REG-TOK-004
def test_published_card_tokenizer(registered_tokenizer_mode, real_card_dir, check):
    from transformers import AutoTokenizer

    blank = json.loads((real_card_dir / "config.json").read_text())["blank_token_id"]
    tok = _load(real_card_dir)
    if check == "special-both-senses":
        _assert_special_both_senses(tok, range(blank, blank + 6))
        assert len(tok) == blank + 6
    elif check == "eos":
        assert tok.eos_token_id == blank + 1
    else:
        raw = AutoTokenizer.from_pretrained(real_card_dir)
        for token_id in range(blank):
            assert tok.convert_ids_to_tokens(token_id) == raw.convert_ids_to_tokens(
                token_id
            )
