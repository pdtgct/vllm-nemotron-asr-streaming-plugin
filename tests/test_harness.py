# SPDX-License-Identifier: Apache-2.0
"""The synthetic card has the published card's shape."""

import json

from support.card import BLANK, LOCALES, NUM_PROMPTS


def test_synthetic_card_shape(card_dir):
    from transformers import AutoTokenizer

    config = json.loads((card_dir / "config.json").read_text())
    assert config["vocab_size"] == BLANK + 1
    assert len(set(LOCALES.values())) < len(LOCALES) < NUM_PROMPTS + len(LOCALES)
    tok = AutoTokenizer.from_pretrained(card_dir)
    assert tok.vocab_size == BLANK
    assert tok.convert_ids_to_tokens(BLANK) == "<pad>"
    assert tok.convert_ids_to_tokens(BLANK + 1) == "<blank>"
    # As on the published card: the language tag and <pad> are listed special,
    # <blank> is flagged special in the backend but not listed.
    assert {1, BLANK} <= set(tok.all_special_ids)
    assert BLANK + 1 not in tok.all_special_ids
    assert tok.decode([BLANK + 1], skip_special_tokens=True) == ""
