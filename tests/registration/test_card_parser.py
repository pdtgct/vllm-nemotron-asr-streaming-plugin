# SPDX-License-Identifier: Apache-2.0
"""The card parser and card translation (REG-CARD-001 to REG-CARD-016).

``translate_card`` and ``schema_errors`` are called directly on dictionaries;
file-level behaviour goes through the parser registered under
``--config-format nemotron_asr_streaming`` exactly as core calls it
(``get_config`` in ``vllm/transformers_utils/config.py``: the parser gets the
flat ``hf_overrides`` keyword; core applies them to the result again, and
``try_get_generation_config`` calls it with no overrides at all).
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
from support.card import (
    BLANK,
    CARD_LOOKAHEADS,
    LOCALES,
    NUM_PROMPTS,
    card_config,
    processor_config,
    write_card,
)

pytestmark = pytest.mark.engine

ARCH = "Nemotron3_5AsrStreamingForRNNT"
OTHER_BLANK = 57

#: Keys that exist only inside the card's ``encoder_config`` and must not be
#: carried into the served configuration (REG-CARD-007).
ENCODER_ONLY_KEYS = (
    "encoder_config",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "subsampling_factor",
    "subsampling_conv_channels",
    "subsampling_conv_kernel_size",
    "subsampling_conv_stride",
    "num_hidden_layers",
    "conv_kernel_size",
    "num_mel_bins",
    "sliding_window",
)


def _card(blank: int = BLANK, **encoder_updates):
    config = card_config(blank_token_id=blank, vocab_size=blank + 1)
    config["encoder_config"].update(encoder_updates)
    return config


def _translate(config=None, processor=None):
    from vllm_nemotron_asr_streaming.card_parser import translate_card

    return translate_card(
        copy.deepcopy(config or card_config()),
        copy.deepcopy(processor or processor_config()),
    )


def _schema_errors(config, processor):
    from vllm_nemotron_asr_streaming.card_parser import schema_errors

    return schema_errors(copy.deepcopy(config), copy.deepcopy(processor))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


# @spec REG-CARD-001, REG-CARD-009
def test_local_path_loads_both_files(card_dir, parse_card):
    result, config = parse_card(card_dir)
    assert result["architectures"] == [ARCH]
    assert result["prompt_dictionary"] == LOCALES
    assert config.prompt_dictionary == LOCALES


# @spec REG-CARD-001
def test_hub_repository_is_read_at_the_requested_revision(tmp_path, run_fresh):
    hub = tmp_path / "hub"
    repo = hub / "models--org--card"
    other_processor = processor_config(
        num_prompts=32, prompt_dictionary={**LOCALES, "fr-FR": 20}
    )
    for revision, commit, config, processor in (
        ("rev1", "a" * 40, card_config(), processor_config()),
        ("rev2", "b" * 40, card_config(num_prompts=32), other_processor),
    ):
        snapshot = write_card(
            repo / "snapshots" / commit, config=config, processor=processor
        )
        assert snapshot.is_dir()
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / revision).write_text(commit)
    out = run_fresh(
        """
import vllm_nemotron_asr_streaming as plugin
plugin.register()
from vllm.transformers_utils.config import get_config_parser
parser = get_config_parser("nemotron_asr_streaming")
result = {}
for revision in ("rev1", "rev2"):
    d, _ = parser.parse("org/card", trust_remote_code=False, revision=revision)
    result[revision] = [d["num_prompts"], "fr-FR" in d["prompt_dictionary"]]
emit(**result)
""",
        env={"HF_HUB_CACHE": str(hub), "HF_HUB_OFFLINE": "1"},
    )
    assert out["rev1"] == [16, False]
    assert out["rev2"] == [32, True]


# @spec REG-CARD-002
def test_card_with_encoder_config_is_translated(card_dir, parse_card):
    result, _ = parse_card(card_dir)
    assert result["architectures"] == [ARCH]
    assert result["d_model"] == 32 and result["n_layers"] == 2
    assert "encoder_config" not in result


# @spec REG-CARD-002
def test_streaming_architecture_with_encoder_config_still_translates(
    tmp_path, parse_card
):
    path = write_card(tmp_path / "c", config=card_config(architectures=[ARCH]))
    result, _ = parse_card(path)
    assert result["d_model"] == 32 and "encoder_config" not in result


# @spec REG-CARD-003
def test_already_translated_configuration_is_used_unchanged(
    tmp_path, write_translated, parse_card
):
    path = write_translated(tmp_path / "translated")
    written = json.loads((path / "config.json").read_text())
    assert "encoder_config" not in written
    result, config = parse_card(path)
    assert result == written
    assert config.d_model == written["d_model"]


# @spec REG-CARD-003, REG-LA-001
def test_already_translated_keeps_its_stored_lookahead_keys(
    tmp_path, write_translated, parse_card
):
    def store(translated):
        translated["supported_num_lookahead_tokens"] = [3, 6]
        translated["num_lookahead_tokens"] = 6

    path = write_translated(tmp_path / "translated", store)
    result, _ = parse_card(path)
    assert result["supported_num_lookahead_tokens"] == [3, 6]
    assert result["num_lookahead_tokens"] == 6


# @spec REG-CARD-004
@pytest.mark.parametrize(
    "config",
    [
        card_config(),  # offline architecture key, encoder_config removed below
        card_config(architectures=["SomethingElse"]),
        card_config(architectures=[]),
    ],
    ids=["offline-key-no-encoder", "other-architecture", "no-architecture"],
)
def test_neither_input_names_both_accepted_inputs(tmp_path, parse_card, config):
    config = copy.deepcopy(config)
    config.pop("encoder_config")
    path = write_card(tmp_path / "c", config=config)
    with pytest.raises(ValueError) as excinfo:
        parse_card(path)
    message = str(excinfo.value)
    assert "encoder_config" in message
    assert ARCH in message


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------


# @spec REG-CARD-005
def test_vocab_size_must_be_blank_plus_one(tmp_path, parse_card):
    for vocab in (BLANK, BLANK + 2, BLANK + 6):
        path = write_card(tmp_path / f"v{vocab}", config=card_config(vocab_size=vocab))
        with pytest.raises(ValueError, match="vocab_size"):
            parse_card(path)
    with pytest.raises(ValueError, match="vocab_size"):
        _translate(card_config(vocab_size=BLANK + 2))


# @spec REG-CARD-005
def test_missing_card_lookahead_set_is_named(tmp_path, parse_card):
    config = _card()
    del config["encoder_config"]["supported_num_lookahead_tokens"]
    with pytest.raises(ValueError, match="supported_num_lookahead_tokens"):
        _translate(config)
    path = write_card(tmp_path / "c", config=config)
    with pytest.raises(ValueError, match="supported_num_lookahead_tokens"):
        parse_card(path)


@pytest.mark.parametrize("blank", [BLANK, OTHER_BLANK])
# @spec REG-CARD-006, REG-CARD-014
def test_control_tokens_follow_blank(blank, control_fields):
    result = _translate(_card(blank))
    ids = [result[name] for name in control_fields]
    assert ids == [blank + 1, blank + 2, blank + 3, blank + 4, blank + 5]
    assert result["vocab_size"] == blank + 6
    # The model reads its logit width from the configuration's vocab_size.
    from vllm_nemotron_asr_streaming.nemotron_asr.configuration_nemotron_asr import (
        NemotronASRConfig,
    )

    config = NemotronASRConfig(**result)
    assert config.vocab_size == blank + 6
    assert config.eos_token_id == blank + 1
    assert config.num_asr_labels == blank


@pytest.mark.parametrize("blank", [BLANK, OTHER_BLANK])
# @spec REG-CARD-006, REG-CARD-014
def test_parser_result_follows_blank(blank, tmp_path, parse_card, control_fields):
    path = write_card(
        tmp_path / "c",
        config=_card(blank),
        blank=blank,
    )
    result, config = parse_card(path)
    assert [result[n] for n in control_fields] == [blank + k for k in range(1, 6)]
    assert config.vocab_size == blank + 6
    assert config.num_asr_labels == blank


# @spec REG-CARD-007
def test_encoder_fields_are_mapped():
    config = _card(
        hidden_size=48,
        num_hidden_layers=3,
        conv_kernel_size=5,
        sliding_window=33,
        num_mel_bins=80,
    )
    config["prompt_intermediate_size"] = 96
    config["decoder_hidden_size"] = 24
    config["num_decoder_layers"] = 2
    result = _translate(config)
    assert result["d_model"] == 48
    assert result["n_layers"] == 3
    assert result["conv_kernel"] == 5
    assert result["att_context_left"] == 32
    assert result["n_mels"] == 80
    assert result["pred_hidden"] == 24
    assert result["pred_rnn_layers"] == 2


# @spec REG-CARD-007
def test_decoder_fields_are_optional():
    config = _card()
    del config["decoder_hidden_size"], config["num_decoder_layers"]
    result = _translate(config)
    assert "pred_hidden" not in result and "pred_rnn_layers" not in result


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_hidden_layers",
        "conv_kernel_size",
        "sliding_window",
        "num_mel_bins",
    ],
)
@pytest.mark.parametrize("bad", ["missing", "32", 3.0, True, None])
# @spec REG-CARD-007
def test_encoder_fields_must_be_integers(field, bad):
    config = _card()
    if bad == "missing":
        del config["encoder_config"][field]
    else:
        config["encoder_config"][field] = bad
    with pytest.raises(ValueError, match=field):
        _translate(config)


# @spec REG-CARD-007
def test_prompt_width_must_be_twice_the_hidden_size():
    config = _card()
    config["prompt_intermediate_size"] = 32 * 2 + 1
    with pytest.raises(ValueError, match="prompt_intermediate_size"):
        _translate(config)
    config["prompt_intermediate_size"] = 2 * 32
    _translate(config)
    del config["prompt_intermediate_size"]
    _translate(config)


# @spec REG-CARD-007
def test_no_other_encoder_field_is_carried():
    result = _translate()
    for key in ENCODER_ONLY_KEYS:
        assert key not in result, key
    assert result.get("hidden_size") != 32


# @spec REG-CARD-008
def test_card_facts_recorded_and_card_defaults_dropped():
    config = card_config(
        sliding_window=57,
        supported_num_lookahead_tokens=[1],
        default_num_lookahead_tokens=3,
        num_lookahead_tokens=6,
        default_prompt_id=9,
    )
    result = _translate(config)
    assert result["architectures"] == [ARCH]
    assert sorted(result["card_num_lookahead_tokens"]) == sorted(CARD_LOOKAHEADS)
    assert result["max_position_embeddings"] == 2**24
    assert result["decode_dispatch_arm"] == "dense-eager"
    for key in (
        "model_type",
        "is_encoder_decoder",
        "default_prompt_id",
        "sliding_window",
        "supported_num_lookahead_tokens",
        "default_num_lookahead_tokens",
        "num_lookahead_tokens",
    ):
        assert key not in result, key
    assert result["num_prompts"] == NUM_PROMPTS


# @spec REG-CARD-008
def test_translated_configuration_object_has_no_default_lookahead_or_window(
    card_dir, parse_card
):
    _, config = parse_card(card_dir)
    assert getattr(config, "sliding_window", None) is None
    assert getattr(config, "num_lookahead_tokens", None) is None
    assert getattr(config, "supported_num_lookahead_tokens", None) is None


# @spec REG-CARD-009
def test_locale_table_and_num_prompts_are_kept(card_dir, parse_card):
    result, _ = parse_card(card_dir)
    assert result["prompt_dictionary"] == LOCALES
    assert result["num_prompts"] == NUM_PROMPTS
    # The table uses fewer distinct rows than the embedding table has.
    assert len(set(LOCALES.values())) < NUM_PROMPTS


# @spec REG-CARD-009
def test_stored_table_must_agree_with_the_file(tmp_path, write_translated, parse_card):
    agreeing = write_translated(
        tmp_path / "same", lambda t: t.update(prompt_dictionary=dict(LOCALES))
    )
    parse_card(agreeing)

    def drift(translated):
        translated["prompt_dictionary"] = {**LOCALES, "en": 7}

    differing = write_translated(tmp_path / "diff", drift)
    with pytest.raises(ValueError, match="prompt_dictionary"):
        parse_card(differing)


# @spec REG-CARD-010
def test_computed_values_win_over_same_named_card_keys(control_fields):
    clash = {name: 1 for name in control_fields}
    clash["architectures"] = ["Nemotron3_5AsrForRNNT"]
    result = _translate(card_config(**clash))
    assert [result[n] for n in control_fields] == [BLANK + k for k in range(1, 6)]
    assert result["vocab_size"] == BLANK + 6
    assert result["architectures"] == [ARCH]


# --------------------------------------------------------------------------
# Schema check (both accepted inputs)
# --------------------------------------------------------------------------

#: (field named in the error, mutation of a valid translated configuration)
_BAD_TRANSLATED = {
    "eos_token_id": lambda t, b: t.update(eos_token_id=b + 2),
    "audio_chunk_token_id": lambda t, b: t.update(audio_chunk_token_id=b + 9),
    "eou_token_id": lambda t, b: t.update(eou_token_id=b + 9),
    "file_carrier_token_id": lambda t, b: t.update(file_carrier_token_id=b + 9),
    "continuation_token_id": lambda t, b: t.update(continuation_token_id=b + 9),
    "vocab_size": lambda t, b: t.update(vocab_size=b + 5),
    "num_prompts-zero": lambda t, b: t.update(num_prompts=0),
    "num_prompts-text": lambda t, b: t.update(num_prompts="16"),
    "num_prompts-bool": lambda t, b: t.update(num_prompts=True),
    "num_prompts-differs": lambda t, b: t.update(num_prompts=NUM_PROMPTS + 1),
    "card_num_lookahead_tokens-empty": lambda t, b: t.update(
        card_num_lookahead_tokens=[]
    ),
    "card_num_lookahead_tokens-two": lambda t, b: t.update(
        card_num_lookahead_tokens=[3, 2]
    ),
    "card_num_lookahead_tokens-fourteen": lambda t, b: t.update(
        card_num_lookahead_tokens=[14]
    ),
}

_BAD_TABLES = {
    "empty": {},
    "text-row": {"en": "0"},
    "bool-row": {"en": True},
    "row-at-size": {"en": NUM_PROMPTS},
    "negative-row": {"en": -1},
    "float-row": {"en": 1.0},
}


def _field(case: str) -> str:
    return case.split("-")[0]


@pytest.mark.parametrize("case", list(_BAD_TRANSLATED))
# @spec REG-CARD-011
def test_schema_errors_name_each_bad_field(case):
    translated = _translate()
    _BAD_TRANSLATED[case](translated, BLANK)
    errors = _schema_errors(translated, processor_config())
    assert any(_field(case) in error for error in errors), errors


# @spec REG-CARD-011
def test_schema_errors_accept_the_translation_of_a_valid_card():
    assert _schema_errors(_translate(), processor_config()) == []


# @spec REG-CARD-011
def test_aliases_and_unused_rows_pass_the_schema():
    processor = processor_config(prompt_dictionary={"en": 0, "en-US": 0, "x": 15})
    assert _schema_errors(_translate(processor=processor), processor) == []


@pytest.mark.parametrize("case", list(_BAD_TABLES))
# @spec REG-CARD-011
def test_schema_errors_name_a_bad_locale_table(case):
    processor = processor_config(prompt_dictionary=copy.deepcopy(_BAD_TABLES[case]))
    errors = _schema_errors(_translate(), processor)
    assert any("prompt_dictionary" in error for error in errors), errors


# @spec REG-CARD-011
def test_schema_errors_name_a_missing_locale_table():
    processor = processor_config()
    del processor["prompt_dictionary"]
    errors = _schema_errors(_translate(), processor)
    assert any("prompt_dictionary" in error for error in errors), errors


@pytest.mark.parametrize("case", list(_BAD_TRANSLATED))
# @spec REG-CARD-011
def test_parser_refuses_an_already_translated_schema_violation(
    case, tmp_path, write_translated, parse_card
):
    path = write_translated(tmp_path / "t", lambda t: _BAD_TRANSLATED[case](t, BLANK))
    with pytest.raises(ValueError, match=_field(case)):
        parse_card(path)


@pytest.mark.parametrize("case", list(_BAD_TABLES))
# @spec REG-CARD-011
def test_parser_refuses_a_bad_locale_table_for_either_input(
    case, tmp_path, write_translated, parse_card
):
    processor = processor_config(prompt_dictionary=copy.deepcopy(_BAD_TABLES[case]))
    hub = write_card(tmp_path / "hub", processor=processor)
    with pytest.raises(ValueError, match="prompt_dictionary"):
        parse_card(hub)
    translated = write_translated(tmp_path / "translated")
    (translated / "processor_config.json").write_text(json.dumps(processor))
    with pytest.raises(ValueError, match="prompt_dictionary"):
        parse_card(translated)


@pytest.mark.parametrize("lookaheads", [[], [3, 2], [14], [0, 4]])
# @spec REG-CARD-011
def test_parser_refuses_a_bad_card_lookahead_set(tmp_path, parse_card, lookaheads):
    config = _card(supported_num_lookahead_tokens=lookaheads)
    path = write_card(tmp_path / "c", config=config)
    with pytest.raises(ValueError, match="num_lookahead_tokens"):
        parse_card(path)


# @spec REG-CARD-011
def test_parser_refuses_num_prompts_that_the_processor_does_not_repeat(
    tmp_path, parse_card
):
    path = write_card(
        tmp_path / "c", processor=processor_config(num_prompts=NUM_PROMPTS + 1)
    )
    with pytest.raises(ValueError, match="num_prompts"):
        parse_card(path)


# @spec REG-CARD-011
def test_parser_refuses_a_non_positive_num_prompts(tmp_path, parse_card):
    path = write_card(
        tmp_path / "c",
        config=card_config(num_prompts=0),
        processor=processor_config(num_prompts=0),
    )
    with pytest.raises(ValueError, match="num_prompts"):
        parse_card(path)


# --------------------------------------------------------------------------
# Overrides, generation configuration, result type
# --------------------------------------------------------------------------


# @spec REG-CARD-012
def test_parser_result_does_not_depend_on_dictionary_overrides(card_dir, parse_card):
    baseline = parse_card(card_dir)[0]
    overrides = {
        "num_lookahead_tokens": 6,
        "supported_num_lookahead_tokens": [1],
        "experimental_native_burst": True,
        "vocab_size": 5,
        "eos_token_id": 2,
    }
    assert parse_card(card_dir, hf_overrides=overrides)[0] == baseline
    assert parse_card(card_dir, hf_overrides={})[0] == baseline
    assert parse_card(card_dir, hf_overrides=None)[0] == baseline


# @spec REG-CARD-012
def test_callable_overrides_are_refused_with_the_dictionary_remedy(
    card_dir, parse_card
):
    with pytest.raises(ValueError) as excinfo:
        parse_card(card_dir, hf_overrides=lambda config: config)
    assert "dictionary" in str(excinfo.value).lower()


# @spec REG-CARD-012
def test_core_applies_flat_overrides_after_the_parser(card_dir, registered):
    from vllm.transformers_utils.config import get_config

    # Core passes the flat overrides to the parser and re-applies them to the
    # returned configuration (get_config: config.update(hf_overrides_kw)).
    config = get_config(
        str(card_dir),
        False,
        config_format="nemotron_asr_streaming",
        hf_overrides_kw={"num_lookahead_tokens": 6},
    )
    assert config.num_lookahead_tokens == 6


# @spec REG-CARD-013
def test_parser_without_overrides_yields_a_park_only_generation_config(
    card_dir, parse_card
):
    from transformers import GenerationConfig

    _, config = parse_card(card_dir)
    derived = GenerationConfig.from_model_config(config)
    diff = derived.to_diff_dict()
    assert derived.eos_token_id in (BLANK + 1, [BLANK + 1])
    assert diff.get("max_new_tokens") is None
    assert diff.get("max_length", 20) == 20


# @spec REG-CARD-013
def test_core_derives_the_same_generation_config_for_a_checkpoint_without_one(
    tmp_path, registered
):
    from vllm.transformers_utils.config import try_get_generation_config

    path = write_card(tmp_path / "c", write_generation=False)
    derived = try_get_generation_config(
        str(path), trust_remote_code=False, config_format="nemotron_asr_streaming"
    )
    assert derived is not None
    diff = derived.to_diff_dict()
    assert diff["eos_token_id"] in (BLANK + 1, [BLANK + 1])
    assert diff.get("max_new_tokens") is None


# @spec REG-CARD-014
def test_parser_returns_dictionary_and_model_configuration(card_dir, parse_card):
    from vllm_nemotron_asr_streaming.nemotron_asr.configuration_nemotron_asr import (
        NemotronASRConfig,
    )

    result, config = parse_card(card_dir)
    assert isinstance(result, dict)
    assert isinstance(config, NemotronASRConfig)
    assert config.hidden_size == 17927
    assert config.num_asr_labels == BLANK
    assert config.d_model == result["d_model"]
    assert config.prompt_dictionary == result["prompt_dictionary"]


# --------------------------------------------------------------------------
# One definition, no constants
# --------------------------------------------------------------------------

_RUNTIME_MODULES = (
    "session/*.py",
    "transcription.py",
    "model.py",
    "streaming_config.py",
    "card_parser.py",
    "tokenizer_mode.py",
    "lookahead.py",
    "locales.py",
    "nemotron_asr/file_rows.py",
    "nemotron_asr/model_state_v2.py",
    "nemotron_asr/nemotron_asr.py",
)


def _runtime_sources(repo_root: Path) -> list[Path]:
    package = repo_root / "vllm_nemotron_asr_streaming"
    return sorted({p for pattern in _RUNTIME_MODULES for p in package.glob(pattern)})


def _int_constants(path: Path) -> dict[int, list[int]]:
    found: dict[int, list[int]] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            found.setdefault(node.value, []).append(node.lineno)
    return found


# @spec REG-CARD-016
def test_realtime_max_tokens_has_one_definition(repo_root):
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    assert REALTIME_MAX_TOKENS == 142
    for path in _runtime_sources(repo_root):
        constants = _int_constants(path)
        for literal in (REALTIME_MAX_TOKENS, REALTIME_MAX_TOKENS + 1):
            assert literal not in constants, (
                f"{path.name}:{constants[literal]} repeats the per-unit token bound"
            )


# @spec REG-CARD-016
def test_model_class_attribute_is_the_manifest_value(run_fresh):
    out = run_fresh(
        """
from vllm_nemotron_asr_streaming.nemotron_asr import manifests
from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT as M
emit(model=M.realtime_max_tokens, manifest=manifests.REALTIME_MAX_TOKENS)
"""
    )
    assert out["model"] == out["manifest"] == 142


@pytest.mark.card
# @spec REG-CARD-015
def test_runtime_modules_hold_no_published_card_token_ids(real_card_dir, repo_root):
    blank = json.loads((real_card_dir / "config.json").read_text())["blank_token_id"]
    ids = {blank + offset for offset in range(0, 7)}
    for path in _runtime_sources(repo_root):
        constants = _int_constants(path)
        clash = {i: constants[i] for i in ids if i in constants}
        assert not clash, f"{path.name} hardcodes control token ids {clash}"


@pytest.mark.card
# @spec REG-CARD-006, REG-CARD-007, REG-CARD-008, REG-CARD-009, REG-CARD-014
def test_published_card_translates(real_card_dir, parse_card, control_fields):
    raw = json.loads((real_card_dir / "config.json").read_text())
    processor = json.loads((real_card_dir / "processor_config.json").read_text())
    blank = raw["blank_token_id"]
    result, config = parse_card(real_card_dir)
    assert [result[n] for n in control_fields] == [blank + k for k in range(1, 6)]
    assert result["vocab_size"] == blank + 6
    assert result["architectures"] == [ARCH]
    assert result["prompt_dictionary"] == processor["prompt_dictionary"]
    assert result["num_prompts"] == raw["num_prompts"]
    assert sorted(result["card_num_lookahead_tokens"]) == sorted(
        raw["encoder_config"]["supported_num_lookahead_tokens"]
    )
    assert result["att_context_left"] == raw["encoder_config"]["sliding_window"] - 1
    assert config.hidden_size == 17927
