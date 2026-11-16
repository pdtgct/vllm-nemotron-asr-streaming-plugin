# SPDX-License-Identifier: Apache-2.0
"""README statements the registration specs require (REG-DOC-001 to
REG-DOC-003, REG-ENG-007, REG-ENG-010). Text checks only: each looks for one
passage (a paragraph or a list item) that carries every element of the
required statement, or reads the README's serve command."""

from __future__ import annotations

import re

import pytest


@pytest.fixture
def readme(repo_root) -> str:
    return (repo_root / "README.md").read_text()


def passages(text: str) -> list[str]:
    """Paragraphs and top-level list items, with line breaks folded."""
    blocks = re.split(r"\n\s*\n|\n(?=- )", text)
    return [re.sub(r"\s*\n\s*", " ", b).strip() for b in blocks if b.strip()]


def find(text: str, *needles: str) -> list[str]:
    """Passages that contain every needle (case-insensitive)."""
    return [p for p in passages(text) if all(n.lower() in p.lower() for n in needles)]


def section(text: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"README has no '{title}' section"
    return match.group(1)


# @spec REG-DOC-001
def test_readme_defines_stage_1(readme):
    assert find(
        readme, "stage 1", "main", "67374", "67375", "67583", "67144", "67151"
    ), (
        "no passage defines stage 1 as upstream main with vllm#67374, #67375, "
        "#67583, #67144 and #67151"
    )


# @spec REG-DOC-001
def test_readme_defines_stage_2(readme):
    assert find(readme, "stage 2", "stage 1", "65546", "67294", "frontend"), (
        "no passage defines stage 2 as stage 1 plus RFC 2 (#65546, #67294) and a "
        "frontend endpoint plugin"
    )


CLAIM = re.compile(r"^(achieved|partly achieved|not met)\b", re.I)


def claims(readme: str) -> list[str]:
    evidence = section(readme, "Evidence")
    return [p.lstrip("- ") for p in passages(evidence) if CLAIM.match(p.lstrip("- "))]


# @spec REG-DOC-002
def test_every_acceptance_claim_names_its_tier_build_and_revisions(readme):
    found = claims(readme)
    assert found, "the Evidence section has no acceptance claim"
    bad = []
    for claim in found:
        historical = "historical" in claim.lower()
        names_build = re.search(r"stage [12]\b", claim, re.I) is not None
        names_revision = re.search(r"\b[0-9a-f]{7,40}\b", claim) is not None
        if not re.search(r"\btier \d", claim, re.I):
            bad.append(f"no tier: {claim}")
        elif not names_revision:
            bad.append(f"no source revision: {claim}")
        elif not (names_build or historical):
            bad.append(f"neither a stage 1/2 build nor marked historical: {claim}")
    assert not bad, "\n".join(bad)


# @spec REG-DOC-002
def test_claims_from_the_superseded_integration_build_are_marked_historical(readme):
    """Evidence gathered on the earlier RFC 1 + RFC 2 integration build is
    historical; it is not a claim about stage 1 or stage 2."""
    for claim in claims(readme):
        if re.search(r"stage [12]\b", claim, re.I):
            continue
        assert "historical" in claim.lower(), claim


# @spec REG-DOC-003
def test_readme_states_the_unguarded_core_risks_of_vllm_67151(readme):
    assert find(readme, "67151", "beam", "max_model_len", "untrusted"), (
        "no passage states that without vllm#67151 the guards are inactive, beam "
        "search can issue up to n × max_model_len engine requests per upload, "
        "and the route must not be exposed to untrusted clients"
    )
    assert find(readme, "67151", "per beam"), (
        "the beam-search bound does not say one request per beam per pass"
    )


# @spec REG-DOC-003
def test_readme_states_the_partial_text_endings_without_67144_and_67151(readme):
    assert find(readme, "67144", "error", "abort", "partial text", "200"), (
        "no passage states that without vllm#67144 a file ending in error or "
        "abort returns partial text with HTTP 200"
    )
    assert find(readme, "67151", "length", "partial text", "200"), (
        "no passage states that without vllm#67151 a file ending in a "
        "server-caused length returns partial text with HTTP 200"
    )


# --------------------------------------------------------------------------
# Serve command (REG-ENG-007) and the guidance beside it (REG-ENG-010)
# --------------------------------------------------------------------------


def serve_command(readme: str) -> str:
    """The first fenced block that runs ``vllm serve``, continuation lines
    joined."""
    blocks = re.findall(r"```[a-z]*\n(.*?)```", readme, flags=re.S)
    commands = [b for b in blocks if "vllm serve" in b]
    assert commands, "README has no `vllm serve` command"
    return re.sub(r"\\\s*\n\s*", " ", commands[0])


def flag_value(command: str, flag: str) -> str | None:
    match = re.search(rf"{re.escape(flag)}[ =](\S+)", command)
    return match.group(1) if match else None


# @spec REG-ENG-007
def test_readme_serve_command_shows_every_required_flag(readme):
    command = serve_command(readme)
    for flag in (
        "--dtype float32",
        "--config-format nemotron_asr_streaming",
        "--tokenizer-mode nemotron_asr_streaming",
    ):
        assert flag in command, f"{flag} missing from: {command}"
    assert flag_value(command, "--max-model-len") == "77000", command
    max_num_seqs = flag_value(command, "--max-num-seqs")
    assert max_num_seqs is not None and max_num_seqs.isdigit(), command


# @spec REG-ENG-007
def test_readme_serve_command_uses_exactly_one_runner_configuration(readme):
    """``--enforce-eager``, or the hook configuration's flags; not both and not
    neither. Until the hook configuration meets tier 2, the design keeps
    ``--enforce-eager``."""
    command = serve_command(readme)
    eager = "--enforce-eager" in command
    hook = "-cc.cudagraph_mode none" in command or "cudagraph_mode" in command
    hook = hook and ("-cc.mode none" in command or "-O0" in command)
    assert eager != hook, command


# @spec REG-ENG-007
def test_readme_serve_command_serves_the_defaults(readme):
    """No geometry, no served set and no segment-arm override: the card's
    default geometry (320 ms) and the default arms."""
    command = serve_command(readme)
    for key in (
        "num_lookahead_tokens",
        "supported_num_lookahead_tokens",
        "encoder_execution_arm",
        "decode_dispatch_arm",
        "chunk_bucket_graph_populations",
    ):
        assert key not in command, f"{key} in: {command}"


# @spec REG-ENG-007
def test_readme_serve_command_turns_on_no_refused_setting(readme):
    """``--no-async-scheduling`` and ``--no-enable-prefix-caching`` may be
    omitted (core resolves both off for this model); nothing turns on a
    setting REG-ENG-001 to REG-ENG-005 refuses."""
    command = serve_command(readme)
    for refused in (
        " --async-scheduling",
        " --enable-prefix-caching",
        "--speculative-config",
        "--kv-transfer-config",
        "--enable-dbo",
        "--enable-sleep-mode",
        "--fault-tolerance-config",
        "--enable-fault-tolerance",
        "--enable-elastic-ep",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
    ):
        assert refused not in command, f"{refused} in: {command}"


# @spec REG-ENG-007
def test_readme_marks_an_unverified_max_num_seqs_as_interim(readme):
    """The ``--max-num-seqs`` value is either one tier 2 verified (a passage
    naming tier 2 and the value) or the interim value of the unpublished A40
    measurement, marked so."""
    value = flag_value(serve_command(readme), "--max-num-seqs")
    assert value is not None
    verified = find(readme, "--max-num-seqs", "tier 2", value)
    interim = find(readme, "--max-num-seqs", "interim", "unpublished", value)
    assert verified or interim, (
        f"no passage says whether --max-num-seqs {value} met tier 2 or is interim "
        "and unpublished"
    )


def _number(value: int) -> str:
    """A regex matching ``value`` with or without thousands separators."""
    text = str(value)
    if len(text) <= 3:
        return rf"\b{text}\b"
    return rf"\b{text[:-3]}[,  ]?{text[-3:]}\b"


def find_re(text: str, *patterns: str) -> list[str]:
    return [p for p in passages(text) if all(re.search(q, p, re.I) for q in patterns)]


# @spec REG-ENG-010
def test_readme_says_how_num_lookahead_tokens_selects_another_geometry(readme):
    found = find_re(
        readme,
        r"num_lookahead_tokens",
        r"--hf-overrides",
        r"\b0\b",
        r"\b1\b",
        r"\b6\b",
        r"\b13\b",
    )
    assert found, (
        "no passage says that num_lookahead_tokens in --hf-overrides selects "
        "0, 1, 6 or 13"
    )


# @spec REG-ENG-010
def test_readme_says_how_max_model_len_scales_with_the_lookahead(readme):
    found = find_re(
        readme, r"--max-model-len", _number(77000), _number(82513), _number(78773)
    )
    assert found, (
        "no passage states the 600 s bounds: 77,000 for lookahead 3, 6 and 13; "
        "82,513 at 0 and 78,773 at 1"
    )
    passage = found[0]
    assert re.search(r"\b3\b", passage) and re.search(r"\b6\b", passage)
    assert re.search(r"\b13\b", passage), passage


# @spec REG-ENG-010
def test_readme_says_segment_graph_memory_grows_faster_than_linearly(readme):
    assert find_re(
        readme, r"--max-num-seqs", r"faster than linear|super-?linear|more than linear"
    ), (
        "no passage says segment-graph memory grows faster than linearly with "
        "--max-num-seqs"
    )


# @spec REG-ENG-010
def test_readme_names_the_eager_arm_overrides_and_their_runner(readme):
    assert find(
        readme,
        "encoder_execution_arm",
        "eager",
        "decode_dispatch_arm",
        "dense-eager",
        "--enforce-eager",
    ), "no passage names the eager-arm overrides with the --enforce-eager they require"


# @spec REG-ENG-010
def test_readme_says_a_fixed_kv_cache_budget_must_leave_room_for_graphs(readme):
    assert find_re(readme, r"--kv-cache-memory-bytes", r"segment.graph"), (
        "no passage says a fixed --kv-cache-memory-bytes must leave room for the "
        "segment graphs"
    )
