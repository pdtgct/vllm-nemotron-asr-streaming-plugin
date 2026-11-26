# SPDX-License-Identifier: Apache-2.0
"""Document statements the registration specs require (REG-DOC-001 to
REG-DOC-006, REG-ENG-007, REG-ENG-010). Text checks only: each looks for one
passage (a paragraph or a list item) that carries every element of the
required statement in the document that owns it, or reads the README's serve
command. The README owns the overview, the build definitions, the serve
command and the performance summary; ``docs/manual.md`` owns the serving
detail and the transcription-route caveats; ``docs/evidence.md`` owns the
acceptance claims."""

from __future__ import annotations

import re

import pytest


@pytest.fixture
def readme(repo_root) -> str:
    return (repo_root / "README.md").read_text()


@pytest.fixture
def manual(repo_root) -> str:
    return (repo_root / "docs" / "manual.md").read_text()


@pytest.fixture
def evidence(repo_root) -> str:
    path = repo_root / "docs" / "evidence.md"
    assert path.exists(), "docs/evidence.md does not exist"
    return path.read_text()


def passages(text: str) -> list[str]:
    """Paragraphs and top-level list items, with line breaks folded."""
    blocks = re.split(r"\n\s*\n|\n(?=- )", text)
    return [re.sub(r"\s*\n\s*", " ", b).strip() for b in blocks if b.strip()]


def find(text: str, *needles: str) -> list[str]:
    """Passages that contain every needle (case-insensitive)."""
    return [p for p in passages(text) if all(n.lower() in p.lower() for n in needles)]


def section(text: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"the document has no '{title}' section"
    return match.group(1)


# @spec REG-DOC-001
def test_readme_defines_stage_1(readme):
    """Stage 1 is ``pdtgct/vllm`` at ``cbf13fed4a`` alone: RFC 1's state
    layer with vllm#67583's model-state hooks and the state-group layout
    accessor."""
    assert find(
        readme,
        "stage 1",
        "pdtgct/vllm",
        "cbf13fed4a",
        "alone",
        "state layer",
        "67583",
        "ModelState.state_group_layouts",
    ), (
        "no passage defines stage 1 as pdtgct/vllm at cbf13fed4a alone, with "
        "RFC 1's state layer, vllm#67583's hooks and the state-group layout accessor"
    )


# @spec REG-DOC-001, REG-DOC-003
def test_readme_says_stage_1_carries_neither_stt_fix_so_the_limits_apply(readme):
    assert find(readme, "neither", "67144", "67151", "limits"), (
        "no passage states that stage 1 carries neither vllm#67144 nor #67151 and "
        "that the limits of a core without them therefore apply"
    )


# @spec REG-DOC-001
def test_readme_says_the_session_factory_is_on_neither_build(readme):
    assert find(readme, "session factory", "not available", "either build"), (
        "no passage states that the session factory is not available on either build"
    )
    assert find(readme, "session factory", "routes", "do not run"), (
        "no passage states that a frontend's routes that use the factory do not run"
    )


# @spec REG-DOC-001
def test_readme_names_neither_state_pool_draft_as_part_of_a_build(readme):
    """vllm#67374 and #67375 are the drafts the state layer replaces: no
    passage that defines stage 1 or stage 2 names them."""
    defining = [
        p
        for p in passages(readme)
        if re.search(r"\bstage [12]\b", p, re.I)
        and re.search(r"\bstage [12]\b[^.]*\b(is|adds|plus)\b", p, re.I)
    ]
    assert defining, "no passage defines a build"
    bad = [p for p in defining if "67374" in p or "67375" in p]
    assert not bad, "a build definition names #67374 or #67375:\n" + "\n".join(bad)


# @spec REG-DOC-001
def test_readme_defines_stage_2(readme):
    assert find(readme, "stage 2", "stage 1", "65546", "67294", "frontend"), (
        "no passage defines stage 2 as stage 1 plus RFC 2 (#65546, #67294) and a "
        "frontend endpoint plugin"
    )


CLAIM = re.compile(r"^(achieved|partly achieved|not met)\b", re.I)


def claims(evidence: str) -> list[str]:
    """Every acceptance claim in ``docs/evidence.md``: a passage that opens
    with its result."""
    return [p.lstrip("- ") for p in passages(evidence) if CLAIM.match(p.lstrip("- "))]


def historical_builds(evidence: str) -> list[str]:
    """The builds the evidence's Builds section marks historical: the name
    that opens each list item reading ``<name> (historical...``."""
    names = re.findall(r"^- ([^(\n]+?) \(historical", section(evidence, "Builds"), re.M)
    return [name.strip().lower() for name in names]


def marked_historical(claim: str, evidence: str) -> bool:
    """A claim is marked historical when it says so or names a build the
    Builds section marks historical."""
    text = claim.lower()
    return "historical" in text or any(b in text for b in historical_builds(evidence))


# @spec REG-DOC-002
def test_every_acceptance_claim_names_its_tier_build_and_revisions(evidence):
    found = claims(evidence)
    assert found, "docs/evidence.md has no acceptance claim"
    bad = []
    for claim in found:
        names_build = re.search(r"stage [12]\b", claim, re.I) is not None
        names_revision = re.search(r"\b[0-9a-f]{7,40}\b", claim) is not None
        if not re.search(r"\btier \d", claim, re.I):
            bad.append(f"no tier: {claim}")
        elif not names_revision:
            bad.append(f"no source revision: {claim}")
        elif not (names_build or marked_historical(claim, evidence)):
            bad.append(f"neither a stage 1/2 build nor marked historical: {claim}")
    assert not bad, "\n".join(bad)


# @spec REG-DOC-002
def test_claims_from_earlier_builds_are_marked_historical(evidence):
    """Evidence gathered on an earlier build (the state-pool build, the
    RFC 1 + RFC 2 integration build) is historical: each such claim says so
    or names a build that the Builds section marks historical."""
    assert "state-pool build" in historical_builds(evidence), historical_builds(
        evidence
    )
    for claim in claims(evidence):
        if re.search(r"stage [12]\b", claim, re.I):
            continue
        assert marked_historical(claim, evidence), claim


# @spec REG-DOC-002
def test_readme_states_no_acceptance_claim(readme):
    """Acceptance claims live in docs/evidence.md, not in the README."""
    found = claims(readme)
    assert not found, "the README carries acceptance claims:\n" + "\n".join(found)


FIGURE = re.compile(r"\d[\d,.]*\s*(%|ms\b|s\b|MiB\b|GiB\b)")


# @spec REG-DOC-004
def test_readme_states_expected_performance_and_links_the_evidence(readme):
    body = section(readme, "Expected performance")
    assert "docs/evidence.md" in body, "the section does not link docs/evidence.md"
    assert FIGURE.search(body), "the section states no measured figure"


# @spec REG-DOC-004
def test_every_readme_performance_figure_names_its_condition(readme):
    """Each passage with a measured figure names its tier, build and GPU, or
    links the docs/evidence.md entry that does."""
    bad = []
    for passage in passages(section(readme, "Expected performance")):
        if not FIGURE.search(passage):
            continue
        linked = re.search(r"docs/evidence\.md#[a-z0-9-]+", passage) is not None
        named = all(
            re.search(q, passage, re.I) for q in (r"\btier \d", r"stage [12]\b", r"A40")
        )
        if not (linked or named):
            bad.append(passage)
    assert not bad, "figures without their condition:\n" + "\n".join(bad)


# @spec REG-DOC-005
def test_readme_names_the_complete_audio_work_and_how_this_plugin_differs(readme):
    assert find(readme, "59827", "Sohaib-Ahmed21/vllm-nemotron-asr-plugin"), (
        "no passage names vllm#59827 and the offline plugin's repository"
    )
    assert find(readme, "59827", "complete", "stream"), (
        "no passage says #59827 serves complete audio and this plugin adds streaming"
    )


# @spec REG-DOC-005
def test_readme_lists_the_rfcs_and_their_prs_as_unmerged(readme):
    for number in ("65543", "65545", "9848"):
        assert number in readme, f"RFC issue {number} missing"
    assert "pstate-impl-20261122" in readme, "RFC 1's state-layer implementation"
    for number in ("67583", "65546", "67294", "65549", "9849"):
        assert number in readme, f"PR {number} missing"
    assert find(readme, "RFC", "under review", "merged"), (
        "no passage says the RFCs are under review and nothing is merged"
    )


# @spec REG-DOC-005
def test_readme_says_what_the_frontend_serves(readme):
    assert find(
        readme, "vllm-riva-frontend", "StreamingRecognize", "Recognize", "file"
    ), "no passage says the frontend serves streams and files"


# @spec REG-DOC-003
def test_manual_states_the_unguarded_core_risks_of_vllm_67151(manual):
    assert find(manual, "67151", "beam", "max_model_len", "untrusted"), (
        "no passage states that without vllm#67151 the guards are inactive, beam "
        "search can issue up to n × max_model_len engine requests per upload, "
        "and the route must not be exposed to untrusted clients"
    )
    assert find(manual, "67151", "per beam"), (
        "the beam-search bound does not say one request per beam per pass"
    )


# @spec REG-DOC-003
def test_manual_states_the_partial_text_endings_without_67144_and_67151(manual):
    assert find(manual, "67144", "error", "abort", "partial text", "200"), (
        "no passage states that without vllm#67144 a file ending in error or "
        "abort returns partial text with HTTP 200"
    )
    assert find(manual, "67151", "length", "partial text", "200"), (
        "no passage states that without vllm#67151 a file ending in a "
        "server-caused length returns partial text with HTTP 200"
    )


# @spec REG-DOC-003
def test_manual_limits_state_how_stage_1_realtime_refuses_a_stream(manual):
    """On stage 1, core's ``/v1/realtime`` sends ``session.created`` when it
    accepts the connection, so a stream core cannot admit (``active_limit``
    or ``engine_paused``) is refused only at the first commit, as an
    ``error`` event with ``code`` ``processing_error`` naming the reason."""
    limits = section(manual, "Limits and the support boundary")
    assert find(
        limits,
        "session.created",
        "first commit",
        "error",
        "processing_error",
        "active_limit",
        "engine_paused",
    ), (
        "the Limits section has no passage stating that stage 1's /v1/realtime "
        "refuses a stream it cannot admit (active_limit or engine_paused) only at "
        "the first commit, as an error event with code processing_error naming "
        "the reason"
    )


# @spec REG-DOC-006
def test_manual_builds_section_gives_each_build_with_its_commit(manual):
    builds = section(manual, "Builds and compatibility")
    stage_1 = [p for p in passages(builds) if re.search(r"stage 1", p, re.I)]
    assert any("cbf13fed4a" in p and "state_group_layouts" in p for p in stage_1), (
        "the stage 1 build is not given with its commit"
    )
    rows = [line for line in builds.splitlines() if line.startswith("| Stage 2")]
    assert rows, "the stage 2 build is not given"
    assert "none recorded" in rows[0] or re.search(r"\b[0-9a-f]{10,40}\b", rows[0]), (
        rows[0]
    )
    assert find(builds, "stage 2", "stage 1", "65546", "67294", "frontend"), (
        "the stage 2 build is not defined as stage 1 plus RFC 2 and a frontend"
    )


# @spec REG-DOC-006
def test_manual_builds_section_says_the_session_factory_is_on_neither(manual):
    builds = section(manual, "Builds and compatibility")
    assert find(builds, "session factory", "not available", "target core"), (
        "the builds section does not state that the session factory is not available"
    )


# @spec REG-DOC-006, REG-002
def test_manual_startup_failure_entry_gives_the_core_checks_message(manual):
    builds = section(manual, "Builds and compatibility")
    message = (
        "requires a vLLM build with RFC 1's state layer and vllm#67583's "
        "model-state hooks"
    )
    assert any(
        "register()" in line and message in line for line in builds.splitlines()
    ), "no startup-failure entry gives register()'s refusal with its message"


# --------------------------------------------------------------------------
# Serve command (REG-ENG-007) and the serving guidance in the manual
# (REG-ENG-010)
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
def test_readme_links_the_serving_guidance_beside_its_serve_command(readme):
    assert "docs/manual.md#serving-configurations" in section(readme, "Serve"), (
        "the README's Serve section does not link the manual's serving configurations"
    )


# @spec REG-ENG-010
def test_manual_says_how_num_lookahead_tokens_selects_another_geometry(manual):
    found = find_re(
        manual,
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
def test_manual_says_how_max_model_len_scales_with_the_lookahead(manual):
    found = find_re(
        manual, r"--max-model-len", _number(77000), _number(82513), _number(78773)
    )
    assert found, (
        "no passage states the 600 s bounds: 77,000 for lookahead 3, 6 and 13; "
        "82,513 at 0 and 78,773 at 1"
    )
    passage = found[0]
    assert re.search(r"\b3\b", passage) and re.search(r"\b6\b", passage)
    assert re.search(r"\b13\b", passage), passage


# @spec REG-ENG-010
def test_manual_says_segment_graph_memory_grows_faster_than_linearly(manual):
    assert find_re(
        manual, r"--max-num-seqs", r"faster than linear|super-?linear|more than linear"
    ), (
        "no passage says segment-graph memory grows faster than linearly with "
        "--max-num-seqs"
    )


# @spec REG-ENG-010
def test_manual_names_the_eager_arm_overrides_and_their_runner(manual):
    assert find(
        manual,
        "encoder_execution_arm",
        "eager",
        "decode_dispatch_arm",
        "dense-eager",
        "--enforce-eager",
    ), "no passage names the eager-arm overrides with the --enforce-eager they require"


# @spec REG-ENG-010
def test_manual_says_a_fixed_kv_cache_budget_must_leave_room_for_graphs(manual):
    assert find_re(manual, r"--kv-cache-memory-bytes", r"segment.graph"), (
        "no passage says a fixed --kv-cache-memory-bytes must leave room for the "
        "segment graphs"
    )
