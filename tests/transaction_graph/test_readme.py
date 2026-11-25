# SPDX-License-Identifier: Apache-2.0
"""The transaction-graph sections of the user manual and the evidence document
(TXG-DOC-001 to TXG-DOC-003). Text checks only: each section is the first
section of ``docs/manual.md`` or ``docs/evidence.md`` whose heading names
transaction graphs. The manual owns the serve command and the limits; the
evidence document owns the measured figures."""

from __future__ import annotations

import json
import re
import shlex

import pytest

KEY = "transaction_graph_populations"


def transaction_graph_section(text: str, name: str) -> str:
    match = re.search(
        r"^(#{2,3}) [^\n]*transaction graph[^\n]*\n(.*?)(?=^#{1,3} |\Z)",
        text,
        re.S | re.M | re.I,
    )
    assert match, f"{name} has no transaction-graph section"
    return match.group(2)


def _section(repo_root, relative: str):
    """The section text, looked up inside each test so that a missing document
    or section fails the test rather than erroring its setup."""

    def read() -> str:
        path = repo_root / relative
        assert path.exists(), f"{relative} does not exist"
        return transaction_graph_section(path.read_text(), relative)

    return read


@pytest.fixture
def section(repo_root):
    """The user manual's transaction-graph section."""
    return _section(repo_root, "docs/manual.md")


@pytest.fixture
def evidence_section(repo_root):
    """The evidence document's transaction-graph section."""
    return _section(repo_root, "docs/evidence.md")


def passages(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n|\n(?=- )", text)
    return [re.sub(r"\s*\n\s*", " ", b).strip() for b in blocks if b.strip()]


def find(text: str, *needles: str) -> list[str]:
    return [p for p in passages(text) if all(n.lower() in p.lower() for n in needles)]


def serve_commands(text: str) -> list[list[str]]:
    commands = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            if "vllm serve" in line:
                commands.append(shlex.split(line))
    return commands


def flag(argv: list[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def the_command(section: str) -> list[str]:
    commands = [c for c in serve_commands(section) if any(KEY in a for a in c)]
    assert len(commands) == 1, "the section has no single serve command with the option"
    return commands[0]


def admission_limit(argv: list[str]) -> int:
    """``--max-num-active-seqs`` when the command sets it, otherwise
    ``--max-num-seqs``."""
    active = flag(argv, "--max-num-active-seqs")
    seqs = flag(argv, "--max-num-seqs")
    assert seqs is not None, "the command sets no --max-num-seqs"
    return int(active if active is not None else seqs)


# @spec TXG-DOC-001
def test_serve_command_runs_the_restricted_startup(section):
    """``-cc.cudagraph_mode none -cc.mode none --no-async-scheduling`` and a
    ``--num-gpu-blocks-override`` of at least the admission limit plus one
    (TXG-CFG-005, TXG-CFG-017 to TXG-CFG-019)."""
    argv = the_command(section())
    assert (flag(argv, "-cc.cudagraph_mode") or "").lower() == "none", argv
    assert (flag(argv, "-cc.mode") or "").lower() == "none", argv
    assert "--no-async-scheduling" in argv, argv
    blocks = flag(argv, "--num-gpu-blocks-override")
    assert blocks is not None, "the command sets no --num-gpu-blocks-override"
    assert int(blocks) >= admission_limit(argv) + 1, (blocks, argv)


# @spec TXG-DOC-001
def test_serve_command_runs_the_hook_configuration(section):
    section = section()
    argv = the_command(section)
    assert "--enforce-eager" not in argv
    overrides = json.loads(flag(argv, "--hf-overrides"))
    assert overrides.get("num_lookahead_tokens") in (0, 1)
    populations = overrides[KEY]
    assert isinstance(populations, list) and populations
    for key, default in (
        ("encoder_execution_arm", "eager-graphed"),
        ("decode_dispatch_arm", "dense-graphed"),
    ):
        assert overrides.get(key, default) == default, key
    assert not overrides.get("chunk_bucket_graph_populations")
    assert flag(argv, "--max-model-len") is not None
    max_num_seqs = flag(argv, "--max-num-seqs")
    assert max_num_seqs is not None
    assert max(populations) <= int(max_num_seqs)
    assert max(populations) <= admission_limit(argv)


# @spec TXG-DOC-001
def test_serve_command_is_the_one_tier_5_verified(section):
    """The command's ``--max-num-seqs``, block count and ``N`` are the ones the
    tier 5 run verified to fit the A40."""
    section = section()
    argv = the_command(section)
    overrides = json.loads(flag(argv, "--hf-overrides"))
    n = str(max(overrides[KEY]))
    seqs = flag(argv, "--max-num-seqs")
    blocks = flag(argv, "--num-gpu-blocks-override")
    assert find(section, "A40", "tier 5", n), "no passage ties N to the tier 5 A40 run"
    assert find(section, "A40", seqs), "no passage ties --max-num-seqs to the A40 run"
    assert blocks is not None and find(section, "A40", "tier 5", blocks), (
        "no passage ties the block count to the tier 5 A40 run"
    )


# @spec TXG-DOC-002
def test_section_states_measured_memory_and_capture_time(evidence_section):
    section = evidence_section()
    assert find(section, "graph memory", "A40"), "no graph-memory figure on the A40"
    assert re.search(r"\d+(\.\d+)?\s*(MiB|GiB|MB|GB)", section), "no memory figure"
    assert find(section, "transient peak"), "no transient peak"
    assert re.search(r"capture time[^.]*\d+(\.\d+)?\s*s", section, re.I), (
        "no capture time"
    )
    assert find(section, "build"), "no build named with the figures"
    assert re.search(r"\b(80|160)\s*ms\b", section), (
        "no geometry named with the figures"
    )
    assert re.search(r"\bN\s*=\s*\d+", section), "no N named with the figures"


# @spec TXG-DOC-002
def test_section_states_the_histogram_n_was_chosen_from(evidence_section):
    section = evidence_section()
    assert find(section, "(K, E)", "histogram"), "no (K, E) histogram"


# @spec TXG-DOC-003
def test_section_states_the_fixed_budget_rule(section):
    section = section()
    assert find(section, "--kv-cache-memory-bytes", "room", "startup log"), (
        "no passage says a fixed budget must leave room for the graphs, which the "
        "startup log reports"
    )


# @spec TXG-DOC-003
@pytest.mark.parametrize(
    "needles",
    [
        ("one geometry", "80", "160"),
        ("`N`", "bounded", "max_num_seqs", "--max-num-active-seqs"),
        ("block count", "admission limit plus one"),
        ("`chunk-uncovered`", "without transaction graphs", "counted"),
        ("`echo-uncovered`", "without transaction graphs", "counted"),
        ("multi-token", "without transaction graphs", "counted"),
        ("--enforce-eager", "refused"),
        ("native burst", "refused"),
        ("restricted startup", "refused"),
        ("frontend", "locale only"),
    ],
    ids=[
        "geometry",
        "bounded-n",
        "block-count",
        "beyond-n-chunk",
        "beyond-n-echo",
        "multi-token",
        "enforce-eager",
        "native-burst",
        "restricted-startup",
        "frontend",
    ],
)
def test_section_states_the_limits(section, needles):
    section = section()
    assert find(section, *needles), f"no passage states {needles}"


def table_rows(text: str) -> list[list[str]]:
    """The cells of every Markdown table row, separator rows excluded."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


# @spec TXG-DOC-003
@pytest.mark.parametrize(
    "setting",
    [
        "--no-async-scheduling",
        "-cc.cudagraph_mode none",
        "-cc.mode none",
        "--num-gpu-blocks-override",
        "--kv-cache-memory-bytes",
    ],
)
def test_section_names_each_restricted_startup_setting_with_its_reason(
    section, setting
):
    """The restricted startup's settings (TXG-CFG-017 to TXG-CFG-019), each
    with the reason it is required."""
    section = section()
    rows = [r for r in table_rows(section) if setting in r[0]]
    assert rows, f"no row names {setting}"
    assert all(len(r) > 1 and len(r[1].split()) >= 5 for r in rows), rows


# @spec TXG-DOC-003
def test_each_claim_names_its_tier_and_build(section):
    section = section()
    claims = [p for p in passages(section) if re.search(r"\btier [1-5]\b", p, re.I)]
    assert claims, "no claim names its acceptance tier"
    for claim in claims:
        assert re.search(r"\bbuild\b|stage [12]", claim, re.I), claim
