# SPDX-License-Identifier: Apache-2.0
"""README statements the registration specs require (REG-DOC-001 to
REG-DOC-003). Text checks only: each looks for one passage (a paragraph or a
list item) that carries every element of the required statement."""

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
    assert find(readme, "stage 1", "main", "67374", "67375", "67144", "67151"), (
        "no passage defines stage 1 as upstream main with vllm#67374, #67375, "
        "#67144 and #67151"
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
