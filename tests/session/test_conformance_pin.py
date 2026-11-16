# SPDX-License-Identifier: Apache-2.0
"""The counterpart pin behind tier 1 seam conformance (SESS-FAC-007).

The pin is the ``test`` dependency group: two direct references, the frontend
and its ``dev/`` distribution, at one full commit of the frontend repository.
These tests need neither vLLM nor the frontend to import.

* A missing distribution fails, never skips: a run without both does not meet
  tier 1.
* Provenance is checked only in the acceptance lane, with
  ``NEMOTRON_CONFORMANCE_ACCEPTANCE=1``: there both installed distributions
  must record, in ``direct_url.json``, a VCS install whose ``commit_id`` is the
  pinned commit. An editable or local install proves nothing about the pinned
  revision, so outside that lane the provenance test skips, reporting the run
  as not qualifying for tier 1 acceptance.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
REPOSITORY = "git+https://github.com/pdtgct/vllm-riva-frontend.git"
ACCEPTANCE = "NEMOTRON_CONFORMANCE_ACCEPTANCE"
_DIRECT = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*@\s*(\S+)\s*$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
DISTRIBUTIONS = ("vllm-riva-frontend", "vllm-riva-frontend-dev")


def _meta() -> dict:
    return tomllib.loads((PKG / "pyproject.toml").read_text())


def _direct(requirements: list[str]) -> dict[str, str]:
    pins = {}
    for requirement in requirements:
        match = _DIRECT.match(requirement)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def _pins() -> dict[str, str]:
    return _direct(_meta()["dependency-groups"]["test"])


def _commit() -> str:
    url = _pins()["vllm-riva-frontend"]
    assert url.startswith(REPOSITORY + "@"), url
    return url[len(REPOSITORY) + 1 :]


# @spec SESS-FAC-007
def test_the_test_group_pins_both_distributions_at_one_commit():
    pins = _pins()
    assert set(pins) == set(DISTRIBUTIONS), pins
    commit = _commit()
    assert _SHA.match(commit), f"{commit!r} is not a full commit SHA"
    assert pins["vllm-riva-frontend"] == f"{REPOSITORY}@{commit}"
    assert pins["vllm-riva-frontend-dev"] == (f"{REPOSITORY}@{commit}#subdirectory=dev")


# @spec SESS-FAC-007
def test_no_direct_reference_reaches_the_package_metadata():
    project = _meta()["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    assert _direct(requirements) == {}


# @spec SESS-FAC-007
def test_both_distributions_are_installed_as_one_release():
    frontend = metadata.distribution("vllm-riva-frontend")
    dev = metadata.distribution("vllm-riva-frontend-dev")
    assert frontend.version == dev.version
    requires = [r.replace(" ", "") for r in (dev.requires or [])]
    assert f"vllm-riva-frontend=={frontend.version}" in requires, requires


# @spec SESS-FAC-007
def test_the_installed_distributions_come_from_the_pinned_commit():
    commit = _commit()
    sources = {}
    for name in DISTRIBUTIONS:
        direct = json.loads(
            metadata.distribution(name).read_text("direct_url.json") or "{}"
        )
        sources[name] = direct
    if os.environ.get(ACCEPTANCE) != "1":
        pytest.skip(
            f"not an acceptance run ({ACCEPTANCE} unset): installed provenance "
            f"is unverified, so this run does not qualify for tier 1 "
            f"acceptance; sources: {sources}"
        )
    for name, direct in sources.items():
        vcs = direct.get("vcs_info") or {}
        assert vcs.get("vcs") == "git", (name, direct)
        assert vcs.get("commit_id") == commit, (name, direct)
        assert direct.get("url", "").endswith("pdtgct/vllm-riva-frontend.git"), (
            name,
            direct,
        )
        if name == "vllm-riva-frontend-dev":
            assert direct.get("subdirectory") == "dev", (name, direct)
