# SPDX-License-Identifier: Apache-2.0
"""The TLA+ system models of the transaction-graph capture lifecycle and the
split step (TXG-MODEL-001 to TXG-MODEL-005, TXG-MODEL-007 to TXG-MODEL-016).

Structural checks run on CPU. The TLC check runs when ``TLA2TOOLS_JAR`` names
a ``tla2tools.jar`` and ``java`` is on the path, and is skipped otherwise.
Each model's header (its leading comment block) explains every invariant and
property its configuration checks, and cites the spec ID it checks, so the
set of models can be traced to TXG-MODEL-003, TXG-MODEL-004 and
TXG-MODEL-007 to TXG-MODEL-016.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INVARIANT_SPECS = ["TXG-MODEL-003"] + [f"TXG-MODEL-{i:03d}" for i in range(7, 17)]
LIVENESS_SPEC = "TXG-MODEL-004"

#: Actions TXG-MODEL-002 requires, as identifier stems (case and underscores
#: ignored).
ACTION_STEMS = [
    # Core binds the profiling and the serving KV cache's views, setting the
    # state-group layout at each binding, and clears it when it releases the
    # profiling KV cache after clear_cudagraphs; a binding after the serving
    # capture with views that fit and with views that do not (TXG-LIFE-009).
    "bindkvcache",
    "stategrouplayout",
    "releaseprofilingkvcache",
    # The pool-access check of TXG-CAP-008, able to fail.
    "poolaccess",
    # The KV-cache sizing, a nondeterministic choice (its fixed settings).
    "blockcount",
    "budget",
    "prepareexecutionmemory",
    "capturecudagraphs",
    "clearcudagraphs",
    "postwarmup",
    "shutdown",
    "split",
    "fallback",
    "bypass",
    "cleanup",
    "dummy",
    "latch",
    "profil",
    # A second serving capture after a successful and after a failed one.
    "secondservingcapture",
    "retry",
    # A failing cancel of the held reservation before the first launch.
    "prelaunchcancel",
    # A stage that raises before and one that raises after publication.
    "stageraises",
    "publish",
    # A declined step's option-off reservation, scatter and stage failures,
    # and its failure path's cancel that itself raises (TXG-TXN-008).
    "optionofffailures",
    "optionoffcancelraises",
    # A step before post-warmup verification, after which the engine may die.
    "premature",
    "enginedies",
]


@pytest.fixture
def models_dir(repo_root) -> Path:
    return repo_root / "models" / "transaction-graph"


def specs(models_dir: Path) -> list[Path]:
    assert models_dir.is_dir(), "models/transaction-graph/ does not exist"
    found = sorted(models_dir.glob("*.tla"))
    assert found, "no TLA+ specification under models/transaction-graph/"
    return found


def header(text: str) -> str:
    match = re.match(r"\s*((?:\\\*[^\n]*\n|\(\*.*?\*\)\s*)+)", text, re.S)
    assert match, "the specification has no leading comment header"
    return match.group(1)


def checked_names(cfg: str) -> list[str]:
    names = []
    for keyword in ("INVARIANT", "INVARIANTS", "PROPERTY", "PROPERTIES"):
        for block in re.findall(
            rf"^{keyword}\b(.*?)(?=^[A-Z]+\b|\Z)", cfg, re.S | re.M
        ):
            names += re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block)
    return names


def normalized(text: str) -> str:
    return re.sub(r"[_\s]", "", text).lower()


# @spec TXG-MODEL-001
def test_each_model_has_a_configuration_and_a_header(models_dir):
    for tla in specs(models_dir):
        cfg = tla.with_suffix(".cfg")
        assert cfg.is_file(), f"{tla.name} has no configuration file"
        head = header(tla.read_text())
        assert re.search(r"TLC[^\n]*\d+\.\d+(\.\d+)?", head), (
            f"{tla.name}: the header states no TLC version"
        )
        names = checked_names(cfg.read_text())
        assert names, f"{cfg.name} checks nothing"
        for name in names:
            assert name in head, f"{tla.name}: the header does not explain {name}"


# @spec TXG-MODEL-001
def test_models_cover_the_lifecycle_and_the_split_step(models_dir):
    text = normalized(" ".join(t.read_text() for t in specs(models_dir)))
    assert "capturecudagraphs" in text and "split" in text


# @spec TXG-MODEL-002
def test_models_have_every_required_action(models_dir):
    text = normalized(" ".join(t.read_text() for t in specs(models_dir)))
    missing = [stem for stem in ACTION_STEMS if stem not in text]
    assert not missing, f"no action for {missing}"


def variables(text: str) -> list[str]:
    """The names a specification declares in its ``VARIABLES`` block."""
    names = []
    for block in re.findall(r"^VARIABLES\b(.*?)(?=^\S)", text, re.S | re.M):
        block = re.sub(r"\\\*[^\n]*", "", block)
        names += re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block)
    return names


def definitions(text: str) -> list[str]:
    """The operator names a specification defines (``Name ==`` or
    ``Name(args) ==`` at the start of a line)."""
    return re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*==", text, re.M)


#: The KV-cache sizings the models choose among (TXG-MODEL-002): the
#: extensible KV cache, a profiled budget with it off, a fixed block count and
#: a fixed budget, as quoted model values (case and underscores ignored).
SIZING_VALUES = {"extensible", "profiled", "blockcount", "budget"}


# @spec TXG-MODEL-002
def test_models_choose_every_kv_cache_sizing(models_dir):
    """The nondeterministic choice of the KV-cache sizing covers all four of
    core's sizings, the extensible KV cache and a profiled budget included."""
    text = " ".join(t.read_text() for t in specs(models_dir))
    values = {normalized(v) for v in re.findall(r'"([A-Za-z_\- ]+)"', text)}
    missing = sorted(SIZING_VALUES - values)
    assert not missing, f"the models choose no KV-cache sizing {missing}"


# @spec TXG-MODEL-002
def test_models_keep_the_state_group_layout_apart_from_the_bound_views(models_dir):
    """The state-group layout and the state layer's bound views are separate
    state: core's extension rebinds the views while the layout stays
    unchanged."""
    for tla in specs(models_dir):
        names = {normalized(n) for n in variables(tla.read_text())}
        if "stategrouplayout" in names:
            assert "boundviews" in names, (
                f"{tla.name} models the state-group layout but not the bound "
                "views as separate state"
            )
            return
    pytest.fail("no model declares the state-group layout as a variable")


# @spec TXG-MODEL-002
def test_models_have_cores_extension_of_the_kv_cache(models_dir):
    """Core's extension under the extensible KV cache, after post-warmup
    verification and before any scheduled step, is an action of the
    capture-lifecycle model."""
    found = [
        name
        for tla in specs(models_dir)
        for name in definitions(tla.read_text())
        if "extend" in normalized(name)
    ]
    assert found, "no model defines an action for core's extension of the KV cache"


# @spec TXG-MODEL-010
def test_the_measured_and_uncharged_cases_are_separate_state(models_dir):
    """A serving capture with no profiling capture before it is reported as
    measured under the extensible KV cache and as uncharged with it off: the
    measured report is a variable of its own beside the uncharged one, not
    the aggregate comparison's reading of the serving total."""
    for tla in specs(models_dir):
        names = {normalized(n) for n in variables(tla.read_text())}
        if "uncharged" in names:
            measured = [n for n in names if "measured" in n and n != "servmeasured"]
            assert measured, (
                f"{tla.name} has no variable for the measured-memory report "
                "beside the uncharged one"
            )
            return
    pytest.fail("no model declares the uncharged-memory report")


# @spec TXG-MODEL-010
def test_the_unprofiled_serving_invariant_explains_both_cases(models_dir):
    """The header of the checked model citing TXG-MODEL-010 explains the
    measured case under the extensible KV cache and the uncharged case with
    it off."""
    paragraphs = []
    for tla in specs(models_dir):
        if not checked_names(tla.with_suffix(".cfg").read_text()):
            continue
        head = header(tla.read_text())
        paragraphs += [
            p for p in re.split(r"\n\\\*\s{2}(?=\S)", head) if "TXG-MODEL-010" in p
        ]
    assert paragraphs, "no checked model cites TXG-MODEL-010"
    text = " ".join(paragraphs).lower()
    for word in ("measured", "extensible", "uncharged"):
        assert re.search(rf"\b{word}\b", text), (
            f"the TXG-MODEL-010 invariant's explanation does not name {word!r}",
            text,
        )


# @spec TXG-MODEL-002, TXG-MODEL-006
def test_models_follow_the_state_layer_not_the_deleted_slot_bindings(models_dir):
    """The capture lifecycle reaches the pools through core's state-group
    layout; the deleted ``initialize_state_slots`` hook and its startup
    access are no longer core's."""
    text = normalized(" ".join(t.read_text() for t in specs(models_dir)))
    stale = [s for s in ("initializestateslots", "startuppoolaccess") if s in text]
    assert not stale, f"the models still model {stale}"


# @spec TXG-MODEL-008
def test_the_profiling_invariant_covers_the_state_group_layout(models_dir):
    """TXG-MODEL-008 also checks that the transaction-graph part reads the
    state-group layout only inside a capture call: a checked model's header
    explains it with the invariant it cites."""
    heads = [
        header(tla.read_text())
        for tla in specs(models_dir)
        if checked_names(tla.with_suffix(".cfg").read_text())
    ]
    citing = [h for h in heads if "TXG-MODEL-008" in h]
    assert citing, "no checked model cites TXG-MODEL-008"
    assert any("stategrouplayout" in normalized(h.replace("-", "")) for h in citing), (
        "no header citing TXG-MODEL-008 explains the state-group layout read"
    )


def assert_cited(models_dir, spec_ids):
    """Each spec ID is cited in the header of a model whose configuration
    checks something."""
    cited = set()
    for tla in specs(models_dir):
        head = header(tla.read_text())
        if checked_names(tla.with_suffix(".cfg").read_text()):
            cited |= {s for s in spec_ids if s in head}
    missing = [s for s in spec_ids if s not in cited]
    assert not missing, f"no checked model cites {missing}"


# @spec TXG-MODEL-003, TXG-MODEL-007, TXG-MODEL-008, TXG-MODEL-009
def test_replay_and_profiling_invariants_are_checked(models_dir):
    assert_cited(models_dir, INVARIANT_SPECS[:4])


# @spec TXG-MODEL-010, TXG-MODEL-011, TXG-MODEL-012, TXG-MODEL-013
def test_memory_and_reservation_invariants_are_checked(models_dir):
    assert_cited(models_dir, INVARIANT_SPECS[4:8])


# @spec TXG-MODEL-014, TXG-MODEL-015, TXG-MODEL-016
def test_latch_and_cleanup_invariants_are_checked(models_dir):
    assert_cited(models_dir, INVARIANT_SPECS[8:])


# @spec TXG-MODEL-004
def test_liveness_is_checked_under_fairness(models_dir):
    assert_cited(models_dir, [LIVENESS_SPEC])
    found = False
    for tla in specs(models_dir):
        cfg = tla.with_suffix(".cfg").read_text()
        if re.search(r"^PROPERT(Y|IES)\b", cfg, re.M):
            found = True
            assert re.search(r"\b(WF|SF)_", tla.read_text()), (
                f"{tla.name} checks a property without fairness"
            )
    assert found, "no model checks a liveness property"


def _tlc() -> tuple[str, str] | None:
    jar = os.environ.get("TLA2TOOLS_JAR")
    java = shutil.which("java")
    if not jar or not Path(jar).is_file() or not java:
        return None
    return java, jar


# @spec TXG-MODEL-005
def test_tlc_finds_no_violation(models_dir, tmp_path):
    tools = _tlc()
    if tools is None:
        pytest.skip("TLA2TOOLS_JAR or java is not available")
    java, jar = tools
    for tla in specs(models_dir):
        result = subprocess.run(
            [
                java,
                "-XX:+UseParallelGC",
                "-cp",
                jar,
                "tlc2.TLC",
                "-workers",
                "1",
                "-metadir",
                str(tmp_path / tla.stem),
                "-config",
                str(tla.with_suffix(".cfg")),
                str(tla),
            ],
            capture_output=True,
            text=True,
            cwd=models_dir,
            timeout=1800,
        )
        assert result.returncode == 0, f"{tla.name}:\n{result.stdout[-4000:]}"
        assert "No error has been found" in result.stdout, result.stdout[-4000:]
