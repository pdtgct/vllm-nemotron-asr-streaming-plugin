# SPDX-License-Identifier: Apache-2.0
"""The server's one lookahead (REG-LA-001, REG-LA-009).

CPU only: ``vllm_nemotron_asr_streaming.lookahead`` imports no vLLM, and the
one-source check is a static scan of the package. Configurations are plain
namespaces with the two fields ``served_lookahead`` reads.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

IMPLEMENTED = (0, 1, 3, 6, 13)
PACKAGE = "vllm_nemotron_asr_streaming"


def _served(**fields):
    from vllm_nemotron_asr_streaming.lookahead import served_lookahead

    return served_lookahead(SimpleNamespace(**fields))


# --------------------------------------------------------------------------
# REG-LA-001
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", IMPLEMENTED)
# @spec REG-LA-001
def test_the_override_is_the_servers_lookahead(value):
    assert _served(num_lookahead_tokens=value, card_default_num_lookahead_tokens=3) == (
        value
    )


@pytest.mark.parametrize("value", IMPLEMENTED)
# @spec REG-LA-001
def test_without_an_override_the_card_default_is_the_servers_lookahead(value):
    assert _served(card_default_num_lookahead_tokens=value) == value


# @spec REG-LA-001
def test_the_card_set_and_a_served_set_play_no_part():
    """Only the two fields decide; neither ``card_num_lookahead_tokens`` nor
    a ``supported_num_lookahead_tokens`` key selects or limits the value."""
    assert (
        _served(
            card_default_num_lookahead_tokens=3,
            card_num_lookahead_tokens=[0, 6],
            supported_num_lookahead_tokens=[13],
        )
        == 3
    )
    assert _served(num_lookahead_tokens=1, card_num_lookahead_tokens=[0, 3]) == 1


# @spec REG-LA-001
def test_the_result_is_a_plain_int():
    value = _served(num_lookahead_tokens=6)
    assert type(value) is int


# @spec REG-LA-001
def test_neither_field_raises_naming_the_field():
    with pytest.raises(ValueError) as excinfo:
        _served()
    assert "num_lookahead_tokens" in str(excinfo.value)


#: A present ``None`` is a present value that is not an integer; it does not
#: fall through to the card default.
_BAD = [None, True, False, 3.0, "3", [3], 2, 4, 14, -1]


@pytest.mark.parametrize("value", _BAD, ids=repr)
# @spec REG-LA-001
def test_a_bad_override_raises_naming_it(value):
    """A ``bool`` is not an integer here (``True == 1``), nor is ``3.0``."""
    with pytest.raises(ValueError) as excinfo:
        _served(num_lookahead_tokens=value, card_default_num_lookahead_tokens=3)
    assert "num_lookahead_tokens" in str(excinfo.value)


@pytest.mark.parametrize("value", _BAD, ids=repr)
# @spec REG-LA-001
def test_a_bad_card_default_raises_naming_it(value):
    with pytest.raises(ValueError) as excinfo:
        _served(card_default_num_lookahead_tokens=value)
    assert "card_default_num_lookahead_tokens" in str(excinfo.value)


# @spec REG-LA-001
def test_served_lookahead_imports_no_vllm(run_fresh):
    out = run_fresh(
        """
import sys
from vllm_nemotron_asr_streaming.lookahead import served_lookahead
emit(vllm=[m for m in sys.modules if m == "vllm" or m.startswith("vllm.")])
"""
    )
    assert out["vllm"] == []


# --------------------------------------------------------------------------
# REG-LA-009: one geometry source
# --------------------------------------------------------------------------

#: Configuration fields from which the server's lookahead is resolved, or
#: which describe lookahead sets. Only ``lookahead.served_lookahead`` reads the
#: first two to obtain the lookahead.
GEOMETRY_FIELDS = (
    "num_lookahead_tokens",
    "card_default_num_lookahead_tokens",
    "supported_num_lookahead_tokens",
    "card_num_lookahead_tokens",
    "default_num_lookahead_tokens",
)

#: Modules that may name those fields for reasons other than reading the
#: server's lookahead: translation and the schema check (the card's fields),
#: post-override validation (it reports the field a bad value came from,
#: REG-LA-002 and REG-LA-004, refuses ``supported_num_lookahead_tokens``,
#: REG-LA-008, and logs an undeclared geometry, REG-LA-003), the
#: configuration class (it declares the fields), and the resolver itself.
FIELD_OWNERS = {
    "lookahead.py",
    "card_parser.py",
    "streaming_config.py",
    "nemotron_asr/configuration_nemotron_asr.py",
}

#: Readers of the server's lookahead named by REG-LA-009, by module: the
#: model's warmed and captured geometry and its per-row geometry check
#: (``nemotron_asr/nemotron_asr.py``), the session factory and
#: ``buffer_realtime_audio`` (``session/``), ``get_generation_prompt``
#: (``transcription.py``), the processor's header check
#: (``nemotron_asr/processor.py``) and native burst's configuration check
#: (``nemotron_asr/native_burst.py``).
READERS = (
    "nemotron_asr/nemotron_asr.py",
    "session/factory.py",
    "session/packer.py",
    "transcription.py",
    "nemotron_asr/processor.py",
    "nemotron_asr/native_burst.py",
)

#: Names of earlier served-set readers; no module may define or use them.
RETIRED = ("served_lookaheads", "default_lookahead", "served_lookahead_source")


def _sources(repo_root: Path) -> dict[str, str]:
    root = repo_root / PACKAGE
    return {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*.py")}


def _field_uses(source: str) -> list[tuple[int, str]]:
    """Attribute reads, ``getattr``/``get``/subscript string keys and other
    string constants that name a geometry field (docstrings excluded)."""
    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in GEOMETRY_FIELDS:
            found.append((node.lineno, node.attr))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value in GEOMETRY_FIELDS
        ):
            found.append((node.lineno, node.value))
        elif isinstance(node, ast.keyword) and node.arg in GEOMETRY_FIELDS:
            found.append((node.value.lineno, node.arg))
    return found


# @spec REG-LA-009
def test_no_component_reads_the_lookahead_fields_itself(repo_root):
    found = []
    for name, source in _sources(repo_root).items():
        if name in FIELD_OWNERS:
            continue
        found.extend(f"{name}:{line}: {field}" for line, field in _field_uses(source))
    assert not found, (
        "components that read a lookahead field instead of calling "
        "served_lookahead:\n" + "\n".join(found)
    )


# @spec REG-LA-009
def test_no_served_set_reader_remains(repo_root):
    found = []
    for name, source in _sources(repo_root).items():
        for node in ast.walk(ast.parse(source)):
            text = None
            if isinstance(node, ast.Name):
                text = node.id
            elif isinstance(node, ast.Attribute):
                text = node.attr
            elif isinstance(node, ast.FunctionDef):
                text = node.name
            elif isinstance(node, ast.alias):
                text = node.name
            if text in RETIRED:
                found.append(f"{name}:{getattr(node, 'lineno', '?')}: {text}")
    assert not found, "\n".join(found)


@pytest.mark.parametrize("module", READERS)
# @spec REG-LA-009
def test_each_named_reader_calls_served_lookahead(repo_root, module):
    source = (repo_root / PACKAGE / module).read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "served_lookahead")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "served_lookahead"
            )
        )
    ]
    assert calls, f"{module} does not call served_lookahead"


# @spec REG-LA-009
def test_served_lookahead_is_defined_once(repo_root):
    defined = [
        name
        for name, source in _sources(repo_root).items()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "served_lookahead"
    ]
    assert defined == ["lookahead.py"], defined


# @spec REG-LA-009
def test_the_scan_sees_each_form_of_field_read():
    """Self-test of ``_field_uses`` on the forms a component could use."""
    source = (
        '"""num_lookahead_tokens in a docstring is prose."""\n'
        "a = config.num_lookahead_tokens\n"
        "b = getattr(config, 'card_default_num_lookahead_tokens', None)\n"
        "c = data.get('supported_num_lookahead_tokens')\n"
        "d = data['card_num_lookahead_tokens']\n"
        "e = f(num_lookahead_tokens=1)\n"
    )
    lines = sorted({line for line, _ in _field_uses(source)})
    assert lines == [2, 3, 4, 5, 6], lines
    assert re.fullmatch(r"\w+", GEOMETRY_FIELDS[0])
