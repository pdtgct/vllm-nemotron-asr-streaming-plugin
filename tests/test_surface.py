# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vllm_nemotron_asr_streaming"


def _python_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_architecture_differs_from_offline_and_hub_card() -> None:
    from vllm_nemotron_asr_streaming.identity import (
        ARCHITECTURE,
        LAZY_MODEL,
        OFFLINE_ARCHITECTURE,
        OFFLINE_MODEL_TYPE,
    )
    from vllm_nemotron_asr_streaming.lookahead import CARD_ARCHITECTURE

    assert ARCHITECTURE != OFFLINE_ARCHITECTURE
    assert ARCHITECTURE != OFFLINE_MODEL_TYPE
    assert CARD_ARCHITECTURE == OFFLINE_ARCHITECTURE
    assert LAZY_MODEL.endswith(":" + ARCHITECTURE)
    assert OFFLINE_ARCHITECTURE not in LAZY_MODEL.split(":")[-1]


def test_package_import_does_not_import_vllm() -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]
        if name == "vllm_nemotron_asr_streaming" or name.startswith(
            "vllm_nemotron_asr_streaming."
        ):
            del sys.modules[name]
    import vllm_nemotron_asr_streaming

    assert vllm_nemotron_asr_streaming.ARCHITECTURE
    assert not any(
        name == "vllm" or name.startswith("vllm.") for name in sys.modules
    )


def test_no_omni_references() -> None:
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "vllm_omni" in text or "vllm.omni" in text or "nemotron_session" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_sources_parse() -> None:
    for path in _python_files():
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_pyproject_entry_point() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "vllm-nemotron-asr-streaming"
    assert data["project"]["dependencies"] == []
    entry = data["project"]["entry-points"]["vllm.general_plugins"]
    assert entry == {
        "nemotron_asr_streaming": "vllm_nemotron_asr_streaming:register"
    }
    assert "nemotron_asr" not in entry


def test_lookahead_trap_is_unresolved() -> None:
    from vllm_nemotron_asr_streaming.lookahead import (
        CARD_DEFAULT_NUM_LOOKAHEAD_TOKENS,
        CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS,
        SOURCE_CADENCE_BY_LOOKAHEAD,
        SOURCE_NATIVE_BURST_REQUIRED_LOOKAHEADS,
        SOURCE_TRANSLATE_PINNED_LOOKAHEAD,
    )

    assert CARD_DEFAULT_NUM_LOOKAHEAD_TOKENS in CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS
    assert SOURCE_TRANSLATE_PINNED_LOOKAHEAD in CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS
    assert SOURCE_TRANSLATE_PINNED_LOOKAHEAD != CARD_DEFAULT_NUM_LOOKAHEAD_TOKENS
    assert SOURCE_NATIVE_BURST_REQUIRED_LOOKAHEADS not in (
        CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS,
    )
    for token in SOURCE_NATIVE_BURST_REQUIRED_LOOKAHEADS:
        assert token not in CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS
    assert set(SOURCE_CADENCE_BY_LOOKAHEAD) >= set(
        CARD_SUPPORTED_NUM_LOOKAHEAD_TOKENS
    )


def test_engine_flag_overrides_are_explicit_and_unused() -> None:
    from vllm_nemotron_asr_streaming.engine_flags import (
        PARALLEL_SIZES_MUST_BE_ONE,
        REFUSED_WHEN_ENABLED,
        REQUIRED_OVERRIDES,
    )

    assert REQUIRED_OVERRIDES == {
        "async_scheduling": False,
        "enable_prefix_caching": False,
    }
    assert "async scheduling" in REFUSED_WHEN_ENABLED
    assert "prefix caching" in REFUSED_WHEN_ENABLED
    assert "tensor_parallel_size" in PARALLEL_SIZES_MUST_BE_ONE
    # The overrides are data. No module in the package reads them to
    # launch an engine.
    for path in _python_files():
        if path.name == "engine_flags.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "REQUIRED_OVERRIDES" not in text


def test_merge_conflict_list_does_not_include_the_model_tree() -> None:
    from vllm_nemotron_asr_streaming.refs import (
        BOTH_ADDED_PATHS,
        MERGE_CONFLICT_PATHS,
        RFC1_TIP_COMMIT,
        SOURCE_COMMIT,
    )

    assert RFC1_TIP_COMMIT.startswith("8b8327e2ee")
    assert SOURCE_COMMIT.startswith("43c0bed5")
    assert len(MERGE_CONFLICT_PATHS) == 20
    assert all(
        not path.startswith("vllm/models/nemotron_asr") for path in MERGE_CONFLICT_PATHS
    )
    assert "vllm/v1/core/stream_admission.py" in BOTH_ADDED_PATHS


@pytest.mark.parametrize(
    "relative",
    [
        "identity.py",
        "plugin.py",
        "model.py",
        "session/factory.py",
    ],
)
def test_module_does_not_register_offline_architecture(relative: str) -> None:
    tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name != "register_model":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "Nemotron3_5AsrForRNNT":
            raise AssertionError(f"{relative} registers the offline architecture")
