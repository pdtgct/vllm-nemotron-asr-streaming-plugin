# SPDX-License-Identifier: Apache-2.0
"""Test tiers.

* unmarked: CPU tests needing only this package and its pure-Python modules.
* ``engine``: CPU tests that import the stage 1 vLLM build (upstream ``main`` with
  vllm#67374, #67375, #67144 and #67151 on ``PYTHONPATH``). Skipped when it does not
  import.
* ``gpu``: installed-engine tests on a CUDA device (``vllm serve`` with the
  plugin installed). Skipped without CUDA.
* ``card``: tests against a local copy of the published Hub card, named by
  ``NEMOTRON_CARD_DIR``. Skipped when unset. The card is never committed.
"""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import pytest


def _engine_available() -> bool:
    try:
        async_llm = importlib.import_module("vllm.v1.engine.async_llm")
    except Exception:
        return False
    generate = getattr(async_llm.AsyncLLM, "generate", None)
    return (
        generate is not None
        and "stream_admission" in inspect.signature(generate).parameters
        and hasattr(async_llm.AsyncLLM, "wait_stream_retired")
    )


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


ENGINE = _engine_available()
CUDA = _cuda_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "engine: needs the stage 1 vLLM build")
    config.addinivalue_line("markers", "gpu: needs an installed engine on CUDA")
    config.addinivalue_line("markers", "card: needs NEMOTRON_CARD_DIR")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    card_dir = os.environ.get("NEMOTRON_CARD_DIR")
    for item in items:
        if "engine" in item.keywords and not ENGINE:
            item.add_marker(
                pytest.mark.skip(reason="stage 1 vLLM build not importable")
            )
        if "gpu" in item.keywords and not CUDA:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "card" in item.keywords and not card_dir:
            item.add_marker(pytest.mark.skip(reason="NEMOTRON_CARD_DIR not set"))


@pytest.fixture
def card_dir(tmp_path: Path) -> Path:
    """A synthetic Hub card directory (see ``support.card``)."""
    from support.card import write_card

    return write_card(tmp_path / "card")


@pytest.fixture
def real_card_dir() -> Path:
    """The local copy of the published card (``card`` tests only)."""
    path = os.environ.get("NEMOTRON_CARD_DIR")
    if not path:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    return Path(path)
