# SPDX-License-Identifier: Apache-2.0
"""CPU register() against a checkout of rfc12-integration.

This does not start an engine and does not load the model class. It
imports the RFC12 tree and calls ``register()``. Gate A (installed
plugin load, admit, chunks, retire) is a separate run.

Exact command, from this repository::

    git clone --filter=blob:none --branch rfc12-integration \\
        https://github.com/pdtgct/vllm.git /tmp/vllm-rfc12
    git -C /tmp/vllm-rfc12 checkout 4c1809692dfa46fcdd7a9891f69a997afa1dbb78
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -e '.[test]'
    VLLM_RFC12_CHECKOUT=/tmp/vllm-rfc12 python -m pytest -q tests/test_rfc12_register.py

``4c180969`` has tree ``465f10ad65c7742fca01e25e3a1a0f58c4608094``.
The test skips when that checkout cannot be imported.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

from vllm_nemotron_asr_streaming.refs import RFC12_COMMIT, RFC12_TREE

CHECKOUT = os.environ.get("VLLM_RFC12_CHECKOUT", "")
COMMAND = (
    "VLLM_RFC12_CHECKOUT=/tmp/vllm-rfc12 "
    "python -m pytest -q tests/test_rfc12_register.py"
)


def _checkout() -> Path | None:
    if not CHECKOUT:
        return None
    root = Path(CHECKOUT)
    if not (root / "vllm" / "__init__.py").is_file():
        return None
    return root


@pytest.mark.skipif(
    _checkout() is None,
    reason=(
        "set VLLM_RFC12_CHECKOUT to a checkout of pdtgct/vllm "
        f"rfc12-integration@{RFC12_COMMIT} (tree {RFC12_TREE}) and run "
        f"{COMMAND}"
    ),
)
def test_register_against_rfc12_checkout() -> None:
    root = _checkout()
    assert root is not None
    sys.path.insert(0, str(root))
    # Drop stub modules installed by other tests in this process.
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            del sys.modules[name]
        if name == "vllm_nemotron_asr_streaming" or name.startswith(
            "vllm_nemotron_asr_streaming."
        ):
            del sys.modules[name]
    try:
        vllm = importlib.import_module("vllm")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            "rfc12 checkout did not import (CPU torch and vLLM's Python "
            f"dependencies are required): {exc.__class__.__name__}: {exc}. "
            f"Command: {COMMAND}"
        )
    assert getattr(vllm, "__file__", "")
    from vllm_nemotron_asr_streaming import register
    from vllm_nemotron_asr_streaming.identity import (
        ARCHITECTURE,
        OFFLINE_ARCHITECTURE,
    )

    register()
    assert "vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr" not in sys.modules
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    assert ARCHITECTURE in ModelRegistry.models
    assert ModelRegistry.models[ARCHITECTURE].__class__.__name__
    registered = ModelRegistry.models[ARCHITECTURE]
    # Lazy registration stores a loader, not the offline class name.
    assert OFFLINE_ARCHITECTURE not in repr(getattr(registered, "class_name", ""))
    assert MODELS_CONFIG_MAP[ARCHITECTURE].__name__ == (
        "Nemotron3_5AsrStreamingConfig"
    )
    if OFFLINE_ARCHITECTURE in MODELS_CONFIG_MAP:
        assert (
            MODELS_CONFIG_MAP[OFFLINE_ARCHITECTURE].__name__
            != "Nemotron3_5AsrStreamingConfig"
        )
