# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

from tests.fakes import drop_cached_plugin_modules, install


def test_register_leaves_offline_architecture_untouched() -> None:
    registry = install()
    registry.models["Nemotron3_5AsrForRNNT"] = "offline-plugin"
    drop_cached_plugin_modules()
    from vllm_nemotron_asr_streaming import register

    register()
    register()
    assert registry.models["Nemotron3_5AsrForRNNT"] == "offline-plugin"
    assert (
        registry.models["Nemotron3_5AsrStreamingForRNNT"]
        == "vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT"
    )
    assert "vllm_nemotron_asr_streaming.model" not in sys.modules


def test_register_refuses_offline_name(monkeypatch) -> None:
    install()
    drop_cached_plugin_modules()
    from vllm_nemotron_asr_streaming import identity, register

    monkeypatch.setattr(
        identity,
        "ARCHITECTURE",
        identity.OFFLINE_ARCHITECTURE,
    )
    try:
        register()
    except RuntimeError as exc:
        assert "offline architecture" in str(exc)
    else:
        raise AssertionError("register() accepted the offline architecture")


def test_register_requires_admit_stream_async(monkeypatch) -> None:
    install()
    drop_cached_plugin_modules()
    from vllm.v1.engine import core_client

    monkeypatch.delattr(core_client.EngineCoreClient, "admit_stream_async")
    from vllm_nemotron_asr_streaming import register

    try:
        register()
    except RuntimeError as exc:
        assert "admit_stream_async" in str(exc)
    else:
        raise AssertionError("register() ignored a missing admit_stream_async")
