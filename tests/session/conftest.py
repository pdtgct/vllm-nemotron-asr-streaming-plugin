# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the session segment."""

from __future__ import annotations

import pytest
from support.fake_engine import FakeEngine, make_hf_config


@pytest.fixture
def build():
    """Return ``build(engine=None, hf=None, **kw) -> (engine, factory)`` with
    small backoff parameters so retries take milliseconds."""

    def _build(engine=None, hf=None, **kw):
        from vllm_nemotron_asr_streaming.session.factory import (
            StreamingSessionFactory,
        )

        engine = engine if engine is not None else FakeEngine()
        kw.setdefault("backoff_initial_s", 0.002)
        kw.setdefault("backoff_max_s", 0.01)
        factory = StreamingSessionFactory(engine, hf or make_hf_config(), **kw)
        return engine, factory

    return _build


@pytest.fixture(autouse=True, params=[40, 57], ids=["blank40", "blank57"])
def layout(request, monkeypatch):
    """Run every session test under two coherent control-token layouts, so a
    session that reads a constant instead of the served configuration fails
    under one of them."""
    import support.fake_engine as fake_engine

    monkeypatch.setattr(fake_engine, "DEFAULT_BLANK", request.param)
    return request.param
