# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the session segment.

The frontend's seam conformance suite is registered for pytest's assertion
rewriting here, before ``test_frontend_conformance`` imports it (SESS-FAC-007).

The session specs hold two kinds of obligation (their scope). The current
model-side and shared ones (SESS-FAC-001 to SESS-FAC-003, server geometry, the
packer, the geometry check in the model, core ``/v1/realtime`` and
SESS-EP-001) are conformance on the target core, ``pdtgct/vllm`` at
``ddf6db2ab8``. The session factory's groups run on the stream-admission
client path, which the target core does not provide; their tests drive the
factory through ``support.fake_engine``, which models that path, and a pass
makes no claim for the target core. Each such module says so in its
docstring.
"""

from __future__ import annotations

import pytest

pytest.register_assert_rewrite("vllm_riva_frontend_dev.seam_conformance")

from support.fake_engine import FakeEngine, make_hf_config  # noqa: E402


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
