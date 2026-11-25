# SPDX-License-Identifier: Apache-2.0
"""A malformed file request fails its own row and nothing else (CPU).

Drives core's request hooks (``add_request``), the model state's screening
(``_screen_rows``), the production planning and the registry binding of
``_stage_v2_projection`` with a malformed file request next to a valid file
request and a valid stream request.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support.card import BLANK

pytestmark = pytest.mark.engine

FILE_TOKEN = BLANK + 4
STREAM_TOKEN = BLANK + 2
LAYER = "nemotron.state"


def _elem(data):
    from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldElem

    return MultiModalFieldElem(data, MultiModalBatchedField(keep_on_cpu=True))


def _request(make_step, req_id, token, fields, uuid=None):
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )

    request = make_step.new_request(req_id, token)
    request.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem({k: _elem(v) for k, v in fields.items()}),
            modality="audio",
            identifier=uuid or f"uuid-{req_id}",
            mm_position=PlaceholderRange(offset=0, length=1),
        )
    ]
    return request


def _file_fields(n, row=0, count=None, geometry=3):
    samples = torch.from_numpy(np.zeros(n, np.float32))
    header = torch.tensor(
        [geometry, row, n if count is None else count], dtype=torch.int64
    )
    return {"samples": samples, "file_header": header}


def _stream_fields():
    row = np.zeros(17_927, np.float32)
    row[:7] = (2, 8960, 3, 0, 0, 0, 1)
    return {"audio": torch.from_numpy(row)}


def _bad_fields(kind, num_prompts):
    good = _file_fields(2000)
    if kind == "missing_samples":
        return {"file_header": good["file_header"]}
    if kind == "missing_header":
        return {"samples": good["samples"]}
    if kind == "bad_row":
        return _file_fields(2000, row=num_prompts)
    if kind == "foreign_geometry":
        # an implemented geometry (index 2, 320 ms) that is not the server's
        # (index 3, lookahead 6): only a direct engine client can send it
        return _file_fields(2000, geometry=2)
    return _file_fields(2000, count=1999)


@pytest.mark.parametrize(
    "kind",
    ["missing_samples", "missing_header", "bad_row", "bad_count", "foreign_geometry"],
)
# @spec FILE-BUF-001, FILE-ROW-001, FILE-PROMPT-007
def test_malformed_file_request_fails_only_its_row(file_model_state, make_step, kind):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileAction
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import RowBinding
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
        PlanContextSlot,
        SessionRegistry,
    )

    state = file_model_state(6)
    model = state.model
    config = model.config
    num_prompts = int(config.num_prompts)
    model._registry = SessionRegistry()
    model._plan_slot = PlanContextSlot()
    model._plan_step = 0
    model._commit_sink = None
    model.core = SimpleNamespace(lid=SimpleNamespace(num_prompts=num_prompts))

    good = _request(make_step, "good", FILE_TOKEN, _file_fields(1280 * 7 * 2))
    bad = _request(make_step, "bad", FILE_TOKEN, _bad_fields(kind, num_prompts))
    stream = _request(make_step, "stream", STREAM_TOKEN, _stream_fields())
    ids = ("good", "bad", "stream")
    handles = {i: make_step.handle(n + 1, 1) for n, i in enumerate(ids)}
    out = make_step.step(new=[good, bad, stream], handles=handles)
    make_step.run(state, out)

    bindings = tuple(RowBinding(i, handles[i][LAYER], True) for i in ids)
    input_batch = SimpleNamespace(
        req_ids=list(ids),
        num_computed_tokens_np=np.zeros(3, np.int64),
        num_scheduled_tokens=np.ones(3, np.int64),
    )
    screen = state._screen_rows(input_batch, {})
    # The file the model state could not take fails at screening, before
    # core's encoder path (REG-STATE-015); the others are clean.
    assert screen.failed == {"bad"}, screen.failed
    token_rows = list(screen.token_rows)
    plan = state._plan_production_step(
        ids,
        bindings,
        SimpleNamespace(pools=None),
        token_rows,
        input_batch,
        headers=screen.headers,
        failed=screen.failed,
    )
    by_id = {r.request_id: r for r in plan.rows}
    assert by_id["bad"].failed
    if kind != "foreign_geometry":
        assert "bad" not in state.file_buffers
    assert by_id["good"].file_plan.action == FileAction.CHUNK
    assert "good" in state.file_buffers

    projection = SimpleNamespace(
        input_batch=input_batch,
        request_metadata=state._metadata,
        req_ids=ids,
        bindings=bindings,
        token_ids=tuple(t[0] for t in token_rows),
        scheduled_encoder_inputs={},
        file_rows={r.request_id: r for r in plan.rows if r.is_file},
        planning_failed=set(screen.failed),
        stream_headers=dict(screen.headers),
        req_states=SimpleNamespace(req_id_to_index={i: n for n, i in enumerate(ids)}),
    )
    context = type(model)._stage_v2_projection(model, projection)
    chunk = dict(zip(context.request_ids, context.is_chunk.tolist(), strict=True))
    assert chunk == {"good": True, "bad": False, "stream": True}


@pytest.mark.parametrize(
    "header, count",
    [
        ([3, 0], 10),  # wrong shape
        ([5, 0, 10], 10),  # geometry outside the table
        ([3, -1, 10], 10),
        ([3, 16, 10], 10),  # row == num_prompts
        ([3, 0, 9], 10),  # sample-count mismatch
    ],
)
# @spec FILE-PROMPT-002
def test_processor_refuses_each_malformed_header(header, count):
    from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
        validate_file_header,
    )

    with pytest.raises(ValueError):
        validate_file_header(header, [count], 16)
    validate_file_header([3, 15, count], [count], 16)
