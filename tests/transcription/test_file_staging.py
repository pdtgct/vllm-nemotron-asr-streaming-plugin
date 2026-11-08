# SPDX-License-Identifier: Apache-2.0
"""Production staging and queue-read paths of FILE rows on CPU.

Regressions: ``advance_model_rows`` staged an ``is_file`` vector through a
``HostStaging`` that did not know the slot; the label-queue read of the step
planner had no test.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support.card import BLANK

FILE_TOKEN = BLANK + 4


# @spec FILE-ROW-001
@pytest.mark.engine
def test_every_slot_the_transaction_stages_is_known_to_host_staging():
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    source = inspect.getsource(advance.advance_model_rows)
    slots = set(re.findall(r'_stage\(\s*"([a-z_]+)"', source))
    assert "is_file" in slots
    staging = advance.HostStaging(4)
    device = torch.device("cpu")
    for name in sorted(slots):
        if name == "fresh_init_book":
            vector = torch.zeros(2, 7, dtype=torch.int64)
        elif name in ("is_chunk", "is_file"):
            vector = torch.tensor([True, False])
        else:
            vector = torch.tensor([1, 2], dtype=torch.int64)
        staging.stage(name, vector, device)


# @spec FILE-ENG-003
@pytest.mark.engine
def test_file_and_chunk_role_vectors_use_separate_storage():
    from vllm_nemotron_asr_streaming.nemotron_asr.advance import HostStaging

    staging = HostStaging(4)
    chunk = torch.tensor([True, True, False])
    file = torch.tensor([False, True, False])  # a mixed FILE/stream batch
    staged_chunk = staging.stage("is_chunk", chunk, torch.device("cpu"))
    staged_file = staging.stage("is_file", file, torch.device("cpu"))
    assert staged_chunk.tolist() == chunk.tolist()
    assert staged_file.tolist() == file.tolist()
    assert staging._is_chunk_buf.data_ptr() != staging._is_file_buf.data_ptr()


# @spec FILE-ROW-002, FILE-BUF-002
@pytest.mark.engine
def test_production_steps_read_the_device_queue_after_an_advance(
    file_model_state, make_step, monkeypatch
):
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import FileAction
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import RowBinding
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import QUEUE_HEAD, QUEUE_LEN

    state = file_model_state(6)
    chunk = 1280 * 7
    total = chunk * 3 + 5
    samples = np.random.default_rng(0).standard_normal(total).astype(np.float32)
    request = make_step.new_request(
        "f1", FILE_TOKEN, samples=samples, header=[3, 0, total]
    )
    handle = make_step.handle(1, 1)
    book = torch.zeros(4, 7, dtype=torch.int32)
    book[1, QUEUE_LEN] = 3  # the device reports two labels still queued
    book[1, QUEUE_HEAD] = 1
    reads = []
    real_index_select = torch.Tensor.index_select

    def counting(self, dim, index):
        if self is book:
            reads.append(index.tolist())
        return real_index_select(self, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", counting)
    batch = SimpleNamespace(pools=SimpleNamespace(replay_book=book))
    input_batch = SimpleNamespace(num_computed_tokens_np=np.zeros(1, np.int64))

    def run(out, fresh):
        state.prepare_step(out, dummy_run=False, is_profile=False)
        binding = RowBinding("f1", handle["nemotron.state"], fresh)
        plan = state._plan_production_step(
            ("f1",), (binding,), batch, [state._scheduled_tokens["f1"]], input_batch
        )
        (row,) = plan.rows
        return row.file_plan.action

    actions = [
        run(make_step.step(new=[request], handles={"f1": handle}), True),
        run(make_step.step(cached=["f1"], handles={"f1": handle}), False),
        run(make_step.step(cached=["f1"], handles={"f1": handle}), False),
        run(make_step.step(cached=["f1"], handles={"f1": handle}), False),
    ]
    # fresh row: empty queue, a chunk; the next step reads the device (two
    # queued labels) and drains; the host then knows the count, drains the
    # second without a read, and the emptied queue lets the next chunk run.
    assert actions == [
        FileAction.CHUNK,
        FileAction.DRAIN,
        FileAction.DRAIN,
        FileAction.CHUNK,
    ]
    assert len(reads) == 1
