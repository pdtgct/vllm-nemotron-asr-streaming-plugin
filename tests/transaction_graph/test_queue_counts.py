# SPDX-License-Identifier: Apache-2.0
"""Remaining queue counts written by the transaction body (TXG-QC-001), with
transaction graphs off, on CPU. The reservation view under transaction
graphs (TXG-QC-002) is in ``test_exec_gpu.py``."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.engine


def remaining_of(book_row) -> int:
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import QUEUE_HEAD, QUEUE_LEN

    return int(book_row[QUEUE_LEN]) - int(book_row[QUEUE_HEAD])


# @spec TXG-QC-001
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
def test_each_row_reports_its_remaining_queue_count(txn_env, lookahead):
    env = txn_env(lookahead)
    drains = env.fixtures[("other", False, False)]
    rows = [
        env.accepted(("chunk", False, False)),
        drains[0],
        env.accepted(("chunk", True, True)),
        drains[min(3, len(drains) - 1)],
        env.accepted(("other", False, True)),
        env.accepted(("other", True, False)),
    ]
    out = env.run(env.compose(rows))
    counts = out.queue_counts
    assert counts is not None, "the commit plan carries no queue_counts"
    assert counts.dtype == torch.int32 and tuple(counts.shape) == (len(rows),)
    books = out.pools.tensors["book_pool"]
    accepted = [i for i in range(len(rows)) if int(out.status[i]) == 0]
    assert accepted
    for i in accepted:
        assert int(counts[i]) == remaining_of(books[i + 1]), rows[i].name
    assert any(int(counts[i]) > 0 for i in accepted), "every queue was empty"


# @spec TXG-QC-001
def test_counts_follow_the_rows_positions(txn_env):
    env = txn_env(0)
    drains = env.fixtures[("other", False, False)]
    rows = [drains[0], env.accepted(("chunk", True, False)), drains[-1]]
    forward = env.run(env.compose(rows, blocks=[1, 2, 3]))
    backward = env.run(env.compose(rows[::-1], blocks=[3, 2, 1]))
    assert forward.queue_counts is not None and backward.queue_counts is not None
    assert forward.queue_counts.tolist() == backward.queue_counts.flip(0).tolist()
