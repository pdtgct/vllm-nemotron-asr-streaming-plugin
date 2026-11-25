# SPDX-License-Identifier: Apache-2.0
"""Real steps whose rows schedule more than one token, for the token-position
specs (TXG-TXN-010, TXG-COV-014, TXG-FAIL-009; REG-STATE-014).

``support.model_harness.Runner`` schedules one token per request.
:func:`allow_multi_token_rows` gives one runner core's general token axis for
the steps that need it: each scheduled new request takes
``num_scheduled_tokens[req_id]`` consecutive tokens of its prompt, from its
``num_computed_tokens``, so row ``i``'s token positions are
``query_start_loc[i]`` to ``query_start_loc[i + 1] - 1`` and core samples
row ``i`` at the last of them (``logits_indices``), as core's
``InputBatch`` lays a step out. A step whose every row schedules one token is
built by the harness unchanged.

:func:`token_step` builds such a step's ``SchedulerOutput`` and runs it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from support import model_harness as h


def allow_multi_token_rows(runner: Any) -> Any:
    """Make ``runner``'s real steps lay out multi-token rows (see the module
    docstring). Only new requests may schedule more than one token. Returns
    ``runner``."""
    one_token = runner._real_batch
    if getattr(one_token, "_txg_multi_token", False):
        return runner

    def real_batch(output: Any, computed: dict[str, int] | None) -> Any:
        new = list(output.scheduled_new_reqs)
        cached = list(output.scheduled_cached_reqs.req_ids)
        req_ids = [r.req_id for r in new] + cached
        counts = [int(output.num_scheduled_tokens[r]) for r in req_ids]
        if all(c == 1 for c in counts):
            return one_token(output, computed)
        assert not cached, "a multi-token step schedules new requests only"
        n, total = len(req_ids), sum(counts)
        batch = h.dummy_batch(runner.state, n, total, is_padding=False)
        tokens: list[int] = []
        fed: list[tuple[str, int, int]] = []
        starts = []
        for request, count in zip(new, counts, strict=True):
            runner._prompts[request.req_id] = list(request.prefill_token_ids)
            runner._sampled.pop(request.req_id, None)
            start = int(request.num_computed_tokens)
            window = [int(v) for v in request.prefill_token_ids[start : start + count]]
            assert len(window) == count, (request.req_id, count, window)
            starts.append(start)
            tokens += window
            fed += [(request.req_id, start + k, v) for k, v in enumerate(window)]
        batch.input_ids[:total] = torch.tensor(tokens)
        runner._step_fed = fed
        query_start_loc_np = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.asarray(counts, dtype=np.int32), out=query_start_loc_np[1:])
        device = runner.state.device
        query_start_loc = torch.from_numpy(query_start_loc_np.copy()).to(device)
        index = np.array(
            [runner.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp
        )
        seq_lens = np.asarray(starts, dtype=np.int32) + np.asarray(counts, np.int32)
        return dataclasses.replace(
            batch,
            req_ids=req_ids,
            idx_mapping=torch.from_numpy(index).to(device),
            idx_mapping_np=index,
            num_scheduled_tokens=np.asarray(counts, dtype=np.int32),
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=torch.from_numpy(seq_lens.copy()).to(device),
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens.copy()),
            logits_indices=query_start_loc[1:] - 1,
            num_computed_tokens_np=np.asarray(starts, dtype=np.int32),
            prefill_len_np=np.array(
                [len(r.prefill_token_ids) for r in new], dtype=np.int32
            ),
        )

    real_batch._txg_multi_token = True  # type: ignore[attr-defined]
    runner._real_batch = real_batch
    return runner


def with_tokens(request: Any, *extra: int) -> Any:
    """``request`` with ``extra`` tokens after its prompt, so a step can
    schedule them with its first one (a multi-token row, which only a direct
    engine client produces)."""
    prompt = [*request.prompt_token_ids, *extra]
    return dataclasses.replace(
        request, prompt_token_ids=prompt, prefill_token_ids=list(prompt)
    )


def token_step(
    runner: Any,
    *new: Any,
    counts: dict[str, int] | None = None,
    handles: dict[str, int] | None = None,
) -> Any:
    """One real step of the new requests ``new``, each scheduling
    ``counts[req_id]`` tokens (default 1) and taking the slot ``handles``
    gives it (default ``1..n``). Returns ``(hidden, completion)``."""
    allow_multi_token_rows(runner)
    counts = counts or {}
    if handles is None:
        handles = {r.req_id: i + 1 for i, r in enumerate(new)}
    output = h.scheduler_output(new=tuple(new), handles=handles)
    scheduled = {r.req_id: int(counts.get(r.req_id, 1)) for r in new}
    output = dataclasses.replace(
        output,
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
    )
    return runner.real_step(output)


def token_positions(counts: list[int]) -> list[list[int]]:
    """Each row's token positions for per-row token counts ``counts``
    (``query_start_loc[i]`` to ``query_start_loc[i + 1] - 1``)."""
    starts = np.concatenate([[0], np.cumsum(counts)]).tolist()
    return [list(range(starts[i], starts[i + 1])) for i in range(len(counts))]
