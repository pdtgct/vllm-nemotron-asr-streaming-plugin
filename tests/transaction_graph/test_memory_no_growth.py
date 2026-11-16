# SPDX-License-Identifier: Apache-2.0
"""No growth after capture (TXG-MEM-014 to TXG-MEM-016), on CUDA.

The measured span is ``execute`` from entry to return, commit staging
included (``support.txgcap_harness.wrap_execute``). A file row's envelope
preparation, which runs before ``execute`` (TXG-EXEC-011), is measured
separately and identically in both arms (TXG-MEM-016)."""

from __future__ import annotations

import gc

import pytest
import torch
from support import txgcap_harness as t

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

STEPS = 24
BLOCK = 512  # the caching allocator's rounding


def round_up(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK * BLOCK


def drive(env, steps=STEPS):
    """One CHUNK step, then echo steps of the same stream: each covered."""
    core = env.core
    core.step(t.stream_unit(env, "s1"), handles={"s1": 1})
    for _ in range(steps - 1):
        core.step(cached=("s1",), handles={"s1": 1})


# @spec TXG-MEM-014
def test_covered_steps_allocate_only_their_escaped_row_tensor(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    env.core.startup()
    spans = []

    def around(call):
        torch.cuda.synchronize()
        allocated, reserved = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )
        result = call()
        torch.cuda.synchronize()
        spans.append(
            (
                torch.cuda.memory_allocated() - allocated,
                torch.cuda.memory_reserved() - reserved,
                result,
            )
        )
        return result

    t.wrap_execute(monkeypatch, env.model, around)
    drive(env)
    assert len(spans) == STEPS, f"{len(spans)} execute calls for {STEPS} steps"
    for index, (allocated, reserved, result) in enumerate(spans[2:], start=2):
        escaped = (
            round_up(result.numel() * result.element_size())
            if torch.is_tensor(result)
            else 0
        )
        assert reserved == 0, f"step {index}: reserved bytes grew by {reserved}"
        assert 0 <= allocated <= escaped, (
            f"step {index}: {allocated} bytes allocated, escaped tensor {escaped}"
        )


# @spec TXG-MEM-015
def test_execute_reads_no_memory_statistics_and_never_synchronizes(
    on_cuda, monkeypatch
):
    env = on_cuda(n=2)
    env.core.startup()
    inside = {"execute": False}
    violations = []
    events = []

    def guard(name, original):
        def spy(*args, **kwargs):
            if inside["execute"]:
                violations.append(name)
            return original(*args, **kwargs)

        return spy

    for name in (
        "synchronize",
        "mem_get_info",
        "memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
        "reset_peak_memory_stats",
        "empty_cache",
    ):
        monkeypatch.setattr(torch.cuda, name, guard(name, getattr(torch.cuda, name)))
    monkeypatch.setattr(
        torch.cuda.Stream,
        "synchronize",
        guard("Stream.synchronize", torch.cuda.Stream.synchronize),
    )
    original_event_sync = torch.cuda.Event.synchronize

    def event_sync(self):
        if inside["execute"]:
            events.append(self)
        return original_event_sync(self)

    monkeypatch.setattr(torch.cuda.Event, "synchronize", event_sync)
    monkeypatch.setattr(gc, "collect", guard("gc.collect", gc.collect))

    def around(call):
        inside["execute"] = True
        before = len(events)
        try:
            return call()
        finally:
            inside["execute"] = False
            assert len(events) - before <= 1, "more than the pinned-arena fence waited"

    t.wrap_execute(monkeypatch, env.model, around)
    drive(env)
    assert violations == [], sorted(set(violations))


# @spec TXG-MEM-016
def test_file_envelope_preparation_is_measured_outside_execute_in_both_arms(
    on_cuda, monkeypatch
):
    """The file row's envelope preparation (its host-to-device copy) is
    measured around ``FileHostBuffers.stage`` the same way with the option on
    and off, and never inside the ``execute`` span."""

    from vllm_nemotron_asr_streaming.nemotron_asr import model_state_v2

    envelope = {True: [], False: []}
    inside = {"execute": False, "arm": True}
    stage = model_state_v2.FileHostBuffers.stage

    def measured_stage(self, *args, **kwargs):
        assert not inside["execute"], "file envelope preparation ran inside execute"
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        result = stage(self, *args, **kwargs)
        torch.cuda.synchronize()
        envelope[inside["arm"]].append(torch.cuda.memory_allocated() - before)
        return result

    monkeypatch.setattr(model_state_v2.FileHostBuffers, "stage", measured_stage)

    def around(call):
        inside["execute"] = True
        try:
            return call()
        finally:
            inside["execute"] = False

    for on in (True, False):
        inside["arm"] = on
        env = on_cuda(n=2)
        if not on:
            env = env.option_off()
        env.core.startup()
        if on:
            t.wrap_execute(monkeypatch, env.model, around)
        env.core.step(t.file_item(env, "f1"), handles={"f1": 1})
    assert envelope[True] and envelope[True] == envelope[False], envelope
