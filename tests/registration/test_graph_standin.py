# SPDX-License-Identifier: Apache-2.0
"""Graphed segment arms on a platform without static graphs (REG-MEM-028).

These run on the real platform runtime of the host when it captures no static
graphs on the CPU device (vLLM's CPU platforms, and a CUDA host for CPU
tensors)."""

from __future__ import annotations

import pytest
import torch

decode_graph = pytest.importorskip(
    "vllm_nemotron_asr_streaming.nemotron_asr.decode_graph"
)


# @spec REG-MEM-028
def test_a_cpu_entry_is_bound_to_the_eager_stand_in():
    runtime = decode_graph.platform_graph_runtime()
    device = torch.device("cpu")
    assert not runtime.captures_on(device)
    descriptor = runtime.descriptor_factory(2)
    calls = []

    def runnable(x):
        calls.append(x)
        return x * 2

    wrapper = runtime.wrap(runnable, None, descriptor, device)
    assert isinstance(wrapper, decode_graph.EagerStaticGraphWrapper)
    assert wrapper.graph_pool is None
    # The entry holds its (non-device) graph from construction, so readiness
    # checks see it as captured.
    assert decode_graph.wrapper_holds_graph(wrapper, descriptor)
    x = torch.arange(4.0)
    assert torch.equal(wrapper(x), runnable(x))
    assert len(calls) == 2
    # Nothing is captured or synchronized on the device.
    with runtime.capture_scope(device):
        pass
    runtime.sync(device)
