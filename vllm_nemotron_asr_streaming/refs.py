# SPDX-License-Identifier: Apache-2.0
"""Pinned source and API commits for this staging package.

These are git commits on ``pdtgct/vllm``. They are not a vLLM release.
"""

from __future__ import annotations

RFC1_REPO = "pdtgct/vllm"
RFC1_TIP_BRANCH = "persistent-state-primitive-v2"
RFC1_TIP_COMMIT = "8b8327e2eece1df39eb6318eff4c9dc207b3ab61"

SOURCE_BRANCH = "rfc1-5212-vllm"
SOURCE_COMMIT = "43c0bed5bf877dc74de2e1494a3a542f497493de"
SOURCE_TREE = "vllm/models/nemotron_asr"

# Named in the staging plan. The branch was not on pdtgct/vllm when this
# scaffold was written, so none of its trees were read.
INTEGRATION_BRANCH = "rfc12-integration"

# ``git merge-tree`` of the RFC1 tip and the source branch. These paths
# changed on both sides. Do not merge the source branch over the tip.
# The model tree itself is not in this list; it exists only on the source
# branch and has to be ported file by file onto the tip symbols.
MERGE_CONFLICT_PATHS: tuple[str, ...] = (
    ".buildkite/test_areas/models_basic.yaml",
    "pyproject.toml",
    "tests/v1/core/test_async_scheduler.py",
    "tests/v1/core/test_kv_cache_utils.py",
    "tests/v1/core/test_scheduler.py",
    "tests/v1/engine/test_engine_core.py",
    "tests/v1/worker/test_attn_utils.py",
    "tests/v1/worker/test_gpu_model_runner_v2.py",
    "vllm/v1/core/kv_cache_planning.py",
    "vllm/v1/core/sched/output.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/engine/__init__.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/metrics/stats.py",
    "vllm/v1/request.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu/cudagraph_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
)

# Added on both sides with different blobs. Same rule: do not merge.
BOTH_ADDED_PATHS: tuple[str, ...] = ("vllm/v1/core/stream_admission.py",)
