# SPDX-License-Identifier: Apache-2.0
"""A world-size-1 distributed environment for in-process model tests.

Core's worker initialises the distributed environment and the model-parallel
groups before profiling; a test that drives the model and runner in process
(without ``AsyncLLM``) must do the same, or ``graph_capture`` finds no
tensor/pipeline/data parallel group. Mirrors core's ``dist_init`` fixture
(``tests/conftest.py``): ``init_distributed_environment``, then
``initialize_model_parallel(1, 1)``, torn down with
``cleanup_dist_env_and_memory``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def single_process_distributed(init_file: Path, backend: str) -> Iterator[None]:
    """Initialise world size 1 (rank 0) over a ``file://`` store, then tear down.

    Runs under a default ``VllmConfig`` when none is current, as core requires.
    """
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    if get_current_vllm_config_or_none() is None:
        config_scope = set_current_vllm_config(VllmConfig())
    else:
        config_scope = _nullscope()
    try:
        with config_scope:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{init_file}",
                local_rank=0,
                backend=backend,
            )
            initialize_model_parallel(1, 1)
            yield
    finally:
        cleanup_dist_env_and_memory()


@contextmanager
def _nullscope() -> Iterator[None]:
    yield
