# SPDX-License-Identifier: Apache-2.0
"""Core's model-state hook order, one call at a time.

``CoreOrder`` drives a ``support.model_harness.Runner`` through the calls
core's GPU worker and model runner make on a model state between loading the
model and shutting the worker down (``vllm/v1/worker/gpu_worker.py``,
``vllm/v1/worker/gpu/model_runner.py``,
``vllm/v1/worker/gpu/cudagraph_utils.py``), in core's order:

1. ``determine_available_memory``: ``prepare_execution_memory``, then the
   profiling forward (``profile_run``, which asks for the preparation again;
   the runner's guard makes that a no-op).
2. ``profile_cudagraph_memory`` (the graph-memory estimate): a temporary
   profiling state pool bound to the state layer with its state-group layout
   (core's ``initialize_kv_cache(..., is_profiling=True)``), then, when
   ``needs_capture()``, ``capture_cudagraphs(profile_only=True)`` into a
   throwaway graph pool, ``clear_cudagraphs`` and core's teardown of the
   profiling KV cache (``_teardown_profiling_state``: the state-group layouts
   emptied, the layer's ``kv_cache`` cleared, a collection).
3. ``initialize_from_config``: the serving state pool, bound the same way.
4. ``compile_or_warm_up_model``: the kernel warm-up dummy step; when
   ``needs_capture()``, ``capture_cudagraphs(profile_only=False)`` into the
   global graph pool; the post-capture dummy step; ``post_warmup``.
5. ``shutdown``.

``needs_capture()`` is core's ``model_state_needs_cudagraph_capture``: the
runner is not eager and the model state asks for a capture.

Every capture runs under ``torch.inference_mode``; on CUDA also inside
``graph_capture``, on its stream. On CPU the stream is the
caller's capture stream (a ``support.fake_cuda.FakeStream``) and the graph
pools are the tokens :data:`PROFILING_POOL` and :data:`SERVING_POOL`; on CUDA
they are ``current_platform.graph_pool_handle()`` and
``current_platform.get_global_graph_pool()``.

``startup`` runs steps 1 to 4; each step is also a method, so a test can stop
before, perturb, repeat or reorder any call.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterator
from typing import Any

import torch

from support import model_harness as h

#: The graph pools a CPU capture receives: the throwaway pool of the
#: graph-memory estimate and core's global pool.
PROFILING_POOL = "profiling-pool"
SERVING_POOL = "global-pool"


class CoreOrder:
    """Core's calls on ``runner``'s model state (see the module docstring).

    ``device`` is ``"cpu"`` or ``"cuda"``; ``capture_stream`` is the stream a
    CPU capture receives; ``capacity`` is the block count of each state pool
    :meth:`bindings` makes; with ``spy``, those pools are wrapped in
    ``model_harness.BindingsSpy``."""

    def __init__(
        self,
        runner: Any,
        *,
        device: Any = "cpu",
        capture_stream: Any = None,
        capacity: int = 8,
        spy: bool = True,
    ) -> None:
        self.runner = runner
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.capture_stream = capture_stream
        self.capacity = capacity
        self.spy = spy
        self.profiling_bindings: Any = None
        self.serving_bindings: Any = None

    @property
    def state(self) -> Any:
        return self.runner.state

    # -- 1. determine_available_memory --------------------------------------

    def prepare(self) -> None:
        """``prepare_execution_memory``, then the profiling forward."""
        self.runner.prepare_execution_memory()
        self.runner.profile_run(core_embeds=False)

    # -- state pools ----------------------------------------------------------

    def bindings(self) -> Any:
        """A fresh state pool for the model's layer."""
        bindings = h.real_bindings(
            h.model_spec(self.runner.model),
            capacity=self.capacity,
            device=self.device.type,
        )
        return h.BindingsSpy(bindings) if self.spy else bindings

    def bind_profiling(self, bindings: Any = None) -> Any:
        """Step 2's temporary profiling pool."""
        self.profiling_bindings = self.runner.bind_pool(bindings or self.bindings())
        return self.profiling_bindings

    def bind_serving(self, bindings: Any = None) -> Any:
        """Step 3's serving pool."""
        self.serving_bindings = self.runner.bind_pool(bindings or self.bindings())
        return self.serving_bindings

    # -- captures -------------------------------------------------------------

    def needs_capture(self) -> bool:
        return (
            not self.runner.vllm_config.model_config.enforce_eager
            and self.state.needs_cudagraph_capture()
        )

    @contextlib.contextmanager
    def _capture_context(self) -> Iterator[Any]:
        with torch.inference_mode():
            if self.cuda:
                from vllm.distributed.parallel_state import graph_capture

                with graph_capture(self.device) as context:
                    yield context.stream
            else:
                yield self.capture_stream

    def _graph_pool(self, profile_only: bool) -> Any:
        if not self.cuda:
            return PROFILING_POOL if profile_only else SERVING_POOL
        from vllm.platforms import current_platform

        return (
            current_platform.graph_pool_handle()
            if profile_only
            else current_platform.get_global_graph_pool()
        )

    def capture(
        self, *, profile_only: bool, graph_pool: Any = None, stream: Any = None
    ) -> None:
        """One ``capture_cudagraphs`` call; ``graph_pool`` and ``stream``
        replace the pool and the stream core would pass."""
        with self._capture_context() as core_stream:
            self.state.capture_cudagraphs(
                graph_pool=(
                    self._graph_pool(profile_only) if graph_pool is None else graph_pool
                ),
                stream=core_stream if stream is None else stream,
                profile_only=profile_only,
            )

    def profiling_capture(self, **kwargs: Any) -> None:
        self.capture(profile_only=True, **kwargs)

    def serving_capture(self, **kwargs: Any) -> None:
        self.capture(profile_only=False, **kwargs)

    def clear(self) -> None:
        """``clear_cudagraphs``, then core's teardown of the profiling KV
        cache (see the module docstring)."""
        self.state.clear_cudagraphs()
        h.release_profiling_kv_cache(self.state, self.runner.model)
        gc.collect()

    # -- 4. compile_or_warm_up_model ----------------------------------------

    def warmup_step(self) -> None:
        """A non-profiling dummy step: the kernel warm-up before the serving
        capture and the dummy forward after it."""
        self.runner.dummy_step(is_profile=False, core_embeds=False)

    def post_warmup(self) -> None:
        with torch.inference_mode():
            self.state.post_warmup()

    # -- the whole startup ----------------------------------------------------

    def startup(
        self, *, profiling: bool = True, warmup: bool = False, verify: bool = True
    ) -> Any:
        """Steps 1 to 4. ``profiling=False`` skips the graph-memory estimate
        (a fixed budget, or the estimate disabled); ``warmup`` adds the dummy
        steps around the serving capture; ``verify=False`` stops before
        ``post_warmup``. Returns the serving pool."""
        self.prepare()
        captures = self.needs_capture()
        if profiling and captures:
            self.bind_profiling()
            self.profiling_capture()
            self.clear()
        self.bind_serving()
        if warmup:
            self.warmup_step()
        if captures:
            self.serving_capture()
        if warmup:
            self.warmup_step()
        if verify:
            self.post_warmup()
        return self.serving_bindings

    def shutdown(self) -> None:
        self.state.shutdown()

    # -- serving --------------------------------------------------------------

    def step(
        self, *new: Any, cached: tuple[str, ...] = (), handles=None, **kw: Any
    ) -> Any:
        """One real step scheduling ``new`` requests (handles ``1..n`` unless
        given) and ``cached`` ones."""
        if handles is None:
            handles = {r.req_id: i + 1 for i, r in enumerate(new)}
        return self.runner.real_step(
            h.scheduler_output(new=tuple(new), cached=cached, handles=handles, **kw)
        )
