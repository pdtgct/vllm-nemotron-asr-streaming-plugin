# SPDX-License-Identifier: Apache-2.0
"""Cells that compute on CPU, and a synchronization detector for their bodies.

The fake-capture seam (``support.txgcap_harness``) captures cells whose
replay runs nothing, so a covered step there has no outputs. Two helpers
close that gap without CUDA:

* :class:`EagerCells` makes each cell of a CPU option-on environment replay
  by running the cell's own body (``TransactionGraphBinding._run_cell``, the
  function a real cell captures) over the step's pools, inside
  ``advance.flat_capture()``, so the nested segment bindings take their
  uncaptured paths as they do inside a real capture. A covered step then
  computes exactly what a real replay records: the arena packing, the
  ``int64`` vectors, the gathers by position, the body, its predicated
  scatters and the ``index_copy_`` of the outputs. What it does not test is
  the graph itself (capture, replay of recorded kernels, allocation in the
  graph pool); those are ``gpu`` tests.
* :class:`SyncDetector` is a ``TorchDispatchMode`` that raises on every
  operation that would synchronize the device, or build a tensor from host
  data, inside a graph capture. On CPU every tensor is a host tensor, so the
  detector cannot tell a device read from a host read by device; it treats
  every tensor the same, and a body that passes it reads no tensor value on
  the host at all.

``sync_free_scatter`` replaces the masked scatter's CPU executor, which reads
each row's status and page ID on the host (the CUDA executor is one kernel
launch and reads nothing).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from support import txg_api as api

#: The transaction's pool arguments, the endpoint pools included.
POOL_ARGUMENTS = (
    "channel_pools",
    "time_pools",
    "len_pools",
    "h_pool",
    "c_pool",
    "queue_pool",
    "book_pool",
    "frontend_raw_pool",
    "frontend_mel_pool",
    "frontend_counter_pool",
    "endpoint_history_pool",
    "endpoint_book_pool",
)


class EagerCells:
    """See the module docstring. ``runs`` lists the cell key of every eager
    replay, in order; ``inside(key)`` wraps one replay's body (default: no
    wrapper), for a test that runs the body under a detector."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.binding = api.binding_of(model)
        assert self.binding is not None, "transaction graphs are off"
        self.pools: dict[str, Any] | None = None
        self.runs: list[tuple[int, str, int]] = []
        self.inside: Any = None
        self._programs: dict[tuple[str, int], Any] = {}

    def install(self, monkeypatch: Any) -> EagerCells:
        """After the serving capture: record each step's pools at ``execute``
        and make every cell's ``replay`` run its body."""
        binding = self.binding
        assert binding.cells, "no cell was captured (install after the capture)"
        execute = binding.execute

        def recording_execute(core, input_ids, inputs_embeds, plan, **transaction):
            self.pools = pools_of(transaction)
            return execute(core, input_ids, inputs_embeds, plan, **transaction)

        monkeypatch.setattr(binding, "execute", recording_execute)
        for key, cell in binding.cells.items():
            real = cell.graph.replay

            def replay(__real=real, __key=key) -> None:
                __real()
                self.run(__key)

            monkeypatch.setattr(cell.graph, "replay", replay)
        return self

    def program(self, kind: str, count: int, pools: dict[str, Any]) -> Any:
        """The cell's program over ``pools`` (``_cell_program``), built once.
        Building it runs the prepare phase on the cell's capture row plan,
        which restages the module-level batch statistics; the step's own are
        put back."""
        advance = api.advance_module()
        key = (kind, count)
        if key not in self._programs:
            staged = advance._batch_stats_slot
            self._programs[key] = self.binding._cell_program(
                kind,
                count,
                capacity=int(pools["queue_pool"].shape[0]),
                resident=advance.resident_layout(**pools),
                pools=(),
            )
            advance._batch_stats_slot = staged
        return self._programs[key]

    def run(self, key: tuple[int, str, int], pools: dict[str, Any] | None = None):
        """Run the body of the cell at ``key`` over ``pools`` (default: the
        pools of the step in progress), as a replay of that cell would."""
        advance = api.advance_module()
        pools = self.pools if pools is None else pools
        assert pools is not None, "no step recorded its pools"
        _geometry, kind, count = key
        program = self.program(kind, count, pools)
        self.runs.append(key)
        with advance.flat_capture():
            if self.inside is None:
                self.binding._run_cell(
                    kind, count, program.prepared, pools, program.encoder_transition
                )
            else:
                with self.inside(key):
                    self.binding._run_cell(
                        kind, count, program.prepared, pools, program.encoder_transition
                    )


class CoveredSteps:
    """Runs composed steps (``support.txg_transaction``) as covered steps:
    through the binding's own ``execute`` over the serving pools, with
    :class:`EagerCells` computing each replay.

    ``env`` is a ``TransactionEnv`` built over the option-on model's runner
    (``txg_transaction.env_from_runner``). A cell bakes in the serving pools'
    addresses, so each step's composed pools are written into the serving
    pools before ``execute`` runs and read back after it."""

    def __init__(self, env: Any, cells: EagerCells) -> None:
        assert env.serving is not None, "the environment has no serving pools"
        self.env = env
        self.cells = cells

    def run(self, step: Any, *, plan: Any = None, **overrides: Any) -> Any:
        """The covered step's outcome (``txg_transaction.Outcome``) with the
        serving pools' content after it, or ``None`` when ``execute``
        declined the step."""
        import dataclasses

        from support import txg_transaction as tx
        from support.txg_fakes import RecordingSink

        env, binding = self.env, self.cells.binding
        plan = step.plan if plan is None else plan
        chunks = int(plan.is_chunk.sum())
        if chunks and env.kwargs.get("graph_covers_decode"):
            # As the model's forward sets it for the step's CHUNK count.
            graphs = env.model._decode_graph_binding
            plan = dataclasses.replace(
                plan, execution_tier=graphs.execution_tier(chunks)
            )
        sink = RecordingSink()
        kwargs = dict(env.kwargs)
        kwargs.update(overrides)
        with torch.inference_mode():
            for (name, live), (other, composed) in zip(
                env.serving.flat(), step.pools.flat(), strict=True
            ):
                assert name == other
                live.copy_(composed)
            out = binding.execute(
                env.core,
                step.input_ids.clone(),
                step.inputs_embeds.clone(),
                plan,
                **env.serving.kwargs(),
                **kwargs,
                commit_sink=sink,
                staging=None,
            )
            if out is None:
                return None
            last = sink.last
            return tx.Outcome(
                rows=out.detach().clone(),
                status=last.staged,
                progress=last.progress,
                queue_counts=last.queue_counts,
                pools=env.serving.clone(),
                sink=sink,
            )

    def compare(self, step: Any) -> list[str]:
        """Every difference between the covered step and the option-off
        transaction of the same row plan on a copy of the same pools: the
        escaped rows, the staged status, progress and remaining queue counts,
        and every resident page of every pool."""
        from support import txg_transaction as tx

        off = self.env.run(step)
        on = self.run(step)
        if on is None:
            return ["execute declined the step"]
        problems = []
        for field in ("rows", "status", "progress", "queue_counts"):
            if not tx.same_bits(getattr(on, field), getattr(off, field)):
                problems.append(f"{field} differs from the option-off transaction")
        differing = on.pools.differences(off.pools)
        if differing:
            problems.append(f"resident pages differ in {differing}")
        if len(on.sink.reservations) != 1 or on.sink.last.cancelled:
            problems.append("the covered step did not hold exactly one staged ticket")
        return problems


def build_covered(monkeypatch: Any, run: Any, *, lookahead: int = 0) -> CoveredSteps:
    """Start ``run`` (an ``on_cpu`` environment and its ``CoreOrder``), build
    the transaction environment over its runner while the binding declines
    every step, then install the eager cells."""
    from support import txg_transaction as tx

    env, core = run.env, run.core
    core.startup()
    binding = api.binding_of(env.model)
    with monkeypatch.context() as patch:
        patch.setattr(binding, "execute", lambda *args, **kwargs: None)
        tenv = tx.env_from_runner(
            env.runner,
            env.model,
            env.state,
            lookahead=lookahead,
            capacity=core.capacity,
            monkeypatch=patch,
        )
    return CoveredSteps(tenv, EagerCells(env.model).install(monkeypatch))


def pools_of(transaction: dict[str, Any]) -> dict[str, Any]:
    """The pool arguments of a transaction's keyword arguments."""
    return {name: transaction[name] for name in POOL_ARGUMENTS}


# ---------------------------------------------------------------------------
# The synchronization detector


class SyncError(AssertionError):
    """An operation that synchronizes the device or builds a tensor from host
    data ran where a graph capture would run."""


#: Operator names (``aten::`` overload packets) that read a tensor's values
#: on the host, or whose output shape depends on values, and so synchronize.
SYNCHRONIZING = (
    "_local_scalar_dense",
    "item",
    "nonzero",
    "nonzero_numpy",
    "nonzero_static",
    "argwhere",
    "masked_select",
    "unique",
    "unique_consecutive",
    "_unique",
    "_unique2",
    "unique_dim",
    "is_nonzero",
    "equal",
    "allclose",
)

#: Operator names that build a tensor from Python data (a host-to-device
#: upload when the target is the device).
FROM_HOST_DATA = ("lift_fresh", "lift_fresh_copy", "lift")

#: ``torch.Tensor`` methods that read values on the host without passing
#: through the dispatcher.
HOST_READ_METHODS = ("tolist", "numpy", "__array__")

#: Functions the body calls that work on host tensors only, by construction:
#: each builds its tensors on the literal CPU device and returns Python
#: values, so inside a real capture it touches no device memory. On CPU the
#: detector cannot see that from a tensor's device, so it is paused inside
#: them (``module``, ``function``).
HOST_ONLY = (
    (
        "vllm_nemotron_asr_streaming.nemotron_asr.encoder_execution",
        "encoder_geometry_shape",
    ),
)

_ACTIVE: list[SyncDetector] = []


class SyncDetector(TorchDispatchMode):
    """Raises :class:`SyncError` on each operation a graph capture cannot
    hold: a host read of tensor values (``_local_scalar_dense`` and the
    operators built on it, reached by ``int()``, ``bool()``, ``float()`` and
    ``item()``; ``tolist()`` and ``numpy()``), a data-dependent shape
    (``nonzero``, ``masked_select``, a boolean-mask index) and a tensor built
    from Python data (``torch.tensor``, ``torch.as_tensor`` of a list).
    ``seen`` counts the operations it let through, so a test can tell that
    the body ran under it. Use it through :func:`detecting`, which also
    pauses it inside the :data:`HOST_ONLY` functions."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0
        self._paused = 0
        self._methods: dict[str, Any] = {}

    def __enter__(self) -> SyncDetector:
        for name in HOST_READ_METHODS:
            original = getattr(torch.Tensor, name)
            self._methods[name] = original

            def guarded(tensor, *args, __name=name, __original=original, **kwargs):
                if _ACTIVE and not _ACTIVE[-1]._paused:
                    raise SyncError(f"Tensor.{__name} reads tensor values on the host")
                return __original(tensor, *args, **kwargs)

            setattr(torch.Tensor, name, guarded)
        _ACTIVE.append(self)
        return super().__enter__()

    def __exit__(self, *exc: Any) -> None:
        try:
            super().__exit__(*exc)
        finally:
            _ACTIVE.remove(self)
            for name, original in self._methods.items():
                setattr(torch.Tensor, name, original)
            self._methods = {}

    @contextmanager
    def host_only(self) -> Iterator[None]:
        """Pause the detector: the block works on host tensors only."""
        from torch.utils._python_dispatch import _disable_current_modes

        self._paused += 1
        try:
            with _disable_current_modes():
                yield
        finally:
            self._paused -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__ if hasattr(func, "overloadpacket") else ""
        if name in SYNCHRONIZING:
            raise SyncError(f"{func} reads tensor values on the host")
        if name in FROM_HOST_DATA:
            raise SyncError(f"{func} builds a tensor from Python data")
        if name in ("index", "index_put", "index_put_", "_index_put_impl_"):
            indices = args[1] if len(args) > 1 else kwargs.get("indices", ())
            for index in indices or ():
                if isinstance(index, torch.Tensor) and index.dtype in (
                    torch.bool,
                    torch.uint8,
                ):
                    raise SyncError(f"{func} indexes with a mask (a nonzero)")
        self.seen += 1
        return func(*args, **kwargs)


def pause_in_host_only_functions(monkeypatch: Any) -> None:
    """Wrap each :data:`HOST_ONLY` function so that an active detector is
    paused while it runs."""
    import importlib

    for module_name, name in HOST_ONLY:
        module = importlib.import_module(module_name)
        original = getattr(module, name)

        def paused(*args, __original=original, **kwargs):
            if not _ACTIVE:
                return __original(*args, **kwargs)
            with _ACTIVE[-1].host_only():
                return __original(*args, **kwargs)

        monkeypatch.setattr(module, name, paused)


@contextmanager
def detecting() -> Iterator[SyncDetector]:
    """Run the block under a fresh :class:`SyncDetector`."""
    detector = SyncDetector()
    with detector:
        yield detector


def sync_free_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    """The masked scatter with the CUDA kernel's semantics and no host read:
    a clean row's scratch page replaces its pool page, a failed row stores
    nothing. Page IDs are unique within a call (the row plan's preflight), so
    gathering the current pages, selecting per row and copying them back is
    the same write."""
    if int(scratch.shape[0]) == 0:
        return
    clean = (row_status == 0).reshape(-1, *([1] * (scratch.dim() - 1)))
    current = pool.index_select(0, block_ids)
    pool.index_copy_(0, block_ids, torch.where(clean, scratch, current))


@contextmanager
def sync_free_scatters(monkeypatch: Any) -> Iterator[None]:
    """Bind :func:`sync_free_scatter` wherever the prevalidated executor is
    looked up by name, for the duration of the block."""
    import importlib

    with monkeypatch.context() as patch:
        for module_name in (
            "vllm_nemotron_asr_streaming.nemotron_asr.state_scatter",
            "vllm_nemotron_asr_streaming.nemotron_asr.advance",
        ):
            module = importlib.import_module(module_name)
            patch.setattr(module, "execute_masked_page_scatter_", sync_free_scatter)
        yield
