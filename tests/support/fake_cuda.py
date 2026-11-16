# SPDX-License-Identifier: Apache-2.0
"""The CPU stand-ins for the CUDA device: memory readings, synchronization,
streams, events, graphs, core's ``capture_pool`` and the CUDA platform claim.

Every CPU test that needs CUDA's surface takes it from this module:

* ``set_cuda_platform(monkeypatch, cuda)``: vLLM's current platform reports
  CUDA (``is_cuda`` / ``is_cuda_alike``) or not.
* ``FakeCuda.install_memory(monkeypatch)``: only the memory readings, the
  device and stream synchronization, the cache release and the peak resets
  (``torch.cuda`` and ``torch.accelerator``).
* ``FakeCuda.install(monkeypatch)``: all of it, for transaction-graph
  capture: the public PyTorch names the specs fix (TXG-CAP-009: each cell
  owns one ``torch.cuda.CUDAGraph``; TXG-MEM-001/REG-MEM-027: retained bytes
  are the drop in ``torch.cuda.mem_get_info`` free memory, each reading after
  a capture-stream synchronize and ``torch.cuda.empty_cache``; TXG-MEM-002:
  the transient peak from ``reset_peak_memory_stats`` and the peak reserved
  bytes), streams and events, core's
  ``vllm.compilation.cudagraph_pool.capture_pool`` and, with ``claim_cuda``,
  the platform claim and ``current_platform.graph_pool_handle``.
* ``FakeEvent``: ``torch.cuda.Event``. An event made by an installed double
  logs; one made directly (as the model harness's ``finish_step`` does) is
  silent.

Every call is logged as ``("cuda.<name>", detail)`` into one ``events`` list,
which the segment-graph double (``model_harness.FakeGraphRuntime``) can share,
so a test reads one ordered record of the hook call.

Memory model (bytes, all host integers):

* ``mem_get_info`` reports ``free``, less ``per_segment_graph`` bytes for each
  live captured graph of ``segment_graphs`` (a ``FakeGraphRuntime``; collected
  before each reading) when one is attached. A cell capture that completes
  retains ``capture_cost`` bytes (its graph's pool memory) until the graph is
  reset or collected. An allocation of exactly ``margin_bytes`` ``uint8``
  elements (the charged margin, TXG-MEM-004) retains ``margin_bytes +
  margin_rounding`` until the tensor is collected. Each reading lowers
  ``free`` by ``jitter`` first, so with ``jitter`` > 0 every reading is a
  distinct value a test can locate in the log.
* ``reserved`` follows retained graph and margin bytes; a capture also
  reserves ``transient`` bytes while it runs, so the peak exceeds the
  retained total. ``reset_peak_memory_stats`` resets the peak.

Failure injection (``fail``): ``"capture"`` raises inside the ``n``-th cell
capture (0-based, counted over the installed lifetime); ``"replay"`` raises
in the ``n``-th replay; ``"margin"`` raises ``torch.OutOfMemoryError`` from the
margin's allocation; ``"pinned"`` raises from the ``n``-th pinned host
allocation (a binding-owned buffer, TXG-MEM-010). ``on_capture`` is called
once a capture has begun, with the graph (a test may issue a forbidden
operation there, TXG-CAP-002). ``on_replay`` is called after each replay with the graph
(a test may write into pool memory there, e.g. to make a validation replay
differ from its reference, TXG-CAP-004).

Seams this double relies on (requirements on the implementation): the
transaction-graph code looks ``torch.cuda.*``, ``torch.empty``/``torch.zeros``
and ``capture_pool`` up at call time (no ``from torch.cuda import ...`` bound
at import), and allocates pinned host memory only through
``pin_memory=True`` or ``Tensor.pin_memory()``, which the double accepts on
CPU.
"""

from __future__ import annotations

import contextlib
import gc
import itertools
import weakref
from collections.abc import Iterator
from typing import Any

import torch

#: The charged margin's size until the first A40 measurement (TXG-MEM-004).
MARGIN_BYTES = 64 << 20


def set_cuda_platform(monkeypatch: Any, cuda: bool, *, cuda_alike: bool = True) -> None:
    """Make vLLM's current platform report CUDA or not: ``is_cuda`` and, with
    ``cuda_alike``, ``is_cuda_alike``."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda, raising=False)
    if cuda_alike:
        monkeypatch.setattr(
            current_platform, "is_cuda_alike", lambda: cuda, raising=False
        )


class InjectedFault(RuntimeError):
    """The exception a ``FakeCuda`` failure injection raises."""


class FakeStream:
    """``torch.cuda.Stream``; a stream made by ``FakeCuda.new_stream`` logs
    its synchronizations."""

    _ids = itertools.count()

    def __init__(self, *args: Any, name: str | None = None, **kwargs: Any) -> None:
        self.name = name or f"stream-{next(self._ids)}"
        self.fake: FakeCuda | None = None

    def synchronize(self) -> None:
        if self.fake is not None:
            self.fake._log("stream.synchronize", self.name)

    def wait_stream(self, other: Any) -> None:
        if self.fake is not None:
            self.fake._log("stream.wait_stream", self.name)

    def wait_event(self, event: Any) -> None:
        if self.fake is not None:
            self.fake._log("stream.wait_event", self.name)

    def record_event(self, event: Any = None) -> Any:
        if event is None:
            event = self.fake.Event() if self.fake is not None else FakeEvent()
        event.record(self)
        return event

    def query(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"FakeStream({self.name})"


class FakeEvent:
    """``torch.cuda.Event``. ``fake`` is the ``FakeCuda`` an event logs to:
    set on the per-double subclass ``FakeCuda.Event``, ``None`` here."""

    fake: FakeCuda | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.recorded = False

    def record(self, stream: Any = None) -> None:
        self.recorded = True
        if self.fake is not None:
            self.fake._log("event.record", getattr(stream, "name", None))

    def synchronize(self) -> None:
        if self.fake is not None:
            self.fake._log("event.synchronize", None)

    def wait(self, stream: Any = None) -> None:
        if self.fake is not None:
            self.fake._log("event.wait", None)

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: Any) -> float:
        return 0.0


class FakeCUDAGraph:
    """``torch.cuda.CUDAGraph``: capture is a log entry and a retained charge;
    replay is a log entry (the body already ran eagerly inside the capture
    block, so the outputs it wrote stay in place)."""

    def __init__(self, fake: FakeCuda, *args: Any, **kwargs: Any) -> None:
        self._fake = fake
        self.index = len(fake.graphs)
        self.captured = False
        self.pool_handle: Any = None
        self.stream: Any = None
        self.replays = 0
        self._retained = 0
        fake.graphs.append(weakref.ref(self))
        weakref.finalize(self, fake._release_graph, self.index)

    # -- capture -------------------------------------------------------------
    def capture_begin(self, pool: Any = None, *args: Any, **kwargs: Any) -> None:
        fake = self._fake
        if fake.capturing is not None:
            raise RuntimeError("nested torch.cuda graph capture")
        number = fake.capture_attempts
        fake.capture_attempts += 1
        fake._log("graph.capture_begin", (self.index, pool))
        self.pool_handle = pool
        fake.capturing = self
        fake.reserved += fake.transient
        fake.peak_reserved = max(fake.peak_reserved, fake.reserved)
        if fake.fail.get("capture") == number:
            fake.capturing = None
            fake.reserved -= fake.transient
            fake._log("fault", ("capture", number))
            raise InjectedFault(f"injected fault in cell capture {number}")
        if fake.on_capture is not None:
            fake.on_capture(self)

    def capture_end(self) -> None:
        fake = self._fake
        fake._log("graph.capture_end", self.index)
        fake.capturing = None
        fake.reserved -= fake.transient
        self.captured = True
        self._retained = fake.capture_cost
        fake.free -= self._retained
        fake.reserved += self._retained
        fake.peak_reserved = max(fake.peak_reserved, fake.reserved)
        fake.capture_order.append(self.index)

    # -- use -------------------------------------------------------------------
    def replay(self) -> None:
        fake = self._fake
        number = fake.replay_count
        fake.replay_count += 1
        fake._log("graph.replay", self.index)
        if not self.captured:
            raise RuntimeError("replay of a graph that holds no capture")
        self.replays += 1
        if fake.fail.get("replay") == number:
            fake._log("fault", ("replay", number))
            raise InjectedFault(f"injected fault in replay {number}")
        if fake.on_replay is not None:
            fake.on_replay(self)

    def reset(self) -> None:
        self._fake._log("graph.reset", self.index)
        self._drop()

    def _drop(self) -> None:
        if self._retained:
            self._fake.free += self._retained
            self._fake.reserved -= self._retained
            self._retained = 0
        self.captured = False

    def pool(self) -> Any:
        return ("pool-of", self.index)

    def enable_debug_mode(self) -> None:
        pass

    def debug_dump(self, path: str) -> None:
        pass


class FakeCuda:
    def __init__(
        self,
        events: list | None = None,
        *,
        total: int = 48 << 30,
        free: int = 40 << 30,
        segment_graphs: Any = None,
        per_segment_graph: int = 0,
    ) -> None:
        self.events: list = events if events is not None else []
        self.total = total
        self.free = free
        #: A ``FakeGraphRuntime`` whose live captured graphs each hold
        #: ``per_segment_graph`` bytes of the free memory ``mem_get_info``
        #: reports.
        self.segment_graphs = segment_graphs
        self.per_segment_graph = per_segment_graph
        #: ``torch.cuda.Event`` while installed: events that log here.
        self.Event = type("FakeEvent", (FakeEvent,), {"fake": self})
        self.reserved = 1 << 30
        self.peak_reserved = self.reserved
        self.allocated = self.reserved
        #: Bytes a completed cell capture retains, and extra bytes a capture
        #: reserves while it runs.
        self.capture_cost = 1 << 20
        self.transient = 3 << 20
        self.margin_bytes = MARGIN_BYTES
        self.margin_rounding = 0
        self.jitter = 0
        self.fail: dict[str, int] = {}
        self.on_replay: Any = None
        self.on_capture: Any = None
        self.capturing: FakeCUDAGraph | None = None
        self.graphs: list[weakref.ref] = []
        self.capture_order: list[int] = []
        self.capture_attempts = 0
        self.replay_count = 0
        self.margin_allocations = 0
        self.margins: list[weakref.ref] = []
        #: ``(reserved at the latest peak reset, peak since that reset)``.
        self.readings: list[int] = []
        self.peak_resets: list[int] = []
        self.capture_pools: list[Any] = []
        self.streams_entered: list[Any] = []
        self.streams_created = 0
        self.pinned_allocations = 0
        self.pinned_requests = 0
        self.default_stream = FakeStream(name="default")
        self.default_stream.fake = self
        self._current = [self.default_stream]

    # -- log -------------------------------------------------------------------
    def _log(self, name: str, detail: Any = None) -> None:
        self.events.append((f"cuda.{name}", detail))

    def names(self) -> list[str]:
        return [e[0] for e in self.events]

    # -- memory readings -------------------------------------------------------
    def mem_get_info(self, device: Any = None) -> tuple[int, int]:
        self.free -= self.jitter
        free = self.free
        if self.segment_graphs is not None:
            gc.collect()
            free -= self.per_segment_graph * self.segment_graphs.live_graphs()
        self.readings.append(free)
        self._log("mem_get_info", free)
        return free, self.total

    def synchronize(self, device: Any = None) -> None:
        self._log("synchronize", None)

    def empty_cache(self) -> None:
        self._log("empty_cache", None)

    def reset_peak_memory_stats(self, device: Any = None) -> None:
        self.peak_reserved = self.reserved
        self.peak_resets.append(self.reserved)
        self._log("reset_peak_memory_stats", self.reserved)

    def reset_max_memory_allocated(self, device: Any = None) -> None:
        self._log("reset_max_memory_allocated", self.reserved)

    def memory_reserved(self, device: Any = None) -> int:
        self._log("memory_reserved", self.reserved)
        return self.reserved

    def max_memory_reserved(self, device: Any = None) -> int:
        self._log("max_memory_reserved", self.peak_reserved)
        return self.peak_reserved

    def memory_allocated(self, device: Any = None) -> int:
        self._log("memory_allocated", self.reserved)
        return self.reserved

    def max_memory_allocated(self, device: Any = None) -> int:
        self._log("max_memory_allocated", self.peak_reserved)
        return self.peak_reserved

    def memory_stats(self, device: Any = None) -> dict[str, int]:
        self._log("memory_stats", None)
        return {
            "reserved_bytes.all.current": self.reserved,
            "reserved_bytes.all.peak": self.peak_reserved,
            "allocated_bytes.all.current": self.reserved,
            "allocated_bytes.all.peak": self.peak_reserved,
        }

    def transient_peak_since_reset(self) -> int:
        """What TXG-MEM-002 defines, from the latest reset: the peak reserved
        bytes less the reserved bytes at the reset."""
        assert self.peak_resets, "no peak reset was made"
        return self.peak_reserved - self.peak_resets[-1]

    # -- graphs ------------------------------------------------------------------
    def new_graph(self, *args: Any, **kwargs: Any) -> FakeCUDAGraph:
        graph = FakeCUDAGraph(self, *args, **kwargs)
        self._log("graph.new", graph.index)
        return graph

    def live_graphs(self) -> list[FakeCUDAGraph]:
        """Graphs still referenced that hold a capture (``gc.collect()``
        first)."""
        return [g for ref in self.graphs if (g := ref()) is not None and g.captured]

    def _release_graph(self, index: int) -> None:
        self.events.append(("cuda.graph.released", index))

    @contextlib.contextmanager
    def graph(
        self, cuda_graph: Any, pool: Any = None, stream: Any = None, **kwargs: Any
    ) -> Iterator[Any]:
        self._log("graph.context", (getattr(cuda_graph, "index", None), pool, stream))
        self.capture_pools.append(pool)
        if stream is not None:
            self.streams_entered.append(stream)
        cuda_graph.stream = stream
        cuda_graph.capture_begin(pool)
        try:
            yield cuda_graph
        except BaseException:
            self.capturing = None
            raise
        cuda_graph.capture_end()

    def is_current_stream_capturing(self) -> bool:
        return self.capturing is not None

    def graph_pool_handle(self) -> Any:
        self._log("graph_pool_handle", None)
        return ("pool-handle", len(self.events))

    # -- streams -----------------------------------------------------------------
    def new_stream(self, *args: Any, **kwargs: Any) -> FakeStream:
        self.streams_created += 1
        stream = FakeStream(*args, **kwargs)
        stream.fake = self
        self._log("stream.new", stream.name)
        return stream

    def current_stream(self, device: Any = None) -> FakeStream:
        return self._current[-1]

    @contextlib.contextmanager
    def stream(self, stream: Any) -> Iterator[None]:
        self._log("stream.enter", getattr(stream, "name", stream))
        self.streams_entered.append(stream)
        self._current.append(stream)
        try:
            yield
        finally:
            self._current.pop()
            self._log("stream.exit", getattr(stream, "name", stream))

    # -- allocations ---------------------------------------------------------------
    def _wrap_alloc(self, original: Any, name: str) -> Any:
        fake = self

        def alloc(*args: Any, **kwargs: Any) -> Any:
            pinned = bool(kwargs.pop("pin_memory", False))
            dtype = kwargs.get("dtype")
            size = _numel(args, kwargs)
            is_margin = dtype is torch.uint8 and size == fake.margin_bytes
            if is_margin:
                number = fake.margin_allocations
                fake.margin_allocations += 1
                fake._log("margin.allocate", size)
                if fake.fail.get("margin") == number:
                    fake._log("fault", ("margin", number))
                    raise torch.OutOfMemoryError("injected fault in margin allocation")
            if pinned:
                number = fake.pinned_requests
                fake.pinned_requests += 1
                if fake.fail.get("pinned") == number:
                    fake._log("fault", ("pinned", number))
                    raise torch.OutOfMemoryError(
                        "injected fault in a pinned allocation"
                    )
            tensor = original(*args, **kwargs)
            if pinned:
                fake.pinned_allocations += 1
                fake._log("pinned.allocate", tuple(tensor.shape))
            if is_margin:
                charge = fake.margin_bytes + fake.margin_rounding
                fake.free -= charge
                fake.reserved += charge
                fake.peak_reserved = max(fake.peak_reserved, fake.reserved)
                fake.margins.append(weakref.ref(tensor))
                weakref.finalize(tensor, fake._release_margin, charge)
            return tensor

        alloc.__name__ = name
        return alloc

    def _release_margin(self, charge: int) -> None:
        self.free += charge
        self.reserved -= charge
        self.events.append(("cuda.margin.released", charge))

    def live_margins(self) -> int:
        return sum(ref() is not None for ref in self.margins)

    # -- installation --------------------------------------------------------------
    def install_memory(self, monkeypatch: Any) -> FakeCuda:
        """Replace the memory readings, the device synchronization, the cache
        release and the peak resets of ``torch.cuda`` and
        ``torch.accelerator``; nothing else."""
        cuda = torch.cuda
        fake = self
        for name in (
            "mem_get_info",
            "synchronize",
            "empty_cache",
            "reset_peak_memory_stats",
            "reset_max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
            "memory_allocated",
            "max_memory_allocated",
            "memory_stats",
        ):
            monkeypatch.setattr(cuda, name, getattr(self, name))
        accelerator = torch.accelerator
        monkeypatch.setattr(
            accelerator, "synchronize", lambda device=None: fake.synchronize()
        )
        monkeypatch.setattr(accelerator, "empty_cache", lambda: fake.empty_cache())
        monkeypatch.setattr(accelerator, "get_memory_info", self.mem_get_info)
        return self

    def install(self, monkeypatch: Any, *, claim_cuda: bool = True) -> FakeCuda:
        """``install_memory``, graphs, streams, events, allocations and core's
        ``capture_pool``; with ``claim_cuda``, the CUDA platform claim."""
        cuda = torch.cuda
        fake = self
        self.install_memory(monkeypatch)
        for name in (
            "is_current_stream_capturing",
            "graph_pool_handle",
            "current_stream",
            "stream",
        ):
            monkeypatch.setattr(cuda, name, getattr(self, name))
        monkeypatch.setattr(
            cuda, "default_stream", lambda device=None: fake.default_stream
        )
        monkeypatch.setattr(cuda, "CUDAGraph", self.new_graph)
        monkeypatch.setattr(cuda, "graph", self.graph)
        monkeypatch.setattr(cuda, "Stream", self.new_stream)
        monkeypatch.setattr(cuda, "Event", self.Event)
        monkeypatch.setattr(cuda, "current_device", lambda: 0)
        monkeypatch.setattr(cuda, "device_count", lambda: 1)
        monkeypatch.setattr(cuda, "get_device_capability", lambda device=None: (8, 6))
        monkeypatch.setattr(cuda, "get_device_name", lambda device=None: "FakeCuda")
        monkeypatch.setattr(torch, "empty", self._wrap_alloc(torch.empty, "empty"))
        monkeypatch.setattr(torch, "zeros", self._wrap_alloc(torch.zeros, "zeros"))
        monkeypatch.setattr(torch.Tensor, "pin_memory", _pin_memory(self))
        if claim_cuda:
            monkeypatch.setattr(cuda, "is_available", lambda: True)
            from vllm.platforms import current_platform

            set_cuda_platform(monkeypatch, True, cuda_alike=False)
            monkeypatch.setattr(
                current_platform,
                "graph_pool_handle",
                self.graph_pool_handle,
                raising=False,
            )
        self._install_capture_pool(monkeypatch)
        return self

    def _install_capture_pool(self, monkeypatch: Any) -> None:
        import importlib

        fake = self

        @contextlib.contextmanager
        def capture_pool(pool: Any, vllm_config: Any = None) -> Iterator[Any]:
            fake._log("capture_pool.enter", pool)
            try:
                yield pool
            finally:
                fake._log("capture_pool.exit", pool)

        from vllm.compilation import cudagraph_pool

        monkeypatch.setattr(cudagraph_pool, "capture_pool", capture_pool)
        for module_name in (
            "vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph",
            "vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr",
            "vllm_nemotron_asr_streaming.nemotron_asr.decode_graph",
            "vllm_nemotron_asr_streaming.nemotron_asr.encoder_execution",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "capture_pool"):
                monkeypatch.setattr(module, "capture_pool", capture_pool)


def _pin_memory(fake: FakeCuda) -> Any:
    def pin_memory(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        fake.pinned_allocations += 1
        fake._log("pinned.allocate", tuple(self.shape))
        return self

    return pin_memory


def _numel(args: tuple, kwargs: dict) -> int | None:
    size = kwargs.get("size")
    if size is None:
        if not args:
            return None
        size = args[0] if len(args) == 1 else args
    if isinstance(size, int):
        return size
    if isinstance(size, torch.Size | tuple | list):
        total = 1
        for value in size:
            if not isinstance(value, int):
                return None
            total *= value
        return total
    return None
