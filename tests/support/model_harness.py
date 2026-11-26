# SPDX-License-Identifier: Apache-2.0
"""The real model and model state on CPU, driven as core's model runner drives
them on the target core, ``pdtgct/vllm`` at ``cbf13fed4a``
(``vllm/v1/worker/gpu/model_runner.py``:
``execute_model``, ``_dummy_run``, ``profile_run``, ``_dummy_sampler_run``;
``vllm/model_executor/warmup/kernel_warmup.py`` for later profiling steps).

Real: the configuration (card parser, post-override validation, the engine
hook), the model class (constructed as core's loader does, frozen for
inference, without loading weights), its ``ModelState``, core's
``EncoderCache`` / ``EncoderRunner`` (so ``prepare_inputs_embeds`` runs the
model's own multimodal embedding), the state layer's ``MambaBase``
``bind_kv_cache`` over a CPU pool (``StatePool``, laid out as core's
``MambaManager`` pool: ``[blocks, 1, 1, page]`` bytes), ``SchedulerOutput``,
``NewRequestData`` (its ``block_ids`` carry the request's slot),
``InputBatch``, ``set_forward_context`` and ``BatchDescriptor``, the profile
transaction and the row transaction (scatter and commit included).

The card is ``write_runnable_card``: the synthetic card with an encoder wide
enough (1024, two layers) for the streaming transition to run on CPU.

Steps run under ``torch.inference_mode()``, as core's ``execute_model``,
``_dummy_run`` and ``_dummy_sampler_run`` do.

Omissions from core's step, by design of a CPU harness:

* Block tables, slot mappings and attention groups: the model declares no
  attention layers, so ``prepare_attn`` receives empty tables and no groups,
  and ``preprocess_state`` (a no-op for this model) receives none.
* ``update_requests`` (cached-request bookkeeping in the runner's own
  ``RequestState``), the LoRA, speculative, pooling, PP and KV-connector
  branches, and core's ``Sampler`` (a GPU module): real steps call
  ``compute_sample_logits`` but sample nothing; dummy sampling uses the
  caller's recording sampler (``RecordingSampler``).
* Core's ``ModelRunnerOutput``: after ``postprocess_state`` the runner asks
  ``take_failed_requests`` once per step, as core's does, and records the
  answer in ``Runner.failed_log`` (REG-MS-004); it builds no output.
* ``StepRequestStates`` stands in for the runner's ``RequestState``; the model
  state reads only ``req_id_to_index``.

Seams the doubles rely on (call-time lookups in the plugin):

* ``profile_execution.run_persistent_state_profile`` (``ProfileSpy`` wraps it
  and calls through, and the tests read its ``num_rows`` and ``geometry_id``
  keyword arguments to identify the transaction's shape, and its
  ``ready_domain=True`` keyword to tell the final, readiness-gated profile
  transaction from a warm-up invocation; see ``is_final`` in
  ``tests/registration/test_execution_memory.py``);
* ``advance.advance_model_rows`` (``AdvanceSpy`` wraps it and calls through);
* ``decode_graph.platform_graph_runtime``, also bound by name in
  ``chunk_bucket_graph`` and ``encoder_execution`` (``FakeGraphRuntime``
  replaces it, since CPU cannot capture graphs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

#: The model's state layer name for ``prefix="nemotron"`` (``f"{prefix}.state"``).
STATE_LAYER = "nemotron.state"
EPOCH = "epoch-1"

#: Encoder fields of the runnable synthetic card.
RUNNABLE_ENCODER = {
    "hidden_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 4096,
}


def write_runnable_card(path: Path) -> Path:
    from support.card import card_config, write_card

    config = card_config()
    config["encoder_config"].update(RUNNABLE_ENCODER)
    config["prompt_intermediate_size"] = 2 * RUNNABLE_ENCODER["hidden_size"]
    return write_card(path, config=config)


# ---------------------------------------------------------------------------
# Configuration, model, model state


def build_vllm_config(
    card_dir: Path,
    *,
    right_context: int = 6,
    max_num_seqs: int = 2,
    max_num_batched_tokens: int = 64,
    overrides: dict[str, Any] | None = None,
    device: str = "cpu",
    enforce_eager: bool = True,
    sizing: str | None = None,
    blocks: int | None = None,
    budget: int | None = None,
) -> Any:
    """The engine configuration of a server at lookahead ``right_context``.

    The server serves one geometry (REG-LA-001), so no served lookahead set
    (``supported_num_lookahead_tokens``, refused by REG-LA-008) is passed;
    ``overrides`` are further ``--hf-overrides`` keys. ``enforce_eager=False``
    builds the hook configuration (REG-ENG-005: ``-cc.mode none`` and
    ``-cc.cudagraph_mode none``), which transaction graphs require
    (TXG-CFG-009); ``support.core_order.CoreOrder`` then runs core's capture
    hooks. The CPU tests keep the eager runner. While ``overrides`` turn
    transaction graphs on, the configuration is the restricted startup
    (TXG-CFG-017, TXG-CFG-018: synchronous scheduling and both modes NONE)
    under core's default KV-cache sizing. ``sizing`` (``txg_api.Sizing``,
    with ``blocks`` for a fixed block count and ``budget`` for a fixed
    budget) passes that sizing's engine arguments
    (``txg_api.sizing_arguments``) to the cache configuration as it is
    built, so core's validators and the plugin's configuration hook see it;
    a test names a sizing only where its spec names one. Core resolves the
    extensible KV cache off on the CPU platform and refuses it requested
    there, so a CPU test sets that sizing on the built configuration
    (``txgcap_harness.set_cache``)."""
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    from support.txg_api import sizing_arguments
    from vllm_nemotron_asr_streaming import register

    register()
    cache: dict[str, Any] = {}
    if sizing is not None:
        cache = sizing_arguments(sizing, blocks=blocks, budget=budget)
    elif blocks is not None or budget is not None:
        raise ValueError("blocks and budget take a named sizing")
    hf_overrides: dict[str, Any] = {"num_lookahead_tokens": right_context}
    hf_overrides.update(overrides or {})
    model_config = ModelConfig(
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        hf_overrides=hf_overrides,
        enforce_eager=enforce_eager,
        dtype="float32",
        max_model_len=2048,
    )
    compilation: dict[str, Any] = {}
    if not enforce_eager:
        compilation["compilation_config"] = CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
        )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device=device),
        cache_config=CacheConfig(enable_prefix_caching=False, **cache),
        **compilation,
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=False,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max(max_num_batched_tokens, max_num_seqs),
            enable_chunked_prefill=True,
        ),
    )


def build_model(vllm_config: Any) -> Any:
    from vllm.config.vllm import set_current_vllm_config

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    device = torch.device(vllm_config.device_config.device)
    with set_current_vllm_config(vllm_config), torch.device(device):
        model = Nemotron3_5AsrStreamingForRNNT(
            vllm_config=vllm_config, prefix="nemotron"
        )
    return model.to(device).requires_grad_(False).eval()


def build_model_state(vllm_config: Any, model: Any) -> Any:
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    device = torch.device(vllm_config.device_config.device)
    state_cls = type(model).get_model_state_cls()
    return state_cls(vllm_config, model, EncoderCache(), device)


def model_spec(model: Any) -> Any:
    """The model's state layout (the one its state layer declares)."""
    return model._persistent_state_layer.layout


# ---------------------------------------------------------------------------
# Spies, traps and fakes


class Events(list):
    """An ordered event log shared by the spies."""

    def kinds(self) -> list[str]:
        return [e[0] for e in self]


class ProfileSpy:
    """Wraps the profile transaction; records each call and calls through.

    ``fail`` makes the next call raise after recording; ``before`` is called
    with the kwargs before the real transaction runs."""

    def __init__(self, events: Events | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events = events if events is not None else Events()
        self.fail: BaseException | None = None
        self.before: Callable[[dict[str, Any]], None] | None = None
        self._real: Any = None

    def __call__(self, model: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        self.events.append(("profile", kwargs))
        if self.before is not None:
            self.before(kwargs)
        if self.fail is not None:
            raise self.fail
        return self._real(model, **kwargs)

    def install(self, monkeypatch: Any) -> ProfileSpy:
        from vllm_nemotron_asr_streaming.nemotron_asr import profile_execution

        self._real = profile_execution.run_persistent_state_profile
        monkeypatch.setattr(profile_execution, "run_persistent_state_profile", self)
        return self


class AdvanceSpy:
    """Wraps the row transaction; records each call and calls through.
    ``before(kwargs)`` runs first (e.g. to make the step stale)."""

    def __init__(self, events: Events | None = None) -> None:
        self.calls: list[Any] = []
        #: The keyword arguments of each call, in order (``calls`` holds the
        #: row plans).
        self.kwargs: list[dict[str, Any]] = []
        self.events = events if events is not None else Events()
        self.before: Callable[[dict[str, Any]], None] | None = None
        self._real: Any = None

    def __call__(self, core, input_ids, inputs_embeds, plan, **kwargs):
        self.calls.append(plan)
        self.kwargs.append(dict(kwargs))
        self.events.append(("advance", plan))
        if self.before is not None:
            self.before(kwargs)
        return self._real(core, input_ids, inputs_embeds, plan, **kwargs)

    def install(self, monkeypatch: Any) -> AdvanceSpy:
        # ``profile_execution`` binds ``advance_model_rows`` by name at import.
        # Import it before patching, so it binds the real row transaction: a
        # binding captured while a spy was installed would outlive the test's
        # monkeypatch, and the spy's records (and through them the model)
        # would stay alive for the rest of the process. The profile
        # transaction therefore never reaches a spy.
        from vllm_nemotron_asr_streaming.nemotron_asr import (  # noqa: F401
            advance,
            profile_execution,
        )

        self._real = advance.advance_model_rows
        monkeypatch.setattr(advance, "advance_model_rows", self)
        return self


def drop_advance_records() -> None:
    """Drop what every installed :class:`AdvanceSpy` recorded (each spy
    installed later wraps the one before it), through both bindings of the
    row transaction. The recorded keyword arguments hold the pools of every
    call the spies saw, so a test that measures the device memory the model
    retains drops them first. The profile transaction calls
    the real function (``profile_execution`` binds it before any spy is
    installed), so no spy records its scratch."""
    from vllm_nemotron_asr_streaming.nemotron_asr import (
        advance,
        profile_execution,
    )

    for spy in (advance.advance_model_rows, profile_execution.advance_model_rows):
        while isinstance(spy, AdvanceSpy):
            spy.calls.clear()
            spy.kwargs.clear()
            spy = spy._real


class Trap:
    """Forwards to ``target`` until ``armed``; then any attribute read fails.
    ``reads`` records every attribute read."""

    def __init__(self, target: Any = None, name: str = "trap") -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "armed", False)
        object.__setattr__(self, "reads", [])

    def __getattr__(self, attr: str) -> Any:
        self.reads.append(attr)
        if self.armed:
            raise AssertionError(f"{self._name}.{attr} was read")
        return getattr(self._target, attr)


class StatePool:
    """A state pool as core's ``MambaManager`` allocates it for the model's
    layer: ``capacity`` pages of ``layout.page_size_bytes`` bytes, viewed as
    ``[capacity, 1, 1, page]`` for ``MambaBase.bind_kv_cache``. Block 0 is the
    null block."""

    def __init__(self, layout: Any, *, capacity: int = 8, device: str = "cpu"):
        self.layout = layout
        self.capacity = capacity
        self.tensor = torch.zeros(
            (capacity, 1, 1, layout.page_size_bytes), dtype=torch.int8, device=device
        )


class BindingsSpy:
    """Wraps a ``StatePool``; records every member the model reads (none:
    the model reads the state layer's views)."""

    def __init__(self, bindings: Any) -> None:
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "bind_calls", [])
        object.__setattr__(self, "reads", [])
        object.__setattr__(self, "row_reads", [])
        object.__setattr__(self, "phase", None)

    def __getattr__(self, attr: str) -> Any:
        self.reads.append((attr, self.phase))
        return getattr(self._bindings, attr)


def core_bindings(bindings: Any) -> Any:
    while True:
        if isinstance(bindings, BindingsSpy):
            bindings = object.__getattribute__(bindings, "_bindings")
        elif isinstance(bindings, Trap):
            bindings = object.__getattribute__(bindings, "_target")
        else:
            return bindings


def real_bindings(spec: Any, *, capacity: int = 8, device: str = "cpu") -> Any:
    """A ``StatePool`` for the model's layer."""
    return StatePool(spec, capacity=capacity, device=device)


def pool_tensor(bindings: Any) -> torch.Tensor:
    """The pool's bytes, flat."""
    return core_bindings(bindings).tensor.view(torch.uint8).reshape(-1)


def slot_bytes(bindings: Any, slot: int) -> torch.Tensor:
    page = core_bindings(bindings).layout.page_size_bytes
    return pool_tensor(bindings)[slot * page : (slot + 1) * page]


def state_layer_of(model: Any) -> Any:
    """The model's state layer, through a ``support.state_layer.ViewTrap``
    standing in for it."""
    layer = model._persistent_state_layer
    return getattr(layer, "_layer", layer)


def state_group_layouts(layer: Any) -> tuple[Any, ...]:
    """What core's ``get_state_group_layouts`` returns for the model's one
    state group: its layer's bound ``kv_cache`` views, in order."""
    from types import MappingProxyType

    from vllm.v1.worker.utils import StateComponentLayout, StateGroupLayout

    components = tuple(StateComponentLayout(view) for view in layer.kv_cache)
    return (StateGroupLayout(0, MappingProxyType({layer.prefix: components})),)


def release_profiling_kv_cache(state: Any, model: Any) -> None:
    """Core's teardown of the profiling KV cache after ``clear_cudagraphs``
    (``_teardown_profiling_state``): the state-group layouts emptied, the
    layer's ``kv_cache`` cleared (``clear_layer_kv_caches``), a collection."""
    import gc

    from vllm.v1.worker.utils import clear_layer_kv_caches

    state._state_group_layouts = ()
    clear_layer_kv_caches([state_layer_of(model)])
    gc.collect()


def kv_cache_config(model: Any, num_blocks: int) -> Any:
    """The ``KVCacheConfig`` fields the model state reads: one group with the
    state layer."""
    layer = model._persistent_state_layer
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=[layer.prefix])],
    )


class RecordingSampler:
    """A working base sampler for dummy sampling: returns a ``SamplerOutput``
    of the decision each logits row selects, and records every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        self.calls.append("sample")
        rows = logits.shape[0]
        return SamplerOutput(
            sampled_token_ids=logits.argmax(dim=-1, keepdim=True),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(rows, dtype=torch.int32, device=logits.device),
            num_rejected=torch.zeros(rows, dtype=torch.int32, device=logits.device),
        )

    def add_request(self, req_index: int, params: Any) -> None:
        self.calls.append("add_request")

    def apply_staged_writes(self) -> None:
        self.calls.append("apply_staged_writes")


# ---------------------------------------------------------------------------
# Graph runtime double


class FakeGraph:
    """Stands for a captured graph."""


class FakeGraphWrapper:
    """Core's static graph wrapper's contract (``CUDAGraphWrapper``):
    ``concrete_cudagraph_entries`` keyed by batch descriptor; under a forward
    context whose mode is the wrapper's runtime mode, an entry without a
    graph is captured, and one with a graph is replayed. Capture and replay
    run the runnable eagerly and are logged. The runtime holds wrappers and
    graphs only weakly, so a discarded inventory can be observed as gone."""

    def __init__(self, runtime: FakeGraphRuntime, runnable, vllm_config, runtime_mode):
        from vllm.compilation.cuda_graph import CUDAGraphEntry

        #: Core's wrapper binds the global pool at construction.
        self.graph_pool = None
        self._runtime = runtime
        self._entry_cls = CUDAGraphEntry
        self.runnable = runnable
        self.runtime_mode = runtime_mode
        self.concrete_cudagraph_entries: dict[Any, Any] = {}
        runtime.wrappers.append(weakref.ref(self))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return self.runnable(*args, **kwargs)
        context = get_forward_context()
        mode = context.cudagraph_runtime_mode
        if mode == CUDAGraphMode.NONE or mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)
        descriptor = context.batch_descriptor
        entry = self.concrete_cudagraph_entries.setdefault(
            descriptor, self._entry_cls(batch_descriptor=descriptor)
        )
        runtime = self._runtime
        if entry.cudagraph is None:
            index = len(runtime.captures)
            key = (id(self), descriptor)
            runtime.events.append(("capture", descriptor))
            runtime.captures.append((weakref.ref(self), descriptor))
            if runtime.skip_capture == index and runtime.skipped is None:
                runtime.skipped = key
                runtime.skipped_ref = (weakref.ref(self), descriptor)
            if (
                runtime.fail_capture_after is not None
                and runtime.captures_completed >= runtime.fail_capture_after
                and runtime.failed_capture is None
            ):
                runtime.failed_capture = index
                raise RuntimeError(f"capture {index} failed")
            output = self.runnable(*args, **kwargs)
            if runtime.skipped != key:
                graph = FakeGraph()
                runtime.graphs.append(weakref.ref(graph))
                entry.cudagraph = graph
            else:
                # immutable evidence, independent of the wrapper's lifetime
                runtime.skip_stored_nothing = True
            entry.output = output
            runtime.captures_completed += 1
            return output
        runtime.events.append(("replay", descriptor))
        runtime.replays.append((weakref.ref(self), descriptor))
        return self.runnable(*args, **kwargs)


class FakeGraphRuntime:
    """Replaces ``platform_graph_runtime`` with core's forward context and
    batch descriptors plus a capture double (``FakeGraphWrapper``)."""

    def __init__(self, events: Events | None = None) -> None:
        self.events = events if events is not None else Events()
        #: Weak references to every wrapper built and every graph stored.
        self.wrappers: list[weakref.ref] = []
        self.graphs: list[weakref.ref] = []
        #: ``(wrapper ref, descriptor)`` of every capture attempt / replay.
        self.captures: list[tuple[weakref.ref, Any]] = []
        self.replays: list[tuple[weakref.ref, Any]] = []
        #: Sticky skip: the entry ``(wrapper, descriptor)`` of capture attempt
        #: ``skip_capture`` is selected, and every capture of it, then or
        #: later, stores no graph.
        self.skip_capture: int | None = None
        self.skipped: tuple[int, Any] | None = None
        self.skipped_ref: tuple[weakref.ref, Any] | None = None
        self.skip_stored_nothing = False
        #: Fail the first capture attempted once ``fail_capture_after``
        #: captures have completed.
        self.fail_capture_after: int | None = None
        self.failed_capture: int | None = None
        self.captures_completed = 0
        #: Fail the first eager warm-up call (forward context in mode NONE
        #: entered through the runtime) made once ``fail_eager_after`` eager
        #: calls have completed.
        self.fail_eager_after: int | None = None
        self.failed_eager: int | None = None
        self.eager_calls = 0
        self.eager_completed = 0
        #: The stream passed to each capture context (``None``: core's own).
        self.capture_streams: list[Any] = []

    def disarm(self) -> None:
        self.skip_capture = self.fail_capture_after = self.fail_eager_after = None

    def skipped_entry_has_graph(self) -> bool | None:
        """Whether the selected skipped entry holds a graph (``None`` if no
        entry was selected or its wrapper is gone)."""
        if self.skipped_ref is None or self.skipped_ref[0]() is None:
            return None
        wrapper, descriptor = self.skipped_ref[0](), self.skipped_ref[1]
        entry = wrapper.concrete_cudagraph_entries.get(descriptor)
        return entry is not None and entry.cudagraph is not None

    def build(self) -> Any:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context

        from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import (
            GraphRuntime,
        )

        runtime = self

        @contextlib.contextmanager
        def capture_context(device, stream=None):
            runtime.events.append(("capture_context", str(device)))
            runtime.capture_streams.append(stream)
            yield

        @contextlib.contextmanager
        def forward_context(*args, **kwargs):
            eager = kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.NONE
            if eager:
                index = runtime.eager_calls
                runtime.eager_calls += 1
                runtime.events.append(("eager", kwargs.get("batch_descriptor")))
                if (
                    runtime.fail_eager_after is not None
                    and runtime.eager_completed >= runtime.fail_eager_after
                    and runtime.failed_eager is None
                ):
                    runtime.failed_eager = index
                    raise RuntimeError(f"eager warm-up {index} failed")
            with set_forward_context(*args, **kwargs):
                yield
            if eager:
                runtime.eager_completed += 1

        values = {
            "wrapper_factory": lambda runnable, vllm_config, runtime_mode=None, **kw: (
                FakeGraphWrapper(runtime, runnable, vllm_config, runtime_mode)
            ),
            "forward_context": forward_context,
            "capture_context": capture_context,
            "descriptor_factory": lambda tier: BatchDescriptor(
                num_tokens=tier, num_reqs=tier, uniform=True
            ),
            "eager_mode": CUDAGraphMode.NONE,
            "graph_mode": CUDAGraphMode.PIECEWISE,
            "synchronize": lambda device: None,
            # Present only while the model's graph runtime still has a
            # capture-enabled switch (REG-MEM-007); filled with a no-op.
            "set_capture_enabled": lambda enabled: None,
        }
        names = {f.name for f in dataclasses.fields(GraphRuntime)}
        return GraphRuntime(**{k: v for k, v in values.items() if k in names})

    def install(self, monkeypatch: Any) -> FakeGraphRuntime:
        from vllm_nemotron_asr_streaming.nemotron_asr import (
            chunk_bucket_graph,
            decode_graph,
            encoder_execution,
        )

        for module in (decode_graph, chunk_bucket_graph, encoder_execution):
            if hasattr(module, "platform_graph_runtime"):
                monkeypatch.setattr(module, "platform_graph_runtime", self.build)
        return self

    def captured(self) -> list[tuple[Any, Any]]:
        """``(wrapper, descriptor)`` of every entry of a live wrapper that
        holds a graph."""
        return [
            (w, d)
            for ref in self.wrappers
            if (w := ref()) is not None
            for d, e in w.concrete_cudagraph_entries.items()
            if e.cudagraph is not None
        ]

    def live_graphs(self) -> int:
        """Graphs still reachable from anything (call ``gc.collect()`` first)."""
        return sum(ref() is not None for ref in self.graphs)


# ---------------------------------------------------------------------------
# Scheduler output and requests


def handle(slot: int, generation: int = 1) -> Any:
    """The model state's ``SlotHandle``. The generation is the model state's
    own counter (1 for the first fresh request it records, and so on)."""
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import SlotHandle

    return SlotHandle(slot, generation)


def _slot(slot: int | tuple[int, int]) -> int:
    return int(slot[0] if isinstance(slot, tuple) else slot)


def with_slot(request: Any, slot: int | tuple[int, int]) -> Any:
    """``request`` with its one ``MambaManager`` block (``block_ids``).
    Attributes set on the request beyond its dataclass fields (a test's
    endpoint policy) are kept: ``dataclasses.replace`` copies fields only."""
    new = dataclasses.replace(request, block_ids=([_slot(slot)],))
    fields = {field.name for field in dataclasses.fields(request)}
    for name, value in vars(request).items():
        if name not in fields:
            setattr(new, name, value)
    return new


def stream_envelope(
    config: Any = None,
    *,
    geometry: int = 3,
    prompt: int = 0,
    sequence: int = 0,
    final: bool = False,
    valid_samples: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """One stream unit's carrier row (SESS-PACK-001): header then samples, the
    header in ``manifests.ENVELOPE_HEADER_FIELDS`` order (version 2, valid
    samples, geometry, final tail, prompt, chunk sequence, admission stamp),
    as the session packer builds it.

    ``final`` sets the final-tail slot. A final tail carries fewer samples
    than a full chunk, so a caller that builds one passes ``valid_samples``
    below the chunk width; the default is a full chunk. The admission stamp
    is the current wall clock modulo the envelope's period. ``config`` is
    accepted for the callers that pass it and is not read."""
    import time

    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        ADMISSION_EPOCH_MODULUS_MS,
        CADENCES,
        ENVELOPE_HEADER_FIELDS,
        RAW_SAMPLES_PER_CHUNK,
    )

    del config
    width = RAW_SAMPLES_PER_CHUNK[tuple(CADENCES)[geometry]]
    valid = width if valid_samples is None else int(valid_samples)
    values = {
        "version": 2,
        "valid_samples": valid,
        "geometry_id": geometry,
        "final_tail": int(final),
        "prompt_index": prompt,
        "chunk_sequence": sequence,
        "admission_ms_mod": int(time.time() * 1000) % ADMISSION_EPOCH_MODULUS_MS,
    }
    row = torch.zeros(7 + width, dtype=torch.float32)
    row[:7] = torch.tensor([float(values[n]) for n in ENVELOPE_HEADER_FIELDS])
    row[7 : 7 + valid] = torch.from_numpy(
        np.random.default_rng(seed + sequence).standard_normal(valid).astype(np.float32)
        * 0.1
    )
    return row


def new_request(
    req_id: str,
    token: int,
    *,
    audio: torch.Tensor | None = None,
    samples: np.ndarray | None = None,
    header: list[int] | None = None,
    identifier: str | None = None,
) -> Any:
    """``NewRequestData`` for a stream unit (``audio``: its envelope row) or a
    complete file (``samples`` and its 3-value ``header``)."""
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFeatureSpec,
        MultiModalFieldElem,
        MultiModalKwargsItem,
        PlaceholderRange,
    )
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData

    field = MultiModalBatchedField(keep_on_cpu=True)
    items = None
    if audio is not None:
        items = {"audio": MultiModalFieldElem(audio, field)}
    elif samples is not None:
        items = {
            "samples": MultiModalFieldElem(torch.from_numpy(samples), field),
            "file_header": MultiModalFieldElem(
                torch.tensor(header, dtype=torch.int64), field
            ),
        }
    features = []
    if items is not None:
        features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem(items),
                modality="audio",
                identifier=identifier or f"uuid-{req_id}",
                mm_position=PlaceholderRange(offset=0, length=1),
            )
        )
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[token],
        mm_features=features,
        sampling_params=SamplingParams(max_tokens=100),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prefill_token_ids=[token],
    )


def with_extra_args(request: Any, extra_args: dict[str, Any] | None) -> Any:
    """``request`` whose sampling parameters (a copy) carry ``extra_args``,
    as a direct engine client sets them; the endpoint policy is read from
    ``extra_args["endpoint_policy"]`` (REG-STATE-020)."""
    from vllm.sampling_params import SamplingParams

    params = request.sampling_params
    params = SamplingParams(max_tokens=100) if params is None else params.clone()
    params.extra_args = extra_args
    return dataclasses.replace(request, sampling_params=params)


def scheduler_output(
    *,
    new: tuple = (),
    cached: tuple[str, ...] = (),
    finished: tuple[str, ...] = (),
    handles: dict[str, int | tuple[int, int]] | None = None,
    cleanup: dict[str, int | tuple[int, int]] | None = None,
    step_seq: int = 1,
    encoder_inputs: dict[str, list[int]] | None = None,
    computed: dict[str, int] | None = None,
) -> Any:
    """A real ``SchedulerOutput`` as core builds it without pipeline
    parallelism: cached requests carry no token slices (``new_token_ids`` is
    empty), and ``computed`` gives each cached request's ``num_computed_tokens``
    (default 1). ``handles`` maps request IDs to a slot or ``(slot,
    generation)``; a new request's slot becomes its ``block_ids`` (the
    generation is the model state's, so the given one is not sent).
    ``cleanup`` and ``step_seq`` are accepted and not sent (the target core
    has neither). When ``encoder_inputs`` is ``None``, every new request with a
    multimodal feature has its feature scheduled, as core schedules it on
    first sight."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    del cleanup, step_seq
    slots = handles or {}
    new = tuple(with_slot(r, slots[r.req_id]) if r.req_id in slots else r for r in new)
    if encoder_inputs is None:
        encoder_inputs = {r.req_id: [0] for r in new if r.mm_features}
    scheduled = {r.req_id: 1 for r in new} | {r: 1 for r in cached}
    output = SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(cached),
            new_token_ids=[],
            new_block_ids=[None for _ in cached],
            num_computed_tokens=[(computed or {}).get(r, 1) for r in cached],
            num_output_tokens=[1 for _ in cached],
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs=dict(encoder_inputs),
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )
    return output


def drive_step(state: Any, output: Any) -> None:
    """The model-state calls core's ``execute_model`` makes before it builds
    the batch: ``remove_request`` for each finished request (sorted), then
    ``remove_request`` and ``add_request`` for each new one; on a step with
    scheduled tokens, ``prepare_attn`` (which decides a pending removal, as
    the step's later calls would)."""
    from vllm.config import CUDAGraphMode

    finished = set(output.finished_req_ids) | set(
        getattr(output, "preempted_req_ids", None) or ()
    )
    for req_id in sorted(finished):
        state.remove_request(req_id)
    for index, request in enumerate(output.scheduled_new_reqs):
        state.remove_request(request.req_id)
        state.add_request(index, request)
    if output.total_num_scheduled_tokens:
        state.prepare_attn(None, CUDAGraphMode.NONE, (), None, [], None)


class StepRequestStates:
    """Stands for the runner's ``RequestState``: ``req_id_to_index``."""

    def __init__(self) -> None:
        self.req_id_to_index: dict[str, int] = {}

    def add(self, req_id: str) -> int:
        free = sorted(
            set(range(len(self.req_id_to_index) + 1))
            - set(self.req_id_to_index.values())
        )
        self.req_id_to_index[req_id] = free[0]
        return free[0]

    def remove(self, req_id: str) -> None:
        self.req_id_to_index.pop(req_id, None)


def dummy_batch(state: Any, num_reqs: int, num_tokens: int, *, is_padding=True):
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    buffers = InputBuffers(state.max_num_reqs, state.max_num_tokens, state.device)
    return InputBatch.make_dummy(num_reqs, num_tokens, buffers, is_padding=is_padding)


# ---------------------------------------------------------------------------
# The runner


def collection_parts(result: Any) -> tuple[dict[str, int], set[str], tuple]:
    """``(statuses, failed, queue_counts)`` of the ``CommitCollection`` that
    ``collect_commit_status`` returns (FILE-QC-005; fields ``statuses``,
    ``failed`` and ``queue_counts``). Any other return value is refused, so
    no test can pass on a model that returns another shape."""
    if not all(
        hasattr(result, name) for name in ("statuses", "failed", "queue_counts")
    ):
        raise TypeError(
            "collect_commit_status must return a CommitCollection, got "
            f"{type(result).__name__}"
        )
    return result.statuses, result.failed, tuple(result.queue_counts)


def replace_collection(result: Any, **changes: Any) -> Any:
    """A copy of a ``CommitCollection`` with fields replaced (for
    ``Runner.collect_hook``)."""
    return dataclasses.replace(result, **changes)


class Runner:
    """Core's model runner calls on the model and model state, one at a time
    (``support.core_order.CoreOrder`` puts them in core's startup order)."""

    def __init__(self, vllm_config: Any, model: Any, state: Any) -> None:
        self.vllm_config = vllm_config
        self.model = model
        self.state = state
        self.bindings: Any = None
        self._execution_memory_prepared = False
        self.req_states = StepRequestStates()
        self._prompts: dict[str, list[int]] = {}
        #: The token fed at each position of each live request's current
        #: prompt, and its multimodal features: the history core compacts
        #: when it folds the request's next turn in.
        self._fed: dict[str, list[int]] = {}
        self._features: dict[str, list[Any]] = {}
        #: Where each request's current turn starts in its prompt: tests
        #: count a turn's cached positions from 0, as an earlier core did;
        #: the harness adds this base (1 after compaction: the frontier).
        self._turn_base: dict[str, int] = {}
        #: The tokens the current step feeds and its new requests' features,
        #: recorded once it completes.
        self._step_fed: list[tuple[str, int, int]] = []
        self._step_features: dict[str, list[Any]] = {}
        # The token the model emitted for each request on its previous step:
        # core's runner feeds it back as the request's next input.
        self._sampled: dict[str, int] = {}
        #: ``(statuses, failed)`` of every real step's commit, in order.
        self.commit_log: list[tuple[dict[str, int], set[str]]] = []
        #: What ``take_failed_requests`` returned after each real step's
        #: ``postprocess_state``, in order (core asks once per step).
        self.failed_log: list[set[str] | None] = []
        #: The ``QueueCount`` values of every real step's collection, in
        #: order (FILE-QC-005), as ``collect_commit_status`` returned them.
        self.queue_count_log: list[tuple[Any, ...]] = []
        #: Optional ``hook(collection) -> collection``: replaces what the
        #: model state receives (fault injection). The logs above record the
        #: model's own collection, before the hook.
        self.collect_hook: Callable[[Any], Any] | None = None
        collect = model.collect_commit_status

        def recording_collect() -> Any:
            result = collect()
            statuses, failed, counts = collection_parts(result)
            self.commit_log.append((dict(statuses), set(failed)))
            self.queue_count_log.append(tuple(counts))
            if self.collect_hook is not None:
                result = self.collect_hook(result)
            return result

        model.collect_commit_status = recording_collect

    # -- preparation ---------------------------------------------------------

    @torch.inference_mode()
    def prepare_execution_memory(self) -> None:
        """core's ``prepare_execution_memory``: the model state's hook, once;
        every later call (also after the first raised) is a no-op."""
        if self._execution_memory_prepared:
            return
        self._execution_memory_prepared = True
        self.state.prepare_execution_memory()

    # -- state slots ---------------------------------------------------------

    def bind_pool(self, bindings: Any) -> Any:
        """Core's ``initialize_kv_cache``: bind the pool to the state layer
        (``bind_kv_cache``), install the state-group layout core builds from
        the layer's bound ``kv_cache`` (``get_state_group_layouts``, as
        ``ModelState._state_group_layouts``), then the model state's
        ``initialize_kv_cache``."""
        pool = core_bindings(bindings)
        layer = state_layer_of(self.model)
        layer.bind_kv_cache(pool.tensor)
        self.state._state_group_layouts = state_group_layouts(layer)
        self.state.initialize_kv_cache(kv_cache_config(self.model, pool.capacity), None)
        self.bindings = bindings
        if isinstance(bindings, BindingsSpy):
            bindings.reads.clear()
        if isinstance(bindings, Trap):
            bindings.reads.clear()
        return bindings

    # -- forward ------------------------------------------------------------

    def _forward(self, batch, embeds, kwargs, attn_metadata=None):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.model_executor.models.interfaces import requires_raw_input_tokens

        num_tokens = batch.num_tokens_after_padding
        input_ids = batch.input_ids[:num_tokens]
        if embeds is not None and not requires_raw_input_tokens(self.model):
            input_ids = None
        inputs = {
            "input_ids": input_ids,
            "positions": batch.positions[:num_tokens],
            "inputs_embeds": embeds,
            "intermediate_tensors": None,
            **kwargs,
        }
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            is_padding=batch.is_padding,
        ):
            return self.model(**inputs)

    def _attn(self, batch):
        from vllm.config import CUDAGraphMode

        return self.state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (),
            torch.zeros(batch.num_tokens, dtype=torch.int64, device=self.state.device),
            [],
            None,
        )

    # -- dummy steps --------------------------------------------------------

    @torch.inference_mode()
    def dummy_step(
        self,
        *,
        is_profile: bool,
        num_tokens: int | None = None,
        randomize_inputs: bool = False,
        core_embeds: bool = True,
        attn: bool = False,
    ) -> tuple[Any, torch.Tensor, Any]:
        """``_dummy_run``. A non-profiling dummy step builds attention
        metadata (core allows ``skip_attn`` only with ``is_profile``);
        ``attn=True`` builds it for a profiling step too. The profiling run
        (``is_profile`` without ``attn``) skips it, which is how the model
        state tells the memory profile. ``core_embeds=False`` takes the
        pre-allocated buffer core would receive without calling
        ``dummy_inputs_embeds`` (so tests of other specs do not depend on
        REG-MS-002)."""
        from vllm.v1.core.sched.output import SchedulerOutput

        state = self.state
        num_tokens = num_tokens or state.max_num_tokens
        num_reqs = min(num_tokens, state.max_num_reqs)
        # core distributes the remainder so no request exceeds
        # ceil(num_tokens / num_reqs) tokens
        per_request = [
            num_tokens // num_reqs + (i >= num_reqs - num_tokens % num_reqs)
            for i in range(num_reqs)
        ]
        assert sum(per_request) == num_tokens
        output = SchedulerOutput.make_empty()
        output.total_num_scheduled_tokens = num_tokens
        output.num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(per_request)
        }
        attn = attn or not is_profile
        batch = dummy_batch(state, num_reqs, num_tokens, is_padding=not is_profile)
        assert batch.num_reqs == len(output.num_scheduled_tokens)
        assert batch.num_tokens == output.total_num_scheduled_tokens
        assert list(batch.num_scheduled_tokens) == per_request
        if randomize_inputs:
            batch.input_ids.random_(0, int(self.model.config.vocab_size))
        attn_metadata = self._attn(batch) if attn else None
        if core_embeds:
            embeds = state.dummy_inputs_embeds(
                batch.num_tokens_after_padding,
                batch.input_ids if randomize_inputs else None,
            )
        else:
            embeds = state.encoder_runner.inputs_embeds[
                : batch.num_tokens_after_padding
            ]
        kwargs = state.prepare_inputs(batch, self.req_states)
        hidden = self._forward(batch, embeds, kwargs, attn_metadata)
        return batch, hidden, embeds

    def profile_run(self, **kwargs: Any) -> tuple[Any, torch.Tensor, Any]:
        """core's ``profile_run``: ``prepare_execution_memory`` first, then
        the profiling dummy step."""
        self.prepare_execution_memory()
        return self.dummy_step(is_profile=True, **kwargs)

    @torch.inference_mode()
    def dummy_sampler_run(self, sampler: Any, hidden: torch.Tensor) -> Any:
        """``_dummy_sampler_run``, including the warm-up-parameter pass core
        runs on GPU platforms."""
        from vllm.sampling_params import SamplingParams

        if self.state.dummy_sampler_run(sampler, hidden):
            return None
        num_reqs = hidden.shape[0]
        logits = self.model.compute_logits(hidden)
        batch = dummy_batch(self.state, num_reqs, num_reqs)
        first = sampler(logits, batch)
        warmup = SamplingParams.for_sampler_warmup()
        for index in range(num_reqs):
            sampler.add_request(index, warmup)
        sampler.apply_staged_writes()
        second = sampler(logits, batch)
        return first, second

    # -- real steps ---------------------------------------------------------

    def _remove_request(self, req_id: str) -> bool:
        """core's ``_remove_request``: the model state first; the encoder cache
        only for a request the runner knows."""
        self.state.remove_request(req_id)
        if req_id not in self.req_states.req_id_to_index:
            return False
        self.req_states.remove(req_id)
        self.state.encoder_cache.remove_request(req_id)
        return True

    def _phase(self, name: str) -> None:
        if isinstance(self.bindings, BindingsSpy):
            object.__setattr__(self.bindings, "phase", name)

    @torch.inference_mode()
    def real_step(self, output: Any, computed: dict[str, int] | None = None) -> Any:
        """``execute_model`` then ``sample_tokens``'s state hooks, for a real
        step; returns ``(hidden, completion)``."""
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

        state = self.state
        self._phase("finish_requests")
        finished = set(output.finished_req_ids) | set(
            getattr(output, "preempted_req_ids", None) or ()
        )
        for req_id in sorted(finished):
            self._remove_request(req_id)
            self._fed.pop(req_id, None)
            self._features.pop(req_id, None)
            self._turn_base.pop(req_id, None)
        output = self._as_session_turns(output)
        for mm_hash in output.free_encoder_mm_hashes:
            state.encoder_cache.free_encoder_cache(mm_hash)
        self._phase("add_requests")
        for request in output.scheduled_new_reqs:
            self._remove_request(request.req_id)
            index = self.req_states.add(request.req_id)
            state.encoder_cache.add_request(request.req_id, request.mm_features)
            state.add_request(index, request)
        if output.scheduled_new_reqs:
            state.apply_staged_writes()
        if output.total_num_scheduled_tokens == 0:
            return None, self._finish(EMPTY_MODEL_RUNNER_OUTPUT)
        batch = self._real_batch(output, computed)
        self._phase("prepare_inputs")
        # core: preprocess_state runs before model_state.prepare_attn
        state.preprocess_state(batch, (), None, None)
        attn_metadata = self._attn(batch)
        embeds = state.prepare_inputs_embeds(
            output.scheduled_encoder_inputs, batch, self.req_states
        )
        kwargs = state.prepare_inputs(batch, self.req_states)
        self._record_binding(kwargs)
        self._phase("forward")
        hidden = self._forward(batch, embeds, kwargs, attn_metadata)
        self._phase("sample")
        logits = state.compute_sample_logits(hidden[batch.logits_indices])
        # the forced logits decide the token: argmax is what the sampler draws
        for req_id, token in zip(batch.req_ids, logits.argmax(dim=-1).tolist()):
            self._sampled[req_id] = int(token)
        state.postprocess_state(
            batch.idx_mapping,
            torch.ones(batch.num_reqs, dtype=torch.int32, device=state.device),
            None,
        )
        # core's runner asks once per step, after ``postprocess_state``
        self.failed_log.append(state.take_failed_requests())
        self._phase("finish_step")
        # A step that raised fed nothing: core would have stopped the engine.
        for req_id, position, token in self._step_fed:
            fed = self._fed.setdefault(req_id, [])
            del fed[position:]
            fed.append(token)
        self._features.update(self._step_features)
        return hidden, self._finish(EMPTY_MODEL_RUNNER_OUTPUT)

    def _record_binding(self, kwargs: dict[str, Any]) -> None:
        """Record the rows the step bound (``BindingsSpy.bind_calls``): the
        layer, the request IDs in row order, their ``SlotHandle``s and the
        phase. The target core has no ``bind_rows``; the model state binds the
        rows when it builds the step's state batch in ``prepare_inputs``."""
        snapshot = kwargs.get("persistent_state_projection")
        spy = self.bindings
        if not isinstance(spy, BindingsSpy) or snapshot is None:
            return
        if getattr(snapshot, "state_batch", None) is None:
            return
        spy.bind_calls.append(
            (
                STATE_LAYER,
                tuple(b.request_id for b in snapshot.bindings),
                tuple(b.handle for b in snapshot.bindings),
                spy.phase,
            )
        )

    def _as_session_turns(self, output: Any) -> Any:
        """Re-deliver a session's next turn as the target core does (the
        registration design, Recorded prompt; REG-STATE-008, REG-STATE-011).
        A test sends a live request again as a new request with no computed
        tokens; core folds the turn into the same request and compacts its
        history: it drops the last sampled token (the park) and every earlier
        token but the frontier (the last token fed), delivers the frontier
        followed by the new input as the prompt with ``num_computed_tokens``
        1 on the same slot, and keeps only the new input's features, at
        offsets past the frontier. The runner's ``remove_request`` before the
        ``add_request`` is core's."""
        from vllm.multimodal.inputs import PlaceholderRange

        new_reqs = []
        self._step_features = {}
        changed = False
        for request in output.scheduled_new_reqs:
            history = self._fed.get(request.req_id)
            self._turn_base[request.req_id] = 0
            if history and int(request.num_computed_tokens) == 0:
                frontier = int(history[-1])
                self._turn_base[request.req_id] = 1
                self._fed[request.req_id] = [frontier]
                shifted = [
                    dataclasses.replace(
                        f,
                        mm_position=PlaceholderRange(
                            offset=int(f.mm_position.offset) + 1,
                            length=int(f.mm_position.length),
                        ),
                    )
                    for f in request.mm_features
                ]
                prompt = [frontier, *request.prefill_token_ids]
                request = dataclasses.replace(
                    request,
                    prompt_token_ids=prompt,
                    prefill_token_ids=prompt,
                    mm_features=shifted,
                    num_computed_tokens=1,
                )
                changed = True
            self._step_features[request.req_id] = list(request.mm_features)
            new_reqs.append(request)
        if not changed:
            return output
        return dataclasses.replace(output, scheduled_new_reqs=new_reqs)

    def _finish(self, output: Any) -> Any:
        """The target core has no step completion: nothing is returned."""
        del output
        return None

    def _real_batch(self, output: Any, computed: dict[str, int] | None) -> Any:
        """One token per scheduled request, in the scheduler output's order
        (new, then cached), agreeing with ``num_scheduled_tokens``."""
        req_ids = [r.req_id for r in output.scheduled_new_reqs]
        req_ids += list(output.scheduled_cached_reqs.req_ids)
        assert [output.num_scheduled_tokens[r] for r in req_ids] == [1] * len(req_ids)
        assert output.total_num_scheduled_tokens == len(req_ids)
        n = len(req_ids)
        batch = dummy_batch(self.state, n, n, is_padding=False)
        tokens = [
            r.prefill_token_ids[r.num_computed_tokens]
            for r in output.scheduled_new_reqs
        ]
        for r in output.scheduled_new_reqs:
            self._prompts[r.req_id] = list(r.prefill_token_ids)
            self._sampled.pop(r.req_id, None)
        cached = output.scheduled_cached_reqs
        bases = self._turn_base
        overrides = dict(computed or {})
        given = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        given |= {r: p for r, p in overrides.items() if r in given}
        absolute = {r: bases.get(r, 0) + int(p) for r, p in given.items()}
        absolute |= {r: int(p) for r, p in overrides.items() if r not in given}
        for req_id in cached.req_ids:
            position = absolute[req_id]
            # A prompt token if the request's prompt reaches this position,
            # else the token the model emitted on this request's previous step.
            prompt = self._prompts[req_id]
            tokens.append(
                prompt[position] if position < len(prompt) else self._sampled[req_id]
            )
        batch.input_ids[:n] = torch.tensor(tokens)
        index = np.array(
            [self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.intp
        )
        computed = {
            r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs
        } | absolute
        self._step_fed = [
            (req_id, int(computed.get(req_id, 0)), int(token))
            for req_id, token in zip(req_ids, tokens, strict=True)
        ]
        prompt_lens = [len(self._prompts[r]) for r in req_ids]
        return dataclasses.replace(
            batch,
            req_ids=req_ids,
            idx_mapping=torch.from_numpy(index).to(self.state.device),
            idx_mapping_np=index,
            num_scheduled_tokens=np.ones(n, dtype=np.int32),
            num_computed_tokens_np=np.array(
                [computed.get(r, 0) for r in req_ids], dtype=np.int32
            ),
            prefill_len_np=np.array(prompt_lens, dtype=np.int32),
        )


def call_with(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Exception | None:
    """Run ``fn``; return the exception it raised, or ``None``."""
    try:
        fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the test inspects it
        return error
    return None


@contextlib.contextmanager
def nothing() -> Iterator[None]:
    yield
