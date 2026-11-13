# SPDX-License-Identifier: Apache-2.0
"""The real model and model state on CPU, driven as core's model runner drives
them at the stage 1 tree (``vllm/v1/worker/gpu/model_runner.py``:
``execute_model``, ``_dummy_run``, ``profile_run``, ``_dummy_sampler_run``;
``vllm/model_executor/warmup/kernel_warmup.py`` for later profiling steps).

Real: the configuration (card parser, post-override validation, the engine
hook), the model class (constructed as core's loader does, frozen for
inference, without loading weights), its ``ModelState``, core's
``EncoderCache`` / ``EncoderRunner`` (so ``prepare_inputs_embeds`` runs the
model's own multimodal embedding), core's ``StateSlotBindings`` over a CPU
pool, ``SchedulerOutput``, ``NewRequestData``, ``InputBatch``,
``StateSlotHandle``, ``set_forward_context`` and ``BatchDescriptor``, the
profile transaction and the row transaction (scatter and commit included).

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
* ``finish_step`` creates a CUDA event; on CPU it runs with an event double
  (``_CpuEvent``), everything else in it is core's.
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

Migration-blocker isolation (each fixture is opt-in and scoped to a test):

* ``spec_without_state_name``: drops the removed ``state_name`` argument the
  layout builder still passes (REG-STATE-001), so model construction is
  reachable.
* ``legacy_state_api``: re-adds the removed ``StateSlotBindings.get_handle``,
  ``StateSlotRows.spec`` / ``.views`` and ``StateSlotRows.validate_current``
  on top of the target core's classes, each part separately, so a test of
  another spec is not masked by REG-STATE-002..005.
"""

from __future__ import annotations

import contextlib
import dataclasses
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

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
) -> Any:
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from vllm_nemotron_asr_streaming import register

    register()
    hf_overrides: dict[str, Any] = {
        "num_lookahead_tokens": right_context,
        "supported_num_lookahead_tokens": [0, 1, 3, 6, 13],
    }
    hf_overrides.update(overrides or {})
    model_config = ModelConfig(
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        hf_overrides=hf_overrides,
        enforce_eager=True,
        dtype="float32",
        max_model_len=2048,
    )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device=device),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=False,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max(max_num_batched_tokens, max_num_seqs),
            enable_chunked_prefill=True,
        ),
    )


def spec_without_state_name(monkeypatch: Any) -> None:
    """See the module docstring."""
    from vllm.v1.kv_cache_interface import PersistentStateSpec

    from vllm_nemotron_asr_streaming.nemotron_asr import state_profile

    def build(**kwargs: Any) -> Any:
        kwargs.pop("state_name", None)
        return PersistentStateSpec(**kwargs)

    monkeypatch.setattr(state_profile, "PersistentStateSpec", build)


def legacy_state_api(
    monkeypatch: Any,
    *,
    get_handle: bool = True,
    rows_layout: bool = True,
    validate_current: bool = True,
) -> None:
    """See the module docstring. Each part is added only if the target core
    lacks it."""
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings, StateSlotRows

    if get_handle and not hasattr(StateSlotBindings, "get_handle"):

        def _get_handle(self, request_id, layer):
            if self._completion is None:
                raise ValueError("No live state step is active")
            return self._completion.handles[request_id][layer]

        monkeypatch.setattr(StateSlotBindings, "get_handle", _get_handle, raising=False)
    if rows_layout and not hasattr(StateSlotRows, "spec"):
        monkeypatch.setattr(
            StateSlotRows,
            "spec",
            property(lambda self: self._bindings.views[self.layer].spec),
            raising=False,
        )
        monkeypatch.setattr(
            StateSlotRows,
            "views",
            property(lambda self: self.component_pools),
            raising=False,
        )
    if validate_current and not hasattr(StateSlotRows, "validate_current"):
        monkeypatch.setattr(
            StateSlotRows,
            "validate_current",
            lambda self: self.component_pools and None,
            raising=False,
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
    """The model's ``PersistentStateSpec`` (the one its state layer declares)."""
    return model._persistent_state_layer._persistent_state_spec


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
        self.events = events if events is not None else Events()
        self.before: Callable[[dict[str, Any]], None] | None = None
        self._real: Any = None

    def __call__(self, core, input_ids, inputs_embeds, plan, **kwargs):
        self.calls.append(plan)
        self.events.append(("advance", plan))
        if self.before is not None:
            self.before(kwargs)
        return self._real(core, input_ids, inputs_embeds, plan, **kwargs)

    def install(self, monkeypatch: Any) -> AdvanceSpy:
        from vllm_nemotron_asr_streaming.nemotron_asr import advance

        self._real = advance.advance_model_rows
        monkeypatch.setattr(advance, "advance_model_rows", self)
        return self


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


class RowsSpy:
    """Wraps core's ``StateSlotRows``; records every member read."""

    def __init__(self, rows: Any, log: list[str]) -> None:
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, attr: str) -> Any:
        self._log.append(attr)
        return getattr(self._rows, attr)


class BindingsSpy:
    """Wraps core's ``StateSlotBindings``; records ``bind_rows`` calls (with
    the step phase), every other member the model reads, and wraps returned
    rows in ``RowsSpy``."""

    def __init__(self, bindings: Any) -> None:
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "bind_calls", [])
        object.__setattr__(self, "reads", [])
        object.__setattr__(self, "row_reads", [])
        object.__setattr__(self, "phase", None)

    def bind_rows(self, layer, ordered_request_ids, expected_handles):
        self.bind_calls.append(
            (layer, tuple(ordered_request_ids), tuple(expected_handles), self.phase)
        )
        rows = self._bindings.bind_rows(layer, ordered_request_ids, expected_handles)
        return RowsSpy(rows, self.row_reads)

    def __getattr__(self, attr: str) -> Any:
        self.reads.append((attr, self.phase))
        return getattr(self._bindings, attr)


def core_bindings(bindings: Any) -> Any:
    return bindings._bindings if isinstance(bindings, BindingsSpy) else bindings


def real_bindings(spec: Any, *, capacity: int = 8, device: str = "cpu") -> Any:
    """Core's ``StateSlotBindings`` over a pool for the model's layer."""
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    group = SimpleNamespace(kv_cache_specs={STATE_LAYER: spec})
    tensor = torch.zeros(
        capacity * spec.page_size_bytes, dtype=torch.uint8, device=device
    )
    return StateSlotBindings([group], {STATE_LAYER: tensor})


def pool_tensor(bindings: Any) -> torch.Tensor:
    return core_bindings(bindings).views[STATE_LAYER]._tensor


def slot_bytes(bindings: Any, slot: int) -> torch.Tensor:
    views = core_bindings(bindings).views[STATE_LAYER]
    page = views.spec.page_size_bytes
    return pool_tensor(bindings)[slot * page : (slot + 1) * page]


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
        def capture_context(device):
            runtime.events.append(("capture_context", str(device)))
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
    from vllm.v1.kv_cache_interface import StateSlotHandle

    return StateSlotHandle(EPOCH, slot, generation)


def _layer(slot: int | tuple[int, int]) -> dict[str, Any]:
    slot, generation = slot if isinstance(slot, tuple) else (slot, 1)
    return {STATE_LAYER: handle(slot, generation)}


def stream_envelope(
    config: Any,
    *,
    geometry: int = 3,
    prompt: int = 0,
    sequence: int = 0,
    final: bool = True,
) -> torch.Tensor:
    """One stream unit's carrier row (SESS-PACK-001): header then samples.
    ``final`` marks the session's last unit (its final tail)."""
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        CADENCES,
        RAW_SAMPLES_PER_CHUNK,
    )

    samples = RAW_SAMPLES_PER_CHUNK[tuple(CADENCES)[geometry]]
    row = torch.zeros(7 + samples, dtype=torch.float32)
    row[:7] = torch.tensor([2, samples, geometry, 0, prompt, sequence, int(final)])
    row[7:] = torch.from_numpy(
        np.random.default_rng(sequence).standard_normal(samples).astype(np.float32)
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
    (default 1). ``handles`` / ``cleanup`` map request IDs
    to a slot (generation 1) or ``(slot, generation)``. When
    ``encoder_inputs`` is ``None``, every new request with a multimodal
    feature has its feature scheduled, as core schedules it on first sight."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

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
        state_slot_handles={r: _layer(s) for r, s in (handles or {}).items()},
    )
    output.state_cleanup_handles = {r: _layer(s) for r, s in (cleanup or {}).items()}
    output.state_step_seq = step_seq
    return output


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


class _CpuEvent:
    """Stands for ``torch.cuda.Event`` in ``finish_step`` on CPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# The runner


class Runner:
    """Drives the model and model state in core's order."""

    def __init__(self, vllm_config: Any, model: Any, state: Any) -> None:
        self.vllm_config = vllm_config
        self.model = model
        self.state = state
        self.bindings: Any = None
        self.req_states = StepRequestStates()
        self._prompts: dict[str, list[int]] = {}
        # The token the model emitted for each request on its previous step:
        # core's runner feeds it back as the request's next input.
        self._sampled: dict[str, int] = {}
        #: ``(statuses, failed)`` of every real step's commit, in order.
        self.commit_log: list[tuple[dict[str, int], set[str]]] = []
        collect = model.collect_commit_status

        def recording_collect() -> Any:
            result = collect()
            self.commit_log.append((dict(result[0]), set(result[1])))
            return result

        model.collect_commit_status = recording_collect

    # -- state slots ---------------------------------------------------------

    def bind_pool(self, bindings: Any) -> Any:
        """``initialize_state_slots``, as core calls it after profiling."""
        self.state.initialize_state_slots(bindings)
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
        """``_dummy_run``. ``attn=True`` builds attention metadata as core's
        kernel warm-up does (no ``skip_attn``); otherwise as the profiling run
        (``skip_attn``). ``core_embeds=False`` takes the pre-allocated buffer
        core would receive without calling ``dummy_inputs_embeds`` (so tests
        of other specs do not depend on REG-MS-002)."""
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
        state.prepare_step(output, dummy_run=True, is_profile=is_profile)
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
        core = core_bindings(self.bindings) if self.bindings is not None else None
        self._phase("prepare_step")
        state.prepare_step(output, dummy_run=False, is_profile=False)
        self._phase("begin_step")
        if core is not None:
            core.begin_step(output, None)
        finished = set(output.finished_req_ids) | set(
            getattr(output, "preempted_req_ids", None) or ()
        )
        for req_id in sorted(finished):
            self._remove_request(req_id)
        for mm_hash in output.free_encoder_mm_hashes:
            state.encoder_cache.free_encoder_cache(mm_hash)
        if core is not None:
            core.activate_requests()
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
        self._phase("finish_step")
        return hidden, self._finish(EMPTY_MODEL_RUNNER_OUTPUT)

    def _finish(self, output: Any) -> Any:
        if self.bindings is None:
            return None
        core = core_bindings(self.bindings)
        with mock.patch.object(torch.cuda, "Event", _CpuEvent):
            return core.finish_step(output, None)

    def _real_batch(self, output: Any, computed: dict[str, int] | None) -> Any:
        """One token per scheduled request, in the scheduler output's order
        (new, then cached), agreeing with ``num_scheduled_tokens``."""
        req_ids = [r.req_id for r in output.scheduled_new_reqs]
        req_ids += list(output.scheduled_cached_reqs.req_ids)
        assert [output.num_scheduled_tokens[r] for r in req_ids] == [1] * len(req_ids)
        assert output.total_num_scheduled_tokens == len(req_ids)
        n = len(req_ids)
        batch = dummy_batch(self.state, n, n, is_padding=False)
        tokens = [r.prefill_token_ids[0] for r in output.scheduled_new_reqs]
        for r in output.scheduled_new_reqs:
            self._prompts[r.req_id] = list(r.prefill_token_ids)
            self._sampled.pop(r.req_id, None)
        cached = output.scheduled_cached_reqs
        for req_id, position in zip(cached.req_ids, cached.num_computed_tokens):
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
        computed = dict(
            zip(cached.req_ids, cached.num_computed_tokens, strict=True)
        ) | (computed or {})
        return dataclasses.replace(
            batch,
            req_ids=req_ids,
            idx_mapping=torch.from_numpy(index).to(self.state.device),
            idx_mapping_np=index,
            num_scheduled_tokens=np.ones(n, dtype=np.int32),
            num_computed_tokens_np=np.array(
                [computed.get(r, 0) for r in req_ids], dtype=np.int32
            ),
            prefill_len_np=np.ones(n, dtype=np.int32),
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
