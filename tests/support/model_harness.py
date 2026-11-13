# SPDX-License-Identifier: Apache-2.0
"""The real model and model state on CPU, driven as core's model runner drives
them (``vllm/v1/worker/gpu/model_runner.py`` of the target core:
``execute_model``, ``_dummy_run``, ``profile_run``, ``_dummy_sampler_run``).

What is real: the configuration (card parser, post-override validation, the
engine hook), the model class constructed without weights, its
``ModelState``, core's ``EncoderCache`` / ``EncoderRunner``, core's
``StateSlotBindings`` over a CPU pool, core's ``SchedulerOutput``,
``NewRequestData``, ``InputBatch`` and ``StateSlotHandle``.

What a test substitutes, and why:

* ``spec_without_state_name``: the model's state-layout builder still passes
  the removed ``state_name`` argument (REG-STATE-001), which would stop model
  construction before any other behaviour is reached. Tests of other specs
  drop that argument; REG-STATE-001 is tested without it.
* ``ProfileRecorder``: the profile transaction (``run_persistent_state_profile``)
  cannot run the synthetic card's tiny encoder on CPU; tests record its calls
  instead. It is the one plugin function the recorder replaces.
* ``stub_advance``: the row transaction (``advance.advance_model_rows``) does
  not run on CPU for the same reason; tests that run a real step past
  readiness replace it with a stub returning a tensor like the input.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

#: The model's state layer name for ``prefix="nemotron"`` (``f"{prefix}.state"``).
STATE_LAYER = "nemotron.state"
EPOCH = "epoch-1"


# ---------------------------------------------------------------------------
# Configuration, model, model state


def build_vllm_config(
    card_dir: Path,
    *,
    right_context: int = 6,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 64,
    overrides: dict[str, Any] | None = None,
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
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=False,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
        ),
    )


def spec_without_state_name(monkeypatch: Any) -> None:
    """Let the model's state-layout builder construct on the target core
    (see the module docstring)."""
    from vllm.v1.kv_cache_interface import PersistentStateSpec

    from vllm_nemotron_asr_streaming.nemotron_asr import state_profile

    def build(**kwargs: Any) -> Any:
        kwargs.pop("state_name", None)
        return PersistentStateSpec(**kwargs)

    monkeypatch.setattr(state_profile, "PersistentStateSpec", build)


def build_model(vllm_config: Any) -> Any:
    """The model class, constructed as core's loader does, without weights."""
    from vllm.config.vllm import set_current_vllm_config

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    with set_current_vllm_config(vllm_config):
        return Nemotron3_5AsrStreamingForRNNT(
            vllm_config=vllm_config, prefix="nemotron"
        )


def build_model_state(vllm_config: Any, model: Any) -> Any:
    """The model's ``ModelState`` as core resolves and builds it, with an
    encoder cache (the model is multimodal), so ``encoder_runner`` and its
    pre-allocated ``inputs_embeds`` buffer exist."""
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    state_cls = type(model).get_model_state_cls()
    return state_cls(vllm_config, model, EncoderCache(), torch.device("cpu"))


def model_spec(model: Any) -> Any:
    """The model's ``PersistentStateSpec`` (the one its state layer declares)."""
    return model._persistent_state_layer._persistent_state_spec


# ---------------------------------------------------------------------------
# Recorders and traps


class ProfileRecorder:
    """Stands in for the profile transaction; records each call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model: Any, **kwargs: Any) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return torch.zeros(1)

    def install(self, monkeypatch: Any) -> ProfileRecorder:
        from vllm_nemotron_asr_streaming.nemotron_asr import profile_execution

        monkeypatch.setattr(profile_execution, "run_persistent_state_profile", self)
        return self


def stub_advance(monkeypatch: Any) -> list[Any]:
    """Replace the row transaction with a stub; return the list of calls."""
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    calls: list[Any] = []

    def advance_model_rows(core, input_ids, inputs_embeds, plan, **kwargs):
        calls.append(plan)
        return torch.zeros_like(inputs_embeds)

    monkeypatch.setattr(advance, "advance_model_rows", advance_model_rows)
    return calls


class Trap:
    """An object that fails any attribute access once armed.

    Wraps ``target`` and forwards to it until ``armed`` is set; ``reads``
    records every attribute name read (armed or not).
    """

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
    """Wraps core's ``StateSlotBindings``; records ``bind_rows`` calls and
    every member the model reads, and wraps returned rows in ``RowsSpy``."""

    def __init__(self, bindings: Any) -> None:
        object.__setattr__(self, "_bindings", bindings)
        object.__setattr__(self, "bind_calls", [])
        object.__setattr__(self, "reads", [])
        object.__setattr__(self, "row_reads", [])
        object.__setattr__(self, "bind_phase", None)

    def bind_rows(self, layer, ordered_request_ids, expected_handles):
        self.bind_calls.append(
            (
                layer,
                tuple(ordered_request_ids),
                tuple(expected_handles),
                self.bind_phase,
            )
        )
        rows = self._bindings.bind_rows(layer, ordered_request_ids, expected_handles)
        return RowsSpy(rows, self.row_reads)

    def __getattr__(self, attr: str) -> Any:
        self.reads.append(attr)
        return getattr(self._bindings, attr)


def real_bindings(spec: Any, *, capacity: int = 8) -> Any:
    """Core's ``StateSlotBindings`` over a CPU pool for the model's layer."""
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    group = SimpleNamespace(kv_cache_specs={STATE_LAYER: spec})
    tensor = torch.zeros(capacity * spec.page_size_bytes, dtype=torch.uint8)
    return StateSlotBindings([group], {STATE_LAYER: tensor})


# ---------------------------------------------------------------------------
# Scheduler output and batches


def handle(slot: int, generation: int = 1) -> Any:
    from vllm.v1.kv_cache_interface import StateSlotHandle

    return StateSlotHandle(EPOCH, slot, generation)


def stream_envelope(
    config: Any, *, geometry: int = 3, prompt: int = 0, sequence: int = 0
) -> torch.Tensor:
    """One stream unit's carrier row (SESS-PACK-001): header then samples."""
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        CADENCES,
        RAW_SAMPLES_PER_CHUNK,
    )

    samples = RAW_SAMPLES_PER_CHUNK[tuple(CADENCES)[geometry]]
    row = torch.zeros(7 + samples, dtype=torch.float32)
    row[:7] = torch.tensor([2, samples, geometry, 0, prompt, sequence, 1])
    return row


def new_request(
    req_id: str,
    token: int,
    *,
    audio: torch.Tensor | None = None,
    samples: np.ndarray | None = None,
    header: list[int] | None = None,
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
                identifier=f"uuid-{req_id}",
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
) -> Any:
    """A real ``SchedulerOutput``; ``handles`` / ``cleanup`` map request IDs
    to a slot (generation 1) or a ``(slot, generation)`` pair for the model's
    state layer."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    scheduled = {r.req_id: 1 for r in new} | {r: 1 for r in cached}
    output = SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(cached),
            new_token_ids=[[0] for _ in cached],
            new_block_ids=[None for _ in cached],
            num_computed_tokens=[1 for _ in cached],
            num_output_tokens=[1 for _ in cached],
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs=dict(encoder_inputs or {}),
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
        state_slot_handles={r: _layer(s) for r, s in (handles or {}).items()},
    )
    output.state_cleanup_handles = {r: _layer(s) for r, s in (cleanup or {}).items()}
    output.state_step_seq = step_seq
    return output


def _layer(slot: int | tuple[int, int]) -> dict[str, Any]:
    slot, generation = slot if isinstance(slot, tuple) else (slot, 1)
    return {STATE_LAYER: handle(slot, generation)}


def dummy_batch(state: Any, num_reqs: int, num_tokens: int) -> Any:
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    buffers = InputBuffers(state.max_num_reqs, state.max_num_tokens, state.device)
    return InputBatch.make_dummy(num_reqs, num_tokens, buffers)


def real_batch(state: Any, output: Any, computed: dict[str, int] | None = None) -> Any:
    """An ``InputBatch`` for a real step: one token per scheduled request, in
    the scheduler output's order (new, then cached)."""
    req_ids = [r.req_id for r in output.scheduled_new_reqs]
    req_ids += list(output.scheduled_cached_reqs.req_ids)
    batch = dummy_batch(state, len(req_ids), len(req_ids))
    tokens = [r.prefill_token_ids[0] for r in output.scheduled_new_reqs]
    tokens += [0 for _ in output.scheduled_cached_reqs.req_ids]
    batch.input_ids[: len(tokens)] = torch.tensor(tokens)
    return dataclasses.replace(
        batch,
        req_ids=req_ids,
        num_scheduled_tokens=np.ones(len(req_ids), dtype=np.int32),
        num_computed_tokens_np=np.array(
            [(computed or {}).get(r, 0) for r in req_ids], dtype=np.int32
        ),
    )


def req_states(req_ids: list[str]) -> Any:
    return SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(req_ids)})


# ---------------------------------------------------------------------------
# Steps, as core's model runner sequences them


def dummy_step(
    state: Any,
    model: Any,
    *,
    is_profile: bool,
    num_tokens: int | None = None,
    randomize_inputs: bool = False,
    core_embeds: bool = True,
) -> tuple[Any, torch.Tensor, Any]:
    """``_dummy_run``: ``prepare_step`` (dummy), ``dummy_inputs_embeds``,
    ``prepare_inputs``, then the forward. Returns ``(batch, output, embeds)``.

    ``core_embeds=False`` takes the embeddings core would receive (the model
    state's pre-allocated buffer, REG-MS-002) without calling
    ``dummy_inputs_embeds``, so tests of other specs do not depend on that
    method's signature, which REG-MS-001/002 test on their own."""
    from vllm.v1.core.sched.output import SchedulerOutput

    num_tokens = num_tokens or state.max_num_tokens
    num_reqs = min(num_tokens, state.max_num_reqs)
    output = SchedulerOutput.make_empty()
    output.total_num_scheduled_tokens = num_tokens
    output.num_scheduled_tokens = {f"_dummy_req_{i}": 1 for i in range(num_reqs)}
    state.prepare_step(output, dummy_run=True, is_profile=is_profile)
    batch = dummy_batch(state, num_reqs, num_tokens)
    if randomize_inputs:
        batch.input_ids.random_(0, int(model.config.vocab_size))
    if core_embeds:
        embeds = state.dummy_inputs_embeds(
            batch.num_tokens_after_padding,
            batch.input_ids if randomize_inputs else None,
        )
    else:
        embeds = state.encoder_runner.inputs_embeds[: batch.num_tokens_after_padding]
    kwargs = state.prepare_inputs(batch, req_states(list(batch.req_ids)))
    hidden = model(
        input_ids=batch.input_ids,
        positions=batch.positions,
        inputs_embeds=embeds,
        intermediate_tensors=None,
        **kwargs,
    )
    return batch, hidden, embeds


def profile_run(state: Any, model: Any, **kwargs: Any) -> tuple[Any, torch.Tensor, Any]:
    """``profile_run``'s forward: a profiling dummy step."""
    return dummy_step(state, model, is_profile=True, **kwargs)


def real_step(
    state: Any,
    model: Any,
    bindings: Any,
    output: Any,
    *,
    forward: bool = True,
    computed: dict[str, int] | None = None,
) -> Any:
    """``execute_model`` for a real step: ``prepare_step``, the state-slot
    ``begin_step`` / ``activate_requests``, ``add_request`` per new request,
    then (when tokens are scheduled) ``prepare_inputs`` and the forward."""
    state.prepare_step(output, dummy_run=False, is_profile=False)
    if bindings is not None:
        core = bindings._bindings if isinstance(bindings, BindingsSpy) else bindings
        core.begin_step(output, None)
        core.activate_requests()
    for index, request in enumerate(output.scheduled_new_reqs):
        state.add_request(index, request)
    if output.total_num_scheduled_tokens == 0 or not forward:
        return None
    batch = real_batch(state, output, computed)
    if bindings is not None and hasattr(bindings, "bind_phase"):
        object.__setattr__(bindings, "bind_phase", "prepare_inputs")
    kwargs = state.prepare_inputs(batch, req_states(list(batch.req_ids)))
    embeds = torch.zeros(
        batch.num_tokens, int(model.config.hidden_size), dtype=torch.float32
    )
    for index, request in enumerate(output.scheduled_new_reqs):
        feature = request.mm_features[0] if request.mm_features else None
        if feature is not None and "audio" in feature.data:
            envelope = feature.data["audio"].data
            embeds[index, : envelope.shape[0]] = envelope
    if bindings is not None and hasattr(bindings, "bind_phase"):
        object.__setattr__(bindings, "bind_phase", "forward")
    return model(
        input_ids=batch.input_ids[: batch.num_tokens],
        positions=batch.positions[: batch.num_tokens],
        inputs_embeds=embeds,
        intermediate_tensors=None,
        **kwargs,
    )


def call_with(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> BaseException | None:
    """Run ``fn``; return the exception it raised, or ``None``."""
    try:
        fn(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the test inspects it
        return error
    return None
