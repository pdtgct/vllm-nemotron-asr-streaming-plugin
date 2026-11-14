# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv2 projection of bounded core state views into Nemotron transactions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
    project_nemotron_state_pools,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    PersistentStateSpec,
    StateSlotHandle,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.state_slots import StateSlotBindings, StateSlotRows
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup
from vllm_nemotron_asr_streaming.engine_flags import check_engine_config
from vllm_nemotron_asr_streaming.nemotron_asr import file_rows
from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import (
    FileAction,
    FileHostBuffers,
    StepPlan,
    StepRow,
)
from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
    CADENCES,
    RAW_SAMPLES_PER_CHUNK,
)
from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
    file_envelope_header,
    resolve_row_envelope_header,
    stream_order_key,
)


#: Host-side stand-in for a scheduled token the sampler produced on the
#: device: never a valid token id, so never the CHUNK or file carrier.
SAMPLED_TOKEN = -1


@dataclass(frozen=True)
class RowBinding:
    request_id: str
    handle: StateSlotHandle
    fresh: bool

    @property
    def slot_id(self) -> int:
        return self.handle.slot

    @property
    def generation(self) -> int:
        return self.handle.generation


@dataclass(frozen=True)
class ProjectionSnapshot:
    epoch: int
    req_ids: tuple[str, ...]
    scheduled_encoder_inputs: dict[str, list[int]]
    encoder_features: object
    bindings: tuple[RowBinding, ...]
    block_ids: tuple[tuple[int, ...], ...]
    input_batch: Any
    req_states: Any
    model_kwargs: dict[str, Any]
    request_metadata: dict[str, NewRequestData] | None = None
    token_ids: tuple[int, ...] = ()
    dummy_run: bool = False
    is_profile: bool = False
    no_page_io: bool = False
    state_batch: StateBatch | None = None
    file_rows: dict[str, StepRow] = field(default_factory=dict)
    geometry_order: tuple[int, ...] = ()


@dataclass
class _FileProgress:
    """A FILE request's progress through its file, kept for its life."""

    position: int = 0
    tail_done: bool = False
    #: Queued labels not yet emitted, when the host knows it without reading
    #: the device (a fresh row, or after a drain step); ``None`` otherwise.
    known_queue: int | None = 0


class _ResidentCommitSink:
    def __init__(
        self, sink: Any, access: StateSlotRows, request_ids: tuple[str, ...]
    ):
        self.sink = sink
        self.access = access
        self.request_ids = request_ids

    # @spec REG-STATE-004, REG-STATE-006
    def reserve(self, plan: Any) -> Any:
        # Core refuses this read outside the live step, and the commit then
        # writes no state.
        self.access.component_pools
        if tuple(b.request_id for b in plan.bindings) != self.request_ids:
            raise ValueError("commit row order differs from resident bindings")
        if tuple(b.block_id for b in plan.bindings) != self.access.slot_indices_cpu:
            raise ValueError("commit slots differ from resident bindings")
        return self.sink.reserve(plan)


class StateBatch:
    """Zero-copy resident pools with a step-scoped pre-commit guard."""

    # @spec REG-STATE-003, REG-STATE-005
    def __init__(
        self,
        slots: StateSlotBindings,
        layer: str,
        bindings: tuple[RowBinding, ...],
        spec: PersistentStateSpec,
        n_layers: int,
    ):
        self.request_ids = tuple(b.request_id for b in bindings)
        self.access = slots.bind_rows(
            layer, self.request_ids, tuple(b.handle for b in bindings)
        )
        self.pools = project_nemotron_state_pools(
            self.access.component_pools, spec, n_layers=n_layers
        )

    def commit_sink(self, sink: Any) -> _ResidentCommitSink:
        return _ResidentCommitSink(sink, self.access, self.request_ids)


class NemotronASRModelState(ModelState):
    """Model-local projection with no slot allocation or retirement authority."""

    # @spec REG-ENG-006
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The final engine configuration may differ from what the
        # configuration hook saw: check it again before any state exists.
        check_engine_config(
            self.vllm_config, "the Nemotron streaming model state", resolved=True
        )
        self.file_buffers = FileHostBuffers()
        self._file_progress: dict[str, _FileProgress] = {}
        self._failed_files: set[str] = set()
        self._scheduler_output: SchedulerOutput | None = None
        self._metadata: dict[str, NewRequestData] = {}
        self._initialized: dict[str, StateSlotHandle] = {}
        self._epoch = 0
        self._snapshot: ProjectionSnapshot | None = None
        self._dummy_run = False
        self._is_profile = False
        self._scheduled_encoder_inputs: dict[str, list[int]] = {}
        self._features: torch.Tensor | None = None
        self._scheduled_tokens: dict[str, tuple[int, ...]] = {}
        #: Each live request's latest prompt, as core delivered it.
        self._prompts: dict[str, tuple[int, ...]] = {}

    # @spec REG-TASK-002
    def get_supported_generation_tasks(self) -> tuple[Any, ...]:
        # Core would infer "generate" from compute_logits; the model emits
        # transcript labels, not text generation.
        return ("transcription", "realtime")

    def initialize_state_slots(self, bindings: StateSlotBindings) -> None:
        super().initialize_state_slots(bindings)
        layer = self.model._persistent_state_layer.prefix
        if set(bindings.views) != {layer}:
            raise ValueError("Nemotron requires exactly its aggregate state group")
        # All resident writes run on the runner stream. Inner graphs also replay
        # there; no auxiliary state-writing stream is created by this consumer.

    # @spec REG-MEM-001, REG-STATE-002, REG-STATE-007, REG-STATE-008
    def prepare_step(
        self, scheduler_output: SchedulerOutput, *, dummy_run: bool, is_profile: bool
    ) -> None:
        # Only a dummy step that is also a profiling step is a profiling
        # dummy step; the model's forward prepares execution on it. The
        # record is replaced on every step and prepares nothing here.
        self._dummy_run = dummy_run
        self._is_profile = is_profile
        self._snapshot = None
        self._scheduled_encoder_inputs = {}
        self._features = None
        self._scheduled_tokens = {}
        self._epoch = getattr(self, "_epoch", 0) + 1
        if dummy_run:
            # Core skips its state-slot bookkeeping on dummy steps: nothing of
            # an earlier step is kept for them to read.
            self._scheduler_output = None
            return
        # Core sends no token slices for cached requests unless pipeline
        # parallelism is on, which this model refuses. The host token of a
        # cached row is therefore read from the prompt this state recorded when
        # core delivered it as a new request (a session turn arrives that way
        # too, with its prompt starting at position zero). A position past the
        # recorded prompt holds a token the sampler produced on the device; the
        # host never reads it, and no sampled token is a CHUNK carrier, so such
        # a row is marked with ``SAMPLED_TOKEN``.
        for req_id in scheduler_output.finished_req_ids:
            self._prompts.pop(req_id, None)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens):
            prompt = self._prompts.get(req_id)
            if prompt is None:
                raise ValueError(
                    f"cached request {req_id!r} has no recorded prompt tokens"
                )
            start = int(computed)
            count = scheduler_output.num_scheduled_tokens[req_id]
            window = prompt[start : start + count]
            if window and len(window) != count:
                raise ValueError(
                    f"cached request {req_id!r} schedules tokens across the end "
                    "of its prompt"
                )
            self._scheduled_tokens[req_id] = window or (SAMPLED_TOKEN,) * count
        for request in scheduler_output.scheduled_new_reqs:
            tokens = request.prefill_token_ids
            if tokens is None:
                raise ValueError("state step omitted new host token slice")
            self._prompts[request.req_id] = tuple(tokens)
            start = request.num_computed_tokens
            count = scheduler_output.num_scheduled_tokens[request.req_id]
            self._scheduled_tokens[request.req_id] = tuple(
                tokens[start : start + count]
            )
        self._scheduler_output = scheduler_output
        for req_id in scheduler_output.finished_req_ids:
            self._metadata.pop(req_id, None)
            self._initialized.pop(req_id, None)
        # FILE-BUF-003: any finished request frees its file buffer, also in a
        # step with no forward work (this hook runs for every step).
        for req_id in self.file_buffers.free_finished(
            scheduler_output.finished_req_ids
        ):
            self._file_progress.pop(req_id, None)
        self._failed_files.difference_update(scheduler_output.finished_req_ids)
        carrier = getattr(getattr(self.model, "config", None), "file_carrier_token_id", None)
        if carrier is not None:
            for request in scheduler_output.scheduled_new_reqs:
                tokens = self._scheduled_tokens.get(request.req_id, ())
                if len(tokens) == 1 and tokens[0] == int(carrier):
                    self._take_file(request, scheduler_output)
        self.model._registry.prune(self._initialized)
        handoff = self.model._native_burst_handoff
        if handoff is not None:
            handoff.clear()

    # @spec FILE-BUF-001, FILE-BUF-004, FILE-BUF-005, FILE-ROW-001
    def _take_file(
        self, request: NewRequestData, scheduler_output: SchedulerOutput
    ) -> None:
        """Take a file request's samples and header into a request-owned
        host buffer at its first schedule and mark it a FILE request.

        A malformed item (the processor refuses these at input processing,
        so only a direct engine client can send one) fails that row alone
        through the transaction's row-failure path; it never raises out of
        the step.
        """
        req_id = request.req_id
        try:
            self._take_file_checked(request, scheduler_output)
        except (ValueError, KeyError, TypeError, AttributeError, IndexError):
            # any missing or unreadable field makes the item malformed
            self.file_buffers.free_finished([req_id])
            self._file_progress.pop(req_id, None)
            self._failed_files.add(req_id)

    def _take_file_checked(
        self, request: NewRequestData, scheduler_output: SchedulerOutput
    ) -> None:
        req_id = request.req_id
        if len(request.mm_features) != 1:
            raise ValueError(f"file request {req_id!r} needs exactly one audio item")
        item = request.mm_features[0].data
        if item is None:
            raise ValueError(f"file request {req_id!r} arrived without its samples")
        layer = self.model._persistent_state_layer.prefix
        handles = scheduler_output.state_slot_handles or {}
        handle = handles.get(req_id, {}).get(layer)
        if handle is None:
            raise ValueError(f"file request {req_id!r} has no state slot handle")
        dropped = self.file_buffers.take(
            req_id,
            handle.slot,
            handle.generation,
            item["samples"].data,
            item["file_header"].data,
        )
        for old in dropped:
            self._file_progress.pop(old, None)
        header = self.file_buffers.header(req_id)
        count = self.file_buffers.num_samples(req_id)
        if header.shape[0] != 3 or int(header[2]) != count:
            raise ValueError(
                f"file request {req_id!r} header {header.tolist()} disagrees "
                f"with its {count} samples"
            )
        num_prompts = int(self.model.config.num_prompts)
        if not 0 <= int(header[0]) < len(CADENCES) or not (
            0 <= int(header[1]) < num_prompts
        ):
            raise ValueError(f"file request {req_id!r} has a bad header")
        self._file_progress[req_id] = _FileProgress()

    # @spec FILE-ENG-001, FILE-ENG-003, FILE-ENG-004, FILE-BUF-002, FILE-ROW-002
    def plan_step(
        self,
        scheduler_output: SchedulerOutput,
        *,
        queue_lens: dict[str, int] | None = None,
        stream_rows: dict[str, tuple[int, int]] | None = None,
    ) -> StepPlan:
        """Plan the step's rows once: FILE rows resolve their action from the
        label queue length, stage the one chunk (or tail) they advance through
        ``file_buffers.stage`` and advance their position.

        ``queue_lens`` maps request IDs to the label queue length the device
        book reports (default 0); ``stream_rows`` maps stream request IDs to
        ``(geometry_index, deadline)``.
        """
        cached = scheduler_output.scheduled_cached_reqs
        req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
        req_ids += list(cached.req_ids)
        rows: list[StepRow] = []
        for req_id in req_ids:
            progress = self._file_progress.get(req_id)
            if req_id in self._failed_files:
                rows.append(
                    StepRow(
                        req_id,
                        0,
                        None,
                        file_rows.FileStepPlan(FileAction.END, 0, 0),
                        failed=True,
                    )
                )
                continue
            if progress is None:
                key = (stream_rows or {}).get(req_id)
                if key is None:
                    rows.append(StepRow(req_id, -1, None, None))
                else:
                    rows.append(StepRow(req_id, int(key[0]), int(key[1]), None))
                continue
            geometry, locale_row, total = (
                int(v) for v in self.file_buffers.header(req_id)
            )
            chunk_samples = list(RAW_SAMPLES_PER_CHUNK.values())[geometry]
            plan = file_rows.plan_file_step(
                queue_len=int((queue_lens or {}).get(req_id, 0)),
                position=progress.position,
                total_samples=total,
                chunk_samples=chunk_samples,
                tail_done=progress.tail_done,
            )
            device_chunk = None
            header = None
            sequence = 0
            if plan.advances_audio:
                if plan.length:
                    view = self.file_buffers.stage(req_id, plan.start, plan.length)
                    device_chunk = file_rows.copy_chunk_to_device(
                        view, device=self.device
                    )
                sequence = plan.start // chunk_samples
                header = file_envelope_header(
                    valid_samples=plan.length,
                    geometry_id=geometry,
                    final_tail=plan.action is FileAction.TAIL,
                    prompt_index=locale_row,
                    chunk_sequence=sequence,
                    now_ms=int(time.time() * 1000),
                )
                progress.position = plan.start + plan.length
                progress.tail_done = plan.action is FileAction.TAIL
            rows.append(
                StepRow(
                    req_id,
                    geometry,
                    None,
                    plan,
                    samples=device_chunk,
                    header=header,
                    chunk_sequence=sequence,
                )
            )
        return StepPlan(rows=rows, groups=file_rows.plan_chunk_groups(rows))

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        del req_index
        handoff = self.model._native_burst_handoff
        if handoff is not None:
            from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
                validate_native_burst_history,
                validate_native_burst_sampling,
            )

            validate_native_burst_sampling(
                new_req_data.sampling_params,
                park_id=int(self.model.config.eos_token_id),
                capacity=handoff.max_tokens - 1,
            )
            validate_native_burst_history(
                new_req_data,
                max_model_len=self.max_model_len,
                burst_tokens=handoff.max_tokens,
            )
        self._metadata[new_req_data.req_id] = new_req_data

    def remove_request(self, req_id: str) -> None:
        # Core also calls this when replacing a parked turn. prepare_step prunes
        # terminal identities; the same live handle preserves its acoustic state.
        self._metadata.pop(req_id, None)

    # @spec REG-MEM-013
    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        handoff = getattr(getattr(self, "model", None), "_native_burst_handoff", None)
        if handoff is None:
            return None
        from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import NativeBurstSampler

        state = self

        class Sampler(NativeBurstSampler):
            def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
                if state._dummy_run:
                    assert handoff is not None
                    with handoff.dummy_sampling(
                        park_id=int(state.model.config.eos_token_id)
                    ):
                        return super().__call__(logits, input_batch)
                return super().__call__(logits, input_batch)

        return Sampler(sampler, handoff), None

    # @spec REG-MS-002
    def dummy_inputs_embeds(
        self, num_tokens: int, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The dummy forward reads no embedding values, so randomized
        # ``input_ids`` need not be embedded.
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        self._scheduled_encoder_inputs = scheduled_encoder_inputs
        self._features = None
        if self.supports_mm_inputs:
            self.execute_mm_encoder(scheduled_encoder_inputs)
            mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
            self._features = self.encoder_runner.get_inputs_embeds(
                input_batch.input_ids[: input_batch.num_tokens], mm_embeds, is_mm_embed
            )[: input_batch.num_tokens_after_padding]
        return self._features

    # @spec REG-MEM-012
    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        req_ids = tuple(input_batch.req_ids)
        if self._dummy_run:
            snapshot = ProjectionSnapshot(
                self._epoch,
                req_ids,
                {},
                None,
                (),
                (),
                input_batch,
                req_states,
                {},
                dummy_run=True,
                is_profile=self._is_profile,
                no_page_io=True,
            )
        else:
            if self.state_slots is None:
                raise RuntimeError("persistent-state bindings are not initialized")
            if any(int(n) != 1 for n in input_batch.num_scheduled_tokens):
                raise ValueError("every streaming row must be single-token")
            layer = self.model._persistent_state_layer.prefix
            # @spec REG-STATE-002: handles come from the scheduler output;
            # rows are bound here, inside the live step.
            assert self._scheduler_output is not None
            handles = self._scheduler_output.state_slot_handles or {}
            bindings = tuple(
                RowBinding(req_id, handle, self._initialized.get(req_id) != handle)
                for req_id in req_ids
                for handle in (handles[req_id][layer],)
            )
            batch = StateBatch(
                self.state_slots,
                layer,
                bindings,
                self.model._state_spec,
                int(self.model.config.n_layers),
            )
            token_rows = [self._scheduled_tokens.get(req_id, ()) for req_id in req_ids]
            if any(len(tokens) != 1 for tokens in token_rows):
                raise ValueError("every streaming row needs one scheduled host token")
            step_plan = self._plan_production_step(
                req_ids, bindings, batch, token_rows, input_batch
            )
            snapshot = ProjectionSnapshot(
                self._epoch,
                req_ids,
                self._scheduled_encoder_inputs,
                self._features,
                bindings,
                tuple((b.slot_id,) for b in bindings),
                input_batch,
                req_states,
                {},
                {r: self._metadata[r] for r in req_ids},
                token_ids=tuple(tokens[0] for tokens in token_rows),
                state_batch=batch,
                file_rows={
                    row.request_id: row for row in step_plan.rows if row.is_file
                },
                geometry_order=step_plan.geometry_order,
            )
        self._snapshot = snapshot
        return {"persistent_state_projection": snapshot}

    def _plan_production_step(
        self,
        req_ids: tuple[str, ...],
        bindings: tuple[RowBinding, ...],
        batch: StateBatch,
        token_rows: list[tuple[int, ...]],
        input_batch: InputBatch,
    ) -> StepPlan:
        """Make the step's plan once: read the label queue lengths FILE rows
        need, resolve stream rows' geometry and deadline, then plan."""
        assert self._scheduler_output is not None
        if not any(
            r in self._file_progress or r in self._failed_files for r in req_ids
        ):
            # A stream-only step: no FILE row, so no planning work.
            return StepPlan(rows=[], groups=[])
        config = self.model.config
        queue_lens: dict[str, int] = {}
        unknown = []
        for binding in bindings:
            if binding.request_id not in self._file_progress:
                continue
            known = self._file_progress[binding.request_id].known_queue
            if binding.fresh:
                queue_lens[binding.request_id] = 0
            elif known is not None:
                queue_lens[binding.request_id] = known
            else:
                unknown.append(binding)
        if unknown:
            from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
                QUEUE_HEAD,
                QUEUE_LEN,
            )

            book = batch.pools.replay_book
            slots = torch.tensor([b.slot_id for b in unknown], device=book.device)
            # One small read per file row after an advancing step; a drain
            # step knows its remaining count without reading.
            picked = book.index_select(0, slots).cpu()
            for binding, row in zip(unknown, picked, strict=True):
                queue_lens[binding.request_id] = int(row[QUEUE_LEN] - row[QUEUE_HEAD])
        now_ns = time.time_ns()
        stream_rows: dict[str, tuple[int, int]] = {}
        for index, req_id in enumerate(req_ids):
            token = token_rows[index][0]
            if req_id in self._file_progress or token != int(
                config.audio_chunk_token_id
            ):
                continue
            header = resolve_row_envelope_header(
                scheduled_token_id=token,
                placeholder_id=int(config.audio_chunk_token_id),
                num_computed_tokens=int(input_batch.num_computed_tokens_np[index]),
                mm_features=self._metadata[req_id].mm_features,
                scheduled_encoder_input_ids=self._scheduled_encoder_inputs.get(
                    req_id, ()
                ),
            )
            assert header is not None
            stream_rows[req_id] = stream_order_key(header, now_ns=now_ns)
        plan = self.plan_step(
            self._scheduler_output, queue_lens=queue_lens, stream_rows=stream_rows
        )
        for row in plan.rows:
            progress = self._file_progress.get(row.request_id)
            if progress is not None and row.file_plan is not None:
                # DRAIN emits one queued label; any other step leaves the
                # count for the device to report.
                if row.file_plan.action is FileAction.DRAIN:
                    progress.known_queue = queue_lens[row.request_id] - 1
                else:
                    progress.known_queue = None
        return plan

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.dummy_run:
            return
        _statuses, failed = self.model.collect_commit_status()
        if failed:
            assert self.state_slots is not None
            self.state_slots.fail_requests(failed)
        self._initialized.update(
            (b.request_id, b.handle)
            for b in snapshot.bindings
            if b.request_id not in failed
        )
        self.model.consume_batch_stats()
        self._snapshot = None

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        raise RuntimeError("Nemotron does not support outer CUDA graph capture")

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        if ubatch_idx or cudagraph_mode == CUDAGraphMode.FULL or for_capture:
            raise RuntimeError("Nemotron does not support DBO or outer graph capture")
        return {}
