# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv2 projection of the state layer's views into Nemotron transactions.

The state is one ``MambaSpec`` page per request (``state_layer``), allocated
by core's ``MambaManager``. Core gives the model state no step record: the
model state rebuilds each step from ``add_request`` / ``remove_request`` and
the ``InputBatch``. Per request it records, at ``add_request``, the prompt,
the slot (``block_ids[0][0]``), the ``NewRequestData`` and an admission
generation; a row is fresh when its ``num_computed_tokens`` is 0 (the Mamba
builder's ``has_initial_states_p``), and a fresh row's page is initialized
before the forward. A dummy step is a step with a row ``add_request`` never
recorded; the profiling dummy step is a dummy step with no ``prepare_attn``
call before ``prepare_inputs``.

``remove_request`` records one pending removal; the next model-state call
decides it: an ``add_request`` of the same request that continues it (same
slot, ``num_computed_tokens > 0``) replaces the turn, every other call retires
the request first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_nemotron_asr_streaming.engine_flags import (
    check_engine_config,
    selects_hook_captured_graph,
)
from vllm_nemotron_asr_streaming.lookahead import geometry_index, served_lookahead
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
from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
    StateLayout,
    project_nemotron_state_pools,
)

#: Host-side stand-in for a scheduled token the sampler produced on the
#: device: never a valid token id, so never the CHUNK or file carrier.
SAMPLED_TOKEN = -1


def _cuda_platform() -> bool:
    """Whether vLLM's current platform is CUDA. The platform module is a
    graph-runtime dependency (REG-014), imported only when a hook-captured
    graph is selected."""
    from vllm.platforms import current_platform

    return bool(current_platform.is_cuda())


@dataclass(frozen=True)
class SlotHandle:
    """A request incarnation's slot and admission generation. The slot is
    the request's ``MambaManager`` block; the generation comes from the model
    state's own counter and is new for every fresh incarnation."""

    slot: int
    generation: int


@dataclass
class _Request:
    """What ``add_request`` recorded for one live request."""

    handle: SlotHandle
    prompt: tuple[int, ...]
    metadata: NewRequestData


@dataclass(frozen=True)
class RowBinding:
    request_id: str
    handle: SlotHandle
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
    #: The step's planning-failed set (TXG-FAIL-008): the model state records
    #: the rows it fails at planning, and the model's forward adds every row
    #: the plan binds with its planned-rejection bit (TXG-COV-010).
    planning_failed: set[str] = field(default_factory=set)
    #: The file requests whose file action this step planned (FILE-QC-006).
    planned_files: tuple[str, ...] = ()
    #: Each row's admission generation in this step's plan, recorded by the
    #: model's forward (FILE-QC-013).
    admission_generations: dict[str, int] = field(default_factory=dict)


@dataclass
class _FileProgress:
    """A FILE request's progress through its file, kept for its life."""

    position: int = 0
    tail_done: bool = False
    #: Queued labels not yet emitted (FILE-QC-007: 0 at take). Only an
    #: accepted count from the step's collection changes it (FILE-QC-006).
    queue_count: int = 0
    #: The count's validity is unknown: a planned row had no accepted count
    #: and no rejection (FILE-QC-009). The retained number is kept.
    queue_count_unknown: bool = False
    #: The admission generation of the incarnation taken at the first
    #: schedule, recorded from its first clean count (FILE-QC-013).
    admission_generation: int | None = None
    #: Steps whose file action was planned.
    planned_steps: int = 0


class _ResidentCommitSink:
    def __init__(self, sink: Any, slots: tuple[int, ...], request_ids: tuple[str, ...]):
        self.sink = sink
        self.slots = slots
        self.request_ids = request_ids

    # @spec REG-STATE-004, REG-STATE-006
    def reserve(self, plan: Any) -> Any:
        if tuple(b.request_id for b in plan.bindings) != self.request_ids:
            raise ValueError("commit row order differs from resident bindings")
        if tuple(b.block_id for b in plan.bindings) != self.slots:
            raise ValueError("commit slots differ from resident bindings")
        return self.sink.reserve(plan)


class StateBatch:
    """Zero-copy resident pools of one step's rows.

    The views are read from the state layer when the step is bound, never
    kept across steps: the extensible KV cache rebinds them."""

    # @spec REG-STATE-003, REG-STATE-005
    def __init__(
        self,
        layer: Any,
        bindings: tuple[RowBinding, ...],
        layout: StateLayout,
        n_layers: int,
    ):
        views = layer.component_pools
        if views is None:
            raise RuntimeError("the state pool is not bound")
        num_blocks = layer.num_blocks
        self.layer = layer
        self.request_ids = tuple(b.request_id for b in bindings)
        self.slot_indices = tuple(int(b.slot_id) for b in bindings)
        self.fresh_slots = tuple(int(b.slot_id) for b in bindings if b.fresh)
        outside = [s for s in self.slot_indices if not 0 < s < num_blocks]
        if outside:
            raise ValueError(
                f"state slots {outside} are outside the bound pool of "
                f"{num_blocks} blocks"
            )
        self.pools = project_nemotron_state_pools(views, layout, n_layers=n_layers)

    def initialize_fresh_pages(self) -> None:
        """Initialize the pages of the step's fresh rows, as a newly taken
        page must be (core does not clear a reused block): the model's
        forward calls this before its transaction, on the current stream."""
        self.layer.initialize_pages(self.fresh_slots)

    def commit_sink(self, sink: Any) -> _ResidentCommitSink:
        return _ResidentCommitSink(sink, self.slot_indices, self.request_ids)


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
        #: Each live request's record, from its latest ``add_request``.
        self._requests: dict[str, _Request] = {}
        #: The request ``remove_request`` named last, until the next call
        #: decides it (a turn replacement or a retirement).
        self._pending_removal: str | None = None
        #: The requests added since the last ``prepare_inputs``, in order: the
        #: step's new requests, which plan before its cached ones.
        self._step_added: list[str] = []
        #: The admission-generation counter (``SlotHandle.generation``).
        self._generation = 0
        #: Each row's scheduled host tokens: a new request's at
        #: ``add_request``, every row's at ``prepare_inputs``.
        self._scheduled_tokens: dict[str, tuple[int, ...]] = {}
        #: Whether ``prepare_attn`` ran since the last ``prepare_inputs``.
        self._attn_prepared = False
        self._epoch = 0
        self._snapshot: ProjectionSnapshot | None = None
        self._dummy_run = False
        self._is_profile = False
        self._scheduled_encoder_inputs: dict[str, list[int]] = {}
        self._features: torch.Tensor | None = None

    # Core's graph hooks (vllm#67583), required in every runner configuration
    # (REG-ENG-009). Under ``--enforce-eager`` core calls none of the three
    # capture hooks, and the model captures on its first profiling step.
    def _graphs_by_core(self) -> bool:
        """Whether core captures this model's graphs through
        ``capture_cudagraphs`` (core asks only when the runner is not eager)."""
        return (
            not self.vllm_config.model_config.enforce_eager
            and self.needs_cudagraph_capture()
        )

    # @spec REG-MEM-018
    def prepare_execution_memory(self) -> None:
        self.model.prepare_execution_memory_hook(
            self.device, graphs_by_core=self._graphs_by_core()
        )

    # @spec REG-MEM-019
    def needs_cudagraph_capture(self) -> bool:
        # Configuration and platform only: the engine check accepts a
        # non-eager runner only for a graph the capture hook captures
        # (``selects_hook_captured_graph``). Both are read on the first call
        # and the answer is kept for the model state's lifetime.
        answer = getattr(self, "_needs_cudagraph_capture", None)
        if answer is None:
            answer = bool(
                selects_hook_captured_graph(self.model.config) and _cuda_platform()
            )
            self._needs_cudagraph_capture = answer
        return answer

    # @spec REG-MEM-020, REG-MEM-022
    def capture_cudagraphs(
        self,
        *,
        graph_pool: Any,
        stream: torch.cuda.Stream,
        profile_only: bool,
    ) -> None:
        self.model.capture_graphs_hook(
            self.device,
            graph_pool=graph_pool,
            stream=stream,
            profile_only=profile_only,
        )

    # @spec REG-MEM-021
    def clear_cudagraphs(self) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "clear_graphs_hook"):
            model.clear_graphs_hook()

    # @spec REG-MEM-025
    def post_warmup(self) -> None:
        self.model.post_warmup_hook()

    # @spec REG-MEM-026
    def shutdown(self) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "shutdown_graphs_hook"):
            model.shutdown_graphs_hook()

    # @spec REG-TASK-002
    def get_supported_generation_tasks(self) -> tuple[Any, ...]:
        # Core would infer "generate" from compute_logits; the model emits
        # transcript labels, not text generation.
        return ("transcription", "realtime")

    @property
    def _metadata(self) -> dict[str, NewRequestData]:
        """Each live request's latest ``NewRequestData``."""
        return {r: record.metadata for r, record in self._requests.items()}

    @property
    def _prompts(self) -> dict[str, tuple[int, ...]]:
        """Each live request's latest prompt, as core delivered it."""
        return {r: record.prompt for r, record in self._requests.items()}

    def _state_layer(self) -> Any:
        return self.model._persistent_state_layer

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: Any
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        layer = self._state_layer()
        groups = [list(group.layer_names) for group in kv_cache_config.kv_cache_groups]
        if groups != [[layer.prefix]]:
            raise ValueError(
                "Nemotron requires exactly its one state layer in one KV cache "
                f"group, got {groups}"
            )
        if layer.component_pools is None:
            raise RuntimeError("core did not bind the state pool to the state layer")

    # -- request lifetime (worker side) --------------------------------------

    def _decide_pending(self) -> None:
        """Retire the pending removal, if any: every model-state call other
        than a continuing ``add_request`` of the same request decides it."""
        pending = self._pending_removal
        if pending is not None:
            self._pending_removal = None
            self._retire(pending)

    def _retire(self, req_id: str) -> None:
        """Drop everything kept for ``req_id``: its record (prompt, metadata,
        slot), its registry entry, its file buffer and progress, and its
        failed-file mark."""
        self._requests.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self.model._registry.prune(tuple(self._requests))
        # FILE-BUF-003: a retired request frees its file buffer.
        for freed in self.file_buffers.free_finished((req_id,)):
            self._file_progress.pop(freed, None)
        self._file_progress.pop(req_id, None)
        self._failed_files.discard(req_id)

    def _continues(self, req_id: str, new_req_data: NewRequestData, slot: int) -> bool:
        """Whether ``new_req_data`` is a turn of the live request ``req_id``:
        the same request, prior state (``num_computed_tokens > 0``), the same
        slot."""
        record = self._requests.get(req_id)
        return (
            record is not None
            and int(new_req_data.num_computed_tokens) > 0
            and record.handle.slot == slot
        )

    # @spec FILE-BUF-001, FILE-BUF-004, FILE-BUF-005, FILE-ROW-001
    def _take_file(self, request: NewRequestData) -> None:
        """Take a file request's samples and header into a request-owned
        host buffer at its first schedule and mark it a FILE request.

        A malformed item (the processor refuses these at input processing,
        so only a direct engine client can send one) fails that row alone
        through the transaction's row-failure path; it never raises out of
        the step.
        """
        req_id = request.req_id
        try:
            self._take_file_checked(request)
        except (ValueError, KeyError, TypeError, AttributeError, IndexError):
            # any missing or unreadable field makes the item malformed
            self.file_buffers.free_finished([req_id])
            self._file_progress.pop(req_id, None)
            self._failed_files.add(req_id)

    def _take_file_checked(self, request: NewRequestData) -> None:
        req_id = request.req_id
        if len(request.mm_features) != 1:
            raise ValueError(f"file request {req_id!r} needs exactly one audio item")
        item = request.mm_features[0].data
        if item is None:
            raise ValueError(f"file request {req_id!r} arrived without its samples")
        record = self._requests.get(req_id)
        if record is None:
            raise ValueError(f"file request {req_id!r} has no state slot")
        handle = record.handle
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
        # FILE-PROMPT-007: a header of another geometry than the server's fails
        # the row at planning.
        if int(header[0]) != self._server_geometry():
            raise ValueError(
                f"file request {req_id!r} header geometry {int(header[0])} is "
                f"not the server's geometry {self._server_geometry()}"
            )
        # FILE-QC-007: count 0 and no recorded generation at take.
        self._file_progress[req_id] = _FileProgress()

    def _server_geometry(self) -> int:
        """The server's geometry index (SESS-GEOM-001), read once."""
        geometry = getattr(self, "_server_geometry_index", None)
        if geometry is None:
            geometry = geometry_index(served_lookahead(self.model.config))
            self._server_geometry_index = geometry
        return geometry

    # @spec FILE-ENG-001, FILE-ENG-003, FILE-BUF-002, FILE-ROW-002
    # @spec FILE-QC-010, FILE-QC-011
    def plan_step(
        self,
        scheduler_output: SchedulerOutput | tuple[str, ...],
        *,
        stream_rows: dict[str, tuple[int, int]] | None = None,
    ) -> StepPlan:
        """Plan the step's rows once: FILE rows resolve their action from the
        file's queue count, stage the one chunk (or tail) they advance through
        ``file_buffers.stage`` and advance their position.

        Planning reads no label book and changes no queue count (FILE-QC-011).
        A file that has had a step and whose count's validity is unknown is
        failed alone before its file action is planned (FILE-QC-010).
        ``stream_rows`` maps stream request IDs to ``(geometry_index,
        deadline)``. The rows are the step's request IDs in planning order
        (new requests, then cached ones), given directly or as a scheduler
        output.
        """
        if isinstance(scheduler_output, tuple):
            req_ids = list(scheduler_output)
        else:
            cached = scheduler_output.scheduled_cached_reqs
            req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
            req_ids += list(cached.req_ids)
        rows: list[StepRow] = []
        for req_id in req_ids:
            progress = self._file_progress.get(req_id)
            if (
                progress is not None
                and progress.queue_count_unknown
                and progress.planned_steps > 0
            ):
                self._failed_files.add(req_id)
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
            progress.planned_steps += 1
            plan = file_rows.plan_file_step(
                queue_len=int(progress.queue_count),
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

    # @spec REG-STATE-002
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        del req_index
        req_id = new_req_data.req_id
        groups = new_req_data.block_ids
        if len(groups) != 1 or not groups[0]:
            raise ValueError(
                f"request {req_id!r} needs one state block in one KV cache group"
            )
        slot = int(groups[0][0])
        continues = self._pending_removal == req_id and self._continues(
            req_id, new_req_data, slot
        )
        if continues:
            # A turn replacement: the session keeps its bookkeeping.
            self._pending_removal = None
        else:
            self._decide_pending()
            if req_id in self._requests:
                # Core removes every request before it adds it again.
                raise RuntimeError(
                    f"request {req_id!r} was added again without a removal"
                )
            if int(new_req_data.num_computed_tokens) != 0:
                raise RuntimeError(
                    f"request {req_id!r} arrived with prior state "
                    f"({new_req_data.num_computed_tokens} computed tokens) but "
                    "continues no live request on its slot"
                )
        tokens = new_req_data.prefill_token_ids
        if tokens is None:
            raise ValueError("state step omitted new host token slice")
        if continues:
            handle = self._requests[req_id].handle
        else:
            self._generation += 1
            handle = SlotHandle(slot, self._generation)
        self._requests[req_id] = _Request(handle, tuple(tokens), new_req_data)
        self._step_added.append(req_id)
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
        carrier = getattr(
            getattr(self.model, "config", None), "file_carrier_token_id", None
        )
        start = int(new_req_data.num_computed_tokens)
        # Every row schedules one token (``prepare_inputs`` refuses others).
        window = tuple(tokens[start : start + 1])
        self._scheduled_tokens[req_id] = window
        if carrier is not None and window == (int(carrier),):
            self._take_file(new_req_data)

    def remove_request(self, req_id: str) -> None:
        # Core calls this for a finished request and before it adds a request
        # again (a session's next turn): the next call decides which.
        self._decide_pending()
        if req_id in self._requests:
            self._pending_removal = req_id

    # @spec REG-MEM-013
    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        handoff = getattr(getattr(self, "model", None), "_native_burst_handoff", None)
        if handoff is None:
            return None
        from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
            NativeBurstSampler,
        )

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

    def _row_tokens(self, input_batch: InputBatch) -> list[tuple[int, ...]]:
        """Each row's scheduled host tokens. Core sends no token slices for
        cached requests unless pipeline parallelism is on, which this model
        refuses. A row's host token is therefore read from the prompt this
        state recorded when core delivered the request (a session turn
        arrives that way too, with its cumulative prompt starting at position
        zero). A position past the recorded prompt holds a token the sampler
        produced on the device; the host never reads it, and no sampled
        token is a CHUNK carrier, so such a row is marked with
        ``SAMPLED_TOKEN``."""
        rows: list[tuple[int, ...]] = []
        for index, req_id in enumerate(input_batch.req_ids):
            prompt = self._requests[req_id].prompt
            start = int(input_batch.num_computed_tokens_np[index])
            count = int(input_batch.num_scheduled_tokens[index])
            window = prompt[start : start + count]
            if window and len(window) != count:
                raise ValueError(
                    f"request {req_id!r} schedules tokens across the end of its prompt"
                )
            rows.append(window or (SAMPLED_TOKEN,) * count)
        return rows

    # @spec REG-MEM-001, REG-MEM-012, REG-STATE-002, REG-STATE-007
    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        self._decide_pending()
        attn_prepared, self._attn_prepared = self._attn_prepared, False
        added, self._step_added = self._step_added, []
        self._epoch = getattr(self, "_epoch", 0) + 1
        self._snapshot = None
        handoff = self.model._native_burst_handoff
        if handoff is not None:
            handoff.clear()
        req_ids = tuple(input_batch.req_ids)
        # A dummy batch carries fresh random request IDs that no add_request
        # recorded; only the memory profile's dummy step (core's
        # ``skip_attn``) builds no attention metadata.
        dummy_run = any(r not in self._requests for r in req_ids)
        is_profile = dummy_run and not attn_prepared
        self._dummy_run = dummy_run
        self._is_profile = is_profile
        if dummy_run:
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
                is_profile=is_profile,
                no_page_io=True,
            )
        else:
            if any(int(n) != 1 for n in input_batch.num_scheduled_tokens):
                raise ValueError("every streaming row must be single-token")
            layer = self._state_layer()
            bindings = tuple(
                RowBinding(
                    req_id,
                    self._requests[req_id].handle,
                    int(input_batch.num_computed_tokens_np[index]) == 0,
                )
                for index, req_id in enumerate(req_ids)
            )
            batch = StateBatch(
                layer,
                bindings,
                self.model._state_spec,
                int(self.model.config.n_layers),
            )
            token_rows = self._row_tokens(input_batch)
            self._scheduled_tokens = dict(zip(req_ids, token_rows, strict=True))
            if any(len(tokens) != 1 for tokens in token_rows):
                raise ValueError("every streaming row needs one scheduled host token")
            # The step's new requests plan first, as core ordered them.
            step_added = set(added)
            plan_order = tuple(r for r in added if r in set(req_ids)) + tuple(
                r for r in req_ids if r not in step_added
            )
            step_plan = self._plan_production_step(
                req_ids, bindings, batch, token_rows, input_batch, plan_order
            )
            # TXG-FAIL-008: the rows the model state failed at planning.
            planning_failed = {row.request_id for row in step_plan.rows if row.failed}
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
                {r: self._requests[r].metadata for r in req_ids},
                token_ids=tuple(tokens[0] for tokens in token_rows),
                state_batch=batch,
                file_rows={
                    row.request_id: row for row in step_plan.rows if row.is_file
                },
                geometry_order=step_plan.geometry_order,
                planning_failed=planning_failed,
                planned_files=tuple(
                    row.request_id
                    for row in step_plan.rows
                    if row.is_file and not row.failed
                ),
            )
        self._scheduled_encoder_inputs = {}
        self._features = None
        self._snapshot = snapshot
        return {"persistent_state_projection": snapshot}

    def _plan_production_step(
        self,
        req_ids: tuple[str, ...],
        bindings: tuple[RowBinding, ...],
        batch: StateBatch,
        token_rows: list[tuple[int, ...]],
        input_batch: InputBatch,
        plan_order: tuple[str, ...] | None = None,
    ) -> StepPlan:
        """Make the step's plan once: resolve stream rows' geometry and
        deadline, then plan the rows in ``plan_order`` (default: the batch
        order). FILE rows plan from their file's queue count; no label book is
        read (FILE-QC-011)."""
        del bindings, batch
        if plan_order is None:
            plan_order = req_ids
        if not any(
            r in self._file_progress or r in self._failed_files for r in req_ids
        ):
            # A stream-only step: no FILE row, so no planning work.
            return StepPlan(rows=[], groups=[])
        config = self.model.config
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
                mm_features=self._requests[req_id].metadata.mm_features,
                scheduled_encoder_input_ids=self._scheduled_encoder_inputs.get(
                    req_id, ()
                ),
            )
            assert header is not None
            stream_rows[req_id] = stream_order_key(header, now_ns=now_ns)
        return self.plan_step(plan_order, stream_rows=stream_rows)

    # @spec SESS-MODEL-003, TXG-FAIL-008, FILE-QC-006, FILE-QC-012
    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.dummy_run:
            return
        collection = self.model.collect_commit_status()
        # Every collected failure (any nonzero status, SESS-MODEL-003), every
        # planning-failed row (TXG-FAIL-008), and every file row with no
        # accepted count (FILE-QC-012) fails a request. Core offers no channel
        # that ends one request with an error, so a failure ends the engine.
        failed = set(collection.failed) | set(snapshot.planning_failed)
        failed |= self._apply_queue_counts(snapshot, collection)
        self.model.consume_batch_stats()
        self._snapshot = None
        if failed:
            raise RuntimeError(
                "Nemotron streaming rows failed and this core has no "
                "per-request failure channel: " + ", ".join(sorted(failed))
            )

    # @spec FILE-QC-006, FILE-QC-008, FILE-QC-009, FILE-QC-012, FILE-QC-013
    def _apply_queue_counts(
        self, snapshot: ProjectionSnapshot, collection: Any
    ) -> set[str]:
        """Apply the step's collected counts to the file progress of the file
        rows the step planned; return the files that had no count accepted.

        A file progress records its admission generation from its first
        step's count of the incarnation the step bound, before any count is
        applied (FILE-QC-013). A count is accepted only for a planned file row
        of the same request and recorded generation whose status resolved to
        0 (FILE-QC-006). A rejected row keeps its count (FILE-QC-008); a row
        with no outcome keeps the number and marks its validity unknown
        (FILE-QC-009). No count is decremented on the host.
        """
        statuses = collection.statuses
        counts = tuple(collection.queue_counts)
        unaccepted: set[str] = set()
        for req_id in snapshot.planned_files:
            progress = self._file_progress.get(req_id)
            if progress is None:
                continue
            status = statuses.get(req_id)
            if progress.admission_generation is None and status == 0:
                bound = snapshot.admission_generations.get(req_id)
                if bound is not None and any(
                    c.request_id == req_id and c.admission_generation == bound
                    for c in counts
                ):
                    progress.admission_generation = bound
            accepted = None
            if status == 0 and progress.admission_generation is not None:
                accepted = next(
                    (
                        c
                        for c in counts
                        if c.request_id == req_id
                        and c.admission_generation == progress.admission_generation
                    ),
                    None,
                )
            if accepted is not None:
                progress.queue_count = int(accepted.count)
                progress.queue_count_unknown = False
                continue
            if status is None or status == 0:
                progress.queue_count_unknown = True
            unaccepted.add(req_id)
        return unaccepted

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
        self._decide_pending()
        # The model builds no Mamba metadata: it plans its rows on the host.
        self._attn_prepared = True
        return {}
