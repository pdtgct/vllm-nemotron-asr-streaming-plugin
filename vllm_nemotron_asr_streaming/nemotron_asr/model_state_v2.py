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

A stream request's ``remove_request`` records one pending removal; the next
model-state call decides it: an ``add_request`` of the same request that
continues it (same slot, ``num_computed_tokens > 0``) replaces the turn, every
other call retires the request first. A file request's ``remove_request``
retires it within the call.

A row whose request causes a planning fault (a request not shaped as the
model's processor builds it, which ``add_request`` records; more than one
scheduled token; a window past its recorded prompt; a malformed envelope or
file item; a malformed endpoint policy) fails alone at planning, before core's
encoder path reads it; ``postprocess_state`` records it with the step's other failed
rows, and core ends those requests through ``take_failed_requests``. A
violation of core's model-state contract raises.

The model owns every graph it runs: core's runner captures no graph over its
forward (``get_additional_cg_support``, ``initialize_kv_cache``,
``prepare_dummy_inputs`` and ``prepare_attn`` refuse every other outcome).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_nemotron_asr_streaming import engine_flags
from vllm_nemotron_asr_streaming.engine_flags import selects_hook_captured_graph
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
    RequestRowFault,
    check_envelope_header,
    encoder_item_fault,
    file_envelope_header,
    request_shape_fault,
    resolve_row_envelope_header,
    stream_order_key,
)
from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
    StateLayout,
    project_nemotron_state_pools,
)

logger = init_logger(__name__)

#: Host-side stand-in for a scheduled token the sampler produced on the
#: device: never a valid token id, so never the CHUNK or file carrier. A row
#: that fails at planning before its tokens are classified carries it too.
SAMPLED_TOKEN = -1

#: The backend name of the model state's runner-graph support (REG-ENG-016).
CG_SUPPORT_NAME = "nemotron_asr_streaming model state (captures its own graphs)"

#: What every refusal of a runner-graph path states (REG-ENG-017, REG-ENG-029).
NO_RUNNER_GRAPH = (
    "the Nemotron streaming model captures its own graphs and supports no runner graph"
)

#: Whether the runner-mode line of REG-ENG-018 was logged in this process.
_runner_modes_logged = False


def _mode_name(mode: Any) -> str:
    """A ``cudagraph_mode`` or compilation mode by its whole name."""
    if mode is None:
        return "unset"
    return str(getattr(mode, "name", mode))


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
    """What ``add_request`` recorded for one held request. ``is_file``: its
    scheduled token at its first ``add_request`` was the file carrier.
    ``fault``: why the request, as its latest ``add_request`` delivered it, is
    not shaped as the model's processor builds it (``request_shape_fault``),
    or ``None``; a request with a fault fails at planning (REG-STATE-015)."""

    handle: SlotHandle
    prompt: tuple[int, ...]
    metadata: NewRequestData
    is_file: bool = False
    fault: str | None = None


@dataclass(frozen=True)
class _Screen:
    """One real step's rows as planning screens them, before core's encoder
    path reads them: each row's scheduled host tokens, the rows failed at
    planning for a fault their request caused (REG-STATE-008, REG-STATE-014,
    REG-STATE-015), each clean stream CHUNK row's validated envelope header,
    and the scheduled encoder inputs core's encoder may run (every input
    whose own item the model's encoder takes)."""

    input_batch: Any
    token_rows: tuple[tuple[int, ...], ...]
    failed: frozenset[str]
    headers: dict[str, tuple[float, ...]]
    encoder_inputs: dict[str, list[int]]


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
    #: Each clean stream CHUNK row's validated envelope header, from the
    #: step's screening (REG-STATE-015).
    stream_headers: dict[str, tuple[float, ...]] = field(default_factory=dict)


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
        # The bound views; the state layer raises when core bound none.
        views = layer.component_pools
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

    # @spec REG-ENG-006, REG-ENG-027
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The final engine configuration may differ from what the
        # configuration hook saw: check it again before any state exists,
        # the hook configuration's conditions through
        # ``engine_flags.runner_mode_refusals`` (stage ``"constructor"``).
        engine_flags.check_engine_config(
            self.vllm_config, "the Nemotron streaming model state", resolved=True
        )
        compilation = getattr(self.vllm_config, "compilation_config", None)
        #: The modes this construction saw (REG-ENG-018).
        self._construction_modes = (
            getattr(compilation, "cudagraph_mode", None),
            getattr(compilation, "mode", None),
        )
        self.file_buffers = FileHostBuffers()
        self._file_progress: dict[str, _FileProgress] = {}
        self._failed_files: set[str] = set()
        #: Each held request's record, from its latest ``add_request``.
        self._requests: dict[str, _Request] = {}
        #: The stream request ``remove_request`` named last, until the next
        #: call decides it (a turn replacement or a retirement).
        self._pending_removal: str | None = None
        #: The ID and slot of the file request ``remove_request`` retired last,
        #: until the next call: core continues a file request only for a
        #: direct engine client, and that continuation is the next call
        #: (REG-STATE-012, REG-STATE-016).
        self._retired_file: tuple[str, int] | None = None
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
        #: The current step's screening, from ``prepare_inputs_embeds``.
        self._screen: _Screen | None = None
        #: The requests ``postprocess_state`` recorded as failed since the
        #: last ``take_failed_requests`` (REG-MS-003, REG-MS-004).
        self._failed_requests: set[str] = set()

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

    # @spec REG-MEM-020, REG-MEM-022, TXG-CAP-008
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
            # TXG-CAP-008: core's state-group layout, read during this call
            # only, is how the transaction-graph part reaches the pools.
            state_group_layouts=lambda: self.state_group_layouts,
        )

    # @spec REG-MEM-021, REG-STATE-018
    def clear_cudagraphs(self) -> None:
        model = getattr(self, "model", None)
        try:
            if model is not None and hasattr(model, "clear_graphs_hook"):
                model.clear_graphs_hook()
        finally:
            # The plugin's own references to the bound KV cache (the profiling
            # one, during core's graph-memory estimate) go here, whether or
            # not anything was captured.
            layer = getattr(model, "_persistent_state_layer", None)
            if layer is not None:
                layer.drop_views()

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

    # @spec REG-ENG-016
    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        # Core's runner captures no graph over the forward: its mode
        # resolution takes the least capable backend and reduces every mode
        # it can reduce to NONE.
        return AttentionCGSupport.NEVER, CG_SUPPORT_NAME

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

    # @spec REG-ENG-017, REG-ENG-018, REG-STATE-013
    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: Any
    ) -> None:
        compilation = getattr(self.vllm_config, "compilation_config", None)
        resolved = getattr(compilation, "cudagraph_mode", None)
        if resolved is not None and resolved != CUDAGraphMode.NONE:
            raise RuntimeError(
                f"core resolved cudagraph_mode {_mode_name(resolved)}, but "
                f"{NO_RUNNER_GRAPH}: core's runner may capture no graph over "
                "its forward (pass -cc.cudagraph_mode none)"
            )
        self._log_runner_modes(resolved)
        super().initialize_kv_cache(kv_cache_config, block_tables)
        layer = self._state_layer()
        groups = [list(group.layer_names) for group in kv_cache_config.kv_cache_groups]
        if groups != [[layer.prefix]]:
            raise ValueError(
                "Nemotron requires exactly its one state layer in one KV cache "
                f"group, got {groups}"
            )
        try:
            layer.component_pools  # noqa: B018 - the bound views, or a raise
        except RuntimeError as error:
            raise RuntimeError(
                "core did not bind the state pages to the state layer of the "
                f"KV cache groups {groups}"
            ) from error

    def _log_runner_modes(self, resolved: Any) -> None:
        """REG-ENG-018: once per process, outside ``--enforce-eager``."""
        global _runner_modes_logged
        if _runner_modes_logged or self.vllm_config.model_config.enforce_eager:
            return
        _runner_modes_logged = True
        graph_mode, compilation_mode = self._construction_modes
        logger.info(
            "Nemotron streaming: core's runner captures no graph over the "
            "model's forward (construction saw cudagraph_mode %s and "
            "compilation mode %s; resolved cudagraph_mode %s), and the model "
            "captures its own graphs through the capture hook "
            "(capture_cudagraphs)",
            _mode_name(graph_mode),
            _mode_name(compilation_mode),
            _mode_name(resolved),
        )

    # -- request lifetime (worker side) --------------------------------------

    def _decide_pending(self) -> None:
        """Retire the pending removal, if any: every model-state call other
        than a continuing ``add_request`` of the same request decides it. The
        call also forgets the file request retired last."""
        self._retired_file = None
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
    # @spec REG-STATE-015
    def _take_file(self, request: NewRequestData) -> None:
        """Take a file request's samples and header into a request-owned
        host buffer at its first schedule and mark it a FILE request.

        ``add_request`` calls it only for a file request that the
        request-shape rule (``request_shape_fault``) accepts: the file prompt
        alone, with one feature on it whose item is a file's and passes the
        item rule (``encoder_item_fault``), by which screening also drops
        encoder inputs. A malformed header fails that row alone through the
        transaction's row-failure path; it never raises out of the step. The
        model's processor builds no malformed file request (it refuses audio
        that is not one-dimensional and a bad header at input processing), so
        only a direct engine client can send one.
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
        (feature,) = request.mm_features
        item = feature.data
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

    # @spec REG-STATE-002, REG-STATE-007, REG-STATE-011, REG-STATE-012
    # @spec REG-STATE-015
    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        del req_index
        req_id = new_req_data.req_id
        groups = new_req_data.block_ids
        if len(groups) != 1 or len(groups[0]) != 1:
            raise ValueError(
                f"request {req_id!r} needs exactly one state block in one KV "
                f"cache group, got block_ids {[list(g) for g in groups]}"
            )
        slot = int(groups[0][0])
        retired_file, self._retired_file = self._retired_file, None
        continues = self._pending_removal == req_id and self._continues(
            req_id, new_req_data, slot
        )
        fault: str | None = None
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
                if retired_file != (req_id, slot):
                    raise RuntimeError(
                        f"request {req_id!r} arrived with prior state "
                        f"({new_req_data.num_computed_tokens} computed tokens) but "
                        "continues no live request on its slot"
                    )
                # REG-STATE-012: the file request remove_request just retired,
                # continued by core because a direct engine client opened it
                # as a stream; the model serves a file as one request.
                fault = "core continued a file request, which is one request"
        tokens = new_req_data.prefill_token_ids
        if tokens is None:
            raise ValueError("state step omitted new host token slice")
        carrier = getattr(
            getattr(self.model, "config", None), "file_carrier_token_id", None
        )
        start = int(new_req_data.num_computed_tokens)
        # The scheduled token; a row of more tokens fails at planning.
        window = tuple(tokens[start : start + 1])
        if continues:
            previous = self._requests[req_id]
            handle, is_file = previous.handle, previous.is_file
        else:
            self._generation += 1
            handle = SlotHandle(slot, self._generation)
            is_file = start == 0 and carrier is not None and window == (int(carrier),)
        if fault is None:
            fault = self._request_shape_fault(tokens, new_req_data)
        self._requests[req_id] = _Request(
            handle, tuple(tokens), new_req_data, is_file=is_file, fault=fault
        )
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
        self._scheduled_tokens[req_id] = window
        if is_file and not continues:
            if fault is None:
                self._take_file(new_req_data)
            else:
                # Not shaped as the processor builds a file request: nothing
                # is taken, and the file fails at planning.
                self._failed_files.add(req_id)

    # @spec REG-STATE-015
    def _request_shape_fault(
        self, tokens: Any, new_req_data: NewRequestData
    ) -> str | None:
        """Why ``new_req_data``, as core delivered it, is not shaped as the
        model's processor builds it (``request_shape_fault``), or ``None``. A
        feature without position metadata raises: core's msgpack decoder
        refuses one, so only core can deliver it (REG-STATE-015)."""
        config = getattr(self.model, "config", None)
        chunk = getattr(config, "audio_chunk_token_id", None)
        file_carrier = getattr(config, "file_carrier_token_id", None)
        if chunk is None or file_carrier is None:
            return None
        return request_shape_fault(
            tokens,
            new_req_data.mm_features,
            num_computed_tokens=int(new_req_data.num_computed_tokens),
            chunk_token_id=int(chunk),
            file_carrier_id=int(file_carrier),
            carrier_width=int(config.hidden_size),
        )

    # @spec REG-STATE-011, REG-STATE-016, FILE-BUF-003
    def remove_request(self, req_id: str) -> None:
        # Core calls this for a finished request and before it adds a request
        # again (a stream's next turn). Core never continues a file request,
        # so it retires now; a stream's removal is decided by the next call.
        self._decide_pending()
        record = self._requests.get(req_id)
        if record is None:
            return
        if record.is_file:
            self._retire(req_id)
            # Kept until the next call, which core's continuation of the file
            # would be (REG-STATE-012, REG-STATE-016).
            self._retired_file = (req_id, record.handle.slot)
        else:
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

    # @spec REG-STATE-015
    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        self._scheduled_encoder_inputs = scheduled_encoder_inputs
        self._features = None
        # A row whose request caused a planning fault leaves core's encoder
        # path, so core never meets the fault (it would raise for the whole
        # step): its features are not gathered, and an encoder input whose
        # own item the model's encoder refuses is not run. Every other input
        # runs, a failed row's included: core's scheduler encodes an item
        # once and treats it as cached for each request that shares it.
        screen = self._screen_rows(input_batch, scheduled_encoder_inputs)
        self._screen = screen
        if screen is None:
            failed, encoder_inputs = frozenset(), scheduled_encoder_inputs
        else:
            failed, encoder_inputs = screen.failed, screen.encoder_inputs
        if self.supports_mm_inputs:
            self.execute_mm_encoder(encoder_inputs)
            mm_embeds, is_mm_embed = self._gather_mm_embeddings(input_batch, failed)
            self._features = self.encoder_runner.get_inputs_embeds(
                input_batch.input_ids[: input_batch.num_tokens], mm_embeds, is_mm_embed
            )[: input_batch.num_tokens_after_padding]
        return self._features

    def _gather_mm_embeddings(
        self, input_batch: InputBatch, failed: frozenset[str]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Core's gather of the step's cached multimodal embeddings, over
        the rows planning did not fail."""
        if not failed:
            return self.gather_mm_embeddings(input_batch)
        keep = [i for i, r in enumerate(input_batch.req_ids) if r not in failed]
        return self.encoder_runner.gather_mm_embeddings(
            [input_batch.req_ids[i] for i in keep],
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens[keep],
            input_batch.query_start_loc_np[keep],
            input_batch.prefill_len_np[keep],
            input_batch.num_computed_tokens_np[keep],
        )

    # @spec REG-STATE-008, REG-STATE-014, REG-STATE-015, REG-STATE-020
    # @spec FILE-PROMPT-007, FILE-BUF-001
    def _screen_rows(
        self,
        input_batch: Any,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> _Screen | None:
        """Classify a real step's rows before any of them is read: each row's
        scheduled host tokens, the rows that fail at planning for a fault
        their request caused (a request not shaped as the model's processor
        builds it, which ``add_request`` recorded; a stream row's features or
        envelope; a file whose item ``add_request`` could not take; any row's
        endpoint policy),
        each clean stream CHUNK row's validated envelope header and the
        encoder inputs core's encoder may run (``_runnable_encoder_inputs``).
        ``None`` for a dummy step (a row the model state does not hold). A
        fault of core's contract raises.

        Core sends no token slices for cached requests unless pipeline
        parallelism is on, which this model refuses, so a row's host tokens
        are read from the prompt recorded at the request's latest
        ``add_request`` (REG-STATE-008). A position past that prompt holds a
        token the sampler produced on the device; the host never reads it,
        and no sampled token is a CHUNK carrier, so such a row is marked
        ``SAMPLED_TOKEN``."""
        req_ids = tuple(input_batch.req_ids)
        if any(r not in self._requests for r in req_ids):
            return None
        config = self.model.config
        placeholder = getattr(config, "audio_chunk_token_id", None)
        # The carrier row's width: a stream envelope fits it (REG-STATE-015).
        carrier_width = int(config.hidden_size)
        endpoint_controls = getattr(self.model, "_endpoint_controls", None)
        token_rows: list[tuple[int, ...]] = []
        failed: set[str] = set()
        headers: dict[str, tuple[float, ...]] = {}
        for index, req_id in enumerate(req_ids):
            record = self._requests[req_id]
            start = int(input_batch.num_computed_tokens_np[index])
            count = int(input_batch.num_scheduled_tokens[index])
            try:
                window = self._row_window(req_id, start, count)
            except RequestRowFault:
                window = ()
            if len(window) != 1:
                # A window crossing the end of the recorded prompt
                # (REG-STATE-008), or more than one token (REG-STATE-014):
                # the row's tokens are not classified.
                token_rows.append((SAMPLED_TOKEN,))
                failed.add(req_id)
                continue
            token = int(window[0])
            token_rows.append((token,))
            if record.fault is not None:
                # A request not shaped as the model's processor builds it, as
                # its latest add_request delivered it (REG-STATE-015): it
                # fails before any check of the row's window could raise.
                failed.add(req_id)
                continue
            try:
                if endpoint_controls is not None:
                    # Every request's endpoint policy is the client's, a
                    # file's too, though a file runs with endpointing
                    # disabled (REG-STATE-020, FILE-ENG-002).
                    endpoint_controls(record.metadata)
                if record.is_file:
                    # A file whose item add_request could not take.
                    if req_id in self._failed_files:
                        failed.add(req_id)
                    continue
                if placeholder is None:
                    continue
                header = resolve_row_envelope_header(
                    scheduled_token_id=token,
                    placeholder_id=int(placeholder),
                    num_computed_tokens=start,
                    mm_features=record.metadata.mm_features,
                    scheduled_encoder_input_ids=scheduled_encoder_inputs.get(
                        req_id, ()
                    ),
                    carrier_width=carrier_width,
                )
                if header is not None:
                    check_envelope_header(
                        header,
                        req_id,
                        num_prompts=int(config.num_prompts),
                        num_geometries=len(CADENCES),
                    )
                    headers[req_id] = header
            except RequestRowFault:
                failed.add(req_id)
        return _Screen(
            input_batch,
            tuple(token_rows),
            frozenset(failed),
            headers,
            self._runnable_encoder_inputs(
                scheduled_encoder_inputs, carrier_width=carrier_width
            ),
        )

    # @spec REG-STATE-015
    def _runnable_encoder_inputs(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        *,
        carrier_width: int,
    ) -> dict[str, list[int]]:
        """The scheduled encoder inputs core's encoder may run: each input
        whose own item the encoder path takes (``encoder_item_fault``).
        An input whose item it refuses (its modality, an item not shaped as
        the processor builds it, an envelope smaller than its header or wider
        than the carrier) is dropped; its row fails at planning, and a
        request whose feature has the same identifier has the same item and
        fails too. A row that fails for its request's prompt or its features'
        placement (``request_shape_fault``, ``feature_placement_fault``) keeps
        its valid inputs, which another request sharing their identifier
        gathers, in this step or a later one. An input that names no feature
        of its request is core's to refuse."""
        runnable: dict[str, list[int]] = {}
        for req_id, inputs in scheduled_encoder_inputs.items():
            record = self._requests.get(req_id)
            if record is None:
                runnable[req_id] = list(inputs)
                continue
            features = record.metadata.mm_features
            kept = [
                index
                for index in inputs
                if not 0 <= int(index) < len(features)
                or encoder_item_fault(features[int(index)], carrier_width=carrier_width)
                is None
            ]
            if kept:
                runnable[req_id] = kept
        return runnable

    # @spec REG-STATE-008
    def _row_window(self, req_id: str, start: int, count: int) -> tuple[int, ...]:
        """The host tokens of a row scheduling ``count`` tokens from position
        ``start``, from the request's recorded prompt: ``SAMPLED_TOKEN`` for
        each position past the prompt. A window crossing the end of the
        prompt is the request's fault (:class:`RequestRowFault`)."""
        prompt = self._requests[req_id].prompt
        window = prompt[start : start + count]
        if window and len(window) != count:
            raise RequestRowFault(
                f"request {req_id!r} schedules tokens across the end of its prompt"
            )
        return tuple(window) or (SAMPLED_TOKEN,) * count

    def _row_tokens(self, input_batch: Any) -> list[tuple[int, ...]]:
        """Each row's scheduled host tokens (``_row_window``)."""
        return [
            self._row_window(
                req_id,
                int(input_batch.num_computed_tokens_np[index]),
                int(input_batch.num_scheduled_tokens[index]),
            )
            for index, req_id in enumerate(input_batch.req_ids)
        ]

    # @spec REG-MEM-001, REG-MEM-012, REG-STATE-002, REG-STATE-003
    # @spec REG-STATE-007, REG-STATE-010, REG-STATE-014
    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        self._decide_pending()
        attn_prepared, self._attn_prepared = self._attn_prepared, False
        added, self._step_added = self._step_added, []
        screen, self._screen = self._screen, None
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
            if screen is None or screen.input_batch is not input_batch:
                screen = self._screen_rows(input_batch, self._scheduled_encoder_inputs)
            assert screen is not None
            layer = self._state_layer()
            bindings = tuple(
                RowBinding(
                    req_id,
                    self._requests[req_id].handle,
                    int(input_batch.num_computed_tokens_np[index]) == 0,
                )
                for index, req_id in enumerate(req_ids)
            )
            # Refuses a slot outside the bound pages before any page is read.
            batch = StateBatch(
                layer,
                bindings,
                self.model._state_spec,
                int(self.model.config.n_layers),
            )
            token_rows = list(screen.token_rows)
            self._scheduled_tokens = dict(zip(req_ids, token_rows, strict=True))
            # A file row that failed at planning is planned as a failed file.
            self._failed_files.update(
                r for r in screen.failed if self._requests[r].is_file
            )
            # The step's new requests plan first, as core ordered them.
            step_added = set(added)
            plan_order = tuple(r for r in added if r in set(req_ids)) + tuple(
                r for r in req_ids if r not in step_added
            )
            step_plan = self._plan_production_step(
                req_ids,
                bindings,
                batch,
                token_rows,
                input_batch,
                plan_order,
                headers=screen.headers,
                failed=screen.failed,
            )
            # TXG-FAIL-008: the rows the model state failed at planning.
            planning_failed = set(screen.failed) | {
                row.request_id for row in step_plan.rows if row.failed
            }
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
                stream_headers=dict(screen.headers),
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
        *,
        headers: dict[str, tuple[float, ...]],
        failed: frozenset[str] = frozenset(),
    ) -> StepPlan:
        """Make the step's plan once: order the stream rows by the geometry
        and deadline of their screened headers (``headers``, from
        ``_screen_rows``), then plan the rows in ``plan_order`` (default: the
        batch order). FILE rows plan from their file's queue count; no label
        book is read (FILE-QC-011). A row in ``failed`` takes no part in
        ordering."""
        del bindings, batch, input_batch
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
            if (
                req_id in failed
                or req_id in self._file_progress
                or token != int(config.audio_chunk_token_id)
            ):
                continue
            header = headers.get(req_id)
            if header is None:
                continue
            stream_rows[req_id] = stream_order_key(header, now_ns=now_ns)
        return self.plan_step(plan_order, stream_rows=stream_rows)

    # @spec SESS-MODEL-003, TXG-FAIL-008, FILE-QC-006, FILE-QC-012
    # @spec REG-MS-003, REG-MS-006, REG-ENG-020
    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        snapshot, self._snapshot = self._snapshot, None
        if snapshot is None or snapshot.dummy_run:
            return
        # The collection waits for the step's commit event before it reads
        # the step's status. Every collected failure (any nonzero status,
        # SESS-MODEL-003), every planning-failed row (TXG-FAIL-008) and every
        # file row with no accepted count (FILE-QC-012) fails its request
        # alone; nothing is recorded for a step whose postprocessing raises.
        collection = self.model.collect_commit_status()
        failed = set(collection.failed) | set(snapshot.planning_failed)
        failed |= self._apply_queue_counts(snapshot, collection)
        self.model.consume_batch_stats()
        self._failed_requests |= failed

    # @spec REG-MS-004
    def take_failed_requests(self) -> set[str] | None:
        failed, self._failed_requests = self._failed_requests, set()
        return failed or None

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

    # @spec REG-ENG-029
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        compilation = getattr(self.vllm_config, "compilation_config", None)
        raise RuntimeError(
            f"core asked for runner-graph capture inputs (cudagraph_mode "
            f"{_mode_name(getattr(compilation, 'cudagraph_mode', None))}), "
            f"but {NO_RUNNER_GRAPH}"
        )

    # @spec REG-ENG-029
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
        if for_capture or cudagraph_mode == CUDAGraphMode.FULL:
            raise RuntimeError(
                f"core prepared a {'capture' if for_capture else 'step'} under "
                f"cudagraph_mode {_mode_name(cudagraph_mode)}, but {NO_RUNNER_GRAPH}"
            )
        if ubatch_idx:
            raise RuntimeError(
                "the Nemotron streaming model does not support micro-batches "
                "(dual batch overlap)"
            )
        self._decide_pending()
        # The model builds no Mamba metadata: it plans its rows on the host.
        self._attn_prepared = True
        return {}
