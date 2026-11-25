# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron cache-aware streaming RNN-T ASR — model class.

Cross-chunk state lives in one engine-managed aggregate page per request.
The MRv2 ModelState projects bounded core slot views into each transaction.
The full-context path survives only as an EVAL-tier parity probe.

RNN-T emission is D-b (PORT-DEC-001/002/003): the forward that ingests
audio runs featurizer -> encoder -> LID -> the complete greedy
label-looping decode in-model, parks the emitted labels in a per-request
replay queue, and every generation step emits the next queued label via
a forced-logits row from ``compute_logits`` (0 at the chosen id, -inf
elsewhere); the park token — the checkpoint's ``eos_token_id`` — ends
the burst through the resumable stop path. Greedy sampling is enforced
by the replay-echo guard (PORT-DEC-005/007).
"""

import dataclasses
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm_nemotron_asr_streaming import engine_flags
from vllm_nemotron_asr_streaming.engine_flags import graphed_execution_configured
from vllm.model_executor.models.interfaces import (
    SupportsPersistentState,
    SupportsTranscription,
)
from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
    DecodeRequest,
    DecodeResolver,
    EmissionAdapter,
    HostStaging,
    ResolvedDecode,
    make_mrv1_adapter,
)
from vllm_nemotron_asr_streaming.nemotron_asr.batch_invariance import (
    BatchInvariantExecution,
    bind_batch_invariant_mode,
    execution_mode,
    installed_batch_invariant_mode,
)
from vllm_nemotron_asr_streaming.nemotron_asr.commit_sink import (
    BoundedCommitSink,
    CommitCollection,
    QueueCount,
    resolve_status_reports,
)
from vllm_nemotron_asr_streaming.nemotron_asr.convert import (
    LID_REQUIRED_PATTERN,
)
from vllm_nemotron_asr_streaming.nemotron_asr.decode_dispatch import (
    SYNC_FREE_ARMS,
)
from vllm_nemotron_asr_streaming.nemotron_asr.encoder import (
    FastConformerEncoder,
)
from vllm_nemotron_asr_streaming.nemotron_asr.encoder_execution import (
    build_encoder_execution,
)
from vllm_nemotron_asr_streaming.nemotron_asr.featurizer import (
    MelFeaturizer,
)
from vllm_nemotron_asr_streaming.nemotron_asr.lid import (
    PromptConditioner,
    resolve_prompt_index,
)
from vllm_nemotron_asr_streaming.locales import (
    supported_languages as _supported_languages,
)
from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
    CADENCES,
    REALTIME_MAX_TOKENS,
)
from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
    ObservedRow,
    PlanContextSlot,
    RequestRowFault,
    SessionRegistry,
    envelope_width_fault,
    prepare_plan_context,
    resolve_row_envelope_header,
)
from vllm_nemotron_asr_streaming.nemotron_asr.precision import (
    FP32_BRINGUP,
    PrecisionPolicy,
    resolve_precision_policy,
)
from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
    NemotronASRDummyInputsBuilder,
    NemotronASRMultiModalProcessor,
    NemotronASRProcessingInfo,
)
from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
    DecodeState,
    Joint,
    Predictor,
    decode_dense_masked_frames,
    greedy_decode_chunk,
)
from vllm_nemotron_asr_streaming.nemotron_asr import graph_memory
from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import NemotronStateLayer
from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
    build_nemotron_state_layout,
)
from vllm_nemotron_asr_streaming.transaction_graph_config import (
    KEY as TRANSACTION_GRAPH_KEY,
    transaction_graphs_on,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

logger = init_logger(__name__)

#: The sampling ``extra_args`` key that carries a request's endpoint policy
#: (REG-STATE-020), and the policy's modes by name, as the row plan's
#: ``endpoint_mode`` column holds them.
ENDPOINT_POLICY_KEY = "endpoint_policy"
ENDPOINT_MODES = {"disabled": 0, "greedy_blank": 1}
#: The largest count the row plan's ``int64`` endpoint columns hold.
_MAX_INT64 = torch.iinfo(torch.int64).max

#: The captures that create segment graphs (REG-MEM-015): the first profiling
#: dummy step under ``--enforce-eager``, and core's capture hook's profiling
#: and serving calls under the hook configuration.
PROFILING_STEP = "profiling dummy step"
PROFILING_CAPTURE = "profiling capture"
SERVING_CAPTURE = "serving capture"


def _synchronize_quietly(device: torch.device | None, stream: Any) -> None:
    """Synchronize the capture stream (or a CUDA device) on a failure path. A
    failure here is logged, never raised over the original error."""
    try:
        if stream is not None:
            stream.synchronize()
        elif device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        logger.exception(
            "synchronizing after a failed Nemotron preparation or capture failed"
        )

#: The largest published chunk (1120 ms) emits 14 encoder frames — the
#: replay queue page's per-chunk worst case.
_MAX_FRAMES_PER_CHUNK = 14
#: The checkpoint's mel-bin count; the featurizer's filterbank is built
#: as a placeholder at this width and overwritten by load_weights.
_N_MELS = 128
_STFT_FREQ_BINS = 512 // 2 + 1
_WIN_LENGTH = 400


# @spec REG-LA-009, SESS-GEOM-001
def _server_geometry(hf_config: Any) -> int:
    """The server's one geometry index: ``geometry_index`` of
    ``served_lookahead``, the one definition the factory, the realtime hook,
    the file prompt and the model share."""
    from vllm_nemotron_asr_streaming.lookahead import (
        geometry_index,
        served_lookahead,
    )

    return geometry_index(served_lookahead(hf_config))


class NemotronASRCore(nn.Module):
    """The assembled cache-aware streaming pipeline.

    The realtime engine-facing class wraps this core; probes drive it
    directly. Weight names follow the converted tree (rules.py):
    ``featurizer.* / encoder.* / lid.* / predictor.* / joint.*``.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        att_context: tuple[int, int] = (56, 13),
        enc_hidden: int = 1024,
        n_layers: int = 24,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        joint_hidden: int = 640,
        num_prompts: int = 128,
        filterbank: torch.Tensor,
        window: torch.Tensor,
        policy: PrecisionPolicy = FP32_BRINGUP,
    ) -> None:
        super().__init__()
        # @spec PORT-PREC-018, PORT-PREC-028: resolve before building components.
        bind_batch_invariant_mode(self, installed_batch_invariant_mode())
        self.policy = policy
        self.vocab_size = vocab_size
        self.blank_id = vocab_size
        self.featurizer = MelFeaturizer(filterbank=filterbank, window=window)
        # n_layers is a checkpoint property, not a fixed default: the
        # served path passes hf_config.n_layers and the dump loader
        # derives it from the converted tensors, so the encoder depth
        # always matches the weights (and the published state manifest).
        self.encoder = FastConformerEncoder(n_layers=n_layers, att_context=att_context)
        self.lid = PromptConditioner(enc_hidden=enc_hidden, num_prompts=num_prompts)
        self.predictor = Predictor(
            vocab_size=vocab_size,
            pred_hidden=pred_hidden,
            pred_rnn_layers=pred_rnn_layers,
        )
        self.joint = Joint(
            enc_hidden=enc_hidden,
            pred_hidden=pred_hidden,
            joint_hidden=joint_hidden,
            vocab_size=vocab_size,
        )
        bind_batch_invariant_mode(self, self.batch_invariant_mode)
        self._pred_layers = pred_rnn_layers
        self._pred_hidden = pred_hidden

    @property
    def batch_invariant_mode(self) -> BatchInvariantExecution:
        return self._batch_invariant_mode

    def fresh_decode_state(self, device: torch.device) -> DecodeState:
        """Zeroed decode state; last label = blank (SOS, PORT-STATE-003)."""
        return DecodeState(
            h=torch.zeros(
                self._pred_layers,
                1,
                self._pred_hidden,
                device=device,
                dtype=self.policy.dtype_for("lstm_state"),
            ),
            c=torch.zeros(
                self._pred_layers,
                1,
                self._pred_hidden,
                device=device,
                dtype=self.policy.dtype_for("lstm_state"),
            ),
            last_label=torch.tensor([self.blank_id], device=device),
        )

    @torch.inference_mode()
    def transcribe_chunk(
        self,
        chunk_conditioned: torch.Tensor,
        state: DecodeState,
    ) -> tuple[list[int], DecodeState]:
        """One streaming chunk-step's decode (D-b queue fill)."""
        return greedy_decode_chunk(chunk_conditioned, self.predictor, self.joint, state)


def apply_policy_dtypes(core: NemotronASRCore) -> NemotronASRCore:
    """Cast compute modules to the policy's weight dtype (PORT-PREC-001).

    @spec PORT-PREC-010: realize encoder precision independently of frontend,
    conditioner, predictor, joint and persistent state.

    LID, predictor, and joint move to ``dtype_for("weights")``; the
    encoder uses the separately declared ``encoder_compute`` override.
    Activations follow at the module entry seams, so this realization
    requires the two classes to agree. The mel front-end stays fp32 —
    it sits upstream of the activations seam and is the golden input
    boundary. Recurrent/cache state dtypes are independent axes read at
    their own construction sites (PORT-PREC-005), and the manual LSTM
    cell up-casts its weights to the state dtype per step, keeping
    ``(h, c)`` accumulation at fp32 under sub-fp32 weights.
    """
    weights = core.policy.dtype_for("weights")
    activations = core.policy.dtype_for("activations")
    if weights != activations:
        raise ValueError(
            "this realization derives activations from the weight "
            f"dtype at the entry seams; policy {core.policy.identifier} "
            f"declares weights={weights} activations={activations}"
        )
    core.encoder.to(core.policy.dtype_for("encoder_compute"))
    for module in (core.lid, core.predictor, core.joint):
        module.to(weights)
    return core


def load_core_from_dump(
    dump_dir: str | Path,
    *,
    device: torch.device,
    att_context: tuple[int, int] = (56, 13),
    policy: PrecisionPolicy = FP32_BRINGUP,
) -> tuple[NemotronASRCore, dict]:
    """Assemble the core from the offline conversion dump.

    The engine path loads the same converted tree through the standard
    HF loader once the checkpoint is published in HF layout
    (PORT-WGT-001); this loader serves probes and the serving prototype
    from the referee dump meanwhile.
    """
    from safetensors.torch import load_file

    from vllm_nemotron_asr_streaming.nemotron_asr.convert import (
        convert_state_dict,
        derive_n_layers,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.rules import NEMO_RULES

    dump = Path(dump_dir)
    state = load_file(str(dump / "nemo_state.safetensors"))
    meta = json.loads((dump / "meta.json").read_text())
    converted, _ = convert_state_dict(state, NEMO_RULES)
    core = NemotronASRCore(
        vocab_size=int(meta["vocab_size"]),
        att_context=att_context,
        # Derive the encoder depth from the converted tensors (the same
        # source publish.py uses), never the module default — this
        # loader must build an encoder that matches the dump's weights.
        n_layers=derive_n_layers(converted),
        filterbank=converted["featurizer.fb"][0],
        window=converted["featurizer.window"],
        policy=policy,
    )
    # fb/window enter via the constructor (checkpoint-buffer rule);
    # exclude them here — NeMo's fb carries a leading batch dim.
    loadable = {
        k: v
        for k, v in converted.items()
        if k not in ("featurizer.fb", "featurizer.window")
    }
    missing, unexpected = core.load_state_dict(loadable, strict=False)
    real_missing = [m for m in missing if not m.endswith(("fb", "window", "pe"))]
    if real_missing or unexpected:
        raise ValueError(
            f"core load mismatch: missing={real_missing} unexpected={unexpected}"
        )
    apply_policy_dtypes(core)
    core.to(device).eval()
    return core, meta


__all__ = [
    "NemotronASRCore",
    "NemotronASRForRNNT",
    "apply_policy_dtypes",
    "load_core_from_dump",
    "resolve_prompt_index",
]


@MULTIMODAL_REGISTRY.register_processor(
    NemotronASRMultiModalProcessor,
    info=NemotronASRProcessingInfo,
    dummy_inputs=NemotronASRDummyInputsBuilder,
)
# @spec REG-TASK-001, FILE-IF-001
class NemotronASRForRNNT(nn.Module, SupportsTranscription, SupportsPersistentState):
    """Engine-facing cache-aware RNN-T over one aggregate state page."""

    supports_multimodal = True
    requires_raw_input_tokens = True
    supports_realtime = True
    supports_transcription_only = False
    supports_segment_timestamp = False
    supports_diarized_transcription = False
    supports_persistent_state = True
    # @spec REG-TASK-003: the forward reads no value of ``positions``.
    persistent_state_reads_positions = False
    # @spec REG-TASK-004
    persistent_state_accepts_synthetic_warmup = False
    # @spec REG-TASK-005: a stop-free EOS (the park) is the only resumable
    # stop.
    persistent_state_resumable_stops = frozenset({"eos"})
    is_attention_free = True
    # @spec FILE-IF-002
    supported_languages = _supported_languages()
    # @spec REG-CARD-016
    realtime_max_tokens = REALTIME_MAX_TOKENS
    realtime_input_tokens = 1
    # The logit width is the served configuration's ``vocab_size``, set in
    # ``__init__`` (REG-CARD-015).
    num_logits: int

    def __init__(self, *, vllm_config: Any = None, prefix: str = "") -> None:
        """@spec PORT-PREC-009, PORT-PREC-010, PORT-PREC-013: bind startup policy."""
        super().__init__()
        if vllm_config is None:
            raise ValueError("NemotronASRForRNNT requires vllm_config")
        max_model_len = getattr(vllm_config.model_config, "max_model_len", None)
        min_history = self.realtime_input_tokens + self.realtime_max_tokens
        if max_model_len is not None and max_model_len < min_history:
            raise ValueError(
                "max_model_len must contain one streaming transaction "
                f"({min_history} tokens)"
            )
        hf_config = vllm_config.model_config.hf_config
        # @spec REG-ENG-006, REG-ENG-027: the final engine configuration may
        # differ from what the configuration hook saw; check it again, the
        # hook configuration's conditions through
        # ``engine_flags.runner_mode_refusals`` (stage ``"constructor"``),
        # before any arm is built or weights load.
        engine_flags.check_engine_config(
            vllm_config, "the Nemotron streaming model", resolved=True
        )
        # @spec REG-014, REG-018: the graph-runtime dependencies are needed
        # only when graphed execution is configured, the transaction-graph
        # members only while transaction graphs are on; name every missing one
        # now, before any arm is built or weights load.
        engine_flags.check_conditional_dependencies(hf_config)
        # Resolve the effective (stage-overridden) selector once, before any
        # state reservation, execution construction or checkpoint loading.
        policy = resolve_precision_policy(hf_config)
        from vllm_nemotron_asr_streaming.nemotron_asr.configuration_nemotron_asr import (
            ensure_prompt_dictionary,
            validate_prompt_dictionary,
        )

        model_path = getattr(vllm_config.model_config, "model", None)
        if model_path:
            ensure_prompt_dictionary(hf_config, model_path)
        validate_prompt_dictionary(
            getattr(hf_config, "prompt_dictionary", None),
            getattr(hf_config, "num_prompts", None),
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
            MAX_SYMBOLS_PER_STEP,
        )

        # PORT-DEC-005: the served declaration is the authority; the
        # manifest-derived emission budgets were computed from the
        # module value, so a disagreeing declaration fails closed
        # instead of being silently overridden.
        declared_symbols = getattr(hf_config, "max_symbols_per_step", None)
        if (
            declared_symbols is not None
            and int(declared_symbols) != MAX_SYMBOLS_PER_STEP
        ):
            raise ValueError(
                "served config declares max_symbols_per_step="
                f"{declared_symbols}, but the manifest-derived emission "
                f"budgets were computed from {MAX_SYMBOLS_PER_STEP}; "
                "requalify the budgets before serving this declaration "
                "(PORT-DEC-005)"
            )
        # PORT-STATE-009: precision is part of the qualified profile.
        # An unqualified --dtype fails loudly, never warn-and-ignore.
        engine_dtype = getattr(vllm_config.model_config, "dtype", None)
        if engine_dtype is not None and engine_dtype != torch.float32:
            raise ValueError(
                f"--dtype {engine_dtype} is not the qualified profile "
                f"for PrecisionPolicy {policy.identifier} (float32); a precision "
                "change requires "
                "requalification (PORT-STATE-009). If no --dtype was "
                "given, vLLM's auto policy downcasts float32 checkpoints "
                "on SM80+ GPUs. Pass --dtype float32."
            )
        # Pool sizing is resolved after vLLM's real memory profile in the
        # common worker. The model constructor deliberately does not invent
        # physical capacity or mutate ``num_gpu_blocks_override``.
        self.config = hf_config
        self.num_logits = int(hf_config.vocab_size)
        self.core = NemotronASRCore(
            vocab_size=hf_config.num_asr_labels,
            att_context=(
                hf_config.att_context_left,
                hf_config.att_context_right,
            ),
            enc_hidden=hf_config.d_model,
            n_layers=hf_config.n_layers,
            pred_hidden=hf_config.pred_hidden,
            pred_rnn_layers=hf_config.pred_rnn_layers,
            joint_hidden=hf_config.joint_hidden,
            num_prompts=hf_config.num_prompts,
            filterbank=torch.zeros(_N_MELS, _STFT_FREQ_BINS),
            window=torch.zeros(_WIN_LENGTH),
            policy=policy,
        )
        if policy is not FP32_BRINGUP:
            apply_policy_dtypes(self.core)
        self._state_spec = build_nemotron_state_layout(hf_config)
        self._persistent_state_layer = NemotronStateLayer(
            self._state_spec,
            prefix=f"{prefix}.state" if prefix else "state",
            vllm_config=vllm_config,
        )
        self._registry = SessionRegistry()
        self._plan_slot = PlanContextSlot()
        self._plan_step = 0
        self._emission_adapter: EmissionAdapter = make_mrv1_adapter(
            hidden_size=hf_config.hidden_size,
            park_id=hf_config.eos_token_id,
            blank_id=self.core.blank_id,
            continuation_id=int(hf_config.continuation_token_id),
        )
        native_burst_enabled = getattr(hf_config, "experimental_native_burst", False)
        if not isinstance(native_burst_enabled, bool):
            raise ValueError("experimental_native_burst must be a boolean")
        self._native_burst_handoff = None
        if native_burst_enabled:
            from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
                NativeBurstHandoff,
                native_burst_token_budget,
                validate_native_burst_config,
            )

            validate_native_burst_config(
                vllm_config,
                hf_config=hf_config,
                require_160ms_profile=True,
            )
            self._native_burst_handoff = NativeBurstHandoff(
                max_tokens=native_burst_token_budget(hf_config)
            )
        self._max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        server_geometry = _server_geometry(hf_config)
        self._server_geometry = server_geometry
        self._graphed_execution = graphed_execution_configured(hf_config)
        self._encoder_execution = build_encoder_execution(
            self.core,
            hf_config,
            maximum_population=self._max_num_seqs,
            warmup_geometries=(server_geometry,),
            vllm_config=vllm_config,
        )
        self._decode_graph_binding = None
        if getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed":
            from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import (
                DenseGraphBinding,
                execution_tiers,
            )

            self._decode_graph_binding = DenseGraphBinding(
                batch_invariant_mode=self.core.batch_invariant_mode,
                decode_fn=decode_dense_masked_frames,
                predictor=self.core.predictor,
                joint=self.core.joint,
                vllm_config=vllm_config,
                frame_widths=tuple(
                    right + 1 if index == server_geometry else None
                    for index, (_left, right) in enumerate(CADENCES.values())
                ),
                tiers=execution_tiers(self._max_num_seqs),
                encoder_hidden=int(hf_config.d_model),
                predictor_layers=int(hf_config.pred_rnn_layers),
                predictor_hidden=int(hf_config.pred_hidden),
                blank_id=int(self.core.blank_id),
            )
        self._decode_resolver = build_decode_resolver(
            hf_config,
            graph_binding=self._decode_graph_binding,
        )
        self._chunk_bucket_binding = None
        chunk_populations = getattr(hf_config, "chunk_bucket_graph_populations", ())
        if not isinstance(chunk_populations, (tuple, list)):
            raise ValueError("chunk_bucket_graph_populations must be a sequence")
        if chunk_populations:
            from vllm_nemotron_asr_streaming.nemotron_asr.chunk_bucket_graph import (
                ExactChunkGraphBinding,
            )
            from vllm_nemotron_asr_streaming.nemotron_asr.native_burst import (
                validate_native_burst_config,
            )

            # Reuse the existing initial-profile configuration guard only.
            # This does not enable native burst; shared decoder scratch requires
            # synchronous, non-speculative single-device worker execution.
            if native_burst_enabled:
                raise ValueError("CHUNK serving pilot requires native burst disabled")
            validate_native_burst_config(vllm_config, hf_config=hf_config)
            self._chunk_bucket_binding = ExactChunkGraphBinding(
                self.core,
                hf_config,
                self._encoder_execution,
                self._decode_graph_binding,
                chunk_populations,
            )
        self._commit_sink: BoundedCommitSink | None = None
        self._host_staging: HostStaging | None = None
        self._vllm_config = vllm_config
        #: The first profiling dummy step's preparation succeeded
        #: (REG-MEM-002, REG-MEM-016) or failed (REG-MEM-004) in this worker.
        self._execution_memory_prepared = False
        self._execution_memory_failed = False
        #: Core's ``prepare_execution_memory`` ran (REG-MEM-018).
        self._working_memory_prepared = False
        #: Set when core's model-state capture hook, not the profiling step,
        #: captures the segment graphs (the hook configuration), by
        #: ``prepare_execution_memory_hook``.
        self._graphs_by_core = False
        #: The capture that created the live segment graphs: ``None``,
        #: ``PROFILING_STEP``, ``PROFILING_CAPTURE`` or ``SERVING_CAPTURE``.
        self._segment_graphs: str | None = None
        self._serving_capture_attempted = False
        #: Whether the eager stand-in line (REG-MEM-028) was logged.
        self._graph_stand_in_logged = False
        #: The profiling capture call's figures, host integers kept after
        #: ``clear_cudagraphs`` (REG-MEM-021, REG-MEM-022); ``None`` until a
        #: profiling capture ran in this worker.
        self._profiling_figures: dict[str, Any] | None = None
        #: The profiling capture's charged margin while transaction graphs
        #: are off (REG-MEM-020), held until ``clear_cudagraphs``.
        self._charged_margin: Any = None
        #: The transaction-graph binding, built while transaction graphs are
        #: on (TXG-CFG-001); read through ``_transaction_graph_binding``,
        #: which is ``None`` while they are off.
        self._txg_binding: Any = None
        if transaction_graphs_on(hf_config):
            self._txg_binding = self._build_transaction_graph_binding(vllm_config)
        if self._txg_binding is not None:
            # TXG-LIFE-009: the state layer reports each binding of its views.
            self._persistent_state_layer.bind_observer = self._state_layer_bound

    def _build_transaction_graph_binding(self, vllm_config: Any) -> Any:
        """The binding for an accepted option value. A configuration the
        option's checks refuse (``transaction_graph_config_errors`` reports a
        condition) yields none: the model state's construction refuses that
        configuration (TXG-CFG-011), and no ``N`` is computed from it
        (TXG-CFG-004). Any other failure to build the binding propagates, so
        an accepted option value never serves without its binding."""
        from vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph import (
            TransactionGraphBinding,
        )
        from vllm_nemotron_asr_streaming.transaction_graph_config import (
            transaction_graph_config_errors,
        )

        if transaction_graph_config_errors(
            vllm_config,
            has_state_group_layouts=engine_flags.model_state_has_state_group_layouts(),
        ):
            return None
        return TransactionGraphBinding(self, vllm_config)

    # @spec TXG-CFG-014, TXG-LIFE-011
    @property
    def _transaction_graph_binding(self) -> Any:
        """The transaction-graph binding while transaction graphs are on,
        else ``None``: every transaction-graph part of the hooks and of a
        step goes through this one read, which asks the one on predicate of
        the served configuration."""
        state = self.__dict__
        binding = state.get("_txg_binding")
        config = state.get("config")
        if binding is None or config is None or not transaction_graphs_on(config):
            return None
        return binding

    # @spec TXG-LIFE-009, TXG-LIFE-011
    def _state_layer_bound(self, layer: Any) -> None:
        """The state layer's report of views core just bound
        (``bind_kv_cache``): the transaction-graph binding compares them with
        the layout the serving capture read, once one was attempted."""
        binding = self._transaction_graph_binding
        if binding is not None:
            binding.check_bindings(layer)

    def _state_pools(self, projection: Any) -> Any:
        return projection.state_batch.pools

    def load_weights(self, weights: Any) -> set[str]:
        """Strictly load the served tree or the public HF card.

        @spec PORT-PREC-017: load the unchanged checkpoint into policy-typed
        targets. Each copy converts its source once without modifying it.

        The served artifact's names load as-is. The public card's
        renamed modules map through ``remap_card_name``; shared leaves
        already match. A card checkpoint persists no featurizer
        buffers, so those two are synthesized after the stream — only
        when card naming was actually seen, never for a served artifact
        (persisted buffers are inputs, not recomputed).
        """
        from vllm_nemotron_asr_streaming.nemotron_asr.rules import (
            remap_card_name,
        )

        if self._encoder_execution._model_state is not None:
            raise RuntimeError(
                "encoder weights are immutable after execution materialization; use a "
                "fresh worker"
            )
        self.core.encoder.invalidate_stream_relative_position_projections()
        expected: dict[str, torch.Tensor] = dict(self.core.named_parameters())
        expected.update(self.core.named_buffers())
        required = set(self.core.state_dict())
        consumed: set[str] = set()
        card_names_seen = False
        for name, tensor in weights:
            mapped = remap_card_name(name)
            if mapped is not None:
                card_names_seen = True
                name = mapped
            if name not in expected:
                raise ValueError(f"unexpected weight {name!r}")
            target = expected[name]
            tensor = self._normalize_checkpoint_tensor(name, tensor)
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name!r}: expected {tuple(target.shape)}, got "
                    f"{tuple(tensor.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor)
            consumed.add(name)
        missing = required - consumed
        featurizer_buffers = {"featurizer.fb", "featurizer.window"}
        if card_names_seen and featurizer_buffers & missing:
            from vllm_nemotron_asr_streaming.nemotron_asr.featurizer import (
                synthesize_card_featurizer_buffers,
            )

            fb, window = synthesize_card_featurizer_buffers(
                n_mels=int(self.config.n_mels),
            )
            for name, tensor in (("featurizer.fb", fb), ("featurizer.window", window)):
                if name not in missing:
                    continue
                target = expected[name]
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"synthesized {name!r} shape {tuple(tensor.shape)} "
                        f"does not match the module ({tuple(target.shape)})"
                    )
                with torch.no_grad():
                    target.copy_(tensor)
                consumed.add(name)
            missing = required - consumed
        lid_missing = sorted(
            name for name in missing if LID_REQUIRED_PATTERN.search(name)
        )
        if lid_missing:
            raise ValueError(
                f"LID weights absent ({lid_missing}); conditioned transcription never "
                "degrades silently"
            )
        if missing:
            raise ValueError(
                f"{len(missing)} expected weights not provided, e.g. "
                f"{sorted(missing)[:3]}"
            )
        return {f"core.{name}" for name in consumed}

    @staticmethod
    def _normalize_checkpoint_tensor(
        name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the publisher's one explicit storage-to-runtime reshape."""
        if name == "featurizer.fb" and tensor.ndim == 3 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    # @spec SESS-RT-001, REG-LA-009: the packer packs at the server's
    # geometry, from ``served_lookahead``.
    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: Any,
        input_stream: Any,
        model_config: Any,
        *,
        observer: Any = None,
        accepted_audio_budget_s: float | None = None,
        session_key: str | None = None,
    ) -> Any:
        """Pack core's realtime audio with the session packer (SESS-RT-001
        to SESS-RT-003)."""
        from vllm_nemotron_asr_streaming.session.packer import pack_realtime

        async for prompt in pack_realtime(
            audio_stream, input_stream, model_config.hf_config
        ):
            yield prompt

    # @spec FILE-IF-003
    @classmethod
    def get_speech_to_text_config(cls, model_config: Any, task_type: Any) -> Any:
        """Core never splits a file: no clip limit, no energy split window."""
        from vllm.config.speech_to_text import SpeechToTextConfig

        del model_config, task_type
        return SpeechToTextConfig(
            sample_rate=16000,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    # @spec FILE-IF-004, FILE-IF-005, FILE-PROMPT-001
    @classmethod
    def get_generation_prompt(cls, stt_params: Any) -> Any:
        from vllm_nemotron_asr_streaming.transcription import build_file_prompt

        return build_file_prompt(stt_params)

    def prepare_row_plan_context(
        self,
        *,
        req_ids: Any,
        token_ids_cpu: Any,
        num_computed_tokens_cpu: Any,
        num_scheduled_tokens: Any,
        requests: Any,
        scheduled_encoder_inputs: Any,
    ) -> None:
        """MRV1 oracle projection into the common transaction plan."""
        rows: list[ObservedRow] = []
        for index, request_id in enumerate(req_ids):
            if int(num_scheduled_tokens[index]) != 1:
                raise ValueError("every streaming row must be single-token")
            computed = int(num_computed_tokens_cpu[index])
            token = int(token_ids_cpu[index, computed])
            request = requests[request_id]
            groups = request.block_ids
            if len(groups) != 1 or len(groups[0]) != 1:
                raise ValueError("persistent state requires one slot per request")
            endpoint_mode, endpoint_threshold, endpoint_residue = (
                self._endpoint_controls(request)
            )
            rows.append(
                ObservedRow(
                    request_id=str(request_id),
                    block_id=int(groups[0][0]),
                    scheduled_token_id=token,
                    has_prior_state=computed > 0,
                    envelope_header=resolve_row_envelope_header(
                        scheduled_token_id=token,
                        placeholder_id=int(self.config.audio_chunk_token_id),
                        num_computed_tokens=computed,
                        mm_features=request.mm_features,
                        scheduled_encoder_input_ids=scheduled_encoder_inputs.get(
                            request_id, ()
                        ),
                        carrier_width=int(self.config.hidden_size),
                    ),
                    endpoint_mode=endpoint_mode,
                    endpoint_threshold_frames=endpoint_threshold,
                    endpoint_residue_frames=endpoint_residue,
                )
            )
        self._stage_plan(
            rows, resident_request_ids=tuple(str(value) for value in requests)
        )

    def _stage_plan(
        self,
        rows: list[ObservedRow],
        *,
        resident_request_ids: tuple[str, ...],
    ) -> Any:
        import time

        if self._commit_sink is not None and self._commit_sink.has_staged:
            raise RuntimeError("previous transaction status was not consumed")
        self._plan_step += 1
        context = prepare_plan_context(
            self._registry,
            rows,
            resident_request_ids=resident_request_ids,
            placeholder_id=int(self.config.audio_chunk_token_id),
            num_prompts=int(self.core.lid.num_prompts),
            num_geometries=len(CADENCES),
            now_ns=time.time_ns(),
            step=self._plan_step,
            server_geometry=_server_geometry(self.config),
            eou_token_id=(
                None
                if getattr(self.config, "eou_token_id", None) is None
                else int(self.config.eou_token_id)
            ),
        )
        self._plan_slot.stage(context)
        return context

    # @spec REG-STATE-020
    @staticmethod
    def _endpoint_controls(request: Any) -> tuple[int, int, int]:
        """The request's endpoint policy as ``(mode, threshold_frames,
        residue_frames)``, from its sampling parameters'
        ``extra_args["endpoint_policy"]`` (core delivers the sampling
        parameters with the request): ``(0, 0, 0)``, endpointing disabled,
        without ``extra_args`` or without the key. A count the policy omits
        is 0. A malformed policy (not a mapping, an unknown mode, or a count
        that is not a non-negative integer the plan's ``int64`` columns hold)
        is the request's own fault (:class:`RequestRowFault`, REG-STATE-015).
        """
        params = getattr(request, "sampling_params", None)
        extra_args = getattr(params, "extra_args", None)
        if extra_args is None:
            return 0, 0, 0
        if not isinstance(extra_args, Mapping):
            raise RequestRowFault("request's sampling extra_args is not a mapping")
        if ENDPOINT_POLICY_KEY not in extra_args:
            return 0, 0, 0
        policy = extra_args[ENDPOINT_POLICY_KEY]
        if not isinstance(policy, Mapping):
            raise RequestRowFault("request's endpoint policy is not a mapping")
        mode_name = policy.get("mode")
        if not isinstance(mode_name, str) or mode_name not in ENDPOINT_MODES:
            raise RequestRowFault(
                f"request carries an unknown endpoint mode {mode_name!r}"
            )
        counts = []
        for name in ("threshold_frames", "residue_frames"):
            value = policy.get(name, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _MAX_INT64
            ):
                raise RequestRowFault(
                    f"request's endpoint policy {name} {value!r} is not a "
                    "non-negative integer"
                )
            counts.append(value)
        return ENDPOINT_MODES[mode_name], counts[0], counts[1]

    # @spec REG-STATE-014, REG-STATE-015, TXG-COV-010, FILE-ENG-002
    def _stage_v2_projection(self, projection: Any) -> Any:
        input_batch = projection.input_batch
        metadata = projection.request_metadata
        if metadata is None:
            raise RuntimeError("MRv2 projection omitted request metadata")
        # The rows the model state failed at planning, and each clean stream
        # CHUNK row's screened header.
        planning_failed = getattr(projection, "planning_failed", None) or ()
        stream_headers = projection.stream_headers
        rows: list[ObservedRow] = []
        for index, (request_id, binding) in enumerate(
            zip(projection.req_ids, projection.bindings)
        ):
            request = metadata[request_id]
            token = projection.token_ids[index]
            file_row = projection.file_rows.get(request_id)
            if request_id in planning_failed or (
                file_row is not None and file_row.failed
            ):
                # Failed at planning: the row stays at its position, bound
                # non-CHUNK with its planned-rejection bit, which the
                # transaction rejects; its tokens are not classified.
                rows.append(
                    ObservedRow(
                        request_id=request_id,
                        block_id=int(binding.slot_id),
                        scheduled_token_id=token,
                        has_prior_state=not bool(binding.fresh),
                        envelope_header=None,
                        is_file=file_row is not None,
                        reject=True,
                    )
                )
                continue
            if int(input_batch.num_scheduled_tokens[index]) != 1:
                raise ValueError(
                    f"row {request_id!r} schedules more than one token but did "
                    "not fail at planning"
                )
            if file_row is not None:
                # FILE-ROW-001: a FILE row's role and header come from the
                # step plan, not from the request's multimodal feature; it runs
                # with endpointing disabled (FILE-ENG-002), and screening has
                # already failed it for a malformed policy.
                plan = file_row.file_plan
                assert plan is not None
                rows.append(
                    ObservedRow(
                        request_id=request_id,
                        block_id=int(binding.slot_id),
                        scheduled_token_id=token,
                        has_prior_state=not bool(binding.fresh),
                        envelope_header=file_row.header,
                        is_file=True,
                        file_advances=plan.advances_audio,
                    )
                )
                continue
            endpoint_mode, endpoint_threshold, endpoint_residue = (
                self._endpoint_controls(request)
            )
            rows.append(
                ObservedRow(
                    request_id=request_id,
                    block_id=int(binding.slot_id),
                    scheduled_token_id=token,
                    has_prior_state=not bool(binding.fresh),
                    envelope_header=stream_headers.get(request_id),
                    endpoint_mode=endpoint_mode,
                    endpoint_threshold_frames=endpoint_threshold,
                    endpoint_residue_frames=endpoint_residue,
                )
            )
        return self._stage_plan(
            rows,
            resident_request_ids=tuple(projection.req_states.req_id_to_index),
        )

    def _require_execution_inventory_ready(
        self, *, capture_pending_ok: bool = False
    ) -> None:
        """Validate complete model-owned inventory without creating allocations.

        ``capture_pending_ok``: accept segment graphs that core's capture hook
        has not captured yet (the hook configuration before its serving
        capture; serving steps never pass it)."""
        execution = self._encoder_execution
        if capture_pending_ok and execution.arm in ("dense-graphed", "eager-graphed"):
            if not execution.warmed:
                raise RuntimeError("encoder execution inventory is incomplete")
        elif not execution.ready:
            raise RuntimeError("encoder execution inventory is incomplete")
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        if chunk_binding is not None and not chunk_binding.ready:
            raise RuntimeError("CHUNK execution inventory is incomplete")
        binding = self._decode_graph_binding
        if binding is None:
            if getattr(self.config, "decode_dispatch_arm", None) == "dense-graphed":
                raise RuntimeError("decoder execution inventory is missing")
            return
        if capture_pending_ok and not binding.captured_keys:
            return
        from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import execution_tiers

        expected = {
            (geometry, tier)
            for geometry in (_server_geometry(self.config),)
            for tier in execution_tiers(self._max_num_seqs)
        }
        expected = {
            execution_mode(self.core).graph_key("decode_fn", key) for key in expected
        }
        if set(binding.captured_keys) != expected:
            raise RuntimeError("decoder execution inventory is incomplete")

    def _graphs_present(self) -> bool:
        """Whether every captured graph entry of every wrapper holds a graph."""
        binding = self._decode_graph_binding
        return self._encoder_execution.graphs_present and (
            binding is None or binding.graphs_present
        )

    def _require_execution_ready(self, *, capture_pending_ok: bool = False) -> None:
        """Readiness of the whole inventory, including that every captured
        graph entry still holds its graph, without creating allocations."""
        self._require_execution_inventory_ready(capture_pending_ok=capture_pending_ok)
        if not self._graphs_present():
            raise RuntimeError(
                "a captured execution graph entry holds no graph; the model "
                "never captures while serving"
            )

    def _segment_graphs_pending(self) -> bool:
        """Whether core's capture hook has yet to capture the serving segment
        graphs (the hook configuration before its serving capture)."""
        return self._graphs_by_core and self._segment_graphs != SERVING_CAPTURE

    # @spec REG-MEM-002, REG-MEM-003, REG-MEM-004, REG-MEM-005, REG-MEM-008
    # @spec REG-MEM-016
    def _profile_dummy_step(self, device: torch.device) -> None:
        """The model's one preparation of its execution memory, run by the
        forward of a profiling dummy step (core's memory profile).

        The first such step of a worker prepares: under ``--enforce-eager`` it
        captures the segment graphs and then, with every graph retained, runs
        the profile transaction; under the hook configuration it runs the
        profile transaction over the arms' uncaptured paths and captures
        nothing (core's capture hook captures later). A failure discards the
        partial state, marks the worker failed and raises, and every later
        profiling step then raises without preparing. Later profiling steps of
        a prepared worker (kernel warm-up after the state pool exists) only
        verify readiness: no capture, no profile transaction, no state read.
        """
        if self._execution_memory_failed:
            raise RuntimeError(
                "execution memory preparation failed; a fresh worker is required"
            )
        if self._execution_memory_prepared:
            # Core's capture hook may not have captured yet (graphs by core).
            self._require_execution_ready(
                capture_pending_ok=self._segment_graphs_pending()
            )
            return
        if not self._working_memory_prepared:
            # Core calls prepare_execution_memory before this step; prepare
            # here only if it did not.
            self.prepare_execution_memory_hook(
                device, graphs_by_core=self._graphs_by_core
            )
        try:
            if self._graphs_by_core:
                self._profile_uncaptured(device)
            else:
                self._capture_on_profiling_step(device)
        except Exception as error:
            self._fail_preparation(device, error)
            raise
        self._execution_memory_prepared = True

    def _fail_preparation(
        self, device: torch.device, error: BaseException | None = None
    ) -> None:
        """REG-MEM-004: discard the partial execution state, synchronize the
        device and mark this worker's preparation failed. The finished frames
        ``error`` holds are cleared first (REG-MEM-017), so the propagated
        error keeps no reference to the tensors or graphs the failed
        preparation or capture created."""
        graph_memory.release_error_frames(error)
        self._execution_memory_failed = True
        try:
            self._encoder_execution._discard()
        finally:
            if self._decode_graph_binding is not None:
                self._decode_graph_binding.discard()
            self._segment_graphs = None
            _synchronize_quietly(device, None)

    # @spec PORT-STATE-003, PORT-PERF-009
    @torch.inference_mode()
    def _prepare_working_memory(self, device: torch.device) -> None:
        """Allocate long-lived working memory and warm uncaptured execution:
        the static encoder domain is warmed and sealed (and a compiled arm
        compiled). Captures nothing and runs no profile transaction."""
        from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
            warmup_static_encoder_execution,
        )

        warmup_static_encoder_execution(self, device=device)

    # @spec PORT-STATE-003, PORT-PERF-009, PORT-PERF-011, REG-MEM-002
    @torch.inference_mode()
    def _capture_on_profiling_step(self, device: torch.device) -> None:
        """``--enforce-eager``: capture the segment graphs, verify the whole
        inventory, then profile its transient headroom with every graph
        retained.

        Profiling uses model-owned scratch state built from the model's
        state layout; it reads no state pool.
        """
        from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
            run_persistent_state_profile,
        )

        self._capture_segment_graphs(
            device, capture=PROFILING_STEP, stream=None, graph_pool=None
        )
        self._segment_graphs = PROFILING_STEP
        self._require_execution_ready()
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        # All retained graph buffers must coexist with this temporary
        # transaction so the profiler sees their combined high-water mark.
        if chunk_binding is not None:
            for geometry, population in sorted(
                self._encoder_execution._chunk_graph_cells
            ):
                run_persistent_state_profile(
                    self,
                    num_rows=population,
                    device=device,
                    geometry_id=geometry,
                    ready_domain=True,
                )
        run_persistent_state_profile(
            self,
            num_rows=self._max_num_seqs,
            device=device,
            geometry_id=_server_geometry(self.config),
            ready_domain=True,
        )
        self._require_execution_ready()

    @torch.inference_mode()
    def _profile_uncaptured(self, device: torch.device) -> None:
        """The hook configuration: verify every configured execution entry
        except the segment graphs and run the profile transaction once, at its
        maximum shape, over the arms' uncaptured paths (the encoder's sealed
        uncaptured transition, the dense eager decode). Captures nothing."""
        from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
            run_persistent_state_profile,
        )

        self._require_execution_inventory_ready(capture_pending_ok=True)
        run_persistent_state_profile(
            self,
            num_rows=self._max_num_seqs,
            device=device,
            geometry_id=_server_geometry(self.config),
            ready_domain=True,
            uncaptured=True,
        )
        self._require_execution_inventory_ready(capture_pending_ok=True)

    # Core's model-state graph hooks: the model state forwards
    # ``prepare_execution_memory``, ``capture_cudagraphs``,
    # ``clear_cudagraphs``, ``post_warmup`` and ``shutdown`` here. The
    # segment-graph part below runs first in each capture call, after the
    # transaction-graph entry checks; the transaction-graph part is called
    # through ``_transaction_graph_binding``, which is ``None`` while
    # transaction graphs are off. The calls it receives:
    # ``check_capture_entry(profile_only=)`` (TXG-LIFE-007, TXG-LIFE-008),
    # ``check_profiling_requirement(requirement)`` (TXG-MEM-007, TXG-MEM-009),
    # ``capture(call)`` after the segment graphs (TXG-LIFE-004, TXG-LIFE-006,
    # with the margin and the comparison through ``graph_memory``; the call
    # carries the model state's ``state_group_layouts`` reader, TXG-CAP-008),
    # ``check_bindings(layer)`` from the state layer's bind observer
    # (TXG-LIFE-009),
    # ``discard_capture(error)`` on a failed call (TXG-LIFE-012),
    # ``release_profiling()`` (TXG-LIFE-005), ``verify_post_warmup()`` and
    # ``shutdown()`` (TXG-LIFE-010), each before the segment-graph part.

    def prepare_execution_memory_hook(
        self, device: torch.device, *, graphs_by_core: bool
    ) -> None:
        """Core's ``prepare_execution_memory``: once, before the profiling
        forward, inside core's memory profile, in both runner configurations.
        Allocates long-lived working memory and warms uncaptured execution;
        captures nothing and runs no profile transaction.

        ``graphs_by_core``: core's capture hook captures the segment graphs
        (the hook configuration); otherwise (``--enforce-eager``) the first
        profiling step captures them.
        """
        if self._working_memory_prepared or self._execution_memory_failed:
            return
        self._graphs_by_core = bool(graphs_by_core)
        self._working_memory_prepared = True
        try:
            self._prepare_working_memory(device)
        except Exception as error:
            self._fail_preparation(device, error)
            raise

    # @spec REG-MEM-017, REG-MEM-023, REG-MEM-024
    @torch.inference_mode()
    def capture_graphs_hook(
        self,
        device: torch.device,
        *,
        graph_pool: Any,
        stream: Any,
        profile_only: bool,
        state_group_layouts: Any = None,
    ) -> None:
        """Core's ``capture_cudagraphs``: capture the segment graphs for the
        server's geometry into ``graph_pool`` on ``stream``, then the
        transaction-graph part; a profiling call (``profile_only``) ends with
        the charged margin, a serving call with the memory comparison.
        ``state_group_layouts`` reads the model state's
        ``state_group_layouts`` during the call: the transaction-graph part's
        only way to the state pools (TXG-CAP-008).

        A refused call captures nothing. A failure after the entry checks
        discards everything the call created and synchronizes ``stream``
        before propagating.
        """
        if self._execution_memory_failed or not self._execution_memory_prepared:
            raise RuntimeError(
                "execution memory is not prepared on this worker (core's "
                "profiling step prepares it before the capture hook)"
            )
        if not self._graphs_by_core:
            raise RuntimeError(
                "core's capture hook was called, but this worker prepared for "
                "--enforce-eager, where the profiling step captures"
            )
        binding = self._transaction_graph_binding
        if binding is not None:
            binding.check_capture_entry(profile_only=profile_only)
        self._check_capture_order(profile_only)
        requirement = None
        if not profile_only:
            if binding is not None:
                # TXG-MEM-007: decided from the worker's effective cache
                # configuration when the serving capture begins, and recorded
                # whether or not a profiling capture ran.
                requirement = graph_memory.profiling_requirement(self._vllm_config)
                binding.check_profiling_requirement(
                    requirement, profiling_ran=self._profiling_figures is not None
                )
            elif self._profiling_figures is None:
                requirement = graph_memory.profiling_requirement(self._vllm_config)
                if requirement.required:
                    raise RuntimeError(
                        "the serving capture found no profiling capture in this "
                        "worker although one was required: "
                        f"{requirement.reason}. Core's graph-memory estimate "
                        "(capture_cudagraphs(profile_only=True)) did not run, so "
                        "the segment graphs would not be charged to the budget; "
                        "restart the engine with the estimate enabled, or set a "
                        "fixed --kv-cache-memory-bytes that leaves room for them"
                    )
            # The entry checks passed: from here the serving capture counts as
            # attempted, whether or not it succeeds (REG-MEM-024). A call
            # refused at its entry captured nothing and is no attempt.
            self._serving_capture_attempted = True
        if binding is not None:
            # The entry checks passed: the call's transient peak starts here,
            # before the segment-graph part captures anything (TXG-MEM-002).
            binding.begin_capture(profile_only=profile_only, device=device)
        capture = PROFILING_CAPTURE if profile_only else SERVING_CAPTURE
        try:
            call_free_before, segment_free_after, seconds = (
                self._capture_segment_graphs(
                    device, capture=capture, stream=stream, graph_pool=graph_pool
                )
            )
            self._segment_graphs = capture
            self._require_execution_ready()
            call = graph_memory.CaptureCall(
                capture=capture,
                profile_only=profile_only,
                device=device,
                stream=stream,
                graph_pool=graph_pool,
                call_free_before=call_free_before,
                segment_free_after=segment_free_after,
                segment_seconds=seconds,
                requirement=requirement,
                profiling_ran=self._profiling_figures is not None,
                state_group_layouts=state_group_layouts,
            )
            if binding is not None:
                binding.capture(call)
                if profile_only:
                    self._profiling_figures = {
                        "segment_total": call.segment_total
                    }
            elif profile_only:
                self._end_profiling_call(call)
            else:
                self._end_serving_call(call)
        except Exception as error:
            self._discard_capture_call(stream, error)
            raise

    def _check_capture_order(self, profile_only: bool) -> None:
        """REG-MEM-024: refuse a capture call that core's lifecycle does not
        make, at the call's entry, before capturing."""
        option = (
            f" ({TRANSACTION_GRAPH_KEY} is set)"
            if transaction_graphs_on(self.config)
            else ""
        )
        if not profile_only and self._serving_capture_attempted:
            raise RuntimeError(
                "a serving capture after a serving capture was attempted in this "
                "worker, whether or not that attempt succeeded: recapture over "
                "live state pools is not supported; recover with an engine "
                f"restart{option}"
            )
        if profile_only and self._serving_capture_attempted:
            raise RuntimeError(
                "a profiling capture after a serving capture was attempted in "
                "this worker is a lifecycle defect (core captures for its "
                "graph-memory estimate before it serves); recover with an engine "
                f"restart{option}"
            )
        if self._segment_graphs == PROFILING_CAPTURE:
            raise RuntimeError(
                f"a {'profiling' if profile_only else 'serving'} capture while "
                "the profiling capture's segment graphs exist is a lifecycle "
                "defect: core releases them (clear_cudagraphs) before it "
                f"captures again{option}"
            )

    def _end_profiling_call(self, call: Any) -> None:
        """Transaction graphs off: end the profiling call with the charged
        margin and keep the call's figures as host integers."""
        margin = graph_memory.allocate_margin(
            call.device, stream=call.stream, free_before=call.segment_free_after
        )
        self._charged_margin = margin
        self._profiling_figures = {
            "segment_total": call.segment_total,
            "call_total": graph_memory.retained(
                call.call_free_before, margin.free_after
            ),
            "margin_bytes": margin.margin_bytes,
            "margin_charge": margin.charge,
            "margin_free_before": margin.free_before,
            "margin_free_after": margin.free_after,
        }

    def _end_serving_call(self, call: Any) -> None:
        """Transaction graphs off: compare the serving call with the profiling
        call, or, when no profiling capture ran and none was required, warn
        that the segment graphs were not charged to the budget."""
        serving_total = call.segment_total
        figures = self._profiling_figures
        if figures is None:
            requirement = call.requirement
            logger.warning(
                "Nemotron segment graphs were not charged to the budget: no "
                "profiling capture ran, and none was required (%s). The serving "
                "capture retains %s; the budget must leave room for it.",
                requirement.reason if requirement is not None else "unknown",
                graph_memory.format_bytes(serving_total),
            )
            return
        profiling_total = figures["call_total"]
        excess = graph_memory.serving_excess(serving_total, profiling_total)
        if excess:
            raise RuntimeError(
                "the serving capture retains more graph memory than the "
                "profiling capture core charged to the budget: serving total "
                f"{graph_memory.format_bytes(serving_total)}, profiling total "
                f"{graph_memory.format_bytes(profiling_total)} (including the "
                "charged margin's charge "
                f"{graph_memory.format_bytes(figures['margin_charge'])} of a "
                f"{graph_memory.format_bytes(figures['margin_bytes'])} margin), "
                f"excess {graph_memory.format_bytes(excess)}. "
                + self._capture_remedies(serving=True)
            )

    # @spec REG-MEM-017, TXG-LIFE-012
    def _discard_capture_call(self, stream: Any, error: BaseException) -> None:
        """Discard everything a capture call that failed with ``error``
        created, the transaction-graph part first, then the segment graphs,
        their buffers and the margin, and synchronize the capture stream.
        With transaction graphs on or off, the finished frames ``error``
        holds, through its cause and its context, are cleared first, so the
        propagated error keeps no reference to the call's tensors or graphs;
        its messages and chain stay."""
        graph_memory.release_error_frames(error)
        binding = self._transaction_graph_binding
        try:
            if binding is not None:
                binding.discard_capture(error)
        finally:
            self._discard_segment_graphs()
            margin = self._charged_margin
            if margin is not None:
                margin.release()
                self._charged_margin = None
            _synchronize_quietly(None, stream)

    def _discard_segment_graphs(self) -> None:
        """Release every segment graph and segment-graph buffer (encoder,
        decoder and CHUNK-bucket entries), keeping the sealed execution
        domain."""
        execution = getattr(self, "_encoder_execution", None)
        if execution is not None:
            execution.release_graphs()
        binding = getattr(self, "_decode_graph_binding", None)
        if binding is not None:
            binding.discard()
        self._segment_graphs = None

    # @spec REG-MEM-015, REG-MEM-017
    def _capture_segment_graphs(
        self,
        device: torch.device,
        *,
        capture: str,
        stream: Any,
        graph_pool: Any,
    ) -> tuple[int | None, int | None, float]:
        """Capture every segment graph for the server's geometry: the encoder
        at each CHUNK count, then the decoder at each tier, then the
        CHUNK-bucket cells, on ``stream`` into ``graph_pool`` (core's capture
        hook) or core's own (``None``). Logs the capture line and returns the
        free-memory readings around the capture and its time.

        A failure raises an error chained from the original that names the
        arms, the geometry, ``max_num_seqs``, the failing graph, what was
        retained so far, the time so far, the budget setting and the remedies;
        the caller discards what was captured.
        """
        from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
            capture_static_encoder_graphs,
        )

        execution = self._encoder_execution
        decoder = self._decode_graph_binding
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        execution.capture_in_progress = None
        if decoder is not None:
            decoder.capture_in_progress = None
        if chunk_binding is not None:
            chunk_binding.capture_in_progress = None
        free_before = graph_memory.free_memory(device, stream)
        started = time.perf_counter()
        try:
            capture_static_encoder_graphs(
                self, device=device, stream=stream, graph_pool=graph_pool
            )
            if decoder is not None:
                decoder.warmup(
                    device,
                    self.core.policy.dtype_for("activations"),
                    stream=stream,
                    graph_pool=graph_pool,
                )
            if chunk_binding is not None:
                chunk_binding.warmup(device)
        except Exception as error:
            seconds = time.perf_counter() - started
            try:
                so_far = graph_memory.retained(
                    free_before, graph_memory.free_memory(device, stream)
                )
            except Exception:
                so_far = None
            raise RuntimeError(
                self._capture_failure_message(
                    capture, graph=self._failing_segment_graph(), retained=so_far,
                    seconds=seconds,
                )
            ) from error
        seconds = time.perf_counter() - started
        free_after = graph_memory.free_memory(device, stream)
        logger.info(
            "Nemotron segment graphs: %s; retained %s, capture time %.3f s",
            self._capture_identity(capture),
            graph_memory.format_bytes(graph_memory.retained(free_before, free_after)),
            seconds,
        )
        self._log_graph_stand_in(device)
        return free_before, free_after, seconds

    # @spec REG-MEM-028
    def _log_graph_stand_in(self, device: torch.device) -> None:
        """Log one INFO line, once per worker, naming each graphed arm that
        the eager stand-in wrapper serves on ``device`` (a platform without
        static graphs), so a run that asked for graphed arms shows that it
        executes them eagerly."""
        if self._graph_stand_in_logged:
            return
        arms: list[str] = []
        execution = self._encoder_execution
        runtime = getattr(execution, "_graph_runtime", None)
        if runtime is not None and not runtime.captures_on(device):
            arms.append(f"encoder_execution_arm {execution.arm}")
            if getattr(self, "_chunk_bucket_binding", None) is not None:
                arms.append("chunk_bucket_graph_populations")
        decoder = self._decode_graph_binding
        runtime = getattr(decoder, "_runtime", None)
        if runtime is not None and not runtime.captures_on(device):
            arm = getattr(self.config, "decode_dispatch_arm", None) or "dense-graphed"
            arms.append(f"decode_dispatch_arm {arm}")
        if not arms:
            return
        self._graph_stand_in_logged = True
        logger.info(
            "Nemotron graphed arms run eagerly on %s, a platform or device "
            "without static graphs: %s are served through the eager stand-in "
            "graph wrapper; nothing is captured, pooled or synchronized",
            device,
            ", ".join(arms),
        )

    def _capture_identity(self, capture: str) -> str:
        """The runner configuration, the capture, the arms, the server's
        geometry and ``max_num_seqs``, as the capture line and the failure
        messages name them."""
        from vllm_nemotron_asr_streaming.lookahead import (
            cadence_label,
            served_lookahead,
        )

        lookahead = served_lookahead(self.config)
        runner = "hook configuration" if self._graphs_by_core else "--enforce-eager"
        encoder = getattr(self._encoder_execution, "arm", None) or "eager"
        decoder = getattr(self.config, "decode_dispatch_arm", None) or "compact-eager"
        return (
            f"{runner} runner, {capture}; encoder_execution_arm {encoder}, "
            f"decode_dispatch_arm {decoder}; geometry {cadence_label(lookahead)} "
            f"(lookahead {lookahead}); max_num_seqs {self._max_num_seqs}"
        )

    def _failing_segment_graph(self) -> str:
        """The segment graph whose staging or capture was in progress."""
        from vllm_nemotron_asr_streaming.lookahead import cadence_label

        labels = {
            index: cadence_label(right)
            for index, (_left, right) in enumerate(CADENCES.values())
        }
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        if chunk_binding is not None and chunk_binding.capture_in_progress:
            geometry, population = chunk_binding.capture_in_progress
            return (
                f"the CHUNK-bucket graph at the {labels[geometry]} geometry with "
                f"{population} rows (chunk_bucket_graph_populations)"
            )
        decoder = self._decode_graph_binding
        if decoder is not None and decoder.capture_in_progress:
            geometry, tier = decoder.capture_in_progress
            return f"the decoder graph at tier {tier} ({labels[geometry]})"
        cell = self._encoder_execution.capture_in_progress
        if cell is not None:
            geometry, count = cell
            return f"the encoder graph at CHUNK count {count} ({labels[geometry]})"
        return "a segment graph (before its first CHUNK count or tier)"

    def _budget_setting(self) -> tuple[str, bool]:
        """The budget setting, and whether it is a fixed budget."""
        cache = getattr(self._vllm_config, "cache_config", None)
        fixed = getattr(cache, "kv_cache_memory_bytes", None)
        if fixed:
            return f"a fixed budget (--kv-cache-memory-bytes {fixed})", True
        utilization = getattr(cache, "gpu_memory_utilization", None)
        return (
            f"a profiled budget (--gpu-memory-utilization {utilization})",
            False,
        )

    def _capture_remedies(self, *, serving: bool) -> str:
        """REG-MEM-017's remedies for this capture and budget."""
        budget, fixed = self._budget_setting()
        remedies = [
            "a smaller --max-num-seqs",
            "--hf-overrides '{\"encoder_execution_arm\": \"eager\", "
            '"decode_dispatch_arm": "dense-eager", '
            '"chunk_bucket_graph_populations": [], '
            '"transaction_graph_populations": []}\' together with --enforce-eager',
        ]
        if fixed:
            remedies.append("a smaller --kv-cache-memory-bytes")
        elif serving:
            remedies.append("a lower --gpu-memory-utilization")
        return f"Budget: {budget}. Remedies: " + "; or ".join(remedies) + "."

    def _capture_failure_message(
        self,
        capture: str,
        *,
        graph: str,
        retained: int | None,
        seconds: float,
    ) -> str:
        """REG-MEM-017: the message of a segment-graph capture failure."""
        return (
            f"Nemotron segment-graph capture failed ({self._capture_identity(capture)}): "
            f"{graph} failed after retaining {graph_memory.format_bytes(retained)} "
            f"in {seconds:.3f} s. "
            + self._capture_remedies(serving=capture == SERVING_CAPTURE)
        )

    def clear_graphs_hook(self) -> None:
        """Core's ``clear_cudagraphs``: release what the profiling capture
        created, the transaction-graph part first, then every segment graph,
        segment-graph buffer and the charged margin, keeping the profiling
        figures as host integers. Tolerates nothing captured."""
        binding = getattr(self, "_transaction_graph_binding", None)
        try:
            if binding is not None:
                binding.release_profiling()
        finally:
            if getattr(self, "_segment_graphs", None) == PROFILING_CAPTURE:
                self._discard_segment_graphs()
            margin = getattr(self, "_charged_margin", None)
            if margin is not None:
                margin.release()
                self._charged_margin = None

    def post_warmup_hook(self) -> None:
        """Core's ``post_warmup``, before the worker serves, in either runner
        configuration: every expected segment-graph entry holds its graph,
        then the transaction-graph verification. Allocates nothing and
        replays nothing."""
        if self._execution_memory_failed:
            raise RuntimeError(
                "execution memory preparation failed; a fresh worker is required"
            )
        if self._graphs_by_core and self._segment_graphs != SERVING_CAPTURE:
            raise RuntimeError(
                "post_warmup before core's serving capture: the segment graphs "
                "are not captured"
            )
        self._require_execution_ready()
        binding = self._transaction_graph_binding
        if binding is not None:
            binding.verify_post_warmup()

    def shutdown_graphs_hook(self) -> None:
        """Core's ``shutdown``: release every graph and binding-owned buffer,
        the transaction-graph part first. Tolerates a model whose
        initialization did not complete or that captured nothing."""
        binding = getattr(self, "_transaction_graph_binding", None)
        try:
            if binding is not None:
                binding.shutdown()
        finally:
            self._discard_segment_graphs()
            margin = getattr(self, "_charged_margin", None)
            if margin is not None:
                margin.release()
                self._charged_margin = None

    def execution_profile_receipt(self) -> dict[str, Any]:
        """Report the resolved execution arms and startup coverage."""
        binding = self._decode_graph_binding
        keys = () if binding is None else binding.captured_keys
        decode_arm = (
            getattr(self.config, "decode_dispatch_arm", None) or "compact-eager"
        )
        receipt = {
            "schema": "nemotron-execution-profile/1",
            "decode": {
                "arm": decode_arm,
                "captured_keys": [list(key) for key in keys],
                "ready": binding is None or bool(keys),
            },
            "encoder": self._encoder_execution.ready_receipt(),
        }
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        if chunk_binding is not None:
            receipt["chunk"] = chunk_binding.receipt()
        return execution_mode(getattr(self, "core", None)).readiness_receipt(receipt)

    def _ensure_commit_sink(self, device: torch.device) -> BoundedCommitSink:
        if self._commit_sink is None:
            self._commit_sink = BoundedCommitSink(
                self._registry,
                max_rows=self._max_num_seqs,
                device=device,
            )
        return self._commit_sink

    def _ensure_host_staging(self) -> HostStaging:
        if self._host_staging is None:
            self._host_staging = HostStaging(self._max_num_seqs)
        return self._host_staging

    # @spec FILE-QC-005, REG-ENG-020
    def collect_commit_status(self) -> CommitCollection:
        """Consume the staged step once: the resolved statuses, the requests
        to fail and one ``QueueCount`` per report resolved to status 0 that
        carries a count; all three empty without a staged commit, after
        waiting for the completion event of a step that recorded one without
        staging (an all-failed step)."""
        sink = self._commit_sink
        if sink is None:
            return CommitCollection()
        if not sink.has_staged:
            sink.wait_fence()
            return CommitCollection()
        reports, _records, lease_ok = sink.collect()
        statuses, failed = resolve_status_reports(
            self._registry, reports, lease_ok=lease_ok
        )
        queue_counts = tuple(
            QueueCount(
                request_id=report.request_id,
                admission_generation=report.admission_generation,
                count=int(report.queue_count),
            )
            for report in reports
            if statuses.get(report.request_id) == 0 and report.queue_count is not None
        )
        self._committed_progress = {}
        for report in reports:
            if statuses.get(report.request_id) != 0 or report.progress_key is None:
                continue
            key = self._registry.publish_progress(
                report.request_id,
                report.admission_generation,
                report.progress_key,
                starts_chunk=report.starts_chunk,
            )
            if key is not None:
                self._committed_progress[report.request_id] = key
        return CommitCollection(
            statuses=statuses, failed=failed, queue_counts=queue_counts
        )

    def consume_committed_progress(self) -> dict[str, tuple[int, int, int]]:
        progress = getattr(self, "_committed_progress", {})
        self._committed_progress = {}
        return progress

    # @spec FILE-PROMPT-004
    def embed_multimodal(self, **kwargs: Any) -> Any:
        header = kwargs.get("file_header")
        if header is not None or "samples" in kwargs:
            # A complete-file item: one placeholder row per item. The samples
            # stay on the host (the model state takes them into the request's
            # buffer) and are never read here; the stream carrier-width check
            # does not apply.
            count = len(header if header is not None else kwargs["samples"])
            device = next(self.parameters()).device
            hidden = int(self.config.hidden_size)
            return [
                torch.zeros(1, hidden, dtype=torch.float32, device=device)
                for _ in range(count)
            ]
        audios = kwargs.get("audio")
        if audios is None:
            raise ValueError("embed_multimodal expects an 'audio' item")
        device = next(self.parameters()).device
        hidden = int(self.config.hidden_size)
        rows = []
        for chunk in audios:
            envelope = (
                (
                    chunk
                    if isinstance(chunk, torch.Tensor)
                    else torch.as_tensor(getattr(chunk, "audio_arrays", chunk))
                )
                .reshape(-1)
                .to(dtype=torch.float32)
            )
            # Screening fails a request whose envelope breaks this rule
            # before core's encoder path runs (REG-STATE-015).
            fault = envelope_width_fault(int(envelope.shape[0]), carrier_width=hidden)
            if fault is not None:
                raise ValueError(f"audio {fault}")
            row = torch.zeros(hidden, dtype=torch.float32)
            row[: envelope.shape[0]] = envelope
            rows.append(row.to(device).unsqueeze(0))
        return rows

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_nemotron_asr_streaming.nemotron_asr.forward_ops import (
            merge_mm_embeddings,
        )

        hidden = int(self.config.hidden_size)
        if multimodal_embeddings is None or is_multimodal is None:
            return torch.zeros(
                input_ids.shape[0],
                hidden,
                dtype=torch.float32,
                device=input_ids.device,
            )
        return merge_mm_embeddings(
            input_ids,
            multimodal_embeddings,
            is_multimodal,
            hidden_size=hidden,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        persistent_state_projection: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, kwargs
        from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
            advance_model_rows,
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
            build_row_plan,
        )
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        if inputs_embeds is None:
            raise RuntimeError("Nemotron forward requires embeddings")
        if (
            persistent_state_projection is not None
            and persistent_state_projection.no_page_io
        ):
            if not persistent_state_projection.dummy_run:
                raise RuntimeError("no-page-I/O projection must be a dummy invocation")
            # @spec REG-MEM-001, REG-MEM-002, REG-MEM-003, REG-MEM-009, REG-MEM-010
            if persistent_state_projection.is_profile:
                self._profile_dummy_step(inputs_embeds.device)
            return torch.zeros_like(inputs_embeds)
        if input_ids is None:
            raise RuntimeError("Nemotron forward requires raw ids")
        if persistent_state_projection is None:
            raise ValueError("Nemotron forward requires an MRv2 state projection")
        binding = self._transaction_graph_binding
        if binding is not None and binding.failed:
            # @spec TXG-FAIL-003, TXG-FAIL-010: a latched worker refuses every
            # real step that schedules a row, before any other check.
            from vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph import (
                LATCHED,
            )

            raise RuntimeError(LATCHED)
        # @spec REG-MEM-011, REG-MEM-006: a real step runs only on a worker
        # whose preparation succeeded, and never captures while serving.
        if not self._execution_memory_prepared:
            raise RuntimeError(
                "execution memory was not prepared on this worker (preparation "
                "runs on core's profiling step and never lazily while serving)"
            )
        if self._graphed_execution:
            # The expected inventory and every graph, not only graph presence;
            # the eager default keeps its steps cheap.
            self._require_execution_ready()
            if self._segment_graphs_pending():
                raise RuntimeError(
                    "a real step before core's serving capture: the serving "
                    "segment graphs are not captured"
                )
        context = self._stage_v2_projection(persistent_state_projection)
        planning_failed = self._record_planning(persistent_state_projection, context)
        state_batch = persistent_state_projection.state_batch
        if state_batch is None:
            raise RuntimeError("MRv2 projection omitted bounded state views")
        # @spec REG-STATE-010: a fresh row's page may hold a previous owner's
        # bytes (core does not clear a reused block): initialize it on the
        # current stream before the transaction reads it, whether or not the
        # row failed at planning, an all-failed step included.
        state_batch.initialize_fresh_pages()
        row_ids, row_embeds, last_positions = self._rows_by_position(
            input_ids, inputs_embeds, persistent_state_projection.input_batch
        )
        if (
            planning_failed
            and len(planning_failed) == len(context.request_ids)
            and self._native_burst_handoff is None
        ):
            output = self._runner_output(
                self._all_failed_bypass(context, row_embeds),
                last_positions,
                inputs_embeds,
            )
            # @spec REG-ENG-020: the step stages no commit, but its device
            # work read the step's state and inputs (the fresh pages and the
            # row gather above); its commit event follows that work, and
            # ``postprocess_state`` waits for it.
            self._ensure_commit_sink(inputs_embeds.device).record_fence()
            return output
        pools = self._state_pools(persistent_state_projection)
        self._write_file_envelopes(row_embeds, persistent_state_projection)
        commit_sink = state_batch.commit_sink(
            self._ensure_commit_sink(inputs_embeds.device)
        )
        num_decodes = 0
        num_prefills = len(context.request_ids)
        graph_binding = self._decode_graph_binding
        graph_covers_decode = graph_binding is not None
        chunk_rows = int(context.is_chunk.sum().item())
        execution_tier = 0
        if graph_binding is not None and chunk_rows:
            execution_tier = graph_binding.execution_tier(chunk_rows)
        plan = build_row_plan(
            context,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            null_block_id=int(NULL_BLOCK_ID),
            num_pool_blocks=int(pools.replay_queue.shape[0]),
            execution_tier=execution_tier,
        )
        if last_positions is not None:
            # @spec TXG-COV-014: a step with a multi-token row; its rows were
            # placed by their token positions above (TXG-TXN-010).
            plan = dataclasses.replace(plan, multi_token=True)
        native_handoff = self._native_burst_handoff
        if native_handoff is not None:
            if persistent_state_projection is None:
                raise RuntimeError("native burst requires a native runner projection")
            native_handoff.prepare(persistent_state_projection)
        binding = self._transaction_graph_binding
        # @spec TXG-COV-012: every transaction of a real step receives the
        # endpoint arguments (the endpoint history and book pools and the EOU
        # token), with transaction graphs on or off, so it runs with
        # ``endpoint_enabled`` true and a row without an endpoint policy takes
        # the forced-EOU path exactly as in ``B_hook``.
        transaction: dict[str, Any] = dict(
            channel_pools=list(pools.channel),
            time_pools=list(pools.convolution),
            len_pools=list(pools.valid_length),
            h_pool=pools.predictor_h,
            c_pool=pools.predictor_c,
            queue_pool=pools.replay_queue,
            book_pool=pools.replay_book,
            frontend_raw_pool=pools.frontend_raw,
            frontend_mel_pool=pools.frontend_mel,
            frontend_counter_pool=pools.frontend_counters,
            endpoint_history_pool=pools.endpoint_history,
            endpoint_book_pool=pools.endpoint_book,
            eou_token_id=int(self.config.eou_token_id),
            adapter=self._emission_adapter,
            decode_resolver=self._decode_resolver,
            encoder_transition=self._encoder_execution.transition,
            bucket_transition=getattr(self, "_chunk_bucket_binding", None),
            placeholder_id=int(self.config.audio_chunk_token_id),
            park_id=int(self.config.eos_token_id),
            commit_sink=commit_sink,
            staging=self._ensure_host_staging(),
            graph_covers_decode=graph_covers_decode,
            native_burst_handoff=native_handoff,
            file_carrier_id=int(self.config.file_carrier_token_id),
            continuation_id=int(self.config.continuation_token_id),
            geometry_order=persistent_state_projection.geometry_order,
        )
        try:
            if binding is not None:
                # @spec TXG-COV-003, TXG-EXEC-012: a scheduled step replays
                # its cells; a step ``execute`` declines (already counted)
                # runs wholly as the option-off transaction below. ``execute``
                # is looked up on the binding at call time.
                result = binding.execute(
                    self.core, row_ids, row_embeds, plan, **transaction
                )
                if result is not None:
                    return self._runner_output(result, last_positions, inputs_embeds)
            result = advance_model_rows(
                self.core,
                row_ids,
                row_embeds,
                plan,
                **transaction,
            )
            return self._runner_output(result, last_positions, inputs_embeds)
        except BaseException:
            if native_handoff is not None:
                native_handoff.clear()
            raise

    # @spec REG-STATE-014, TXG-TXN-010
    @staticmethod
    def _rows_by_position(
        input_ids: torch.Tensor, inputs_embeds: torch.Tensor, input_batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Each row's runner inputs at its last token position, the position
        core samples. In a one-token step row ``i`` is position ``i`` and the
        runner's tensors pass through; otherwise (a row that failed at
        planning for scheduling several tokens) each row's ``input_ids`` and
        ``inputs_embeds`` are gathered from ``query_start_loc[i + 1] - 1``,
        whose index is returned for the output."""
        counts = getattr(input_batch, "num_scheduled_tokens", None)
        if counts is None or all(int(c) == 1 for c in counts):
            return input_ids.long(), inputs_embeds, None
        ends = torch.as_tensor(counts, dtype=torch.int64).cumsum(0) - 1
        last = ends.to(inputs_embeds.device)
        return input_ids.long()[last], inputs_embeds[last], last

    def _runner_output(
        self,
        rows: torch.Tensor,
        last_positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """The runner-facing output: one row per scheduled token, each row's
        output at its last token position and the rejected-row output at
        every other position (TXG-TXN-010)."""
        if last_positions is None:
            return rows
        from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
            rejected_row_output,
        )

        out = rejected_row_output(
            int(inputs_embeds.shape[0]),
            int(rows.shape[-1]),
            park_id=int(self.config.eos_token_id),
            dtype=rows.dtype,
            device=rows.device,
        )
        out[last_positions] = rows[: int(last_positions.shape[0])]
        return out

    # @spec TXG-FAIL-008
    @staticmethod
    def _record_planning(projection: Any, context: Any) -> tuple[str, ...]:
        """Record the step's planning-failed requests (rows bound non-CHUNK
        with their planned-rejection bit set, TXG-COV-010) and each row's
        admission generation in the model state's step record, for
        ``postprocess_state``. Returns the planning-failed request IDs."""
        bits = context.planned_rejection.tolist()
        failed: tuple[str, ...] = ()
        if bits:
            failed = tuple(
                request_id
                for request_id, bit in zip(context.request_ids, bits, strict=True)
                if bit
            )
        record = getattr(projection, "planning_failed", None)
        if record is not None:
            record.update(failed)
        generations = getattr(projection, "admission_generations", None)
        if generations is not None:
            generations.update(
                (binding.request_id, int(binding.admission_generation))
                for binding in context.bindings
            )
        return failed

    # @spec TXG-FAIL-009, TXG-FAIL-010, TXG-COV-013
    def _all_failed_bypass(self, context: Any, inputs_embeds: torch.Tensor) -> Any:
        """A real step whose every row failed at planning: no transaction runs
        (neither the binding's ``execute`` nor the option-off transaction),
        nothing is reserved or staged, and each scheduled row gets the
        rejected-row output in the runner's order. ``postprocess_state`` fails
        the requests from the planning-failed set (TXG-FAIL-008).

        The forward refuses a real step on a latched worker before it plans
        one (TXG-FAIL-003, TXG-FAIL-010), so a latched worker never gets
        here. Where transaction graphs are on, the binding counts the step as
        a bypass, not a fallback (TXG-COV-013)."""
        from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
            rejected_row_output,
        )

        binding = getattr(self, "_transaction_graph_binding", None)
        if binding is not None:
            record_bypass = getattr(binding, "record_all_failed_step", None)
            if record_bypass is not None:
                record_bypass(len(context.request_ids))
        return rejected_row_output(
            len(context.request_ids),
            int(self.config.hidden_size),
            park_id=int(self.config.eos_token_id),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    @staticmethod
    def _write_file_envelopes(inputs_embeds: torch.Tensor, projection: Any) -> None:
        """Write each advancing FILE row's chunk envelope (the stream unit's
        header and samples) into its carrier row, so the row runs the
        streaming transition (FILE-ROW-003, FILE-ROW-004)."""
        for index, request_id in enumerate(projection.req_ids):
            row = projection.file_rows.get(request_id)
            if row is None or row.header is None:
                continue
            carrier = inputs_embeds[index]
            carrier.zero_()
            carrier[: len(row.header)] = torch.tensor(
                row.header, dtype=carrier.dtype, device=carrier.device
            )
            if row.samples is not None:
                count = int(row.samples.shape[0])
                start = len(row.header)
                carrier[start : start + count] = row.samples.to(carrier.dtype)

    def consume_batch_stats(self) -> list[tuple[str, int]] | None:
        from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
            consume_batch_stats,
        )

        return consume_batch_stats()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        del sampling_metadata
        from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
            forced_logits_rows,
            read_decision_carrier,
        )

        ids = read_decision_carrier(hidden_states)
        return forced_logits_rows(ids, num_logits=self.num_logits)

    @classmethod
    def get_model_state_cls(cls) -> type[Any]:
        from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
            NemotronASRModelState,
        )

        return NemotronASRModelState


# @spec PORT-DEC-008, PORT-PERF-004
def build_decode_resolver(
    hf_config: Any,
    *,
    graph_binding: Any = None,
) -> DecodeResolver:
    """Startup decode-dispatch resolution (PORT-DEC-008: no default).

    The SERVED config must declare exactly one of:

    - ``decode_dispatch_arm``: an explicitly configured eager arm for
      the bring-up/correctness lane. More than one ready decode bucket
      still forces the sync-free eager arm — a declared arm is a
      preference, never a license to serialize a busy engine. A
      ``dense-graphed`` declaration requires a startup-captured regional
      graph binding; the outer model runner remains eager.
    - ``decode_dispatch_table``: a measured, validated table artifact.
      Loading it requires the runtime-fingerprint tooling owned by the
      deferred A100 work packet; until that lands this path fails
      closed rather than validating against a fabricated fingerprint.

    Raises:
        ValueError: neither or both declared, an unknown arm, or the
            not-yet-supported table path.

    """
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import (
        decode_compact_active_frames,
        decode_dense_masked_frames,
    )

    arm = getattr(hf_config, "decode_dispatch_arm", None)
    table_path = getattr(hf_config, "decode_dispatch_table", None)
    if arm and table_path:
        raise ValueError(
            "declare decode_dispatch_arm OR decode_dispatch_table, not both"
        )
    if not arm and not table_path:
        raise ValueError(
            "the served config must declare decode_dispatch_arm or "
            "decode_dispatch_table — decode dispatch is never a "
            "hardcoded default (PORT-DEC-008)"
        )
    if table_path:
        raise ValueError(
            "decode_dispatch_table loading requires the deferred A100 "
            "work packet's runtime-fingerprint tooling; declare "
            "decode_dispatch_arm for the bring-up lane"
        )
    arms = {
        "dense-eager": decode_dense_masked_frames,
        "compact-eager": decode_compact_active_frames,
    }
    if arm == "dense-graphed" and graph_binding is None:
        raise ValueError(
            "decode_dispatch_arm='dense-graphed' requires a captured graph binding"
        )
    known_arms = {*arms, "dense-graphed"}
    if arm not in known_arms:
        raise ValueError(
            f"unknown decode_dispatch_arm {arm!r} (known: {sorted(known_arms)})"
        )

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        if arm == "dense-graphed":
            if request.memory_profile:
                if request.graph_covers_decode:
                    raise ValueError(
                        "pre-capture memory profile cannot claim graph coverage"
                    )
                return ResolvedDecode(
                    arm="dense-eager",
                    decode_fn=arms["dense-eager"],
                    override_reason="pre-capture-memory-profile",
                )
            if not request.graph_covers_decode:
                raise ValueError(
                    "dense-graphed dispatch requires regional graph coverage"
                )
            tier = graph_binding.execution_tier(request.execution_batch_size)
            if request.execution_tier_limit > 0 and tier > request.execution_tier_limit:
                raise ValueError(
                    "decode bucket tier exceeds RowPlan execution authority"
                )
            return ResolvedDecode(
                arm="dense-graphed",
                decode_fn=graph_binding.decode_fn(
                    geometry=request.geometry,
                    tier=tier,
                ),
                override_reason="regional-graph-coverage",
            )
        if request.graph_covers_decode:
            raise ValueError(
                "graph-covered decode requires the dense-graphed served arm"
            )
        if request.ready_decode_buckets > 1 and arm not in SYNC_FREE_ARMS:
            return ResolvedDecode(
                arm="dense-eager",
                decode_fn=arms["dense-eager"],
                override_reason="multi-bucket-serialization-guard",
            )
        return ResolvedDecode(arm=arm, decode_fn=arms[arm])

    return resolve
