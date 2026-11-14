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

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsTranscription
from vllm.model_executor.layers.state import PersistentStateLayerBase
from vllm_nemotron_asr_streaming.nemotron_asr.advance import (
    ENVELOPE_HEADER_SLOTS,
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
    resolve_status_reports,
)
from vllm_nemotron_asr_streaming.nemotron_asr.convert import (
    LID_REQUIRED_PATTERN,
)
from vllm_nemotron_asr_streaming.nemotron_asr.decode_dispatch import (
    SYNC_FREE_ARMS,
)
from vllm_nemotron_asr_streaming.nemotron_asr import decode_graph
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
    file_envelope_header,
    PlanContextSlot,
    SessionRegistry,
    prepare_plan_context,
    reject_unsupported_outer_graph_mode,
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
from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import build_nemotron_persistent_state_spec
from vllm.multimodal import MULTIMODAL_REGISTRY

logger = init_logger(__name__)

#: The largest published chunk (1120 ms) emits 14 encoder frames — the
#: replay queue page's per-chunk worst case.
_MAX_FRAMES_PER_CHUNK = 14
#: The checkpoint's mel-bin count; the featurizer's filterbank is built
#: as a placeholder at this width and overwritten by load_weights.
_N_MELS = 128
_STFT_FREQ_BINS = 512 // 2 + 1
_WIN_LENGTH = 400


def _served_geometry_ids(hf_config: Any) -> tuple[int, ...]:
    """Geometry indices of the served lookahead set, the one definition the
    factory, the realtime hook and the model share."""
    from vllm_nemotron_asr_streaming.lookahead import (
        geometry_index,
        served_lookaheads,
    )

    geometries = tuple(geometry_index(right) for right in served_lookaheads(hf_config))
    if not geometries:
        raise ValueError("served configuration declares no supported encoder geometry")
    return geometries


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
class NemotronASRForRNNT(nn.Module, SupportsTranscription):
    """Engine-facing cache-aware RNN-T over one aggregate state page."""

    supports_multimodal = True
    requires_raw_input_tokens = True
    supports_realtime = True
    supports_transcription_only = False
    supports_segment_timestamp = False
    supports_diarized_transcription = False
    supports_persistent_state = True
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
        reject_unsupported_outer_graph_mode(
            getattr(vllm_config, "compilation_config", None)
        )
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
        self._state_spec = build_nemotron_persistent_state_spec(hf_config)
        self._persistent_state_layer = PersistentStateLayerBase(
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
        served_geometry_ids = _served_geometry_ids(hf_config)
        # @spec REG-014: the graph-runtime dependencies are needed only when
        # graphed execution is configured; name any that is missing now,
        # before weights load.
        self._graphed_execution = bool(
            getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed"
            or getattr(hf_config, "chunk_bucket_graph_populations", ())
            or getattr(hf_config, "encoder_execution_arm", None)
            in {"dense-graphed", "eager-graphed"}
        )
        if self._graphed_execution:
            decode_graph.platform_graph_runtime()
        self._encoder_execution = build_encoder_execution(
            self.core,
            hf_config,
            maximum_population=self._max_num_seqs,
            warmup_geometries=served_geometry_ids,
            vllm_config=vllm_config,
        )
        self._decode_graph_binding = None
        if getattr(hf_config, "decode_dispatch_arm", None) == "dense-graphed":
            from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import (
                DenseGraphBinding,
                execution_tiers,
            )

            served_geometry_set = set(served_geometry_ids)
            self._decode_graph_binding = DenseGraphBinding(
                batch_invariant_mode=self.core.batch_invariant_mode,
                decode_fn=decode_dense_masked_frames,
                predictor=self.core.predictor,
                joint=self.core.joint,
                vllm_config=vllm_config,
                frame_widths=tuple(
                    right + 1 if index in served_geometry_set else None
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
        self._execution_memory_prepared = False
        self._execution_memory_failed = False
        #: Set when core's model-state capture hook, not the profiling step,
        #: captures the decoder graphs (``prepare_execution_memory_hook``).
        self._graphs_by_core = False

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
            served_geometries=_served_geometry_ids(self.config),
        )
        self._plan_slot.stage(context)
        return context

    @staticmethod
    def _endpoint_controls(request: Any) -> tuple[int, int, int]:
        information = getattr(request, "additional_information", None)
        if not isinstance(information, dict):
            information = getattr(information, "entries", None)
        if not isinstance(information, dict):
            return 0, 0, 0
        policy = information.get("endpoint_policy")
        if not isinstance(policy, dict):
            return 0, 0, 0
        mode_name = policy.get("mode")
        if mode_name not in ("disabled", "greedy_blank"):
            raise ValueError("request carries an unknown endpoint mode")
        mode = 1 if mode_name == "greedy_blank" else 0
        threshold = int(policy.get("threshold_frames", 0))
        residue = int(policy.get("residue_frames", 0))
        if threshold < 0 or residue < 0:
            raise ValueError("request carries an invalid endpoint policy")
        return mode, threshold, residue

    def _stage_v2_projection(self, projection: Any) -> Any:
        input_batch = projection.input_batch
        metadata = projection.request_metadata
        if metadata is None:
            raise RuntimeError("MRv2 projection omitted request metadata")
        rows: list[ObservedRow] = []
        for index, (request_id, binding) in enumerate(
            zip(projection.req_ids, projection.bindings)
        ):
            if int(input_batch.num_scheduled_tokens[index]) != 1:
                raise ValueError("every streaming row must be single-token")
            request = metadata[request_id]
            computed = int(input_batch.num_computed_tokens_np[index])
            token = projection.token_ids[index]
            endpoint_mode, endpoint_threshold, endpoint_residue = (
                self._endpoint_controls(request)
            )
            file_row = projection.file_rows.get(request_id)
            if file_row is not None:
                # FILE-ROW-001: a FILE row's role and header come from the
                # step plan, not from the request's multimodal feature.
                plan = file_row.file_plan
                assert plan is not None
                if file_row.failed:
                    # A malformed file item fails this row alone: a CHUNK
                    # bound as non-CHUNK, which the transaction masks.
                    rows.append(
                        ObservedRow(
                            request_id=request_id,
                            block_id=int(binding.slot_id),
                            scheduled_token_id=token,
                            has_prior_state=not bool(binding.fresh),
                            envelope_header=file_envelope_header(
                                valid_samples=0,
                                geometry_id=0,
                                final_tail=False,
                                prompt_index=0,
                                chunk_sequence=0,
                                now_ms=int(time.time() * 1000),
                            ),
                            is_file=True,
                            file_advances=True,
                            reject=True,
                        )
                    )
                    continue
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
            rows.append(
                ObservedRow(
                    request_id=request_id,
                    block_id=int(binding.slot_id),
                    scheduled_token_id=token,
                    has_prior_state=not bool(binding.fresh),
                    envelope_header=resolve_row_envelope_header(
                        scheduled_token_id=token,
                        placeholder_id=int(self.config.audio_chunk_token_id),
                        num_computed_tokens=computed,
                        mm_features=request.mm_features,
                        scheduled_encoder_input_ids=projection.scheduled_encoder_inputs.get(
                            request_id, ()
                        ),
                    ),
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

        ``capture_pending_ok``: accept a decoder that core's capture hook has
        not captured yet (startup only; serving steps never pass it)."""
        if not self._encoder_execution.ready:
            raise RuntimeError("encoder execution inventory is incomplete")
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        if chunk_binding is not None and not chunk_binding.ready:
            raise RuntimeError("CHUNK execution inventory is incomplete")
        binding = self._decode_graph_binding
        if binding is None:
            if getattr(self.config, "decode_dispatch_arm", None) == "dense-graphed":
                raise RuntimeError("decoder execution inventory is missing")
            return
        if capture_pending_ok and self._graphs_by_core and not binding.captured_keys:
            return
        from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import execution_tiers

        expected = {
            (geometry, tier)
            for geometry in _served_geometry_ids(self.config)
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

    # @spec REG-MEM-002, REG-MEM-003, REG-MEM-004, REG-MEM-005, REG-MEM-008
    def _profile_dummy_step(self, device: torch.device) -> None:
        """The model's one preparation of its execution memory, run by the
        forward of a profiling dummy step (core's memory profile).

        The first such step of a worker prepares; a failure discards the
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
            self._require_execution_ready(capture_pending_ok=True)
            return
        try:
            self._prepare_execution_memory(device)
        except Exception:
            self._execution_memory_failed = True
            try:
                self._encoder_execution._discard()
            finally:
                if self._decode_graph_binding is not None:
                    self._decode_graph_binding.discard()
            raise
        self._execution_memory_prepared = True

    # @spec PORT-STATE-003, PORT-PERF-009, PORT-PERF-011
    @torch.inference_mode()
    def _prepare_execution_memory(
        self, device: torch.device, *, capture: bool = True
    ) -> None:
        """Retain execution inventory and profile its transient headroom.

        Profiling uses model-owned scratch state built from the model's
        ``PersistentStateSpec``; it reads no state-slot bindings or pool.
        ``capture=False`` (core's capture hook captures the decoder graphs
        later) skips the decoder capture; the profile transaction then runs
        the dense eager decoder, as it does today when no CHUNK graphs exist.
        """
        from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
            run_persistent_state_profile,
            warmup_static_encoder_execution,
        )

        warmup_static_encoder_execution(self, device=device)
        chunk_binding = getattr(self, "_chunk_bucket_binding", None)
        if not capture:
            if chunk_binding is not None:
                raise RuntimeError("CHUNK graphs are captured on the profiling step")
        elif self._decode_graph_binding is not None:
            self._decode_graph_binding.warmup(
                device, self.core.policy.dtype_for("activations")
            )
        if chunk_binding is not None:
            chunk_binding.warmup(device)
        self._require_execution_ready(capture_pending_ok=not capture)
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
            geometry_id=max(_served_geometry_ids(self.config)),
            ready_domain=True,
        )
        self._require_execution_ready(capture_pending_ok=not capture)

    # Core's model-state graph hooks (a core that defines them; the model
    # state forwards ``prepare_execution_memory``, ``capture_cudagraphs``,
    # ``clear_cudagraphs`` and ``shutdown`` here).
    def prepare_execution_memory_hook(
        self, device: torch.device, *, graphs_by_core: bool
    ) -> None:
        """Core's ``prepare_execution_memory``: once, before the profiling
        forward, inside core's memory profile. Captures nothing.

        With ``graphs_by_core`` (core's capture hook captures the decoder
        graphs), prepare everything but those graphs. Otherwise a
        configuration that captures nothing prepares in full here, and one
        that captures leaves preparation to the first profiling step, as on
        a core without the hooks. The profiling step never prepares again.
        """
        if self._execution_memory_prepared or self._execution_memory_failed:
            return
        if not graphs_by_core:
            if not self._graphed_execution:
                self._profile_dummy_step(device)
            return
        if (
            self._decode_graph_binding is None
            or getattr(self, "_chunk_bucket_binding", None) is not None
            or getattr(self._encoder_execution, "arm", None)
            in ("dense-graphed", "eager-graphed")
        ):
            raise RuntimeError(
                "core's capture hook captures only the dense-graphed decoder; "
                "graphed encoder and CHUNK arms are captured on the profiling step"
            )
        self._graphs_by_core = True
        try:
            self._prepare_execution_memory(device, capture=False)
        except Exception:
            self._execution_memory_failed = True
            try:
                self._encoder_execution._discard()
            finally:
                self._decode_graph_binding.discard()
            raise
        self._execution_memory_prepared = True

    def capture_graphs_hook(
        self,
        device: torch.device,
        *,
        graph_pool: Any,
        stream: Any,
        profile_only: bool,
    ) -> None:
        """Core's ``capture_cudagraphs``: capture the decoder graphs into
        ``graph_pool`` on ``stream``. The profile-only capture is the same
        capture (same keys and buffers), released by ``release_core_graphs``.
        """
        del profile_only  # the measured capture is the serving capture
        if not self._graphs_by_core:
            raise RuntimeError("core's capture hook was not selected at preparation")
        if self._execution_memory_failed or not self._execution_memory_prepared:
            raise RuntimeError("execution memory is not prepared on this worker")
        binding = self._decode_graph_binding
        assert binding is not None
        if binding.captured_keys:
            raise RuntimeError(
                "the decoder graphs are already captured; core releases the "
                "profiling capture before it captures again"
            )
        try:
            binding.warmup(
                device,
                self.core.policy.dtype_for("activations"),
                stream=stream,
                graph_pool=graph_pool,
            )
            self._require_execution_ready()
        except Exception:
            binding.discard()
            raise

    def release_core_graphs(self) -> None:
        """Core's ``clear_cudagraphs`` / ``shutdown``: drop the graphs and
        buffers core's capture hook created. Tolerates nothing captured."""
        binding = getattr(self, "_decode_graph_binding", None)
        if binding is not None and getattr(self, "_graphs_by_core", False):
            binding.discard()

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

    def collect_commit_status(self) -> tuple[dict[str, int], set[str]]:
        sink = self._commit_sink
        if sink is None or not sink.has_staged:
            return {}, set()
        reports, _records, lease_ok = sink.collect()
        statuses, failed = resolve_status_reports(
            self._registry, reports, lease_ok=lease_ok
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
        return statuses, failed

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
            if envelope.shape[0] < ENVELOPE_HEADER_SLOTS:
                raise ValueError("audio envelope is smaller than its header")
            if envelope.shape[0] > hidden:
                raise ValueError("audio envelope exceeds carrier width")
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
        context = self._stage_v2_projection(persistent_state_projection)
        state_batch = persistent_state_projection.state_batch
        if state_batch is None:
            raise RuntimeError("MRv2 projection omitted bounded state views")
        pools = self._state_pools(persistent_state_projection)
        self._write_file_envelopes(inputs_embeds, persistent_state_projection)
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
        native_handoff = self._native_burst_handoff
        if native_handoff is not None:
            if persistent_state_projection is None:
                raise RuntimeError("native burst requires a native runner projection")
            native_handoff.prepare(persistent_state_projection)
        try:
            result = advance_model_rows(
                self.core,
                input_ids.long(),
                inputs_embeds,
                plan,
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
            return result
        except BaseException:
            if native_handoff is not None:
                native_handoff.clear()
            raise

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
