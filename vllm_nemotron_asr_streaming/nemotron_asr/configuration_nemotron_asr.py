# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for the streaming ASR model (PORT-WGT-004).

The served checkpoint has no upstream HF modeling code. The card parser
(``vllm_nemotron_asr_streaming.card_parser``, selected with
``--config-format nemotron_asr_streaming``) translates the published
card into this class's fields, and the conversion publisher
(``convert.author_config``) authors the same schema. Only the fields
vLLM's config machinery and this model's ``__init__`` actually read live
here; the values are produced by one of those two, never hand-authored.
"""

import json
from pathlib import Path

from transformers import PreTrainedConfig

from vllm_nemotron_asr_streaming.nemotron_asr.identity import ARCHITECTURE
from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
)

MODEL_TYPE = "nemotron_asr"

#: The raw chunk-envelope carrier width: header slots plus the largest
#: admitted raw cadence (1120 ms at 16 kHz) — DERIVED from the
#: manifests, never a copied constant (PORT-INT-004; the retired mel
#: default 15,617 is superseded by this envelope value, 17,926).
_CARRIER_WIDTH = len(ENVELOPE_HEADER_FIELDS) + max(RAW_SAMPLES_PER_CHUNK.values())


def validate_prompt_dictionary(
    prompt_dictionary: object,
    num_prompts: object,
) -> dict[str, int]:
    """Validate the published locale-to-conditioning-row authority.

    Config construction remains permissive so Transformers can create an
    empty default config for registry/introspection work. The serving model
    calls this guard before allocating model state; a served artifact may
    never start without a complete, in-range prompt binding.

    Args:
        prompt_dictionary: Candidate locale-to-row mapping.
        num_prompts: Number of conditioning rows in the checkpoint.

    Returns:
        A defensive copy with normalized static types.

    Raises:
        ValueError: If the row count or any mapping entry is invalid.

    """
    if (
        isinstance(num_prompts, bool)
        or not isinstance(num_prompts, int)
        or num_prompts <= 0
    ):
        raise ValueError("num_prompts must be a positive integer")
    if not isinstance(prompt_dictionary, dict) or not prompt_dictionary:
        raise ValueError("prompt_dictionary must be a non-empty object")
    validated: dict[str, int] = {}
    for locale, index in prompt_dictionary.items():
        if (
            not isinstance(locale, str)
            or not locale
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_prompts
        ):
            raise ValueError(
                "prompt_dictionary must map non-empty locale strings to integer rows "
                f"in [0, {num_prompts})"
            )
        validated[locale] = index
    return validated


class NemotronASRConfig(PreTrainedConfig):
    """Config for the FastConformer RNN-T streaming ASR model.

    ``hidden_size`` is the multimodal-carrier row width (D-BU-2), not
    an LM width — this model has no LM stack; the encoder's true width
    is ``d_model``. ``vocab_size`` (also the model's logit width) covers
    the checkpoint's label set, the blank, and the five minted control
    tokens at ``blank + 1`` to ``blank + 5``: park (``eos_token_id``),
    stream chunk carrier (``audio_chunk_token_id``), end of utterance
    (``eou_token_id``), file carrier (``file_carrier_token_id``) and
    continuation (``continuation_token_id``).
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        vocab_size: int = 13093,
        num_asr_labels: int = 13087,
        hidden_size: int = _CARRIER_WIDTH,
        eos_token_id: int | None = None,
        audio_chunk_token_id: int | None = None,
        eou_token_id: int | None = None,
        file_carrier_token_id: int | None = None,
        continuation_token_id: int | None = None,
        card_num_lookahead_tokens: list[int] | None = None,
        endpoint_history_capacity_frames: int = 12,
        encoder_execution_arm: str | None = None,
        decode_dispatch_arm: str | None = None,
        decode_dispatch_table: str | None = None,
        performance_gated: bool = False,
        prompt_dictionary: dict[str, int] | None = None,
        d_model: int = 1024,
        n_layers: int = 24,
        conv_kernel: int = 9,
        att_context_left: int = 56,
        att_context_right: int = 13,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        joint_hidden: int = 640,
        num_prompts: int = 128,
        n_mels: int = 128,
        max_position_embeddings: int = 32768,
        torch_dtype: str = "float32",
        **kwargs: object,
    ) -> None:
        self.vocab_size = vocab_size
        #: The decoded label-set size V (blank = V, joint emits V+1);
        #: distinct from ``vocab_size``, the engine logit width that
        #: also covers the minted park/placeholder specials. NOT named
        #: ``num_labels`` — that is a reserved ``PreTrainedConfig``
        #: field (it drives ``id2label`` and is reset to a 2-label
        #: default by ``super().__init__``, silently clobbering ours).
        self.num_asr_labels = num_asr_labels
        self.hidden_size = hidden_size
        self.audio_chunk_token_id = audio_chunk_token_id
        self.eou_token_id = eou_token_id
        self.file_carrier_token_id = file_carrier_token_id
        self.continuation_token_id = continuation_token_id
        #: The right contexts the published card declares. The served set
        #: and the default lookahead are launch options, resolved in the
        #: model-configuration hook, so no default is stored here.
        self.card_num_lookahead_tokens = (
            None
            if card_num_lookahead_tokens is None
            else list(card_num_lookahead_tokens)
        )
        self.endpoint_history_capacity_frames = endpoint_history_capacity_frames
        controls = (
            eos_token_id,
            audio_chunk_token_id,
            eou_token_id,
            file_carrier_token_id,
            continuation_token_id,
        )
        if any(value is not None for value in controls):
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in controls
            ):
                raise ValueError("all five control token ids must be integers")
            typed_controls = tuple(
                int(value) for value in controls if value is not None
            )
            if len(set(typed_controls)) != len(typed_controls):
                raise ValueError("control token ids must be pairwise distinct")
            if min(typed_controls) <= num_asr_labels:
                raise ValueError(
                    "control token ids must be outside the label and blank space"
                )
            if vocab_size <= max(typed_controls):
                raise ValueError("vocab_size must cover every control token id")
        if endpoint_history_capacity_frames <= 0:
            raise ValueError("endpoint history capacity must be positive")
        # The encoder execution gate is an analysis-only startup choice.
        # Absence retains the compatibility eager baseline; an experiment
        # stamps the resolved value into its evidence fingerprint.
        self.encoder_execution_arm = encoder_execution_arm
        # Startup decode policy is part of the served artifact, not a
        # process-local default. Preserve all three fields through HF
        # serialization so build_decode_resolver sees the declaration
        # authored by the publisher (PORT-WGT-004 / PORT-DEC-008).
        self.decode_dispatch_arm = decode_dispatch_arm
        self.decode_dispatch_table = decode_dispatch_table
        self.performance_gated = performance_gated
        # Complete locale -> prompt-row authority, authored from the
        # source checkpoint metadata. Serving must never rely on a
        # sidecar path or reconstruct this binding from labels.
        self.prompt_dictionary = dict(prompt_dictionary or {})
        self.d_model = d_model
        self.n_layers = n_layers
        self.conv_kernel = conv_kernel
        self.att_context_left = att_context_left
        self.att_context_right = att_context_right
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.joint_hidden = joint_hidden
        self.num_prompts = num_prompts
        self.n_mels = n_mels
        self.max_position_embeddings = max_position_embeddings
        # ``architectures`` rides kwargs, never an explicit arg: a
        # config.json (from author_config or to_dict) already carries
        # it, and ``from_dict`` re-passes every key — an explicit
        # ``architectures=`` would then collide with **kwargs and raise
        # on every reload.
        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(
            eos_token_id=eos_token_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )


def ensure_prompt_dictionary(config: PreTrainedConfig, model_path: object) -> None:
    """Fill ``prompt_dictionary`` from the card's processor sidecar.

    The public card ships the complete locale-to-row authority in
    ``processor_config.json``, not ``config.json``. Idempotent: a config
    that already carries a non-empty mapping (the served artifact) is
    left untouched, and an absent sidecar leaves the config unchanged —
    the model's existing ``validate_prompt_dictionary`` gate then fails
    admission with its own error.
    """
    if getattr(config, "prompt_dictionary", None):
        return
    sidecar = None
    local = Path(str(model_path)) / "processor_config.json"
    if local.is_file():
        sidecar = json.loads(local.read_text())
    else:
        try:
            from transformers.utils import cached_file

            resolved = cached_file(
                str(model_path),
                "processor_config.json",
                _raise_exceptions_for_missing_entries=False,
            )
            if resolved:
                sidecar = json.loads(Path(resolved).read_text())
        except Exception:
            sidecar = None
    if not sidecar:
        return
    mapping = sidecar.get("prompt_dictionary")
    if mapping:
        config.prompt_dictionary = validate_prompt_dictionary(
            mapping,
            getattr(config, "num_prompts", None),
        )
