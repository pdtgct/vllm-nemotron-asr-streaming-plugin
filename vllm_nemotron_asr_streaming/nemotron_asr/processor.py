# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The multimodal processor for the streaming ASR model (BU-c2, D-BUc-2).

This model has **no HF processor**: the mel front-end is a torch module
inside the model (``embed_multimodal`` runs the featurizer +
``pack_audio_carrier``), so the processor's only job is to hand the raw
audio chunk through as the ``audio`` mm-kwarg and register the single
carrier placeholder. One realtime chunk = one prompt = one placeholder
token (``audio_chunk_token_id``) = one carrier row (BU-c1 measured
width). Modeled on vLLM's realtime precedent (manual single
placeholder, no placeholder tokens minted into text) and the ASR
field-config precedent (``MultiModalFieldConfig.batched("audio")``).

The native processor separates multimodal field processing from prompt
updates. Raw audio bypasses HF processing; dummy inputs receive their
carrier token during the prompt-update stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    MultiModalProcessingResult,
    PlaceholderFeaturesInfo,
)

#: One second of 16 kHz audio — a safe dummy chunk for warmup profiling
#: (the featurizer only needs a valid non-empty waveform; the realtime
#: route never serves this).
_DUMMY_SAMPLES = 16_000

#: The ``mm_processor_kwargs`` key (and the multimodal field) that carries a
#: complete file's header: geometry index, locale row and sample count.
FILE_HEADER_KEY = "file_header"


def validate_file_header(
    header: object, sample_counts: Sequence[int], num_prompts: int | None = None
) -> np.ndarray:
    """Check a file header at input processing, so a malformed request is a
    ``ValueError`` (a 400) and never reaches the worker step.

    The header is ``(geometry index, locale row, sample count)``.
    """
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import CADENCES

    try:
        values = np.asarray(header, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"file_header must be integers, got {header!r}") from exc
    if values.shape != (3,):
        raise ValueError(f"file_header must have 3 values, got shape {values.shape}")
    geometry, row, count = (int(v) for v in values)
    if not 0 <= geometry < len(CADENCES):
        raise ValueError(
            f"file_header geometry {geometry} is not in [0, {len(CADENCES)})"
        )
    if row < 0 or (num_prompts is not None and row >= num_prompts):
        raise ValueError(
            f"file_header locale row {row} is outside the locale table "
            f"(0 <= row < {num_prompts})"
        )
    if len(sample_counts) != 1 or count != int(sample_counts[0]):
        raise ValueError(
            f"file_header sample count {count} disagrees with the audio item "
            f"({list(sample_counts)} samples)"
        )
    return values


class NemotronASRProcessingInfo(BaseProcessingInfo):
    """Processing metadata: one audio item per streaming prompt."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}


class NemotronASRDummyInputsBuilder(BaseDummyInputsBuilder[NemotronASRProcessingInfo]):
    """Warmup inputs: one placeholder-free text + one dummy chunk.

    The carrier placeholder id is a minted special with no text form, so
    the dummy text is empty; the placeholder is registered structurally
    by the processor, not tokenized from text.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(
                length=_DUMMY_SAMPLES, num_audios=num_audios
            )
        }


class NemotronASRMultiModalProcessor(
    BaseMultiModalProcessor[NemotronASRProcessingInfo]
):
    """Raw-audio passthrough + single carrier placeholder.

    A stream unit carries its audio envelope as the ``audio`` field; a
    complete file carries ``samples`` and ``file_header`` instead
    (FILE-PROMPT-003).

    No HF processor: ``_apply_hf_processor_main`` builds the field batch from
    the raw chunk directly (the model's ``embed_multimodal`` owns the
    mel front-end). The realtime prompt carries exactly one placeholder
    token, so placeholder positions are built manually (one carrier row
    per audio) rather than by text-target replacement.
    """

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingResult:
        # Realtime cannot use the content-hash cache (streams are unique).
        return self._apply_hf_processor(inputs, timing_ctx)

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # No HF processor or featurization: the model owns the mel frontend.
        mm_data, _, passthrough_data = self._get_hf_mm_inputs(mm_items, hf_kwargs)
        # ``_get_hf_mm_inputs`` remaps the item key ``audios`` to ``audio``,
        # which is also the output field ``embed_multimodal`` reads.
        audios = mm_data.get("audio", [])
        if not isinstance(audios, list):
            audios = [audios]
        # torch tensors, not np arrays: the engine serializes the mm field
        # via _encode_nested_tensors, which only handles torch tensors
        # (it recurses into an ndarray and chokes on the scalar leaves).
        arrays = [torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in audios]
        header = hf_kwargs.get(FILE_HEADER_KEY)
        if header is not None:
            # A complete-file prompt: the samples and the file header travel
            # as two host-resident fields; the model state takes them into a
            # request-owned buffer at the request's first schedule.
            values = torch.as_tensor(
                validate_file_header(
                    header,
                    [a.shape[0] for a in arrays],
                    int(self.info.get_hf_config().num_prompts),
                ),
                dtype=torch.int64,
            )
            return BatchFeature(
                {
                    **passthrough_data,
                    "samples": arrays,
                    FILE_HEADER_KEY: [values for _ in arrays],
                }
            )
        return BatchFeature({**passthrough_data, "audio": arrays})

    # @spec FILE-PROMPT-003
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # A stream unit: one batched field named ``audio`` — the key
        # ``embed_multimodal`` reads (``kwargs.get("audio")``). A complete
        # file: ``samples`` and ``file_header``, both kept on the host so
        # core's batching never moves the file to the device.
        fields: dict[str, MultiModalFieldConfig] = {}
        if "audio" in hf_inputs:
            fields["audio"] = MultiModalFieldConfig.batched("audio")
        for name in ("samples", FILE_HEADER_KEY):
            if name in hf_inputs:
                fields[name] = MultiModalFieldConfig.batched("audio", keep_on_cpu=True)
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # The realtime prompt already holds the single placeholder token
        # at the right position; placeholders are built manually in
        # ``_maybe_apply_prompt_updates``, so no text-target replacement.
        return []

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        mm_res: MultiModalProcessingResult,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        prompt_ids = mm_res.prompt_ids
        # One carrier row per chunk: the whole prompt is the single
        # placeholder token, so the placeholder spans one position at
        # index 0. ``tokens`` is only used for length accounting.
        if mm_items.get_all_counts().get("audio", 0) == 0:
            return prompt_ids, {}
        if not prompt_ids:
            # The dummy builder supplies empty text: its minted carrier has
            # no tokenizer text form. Serving already supplies the real ID.
            prompt_ids = [self.info.get_hf_config().audio_chunk_token_id]
        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=0,
            tokens=[prompt_ids[0]],
            is_embed=None,
        )
        return prompt_ids, {"audio": [features_info]}
