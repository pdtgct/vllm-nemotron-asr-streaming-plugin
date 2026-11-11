# SPDX-License-Identifier: Apache-2.0
"""Complete-file transcription: the prompt of a file request.

Core sends a file as one ordinary request. ``build_file_prompt`` refuses what
the model does not serve (including beam search and client completion limits), binds the request's language to a
row of the served locale table and returns a prompt of one file-carrier token
with the file's samples as its audio item. The model imposes no limit on a
file's length. Importing this module imports no vLLM.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from vllm_nemotron_asr_streaming.locales import resolve_file_locale
from vllm_nemotron_asr_streaming.lookahead import geometry_index


# @spec FILE-IF-004, FILE-IF-005, FILE-LANG-001, FILE-LANG-002, FILE-LANG-003
# @spec FILE-LANG-004, FILE-LANG-005, FILE-LANG-006, FILE-PROMPT-001
# @spec FILE-PROMPT-002, FILE-PROMPT-005, FILE-GRD-001, FILE-GRD-002
# @spec FILE-GRD-003, FILE-GRD-004, FILE-GRD-005, FILE-GRD-006, REG-CARD-015
def build_file_prompt(stt_params: Any) -> dict[str, Any]:
    """Build the prompt of a complete-file request, or raise ``ValueError``."""
    if stt_params.task_type != "transcribe":
        raise ValueError(
            f"unsupported task {stt_params.task_type!r}: complete-file "
            "transcription serves only the transcribe task"
        )
    if stt_params.request_prompt:
        raise ValueError("a text prompt is not supported for file transcription")
    if stt_params.hotwords:
        raise ValueError("hotwords are not supported for file transcription")
    if stt_params.to_language:
        raise ValueError(
            "a target language (to_language) is not supported for file transcription"
        )
    model_config = stt_params.model_config
    hf_config = model_config.hf_config
    if getattr(hf_config, "experimental_native_burst", False):
        raise ValueError(
            "complete-file transcription is not served in native-burst mode: "
            "a native-burst row cannot run the one-token-per-step file cycle"
        )
    # The two guard fields come from a core change that may not be present; an
    # absent attribute and ``None`` both mean "not given" (FILE-GRD-005).
    if getattr(stt_params, "use_beam_search", False):
        raise ValueError(
            "use_beam_search is not supported for this model"
        )
    if getattr(stt_params, "max_completion_tokens", None) is not None:
        raise ValueError(
            "max_completion_tokens is not supported for this model"
        )

    extra_args = stt_params.extra_args or {}
    exact = extra_args.get("locale")
    if exact is not None and not isinstance(exact, str):
        raise ValueError(f"the locale argument must be a string, got {exact!r}")
    table = hf_config.prompt_dictionary
    locale = resolve_file_locale(
        table,
        language=stt_params.language,
        exact_locale=exact,
        language_locales=getattr(hf_config, "language_locales", None),
    )
    row = int(table[locale])

    right_context = int(hf_config.num_lookahead_tokens)
    geometry = geometry_index(right_context)
    samples = np.ascontiguousarray(stt_params.audio, dtype=np.float32).reshape(-1)
    count = int(samples.shape[0])

    return {
        "prompt_token_ids": [int(hf_config.file_carrier_token_id)],
        "multi_modal_data": {"audio": samples},
        "mm_processor_kwargs": {"file_header": [geometry, row, count]},
        "multi_modal_uuids": {"audio": [f"file-{uuid.uuid4().hex}"]},
    }
