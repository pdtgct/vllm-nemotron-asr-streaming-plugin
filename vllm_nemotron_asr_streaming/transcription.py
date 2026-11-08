# SPDX-License-Identifier: Apache-2.0
"""Complete-file transcription: the prompt and token budget of a file request.

Core sends a file as one ordinary request. ``build_file_prompt`` refuses what
the model does not serve, binds the request's language to a row of the served
locale table, checks the worst-case token budget and returns a prompt of one
file-carrier token with the file's samples as its audio item. Importing this
module imports no vLLM.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from vllm_nemotron_asr_streaming.locales import resolve_file_locale
from vllm_nemotron_asr_streaming.lookahead import cadence_label, geometry_index
from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
    RAW_SAMPLES_PER_CHUNK,
    SESSION_LIMITS,
)


def file_chunk_samples(right_context: int) -> int:
    """Samples per regular chunk of a geometry (``s``), from the manifest."""
    return RAW_SAMPLES_PER_CHUNK[cadence_label(right_context)]


def file_queue_bound(right_context: int) -> int:
    """The most tokens one chunk or tail queues (``Q``): labels, at most
    ``max_symbols_per_step`` per encoder frame, plus an optional EOU."""
    return (right_context + 1) * SESSION_LIMITS["max_symbols_per_step"] + 1


# @spec FILE-BUD-001
def file_token_budget(num_samples: int, right_context: int) -> int:
    """The worst-case tokens a file request emits, counting the prompt token.

    ``B = 1 + (C + 1) * Q + 1``: the prompt token, each of the ``C`` regular
    chunks and the tail at their bound ``Q``, and the end token.
    """
    chunks = int(num_samples) // file_chunk_samples(right_context)
    return 1 + (chunks + 1) * file_queue_bound(right_context) + 1


# @spec FILE-IF-004, FILE-IF-005, FILE-LANG-001, FILE-LANG-002, FILE-LANG-003
# @spec FILE-LANG-004, FILE-LANG-005, FILE-LANG-006, FILE-PROMPT-001
# @spec FILE-PROMPT-002, FILE-PROMPT-005, FILE-BUD-001, REG-CARD-015
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

    budget = file_token_budget(count, right_context)
    max_model_len = int(model_config.max_model_len)
    if budget > max_model_len:
        raise ValueError(
            f"the file is {count / 16000:.1f} s long and needs max_model_len "
            f">= {budget} tokens at the {right_context}-frame right context, "
            f"but max_model_len is {max_model_len}"
        )

    return {
        "prompt_token_ids": [int(hf_config.file_carrier_token_id)],
        "multi_modal_data": {"audio": samples},
        "mm_processor_kwargs": {"file_header": [geometry, row, count]},
        "multi_modal_uuids": {"audio": [f"file-{uuid.uuid4().hex}"]},
    }
