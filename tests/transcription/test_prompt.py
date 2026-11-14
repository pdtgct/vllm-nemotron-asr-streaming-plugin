# SPDX-License-Identifier: Apache-2.0
"""FILE-PROMPT, FILE-GRD-006, FILE-ROW-008: the prompt a file request becomes."""

from __future__ import annotations

import numpy as np
import pytest
from support.card import BLANK, LOCALES

GEOMETRIES = ((0, 0), (1, 1), (3, 2), (6, 3), (13, 4))  # (right context, index)
RIGHT_CONTEXTS = [rc for rc, _ in GEOMETRIES]


def chunk_samples(rc: int) -> int:
    return 1280 * (rc + 1)


def queue_bound(rc: int) -> int:
    """Q = (r + 1) * 10 + 1, the design's per-chunk emission bound."""
    return (rc + 1) * 10 + 1


# ---------------------------------------------------------------------------
# FILE-PROMPT


# @spec FILE-PROMPT-001
@pytest.mark.engine
@pytest.mark.parametrize("n_samples", [0, 1, 20_000])
def test_prompt_is_one_file_carrier_token_and_one_audio_item(
    model_cls, stt_params, prompt_fields, n_samples
):
    samples = np.linspace(-1, 1, n_samples, dtype=np.float32)
    prompt = model_cls.get_generation_prompt(stt_params(samples=samples))
    assert prompt["prompt_token_ids"] == [BLANK + 4]
    got, header = prompt_fields(prompt)  # also checks the prompt's shape
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, samples)
    assert len(header) == 3
    assert set(prompt["multi_modal_data"]) == {"audio"}


# @spec FILE-PROMPT-002
@pytest.mark.engine
@pytest.mark.parametrize("right_context, geometry", GEOMETRIES)
def test_header_is_geometry_locale_row_and_sample_count(
    model_cls, stt_params, prompt_fields, right_context, geometry
):
    prompt = model_cls.get_generation_prompt(
        stt_params(
            n_samples=12_345,
            right_context=right_context,
            language="en",
            extra_args={"locale": "en-GB"},
        )
    )
    _, header = prompt_fields(prompt)
    assert header == [geometry, LOCALES["en-GB"], 12_345]


# @spec FILE-PROMPT-003
@pytest.mark.engine
def test_processor_declares_both_fields_keep_on_cpu():
    from transformers import BatchFeature

    from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
        NemotronASRMultiModalProcessor,
    )

    # _get_mm_fields_config reads no instance state: a real instance made
    # without a processing context is enough.
    processor = object.__new__(NemotronASRMultiModalProcessor)
    inputs = BatchFeature(
        {
            "samples": [np.zeros(8, np.float32)],
            "file_header": [np.zeros(3, np.int64)],
        }
    )
    # vllm/multimodal/inputs.py:432 (keep_on_cpu on the field)
    fields = processor._get_mm_fields_config(inputs, {})
    for name in ("samples", "file_header"):
        assert fields[name].field.keep_on_cpu is True


# @spec FILE-PROMPT-004
@pytest.mark.engine
@pytest.mark.parametrize("n_samples", [8, 100_000])
def test_embed_multimodal_file_item_is_a_placeholder(
    model_cls, served_config, n_samples
):
    import torch

    class Unread:
        """Samples that fail the test if the model touches them."""

        def __getattr__(self, name):
            raise AssertionError(f"embed_multimodal read the samples ({name})")

    # the real model class without weights: only a placeholder parameter (the
    # device probe) and the configuration are set
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = served_config()
    model.register_parameter("probe", torch.nn.Parameter(torch.zeros(1)))
    samples = Unread() if n_samples == 8 else torch.zeros(n_samples)
    header = torch.tensor([3, 0, n_samples], dtype=torch.int64)
    # n_samples=100_000 is wider than the stream carrier row (17,927): the
    # carrier-width check must not apply to file items.
    out = model.embed_multimodal(samples=[samples], file_header=[header])
    rows = out if isinstance(out, (list, tuple)) else [out]
    total = sum(int(r.reshape(-1, r.shape[-1]).shape[0]) for r in rows)
    assert total == 1


# @spec FILE-PROMPT-005
@pytest.mark.engine
def test_identical_samples_hash_differently(model_cls, stt_params):
    from vllm.multimodal.parse import MultiModalDataParser
    from vllm.multimodal.processing.inputs import ProcessorInputs

    samples = np.random.default_rng(1).standard_normal(5000).astype(np.float32)
    prompts = [
        model_cls.get_generation_prompt(stt_params(samples=samples)) for _ in range(2)
    ]
    uuids = [p["multi_modal_uuids"]["audio"] for p in prompts]
    assert all(len(u) == 1 and isinstance(u[0], str) and u[0] for u in uuids)
    assert uuids[0] != uuids[1]

    # The narrowest real seam: core's per-item hash (the multimodal cache key)
    # vllm/multimodal/processing/inputs.py:34
    hashes = []
    for prompt in prompts:
        items = MultiModalDataParser(target_sr=16000).parse_mm_data(
            prompt["multi_modal_data"]
        )
        inputs = ProcessorInputs(
            prompt=prompt["prompt_token_ids"],
            mm_data_items=items,
            mm_uuid_items=prompt["multi_modal_uuids"],
            hf_processor_mm_kwargs=prompt["mm_processor_kwargs"],
        )
        hashes.append(inputs.get_mm_hashes("model", "blake3")["audio"])
    assert hashes[0] != hashes[1]
    # control: the UUID is what differs (same samples and header)
    same = ProcessorInputs(
        prompt=prompts[0]["prompt_token_ids"],
        mm_data_items=MultiModalDataParser(target_sr=16000).parse_mm_data(
            prompts[0]["multi_modal_data"]
        ),
        mm_uuid_items=prompts[0]["multi_modal_uuids"],
        hf_processor_mm_kwargs=prompts[0]["mm_processor_kwargs"],
    )
    assert same.get_mm_hashes("model", "blake3")["audio"] == hashes[0]


# ---------------------------------------------------------------------------
# FILE-GRD-006: no refusal on a file's length (FILE-BUD-001 and 003 are retired)


# @spec FILE-GRD-006
@pytest.mark.engine
@pytest.mark.parametrize("right_context", [0, 6, 13])
def test_no_file_is_refused_for_its_length(model_cls, stt_params, right_context):
    # the retired worst-case check refused this file at max_model_len = 100:
    # its worst case is far above 100 tokens. The model now imposes no
    # input-length limit; core's actual-token stop is the only bound.
    s = chunk_samples(right_context)
    for n in (0, 1, s, 40 * s + 17, 3_600 * 16_000):
        prompt = model_cls.get_generation_prompt(
            stt_params(n_samples=n, right_context=right_context, max_model_len=100)
        )
        assert len(prompt["prompt_token_ids"]) == 1
        assert prompt["mm_processor_kwargs"]["file_header"][2] == n


# @spec FILE-GRD-006
def test_the_worst_case_budget_check_is_gone():
    import vllm_nemotron_asr_streaming.transcription as transcription

    assert not hasattr(transcription, "file_token_budget")


# ---------------------------------------------------------------------------
# FILE-ROW-008


# @spec FILE-ROW-008
@pytest.mark.parametrize("right_context", RIGHT_CONTEXTS)
def test_queue_bound_fits_the_label_queue(right_context):
    from vllm_nemotron_asr_streaming.nemotron_asr import manifests

    capacity = manifests.SESSION_LIMITS["queue_capacity"]
    assert capacity == 141
    assert queue_bound(right_context) <= capacity
    assert right_context in {right for _, right in manifests.CADENCES.values()}
    # the largest geometry attains the capacity exactly
    assert max(queue_bound(rc) for rc in RIGHT_CONTEXTS) == 141
