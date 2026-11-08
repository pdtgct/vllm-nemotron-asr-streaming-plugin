# SPDX-License-Identifier: Apache-2.0
"""Whole-file requests through a real engine on the published card.

These tests need CUDA, the installed plugin and the published card
(``NEMOTRON_CARD_DIR``). The audio is a synthetic signal, or a 16 kHz mono
16-bit WAV file named by ``NEMOTRON_TEST_AUDIO`` (speech makes the
equivalence checks non-vacuous; the synthetic signal may transcribe to
nothing, which still has to be identical on both paths).

The in-process staging test (marker ``inprocess_engine``) builds an in-process
EngineCore, whose ``enable_envs_cache()`` freezes ``vllm.envs`` for the rest of
the session: in the GPU round run it in its own pytest invocation
(``-m inprocess_engine``) and the others with ``-m "not inprocess_engine"``.

The A40 round installs the plugin with ``pip install -e .`` so that the
engine-core process loads the ``vllm.general_plugins`` entry point; the
in-process fixture also calls ``register()`` (idempotent) for the front end.
Both the in-process ``AsyncLLM`` and the ENG-006 ``vllm serve`` subprocess are
alive at once, each at ``gpu_memory_utilization`` 0.3, so the round needs a
48 GB card. The model refuses non-fp32 weights and core downcasts fp32 under
``dtype=auto``, so both pass ``float32`` explicitly.

The default lookahead is 6 (560 ms, ``s`` = 8,960 samples, ``Q`` = 71).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.card]

RIGHT_CONTEXT = 6
CADENCE = "560ms"
CHUNK = 1280 * (RIGHT_CONTEXT + 1)
MAX_MODEL_LEN = 16_384
MAX_NUM_SEQS = 2


# ---------------------------------------------------------------------------
# audio


def synthetic_audio(seconds: float, seed: int = 0) -> np.ndarray:
    """Speech-shaped bursts with pauses, 16 kHz float32."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    t = np.arange(n) / 16000
    voiced = sum(
        np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1)
    )
    envelope = (np.sin(2 * np.pi * 2.7 * t) > 0.1).astype(np.float64)
    signal = 0.3 * voiced * envelope + 0.01 * rng.standard_normal(n)
    return signal.astype(np.float32)


def load_audio(seconds: float) -> np.ndarray:
    """``seconds`` of audio: the file named by NEMOTRON_TEST_AUDIO, else synthetic."""
    path = os.environ.get("NEMOTRON_TEST_AUDIO")
    n = int(seconds * 16000)
    if not path:
        return synthetic_audio(seconds)
    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    audio = data.astype(np.float32) / 32768.0
    if len(audio) < n:
        audio = np.tile(audio, n // max(len(audio), 1) + 1)
    return audio[:n].copy()


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# engine


def serve_flags() -> list[str]:
    """The flags REG-ENG-007 requires, plus the default lookahead."""
    return [
        "--config-format", "nemotron_asr_streaming",
        "--tokenizer-mode", "nemotron_asr_streaming",
        "--hf-overrides", json.dumps({"num_lookahead_tokens": RIGHT_CONTEXT}),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--dtype", "float32",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", "0.3",
    ]  # fmt: skip


class Runner:
    """Runs coroutines on one persistent loop, so one AsyncLLM serves a module."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()

    def __call__(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)


@pytest.fixture(scope="module")
def run():
    runner = Runner()
    yield runner
    runner.loop.close()


@pytest.fixture(scope="module")
def engine(run):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_nemotron_asr_streaming import register

    register()
    card = os.environ["NEMOTRON_CARD_DIR"]

    async def build():
        args = AsyncEngineArgs(
            model=card,
            config_format="nemotron_asr_streaming",
            tokenizer_mode="nemotron_asr_streaming",
            hf_overrides={"num_lookahead_tokens": RIGHT_CONTEXT},
            async_scheduling=False,
            enable_prefix_caching=False,
            enforce_eager=True,
            dtype="float32",
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=0.3,
        )
        return AsyncLLM.from_engine_args(args)

    llm = run(build())
    yield llm
    llm.shutdown()


@dataclass
class FileResult:
    text: str
    token_ids: list[int]
    finish_reason: str | None


def blank_of(engine) -> int:
    return int(engine.model_config.hf_config.num_asr_labels)


async def transcribe_file(
    engine, samples: np.ndarray, *, extra_args: dict | None = None
) -> FileResult:
    """Submit a file the way core's transcription serving does."""
    from fastapi import UploadFile
    from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )

    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    stt = SpeechToTextParams(
        audio=samples,
        stt_config=SpeechToTextConfig(
            sample_rate=16000, max_audio_clip_s=None, min_energy_split_window_size=None
        ),
        model_config=engine.model_config,
        extra_args=extra_args,
    )
    prompt = Nemotron3_5AsrStreamingForRNNT.get_generation_prompt(stt)
    request = TranscriptionRequest(
        file=UploadFile(io.BytesIO(b"x"), filename="a.wav"), model="m"
    )
    # core: max_tokens = max_model_len minus the prompt length
    # (speech_to_text/base/serving.py:517, entrypoints/serve/utils/api_utils.py
    # get_max_tokens); the prompt is the one carrier token
    max_tokens = engine.model_config.max_model_len - len(prompt["prompt_token_ids"])
    params = request.to_sampling_params(max_tokens)
    final = None
    async for out in engine.generate(prompt, params, f"file-{uuid.uuid4().hex}"):
        final = out
    assert final is not None
    completion = final.outputs[0]
    return FileResult(
        completion.text, list(completion.token_ids), completion.finish_reason
    )


class RecordingEngine:
    """Engine client that records every token ID the engine emits.

    The session factory checks ``generate``'s signature (SESS-FAC-003), so the
    wrapper carries the real signature.
    """

    def __init__(self, engine) -> None:
        import functools
        import inspect

        self._engine = engine
        self.token_ids: list[int] = []

        @functools.wraps(engine.generate)
        async def generate(*args, **kwargs):
            async for out in engine.generate(*args, **kwargs):
                for completion in out.outputs:
                    self.token_ids.extend(completion.token_ids)  # delta output
                yield out

        generate.__signature__ = inspect.signature(engine.generate)  # type: ignore[attr-defined]
        self.generate = generate

    def __getattr__(self, name):
        return getattr(self._engine, name)


async def transcribe_stream(
    engine,
    samples: np.ndarray,
    piece: int = 4000,
    recorder: RecordingEngine | None = None,
) -> str:
    """Feed the same samples through a session at the default cadence."""
    from vllm_nemotron_asr_streaming.session import create_session_factory

    client = recorder if recorder is not None else engine
    factory = create_session_factory(client, engine.model_config)
    assert CADENCE in factory.served_cadences
    lease = await factory.open(cadence=CADENCE, locale=factory.default_locale)
    try:
        for start in range(0, len(samples), piece):
            await lease.feed(samples[start : start + piece], on_accepted=lambda n: None)
        text = await lease.flush()
        await lease.finish()
    finally:
        await lease.release()
    return text


def expected_chunks(n: int) -> tuple[int, int]:
    return n // CHUNK, n - (n // CHUNK) * CHUNK


def labels(engine, token_ids: list[int]) -> list[int]:
    return [t for t in token_ids if t < blank_of(engine)]


# ---------------------------------------------------------------------------
# FILE-ROW


# @spec FILE-ROW-001, FILE-ROW-002, FILE-ROW-005
def test_file_request_runs_to_eos_after_every_advance(run, engine):
    blank = blank_of(engine)
    audio = load_audio(12.0)
    c, _ = expected_chunks(len(audio))
    result = run(transcribe_file(engine, audio))
    # a FILE row (the file carrier took the row role) ends on the park/EOS
    # token, once, last, after at least one step per chunk and the tail
    assert result.finish_reason == "stop"
    assert result.token_ids[-1] == blank + 1
    assert result.token_ids.count(blank + 1) == 1
    assert len(result.token_ids) >= c + 1 + 1
    # no stream-only control token appears in a file's output
    assert (blank + 2) not in result.token_ids


# @spec FILE-ROW-002, FILE-ENG-001
def test_one_chunk_and_one_token_per_step_on_silence(run, engine):
    blank = blank_of(engine)
    n = CHUNK * 9 + 1234
    c, _ = expected_chunks(n)
    result = run(transcribe_file(engine, np.zeros(n, dtype=np.float32)))
    advances = c + 1  # C regular chunks and the tail, one step each
    non_eos = result.token_ids[:-1]
    n_labels = len(labels(engine, non_eos))
    n_continuation = non_eos.count(blank + 5)
    # every step emits one token: advances that queued nothing emit the
    # continuation token, the others emit labels (first at the advance, the
    # rest in drain steps), so tokens = continuations + labels (+ EOU) + EOS
    assert n_continuation <= advances
    assert len(result.token_ids) == (
        n_continuation + n_labels + non_eos.count(blank + 3) + 1
    )
    if n_labels == 0:
        assert n_continuation == advances  # a chunk never takes two steps


# @spec FILE-ROW-003
def test_file_of_whole_chunks_equals_stream(run, engine):
    audio = load_audio(10.0)[: CHUNK * 8]
    assert len(audio) == CHUNK * 8  # T = 0
    result = run(transcribe_file(engine, audio))
    assert result.finish_reason == "stop"
    assert result.text == run(transcribe_stream(engine, audio))


# @spec FILE-ROW-004
@pytest.mark.parametrize("tail", [1, 1280, CHUNK - 1])
def test_final_tail_equals_stream_final_tail(run, engine, tail):
    audio = load_audio(10.0)[: CHUNK * 6 + tail]
    result = run(transcribe_file(engine, audio))
    assert result.finish_reason == "stop"
    assert result.text == run(transcribe_stream(engine, audio))


# @spec FILE-ROW-006
def test_empty_file_runs_tail_and_end_only(run, engine):
    blank = blank_of(engine)
    result = run(transcribe_file(engine, np.zeros(0, dtype=np.float32)))
    assert result.text == ""
    assert result.finish_reason == "stop"
    # the tail step (no labels: the continuation token), then the end step
    assert result.token_ids == [blank + 5, blank + 1]


# @spec FILE-ROW-007
@pytest.mark.skipif(
    not os.environ.get("NEMOTRON_TEST_AUDIO"),
    reason="needs a speech fixture named by NEMOTRON_TEST_AUDIO: silence or "
    "synthetic audio would make both label sequences empty",
)
@pytest.mark.parametrize(
    "n_samples",
    [16000 * 3, CHUNK * 10, CHUNK * 10 + 3333, 16000 * 7, 16000 * 20],
)
def test_file_label_sequence_equals_session(run, engine, n_samples):
    blank = blank_of(engine)
    audio = load_audio(n_samples / 16000)
    assert len(audio) == n_samples
    result = run(transcribe_file(engine, audio))
    assert result.finish_reason == "stop"
    recorder = RecordingEngine(engine)
    run(transcribe_stream(engine, audio, recorder=recorder))
    file_labels = [t for t in result.token_ids if t < blank]
    stream_labels = [t for t in recorder.token_ids if t < blank]
    assert file_labels  # speech: a nonempty label sequence on both paths
    assert file_labels == stream_labels


# ---------------------------------------------------------------------------
# FILE-ENG-001, FILE-BUF-002: staging per step on the production path


# @spec FILE-ENG-001, FILE-BUF-002, FILE-ROW-002
@pytest.mark.inprocess_engine
@pytest.mark.parametrize("right_context", [3, 6])
def test_engine_stages_copies_and_encodes_at_most_one_chunk_per_file_step(
    monkeypatch, right_context
):
    """Synchronous in-process engine: ``LLM`` builds ``LLMEngine``, which honours
    VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient, worker in this process), so
    the wrapped boundaries see the real worker.

    Layout constraints the probes rely on (Phase 6 must keep them):
    - ``FileHostBuffers.stage`` is the single host-to-device route for FILE
      samples, and the device copy is ``file_rows.copy_chunk_to_device(view,
      *, device)``, looked up as a module attribute at call time;
    - FILE rows' encoder chunks run through ``advance.advance_chunk_bucket``
      (one call per geometry group; ``env.shape[0]`` rows = chunks encoded),
      looked up as a module global at call time;
    - ``NemotronASRModelState.plan_step`` runs once per step.
    """
    import gc

    from fastapi import UploadFile
    from vllm import LLM
    from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.entrypoints.speech_to_text.transcription.protocol import (
        TranscriptionRequest,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.file_rows import (
        FileAction,
        FileHostBuffers,
    )

    from vllm_nemotron_asr_streaming import register
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT
    from vllm_nemotron_asr_streaming.nemotron_asr import advance, file_rows
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        NemotronASRModelState,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    register()
    step = [0]
    staged: list[tuple[str, int, int, int]] = []  # (request_id, step, start, length)
    copies: list[tuple[int, int]] = []  # (step, samples copied to the device)
    encoded: list[tuple[int, int]] = []  # (step, chunk rows run by the encoder)
    plans: dict[int, Any] = {}  # step -> StepPlan
    real_stage = FileHostBuffers.stage
    real_prepare = NemotronASRModelState.prepare_step
    real_plan = NemotronASRModelState.plan_step
    real_copy = file_rows.copy_chunk_to_device
    real_bucket = advance.advance_chunk_bucket

    def stage(self, request_id, start, length):
        staged.append((request_id, step[0], int(start), int(length)))
        return real_stage(self, request_id, start, length)

    def prepare_step(self, *args, **kwargs):
        step[0] += 1
        return real_prepare(self, *args, **kwargs)

    def plan_step(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        plans[step[0]] = plan
        return plan

    def copy_chunk_to_device(view, *, device):
        copies.append((step[0], int(view.shape[0])))
        return real_copy(view, device=device)

    def advance_chunk_bucket(core, env, state, **kwargs):
        encoded.append((step[0], int(env.shape[0])))
        return real_bucket(core, env, state, **kwargs)

    monkeypatch.setattr(FileHostBuffers, "stage", stage)
    monkeypatch.setattr(NemotronASRModelState, "prepare_step", prepare_step)
    monkeypatch.setattr(NemotronASRModelState, "plan_step", plan_step)
    monkeypatch.setattr(file_rows, "copy_chunk_to_device", copy_chunk_to_device)
    monkeypatch.setattr(advance, "advance_chunk_bucket", advance_chunk_bucket)

    llm = LLM(
        model=os.environ["NEMOTRON_CARD_DIR"],
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": right_context},
        async_scheduling=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        dtype="float32",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.2,
    )
    try:
        model_config = llm.llm_engine.model_config
        audio = load_audio(10.0)
        n = len(audio)
        prompt = Nemotron3_5AsrStreamingForRNNT.get_generation_prompt(
            SpeechToTextParams(
                audio=audio,
                stt_config=SpeechToTextConfig(
                    sample_rate=16000,
                    max_audio_clip_s=None,
                    min_energy_split_window_size=None,
                ),
                model_config=model_config,
            )
        )
        request = TranscriptionRequest(
            file=UploadFile(io.BytesIO(b"x"), filename="a.wav"), model="m"
        )
        params = request.to_sampling_params(
            model_config.max_model_len - len(prompt["prompt_token_ids"])
        )
        # warmup and profile runs recorded nothing of interest: start clean
        step[0] = 0
        staged.clear()
        copies.clear()
        encoded.clear()
        plans.clear()
        (output,) = llm.generate([prompt], params)
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()
    assert output.outputs[0].finish_reason == "stop"

    chunk = 1280 * (right_context + 1)
    c, _ = divmod(n, chunk)
    assert staged
    per_step: dict[int, list[tuple[int, int]]] = {}
    for _, at_step, start, length in staged:
        per_step.setdefault(at_step, []).append((start, length))
    # at most one stage call per file per step, of at most one chunk
    assert all(len(v) == 1 and v[0][1] <= chunk for v in per_step.values())
    # starts contiguous, covering every sample exactly once
    position = 0
    for key in sorted(per_step):
        start, length = per_step[key][0]
        assert start == position
        position += length
    assert position == n
    assert c <= len(per_step) <= c + 1

    # the device copy and the encoder, per engine step
    copied_by_step: dict[int, list[int]] = {}
    for at_step, samples in copies:
        copied_by_step.setdefault(at_step, []).append(samples)
    encoded_by_step: dict[int, int] = {}
    for at_step, rows in encoded:
        encoded_by_step[at_step] = encoded_by_step.get(at_step, 0) + rows
    for at_step, plan in plans.items():
        (row,) = [r for r in plan.rows if r.file_plan is not None]
        action = row.file_plan.action
        copied = copied_by_step.get(at_step, [])
        rows = encoded_by_step.get(at_step, 0)
        if action in (FileAction.DRAIN, FileAction.END):
            assert copied == [] and rows == 0  # nothing while draining
        else:
            assert rows == 1  # one encoder chunk for the file row
            assert len(copied) <= 1  # never more than one chunk's samples
            staged_len = per_step.get(at_step, [(0, 0)])[0][1]
            assert sum(copied) == staged_len <= chunk  # copy equals the view
    assert set(copied_by_step) <= set(plans) and set(encoded_by_step) <= set(plans)


# ---------------------------------------------------------------------------
# FILE-BUD, FILE-ENG


# @spec FILE-BUD-002
def test_worst_case_budget_file_ends_with_eos_not_length(run, engine):
    # the longest file whose worst-case budget fits max_model_len
    from vllm_nemotron_asr_streaming.transcription import file_token_budget

    n = CHUNK * 220
    while file_token_budget(n + CHUNK, RIGHT_CONTEXT) <= MAX_MODEL_LEN:
        n += CHUNK
    assert file_token_budget(n, RIGHT_CONTEXT) <= MAX_MODEL_LEN
    audio = np.random.default_rng(3).standard_normal(n).astype(np.float32) * 0.2
    result = run(transcribe_file(engine, audio))
    assert result.finish_reason == "stop"
    assert result.token_ids[-1] == blank_of(engine) + 1


# @spec FILE-ENG-002
def test_file_requests_never_emit_the_endpoint_token(run, engine):
    blank = blank_of(engine)
    # speech, a long pause, speech: a stream with endpointing would end the
    # utterance in the pause
    audio = np.concatenate(
        [load_audio(4.0), np.zeros(16000 * 4, np.float32), load_audio(4.0)]
    )
    result = run(transcribe_file(engine, audio))
    assert result.finish_reason == "stop"
    assert (blank + 3) not in result.token_ids


# @spec FILE-ENG-003
def test_file_and_stream_share_steps_without_changing_transcripts(run, engine):
    file_audio = load_audio(14.0)
    stream_audio = load_audio(9.0)[::-1].copy()

    alone_file = run(transcribe_file(engine, file_audio))
    alone_stream = run(transcribe_stream(engine, stream_audio))

    async def together():
        return await asyncio.gather(
            transcribe_file(engine, file_audio),
            transcribe_stream(engine, stream_audio),
            transcribe_file(engine, file_audio),
        )

    first, stream_text, second = run(together())
    assert first.finish_reason == second.finish_reason == "stop"
    assert first.text == second.text == alone_file.text
    assert stream_text == alone_stream


# @spec FILE-ENG-005
def test_files_beyond_the_slot_count_wait_and_complete(run, engine):
    audio = load_audio(6.0)
    count = MAX_NUM_SEQS * 3

    async def many():
        return await asyncio.gather(
            *(transcribe_file(engine, audio) for _ in range(count))
        )

    results = run(many())
    assert len(results) == count
    assert {r.finish_reason for r in results} == {"stop"}
    assert len({r.text for r in results}) == 1


# ---------------------------------------------------------------------------
# FILE-ENG-006: core's streaming response, over HTTP


@pytest.fixture(scope="module")
def server():
    import httpx

    card = Path(os.environ["NEMOTRON_CARD_DIR"])
    port = int(os.environ.get("NEMOTRON_TEST_PORT", "18765"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(card),
         "--port", str(port), *serve_flags()],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "1"},
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 900
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("vllm serve exited during startup")
            try:
                if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            pytest.fail("vllm serve did not become healthy")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=60)


# @spec FILE-ENG-006
def test_streaming_response_carries_only_label_text(server):
    import httpx

    audio = load_audio(12.0)
    files = {"file": ("a.wav", wav_bytes(audio), "audio/wav")}
    model = os.environ["NEMOTRON_CARD_DIR"]
    plain = httpx.post(
        f"{server}/v1/audio/transcriptions",
        files=files,
        data={"model": model},
        timeout=300,
    )
    plain.raise_for_status()
    expected = plain.json()["text"]

    events: list[dict] = []
    done = False
    with httpx.stream(
        "POST",
        f"{server}/v1/audio/transcriptions",
        files=files,
        data={"model": model, "stream": "true"},
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                done = True
                continue
            assert not done, "event after [DONE]"
            events.append(json.loads(payload))
    assert done
    choices = [e["choices"][0] for e in events if e.get("choices")]
    contents = [(c.get("delta") or {}).get("content") for c in choices]
    terminal = [c for c in choices if c.get("finish_reason")]
    # core may carry text on the finish event itself (base/serving.py), so the
    # transcript includes terminal delta content; the non-empty rule applies
    # to non-terminal events only
    non_terminal = [
        (c.get("delta") or {}).get("content")
        for c in choices
        if not c.get("finish_reason")
    ]
    assert all(isinstance(t, str) and t != "" for t in non_terminal)
    assert "".join(t or "" for t in contents) == expected
    assert len(terminal) == 1
    assert terminal[0]["finish_reason"] == "stop"
