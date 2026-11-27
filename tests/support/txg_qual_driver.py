# SPDX-License-Identifier: Apache-2.0
"""The transaction-level tier 5 qualification driver (TXG-QUAL-001 to -007,
-010 to -014, -016, -019, TXG-CAL-001 to -003).

One process per run: ``python -m support.txg_qual_driver SPEC.json OUT.json``
(``PYTHONPATH`` naming ``tests``). Every engine runs the restricted startup
(``txgcap_qual.restricted_startup``) with core's graph-memory estimate
enabled, under the run's KV-cache sizing (:func:`run_sizing`): core's
default, the extensible KV cache, except the step rule, which runs under a
fixed ``--num-gpu-blocks-override`` (TXG-QUAL-001, -010), and a spec that
names another (``kv_cache_sizing``, ``num_gpu_blocks_override``,
``kv_cache_memory_bytes``). ``B_hook`` is the same settings, the sizing
included, with ``transaction_graph_populations`` absent. The
transaction-level modes start one engine in this process (``LLMEngine`` with
``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so the engine core and its worker live
here) and drive the worker through ``support.txg_qual_adapter.QualAdapter``
(TXG-QUAL-026), assigning blocks below the final committed block count. The
``uapi`` and ``alone`` modes serve through a multiprocess engine
(``AsyncLLM``) on the client path. Every pass/fail decision is
``support.txgcap_qual``'s.

The reply records, for every run, the KV-cache sizing, each capture
layout's page count and the final bound count the driver reads after
core's extension (``identity``; TXG-QUAL-001), and the startup probes
(``probes``: the plugin's INFO and WARNING lines, core's logged torch
memory limit, and, under the extensible KV cache, the committed block count
at the serving capture and the state layer's views at that capture and after
core's extension; TXG-QUAL-028). A spec with ``headroom``
(``txgcap_qual.preregistered_headroom``: ``max_requests``,
``page_bytes``, ``copy_bytes``, ``margin_bytes``, computed before the run)
runs an exactness run under the extensible KV cache with that headroom left
free through ``--gpu-memory-utilization`` and records the allocator around
the detached copies (``exactness_memory``; TXG-QUAL-029). In the ``qm``
and ``fallback`` modes the detached copies hold block 0 and one page per
request the mode can hold (``txgcap_qual.copy_extent`` of
``mode_max_requests``: pages 0 to the admission limit, and 0 to ``N + 4``),
a final committed count below that extent fails the run before its first
step, and the other committed pages are digested in place before the first
step and after the last; a changed digest fails the run (TXG-QUAL-003,
-007).

Modes (``spec["mode"]``):

* ``qm``: the qualification workload ``Q`` at each concurrency, then the
  mixed workload ``M``, every step run replayed and as the option-off
  transaction on a copy of the pools (TXG-QUAL-002 to -006). With
  ``replay`` (a ``qm`` output of the option-on run) the clock replays that
  run's recorded step durations, so an option-off (``B_hook``) process runs
  the same schedule and its transcripts can be compared (TXG-QUAL-006).
* ``fallback``: steps with ``K > N``, ``E > N``, forced signature
  mismatches and a multi-token row among one-token rows, each compared with
  the option-off transaction (TXG-QUAL-007).
* ``steprule``: the step rule for one population (TXG-QUAL-010 to -014);
  its first composition at population 1 is also the calibration run's U(1)
  (TXG-CAL-001), which the calibration runs once under the step rule's fixed
  block count and once with ``kv_cache_sizing`` ``"extensible"``, core's
  default (TXG-CAL-001, -002).
* ``alone``: TXG-QUAL-028's served part in a fresh multiprocess engine
  (``AsyncLLM``): one stream served alone on the client path of core's
  ``/v1/realtime``, then each of TXG-QUAL-004's files transcribed alone on
  the idle server as core's transcription serving submits it
  (:func:`run_alone`); per item, its outcome, token IDs, cell replays and
  fallbacks by reason. Its worker extension (:data:`ALONE_READER`) installs
  the startup probes in the engine-core process.
* ``uapi``: one leg of ``U_api`` (TXG-QUAL-016) in a multiprocess engine
  (``AsyncLLM``): repeats 0 to 3, each a fresh stream opened with
  ``AsyncLLM.generate`` and an input generator that yields the plugin
  packer's ``StreamingInput`` units (:func:`run_u_api_leg`).

Audio is LibriSpeech test-clean (``spec["corpus"]``: the directory holding
``test-clean``), in a seeded permutation of the sorted utterance IDs. Stream
units are built by the plugin's packer (``session.packer.Packer``) at the
server's geometry and default locale, file prompts by the model's
``get_generation_prompt``, and both pass through the engine's own input
processor: the input is plugin-packed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

LOOKAHEADS = (0, 1, 3, 6, 13)
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280

# ---------------------------------------------------------------------------
# Audio


class Clips:
    """One run's clips: ``rng = default_rng(seed)``, the permutation
    ``rng.permutation(count)`` of the corpus order, then the same generator for
    the run's stream offsets (``rng.uniform(0, c)`` as each stream is taken),
    the served ``Q`` run's convention (lane B, TXG-QUAL-008: same manifest,
    seed and offsets)."""

    def __init__(self, corpus: Any, seed: int) -> None:
        self.corpus = corpus
        self.rng = np.random.default_rng(seed)
        self.order = [corpus.ids[i] for i in self.rng.permutation(len(corpus.ids))]
        self._next = 0

    def take(self) -> tuple[str, np.ndarray]:
        utt = self.order[self._next % len(self.order)]
        self._next += 1
        return utt, self.corpus.load(utt)


class Corpus:
    """LibriSpeech test-clean, its utterance IDs sorted (the harness
    manifest's order: chapters by path, utterances by ID). ``take`` draws from
    the process's own permutation (``seed``); ``clips(seed)`` starts a run's
    permutation and offset generator (:class:`Clips`)."""

    def __init__(self, root: str | Path, seed: int) -> None:
        base = Path(root)
        files = sorted(base.glob("test-clean/*/*/*.flac"))
        if not files:
            files = sorted(base.glob("LibriSpeech/test-clean/*/*/*.flac"))
        if not files:
            raise FileNotFoundError(f"no test-clean FLAC files under {base}")
        self.paths = {p.stem: p for p in files}
        self.ids = sorted(self.paths)
        self.manifest_sha256 = hashlib.sha256("\n".join(self.ids).encode()).hexdigest()
        self.count = len(self.ids)
        self._own = Clips(self, seed)

    def load(self, utt: str) -> np.ndarray:
        import soundfile

        data, rate = soundfile.read(str(self.paths[utt]), dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError(f"{utt} has sample rate {rate}")
        return np.ascontiguousarray(data.reshape(-1))

    def take(self) -> tuple[str, np.ndarray]:
        return self._own.take()

    def clips(self, seed: int) -> Clips:
        return Clips(self, seed)


class SyntheticCorpus(Corpus):
    """Seeded noise clips (CPU tests only)."""

    def __init__(self, seed: int, lengths: Sequence[int]) -> None:
        self.lengths = list(lengths)
        self.ids = [f"synthetic-{i}" for i in range(len(self.lengths))]
        self.manifest_sha256 = "synthetic"
        self.count = len(self.ids)
        self._own = Clips(self, seed)

    def load(self, utt: str) -> np.ndarray:
        i = int(utt.rsplit("-", 1)[1])
        rng = np.random.default_rng(1000 + i)
        return (rng.standard_normal(self.lengths[i]) * 0.1).astype(np.float32)


# ---------------------------------------------------------------------------
# Requests


class Requests:
    """Builds the worker's ``NewRequestData`` from plugin-packed prompts:
    through the engine's input processor when an engine is given, else
    directly (CPU tests, the runnable synthetic card)."""

    def __init__(self, model: Any, *, lookahead: int, engine: Any = None) -> None:
        self.engine = engine
        self.model = model
        self.config = model.config
        self.lookahead = lookahead
        self.geometry = LOOKAHEADS.index(lookahead)
        self.unit_samples = FRAME_SAMPLES * (lookahead + 1)
        self.chunk_seconds = self.unit_samples / SAMPLE_RATE
        table = getattr(self.config, "prompt_dictionary", None) or {}
        self.locale_row = _default_locale_row(table)
        self._tasks = None if engine is None else engine.get_supported_tasks()
        #: Files the input processor refused (built directly instead).
        self.direct_files: list[dict[str, str]] = []
        c = self.config
        self.park = int(c.eos_token_id)
        self.controls = {
            int(v)
            for v in (
                c.eos_token_id,
                getattr(c, "continuation_token_id", None),
                getattr(c, "file_carrier_token_id", None),
                c.audio_chunk_token_id,
                getattr(c, "eou_token_id", None),
            )
            if v is not None
        }

    def packer(self) -> Any:
        from vllm_nemotron_asr_streaming.session.packer import Packer

        return Packer(
            right_context=self.lookahead,
            geometry_index=self.geometry,
            locale_row=self.locale_row,
            carrier_token_id=int(self.config.audio_chunk_token_id),
        )

    def units(self, samples: np.ndarray) -> list[Any]:
        packer = self.packer()
        return [*packer.append(samples), packer.close()]

    def _params(self) -> Any:
        from vllm.sampling_params import SamplingParams

        return SamplingParams(temperature=0.0, max_tokens=64)

    def _new(self, req_id: str, processed: Any) -> Any:
        from vllm.v1.core.sched.output import NewRequestData

        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(processed.prompt_token_ids),
            mm_features=processed.mm_features,
            sampling_params=processed.sampling_params,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(processed.prompt_token_ids),
        )

    def unit(self, req_id: str, unit: Any, tag: str) -> Any:
        if self.engine is None:
            import torch

            from support import model_harness as h

            return h.new_request(
                req_id,
                int(unit.prompt_token_ids[0]),
                audio=torch.from_numpy(unit.row.copy()),
                identifier=f"{req_id}-{tag}",
            )
        processed = self.engine.input_processor.process_inputs(
            f"{req_id}-{tag}", unit.prompt, self._params(), supported_tasks=self._tasks
        )
        return self._new(req_id, processed)

    def file(self, req_id: str, samples: np.ndarray) -> Any:
        if self.engine is None:
            from support import model_harness as h

            return h.new_request(
                req_id,
                int(self.config.file_carrier_token_id),
                samples=samples.astype(np.float32),
                header=[self.geometry, self.locale_row, int(samples.shape[0])],
            )
        stt = SimpleNamespace(
            task_type="transcribe",
            request_prompt="",
            hotwords=None,
            to_language=None,
            model_config=self.engine.model_config,
            extra_args=None,
            language=None,
            audio=samples,
        )
        prompt = type(self.model).get_generation_prompt(stt)
        try:
            processed = self.engine.input_processor.process_inputs(
                req_id, prompt, self._params(), supported_tasks=self._tasks
            )
        except Exception as error:  # noqa: BLE001 - recorded, then built directly
            if samples.shape[0]:
                raise
            # The input processor refuses an empty audio item; the model's
            # prompt (samples and header) is delivered as an engine client
            # would deliver it, and the substitution is recorded.
            self.direct_files.append({"req_id": req_id, "error": repr(error)[:300]})
            from support import model_harness as h

            header = prompt["mm_processor_kwargs"]["file_header"]
            return h.new_request(
                req_id,
                int(self.config.file_carrier_token_id),
                samples=samples.astype(np.float32),
                header=list(header),
                identifier=prompt["multi_modal_uuids"]["audio"][0],
            )
        return self._new(req_id, processed)

    def special(self, req_id: str, token: int) -> Any:
        """A request with no audio whose scheduled token is ``token``, as only
        a direct engine stream client sends one."""
        from support import model_harness as h

        return h.new_request(req_id, int(token))

    def label_tokens(self, count: int) -> list[int]:
        """``count`` label token IDs: no control token, no blank."""
        blank = getattr(self.config, "blank_token_id", None)
        out = [
            t
            for t in range(1, int(self.config.vocab_size))
            if t not in self.controls and t != blank
        ]
        return out[:count]

    def multi_token(self, req_id: str, tokens: Sequence[int]) -> Any:
        """A request with no audio whose prompt is ``tokens``, all scheduled
        in one step: a multi-token row, as only a direct engine client sends
        one (REG-STATE-014, TXG-COV-014)."""
        import dataclasses as dc

        from support import model_harness as h

        tokens = [int(t) for t in tokens]
        request = h.new_request(req_id, tokens[0])
        return dc.replace(request, prompt_token_ids=tokens, prefill_token_ids=tokens)


def _default_locale_row(table: dict) -> int:
    if not table:
        return 0
    try:
        from vllm_nemotron_asr_streaming.locales import default_locale

        locale = default_locale(table)
    except Exception:  # noqa: BLE001 - an older plugin
        locale = "auto" if "auto" in table else None
    if locale is None:
        locale = sorted(table)[0]
    return int(table[locale])


def mm_hashes(request: Any) -> list[str]:
    return [f.identifier for f in (request.mm_features or ()) if f.identifier]


# ---------------------------------------------------------------------------
# Blocks


class Blocks:
    """The adapter's block authority (TXG-QUAL-026): each request takes the
    lowest free block at its first step, keeps it until its terminal finish
    and frees it only then, so a block is reused only after its request
    finished."""

    def __init__(self, capacity: int) -> None:
        self.free = list(range(1, capacity))
        self.held: dict[str, int] = {}

    def take(self, req_id: str) -> int:
        if req_id not in self.held:
            if not self.free:
                raise RuntimeError("no free state block")
            self.free.sort()
            self.held[req_id] = self.free.pop(0)
        return self.held[req_id]

    def of(self, req_ids: Sequence[str]) -> dict[str, int]:
        """The blocks of ``req_ids``, each taken at its first step."""
        return {r: self.take(r) for r in req_ids}

    def release(self, req_id: str) -> None:
        block = self.held.pop(req_id, None)
        if block is not None:
            self.free.append(block)


# ---------------------------------------------------------------------------
# Streams and files


@dataclasses.dataclass(eq=False)
class Stream:
    req_id: str
    utt: str
    units: list[Any]
    arrivals: list[float]
    delivered: int = 0
    busy: bool = False
    position: int = 0
    tokens: list[int] = dataclasses.field(default_factory=list)
    retired: bool = False
    failed: bool = False

    def next_arrival(self) -> float | None:
        if self.busy or self.delivered >= len(self.units):
            return None
        return self.arrivals[self.delivered]

    @property
    def exhausted(self) -> bool:
        return self.delivered >= len(self.units)


@dataclasses.dataclass(eq=False)
class FileItem:
    req_id: str
    name: str
    samples: np.ndarray
    started: bool = False
    position: int = 0
    tokens: list[int] = dataclasses.field(default_factory=list)
    done: bool = False
    failed: bool = False


def _bits(plan: Any) -> tuple[list[bool], list[bool], list[bool]]:
    chunk = [bool(v) for v in plan.is_chunk.tolist()]
    fresh = [not bool(v) for v in plan.has_initial_states_p.tolist()]
    file_mask = getattr(plan, "is_file", None)
    if file_mask is None or not file_mask.numel():
        file = [False] * len(chunk)
    else:
        file = [bool(v) for v in file_mask.tolist()]
    return chunk, fresh, file


def step_cases(call: Any, requests: Requests, zero_files: set[str]) -> set[str]:
    """The TXG-EQ-004 cases (``txgcap_qual.EQ004_CASES`` names) a compared
    step exercises, from its row plan, inputs and outputs."""
    from support import txg_transaction as tx
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    if call is None or call.plan is None or call.rows is None:
        return set()
    plan, config = call.plan, requests.config
    chunk, fresh, file = _bits(plan)
    ids = call.input_ids.tolist()
    header = call.inputs_embeds[:, : tx.HEADER_SLOTS].cpu()
    final = [bool(header[i, tx.H_FINAL]) for i in range(len(chunk))]
    samples = [int(header[i, tx.H_SAMPLES]) for i in range(len(chunk))]
    out = [int(t) for t in read_decision_carrier(call.rows).tolist()]
    park, cont = requests.park, int(getattr(config, "continuation_token_id", -1))
    label = [t not in requests.controls and t >= 0 for t in ids]
    phase = (
        call.progress[:, 2].tolist()
        if call.progress is not None
        else [None] * len(chunk)
    )
    flush = [not chunk[i] and phase[i] == 3 for i in range(len(chunk))]
    rows = range(len(chunk))
    cases = set()

    def any_(pred):
        return any(pred(i) for i in rows)

    if any_(lambda i: chunk[i] and fresh[i] and not file[i]) and any_(
        lambda i: chunk[i] and not fresh[i]
    ):
        cases.add("stream-first-chunk-among-continuing")
    if any_(lambda i: chunk[i] and fresh[i] and file[i]):
        cases.add("file-first-step")
    if any_(lambda i: chunk[i] and not fresh[i] and not final[i]):
        cases.add("regular-chunk")
    if any_(lambda i: chunk[i] and out[i] == cont):
        cases.add("silent-chunk")
    if any_(lambda i: chunk[i] and final[i] and samples[i] > 0):
        cases.add("final-tail-with-samples")
    if any_(lambda i: chunk[i] and final[i] and samples[i] == 0):
        cases.add("empty-final-tail")
    if any(r in zero_files for r in plan.request_ids):
        cases.add("zero-sample-file")
    if any_(lambda i: not chunk[i] and label[i]):
        cases.add("drain-step")
    if any_(lambda i: not chunk[i] and file[i] and not flush[i] and out[i] == park):
        cases.add("end-step-replay")
    if any_(lambda i: file[i] and flush[i] and out[i] == park):
        cases.add("end-step-flush")
    if any_(lambda i: chunk[i] and file[i] and label[i]):
        cases.add("armed-echo-on-file-chunk")
    if (
        any_(lambda i: chunk[i] and file[i])
        and any_(lambda i: chunk[i] and not file[i])
        and any_(lambda i: not chunk[i] and file[i])
        and any_(lambda i: not chunk[i] and not file[i])
    ):
        cases.add("split-mixed-file-and-stream")
    if any_(lambda i: not chunk[i] and fresh[i]):
        cases.add("fresh-non-chunk-row")
    return cases


def special_files(requests: Requests, corpus: Any) -> Iterator[tuple[str, np.ndarray]]:
    """``M``'s files in order (TXG-QUAL-004): leading and trailing silence, a
    whole number of chunks, zero samples, shorter than one chunk, then
    test-clean utterances from the same permutation."""
    unit = requests.unit_samples
    utt, clip = corpus.take()
    pad = np.zeros(SAMPLE_RATE, dtype=np.float32)
    yield f"silence-padded:{utt}", np.concatenate([pad, clip, pad])
    utt, clip = corpus.take()
    whole = max(1, clip.shape[0] // unit) * unit
    yield f"whole-chunks:{utt}", np.ascontiguousarray(clip[:whole])
    yield "zero-samples", np.zeros(0, dtype=np.float32)
    utt, clip = corpus.take()
    yield f"short:{utt}", np.ascontiguousarray(clip[: unit // 2])
    while True:
        utt, clip = corpus.take()
        yield f"utterance:{utt}", clip


# ---------------------------------------------------------------------------
# The free-running workloads (TXG-QUAL-002 to -006)


class Workload:
    """``Q`` (``n_files = 0``) or ``M``. ``durations`` replays a recorded
    run's clock (the ``B_hook`` transcript run)."""

    def __init__(
        self,
        ctx: Any,
        *,
        name: str,
        n_streams: int,
        n_files: int,
        steps: int,
        seed: int,
        files: Iterator[tuple[str, np.ndarray]] | None = None,
        durations: Sequence[float] | None = None,
        clips: Clips | None = None,
    ) -> None:
        self.ctx = ctx
        self.name = name
        self.n_streams = n_streams
        self.n_files = n_files
        self.steps = steps
        self.clips = ctx.corpus.clips(seed) if clips is None else clips
        self.rng = self.clips.rng
        self.files = files
        self.durations = durations
        self.clock = 0.0
        self.streams: list[Stream | None] = [None] * n_streams
        self.file_slots: list[FileItem | None] = [None] * n_files
        self.finished_streams: list[Stream] = []
        self.finished_files: list[FileItem] = []
        self.pending_finish: list[str] = []
        self.pending_free: list[str] = []
        self.counter = 0
        self.records: list[dict[str, Any]] = []
        self.problems: list[dict[str, Any]] = []
        self.histogram: dict[tuple[int, int], int] = {}
        self.cases: dict[str, int] = {}
        self.counts = {"split": 0, "fresh_among_continuing": 0, "split_with_file": 0}
        self.fallback_steps = 0
        self.failed_rows = 0
        self.zero_files: set[str] = set()
        for i in range(n_streams):
            self._start_stream(i, 0.0)

    # -- arrivals --------------------------------------------------------------

    def _start_stream(self, slot: int, after: float) -> None:
        requests = self.ctx.requests
        c = requests.chunk_seconds
        start = after + float(self.rng.uniform(0.0, c))
        utt, clip = self.clips.take()
        units = requests.units(clip)
        arrivals = [start + (k + 1) * c for k in range(len(units) - 1)]
        arrivals.append(max(start + clip.shape[0] / SAMPLE_RATE, *arrivals[-1:], start))
        self.counter += 1
        self.streams[slot] = Stream(
            f"{self.name}-s{slot}-{self.counter}", utt, units, arrivals
        )

    def _admit_files(self) -> None:
        if self.files is None or len(self.records) >= self.steps:
            return
        if any(s is None or s.retired for s in self.streams):
            return
        for i, item in enumerate(self.file_slots):
            if item is None or item.done:
                name, samples = next(self.files)
                self.counter += 1
                req_id = f"{self.name}-f{i}-{self.counter}"
                self.file_slots[i] = FileItem(req_id, name, samples)
                if samples.shape[0] == 0:
                    self.zero_files.add(req_id)

    def finished(self) -> bool:
        if len(self.records) < self.steps:
            return False
        return all(f is None or f.done for f in self.file_slots)

    # -- one step ----------------------------------------------------------------

    def _schedule(self) -> tuple[list, list, dict, list]:
        ctx = self.ctx
        new, cached, computed, scheduled = [], [], {}, []
        for s in self.streams:
            if s is None or s.retired:
                continue
            if s.busy:
                cached.append(s.req_id)
                computed[s.req_id] = s.position
                scheduled.append(s)
            elif not s.exhausted and s.arrivals[s.delivered] <= self.clock + 1e-12:
                unit = s.units[s.delivered]
                new.append(ctx.requests.unit(s.req_id, unit, str(s.delivered)))
                s.delivered += 1
                s.position = 0
                scheduled.append(s)
        for f in self.file_slots:
            if f is None or f.done:
                continue
            if not f.started:
                new.append(ctx.requests.file(f.req_id, f.samples))
                f.started = True
                f.position = 0
            else:
                cached.append(f.req_id)
                computed[f.req_id] = f.position
            scheduled.append(f)
        return new, cached, computed, scheduled

    def _next_arrival(self) -> float | None:
        times = [
            t
            for s in self.streams
            if s is not None and (t := s.next_arrival()) is not None
        ]
        return min(times) if times else None

    def run(self) -> None:
        ctx = self.ctx
        while not self.finished():
            self._admit_files()
            new, cached, computed, scheduled = self._schedule()
            if not scheduled:
                upcoming = self._next_arrival()
                if upcoming is None:
                    break
                self.clock = max(self.clock, upcoming)
                continue
            finished = list(self.pending_finish)
            for r in finished:
                ctx.blocks.release(r)
            blocks = ctx.blocks.of([*(x.req_id for x in new), *cached])
            free = sorted({h for r in new for h in mm_hashes(r)})
            ctx.monitor.begin_step()
            result = ctx.adapter.step(
                new=new,
                cached=cached,
                finished=finished,
                blocks=blocks,
                computed=computed,
                free_mm_hashes=ctx.free_next,
            )
            ctx.free_next = free
            self.pending_finish = []
            self._account(result, scheduled)

    def _account(self, result: Any, scheduled: list) -> None:
        ctx = self.ctx
        calls = list(ctx.monitor.calls)
        index = len(self.records)
        # Rounded once, so a replay of the recorded durations advances its
        # clock by exactly the same amounts and runs the same schedule.
        seconds = round(sum(c.seconds for c in calls), 6)
        if self.durations is not None:
            if index < len(self.durations):
                self.clock += float(self.durations[index])
            else:
                self.clock += seconds
        else:
            # The clock advances by the step's measured option-on duration.
            self.clock += seconds
        covered = any(c.kind == "execute" and c.covered for c in calls)
        declined = any(c.kind == "execute" and not c.covered for c in calls)
        main = next((c for c in calls if c.kind == "execute" and c.covered), None)
        if main is None:
            main = next((c for c in calls if c.kind == "off"), None)
        problems = [p for c in calls for p in c.problems]
        k = main.chunk_count if main is not None else 0
        p = main.population if main is not None else len(result.req_ids)
        e = p - k
        self.histogram[(k, e)] = self.histogram.get((k, e), 0) + 1
        record = {
            "i": index,
            "t": round(self.clock, 6),
            "P": p,
            "K": k,
            "covered": covered,
            "declined": declined,
            "s": seconds,
            "rows": list(result.req_ids),
        }
        if declined:
            self.fallback_steps += 1
        if problems:
            record["problems"] = problems
            self.problems.append({"i": index, "problems": problems})
        if result.failed:
            record["failed"] = sorted(result.failed)
            self.failed_rows += len(result.failed)
        if main is not None and main.plan is not None:
            cases = step_cases(main, ctx.requests, self.zero_files)
            for case in cases:
                self.cases[case] = self.cases.get(case, 0) + 1
            chunk, fresh, file = _bits(main.plan)
            if k and e:
                self.counts["split"] += 1
                if any(file):
                    self.counts["split_with_file"] += 1
            if any(c and f for c, f in zip(chunk, fresh, strict=True)) and any(
                c and not f for c, f in zip(chunk, fresh, strict=True)
            ):
                self.counts["fresh_among_continuing"] += 1
            if main.reference_rows is not None:
                record["ref"] = True
        self.records.append(record)
        for item in scheduled:
            token = result.sampled[item.req_id]
            if token not in ctx.requests.controls and token >= 0:
                item.tokens.append(int(token))
            item.position += 1
            if isinstance(item, FileItem):
                if item.req_id in result.failed:
                    item.failed = True
                if token == ctx.requests.park or item.failed:
                    item.done = True
                    self.finished_files.append(item)
                    self.pending_finish.append(item.req_id)
                continue
            if item.req_id in result.failed:
                item.failed = True
            if token == ctx.requests.park or item.failed:
                item.busy = False
                if item.exhausted or item.failed:
                    item.retired = True
                    self.finished_streams.append(item)
                    self.pending_finish.append(item.req_id)
                    slot = self.streams.index(item)
                    if len(self.records) < self.steps:
                        self._start_stream(slot, self.clock)
            else:
                item.busy = True

    def close(self) -> None:
        """Report every live request finished (one cleanup-only step)."""
        ctx = self.ctx
        live = [s.req_id for s in self.streams if s is not None and not s.retired]
        live += [f.req_id for f in self.file_slots if f is not None and not f.done]
        finished = list(self.pending_finish) + live
        if finished:
            ctx.monitor.begin_step()
            ctx.adapter.step(finished=finished, blocks={}, free_mm_hashes=ctx.free_next)
            ctx.free_next = []
            for r in finished:
                ctx.blocks.release(r)
        self.pending_finish = []

    def summary(self) -> dict[str, Any]:
        transcripts = {}
        for s in [*self.finished_streams, *(x for x in self.streams if x)]:
            transcripts.setdefault(s.req_id, {"utt": s.utt, "tokens": s.tokens})
        for f in [*self.finished_files, *(x for x in self.file_slots if x)]:
            transcripts.setdefault(f.req_id, {"utt": f.name, "tokens": f.tokens})
        steps = len(self.records)
        covered = sum(1 for r in self.records if r["covered"])
        return {
            "name": self.name,
            "streams": self.n_streams,
            "files": self.n_files,
            "steps": steps,
            "covered_steps": covered,
            "fallback_steps": self.fallback_steps,
            "failed_rows": self.failed_rows,
            "mismatched_steps": len(self.problems),
            "problems": self.problems[:20],
            "counts": dict(self.counts),
            "cases": dict(sorted(self.cases.items())),
            "histogram": {
                f"{k},{e}": v for (k, e), v in sorted(self.histogram.items())
            },
            "sim_seconds": round(self.clock, 3),
            "durations": [r["s"] for r in self.records],
            "schedule_sha256": hashlib.sha256(
                json.dumps([r["rows"] for r in self.records]).encode()
            ).hexdigest(),
            "transcripts": transcripts,
        }


# ---------------------------------------------------------------------------
# Requests alone on the client path (TXG-QUAL-028)

#: The worker extension of the ``alone`` mode (core's
#: ``--worker-extension-cls``): core imports its module in the engine-core
#: process before it builds the model runner, and the import installs the
#: startup probes there (``support.txg_alone_worker``).
ALONE_READER = "support.txg_alone_worker.AloneReader"


def alone_items(
    requests: Requests, corpus: Any, spec: dict[str, Any]
) -> list[tuple[str, str, str, np.ndarray]]:
    """TXG-QUAL-028's items in order, each ``(name, kind, utterance,
    samples)``: one stream (the run's first clip), then TXG-QUAL-004's files
    from the same permutation (leading and trailing silence, a whole number
    of chunks, zero samples, shorter than one chunk), then
    ``alone_utterances`` test-clean utterances, default 1."""
    clips = corpus.clips(int(spec["seed"]))
    utt, clip = clips.take()
    items = [("alone-stream", "stream", utt, clip)]
    files = special_files(requests, clips)
    for index in range(4 + int(spec.get("alone_utterances", 1))):
        name, samples = next(files)
        items.append((f"alone-file{index}", "file", name, samples))
    return items


def alone_counters(worker: Any) -> dict[str, Any]:
    """The worker's cell replays (their total) and fallbacks by reason, as
    plain values (TXG-OBS-008); none while transaction graphs are off."""
    from support import txg_api

    reading = txg_api.counters(worker)
    if not reading:
        return {"replays": 0, "fallbacks": {}}
    counters = reading["counters"]
    fallbacks: dict[str, int] = {}
    for key, value in dict(counters["fallbacks"]).items():
        reason = str(key[0] if isinstance(key, tuple) else key)
        fallbacks[reason] = fallbacks.get(reason, 0) + int(value)
    replays = int(sum(dict(counters["replays"]).values()))
    return {"replays": replays, "fallbacks": fallbacks}


def alone_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The counters' growth over one item: its cell replays and its
    fallbacks by reason."""
    reasons = set(before["fallbacks"]) | set(after["fallbacks"])
    grown = {
        r: int(after["fallbacks"].get(r, 0)) - int(before["fallbacks"].get(r, 0))
        for r in reasons
    }
    return {
        "replays": int(after["replays"]) - int(before["replays"]),
        "fallbacks": {r: v for r, v in sorted(grown.items()) if v},
    }


def alone_startup(worker: Any, spec: dict[str, Any], probes: Any) -> dict[str, Any]:
    """Read on the worker after startup: the startup probes, the capture
    record, the run record's fields (:func:`run_record`, read after core's
    extension) and the state layer's page count and page bytes."""
    import torch

    from support.txg_qual_adapter import QualAdapter

    adapter = QualAdapter.from_worker(worker)
    ctx = Context(adapter=adapter, model=adapter.model, probes=probes)
    return _plain(
        {
            "probes": probes.reply(),
            "capture_record": capture_record(adapter.model),
            "record": run_record(ctx, spec),
            "state_capacity": int(adapter.capacity),
            "state_page_bytes": adapter.page_bytes,
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        }
    )


async def serve_alone_item(llm: Any, kind: str, samples: np.ndarray) -> dict:
    """One item alone on the idle server, through the client path: a stream
    as core's ``/v1/realtime`` serving feeds it, a file as core's
    transcription serving submits it (``support.served_clients``). Its
    finish reason, token IDs and failed rows (an ``error`` finish), or the
    error the client received."""
    from support import served_clients as served

    try:
        if kind == "stream":
            result = await served.transcribe_stream(llm, samples)
        else:
            result = await served.transcribe_file(llm, samples)
    except Exception as error:  # noqa: BLE001 - the client's error, recorded
        return {
            "finish_reason": None,
            "token_ids": [],
            "failed_rows": 1,
            "error": f"{type(error).__name__}: {error}"[:500],
        }
    return {
        "finish_reason": result.finish_reason,
        "token_ids": [int(t) for t in result.token_ids],
        "failed_rows": int(result.finish_reason == "error"),
        "error": None,
    }


async def run_alone(spec: dict[str, Any]) -> dict[str, Any]:
    """TXG-QUAL-028's served part in a fresh multiprocess engine
    (``AsyncLLM``) with the run's KV-cache sizing: each of
    :func:`alone_items` served alone, in order, each its own request run to
    its end before the next starts, with the worker's counters read before
    and after it (outside the request). Returns the run's ``identity``, the
    worker's ``capture_record`` and startup ``probes``, and per item its
    outcome, token IDs, cell replays and fallbacks by reason (``result``)."""
    import importlib.metadata as md

    import torch
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    import vllm_nemotron_asr_streaming as plugin
    from vllm_nemotron_asr_streaming import register

    os.environ.update(engine_env(spec))
    register()
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**engine_settings(spec), worker_extension_cls=ALONE_READER)
    )
    try:

        async def read() -> dict[str, Any]:
            return (await llm.collective_rpc("alone_counters"))[0]

        worker = (await llm.collective_rpc("alone_startup", args=(spec,)))[0]
        requests = Requests(
            SimpleNamespace(config=llm.model_config.hf_config),
            lookahead=int(spec["lookahead"]),
        )
        corpus = corpus_of(spec)
        items = []
        for name, kind, utt, samples in alone_items(requests, corpus, spec):
            before = await read()
            outcome = await serve_alone_item(llm, kind, samples)
            after = await read()
            items.append(
                {
                    "name": name,
                    "kind": kind,
                    "utt": utt,
                    "samples": int(samples.shape[0]),
                    **outcome,
                    **alone_deltas(before, after),
                }
            )
        scheduler = llm.vllm_config.scheduler_config
    finally:
        llm.shutdown()
    identity = {
        "plugin_file": plugin.__file__,
        "plugin_version": md.version("vllm-nemotron-asr-streaming"),
        "vllm_file": vllm.__file__,
        "vllm_version": md.version("vllm"),
        "torch": str(torch.__version__),
        "device": worker["device"],
        "settings": engine_settings(spec),
        "env": engine_env(spec),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        **worker["record"],
        "state_capacity": worker["state_capacity"],
        "state_page_bytes": worker["state_page_bytes"],
        "locale_row": requests.locale_row,
        "corpus_sha256": corpus.manifest_sha256,
        "corpus_count": corpus.count,
        "git_head": os.environ.get("TXG_QUAL_PLUGIN_HEAD"),
    }
    return {
        "identity": identity,
        "capture_record": worker["capture_record"],
        "probes": worker["probes"],
        "result": {"items": items},
    }


# ---------------------------------------------------------------------------
# Forced fallbacks (TXG-QUAL-007)


#: TXG-QUAL-007's multi-token fallback reason (TXG-COV-014). The specs do
#: not name its counter key's kind and count, so the check reads the reason.
MULTI_TOKEN = "multi-token"


def run_fallbacks(ctx: Any, n: int) -> dict[str, Any]:
    """TXG-QUAL-007: ``K = N + 1`` fresh CHUNK rows, then the same requests as
    ``N + 1`` non-CHUNK rows, then a row with an endpoint policy and a
    forced-EOU candidate (the signature fallbacks of a direct engine client),
    then a multi-token row among one-token rows (TXG-COV-014), which planning
    fails alone. Every declined step's outputs are compared with the
    option-off transaction on a detached pool copy. The run holds
    ``txgcap_qual.mode_max_requests("fallback", ...)`` requests, ``N + 4``;
    the committed pages above the copy's extent are digested before its
    first step and after its last, and a changed digest fails it."""
    from support import txg_api
    from support import txg_transaction as tx

    requests = ctx.requests
    before = _fallbacks(ctx.model)
    out: dict[str, Any] = {"steps": []}
    ctx.monitor.watch_untouched(tx.Pools({"views": list(ctx.adapter.views)}))

    def step(new=(), cached=(), computed=None, tokens=None, label=""):
        blocks = ctx.blocks.of([*(x.req_id for x in new), *cached])
        ctx.monitor.begin_step()
        result = ctx.adapter.step(
            new=list(new),
            cached=list(cached),
            blocks=blocks,
            computed=computed or {},
            tokens=tokens,
        )
        calls = ctx.monitor.calls
        out["steps"].append(
            {
                "label": label,
                "P": len(result.req_ids),
                "calls": [
                    {
                        "kind": c.kind,
                        "covered": c.covered,
                        "compared": c.compared,
                        "problems": c.problems,
                        "K": c.chunk_count,
                    }
                    for c in calls
                ],
                "failed": sorted(result.failed),
            }
        )
        return result

    ids = [f"fb-k{i}" for i in range(n + 1)]
    clips = [ctx.corpus.take() for _ in ids]
    units = {r: requests.units(clip) for r, (_, clip) in zip(ids, clips, strict=True)}
    step(new=[requests.unit(r, units[r][0], "0") for r in ids], label="K=N+1")
    step(cached=ids, computed={r: 1 for r in ids}, label="E=N+1")
    _, clip = ctx.corpus.take()
    policy = requests.unit("fb-policy", requests.units(clip)[0], "0")
    # The policy rides in the sampling parameters' extra_args (REG-STATE-020).
    from support import model_harness as h

    policy = h.with_extra_args(
        policy,
        {
            **(policy.sampling_params.extra_args or {}),
            "endpoint_policy": {
                "mode": "greedy_blank",
                "threshold_frames": 2,
                "residue_frames": 0,
            },
        },
    )
    step(new=[policy], label="endpoint policy")
    # A forced-EOU candidate: the EOU token for a stream the worker serves,
    # parked or not (the planner refuses a non-CHUNK row of an unseen stream).
    eou = requests.special(ids[0], int(requests.config.eou_token_id))
    step(new=[eou], label="forced EOU")
    # A multi-token row among one-token rows: a stream's next chunk and a
    # direct client's two-token request, scheduled whole (last, where core's
    # batch layout puts a row's extra token).
    _, clip = ctx.corpus.take()
    one = requests.unit("fb-one-token", requests.units(clip)[0], "0")
    multi = requests.multi_token("fb-multi-token", requests.label_tokens(2))
    result = step(
        new=[one, multi],
        tokens={multi.req_id: 2},
        label="multi-token row",
    )
    out["multi_token_failed"] = sorted(result.failed)
    after = _fallbacks(ctx.model)
    grown = {
        "|".join(str(x) for x in key): value - before.get(key, 0)
        for key, value in after.items()
        if value != before.get(key, 0)
    }
    expected = {
        "|".join(str(x) for x in txg_api.fallback_key(reason=r, kind=k, count=c)): 1
        for r, k, c in (
            ("chunk-uncovered", txg_api.CHUNK, n + 1),
            ("echo-uncovered", txg_api.ECHO, n + 1),
            ("signature", txg_api.CHUNK, 1),
            ("signature", txg_api.ECHO, 1),
        )
    }
    multi_grown = sum(
        value for key, value in grown.items() if key.split("|")[0] == MULTI_TOKEN
    )
    others = {k: v for k, v in grown.items() if k.split("|")[0] != MULTI_TOKEN}
    problems = [p for s in out["steps"] for c in s["calls"] for p in c["problems"]]
    uncompared = [
        s["label"]
        for s in out["steps"]
        if not any(c["compared"] for c in s["calls"] if c["kind"] == "off")
    ]
    out.update(
        fallbacks_grown=grown,
        fallbacks_expected=expected,
        multi_token_fallbacks=multi_grown,
        problems=problems,
        uncompared=uncompared,
        passed=(
            others == expected
            and multi_grown == 1
            and out["multi_token_failed"] == [multi.req_id]
            and not problems
            and not uncompared
        ),
    )
    finished = list(ctx.blocks.held)
    ctx.monitor.begin_step()
    ctx.adapter.step(finished=finished, blocks={})
    for r in finished:
        ctx.blocks.release(r)
    untouched = ctx.monitor.untouched_problems()
    out["untouched_problems"] = untouched
    out["passed"] = out["passed"] and not untouched
    return out


def _fallbacks(model: Any) -> dict[tuple, int]:
    from support import txg_api

    reading = txg_api.counters(model)
    return dict(reading["counters"]["fallbacks"]) if reading else {}


# ---------------------------------------------------------------------------
# The step rule (TXG-QUAL-010 to -014)


def drain_positions(population: int, drains: int) -> list[int]:
    """The preregistered evenly spaced step positions of ``E`` REPLAY drains
    among the CHUNK rows: ``floor((2j + 1) P / (2E))``, ``j = 0 … E-1``."""
    if drains == 0:
        return []
    out = [((2 * j + 1) * population) // (2 * drains) for j in range(drains)]
    if len(set(out)) != drains:
        raise ValueError(f"positions collide for P={population}, E={drains}")
    return out


class PlanStream:
    """A stream of the step rule's prefix (stepped alone)."""

    def __init__(self, req_id: str, utt: str, units: list) -> None:
        self.req_id = req_id
        self.utt = utt
        self.units = units
        self.delivered = 0
        self.position = 0
        self.busy = False
        self.request = None  # the CHUNK row's unit request
        self.sampled = None  # the drain row's input token


def _one(ctx: Any, stream: PlanStream) -> tuple[Any, Any]:
    """One prefix step of ``stream`` alone; returns (result, call)."""
    if stream.busy:
        new, cached, computed = [], [stream.req_id], {stream.req_id: stream.position}
    else:
        request = ctx.requests.unit(
            stream.req_id, stream.units[stream.delivered], str(stream.delivered)
        )
        stream.delivered += 1
        stream.position = 0
        new, cached, computed = [request], [], {}
    ctx.monitor.begin_step()
    result = ctx.adapter.step(
        new=new,
        cached=cached,
        blocks=ctx.blocks.of([stream.req_id]),
        computed=computed,
        free_mm_hashes=ctx.free_next,
    )
    ctx.free_next = sorted({h for r in new for h in mm_hashes(r)})
    stream.position += 1
    token = result.sampled[stream.req_id]
    stream.busy = token != ctx.requests.park
    call = ctx.monitor.calls[-1] if ctx.monitor.calls else None
    if result.failed:
        raise RuntimeError(f"prefix row {stream.req_id} was rejected")
    return token, call


def _remaining(call: Any) -> int | None:
    if call is None or call.queue_counts is None:
        return None
    return int(call.queue_counts.reshape(-1)[0].item())


def prefix(ctx: Any, kind: str, tag: str, limit: int = 4000) -> PlanStream:
    """Step one stream alone to its target state (TXG-QUAL-011): a CHUNK-row
    stream has no armed echo, zero remaining queue and a regular next chunk
    (it has delivered at least one chunk, so its row is continuing); a
    drain-row stream has an armed echo and at least one remaining label.
    A clip that never reaches the target is ended and the next one taken."""
    for _attempt in range(50):
        utt, clip = ctx.corpus.take()
        units = ctx.requests.units(clip)
        ctx.counter += 1
        stream = PlanStream(f"{tag}-{ctx.counter}", utt, units)
        regular = len(units) - 1
        for _ in range(limit):
            if not stream.busy and stream.delivered >= regular:
                break
            starting_chunk = not stream.busy
            token, call = _one(ctx, stream)
            if kind == "chunk" and not stream.busy and stream.delivered >= 1:
                if stream.delivered < regular and _remaining(call) in (0, None):
                    stream.request = ctx.requests.unit(
                        stream.req_id,
                        units[stream.delivered],
                        f"plan-{stream.delivered}",
                    )
                    stream.chunk_index = stream.delivered
                    return stream
            if kind == "drain" and starting_chunk and stream.busy:
                remaining = _remaining(call)
                if remaining is not None and remaining >= 1:
                    stream.sampled = int(token)
                    stream.chunk_index = stream.delivered - 1
                    return stream
        _finish(ctx, [stream.req_id])
    raise RuntimeError(f"no clip reached the {kind} target")


def _finish(ctx: Any, req_ids: Sequence[str]) -> None:
    ctx.monitor.begin_step()
    ctx.adapter.step(finished=list(req_ids), blocks={}, free_mm_hashes=ctx.free_next)
    ctx.free_next = []
    for r in req_ids:
        ctx.blocks.release(r)


def run_composition(
    ctx: Any,
    k: int,
    e: int,
    *,
    legs: Sequence[str],
    warmup: int,
    counted: int,
) -> dict[str, Any]:
    """The prefix, snapshot and legs of one composition ``(K, E)``. ``legs``
    is the leg order (``OFF``/``ON`` names, e.g. two blocks of OFF1, ON,
    OFF2). Every step runs the one recorded plan from the snapshot, restored
    in place at the blocks of the plan's rows (TXG-QUAL-011)."""
    from support.txg_qual_adapter import BlockSnapshot

    p = k + e
    positions = drain_positions(p, e)
    chunk_streams = [prefix(ctx, "chunk", f"c{k}x{e}") for _ in range(k)]
    drain_streams = [prefix(ctx, "drain", f"d{k}x{e}") for _ in range(e)]
    order: list[str] = []
    ci, di = iter(chunk_streams), iter(drain_streams)
    for position in range(p):
        order.append((next(di) if position in positions else next(ci)).req_id)
    streams = {s.req_id: s for s in [*chunk_streams, *drain_streams]}
    blocks = ctx.blocks.of(order)
    snapshot = BlockSnapshot()
    legs_out = []
    rejected = []
    uncovered = []
    for leg in legs:
        # Every leg but ON runs the option-off transaction.
        off = leg.upper() != "ON"
        samples = []
        for index in range(warmup + counted):
            for s in drain_streams:
                ctx.adapter.sampled[s.req_id] = s.sampled
            ctx.monitor.force_off = off
            ctx.monitor.begin_step()
            result = ctx.adapter.step(
                new=[s.request for s in chunk_streams],
                cached=[s.req_id for s in drain_streams],
                blocks=blocks,
                computed={s.req_id: 1 for s in drain_streams},
                order=order,
                free_mm_hashes=ctx.free_next,
                before_forward=snapshot,
            )
            ctx.free_next = []
            ctx.monitor.force_off = False
            calls = ctx.monitor.calls
            want = "off" if off else "execute"
            call = next((c for c in calls if c.kind == want), None)
            if call is None or (not off and not call.covered):
                uncovered.append((leg, index))
            if result.failed or (
                call is not None
                and call.status is not None
                and bool((call.status != 0).any().item())
            ):
                rejected.append((leg, index))
            samples.append(None if call is None else call.seconds * 1000.0)
        legs_out.append({"leg": leg, "ms": samples})
    _finish(ctx, order)
    void = bool(rejected) or bool(uncovered)
    return {
        "K": k,
        "E": e,
        "P": p,
        "order": order,
        "drain_positions": positions,
        "audio": [
            {
                "req": s.req_id,
                "utterance_id": s.utt,
                "chunk_index": s.chunk_index,
                "role": "chunk" if s.request is not None else "drain",
            }
            for s in (streams[r] for r in order)
        ],
        "legs": legs_out,
        "blocks": [blocks[r] for r in order],
        "restores": snapshot.restores,
        "restored": [list(b) for b in snapshot.restored],
        "rejected": rejected[:10],
        "uncovered": uncovered[:10],
        "void": void,
    }


def decide_composition(result: dict[str, Any], warmup: int) -> dict[str, Any]:
    """TXG-QUAL-013 for a composition run with two blocks OFF1, ON, OFF2
    (``txgcap_qual.block_material`` when every leg has 2 warm-up and 21
    counted steps; a shorter run reports the same figures, undecided)."""
    from support import txgcap_qual as q

    legs = [leg["ms"][warmup:] for leg in result["legs"]]
    if len(legs) != 6 or result["void"]:
        return {"material": None, "blocks": []}
    full = warmup == q.WARMUP and all(len(x) == q.COUNTED for x in legs)
    detail = []
    blocks = []
    for first in (0, 3):
        off1, on, off2 = (statistics.median(x) for x in legs[first : first + 3])
        block = q.Block(*legs[first : first + 3]) if full else None
        blocks.append(block)
        detail.append(
            {
                "off1": off1,
                "on": on,
                "off2": off2,
                "delta": min(off1, off2) - on,
                "drift": abs(off1 - off2),
                "material": q.block_material(block) if full else None,
            }
        )
    return {"material": q.material(blocks) if full else None, "blocks": detail}


# ---------------------------------------------------------------------------
# U_api (TXG-QUAL-016)


def _u_api_reading(worker: Any) -> dict[str, Any]:
    """The worker's replay and fallback counters as plain values (TXG-OBS-008),
    read on the worker through ``AsyncLLM.collective_rpc`` as the named worker
    method of :class:`UApiReader`."""
    from support import txg_api

    reading = txg_api.counters(worker)
    if not reading:
        return {"replays": [], "fallbacks": 0}
    counters = reading["counters"]
    return {
        "replays": [[list(k), int(v)] for k, v in dict(counters["replays"]).items()],
        "fallbacks": int(sum(dict(counters["fallbacks"]).values())),
    }


class UApiReader:
    """The worker extension of a ``U_api`` leg (core's
    ``--worker-extension-cls``, :data:`U_API_READER`): it gives the worker the
    method ``u_api_reading``, so the leg reads the counters by name through
    ``AsyncLLM.collective_rpc``. Core's message serializer sends a method name
    but refuses a function unless insecure serialization is on, which the leg
    does not set. The worker process imports this module, so its
    ``PYTHONPATH`` names ``tests``."""

    def u_api_reading(self) -> dict[str, Any]:
        return _u_api_reading(self)


#: The qualified name core resolves in the worker process (spelled out: the
#: driver runs as ``__main__``).
U_API_READER = "support.txg_qual_driver.UApiReader"


def u_api_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    """Counter growth over one repeat: replays of the chunk and echo cells
    at count 1, replays of every other cell, and fallbacks. A cell key is
    ``(geometry_index, kind, count)``."""
    old = {tuple(k): v for k, v in before["replays"]}
    new = {tuple(k): v for k, v in after["replays"]}
    grown = {k: new.get(k, 0) - old.get(k, 0) for k in set(old) | set(new)}

    def at_one(kind: str) -> int:
        return sum(v for k, v in grown.items() if k[1] == kind and int(k[2]) == 1)

    chunk, echo = at_one("chunk"), at_one("echo")
    return {
        "chunk1": chunk,
        "echo1": echo,
        "other_replays": sum(grown.values()) - chunk - echo,
        "fallbacks": int(after["fallbacks"]) - int(before["fallbacks"]),
    }


async def u_api_repeat(llm: Any, requests: Requests, samples: np.ndarray) -> dict:
    """One repeat: a fresh stream opened with ``AsyncLLM.generate`` and an
    input generator that yields ``StreamingInput`` units the plugin's packer
    built, timed from before the generator yields the stream's first unit (a
    fresh stream's first chunk, not final) to the arrival of that unit's
    parked output (the output whose new token IDs end with the park token).
    The stream's admission, the counter reads, its close and its terminal
    output are outside the timer."""
    import asyncio
    import uuid

    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    units = requests.packer().append(samples)
    if not units:
        raise ValueError("the clip is shorter than one chunk")
    unit = units[0]

    async def read() -> dict[str, Any]:
        return (await llm.collective_rpc("u_api_reading"))[0]

    before = await read()
    parked = asyncio.Event()
    marks: dict[str, float] = {}

    async def inputs():
        marks["start"] = time.perf_counter()
        yield StreamingInput(prompt=unit.prompt)
        await parked.wait()

    params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=REALTIME_MAX_TOKENS,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=False,
    )
    tokens: list[int] = []
    parks = errors = 0
    async for out in llm.generate(inputs(), params, f"uapi-{uuid.uuid4()}"):
        completion = out.outputs[0]
        ids = [int(t) for t in completion.token_ids]
        if not parks:
            tokens.extend(ids)
        if ids[-1:] == [requests.park]:
            parks += 1
            marks.setdefault("stop", time.perf_counter())
            parked.set()
        if completion.finish_reason == "error":
            errors += 1
    after = await read()
    stop = marks.get("stop", time.perf_counter())
    return {
        "ms": (stop - marks["start"]) * 1000.0,
        "parks": parks,
        "errors": errors,
        "tokens": len(tokens),
        "deltas": u_api_deltas(before, after),
    }


async def run_u_api_leg(spec: dict[str, Any]) -> dict[str, Any]:
    """One ``U_api`` leg in a fresh multiprocess engine (``AsyncLLM``):
    repeats 0 to 3 (repeat 0 discarded by the rule), each on the first chunk
    of the next clip. ``spec["arm"]`` is ``"on"`` or ``"off"``;
    ``txgcap_qual.u_api_schedule`` orders the legs and their washouts."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from support import txgcap_qual as q
    from vllm_nemotron_asr_streaming import register

    os.environ.update(engine_env(spec))
    register()
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**engine_settings(spec), worker_extension_cls=U_API_READER)
    )
    try:
        requests = Requests(
            SimpleNamespace(config=llm.model_config.hf_config),
            lookahead=int(spec["lookahead"]),
        )
        corpus = corpus_of(spec)
        repeats = []
        for _ in range(q.U_API_REPEATS):
            _, clip = corpus.take()
            repeats.append(await u_api_repeat(llm, requests, clip))
    finally:
        llm.shutdown()
    leg = {
        "arm": spec["arm"],
        "leg": spec.get("leg"),
        "washout": bool(spec.get("washout")),
        "repeats": repeats,
    }
    leg["problems"] = q.u_api_leg_problems(leg)
    return leg


def corpus_of(spec: dict[str, Any]) -> Corpus:
    """The spec's corpus: LibriSpeech test-clean under ``spec["corpus"]``,
    or seeded noise clips of ``spec["synthetic_lengths"]`` samples."""
    if spec.get("corpus"):
        return Corpus(spec["corpus"], int(spec["seed"]))
    return SyntheticCorpus(int(spec["seed"]), spec["synthetic_lengths"])


# ---------------------------------------------------------------------------
# The process


class Context(SimpleNamespace):
    pass


def run_sizing(spec: dict[str, Any]) -> str:
    """The run's KV-cache sizing (``txg_api.Sizing``): the spec's
    ``kv_cache_sizing``; else a fixed block count or budget when the spec
    names one; else a fixed block count for the step rule (TXG-QUAL-010) and
    core's default, the extensible KV cache, for every other mode
    (TXG-QUAL-001)."""
    from support.txg_api import Sizing

    if spec.get("kv_cache_sizing"):
        sizing = str(spec["kv_cache_sizing"])
        if sizing not in Sizing.ALL:
            raise ValueError(f"unknown KV-cache sizing {sizing!r}")
        return sizing
    if spec.get("kv_cache_memory_bytes"):
        return Sizing.BUDGET
    if spec.get("num_gpu_blocks_override") or spec.get("mode") == "steprule":
        return Sizing.BLOCK_COUNT
    return Sizing.EXTENSIBLE


def engine_settings(spec: dict[str, Any]) -> dict[str, Any]:
    """The run's ``AsyncEngineArgs`` keywords: the restricted startup under
    the run's KV-cache sizing (:func:`run_sizing`); under a fixed block count
    ``--num-gpu-blocks-override`` is the spec's, or the admission limit plus
    one. ``B_hook`` is the same spec with ``transaction_graph_populations``
    absent from ``hf_overrides``: its KV-cache sizing is the option-on run's.
    A spec with ``headroom`` (TXG-QUAL-029) also sets the
    ``--gpu-memory-utilization`` that leaves the preregistered headroom free
    of a device of ``device_total_bytes``; it is refused with any sizing but
    the extensible KV cache. A run that compares with a detached copy
    (``compare`` in the ``qm`` or ``fallback`` mode) under the extensible KV
    cache is refused without a preregistered headroom. A ``fallback`` spec
    whose ``max_num_seqs`` is below ``N + 1`` is refused: its ``K = N + 1``
    step would exceed the commit sink's row bound."""
    from support import txgcap_qual as q
    from support.txg_api import Sizing

    max_num_seqs = int(spec["max_num_seqs"])
    active = spec.get("max_num_active_seqs")
    limit = max_num_seqs if active is None else int(active)
    sizing = run_sizing(spec)
    blocks = budget = None
    if sizing == Sizing.BLOCK_COUNT:
        blocks = int(spec.get("num_gpu_blocks_override") or limit + 1)
    elif sizing == Sizing.BUDGET:
        budget = int(spec["kv_cache_memory_bytes"])
    settings = dict(
        model=spec["card"],
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        max_model_len=int(spec["max_model_len"]),
        max_num_seqs=max_num_seqs,
        enforce_eager=bool(spec.get("enforce_eager", False)),
        hf_overrides=dict(spec["hf_overrides"]),
        **q.restricted_startup(),
        **q.kv_cache_sizing(sizing, blocks=blocks, budget=budget),
    )
    if active is not None:
        settings["max_num_active_seqs"] = int(active)
    settings.update(spec.get("engine_extra") or {})
    if spec.get("mode") == "fallback":
        rows = int(spec.get("n") or 0) + 1
        if max_num_seqs < rows:
            raise ValueError(
                f"the fallback steps' K = N + 1 step holds {rows} rows, more than "
                f"max_num_seqs {max_num_seqs}, which bounds the commit sink's rows: "
                "set max_num_seqs to at least N + 1 (TXG-QUAL-007)"
            )
    headroom = spec.get("headroom")
    max_requests = q.spec_max_requests(spec)
    if (
        not headroom
        and spec.get("compare")
        and max_requests is not None
        and sizing == Sizing.EXTENSIBLE
    ):
        raise ValueError(
            f"a {spec.get('mode')} run that compares with a detached copy under "
            "the extensible KV cache needs a preregistered headroom: set the "
            "spec's headroom to txgcap_qual.preregistered_headroom of the "
            "mode's most requests (txgcap_qual.spec_max_requests) and the "
            "card's bytes per page (TXG-QUAL-029)"
        )
    if headroom:
        if sizing != Sizing.EXTENSIBLE or "kv_cache_memory_bytes" in settings:
            raise ValueError(
                "the exactness runs' headroom is left through "
                "--gpu-memory-utilization under the extensible KV cache, never "
                "a fixed --kv-cache-memory-bytes (TXG-QUAL-029)"
            )
        problems = q.preregistration_problems(headroom, max_requests)
        if problems:
            raise ValueError(
                "the exactness run's headroom is not preregistered as "
                "TXG-QUAL-029 computes it: " + "; ".join(problems)
            )
        settings["gpu_memory_utilization"] = q.headroom_utilization(
            int(headroom["copy_bytes"]),
            int(headroom["margin_bytes"]),
            int(spec["device_total_bytes"]),
        )
    return settings


def card_page_bytes(card: str | os.PathLike[str]) -> int:
    """The state page's bytes for ``card``, read before any engine starts
    from the card's configuration through the plugin's parser and state
    manifest: the bytes per page of the state layer's bound views, which the
    plugin checks against core's state page (TXG-QUAL-029)."""
    from vllm_nemotron_asr_streaming.card_parser import (
        NemotronAsrStreamingConfigParser,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.state_profile import (
        manifest_page_bytes,
    )

    _, config = NemotronAsrStreamingConfigParser().parse(
        str(card), trust_remote_code=False
    )
    return manifest_page_bytes(config)


def engine_env(spec: dict[str, Any]) -> dict[str, str]:
    """The run's environment: core's graph-memory estimate enabled
    (TXG-QUAL-010, -021, -025) unless the spec sets it."""
    from support import txgcap_qual as q

    return {**q.estimate_enabled(), **dict(spec.get("env") or {})}


def _worker(worker: Any) -> Any:
    return worker


class StartupProbes:
    """Readings taken while the engine starts in this process (TXG-QUAL-001,
    TXG-QUAL-028, TXG-QUAL-029): the plugin's log lines at INFO and above
    (``records``); the torch memory limit core sets, and logs, when it sizes
    the extensible KV cache (``torch_memory_limit``, bytes; read at core's
    setter, since the engine's start reapplies vLLM's logging configuration,
    which drops handlers added to ``vllm`` loggers); at each capture's entry
    (``captures``, by ``"profiling"`` and ``"serving"``) the extensible KV
    cache's committed block count (``None`` when it is off) and the state
    layer's bound views; per capture, the sorted distinct block IDs of every
    capture-only row plan (``capture_rows``); and after core's extension
    (``extension``) the same views, the committed and configured block counts
    and the binding's latch. A view is plain values
    ``[address, shape, strides, dtype]``."""

    def __init__(self) -> None:
        self.records: list[list[str]] = []
        self.torch_memory_limit: int | None = None
        self.captures: dict[str, dict[str, Any]] = {}
        #: Per capture, the block IDs of each capture-only plan the cell
        #: programs pass to ``prepare_model_rows`` (TXG-CAP-005, -017).
        self.capture_rows: dict[str, list[list[int]]] = {}
        self.extension: dict[str, Any] | None = None
        self._in_capture: str | None = None
        self._runners: list[Any] = []

    def install(self) -> StartupProbes:
        import logging

        probes = self

        class Plugin(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                probes.records.append([record.levelname, record.getMessage()])

        plugin = logging.getLogger("vllm_nemotron_asr_streaming")
        plugin.addHandler(Plugin(logging.INFO))
        if plugin.getEffectiveLevel() > logging.INFO:
            plugin.setLevel(logging.INFO)
        self._wrap_core()
        return self

    @staticmethod
    def views(state: Any) -> dict[str, list[Any]]:
        layer = state.model._persistent_state_layer
        return {
            name: [
                int(view.data_ptr()),
                list(view.shape),
                list(view.stride()),
                str(view.dtype),
            ]
            for name, view in layer.component_pools.items()
        }

    def _committed(self) -> int | None:
        for runner in self._runners:
            cache = getattr(runner, "extensible_kv_cache", None)
            if cache is not None:
                return int(cache.num_committed_blocks)
        return None

    def _wrap_core(self) -> None:
        from vllm.v1.worker import extensible_kv_cache, gpu_worker
        from vllm.v1.worker.gpu import model_runner

        from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
            NemotronASRModelState,
        )

        probes = self
        init = model_runner.GPUModelRunner.__init__

        def runner_init(runner: Any, *args: Any, **kwargs: Any) -> None:
            init(runner, *args, **kwargs)
            probes._runners.append(runner)

        model_runner.GPUModelRunner.__init__ = runner_init
        cache_class = extensible_kv_cache.ExtensibleKVCache
        set_limit = cache_class.set_torch_memory_limit

        def set_torch_memory_limit(cache: Any, num_bytes: int | None) -> None:
            # Core lifts the cap (None) when the cache is freed; keep the
            # limit it set when sizing the cache (TXG-QUAL-029).
            if num_bytes is not None:
                probes.torch_memory_limit = int(num_bytes)
            set_limit(cache, num_bytes)

        cache_class.set_torch_memory_limit = set_torch_memory_limit
        capture = NemotronASRModelState.capture_cudagraphs

        def capture_cudagraphs(state: Any, *args: Any, **kwargs: Any) -> Any:
            name = "profiling" if kwargs.get("profile_only") else "serving"
            if name not in probes.captures:
                probes.captures[name] = {
                    "num_committed_blocks": probes._committed(),
                    "views": probes.views(state),
                }
            probes._in_capture = name
            try:
                return capture(state, *args, **kwargs)
            finally:
                probes._in_capture = None

        NemotronASRModelState.capture_cudagraphs = capture_cudagraphs
        from support.txg_api import CAPTURE_ONLY_KEYWORD, advance_module

        advance = advance_module()
        prepare = advance.prepare_model_rows

        def prepare_model_rows(*args: Any, **kwargs: Any) -> Any:
            if probes._in_capture is not None and kwargs.get(CAPTURE_ONLY_KEYWORD):
                plan = args[3] if len(args) > 3 else kwargs.get("plan")
                blocks = sorted({int(b) for b in plan.state_indices_p.tolist()})
                probes.capture_rows.setdefault(probes._in_capture, []).append(blocks)
            return prepare(*args, **kwargs)

        advance.prepare_model_rows = prepare_model_rows
        extend = gpu_worker.Worker.extend_kv_cache

        def extend_kv_cache(worker: Any, num_blocks: int) -> None:
            reading: dict[str, Any] = {"num_blocks": int(num_blocks)}
            probes.extension = reading
            try:
                extend(worker, num_blocks)
            except BaseException as error:
                reading["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                runner = worker.model_runner
                state = runner.model_state
                binding = getattr(state.model, "_transaction_graph_binding", None)
                reading["latched"] = None if binding is None else bool(binding.failed)
                reading["views"] = probes.views(state)
                reading["num_committed_blocks"] = probes._committed()
                config = getattr(runner, "kv_cache_config", None)
                reading["kv_cache_config_blocks"] = getattr(config, "num_blocks", None)

        gpu_worker.Worker.extend_kv_cache = extend_kv_cache

    def page_counts(self) -> dict[str, int]:
        """Each capture layout's page count, as the instrumentation read it:
        the first dimension of the views bound at the capture's entry."""
        return {
            name: int(next(iter(reading["views"].values()))[1][0])
            for name, reading in self.captures.items()
            if reading["views"]
        }

    def reply(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "torch_memory_limit": self.torch_memory_limit,
            "captures": self.captures,
            "capture_rows": self.capture_rows,
            "extension": self.extension,
        }


def device_context() -> dict[str, Any]:
    """Device-level memory figures, recorded as context only and used by no
    check (TXG-QUAL-029): ``torch.cuda.mem_get_info`` and ``nvidia-smi``'s
    used and total memory."""
    import subprocess

    import torch

    out: dict[str, Any] = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["mem_get_info"] = [int(free), int(total)]
    try:
        out["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        out["nvidia_smi"] = f"unavailable: {error}"
    return out


def start(spec: dict[str, Any]) -> tuple[Any, Any]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.update(engine_env(spec))
    if spec.get("headroom") and not spec.get("device_total_bytes"):
        import torch

        spec["device_total_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    from vllm_nemotron_asr_streaming import register

    register()
    engine = LLMEngine.from_engine_args(AsyncEngineArgs(**engine_settings(spec)))
    worker = engine.collective_rpc(_worker)[0]
    return engine, worker


def identity(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    import importlib.metadata as md

    import torch
    import vllm

    import vllm_nemotron_asr_streaming as plugin
    from support.txg_qual_adapter import adapter_identity

    out = {
        "plugin_file": plugin.__file__,
        "plugin_version": md.version("vllm-nemotron-asr-streaming"),
        "vllm_file": vllm.__file__,
        "vllm_version": md.version("vllm"),
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "settings": engine_settings(spec),
        "env": engine_env(spec),
        "adapter_identity": adapter_identity(),
        "max_num_batched_tokens": int(
            ctx.adapter.vllm_config.scheduler_config.max_num_batched_tokens
        ),
        **run_record(ctx, spec),
        "state_capacity": int(ctx.capacity),
        "state_page_bytes": ctx.adapter.page_bytes,
        "locale_row": ctx.requests.locale_row,
        "corpus_sha256": ctx.corpus.manifest_sha256,
        "corpus_count": ctx.corpus.count,
        "git_head": os.environ.get("TXG_QUAL_PLUGIN_HEAD"),
        # The exactness margin's allocator terms assume PyTorch's default
        # caching-allocator configuration (TXG-QUAL-029); the record shows it.
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    return out


def run_record(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """TXG-QUAL-001's fields of the run's record, read after core's
    extension: the KV-cache sizing, the effective cache values, each capture
    layout's page count (the capture record's), and the state group's final
    bound count, the first dimension of the state layer's bound views, which
    is a field separate from the page counts (``None`` unless the extensible
    KV cache is on)."""
    from support import txg_api
    from support.txg_api import Record, Sizing

    cache = ctx.adapter.vllm_config.cache_config
    sizing = run_sizing(spec)
    counts: dict[str, Any] = {}
    reading = capture_record(ctx.model) or {}
    record = reading.get(Record.RECORD) or {}
    for capture in (Record.PROFILING, Record.SERVING):
        figures = record.get(capture)
        if isinstance(figures, dict) and Record.PAGE_COUNT in figures:
            counts[capture] = figures[Record.PAGE_COUNT]
    probes = getattr(ctx, "probes", None)
    if not counts and probes is not None:
        # Option off (B_hook): the instrumentation's reading of the views
        # bound at each capture.
        counts.update(probes.page_counts())
    return {
        "kv_cache_sizing": sizing,
        "effective_cache": {
            "num_gpu_blocks_override": getattr(cache, "num_gpu_blocks_override", None),
            "kv_cache_memory_bytes": getattr(cache, "kv_cache_memory_bytes", None),
            "enable_extensible_kv_cache": getattr(
                cache, "enable_extensible_kv_cache", None
            ),
        },
        "capture_page_counts": counts,
        "final_bound_count": int(ctx.adapter.capacity)
        if sizing == Sizing.EXTENSIBLE
        else None,
        "final_committed_blocks": int(ctx.adapter.final_committed_blocks),
        "plugin_final_bound_count": record.get(txg_api.Record.FINAL_BOUND_COUNT),
    }


def capture_record(model: Any) -> dict[str, Any] | None:
    try:
        from support import txg_api

        reading = txg_api.counters(model)
    except Exception:  # noqa: BLE001 - an older plugin has no counters
        return None
    if not reading:
        return None
    return _plain(reading)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            (
                k
                if isinstance(k, str)
                else "|".join(map(str, k))
                if isinstance(k, tuple)
                else str(k)
            ): _plain(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None:
        return value
    # The exact builtin type: core's message serializer refuses a subclass
    # (``torch.__version__`` is a ``str`` subclass).
    for kind in (bool, int, float, str):
        if isinstance(value, kind):
            return kind(value)
    return repr(value)


def build_context(spec: dict[str, Any], engine: Any, worker: Any) -> Context:
    from support import txgcap_qual as q
    from support.txg_qual_adapter import QualAdapter, TransactionMonitor

    adapter = QualAdapter.from_worker(worker)
    model = adapter.model
    corpus = corpus_of(spec)
    final = adapter.final_committed_blocks
    monitor = TransactionMonitor(
        model,
        compare=bool(spec.get("compare")),
        final_blocks=final,
        # The ``qm`` and ``fallback`` modes copy block 0 and one page per
        # request they can hold (TXG-QUAL-003, -007); construction raises,
        # before the first step, when fewer pages are committed.
        max_requests=q.mode_max_requests(
            spec["mode"],
            admission_limit=admission_limit(adapter),
            n=int(spec.get("n") or 0),
        ),
        memory=bool(spec.get("headroom")),
    )
    return Context(
        spec=spec,
        engine=engine,
        adapter=adapter,
        model=model,
        monitor=monitor.install(),
        requests=Requests(model, lookahead=int(spec["lookahead"]), engine=engine),
        corpus=corpus,
        # Blocks and copies stay below the final committed block count
        # (TXG-QUAL-003, TXG-QUAL-026).
        capacity=final,
        blocks=Blocks(final),
        free_next=[],
        counter=0,
    )


def admission_limit(adapter: Any) -> int:
    """The engine's admission limit (TXG-CFG-020): ``max_num_active_seqs``
    when set, otherwise ``max_num_seqs``."""
    config = adapter.vllm_config.scheduler_config
    active = getattr(config, "max_num_active_seqs", None)
    return int(config.max_num_seqs if active is None else active)


def qm_result(
    runs: list[dict[str, Any]], n: int, untouched: Sequence[str] = ()
) -> dict[str, Any]:
    """The ``qm`` mode's result over its ``Q`` and ``M`` summaries. A changed
    digest of the committed pages above the copy's extent (``untouched``,
    ``TransactionMonitor.untouched_problems``) counts as a mismatch, so it
    fails the run as a page mismatch does (TXG-QUAL-003, -005)."""
    from support import txgcap_qual as q

    at_n = next((r for r in runs if r["name"] == f"Q{n}"), runs[0])
    counts = {
        "q_split_at_n": at_n["counts"]["split"],
        "q_fresh_among_continuing_at_n": at_n["counts"]["fresh_among_continuing"],
        "m_split_with_file": runs[-1]["counts"]["split_with_file"],
        "m_cases": sorted(runs[-1]["cases"]),
    }
    return {
        "runs": runs,
        "counts": counts,
        "void": q.run_void(counts),
        "untouched_problems": list(untouched),
        "mismatches": sum(r["mismatched_steps"] for r in runs) + len(untouched),
        "fallbacks": sum(r["fallback_steps"] for r in runs),
        "failed_rows": sum(r["failed_rows"] for r in runs),
    }


def run_mode(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    from support import txg_transaction as tx
    from support import txgcap_qual as q

    mode = spec["mode"]
    n = int(spec.get("n") or 0)
    if mode == "qm":
        replay = None
        if spec.get("replay"):
            replay = json.loads(Path(spec["replay"]).read_text())["result"]
        runs = []
        # The committed pages above the copy's extent, digested before the
        # first step (TXG-QUAL-003).
        ctx.monitor.watch_untouched(tx.Pools({"views": list(ctx.adapter.views)}))
        for index, c in enumerate(spec["concurrencies"]):
            name = f"Q{c}"
            durations = replay["runs"][index]["durations"] if replay else None
            w = Workload(
                ctx,
                name=name,
                n_streams=int(c),
                n_files=0,
                steps=int(spec["steps"]),
                # Every concurrency plays the seed's permutation and offsets
                # from the start, as the served Q run does at N.
                seed=int(spec["seed"]),
                durations=durations,
            )
            w.run()
            w.close()
            runs.append(w.summary())
        files = max(1, n // 4)
        durations = replay["runs"][len(runs)]["durations"] if replay else None
        # M's streams and files draw from one permutation (TXG-QUAL-004).
        m_clips = ctx.corpus.clips(int(spec["seed"]))
        m = Workload(
            ctx,
            name="M",
            n_streams=n - files,
            n_files=files,
            steps=int(spec["m_steps"]),
            seed=int(spec["seed"]),
            files=special_files(ctx.requests, m_clips),
            durations=durations,
            clips=m_clips,
        )
        m.run()
        m.close()
        runs.append(m.summary())
        return qm_result(runs, n, ctx.monitor.untouched_problems())
    if mode == "fallback":
        return run_fallbacks(ctx, n)
    if mode == "steprule":
        out = []
        for k, e in spec["compositions"]:
            result = run_composition(
                ctx,
                int(k),
                int(e),
                legs=spec["legs"],
                warmup=int(spec.get("warmup", q.WARMUP)),
                counted=int(spec.get("counted", q.COUNTED)),
            )
            result["decision"] = decide_composition(
                result, int(spec.get("warmup", q.WARMUP))
            )
            out.append(result)
        return {"compositions": out}
    raise ValueError(f"unknown mode {mode!r}")


def main(spec_path: str, out_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    reply: dict[str, Any] = {"spec": spec, "ok": False}
    t0 = time.time()
    if spec["mode"] in ("uapi", "alone"):
        import asyncio

        try:
            if spec["mode"] == "uapi":
                reply["result"] = _plain(asyncio.run(run_u_api_leg(spec)))
            else:
                reply.update(_plain(asyncio.run(run_alone(spec))))
            reply["ok"] = True
        except BaseException as error:  # noqa: BLE001 - reported in the reply
            reply["error"] = {"type": type(error).__name__, "message": str(error)}
            reply["traceback"] = traceback.format_exc()
        reply["elapsed_s"] = round(time.time() - t0, 2)
        Path(out_path).write_text(json.dumps(reply))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if reply["ok"] else 1)
    probes = None
    try:
        probes = StartupProbes().install()
        engine, worker = start(spec)
        reply["ready_s"] = round(time.time() - t0, 2)
        ctx = build_context(spec, engine, worker)
        ctx.probes = probes
        reply["identity"] = identity(ctx, spec)
        reply["capture_record"] = capture_record(ctx.model)
        if spec.get("headroom"):
            reply["exactness_memory"] = {"context_before": device_context()}
        reply["first_counted_at"] = time.time()
        reply["result"] = _plain(run_mode(ctx, spec))
        reply["counters_after"] = capture_record(ctx.model)
        if spec.get("headroom"):
            # TXG-QUAL-029: the allocator figures the checks read, and the
            # device-level figures as context only.
            reply["exactness_memory"].update(
                {
                    "torch_memory_limit": probes.torch_memory_limit,
                    **ctx.monitor.memory_summary(),
                    "context_after": device_context(),
                }
            )
        reply["ok"] = True
    except BaseException as error:  # noqa: BLE001 - reported in the reply
        reply["error"] = {"type": type(error).__name__, "message": str(error)}
        reply["traceback"] = traceback.format_exc()
    if probes is not None:
        reply["probes"] = _plain(probes.reply())
    reply["elapsed_s"] = round(time.time() - t0, 2)
    if "ctx" in locals():
        reply["direct_files"] = ctx.requests.direct_files
    Path(out_path).write_text(json.dumps(reply))
    sys.stdout.flush()
    sys.stderr.flush()
    # The engine's state no longer matches what core scheduled; leave without
    # running its shutdown (the device is released with the process).
    os._exit(0 if reply["ok"] else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
