# SPDX-License-Identifier: Apache-2.0
"""The transaction-level tier 5 qualification driver (TXG-QUAL-002 to -007,
-010 to -015, -019, TXG-CAL-001 to -003).

One process per run: ``python -m support.txg_qual_driver SPEC.json OUT.json``
starts one engine in this process (``LLMEngine`` with
``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so the engine core and its worker live
here), with the configuration the spec names, and drives the worker through
``support.txg_qual_adapter.QualAdapter`` (TXG-QUAL-026). Every pass/fail
decision is ``support.txgcap_qual``'s.

Modes (``spec["mode"]``):

* ``qm``: the qualification workload ``Q`` at each concurrency, then the
  mixed workload ``M``, every step run replayed and as the option-off
  transaction on a copy of the pools (TXG-QUAL-002 to -006). With
  ``replay`` (a ``qm`` output of the option-on run) the clock replays that
  run's recorded step durations, so an option-off (``B_hook``) process runs
  the same schedule and its transcripts can be compared (TXG-QUAL-006).
* ``fallback``: steps with ``K > N``, ``E > N`` and forced signature
  mismatches, each compared with the option-off transaction (TXG-QUAL-007).
* ``steprule``: the step rule for one population (TXG-QUAL-010 to -014);
  its first composition at population 1 is also the calibration run's U(1)
  (TXG-CAL-001).
* ``body``: one leg of the body non-regression check (TXG-QUAL-015):
  option-off one-kind plans at the given populations.

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
# Slots


class Slots:
    """The adapter's slot authority: the lowest free slot, its generation
    raised on every reuse (core's scheduler does this in serving)."""

    def __init__(self, capacity: int) -> None:
        self.free = list(range(1, capacity))
        self.generation: dict[int, int] = {}
        self.handles: dict[str, tuple[int, int]] = {}

    def take(self, req_id: str) -> tuple[int, int]:
        if req_id not in self.handles:
            if not self.free:
                raise RuntimeError("no free state slot")
            self.free.sort()
            slot = self.free.pop(0)
            generation = self.generation.get(slot, 0) + 1
            self.generation[slot] = generation
            self.handles[req_id] = (slot, generation)
        return self.handles[req_id]

    def release(self, req_id: str) -> None:
        handle = self.handles.pop(req_id, None)
        if handle is not None:
            self.free.append(handle[0])


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
            for r in [*(x.req_id for x in new), *cached]:
                ctx.slots.take(r)
            handles = {
                r: ctx.slots.handles[r] for r in [*(x.req_id for x in new), *cached]
            }
            finished = list(self.pending_finish)
            free = sorted({h for r in new for h in mm_hashes(r)})
            ctx.monitor.begin_step()
            result = ctx.adapter.step(
                new=new,
                cached=cached,
                finished=finished,
                handles=handles,
                computed=computed,
                free_mm_hashes=ctx.free_next,
            )
            ctx.free_next = free
            for r in finished:
                ctx.slots.release(r)
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
            ctx.adapter.step(
                finished=finished, handles={}, free_mm_hashes=ctx.free_next
            )
            ctx.free_next = []
            for r in finished:
                ctx.slots.release(r)
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
# Forced fallbacks (TXG-QUAL-007)


def run_fallbacks(ctx: Any, n: int) -> dict[str, Any]:
    """``K = N + 1`` fresh CHUNK rows, then the same requests as ``N + 1``
    non-CHUNK rows, then a row with an endpoint policy and a forced-EOU
    candidate (the signature fallbacks of a direct engine client)."""
    from support import txg_api

    requests = ctx.requests
    before = _fallbacks(ctx.model)
    out: dict[str, Any] = {"steps": []}

    def step(new=(), cached=(), computed=None, label=""):
        for r in [*(x.req_id for x in new), *cached]:
            ctx.slots.take(r)
        handles = {r: ctx.slots.handles[r] for r in [*(x.req_id for x in new), *cached]}
        ctx.monitor.begin_step()
        result = ctx.adapter.step(
            new=list(new), cached=list(cached), handles=handles, computed=computed or {}
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
    policy.additional_information = {
        "endpoint_policy": {
            "mode": "greedy_blank",
            "threshold_frames": 2,
            "residue_frames": 0,
        }
    }
    step(new=[policy], label="endpoint policy")
    # A forced-EOU candidate: the EOU token for a stream the worker serves,
    # parked or not (the planner refuses a non-CHUNK row of an unseen stream).
    eou = requests.special(ids[0], int(requests.config.eou_token_id))
    step(new=[eou], label="forced EOU")
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
    problems = [p for s in out["steps"] for c in s["calls"] for p in c["problems"]]
    uncompared = [
        s["label"]
        for s in out["steps"]
        if not any(c["compared"] for c in s["calls"] if c["kind"] == "off")
    ]
    out.update(
        fallbacks_grown=grown,
        fallbacks_expected=expected,
        problems=problems,
        uncompared=uncompared,
        passed=grown == expected and not problems and not uncompared,
    )
    finished = list(ctx.slots.handles)
    ctx.monitor.begin_step()
    ctx.adapter.step(finished=finished, handles={})
    for r in finished:
        ctx.slots.release(r)
    return out


def _fallbacks(model: Any) -> dict[tuple, int]:
    from support import txg_api

    reading = txg_api.counters(model)
    return dict(reading["counters"]["fallbacks"]) if reading else {}


# ---------------------------------------------------------------------------
# The step rule (TXG-QUAL-010 to -014) and the body check (TXG-QUAL-015)


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
    ctx.slots.take(stream.req_id)
    ctx.monitor.begin_step()
    result = ctx.adapter.step(
        new=new,
        cached=cached,
        handles={stream.req_id: ctx.slots.handles[stream.req_id]},
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
    ctx.adapter.step(finished=list(req_ids), handles={}, free_mm_hashes=ctx.free_next)
    ctx.free_next = []
    for r in req_ids:
        ctx.slots.release(r)


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
    OFF2). Every step runs the one recorded plan from the snapshot."""
    from support.txg_qual_adapter import SlotSnapshot

    p = k + e
    positions = drain_positions(p, e)
    chunk_streams = [prefix(ctx, "chunk", f"c{k}x{e}") for _ in range(k)]
    drain_streams = [prefix(ctx, "drain", f"d{k}x{e}") for _ in range(e)]
    order: list[str] = []
    ci, di = iter(chunk_streams), iter(drain_streams)
    for position in range(p):
        order.append((next(di) if position in positions else next(ci)).req_id)
    streams = {s.req_id: s for s in [*chunk_streams, *drain_streams]}
    handles = {r: ctx.slots.handles[r] for r in order}
    snapshot = SlotSnapshot()
    legs_out = []
    rejected = []
    uncovered = []
    for leg in legs:
        # Every leg but ON runs the option-off transaction (OFF1, OFF2, and
        # the body check's OLD and NEW legs, which run with the option off).
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
                handles=handles,
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
        "restores": snapshot.restores,
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
# The process


class Context(SimpleNamespace):
    pass


def engine_settings(spec: dict[str, Any]) -> dict[str, Any]:
    settings = dict(
        model=spec["card"],
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        max_model_len=int(spec["max_model_len"]),
        max_num_seqs=int(spec["max_num_seqs"]),
        enforce_eager=bool(spec.get("enforce_eager", False)),
        compilation_config={"cudagraph_mode": "NONE", "mode": 0},
        hf_overrides=dict(spec["hf_overrides"]),
    )
    if spec.get("num_gpu_blocks_override"):
        settings["num_gpu_blocks_override"] = int(spec["num_gpu_blocks_override"])
    settings.update(spec.get("engine_extra") or {})
    return settings


def _worker(worker: Any) -> Any:
    return worker


def start(spec: dict[str, Any]) -> tuple[Any, Any]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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

    out = {
        "plugin_file": plugin.__file__,
        "plugin_version": md.version("vllm-nemotron-asr-streaming"),
        "vllm_file": vllm.__file__,
        "vllm_version": md.version("vllm"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "settings": engine_settings(spec),
        "max_num_batched_tokens": int(
            ctx.adapter.vllm_config.scheduler_config.max_num_batched_tokens
        ),
        "state_capacity": int(ctx.capacity),
        "state_page_bytes": int(
            ctx.adapter.bindings.views[ctx.adapter.layer].spec.page_size_bytes
        ),
        "locale_row": ctx.requests.locale_row,
        "corpus_sha256": ctx.corpus.manifest_sha256,
        "corpus_count": ctx.corpus.count,
        "git_head": os.environ.get("TXG_QUAL_PLUGIN_HEAD"),
    }
    return out


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
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def build_context(spec: dict[str, Any], engine: Any, worker: Any) -> Context:
    from support.txg_qual_adapter import QualAdapter, TransactionMonitor

    adapter = QualAdapter.from_worker(worker)
    model = adapter.model
    views = adapter.bindings.views[adapter.layer]
    corpus = Corpus(spec["corpus"], int(spec["seed"]))
    return Context(
        spec=spec,
        engine=engine,
        adapter=adapter,
        model=model,
        monitor=TransactionMonitor(model, compare=bool(spec.get("compare"))).install(),
        requests=Requests(model, lookahead=int(spec["lookahead"]), engine=engine),
        corpus=corpus,
        capacity=int(views.capacity),
        slots=Slots(int(views.capacity)),
        free_next=[],
        counter=0,
    )


def run_mode(ctx: Any, spec: dict[str, Any]) -> dict[str, Any]:
    from support import txgcap_qual as q

    mode = spec["mode"]
    n = int(spec.get("n") or 0)
    if mode == "qm":
        replay = None
        if spec.get("replay"):
            replay = json.loads(Path(spec["replay"]).read_text())["result"]
        runs = []
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
            "mismatches": sum(r["mismatched_steps"] for r in runs),
            "fallbacks": sum(r["fallback_steps"] for r in runs),
            "failed_rows": sum(r["failed_rows"] for r in runs),
        }
    if mode == "fallback":
        return run_fallbacks(ctx, n)
    if mode in ("steprule", "body"):
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
            if mode == "steprule":
                result["decision"] = decide_composition(
                    result, int(spec.get("warmup", q.WARMUP))
                )
            else:
                result["medians"] = [
                    statistics.median(leg["ms"][int(spec.get("warmup", q.WARMUP)) :])
                    for leg in result["legs"]
                ]
            out.append(result)
        return {"compositions": out}
    raise ValueError(f"unknown mode {mode!r}")


def main(spec_path: str, out_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    reply: dict[str, Any] = {"spec": spec, "ok": False}
    t0 = time.time()
    try:
        engine, worker = start(spec)
        reply["ready_s"] = round(time.time() - t0, 2)
        ctx = build_context(spec, engine, worker)
        reply["identity"] = identity(ctx, spec)
        reply["capture_record"] = capture_record(ctx.model)
        reply["first_counted_at"] = time.time()
        reply["result"] = _plain(run_mode(ctx, spec))
        reply["counters_after"] = capture_record(ctx.model)
        reply["ok"] = True
    except BaseException as error:  # noqa: BLE001 - reported in the reply
        reply["error"] = {"type": type(error).__name__, "message": str(error)}
        reply["traceback"] = traceback.format_exc()
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
