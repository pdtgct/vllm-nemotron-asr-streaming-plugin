---
parent: high-level-design
prefix: SESS
---

# Session

## Context and Design Philosophy

A session is one live audio stream on one RFC 1 state slot. This component gives protocol
frontends a session API (open, feed, flush, abort, release) over vLLM's admitted-stream client
path. It owns the packer that turns 16 kHz audio into the model's input units, and the projection
from the engine's outputs to transcript text. The same packer serves core's native `/v1/realtime`
through the model's `buffer_realtime_audio` hook, so every streaming path segments audio
identically.

Guiding principles:
- **No unowned engine work.** Every task that talks to the engine belongs to the factory's session
  table from before it starts until its slot has retired, or until the factory is torn down.
- **A transcript is final only after a normal close.** Any other end of a stream fails the session;
  a truncated transcript is never returned as final.
- **One input in flight per session.** Concurrency comes from many sessions, not from pipelining
  one, which keeps retries order-safe and output matching trivial.

## Discovery and construction

Frontends find the factory through the entry-point group `vllm_riva_frontend.session_factories`,
name `nemotron_asr_streaming`, value
`vllm_nemotron_asr_streaming.session:create_session_factory`. Neither package imports the other;
the protocol is structural.

`create_session_factory(engine_client, model_config)`:
- returns `None` when the served architecture is not `Nemotron3_5AsrStreamingForRNNT`;
- raises when the engine client lacks `generate(..., stream_admission=True)` or
  `wait_stream_retired` (typed against `EngineClient` where the engine declares them there, and
  otherwise requiring `AsyncLLM`);
- otherwise returns a `SessionFactory`. `served_cadences` is the set of cadence labels of the
  served lookaheads, `locales` the model's locale table keys, and `default_locale` the locale for a
  stream whose client names none: `auto` when the table has it, else the table's only locale, else
  `None` (the client must name one).

## Protocol

```python
class SessionFactory(Protocol):
    served_cadences: frozenset[str]
    locales: frozenset[str]
    default_locale: str | None
    async def open(self, *, cadence: str, locale: str) -> SessionLease: ...
    async def drain(self, deadline: float) -> list[BaseException]: ...
    def teardown_all(self) -> set[asyncio.Task]: ...

class SessionLease(Protocol):
    async def feed(self, samples: np.ndarray, *, on_accepted: PieceAccepted) -> list[str]: ...
    async def update_locale(self, locale: str) -> None: ...
    async def flush(self) -> str: ...
    async def finish(self) -> None: ...
    async def abort(self) -> None: ...
    async def release(self) -> None: ...
    def teardown(self) -> None: ...
```

`samples` is a one-dimensional `float32` array of 16 kHz mono audio; other shapes or dtypes and
non-finite values raise `ValueError` and leave the lease unchanged; an empty array is accepted (`on_accepted(0)`) and sends nothing. A valid
`feed` appends all its samples, calls `on_accepted` exactly once with their count, sends every unit
they complete in order, and returns only after the last of those units' outcomes; it returns the
cumulative transcript strings, one per unit whose text changed. `flush` returns the final
cumulative transcript.

**Time bounds.** Calls take no deadline argument; the caller bounds them by cancellation (for
example `asyncio.timeout`). Cancelling `feed` or `flush` while a unit may be in
flight leaves the session's position unknown, so the lease becomes **broken**: every later call
except `abort`, `release`, `finish` and `teardown` raises `SessionFailed(reason="cancelled")`.

## Lease states

| State | Entered by | Allowed calls |
| --- | --- | --- |
| `open` | `open` returns | `feed`, `update_locale`, `flush`, `abort`, `teardown` |
| `flushed` | `flush` succeeds | `finish`, `release`, `teardown` |
| `failed` | any failure, or a cancelled call | `abort`, `finish`, `release`, `teardown` |
| `aborted` | `abort` is called, even if that call is then cancelled | `release`, `teardown` |
| `released` | `release` returns | `teardown` |

A lease call checks, in order:
1. **Serialization.** A call made while another call on the same lease is running raises
   `RuntimeError` (a caller bug) and changes nothing. `teardown()` is exempt: it is synchronous and
   may be called at any time, including during another call.
2. **Closed factory.** After `teardown_all()`, every call other than `teardown()` raises
   `SessionFailed(reason="cancelled")`.
3. **State.** A call not allowed in the current state raises `SessionFailed`: in `failed`, with the
   reason of the failure that put the lease there (for example `cancelled` or `expired`); in any
   other state, with `reason="state"`. `finish` awaits the end of the `generate()` task, which in `aborted`
   may still be ending after its cancellation; `release` after it has returned returns without effect.

The cleanup calls `abort`, `finish`, `release` and `teardown` never raise a failure already
reported through the lease. `finish` returns `None` once the task has ended, retrieving and not
re-raising the task's exception.

## Engine binding

Each session is one admitted stream: `generate(input_generator, sampling_params, caller_id,
stream_admission=True)` running as a factory-owned task.

- **Caller ID.** A fresh `uuid4` per session, never reused.
- **Sampling.** `n=1`, `temperature=0`, `max_tokens = realtime_max_tokens` (the largest per-unit
  emission bound over all geometries: frames per chunk × symbols per frame + 2 for the EOU and park
  tokens, 142 for the implemented geometries), delta output, special tokens skipped. The stream
  ends on any non-EOS stop, so a smaller bound would end the session.
- **Inputs.** The input generator yields one `StreamingInput` per packed unit, and only after the
  previous unit's outcome has arrived. The engine numbers inputs (`update_seq`); the session never
  builds admission contexts. Exhausting the generator is the close.
- **Outcome of a unit.** Either its parked output, which is complete at the output whose new token
  IDs end with the park token, or a rejection output carrying `stream_rejection`, whose
  `input_index` is the generator position of the rejected unit. Core numbers every yielded input,
  so a resent unit occupies a new position; the session matches outcomes by position, never by
  `chunk_sequence`.

## Lease calls

| Call | Behaviour |
| --- | --- |
| `open(cadence, locale)` | Refuse a cadence not in `served_cadences` (`refused`, `cadence`) or a locale not in `locales` (`refused`, `locale`). Mint the caller ID, insert the session record, start the `generate()` task. Admission is signalled by the input generator's first read. Return the lease on that signal. If the task ends first: `StreamAdmissionError` → `refused` with its reason; `EngineDeadError` → `failed` (`engine_dead`); another exception or a normal end → `failed` (`abnormal_end`). |
| cancellation of `open` | Cancel the `generate()` task and hand the session to the factory's cleanup (below), which owns it through the cancellation and the retirement-wait outcome. The record is removed at once only if the task never started. Once `open` has returned the lease, the caller owns cleanup. |
| stream end while a unit is outstanding | If the `generate()` task ends while `feed` or `flush` awaits a unit's outcome, the lease fails with `engine_dead` for `EngineDeadError`, otherwise `abnormal_end` (an `error` or `abort` finish, or any end without the unit's park). |
| `feed(samples, on_accepted)` | Append to the packer and call `on_accepted(len(samples))`. For each complete unit: send it, await its outcome, project it. Return the cumulative transcript after each unit whose text changed; with no complete unit, return `[]` at once. |
| rejection of a unit | `engine_paused`: resend after a backoff (a factory parameter; default 50 ms, doubling to at most 1 s), until the caller cancels. `mm_cache_miss`: resend once. Any other reason, or a second `mm_cache_miss`: the lease fails with `rejected_input`. A resent unit keeps its `chunk_sequence` and timestamp. |
| `update_locale(locale)` | Refuse a locale not in `locales` (`refused`, `locale`), leaving the lease open with its current locale; otherwise every unit packed after the call, including one built from samples already buffered and the final tail, uses the new locale. |
| `flush()` | Pack and send the final-tail unit, await its outcome, end the input generator, and await the stream's end. Success only for a normal close: `finished=True`, finish reason `stop`, after our close; returns the final cumulative transcript. Any other end fails the lease: a rejected final tail (`rejected_input`); an `error` or `abort` finish (preemption, a failed state step, a rejected close, which core turns into an abort) or any other end (`abnormal_end`); a dead engine (`engine_dead`); cancellation (`cancelled`). |
| `finish()` | Await the `generate()` task's single end. |
| `abort()` | Cancel the `generate()` task (the engine client then aborts the request), move the lease to `aborted`, and await the task, bounded by the caller. The lease is `aborted` before the wait, so a caller whose bound cancels `abort` can still call `release`, which keeps the session owned until the task ends. |
| `release()` | Await `wait_stream_retired(caller_id)`, bounded by the caller, and resolve it as in Factory ownership. `EngineDeadError` removes the record and fails `release` with `engine_dead`, a cleanup fault the frontend escalates, unless the lease already failed with `engine_dead`; then `release` removes the record and returns. A timeout hands the session to the factory's cleanup. |
| `teardown()` | Cancel the `generate()` task; no awaits and no direct engine calls. |

**Engine aborts while idle.** The engine aborts an admitted stream that is never fed, parked idle
or draining past its configured deadlines; an operator abort looks the same to the client. When the
`generate()` task ends with an `abort` finish while the session has no unit outstanding, no `flush`
is in progress and its input is not closed, the lease fails with `expired` (the engine ended an idle stream; deadlines are the
usual cause), and the next call raises it. An idle end by `EngineDeadError` fails it with
`engine_dead`, and any other idle end with `abnormal_end`.

## Factory ownership

- **Session table.** One record per session, keyed by caller ID, inserted before `generate()`
  starts. Its admission state is `pending` until the outcome is known, then `accepted` (the input
  generator's first read) or `refused` (`StreamAdmissionError` from the task). A task that is
  cancelled, or ends with another exception, before either signal leaves the state `pending`: core
  may still have admitted the stream and will abort it.
- **Retirement wait.** For any record not `refused`, cleanup awaits `wait_stream_retired`. Its
  terminal results: it returns (the slot retired); `StreamAdmissionError` (admission was refused);
  `ValueError` (admission was never sent, or the retirement already happened and is no longer
  tracked); `EngineDeadError` (the engine died, reported as a cleanup fault). Every terminal result
  removes the record; a `refused` record is removed at once.
- **Cleanup tasks.** A session whose owner is gone (a cancelled or failed `open`, a `release` that
  timed out) gets one tracked cleanup task running the retirement wait with no deadline of its own.
  Cleanup tasks run until they finish or the factory is torn down.
- **Cleanup faults.** A cleanup task whose retirement wait raises `EngineDeadError`, or ends with an
  unexpected exception, records the exception as a cleanup fault and logs it. No caller is waiting
  for an orphaned session, so faults are reported through `drain`.
- **`drain(deadline)`** awaits the cleanup tasks until `deadline`, an absolute time on
  `time.monotonic()`, and returns the cleanup faults recorded since the previous `drain`, clearing
  them; tasks still running stay owned by the factory.
- **`teardown_all()`** is terminal: it sets `closed` synchronously, cancels every task it owns and
  returns them. After it, `open` raises (`failed`, `cancelled`), pending-open cancellation paths
  only cancel and drop (never start cleanup), and lease calls other than `teardown()` raise. Every
  cancelled task carries a done-callback that retrieves its exception.

## Error contract

Exceptions raised by `open` or a lease carry two attributes the frontend reads with `getattr`:
- `session_error = "refused"`, with `reason` one of `capacity`, `active_limit`, `engine_paused`,
  `request_id_in_use`, `cadence`, `locale`;
- `session_error = "failed"`, with `reason` one of `rejected_input`, `abnormal_end`, `expired`,
  `engine_dead`, `cancelled`, `output_capacity`, `input_capacity`, `state`.

The plugin defines `SessionRefused` and `SessionFailed` carrying them. Other exceptions
(`ValueError` for bad audio, `RuntimeError` for concurrent calls) carry neither and indicate a
caller bug.

## Packer

The packer turns a stream of samples into units of the model's envelope. One packer belongs to one
session (or one core realtime stream) and is fixed to one geometry.

**Envelope.** Each unit is a prompt of exactly one token, the stream chunk-carrier token
(`blank + 2`), plus one audio row of width 17,927: a 7-value header
`[version=2, valid_samples, geometry_id, final_tail, prompt_index, chunk_sequence,
admission_ms_mod]` followed by samples, zero-padded. Every header value is an integer below 2^24 so
it is exact in float32.

**Segmentation.** For geometry `g` with right context `r`, a regular unit carries exactly
`s_g = 1280 × (r + 1)` samples (1,280 at 80 ms, 8,960 at 560 ms, 17,920 at 1120 ms).
`chunk_sequence` counts units from 0. The final tail is the one unit with `final_tail = 1`; it
carries the remaining samples, strictly fewer than `s_g` and possibly zero, and is always the last
unit.

**Fields.** The row is `float32`. `valid_samples` is the number of samples the unit carries;
`final_tail` is 1 for the final tail and 0 for every other unit. `geometry_id` is the geometry's index in the implemented order. `prompt_index` is the
locale's row in the locale table, as of when the unit is packed. `admission_ms_mod` is the
wall-clock time, in epoch milliseconds modulo 2^24, at which the unit became complete (its last
sample was appended, or `flush` was called for the final tail); a resent unit keeps it. The model
reconstructs each unit's deadline from it, which is valid while a unit is used within 2^24 ms
(about 4.6 hours) of being stamped.

**Limits.** A unit whose `chunk_sequence` would reach 2^24 is not emitted: the packer raises its
capacity error rather than wrap (about 15.5 days at 80 ms). The session maps it to
`SessionFailed(reason="input_capacity")`; on core `/v1/realtime` the error ends the stream.

## Served-set enforcement in the model

The model fixes a stream's geometry from its first unit and fails the row of a later unit whose
geometry differs. It also fails the row of a first unit whose geometry is not in the served set.
A failed row ends the request with finish reason `error`, which a session reports as
`abnormal_end`. Neither plugin path can send such a unit (`open` refuses an unserved cadence, and
`buffer_realtime_audio` packs at the default lookahead, which is served); the check protects direct
clients of the engine.

## Core `/v1/realtime`

The model's `buffer_realtime_audio(audio_stream, input_stream, model_config)` hook runs the packer at
the server's default lookahead, with the locale `auto` when the locale table has it, else its only
locale; a multi-locale table without `auto` makes the hook raise `ValueError` naming the locales.
It yields one prompt per unit and the final tail when `audio_stream` ends. Core puts every output's
token IDs on `input_stream`; the model needs no generated-token context, so the hook drains that
queue without using it, keeping it from growing for the life of the connection. Core drives this route as a streaming request without stream admission,
so a rejected unit ends that stream; that is core's contract for this route.

## Transcript projection

The session reads the engine's delta outputs:
- **Text.** Core detokenizes with special tokens skipped; the session appends each output's delta
  text to the current transcript, so word-piece spacing is handled by core's detokenizer.
- **End of unit.** The park token ends the unit and carries no text.

The cumulative transcript is held in a bounded transcript (default retained limit 1 MiB of UTF-8
per session); exceeding it fails the lease with `output_capacity`.

## Endpointing

The model's endpoint detector runs only when a request carries an endpoint policy, which the model
reads from a request field that core does not populate. This plugin defines no binding for the
policy on core, and neither the session nor `buffer_realtime_audio` nor the file prompt sets one (core's generic channels, such as sampling `extra_args`, could carry one), so every
stream and file runs with endpointing disabled and no EOU token is emitted. Binding a policy is
deferred until per-utterance finals are designed with the frontend, since a policy without a way to
report segments to the frontend changes nothing it can use. Segment boundaries from EOU tokens are
part of that later design.

## Implementation notes

- **Ownership until both ends.** A session record stays owned by the factory until its
  `generate()` task has ended and its retirement outcome is known; cancelling a real `generate()`
  can itself await the engine's abort, so retirement alone does not release ownership.
- **Close before success.** `flush` accepts a terminal output as success only when it is observed
  after the input generator is exhausted; a terminal output that arrives earlier is an abnormal
  end.
- **Layout.** The session package imports neither vLLM nor the model at import time; the packer
  and `pack_realtime` are pure NumPy, so one implementation serves the factory and core realtime.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Engine path | `generate(..., stream_admission=True)` and `wait_stream_retired` | Direct `engine_core` calls | The admitted path goes through the output processor, numbers inputs and reports retirement |
| Inputs in flight | One per session | Pipelining several | Retries stay order-safe and outputs match trivially; throughput comes from concurrent sessions |
| Rejection retries | `engine_paused`: resend with backoff until cancelled; `mm_cache_miss`: resend once; otherwise fail | A fixed retry count for both; fail on any rejection | A pause lasts as long as it lasts; a cache miss is fixed by one resend; any other lost unit corrupts the transcript |
| Unit timestamp | When the unit became complete, kept on resend | Session admission time | The model derives each unit's deadline from it; a fixed time makes later units look ever more overdue |
| Cancelled calls | Lease becomes broken | Try to resynchronize | A cancelled call may leave a unit in flight with unknown outcome |
| Orphan cleanup faults | Recorded and returned by `drain` | A fault callback given to the factory; logging only | `drain` is already the frontend's shutdown step, and its result reaches the frontend without a new registration path |
| Idle abort | `expired`, whatever its cause | Distinguishing deadlines from operator aborts | Core does not tell the client which deadline fired; both mean the stream ended while idle |
| Text | Core's detokenized delta text | Detokenizing label IDs in the session | Core already detokenizes incrementally and handles word-piece spacing |
| Final tail | One explicit final-tail unit, then close | A flush control token | The model finalizes from the final-tail unit and the close |
| Packer ownership | One implementation shared by the factory and `buffer_realtime_audio` | Separate packers | Two packers could segment the same audio differently |
| Caller IDs | Fresh `uuid4` per session | Caller-chosen, reused IDs | An ID is reserved until retirement on this path |

## Open Questions & Future Decisions

### Deferred
1. **Utterance finals.** This plugin defines neither a binding for the endpoint policy on core nor
   a way for the lease to report segment completions. Both are settled together with the frontend,
   since either alone gives the frontend nothing it can use for per-utterance finals.

## References

- RFC 1 client path: `docs/design/persistent_state.md` (criteria A1-A10, R1-R5) and
  `vllm/v1/engine/async_llm.py` on the RFC 1 branch.
- Core realtime: `SupportsRealtime.buffer_realtime_audio` in
  `vllm/model_executor/models/interfaces.py`; `vllm/entrypoints/speech_to_text/realtime/`.
- Envelope and geometries: `nemotron_asr/advance.py`, `nemotron_asr/manifests.py`; unit timestamps:
  `nemotron_asr/accepted_audio.py`, `nemotron_asr/plan.py`; transcript bound:
  `nemotron_asr/transcript.py`; endpointing: `nemotron_asr/endpointing.py`.
