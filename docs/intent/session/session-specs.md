# Session specs

Design: [session-design.md](session-design.md). Prefix `SESS`.

A session is one live audio stream served through the plugin's `SessionFactory` on one RFC 1
admitted stream; a lease is the handle `open` returns for it.

## Discovery and construction

- [x] **SESS-FAC-001**: The distribution shall expose an entry point in the group `vllm_riva_frontend.session_factories`, named `nemotron_asr_streaming`, with the value `vllm_nemotron_asr_streaming.session:create_session_factory`.
- [x] **SESS-FAC-002**: When the served architecture is not `Nemotron3_5AsrStreamingForRNNT`, `create_session_factory(engine_client, model_config)` shall return `None`.
- [x] **SESS-FAC-003**: If the engine client given to `create_session_factory` has no `generate` accepting `stream_admission` or no `wait_stream_retired`, then `create_session_factory` shall raise an error naming the missing call.
- [x] **SESS-FAC-004**: The session factory's `served_cadences` shall be the cadence labels of exactly the served lookahead set, using the one table right context 0, 1, 3, 6, 13 ↔ label `80ms`, `160ms`, `320ms`, `560ms`, `1120ms` ↔ geometry index 0 to 4; its `locales` shall be exactly the keys of the served locale table; and its `default_locale` shall be `auto` when the table has it, else the table's only locale, else `None`.
- [x] **SESS-FAC-005**: The plugin shall not import the frontend package, and the session factory and lease shall satisfy the protocol in the session design (§Protocol) structurally.

## Audio input to `feed`

- [x] **SESS-IN-001**: If the samples given to `feed` are not a one-dimensional `float32` array, or contain a non-finite value, then `feed` shall raise `ValueError` and leave the lease's state and buffered audio unchanged.
- [x] **SESS-IN-002**: When `feed` is given an empty array on an open lease, it shall call `on_accepted(0)`, send nothing to the engine and return `[]`.
- [x] **SESS-IN-003**: When `feed` is given valid samples on an open lease, it shall append all of them to the session's packer, call `on_accepted(len(samples))` exactly once before sending any unit built from them, and send every unit they complete, in order, before returning.
- [x] **SESS-IN-004**: When `feed` returns, it shall have received the outcome of every unit it sent, and shall return, in order, the session's cumulative transcript after each of those units whose outcome changed the transcript text (`[]` when it completed no unit, without awaiting the engine).

## Lease states and call discipline

- [x] **SESS-LEASE-001**: The lease shall allow, in state `open`: `feed`, `update_locale`, `flush`, `abort`, `teardown`; in `flushed`: `finish`, `release`, `teardown`; in `failed`: `abort`, `finish`, `release`, `teardown`; in `aborted`: `release`, `teardown`; in `released`: `teardown`.
- [x] **SESS-LEASE-002**: If a lease call is not allowed in the lease's current state, then it shall raise `SessionFailed` with, in state `failed`, the reason of the failure that put the lease there, and in any other state `reason="state"`; except that `finish` after the `generate()` task has ended and `release` after `release` has returned shall return without effect.
- [x] **SESS-LEASE-003**: If a lease call other than `teardown` is made while another call on the same lease is still running, then the second call shall raise `RuntimeError` without changing the lease's state; `teardown` shall be accepted at any time.
- [x] **SESS-LEASE-004**: If `feed` or `flush` is cancelled, then the lease shall enter `failed` with reason `cancelled`.
- [x] **SESS-LEASE-005**: When any session failure occurs on an open lease (a failed unit, an end of the stream while a unit is outstanding, an abnormal end, an idle abort, a dead engine, a transcript or input limit), the lease shall enter `failed` with that failure's reason, and the call that observed it, or the next call other than `abort`, `finish`, `release` and `teardown` if none was running, shall raise the corresponding `SessionFailed`.
- [x] **SESS-LEASE-006**: A lease call shall apply its checks in this order: serialization (SESS-LEASE-003), closed factory (SESS-OWN-006), state (SESS-LEASE-002).
- [x] **SESS-LEASE-007**: The cleanup calls `abort`, `finish`, `release` and `teardown` shall not raise a failure already reported through the lease.

## Opening a session

- [x] **SESS-OPEN-001**: If `open` is given a cadence not in `served_cadences`, then it shall raise `SessionRefused(reason="cadence")`; if given a locale not in `locales`, then it shall raise `SessionRefused(reason="locale")`; in both cases before any engine call.
- [x] **SESS-OPEN-002**: When `open` is called with a served cadence and locale, it shall mint a fresh `uuid4` caller ID, insert the session record in the factory's session table, and only then start the `generate(..., stream_admission=True)` task.
- [x] **SESS-OPEN-003**: When the input generator of a session's `generate()` task is first read, `open` shall return the lease in state `open`.
- [x] **SESS-OPEN-004**: If the `generate()` task raises `StreamAdmissionError` before its input generator is first read, then `open` shall raise `SessionRefused` with that error's reason (`capacity`, `active_limit`, `engine_paused` or `request_id_in_use`).
- [x] **SESS-OPEN-005**: If the `generate()` task raises `EngineDeadError` before its input generator is first read, then `open` shall raise `SessionFailed(reason="engine_dead")`; if it raises any other exception or ends normally before that read, then `open` shall raise `SessionFailed(reason="abnormal_end")`.
- [x] **SESS-OPEN-006**: If `open` is cancelled before it returns, then it shall cancel the session's `generate()` task; if the task never started, it shall remove the session record and start no cleanup task, and otherwise it shall hand the session to exactly one factory cleanup task (SESS-OWN-003).

## Engine binding

- [x] **SESS-ENG-001**: The session's sampling parameters shall be `n=1`, `temperature=0`, `max_tokens=realtime_max_tokens` (142 for the implemented geometries), delta output kind, and `skip_special_tokens=True`.
- [x] **SESS-ENG-002**: The session's input generator shall yield one `StreamingInput` per packed unit, and shall not yield a unit until the previous unit's outcome (SESS-ENG-003) has arrived.
- [x] **SESS-ENG-003**: The session shall take a unit's outcome to be either the output whose new token IDs end with the park token (`blank + 1`), or a rejection output whose `stream_rejection.input_index` equals the 0-based position at which the input generator yielded that unit; a resent unit has a new position, and outcomes shall be matched by position, never by `chunk_sequence`.
- [x] **SESS-ENG-004**: When a session's unit is rejected with reason `engine_paused`, the session shall resend the same unit after a backoff (a factory parameter, default 50 ms doubling to at most 1 s), repeatedly, until the unit is accepted or the caller cancels.
- [x] **SESS-ENG-005**: When a session's unit is rejected with reason `mm_cache_miss`, the session shall resend the same unit once; if the resent unit is rejected with `mm_cache_miss` again, then the lease shall fail with `rejected_input`.
- [x] **SESS-ENG-006**: If a session's unit is rejected with any reason other than `engine_paused` or `mm_cache_miss`, then the lease shall fail with `rejected_input`.
- [x] **SESS-ENG-007**: When the session resends a rejected unit, the resent unit shall be identical to the original, including its `chunk_sequence` and `admission_ms_mod`.
- [x] **SESS-ENG-008**: The session shall never set the engine's input numbering or build stream-admission contexts itself; ending the input generator shall be the session's only way to close the stream.
- [x] **SESS-ENG-009**: If the `generate()` task ends while `feed` or `flush` awaits a unit's outcome, then the lease shall fail with `engine_dead` for `EngineDeadError`, and otherwise with `abnormal_end` (an `error` or `abort` finish, or any end without the unit's park token).

## Locale

- [x] **SESS-LOC-001**: If `update_locale` is given a locale not in `locales`, then it shall raise `SessionRefused(reason="locale")` and leave the lease open with its current locale.
- [x] **SESS-LOC-002**: When `update_locale` succeeds, every unit packed after the call, including units built from samples already buffered and the final tail, shall carry the new locale's row as `prompt_index`.
- [x] **SESS-LOC-003**: `update_locale` shall return `None`.

## Flush, finish, abort, release, teardown

- [x] **SESS-FLUSH-001**: When `flush` is called on an open lease, it shall pack and send the final-tail unit, await its outcome, end the input generator, and await the end of the stream.
- [x] **SESS-FLUSH-002**: When the stream ends with `finished=True` and finish reason `stop` after the session ended its input generator, `flush` shall move the lease to `flushed` and return the final cumulative transcript.
- [x] **SESS-FLUSH-003**: If the stream ends any other way during `flush`, then the lease shall fail: a rejected final tail with `rejected_input` (as SESS-ENG-004 to SESS-ENG-006); an `error` or `abort` finish (including a rejected close, which core turns into an abort) or any other end with `abnormal_end`; a dead engine with `engine_dead`.
- [x] **SESS-FIN-001**: When `finish` is called, it shall await the end of the session's `generate()` task and return `None`, retrieving and not re-raising any exception the task ended with.
- [x] **SESS-ABORT-001**: When `abort` is called, it shall cancel the session's `generate()` task, await its end, and move the lease to `aborted`.
- [x] **SESS-REL-001**: When `release` is called, it shall await `wait_stream_retired(caller_id)` and resolve the result as SESS-OWN-002 does, then move the lease to `released`.
- [x] **SESS-REL-002**: If `wait_stream_retired` raises `EngineDeadError` during `release`, then `release` shall remove the session record and raise `SessionFailed(reason="engine_dead")`, unless the lease already failed with `engine_dead`, in which case it shall remove the record and return.
- [x] **SESS-REL-003**: If `release` is cancelled before the retirement wait finishes, then it shall hand the session to the factory's cleanup (SESS-OWN-003).
- [x] **SESS-TEAR-001**: When `teardown` is called, it shall cancel the session's `generate()` task without awaiting anything and without calling the engine directly.
- [x] **SESS-EXP-001**: When a session's `generate()` task ends while the session has no unit outstanding, no `flush` is in progress and its input is not closed, the lease shall enter `failed` with reason `expired` for an `abort` finish (an engine deadline for a stream never fed, parked idle or draining, or an operator abort), `engine_dead` for `EngineDeadError`, and `abnormal_end` for any other exception, an `error` finish or any other end.

## Factory ownership

- [x] **SESS-OWN-001**: The factory shall keep one session record per session, keyed by caller ID, with admission state `pending` until the input generator's first read (`accepted`) or a `StreamAdmissionError` from the task (`refused`); a task cancelled or failed before either shall leave the state `pending`.
- [x] **SESS-OWN-002**: For every session record whose admission state is not `refused`, the factory's cleanup shall await `wait_stream_retired` and remove the record when it returns or raises `StreamAdmissionError`, `ValueError` or `EngineDeadError`; a `refused` record shall be removed at once.
- [x] **SESS-OWN-008**: When a factory cleanup task's retirement wait raises `EngineDeadError` or any exception other than `StreamAdmissionError` and `ValueError`, the factory shall log it and record it as a cleanup fault.
- [x] **SESS-OWN-003**: When a session's owner is gone (a cancelled or failed `open`, a cancelled `release`), the factory shall start exactly one tracked cleanup task for it that runs SESS-OWN-002 with no deadline of its own.
- [x] **SESS-OWN-004**: When `drain(deadline)` is called with an absolute `time.monotonic()` deadline, the factory shall await its cleanup tasks until they finish or the deadline passes, keep ownership of any still running, and return the cleanup faults recorded since the previous `drain`, clearing them.
- [x] **SESS-OWN-005**: When `teardown_all()` is called, the factory shall mark itself closed before returning, cancel every task it owns, and return the set of those tasks.
- [x] **SESS-OWN-006**: While the factory is closed, `open` shall raise `SessionFailed(reason="cancelled")`, every lease call other than `teardown` shall raise `SessionFailed(reason="cancelled")`, and a cancelled pending `open` shall cancel its task without starting cleanup.
- [x] **SESS-OWN-007**: Every task the factory cancels shall carry a done-callback that retrieves the task's exception.

## Error contract

- [x] **SESS-ERR-001**: `SessionRefused` shall carry `session_error = "refused"` and `reason`, one of `capacity`, `active_limit`, `engine_paused`, `request_id_in_use`, `cadence`, `locale`.
- [x] **SESS-ERR-002**: `SessionFailed` shall carry `session_error = "failed"` and `reason`, one of `rejected_input`, `abnormal_end`, `expired`, `engine_dead`, `cancelled`, `output_capacity`, `input_capacity`, `state`.
- [x] **SESS-ERR-003**: Exceptions from `open` or a lease other than `SessionRefused` and `SessionFailed` (`ValueError` for bad audio, `RuntimeError` for concurrent calls) shall carry neither `session_error` nor `reason`.

## Packer

These specs apply both to the session factory and to core `/v1/realtime` (SESS-RT), which use one
packer implementation.

- [x] **SESS-PACK-001**: Each unit the packer emits shall be a prompt of exactly one token, the stream chunk-carrier token (`blank + 2`), plus one `float32` audio row of width 17,927 holding the 7-value header `[version=2, valid_samples, geometry_id, final_tail, prompt_index, chunk_sequence, admission_ms_mod]` followed by the unit's samples, zero-padded, where `valid_samples` is the number of samples the unit carries and `final_tail` is 0 for every unit except the final tail.
- [x] **SESS-PACK-002**: For a packer fixed to geometry `g` with right context `r`, every unit other than the final tail shall carry exactly `1280 × (r + 1)` samples, and the samples of consecutive units shall be consecutive in the input audio.
- [x] **SESS-PACK-003**: The packer shall set `chunk_sequence` to 0 for its first unit and increase it by 1 for each later unit.
- [x] **SESS-PACK-004**: When the stream is closed (`flush` on a session, or the end of `audio_stream` on core `/v1/realtime`), the packer shall emit exactly one final-tail unit with `final_tail = 1` carrying the remaining samples (fewer than `1280 × (r + 1)`, possibly zero), and shall emit no unit after it.
- [x] **SESS-PACK-005**: The packer shall set `geometry_id` to the geometry's index in the implemented order (80, 160, 320, 560, 1120 ms) and `prompt_index` to the current locale's row in the locale table at the time the unit is packed.
- [x] **SESS-PACK-006**: The packer shall set `admission_ms_mod` to the wall-clock time, in epoch milliseconds modulo 2^24, at which the unit became complete (its last sample was appended, or the stream was closed, for the final tail).
- [x] **SESS-PACK-007**: Every header value the packer writes shall be a non-negative integer below 2^24.
- [x] **SESS-PACK-008**: If a unit would need a `chunk_sequence` of 2^24 or more, then the packer shall raise its capacity error instead of emitting the unit; the session factory shall report it as `SessionFailed(reason="input_capacity")`, and on core `/v1/realtime` `buffer_realtime_audio` shall let the error propagate (ending the stream as failed, never with a normal close).

## Served-set enforcement in the model

- [ ] **SESS-MODEL-001**: When the model receives a stream's first unit, it shall fix the stream's geometry from that unit's `geometry_id`; if a later unit of the stream carries a different `geometry_id`, then the model shall fail that row, ending the request with finish reason `error`.
- [ ] **SESS-MODEL-002**: If a stream's first unit carries a geometry that is not in the served lookahead set, then the model shall fail that row, ending the request with finish reason `error` (reachable only by direct engine clients; both plugin paths send served geometries).

## Core `/v1/realtime`

- [x] **SESS-RT-001**: The model's `buffer_realtime_audio(audio_stream, input_stream, model_config)` shall pack the audio from `audio_stream` with the session packer at the server's default lookahead and yield one prompt per unit, followed by the final-tail unit when `audio_stream` ends.
- [x] **SESS-RT-002**: The model's `buffer_realtime_audio` shall use the same default locale as SESS-FAC-004's `default_locale`; if that is `None`, then it shall raise `ValueError` naming the locales.
- [x] **SESS-RT-003**: The model's `buffer_realtime_audio` shall drain `input_stream` for the life of the connection without using its contents.

## Transcript projection

- [x] **SESS-TXT-001**: The session shall build its cumulative transcript by appending the delta text of each engine output, as detokenized by core with special tokens skipped.
- [x] **SESS-TXT-003**: The session shall treat the park token as the end of the current unit and add no text for it.
- [x] **SESS-TXT-004**: If the session's cumulative transcript would exceed its retained limit (default 1 MiB of UTF-8), then the lease shall fail with `output_capacity`.

## Endpointing

- [x] **SESS-EP-001**: Neither the session factory nor `buffer_realtime_audio` shall set an endpoint policy on the requests they create, so every stream they serve runs with the model's endpointing disabled.
- [D] **SESS-EP-002**: Where per-utterance finals are designed with the frontend, the session shall bind an endpoint policy on core, close a transcript segment at each EOU token (`blank + 3`), and report segment completions to the frontend.
