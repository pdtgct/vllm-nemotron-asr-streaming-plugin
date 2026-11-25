# Transcription specs

Design: [transcription-design.md](transcription-design.md). Prefix `FILE`.

A file request is core's complete-file transcription request (`POST /v1/audio/transcriptions`, or
the frontend's `Recognize` through the same service) for this model. The translation route shares
the same base serving and `SpeechToTextParams`; its requests reach `get_generation_prompt` too and
are covered where stated.

## Interface

- [x] **FILE-IF-001**: The model class shall implement `SupportsTranscription` with `supports_transcription_only = False`, `supports_segment_timestamp = False` and `supports_diarized_transcription = False`.
- [x] **FILE-IF-002**: The model class's `supported_languages` shall be a static mapping whose keys are exactly the two-letter languages of the supported checkpoint's locale table that core's language map (Whisper's `LANGUAGES`) contains.
- [x] **FILE-IF-003**: The model's `get_speech_to_text_config` shall return `sample_rate = 16000`, `max_audio_clip_s = None` and `min_energy_split_window_size = None`.
- [x] **FILE-IF-004**: If a file request's task is not transcription, or it carries a text prompt, hotwords or a target language, then `get_generation_prompt` shall raise `ValueError` naming the unsupported field.
- [x] **FILE-IF-005**: If the served configuration has `experimental_native_burst` true, then `get_generation_prompt` shall raise `ValueError` stating that complete-file transcription is not served in native-burst mode.

## Request guards

Beam search and a completion limit are text-decoding options. They do not apply to audio
transcription with this model, which decodes by a fixed transducer rule, not by searching or
sampling text, and whose token count is an internal step count, not a transcript length. A file
request that sets them is refused with vLLM's usual message, `<field> is not supported for this
model`, which leaves other models free to support the field.

Core hands `get_generation_prompt` a `SpeechToTextParams`; for these specs it carries
`use_beam_search` (bool, default false) and `max_completion_tokens` (int or none, default none),
copied from the request.

A guard field is *present* when its value is not `None` (and, for `use_beam_search`, true).

- [x] **FILE-GRD-001**: If a file request's `use_beam_search` is true, whatever its `n`, then `get_generation_prompt` shall raise `ValueError` whose message begins with `use_beam_search is not supported for this model`.
- [x] **FILE-GRD-002**: If a file request's `max_completion_tokens` is present (not `None`), including `0`, a negative number and any generous value, then `get_generation_prompt` shall raise `ValueError` whose message begins with `max_completion_tokens is not supported for this model`.
- [x] **FILE-GRD-003**: When `use_beam_search` is false and `max_completion_tokens` is `None`, `get_generation_prompt` shall not refuse the request on those fields.
- [x] **FILE-GRD-004**: When more than one refusal applies, `get_generation_prompt` shall evaluate them in this order and raise the first: FILE-IF-004, FILE-IF-005, FILE-GRD-001, FILE-GRD-002, then the language refusals (FILE-LANG); it shall do so before the audio is staged or the prompt is built.
- [x] **FILE-GRD-005**: Where the `SpeechToTextParams` it receives has no `use_beam_search` or `max_completion_tokens` attribute, `get_generation_prompt` shall treat the missing field as false or `None` (FILE-GRD-003 then applies).
- [x] **FILE-GRD-006**: The model shall impose no limit on a file's input length and no file-size cap, and shall not refuse a file on its duration or sample count. `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` and core's decode limits are the operator's upload and server settings, and the only length bound is core's actual-token `max_model_len` stop, which the operator sizes (about 4 bytes per token per slot, no KV memory).
- [ ] **FILE-GRD-007**: The sampling-shaping fields (`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `watermarking`, and the repetition, frequency and presence penalties) shall not change the tokens a file request emits. (To be verified by test; if a penalty can, the design refuses that field.)
- [ ] **FILE-GRD-008**: A refusal under FILE-GRD-001 or FILE-GRD-002 shall reach the client as HTTP 400 in a JSON error body, before any event stream starts and before the request is submitted to the engine, and shall leave no state slot, host buffer or other state attributable to the request. Where core refuses a request itself before it calls `get_generation_prompt` (for example an oversized upload, an invalid language, over-limit or invalid audio, beam search with `stream=true`, an unsupported response format), the model's message prefixes need not apply.

## Language

- [x] **FILE-LANG-001**: When a file request names a language that is itself a key of the served locale table (for example `en`), `get_generation_prompt` shall use that locale.
- [x] **FILE-LANG-002**: When a file request names a language that is not a key of the served locale table, `get_generation_prompt` shall consider the locales whose language (the lowercase letters before the first `-` or uppercase letter, when there are two or three of them; `auto` has none) it is: if they all use one row, it shall use that row; otherwise it shall use the row of the locale that `language_locales` gives for the language, and if there is no entry, then it shall raise `ValueError` naming the language and the `language_locales` override.
- [x] **FILE-LANG-003**: If a file request names a language that has no locale in the served locale table, then `get_generation_prompt` shall raise `ValueError` naming the language.
- [x] **FILE-LANG-004**: When a file request names no language and no exact locale, `get_generation_prompt` shall use the locale `auto` if the served locale table has it, else the table's only locale; if neither exists, then it shall raise `ValueError`.
- [x] **FILE-LANG-005**: When a file request's `extra_args` (core's `vllm_xargs`) has `locale` naming a key of the served locale table, `get_generation_prompt` shall use that locale, ahead of FILE-LANG-001 to FILE-LANG-004.
- [x] **FILE-LANG-006**: If a file request's `extra_args` `locale` is not a key of the served locale table, or the request also names a language that is not that locale's language (as FILE-LANG-002 defines it), then `get_generation_prompt` shall raise `ValueError` naming both.

## File prompt

- [x] **FILE-PROMPT-001**: `get_generation_prompt` shall return a prompt whose `prompt_token_ids` is exactly one token, the file-carrier token (`file_carrier_token_id`, `blank + 4`), whose `multi_modal_data` is one audio item holding the file's 16 kHz `float32` samples, and whose `mm_processor_kwargs` carries `file_header`; the model's processor shall turn them into the `keep_on_cpu` fields `samples` and `file_header`.
- [ ] **FILE-PROMPT-002**: The `file_header` field shall be three `int64` values: the geometry index, in SESS-PACK-005's order, of the server's geometry, which `get_generation_prompt` shall obtain from `vllm_nemotron_asr_streaming.lookahead.served_lookahead` (REG-LA-009, SESS-GEOM-001); the locale's row in the locale table; and the number of samples.
- [x] **FILE-PROMPT-005**: `get_generation_prompt` shall give the audio item a random per-request multimodal UUID, so that two file requests with identical samples have distinct multimodal hashes.
- [x] **FILE-PROMPT-003**: The model's multimodal processor shall declare `samples` and `file_header` as `keep_on_cpu` fields.
- [x] **FILE-PROMPT-004**: When `embed_multimodal` receives a file item, it shall return a one-row placeholder embedding without reading the item's samples and without applying the stream unit's carrier-width check.
- [ ] **FILE-PROMPT-006**: If a file item's `file_header` carries a geometry index other than the server's geometry index (SESS-GEOM-001), then the model's processor shall refuse the item at input processing with a `ValueError` naming both indices.
- [ ] **FILE-PROMPT-007**: If a file item whose `file_header` carries a geometry index other than the server's geometry index reaches the model state at the request's first schedule, then the model state shall fail that request's row alone at planning, through the row-failure path (TXG-COV-010, TXG-FAIL-008), before the step's CHUNK bucket forms, and shall not fail the step.

## Request-owned host buffer

- [x] **FILE-BUF-001**: When a file request is first scheduled, the model state shall copy the samples and header from the request's multimodal features into a host buffer owned by that request and recorded with its state slot handle.
- [x] **FILE-BUF-002**: While a file request is running, each step shall copy to the device at most the samples of the one chunk or tail it advances.
- [x] **FILE-BUF-003**: When a file request's ID appears in a step's finished request IDs, for any finish reason, the model state shall free that request's host buffer, including in a step with no forward work.
- [x] **FILE-BUF-004**: When a state slot handle is seen with a new generation, the model state shall drop any host buffer recorded for that slot's previous generation.
- [x] **FILE-BUF-005**: If a file request is aborted before its first schedule, then the model state shall create no host buffer for it.

## Whole-file rows

Let a file have `N` samples, and let the server's geometry have right context `r` and chunk size
`s = 1280 × (r + 1)` samples; then `C = floor(N / s)` and `T = N − C × s`. A *file row* is a row of
a file request in one step. A file's *queue count* is the number of labels queued on its slot and
not yet emitted, as the model state holds it (Queue counts).

- [ ] **FILE-ROW-001**: When the token scheduled at a request's first schedule is the file-carrier token, the model state shall mark the request as a file request for its whole life, set that request's bit in the step's per-row file mask on every step it is scheduled, and give its rows no unit deadlines.
- [ ] **FILE-ROW-002**: In each step, a file row shall do exactly one of the following, checked in this order: (1) drain: if the file's queue count is not zero, run as a REPLAY row that emits the next queued token without advancing audio; (2) regular chunk: else, if a full regular chunk remains, run as a CHUNK row; (3) final tail: else, if the tail has not been processed, run as a CHUNK row; (4) end: else run as a REPLAY row when the file's last emitted token was a label and as a FLUSH row when it was the continuation token, and emit EOS.
- [ ] **FILE-ROW-003**: When a file row advances a regular chunk, it shall run the chunk as a CHUNK row (fresh on the file's first step) through the streaming transition at the server's geometry, queue its labels, and emit the first queued token, or the continuation token if the chunk queued none.
- [ ] **FILE-ROW-004**: When a file row advances its final tail, it shall run as a CHUNK row with the final-tail flag set, fresh when the tail is the file's first step (a file shorter than one chunk, including a file with no samples), advance the `T` remaining samples (possibly none) with the final-tail handling and finalization a stream receives at its final tail and close, record the tail as processed, queue its labels, and emit the first queued token, or the continuation token if it queued none.
- [x] **FILE-ROW-005**: A file row shall emit EOS only after its file's final tail has been processed and its queue count is zero.
- [x] **FILE-ROW-006**: When a file has no samples, its request shall run only the final-tail and end steps and return empty text.
- [ ] **FILE-ROW-007**: For a file request that ends with EOS, its label sequence (emitted token IDs below `blank`) shall equal the label sequence of a session fed the same samples at the server's geometry and the same locale with no `update_locale`, with the same precision settings and endpointing disabled on both, and flushed successfully.
- [x] **FILE-ROW-008**: For every implemented geometry with right context `r`, `Q = (r + 1) × 10 + 1` shall be at most the per-slot label queue capacity of 141, and a file row shall advance audio only while its file's queue count is zero.
- [ ] **FILE-ROW-009**: On every step, the streaming transaction shall receive a per-row file mask with one bit per row, all false when the step holds no file row, and shall apply FILE-ROW-011 to FILE-ROW-014 and the continuation emission of FILE-ROW-003 and FILE-ROW-004 only to rows whose bit is set, applying its role's stream rule to every row whose bit is clear.
- [ ] **FILE-ROW-010**: The streaming transaction shall receive the file-carrier and continuation token IDs on every step, whether or not the step holds a file row.
- [ ] **FILE-ROW-011**: When a file row runs as a CHUNK row while the echo of its last emitted label is armed, the transaction shall set the row's echo-mismatch status unless the row's scheduled token equals that label.
- [ ] **FILE-ROW-012**: When a file row runs as a CHUNK row while the echo of its last emitted label is armed and its queue count is zero, the transaction shall not set the row's undrained-queue status.
- [ ] **FILE-ROW-013**: The transaction shall treat the continuation token as an illegal decision on every row whose file-mask bit is clear, including on steps where the continuation token ID is supplied.
- [ ] **FILE-ROW-014**: The transaction shall set a file row's role-mismatch status, in any role, if its scheduled token is the stream carrier, and, when the row runs as a CHUNK row, also if no echo is pending and its scheduled token is neither the file-carrier token nor the continuation token.

## Queue counts

A row's *count* is `QUEUE_LEN − QUEUE_HEAD` of its proposed label book after the step. A *clean*
report is a `SessionStatusReport` that `resolve_status_reports` resolves to status 0. These specs
hold with transaction graphs on and off.

- [ ] **FILE-QC-001**: `CommitPlan` shall have a field `queue_counts`, `None` or a device `int32` tensor of shape `(rows,)` holding each binding's count in binding order, and the streaming transaction shall fill it, from its per-row remaining-count output (`remaining_step[:P]` under transaction graphs; TXG-QC-001), for every row of every step it reserves.
- [ ] **FILE-QC-002**: If a commit plan's `queue_counts` is not `None` and its shape is not `(rows,)` or its dtype is not `int32`, then `BoundedCommitSink.reserve` shall raise `ValueError` before validating the lease and hold no reservation.
- [ ] **FILE-QC-003**: `BoundedCommitSink` shall allocate at construction a host `int32` buffer of `max_rows` counts, pinned when its device is CUDA, and when a reservation's `stage` runs for a plan whose `queue_counts` is not `None`, it shall copy them into that buffer with one non-blocking device-to-host copy issued before the completion event is recorded, allocating no tensor storage and not synchronizing.
- [ ] **FILE-QC-004**: When `collect` builds a row's `SessionStatusReport`, it shall set `queue_count` to the row's staged count if the row status is 0, the stage-time lease revalidation held and the plan's `queue_counts` was not `None`, and to `None` otherwise, whatever the row's `progress_key`.
- [ ] **FILE-QC-005**: `collect_commit_status` shall return, from one consumption of the sink's reports, a `CommitCollection` holding the status dictionary, the failed-request set and one `QueueCount` (request ID, admission generation, count) for each clean report whose `queue_count` is not `None`; when the sink holds no staged commit, all three shall be empty.
- [ ] **FILE-QC-006**: The model state shall apply a collected `QueueCount` only in `postprocess_state`, after `collect_commit_status` returns, and only to the file progress of a file request that the step being post-processed scheduled, whose request ID equals the count's and whose recorded admission generation equals the count's; it shall not apply any other count. A count so applied is *accepted*.
- [ ] **FILE-QC-007**: When the model state takes a file request at its first schedule, it shall set the file's queue count to 0, and the file's first step shall be planned with that count.
- [ ] **FILE-QC-008**: When a step's report for a file row is not clean, the model state shall leave the file's queue count as it was before that step.
- [ ] **FILE-QC-009**: When a step planned a file row, no count was accepted for that file (FILE-QC-006), and the row's report was not a rejection (there was no report, or its status resolved to 0), the model state shall mark the file's queue-count validity unknown without changing its retained numeric count.
- [ ] **FILE-QC-010**: If a file request that has had at least one step is scheduled while its queue-count validity is unknown, then the model state shall fail that row alone through the row-failure path (TXG-COV-010, TXG-FAIL-008) before planning the row's file action, so that the request ends with finish reason `error`.
- [ ] **FILE-QC-011**: The model state shall not read the label book on the host to plan a step, and shall not change a file's queue count while planning a step.
- [ ] **FILE-QC-012**: When a step planned a file row and, after that step's collection, no count was accepted for that file (FILE-QC-006) or the row's status resolved to a nonzero value of any status class, the model state shall fail that file request alone by recording it in `postprocess_state` for the model state's `take_failed_requests` (REG-MS-003, REG-MS-004), keeping its queue count as it was before the step, and shall not fail the step or any other row.
- [ ] **FILE-QC-013**: When a file progress that has no recorded admission generation is post-processed for the step that ran its first row, the model state shall record the admission generation of that row's clean report, the report bound to the request incarnation taken at its first schedule, and shall record no generation from any other report; for that first clean report, `postprocess_state` shall record its admission generation before applying FILE-QC-006 or evaluating FILE-QC-012.

## Tokens

- [x] **FILE-TOK-001**: The continuation token (`blank + 5`) shall be marked special in the served tokenizer and shall not be an EOS or stop token; the park token (`blank + 1`) shall be the only EOS or stop token a file request carries.

## Token count

With `Q = (r + 1) × 10 + 1` tokens at most per chunk or tail (labels plus an optional EOU), a file
can emit at most `B = 1 + (C + 1) × Q + 1` tokens. `B` is an upper bound for sizing `max_model_len`,
not an admission limit. Core stops a request at its actual token count reaching `max_model_len`.

- [ ] **FILE-BUD-002**: When a file's actual token count stays below `max_model_len`, no client completion limit was given (FILE-GRD-002 refuses any) and the deployment sets no generation-config `max_tokens`, and the request is not aborted and does not fail, the request shall end with EOS and never with finish reason `length`.
- [ ] **FILE-BUD-004**: Where core carries vllm#67144 and vllm#67151 (neither build the plugin targets does: stage 1 is the target core alone, so on it REG-DOC-003's limits apply), on the transcription route (the translation route refuses every request, FILE-IF-004): if a file request ends with a `length` caused by a server limit (the request reaching `max_model_len`; EOS at the boundary is not one, since core checks EOS first), then non-streaming it shall be reported as HTTP 422 with no partial text, and streaming it shall carry an error event after any partial deltas, followed only by `data: [DONE]`; if it ends with `error`, then non-streaming it shall be reported as HTTP 500 with no partial text, and streaming as an error event after any partial deltas followed only by `data: [DONE]`; if it ends with `abort`, then non-streaming it shall be reported as HTTP 503 with no partial text, and streaming its last transcript chunk shall carry `finish_reason` `abort`, followed by at most a usage chunk and `data: [DONE]`. (On a core without these PRs, such a file returns partial text with HTTP 200.)

## Engine interaction

- [x] **FILE-ENG-001**: A file row shall contribute at most one encoder chunk to each engine step.
- [ ] **FILE-ENG-002**: File requests shall run with the model's endpointing disabled.
- [ ] **FILE-ENG-003**: A step may hold file rows and stream rows together; a file row that runs as a CHUNK row shall run in the step's one CHUNK bucket beside the step's stream CHUNK rows, and a file row that runs as a REPLAY or FLUSH row shall run beside the step's stream rows of the same role.
- [ ] **FILE-ENG-005**: When no state slot is free, a file request shall wait in core's queue and be scheduled when a slot frees, not be refused.
- [ ] **FILE-ENG-006**: With `stream=true`, core's streaming transcription response for a file request shall carry only label text in its events before the finish, with no event for a step that emitted only a special token, and shall end with core's usual terminal events when the request finishes with EOS; if it ends in a server-caused `length`, `error` or `abort`, the terminal form is that of FILE-BUD-004.

## Deferred

- [D] **FILE-DEF-003**: Where core lets a model-state request run past `max_model_len` or the transcription route feeds a file as a resumable stream, a file of any length shall be transcribed. (Needs core; see the `transcription` design, Input length.)
- [D] **FILE-DEF-002**: Where measured file latency requires it, a file row shall advance up to `k > 1` chunks per step.

## Future roadmap

The deferred behaviour of this segment, with each item's trigger and shape, is the transcription design's Future roadmap; the `[D]` specs above are its obligations once their triggers fire.
