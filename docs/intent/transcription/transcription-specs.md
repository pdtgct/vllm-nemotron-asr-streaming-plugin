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
request that sets them is refused as not applicable, not as unsupported capacity.

Core hands `get_generation_prompt` a `SpeechToTextParams`; for these specs it carries
`use_beam_search` (bool, default false) and `max_completion_tokens` (int or none, default none),
copied from the request.

A guard field is *present* when its value is not `None` (and, for `use_beam_search`, true).

- [x] **FILE-GRD-001**: If a file request's `use_beam_search` is true, whatever its `n`, then `get_generation_prompt` shall raise `ValueError` whose message begins with `use_beam_search does not apply to audio transcription with this model`.
- [x] **FILE-GRD-002**: If a file request's `max_completion_tokens` is present (not `None`), including `0`, a negative number and any generous value, then `get_generation_prompt` shall raise `ValueError` whose message begins with `max_completion_tokens does not apply to audio transcription with this model`.
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
- [x] **FILE-PROMPT-002**: The `file_header` field shall be `int64` values: the geometry index (SESS-FAC-004's table) of the server's default lookahead, the locale's row in the locale table, and the number of samples.
- [x] **FILE-PROMPT-005**: `get_generation_prompt` shall give the audio item a random per-request multimodal UUID, so that two file requests with identical samples have distinct multimodal hashes.
- [x] **FILE-PROMPT-003**: The model's multimodal processor shall declare `samples` and `file_header` as `keep_on_cpu` fields.
- [x] **FILE-PROMPT-004**: When `embed_multimodal` receives a file item, it shall return a one-row placeholder embedding without reading the item's samples and without applying the stream unit's carrier-width check.

## Request-owned host buffer

- [x] **FILE-BUF-001**: When a file request is first scheduled, the model state shall copy the samples and header from the request's multimodal features into a host buffer owned by that request and recorded with its state slot handle.
- [x] **FILE-BUF-002**: While a file request is running, each step shall copy to the device at most the samples of the one chunk or tail it advances.
- [x] **FILE-BUF-003**: When a file request's ID appears in a step's finished request IDs, for any finish reason, the model state shall free that request's host buffer, including in a step with no forward work.
- [x] **FILE-BUF-004**: When a state slot handle is seen with a new generation, the model state shall drop any host buffer recorded for that slot's previous generation.
- [x] **FILE-BUF-005**: If a file request is aborted before its first schedule, then the model state shall create no host buffer for it.

## Whole-file rows

Let a file have `N` samples, and let the server's default geometry have right context `r` and chunk
size `s = 1280 × (r + 1)` samples; then `C = floor(N / s)` and `T = N − C × s`.

- [x] **FILE-ROW-001**: When the token scheduled at a request's first schedule is the file-carrier token, the model state shall mark the request as a file request (row role FILE) for its whole life, and FILE rows shall carry no unit deadlines.
- [x] **FILE-ROW-002**: In each step, a FILE row shall do exactly one of the following, checked in this order: (1) drain: if its label queue is not empty, emit the next queued token without advancing audio; (2) regular chunk: else, if a full regular chunk remains, advance it; (3) final tail: else, if the tail has not been processed, advance it; (4) end: else emit EOS.
- [x] **FILE-ROW-003**: When a FILE row advances a regular chunk, it shall run the chunk through the streaming transition at the default geometry, queue its labels, and emit the first queued token, or the continuation token if the chunk queued none.
- [x] **FILE-ROW-004**: When a FILE row advances its final tail, it shall advance the `T` remaining samples (possibly none) with the final-tail handling and finalization a stream receives at its final tail and close, record the tail as processed, queue its labels, and emit the first queued token, or the continuation token if it queued none.
- [x] **FILE-ROW-005**: A FILE row shall emit EOS only after its final tail has been processed and its label queue is empty.
- [x] **FILE-ROW-006**: When a file has no samples, its request shall run only the final-tail and end steps and return empty text.
- [ ] **FILE-ROW-007**: For a file request that ends with EOS, its label sequence (emitted token IDs below `blank`) shall equal the label sequence of a session fed the same samples at the same geometry and locale with no `update_locale`, with the same precision settings and endpointing disabled on both, and flushed successfully.
- [x] **FILE-ROW-008**: For every implemented geometry with right context `r`, `Q = (r + 1) × 10 + 1` shall be at most the per-slot label queue capacity of 141, and a FILE row shall advance audio only while its label queue is empty.

## Tokens

- [x] **FILE-TOK-001**: The continuation token (`blank + 5`) shall be marked special in the served tokenizer and shall not be an EOS or stop token; the park token (`blank + 1`) shall be the only EOS or stop token a file request carries.

## Token count

With `Q = (r + 1) × 10 + 1` tokens at most per chunk or tail (labels plus an optional EOU), a file
can emit at most `B = 1 + (C + 1) × Q + 1` tokens. `B` is an upper bound for sizing `max_model_len`,
not an admission limit. Core stops a request at its actual token count reaching `max_model_len`.

- [R] **FILE-BUD-001** (retired, Pete 2026-11-11): the worst-case admission refusal is dropped because the model imposes no input-length limit; the only bound is core's actual-token `max_model_len` stop.
- [R] **FILE-BUD-003** (retired with FILE-BUD-001): its fixed message prefix has no refusal to identify.
- [ ] **FILE-BUD-002**: When a file's actual token count stays below `max_model_len`, no client completion limit was given (FILE-GRD-002 refuses any) and the deployment sets no generation-config `max_tokens`, and the request is not aborted and does not fail, the request shall end with EOS and never with finish reason `length`.
- [ ] **FILE-BUD-004**: Where core reports a request that ends with a `length` caused by a server limit (the request reaching `max_model_len`), or with `error` or `abort`, as an error on the transcription and translation routes: if a file request ends with such a `length` (EOS at the boundary is not one: core checks EOS first), `error` or `abort`, then non-streaming it shall be reported as an error response with no partial text, and streaming it shall carry an error event after any partial deltas and no normal completion event, followed only by core's stream terminator `data: [DONE]` (FILE-ENG-006). (Depends on the core change, part (b); on a core without it, such a file returns partial text with HTTP 200.)

## Engine interaction

- [x] **FILE-ENG-001**: A FILE row shall contribute at most one encoder chunk to each engine step.
- [ ] **FILE-ENG-002**: File requests shall run with the model's endpointing disabled.
- [x] **FILE-ENG-003**: A step may hold FILE rows and stream rows together; a FILE row shall advance in the same per-geometry group as stream rows of its geometry.
- [x] **FILE-ENG-004**: Within a step, the model shall run chunk groups in order of the earliest unit deadline among each group's stream rows, FILE rows contributing no deadline, then groups holding only FILE rows, breaking ties by geometry index.
- [ ] **FILE-ENG-005**: When no state slot is free, a file request shall wait in core's queue and be scheduled when a slot frees, not be refused.
- [ ] **FILE-ENG-006**: With `stream=true`, core's streaming transcription response for a file request shall carry only label text in its events before the finish, with no event for a step that emitted only a special token, and shall end with core's usual terminal events when the request finishes with EOS; if it ends in a server-caused `length`, `error` or `abort`, the terminal form is that of FILE-BUD-004.

## Deferred

- [D] **FILE-DEF-001**: Where a request-chosen cadence for file requests is introduced, `get_generation_prompt` shall accept a served cadence from the request instead of the default lookahead.
- [D] **FILE-DEF-003**: Where core lets a model-state request run past `max_model_len` or the transcription route feeds a file as a resumable stream, a file of any length shall be transcribed. (Needs core; see the `transcription` design, Input length.)
- [D] **FILE-DEF-002**: Where measured file latency requires it, a FILE row shall advance up to `k > 1` chunks per step.
