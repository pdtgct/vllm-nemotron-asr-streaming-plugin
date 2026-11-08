# Transcription specs

Design: [transcription-design.md](transcription-design.md). Prefix `FILE`.

A file request is core's complete-file transcription request (`POST /v1/audio/transcriptions`, or
the frontend's `Recognize` through the same service) for this model.

## Interface

- [ ] **FILE-IF-001**: The model class shall implement `SupportsTranscription` with `supports_transcription_only = False`, `supports_segment_timestamp = False` and `supports_diarized_transcription = False`.
- [ ] **FILE-IF-002**: The model class's `supported_languages` shall be a static mapping of the ISO 639-1 languages of the checkpoints the plugin supports.
- [ ] **FILE-IF-003**: The model's `get_speech_to_text_config` shall return `sample_rate = 16000`, `max_audio_clip_s = None` and `min_energy_split_window_size = None`.
- [ ] **FILE-IF-004**: If a file request's task is not transcription, or it carries a text prompt, hotwords or a target language, then `get_generation_prompt` shall raise `ValueError` naming the unsupported field.

## Language

- [ ] **FILE-LANG-001**: When a file request names a language that has exactly one locale in the served locale table (a locale `ll` or `ll-RR` belongs to language `ll`), `get_generation_prompt` shall use that locale.
- [ ] **FILE-LANG-002**: When a file request names a language that has several locales in the served locale table, `get_generation_prompt` shall use the locale that the served configuration's `language_locales` gives for it; if there is no entry, then it shall raise `ValueError` naming the language and the `language_locales` override.
- [ ] **FILE-LANG-003**: If a file request names a language that has no locale in the served locale table, then `get_generation_prompt` shall raise `ValueError` naming the language.
- [ ] **FILE-LANG-004**: When a file request names no language, `get_generation_prompt` shall use the locale `auto` if the served locale table has it, else the table's only locale; if neither exists, then it shall raise `ValueError`.

## File prompt

- [ ] **FILE-PROMPT-001**: `get_generation_prompt` shall return a prompt of exactly one token, the file-carrier token (`blank + 4`), plus one audio item with the `keep_on_cpu` fields `samples` (the file's 16 kHz `float32` samples) and `file_header`.
- [ ] **FILE-PROMPT-002**: The `file_header` field shall be `int64` values: the geometry index of the server's default lookahead, the locale's row in the locale table, and the number of samples.
- [ ] **FILE-PROMPT-005**: `get_generation_prompt` shall give the audio item a random per-request multimodal UUID, so that two file requests with identical samples have distinct multimodal hashes.
- [ ] **FILE-PROMPT-003**: The model's multimodal processor shall declare `samples` and `file_header` as `keep_on_cpu` fields.
- [ ] **FILE-PROMPT-004**: When `embed_multimodal` receives a file item, it shall return a one-row placeholder embedding without reading the item's samples and without applying the stream unit's carrier-width check.

## Request-owned host buffer

- [ ] **FILE-BUF-001**: When a file request is first scheduled, the model state shall copy the samples and header from the request's multimodal features into a host buffer owned by that request and recorded with its state slot handle.
- [ ] **FILE-BUF-002**: While a file request is running, each step shall copy to the device at most the samples of the one chunk or tail it advances.
- [ ] **FILE-BUF-003**: When a file request's ID appears in a step's finished request IDs, for any finish reason, the model state shall free that request's host buffer, including in a step with no forward work.
- [ ] **FILE-BUF-004**: When a state slot handle is seen with a new generation, the model state shall drop any host buffer recorded for that slot's previous generation.
- [ ] **FILE-BUF-005**: If a file request is aborted before its first schedule, then the model state shall create no host buffer for it.

## Whole-file rows

Let a file have `N` samples, and let the server's default geometry have right context `r` and chunk
size `s = 1280 × (r + 1)` samples; then `C = floor(N / s)` and `T = N − C × s`.

- [ ] **FILE-ROW-001**: When the token scheduled at a request's first schedule is the file-carrier token, the model state shall mark the request as a file request (row role FILE) for its whole life, and FILE rows shall carry no unit deadlines.
- [ ] **FILE-ROW-002**: In each step, a FILE row shall do exactly one of the following, checked in this order: (1) drain: if its label queue is not empty, emit the next queued token without advancing audio; (2) regular chunk: else, if a full regular chunk remains, advance it; (3) final tail: else, if the tail has not been processed, advance it; (4) end: else emit EOS.
- [ ] **FILE-ROW-003**: When a FILE row advances a regular chunk, it shall run the chunk through the streaming transition at the default geometry, queue its labels, and emit the first queued token, or the continuation token if the chunk queued none.
- [ ] **FILE-ROW-004**: When a FILE row advances its final tail, it shall advance the `T` remaining samples (possibly none) with the final-tail handling and finalization a stream receives at its final tail and close, record the tail as processed, queue its labels, and emit the first queued token, or the continuation token if it queued none.
- [ ] **FILE-ROW-005**: A FILE row shall emit EOS only after its final tail has been processed and its label queue is empty.
- [ ] **FILE-ROW-006**: When a file has no samples, its request shall run only the final-tail and end steps and return empty text.
- [ ] **FILE-ROW-007**: For a file request that ends with EOS, its label sequence (emitted token IDs below `blank`) shall equal the label sequence of a session fed the same samples at the same geometry and locale, with the same precision settings and endpointing disabled on both, and flushed successfully.
- [ ] **FILE-ROW-008**: For every implemented geometry with right context `r`, `Q = (r + 1) × 10 + 1` shall be at most the per-slot label queue capacity of 141, and a FILE row shall advance audio only while its label queue is empty.

## Tokens

- [ ] **FILE-TOK-001**: The continuation token (`blank + 5`) shall be marked special in the served tokenizer and shall not be an EOS or stop token; the park token (`blank + 1`) shall be the only EOS or stop token a file request carries.

## Token budget

With `Q = (r + 1) × 10 + 1` tokens at most per chunk or tail (labels plus an optional EOU), the
worst-case budget is `B = 1 + (C + 1) × Q + 1`.

- [ ] **FILE-BUD-001**: If a file's worst-case budget `B` exceeds `max_model_len`, then `get_generation_prompt` shall raise `ValueError` stating the file's duration and the `max_model_len` it would need.
- [ ] **FILE-BUD-002**: When a file's worst-case budget `B` is at most `max_model_len`, no completion limit below `max_model_len` applies, and the request is not aborted and does not fail, the request shall end with EOS and never with finish reason `length`.

## Engine interaction

- [ ] **FILE-ENG-001**: A FILE row shall contribute at most one encoder chunk to each engine step.
- [ ] **FILE-ENG-002**: File requests shall run with the model's endpointing disabled.

## Deferred

- [D] **FILE-DEF-001**: Where a request-chosen cadence for file requests is introduced, `get_generation_prompt` shall accept a served cadence from the request instead of the default lookahead.
- [D] **FILE-DEF-002**: Where measured file latency requires it, a FILE row shall advance up to `k > 1` chunks per step.
