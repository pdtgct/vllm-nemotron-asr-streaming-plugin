---
parent: high-level-design
prefix: FILE
---

# Transcription

## Context and Design Philosophy

Core vLLM serves complete-file transcription (`POST /v1/audio/transcriptions`, and the frontend's
gRPC `Recognize` through the same service) for models implementing `SupportsTranscription`. Core
sends a file as one ordinary request: a prompt from the model's `get_generation_prompt`, decoded
token by token until EOS. This component makes that request walk the whole file through the same
streaming transition a live stream uses, on the request's own state slot.

Guiding principles:
- **A file is a stream that is already fully buffered.** It runs the same transition at the
  server's default cadence, with the same final-tail handling and numerics as a stream at that
  cadence, so its transcript can equal that stream's transcript of the same audio.
- **A file costs about what a stream costs per step.** Each engine step advances a file by at most
  one cadence chunk and emits one token.
- **Complete or refused.** A file whose token budget cannot fit is refused before submission; a
  file is reported complete only when its request ends with EOS after its final tail.

## Interface

The model class adds `SupportsTranscription` with:
- `supports_transcription_only = False` (the model also serves `realtime`);
- `supports_segment_timestamp = False` and `supports_diarized_transcription = False`;
- `supported_languages`: a static class table of the two-letter languages of the supported
  checkpoint's locale table that core's language map (Whisper's `LANGUAGES`) contains. Core refuses
  the class at definition time if the table names any other code, and refuses a request language
  outside its map before the model sees it; a code in the map but not in the table passes core with
  a warning and the model refuses it (Language). The checkpoint's other languages (for example
  Norwegian Bokmål `nb`, Zulu `zu`, and the three-letter `nah`) are reachable only by
  exact locale with no `language`;
- `get_speech_to_text_config(model_config, task_type)`: `sample_rate = 16000`,
  `max_audio_clip_s = None` and `min_energy_split_window_size = None`, so core never splits a file;
- `get_generation_prompt(stt_params)`: refuses unsupported requests, maps the language, checks the
  budget and builds the file prompt.

`get_generation_prompt` raises `ValueError` (core returns 400; `Recognize` returns
`INVALID_ARGUMENT`) for: a task other than transcription, a text prompt, hotwords, a target
language, an unmappable language, a budget over `max_model_len`, or a server running experimental
native burst. Core also copies `vllm_xargs` into the request's sampling `extra_args`, which native
burst refuses, and a native-burst row cannot run the one-token-per-step file cycle, so complete
files are not served in that mode. Core itself refuses
`verbose_json` and timestamps for this model (it declares no segment timestamps).

Core decodes the upload to mono 16 kHz float32 within its own limits
(`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `VLLM_MAX_AUDIO_DECODE_DURATION_S`,
`VLLM_MAX_AUDIO_DECODE_BYTES`), sets `max_tokens` to `max_model_len` minus the prompt length (no client or generation-config
limit is present in the supported deployment), and returns the request's output text with special
tokens skipped.

## Language

Core's `validate_language` accepts any code in the class table (and other known codes with a
warning), so `get_generation_prompt` makes the binding decision against the served configuration,
`stt_params.model_config.hf_config`, which holds the locale table (`prompt_dictionary`) and the
optional `language_locales` mapping:
- **Exact locale.** A request may name a locale of the table in `vllm_xargs` as `{"locale": "en-GB"}`
  (core passes `vllm_xargs` to the model as `stt_params.extra_args`). It takes precedence over
  language mapping; an unknown locale is refused, and a `language` also given must be that
  locale's language. This is how a frontend whose clients name locales (Riva `Recognize`) keeps the
  client's locale.
- A locale's language is the lowercase letters before the first `-` or uppercase letter, when there are two or three of them; `auto` has none (`en-US`, `enGB` and `en` are `en`; `haw-US` is `haw`). The
  table has aliases: several locale names may share a row.
- A language that is itself a locale of the table (`en`, `de`) maps to that locale; the card's own
  processor accepts bare codes the same way.
- Otherwise, a language whose locales all use one row maps to that row (the prompt carries only the
  row, so which alias names it does not matter).
- Otherwise (its locales use several rows, as `zh-CN` and `zh-TW` do) the language maps to the locale
  named for it in `language_locales`, given as `--hf-overrides '{"language_locales": {"zh": "zh-CN"}}'`;
  core applies it to the served configuration, and post-override validation checks every entry
  (see the `registration` design). Without an entry, that language is refused, and startup logs the
  unmapped languages.
- A language with no locale is refused.
- No language: the `auto` locale when the table has it, else the table's only locale, else the
  request is refused.

## File prompt

The prompt is exactly one token, the **file-carrier token**, with the file's samples as its one
audio item (`multi_modal_data`) and the header in `mm_processor_kwargs`, the standard channels of
a vLLM prompt. The model's processor turns them into two `keep_on_cpu` fields, so core's batching
never moves the file to the device:
- `samples`: the file's 16 kHz float32 samples;
- `file_header`: int64 values: geometry (the server's default lookahead), locale row and sample
  count.

The prompt gives the audio item a random per-request multimodal UUID (`multi_modal_uuids`). Core
hashes an item from its UUID and the request's processor arguments, so every file item's hash is
unique and core's multimodal cache never recognizes a repeated upload and drops its data before the
model sees it.

Core schedules the item as an encoder input of one placeholder. The model's multimodal embedding
for a file item returns a one-row placeholder embedding without reading the samples; the
stream-unit carrier-width check does not apply to file items.

## Request-owned host buffer

At the request's first schedule, the model state takes the samples and header from the scheduled
request's multimodal features into a **host buffer owned by the request**, recorded with the
request's state slot handle. The buffer is independent of core's encoder cache, which may drop the
item once the one-token prompt is consumed. Each later step copies at most one chunk's samples to
the device.

The buffer is freed when the request leaves the engine for any reason (finish, abort, preemption,
error), which the model state learns from the next step's finished request IDs; the engine
delivers that cleanup even when the step has no forward work. Under synchronous scheduling no step
that reads the buffer is still in flight then. A slot handle seen again with a new generation
drops any stale buffer recorded for the old one. A request aborted before its first schedule never
creates one.

Memory: 64,000 bytes per second of audio (about 3.8 MB per minute) per file in flight. The
frontend's in-flight file cap bounds the number of buffers; their total is that cap times the
longest accepted file.

## Whole-file rows

A file request is a new row role, **FILE**, beside the stream roles (CHUNK, REPLAY, EOU, FLUSH). The
role comes from host authority: the scheduled token is the file-carrier token at the first
schedule, and the model state marks the request as a file request for its life. FILE rows carry no
unit deadlines.

Let the file have `N` samples and the default geometry `s` samples per chunk. Then
`C = floor(N / s)` regular chunks and a tail of `T = N − C × s` samples (`0 ≤ T < s`).

Each step for a FILE row does exactly one of the following, checked in order:
1. **Drain.** If the row's label queue is not empty, emit its next token (a label, or the EOU token
   if endpointing ever fires). No audio advances.
2. **Regular chunk.** Else, if a full regular chunk remains, advance it through the streaming
   transition, queue its labels, and emit the first queued token, or the continuation token if it
   queued none.
3. **Final tail.** Else, if the tail has not been processed, advance the `T` remaining samples
   (possibly none) with the same final-tail rule and finalization a stream gets at its final tail
   and close, record that the tail is done, queue its labels, and emit the first queued token, or
   the continuation token if it queued none.
4. **End.** Else emit EOS.

The label queue is the same per-slot queue streams use, with capacity 141. A FILE row advances audio
only when it is empty, and one chunk, or the tail, queues at most `Q_g = (r + 1) × 10 + 1` tokens
for geometry `g` with right context `r` (labels, at most 10 per encoder frame, plus an optional
EOU), at most 141. So the queue cannot overflow.

A file shorter than one chunk has `C = 0` and is all tail; an empty file runs only the tail step
and the end step and returns empty text.

## Continuation token

A dedicated special token emitted when a step advanced audio but queued no token. It is not EOS, so
it does not end the request, and core's `skip_special_tokens` drops it from the text.

## Control tokens

| Token | ID |
| --- | --- |
| blank (never emitted) | blank |
| park / EOS | blank + 1 |
| stream chunk carrier | blank + 2 |
| EOU | blank + 3 |
| file carrier | blank + 4 |
| continuation | blank + 5 |

`vocab_size = blank + 6`. The tokenizer mode (see the `registration` design) marks all of them
special.

## Token budget

Every step emits one token, and core counts the prompt token too. Each regular chunk and the tail
produce at most `Q_g` tokens (at least one), plus one EOS:

`B = 1 + (C + 1) × Q_g + 1`

`get_generation_prompt` refuses the file when `B > max_model_len`. With `B ≤ max_model_len`, and
no smaller completion limit (the frontend refuses client completion limits and a generation-config
`max_tokens`), the request reaches EOS before core's length limit, so a `length` ending cannot
occur. An abort or error can still end it earlier (Completion).

At 560 ms (`s = 8,960`, `Q = 71`), one hour is `N = 57,600,000` samples: `C = 6,428`, a tail of
5,120 samples, and `B = 456,461`. Typical output is a small fraction of that. A deployment sets
`max_model_len` for the longest file it accepts; for this attention-free model it costs no KV
memory, but core preallocates token history for `max_num_seqs × max_model_len` tokens (about
4 bytes each).

## Equivalence with streams

A file's **label sequence** is the sequence of emitted token IDs below `blank` (the park, EOU,
continuation and other control tokens excluded). For a file request that ends with EOS, it equals
the label sequence of a session that is fed the same samples at the same geometry and locale (with
no `update_locale`), with
the same precision settings and endpointing disabled on both, and then flushed successfully.

## Streaming responses

With `stream=true`, core sends each non-empty text delta as an event; continuation and other special
tokens produce no text, so the stream carries only label text before it finishes. Core's terminal
events (the finish event, optional usage, and `[DONE]`) are sent as for any model.

## Completion

A file's transcript is complete only when its request finishes with EOS after the final tail. Core's
serving returns the text for any ending, so an `error` or `abort` ending (for example a preempted
slot, whose queued labels are lost) still returns partial text until core treats those endings as
failures, which is independent core work. Beam search and client completion limits are refused by
the frontend, not by the model; without the frontend this route is unsupported for this model.

## Engine interaction

- **Step cost.** A FILE row contributes at most one encoder chunk per step, the same as a stream
  row, so the encoder's captured populations (up to `max_num_seqs` rows) and the profiled step
  memory need no change.
- **Mixed steps.** One step may hold FILE rows and stream rows. The model groups chunk rows by
  geometry, and a FILE row advances in the group for its geometry alongside stream rows of the
  same geometry.
- **Admission.** A file request is an ordinary request: it waits in core's queue until a state slot
  is free, takes it at first schedule, and counts against the engine's active-request limit, where
  it can delay or refuse streams. The frontend's file cap bounds file admission without reserving
  capacity for streams: a finished file's slot is held until it retires, so the next file can be
  admitted while it is still occupied.
- **Step order.** Within a step, chunk groups (one per geometry) run in order of urgency: a group's
  urgency is the earliest unit deadline among its stream rows; FILE rows, which have no deadlines,
  add nothing to it; groups holding only FILE rows run after every group with a stream row, and
  ties are broken by geometry index.
- **Endpointing.** Disabled, as for streams on core.

## Current source

The vendored model differs from this design: its row planner rejects a non-chunk row that carries
a multimodal feature, header resolution assumes the stream carrier, `embed_multimodal` enforces the
carrier width for every item, the processor declares no `keep_on_cpu` fields, and the class does
not implement `SupportsTranscription`.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Chunks per step | At most one | Up to `k > 1` | One chunk keeps a file row within the encoder's captured populations and profiled memory and fits the one-token-per-step queue; `k > 1` needs capture and profiling for `k ×` rows and a queue holding `k` bursts |
| Cadence | The server's default lookahead | Request-chosen | Core's transcription request has no lookahead field |
| Where samples live | Request-owned host buffer, one chunk staged per step | A device copy of the file; reading from the encoder cache | A device copy grows with file length; core may drop encoder-cache entries once the prompt is consumed |
| When the buffer is freed | When the request is reported finished (any reason) | A release-fence hook | The engine exposes no model hook at the release fence; under synchronous scheduling no in-flight step reads the buffer |
| Repeated uploads | A random per-request multimodal UUID | A nonce in the header; disabling the multimodal cache | Core hashes items by their UUID, so no header field is needed, and the server's cache settings stay untouched |
| Silence | A continuation token | Emitting nothing; the park/EOS token | Every step emits a token; EOS would end the request |
| Distinguishing file rows | A dedicated file-carrier prompt token | The stream carrier with a different field | Row roles come from the scheduled token |
| Budget | Worst case, checked before submission | A typical-output bound | Only the worst case makes a `length` ending impossible |
| Regional languages | Explicit `language_locales` mapping, and an exact locale in `vllm_xargs` | Picking a default region | A guessed region changes transcripts silently; a client that names a locale must get that locale |

## Open Questions & Future Decisions

### Deferred
1. A request-chosen cadence (for example through `vllm_xargs`), if deployments need one.
2. `k > 1` chunks per step, if measured file latency needs it.

## References

- `SupportsTranscription`, `SpeechToTextConfig`: `vllm/model_executor/models/interfaces.py`,
  `vllm/config/speech_to_text.py`.
- Core serving: `vllm/entrypoints/speech_to_text/base/serving.py`.
- `keep_on_cpu` fields: `vllm/multimodal/inputs.py`; token history: `vllm/v1/worker/gpu/states.py`.
- Row roles and emission: `nemotron_asr/plan.py`, `nemotron_asr/advance.py`; queue:
  `nemotron_asr/state_profile.py`, `nemotron_asr/manifests.py`.
