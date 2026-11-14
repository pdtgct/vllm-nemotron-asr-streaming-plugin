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
- **Complete or an error.** The model refuses no file for its length; a file is reported complete
  only when its request ends with EOS after its final tail, and one that core stops at
  `max_model_len` is an error where core reports it (Input length).

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
- `get_generation_prompt(stt_params)`: refuses unsupported requests (including beam search and
  completion limits, Request guards), maps the language and builds the file prompt.

`get_generation_prompt` raises `ValueError` (core returns 400; a frontend maps it to its own
status codes) for: a task other than transcription, a text prompt, hotwords, a target
language, an unmappable language, beam search, or any client completion limit,
or a server running experimental native burst. Core also copies `vllm_xargs` into the request's sampling `extra_args`, which native
burst refuses, and a native-burst row cannot run the one-token-per-step file cycle, so complete
files are not served in that mode. Core itself refuses
`verbose_json` and timestamps for this model (it declares no segment timestamps).

Core reads the upload under its own size cap and decodes it to mono 16 kHz float32 within its
other limits (`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `VLLM_MAX_AUDIO_DECODE_DURATION_S`,
`VLLM_MAX_AUDIO_DECODE_BYTES`), sets `max_tokens` to `max_model_len` (non-beam requests pass `input_len=0` to `get_max_tokens`,
`serve/utils/api_utils.py:253`; the scheduler enforces total length separately, at
`num_tokens ≥ max_model_len`, `v1/core/sched/utils.py:115`, checked after EOS, `:105`; a generation-config `max_tokens` is already ignored for a model whose audio core does
not chunk, `base/serving.py:517-524`), and returns the request's output text with special
tokens skipped.

## Request guards

Where a refusal can live. The model sees a file request at three points, and only the first
refuses it cleanly:
- **`get_generation_prompt`** runs in core's front end before the sampling parameters exist
  (`speech_to_text/base/serving.py` `_preprocess_speech_to_text`, which calls it at line 308; the
  parameters are built after it, at lines 519-533). A `ValueError` raised there is returned to
  the client as HTTP 400 (`serve/exception_handling/register.py:36` with
  `error_response.py:80-82`), the engine never sees the request, and nothing in the engine
  changes state.
- **The model state's `add_request`** sees the engine's `SamplingParams` but runs inside the
  engine step (`vllm/v1/worker/gpu/model_runner.py:1377`, no handler): an exception there ends
  the engine, so it is not a refusal point. Failing the row through the state transaction instead
  ends that request with finish reason `error`, which upstream `main` returns as HTTP 200 with
  empty or partial text and vllm#67144 returns as HTTP 500: a server error, not a refusal of the
  client's option. Neither is acceptable.
- **Logits-processor `validate_params`** sees only `SamplingParams`, needs the operator to list the
  processor, and cannot tell a file request from a stream.

So the guards belong in `get_generation_prompt`, which is the only point that is both a clean 400
and self-contained. On upstream `main` without vllm#67151, the parameters it
receives are `SpeechToTextParams` (`vllm/config/speech_to_text.py:17-54`), built by
`build_stt_params` (`transcription/protocol.py:195-212`), and they carry neither
`use_beam_search` nor `max_completion_tokens`. **This needs a small core change:** two optional
fields on `SpeechToTextParams`, `use_beam_search: bool = False` and
`max_completion_tokens: int | None = None`, filled by `build_stt_params` of the transcription (and
translation) request. The class docstring already provides for adding fields without changing the
`get_generation_prompt` signature. The guards activate when the fields are present; against a core
without them, `getattr` finds neither and the model behaves as before. vllm#67151 adds the fields
(head `dbae7c72cf`, stacked on #67144) and is part of the stage 1 build (HLD Constraints); on a
core without it the guards are inactive and the README states the limits.

What reaches the model, by core request field:
- Beam search and a completion limit are text-decoding options. They do not apply to audio
  transcription with this model, so the refusal says so and does not present them as missing
  capacity. A completion limit is refused whatever its value rather than compared with the
  budget: a token count here counts engine steps, not words, so a client cannot choose a
  meaningful value, `max_completion_tokens` is not part of the audio-transcription API this
  route mimics, and a comparison with a bound would refuse the generous limits clients send for
  short files. Messages use vLLM's usual form, `<field> is not supported for this model`, which leaves other models free to support the field.
- `use_beam_search` (any `n`, including 1). Core then runs its beam-search loop
  (`speech_to_text/base/serving.py:526-528, 560`): up to `max_tokens` passes
  (`generate/beam_search/online.py:93-110`), each a new engine request with the prompt plus the
  beam's tokens and the same multimodal hashes (`generate/beam_search/utils.py:56-63`) and
  `max_tokens=1`. On a core without the fields (nothing refuses it) this restarts the file on every
  pass, can issue up to `n × max_model_len` engine requests (one per beam per pass) per upload, and ties up the server; it
  yields no usable transcript. Refused as above. Core itself refuses beam search first when
  `stream=true` (`serving.py:437-440`).
- `max_completion_tokens` becomes `SamplingParams.max_tokens` through core's `get_max_tokens`
  (`serve/utils/api_utils.py:234-271`), the minimum of the model's remaining length and the client
  limit, and would end a file early with `length`. Refused whatever its value.
- Sampling-shaping fields (`temperature`, `top_p`, `top_k`, `min_p`, `seed`, penalties,
  `watermarking`): not refused. Every token is forced by the model's logits
  (`forced_logits_rows`), so they cannot change the transcript; they are ignored.
  Unverified: a repetition, frequency or presence penalty might move the forced choice; the
  test phase shall verify it (FILE-GRD-007) and, if it can, this design refuses those fields too.
- `n`, `length_penalty`, `include_stop_str_in_output` without beam search: core does not forward
  them (`transcription/protocol.py:240-290`); nothing to refuse.
- `vllm_xargs`: only `locale` is read (Language); other keys are ignored on the non-native-burst
  path and refused in native-burst mode (FILE-IF-005).
- File size and length: the model imposes none (Input length). Core caps the upload bytes
  (`speech_to_text/base/utils.py:16-56`, route `transcription/api_router.py:49`, default 25 MB,
  `vllm/envs.py:83`) and the decode (`VLLM_MAX_AUDIO_DECODE_DURATION_S` 600 s,
  `VLLM_MAX_AUDIO_DECODE_BYTES` 256 MiB, `envs.py:84-85`, applied at `base/serving.py:126-127,
  182-183`). These are the operator's upload and server settings, not model limits.
- A generation-config `max_tokens` is the deployment's concern and is refused at startup
  (`streaming_config.py`).

Semantics of the guards:
- "Present" means a non-`None` value. A missing attribute (a core without the fields) and `None`
  mean absent; `0`, a negative number and a generous value are all present and refused.
  `use_beam_search` is present when true.
- Both set: beam search is reported first.
- Order inside `get_generation_prompt`: task, text prompt, hotwords and target language
  (FILE-IF-004), native burst (FILE-IF-005), beam search, completion limit, language, then the prompt. A translation request with beam search is therefore
  refused as a wrong task.
- Behaviour: HTTP 400, returned as JSON before any event stream starts and before engine
  submission; the refusal leaves no slot, host buffer or other state attributable to the request.
  Core makes some refusals before it calls `get_generation_prompt`, with its own messages, also as
  400 (the model's prefixes do not apply); the list is open. For example: an oversized upload
  (`transcription/api_router.py:49`), an invalid `language` or `to_language` (`serving.py:275-280`),
  audio over the decode limits or invalid audio (`serving.py:283, 190-193`), beam search with
  `stream=true` (`serving.py:437`), and unsupported response formats (`serving.py:451-460`,
  `verbose_json` `:462-468`, `diarized_json` `:470-476`, a streamed `verbose_json` or
  `diarized_json` `:478-484`).
- Scope: the transcription route and the translation route, which share the base serving and
  `SpeechToTextParams`; the translation route's requests are refused by FILE-IF-004 anyway.

Realtime (`/v1/realtime`) never calls `get_generation_prompt` and is unaffected.

## Input length

A streaming model can in principle process input of any length, and the model state holds a
constant amount of device state per file (one state slot); the host buffer grows with duration.
Five things bound a file or set its cost today:
- **Token count (core constraint).** A file emits one token per engine step, so its request grows
  with its length. Core passes `max_tokens = max_model_len` for a non-beam request
  (`serve/utils/api_utils.py:253`), stops the request when `num_tokens ≥ max_model_len`
  (`v1/core/sched/utils.py:115`, finish `length`, checked after EOS at `:105`, so EOS exactly at the
  boundary succeeds), and sizes the sampled-token history at `max_num_reqs × max_model_len`
  (`v1/worker/gpu/states.py:38`). Core's stop counts actual tokens. Raising `max_model_len` costs
  about 4 bytes per token per slot and no KV memory.
- **No model-side admission check.** The worst-case refusal of earlier drafts (FILE-BUD-001, 71
  tokens per chunk at 560 ms, about 1/71 of what core's actual-token stop allows) is retired (Pete,
  2026-11-11). The model imposes no input-length limit; the operator sizes `max_model_len`.
- **How core reports the stop.** On upstream `main` a request that ends with `length`, `error` or
  `abort` returns HTTP 200 with partial text. Two upstream pull requests, both in the stage 1
  build, change that on the transcription and translation routes, which share core's
  speech-to-text serving:
  - vllm#67144 fails an `error` ending (non-streaming: HTTP 500; streaming: an error event after
    any partial deltas, then `data: [DONE]`) and an `abort` ending (non-streaming: HTTP 503,
    retryable; streaming: a final chunk with `finish_reason` `abort`, after which the remaining
    chunks are skipped, a usage chunk may follow, then `data: [DONE]`).
  - vllm#67151 fails a `length` ending caused by a server limit (the request reaching
    `max_model_len`) with HTTP 422 non-streaming and an error event streaming. A `length` caused by
    a client's own `max_completion_tokens` stays HTTP 200 for other models; this plugin refuses
    that field, so every `length` ending of its files is server-caused.

  On a core without these PRs such a file returns partial text with HTTP 200, and the README
  limits say so (FILE-BUD-004).
- **Upload and decode (core settings).** Core reads the upload into bytes and decodes the whole
  file to float32 under the limits above, then carries every sample in the request's
  `multi_modal_data`.
- **Host buffer (plugin choice).** The request-owned buffer holds the whole file (64,000 bytes
  per second; it grows with duration). It could be replaced by chunked intake only after the
  core change of FILE-DEF-003 (the file arriving in segments).

Making length unbounded cannot be done by the plugin alone: while the route issues one request
for the whole file, steps and tokens grow with length, and `k > 1` chunks per step (FILE-DEF-002)
only multiplies the limit. It needs one of two core changes: the transcription route feeds the
file as a resumable stream in segments (as the realtime route does with its bounded units), or a
model-state request is exempted from the `max_model_len` stop with a bounded token history. Not
designed here; recorded as FILE-DEF-003. Until then the deployment sets `max_model_len` for the
longest file it accepts, using the upper bound of the Token count section.

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

Memory: 64,000 bytes per second of audio (about 3.8 MB per minute) per file in flight. Without a
frontend, `max_num_seqs` bounds the files actively scheduled, not uploads: requests queued in core
hold their decoded audio in the API server with no bound; a frontend may add a cap.

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

## Token count

Every step emits one token, and core counts the prompt token too. Each regular chunk and the tail
produce at most `Q_g` tokens (at least one), plus one EOS:

`B = 1 + (C + 1) × Q_g + 1`

`get_generation_prompt` does not check `B` against `max_model_len` (FILE-BUD-001 and 003 are retired):
the bound is pessimistic, and core's stop counts actual tokens. A file whose actual count stays
below `max_model_len`, with no completion limit (Request guards refuse every client limit; startup
refuses a generation-config `max_tokens`), reaches EOS before core's length limit. An abort or
error can still end it earlier (Completion).

At 560 ms (`s = 8,960`, `Q = 71`), one hour is `N = 57,600,000` samples: `C = 6,428`, a tail of
5,120 samples, and `B = 456,461`. Typical output is a small fraction of that. A deployment sets
`max_model_len` for the longest file it accepts, using `B` as a safe upper bound; for this attention-free model it costs no KV
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
slot, whose queued labels are lost) returns partial text on a core without vllm#67144, and a
server-caused `length` on a core without vllm#67151 (Input length). Beam search and every client completion limit are refused by the model before
submission (Request guards), so neither can truncate a transcript.

## Engine interaction

- **Step cost.** A FILE row contributes at most one encoder chunk per step, the same as a stream
  row, so the encoder's captured populations (up to `max_num_seqs` rows) and the profiled step
  memory need no change.
- **Mixed steps.** One step may hold FILE rows and stream rows. The model groups chunk rows by
  geometry, and a FILE row advances in the group for its geometry alongside stream rows of the
  same geometry.
- **Admission.** A file request is an ordinary request: it waits in core's queue until a state slot
  is free, takes it at first schedule, and counts against the engine's active-request limit, where
  it can delay or refuse streams. Without a frontend nothing bounds file admission beyond `max_num_seqs`, and
  it does not reserve capacity for streams: a finished file's slot is held until it retires, so the next file can be
  admitted while it is still occupied.
- **Step order.** Within a step, chunk groups (one per geometry) run in order of urgency: a group's
  urgency is the earliest unit deadline among its stream rows; FILE rows, which have no deadlines,
  add nothing to it; groups holding only FILE rows run after every group with a stream row, and
  ties are broken by geometry index.
- **Endpointing.** Disabled, as for streams on core.

## Implementation notes

- **Host view of the label queue.** The step rule needs each FILE row's queued-token count. After a
  step in which a FILE row advanced audio, the model state reads the counts of those rows from the
  device-resident label book in one batched copy; after a drain step the host already knows the
  count. This is one small synchronous read per step that advances a file. The planned
  replacement carries each row's remaining count in the per-step status the device already
  returns, which removes the read.
- **Where malformed file items are refused.** The processor checks a file item's header against
  its samples (geometry index in range, locale row non-negative, sample count equal to the samples'
  length), so a malformed request fails at input processing. The model state repeats the checks
  and fails only that row if one gets through; it never fails the step.
- **Device route.** File samples leave the host buffer only through `FileHostBuffers.stage` and
  reach the device only through `file_rows.copy_chunk_to_device`; their encoder chunks run through
  the same per-geometry chunk path as stream rows.
- **Served geometries.** The model, the session factory and `buffer_realtime_audio` take the served
  set from one function, `lookahead.served_lookaheads`.

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
| Where to refuse beam search and completion limits | `get_generation_prompt`, with two request fields added to `SpeechToTextParams` | The model state's `add_request`; row failure; a logits processor | `add_request` cannot raise without ending the engine; row failure becomes a server error (HTTP 500 with vllm#67144, HTTP 200 with partial text without it); a logits processor cannot see the prompt |
| Budget | No admission check; `B` is sizing guidance only (Pete, 2026-11-11) | Worst-case refusal (FILE-BUD-001, retired); a typical-output bound | The model imposes no input-length limit; core's actual-token stop is the bound, and reporting a server-caused `length` as an error is core work (Open 1) |
| Regional languages | Explicit `language_locales` mapping, and an exact locale in `vllm_xargs` | Picking a default region | A guessed region changes transcripts silently; a client that names a locale must get that locale |

## Open Questions & Future Decisions

### Open
1. The core changes are open upstream as vllm#67144 (`error` and `abort` endings) and vllm#67151
   (the two request fields and server-caused `length` endings), and both are in the stage 1 build.
   Until they merge, the README states the behaviour on a core without them: guards inactive, and
   partial text with HTTP 200 for these endings.
2. For the README limits: on a core without vllm#67151, beam search can issue up to
   `n × max_model_len` engine requests per upload, one per beam per pass (Request guards), so the route is not to be exposed to untrusted
   clients on such a core.

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
