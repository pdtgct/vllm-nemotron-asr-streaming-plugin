# vllm-nemotron-asr-streaming

Out-of-tree plugin for cache-aware streaming Nemotron ASR on vLLM core. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card unmodified, for streaming sessions and for complete-file transcription. It targets two builds of
upstream vLLM `main` with open pull requests; a released vLLM does not provide the symbols it needs,
and `register()` fails with a message saying so.

- **Stage 1** (the model alone, on core's `/v1/realtime` and `/v1/audio/transcriptions`) is upstream
  vLLM `main` with vllm#67374, #67375, #67144 and #67151. #67374 and #67375 are RFC 1 (the
  persistent-state primitive and stream admission, `AsyncLLM.generate(..., stream_admission=True)` and
  `wait_stream_retired`).
- **Stage 2** (with a frontend) is stage 1 plus RFC 2 (#65546 and #67294, the endpoint plugin
  lifecycle) and a frontend endpoint plugin.

The design is in `docs/high-level-design.md` and `docs/intent/`; module names are in
`docs/implementation-layout.md`.

## What is registered

Distribution `vllm-nemotron-asr-streaming`, import package `vllm_nemotron_asr_streaming`. Two entry
points:

```toml
[project.entry-points."vllm.general_plugins"]
nemotron_asr_streaming = "vllm_nemotron_asr_streaming:register"

[project.entry-points."vllm_riva_frontend.session_factories"]
nemotron_asr_streaming = "vllm_nemotron_asr_streaming.session:create_session_factory"
```

`register()` checks the client calls it uses (`AsyncLLM.generate(stream_admission=...)` and
`AsyncLLM.wait_stream_retired`), then registers, without importing the model or touching a device:

- the architecture `Nemotron3_5AsrStreamingForRNNT` (a lazy reference) and its model-configuration hook;
- the config format `nemotron_asr_streaming`, which translates the Hub card;
- the tokenizer mode `nemotron_asr_streaming` (and the renderer of the same name), which makes the
  control tokens special and the park token the EOS.

The Hub card names the architecture `Nemotron3_5AsrForRNNT`, which is the offline plugin's key. This
plugin never writes that key or its configuration hook, so both plugins can be installed together.
When `VLLM_PLUGINS` is an allowlist, include `nemotron_asr_streaming`.

## Serve

<!-- @spec REG-ENG-007 -->
```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --enforce-eager \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    --max-model-len 76800 \
    --hf-overrides '{"supported_num_lookahead_tokens": [6, 13], "num_lookahead_tokens": 13}'
```

- The persistent-state model needs synchronous scheduling, no prefix caching, a single device and an
  eager outer runner (it runs its own captured graphs inside the step). Core resolves synchronous
  scheduling and no prefix caching for this model, so the command omits `--no-async-scheduling` and
  `--no-enable-prefix-caching`; `--enforce-eager` is required because core's default is not eager.
  Other settings it refuses (speculative decoding, KV transfer, dual-batch overlap, sleep mode, the
  cumem allocator, hierarchical sparse attention, fault tolerance, any parallel size other than 1)
  fail at startup with a message naming each.
- The model prepares its execution (and any captured graphs) on core's memory-profiling step, before
  core sizes the KV cache and state pool, so the memory it retains is budgeted. If you set
  `--kv-cache-memory-bytes`, core takes that budget as given: leave room for the model's execution
  memory.
- `num_lookahead_tokens` is required: the card has no default the plugin accepts. It sets the cadence for
  complete files and for `/v1/realtime`, and must be in the served set.
- `supported_num_lookahead_tokens` is the served set. It defaults to the card's own set (0, 3, 6, 13) and
  accepts any of the implemented right contexts 0, 1, 3, 6 and 13. The reference deployment serves
  `[6, 13]` with default 13.
- The model imposes no limit on a file's length. Each engine step advances a file by one chunk and
  emits one token, so a file's token count grows with its length, and core stops a request when it
  reaches `--max-model-len` tokens. Size `--max-model-len` for the longest file you serve: a file of
  `N` samples emits at most `1 + (C + 1) × Q + 1` tokens, with `C = floor(N / s)`,
  `s = 1280 × (r + 1)` samples per chunk and `Q = 10 × (r + 1) + 1` at the default lookahead `r`
  (typical output is far smaller). 76,800 covers files up to 600 s (core's default decode limit) at
  either served cadence; streams need only 143. Each token costs about 4 bytes per slot of token
  history (`max_num_seqs × max_model_len` tokens) and no KV memory.
- Pass `--generation-config vllm` if the checkpoint's generation configuration or an
  `--override-generation-config` sets a token limit; the startup error says when.

Support boundary:

- Core's `/v1/realtime` is served directly, with no frontend. A unit core rejects ends that stream, as
  core's contract says.
- Complete-file transcription through core's `/v1/audio/transcriptions` is served directly, with no
  frontend. Limits of the route today:
  - Core's upload and decode caps are the operator's settings, not the model's: raise
    `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` (default 25), `VLLM_MAX_AUDIO_DECODE_DURATION_S` (default 600 s)
    and `VLLM_MAX_AUDIO_DECODE_BYTES` (default 256 MiB) for long files.
  - Requests queued in core hold their decoded audio in the API server with no bound; `max_num_seqs`
    bounds only the files actively scheduled. Put a cap in front of the server if clients are not
    trusted.
  - The model refuses `use_beam_search` and `max_completion_tokens` as options that do not apply to
    audio transcription, from the `SpeechToTextParams` fields that vllm#67151 adds. On a core without
    vllm#67151 the request guards are inactive: a client's `max_completion_tokens` ends the file early
    with `length`, and beam search can issue up to `n × max_model_len` engine requests per upload (one
    per beam per pass), so the transcription route must not be exposed to untrusted clients on such a
    core.
  - On a core without vllm#67144, a file ending in `error` or `abort` returns partial text with HTTP
    200. On a core without vllm#67151, a file ending in a server-caused `length` (the request
    reaching `--max-model-len`) returns partial text with HTTP 200 as well.
- The session factory, found through the second entry point, is for frontends.

## Evidence

Every claim names its tier, the build and the source revisions it was met on. The measurements below
were made on the earlier integration build of RFC 1 (the superseded #65544 API) and RFC 2, not on
stage 1 or stage 2; they are historical and are not claims about either. No tier has yet been met on
a stage 1 or stage 2 build.

- Historical build: the merge of RFC 1 `persistent-state-primitive-v2` @ `c384f178cd` and RFC 2
  `endpoint-plugin-lifecycle` @ `6b1ce05706` on `pdtgct/vllm` (branch `rfc12-integration-v2` @
  `c7cf80ac45`).
- Achieved (historical, integration build `c7cf80ac45`), tier 1 (CPU conformance): the CPU-tier
  tests passed, including the request guards of the transcription route (run against a core
  `SpeechToTextParams` that carries the two fields of vllm#67151). The frontend-free file route was
  not run at tier 2 (installed engine).
- Achieved (historical, integration build `c7cf80ac45`), tier 2 (installed-engine cycle), on one A40
  with the Riva frontend at `782dec8`: this repository's GPU tier passed, and unmodified Riva clients
  completed streaming, `Recognize`, Realtime WebSocket and file-route requests, for one session and
  for eight concurrent sessions.
- Partly achieved (historical, integration build `c7cf80ac45`), tier 3 (equivalence), same A40: on
  eight public clips from 20 s to 428 s, the core transcription route (sent with `language`) and
  `Recognize` returned identical transcripts, and the whole-file transcript equaled a session at the
  1120 ms file cadence. The comparison with the offline plugin has not been run on these clips.
- Not met (historical, integration build `c7cf80ac45`), tier 4 (mixed load): with one 293 s file
  request in flight, eight concurrent streams' p95 final latency rose from 0.55 s to 3.64 s, with no
  preemptions. Run file traffic on a separate server, or keep files short, until this is fixed.
- These are exploratory measurements on one GPU, not a qualification run.

## Install and test

```bash
pip install -e '.[test]'
pytest
```

Tests are tiered: unmarked tests need only this package; `engine` tests need the stage 1 vLLM build
on `PYTHONPATH`; `gpu` tests need an installed engine on CUDA; `card` tests need `NEMOTRON_CARD_DIR` set to
a local copy of the published card, which is never committed.
