# vllm-nemotron-asr-streaming

Out-of-tree plugin for cache-aware streaming Nemotron ASR on vLLM core. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card unmodified, for streaming sessions and for complete-file transcription. It targets two builds of
upstream vLLM `main` with open pull requests; a released vLLM does not provide the symbols it needs,
and `register()` fails with a message saying so.

- **Stage 1** (the model alone, on core's `/v1/realtime` and `/v1/audio/transcriptions`) is upstream
  vLLM `main` with vllm#67374, #67375, #67583, #67144 and #67151. #67374, #67375 and #67583 are RFC 1
  (the persistent-state primitive; stream admission, `AsyncLLM.generate(..., stream_admission=True)`
  and `wait_stream_retired`; and the model-state hooks through which core captures the model's
  graphs).
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

The configuration hook refuses a core whose `ModelState` lacks any of #67583's model-state hooks,
naming each missing hook.

## Serve

The default configuration serves the checkpoint's default lookahead with the default segment arms:

<!-- @spec REG-ENG-007 -->
```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --enforce-eager \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    --max-num-seqs 64 \
    --max-model-len 77000
```

- `--max-num-seqs 64` is interim and unpublished: it comes from an unpublished A40 measurement at
  320 ms under `--enforce-eager`, and qualifies neither runner configuration. The tier 2 measurement
  that verifies a value for the A40 at 320 ms is owed.
- A server serves one geometry, fixed at startup. The default is the checkpoint's own default
  lookahead, which is 3 (320 ms) on the published card; another checkpoint can declare another
  default. `num_lookahead_tokens` in `--hf-overrides` selects one of the other implemented
  geometries: 0 (80 ms), 1 (160 ms), 6 (560 ms) or 13 (1120 ms). The card declares 0, 3, 6 and 13;
  the server also accepts 1, and logs that the checkpoint does not declare it. Every stream and
  every whole file runs at the server's geometry. `supported_num_lookahead_tokens` is refused:
  remove it and select the geometry with `num_lookahead_tokens`.
- The model imposes no limit on a file's length. Each engine step advances a file by one chunk and
  emits at most `Q` tokens, and core stops a request when it reaches `--max-model-len` tokens, so
  size `--max-model-len` for the longest file you serve at the server's lookahead `r`: a file of `N`
  samples emits at most `1 + (floor(N / s) + 1) × Q + 1` tokens, with `s = 1280 × (r + 1)` samples
  per chunk and `Q = 10 × (r + 1) + 1` (typical output is far smaller). A 600 s file (core's
  default decode limit) needs at most 82,513 tokens at lookahead 0, 78,773 at 1, 76,918 at 3,
  76,114 at 6 and 75,578 at 13. `--max-model-len 77000` therefore holds a 600 s file at lookahead
  3, 6 and 13; an override to 0 or 1 needs a correspondingly larger `--max-model-len`, at least
  82,513 or 78,773. Streams need only 143. Each token costs about 4 bytes per slot of token history
  (`max_num_seqs × max_model_len` tokens) and no KV memory.
- The default segment arms are the `eager-graphed` encoder, captured at every CHUNK count from 1 to
  `max_num_seqs`, and the `dense-graphed` RNN-T decoder, captured at padded population tiers; card
  translation declares both. Their graphs (the segment graphs) are captured for the server's one
  geometry only. Segment-graph memory grows faster than linearly with `--max-num-seqs`, one encoder
  graph per CHUNK count. In an unpublished A40 measurement at 320 ms under `--enforce-eager`, the
  default arms retained about 1.1, 2.3 and 17.4 GiB at `--max-num-seqs` 32, 64 and 128; device
  memory in use during capture, weights included, peaked at 7.0, 18.3 and 44.4 GiB of the card's
  45 GiB, and startup took about 41, 78 and 182 s, against about 25 s for the eager arms. At 64
  rows the 160 ms and 80 ms geometries retained 1.8 and 1.6 GiB. The tier 2 figures for these arms
  are owed at every geometry, and under the hook configuration (below) every figure is owed: the
  retained memory, capture peak and startup time there are unmeasured.
- `--hf-overrides '{"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"}'`
  selects the eager arms, the reference the graphed arms are tested against. The eager arms capture
  nothing, so they need `--enforce-eager`. The compiled `dense-graphed` encoder and
  `chunk_bucket_graph_populations` also need `--enforce-eager`.
- The persistent-state model needs synchronous scheduling, no prefix caching and a single device.
  Core resolves synchronous scheduling and no prefix caching for this model, so the command omits
  `--no-async-scheduling` and `--no-enable-prefix-caching`. Other settings it refuses (speculative
  decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator, hierarchical sparse
  attention, fault tolerance, elastic expert parallelism, any parallel size other than 1) fail at
  startup with a message naming each.
- The outer runner captures no graphs and compiles nothing. Two runner configurations are accepted.
  Under `--enforce-eager`, which the command above uses, the model captures its segment graphs itself
  on core's first memory-profiling step. Under the hook configuration (no `--enforce-eager`, with
  `-cc.cudagraph_mode none` and `-cc.mode none`, on CUDA), core captures them through #67583's
  capture hook and charges them in its graph-memory estimate. The hook configuration requires at
  least one graph the hook captures (the `eager-graphed` encoder, the `dense-graphed` decoder or
  transaction graphs) and no other graph, and has not met tier 2 on any build. A command that gives neither
  configuration is refused, naming each unmet condition and its remedy.
- Each capture of the segment graphs logs one INFO line naming the runner configuration, the arms,
  the geometry, `max_num_seqs`, the retained bytes and the capture time. A capture that cannot
  allocate fails startup with those figures and a remedy: a smaller `--max-num-seqs`, or the eager
  arms with `--enforce-eager`.
- The model prepares its execution memory on core's memory-profiling step, before core sizes the KV
  cache and state pool, so that memory is budgeted. If you set `--kv-cache-memory-bytes`, core takes
  that budget as given and charges neither the model's execution memory nor the segment graphs:
  leave room for both. Under the hook configuration the startup log then carries a WARNING with the
  uncharged bytes.
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
- The session factory, found through the second entry point, is for frontends. Its `open` takes a
  locale only; it needs a frontend release that opens streams by locale only (Install and test).

## Transaction graphs (opt-in)

Transaction graphs replay a streaming step's device body and its scatters from CUDA graphs. A
step's CHUNK rows replay a chunk cell captured at exactly their count `K`, and its other rows
(REPLAY, park echo, FLUSH) replay an echo cell captured at exactly their count `E`. A step that the
cells do not cover runs wholly without transaction graphs, as the option-off transaction, and is
counted. Every covered step's outputs and resident state must equal the option-off transaction's
bit for bit. The option is `transaction_graph_populations`, off by default; core captures the cells
through #67583's capture hook, into its own graph pool.

#### Serve command

<!-- @spec TXG-DOC-001 -->
```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    -cc.cudagraph_mode none \
    -cc.mode none \
    --max-num-seqs 64 \
    --max-model-len 82513 \
    --hf-overrides '{"num_lookahead_tokens": 0, "transaction_graph_populations": [64]}'
```

- The command runs the hook configuration (no `--enforce-eager`) at 80 ms (lookahead 0), with the
  default segment arms and native burst off. `--max-model-len 82513` holds a 600 s file at 80 ms.
  For 160 ms, set `num_lookahead_tokens` to 1 and `--max-model-len` to at least 78,773.
- `transaction_graph_populations` lists distinct step populations, each an integer from 1 to
  `min(max_num_seqs, max_num_batched_tokens)`. The largest entry, `N`, fixes the cell inventory: a
  chunk cell and an echo cell at every count from 1 to `N`, `2N` cells in all. The other entries
  change no capture, so `[63]` and `[1, 31, 63]` capture the same cells. An absent key or an empty
  list is off; any other malformed value is refused.
- `N` = 64 and `--max-num-seqs 64` are the values of the design's first A40 measurement plan, not
  verified values (owed: the `N` and `--max-num-seqs` that the tier 5 run on the A40 verified to
  fit, on the stage 1 build). For tier 5, `N` is the smallest count of at least 63 that fits and
  for which steps with `K ≤ N` and `E ≤ N` make up at least 99 % of served steps.

#### Measured figures

Every figure below is owed until tier 5 is met on the A40 at FP32 on the stage 1 build, separately
at 80 ms and at 160 ms.

- Build, device, geometry and `N` of the figures: (owed: tier 5 on A40, stage 1 build; the build's
  and plugin's revisions, 80 ms and 160 ms, and `N` for each).
- Graph memory on the A40 for each capture, profiling and serving (call total, transaction total
  and per-kind totals): (owed: tier 5 on A40, stage 1 build).
- Transient peak of each capture: (owed: tier 5 on A40, stage 1 build).
- Capture time of each capture: (owed: tier 5 on A40, stage 1 build).
- The `(K, E)` histogram of the served qualification workload from which `N` was chosen: (owed:
  tier 5 on A40, stage 1 build).

On an earlier in-tree copy of this model, a whole-transaction prototype on an A40 at 80 ms saved
5.2 to 6.4 ms per homogeneous continuing CHUNK step at populations 1, 31 and 63 (unpublished
project measurement). It predicts neither the saving on split steps, at 160 ms nor in this plugin;
that saving is owed (tier 5 on the A40, stage 1 build).

#### Memory and the startup log

- A fixed `--kv-cache-memory-bytes` must leave room for the transaction graphs and the segment
  graphs, which core does not charge under a fixed budget; the startup log reports the uncharged
  bytes in a WARNING. With a profiled budget, core's graph-memory estimate charges the profiling
  capture, and startup fails if the serving capture retains more than the profiling capture.
- Each capture logs one INFO line with the cells, `N`, the call and transaction totals, the
  transient peak and the capture time. Post-warmup verification logs one INFO line when it passes,
  and shutdown logs one INFO line with replays, split steps and fallbacks by reason.
- Startup captures `2N` cells twice, beside the segment graphs' two captures. If startup exceeds
  `VLLM_ENGINE_READY_TIMEOUT_S` (default 600 s), raise it.

#### Limits

- One geometry per server: transaction graphs need 80 or 160 ms (lookahead 0 or 1). The published
  card's default, 320 ms, and 560 and 1120 ms are refused, with the remedy `num_lookahead_tokens` 0
  or 1.
- `N` is bounded by `max_num_seqs` (and by `max_num_batched_tokens`, and under
  `num_gpu_blocks_override` by the override minus 1).
- Steps with more than `N` CHUNK rows (`chunk-uncovered`) or more than `N` other rows
  (`echo-uncovered`) run without transaction graphs and are counted. A step whose structural inputs
  differ from its cells' (`signature`) also runs without them and is counted; a direct engine
  stream client that sets an endpoint policy causes one; on plugin-packed input a nonzero `signature`
  count is a defect. The
  counters, including the `(K, E)` histogram, are read with `transaction_graph_counters`.
- `--enforce-eager` with `transaction_graph_populations` is refused: core runs no capture hook
  there.
- Native burst (`experimental_native_burst: true`) with transaction graphs is refused.
- Segment arms other than the defaults (`eager-graphed` encoder, `dense-graphed` decoder) are
  refused, and so is `chunk_bucket_graph_populations`.
- The server needs a frontend release that opens streams by locale only; the pinned release is
  v0.4.0b3 (Install and test).
- Tier 4 (mixed load) is claimed separately with the option on; it has not been run with the option
  on, on any build.

## Evidence

Every claim names its tier, the build and the source revisions it was met on.

- Owed: tiers 1 to 4 with the default segment arms on the stage 1 build (with #67583), at the current
  plugin, for each runner configuration claimed; tier 2 also verifies the serve command's
  `--max-num-seqs`. Tier 5 is owed at 80 ms and at 160 ms (Transaction graphs).

Every entry below is historical. The stage 1 entries were made on a stage 1 build without #67583,
at plugin revisions before the default graphed arms and the one-geometry server; the other entries
were made on the earlier integration build of RFC 1 (the superseded #65544 API) and RFC 2. None is
a claim about the current plugin on stage 1 or stage 2.

- Historical stage 1 build (without #67583): `pdtgct/vllm` `stage1-build-20261113` @ `b7e7a20c26` (upstream `main` `cf8fce503c` +
  vllm#67375 + vllm#67151), precompiled wheel `191e5d5df2` (cu132), one NVIDIA A40. Evidence:
  `pdtgct/nemotron-omni-port` `experiments/2026-11-13-two-stage-stage1-a40/`.
- Achieved (historical, stage 1 build without #67583, plugin `5acd9d2`), tier 1 (CPU conformance): 1203 passed, 43 skipped.
- Achieved (historical, stage 1 build without #67583; serving at plugin `a850916`, GPU tier at
  `5acd9d2`, which changes only tests since),
  tier 2 (installed-engine cycle): the GPU tier passed (35 passed, 1 skipped), and 5559 LibriSpeech
  `/v1/realtime` sessions at concurrency 32 each reached a final transcript with none failed.
  Tiers 3 and 4 have not been run on stage 1.

- Historical build: the merge of RFC 1 `persistent-state-primitive-v2` @ `c384f178cd` and RFC 2
  `endpoint-plugin-lifecycle` @ `6b1ce05706` on `pdtgct/vllm` (branch `rfc12-integration-v2` @
  `c7cf80ac45`).
- Achieved (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 1 (CPU
  conformance): the CPU-tier
  tests passed, including the request guards of the transcription route (run against a core
  `SpeechToTextParams` that carries the two fields of vllm#67151). The frontend-free file route was
  not run at tier 2 (installed engine).
- Achieved (historical, integration build `c7cf80ac45`, plugin `4e4fd6f` for the GPU tier; the
  unmodified-client runs record no plugin revision), tier 2 (installed-engine cycle), on one A40
  with the Riva frontend at `782dec8`: this repository's GPU tier passed, and unmodified Riva clients
  completed streaming, `Recognize`, Realtime WebSocket and file-route requests, for one session and
  for eight concurrent sessions.
- Partly achieved (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 3 (equivalence), same A40: on
  eight public clips from 20 s to 428 s, the core transcription route (sent with `language`) and
  `Recognize` returned identical transcripts, and the whole-file transcript equaled a session at the
  1120 ms file cadence. The comparison with the offline plugin has not been run on these clips.
- Not met (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 4 (mixed load): with one 293 s file
  request in flight, eight concurrent streams' p95 final latency rose from 0.55 s to 3.64 s, with no
  preemptions. Run file traffic on a separate server, or keep files short, until this is fixed.
- These are exploratory measurements on one GPU, not a qualification run.

## Install and test

Install the package with its PEP 735 `test` dependency group. The group pins `vllm-riva-frontend`
and its seam conformance suite, `vllm-riva-frontend-dev`, at one commit of the frontend repository;
the `test` extra installs only pytest.

Compatibility: the pinned commit, `125ac198846d27ca17a1aaa9f1e17c009b40c676`, is the frontend
release tag v0.4.0b3, the counterpart release, which opens streams by locale only. The protocol has
no version negotiation, so the plugin and the frontend are released and qualified as a pair. An
earlier frontend that reads the factory's served cadences at startup, such as v0.4.0b1, fails its
own startup against this plugin. The v0.4.0b3 release notes state that plugin `main` at `d5c5197`
is incompatible with it: that revision's streams end `internal`.

```bash
# pip 25.1 or later
pip install -e . --group test
# or uv
uv sync --group test
```

Either command installs the two frontend distributions from the pinned commit as non-editable VCS
installs, as tier 1 acceptance requires.

Run the suite:

```bash
pytest
```

Tests are tiered: unmarked tests need only this package; `engine` tests need the stage 1 vLLM build
on `PYTHONPATH`; `gpu` tests need an installed engine on CUDA; `card` tests need `NEMOTRON_CARD_DIR` set to
a local copy of the published card, which is never committed.

Tier 1 acceptance runs the seam conformance tests and the pin tests together, with
`NEMOTRON_CONFORMANCE_ACCEPTANCE=1`:

```bash
NEMOTRON_CONFORMANCE_ACCEPTANCE=1 PYTHONPATH=<stage 1 vLLM tree> \
    python -m pytest -q -p no:cacheprovider \
    tests/session/test_frontend_conformance.py tests/session/test_conformance_pin.py
```

The acceptance run fails when either frontend distribution is an editable or local install, or was
installed from a commit other than the pinned one. Tier 1 is met only by an acceptance run in which
no test fails or is skipped.
