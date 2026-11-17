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

To install and serve, start with the [quick start](docs/quickstart.md), which covers both stages.
The [user manual](docs/manual.md) explains the builds and their compatibility, every serving
setting, the clients, operations and troubleshooting. This README remains the source of record for
the serve commands, the measured figures and the evidence; the manual links back to it.

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

- `--max-num-seqs 64` met tier 2's serving run on one A40 at 320 ms under the hook configuration
  (stage 1 build `hook-build-20261114b` @ `9c40a6ffd6`, plugin `8166d85`): the server started,
  both segment-graph captures fit, the KV budget admits 65 requests, and 64 concurrent sessions
  completed. The largest sampled step held 38 rows; a 64-row step was not observed. Under
  `--enforce-eager`, which the command above uses, the value is interim: it comes from an
  unpublished A40 measurement at 320 ms.
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
  rows the 160 ms and 80 ms geometries retained 1.8 and 1.6 GiB. Under `--enforce-eager` the
  tier 2 figures for these arms are owed at every geometry.
- Under the hook configuration at 320 ms with `--max-num-seqs 64` (tier 2 serving run, one A40,
  stage 1 build `9c40a6ffd6`, plugin `8166d85`), the segment graphs retained 1,544 MiB in the
  profiling capture and 716 MiB in the serving capture, which took 42.6 and 41.4 s; core's graph
  pool held 0.70 GiB against its 1.57 GiB estimate. Device memory in use (`nvidia-smi`) peaked at
  16,445 MiB, on the memory-profiling step, 35.74 GiB was available for KV memory, and the server
  was ready 130.6 s after launch. The tier 2 figures at the other geometries are owed.
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
  transaction graphs) and no other graph; Evidence gives its tier 2 result on the stage 1 build. A
  command that gives neither configuration is refused, naming each unmet condition and its remedy.
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

Release 0.1.0b2 ships the option opt-in without tier 5: on the stage 1 build the step rule and
`U_api` fail at 80 ms and at 160 ms, and the body check fails at population 1 (Measured figures).

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
    --hf-overrides '{"num_lookahead_tokens": 0, "transaction_graph_populations": [63]}'
```

- The command runs the hook configuration (no `--enforce-eager`) at 80 ms (lookahead 0), with the
  default segment arms and native burst off. `--max-model-len 82513` holds a 600 s file at 80 ms.
  For 160 ms, set `num_lookahead_tokens` to 1 and `--max-model-len` to at least 78,773.
- `transaction_graph_populations` lists distinct step populations, each an integer from 1 to
  `min(max_num_seqs, max_num_batched_tokens)`. The largest entry, `N`, fixes the cell inventory: a
  chunk cell and an echo cell at every count from 1 to `N`, `2N` cells in all. The other entries
  change no capture, so `[63]` and `[1, 31, 63]` capture the same cells. An absent key or an empty
  list is off; any other malformed value is refused.
- `N` = 63 and `--max-num-seqs 64` are the values the tier 5 run on one A40 verified to fit at
  80 ms and at 160 ms (stage 1 build `9c40a6ffd6`, plugin `8166d85`): the `N = 63` engines captured
  126 cells per capture, the KV budget admitted 65 requests, and the engines served the tier 5 `Q`
  and `M` workloads with no fallback. The inventory runs, at `N` = 64, served 512 streams at concurrency 64
  in steps of up to 64 rows. Tier 5 takes `N` as the smallest count of at least 63 that fits and
  for which steps with `K ≤ N` and `E ≤ N` make up at least 99 % of served steps; at `N` = 63 they
  make up 100 % at 80 ms and 99.84 % at 160 ms. The same rule below 63, which carries no tier 5
  claim, gives a deployment `N` of 44 at 80 ms (99.10 %; serving capture 832 MiB call total, 268
  MiB transaction total) and of 36 at 160 ms (99.07 %; 794 and 180 MiB).

#### Measured figures

The tier 5 run used one NVIDIA A40 at FP32, the stage 1 build `hook-build-20261114b`
(`pdtgct/vllm` @ `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee`), card revision `ea30d66d`, `N = 63`
and `--max-num-seqs 64`, at 80 ms and at 160 ms. The transaction-level checks, the capture figures,
the body check and the revised served comparison of TXG-QUAL-008 ran at plugin `adfb24b`, whose
production code equals this release's. The first served runs, the inventory and `U_api` ran at plugin
`8166d85`, before `adfb24b` changed how fresh rows take their zero initial state; the forced fallbacks
and the step rule ran at plugin `a19621e`, with `8166d85`'s production code. Tier 5 is not met at
either geometry.

- Passed: the inventory (TXG-QUAL-025); every transaction-level step compared bit for bit at both
  geometries (TXG-QUAL-003); `Q` and `M` at 80 ms and at 160 ms (TXG-QUAL-002 to -006); forced
  fallbacks (TXG-QUAL-007); the revised served comparison (TXG-QUAL-008) at both geometries; the
  capture figures (TXG-QUAL-019).
- Failed: the step rule (TXG-QUAL-010 to -014), both groups at both geometries; `U_api`
  (TXG-QUAL-016, -017) at both geometries; the body check (TXG-QUAL-015) at population 1, one block
  per geometry. The first served `M` run at 80 ms (plugin `8166d85`) failed its original file
  criterion, equality with one `B_hook` run (123 of 124 files); TXG-QUAL-008 as revised replaced that
  criterion (see the note after the served-run table).
- Not run: tier 4 with the option on, on any build.

Step rule: `Δ` = min(median OFF1, median OFF2) − median ON, in ms, per block of 21 counted steps per
leg; a composition is material when `Δ ≥ max(5 ms, |median OFF1 − median OFF2|)` in both blocks
(TXG-QUAL-013). A group passes only when every one of its compositions is material.

| Group | P | (K, E) | 80 ms Δ, block 1 / 2 | Material | 160 ms Δ, block 1 / 2 | Material |
|---|---|---|---|---|---|---|
| one kind | 1 | (1, 0) | 6.85 / 6.72 | yes | 6.82 / 6.83 | yes |
| one kind | 31 | (31, 0) | 5.66 / 6.11 | yes | 6.00 / 5.89 | yes |
| one kind | 63 | (63, 0) | 4.38 / 4.30 | no | 4.28 / 4.26 | no |
| split | 31 | (27, 4) | 5.47 / 5.32 | yes | 5.04 / 4.92 | no |
| split | 31 | (16, 15) | 4.78 / 4.72 | no | 4.64 / 5.03 | no |
| split | 63 | (55, 8) | 3.75 / 6.42 | no | 4.10 / 3.58 | no |
| split | 63 | (32, 31) | 5.07 / 5.00 | yes | 5.25 / 5.01 | yes |

Echo-only steps, reported and not gated, gained 4.42 to 4.89 ms. U(1), the (1, 0) composition's
block 1 median (TXG-CAL-001): 15.06 ms on and 21.97 ms off at 80 ms, 17.46 ms on and 24.29 ms off at
160 ms.

Body check (TXG-QUAL-015): the option-off step time of this release (`adfb24b`) against plugin
`4cc1381`, both under `--enforce-eager` with transaction graphs off, in two blocks of legs OLD1, NEW,
OLD2 per population. Medians of 21 counted steps (ms); a block passes when NEW − min(OLD1, OLD2) ≤
max(1 ms, |OLD1 − OLD2|). These are engine step times, not served latencies.

| ms | P | Block 1: OLD1 / NEW / OLD2 | Block 2: OLD1 / NEW / OLD2 | NEW − mean(OLD), block 1 / 2 |
|---|---|---|---|---|
| 80 | 1 | 29.59 / 26.20 / 30.41 | 25.80 / 26.85 / 25.08 (**fail**) | −3.80 / +1.41 |
| 80 | 31 | 48.18 / 47.95 / 48.98 | 48.35 / 48.55 / 47.74 | −0.62 / +0.51 |
| 80 | 63 | 73.37 / 72.79 / 72.43 | 72.78 / 73.28 / 73.19 | −0.12 / +0.29 |
| 160 | 1 | 26.17 / 28.68 / 27.79 (**fail**) | 26.23 / 25.85 / 27.30 | +1.69 / −0.92 |
| 160 | 31 | 50.82 / 50.69 / 51.15 | 51.14 / 51.67 / 51.54 | −0.29 / +0.33 |
| 160 | 63 | 78.72 / 78.66 / 79.12 | 78.50 / 78.34 / 78.76 | −0.25 / −0.29 |

The check is not met: at population 1 one block per geometry fails (80 ms: NEW − min(OLD) 1.77 ms
against a bound of 1.00 ms; 160 ms: 2.50 ms against 1.62 ms), while in the other block of the same
geometry NEW is faster than both OLD legs. At populations 31 and 63 every block passes. In an
ungated bridge block (option off, `--enforce-eager` legs around one leg in the hook configuration),
the hook configuration was within the eager legs' drift at 160 ms and 0.7 ms (P 1) and 0.9 ms (P 63)
above it at 80 ms.

- `U_api` (TXG-QUAL-016, -017; plugin `8166d85`), five brackets of OFF1, ON, OFF2 per geometry at
  `N = 63`: not met at either geometry. The per-bracket rule failed in three brackets at 80 ms and in
  two at 160 ms; over the brackets, the counted median was 28.49 ms on and 37.45 ms off at 80 ms, and
  30.59 ms on and 38.68 ms off at 160 ms.

| Exactness and served runs | 80 ms | 160 ms |
|---|---|---|
| Transaction-level `Q` (concurrency 1, 32 and 63, 2,000 steps each) and `M`, plugin `adfb24b`: steps covered and equal to the option-off transaction in outputs, status, progress, queue counts and every page | 8,151 of 8,151 (`M` 2,151); 0 fallbacks | 8,158 of 8,158 (`M` 2,158); 0 fallbacks |
| `B_hook` replay of that run's schedule: requests with equal label tokens | 2,036 of 2,036 | 1,700 of 1,700 |
| Forced fallbacks (`K` > `N`, `E` > `N`, two signature mismatches; plugin `a19621e`): each counted once, outputs equal to the option-off transaction's | pass | pass |
| Served `Q` (plugin `8166d85`), 512 streams at concurrency 63: transcripts equal to `B_hook`'s | 512 of 512; 2,251 steps, all replayed | 512 of 512; 1,978 steps, all replayed |
| Served `M` (plugin `8166d85`), 48 stream slots and 15 file slots, 512 streams and 124 files: transcripts equal to `B_hook`'s | streams 512 of 512, files 123 of 124; 2,221 steps, all replayed | streams 512 of 512, files 124 of 124; 2,494 steps, all replayed |

The 80 ms file `1284-1180-0011` differs from the `B_hook` run by one comma ("I am, my dear, and all
strangers …" against "I am, my dear and all strangers …"). A second, nonqualifying `B_hook` run
differed from the first on that file and equalled the option-on run on all 124 files, which points
to `B_hook`'s own run-to-run variation under wall-clock pacing. Revised TXG-QUAL-008 comparison (two
`B_hook` runs, and each file alone; plugin `adfb24b`): pass at both geometries. Every request
completed, the option-on engine replayed every step with no fallback, and every file sent alone to an
idle server equalled both `B_hook` runs (124 of 124). Under load the option-on transcripts equalled
both `B_hook` runs on every stream and file at 160 ms and on all but `1284-1180-0011` at 80 ms, where
the option-on run kept the comma and both `B_hook` runs dropped it; alone, all three dropped it.

Graph memory on the A40 at `N = 63`, per capture (MiB unless stated; the call total covers the whole
capture call: segment graphs, cells and, in the profiling capture, a 64 MiB margin). The captures are
plugin `adfb24b`'s; available KV memory and ready time are from the served runs at plugin `8166d85`,
whose profiling transaction totals were 2,462 MiB (80 ms) and 2,702 MiB (160 ms):

| | 80 ms | 160 ms |
|---|---|---|
| Profiling capture: call / transaction total (chunk / echo) | 3,832 / 2,410 (2,144 / 266) | 4,148 / 2,676 (2,410 / 266) |
| Serving capture: call / transaction total (chunk / echo) | 986 / 422 (422 / 0) | 1,042 / 428 (428 / 0) |
| Transient peak, each capture | 2,104 | 2,154 |
| Capture time, each capture | 69.5 s and 70.6 s | 73.1 s and 74.0 s |
| Available KV memory, option on (`B_hook`) | 33.63 GiB (36.03 GiB) | 33.31 GiB (35.95 GiB) |
| `vllm serve` ready after launch (`B_hook`) | 220.8 s (100.6 s) | 179.4 s (110.4 s) |

The `(K, E)` histogram from which `N` was chosen comes from the served `Q` run on an `N` = 64 engine
(512 streams at concurrency 64). The three 160 ms steps outside `N` = 63 are (64, 0); an `N` = 63
engine runs them without transaction graphs (`chunk-uncovered`).

| | 80 ms | 160 ms |
|---|---|---|
| Served steps (split / one-kind CHUNK / echo only) | 2,654 (2,262 / 270 / 122) | 1,927 (1,554 / 182 / 191) |
| Largest `K` / `E` | 60 / 33 | 64 / 40 |
| Fraction covered at `N` = 32 / 40 / 48 / 63 | 0.8873 / 0.9717 / 0.9962 / 1.0000 | 0.9725 / 0.9974 / 0.9979 / 0.9984 |

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
  counters, including the `(K, E)` histogram, are read with `transaction_graph_counters`; under a
  multiprocess engine it runs on the worker through `AsyncLLM.collective_rpc`, which needs
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` in the engine's environment to send a function.
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

- Stage 1 build: `pdtgct/vllm` `hook-build-20261114b` @ `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee`
  (with #67583), precompiled wheel `191e5d5df2` (cu132), one NVIDIA A40 at FP32, card revision
  `ea30d66d`, frontend v0.4.0b3. Evidence: `pdtgct/nemotron-omni-port`
  `experiments/2026-11-15-plugin-tier2-hook-build/`, `experiments/2026-11-15-plugin-h1-txg-tier5-a/`
  and `experiments/2026-11-15-plugin-h1-txg-tier5-b/`.
- Partly achieved (stage 1 build `9c40a6ffd6`, plugin `8166d85`, hook configuration, default
  segment arms, transaction graphs off), tier 2 (installed-engine cycle) at 320 ms: with
  `--max-num-seqs 64` and `--max-model-len 77000`, 5,559 LibriSpeech `/v1/realtime` sessions at
  concurrency 32 all ended `stop`, with WER 3.57 % on test-clean and 7.99 % on test-other. One final,
  for `2414-128291-0020` (test-other, 1.25 s, a known empty final), was empty, and the WER counts it
  as deletions. 640 test-clean sessions at concurrency 64 all reached a final transcript (WER
  2.89 %). The server logged no preemption and no error. GPU tier at plugin `adfb24b`: 173 of 175
  tests passed, 2 skipped, 0 failed.
- Not met (stage 1 build `9c40a6ffd6`; transaction level, body check and the revised TXG-QUAL-008 at
  plugin `adfb24b`, forced fallbacks and step rule at plugin `a19621e`, the first served runs,
  inventory and `U_api` at plugin `8166d85`), tier 5 for transaction graphs at 80 ms and at 160 ms: the step rule and `U_api`
  fail at both geometries, and the body check fails at population 1. Measured figures, under
  Transaction graphs, lists what passed and what failed.
- Owed: tier 1 on the stage 1 build at the current plugin; tier 2 under `--enforce-eager` and at
  geometries other than 320 ms; tiers 3 and 4 on the stage 1 build, tier 4 also with transaction
  graphs on.

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
