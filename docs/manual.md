# User manual

This manual covers `vllm-nemotron-asr-streaming` at `main` on the target core, RFC 1's state
layer (`pdtgct/vllm` at `cbf13fed4a`). For the shortest path to a running server, use the
[quick start](quickstart.md).

No acceptance tier has been met on the target core. The last release, `v0.1.0b2`, ran on the
earlier state-pool build (`hook-build-20261114b`), where it partly met tier 2 (the installed-engine
cycle) at 320 ms in the hook configuration and did not meet tier 5 (transaction graphs), on one
NVIDIA A40 at FP32. Those runs are a historical record for this plugin until they are repeated on
the target core. The [evidence document](evidence.md) defines the tiers and lists the revisions of
each run. A command in this manual has been run as written only if the evidence document records
it.

## Contents

- [What the plugin is](#what-the-plugin-is)
- [Builds and compatibility](#builds-and-compatibility)
- [Installation](#installation)
- [Serving configurations](#serving-configurations)
- [Clients](#clients)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)
- [Limits and the support boundary](#limits-and-the-support-boundary)
- [Testing the plugin](#testing-the-plugin)
- [Evidence](#evidence)
- [Future roadmap](#future-roadmap)

## What the plugin is

`vllm-nemotron-asr-streaming` is an out-of-tree vLLM plugin. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card, unmodified, to cache-aware streaming sessions and to complete-file transcription. The model
keeps its encoder caches and decoder state in one page of its state group, which core allocates
per request, keeps while a stream waits for its next input and frees when the request ends. The
same model serves:

- live streams, on core's `/v1/realtime`;
- complete files, on core's `POST /v1/audio/transcriptions`.

A frontend's streaming routes (Riva gRPC `StreamingRecognize`, the Riva Realtime WebSocket) use the
session factory, which is not available on the target core
([The session factory](#the-session-factory)).

The distribution `vllm-nemotron-asr-streaming` (import package `vllm_nemotron_asr_streaming`) has
two entry points:

| Entry-point group | Name | Target |
| --- | --- | --- |
| `vllm.general_plugins` | `nemotron_asr_streaming` | `vllm_nemotron_asr_streaming:register` |
| `vllm_riva_frontend.session_factories` | `nemotron_asr_streaming` | `vllm_nemotron_asr_streaming.session:create_session_factory` |

`register()` checks that the installed vLLM carries the state layer the model uses: the six
model-state hooks of vllm#67583 and `take_failed_requests` on `ModelState`, the `recomputable`
parameter of `MambaManager.layer_config`, the `PERSISTENT_STATE` member of
`MambaAttentionBackendEnum` and `SupportsPersistentState`. It then registers the following, without
importing the model or touching a device:

- the architecture `Nemotron3_5AsrStreamingForRNNT` (a lazy reference) and its
  model-configuration hook;
- the config format `nemotron_asr_streaming`, which translates the Hub card;
- the tokenizer mode `nemotron_asr_streaming` and the renderer of the same name, which make the
  control tokens special and the park token the EOS.

The card names the architecture `Nemotron3_5AsrForRNNT`, which is the offline plugin's key. This
plugin never writes that key or its configuration hook, so both plugins can be installed in one
environment. When `VLLM_PLUGINS` is an allowlist, include `nemotron_asr_streaming`. The
configuration hook refuses a core whose `ModelState` lacks any of #67583's model-state hooks,
and names each missing hook.

Frontends load the session factory through the second entry point. On the target core
`create_session_factory` refuses ([The session factory](#the-session-factory)).

## Builds and compatibility

### The builds

No released vLLM provides the interfaces the plugin needs. The plugin targets RFC 1's state layer
on a fork branch of upstream vLLM `main`, which carries changes that upstream has not merged
([README, Builds](../README.md#builds)):

| Build | `pdtgct/vllm` branch | Commit | Carries |
| --- | --- | --- | --- |
| Stage 1 (the target core) | `pstate-impl-20261122` | `cbf13fed4aff1a72cf0dfbb5cd4cb95d43773e52` | RFC 1's state layer, vllm#67583's model-state hooks and the state-group layout accessor `ModelState.state_group_layouts`, on upstream `main` at `842f74fa8dff2c6792bd68b021c92d8fbf2b93bc` |
| Stage 2 | none recorded | none recorded | stage 1 plus RFC 2 (#65546 and #67294) and a frontend endpoint plugin |

RFC 1's state layer keeps each request's state in one page of a non-recomputable state group:
core allocates the page when it first schedules the request, never preempts the request, keeps
the page while a stream waits for its next input, admits a stream when `AsyncLLM.generate` opens
it, and folds each later input into the same request. vllm#67583's hooks let core capture the
model's own graphs. The accessor gives transaction graphs the state pages' addresses.

Stage 1 carries neither vllm#67144 nor #67151, so core's transcription route has the limits of a
core without them ([Limits](#limits-and-the-support-boundary)). It changes no compiled code, so it
installs over vLLM's precompiled wheel of its upstream base. No stage 2 build of the target core is
recorded, and the frontend's streaming routes need the session factory, which is not available on
the target core.

### The session factory

The session factory serves a frontend's streams. Its design binds a session to a stream-admission
client path (`generate(..., stream_admission=True)` and `wait_stream_retired`) that the target core
does not have: core admits a state stream itself when `AsyncLLM.generate` is given an input
generator. On the target core `create_session_factory` therefore refuses, so a frontend's routes
that use it do not run. Porting the factory to core's admission is planned
([high-level design, Future roadmap](high-level-design.md#future-roadmap)).

The factory's `open` takes a locale only. The plugin and the frontend are released and qualified
as a pair, because their protocol has no version negotiation
([README, Install and test](../README.md#install-and-test)). The last released pair, plugin
`v0.1.0b2` with frontend `v0.4.0b3`, ran on the state-pool build. A frontend that reads the
factory's served cadences at startup, such as `v0.4.0b1`, fails its own startup against this
plugin.

### Mismatched vLLM builds

| vLLM | Result |
| --- | --- |
| A released vLLM, the state-pool build (`hook-build-20261114b`), or another build without RFC 1's state layer or vllm#67583's hooks | `register()` raises `RuntimeError`: `vllm-nemotron-asr-streaming requires a vLLM build with RFC 1's state layer and vllm#67583's model-state hooks; this build lacks ...`, naming each missing item |
| A state-layer build without the state-group layout accessor | the model serves; transaction graphs are refused, naming `ModelState.state_group_layouts` |
| Stage 1 (no #67144 or #67151) | the server starts, but core's transcription route returns partial text with HTTP 200 for some incomplete files, and the request guards are inactive ([Limits](#limits-and-the-support-boundary)) |
| Stage 1 with the frontend | the frontend needs RFC 2, which stage 1 does not carry |

## Installation

Use a fresh virtual environment, with Python 3.11 or 3.12. The plugin declares Python
`>=3.11,<3.15`.

1. Install the target core. `setup.py` reads `VLLM_PRECOMPILED_WHEEL_COMMIT` and
   `VLLM_PRECOMPILED_WHEEL_VARIANT`. The commit must be the full 40-character hash of the upstream
   base in the table above. The variant `cu132` has not been checked against this commit.

   ```bash
   git clone --branch pstate-impl-20261122 --single-branch https://github.com/pdtgct/vllm.git
   cd vllm
   git checkout cbf13fed4aff1a72cf0dfbb5cd4cb95d43773e52
   VLLM_USE_PRECOMPILED=1 \
   VLLM_PRECOMPILED_WHEEL_COMMIT=842f74fa8dff2c6792bd68b021c92d8fbf2b93bc \
   VLLM_PRECOMPILED_WHEEL_VARIANT=cu132 \
       pip install -e .
   cd ..
   ```

2. Install the plugin from `main`. Release `v0.1.0b2` needs the state-pool build and does not run
   on the target core.

   ```bash
   pip install "git+https://github.com/pdtgct/vllm-nemotron-asr-streaming-plugin.git@main"
   ```

   Do not install `vllm-riva-frontend-dev` into a serving environment. It is the frontend's
   conformance suite, which only this plugin's tests use.

If `VLLM_PLUGINS` is set, it is an allowlist: include `nemotron_asr_streaming`.

## Serving configurations

This section explains each flag of the [README's serve command](../README.md#serve) and groups the
other serving settings by the decision each one makes:

```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --enforce-eager \
    --no-async-scheduling \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    --max-num-seqs 64 \
    --max-model-len 77000
```

`--max-num-seqs 64` is an interim value from an unpublished A40 measurement at 320 ms under
`--enforce-eager`; no tier has verified a value on the target core. On the earlier state-pool
build, `--max-num-seqs 64` fit one A40 at 320 ms in the hook configuration
([evidence](evidence.md#tier-2-on-the-state-pool-build)).

`--config-format nemotron_asr_streaming` and `--tokenizer-mode nemotron_asr_streaming` are required.
They select the plugin's translation of the Hub card and its tokenizer mode. Core resolves no
prefix caching for this model, so the command omits `--no-enable-prefix-caching`; passing it is
harmless.

### Scheduling

With transaction graphs off, the plugin accepts asynchronous scheduling: core's per-request barrier
for state models advances each stream one step at a time. No run has qualified asynchronous
scheduling with this model, so the serve command sets `--no-async-scheduling`. On the target core
an unset option means asynchronous scheduling. With transaction graphs on, the plugin requires
`--no-async-scheduling` ([Transaction graphs](#transaction-graphs-opt-in)).

### Runner configuration

The outer model runner must capture no graphs and compile nothing. Two runner configurations do
that:

| | Hook configuration | `--enforce-eager` |
| --- | --- | --- |
| Flags | no `--enforce-eager`; `-cc.cudagraph_mode none -cc.mode none`; a CUDA platform | `--enforce-eager` |
| Who captures the segment graphs | core, through #67583's capture hook | the model, on core's first memory-profiling step |
| Graph memory charged by | core's graph-memory estimate | the profiled budget |
| Transaction graphs | accepted (opt-in) | refused |
| Eager arms | refused (no graph to capture) | accepted |
| Measured | tier 2 partly met at 320 ms on the earlier state-pool build; not run on the target core | `--max-num-seqs 64` is interim, from an unpublished A40 measurement |

The hook configuration requires at least one graph that the hook captures: the `eager-graphed`
encoder, the `dense-graphed` decoder or transaction graphs. The default segment arms include two of
these. Startup refuses a command that gives neither configuration, and the message names each unmet
condition and its remedy ([Troubleshooting](#troubleshooting)).

To run the hook configuration, replace `--enforce-eager` in the command above with
`-cc.cudagraph_mode none -cc.mode none`. Use `--enforce-eager` for the eager arms, the compiled
`dense-graphed` encoder or `chunk_bucket_graph_populations`, or on a platform other than CUDA.

### Lookahead selection

The lookahead, `num_lookahead_tokens`, fixes the chunk duration, and this manual names each
lookahead by that duration: lookahead 3 is 320 ms. A server serves one lookahead, fixed
at startup. Clients cannot choose it per request: every stream and every whole file runs at the
server's lookahead.

- The default is the lookahead that the checkpoint's model configuration sets. The published card
  sets 3 (320 ms); another checkpoint can set another default.
- `num_lookahead_tokens` in `--hf-overrides` selects another lookahead. The implemented values are:

  | `num_lookahead_tokens` | Chunk duration | Declared by the published card |
  | --- | --- | --- |
  | 0 | 80 ms | yes |
  | 1 | 160 ms | no; the server accepts it and logs a WARNING |
  | 3 | 320 ms | yes (the card's default) |
  | 6 | 560 ms | yes |
  | 13 | 1120 ms | yes |

  To select 0 (80 ms), 1 (160 ms), 6 (560 ms) or 13 (1120 ms), set `num_lookahead_tokens` in
  `--hf-overrides`, for example `--hf-overrides '{"num_lookahead_tokens": 13}'`.
- `supported_num_lookahead_tokens` is refused. Remove it and select the lookahead with
  `num_lookahead_tokens`.

A larger lookahead gives the encoder more right context per chunk, so two lookaheads can give
different transcripts for the same audio. Tier 2 has been measured only at 320 ms.

### Sizing `--max-model-len`

The model sets no limit on a file's length. Each engine step advances a file by one chunk and emits
at most `Q` tokens. Core stops a request when it reaches `--max-model-len` tokens. Size
`--max-model-len` for the longest file you serve at the server's lookahead `r`. Typical output is
far smaller than this bound.

- a file of `N` samples at 16 kHz emits at most `1 + (floor(N / s) + 1) × Q + 1` tokens, with
  `s = 1280 × (r + 1)` samples per chunk and `Q = 10 × (r + 1) + 1`;
- a 600 s file, core's default decode limit, needs at most:

  | Lookahead | 0 | 1 | 3 | 6 | 13 |
  | --- | --- | --- | --- | --- | --- |
  | Tokens for 600 s | 82,513 | 78,773 | 76,918 | 76,114 | 75,578 |

`--max-model-len 77000` therefore holds a 600 s file at lookahead 3, 6 and 13. An override to 0
needs at least 82,513, and an override to 1 at least 78,773. Streams need only 144 tokens, and a `--max-model-len` below 144 is refused. If you raise
`VLLM_MAX_AUDIO_DECODE_DURATION_S`, raise `--max-model-len` with it. Each token costs no KV memory
and about 4 bytes of token history per slot (`max_num_seqs × max_model_len` tokens in all).

### `--max-num-seqs` and graph memory

`--max-num-seqs` sets the number of requests the runner holds at once, and so bounds concurrent
streams and actively scheduled files; `--max-num-active-seqs`, when set, lowers the number core
admits. Each running request holds one state page, and core refuses to start when the KV cache
holds fewer pages than requests that can be active. `--max-num-seqs` also sets the size of the
segment graphs. The default `eager-graphed` encoder is
captured at every CHUNK count from 1 to `max_num_seqs`, so segment-graph memory grows faster than
linearly with `--max-num-seqs`.

The table gives two memory figures. Retained memory is what the captured graphs hold for the life
of the server. The capture peak is the device memory in use during capture, weights included.
Capture fails when the peak does not fit, even when the retained memory would.

Figures at 320 ms on one A40, measured with release `v0.1.0b2` on the earlier state-pool build
([evidence](evidence.md#tier-2-on-the-state-pool-build)):

| Runner configuration | `--max-num-seqs` | Retained | Capture peak (device memory in use) | Startup | Status |
| --- | --- | --- | --- | --- | --- |
| Hook configuration | 64 | 1,544 MiB (profiling capture), 716 MiB (serving capture) | 16,445 MiB (`nvidia-smi`, on the memory-profiling step) | ready 130.6 s after launch | tier 2 serving run, state-pool build |
| `--enforce-eager` | 32 | about 1.1 GiB | 7.0 GiB | about 41 s | unpublished measurement |
| `--enforce-eager` | 64 | about 2.3 GiB | 18.3 GiB | about 78 s | unpublished measurement |
| `--enforce-eager` | 128 | about 17.4 GiB | 44.4 GiB of 45 GiB | about 182 s | unpublished measurement |

In the hook configuration run, 35.74 GiB was available for KV memory, and the KV budget admitted 65
requests. The largest sampled step held 38 rows; no 64-row step was observed. Under
`--enforce-eager` at 64 rows, 160 ms retained 1.8 GiB and 80 ms retained 1.6 GiB (unpublished
measurement). Tier 2 figures under `--enforce-eager`, and at lookaheads other than 320 ms, are owed.

If a segment-graph capture cannot allocate, startup fails and reports the capture's figures. The
remedies are a smaller `--max-num-seqs`, or the eager arms with `--enforce-eager`.

### The eager arms

A segment arm is one execution variant of the encoder (`encoder_execution_arm`) or the RNN-T
decoder (`decode_dispatch_arm`). The default arms are the `eager-graphed` encoder, captured at every
CHUNK count from 1 to `max_num_seqs`, and the `dense-graphed` decoder, captured at padded population
tiers. Card translation declares both. Their graphs, the segment graphs, are captured for the
server's one lookahead only. The eager arms capture nothing. The tests use them as the reference for the graphed arms:

```bash
--hf-overrides '{"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"}'
```

The overrides `encoder_execution_arm` `eager` and `decode_dispatch_arm` `dense-eager` select the
eager arms. Because they capture nothing, they need `--enforce-eager`. With the eager arms, startup
took about 25 s in the unpublished A40 measurement at 320 ms. The compiled `dense-graphed` encoder and `chunk_bucket_graph_populations` also need
`--enforce-eager`. To combine the eager arms with another lookahead, put all the keys in one
`--hf-overrides` object.

### `--kv-cache-memory-bytes`

With the extensible KV cache off (`--no-enable-extensible-kv-cache`), core profiles memory and
sizes the KV cache, whose one group holds the state pages, from the remaining budget. The model
prepares its execution memory on core's memory-profiling step, so the budget includes that memory.
In the hook configuration, core's graph-memory estimate there also charges the profiling capture.

A fixed `--kv-cache-memory-bytes` changes this. Core takes the budget as given and charges neither
the model's execution memory nor the segment graphs (nor transaction graphs). Leave room for the
segment graphs and the execution memory. In the hook configuration, the startup log then carries a
WARNING with the uncharged bytes. The same WARNING appears when core's graph-memory estimate is
disabled (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) with the extensible KV cache off. Under the
extensible KV cache, core's default on CUDA, core sizes the KV cache from the memory left after
the capture, whatever the budget or estimate setting, and the startup log reports the same bytes
as measured in an INFO line.

### Refused settings

The persistent-state model needs no prefix caching and a single device. Startup fails and names
each refused setting. The refused settings are:

- prefix caching set explicitly on;
- native burst (`experimental_native_burst`), which the state layer does not support;
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the CuMem allocator,
  hierarchical sparse attention, fault tolerance or elastic expert parallelism;
- any of `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`,
  `decode_context_parallel_size` or `prefill_context_parallel_size` other than 1;
- an outer runner that is in neither runner configuration;
- `supported_num_lookahead_tokens`, or a `num_lookahead_tokens` that is not 0, 1, 3, 6 or 13;
- a generation configuration with an EOS other than the park token, or a token limit
  (`max_tokens`) from the checkpoint's generation configuration or `--override-generation-config`;
  pass `--generation-config vllm`;
- with transaction graphs on, every setting outside their restricted startup
  ([Transaction graphs](#transaction-graphs-opt-in)).

### Transaction graphs (opt-in)

Transaction graphs replay a streaming step's device body and its scatters from CUDA graphs. The
option is `transaction_graph_populations`, off by default. Core captures the graphs, called cells,
through #67583's capture hook into its own graph pool. A step's CHUNK rows replay a chunk cell
captured at exactly their count `K`. Its other rows (REPLAY, park echo, FLUSH) replay an echo cell
captured at exactly their count `E`. A step that no cell covers runs wholly without transaction
graphs, as the option-off transaction, and is counted. Every covered step's outputs and resident
state must equal the option-off transaction's bit for bit.

The option is opt-in and has met no tier. Release `v0.1.0b2` did not meet tier 5, the
transaction-graph acceptance tier, on the state-pool build: the step rule and `U_api` failed at
80 ms and at 160 ms. Tier 5 has not been run on the target core. The
[evidence document](evidence.md#transaction-graphs-tier-5) gives the measured figures and what
passed and failed.

#### Serve command

<!-- @spec TXG-DOC-001 -->
```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    -cc.cudagraph_mode none \
    -cc.mode none \
    --no-async-scheduling \
    --max-num-seqs 64 \
    --num-gpu-blocks-override 65 \
    --max-model-len 82513 \
    --hf-overrides '{"num_lookahead_tokens": 0, "transaction_graph_populations": [63]}'
```

The command runs the hook configuration (no `--enforce-eager`) at 80 ms (lookahead 0), with the
default segment arms and native burst off. `--max-model-len 82513` holds a 600 s file at 80 ms. For
160 ms, set `num_lookahead_tokens` to 1 and `--max-model-len` to at least 78,773.

On the target core the option starts only under a restricted startup, which the command sets:

| Setting | Why |
| --- | --- |
| `--no-async-scheduling` | the baseline and the qualification run synchronously, and the worker's call order under asynchronous scheduling is unverified with cells; an unset value means asynchronous scheduling and is refused |
| `-cc.cudagraph_mode none -cc.mode none`, both explicit | the qualification covers explicit NONE only: a mode that core reduces to NONE only when it initializes the KV cache leaves the runner capturing nothing, as explicit NONE does, but no transaction-graph run has shown its served equality; an unset mode is refused |

The option starts under every KV-cache sizing core offers: core's default, the extensible KV
cache; a profiled budget with `--no-enable-extensible-kv-cache`; a fixed
`--num-gpu-blocks-override`; and a fixed `--kv-cache-memory-bytes`. Every capture addresses only
block 0 of the state pages, the one page core backs at every capture, and the cells reach each
held page by its block ID when they replay. The command fixes the block count:
`--num-gpu-blocks-override 65` holds the null block and one page for each of the 64 requests that
`--max-num-seqs 64` lets be active. No KV-cache sizing, core's default included, carries a
verified claim on the target core (below).

The state pages must number at least the admission limit plus one (`--max-num-active-seqs`, or
`--max-num-seqs` when it is unset). Core refuses fewer when it builds its scheduler, after the
captures, so the plugin refuses an override below that count at startup; to run a smaller
override, lower the admission limit with it. Under a fixed or profiled budget the plugin refuses
a smaller page count when the serving capture first reads the state pages, before it captures any
cell. Under the extensible KV cache the final count is known only after the capture, and core's
scheduler refuses a count that is too small ([Troubleshooting](#troubleshooting)). Under the
override, core still runs its graph-memory estimate, over a profiling KV cache of 65 blocks, and
under a profiled budget with the extensible KV cache off, over a profiling KV cache of one block.
A fixed `--kv-cache-memory-bytes` runs no estimate, and under core's default sizing core sizes
the KV cache after the captures, from the memory they leave
([Memory and the startup log](#memory-and-the-startup-log)).

`transaction_graph_populations` lists distinct step populations. Each is an integer from 1 to
`min(max_num_seqs, max_num_batched_tokens)` and at most core's admission limit
(`--max-num-active-seqs`, or `--max-num-seqs` when it is unset): core never runs a step with more
rows than that limit, so a larger entry would capture cells that never replay, and startup
refuses it. The largest entry, `N`, fixes the cell inventory: a chunk cell and
an echo cell at every count from 1 to `N`, `2N` cells in all. The other entries change no capture,
so `[63]` and `[1, 31, 63]` capture the same cells. An absent key or an empty list turns the option
off. Any other malformed value is refused.

On the state-pool build, the tier 5 run on one A40 verified that `N` = 63 and `--max-num-seqs 64`
fit at 80 ms and at 160 ms: the `N = 63` engines captured 126 cells per capture and served the
tier 5 `Q` and `M` workloads with no fallback. No block count or `N` has been verified on the target
core.

Tier 5 chooses `N` as the smallest count of at least 63 that fits and for which steps with `K ≤ N`
and `E ≤ N` make up at least 99 % of served steps. On the state-pool build at `N` = 63, they made
up 100 % at 80 ms and 99.84 % at 160 ms. The
[evidence document](evidence.md#the-k-e-histogram-and-the-choice-of-n) gives the `(K, E)`
histogram.

The same rule below 63 carries no tier 5 claim. On the same build it gave a deployment `N` of 44
at 80 ms (99.10 %; serving capture 832 MiB call total, 268 MiB transaction total) and of 36 at
160 ms (99.07 %; 794 and 180 MiB).

#### Memory and the startup log

A fixed `--kv-cache-memory-bytes` must leave room for the transaction graphs and the segment
graphs. Core does not charge them under a fixed budget, and the startup log reports the uncharged
bytes in a WARNING. Under core's default KV-cache sizing, the extensible KV cache, core runs no
estimate: it sizes the KV cache after the captures, from the device memory they leave, and the
startup log reports the graph totals as measured in an INFO line, a budget that a startup plan
sets there included. A fixed or profiled budget that holds fewer state pages than the admission
limit plus one fails startup when the serving capture first reads the state pages, before it
captures any cell; the remedies are a larger `--kv-cache-memory-bytes`, a larger
`--num-gpu-blocks-override`, under a profiled budget a larger `--gpu-memory-utilization`, or a
lower admission limit, never a smaller budget. Under `--num-gpu-blocks-override`, and under a
profiled budget with the extensible KV cache off, core's graph-memory estimate charges the
profiling capture, and startup fails if the serving capture retains more than the profiling
capture.

Each capture logs one INFO line with the cells, `N`, the call and transaction totals, the transient
peak and the capture time. Post-warmup verification logs one INFO line when it passes. Shutdown
logs one INFO line with replays, split steps and fallbacks by reason. Startup captures the `2N`
cells twice when core's graph-memory estimate runs, in addition to the segment graphs' two
captures, and once otherwise. On the A40 with the state-pool build,
each capture took about 70 to 74 s. If startup exceeds `VLLM_ENGINE_READY_TIMEOUT_S` (default
600 s), raise it ([Operations](#engine-ready-timeout)).

#### Limits

- One geometry (lookahead and chunk duration) per server. Transaction graphs need 80 or 160 ms
  (lookahead 0 or 1). The published card's default, 320 ms, and 560 and 1120 ms are refused, with
  the remedy `num_lookahead_tokens` 0 or 1.
- `N` is bounded by `max_num_seqs`, by `max_num_batched_tokens` and by core's admission limit
  (`--max-num-active-seqs`). A fixed block count or budget, when set, must hold at least the
  admission limit plus one state pages.
- Startup outside the restricted startup is refused, naming each unmet setting and its remedy.
- Steps with more than `N` CHUNK rows (`chunk-uncovered`) or more than `N` other rows
  (`echo-uncovered`) run without transaction graphs and are counted.
- A step whose structural inputs differ from its cells' (`signature`) also runs without transaction
  graphs and is counted. A direct engine stream client that sets an endpoint policy causes such a
  step. On plugin-packed input, a nonzero `signature` count is a defect.
- A step that holds a row of more than one token (`multi-token`) runs without transaction graphs and
  is counted; that row's request fails. Only a direct engine client sends such a row.
- `transaction_graph_counters` reads the counters, including the `(K, E)` histogram. Under a
  multiprocess engine it runs on the worker through `AsyncLLM.collective_rpc`, which needs
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` in the engine's environment to send a function.
- `--enforce-eager` with `transaction_graph_populations` is refused, because core runs no capture
  hook under `--enforce-eager`.
- Native burst is refused on the target core, with or without transaction graphs.
- Segment arms other than the defaults (`eager-graphed` encoder, `dense-graphed` decoder) are
  refused, and so is `chunk_bucket_graph_populations`.
- On a core whose `ModelState` lacks the state-group layout accessor `state_group_layouts`, the
  option is refused.
- The session factory is not available on the target core, so no frontend serves streams with
  the option; once it is, the server needs a frontend release that opens streams by locale only.
- Tier 4 (mixed load) must be claimed separately with the option on. It has not been run with the
  option on, on any build.

## Clients

The examples use a 16 kHz mono 16-bit PCM WAV file, `sample.wav`. To convert another file:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le sample.wav
```

Clients name the model as the server serves it: `nvidia/nemotron-3.5-asr-streaming-0.6b`, or the
first name given to `--served-model-name`. The client examples below have not been run against the
target core.

### Core `/v1/realtime`

Core's realtime WebSocket is at `ws://<host>:8000/v1/realtime`. vLLM's example
client
[`openai_realtime_client.py`](https://github.com/vllm-project/vllm/blob/main/examples/speech_to_text/realtime/openai_realtime_client.py)
uses this event sequence:

1. The server sends `session.created`.
2. The client sends `session.update` with `"model"` set to the served model name.
3. The client sends `input_audio_buffer.commit` to start the stream.
4. The client sends `input_audio_buffer.append` events, each carrying base64 PCM16 audio at 16 kHz
   in `"audio"`.
5. The client sends `input_audio_buffer.commit` with `"final": true` to end the input.
6. The server sends `transcription.delta` events with text in `"delta"`, then one
   `transcription.done` with the whole transcript in `"text"`. A failure arrives as an `error`
   event.

On this route the model runs at the server's lookahead. It uses the locale `auto` when the card's
locale table has it, and otherwise the table's only locale. Core's behaviour on this route
matters in three ways:

- Core sends `session.created` when it accepts the connection, before it admits the stream. It
  admits the stream when the first `input_audio_buffer.commit` starts generation. A server that
  cannot admit it (reason `active_limit`, at core's admission limit, or `engine_paused`) answers
  that commit with an `error` event (`"type": "error"`) whose `code` is `processing_error` and
  whose message, in `error`, names the reason. A
  server at its admission limit therefore does not refuse a client at connection.
- An input that core refuses ends the stream with an error. When the model fails a stream's row
  (for example a unit at another lookahead, or a malformed unit from a client that bypasses the
  plugin's packer), core ends that request alone with finish reason `error`; the server's other
  streams continue.
- Core sends `transcription.done` whenever generation ends, whatever the finish reason. A stream
  stopped by `length`, `error` or `abort` is also reported done.

You can run vLLM's example against the server directly. It imports from `vllm`, so run it in the
vLLM environment:

```bash
python openai_realtime_client.py --model nvidia/nemotron-3.5-asr-streaming-0.6b --audio_path sample.wav
```

The following client is adapted from that example. It reads the WAV file with the standard library
and sends audio at real-time pace, as a live source does:

```python
import asyncio
import base64
import json
import wave

import websockets

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
CHUNK_BYTES = 3200  # 100 ms of 16 kHz 16-bit mono audio


async def main() -> None:
    with wave.open("sample.wav", "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())

    async with websockets.connect("ws://localhost:8000/v1/realtime") as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "session.created", created
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def send_audio() -> None:
            for start in range(0, len(pcm), CHUNK_BYTES):
                chunk = base64.b64encode(pcm[start : start + CHUNK_BYTES]).decode()
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": chunk})
                )
                await asyncio.sleep(0.1)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        sender = asyncio.create_task(send_audio())
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "transcription.delta":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "transcription.done":
                print(f"\nFinal: {event['text']}")
                break
            elif event["type"] == "error":
                print(f"\nError: {event['error']}")
                break
        await sender


asyncio.run(main())
```

Install its one dependency with `pip install websockets`.

### Core `/v1/audio/transcriptions`

Core's file route is `POST /v1/audio/transcriptions`. With `curl`:

```bash
curl -s localhost:8000/v1/audio/transcriptions \
    -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
    -F file=@sample.wav \
    -F language=en
```

With the OpenAI Python client (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

with open("sample.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        file=audio,
        language="en",
    )
print(result.text)
```

`language` is optional. Without it, the model uses the `auto` locale when the card's locale table
has it, otherwise the table's only locale; if neither exists, it refuses the request. With
`language`, the model maps the language to a locale of the table.

A language whose locales use several rows of the table, such as `zh`, is refused unless
`language_locales` in `--hf-overrides` names its locale, for example
`{"language_locales": {"zh": "zh-CN"}}`. Startup logs a WARNING for each such language. A request
may also name an exact locale in `vllm_xargs` as `{"locale": "en-GB"}`.

The file runs at the server's lookahead. The model refuses with HTTP 400 a task other than
transcription, a text prompt, hotwords, a target language and an unmappable language. It refuses
beam search (`use_beam_search`) and any client completion limit (`max_completion_tokens`) only on a
core that carries vllm#67151, which the target core does not
([Limits](#limits-and-the-support-boundary)). Core refuses `verbose_json` and timestamps for this
model.

### Stage 2: the frontend's routes

The [`vllm-riva-frontend`](https://github.com/pdtgct/vllm-riva-frontend) plugin serves Riva gRPC
`StreamingRecognize` and `Recognize`, the Riva Realtime WebSocket and a file route,
`POST /plugins/riva/v1/audio/transcriptions`, in the same process as core's routes
([frontend `docs/deployment.md`](https://github.com/pdtgct/vllm-riva-frontend/blob/v0.4.0b3/docs/deployment.md)).
Its streaming routes use this plugin's session factory, which is not available on the target core,
and no stage 2 build of the target core is recorded, so this manual gives no stage 2 client
example. The frontend's deployment guide describes its routes, its drain on shutdown and its
security settings.

## Operations

### Startup log lines

The plugin logs these lines at startup, among core's own lines:

| Level | Message begins | When |
| --- | --- | --- |
| INFO | `Starting Nemotron persistent-state profile execution with <n> rows` | the model's profile execution on core's memory-profiling step |
| INFO | `Nemotron segment graphs: <runner> runner, <capture>; encoder_execution_arm ..., decode_dispatch_arm ...; geometry ... (lookahead ...); max_num_seqs ...; retained ..., capture time ... s` | each segment-graph capture |
| INFO | `Nemotron graphed arms run eagerly on <device>, a platform or device without static graphs` | graphed arms requested on a device that cannot capture graphs |
| WARNING | `Serving lookahead <r> (<ms>), a geometry the checkpoint does not declare` | lookahead 1 on the published card |
| WARNING | `Nemotron segment graphs were not charged to the budget` | the hook configuration with the extensible KV cache off and either a fixed budget or core's estimate disabled; under the extensible KV cache an INFO line reports the same figures as measured, because core sizes the KV cache from the memory left after the capture |
| WARNING | `Complete-file requests for language <code> cannot be mapped to a prompt row` | each language that needs a `language_locales` entry |
| INFO | `Nemotron transaction graphs (...): the <capture> capture captured <n> cells ...` | each transaction-graph capture, with the call and transaction totals, the transient peak and the capture time |
| INFO | `Nemotron transaction graphs (...): post-warmup verification passed for <n> cells ...` | after warmup, with transaction graphs on |

In the hook configuration with core's graph-memory estimate running, each segment-graph capture
runs twice: a profiling capture, which the estimate charges, and a serving capture. Under core's
default KV-cache sizing and under a fixed budget it runs once, as the serving capture. At shutdown with transaction graphs on, the
plugin logs one INFO line with replays, split steps and fallbacks by reason.

### Readiness checks

| Check | Ready when |
| --- | --- |
| `curl -sf localhost:8000/health` | HTTP 200 |
| `curl -s localhost:8000/v1/models` | lists the served model |

The target core has no `/ready` route; RFC 2 adds it. `/health` stays unauthenticated under
`--api-key`, so use it for a liveness probe that cannot send a token.

Startup takes a minute or more, because core profiles memory and the model captures its graphs.
[`--max-num-seqs` and graph memory](#--max-num-seqs-and-graph-memory) gives measured startup times.

### Core's audio file caps

Core's upload and decode caps are operator settings, not the model's:

| Variable | Default |
| --- | --- |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | 25 |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | 600 s |
| `VLLM_MAX_AUDIO_DECODE_BYTES` | 256 MiB |

Raise them for long files, and raise `--max-model-len` with the duration. A frontend checks its own
file limits against these at startup; its deployment guide states the rules.

### Engine ready timeout

`VLLM_ENGINE_READY_TIMEOUT_S` (default 600 s) bounds how long the API server waits for the engine to
become ready. Graph capture counts against it. With transaction graphs on, startup captures `2N`
cells twice when core's graph-memory estimate runs, in addition to the segment graphs' two
captures, and once otherwise; on the state-pool build the tier 5
servers were ready 179 to 221 s after launch. If startup exceeds the timeout, raise it.

## Troubleshooting

The plugin collects configuration refusals and raises them together as one `ValueError`, so one
failed start names every problem. The message begins with one of these prefixes:

- `Nemotron3_5AsrStreamingForRNNT configuration refused: `: the served model configuration;
- `Nemotron3_5AsrStreamingForRNNT refuses this engine configuration: `: the engine settings;
- `the Nemotron streaming model state refuses this engine configuration: `: the engine settings,
  checked again when the model state is built.

| Message fragment | Meaning | Remedy |
| --- | --- | --- |
| `requires a vLLM build with RFC 1's state layer and vllm#67583's model-state hooks; this build lacks ...` (raised by `register()`) | the installed vLLM is not the target core: a released vLLM, the state-pool build, or another build without the state layer | install the build of [Builds and compatibility](#builds-and-compatibility) |
| `the installed vLLM lacks adapter dependencies of vllm-nemotron-asr-streaming: ...` | a vLLM interface the model adapter imports is missing | the same |
| `the installed vLLM's ModelState lacks the model-state hooks ...; the plugin requires vllm#67583's model-state hooks` | a build without #67583 | the same |
| `enforce_eager is off, so the hook configuration is required; unmet: ...` | no `--enforce-eager`, and the hook configuration is incomplete; the message lists each unmet condition | add `--enforce-eager`, or the missing `-cc.cudagraph_mode none` / `-cc.mode none` |
| `no graph core's capture hook captures is selected` | the hook configuration with only eager arms | add `--enforce-eager` |
| `encoder_execution_arm dense-graphed (compiled) is not captured through core's capture hook`, or the same for `chunk_bucket_graph_populations` | these need `--enforce-eager` | add `--enforce-eager` |
| `prefix caching must be False` | set explicitly on | remove the setting |
| `experimental_native_burst` ... not supported on the state layer | native burst set | remove `experimental_native_burst` |
| `The KV cache holds <n> requests of state that can't be recomputed, fewer than the <m> that can be active` (core's message) | the KV cache holds fewer state pages than requests that can be active | lower `--max-num-seqs` or `--max-num-active-seqs`, or raise `--num-gpu-blocks-override` or the budget |
| `speculative decoding must be off`, `KV transfer must be off`, `sleep mode must be off`, `CuMem must be off`, and similar | a [refused setting](#refused-settings) | remove the setting |
| `<name> must be 1` | a parallel size other than 1 | serve on one device |
| `supported_num_lookahead_tokens ... is refused` | a server serves one lookahead | remove it; use `num_lookahead_tokens` |
| `num_lookahead_tokens ... is not an implemented right context (one of 0, 1, 3, 6, 13)` | an unimplemented lookahead | use one of the listed values |
| `the server's lookahead is not set` | the checkpoint sets no default lookahead | set `num_lookahead_tokens` in `--hf-overrides` |
| `max_model_len <n> is below 144` | too small for one stream unit | raise `--max-model-len` ([sizing](#sizing---max-model-len)) |
| `the default sampling parameters set max_tokens` | a token limit from the generation configuration | pass `--generation-config vllm` and remove any token-limit `--override-generation-config` |
| `the generation configuration has eos_token_id ...` | an EOS other than the park token | pass `--generation-config <dir>` whose `generation_config.json` has no other EOS |
| `language_locales entry ...` | an entry that names no locale of the table, or a locale of another language | correct the entry |
| `Nemotron segment-graph capture failed (...)` | a segment-graph capture could not allocate; the message names the graph, the memory retained so far, the budget setting and remedies | a smaller `--max-num-seqs`; the eager arms with `--enforce-eager`; with a fixed budget, a smaller `--kv-cache-memory-bytes`; in the serving capture, a lower `--gpu-memory-utilization` |
| `the serving capture retains more graph memory than the profiling capture core charged to the budget` | the hook configuration's serving capture exceeded its charge | the remedies the message lists |
| `transaction_graph_populations is refused under --enforce-eager` | core runs no capture hook there | remove `--enforce-eager` (hook configuration) or remove the key |
| `transaction_graph_populations is refused at the server's lookahead <r>` | transaction graphs need lookahead 0 or 1 | set `num_lookahead_tokens` 0 or 1, or remove the key |
| `transaction_graph_populations needs the default segment arms` | a non-default arm with transaction graphs | remove the arm override |
| `transaction_graph_populations cannot be combined with experimental_native_burst` | both are set | remove `experimental_native_burst`, which the target core refuses in any case |
| `transaction_graph_populations ... is refused: ...` | a malformed population list (not an int, duplicate, out of range) | correct the list |

With transaction graphs on, these conditions are also refused. Each refusal names
`transaction_graph_populations`, the condition, the rejected value and the remedy:

| Condition | Meaning | Remedy |
| --- | --- | --- |
| state layer | the core's `ModelState` lacks the state-group layout accessor `state_group_layouts` | remove `transaction_graph_populations`, or install the target core |
| scheduling | `async_scheduling` is on or unset (unset means asynchronous on the target core) | `--no-async-scheduling` |
| runner modes | `cudagraph_mode` or the compilation mode is not an explicit NONE | `-cc.cudagraph_mode none -cc.mode none` |
| block count | `--num-gpu-blocks-override` minus 1 is below core's admission limit or a population; core would refuse it only after both captures | `--num-gpu-blocks-override` of at least the admission limit plus one, or a lower admission limit with the largest population at most it |
| admission limit | a population above core's admission limit (`--max-num-active-seqs`, or `--max-num-seqs` when it is unset) | a largest population of at most the limit, or a larger `--max-num-active-seqs` (at most `--max-num-seqs`) or none |

Under core's default KV-cache sizing, a state page count below the admission limit plus one is
refused by core's scheduler after the captures with "Lower max_num_seqs or enlarge the cache",
even when `--max-num-active-seqs` sets the limit. The startup log's measured graph-memory line,
logged before it, gives the graph totals. The remedies are a lower admission limit, a smaller
largest population, a larger `--gpu-memory-utilization`, or transaction graphs off.

If startup times out during graph capture, raise `VLLM_ENGINE_READY_TIMEOUT_S`.

A request that the model fails during serving does not stop the server. The model reports the
request through core's failed-request channel (`take_failed_requests`), and core ends that request
alone with finish reason `error`; every other request continues. Causes include a unit at another
lookahead, a malformed unit or file item, and a prompt or input of more than one token, all of
which only a client that bypasses the plugin's packer and processor can send.

## Limits and the support boundary

- A server serves one lookahead. Clients cannot select one; serve each lookahead from its own
  server.
- The server runs on one device. Every parallel size must be 1.
- Core's `/v1/realtime` serves streams directly, with no frontend. Core sends `session.created`
  before it admits the stream, so a server that cannot admit it (reason `active_limit`, at core's
  admission limit, or `engine_paused`) refuses a stream only after the client's first commit, as an
  `error` event with code `processing_error` whose message names the reason. An input that core
  refuses ends the stream, and core reports a stream done whatever its finish reason.
- The session factory is not available on the target core, so no frontend serves streams.
- A direct engine client's prompt or streaming input of more than one token fails its request;
  every unit the plugin packs is one token.
- Core's `/v1/audio/transcriptions` serves files directly, with no frontend:
  - Requests queued in core hold their decoded audio in the API server with no bound.
    `--max-num-seqs` bounds only the files actively scheduled. If clients are not trusted, put a cap
    in front of the server.
  - The model refuses `use_beam_search` and `max_completion_tokens` as options that do not apply to
    audio transcription. It reads them from the `SpeechToTextParams` fields that vllm#67151 adds.
  - On a core without vllm#67151, the request guards are inactive. A client's
    `max_completion_tokens` ends the file early with `length`, and beam search can issue up to
    `n × max_model_len` engine requests per upload (one per beam per pass). Do not expose the route
    to untrusted clients on such a core.
  - On a core without vllm#67144, a file ending in `error` or `abort` returns partial text with
    HTTP 200. On a core without vllm#67151, so does a file ending in a server-caused `length`.
  - The target core, and so stage 1 and stage 2, carries neither vllm#67144 nor vllm#67151, so every
    limit in the three items above applies to it.
- Whole-file requests slow concurrent streams on the same server. The last mixed-load measurement
  ran on an earlier build ([evidence, historical entries](evidence.md#historical-entries)). With one
  293 s file in flight, the p95 final latency of eight concurrent streams rose from 0.55 s to
  3.64 s. Tier 4 (mixed load) has not been run on the target core. Serve file traffic from a
  separate server, or keep files short.
- The plugin's routes set no endpoint policy, so every stream and file they serve runs with
  endpointing disabled and the model emits no end-of-utterance token. A direct engine client may set
  one in the request's sampling parameters: `extra_args={"endpoint_policy": {"mode": "greedy_blank",
  "threshold_frames": <n>, "residue_frames": <n>}}`; a malformed policy fails that request.
- Native burst (`experimental_native_burst`) is refused on the target core.
- The plugin was measured only on one NVIDIA A40 at FP32 with CUDA, on the earlier state-pool
  build. On a device without static graphs, the graphed arms run eagerly and log one INFO line.

## Testing the plugin

The [README's Install and test section](../README.md#install-and-test) gives the test markers, the
install and test commands, and the tier 1 acceptance run. That run tests frontend distributions
installed from the pinned commit.

## Evidence

The [evidence document](evidence.md) lists every acceptance claim with its tier, build and
revisions, and the measured figures. It marks earlier results historical. No run on the target core
is recorded yet. The raw evidence for release `v0.1.0b2` on the state-pool build is in
`pdtgct/nemotron-omni-port`:

- `experiments/2026-11-15-plugin-tier2-hook-build/`: tier 2;
- `experiments/2026-11-15-plugin-h1-txg-tier5-a/` and `experiments/2026-11-15-plugin-h1-txg-tier5-b/`:
  tier 5 for transaction graphs.

The design is in [`docs/high-level-design.md`](high-level-design.md) and [`docs/intent/`](intent/).

## Future roadmap

Each item names a limit this manual states, the trigger that lifts it and what changes here. The
[high-level design's Future roadmap](high-level-design.md#future-roadmap) owns the design of each.

- **A verified serve command.** `--max-num-seqs 64` is interim, and the serve command runs
  `--enforce-eager` and `--no-async-scheduling`. Trigger: tier 2 met on the target core, in the hook
  configuration and with asynchronous scheduling (`registration`, REG-ENG-007, REG-ENG-014).
  Change: the serve command and Serving configurations give the verified values and flags.
- **Frontend streaming on the target core.** The session factory is not available there. Trigger:
  the session factory runs on core's automatic admission. Change: Builds and compatibility records
  a stage 2 commit, and Clients gains the frontend's streaming examples.
- **Transaction graphs beyond the restricted startup.** Trigger: the qualification covers the
  reduced modes and asynchronous scheduling. Change: the restricted startup's rows leave the
  Transaction graphs section.
- **The transcription route's guards and terminal forms.** Trigger: a build the plugin targets
  carries vllm#67144 and vllm#67151. Change: Limits and the support boundary drops the
  partial-text and inactive-guard limits for that build.
