# vllm-nemotron-asr-streaming

An out-of-tree vLLM plugin for cache-aware streaming Nemotron ASR. One `vllm serve` process serves
the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card, unmodified, to live streams and to complete files.

The model keeps its encoder caches and RNN-T decoder state in one page of a state group that the
engine allocates per request. A stream's state persists between engine steps for as long as the
client sends audio. No released vLLM provides the interfaces the plugin needs. The plugin runs on
the RFC 1 state-layer build described below. On a build without the state layer or vllm#67583's
model-state hooks, `register()` fails with a message that names what is missing.

## Relation to complete-audio transcription in vLLM

Sohaib Ahmad's [vllm#59827](https://github.com/vllm-project/vllm/pull/59827), "[Model] Add Nemotron
3.5 ASR transcription support", merged on 2026-10-05. It adds complete-audio transcription for this
checkpoint on `/v1/audio/transcriptions`, and its description puts native cache-aware realtime
streaming out of scope. He also publishes the out-of-tree plugin
[`Sohaib-Ahmed21/vllm-nemotron-asr-plugin`](https://github.com/Sohaib-Ahmed21/vllm-nemotron-asr-plugin)
for complete-audio transcription on released vLLM.

This plugin adds streaming. The engine manages each streaming session's state from chunk to chunk.
Complete files take the same stateful path on the same server, so the model sets no limit on a
file's length; core's `--max-model-len` still bounds it.

The Hub card names the architecture `Nemotron3_5AsrForRNNT`, which is the offline plugin's key. This
plugin registers `Nemotron3_5AsrStreamingForRNNT` and never writes the card's key, so you can install
both plugins together. The [high-level design](docs/high-level-design.md) compares the two.

## Contribution in progress

Three RFCs propose the interfaces this plugin needs to vLLM. All three RFCs and their pull requests
are open and under review; none is merged.

| RFC | Proposal | Pull requests |
| --- | --- | --- |
| RFC 1, [vllm#65543](https://github.com/vllm-project/vllm/issues/65543) | persistent state as a state layer | the state-layer implementation, `pdtgct/vllm` branch `pstate-impl-20261122`; [#67583](https://github.com/vllm-project/vllm/pull/67583) (model-state hooks); the drafts the state layer replaces, [vllm#67374](https://github.com/vllm-project/vllm/pull/67374) and [#67375](https://github.com/vllm-project/vllm/pull/67375), which this plugin does not support. Speech-to-text fixes [#67144](https://github.com/vllm-project/vllm/pull/67144) and [#67151](https://github.com/vllm-project/vllm/pull/67151), in neither build |
| RFC 2, [vllm#65545](https://github.com/vllm-project/vllm/issues/65545) | endpoint plugin lifecycle | [vllm#65546](https://github.com/vllm-project/vllm/pull/65546), [#67294](https://github.com/vllm-project/vllm/pull/67294) |
| RFC 3, [vllm-omni#9848](https://github.com/vllm-project/vllm-omni/issues/9848) | vLLM-Omni on Model Runner V2 | [vllm#65549](https://github.com/vllm-project/vllm/pull/65549), [vllm-omni#9849](https://github.com/vllm-project/vllm-omni/pull/9849) (both drafts) |

The builds below are defined on `pdtgct/vllm`. Neither includes RFC 3.

## Builds

Stage 1 serves the model alone, on core's `/v1/realtime` and `/v1/audio/transcriptions`. It is
`pdtgct/vllm` at `cbf13fed4a` (branch `pstate-impl-20261122`) alone, called the target core: RFC 1's state layer with vllm#67583's
model-state hooks, through which core captures the model's graphs, and the state-group layout
accessor `ModelState.state_group_layouts`, through which transaction graphs address state pages.
It carries neither vllm#67144 nor #67151, so core's transcription route has the limits of a core
without them: the request guards are inactive, and some incomplete files return partial text with
HTTP 200 ([manual, Limits](docs/manual.md#limits-and-the-support-boundary)).

Stage 2 adds a frontend. It is stage 1 plus RFC 2 (#65546 and #67294, the endpoint plugin
lifecycle) and a frontend endpoint plugin. The session factory, through which a frontend serves
streams, is not available on either build: on the target core `create_session_factory` refuses,
so a frontend's routes that use it do not run. Porting the factory to the target core is planned
([high-level design, Future roadmap](docs/high-level-design.md#future-roadmap)).

The [user manual](docs/manual.md#builds-and-compatibility) gives the stage 1 commit and how to
install it.

## What you can serve

On stage 1, core's own routes serve the model with no frontend: live streams on `/v1/realtime` and
complete files on `POST /v1/audio/transcriptions`.

The [`vllm-riva-frontend`](https://github.com/pdtgct/vllm-riva-frontend) plugin serves Riva gRPC
`StreamingRecognize` and `Recognize`, the Riva Realtime WebSocket, and a file route,
`POST /plugins/riva/v1/audio/transcriptions`, in the same `vllm serve` process, so Riva clients can
send both streaming audio and whole files. Its streaming routes use this plugin's session factory,
which is not available on the target core, so they do not run on stage 2 until the factory is
ported. No stage 2 build of the target core is recorded.

The [user manual's Clients section](docs/manual.md#clients) gives an example for each route.

## Quick start

The [quick start](docs/quickstart.md) installs the stage 1 build and the plugin, serves the model
and checks that the server answers.

## Serve

This command serves the checkpoint's default lookahead with the default segment arms (the encoder
and decoder execution variants):

<!-- @spec REG-ENG-007 -->
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

- `--max-num-seqs 64` is an interim value from an unpublished A40 measurement at 320 ms under
  `--enforce-eager`. No tier has been met on the target core, so no value is verified for it.
- The hook configuration replaces `--enforce-eager` with `-cc.cudagraph_mode none -cc.mode none`;
  core then captures the model's graphs through vllm#67583's capture hook. On the earlier
  state-pool build, `--max-num-seqs 64` fit one A40 at 320 ms in that configuration.
- Asynchronous scheduling is accepted with transaction graphs off, but no run has qualified it, so
  the command sets `--no-async-scheduling`. On the target core an unset option means asynchronous
  scheduling.
- The server fixes one lookahead at startup. The default is the checkpoint's: 3 (320 ms) on the
  published card. At that lookahead, `--max-model-len 77000` holds a 600 s file.
- The [user manual's Serving configurations](docs/manual.md#serving-configurations) explains each
  flag, other lookaheads, `--max-model-len` sizing, the memory cost of `--max-num-seqs`, the eager
  arms and the refused settings.
- Transaction graphs are opt-in and start only under a restricted startup. The
  [manual's Transaction graphs section](docs/manual.md#transaction-graphs-opt-in) gives their serve
  command and limits.

## Expected performance

No acceptance tier has been measured on the target core. All figures below come from one serving run of the
last release, `v0.1.0b2`, on the earlier state-pool build (`hook-build-20261114b`, which carried
the state pool and stream admission of vllm#67374 and #67375), on one NVIDIA A40 at FP32, at
320 ms, in the hook configuration with `--max-num-seqs 64` and the default segment arms
([evidence](docs/evidence.md#tier-2-on-the-state-pool-build)). They are a historical record for this
plugin until tier 2 is met on the target core. Other GPUs, other lookaheads and stage 2 are
unmeasured.

- Accuracy: 5,559 LibriSpeech `/v1/realtime` sessions at concurrency 32 all ended `stop`. WER was
  3.57 % on test-clean and 7.99 % on test-other. 640 test-clean sessions at concurrency 64 gave WER
  2.89 %. The server logged no preemption
  ([evidence](docs/evidence.md#tier-2-on-the-state-pool-build)).
- Final latency, from the client's final commit to `transcription.done`, with audio sent at
  real-time pace: p50 206 ms and p99 430 ms on test-clean at concurrency 32; p50 234 ms and p99
  499 ms at concurrency 64 ([evidence](docs/evidence.md#tier-2-on-the-state-pool-build)).
- Startup and memory: the server was ready 130.6 s after launch. Device memory in use peaked at
  16,445 MiB during startup, and 35.74 GiB remained for KV memory
  ([evidence](docs/evidence.md#tier-2-on-the-state-pool-build)).
- `--max-num-seqs` bounds concurrent streams and actively scheduled files. Segment-graph memory
  grows faster than linearly with it. Under `--enforce-eager`, an unpublished A40 measurement put
  the capture peak at 18.3 GiB for 64 and 44.4 GiB of the card's 45 GiB for 128
  ([evidence](docs/evidence.md#unpublished-measurement-under---enforce-eager)). The startup log
  reports each capture's retained memory and capture time.
- Mixed load: whole files slow concurrent streams on the same server. On an earlier build, one
  293 s file in flight raised the p95 final latency of eight streams from 0.55 s to 3.64 s. Tier 4
  (mixed load) has not been run on the target core. Serve file traffic from a separate server, or keep files
  short ([evidence](docs/evidence.md#historical-entries)).
- Transaction graphs, at 80 or 160 ms, are opt-in. They did not meet tier 5 on the state-pool
  build and have not been run on the target core
  ([evidence](docs/evidence.md#transaction-graphs-tier-5)).

## Documentation

- [Quick start](docs/quickstart.md): install, serve and check a server.
- [User manual](docs/manual.md): builds, serving settings, clients, operations, troubleshooting
  and limits.
- [Evidence](docs/evidence.md): the acceptance tiers, every claim with its build and revisions,
  and the measured figures.
- [High-level design](docs/high-level-design.md), component designs and EARS specs in
  [`docs/intent/`](docs/intent/), and module names in
  [`docs/implementation-layout.md`](docs/implementation-layout.md).

## Install and test

Install the package with its PEP 735 `test` dependency group. The group pins `vllm-riva-frontend`
and its conformance suite, `vllm-riva-frontend-dev`, at one commit of the frontend repository. The
`test` extra installs only pytest.

The pinned commit, `125ac198846d27ca17a1aaa9f1e17c009b40c676`, is the frontend release tag
v0.4.0b3, which opens streams by locale only. The plugin–frontend protocol has no version
negotiation, so the two are released and qualified as a pair. An earlier frontend that reads the
factory's served cadences at startup, such as v0.4.0b1, fails its own startup against this plugin.
The conformance suite runs the session factory against this repository's engine double, which
models the stream-admission client path; the factory is not available on the target core, so a
pass makes no claim for it.

```bash
# pip 25.1 or later
pip install -e . --group test
# or uv
uv sync --group test
```

Either command installs the two frontend distributions from the pinned commit as non-editable VCS
installs, as tier 1 requires.

Run the suite with `pytest`.

Unmarked tests need only this package. Tests marked `engine` need the stage 1 vLLM build on
`PYTHONPATH`, `gpu` tests need an installed engine on CUDA, and `card` tests need
`NEMOTRON_CARD_DIR` set to a local copy of the published card, which is never committed.

The tier 1 acceptance run executes the frontend conformance tests and the pin tests together, with
`NEMOTRON_CONFORMANCE_ACCEPTANCE=1`:

```bash
NEMOTRON_CONFORMANCE_ACCEPTANCE=1 PYTHONPATH=<stage 1 vLLM tree> \
    python -m pytest -q -p no:cacheprovider \
    tests/session/test_frontend_conformance.py tests/session/test_conformance_pin.py
```

The run fails when either frontend distribution is an editable or local install, or came from a
commit other than the pinned one. Tier 1 is met only by a run in which no test fails or is skipped.

## Future roadmap

The [high-level design's Future roadmap](docs/high-level-design.md#future-roadmap) gives each
deferred item with its trigger and shape. The items that change this README:

- **Measured figures on the target core.** Expected performance cites only the state-pool build.
  Trigger: tiers 2, 4 and 5 met on the target core. Shape: Expected performance cites those
  records, and the serve command gives the `--max-num-seqs` and runner flags tier 2 verified.
- **Frontend streaming.** A frontend's streaming routes do not run on either build. Trigger: the
  session factory runs on core's automatic admission. Shape: Builds and What you can serve name a
  stage 2 commit that serves them.
- **Transaction graphs beyond the restricted startup.** Trigger: core sizes its profiling and
  extensible KV caches for a model-state capture over state pages. Shape: the option starts with
  the default serve command's settings.
