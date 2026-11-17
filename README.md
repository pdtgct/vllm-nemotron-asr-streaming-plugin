# vllm-nemotron-asr-streaming

An out-of-tree vLLM plugin for cache-aware streaming Nemotron ASR. One `vllm serve` process serves
the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card, unmodified, to live streams and to complete files.

The model keeps its encoder caches and RNN-T decoder state in a state slot that the engine owns. A
stream's state persists between engine steps for as long as the client sends audio. No released
vLLM provides the interfaces the plugin needs. The plugin runs on vLLM `main` with the open pull
requests listed below. On any other build, `register()` fails with a message that says so.

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
| RFC 1, [vllm#65543](https://github.com/vllm-project/vllm/issues/65543) | scheduler-owned persistent state | [vllm#67374](https://github.com/vllm-project/vllm/pull/67374), [#67375](https://github.com/vllm-project/vllm/pull/67375), [#67583](https://github.com/vllm-project/vllm/pull/67583); speech-to-text fixes [#67144](https://github.com/vllm-project/vllm/pull/67144), [#67151](https://github.com/vllm-project/vllm/pull/67151) |
| RFC 2, [vllm#65545](https://github.com/vllm-project/vllm/issues/65545) | endpoint plugin lifecycle | [vllm#65546](https://github.com/vllm-project/vllm/pull/65546), [#67294](https://github.com/vllm-project/vllm/pull/67294) |
| RFC 3, [vllm-omni#9848](https://github.com/vllm-project/vllm-omni/issues/9848) | vLLM-Omni on Model Runner V2 | [vllm#65549](https://github.com/vllm-project/vllm/pull/65549), [vllm-omni#9849](https://github.com/vllm-project/vllm-omni/pull/9849) (both drafts) |

The two builds below are fork branches of `pdtgct/vllm` that merge these pull requests onto vLLM
`main`. Neither build includes RFC 3.

## Builds

Stage 1 serves the model alone, on core's `/v1/realtime` and `/v1/audio/transcriptions`. It is
upstream vLLM `main` with vllm#67374, #67375, #67583, #67144 and #67151. Of these, #67374, #67375
and #67583 are RFC 1: the persistent-state primitive, stream admission
(`AsyncLLM.generate(..., stream_admission=True)` and `wait_stream_retired`), and the model-state
hooks through which core captures the model's graphs.

Stage 2 adds a frontend. It is stage 1 plus RFC 2 (#65546 and #67294, the endpoint plugin
lifecycle) and a frontend endpoint plugin.

The [user manual](docs/manual.md#builds-and-compatibility) gives each build's branch and commit.

## What you can serve

On stage 1, core's own routes serve the model with no frontend: live streams on `/v1/realtime` and
complete files on `POST /v1/audio/transcriptions`.

On stage 2, the [`vllm-riva-frontend`](https://github.com/pdtgct/vllm-riva-frontend) plugin
`v0.4.0b3` runs in the same `vllm serve` process. It adds Riva gRPC `StreamingRecognize` and
`Recognize`, the Riva Realtime WebSocket, and a file route,
`POST /plugins/riva/v1/audio/transcriptions`, so Riva clients can send both streaming audio and
whole files. Core's routes stay available.

The [user manual's Clients section](docs/manual.md#clients) gives an example for each route.

## Quick start

The [quick start](docs/quickstart.md) installs a stage 1 or stage 2 build, the plugin and, for
stage 2, the frontend. It then serves the model and checks that the server answers.

## Serve

This command serves the checkpoint's default lookahead with the default segment arms (the encoder
and decoder execution variants):

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

- With `--enforce-eager`, as here, `--max-num-seqs 64` is an interim value from an unpublished A40
  measurement at 320 ms. The quick start uses the hook configuration instead: replace
  `--enforce-eager` with `-cc.cudagraph_mode none -cc.mode none`. In that configuration
  `--max-num-seqs 64` fit one A40 at 320 ms in the tier 2 serving run.
- The server fixes one lookahead at startup. The default is the checkpoint's: 3 (320 ms) on the
  published card. At that lookahead, `--max-model-len 77000` holds a 600 s file.
- The [user manual's Serving configurations](docs/manual.md#serving-configurations) explains each
  flag, other lookaheads, `--max-model-len` sizing, the memory cost of `--max-num-seqs`, the eager
  arms and the refused settings.
- Transaction graphs are opt-in. The
  [manual's Transaction graphs section](docs/manual.md#transaction-graphs-opt-in) gives their serve
  command and limits.

## Expected performance

All figures come from one serving run on one NVIDIA A40 at FP32, on the stage 1 build, at 320 ms,
in the hook configuration with `--max-num-seqs 64` and the default segment arms. That run partly met
tier 2 ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)). Other GPUs, other lookaheads and
stage 2 are unmeasured.

- Accuracy: 5,559 LibriSpeech `/v1/realtime` sessions at concurrency 32 all ended `stop`. WER was
  3.57 % on test-clean and 7.99 % on test-other. 640 test-clean sessions at concurrency 64 gave WER
  2.89 %. The server logged no preemption
  ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- Final latency, from the client's final commit to `transcription.done`, with audio sent at
  real-time pace: p50 206 ms and p99 430 ms on test-clean at concurrency 32; p50 234 ms and p99
  499 ms at concurrency 64 ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- Startup and memory: the server was ready 130.6 s after launch. Device memory in use peaked at
  16,445 MiB during startup, and 35.74 GiB remained for KV memory
  ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- `--max-num-seqs` bounds concurrent streams and actively scheduled files. Segment-graph memory
  grows faster than linearly with it. Under `--enforce-eager`, an unpublished A40 measurement put
  the capture peak at 18.3 GiB for 64 and 44.4 GiB of the card's 45 GiB for 128
  ([evidence](docs/evidence.md#unpublished-measurement-under---enforce-eager)). The startup log
  reports each capture's retained memory and capture time.
- Mixed load: whole files slow concurrent streams on the same server. On an earlier build, one
  293 s file in flight raised the p95 final latency of eight streams from 0.55 s to 3.64 s. Tier 4
  (mixed load) has not been run on stage 1. Serve file traffic from a separate server, or keep files
  short ([evidence](docs/evidence.md#historical-entries)).
- Transaction graphs, at 80 or 160 ms, are opt-in and have not met tier 5
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
The v0.4.0b3 release notes state that plugin `main` at `d5c5197` is incompatible with it: that
revision's streams end `internal`.

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
