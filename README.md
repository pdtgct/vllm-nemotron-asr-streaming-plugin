# vllm-nemotron-asr-streaming

Out-of-tree vLLM plugin for cache-aware streaming Nemotron ASR. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card, unmodified, as live streams and as complete files from one `vllm serve` process. The model
keeps its encoder caches and RNN-T decoder state in a state slot that the engine owns, so a stream's
state persists between engine steps for as long as the client sends audio. A released vLLM does not
provide the interfaces it needs: the plugin runs on vLLM `main` with the open pull requests below,
and `register()` fails with a message saying so on any other build.

## Relation to complete-audio transcription in vLLM

Sohaib Ahmad's [vllm#59827](https://github.com/vllm-project/vllm/pull/59827), "[Model] Add Nemotron
3.5 ASR transcription support", merged on 2026-10-05, adds complete-audio transcription for this
checkpoint through `/v1/audio/transcriptions`; its description puts native cache-aware realtime
streaming out of scope. He also publishes the out-of-tree plugin
[`Sohaib-Ahmed21/vllm-nemotron-asr-plugin`](https://github.com/Sohaib-Ahmed21/vllm-nemotron-asr-plugin)
for complete-audio transcription on released vLLM.

This plugin adds streaming. It serves streaming sessions whose state the engine manages from chunk
to chunk, and it serves complete files on the same server through the same stateful path, so the
model imposes no limit on a file's length; core's `--max-model-len` still bounds it. The Hub card names the architecture
`Nemotron3_5AsrForRNNT`, which is the offline plugin's key. This plugin registers
`Nemotron3_5AsrStreamingForRNNT` and never writes the card's key, so both plugins can be installed
together. The [high-level design](docs/high-level-design.md) compares the two implementations.

## Contribution in progress

The interfaces this plugin needs are proposed to vLLM in three RFCs. All three RFCs and their pull
requests are open and under review; none is merged.

| RFC | Proposal | Pull requests |
| --- | --- | --- |
| RFC 1, [vllm#65543](https://github.com/vllm-project/vllm/issues/65543) | scheduler-owned persistent state | [vllm#67374](https://github.com/vllm-project/vllm/pull/67374), [#67375](https://github.com/vllm-project/vllm/pull/67375), [#67583](https://github.com/vllm-project/vllm/pull/67583); speech-to-text fixes [#67144](https://github.com/vllm-project/vllm/pull/67144), [#67151](https://github.com/vllm-project/vllm/pull/67151) |
| RFC 2, [vllm#65545](https://github.com/vllm-project/vllm/issues/65545) | endpoint plugin lifecycle | [vllm#65546](https://github.com/vllm-project/vllm/pull/65546), [#67294](https://github.com/vllm-project/vllm/pull/67294) |
| RFC 3, [vllm-omni#9848](https://github.com/vllm-project/vllm-omni/issues/9848) | vLLM-Omni on Model Runner V2 | [vllm#65549](https://github.com/vllm-project/vllm/pull/65549), [vllm-omni#9849](https://github.com/vllm-project/vllm-omni/pull/9849) (both drafts) |

The builds below are fork branches of `pdtgct/vllm` that merge these pull requests onto vLLM
`main`. RFC 3 is not part of either build.

## Builds

- **Stage 1** (the model alone, on core's `/v1/realtime` and `/v1/audio/transcriptions`) is upstream
  vLLM `main` with vllm#67374, #67375, #67583, #67144 and #67151. #67374, #67375 and #67583 are RFC 1
  (the persistent-state primitive; stream admission, `AsyncLLM.generate(..., stream_admission=True)`
  and `wait_stream_retired`; and the model-state hooks through which core captures the model's
  graphs).
- **Stage 2** (with a frontend) is stage 1 plus RFC 2 (#65546 and #67294, the endpoint plugin
  lifecycle) and a frontend endpoint plugin.

The [user manual](docs/manual.md#builds-and-compatibility) gives each build's branch and commit.

## What you can serve

On stage 1, core's own routes serve the model with no frontend: live streams on `/v1/realtime` and
complete files on `POST /v1/audio/transcriptions`.

On stage 2, the [`vllm-riva-frontend`](https://github.com/pdtgct/vllm-riva-frontend) plugin
`v0.4.0b3` runs in the same `vllm serve` process and adds Riva gRPC `StreamingRecognize` and
`Recognize`, the Riva Realtime WebSocket, and a file route,
`POST /plugins/riva/v1/audio/transcriptions`, so Riva clients can send streaming audio and whole
files. Core's routes stay available beside them.

The [user manual's Clients section](docs/manual.md#clients) gives an example for each route.

## Quick start

The [quick start](docs/quickstart.md) installs a stage 1 or stage 2 build, the plugin and, for
stage 2, the frontend, then serves the model and checks that the server answers.

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

- Under `--enforce-eager`, which this command uses, `--max-num-seqs 64` is interim: it comes from an
  unpublished A40 measurement at 320 ms. Under the hook configuration (replace `--enforce-eager`
  with `-cc.cudagraph_mode none -cc.mode none`), `--max-num-seqs 64` met tier 2's serving run on
  one A40 at 320 ms; the quick start uses that configuration.
- The server serves one geometry, fixed at startup: the checkpoint's default lookahead, 3
  (320 ms) on the published card. `--max-model-len 77000` holds a 600 s file at that geometry.
- The [user manual's Serving configurations](docs/manual.md#serving-configurations) gives the
  reason for each flag, how to select another geometry and size `--max-model-len`, the memory that
  `--max-num-seqs` costs, the eager arms and the settings the model refuses.
- Transaction graphs are opt-in; the
  [manual's Transaction graphs section](docs/manual.md#transaction-graphs-opt-in) gives their serve
  command and limits.

## Expected performance

The figures below come from one tier 2 serving run (partly met) on one NVIDIA A40 at FP32, on the
stage 1 build, at 320 ms, in the hook configuration with `--max-num-seqs 64` and the default segment
arms ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)). Other GPUs, geometries and stage 2
have not been measured.

- Accuracy and completion: 5,559 LibriSpeech `/v1/realtime` sessions at concurrency 32 all ended `stop`, with WER
  3.57 % on test-clean and 7.99 % on test-other; 640 test-clean sessions at concurrency 64 gave WER
  2.89 %. The server logged no preemption
  ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- Final latency, from the client's final commit to `transcription.done` with audio sent at
  real-time pace: p50 206 ms and p99 430 ms on test-clean at concurrency 32, and p50 234 ms and
  p99 499 ms at concurrency 64 ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- Startup and memory: the server was ready 130.6 s after launch; device memory in use peaked at
  16,445 MiB during startup, and 35.74 GiB remained for KV memory
  ([evidence](docs/evidence.md#tier-2-on-the-stage-1-build)).
- `--max-num-seqs` bounds concurrent streams and actively scheduled files. Segment-graph memory
  grows faster than linearly with it: under `--enforce-eager`, an unpublished A40 measurement put
  the capture peak at 18.3 GiB for 64 and 44.4 GiB of the card's 45 GiB for 128
  ([evidence](docs/evidence.md#unpublished-measurement-under---enforce-eager)). The startup log
  reports each capture's retained memory and capture time.
- Mixed load: whole files slow concurrent streams on the same server. On an earlier build, one
  293 s file in flight raised eight streams' p95 final latency from 0.55 s to 3.64 s; tier 4 has not
  been run on stage 1. Serve file traffic from a separate server, or keep files short
  ([evidence](docs/evidence.md#historical-entries)).
- Transaction graphs, at 80 or 160 ms, are opt-in and have not met tier 5
  ([evidence](docs/evidence.md#transaction-graphs-tier-5)).

## Documentation

- [Quick start](docs/quickstart.md): install, serve and check a server.
- [User manual](docs/manual.md): builds, every serving setting, clients, operations,
  troubleshooting and limits.
- [Evidence](docs/evidence.md): every acceptance claim with its tier, build and revisions, and the
  measured figures.
- [High-level design](docs/high-level-design.md), component designs and EARS specs in
  [`docs/intent/`](docs/intent/), and module names in
  [`docs/implementation-layout.md`](docs/implementation-layout.md).

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
on `PYTHONPATH`; `gpu` tests need an installed engine on CUDA; `card` tests need
`NEMOTRON_CARD_DIR` set to a local copy of the published card, which is never committed.

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
