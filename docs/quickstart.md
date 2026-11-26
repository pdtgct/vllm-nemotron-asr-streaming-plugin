# Quick start

This page installs `vllm-nemotron-asr-streaming` on the stage 1 build, the target core, serves
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
from one `vllm serve` process, and checks that the server answers. Stage 1 runs the plugin alone:
core's `/v1/realtime` and `POST /v1/audio/transcriptions` serve requests directly.

Stage 2, which adds the Riva frontend, does not serve streams on the target core: the frontend's
streaming routes use this plugin's session factory, which is not available on the target core
([manual, Builds and compatibility](manual.md#builds-and-compatibility)). This page does not
install it.

The [user manual](manual.md) covers every setting, the clients, operations and troubleshooting.

## What has been measured

No acceptance tier has been met on the target core. The last release, `v0.1.0b2`, ran on the
earlier state-pool build, where it partly met tier 2, the installed-engine acceptance tier, on one
NVIDIA A40 at FP32 at 320 ms in the hook configuration
([evidence](evidence.md#tier-2-on-the-state-pool-build)). That release does not run on the target
core, and the plugin's `main` requires the target core and no other RFC 1 build
([high-level design, Constraints](high-level-design.md#constraints)).

The install and serve commands below have not been run as written. The
[evidence document](evidence.md) records the runs that were.

## Versions

| Component | Repository | Revision |
| --- | --- | --- |
| This plugin | `pdtgct/vllm-nemotron-asr-streaming-plugin` | `main` (unreleased for the target core) |
| vLLM, stage 1 (the target core) | `pdtgct/vllm`, branch `pstate-impl-20261122` | `cbf13fed4aff1a72cf0dfbb5cd4cb95d43773e52` |

No released vLLM provides the interfaces the plugin needs. The target core is upstream vLLM `main`
at `842f74fa8dff2c6792bd68b021c92d8fbf2b93bc` with RFC 1's state layer, vllm#67583's model-state
hooks and the state-group layout accessor on top. It changes no compiled code, so it installs over
the precompiled wheel of that upstream commit.

## Requirements

- One NVIDIA GPU with a driver that supports CUDA 13.2. The target core pins torch `2.14.0`; the
  commands below select vLLM's precompiled wheel variant `cu132`, which has not been checked
  against this commit.
- Python 3.11 or 3.12.
- Network access to GitHub and Hugging Face.

## Install

Use a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

git clone --branch pstate-impl-20261122 --single-branch https://github.com/pdtgct/vllm.git
cd vllm
git checkout cbf13fed4aff1a72cf0dfbb5cd4cb95d43773e52
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=842f74fa8dff2c6792bd68b021c92d8fbf2b93bc \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu132 \
    pip install -e .
cd ..

pip install "git+https://github.com/pdtgct/vllm-nemotron-asr-streaming-plugin.git@main"
```

`VLLM_PRECOMPILED_WHEEL_COMMIT` must be the full 40-character commit. Do not install
`vllm-riva-frontend-dev` into a serving environment; it is the frontend's test suite.

## Serve

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

This is the README's serve command. It runs `--enforce-eager`, so the model captures its own segment
graphs on core's memory-profiling step, inside the profiled budget. `--max-num-seqs 64` is an
interim value from an unpublished A40 measurement at 320 ms under `--enforce-eager`.
`--no-async-scheduling` keeps the step order every earlier run used; on the target core an unset
option means asynchronous scheduling, which the plugin accepts but no run has qualified. The command
serves the checkpoint's default lookahead: 3 (320 ms) on the published card.

To run the hook configuration instead, replace `--enforce-eager` with
`-cc.cudagraph_mode none -cc.mode none`. Core then captures the model's segment graphs through
vllm#67583's capture hook ([manual, Runner configuration](manual.md#runner-configuration)).

If `VLLM_PLUGINS` is set, it must include `nemotron_asr_streaming`.

Check the server. The target core has no `/ready` route; use `/health` and `/v1/models`:

```bash
curl -sf localhost:8000/health && echo healthy
curl -s localhost:8000/v1/models
```

`/v1/models` lists `nvidia/nemotron-3.5-asr-streaming-0.6b`. Then transcribe a 16 kHz mono PCM16
WAV file:

```bash
curl -s localhost:8000/v1/audio/transcriptions \
    -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
    -F file=@sample.wav
```

The response is JSON with the transcript in `text`.

## Next

- [Serving configurations](manual.md#serving-configurations): lookahead, `--max-model-len`,
  `--max-num-seqs`, graph memory and transaction graphs.
- [Clients](manual.md#clients): realtime and file transcription examples.
- [Troubleshooting](manual.md#troubleshooting): what each startup refusal means.

## Future roadmap

- **A measured quick start.** No command on this page has been run as written on the target
  core. Trigger: tier 2 met on the target core. Shape: the evidence document records the install
  and serve commands, and this page gives the `--max-num-seqs` that tier 2 verified.
- **Stage 2.** This page installs no frontend. Trigger: the session factory runs on core's
  automatic admission ([high-level design, Future roadmap](high-level-design.md#future-roadmap)).
  Shape: an optional step installs the RFC 2 build and the frontend and checks a streaming route.
