# Quick start

This page installs release `v0.1.0b2` of `vllm-nemotron-asr-streaming`, serves
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
from one `vllm serve` process, and checks that the server answers. It covers two builds:

- **Stage 1**: the plugin alone. Core's `/v1/realtime` and `POST /v1/audio/transcriptions` serve
  requests directly.
- **Stage 2**: the plugin with the Riva frontend `v0.4.0b3`. The same process also serves Riva
  gRPC, the Riva Realtime WebSocket and the frontend's file route.

The [user manual](manual.md) explains every setting, the clients, operations and troubleshooting.

## What has been measured

This release was measured only on the stage 1 build, on one NVIDIA A40 at FP32. Tier 2 is partly
met at 320 ms, in the hook configuration: the serving run used plugin `8166d85` and the GPU tests
used plugin `adfb24b`, whose production code equals this release's
([evidence](evidence.md#tier-2-on-the-stage-1-build)). Stage 2 has not been measured with this
release, and the frontend `v0.4.0b3` passed CI on CPU only. The install and serve commands below
have not been run as written; the [evidence document](evidence.md) records the runs that were.

## Versions

| Component | Repository | Revision |
| --- | --- | --- |
| This plugin | `pdtgct/vllm-nemotron-asr-streaming-plugin` | tag `v0.1.0b2` (`d93dbb8399`) |
| Riva frontend (stage 2 only) | `pdtgct/vllm-riva-frontend` | tag `v0.4.0b3` (`125ac19884`) |
| vLLM, stage 1 | `pdtgct/vllm`, branch `hook-build-20261114b` | `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee` |
| vLLM, stage 2 | `pdtgct/vllm`, branch `stage2-build-20261114b` | `54b457dd8f58e1011c5ec2a9d1e3a16c6c78b498` |

No released vLLM provides the interfaces the plugin needs. The plugin and the frontend are released
as a pair: plugin `v0.1.0b2` needs frontend `v0.4.0b3`. Plugin `main` at `d5c5197` and frontend
`v0.4.0b2` are incompatible with them ([manual, Builds and compatibility](manual.md#builds-and-compatibility)).

## Requirements

- One NVIDIA GPU with a driver that supports CUDA 13.2. The measured environment used vLLM's
  precompiled wheel `191e5d5df2ec693d5536a635079d58864dfacebb`, variant `cu132`, with torch
  `2.14.0+cu132`.
- Python 3.11 or 3.12.
- Network access to GitHub and Hugging Face.

## Install

Use one virtual environment per build. Set `VLLM_BRANCH` to the stage 1 or the stage 2 branch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

VLLM_BRANCH=hook-build-20261114b      # stage 1
# VLLM_BRANCH=stage2-build-20261114b  # stage 2

git clone --branch "$VLLM_BRANCH" --single-branch https://github.com/pdtgct/vllm.git
cd vllm
git rev-parse HEAD   # 9c40a6ffd6... for stage 1, 54b457dd8f... for stage 2
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=191e5d5df2ec693d5536a635079d58864dfacebb \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu132 \
    pip install -e .
cd ..

pip install "git+https://github.com/pdtgct/vllm-nemotron-asr-streaming-plugin.git@v0.1.0b2"

# Stage 2 only:
pip install "git+https://github.com/pdtgct/vllm-riva-frontend.git@v0.4.0b3"
```

`VLLM_PRECOMPILED_WHEEL_COMMIT` must be the full 40-character commit. Do not install
`vllm-riva-frontend-dev` into a serving environment; it is the frontend's test suite.

## Serve, stage 1

```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    -cc.cudagraph_mode none \
    -cc.mode none \
    --max-num-seqs 64 \
    --max-model-len 77000
```

This command runs the hook configuration: no `--enforce-eager`, and `-cc.cudagraph_mode none`
with `-cc.mode none`, on CUDA. Core captures the model's segment graphs through vllm#67583's capture
hook and charges them in its graph-memory estimate. It is the configuration of the tier 2 serving
run, in which `--max-num-seqs 64` fit one A40 at 320 ms. The command serves the checkpoint's default
lookahead, which is 3 (320 ms) on the published card. The README's serve command uses the other
runner configuration, `--enforce-eager`, in which `--max-num-seqs 64` is an interim value from an
unpublished A40 measurement ([manual, Runner configuration](manual.md#runner-configuration)). If
`VLLM_PLUGINS` is set, it must include `nemotron_asr_streaming`.

Check the server. Stage 1 core has no `/ready` route; use `/health` and `/v1/models`:

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

## Serve, stage 2

```bash
export VLLM_PLUGINS=riva_frontend,nemotron_asr_streaming

vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    -cc.cudagraph_mode none \
    -cc.mode none \
    --max-num-seqs 64 \
    --max-model-len 77000 \
    --shutdown-timeout 60 \
    --additional-config '{"riva_frontend": {"grpc_bind": "0.0.0.0:50051"}}'
```

`VLLM_PLUGINS` must name both plugins: core loads no endpoint plugin when it is unset. The model
flags are stage 1's. `--shutdown-timeout 60` is the frontend's drain budget, and `grpc_bind` sets
the Riva gRPC listener, which has no transport security of its own.

Check the server:

```bash
curl -sf localhost:8000/ready && echo ready
curl -sf localhost:8000/plugins/riva/v1/health/ready && echo riva-ready
curl -s localhost:8000/plugins/riva/v1/version
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

`/plugins/riva/v1/version` returns `{"release": "0.4.0b3", "api": "v1"}`, and the gRPC health
check returns `"status": "SERVING"`. Stop the server by sending `SIGINT` or `SIGTERM` to the
API-server process only (`kill -INT <api-server-pid>`), not with Ctrl-C
([manual, Shutdown](manual.md#shutdown-stage-2)).

## Next

- [Serving configurations](manual.md#serving-configurations): lookahead, `--max-model-len`,
  `--max-num-seqs` and graph memory.
- [Clients](manual.md#clients): realtime, file transcription and Riva gRPC examples.
- [Troubleshooting](manual.md#troubleshooting): what each startup refusal means.
