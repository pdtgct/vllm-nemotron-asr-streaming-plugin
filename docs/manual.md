# User manual

This manual covers release `v0.1.0b2` of `vllm-nemotron-asr-streaming`. The
[quick start](quickstart.md) gives the shortest path to a running server. The
[README](../README.md) is the source of record for the serve commands, the measured figures and
the evidence; where this manual restates a README statement, it links the README section.

Measurement status, stated once for the whole manual: this release was measured only on the
stage 1 build, on one NVIDIA A40 at FP32. Tier 2 is partly met at 320 ms in the hook configuration
(stage 1 build `hook-build-20261114b` @ `9c40a6ffd6`; serving run at plugin `8166d85`, GPU tests at
plugin `adfb24b`). Tier 5 for transaction graphs is not met (same build; plugins `adfb24b`,
`a19621e` and `8166d85`, as the README's Evidence section lists). Stage 2 has not been measured
with this release, and frontend `v0.4.0b3` passed CI on CPU only. No command in this manual is
claimed to have been run as written unless the README's Evidence section records it.

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

## What the plugin is

`vllm-nemotron-asr-streaming` is an out-of-tree vLLM plugin. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card, unmodified, for cache-aware streaming sessions and for complete-file transcription. The model
keeps its encoder caches and decoder state in a persistent state slot, so the same model serves:

- live streams, on core's `/v1/realtime` and, on stage 2, on the frontend's Riva gRPC
  `StreamingRecognize` and Riva Realtime WebSocket;
- complete files, on core's `POST /v1/audio/transcriptions` and, on stage 2, on the frontend's file
  route and Riva gRPC `Recognize`.

The distribution has two entry points
([README, What is registered](../README.md#what-is-registered)):

| Entry-point group | Name | Target |
| --- | --- | --- |
| `vllm.general_plugins` | `nemotron_asr_streaming` | `vllm_nemotron_asr_streaming:register` |
| `vllm_riva_frontend.session_factories` | `nemotron_asr_streaming` | `vllm_nemotron_asr_streaming.session:create_session_factory` |

`register()` checks that the installed vLLM has `AsyncLLM.generate(stream_admission=...)` and
`AsyncLLM.wait_stream_retired`, then registers, without importing the model or touching a device:

- the architecture `Nemotron3_5AsrStreamingForRNNT` and its model-configuration hook;
- the config format `nemotron_asr_streaming`, which translates the Hub card;
- the tokenizer mode `nemotron_asr_streaming` and the renderer of the same name.

The card names the architecture `Nemotron3_5AsrForRNNT`, which belongs to the offline plugin. This
plugin never writes that key, so both plugins can be installed in one environment. The session
factory is for frontends: its `open` takes a locale only.

## Builds and compatibility

### The two builds

No released vLLM provides the interfaces the plugin needs. It targets two fork branches of upstream
vLLM `main` that carry pull requests still under upstream review
([README](../README.md#vllm-nemotron-asr-streaming)):

| Build | `pdtgct/vllm` branch | Commit | Carries |
| --- | --- | --- | --- |
| Stage 1 | `hook-build-20261114b` | `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee` | vllm#67374, #67375 and #67583 (RFC 1), #67144 and #67151 |
| Stage 2 | `stage2-build-20261114b` | `54b457dd8f58e1011c5ec2a9d1e3a16c6c78b498` | stage 1 plus RFC 2 (#65546 and #67294) |

RFC 1 provides the persistent-state primitive (#67374), stream admission (#67375) and the
model-state hooks through which core captures the model's graphs (#67583). #67144 and #67151 make
core's transcription route report incomplete files and add the request fields the model's request
guards read. RFC 2 provides the endpoint plugin lifecycle that the frontend needs. The stage 1
commit is an ancestor of the stage 2 commit.

Use stage 1 to serve the model alone, and stage 2 to serve it with the frontend. Both builds install
over vLLM's precompiled wheel `191e5d5df2ec693d5536a635079d58864dfacebb`, variant `cu132`; the
measured environment used torch `2.14.0+cu132`.

### The frontend pairing

The session protocol between the plugin and the frontend has no version negotiation, so the two are
released and qualified as a pair ([README, Install and test](../README.md#install-and-test)):

| Plugin | Frontend | Result |
| --- | --- | --- |
| `v0.1.0b2` | `v0.4.0b3` | the released pair |
| `v0.1.0b2` | `v0.4.0b1`, or another frontend that reads the factory's served cadences at startup | the frontend fails its own startup |
| `main` at `d5c5197` | `v0.4.0b3` | the server starts, and every stream ends with `internal` |

The plugin's release notes state that earlier frontends, which read served cadences at startup,
fail their own startup against this plugin. The frontend `v0.4.0b3` release notes state that plugin
`main` at `d5c5197` is incompatible with it.

### Mismatched vLLM builds

| vLLM | Result |
| --- | --- |
| A released vLLM, or a build without #67375 on #67374 | `register()` raises `RuntimeError`: `vllm-nemotron-asr-streaming requires a vLLM build with RFC 1 stream admission (vllm#67375 on #67374); this build lacks ...` |
| A build without #67583 | the configuration hook refuses the configuration, naming each missing model-state hook and stating that the plugin requires vllm#67583's model-state hooks |
| A build without #67144 or #67151 | the server starts, but core's transcription route returns partial text with HTTP 200 for some incomplete files, and the request guards are inactive ([Limits](#limits-and-the-support-boundary)) |
| Stage 1 with the frontend | the frontend needs RFC 2, which only the stage 2 build carries |

## Installation

Use one virtual environment per build, with Python 3.11 or 3.12. The plugin declares Python
`>=3.11`; the frontend's CI runs 3.11 and 3.12.

1. Install the vLLM build. `setup.py` reads `VLLM_PRECOMPILED_WHEEL_COMMIT`, which must be the full
   40-character commit, and `VLLM_PRECOMPILED_WHEEL_VARIANT`.

   ```bash
   VLLM_BRANCH=hook-build-20261114b      # stage 1
   # VLLM_BRANCH=stage2-build-20261114b  # stage 2

   git clone --branch "$VLLM_BRANCH" --single-branch https://github.com/pdtgct/vllm.git
   cd vllm
   git rev-parse HEAD
   VLLM_USE_PRECOMPILED=1 \
   VLLM_PRECOMPILED_WHEEL_COMMIT=191e5d5df2ec693d5536a635079d58864dfacebb \
   VLLM_PRECOMPILED_WHEEL_VARIANT=cu132 \
       pip install -e .
   cd ..
   ```

   `git rev-parse HEAD` prints the commit in the table above. If the branch has moved, check out
   that commit.

2. Install the plugin, from the tag or from the wheel attached to the release:

   ```bash
   pip install "git+https://github.com/pdtgct/vllm-nemotron-asr-streaming-plugin.git@v0.1.0b2"
   # or
   pip install https://github.com/pdtgct/vllm-nemotron-asr-streaming-plugin/releases/download/v0.1.0b2/vllm_nemotron_asr_streaming-0.1.0b2-py3-none-any.whl
   ```

3. For stage 2, install the frontend:

   ```bash
   pip install "git+https://github.com/pdtgct/vllm-riva-frontend.git@v0.4.0b3"
   ```

   Do not install `vllm-riva-frontend-dev` into a serving environment. It is the frontend's
   conformance suite, which only this plugin's tests use.

4. Check the installed versions:

   ```bash
   pip show vllm-nemotron-asr-streaming vllm-riva-frontend | grep -E '^(Name|Version)'
   ```

   The output lists `vllm-nemotron-asr-streaming` `0.1.0b2` and, for stage 2,
   `vllm-riva-frontend` `0.4.0b3`.

If `VLLM_PLUGINS` is set, it is an allowlist: include `nemotron_asr_streaming`, and for stage 2 also
`riva_frontend`.

## Serving configurations

The [README's Serve section](../README.md#serve) gives the reasons for each flag. This section
groups the same settings by the decision each one makes. This manual leads with the hook
configuration, the runner configuration of the tier 2 serving run:

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

The README's serve command is the same except that it uses `--enforce-eager` in place of the two
`-cc` flags ([Runner configuration](#runner-configuration)).

`--config-format nemotron_asr_streaming` and `--tokenizer-mode nemotron_asr_streaming` are
required: they select the plugin's translation of the Hub card and its tokenizer mode. Core resolves
synchronous scheduling and no prefix caching for this model, so the command omits
`--no-async-scheduling` and `--no-enable-prefix-caching`; passing them is harmless.

### Runner configuration

The outer model runner must capture no graphs and compile nothing. Two runner configurations meet
that condition:

| | Hook configuration | `--enforce-eager` |
| --- | --- | --- |
| Flags | no `--enforce-eager`; `-cc.cudagraph_mode none -cc.mode none`; a CUDA platform | `--enforce-eager` |
| Who captures the segment graphs | core, through #67583's capture hook | the model, on core's first memory-profiling step |
| Graph memory charged by | core's graph-memory estimate | the profiled budget |
| Transaction graphs | accepted (opt-in) | refused |
| Eager arms | refused (no graph to capture) | accepted |
| Measured | tier 2 partly met at 320 ms on the stage 1 build | `--max-num-seqs 64` is interim, from an unpublished A40 measurement |

The hook configuration requires at least one graph the hook captures: the `eager-graphed` encoder,
the `dense-graphed` decoder or transaction graphs. The default segment arms meet that requirement.
A command that gives neither configuration is refused at startup, naming each unmet condition and
its remedy ([Troubleshooting](#troubleshooting)).

To run under `--enforce-eager`, replace `-cc.cudagraph_mode none -cc.mode none` in the command
above with `--enforce-eager`. Use it for the eager arms, the compiled `dense-graphed` encoder or
`chunk_bucket_graph_populations`, or on a platform other than CUDA.

### Lookahead selection

A server serves one geometry, fixed at startup. Clients cannot choose it per request: every stream
and every whole file runs at the server's geometry.

- The default is the lookahead that the checkpoint's model configuration sets. The published card
  sets 3 (320 ms); another checkpoint can set another default.
- `num_lookahead_tokens` in `--hf-overrides` selects another geometry. The implemented values are:

  | `num_lookahead_tokens` | Geometry | Declared by the published card |
  | --- | --- | --- |
  | 0 | 80 ms | yes |
  | 1 | 160 ms | no; the server accepts it and logs a WARNING |
  | 3 | 320 ms | yes (the card's default) |
  | 6 | 560 ms | yes |
  | 13 | 1120 ms | yes |

  For example: `--hf-overrides '{"num_lookahead_tokens": 13}'`.
- `supported_num_lookahead_tokens` is refused. Remove it and select the geometry with
  `num_lookahead_tokens`.

A larger lookahead gives the encoder more right context per chunk, so different geometries can give
different transcripts for the same audio. Tier 2 has been measured only at 320 ms.

### Sizing `--max-model-len`

The model imposes no limit on a file's length. Each engine step advances a file by one chunk, and
core stops a request when it reaches `--max-model-len` tokens. Size `--max-model-len` for the
longest file you serve at the server's lookahead `r`
([README, Serve](../README.md#serve)):

- a file of `N` samples at 16 kHz emits at most `1 + (floor(N / s) + 1) × Q + 1` tokens, with
  `s = 1280 × (r + 1)` samples per chunk and `Q = 10 × (r + 1) + 1`;
- a 600 s file, core's default decode limit, needs at most:

  | Lookahead | 0 | 1 | 3 | 6 | 13 |
  | --- | --- | --- | --- | --- | --- |
  | Tokens for 600 s | 82,513 | 78,773 | 76,918 | 76,114 | 75,578 |

`--max-model-len 77000` therefore holds a 600 s file at lookahead 3, 6 and 13. An override to 0
needs at least 82,513, and an override to 1 at least 78,773. Streams need only 143 tokens. If you
raise `VLLM_MAX_AUDIO_DECODE_DURATION_S`, raise `--max-model-len` with it. Each token costs about
4 bytes per slot of token history (`max_num_seqs × max_model_len` tokens) and no KV memory. A
`--max-model-len` below 143 is refused.

### `--max-num-seqs` and graph memory

`--max-num-seqs` sets the number of state slots, which bounds concurrent streams and actively
scheduled files. It also sets the size of the segment graphs. The default `eager-graphed` encoder is
captured at every CHUNK count from 1 to `max_num_seqs`, so segment-graph memory grows faster than
linearly with `--max-num-seqs`.

Two figures matter, and they differ:

- **Retained memory**: what the captured graphs hold for the life of the server.
- **Capture peak**: device memory in use during capture, weights included. Capture fails when this
  peak does not fit, even when the retained memory would.

Figures at 320 ms on one A40 ([README, Serve](../README.md#serve)):

| Runner configuration | `--max-num-seqs` | Retained | Capture peak (device memory in use) | Startup | Status |
| --- | --- | --- | --- | --- | --- |
| Hook configuration | 64 | 1,544 MiB (profiling capture), 716 MiB (serving capture) | 16,445 MiB (`nvidia-smi`, on the memory-profiling step) | ready 130.6 s after launch | tier 2 serving run, stage 1 build `9c40a6ffd6`, plugin `8166d85` |
| `--enforce-eager` | 32 | about 1.1 GiB | 7.0 GiB | about 41 s | unpublished measurement |
| `--enforce-eager` | 64 | about 2.3 GiB | 18.3 GiB | about 78 s | unpublished measurement |
| `--enforce-eager` | 128 | about 17.4 GiB | 44.4 GiB of 45 GiB | about 182 s | unpublished measurement |

In the hook configuration run, 35.74 GiB was available for KV memory, and the KV budget admitted 65
requests. The largest sampled step held 38 rows; a 64-row step was not observed. Under
`--enforce-eager` at 64 rows, the 160 ms and 80 ms geometries retained 1.8 and 1.6 GiB (unpublished
measurement). Tier 2 figures under `--enforce-eager`, and at geometries other than 320 ms, are owed.

If a segment-graph capture cannot allocate, startup fails with the capture's figures and these
remedies: a smaller `--max-num-seqs`, or the eager arms with `--enforce-eager`.

### The eager arms

The default segment arms are the `eager-graphed` encoder and the `dense-graphed` RNN-T decoder. Their
graphs are captured for the server's one geometry only. The eager arms capture nothing; they are the
reference against which the graphed arms are tested:

```bash
--hf-overrides '{"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"}'
```

The eager arms need `--enforce-eager`. With the eager arms, startup took about 25 s in the
unpublished A40 measurement at 320 ms. The compiled `dense-graphed` encoder and
`chunk_bucket_graph_populations` also need `--enforce-eager`. To combine the eager arms with another
geometry, put both keys in one `--hf-overrides` object.

### `--kv-cache-memory-bytes`

By default core profiles memory and sizes the KV cache and the state pool from the remaining budget.
The model prepares its execution memory on core's memory-profiling step, so that memory is
budgeted, and in the hook configuration core's graph-memory estimate charges the profiling capture.

A fixed `--kv-cache-memory-bytes` changes this: core takes the budget as given and charges neither
the model's execution memory nor the segment graphs (nor transaction graphs). Leave room for both.
In the hook configuration the startup log then carries a WARNING with the uncharged bytes. The same
WARNING appears when core's graph-memory estimate is disabled
(`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`).

### Refused settings

The persistent-state model needs synchronous scheduling, no prefix caching and a single device.
Startup fails, naming each refused setting, for:

- asynchronous scheduling or prefix caching set explicitly on;
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the CuMem allocator,
  hierarchical sparse attention, fault tolerance or elastic expert parallelism;
- any of `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`,
  `decode_context_parallel_size` or `prefill_context_parallel_size` other than 1;
- an outer runner that is in neither runner configuration;
- `supported_num_lookahead_tokens`, or a `num_lookahead_tokens` that is not 0, 1, 3, 6 or 13;
- a generation configuration with an EOS other than the park token, or a token limit
  (`max_tokens`) from the checkpoint's generation configuration or `--override-generation-config`;
  pass `--generation-config vllm`.

### Transaction graphs (opt-in)

Transaction graphs replay a streaming step's device body and its scatters from CUDA graphs. They are
off by default, and this release ships them opt-in without tier 5: on the stage 1 build the step
rule and `U_api` fail at 80 ms and at 160 ms, and the body check fails at population 1. The
[README section Transaction graphs (opt-in)](../README.md#transaction-graphs-opt-in) gives the serve
command, the measured figures and every limit. In short:

- Set `transaction_graph_populations` in `--hf-overrides`. Its largest entry, `N`, fixes the cell
  inventory: `2N` cells. The tier 5 run verified `N = 63` with `--max-num-seqs 64` to fit the A40.
- They need the hook configuration, lookahead 0 or 1 (80 or 160 ms), the default segment arms and
  native burst off; each other combination is refused.
- Steps with more than `N` CHUNK rows or `N` other rows run without transaction graphs and are
  counted.
- Startup captures `2N` cells twice; on the A40 each capture took about 70 to 74 s. If startup
  exceeds `VLLM_ENGINE_READY_TIMEOUT_S`, raise it ([Operations](#engine-ready-timeout)).

## Clients

The examples use a 16 kHz mono 16-bit PCM WAV file, `sample.wav`. To convert another file:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le sample.wav
```

Clients name the model as the server serves it: `nvidia/nemotron-3.5-asr-streaming-0.6b`, or the
first name given to `--served-model-name`. The client examples have not been run against this
release.

### Core `/v1/realtime`

Core's realtime WebSocket is at `ws://<host>:8000/v1/realtime`, on both stages. The event sequence,
as vLLM's example client
[`openai_realtime_client.py`](https://github.com/vllm-project/vllm/blob/main/examples/speech_to_text/realtime/openai_realtime_client.py)
uses it:

1. The server sends `session.created`.
2. The client sends `session.update` with `"model"` set to the served model name.
3. The client sends `input_audio_buffer.commit` to start the stream.
4. The client sends `input_audio_buffer.append` events, each carrying base64 PCM16 audio at 16 kHz
   in `"audio"`.
5. The client sends `input_audio_buffer.commit` with `"final": true` to end the input.
6. The server sends `transcription.delta` events with text in `"delta"`, then one
   `transcription.done` with the whole transcript in `"text"`. A failure arrives as an `error`
   event.

On this route the model runs at the server's geometry with the locale `auto` when the card's locale
table has it, and otherwise the table's only locale. Two of core's behaviors on this route matter:

- Core drives the stream without stream admission, so an input unit that core rejects ends the
  stream.
- Core sends `transcription.done` whenever generation ends, whatever the finish reason, so a
  stream stopped by `length`, `error` or `abort` is also reported done.

You can run vLLM's example against the server directly; it needs the vLLM environment because it
imports from `vllm`:

```bash
python openai_realtime_client.py --model nvidia/nemotron-3.5-asr-streaming-0.6b --audio_path sample.wav
```

The following client is adapted from that example. It reads the WAV file with the standard library
and sends audio at real-time pace, as a live source would:

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

Core's file route is `POST /v1/audio/transcriptions`, on both stages. With `curl`:

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

`language` is optional. Without it the model uses the `auto` locale when the card's locale table
has it, otherwise the table's only locale, and otherwise refuses the request. With it, the model
maps the language to a locale of the table. A language whose locales use several rows of the table, such as `zh`, is refused unless
`language_locales` in `--hf-overrides` names its locale, for example
`{"language_locales": {"zh": "zh-CN"}}`; startup logs a WARNING for each such language. A request
may also name an exact locale in `vllm_xargs` as `{"locale": "en-GB"}`; this manual gives no client
example for that field.

The file runs at the server's geometry. The model refuses, with HTTP 400, a task other than
transcription, a text prompt, hotwords, a target language, an unmappable language, beam search
(`use_beam_search`) and any client completion limit (`max_completion_tokens`). Core refuses
`verbose_json` and timestamps for this model.

### Stage 2: the frontend's routes

On stage 2 one process serves core's routes unchanged and the frontend's routes
([frontend `docs/deployment.md`](https://github.com/pdtgct/vllm-riva-frontend/blob/v0.4.0b3/docs/deployment.md)):

| Surface | Address |
| --- | --- |
| Riva Realtime WebSocket | `ws://<host>:8000/plugins/riva/v1/realtime?intent=transcription` |
| Its bootstrap | `POST /plugins/riva/v1/realtime/transcription_sessions` |
| Complete files | `POST /plugins/riva/v1/audio/transcriptions` |
| Operational routes | `GET /plugins/riva/v1/health/ready`, `/health/live`, `/version`, `/metadata` |
| Riva gRPC | the `grpc_bind` listener (`0.0.0.0:50051` by default): `nvidia.riva.asr.RivaSpeechRecognition` (`StreamingRecognize`, `Recognize`, `GetRivaSpeechRecognitionConfig`), with gRPC health and reflection |

- The frontend's file route takes the same request as core's route, behind the frontend's ingress
  cap, file gate and request guard. To use it from the OpenAI client, set
  `base_url="http://localhost:8000/plugins/riva/v1"`.
- The Riva Realtime WebSocket speaks the Riva Realtime protocol, not the event set of core's
  `/v1/realtime`. The stock Riva Realtime client reaches it with `--server <host>:8000/plugins/riva`.
- The streaming and `Recognize` surfaces accept mono PCM16 at 8 or 16 kHz and mono G.711 mu-law and
  A-law at 8 kHz. The frontend refuses other formats, translation, diarization, word boosting, word
  offsets, endpointing and multi-utterance session reuse, naming the field.

### Stage 2: Riva gRPC through `nvidia-riva-client`

Install the public client with `pip install nvidia-riva-client`. One unary `Recognize` call for a
whole file:

```python
import riva.client

auth = riva.client.Auth(uri="localhost:50051")  # plaintext; secure the listener at the network layer
asr = riva.client.ASRService(auth)

config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    audio_channel_count=1,
    language_code="en-US",
    max_alternatives=1,
)

with open("sample.wav", "rb") as audio:
    response = asr.offline_recognize(audio.read(), config)

for result in response.results:
    print(result.alternatives[0].transcript)
```

`language_code` must be a locale of the card's locale table; an empty `language_code` selects the
model's default locale when it has one. For streaming gRPC, run
[`transcribe_file.py`](https://github.com/nvidia-riva/python-clients/blob/main/scripts/asr/transcribe_file.py)
from `nvidia-riva/python-clients`:

```bash
python scripts/asr/transcribe_file.py --server localhost:50051 --input-file sample.wav
```

## Operations

### Startup log lines

The plugin logs these lines at startup; core's own lines surround them:

| Level | Message begins | When |
| --- | --- | --- |
| INFO | `Starting Nemotron persistent-state profile execution with <n> rows` | the model's profile execution on core's memory-profiling step |
| INFO | `Nemotron segment graphs: <runner> runner, <capture>; encoder_execution_arm ..., decode_dispatch_arm ...; geometry ... (lookahead ...); max_num_seqs ...; retained ..., capture time ... s` | each segment-graph capture |
| INFO | `Nemotron graphed arms run eagerly on <device>, a platform or device without static graphs` | graphed arms requested on a device that cannot capture graphs |
| WARNING | `Serving lookahead <r> (<ms>), a geometry the checkpoint does not declare` | lookahead 1 on the published card |
| WARNING | `Nemotron segment graphs were not charged to the budget` | the hook configuration with a fixed budget or core's estimate disabled |
| WARNING | `Complete-file requests for language <code> cannot be mapped to a prompt row` | each language that needs a `language_locales` entry |
| INFO | `Nemotron transaction graphs (...): the <capture> capture captured <n> cells ...` | each transaction-graph capture, with the call and transaction totals, the transient peak and the capture time |
| INFO | `Nemotron transaction graphs (...): post-warmup verification passed for <n> cells ...` | after warmup, with transaction graphs on |

In the hook configuration each segment-graph capture runs twice: a profiling capture, which core's
graph-memory estimate charges, and a serving capture. At shutdown with transaction graphs on, the
plugin logs one INFO line with replays, split steps and fallbacks by reason.

### Readiness checks

| Stage | Check | Ready when |
| --- | --- | --- |
| 1 | `curl -sf localhost:8000/health` | HTTP 200 |
| 1 | `curl -s localhost:8000/v1/models` | lists the served model |
| 2 | `curl -sf localhost:8000/ready` | HTTP 200: the engine is healthy and every endpoint plugin reports ready; 503 while the server drains |
| 2 | `curl -sf localhost:8000/plugins/riva/v1/health/ready` | HTTP 200 |
| 2 | `grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check` | `"status": "SERVING"` |

Stage 1 core has no `/ready` route; the stage 2 build adds it with RFC 2. `/health` stays
unauthenticated under `--api-key`, so a liveness probe that cannot send a token uses it. On stage 2,
orchestrators probe `/ready` for readiness. `GET /plugins/riva/v1/version` returns
`{"release": "0.4.0b3", "api": "v1"}`.

Startup takes a minute or more: core profiles memory and the model captures its graphs. The figures
in [`--max-num-seqs` and graph memory](#--max-num-seqs-and-graph-memory) give measured startup times.

### Shutdown (stage 2)

The frontend drains on shutdown: open streams are finalized, files settle and the gRPC listener
stops before the engine shuts down
([frontend `docs/deployment.md`](https://github.com/pdtgct/vllm-riva-frontend/blob/v0.4.0b3/docs/deployment.md)).

- `--shutdown-timeout` is the frontend's whole drain budget. It must exceed the engine's
  `draining_timeout` (40 s by default); 60 s is the documented value. At core's default of 0, no
  drain runs, and the frontend logs a warning at startup.
- Send `SIGTERM` or `SIGINT` to the `vllm serve` API-server process only, for example
  `kill -INT <api-server-pid>`. Ctrl-C signals the whole process group, which also reaches the
  engine process: the engine stops without waiting for the drain, and open streams end with an
  error and no final result.
- Under Kubernetes or `docker stop`, run `vllm serve` as the container's main process (exec form),
  and set the grace period above `--shutdown-timeout`. The defaults (30 s for Kubernetes'
  `terminationGracePeriodSeconds`, 10 s for `docker stop --time`) are below 60 s.
- Stage 2 needs one API server (`--api-server-count` at most 1) and data parallel size 1, because
  every API-server process would bind the gRPC port.

### Core's audio file caps

Core's upload and decode caps are operator settings, not the model's
([README, Serve](../README.md#serve)):

| Variable | Default |
| --- | --- |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | 25 |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | 600 s |
| `VLLM_MAX_AUDIO_DECODE_BYTES` | 256 MiB |

Raise them for long files, and raise `--max-model-len` with the duration. On stage 2 the frontend
checks its own file limits against these at startup; the frontend's deployment guide states the
rules.

### Engine ready timeout

`VLLM_ENGINE_READY_TIMEOUT_S` (default 600 s) bounds how long the API server waits for the engine to
become ready. Graph capture counts against it: with transaction graphs on, startup captures `2N`
cells twice beside the segment graphs' two captures, and the tier 5 servers were ready 179 to
221 s after launch. If startup exceeds the timeout, raise it.

## Troubleshooting

Configuration refusals are collected and raised together as one `ValueError`, so one failed start
names every problem. The message begins with one of these prefixes:

- `Nemotron3_5AsrStreamingForRNNT configuration refused: `: the served model configuration;
- `Nemotron3_5AsrStreamingForRNNT refuses this engine configuration: `: the engine settings;
- `the Nemotron streaming model state refuses this engine configuration: `: the engine settings,
  checked again when the model state is built.

| Message fragment | Meaning | Remedy |
| --- | --- | --- |
| `requires a vLLM build with RFC 1 stream admission (vllm#67375 on #67374); this build lacks ...` | the installed vLLM is not a stage 1 or stage 2 build | install a build from [Builds and compatibility](#builds-and-compatibility) |
| `the installed vLLM lacks adapter dependencies of vllm-nemotron-asr-streaming: ...` | a vLLM interface the model adapter imports is missing | the same |
| `the installed vLLM's ModelState lacks the model-state hooks ...; the plugin requires vllm#67583's model-state hooks` | a build without #67583 | the same |
| `enforce_eager is off, so the hook configuration is required; unmet: ...` | no `--enforce-eager`, and the hook configuration is incomplete; the message lists each unmet condition | add `--enforce-eager`, or the missing `-cc.cudagraph_mode none` / `-cc.mode none` |
| `no graph core's capture hook captures is selected` | the hook configuration with only eager arms | add `--enforce-eager` |
| `encoder_execution_arm dense-graphed (compiled) is not captured through core's capture hook`, or the same for `chunk_bucket_graph_populations` | these need `--enforce-eager` | add `--enforce-eager` |
| `async scheduling must be False`, `prefix caching must be False` | set explicitly on | remove the setting |
| `speculative decoding must be off`, `KV transfer must be off`, `sleep mode must be off`, `CuMem must be off`, and similar | a [refused setting](#refused-settings) | remove the setting |
| `<name> must be 1` | a parallel size other than 1 | serve on one device |
| `supported_num_lookahead_tokens ... is refused` | a server serves one geometry | remove it; use `num_lookahead_tokens` |
| `num_lookahead_tokens ... is not an implemented right context (one of 0, 1, 3, 6, 13)` | an unimplemented lookahead | use one of the listed values |
| `the server's lookahead is not set` | the checkpoint sets no default lookahead | set `num_lookahead_tokens` in `--hf-overrides` |
| `max_model_len <n> is below 143` | too small for one stream unit | raise `--max-model-len` ([sizing](#sizing---max-model-len)) |
| `the default sampling parameters set max_tokens` | a token limit from the generation configuration | pass `--generation-config vllm` and remove any token-limit `--override-generation-config` |
| `the generation configuration has eos_token_id ...` | an EOS other than the park token | pass `--generation-config <dir>` whose `generation_config.json` has no other EOS |
| `language_locales entry ...` | an entry that names no locale of the table, or a locale of another language | correct the entry |
| `experimental_native_burst requires the server's lookahead to be 1 (160ms)` | native burst at another geometry | select lookahead 1 or turn native burst off |
| `Nemotron segment-graph capture failed (...)` | a segment-graph capture could not allocate; the message names the graph, the memory retained so far, the budget setting and remedies | a smaller `--max-num-seqs`; the eager arms with `--enforce-eager`; with a fixed budget, a smaller `--kv-cache-memory-bytes`; in the serving capture, a lower `--gpu-memory-utilization` |
| `the serving capture retains more graph memory than the profiling capture core charged to the budget` | the hook configuration's serving capture exceeded its charge | the remedies the message lists |
| `transaction_graph_populations is refused under --enforce-eager` | core runs no capture hook there | remove `--enforce-eager` (hook configuration) or remove the key |
| `transaction_graph_populations is refused at the server's lookahead <r>` | transaction graphs need lookahead 0 or 1 | set `num_lookahead_tokens` 0 or 1, or remove the key |
| `transaction_graph_populations needs the default segment arms` | a non-default arm with transaction graphs | remove the arm override |
| `transaction_graph_populations cannot be combined with experimental_native_burst` | both are set | remove one |
| `transaction_graph_populations ... is refused: ...` | a malformed population list (not an int, duplicate, out of range) | correct the list |

If startup ends with a timeout while graphs are still being captured, raise
`VLLM_ENGINE_READY_TIMEOUT_S`. If every stage 2 stream ends with `internal` and the server started,
check the plugin and frontend versions against [the frontend pairing](#the-frontend-pairing).

## Limits and the support boundary

The [README's Serve section](../README.md#serve) lists the support boundary:

- **One geometry per server.** Clients cannot select a lookahead. Serve each geometry from its own
  server.
- **One device.** Every parallel size must be 1.
- **Core's `/v1/realtime`** is served directly, with no frontend. An input unit that core rejects
  ends the stream, and core reports a stream done whatever its finish reason.
- **Core's `/v1/audio/transcriptions`** is served directly, with no frontend:
  - requests queued in core hold their decoded audio in the API server with no bound;
    `--max-num-seqs` bounds only the files actively scheduled. Put a cap in front of the server if
    clients are not trusted. On stage 2, untrusted clients use the frontend's file route or
    `Recognize`, which are behind the frontend's ingress cap and file gate.
  - on a core without vllm#67151 the request guards are inactive: a client's `max_completion_tokens`
    ends the file early with `length`, and beam search can issue up to `n × max_model_len` engine
    requests per upload, so do not expose the route to untrusted clients on such a core.
  - on a core without vllm#67144, a file ending in `error` or `abort` returns partial text with
    HTTP 200; on a core without vllm#67151, so does a file ending in a server-caused `length`. Both
    stage builds carry these pull requests.
- **Mixed load.** Whole-file requests slow concurrent streams on the same server. The last mixed-load
  measurement was made on an earlier build (README, Evidence, historical); tier 4 has not been run
  on the stage 1 build. Serve file traffic from a separate server, or keep files short.
- **Endpointing.** The plugin binds no endpoint policy, so every stream and file runs with
  endpointing disabled and the model emits no end-of-utterance token.
- **Stage 2 security.** The gRPC listener has no transport security and is outside `--api-key`;
  secure it at the network layer. Under `--api-key`, the frontend checks the token on every request
  under `/plugins/riva/`.
- **Platforms.** Measured only on one NVIDIA A40 at FP32 with CUDA. On a device without static
  graphs, the graphed arms run eagerly and log one INFO line.
- **Not supported.** The serve mode that runs core's Rust frontend loads no endpoint plugin, so
  stage 2 does not run under it.

## Testing the plugin

The tests are tiered: unmarked tests need only this package; `engine` tests need the stage 1 vLLM
build on `PYTHONPATH`; `gpu` tests need an installed engine on CUDA; `card` tests need
`NEMOTRON_CARD_DIR` set to a local copy of the published card. Tier 1 acceptance runs the frontend
conformance tests and the pin tests together with `NEMOTRON_CONFORMANCE_ACCEPTANCE=1`, against
frontend distributions installed from the pinned commit.

The [README's Install and test section](../README.md#install-and-test) gives the install commands,
the test commands and the acceptance rule.

## Evidence

The [README's Evidence section](../README.md#evidence) lists every acceptance claim with its tier,
build and revisions, and marks earlier results as historical. The raw evidence for this release is
in `pdtgct/nemotron-omni-port`:

- `experiments/2026-11-15-plugin-tier2-hook-build/`: tier 2 on the stage 1 build;
- `experiments/2026-11-15-plugin-h1-txg-tier5-a/` and `experiments/2026-11-15-plugin-h1-txg-tier5-b/`:
  tier 5 for transaction graphs.

The design is in [`docs/high-level-design.md`](high-level-design.md) and [`docs/intent/`](intent/).
