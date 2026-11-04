# vllm-nemotron-asr-streaming

Staging out-of-tree plugin for cache-aware streaming Nemotron ASR on vLLM core. The work is aimed at a later fold into the offline plugin at [Sohaib-Ahmed21/vllm-nemotron-asr-plugin](https://github.com/Sohaib-Ahmed21/vllm-nemotron-asr-plugin). This repository does not commit to that repo.

**Gate A is not met.** `register()` has not been loaded inside a vLLM process built from the RFC1 tip. The tests stub those imports. Installing this skeleton is not evidence that the plugin serves audio.

## What is registered

Distribution `vllm-nemotron-asr-streaming`, import package `vllm_nemotron_asr_streaming`.

Entry point group `vllm.general_plugins`:

```toml
nemotron_asr_streaming = "vllm_nemotron_asr_streaming:register"
```

`register()` imports these RFC1 tip symbols and then registers one architecture:

- `vllm.model_executor.layers.state.PersistentStateLayerBase`
- `vllm.v1.engine.async_llm.AsyncLLM`
- `vllm.v1.engine.core_client.EngineCoreClient.admit_stream_async`
- `vllm.v1.kv_cache_interface.StateCompletion`
- `vllm.v1.worker.gpu.model_states.resolve_model_state_cls` (the hook that calls `get_model_state_cls`)

The registered architecture is `Nemotron3_5AsrStreamingForRNNT`.

The Hub config of [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) has `architectures: ["Nemotron3_5AsrForRNNT"]` and `model_type: "nemotron3_5_asr"`. That architecture string is the offline plugin's `ModelRegistry` key. `register_model` overwrites an existing key, so this package never passes `Nemotron3_5AsrForRNNT` to it, and it does not register the offline tokenizer or renderer key `nemotron3_5_asr`. A server started on the Hub repo id still resolves the offline class when that plugin is installed. This package does not rewrite the card.

The model class and the session methods are stubs. See [TODO.md](TODO.md).

## API target

Import target, not a release:

| | |
| --- | --- |
| Repo | `pdtgct/vllm` |
| RFC1 tip | branch `persistent-state-primitive-v2` @ `8b8327e2eece1df39eb6318eff4c9dc207b3ab61` |
| Model source | branch `rfc1-5212-vllm` @ `43c0bed5bf877dc74de2e1494a3a542f497493de`, tree `vllm/models/nemotron_asr/` (19,941 Python lines at that commit) |
| Integration branch | `rfc12-integration` was not on the remote when this scaffold was written, so it was not read |

`pyproject.toml` does not depend on PyPI `vllm`. A released vLLM does not provide these symbols.

Do not merge `rfc1-5212-vllm` onto the tip. Against `8b8327e2ee`, `git merge-tree` reports 20 paths changed on both sides, plus `vllm/v1/core/stream_admission.py` added on both sides. The model tree is not in that set. The port is file by file onto the tip symbols. The path list is in `vllm_nemotron_asr_streaming/refs.py`.

## Session binding

`vllm_nemotron_asr_streaming.session` accepts only `vllm.v1.engine.async_llm.AsyncLLM` and reads `llm.engine_core.admit_stream_async`. On this tip `AsyncLLM` does not define `admit_stream_async`; the method is on the `EngineCoreClient` stored as `engine_core`.

`open`, `feed`, `flush`, `finish`, `release`, and `abort` raise `NotImplementedError`. Nothing in this package imports `vllm_omni`.

## Install

```bash
pip install -e .
```

When `VLLM_PLUGINS` is an allowlist, include the entry-point name:

```bash
export VLLM_PLUGINS=nemotron_asr_streaming
```

That name is separate from the offline plugin's `nemotron_asr` entry.

## Tests

```bash
pip install -e '.[test]'
pytest
```

The tests check identity, the entry point, the stubbed `register()` path, and that the session factory binds `admit_stream_async`. They do not start vLLM.

An editable install of this skeleton resolves the `vllm.general_plugins` entry point to `register`. That check does not import vLLM and does not meet Gate A.
