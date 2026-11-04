# vllm-nemotron-asr-streaming

Staging out-of-tree plugin for cache-aware streaming Nemotron ASR on vLLM core. The work is aimed at a later fold into the offline plugin at [Sohaib-Ahmed21/vllm-nemotron-asr-plugin](https://github.com/Sohaib-Ahmed21/vllm-nemotron-asr-plugin). This repository does not commit to that repo.

**Gate A is not met.** This plugin has not been loaded inside a running engine, and it has not admitted audio, fed chunks, or retired a slot. Unit tests stub the engine. An editable install resolving the entry point is not that cycle. See [TODO.md](TODO.md).

## What is registered

Distribution `vllm-nemotron-asr-streaming`, import package `vllm_nemotron_asr_streaming`.

Entry point group `vllm.general_plugins`:

```toml
nemotron_asr_streaming = "vllm_nemotron_asr_streaming:register"
```

`register()` imports these symbols from the RFC12 tree and then registers one architecture:

- `vllm.model_executor.layers.state.PersistentStateLayerBase`
- `vllm.v1.engine.async_llm.AsyncLLM`
- `vllm.v1.engine.core_client.EngineCoreClient.admit_stream_async`
- `vllm.v1.kv_cache_interface.StateCompletion`
- `vllm.v1.worker.gpu.model_states.resolve_model_state_cls` (the hook that calls `get_model_state_cls`)

The registered architecture is `Nemotron3_5AsrStreamingForRNNT`. The lazy class is the ported `NemotronASRForRNNT` (`supports_persistent_state`, `get_model_state_cls`). `Nemotron3_5AsrStreamingConfig` is stored on `MODELS_CONFIG_MAP` under that same name and refuses an engine left on the RFC12 async-scheduling or prefix-caching defaults.

The Hub config of [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) has `architectures: ["Nemotron3_5AsrForRNNT"]` and `model_type: "nemotron3_5_asr"`. That architecture string is the offline plugin's `ModelRegistry` key. `register()` never passes it to `register_model` and does not register the tokenizer or renderer key `nemotron3_5_asr`. A server started on the Hub repo id still resolves the offline class when that plugin is installed. `translate_card_config` writes `Nemotron3_5AsrStreamingForRNNT` into a translated config and requires an explicit `num_lookahead_tokens` from the card's supported set. It does not apply the card default `3` or the old source pin `13`.

## API target

Import target, not a release:

| | |
| --- | --- |
| Repo | `pdtgct/vllm` |
| Interface tree | `rfc12-integration` @ `4c1809692dfa46fcdd7a9891f69a997afa1dbb78`, tree `465f10ad65c7742fca01e25e3a1a0f58c4608094` |
| RFC1 tip | branch `persistent-state-primitive-v2` @ `8b8327e2eece1df39eb6318eff4c9dc207b3ab61` |
| Model source | branch `rfc1-5212-vllm` @ `43c0bed5bf877dc74de2e1494a3a542f497493de`, tree `vllm/models/nemotron_asr/` (19,941 Python lines), copied under `vllm_nemotron_asr_streaming/nemotron_asr/` |

`pyproject.toml` does not depend on PyPI `vllm`. A released vLLM does not provide these symbols.

Do not merge `rfc1-5212-vllm` onto the tip. Against `8b8327e2ee`, `git merge-tree` reports 20 paths changed on both sides, plus `vllm/v1/core/stream_admission.py` added on both sides. The model tree is not in that set. The port is file by file onto the tip symbols. The path list is in `vllm_nemotron_asr_streaming/refs.py`.

## Session binding

`vllm_nemotron_asr_streaming.session` accepts only `vllm.v1.engine.async_llm.AsyncLLM`. On this tree `admit_stream_async` is on the `EngineCoreClient` stored as `engine_core`.

`open` calls `admit_stream_async` with an empty resumable request. `feed` and `flush` call `add_request_async` with `StreamAdmissionContext`. `finish` sends a non-resumable update with `input_closed=True`. `abort` calls `abort_requests_async`. `release` returns a `StateCompletion` only when `engine_core.wait_state_completion` exists; the RFC12 scheduler consumes that fence internally and does not forward it. The factory raises when `async_scheduling` is not `False` or prefix caching is on.

```
vllm serve <checkpoint-with-streaming-architecture> \
    --dtype float32 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching
```

Nothing in this package imports `vllm_omni`.

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

The default tests stub vLLM. They check identity, registration, lookahead selection, the F7 refusal, and the session calls against that stub. They do not start an engine.

`tests/test_rfc12_register.py` calls `register()` against a checkout of `rfc12-integration` @ `4c180969`. It skips until the checkout imports. The command is in [TODO.md](TODO.md). A pass of that test still does not load the model or admit a stream.
