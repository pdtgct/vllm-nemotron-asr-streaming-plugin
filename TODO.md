# Port map

Interface tree: `pdtgct/vllm` `rfc12-integration` @ `4c1809692dfa46fcdd7a9891f69a997afa1dbb78`, tree `465f10ad65c7742fca01e25e3a1a0f58c4608094`. The RFC1 tip `persistent-state-primitive-v2` @ `8b8327e2ee` is the ancestor those symbols came from. This is not a vLLM release.

The model package from `rfc1-5212-vllm` @ `43c0bed5` `vllm/models/nemotron_asr/` is copied under `vllm_nemotron_asr_streaming/nemotron_asr/` with imports rewritten to this package. It was not merged onto the tip. `refs.MERGE_CONFLICT_PATHS` is still the 20 paths `git merge-tree` reported, plus `vllm/v1/core/stream_admission.py` added on both sides. Those core files were not copied.

`register()` loads the RFC12 symbols and registers `Nemotron3_5AsrStreamingForRNNT` only. The lazy target is the ported `NemotronASRForRNNT` class (alias `Nemotron3_5AsrStreamingForRNNT`), which sets `supports_persistent_state` and `get_model_state_cls` → `NemotronASRModelState`. `PersistentStateLayerBase` is constructed in that class from `build_nemotron_persistent_state_spec`.

## Done

| Piece | Where |
| --- | --- |
| Model modules, file by file, import path rewritten | `vllm_nemotron_asr_streaming/nemotron_asr/` |
| Architecture string is `Nemotron3_5AsrStreamingForRNNT` | `identity.py`, and `translate_card_config` overwrites a Hub card's `architectures` field with that string |
| Offline key and `MODELS_CONFIG_MAP["Nemotron3_5AsrForRNNT"]` left untouched | `plugin.register` |
| F6: no silent pin of lookahead 13, no requirement that the supported set equal `(1,)` | `lookahead.resolve_served_lookahead`, `translate_card_config`, `validate_native_burst_config(..., require_160ms_profile=)` |
| F7: async scheduling must be `False` (not `None`) and prefix caching must be off, plus the other `validate_state_config` refusals | `engine_flags.persistent_state_config_errors`, `Nemotron3_5AsrStreamingConfig`, `StreamingSessionFactory` |
| Session `open` / `feed` / `flush` / `finish` / `abort` call `admit_stream_async`, `add_request_async`, and `abort_requests_async` | `session/factory.py` |
| `release` returns a `StateCompletion` only from `engine_core.wait_state_completion` | same |

`open` sends an empty resumable `EngineCoreRequest` (`n=1`, no stop strings, no stream-admission context). A second `open` of the same id resends that payload. `feed` sends a resumable update with `StreamAdmissionContext`. `flush` sends `flush_token_id` (the served config's `blank + 4`) without closing. `finish` sends a non-resumable update with `input_closed=True`, because a resumable empty update is an `empty_update` before close is applied. `abort` calls `abort_requests_async` and does not reuse the id.

Serve flags, also checked in code:

```
vllm serve <checkpoint-with-streaming-architecture> \
    --dtype float32 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching
```

## Remaining for installed plugin load → admit → chunks → retire

Gate A is not met. Nothing here has loaded this plugin inside a running RFC12 engine, admitted a stream, fed audio chunks, or retired a slot.

- `register()` against a real RFC12 import is `tests/test_rfc12_register.py`. It skips unless `VLLM_RFC12_CHECKOUT` points at a tree that imports. One run against `/tmp/vllm-rfc12` at `4c180969` (tree `465f10ad65`) imported that tree and `register()` returned. The model module was not imported. That is not an installed engine, and it is not admit → chunks → retire.
- The Hub repo id still has `architectures: ["Nemotron3_5AsrForRNNT"]`. vLLM will select the offline plugin for that string. This package selects the streaming class only after `translate_card_config` (which requires an explicit `num_lookahead_tokens` in the card's supported set `{0, 3, 6, 13}`) or an explicit architecture override. `AutoConfig` is not registered for `nemotron3_5_asr`.
- `AsyncLLM`'s output processor does not read `stream_output`. A rejected update can be treated as an `ERROR` finish and abort the session. `feed` does not yet install a processor queue that preserves the slot across that rejection.
- `StateCompletion` is applied in `StateScheduler` from `ModelRunnerOutput.state_completion` and is not forwarded to `AsyncLLM`. `release()` raises if `engine_core.wait_state_completion` is absent rather than inventing a fence. The retire step of the installed cycle needs that forward, or an equivalent client-visible signal that the cleanup handle has been acknowledged.
- `flush_token_id` is minted by config translation and sent as a prompt token. The ported forward path does not branch on that id.
- Audio chunks still have to be packed into the envelope the ported frontend expects (`accepted_audio`, `frontend`, `plan`). The session accepts token ids and optional `mm_features`; it does not run that packer.
- No weight load, no GPU forward, no `StateCompletion` observed from a real worker.

## CPU register command

```
git clone --filter=blob:none --branch rfc12-integration \
    https://github.com/pdtgct/vllm.git /tmp/vllm-rfc12
git -C /tmp/vllm-rfc12 checkout 4c1809692dfa46fcdd7a9891f69a997afa1dbb78
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
VLLM_RFC12_CHECKOUT=/tmp/vllm-rfc12 python -m pytest -q tests/test_rfc12_register.py
```

`git -C /tmp/vllm-rfc12 rev-parse HEAD^{tree}` should print `465f10ad65c7742fca01e25e3a1a0f58c4608094`.
