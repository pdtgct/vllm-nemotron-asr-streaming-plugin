# Port map

This package is a skeleton. Nothing below is implemented. Gate A (the plugin loaded inside an installed RFC1-tip vLLM) has not been run.

Source to port, file by file, from `pdtgct/vllm` `rfc1-5212-vllm` @ `43c0bed5` `vllm/models/nemotron_asr/` onto tip symbols at `persistent-state-primitive-v2` @ `8b8327e2ee`. Do not merge that branch. `git merge-tree` of the two tips reports the paths in `vllm_nemotron_asr_streaming.refs.MERGE_CONFLICT_PATHS` (20 paths changed on both sides) and `BOTH_ADDED_PATHS` (`vllm/v1/core/stream_admission.py`). The model tree is only on the source branch.

`rfc12-integration` was absent (`404`) at scaffold time. When it exists, use it as an interface reference for the RFC1+RFC2 tree. Do not treat a missing branch as a commit that was read.

Measured source line counts at `43c0bed5`:

| Module | Lines | Port note |
| --- | ---: | --- |
| `advance.py` | 3744 | Decode advance. Rebase onto tip scheduler output and `StateCompletion`. |
| `encoder_execution.py` | 2164 | Encoder execution. |
| `nemotron_asr.py` | 1190 | Model class. Becomes `Nemotron3_5AsrStreamingForRNNT`, not the offline name. |
| `encoder.py` | 1111 | FastConformer encoder. |
| `plan.py` | 769 | Row plan and session registry. |
| `kernel_profile.py` | 751 | |
| `rnnt.py` | 709 | Predictor, joint, greedy decode. |
| `decode_dispatch.py` | 661 | |
| `manifests.py` | 608 | Cadence table used by the F6 trap. Stdlib-only in the source. |
| `endpointing.py` | 604 | |
| `batch_invariance.py` | 582 | |
| `accepted_audio.py` | 566 | |
| `chunk_bucket_graph.py` | 511 | |
| `decode_graph.py` | 468 | |
| `frontend.py` | 429 | |
| `configuration_nemotron_asr.py` | 398 | Contains `_PINNED_LOOKAHEAD = 13`. Do not copy that pin blindly. See F6. |
| `native_burst.py` | 398 | `validate_native_burst_config` requires lookaheads `(1,)`. See F6. |
| `profile_execution.py` | 381 | |
| `state_scatter.py` | 366 | |
| `convert.py` | 325 | Publishes `architectures: [Nemotron3_5AsrForRNNT]`. The streaming plugin must not serve that string. |
| `model_state_v2.py` | 312 | Source `NemotronASRModelState`. Rebase onto tip `ModelState` (`initialize_state_slots`, `prepare_step`, `get_model_state_cls`). |
| `commit_sink.py` | 301 | |
| `publish.py` | 292 | |
| `bi_ops.py` | 266 | |
| `state_profile.py` | 256 | Builds the `PersistentStateSpec`. |
| `precision.py` | 244 | |
| `rules.py` | 182 | |
| `featurizer.py` | 185 | |
| `transcript.py` | 170 | |
| `processor.py` | 166 | Multimodal processor. Keep registry keys off `nemotron3_5_asr`. |
| `bi_evidence.py` | 138 | |
| `forward_ops.py` | 136 | |
| `profiling.py` | 114 | |
| `lid.py` | 109 | |
| `publication_identity.py` | 107 | |
| `rnnt_cell.py` | 102 | |
| `staging.py` | 60 | |
| `masks.py` | 55 | |
| `identity.py` | 8 | Source sets `ARCHITECTURE = "Nemotron3_5AsrForRNNT"`. Replace with `Nemotron3_5AsrStreamingForRNNT` in this package. |
| `__init__.py` | 3 | |

The current `model.py` only reserves the class name, sets `supports_persistent_state`, and returns `NemotronAsrStreamingModelState` from `get_model_state_cls`. Both constructors raise.

## Session factory

Module `vllm_nemotron_asr_streaming.session`. Bind only to `AsyncLLM.engine_core.admit_stream_async`. No `vllm_omni` import and no copy of an Omni `nemotron_session`.

The factory stores the bound method. It does not call it.

| Method | Contract to implement |
| --- | --- |
| `StreamingSessionFactory.open(request_id)` | Build an empty resumable `EngineCoreRequest` (`n=1`, no stop strings, no structured outputs, no transfer connectors, token ids only, no prompt embeddings) and call `admit_stream_async`. The caller supplies a globally unique id. `admit_stream` does not go through the input processor, so there is no random suffix, and `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` does not apply here. A retry resends the identical payload, including `arrival_time`. Any other use of a live or still-retiring id is `request_id_in_use`. Refusals (`capacity`, `active_limit`, `engine_paused`) allocate nothing. |
| `StreamingSession.feed` | Resumable update for the same id with `stream_admission` set. Each input replaces the prompt. Multimodal positions are relative to that input. Rejected updates arrive on `EngineCoreOutput.stream_output` and must not be treated as session end. The `AsyncLLM` output processor on this tip still treats an `ERROR` finish as the end of the request; the feed path has to account for that before it can report a rejection without dropping the slot. |
| `StreamingSession.flush` | Source mints `flush_token_id` as `blank + 4` in `translate_card_config`. Flush drains the current emission without closing the slot and without freeing it. It is not `finish`. |
| `StreamingSession.finish` | Close input with `StreamAdmissionContext(input_closed=True)` or a non-resumable update, then wait for drain (`draining_timeout` on the engine). |
| `StreamingSession.release` | The slot stays occupied until a `StateCompletion` covers the last scheduled step and the cleanup handle. `release` waits on that fence. It does not free the slot from the client. |
| `StreamingSession.abort` | `abort_requests_async` for this id, then the same fence. After a timed-out or cancelled admission, abandon the id and admit again with a new one. A late input for the old id must not reach the new session. |

Engine launch, when it exists, has to pass the flags in `engine_flags.py`. This package does not launch an engine.

## F6 lookahead pin trap

Hub encoder config (`nvidia/nemotron-3.5-asr-streaming-0.6b` `config.json`):

- `default_num_lookahead_tokens`: `3`
- `supported_num_lookahead_tokens`: `[3, 0, 6, 13]`

Source `manifests.CADENCES` right-context, recorded in `lookahead.py`:

| Lookahead tokens | Cadence label |
| ---: | --- |
| 0 | 80ms |
| 1 | 160ms |
| 3 | 320ms |
| 6 | 560ms |
| 13 | 1120ms |

Three different selections are sitting in the source and the card:

1. The card default is `3` (320ms). `default_num_lookahead_tokens` is the default arm, not a bound.
2. `translate_card_config` writes `_PINNED_LOOKAHEAD = 13` (1120ms) into `att_context_right` when `13` is in the supported set. That passes the card check and ignores the default.
3. `validate_native_burst_config` requires `supported_num_lookahead_tokens == (1,)` (160ms). `1` is not in this card's supported set, so that check rejects the Hub card.

The port has to take the requested arm from the caller, check it against the card's supported set, and refuse a pin that the card does not list. This scaffold does not choose an arm. Do not copy `_PINNED_LOOKAHEAD` or the native-burst `(1,)` check forward as behavior.

## F7 async scheduling and prefix caching

On the RFC1 tip, prefix caching defaults to on, and async scheduling resolves from `None` to on when the executor supports it. `validate_state_config` refuses both for a persistent-state model, and also refuses speculative decoding, KV transfer, dual batch overlap, sleep mode, CuMem, HiSparse, and any parallel size other than 1.

`engine_flags.REQUIRED_OVERRIDES` records `async_scheduling=False` and `enable_prefix_caching=False`. No call site applies them. A future engine launch passes them explicitly. Do not describe either flag as already off in a vLLM release, and do not quote benchmark figures as core behavior.

## Out of scope for the fold

- Registering `Nemotron3_5AsrForRNNT` or `nemotron3_5_asr`.
- Importing `vllm_omni` or borrowing its session type.
- Commits on `Sohaib-Ahmed21/vllm-nemotron-asr-plugin`.
- A PyPI `vllm` dependency or a claim that a release contains `admit_stream_async`.
- Serving `nvidia/nemotron-3.5-asr-streaming-0.6b` until something explicit selects `Nemotron3_5AsrStreamingForRNNT` without overwriting the offline key. Card rewrite and `hf_overrides` are not in this package.
