# Implementation layout

The module layout the tests are written against. The designs and specs under `docs/intent/` state
the behaviour; this file names where it lives so tests and code agree. Changing a name here means
changing the tests that use it.

## Package `vllm_nemotron_asr_streaming`

| Module | Public names |
| --- | --- |
| `__init__` | `register` (entry point `vllm_nemotron_asr_streaming:register`), `ARCHITECTURE` |
| `plugin` | `register()` |
| `identity` | `ARCHITECTURE`, `OFFLINE_ARCHITECTURE`, `LAZY_MODEL` |
| `card_parser` | Translation records `card_default_num_lookahead_tokens` from the card's `default_num_lookahead_tokens` and the default arms `encoder_execution_arm` `eager-graphed`, `decode_dispatch_arm` `dense-graphed`. `NemotronAsrStreamingConfigParser` (a `ConfigParserBase`, registered as `nemotron_asr_streaming`); `translate_card(config, processor) -> dict`; `schema_errors(config, processor) -> list[str]` |
| `tokenizer_mode` | `NemotronAsrStreamingTokenizer` with classmethod `from_pretrained(path_or_repo, *args, **kwargs)`, registered in `TokenizerRegistry` as `nemotron_asr_streaming`; the refusal for an appended token at a wrong ID says "unexpected ID" |
| `streaming_config` | `Nemotron3_5AsrStreamingConfig(VerifyAndUpdateConfig)` with static `verify_and_update_model_config` and `verify_and_update_config` |
| `lookahead` | `GEOMETRIES` (right context, label, geometry index): (0, `80ms`, 0), (1, `160ms`, 1), (3, `320ms`, 2), (6, `560ms`, 3), (13, `1120ms`, 4); `cadence_label`, `geometry_index`, `right_context_for_label`, `served_lookahead(hf_config) -> int` (the server's one lookahead: `num_lookahead_tokens` when present, else `card_default_num_lookahead_tokens`; REG-LA-001, REG-LA-009). No served set and no separate default reader exist. |
| `locales` | `locale_language(name)`, `default_locale(table)`, `resolve_file_locale(table, *, language, exact_locale, language_locales)`, `supported_languages()` |
| `engine_flags` | `persistent_state_config_errors(vllm_config) -> list[str]`; `ADAPTER_DEPENDENCIES` (module -> names, members written dotted, e.g. `ModelState.post_warmup`, `PersistentStateSpec.memory_contributors`); `MODEL_STATE_GRAPH_HOOKS` (the six model-state hooks) |
| `transaction_graph_config` | Imports no vLLM. `transaction_graphs_on(hf_config)`, `largest_population(hf_config)`, `transaction_graph_config_errors(vllm_config) -> list[str]` (TXG-CFG-001, TXG-CFG-003) |
| `transcription` | `build_file_prompt(stt_params) -> dict` |
| `model` | `Nemotron3_5AsrStreamingForRNNT` (`get_generation_prompt`, `get_speech_to_text_config`, `supported_languages`, `buffer_realtime_audio`, `realtime_max_tokens`, `get_model_state_cls`) |

Served-configuration fields for the control tokens: `eos_token_id` (park), `audio_chunk_token_id`,
`eou_token_id`, `file_carrier_token_id`, `continuation_token_id`; `vocab_size` is also the logit
width; `blank_token_id` and `num_asr_labels` are `blank`.

## Package `vllm_nemotron_asr_streaming.session`

Importing `session` imports neither vLLM nor the model class modules; the factory imports vLLM
lazily.

| Module | Public names |
| --- | --- |
| `session` | `create_session_factory(engine_client, model_config)`, `SessionRefused`, `SessionFailed`, `PackerCapacityError` |
| `session.packer` | `Unit` (`prompt_token_ids`, `row` float32 of width 17,927, `chunk_sequence`, `final_tail`); `Packer(*, right_context, geometry_index, locale_row, carrier_token_id, clock_ms, first_chunk_sequence=0)` with `append(samples) -> list[Unit]`, `close() -> Unit`, `set_locale_row(row)`, `buffered` |
| `session.factory` | `StreamingSessionFactory(engine_client, hf_config, *, backoff_initial_s=0.05, backoff_max_s=1.0)`; `open(*, locale)`; `sessions` maps caller ID to a record whose `admission` is `pending`, `accepted` or `refused`; `_faults` holds the cleanup faults the next `drain` returns (the seam conformance adapter reads both) |

## Model internals (`vllm_nemotron_asr_streaming.nemotron_asr`)

| Module or object | Public names and constraints |
| --- | --- |
| `manifests` | `REALTIME_MAX_TOKENS`, the one definition of the per-unit emission bound |
| `file_rows` | `FileAction` (DRAIN, CHUNK, TAIL, END); `plan_file_step(*, queue_len, position, total_samples, chunk_samples, tail_done) -> FileStepPlan(action, start, length)`; `StepRow(request_id, geometry_index, deadline, file_plan)`; `FileHostBuffers` (`take`, `stage`, `header`, `free_finished`, `__contains__`); `copy_chunk_to_device(view, *, device)` |
| model state | `file_buffers` (a `FileHostBuffers`); `prepare_step` takes a file request's buffer at first schedule; `plan_step(scheduler_output, *, stream_rows=None)` returns `.rows` (planning-failed rows stay, `failed=True`), runs once per step, and stages each advancing FILE row through `file_buffers.stage`; a file's `_FileProgress` carries `queue_count`, `queue_count_unknown` and `admission_generation` (FILE-QC-008 to FILE-QC-013) |
| `commit_sink` | `CommitPlan.queue_counts` (device `int32` `(P,)`); `SessionStatusReport.queue_count`; `QueueCount(request_id, admission_generation, count)`; `CommitCollection(statuses, failed, queue_counts)`, which the model's `collect_commit_status` returns (FILE-QC-001 to FILE-QC-007) |
| `encoder_execution` | `uncaptured_transition(geometry=, count=)`, the encoder binding's uncaptured transition at one CHUNK count (TXG-CAP-003); the binding's `warmed`, `capture_graph_domain`, `release_graphs` and `profile_uncaptured_cell` |
| `decode_graph` | `GraphRuntime` (`captures_on(device)`, `wrap`, `capture_scope`, `sync`), `platform_graph_runtime()`, `EagerStaticGraphWrapper` (REG-MEM-028), `wrapper_holds_graph` |
| `graph_memory` | The one capture-memory implementation (REG-MEM-027): `free_memory`, `retained`, `ChargedMargin`, `allocate_margin`, `ProfilingRequirement`, `profiling_requirement(vllm_config)`, `CaptureCall`, `serving_excess`; and, for the transaction-graph part only, the capture call's transient peak: `begin_transient_peak(device)`, `transient_peak(device, reserved_at_start)` (TXG-MEM-002) |
| `transaction_graph`, `advance` seams | Listed in `tests/support/txg_api.py`, the one module through which tests reach transaction-graph names: the cell key `(geometry_index, kind, count)`, `classify`, `cell_keys`, `arena_rows`, `transaction_graph_counters` with the capture-record keys (`Record`) and the fallback key `(reason, kind, count)`, `model._transaction_graph_binding` (`cells`, `failed`, `verified`, `execute`, which returns `None` when it declines a step), `RowPlan.planned_rejection` and `RowPlan.is_fresh`, and the phase functions `prepare_model_rows`, `model_rows_body`, `commit_model_rows`. The binding's `execute` and the phase functions are looked up at call time. |
| `transaction_graph` | `TransactionGraphBinding(model, vllm_config)`: `cells` (key -> `Cell(key, graph, signature, decoder_tier)`), `failed`, `verified`, `n`, `p_max`, `execute`, `counters()`, `record_all_failed_step(P)`, and the hook seam the model and model state call: `check_bindings` / `record_bindings` (`initialize_state_slots`), `check_capture_entry`, `check_profiling_requirement(requirement, profiling_ran=)`, `begin_capture`, `capture(call)`, `discard_capture`, `release_profiling`, `verify_post_warmup`, `shutdown`. Also `Classification`, `callable_identity`, `process_constants(core)`. The model builds the binding at construction for an accepted option value and reads it through the property `_transaction_graph_binding`, which is `None` whenever `transaction_graphs_on(config)` is false. |
| `advance` (flat capture) | `flat_capture()` (the scope of a cell's warm-ups and outer capture), `flat_capture_active()`; `HostStaging.stage`, `stage_bucket` and the un-pooled staging raise inside it and while the current CUDA stream is capturing (TXG-CAP-001, TXG-CAP-002) |

Routing constraints the tests observe:
- `FileHostBuffers.stage` is the only way FILE samples leave the host buffer, and
  `file_rows.copy_chunk_to_device` is the only host-to-device copy of FILE samples; both are
  looked up at call time.
- FILE rows' encoder chunks run through `advance.advance_chunk_bucket` with the step's other CHUNK
  rows, at the server's one geometry.

Tests assert the information a spec requires in a log line or error, not its exact wording.
