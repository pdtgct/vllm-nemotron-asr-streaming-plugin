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
| `engine_flags` | `persistent_state_config_errors(vllm_config) -> list[str]`; `runner_mode_refusals(vllm_config, *, stage) -> list[str]`, the one function that evaluates REG-ENG-021 to REG-ENG-025 for `verify_and_update_config` (`stage="hook"`) and for model and model-state construction (`stage="constructor"`) (REG-ENG-027); `model_state_has_state_group_layouts() -> bool`, whether the running core's `ModelState` defines `state_group_layouts` (a property counts), which both callers of `transaction_graph_config_errors` pass as `has_state_group_layouts` (TXG-CFG-016); `ADAPTER_DEPENDENCIES` (module -> names, members written dotted, e.g. `ModelState.post_warmup`, `ModelState.take_failed_requests`); `MODEL_STATE_GRAPH_HOOKS` (the six model-state hooks) |
| `transaction_graph_config` | Imports no vLLM. `transaction_graphs_on(hf_config)`, `largest_population(hf_config)`, `transaction_graph_config_errors(vllm_config, *, has_state_group_layouts) -> list[str]` (TXG-CFG-001, TXG-CFG-003; the keyword says whether the running core's model state offers `state_group_layouts`, which TXG-CFG-016 reads; its callers pass it) |
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
| `plan` (request rules) | `request_shape_fault(prompt, mm_features, *, num_computed_tokens, chunk_token_id, file_carrier_id, carrier_width)`, the request rule, which applies `feature_placement_fault(mm_features)` and `encoder_item_fault(feature, *, carrier_width)`; each returns the fault or `None` (REG-STATE-015) |
| model state | `file_buffers` (a `FileHostBuffers`); `add_request` records each request's verdict of `plan.request_shape_fault` and takes a file request's buffer at first schedule; `prepare_inputs_embeds` screens each real step's rows before core's encoder path and `prepare_inputs` classifies them; `plan_step(scheduler_output, *, stream_rows=None)` returns `.rows` (planning-failed rows stay, `failed=True`), runs once per step, and stages each advancing FILE row through `file_buffers.stage`; a file's `_FileProgress` carries `queue_count`, `queue_count_unknown` and `admission_generation` (FILE-QC-008 to FILE-QC-013) |
| `commit_sink` | `CommitPlan.queue_counts` (device `int32` `(P,)`); `SessionStatusReport.queue_count`; `QueueCount(request_id, admission_generation, count)`; `CommitCollection(statuses, failed, queue_counts)`, which the model's `collect_commit_status` returns (FILE-QC-001 to FILE-QC-007) |
| `encoder_execution` | `uncaptured_transition(geometry=, count=)`, the encoder binding's uncaptured transition at one CHUNK count (TXG-CAP-003); the binding's `warmed`, `capture_graph_domain`, `release_graphs` and `profile_uncaptured_cell` |
| `decode_graph` | `GraphRuntime` (`captures_on(device)`, `wrap`, `capture_scope`, `sync`), `platform_graph_runtime()`, `EagerStaticGraphWrapper` (REG-MEM-028), `wrapper_holds_graph` |
| `graph_memory` | The one capture-memory implementation (REG-MEM-027): `free_memory`, `retained`, `ChargedMargin`, `allocate_margin`, `ProfilingRequirement`, `profiling_requirement(vllm_config)`, `CaptureCall`, `serving_excess`; and, for the transaction-graph part only, the capture call's transient peak: `begin_transient_peak(device)`, `transient_peak(device, reserved_at_start)` (TXG-MEM-002) |
| `transaction_graph`, `advance` seams | Listed in `tests/support/txg_api.py`, the one module through which tests reach transaction-graph names: the cell key `(geometry_index, kind, count)`, `classify`, `cell_keys`, `arena_rows`, `transaction_graph_counters` with the capture-record keys (`Record`) and the fallback key `(reason, kind, count)`, `model._transaction_graph_binding` (`cells`, `failed`, `verified`, `execute`, which returns `None` when it declines a step, and `static_inputs`, the binding-owned `(ids_static, embeds_static)` of TXG-EXEC-007), `RowPlan.planned_rejection`, `RowPlan.is_fresh` and `RowPlan.multi_token` (a row scheduled with more than one token; TXG-COV-014), and the phase functions `prepare_model_rows`, `model_rows_body`, `commit_model_rows`. The binding's `execute` and the phase functions are looked up at call time. |
| `transaction_graph` | `TransactionGraphBinding(model, vllm_config)`: `cells` (key -> `Cell(key, graph, signature, decoder_tier)`), `failed`, `verified`, `n`, `p_max`, `execute`, `static_inputs` (the step static inputs `(ids_static[P_max], embeds_static[P_max, H])` a covered step's two copies write and its cells gather from by step position; `None` before a capture allocates them and after their release; TXG-EXEC-007), `counters()` (fallback reasons `multi-token`, `chunk-uncovered`, `echo-uncovered`, `signature`, `multi-geometry`, `uncaptured`, `empty`, `memory-profile`, `native-burst`), `record_all_failed_step(P)`, and the hook seam the model and model state call: `check_bindings(layer)` (TXG-LIFE-009, called through the state layer's `bind_observer` after each `bind_kv_cache` while the option is on), pool access inside the capture call through `CaptureCall.state_group_layouts()` (TXG-CAP-008; the binding keeps only plain values), `static_inputs`, `check_capture_entry`, `check_profiling_requirement(requirement, profiling_ran=)`, `begin_capture`, `capture(call)`, `discard_capture`, `release_profiling`, `verify_post_warmup`, `shutdown`. Also `Classification`, `callable_identity`, `process_constants(core)`. The model builds the binding at construction for an accepted option value and reads it through the property `_transaction_graph_binding`, which is `None` whenever `transaction_graphs_on(config)` is false. |
| `transaction_graph` (cell seams) | `_cell_program(kind, count, *, capacity, resident, pools)` (host only: the prepare phase's output for a cell's capture row plan, its nested launch sites bound, its signature and decoder tier) and `_run_cell(kind, count, prepared, pool_kwargs, encoder_transition)`, the body a cell captures; the CPU tests run it eagerly (`tests/support/txg_eager.py`). Arena field indices `F_POSITION`, `F_ENDPOINT_MODE`, `F_ENDPOINT_THRESHOLD`, `F_ENDPOINT_RESIDUE`; `_pack_arena`. Signature fields include `endpoint_enabled`, `endpoint_policy`, `forced_eou`. `discard_capture(error)` takes the call's error. |
| `advance` (endpoint, pools) | `forced_eou_candidates(scheduled_token_ids, is_chunk, eou_token_id) -> list[bool]` (TXG-COV-012); `RowPlan.forced_eou_candidate` and `PlanContext.forced_eou_candidate` (per-row `bool`, empty when no row is a candidate; `plan.prepare_plan_context(..., eou_token_id=)` fills it); `resident_pool_sequence(**pools)`, the projected resident-pool views in commit order |
| `state_layer` | `NemotronStateLayer(MambaBase)`: one `MambaSpec` state group of `mamba_type` `PERSISTENT_STATE`, `block_size = max_model_len`, `recomputable=False` (`get_kv_cache_config`); `bind_kv_cache`; the view cache `pages` / `component_pools`, held by reference and dropped on rebind, on an identity mismatch with the bound `kv_cache` and in the model state's `clear_cudagraphs` (REG-STATE-018); `initialize_pages(slots)`. `state_spec(layout, *, block_size)`, `check_page_size(spec, layout)`. |
| `state_scatter` | `execute_masked_page_scatter_` (the prevalidated executor, looked up by name where it is called), `launch_rejected_row_scatter(pool, page=1)` (one rejected row; records the argument, alignment included, as warmed; TXG-CAP-006, TXG-CAP-013) |
| `commit_sink` | `CommitTicket`: done only after its `stage` returned, so a stage that raises leaves it cancellable and a cancel after a successful stage does nothing (TXG-EXEC-010) |
| `graph_memory` (errors) | `release_error_frames(error)`: clears the finished frames an error holds through its cause and context (REG-MEM-017, TXG-LIFE-012) |
| `advance` (flat capture) | `flat_capture()` (the scope of a cell's warm-ups and outer capture), `flat_capture_active()`; `HostStaging.stage`, `stage_bucket` and the un-pooled staging raise inside it and while the current CUDA stream is capturing (TXG-CAP-001, TXG-CAP-002) |

Routing constraints the tests observe:
- `FileHostBuffers.stage` is the only way FILE samples leave the host buffer, and
  `file_rows.copy_chunk_to_device` is the only host-to-device copy of FILE samples; both are
  looked up at call time.
- FILE rows' encoder chunks run through `advance.advance_chunk_bucket` with the step's other CHUNK
  rows, at the server's one geometry.

Tests assert the information a spec requires in a log line or error, not its exact wording.

## Future roadmap

- **Session factory on core's automatic admission.** Trigger: the port in the `session` design's
  Future roadmap 1. Shape: `session.factory`'s session record loses its `pending`, `accepted` and
  `refused` admission states and takes retirement from the stream's terminal output; the seam
  conformance adapter's names change with it.
- **Page initial values in core.** Trigger: the `registration` design's Future roadmap 11. Shape:
  `state_layer` drops `initialize_pages`, and the forward's call to it goes.
- **Multi-unit rows.** Trigger: the `registration` design's Future roadmap 14. Shape: the
  transaction-graph seams address units instead of rows.
