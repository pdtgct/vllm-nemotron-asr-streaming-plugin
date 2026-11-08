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
| `card_parser` | `NemotronAsrStreamingConfigParser` (a `ConfigParserBase`, registered as `nemotron_asr_streaming`); `translate_card(config, processor) -> dict`; `schema_errors(config, processor) -> list[str]` |
| `tokenizer_mode` | `NemotronAsrStreamingTokenizer` with classmethod `from_pretrained(path_or_repo, *args, **kwargs)`, registered in `TokenizerRegistry` as `nemotron_asr_streaming`; the refusal for an appended token at a wrong ID says "unexpected ID" |
| `streaming_config` | `Nemotron3_5AsrStreamingConfig(VerifyAndUpdateConfig)` with static `verify_and_update_model_config` and `verify_and_update_config` |
| `lookahead` | `GEOMETRIES` (right context, label, geometry index): (0, `80ms`, 0), (1, `160ms`, 1), (3, `320ms`, 2), (6, `560ms`, 3), (13, `1120ms`, 4); `cadence_label`, `geometry_index`, `right_context_for_label`, `served_lookaheads(hf_config)`, `default_lookahead(hf_config)` |
| `locales` | `locale_language(name)`, `default_locale(table)`, `resolve_file_locale(table, *, language, exact_locale, language_locales)`, `supported_languages()` |
| `engine_flags` | `persistent_state_config_errors(vllm_config) -> list[str]` |
| `transcription` | `file_token_budget(num_samples, right_context) -> int`; `build_file_prompt(stt_params) -> dict` |
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
| `session.factory` | `StreamingSessionFactory(engine_client, hf_config, *, backoff_initial_s=0.05, backoff_max_s=1.0)`; `sessions` maps caller ID to a record whose `admission` is `pending`, `accepted` or `refused` |

## Model internals (`vllm_nemotron_asr_streaming.nemotron_asr`)

| Module or object | Public names and constraints |
| --- | --- |
| `manifests` | `REALTIME_MAX_TOKENS`, the one definition of the per-unit emission bound |
| `file_rows` | `FileAction` (DRAIN, CHUNK, TAIL, END); `plan_file_step(*, queue_len, position, total_samples, chunk_samples, tail_done) -> FileStepPlan(action, start, length)`; `StepRow(request_id, geometry_index, deadline, file_plan)`; `plan_chunk_groups(rows)`, which orders through `order_chunk_groups(groups)`; `FileHostBuffers` (`take`, `stage`, `header`, `free_finished`, `__contains__`); `copy_chunk_to_device(view, *, device)` |
| model state | `file_buffers` (a `FileHostBuffers`); `prepare_step` takes a file request's buffer at first schedule; `plan_step(scheduler_output, *, queue_lens=None, stream_rows=None)` returns `.rows` and `.groups`, runs once per step, and stages each advancing FILE row through `file_buffers.stage` |

Routing constraints the tests observe:
- `FileHostBuffers.stage` is the only way FILE samples leave the host buffer, and
  `file_rows.copy_chunk_to_device` is the only host-to-device copy of FILE samples; both are
  looked up at call time.
- FILE rows' encoder chunks run through `advance.advance_chunk_bucket`, once per geometry group,
  like stream rows.
