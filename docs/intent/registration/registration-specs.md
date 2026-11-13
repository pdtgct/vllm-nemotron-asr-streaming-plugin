# Registration specs

Design: [registration-design.md](registration-design.md). Prefix `REG`.

## Entry point and `register()`

- [x] **REG-001**: The distribution `vllm-nemotron-asr-streaming` shall expose exactly one `vllm.general_plugins` entry point, named `nemotron_asr_streaming`, whose target is the plugin's `register` function.
- [ ] **REG-002**: If `AsyncLLM.generate` has no `stream_admission` parameter, or `AsyncLLM.wait_stream_retired` does not exist, then `register()` shall raise an error that names each missing call and states that the plugin requires a vLLM build with RFC 1 stream admission (vllm#67375 on #67374), and shall leave `ModelRegistry`, `MODELS_CONFIG_MAP`, the config parsers and the tokenizer modes unchanged.
- [x] **REG-003**: When `register()` runs, it shall register the architecture `Nemotron3_5AsrStreamingForRNNT` with `ModelRegistry` as the lazy reference `vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT`, without importing the model module.
- [x] **REG-004**: When `register()` runs, it shall set `MODELS_CONFIG_MAP["Nemotron3_5AsrStreamingForRNNT"]` to the plugin's verify-and-update class.
- [x] **REG-005**: When `register()` runs, it shall register the card parser under the config format name `nemotron_asr_streaming`.
- [x] **REG-006**: When `register()` runs, it shall register the tokenizer mode `nemotron_asr_streaming`.
- [x] **REG-007**: If the `ModelRegistry` entry or the `MODELS_CONFIG_MAP` entry for the offline architecture `Nemotron3_5AsrForRNNT` differs after `register()` from its state before `register()`, then `register()` shall raise.
- [x] **REG-008**: When `register()` is called more than once in one process, every call after the first shall leave the registrations of REG-003 to REG-006 unchanged and shall not raise.
- [x] **REG-009**: `register()` shall initialize no device and load no model weights.
- [ ] **REG-010**: If any unconditional adapter dependency listed in the registration design (module and name; every row except the conditional rows of REG-014 and REG-015) cannot be imported when the model class module is imported, then the import shall raise an error naming every missing module and name, before any weights load.
- [ ] **REG-014**: The conditional graph-runtime dependencies shall be `vllm.distributed.parallel_state.graph_capture`, `vllm.forward_context.BatchDescriptor` and `set_forward_context`, `vllm.platforms.current_platform` and `vllm.utils.import_utils.resolve_obj_by_qualname`. When graphed execution is not configured, their absence shall not prevent model import or construction; if any is missing when graphed execution is configured, then model construction shall raise an error naming it.
- [ ] **REG-015**: The conditional batch-invariant dependency shall be `vllm.model_executor.determinism.batch_invariant`. When the batch-invariant mode is not configured, its absence shall not prevent model import or construction; if it is missing when that mode is configured, then model construction shall raise an error naming it.
- [ ] **REG-016**: The model adapter shall use no vLLM internal dependency outside the registration design's dependency table, except the interfaces that the design explicitly excludes from that table.
- [x] **REG-012**: After importing `vllm_nemotron_asr_streaming`, running `register()` and importing `vllm_nemotron_asr_streaming.session`, neither model class module (`vllm_nemotron_asr_streaming.model`, `vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr`) shall be in `sys.modules`; the card parser's configuration module and engine modules imported by the public client and configuration APIs are allowed.
- [x] **REG-013**: Importing the model class module shall initialize no device and load no weights.

## Card parser

- [x] **REG-CARD-001**: When the card parser runs, it shall load `config.json` and `processor_config.json` from the local path given, or from the Hub repository at the requested `revision`.
- [x] **REG-CARD-002**: When the card parser receives a `config.json` that has `encoder_config`, it shall translate it as a Hub card (REG-CARD-005 to REG-CARD-010).
- [x] **REG-CARD-003**: When the card parser receives a `config.json` whose `architectures` is `["Nemotron3_5AsrStreamingForRNNT"]` and that has no `encoder_config`, it shall use it unchanged as an already-translated configuration.
- [x] **REG-CARD-004**: If the card parser receives a `config.json` that is neither a Hub card nor an already-translated configuration, then it shall raise an error naming both accepted inputs.
- [x] **REG-CARD-005**: If a Hub card's `vocab_size` is not `blank_token_id + 1`, or its `encoder_config` has no `supported_num_lookahead_tokens`, then the card parser shall raise an error naming the field.
- [x] **REG-CARD-006**: When translating a Hub card, the card parser shall set `eos_token_id` (park), `audio_chunk_token_id` (stream chunk carrier), `eou_token_id`, `file_carrier_token_id` and `continuation_token_id` to `blank + 1` through `blank + 5` in that order, and `vocab_size` to `blank + 6`, which is also the model's logit width (the model has no separate logit-width field).
- [x] **REG-CARD-007**: When translating a Hub card, the card parser shall map the encoder fields `hidden_size`, `num_hidden_layers`, `conv_kernel_size`, `sliding_window` and `num_mel_bins` to `d_model`, `n_layers`, `conv_kernel`, `att_context_left = sliding_window − 1` and `n_mels`, raising if any is missing or not an integer; map `decoder_hidden_size` and `num_decoder_layers`, when present, to `pred_hidden` and `pred_rnn_layers`; raise if `prompt_intermediate_size` is present and not `2 × hidden_size`; and carry no other `encoder_config` field.
- [x] **REG-CARD-008**: When translating a Hub card, the card parser shall set `architectures` to `["Nemotron3_5AsrStreamingForRNNT"]`, record the card's `supported_num_lookahead_tokens` as `card_num_lookahead_tokens`, set `max_position_embeddings` to 2^24, set `decode_dispatch_arm` to `"dense-eager"`, and remove `model_type`, `is_encoder_decoder`, `default_prompt_id`, any top-level `sliding_window`, any top-level `supported_num_lookahead_tokens` and any default lookahead the card carries.
- [x] **REG-CARD-009**: For either accepted input, the card parser shall store the locale table `prompt_dictionary` from `processor_config.json` in the configuration and keep the card's `num_prompts`; if an already-translated configuration stores a table that differs from the file's, then it shall raise.
- [x] **REG-CARD-010**: When translating a Hub card that has a key with the same name as a value the translation computes (control-token IDs, `eos_token_id`, `vocab_size`, logit width, `architectures`), the card parser shall keep the computed value.
- [x] **REG-CARD-011**: If, for either accepted input, the control-token IDs, `eos_token_id` or `vocab_size` differ from REG-CARD-006; `num_prompts` is not a positive integer or differs from `processor_config.json`'s; the locale table is missing or empty, or any of its values is not an integer row in `0 … num_prompts − 1` (several locales may share a row); or `card_num_lookahead_tokens` is empty or contains a value other than 0, 1, 3, 6 and 13, then the card parser shall raise an error naming the field.
- [x] **REG-CARD-012**: The card parser's result shall depend only on the checkpoint files: it shall not read the `hf_overrides` keyword argument, except that if that argument is callable, then it shall raise an error stating that only the dictionary form of `--hf-overrides` is supported.
- [x] **REG-CARD-013**: When the card parser is called with no `hf_overrides` (as core does when deriving a generation configuration for a checkpoint without `generation_config.json`), it shall succeed, and the configuration it returns shall yield a generation configuration whose only EOS is `blank + 1` and that sets no token limit.
- [x] **REG-CARD-014**: The card parser shall return the configuration dictionary and an instance of the model's configuration class `NemotronASRConfig` built from it, whose `hidden_size` is the stream carrier row width 17,927 and whose `num_asr_labels` is `blank`.
- [x] **REG-CARD-015**: The session, the packer, the file prompt and the model shall take every control-token ID from the served configuration, never from a constant.
- [x] **REG-CARD-016**: The model class attribute `realtime_max_tokens`, the session's `max_tokens` (SESS-ENG-001) and the threshold of REG-VAL-001 shall all take their value from one definition in the geometry manifest module.

## Tokenizer mode

- [x] **REG-TOK-001**: When the tokenizer mode `nemotron_asr_streaming` loads a tokenizer, it shall load the card's tokenizer, keep any special tokens the card already has at IDs `blank` to `blank + 5` (the published card has `<pad>` at `blank` and `<blank>` at `blank + 1`), and append tokens for the remaining IDs up to `blank + 5`, so that every ID from `blank` to `blank + 5` is special in both senses core's detokenizers use: flagged special in the tokenizer backend, and listed in the tokenizer's `all_special_ids`.
- [x] **REG-TOK-002**: The tokenizer loaded by the tokenizer mode `nemotron_asr_streaming` shall set its EOS to the token at `blank + 1` (`eos_token_id == blank + 1`), replacing any EOS the card tokenizer declares, and shall report no other EOS token.
- [x] **REG-TOK-003**: If any token appended by the tokenizer mode `nemotron_asr_streaming` is assigned an ID other than its expected ID, then loading the tokenizer shall raise.
- [x] **REG-TOK-004**: The tokenizer mode `nemotron_asr_streaming` shall take `blank` from `blank_token_id` in the `config.json` beside the tokenizer; if that file has none, the card tokenizer's base vocabulary size differs from it, or the card tokenizer has a token at an ID of `blank` or more that is not special or is above `blank + 5`, then loading the tokenizer shall raise.

## Lookahead selection and post-override validation

These checks run in `verify_and_update_model_config`, after core has applied every
`--hf-overrides` key, and apply to both accepted inputs.

- [x] **REG-LA-001**: The served lookahead set shall be `supported_num_lookahead_tokens` when the final configuration has it (from `--hf-overrides`, or stored by an already-translated configuration), and `card_num_lookahead_tokens` otherwise.
- [x] **REG-LA-002**: If the served lookahead set is empty, or contains a value other than the implemented right contexts 0, 1, 3, 6 and 13, then startup shall fail with an error naming the value.
- [x] **REG-LA-003**: When the served lookahead set contains a value that `card_num_lookahead_tokens` does not contain, startup shall log a warning naming the value as a geometry the checkpoint does not declare.
- [x] **REG-LA-004**: If the final configuration has no `num_lookahead_tokens` (for a Hub card: none was given in `--hf-overrides`), then startup shall fail with an error stating that the default lookahead must be set with `--hf-overrides`.
- [x] **REG-LA-005**: If the default lookahead (`num_lookahead_tokens`) is not in the served lookahead set, then startup shall fail with an error naming both.
- [x] **REG-LA-006**: When the default lookahead is valid, `verify_and_update_model_config` shall set the served configuration's `att_context_right` to it.
- [x] **REG-LA-007**: If `experimental_native_burst` is true and the default lookahead is not 1, then startup shall fail with an error naming both values.
- [x] **REG-VAL-001**: If, in the final configuration, the control-token IDs, `eos_token_id` or `vocab_size` differ from REG-CARD-006, any schema condition of REG-CARD-011 fails, or `max_model_len` is less than `1 + realtime_max_tokens` (143 for the implemented geometries), then startup shall fail with an error naming each.
- [x] **REG-VAL-002**: If `language_locales` is given and any entry's value is not a key of the locale table, or its key is not that locale's language (the lowercase letters before the first `-` or uppercase letter, when there are two or three of them; `auto` has none), then startup shall fail with an error naming the entry.
- [x] **REG-VAL-003**: If any check of REG-LA-002, REG-LA-004, REG-LA-005, REG-LA-007, REG-VAL-001, REG-VAL-002, REG-VAL-005, REG-VAL-006 or REG-ENG-001 to REG-ENG-005 fails, then the hook running it shall raise one `ValueError` that lists every failed check, not only the first.
- [x] **REG-VAL-004**: When `verify_and_update_model_config` runs, it shall log each language that FILE-LANG-002 cannot map (its locales use several distinct rows, no locale is named for the language itself, and `language_locales` has no entry for it).
- [x] **REG-VAL-005**: If `model_config.try_get_generation_config()` has an `eos_token_id` other than `blank + 1` alone, then startup shall fail with an error naming `--generation-config <dir>` with a generation configuration that has no other EOS.
- [x] **REG-VAL-006**: If `model_config.get_diff_sampling_param()` contains `max_tokens`, then startup shall fail with an error naming `--generation-config vllm` and the removal of any token-limit `--override-generation-config`.

## Engine settings

- [ ] **REG-ENG-001**: If `async_scheduling` is `True`, then `verify_and_update_config` shall refuse the configuration; an unset `async_scheduling` (`None`) shall not be refused there.
- [x] **REG-ENG-002**: If `enable_prefix_caching` is `True` in the configuration, then `verify_and_update_config` shall refuse the configuration.
- [ ] **REG-ENG-003**: If speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator, hierarchical sparse attention or fault tolerance (`parallel_config.enable_fault_tolerance`) is enabled, then `verify_and_update_config` shall refuse the configuration, naming each enabled setting.
- [ ] **REG-ENG-008**: The native-burst configuration check shall count a parallel setting that the running build does not define as 1, and an undefined expert-parallel setting as off.
- [x] **REG-ENG-004**: If any of the tensor, pipeline, data, decode-context or prefill-context parallel sizes that the running vLLM build defines is other than 1, then `verify_and_update_config` shall refuse the configuration; a parallel setting the build does not define shall count as 1.
- [x] **REG-ENG-005**: If `enforce_eager` is not set, then `verify_and_update_config` shall refuse the configuration.
- [ ] **REG-ENG-006**: When the model state is constructed, it shall re-check REG-ENG-002 to REG-ENG-005 against the final engine configuration, and shall refuse a resolved `async_scheduling` that is not `False` (including `None`), raising as in REG-VAL-003 if any fails.
- [ ] **REG-ENG-007**: The README's serve command shall show `--enforce-eager` and every other flag REG-ENG-003 to REG-ENG-005 require, and may omit `--no-async-scheduling` and `--no-enable-prefix-caching` (core resolves both off for this model), together with `--dtype float32` (the model's qualified precision),  `--config-format nemotron_asr_streaming`, `--tokenizer-mode nemotron_asr_streaming` and the `num_lookahead_tokens` override.

## Execution memory

A real step is a step whose `prepare_step` was called with `dummy_run` false; a dummy step is one called
with `dummy_run` true.

- [ ] **REG-MEM-001**: When `prepare_step` is called with `dummy_run` and `is_profile` both true, it shall record the step as a profiling dummy step and shall not prepare execution itself; when it is called with `dummy_run` or `is_profile` false, it shall record the step as not a profiling dummy step, replacing any earlier record.
- [ ] **REG-MEM-002**: When the model's forward runs on the first profiling dummy step of a worker, it shall warm and capture the configured execution inventory, verify readiness of every configured execution entry and capture of every configured graph entry (each served geometry and batch tier of a graphed arm, and each configured CHUNK bucket), then, with every captured graph retained, run the profile transaction at its maximum shape, and return the dummy output core expects (a tensor shaped like the input embeddings).
- [ ] **REG-MEM-003**: When the model's forward runs on a later profiling dummy step of a worker whose preparation succeeded, including one that core runs after the state pool exists with real token IDs and attention metadata, it shall verify readiness and capture as in REG-MEM-002 and return the dummy output without capturing, re-running the profile transaction or reading state slots.
- [ ] **REG-MEM-004**: If any step of REG-MEM-002 fails, then the model shall discard its partial execution state, mark the worker's preparation failed and raise; any later profiling dummy step in that worker shall raise without preparing.
- [ ] **REG-MEM-005**: Preparation (REG-MEM-002) shall use model-owned scratch state built from the model's `PersistentStateSpec` and shall not read the state-slot bindings or the state pool.
- [ ] **REG-MEM-006**: If a real step would use graphed execution and any expected graph entry is not captured in its graph wrapper (the wrapper's entry holds no graph), then the model shall raise without capturing.
- [ ] **REG-MEM-007**: The model's graph runtime shall capture through core's graph-capture context, forward context and the platform's static graph wrapper, and shall neither import nor call a capture-enabled switch.
- [ ] **REG-MEM-009**: When the model's forward runs on a dummy step that is not a profiling dummy step, it shall return the dummy output without capturing, without running the profile transaction and without reading state-slot handles or rows.
- [ ] **REG-MEM-012**: When `prepare_inputs` runs on a dummy step (profiling or not, before or after the state pool exists), the model state shall prepare dummy inputs without reading state-slot handles, state-slot rows or scheduler output retained from an earlier step.
- [ ] **REG-MEM-010**: The dummy output of REG-MEM-002, REG-MEM-003 and REG-MEM-009 shall have the input embeddings' token count, width, dtype and device, with every carrier-column value a valid decision ID (zero).
- [ ] **REG-MEM-013**: When core samples a dummy output, the model's sampler hooks, including the native-burst sampler wrapper, shall read no state slots.
- [ ] **REG-MEM-011**: If a real step schedules any row for model execution and preparation has not succeeded in this worker, then the model shall raise without preparing.
- [ ] **REG-MEM-014**: When a real step schedules no row for model execution (cleanup only), the model shall complete it whether or not preparation has succeeded.
- [ ] **REG-MEM-008**: The model's forward shall not run a forward-time profile of an unprepared encoder on a profiling dummy step; REG-MEM-002 is the only profile the model runs.

## Model-state interface

- [ ] **REG-MS-001**: Every `ModelState` method the model state overrides shall accept every call that the base method's signature at the target core accepts (same positional parameters, keyword-only parameters and defaults).
- [ ] **REG-MS-002**: When `dummy_inputs_embeds(num_tokens, input_ids)` is called, with `input_ids` a tensor or `None`, the model state shall return its pre-allocated embeddings buffer sliced to `num_tokens`.

## State-slot access

- [ ] **REG-STATE-001**: The persistent-state layer shall build its spec as `PersistentStateSpec(descriptors=..., page_size_bytes=..., block_size=1)`, passing no other argument.
- [ ] **REG-STATE-002**: `prepare_step` shall read state-slot handles only from the scheduler output and shall not call `bind_rows` or read `component_pools`.
- [ ] **REG-STATE-003**: The model shall read state-slot rows only through `bind_rows` with the model's state layer name, the step's ordered request IDs and their scheduler-output handles, and then only `component_pools` and `slot_indices_cpu` of the result.
- [ ] **REG-STATE-004**: When the model commits state for a step, it shall first re-read the rows' `component_pools`; if that read raises, then the model shall write no state.
- [ ] **REG-STATE-005**: The model's projection of state pools into named state tensors shall take the component pools and the model's `PersistentStateSpec` and use no other member of the rows.
- [ ] **REG-STATE-006**: File-row planning, initialization and commits shall follow REG-STATE-003 and REG-STATE-004 as stream rows do.
- [ ] **REG-STATE-007**: When `prepare_step` takes a file request, it shall read only the scheduler output and host data.

## README

- [ ] **REG-DOC-001**: The README shall define stage 1 as upstream vLLM `main` with vllm#67374, #67375, #67144 and #67151, and stage 2 as stage 1 plus RFC 2 (#65546, #67294) and a frontend endpoint plugin.
- [ ] **REG-DOC-002**: Every README acceptance claim shall name the tier it met and the build and source revisions it was met on, and shall mark evidence from earlier builds as historical.
- [ ] **REG-DOC-003**: The README shall state that on a core without vllm#67151 the request guards are inactive, beam search can issue up to `n × max_model_len` engine requests per upload (one per beam per pass), and the transcription route must not be exposed to untrusted clients on such a core; that on a core without vllm#67144 a file ending in `error` or `abort` returns partial text with HTTP 200; and that on a core without vllm#67151 a file ending in a server-caused `length` does.

## Supported tasks

- [x] **REG-TASK-001**: The model class shall declare `supports_realtime`, `supports_persistent_state` and `SupportsTranscription` with `supports_transcription_only = False`, `supports_segment_timestamp = False` and `supports_diarized_transcription = False`.
- [x] **REG-TASK-002**: The model state's `get_supported_generation_tasks()` shall return exactly `("transcription", "realtime")`.

## Deferred

- [D] **REG-011**: Where vLLM releases include RFC 1's stream admission, `register()` shall check a minimum vLLM version in addition to the client-path check of REG-002.
