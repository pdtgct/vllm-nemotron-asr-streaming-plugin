# Registration specs

Design: [registration-design.md](registration-design.md). Prefix `REG`.

## Entry point and `register()`

- [ ] **REG-001**: The distribution `vllm-nemotron-asr-streaming` shall expose exactly one `vllm.general_plugins` entry point, named `nemotron_asr_streaming`, whose target is the plugin's `register` function.
- [ ] **REG-002**: If `AsyncLLM.generate` has no `stream_admission` parameter, or `AsyncLLM.wait_stream_retired` does not exist, then `register()` shall raise an error that names each missing call and states that the plugin requires an RFC 1 + RFC 2 vLLM build, and shall leave `ModelRegistry`, `MODELS_CONFIG_MAP`, the config parsers and the tokenizer modes unchanged.
- [ ] **REG-003**: When `register()` runs, it shall register the architecture `Nemotron3_5AsrStreamingForRNNT` with `ModelRegistry` as the lazy reference `vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT`, without importing the model module.
- [ ] **REG-004**: When `register()` runs, it shall set `MODELS_CONFIG_MAP["Nemotron3_5AsrStreamingForRNNT"]` to the plugin's verify-and-update class.
- [ ] **REG-005**: When `register()` runs, it shall register the card parser under the config format name `nemotron_asr_streaming`.
- [ ] **REG-006**: When `register()` runs, it shall register the tokenizer mode `nemotron_asr_streaming`.
- [ ] **REG-007**: If the `ModelRegistry` entry or the `MODELS_CONFIG_MAP` entry for the offline architecture `Nemotron3_5AsrForRNNT` differs after `register()` from its state before `register()`, then `register()` shall raise.
- [ ] **REG-008**: When `register()` is called more than once in one process, every call after the first shall leave the registrations of REG-003 to REG-006 unchanged and shall not raise.
- [ ] **REG-009**: `register()` shall initialize no device and load no model weights.
- [ ] **REG-010**: If any adapter dependency listed in the registration design (module and name) cannot be imported when the model class module is imported, then the import shall raise an error naming every missing module and name, before any weights load.
- [ ] **REG-012**: After importing `vllm_nemotron_asr_streaming`, running `register()` and importing `vllm_nemotron_asr_streaming.session`, neither model class module (`vllm_nemotron_asr_streaming.model`, `vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr`) shall be in `sys.modules`; the card parser's configuration module and engine modules imported by the public client and configuration APIs are allowed.
- [ ] **REG-013**: Importing the model class module shall initialize no device and load no weights.

## Card parser

- [ ] **REG-CARD-001**: When the card parser runs, it shall load `config.json` and `processor_config.json` from the local path given, or from the Hub repository at the requested `revision`.
- [ ] **REG-CARD-002**: When the card parser receives a `config.json` that has `encoder_config`, it shall translate it as a Hub card (REG-CARD-005 to REG-CARD-010).
- [ ] **REG-CARD-003**: When the card parser receives a `config.json` whose `architectures` is `["Nemotron3_5AsrStreamingForRNNT"]` and that has no `encoder_config`, it shall use it unchanged as an already-translated configuration.
- [ ] **REG-CARD-004**: If the card parser receives a `config.json` that is neither a Hub card nor an already-translated configuration, then it shall raise an error naming both accepted inputs.
- [ ] **REG-CARD-005**: If a Hub card's `vocab_size` is not `blank_token_id + 1`, or its `encoder_config` has no `supported_num_lookahead_tokens`, then the card parser shall raise an error naming the field.
- [ ] **REG-CARD-006**: When translating a Hub card, the card parser shall set the park token (`eos_token_id`), stream chunk-carrier, EOU, file-carrier and continuation token IDs to `blank + 1` through `blank + 5` in that order, `vocab_size` to `blank + 6`, and the model's logit width to `vocab_size`.
- [ ] **REG-CARD-007**: When translating a Hub card, the card parser shall map the encoder fields `hidden_size`, `num_hidden_layers`, `conv_kernel_size`, `sliding_window` and `num_mel_bins` to `d_model`, `n_layers`, `conv_kernel`, `att_context_left = sliding_window − 1` and `n_mels`, raising if any is missing or not an integer; map `decoder_hidden_size` and `num_decoder_layers`, when present, to `pred_hidden` and `pred_rnn_layers`; raise if `prompt_intermediate_size` is present and not `2 × hidden_size`; and carry no other `encoder_config` field.
- [ ] **REG-CARD-008**: When translating a Hub card, the card parser shall set `architectures` to `["Nemotron3_5AsrStreamingForRNNT"]`, record the card's `supported_num_lookahead_tokens` as `card_num_lookahead_tokens`, set `max_position_embeddings` to 2^24, set `decode_dispatch_arm` to `"dense-eager"`, and remove `model_type`, `is_encoder_decoder`, `default_prompt_id`, any top-level `sliding_window`, any top-level `supported_num_lookahead_tokens` and any default lookahead the card carries.
- [ ] **REG-CARD-009**: For either accepted input, the card parser shall store the locale table `prompt_dictionary` from `processor_config.json` in the configuration and keep the card's `num_prompts`; if an already-translated configuration stores a table that differs from the file's, then it shall raise.
- [ ] **REG-CARD-010**: When translating a Hub card that has a key with the same name as a value the translation computes (control-token IDs, `eos_token_id`, `vocab_size`, logit width, `architectures`), the card parser shall keep the computed value.
- [ ] **REG-CARD-011**: If, for either accepted input, the control-token IDs, `eos_token_id` or `vocab_size` differ from REG-CARD-006; `num_prompts` is not a positive integer or differs from `processor_config.json`'s; the locale table is missing or empty, or any of its values is not an integer row in `0 … num_prompts − 1` (several locales may share a row); or `card_num_lookahead_tokens` is empty or contains a value other than 0, 1, 3, 6 and 13, then the card parser shall raise an error naming the field.
- [ ] **REG-CARD-012**: The card parser's result shall depend only on the checkpoint files: it shall not read the `hf_overrides` keyword argument, except that if that argument is callable, then it shall raise an error stating that only the dictionary form of `--hf-overrides` is supported.
- [ ] **REG-CARD-013**: When the card parser is called with no `hf_overrides` (as core does when deriving a generation configuration for a checkpoint without `generation_config.json`), it shall succeed, and the configuration it returns shall yield a generation configuration whose only EOS is `blank + 1` and that sets no token limit.
- [ ] **REG-CARD-014**: The card parser shall return the configuration dictionary and an instance of the model's configuration class `NemotronASRConfig` built from it, whose `hidden_size` is the stream carrier row width 17,927 and whose `num_asr_labels` is `blank`.
- [ ] **REG-CARD-015**: The session, the packer, the file prompt and the model shall take every control-token ID from the served configuration, never from a constant.
- [ ] **REG-CARD-016**: The model class attribute `realtime_max_tokens`, the session's `max_tokens` (SESS-ENG-001) and the threshold of REG-VAL-001 shall all take their value from one definition in the geometry manifest module.

## Tokenizer mode

- [ ] **REG-TOK-001**: When the tokenizer mode `nemotron_asr_streaming` loads a tokenizer, it shall load the card's tokenizer and append, as special tokens, a blank placeholder at ID `blank` followed by the five control tokens of REG-CARD-006 at IDs `blank + 1` to `blank + 5`.
- [ ] **REG-TOK-002**: The tokenizer loaded by the tokenizer mode `nemotron_asr_streaming` shall report `eos_token_id == blank + 1` (the park token) and no other EOS token.
- [ ] **REG-TOK-003**: If any token appended by the tokenizer mode `nemotron_asr_streaming` is assigned an ID other than its expected ID, then loading the tokenizer shall raise.
- [ ] **REG-TOK-004**: The tokenizer mode `nemotron_asr_streaming` shall take `blank` from `blank_token_id` in the `config.json` beside the tokenizer; if that file has none, or the card tokenizer's vocabulary size differs from it, then loading the tokenizer shall raise.

## Lookahead selection and post-override validation

These checks run in `verify_and_update_model_config`, after core has applied every
`--hf-overrides` key, and apply to both accepted inputs.

- [ ] **REG-LA-001**: The served lookahead set shall be `supported_num_lookahead_tokens` when the final configuration has it (from `--hf-overrides`, or stored by an already-translated configuration), and `card_num_lookahead_tokens` otherwise.
- [ ] **REG-LA-002**: If the served lookahead set is empty, or contains a value other than the implemented right contexts 0, 1, 3, 6 and 13, then startup shall fail with an error naming the value.
- [ ] **REG-LA-003**: When the served lookahead set contains a value that `card_num_lookahead_tokens` does not contain, startup shall log a warning naming the value as a geometry the checkpoint does not declare.
- [ ] **REG-LA-004**: If the final configuration has no `num_lookahead_tokens` (for a Hub card: none was given in `--hf-overrides`), then startup shall fail with an error stating that the default lookahead must be set with `--hf-overrides`.
- [ ] **REG-LA-005**: If the default lookahead (`num_lookahead_tokens`) is not in the served lookahead set, then startup shall fail with an error naming both.
- [ ] **REG-LA-006**: When the default lookahead is valid, `verify_and_update_model_config` shall set the served configuration's `att_context_right` to it.
- [ ] **REG-LA-007**: If `experimental_native_burst` is true and the default lookahead is not 1, then startup shall fail with an error naming both values.
- [ ] **REG-VAL-001**: If, in the final configuration, the control-token IDs, `eos_token_id` or `vocab_size` differ from REG-CARD-006, any schema condition of REG-CARD-011 fails, or `max_model_len` is less than `1 + realtime_max_tokens` (143 for the implemented geometries), then startup shall fail with an error naming each.
- [ ] **REG-VAL-002**: If `language_locales` is given and any entry's value is not a key of the locale table, or its key is not that locale's language (the lowercase letters before the first `-` or uppercase letter, when there are two or three of them; `auto` has none), then startup shall fail with an error naming the entry.
- [ ] **REG-VAL-003**: If any check of REG-LA-002, REG-LA-004, REG-LA-005, REG-LA-007, REG-VAL-001, REG-VAL-002, REG-VAL-005, REG-VAL-006 or REG-ENG-001 to REG-ENG-005 fails, then the hook running it shall raise one `ValueError` that lists every failed check, not only the first.
- [ ] **REG-VAL-004**: When `verify_and_update_model_config` runs, it shall log each language that FILE-LANG-002 cannot map (its locales use several distinct rows, no locale is named for the language itself, and `language_locales` has no entry for it).
- [ ] **REG-VAL-005**: If `model_config.try_get_generation_config()` has an `eos_token_id` other than `blank + 1` alone, then startup shall fail with an error naming `--generation-config <dir>` with a generation configuration that has no other EOS.
- [ ] **REG-VAL-006**: If `model_config.get_diff_sampling_param()` contains `max_tokens`, then startup shall fail with an error naming `--generation-config vllm` and the removal of any token-limit `--override-generation-config`.

## Engine settings

- [ ] **REG-ENG-001**: If asynchronous scheduling is enabled or not explicitly set to `False`, then `verify_and_update_config` shall refuse the configuration.
- [ ] **REG-ENG-002**: If prefix caching is enabled, then `verify_and_update_config` shall refuse the configuration.
- [ ] **REG-ENG-003**: If speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator or hierarchical sparse attention is enabled, then `verify_and_update_config` shall refuse the configuration, naming each enabled setting.
- [ ] **REG-ENG-004**: If any of the tensor, pipeline, data, decode-context or prefill-context parallel sizes that the running vLLM build defines is other than 1, then `verify_and_update_config` shall refuse the configuration; a parallel setting the build does not define shall count as 1.
- [ ] **REG-ENG-005**: If `enforce_eager` is not set, then `verify_and_update_config` shall refuse the configuration.
- [ ] **REG-ENG-006**: When the model state is constructed, it shall re-check REG-ENG-001 to REG-ENG-005 against the final engine configuration and raise as in REG-VAL-003 if any fails.
- [ ] **REG-ENG-007**: The README's serve command shall show every flag REG-ENG-001 to REG-ENG-005 require, including an explicit `--no-async-scheduling`, together with `--config-format nemotron_asr_streaming`, `--tokenizer-mode nemotron_asr_streaming` and the `num_lookahead_tokens` override.

## Supported tasks

- [ ] **REG-TASK-001**: The model class shall declare `supports_realtime`, `supports_persistent_state` and `SupportsTranscription` with `supports_transcription_only = False`, `supports_segment_timestamp = False` and `supports_diarized_transcription = False`.
- [ ] **REG-TASK-002**: The model state's `get_supported_generation_tasks()` shall return exactly `("transcription", "realtime")`.

## Deferred

- [D] **REG-011**: Where vLLM releases include RFC 1 and RFC 2, `register()` shall check a minimum vLLM version in addition to the client-path check of REG-002.
