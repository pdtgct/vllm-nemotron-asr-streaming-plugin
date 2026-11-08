---
parent: high-level-design
prefix: REG
---

# Registration

## Context and Design Philosophy

Registration is everything vLLM sees of this plugin before a request arrives: the entry point,
the architecture key, how the Hub checkpoint's configuration and tokenizer become a served
configuration, and the engine settings the model refuses. Its job is to make a misconfigured
server fail at startup with a message naming the setting, rather than serve wrong transcripts or
fail on the first stream.

Two principles govern it:
- **Coexist without touching the offline plugin.** The Hub card names `Nemotron3_5AsrForRNNT`,
  which is the offline plugin's architecture key. This plugin registers a different key and never
  writes the offline key or its configuration hook.
- **Check the surface actually used, where it is used.** Startup checks test for the calls the
  plugin makes, in the process that makes them, so a check failure means the plugin genuinely
  cannot run.

## Entry point and `register()`

The distribution `vllm-nemotron-asr-streaming` exposes one `vllm.general_plugins` entry point,
`nemotron_asr_streaming`. vLLM calls it once per process. In the API server it runs before
argument parsing and model configuration, so the parser and tokenizer mode exist when
`--config-format` and `--tokenizer-mode` are resolved; engine-core and worker processes receive
the resolved configuration and load the plugin so the architecture and model hooks are
registered before the model is built. When `VLLM_PLUGINS` is set it is an allowlist of
entry-point names and must include `nemotron_asr_streaming`; otherwise the parser and tokenizer
mode below do not exist and core rejects `--config-format nemotron_asr_streaming` as unknown.

`register()` performs, in order:
1. **Client-path check.** `AsyncLLM.generate` must accept `stream_admission` and
   `AsyncLLM.wait_stream_retired` must exist. A missing one raises, naming it and stating that the
   plugin targets an RFC 1 + RFC 2 build. `register()` initializes no device and loads no model;
   the imports it needs may pull in engine modules transitively.
2. **Architecture.** `ModelRegistry.register_model("Nemotron3_5AsrStreamingForRNNT",
   "vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT")`, a lazy reference.
3. **Configuration hook.** `MODELS_CONFIG_MAP["Nemotron3_5AsrStreamingForRNNT"]` set to the
   plugin's verify-and-update class (post-override validation, below).
4. **Config parser.** `register_config_parser("nemotron_asr_streaming")` installs the card parser.
5. **Tokenizer mode.** Registers the tokenizer mode `nemotron_asr_streaming` (below).
6. **Offline-key guard.** The offline registry entry and its `MODELS_CONFIG_MAP` entry are
   compared before and after; any change raises.

`register()` is idempotent within a process. The model-adapter dependencies (below) are checked
when the model class is imported in the worker, before any weights load.

## Card parser

The Hub card's `model_type` (`nemotron3_5_asr`) has no vLLM configuration class, and its
`architectures` names the offline key. The served configuration is produced by a config parser
selected with `--config-format nemotron_asr_streaming`.

`parse(model, trust_remote_code, revision, code_revision, **kwargs)` loads `config.json` (local
path or Hub revision) and accepts one of two inputs:
- **A Hub card** (has `encoder_config`): translated as below.
- **An already-translated configuration** (its `architectures` is the streaming key): used as is,
  then validated.
Anything else raises, naming the expected inputs.

Translation of a card:
- requires `encoder_config`, `vocab_size == blank_token_id + 1` (labels are `0 … blank-1`) and
  `encoder_config.supported_num_lookahead_tokens`;
- mints the control tokens after the blank (see the `transcription` design for the table) and sets
  `vocab_size = blank + 6` and the model's logit width to `vocab_size`;
- maps the encoder fields, writes the streaming architecture key, and drops `model_type` and
  `is_encoder_decoder`;
- reads the locale table `prompt_dictionary` from the card's `processor_config.json` and sets
  `num_prompts` to its size; a missing or invalid table raises;
- sets `max_position_embeddings` to 2^24. The model has no positional limit; this value only lifts
  core's derived context ceiling, so the deployment's `max_model_len` is the effective bound;
- records the served lookahead set and the default lookahead (below).

Precedence is fixed: values the translation computes (control tokens, vocabulary, logit width,
architecture) win over any same-named card key, and an override of any of them is refused.

**Overrides.** Core applies `--hf-overrides` only after the parser returns, but translation needs
the lookahead while it builds the configuration. So the parser reads `num_lookahead_tokens`,
`supported_num_lookahead_tokens`, `experimental_native_burst` and `language_locales` (see the
`transcription` design) from the `hf_overrides` keyword it receives. Only dictionary overrides are supported; a callable override is refused with a message
to use the dictionary form. Core then applies the same overrides, and post-override validation
confirms they agree.

The parser returns the translated dictionary and a `PretrainedConfig` built from it. It provides no
generation configuration: the served model has no generation-config `max_tokens`.

## Tokenizer mode

The card's tokenizer holds the label pieces only. Core detokenizes every emitted token, and
`skip_special_tokens` drops only tokens the tokenizer marks special. The tokenizer mode
`nemotron_asr_streaming` (selected with `--tokenizer-mode nemotron_asr_streaming`) loads the card's
tokenizer and appends, in order, a blank placeholder at `blank` and the control tokens at
`blank + 1 … blank + 5`, all special, and checks that each lands at its expected ID. Startup fails
otherwise.

## Lookahead selection

The model implements five encoder geometries, `80ms`, `160ms`, `320ms`, `560ms` and `1120ms`
(left context 56 frames, right context 0, 1, 3, 6 and 13). The card declares the lookaheads it was
trained for (0, 3, 6 and 13).

- **Served set.** The card's set, or the `supported_num_lookahead_tokens` override when given. An
  override replaces the card's set; every value must be one of the five right contexts, and the set
  must not be empty. A value outside the card's set (lookahead 1) is allowed but logged as a
  geometry the checkpoint does not declare.
- **Default lookahead.** `num_lookahead_tokens` must be given explicitly in `--hf-overrides` and
  must be in the served set. It is the cadence for complete files and core's `/v1/realtime`, and the
  configuration's nominal `att_context_right`. A card's own default or top-level value is never
  used.
- **Native burst.** `experimental_native_burst: true` requires the default lookahead to be 1; it
  does not restrict the served set.

## Post-override validation

The configuration hook runs twice during startup:
- `verify_and_update_model_config(model_config)` runs at the end of model configuration, after
  overrides. It checks `num_lookahead_tokens == att_context_right`, membership in the served set,
  every served value being one of the five right contexts, the architecture being the streaming
  key, the control-token layout, and `max_model_len ≥ 1 + realtime_max_tokens` (the smallest
  context in which one stream unit fits).
- `verify_and_update_config(vllm_config)` runs once while the engine configuration is resolved and
  refuses the engine settings below. Settings that core resolves only later are re-checked when the
  model state is constructed.

Both raise `ValueError` listing every violated condition.

## Engine settings

The persistent-state model needs one scheduler step at a time, one copy of each session's state
and a single device. The plugin refuses:
- asynchronous scheduling (`async_scheduling` must be `False`; unset counts as a refusal);
- prefix caching;
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator and
  hierarchical sparse attention;
- any tensor, pipeline, data, decode-context or prefill-context parallel size other than 1 (a
  setting the build does not have counts as 1);
- an outer runner that is not eager (`enforce_eager` must be set). The model runs its own
  captured encoder and decoder graphs inside the eager outer step.

The README's serve command shows the required flags.

## Supported tasks

The model class declares `supports_realtime`, `supports_persistent_state` and
`SupportsTranscription` with `supports_transcription_only = False`. Core would also infer
`generate` from the presence of `compute_logits`, so the model's model state overrides
`get_supported_generation_tasks()` to return exactly `("transcription", "realtime")`. A frontend
requiring `realtime` loads; one that needs complete-file transcription checks for
`transcription`.

## Model-adapter dependencies

The model adapter integrates with vLLM's model-state runner and depends on these internal symbols
beyond the public client surface. This is the complete permitted set, checked when the model class
is imported:

| Module | Names |
| --- | --- |
| `vllm.config.compilation` | `CUDAGraphMode` |
| `vllm.v1.core.sched.output` | `NewRequestData`, `SchedulerOutput` |
| `vllm.v1.kv_cache_interface` | `KVCacheConfig`, `StateSlotHandle` |
| `vllm.v1.worker.gpu.input_batch` | `InputBatch` |
| `vllm.v1.worker.gpu.model_states.interface` | `ModelState` |
| `vllm.v1.worker.gpu.state_slots` | `StateSlotBindings`, `StateSlotRows` |
| `vllm.v1.worker.gpu.states` | `RequestState` |
| `vllm.v1.worker.utils` | `AttentionGroup` |
| `vllm.model_executor.layers.state` | `PersistentStateLayerBase` |
| `vllm.multimodal` | `MULTIMODAL_REGISTRY` |
| `vllm.v1.attention.backends.utils` | `NULL_BLOCK_ID` |

Adding a dependency means adding it here and to the import-time check.

## Current source

The vendored code differs from this design in ways the implementation changes: today's
registration checks `StateCompletion` and `EngineCoreClient.admit_stream_async`, registers no
parser or tokenizer mode, and has an empty model-config hook; the translation mints four control
tokens with `vocab_size = blank + 5`, lets leftover card keys overwrite computed fields, falls back
to the card's lookahead, and validates an override set against the card's set; the logit width is a
constant; the engine-flag check has no eager-runner refusal and treats a missing parallel setting
as a refusal.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| How the card becomes a served config | A registered config parser selected by `--config-format nemotron_asr_streaming` | `AutoConfig` registration for `nemotron3_5_asr`; `--hf-overrides` alone | Registering the card's `model_type` would claim the offline plugin's configuration; overrides apply after the configuration is built, too late for translation |
| Where the parser gets the lookahead | The dictionary `hf_overrides` passed to `parse`, re-validated after core applies them | A plugin-specific environment variable or CLI option | One user-facing setting, and post-override validation catches disagreement |
| Control tokens in the tokenizer | A registered tokenizer mode that appends them as special tokens | Publishing a modified tokenizer; leaving them unknown | Core detokenizes every token; unknown IDs break detokenization and unmarked ones appear in text; the card must stay usable unmodified |
| Context ceiling | Translation sets `max_position_embeddings` to 2^24 | Requiring `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | The model has no positional limit; long files need a context far above the card's 32,768 |
| Override of the served set | Replaces the card's set; values limited to the five geometries | Union with the card's set; any value | Replacement lets a deployment narrow the warmed set; values without a geometry cannot run |
| Load-time checks | Client calls in `register()`; adapter symbols at model import | All checks in `register()` | Worker symbols belong to the worker; the API server should not import them |
| `generate` task | Not advertised (model-state override) | Advertise for compatibility | The model emits transcript labels, not text generation |

## Open Questions & Future Decisions

### Deferred
1. A version gate for clean vLLM releases once RFC 1 and RFC 2 ship; until then the client-path
   check is the gate.

## References

- vLLM config parsers: `vllm/transformers_utils/config.py` (`register_config_parser`, the parser
  call and override application), `vllm/transformers_utils/config_parser_base.py`.
- Model configuration hooks: `vllm/model_executor/models/config.py` (`MODELS_CONFIG_MAP`).
- Supported tasks: `vllm/v1/worker/gpu/model_states/interface.py`.
- Plugin loading: `vllm/plugins/__init__.py`.
- Card translation: `nemotron_asr/configuration_nemotron_asr.py`; geometries:
  `nemotron_asr/manifests.py`.
