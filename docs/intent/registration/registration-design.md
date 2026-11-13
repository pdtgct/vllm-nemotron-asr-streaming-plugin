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

`register()` performs the following, and changes no registry until the client-path check has
passed:
1. **Client-path check.** `AsyncLLM.generate` must accept `stream_admission` and
   `AsyncLLM.wait_stream_retired` must exist. A missing one raises, naming it and stating that the
   plugin needs a vLLM build with RFC 1's stream admission (vllm#67375 on #67374). `register()` initializes no device and loads no model;
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

`parse(model, trust_remote_code, revision, code_revision, **kwargs)` loads `config.json` and
`processor_config.json` from the local path or the requested Hub revision, and accepts one of two
inputs:
- **A Hub card** (has `encoder_config`): translated as below.
- **An already-translated configuration** (its `architectures` is the streaming key, and it has no
  `encoder_config`): used as is.
Anything else raises, naming the expected inputs. For both inputs the parser stores the locale
table from `processor_config.json` in the configuration as `prompt_dictionary` (an already-translated
configuration that stores its own table must agree with the file), and both pass the same schema
check: the control-token layout, `eos_token_id` and `vocab_size`; a positive integer `num_prompts`
(the size of the model's prompt embedding table) that `processor_config.json` repeats; a non-empty
locale table whose values are integer rows in `0 … num_prompts − 1`, where several locale names may
share a row (aliases such as `en` and `en-US`) and some rows may be unused; and a non-empty
`card_num_lookahead_tokens` drawn from the five implemented right contexts. Post-override validation repeats this check on the final
configuration, since overrides can change any of these fields.

The returned `PretrainedConfig` is an instance of the model's configuration class
(`NemotronASRConfig`), which supplies the model's remaining fields and defaults. In particular its
`hidden_size` is the stream carrier row width (17,927), the width core's architecture conversion
reads, and `num_asr_labels` is `blank`. The configuration carries no top-level `sliding_window`:
core would cap the derived context at it, and the encoder's attention window is the model's own
`att_context_left`.

The parser is a function of the checkpoint alone. It does not read `hf_overrides`: core passes it
only the flat overrides, applies them again to the returned configuration, applies dictionary-valued
overrides such as `language_locales` later still, and also calls the parser with no overrides at all
when it derives a generation configuration from a checkpoint that has no `generation_config.json`.
Every rule that depends on launch options runs in post-override validation (below). A callable
`hf_overrides` given to the parser is refused with a message to use the dictionary form.

Translation of a card:
- requires `encoder_config`, `vocab_size == blank_token_id + 1` (labels are `0 … blank-1`) and
  `encoder_config.supported_num_lookahead_tokens`;
- mints the control tokens after the blank (see the `transcription` design for the table), sets
  `eos_token_id` to the park token `blank + 1` as the only EOS, `vocab_size = blank + 6` and the
  model's logit width to `vocab_size`;
- maps the encoder fields, each required to be an integer: `hidden_size` → `d_model`,
  `num_hidden_layers` → `n_layers`, `conv_kernel_size` → `conv_kernel`, `sliding_window` →
  `att_context_left` (`sliding_window − 1`), `num_mel_bins` → `n_mels`; maps
  `decoder_hidden_size` → `pred_hidden` and `num_decoder_layers` → `pred_rnn_layers` when present;
  refuses a `prompt_intermediate_size` other than `2 × hidden_size`; carries no other encoder
  field; writes the streaming architecture key; and drops `model_type` and `is_encoder_decoder`;
- keeps the card's `num_prompts` (the prompt embedding table has that many rows, which can exceed
  the number of distinct rows the locale table uses);
- sets `max_position_embeddings` to 2^24. The model has no positional limit; this value only lifts
  core's derived context ceiling, so the deployment's `max_model_len` is the effective bound;
- records the card's lookahead set as `card_num_lookahead_tokens`, drops `default_prompt_id` (the
  default locale is the rule in the `session` and `transcription` designs), drops any default
  lookahead and
  any top-level `supported_num_lookahead_tokens` the card carries, and declares
  `decode_dispatch_arm = "dense-eager"`, the decoder dispatch the model requires a configuration to
  declare.

Every component (the session, the packer, the file prompt and the model) takes control-token IDs
from the served configuration, never from constants. The per-unit emission bound
`realtime_max_tokens` (142 for the implemented geometries) has one definition, in the geometry
manifest module, which the model class attribute, the session's `max_tokens` and post-override
validation all use.

Values the translation computes (control-token IDs, `eos_token_id`, vocabulary, logit width,
architecture) win over any same-named card key. Post-override validation refuses a final
configuration in which a computed token or vocabulary value differs from the computed layout,
which catches an override of one. An override of `architectures` selects a different model, whose
own hook runs instead; this plugin does not govern it.

The parser returns the translated dictionary and a `PretrainedConfig` built from it. A generation
configuration derived from it carries only the park token as EOS and no length limit.

## Tokenizer mode

The card's tokenizer has a base vocabulary of the label pieces (IDs `0 … blank − 1`; some, such as
the language tags, are special), and may already carry added special tokens at and after `blank`:
the published card has `<pad>` at `blank` and `<blank>` at `blank + 1`. Core detokenizes every
emitted token, and `skip_special_tokens` drops only tokens the tokenizer marks special.

The tokenizer mode `nemotron_asr_streaming` (selected with `--tokenizer-mode nemotron_asr_streaming`)
loads the card's tokenizer and takes `blank` from `blank_token_id` in the `config.json` beside the
tokenizer (so `--tokenizer`, when given, must name the checkpoint, and an already-translated
configuration keeps `blank_token_id`). It requires the base vocabulary size to equal `blank`, and
every token the card already has at an ID of `blank` or more to be special and at most `blank + 5`.
It keeps those tokens and appends special tokens for the remaining IDs up to `blank + 5`, so that
IDs `blank … blank + 5` are the blank placeholder, park, stream chunk carrier, EOU, file carrier and
continuation, all special in both senses core's detokenizers use: flagged special in the tokenizer
backend (the fast incremental path) and listed in `all_special_ids` (the slow path). The published
card's `<blank>` is flagged special but not listed, so the mode lists it. It sets the token at `blank + 1` as the tokenizer's EOS (`eos_token_id =
blank + 1`), whatever its name. Core takes a request's primary EOS from the tokenizer, and a
persistent-state stream parks only on that EOS (any other stop ends the stream), so this binding is
what lets a stream resume after each unit. Loading fails if any of these conditions does not hold,
if any token lands at an unexpected ID, or if the tokenizer reports any other EOS.

## Lookahead selection

The model implements five encoder geometries, `80ms`, `160ms`, `320ms`, `560ms` and `1120ms`
(left context 56 frames, right context 0, 1, 3, 6 and 13). The card declares the lookaheads it was
trained for (0, 3, 6 and 13).

The launch options are flat `--hf-overrides` keys, which core applies to the served configuration:
`supported_num_lookahead_tokens`, `num_lookahead_tokens` and `experimental_native_burst`.

- **Served set.** `supported_num_lookahead_tokens` when the final configuration has it (from
  `--hf-overrides`, or stored by an already-translated configuration), else
  `card_num_lookahead_tokens`. Translation removes a top-level `supported_num_lookahead_tokens` a Hub
  card carries, so for a card only the override can set it. It replaces the card's set; every value must be one of the five right contexts, and the
  set must not be empty. A value outside the card's set (lookahead 1) is allowed but logged as a
  geometry the checkpoint does not declare.
- **Default lookahead.** `num_lookahead_tokens` must be present in the final configuration and in
  the served set. For a Hub card it can only come from `--hf-overrides`, since translation drops the
  card's own; an already-translated configuration may store it. It is the cadence for complete
  files and core's `/v1/realtime`; validation sets the configuration's nominal `att_context_right`
  to it.
- **Native burst.** `experimental_native_burst: true` requires the default lookahead to be 1; it
  does not restrict the served set.

## Post-override validation

The configuration hook runs twice during startup:
- `verify_and_update_model_config(model_config)` runs at the end of model configuration, after
  core has applied every override. It resolves the served set and the default lookahead (above),
  sets `att_context_right`, and checks: the default lookahead is present and served; every served
  value is one of the five right contexts; native burst; the
  control-token layout, `eos_token_id` and `vocab_size` equal the computed values;
  `language_locales`, when given, maps language codes to locales of that language in the locale
  table (a locale's language is the lowercase letters before the first `-` or uppercase letter,
  when there are two or three of them; `auto` has none); and
  `max_model_len ≥ 1 + realtime_max_tokens` (the smallest context in which one stream unit fits). It repeats the parser's schema check, and checks generation defaults in two places, because core
reads them in two places. Request EOS and stop tokens come from
`model_config.try_get_generation_config()`, which loads the checkpoint's `generation_config.json`
under both `--generation-config auto` and `vllm` (a directory selects another file): any `eos_token_id` there must be the park token alone, because
core turns any other EOS into a stop token that ends a stream at its first park; the remedy is
`--generation-config <dir>` naming a directory whose generation configuration has no other EOS.
Default sampling limits come from `model_config.get_diff_sampling_param()`, which also merges
`--override-generation-config`: it must contain no `max_tokens`; the remedy is
`--generation-config vllm` and no token-limit override. It logs each language
  that complete-file transcription cannot map (several distinct rows, no locale named for the
  language itself, and no `language_locales` entry).
- `verify_and_update_config(vllm_config)` runs once while the engine configuration is resolved and
  refuses the engine settings below. Settings that core resolves only later are re-checked when the
  model state is constructed.

Both raise `ValueError` listing every violated condition.

## Engine settings

The persistent-state model needs one scheduler step at a time, one copy of each session's state
and a single device. The plugin refuses:
- asynchronous scheduling: an explicit `async_scheduling = True`. Unset is accepted:
  `verify_and_update_config` runs before core resolves the option and sees it unset, and core
  then resolves it to `False` for a model that declares persistent state (and itself raises on
  `True`). The model state re-checks the resolved value when it is constructed and refuses
  anything but `False`;
- prefix caching: `enable_prefix_caching = True` in the resolved configuration. An unset command-line
  option arrives already `False`: `EngineArgs` resolves it from the model, which reports prefix
  caching as unsupported when it declares persistent state. A configuration built directly with
  `True` is refused here (and by core's state validation);
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator and
  hierarchical sparse attention;
- any tensor, pipeline, data, decode-context or prefill-context parallel size other than 1 (a
  setting the build does not have counts as 1);
- an outer runner that is not eager: `enforce_eager` must be set explicitly, since core's default
  is `False`. With it, core normalizes `cudagraph_mode` to `NONE` before its persistent-state
  check, so core captures no graphs of its own; the model runs its own encoder and decoder graphs,
  where its configuration selects graphed execution, inside the eager outer step (Execution
  memory).

The README's serve command shows the required flags.

## Execution memory

The model's own execution (static encoder execution; when its configuration selects them, the
decoder graphs per served geometry and batch tier and the CHUNK buckets) retains device memory,
and its largest transaction needs transient memory beyond that. Core sizes the KV cache and the
state pool from the memory profile of its profiling run, and binds state slots only after it
allocates them (`initialize_state_slots`). The RFC 1 PRs have no hook dedicated to model execution
memory, so the model prepares its execution on core's profiling step:

- **One owner.** `prepare_step` records that the step is a profiling dummy step; the model's
  forward on that step is the only place that prepares execution. On the first profiling forward
  it warms and captures the execution inventory, checks the inventory is complete (every served
  geometry and batch tier), and then, with every captured graph retained, runs the profile
  transaction at its maximum shape so the measurement includes the transient peak. It returns
  the dummy output core expects. The earlier forward-time profile of an unprepared encoder is
  retired: it cannot run once the inventory is sealed.
- **No state pool needed.** Capture and profiling use model-owned scratch state built from the
  model's `PersistentStateSpec`: captured graphs copy state into and out of their own static
  buffers, and the gather from and scatter to the pool run outside any graph, so preparation needs
  neither the pool nor its addresses.
- **Once per worker.** A later profiling step checks the inventory and returns. A failure marks the
  worker failed, discards partial execution state and raises, so startup fails; the next attempt
  is a fresh worker.
- **Serving guard.** Before a serving step uses graphed execution, the model checks that every
  expected graph entry is captured in its graph wrapper (the wrapper's entry holds a graph), not
only listed in the model's inventory and not merely that preparation ran, and raises otherwise rather
  than capturing during serving, where the memory is not budgeted. This also covers a core that
  clears graph wrappers after its profiling run: at these heads core clears registered graph
  wrappers only when it profiles its own graph memory, which it skips under `enforce_eager` (mode
  `NONE` and no encoder graph manager).
- **Budget.** Core measures the profile only when it computes the budget itself. When the operator
  sets `kv_cache_memory_bytes`, core still runs the profiling step, so preparation happens, but
  takes the given budget as is; the operator then leaves room for the model's execution memory.
  Core's elastic expert-parallel scale-up launch (an environment setting) skips profiling and is
  not supported: the plugin refuses any data-parallel size other than 1, and the serving guard
  stops a step that reaches serving unprepared. Core's no-KV-cache path also skips profiling, and
  cannot apply, because the model declares a state layer. The pool is `max_num_seqs + 1` pages unless the
  operator overrides the block count.
- **Graph runtime.** Graph capture uses core's graph-capture context, forward context and the
  platform's static graph wrapper. At these heads core has no capture-enabled switch (removed with
  Model Runner V1), so the model's graph runtime has none; capture legality is the model's own
  ordering above.

## Model-state interface

Every `ModelState` method the model state overrides keeps the base signature at the target heads,
because core's model runner calls them positionally; a test binds each override against the base
signature. In particular the profiling path reaches the forward through
`dummy_inputs_embeds(num_tokens, input_ids=None)`. The model state returns its pre-allocated
embeddings buffer sliced to `num_tokens` in both cases: when core passes randomized `input_ids`
the contents stay unused, because the model's dummy forward reads no embedding values and returns
the dummy output.

## State-slot access

- `prepare_step` reads only the scheduler output, including its state-slot handles; the bindings
  give no row access there.
- The model binds rows inside the step, after core begins the step and activates its requests:
  `bind_rows` for the step's ordered requests and their handles, then
  `StateSlotRows.component_pools` and `slot_indices_cpu`.
- Before committing state, the model re-reads `component_pools`, which core refuses outside the live
  step; that replaces the removed current-step check.
- The model's projection of pools into named state tensors takes the component pools and the
  model's spec, not members that the RFC 1 PRs removed (the rows' spec and views, and handle lookup
  by request).
- The persistent-state spec is layout only: `PersistentStateSpec(descriptors=..., page_size_bytes=...,
  block_size=1)`, with no state name.

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
| `vllm.v1.kv_cache_interface` (state layout) | `PersistentStateSpec`, `PersistentStateDescriptor` |
| `vllm.v1.worker.gpu.sample.output` | `SamplerOutput` |
| `vllm.distributed.parallel_state` | `graph_capture` |
| `vllm.forward_context` | `BatchDescriptor`, `set_forward_context` |
| `vllm.platforms` | `current_platform` (its static graph wrapper class) |
| `vllm.utils.import_utils` | `resolve_obj_by_qualname` |
| `vllm.utils.torch_utils` | `direct_register_custom_op` |
| `vllm.model_executor.determinism.batch_invariant` | the batch-invariant matmul used by the batch-invariant mode |

The graph-runtime rows are imported when graphed execution is configured, and the batch-invariant
row when that mode is; the import-time check covers the rest. vLLM's model-registration, config
parser, tokenizer and renderer registries, model configuration hooks (`VerifyAndUpdateConfig`,
`MODELS_CONFIG_MAP`), speech-to-text configuration, multimodal processing, `SupportsTranscription`,
logging and Hub file helpers, which out-of-tree models use, are outside this table. Adding a dependency means
adding it here and to the import-time check.

## Implementation notes

- **Renderer.** In the target engine a tokenizer mode also selects a renderer; `register()`
  registers the standard Hugging Face renderer for `nemotron_asr_streaming`.
- **One served-set source.** `lookahead.served_lookaheads` is the served set for the model's warmed
  geometries, the session factory and `buffer_realtime_audio`.
- **Precision.** The supported deployment runs at float32 (`--dtype float32`); the model's
  precision check at load refuses any other dtype.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| How the card becomes a served config | A registered config parser selected by `--config-format nemotron_asr_streaming` | `AutoConfig` registration for `nemotron3_5_asr`; `--hf-overrides` alone | Registering the card's `model_type` would claim the offline plugin's configuration; overrides apply after the configuration is built, too late for translation |
| Where launch options are resolved | Post-override validation of the final configuration; the parser reads only the checkpoint | Reading `hf_overrides` inside the parser; a plugin-specific environment variable | Core gives the parser only flat overrides, applies dictionary overrides after it, and also calls it with none when deriving a generation configuration |
| Control tokens in the tokenizer | A registered tokenizer mode that appends them as special tokens, with park as the tokenizer's EOS | Publishing a modified tokenizer; leaving them unknown | Core detokenizes every token; unknown IDs break detokenization and unmarked ones appear in text; the card must stay usable unmodified |
| Context ceiling | Translation sets `max_position_embeddings` to 2^24 | Requiring `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | The model has no positional limit; long files need a context far above the card's 32,768 |
| Where model execution memory is prepared | The model's forward on core's first profiling dummy step (flagged by `prepare_step`), before the state pool, inside core's memory profile | `initialize_state_slots` or first serving step: graphs created after core sized the KV cache and state pool, so their memory is unbudgeted. A new core hook: core adds no interface the RFC 1 PRs do not need (Pete, 2026-11-12) | Only the profiling step is both budgeted and available without a core change |
| Override of the served set | Replaces the card's set; values limited to the five geometries | Union with the card's set; any value | Replacement lets a deployment narrow the warmed set; values without a geometry cannot run |
| Load-time checks | Client calls in `register()`; adapter symbols at model import | All checks in `register()` | `register()` and the session factory do not import the model class modules, so loading the plugin stays cheap; core's speech-to-text serving imports the model class in the API server when it starts, so importing it initializes no device and loads no weights |
| `generate` task | Not advertised (model-state override) | Advertise for compatibility | The model emits transcript labels, not text generation |

## Open Questions & Future Decisions

### Deferred
1. A version gate for clean vLLM releases once RFC 1's stream admission ships; until then the client-path
   check is the gate.

## References

- vLLM config parsers: `vllm/transformers_utils/config.py` (`register_config_parser`, the parser
  call and override application), `vllm/transformers_utils/config_parser_base.py`.
- Model configuration hooks: `vllm/model_executor/models/config.py` (`MODELS_CONFIG_MAP`).
- Supported tasks: `vllm/v1/worker/gpu/model_states/interface.py`.
- Plugin loading: `vllm/plugins/__init__.py`.
- Card translation: `nemotron_asr/configuration_nemotron_asr.py`; geometries:
  `nemotron_asr/manifests.py`.
