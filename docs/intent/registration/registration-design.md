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
share a row (aliases such as `en` and `en-US`) and some rows may be unused; a non-empty
`card_num_lookahead_tokens` drawn from the five implemented right contexts; and, when present, a
`card_default_num_lookahead_tokens` that is one of `card_num_lookahead_tokens`. Post-override
validation repeats this check on the final configuration, since overrides can change any of these
fields.

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
- requires `encoder_config`, `vocab_size == blank_token_id + 1` (labels are `0 … blank-1`),
  `encoder_config.supported_num_lookahead_tokens` and `encoder_config.default_num_lookahead_tokens`;
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
- records the card's lookahead set as `card_num_lookahead_tokens` and the card's default lookahead
  (`encoder_config.default_num_lookahead_tokens`; 3, which is 320 ms, on the published card) as
  `card_default_num_lookahead_tokens`; drops `default_prompt_id` (the default locale is the rule in
  the `session` and `transcription` designs) and any top-level `default_num_lookahead_tokens`,
  `num_lookahead_tokens` and `supported_num_lookahead_tokens` the card carries, so that only the
  card's encoder configuration and `--hf-overrides` choose the geometry; the copy of the default
  in `processor_config.json` is not read;
- declares the default segment arms (Segment arms, below): `encoder_execution_arm =
  "eager-graphed"` and `decode_dispatch_arm = "dense-graphed"`.

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

## Server geometry

The model implements five encoder geometries, `80ms`, `160ms`, `320ms`, `560ms` and `1120ms`
(left context 56 frames, right context 0, 1, 3, 6 and 13). The card declares the lookaheads it was
trained for (0, 3, 6 and 13) and a default among them (3). A server serves exactly one geometry,
fixed at startup, as a model loaded in Hugging Face Transformers is fixed by its
`num_lookahead_tokens`; every stream and every whole file runs at it.

The launch options are flat `--hf-overrides` keys, which core applies to the served configuration:
`num_lookahead_tokens` and `experimental_native_burst`.

- **Resolution.** The server's lookahead is `num_lookahead_tokens` when the final configuration has
  it (from `--hf-overrides`, or stored by an already-translated configuration), else
  `card_default_num_lookahead_tokens`. One function, `lookahead.served_lookahead(hf_config)`, reads
  those two fields in that order and returns the right context; it raises when neither is present
  or the value is not one of the five implemented right contexts. Every consumer of the geometry
  calls it on the served configuration (One geometry source, below). Translation requires the
  card's default, so a missing lookahead (both fields absent) can arise only from an
  already-translated configuration that stores neither.
- **Implemented, not declared.** Any of the five implemented right contexts is accepted, including
  lookahead 1 (160 ms), which the card does not declare; a value outside the card's set is logged
  as a geometry the checkpoint does not declare. The card's set therefore limits nothing; it is
  recorded for that log line and for the schema check.
- **No served set.** A server serves one geometry, and `supported_num_lookahead_tokens` names a set
  of them. A final configuration that has it is refused, with the remedy to remove it and
  select the geometry with `num_lookahead_tokens`. Translation removes the key from a Hub card, so
  only `--hf-overrides` or an already-translated configuration can supply it.
- **Nominal context.** Validation sets the configuration's `att_context_right` to the server's
  lookahead.
- **Native burst.** `experimental_native_burst: true` requires the server's lookahead to be 1.
- **Transaction graphs.** The geometry rule of the transaction-graph option (80 or 160 ms) is one of
  that option's configuration checks, which validation calls (Post-override validation); this
  design does not restate it.

## Post-override validation

The configuration hook runs twice during startup:
- `verify_and_update_model_config(model_config)` runs at the end of model configuration, after
  core has applied every override. It resolves the server's geometry (above), sets
  `att_context_right`, and checks: a lookahead is present and is one of the five right contexts;
  no `supported_num_lookahead_tokens` is present; native burst; the
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
- `verify_and_update_config(vllm_config)` runs once while the engine configuration is resolved,
  after `verify_and_update_model_config`, and refuses the engine settings below. It also calls the
  transaction-graph component's configuration checks (`transaction_graph_config_errors`; the
  `transaction-graph` design, Configuration checks and Where the checks run) on the engine
  configuration, whose geometry is the one resolved above, and includes every condition they
  return. Those checks own every rule of the option, among them its geometry and its refusal of
  native burst; this design does not duplicate them. Settings that core resolves only later are
  re-checked, with the option's checks, when the model state is constructed.

Both raise `ValueError` listing every violated condition, so a misconfiguration fails in the API
server before any worker starts.

Validation never changes `encoder_execution_arm` or `decode_dispatch_arm`: a value given in
`--hf-overrides` is the served value.

## Engine settings

The persistent-state model needs one copy of each session's state and a single device. It
accepts asynchronous scheduling in every setting (`async_scheduling` `True`, `False` or unset):
core's per-request barrier for state models schedules no state request while the output of its
previous step is still in flight, so each stream still advances one step at a time. The worker
also stays one step at a time for the model under either setting: it runs a step's
`postprocess_state` before it executes the next step, and there the model waits for the step's
commit event, which follows every device read of the step's staged inputs and state, then reads
the step's commit status. A request that leaves the engine (finish, abort, preemption, error)
reaches the model state in the next step's finished request IDs, which core delivers even on a
step with no forward work, so cleanup never races a step that still reads the request's
buffers. Native burst, whose sampler reads ahead of the scheduler, is refused on the state layer
until its port there is specified (Future roadmap 9), so it meets asynchronous scheduling nowhere
yet. The plugin refuses:
- prefix caching: `enable_prefix_caching = True` in the resolved configuration. An unset command-line
  option arrives already `False`: `EngineArgs` resolves it from the model, which reports prefix
  caching as unsupported when it declares persistent state. A configuration built directly with
  `True` is refused here (and by core's state validation);
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator,
  hierarchical sparse attention, fault tolerance (`parallel_config.enable_fault_tolerance`, which
  keeps core's JIT kernel warm-up on under `enforce_eager`; that warm-up runs synthetic real steps
  without state-slot handles; `--fault-tolerance-config` turns it on, and the refusal names that
  cause) and elastic expert parallelism (`parallel_config.enable_elastic_ep`, whose reconfiguration
  reruns the worker's warmup and so would call the serving capture again over live state pools;
  A second serving capture, below). Moving execution preparation into core's profiling step does
  not make sleep mode or the cumem allocator usable: both stay refused, so there is no wake-up
  re-preparation;
- any tensor, pipeline, data, decode-context or prefill-context parallel size other than 1 (a
  setting the build does not have counts as 1);
- a runner configuration other than the two of Runner configurations, below.

The model state's construction re-checks these rules against the final engine configuration, where
an unset `cudagraph_mode` or compilation mode is not accepted.

Native burst's own configuration check, run at model construction, applies the same rule for a
setting the build does not define: a parallel size counts as 1 and expert parallelism as off.

### Runner configurations

The model's forward plans each step on the host. It reads device values back, builds each step's
row plan as CPU tensors and branches on them, stages inputs into device tensors it allocates per
step, records a commit event, and replays its own graphs. A runner graph of core's records a
forward once and replays it without running its Python, and a full graph replays without calling
the forward at all, so none of this host work would run on replay; and CUDA does not allow the
model to capture its own graphs inside a capture of core's. The model therefore owns every graph
it runs, and core captures no graph over its forward in any accepted configuration.

The model state states this to core in the standard way: its `get_additional_cg_support` returns
`AttentionCGSupport.NEVER`, with the name `nemotron_asr_streaming model state (captures its own
graphs)`. Core's mode resolution takes the least capable of the model's backends, so it reduces
every graphed `cudagraph_mode` it can reduce to NONE and logs the reduction; FULL, whose
mixed-batch full graphs it cannot reduce, it refuses. (A piecewise mode under compilation mode
NONE is already reduced when the engine configuration is built, because no compiled region
exists to split.) The model class carries no
`@support_torch_compile`, so no compilation mode compiles any of it except `STOCK_TORCH_COMPILE`,
under which the runner compiles the whole model with `fullgraph=True`.

Two runner configurations are accepted:
- `--enforce-eager`. Core normalizes `cudagraph_mode` to `NONE` and runs none of the three
  graph-capture hooks; the model captures its own segment
  graphs on the first profiling dummy step (Execution memory). Every segment arm is accepted.
- **The hook configuration**: `enforce_eager` off, on CUDA. Core captures every graph the model
  runs through the model-state capture hook, and none is captured on the profiling dummy step.
  Its conditions, each with the remedy the refusal names:
  - `cudagraph_mode` NONE, PIECEWISE, FULL_DECODE_ONLY or FULL_AND_PIECEWISE, or unset before
    core resolves it; each resolves to NONE for this model. FULL is refused, because core
    cannot reduce it and would refuse it itself only after the weights load (remedy
    `-cc.cudagraph_mode none`, or leave the option unset).
  - not PIECEWISE while breakable CUDA graphs are on (core's `is_breakable_cudagraph_enabled`:
    `use_breakable_cudagraph`, or `VLLM_USE_BREAKABLE_CUDAGRAPH` while that is unresolved).
    Breakable graphs give core a piecewise capture path without compilation, so PIECEWISE would
    survive resolution and core would capture the forward as one breakable segment (remedy
    `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, or `-cc.cudagraph_mode none`). Enabling breakable graphs
    sets the compilation mode to NONE, and FULL_AND_PIECEWISE then resolves to NONE, so it is
    accepted. An unset mode whose default under the optimization level is PIECEWISE is refused
    only at model-state construction, where the default is known.
  - compilation mode NONE, `DYNAMO_TRACE_ONCE` or `VLLM_COMPILE`, or unset before core resolves
    it. `STOCK_TORCH_COMPILE` is refused while breakable CUDA graphs are off (remedy `-cc.mode
    none`, or leave the option unset); with breakable graphs on, core sets the compilation mode
    to NONE when it enables them, so the requested mode compiles nothing and is accepted.
  - a CUDA platform (remedy `--enforce-eager`).
  - at least one graph the hook captures: the `eager-graphed` encoder, the `dense-graphed`
    decoder, or transaction graphs (remedy: select one of them, or `--enforce-eager`).
  - no graph it does not capture: the compiled `dense-graphed` encoder arm and CHUNK-bucket
    graphs (`chunk_bucket_graph_populations`) are refused with the remedy `--enforce-eager`.
  The eager arms alone capture nothing, so they need `--enforce-eager`.

`enforce_eager` is not core's default, so the hook configuration is the configuration a serve
command without runner options selects. A refusal names each unmet condition with its remedy;
every remedy that names `--enforce-eager` also names removing `transaction_graph_populations`
when it is set (the option is refused under `--enforce-eager`).

The configuration hooks and the model's and model state's constructions all judge the modes
through the one rule above; core resolves the runner mode only when it initializes the KV cache,
after the model state exists. Two guards hold the result:
- the model state's `initialize_kv_cache`, which core calls after it resolves the mode and before
  any capture, in the profiling and the serving KV-cache initialization alike (also under a
  budget fixed through `--kv-cache-memory-bytes`), checks that the resolved `cudagraph_mode` is
  NONE and fails startup naming the resolved mode otherwise;
- every path by which core would capture the forward passes through the model state before the
  forward runs, and the model state refuses each: `prepare_dummy_inputs` (core's capture inputs)
  and `prepare_attn` for a capture or a FULL step raise, naming the mode and stating that the model
  supports no runner graph.
On its first `initialize_kv_cache` call in the process, the model state logs once that core's
runner captures no graph over the model's forward (resolved `cudagraph_mode` NONE, no runner-graph
support reported) and that the model captures its own graphs through the hook. The requested mode
appears in core's own log of its reduction. `prepare_execution_memory` and the profiling dummy
step run before core resolves the mode, so neither depends on the resolved mode.

A graphed mode that reduces to NONE is not identical to an explicit NONE at startup: the engine
configuration still sizes core's graph capture sizes from the requested mode, and core's profiling
KV-cache initialization, with it the temporary state pools of the profiling capture, holds
`min(max_num_seqs, the largest capture size)` pages where an explicit NONE holds one. The
model's graphs and its serving pools do not change; the profiling capture's transient memory
does (Future roadmap 3).

### Qualification of the runner configurations

Accepting a configuration and qualifying it are separate. Acceptance is decided at startup by the
rule above. Qualification is a GPU tier on an A40 at every chunk size the model configuration
allows (80, 160, 320, 560 and 1120 ms; one per server, fixed at startup), on the realtime workload
(the first 200 LibriSpeech test-clean utterances through `/v1/realtime`, 32 concurrent sessions,
paced at real time), compared utterance by utterance with `--enforce-eager` under synchronous
scheduling:
- each accepted runner configuration produces the same final transcripts. Every accepted
  `cudagraph_mode` and compilation mode resolves to the same runner (NONE, nothing compiled) and
  the same model-captured graphs, which the startup record shows per mode; the transcript tier
  runs the two distinct requests, explicit NONE with compilation off and a serve command without
  runner options;
- asynchronous scheduling on produces the same final transcripts;
- under `VLLM_BATCH_INVARIANT=1`, a session's final transcript does not depend on which other
  sessions share its batches: the workload at 32 concurrent sessions and at 8 gives the same
  transcripts. At float32 core enables no breakable graphs for batch invariance (it does so only
  for bf16), so the runner conditions are unchanged under it.
A configuration that fails its tier is refused with its reason and remedy, and the failure is a
Future roadmap item, until a fix passes.

### Core hooks

The plugin needs #67583's model-state hooks in every configuration: `prepare_execution_memory`,
`needs_cudagraph_capture`, `capture_cudagraphs`, `clear_cudagraphs`, `post_warmup` and `shutdown`
(`engine_flags.MODEL_STATE_GRAPH_HOOKS`). `verify_and_update_config` refuses a core whose
`ModelState` does not define each of them, naming every missing hook and #67583, and the
model-adapter dependency check (below) lists them, so a worker on such a core also fails before
any weights load. Under `--enforce-eager` core still calls `prepare_execution_memory`,
`post_warmup` and `shutdown`.

### README serve command

The README's serve command is the default configuration: the default segment arms and the
checkpoint's default lookahead (3, which is 320 ms, on the published card), so it sets no `num_lookahead_tokens` and no
`supported_num_lookahead_tokens`. It shows `--dtype float32`, `--config-format
nemotron_asr_streaming`, `--tokenizer-mode nemotron_asr_streaming`, one of the two runner
configurations, chosen from those that acceptance tier 2 has met with the default arms (until the
hook configuration meets it, `--enforce-eager`) with the runner and scheduling flags tier 2 was met
with (`--no-async-scheduling` until a tier is met with asynchronous scheduling, which core now
resolves on when the option is unset), an explicit `--max-num-seqs` that tier 2 verified
to fit the A40 at 320 ms in that runner configuration, and `--max-model-len 77000`, which holds a 600 s
file at 320 ms (core's default decode limit; at most 76,918 tokens: 1,875 whole chunks of 5,120
samples, the tail chunk, at most 41 tokens per chunk, and the first and last tokens). Beside it the README links the user manual's Serving configurations section (`docs/manual.md`), which states how
`num_lookahead_tokens` selects one of the other four geometries and how `--max-model-len` scales
with it: by the same bound, `1 + (floor(N / s) + 1) × Q + 1` tokens for `N` samples with
`s = 1280 × (r + 1)` and `Q = 10 × (r + 1) + 1` at lookahead `r`, a 600 s file needs at most
82,513 tokens at lookahead 0 (80 ms), 78,773 at 1 (160 ms), 76,918 at 3, 76,114 at 6 and 75,578
at 13, so 77,000 covers 3, 6 and 13, and an override to 0 or 1 needs a correspondingly larger
`--max-model-len` (at least 82,513 or 78,773 for 600 s files); that encoder graph memory grows faster than linearly with `--max-num-seqs`; the
`--hf-overrides` that select the eager arms and the `--enforce-eager` they need; and that a fixed
`--kv-cache-memory-bytes` must leave room for the segment graphs. The transaction-graph section of
the user manual belongs to that component. The README carries the overview, the build definitions, this serve
command and a performance summary; acceptance claims and their figures are in `docs/evidence.md` (REG-DOC-002, REG-DOC-004).

Until the tier 2 measurement verifies a value, the serve command may carry an interim
`--max-num-seqs` taken from an unpublished A40 measurement (64 at 320 ms under `--enforce-eager`),
marked in the README as interim and unpublished, as REG-DOC-002 marks evidence that has not met its
tier.

## Segment arms

The encoder and the RNN-T decoder each have execution arms, selected by `encoder_execution_arm`
and `decode_dispatch_arm` in the served configuration:
- encoder: `eager`, `compiled-static` (compiled, warmed, not graphed), `dense-graphed` (compiled
  and graphed) and `eager-graphed` (the uncompiled transition captured at every exact CHUNK count
  from 1 to `max_num_seqs`); every arm runs the first pointwise convolution with batch and frames
  folded into one axis;
- decoder: `dense-eager` and `dense-graphed` (captured at padded population tiers), and the
  `compact-eager` dispatch.

Translation declares the default arms, `eager-graphed` and `dense-graphed`, the fastest pair
measured with equal transcripts (HLD, Approach section 5). Core applies `--hf-overrides` after the
parser, so an override of either key selects another arm, and validation keeps it. The eager arms
(`eager` and `dense-eager`) are the reference the graphed arms are tested against; they capture
nothing and so need `--enforce-eager` (Engine settings). An already-translated configuration is
used as it stands: an absent `encoder_execution_arm` is the configuration class's `eager`.

**Segment graphs** are the graphs of the graphed arms: the encoder at every CHUNK count from 1 to
`max_num_seqs` and the decoder at each tier, both for the server's one geometry, and under
`--enforce-eager` any configured CHUNK-bucket graphs. Graphed execution is configured exactly when
`decode_dispatch_arm` is `dense-graphed`, `encoder_execution_arm` is `dense-graphed` or
`eager-graphed`, or `chunk_bucket_graph_populations` is non-empty.

## Execution memory

The model's execution retains device memory (long-lived working buffers and the segment graphs),
and its largest transaction needs transient memory beyond that. Core sizes the KV cache and the
state pool from the memory profile of its profiling run, and binds state slots only after it
allocates them (`initialize_state_slots`). #67583 divides model execution memory between two
hooks, and the model keeps them apart: long-lived working memory outside any graph belongs to
`prepare_execution_memory`, which every profiled budget counts; graphs captured through
`capture_cudagraphs` are counted only by core's graph-memory estimate. Under `--enforce-eager`,
where core runs no capture hook, the model captures its segment graphs on core's profiling dummy
step, inside the profiled budget.

The model prepares in three places, by runner configuration:

| Where | `--enforce-eager` | Hook configuration |
| --- | --- | --- |
| `prepare_execution_memory()` | Allocates long-lived working memory and warms uncaptured execution (static encoder execution, compilation of a compiled arm). Captures nothing. | The same. |
| First profiling dummy step | Captures the segment graphs, verifies the inventory, then, with every graph retained, runs the profile transaction at its maximum shape. | Verifies the uncaptured inventory and runs the profile transaction once at its maximum shape over the arms' uncaptured paths (the `eager-graphed` binding's uncaptured transition and the dense decoder's uncaptured decode, at that shape's row count). Captures nothing. |
| `capture_cudagraphs()` | Not called. | Captures the segment graphs (below). |

The profile transaction's maximum shape is `max_num_seqs` rows at the server's geometry and each
configured CHUNK-bucket graph population; transaction graphs add no shape to it. It sizes the option-off transaction's transient use and its use of core's
workspace, which core locks after the serving capture.

- **One owner per stage.** `prepare_step` records whether the step is a profiling dummy step; the
  model's forward on that step is the only place that runs the profile transaction, and under
  `--enforce-eager` the only place that captures. It returns the dummy output core expects. The
  forward runs no forward-time profile of an unprepared encoder: none can run once the inventory
  is sealed.
- **Dummy output.** Core indexes the profiling step's forward output by its logits indices, computes
  logits and runs the sampler on it. The dummy output matches the input embeddings' token count,
  width, dtype and device, and its carrier column holds a valid decision ID (zeros), so dummy sampling
  needs no state; dummy sampling, including the native-burst sampler wrapper, reads no state slots.
- **Repeated profiling steps.** Core runs one profiling run per worker, but its kernel warm-up may run
  further profiling dummy steps after the state pool exists (FlashInfer autotuning on recent GPUs),
  with real token IDs and attention metadata, possibly several times. Each is handled as a later
  profiling step: the inventory is checked (under the hook configuration, segment graphs not yet
  captured by the serving capture are accepted as pending) and the dummy output returned, with no
  capture and no pool access.
- **Other dummy steps.** Core skips its state-slot step bookkeeping on every dummy step, and calls
  `prepare_inputs` after `prepare_step` returned early, so on a dummy step the model reads no handles,
  no rows and nothing left from an earlier real step.
- **No state pool needed.** Preparation, capture and profiling use model-owned scratch state built
  from the model's `PersistentStateSpec`: segment graphs copy state into and out of their own
  static buffers, and the gather from and scatter to the pool run outside any segment graph, so
  neither preparation, the profile transaction nor segment-graph capture reads the state-slot
  bindings, the pool or its addresses, in either runner configuration. The one exception is the
  transaction-graph cells, which reach the pools inside `capture_cudagraphs` only through
  `StateSlotBindings.startup_pool_access`, as the `transaction-graph` design authorizes.
- **Once per worker.** A failure of preparation marks the worker failed, discards partial execution
  state and raises, so startup fails; the next attempt is a fresh worker.
- **Readiness on every real step.** The model refuses any real step that runs model execution unless
  preparation succeeded in this worker, whatever the execution arm; a cleanup-only step (no scheduled
  rows) is allowed. It never prepares lazily during serving. Normal startup cannot admit a request
  before preparation, because core profiles and initializes caches before it constructs the scheduler.
- **Serving guard for graphs.** Before a serving step uses graphed execution, the model checks that every
  expected segment-graph entry is captured in its graph wrapper (the wrapper's entry holds a graph), not
  only listed in the model's inventory and not merely that preparation ran, and raises otherwise rather
  than capturing during serving, where the memory is not budgeted. `post_warmup` makes the same check
  once before the worker serves (below). A transaction-graph cell without its graph follows the
  `transaction-graph` design's own rules (TXG-COV-006, TXG-EXEC-013), not this guard.
- **Budget.** Core measures the profile only when it computes the budget itself. When the operator
  sets `kv_cache_memory_bytes`, core still runs the profiling step, so preparation happens, but
  takes the given budget as is; the operator then leaves room for the model's execution memory and,
  under the hook configuration, for its segment graphs, which core does not charge then.
  Core's elastic expert-parallel scale-up launch skips profiling and is not supported: the plugin
  refuses elastic expert parallelism and any data-parallel size other than 1, and the serving
  guard stops a step that reaches serving unprepared. Core's no-KV-cache path also skips profiling,
  and cannot apply, because the model declares a state layer. The pool is `max_num_seqs + 1` pages
  unless the operator overrides the block count.
- **Graph runtime.** Graph capture uses core's graph-capture context, forward context and the
  platform's static graph wrapper. Under the hook configuration every capture runs on the `stream`
  core passes, inside `vllm.compilation.cudagraph_pool.capture_pool` with the `graph_pool` core
  passes; the model creates no stream and no graph pool of its own. At the target heads core has
  no capture-enabled switch, so the model's graph runtime has none;
  capture legality is the model's own ordering above.
- **Platforms without static graphs.** Where vLLM's platform reports no static graph mode
  (`support_static_graph_mode()`), or a graph entry's device is not the platform's device type,
  the graphed arms keep their inventory, staging, readiness checks and serving path. Each entry's
  wrapper is a stand-in that holds a non-device graph from construction and runs the entry's
  runnable eagerly on every call; nothing is captured, pooled or synchronized. Outputs equal the
  eager arms'. The model logs one INFO line naming each graphed arm the stand-in serves, so a run
  that asked for graphed arms shows that it executes them eagerly. Only `--enforce-eager` reaches
  this case, because the hook configuration requires a CUDA platform.
- **Memory contributors.** The model's `PersistentStateSpec` lists in `memory_contributors` each
  configured option that enlarges its execution memory or graphs: `max_num_seqs`,
  `encoder_execution_arm` and `decode_dispatch_arm` when graphed execution is configured,
  `chunk_bucket_graph_populations` when it is non-empty, and `transaction_graph_populations` when
  the option is on. Whether transaction graphs are on is decided only by the `transaction-graph`
  design's `transaction_graphs_on` (TXG-CFG-014), which every check, hook and dependency gate here
  calls, so a malformed value such as `{}` is on, and refused, everywhere. Core adds a note naming them to a failure of its temporary profiling pool, of a
  capture and of its capacity planning.

### Segment-graph capture on the hook

Under the hook configuration the segment graphs follow #67583's capture lifecycle. The
`transaction-graph` design owns the transaction-graph part of each hook; the segment-graph part
below runs first in every capture call, after the transaction-graph entry checks (the
`transaction-graph` design, Rules of the capture call), which run at the call's entry while
transaction graphs are on. A call that an entry check refuses captures no
segment graph.

| Hook | Segment-graph behaviour |
| --- | --- |
| `needs_cudagraph_capture()` | True when the configuration selects a hook-captured graph (Engine settings) and the platform is CUDA. Configuration only, the same value for the model state's lifetime. |
| `capture_cudagraphs(profile_only=True)` | Runs only when core's graph-memory estimate runs, over core's temporary profiling state pools and into the estimate's throwaway pool. Captures the segment graphs for the server's geometry, then the transaction-graph part; with transaction graphs off, it ends with the charged margin. Logs the capture line. |
| `clear_cudagraphs()` | Releases every segment graph and its buffers, after the transaction-graph part, and, with transaction graphs off, the margin the profiling capture created (with them on, the transaction-graph part releases it); keeps the profiling figures as host integers. Core calls it also when the profiling capture raised or was never reached, so it tolerates nothing captured. Core also clears every registered static graph wrapper after the estimate, so no profiling entry survives either way. |
| `capture_cudagraphs(profile_only=False)` | The serving capture: rebuilds the segment graphs by the same procedure, into core's global graph pool, then the transaction-graph part; then the memory comparison. Logs the capture line. |
| `post_warmup()` | Before the worker serves: every expected segment-graph entry holds its graph; then the transaction-graph verification. The segment-graph check allocates nothing; neither part allocates device memory or replays a graph. Under `--enforce-eager` it makes the same segment-graph check. |
| `shutdown()` | Releases every segment graph and binding-owned buffer; tolerates a model state whose initialization did not complete. |

A serving capture is *attempted* once a serving call has passed every check at its entry (the
transaction-graph entry checks while transaction graphs are on, the checks below, and the
required-capture decision), whether or not the call then succeeds.

**Ordering.** A capture call of either kind while profiling segment graphs exist, or a profiling
capture after a serving capture was attempted, whether or not it succeeded, is a lifecycle defect
and raises at the call's entry, before capturing; while transaction graphs are on, the error names
`transaction_graph_populations`.

**A second serving capture.** A serving capture after a serving capture was attempted, whether or
not the attempt succeeded, raises at the call's entry, stating that recapture over live state
pools is not supported and that the recovery is an engine restart, and naming
`transaction_graph_populations` while transaction graphs are on. The refusal is keyed on the
attempt, not on segment graphs existing, because a failed attempt discards the graphs it captured.
Core reaches it only through an elastic expert-parallel reconfiguration, which the plugin refuses.

### Segment-graph memory

The capture rules are one procedure per capture call, the same with transaction graphs on or off;
with the option off the call holds only the segment graphs. The `transaction-graph` design
(Graph memory) states the procedure in full for a call that also holds cells; this section states
it for the option-off call, which does not depend on that component. The measurement, the charged
margin's allocation and retained charge, the required-capture predicate and the aggregate
comparison below have one implementation, owned here because the option-off call needs it; the
transaction-graph part calls it for its own spans (REG-MEM-027). With transaction graphs on, the
capture line's retained figure is the span of the call's segment-graph part.

- **Measurement.** A call's retained bytes are the drop in driver free memory
  (`torch.cuda.mem_get_info`) across it, each reading taken after synchronizing the capture stream
  and releasing the allocator's unused cached blocks; its capture time is wall time over the call.
  The measurement reads free memory only and resets no allocator peak statistics: under
  `--enforce-eager` the capture runs inside core's memory-profiling window, which reads them. (The
  transaction-graph procedure resets them in a capture hook call, outside that window.)
- **When core's estimate runs.** Core subtracts what the profiling capture retains, read when the
  hook returns. The profiling call ends with one **charged margin**, a device buffer held until
  `clear_cudagraphs`, whose retained charge `r` is measured around its allocation; it absorbs the
  allocator rounding by which a serving call can retain more than an identical profiling call. The
  margin is one per call, with the `transaction-graph` design's value (64 MiB until the first A40
  measurement sets it). After the serving capture, startup fails when the serving call's retained
  bytes exceed the profiling call's, which include `r`. The comparison is aggregate because core's
  charge is aggregate.
- **When core's estimate does not run.** At the serving capture the model decides from the
  worker's effective cache configuration whether a profiling capture was required: exactly when
  the effective `kv_cache_memory_bytes` is unset or 0, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`
  is not 0 and the extensible KV cache is off. Not required: one WARNING that the segment graphs
  were not charged to the budget, with the reason, the retained bytes and that the budget must
  leave room for them. Required but absent: startup fails at the call's entry, before any segment
  graph is captured, naming the configuration that should have run the estimate.
- **Capture line.** Each capture of the segment graphs logs one INFO line naming the runner
  configuration and capture (profiling dummy step under `--enforce-eager`; profiling or serving
  capture under the hook configuration), the arms, the geometry, `max_num_seqs`, the retained
  bytes and the capture time. Under `--enforce-eager`, and under the hook configuration when core
  runs its estimate, the line precedes core's capacity planning; otherwise it follows it. A later
  capacity failure is core's own, and the logged line is its diagnosis.
- **Capture failure.** A segment-graph capture that cannot allocate, or any other failure while
  capturing segment graphs, discards what that call or step captured, synchronizes and raises an
  error chained from the original, with the message below. Under the hook configuration the
  cleanup covers the whole call: a failure after the segment graphs were captured, in the
  transaction-graph part or in the aggregate memory comparison, also discards every segment graph
  and segment-graph buffer the call created and synchronizes the capture stream, and the original
  error then propagates unchanged with its own diagnostics (the `transaction-graph` design, Capture
  failure messages, for the transaction-graph part; the serving-over-profiling failure, below, for
  the aggregate comparison), not with this message. Entry-check refusals keep their own messages
  (Ordering, A second serving capture). The message of a segment-graph capture failure names the
  arms, the geometry, `max_num_seqs`, the graph whose capture
  failed (encoder CHUNK count, decoder tier, or CHUNK-bucket geometry and count), the bytes
  retained so far, the capture time so far, the budget setting and a complete remedy: a smaller
  `--max-num-seqs`; or `--hf-overrides` selecting `encoder_execution_arm` `eager` and
  `decode_dispatch_arm` `dense-eager` with `chunk_bucket_graph_populations` and
  `transaction_graph_populations` empty, together with `--enforce-eager`; under a fixed budget,
  also a smaller `--kv-cache-memory-bytes`; for a serving capture
  under a profiled budget, also a lower `--gpu-memory-utilization`. The serving-over-profiling
  failure names both call totals, `r` and the excess, with the same remedies. Whatever the
  failure, the propagated error holds no reference to the tensors or graphs the capture created:
  the cleanup clears the local variables of the finished frames in its traceback and in the
  tracebacks of its cause and context, keeping the messages and traceback lines, so that a
  retained error does not keep the discarded memory alive.

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
- During scheduled execution, the model binds rows inside the step, after core begins the step and
  activates its requests:
  `bind_rows` for the step's ordered requests and their handles, then
  `StateSlotRows.component_pools` and `slot_indices_cpu`. Transaction-graph capture reaches the
  pools only through the startup access the `transaction-graph` design governs (TXG-CAP-008).
- Before committing state, the model re-reads `component_pools`, which core refuses outside the live
  step, so a commit outside the live step fails before it writes.
- The model's projection of pools into named state tensors takes the component pools and the
  model's spec and reads no other member of the rows.
- The persistent-state spec is the layout and the memory contributors:
  `PersistentStateSpec(descriptors=..., page_size_bytes=..., block_size=1, memory_contributors=...)`,
  with no state name (Execution memory, Memory contributors).
- Core sends no token IDs for scheduled cached requests unless pipeline parallelism is on, which the
  plugin refuses. `prepare_step` therefore records each request's prompt when core delivers it as a
  new request (a session turn arrives that way, with its prompt from position zero) and reads a
  cached row's scheduled tokens from that record. A position past the recorded prompt holds a token
  the sampler produced on the device; the host does not read it (that would synchronize with the
  device), and no sampled token is a CHUNK carrier, so the row is planned as a non-CHUNK row. The
  record is dropped when the request finishes. For a file request this classifies only the
  scheduled token's origin: the file step rule (`transcription`, FILE-ROW-002) decides whether the
  row executes as CHUNK, before transaction classification.

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
| `vllm.config.compilation` | `CUDAGraphMode`, `CompilationMode` |
| `vllm.compilation.breakable_cudagraph` | `is_breakable_cudagraph_enabled` |
| `vllm.v1.attention.backend` | `AttentionCGSupport` |
| `vllm.v1.core.sched.output` | `NewRequestData`, `SchedulerOutput` |
| `vllm.v1.kv_cache_interface` | `KVCacheConfig`, `StateSlotHandle` |
| `vllm.v1.worker.gpu.input_batch` | `InputBatch` |
| `vllm.v1.worker.gpu.model_states.interface` | `ModelState`, with its hooks `prepare_execution_memory`, `needs_cudagraph_capture`, `capture_cudagraphs`, `clear_cudagraphs`, `post_warmup` and `shutdown`, and its `get_additional_cg_support` |
| `vllm.v1.worker.gpu.state_slots` | `StateSlotBindings`, `StateSlotRows` |
| `vllm.v1.worker.gpu.states` | `RequestState` |
| `vllm.v1.worker.utils` | `AttentionGroup` |
| `vllm.model_executor.layers.state` | `PersistentStateLayerBase` |
| `vllm.multimodal` | `MULTIMODAL_REGISTRY` |
| `vllm.v1.attention.backends.utils` | `NULL_BLOCK_ID` |
| `vllm.v1.kv_cache_interface` (state layout) | `PersistentStateSpec` (with its `memory_contributors` argument), `PersistentStateDescriptor` |
| `vllm.v1.worker.gpu.sample.output` | `SamplerOutput` |
| `vllm.distributed.parallel_state` | `graph_capture` |
| `vllm.forward_context` | `BatchDescriptor`, `set_forward_context` |
| `vllm.platforms` | `current_platform` (its static graph wrapper class) |
| `vllm.utils.import_utils` | `resolve_obj_by_qualname` |
| `vllm.compilation.cudagraph_pool` | `capture_pool` |
| `vllm.envs` | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` |
| `vllm.v1.worker.gpu.state_slots` (transaction graphs) | `StateSlotBindings.startup_pool_access`, `StateSlotBindings.pool_layout` |
| `vllm.utils.torch_utils` | `direct_register_custom_op` |
| `vllm.model_executor.determinism.batch_invariant` | the batch-invariant matmul used by the batch-invariant mode |

Three groups of rows are conditional. The graph-runtime rows (`graph_capture`, `BatchDescriptor`,
`set_forward_context`, `current_platform`, `resolve_obj_by_qualname`, `capture_pool` and
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`) are imported when graphed execution is configured; the
transaction-graph rows while transaction graphs are on (TXG-CFG-014; the `transaction-graph`
design, Permitted dependencies, lists their use); the batch-invariant row when that mode is. The
import-time check covers the rest, the six hooks of `ModelState` and its `get_additional_cg_support` included. vLLM's model-registration, config
parser, tokenizer and renderer registries, model configuration hooks (`VerifyAndUpdateConfig`,
`MODELS_CONFIG_MAP`), speech-to-text configuration, multimodal processing, `SupportsTranscription`,
logging and Hub file helpers, which out-of-tree models use, are outside this table, as is the public
client surface that the session and `register()` use (`AsyncLLM`, `StreamingInput`, `SamplingParams`,
`RequestOutputKind`, `StreamAdmissionError`, `EngineDeadError`; see the HLD's Engine surface constraint). Adding a dependency means
adding it here and to the import-time check.

## Implementation notes

- **Renderer.** In the target engine a tokenizer mode also selects a renderer; `register()`
  registers the standard Hugging Face renderer for `nemotron_asr_streaming`.
- **One geometry source.** `lookahead.served_lookahead(hf_config)` (Server geometry) is the only
  reader of the server's lookahead: the model's warmed and captured geometry, its geometry check on
  each row, the session factory, `buffer_realtime_audio` and `get_generation_prompt` all call it on
  the served configuration, and convert with `lookahead.geometry_index` or
  `lookahead.cadence_label`. No component keeps a served set.
- **Hook routing.** `engine_flags` holds the one predicate of hook-captured graphs, which both the
  engine check (Engine settings) and the model state's `needs_cudagraph_capture` use, and the one
  rule of runner configurations (Runner configurations), which the configuration hook, the model
  state's construction and model construction call; no other module judges a runner mode.
- **Precision.** The supported deployment runs at float32 (`--dtype float32`); the model's
  precision check at load refuses any other dtype.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| How the card becomes a served config | A registered config parser selected by `--config-format nemotron_asr_streaming` | `AutoConfig` registration for `nemotron3_5_asr`; `--hf-overrides` alone | Registering the card's `model_type` would claim the offline plugin's configuration; overrides apply after the configuration is built, too late for translation |
| Where launch options are resolved | Post-override validation of the final configuration; the parser reads only the checkpoint | Reading `hf_overrides` inside the parser; a plugin-specific environment variable | Core gives the parser only flat overrides, applies dictionary overrides after it, and also calls it with none when deriving a generation configuration |
| Control tokens in the tokenizer | A registered tokenizer mode that appends them as special tokens, with park as the tokenizer's EOS | Publishing a modified tokenizer; leaving them unknown | Core detokenizes every token; unknown IDs break detokenization and unmarked ones appear in text; the card must stay usable unmodified |
| Context ceiling | Translation sets `max_position_embeddings` to 2^24 | Requiring `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | The model has no positional limit; long files need a context far above the card's 32,768 |
| Where model execution memory is prepared | Long-lived working memory in `prepare_execution_memory`; the profile transaction on core's first profiling dummy step (flagged by `prepare_step`); segment graphs on that step under `--enforce-eager` and in `capture_cudagraphs` under the hook configuration | `initialize_state_slots` or first serving step: graphs created after core sized the KV cache and state pool, so their memory is unbudgeted. Capturing in `prepare_execution_memory`: the hook's contract forbids capture there. Capturing on the profiling step under the hook configuration too: core's estimate would not see those graphs as graph memory, and the profiling and serving captures would differ | Each stage runs where #67583 budgets it. A failed preparation ends worker startup and the engine is restarted: partial preparation is not safely retryable in the worker, and no serving state outlives the failure |
| Served geometry | One per server, fixed at startup: `num_lookahead_tokens` when given, else the card's `default_num_lookahead_tokens`; any of the five implemented lookaheads; `supported_num_lookahead_tokens` refused | A served set with a default (every geometry warmed and captured, multiplying graph memory and startup time); requiring the override (ignores what the checkpoint declares); ignoring `supported_num_lookahead_tokens` silently (an old serve command would start at a geometry its operator did not choose); only the card's four lookaheads (drops the tested 160 ms geometry) | One geometry keeps graph memory and capture time to one set of segment graphs, and matches how a Hugging Face Transformers model fixes its lookahead |
| Default segment arms | `eager-graphed` encoder and `dense-graphed` decoder, declared at translation; overrides kept | The eager arms by default; the compiled `dense-graphed` encoder | The pair measured fastest with equal transcripts (HLD, Key Design Decisions); the eager arms stay selectable as the reference |
| Arms under the hook configuration | The `eager-graphed` encoder and `dense-graphed` decoder (and transaction graphs); the compiled `dense-graphed` encoder and CHUNK-bucket graphs refused there | Capture every graphed arm in the hook; capture the unsupported ones on the profiling step | Capture of the compiled encoder and of CHUNK-bucket graphs through the hook is not yet specified or qualified, and the HLD forbids profiling-step capture in the hook configuration; both stay available under `--enforce-eager`. The refusal holds until their hook capture is specified and measured (Future roadmap 4); nothing technical prevents it |
| Segment-graph memory under the hook | Core's estimate through the profiling capture, a flat charged margin per call and an aggregate serving-over-profiling check, the same procedure as the transaction-graph cells | A per-graph comparison; a warning only | Core's charge is aggregate and allocator rounding arises per segment over the call; a warning would serve with uncharged memory where core estimated |
| Graphed arms without static graphs | A stand-in wrapper that runs each entry eagerly, keeping the graphed arms' inventory and serving path | Refusing graphed arms off CUDA; resolving the default arms to the eager arms there | The card's default arms are graphed, so a refusal would fail every non-CUDA run of the default configuration, and silently switching arms would leave the graphed staging and checks untested off CUDA; the stand-in keeps one path whose outputs equal the eager arms' |
| Core's runner graphs over the forward | None: the model state reports `AttentionCGSupport.NEVER`, core reduces every graphed mode it can to NONE, and the plugin refuses FULL and a piecewise mode under breakable graphs | Refuse every graphed mode; register the stateful segment operations as splitting ops; breakable graphs with host planning inside break operations; core full graphs with the model running eagerly during core's capture | Refusing would make every serve command without runner options name `-cc.cudagraph_mode none` while core can already reduce the mode itself, as it does for encoder-only model states. Splitting ops split a compiled graph, and the forward is not traceable; the code between its stateful segments is index uploads and a projection. Inside a breakable graph the whole forward would be one break operation, so core would capture nothing of value until planning and staging move out of the forward (Future roadmap 5). Core's full graphs replay without calling the forward, so its host planning would not run. The model's own graphs, captured through the hook, already key on its step shapes |
| Compilation modes | Accept NONE, `DYNAMO_TRACE_ONCE` and `VLLM_COMPILE`, which compile nothing in an undecorated model; refuse `STOCK_TORCH_COMPILE` | Require compilation off; accept every mode | The model has no `CustomOp` subclass, whose dispatch the compilation mode's `custom_ops` selects, so a compilation mode changes no operator dispatch in the model; `STOCK_TORCH_COMPILE` compiles the whole model with `fullgraph=True`, which a forward with host reads cannot pass |
| Asynchronous scheduling | Accepted | Refuse it | Core's per-request barrier for state models runs each stream one step at a time, which is the property the model needs; the model reads commit status in `postprocess_state`, before the next step executes |
| Core hooks | Required in every configuration; a core without them is refused | Detect them and fall back to profiling-step capture | One core API to test (HLD, Key Design Decisions, Core API) |
| Load-time checks | Client calls in `register()`; adapter symbols at model import | All checks in `register()` | `register()` and the session factory do not import the model class modules, so loading the plugin stays cheap; core's speech-to-text serving imports the model class in the API server when it starts, so importing it initializes no device and loads no weights |
| `generate` task | Not advertised (model-state override) | Advertise for compatibility | The model emits transcript labels, not text generation |

## References

- vLLM config parsers: `vllm/transformers_utils/config.py` (`register_config_parser`, the parser
  call and override application), `vllm/transformers_utils/config_parser_base.py`.
- Model configuration hooks: `vllm/model_executor/models/config.py` (`MODELS_CONFIG_MAP`).
- Supported tasks and model-state hooks: `vllm/v1/worker/gpu/model_states/interface.py` (hooks from
  vllm#67583); graph-pool routing: `vllm/compilation/cudagraph_pool.py`.
- Plugin loading: `vllm/plugins/__init__.py`.
- Card translation: `nemotron_asr/configuration_nemotron_asr.py`; geometries:
  `nemotron_asr/manifests.py` and `lookahead.py`.

## Future roadmap

Each item names what is deferred, the trigger that reopens it and the shape of the change.

1. **Version gate.** A version check for clean vLLM releases. Trigger: RFC 1's stream admission
   ships in a release. Shape: `register()` checks the release version; until then the client-path
   check is the gate.
2. **Projected capture failure.** Projecting the segment graphs' total from the first captured
   encoder counts and failing before capturing the rest, to save startup time on a
   `max_num_seqs` that cannot fit. Trigger: A40 measurements that validate a projection rule.
   Shape: a check after the first count's capture.
3. **Hook-configuration fit.** The retained memory, capture peak and startup time of the default
   arms under the hook configuration, where they are captured twice, are unmeasured, as is the
   larger profiling KV cache (more temporary state pages) when a graphed mode reduces to NONE.
   Trigger:
   acceptance tier 2 met in the hook configuration. Shape: the README's serve command moves to
   that configuration.
4. **Hook capture of the other graphed arms.** The compiled `dense-graphed` encoder and
   CHUNK-bucket graphs. Trigger: their exactness, fit and serving benefit are measured. Shape:
   compilation and warm-up in `prepare_execution_memory`, profiling capture over core's profiling
   pools, release in `clear_cudagraphs`, serving capture into core's graph pool, the same memory
   accounting and serving-over-profiling check; the hook-configuration refusal and the arm table
   change, and their capture specs join REG-MEM-016 to REG-MEM-022.
5. **Runner piecewise graphs through breakable CUDA graphs.** Core capturing the dense work
   between the model's stateful operations, with the model's graphs replayed inside core's eager
   breaks. Trigger: a bounded prototype shows a step-time gain over the model's own graphs at a
   configured chunk size, or a maintainer asks for runner-captured graphs. Shape: host planning
   moves into `prepare_inputs` and writes fixed-address buffers; staging writes into persistent
   buffers; the stateful segment operations become operations that end core's capture, run
   eagerly and write their outputs in place; the model keeps reporting `NEVER`, which declines
   full graphs, and PIECEWISE under breakable graphs is accepted once its transcripts equal
   `--enforce-eager`'s at every configured chunk size. Full decode graphs wait for item 6.
6. **Runner full graphs (FULL, and FULL_DECODE_ONLY without reduction).** Trigger: core gains a
   model-state hook that extends the full-graph dispatch key or declines a full graph for a step.
   Shape: item 5's host-planning move, plus one predicated device body per row count that holds
   for every chunk/echo split, geometry, file row and endpoint mix, a capture-capable
   `prepare_dummy_inputs`, and the model reporting `UNIFORM_BATCH` support; accepted once
   transcripts equal `--enforce-eager`'s at every configured chunk size.
7. **`STOCK_TORCH_COMPILE`.** Trigger: item 5 leaves a forward that Dynamo can trace whole.
   Shape: accept the mode once the traced forward's transcripts equal `--enforce-eager`'s at every
   configured chunk size.
8. **Transaction graphs on the state layer.** The model refuses `transaction_graph_populations`
   on this core, which has no state-group layout accessor to capture its state pages through.
   Trigger: core provides that accessor with fixed page addresses. Shape: capture through the
   accessor in the hook configuration at the 80 and 160 ms geometries, and the graph-ownership and
   transcript requirements of the hook configuration extend to the cells.
9. **Native burst with asynchronous scheduling.** Native burst is refused on this core. Trigger:
   native burst returns on the state layer. Shape: decide whether its read-ahead sampler composes
   with core's per-request barrier; refuse the pair until a GPU tier shows equal transcripts.
10. **Overlap under asynchronous scheduling.** The commit sink waits for each step's commit event
    in `postprocess_state`, so asynchronous scheduling overlaps little for this model. Trigger: a
    latency A/B that shows the wait on the critical path. Shape: read the commit status one step
    later, from a ring of pinned buffers.
