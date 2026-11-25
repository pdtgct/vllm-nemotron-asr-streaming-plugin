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

`register()` performs the following, and changes no registry until the core check has passed:
1. **Core check.** The installed vLLM must carry the state-layer surface the model uses:
   `ModelState` defines the six model-state hooks of vllm#67583 (Core hooks, below) and
   `take_failed_requests`; `MambaManager.layer_config` accepts `recomputable`, the parameter
   through which a state layer declares a KV-cache group that its tokens cannot rebuild;
   `MambaAttentionBackendEnum` has the member `PERSISTENT_STATE`; and
   `vllm.model_executor.models.interfaces` defines `SupportsPersistentState`. None of these checks
   imports a model class module. A missing item raises, naming each and stating that the plugin
   needs a vLLM build with RFC 1's state layer and vllm#67583's model-state hooks. The whole
   model-adapter dependency table is checked again when the model class is imported
   (Model-adapter dependencies). `register()` checks no client call: on the target core
   `AsyncLLM.generate` admits a persistent-state stream without an admission argument, and the
   session factory checks its own calls (the `session` design). `register()` initializes no device
   and loads no model; the imports it needs may pull in engine modules transitively.
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
- **Native burst.** On a core that runs native burst (not the target core, whose state layer it is
  refused on; Engine settings), `experimental_native_burst: true` requires the server's lookahead
  to be 1.
- **Transaction graphs.** The geometry rule of the transaction-graph option (80 or 160 ms) is one of
  that option's configuration checks, which validation calls (Post-override validation); this
  design does not restate it.

## Post-override validation

The configuration hook runs twice during startup:
- `verify_and_update_model_config(model_config)` runs at the end of model configuration, after
  core has applied every override. It resolves the server's geometry (above), sets
  `att_context_right`, and checks: a lookahead is present and is one of the five right contexts;
  no `supported_num_lookahead_tokens` is present; native burst's lookahead, on a core that runs
  native burst; the
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

The persistent-state model needs one copy of each session's state and a single device. With
transaction graphs off it accepts asynchronous scheduling in every setting (`async_scheduling`
`True`, `False` or unset): core's per-request barrier for state models schedules no state request
while the output of its previous step is still in flight, so each stream still advances one step
at a time. With transaction graphs on it accepts only `False` (the restricted startup, below). The worker
also stays one step at a time for the model under either setting: it runs a step's
`postprocess_state` before it executes the next step, and there the model waits for the step's
commit event, which follows every device read of the step's staged inputs and state, then reads
the step's commit status. A request that leaves the engine (finish, abort, error; core never
preempts a request whose state cannot be recomputed) reaches the model state through
`remove_request`, which core calls when it applies the next step's finished request IDs, also on
a step with no forward work. The model state retires a file request in that call and a stream
request at its next model-state call (State-slot access, Retirement and turn continuation). By
then the worker has run the previous step's `postprocess_state`, so retirement never races a step
that still reads the request's buffers. Native burst, whose sampler reads ahead of the scheduler, is refused on the state layer until its port there is
specified (Future roadmap 9), so it meets asynchronous scheduling nowhere yet. The plugin
refuses:
- on this core, native burst (`experimental_native_burst`), at model construction and in the
  configuration hook, naming the state layer and the remedy of removing it;
- while transaction graphs are on, every startup outside the **restricted startup** of the
  `transaction-graph` design's configuration checks: an `async_scheduling` other than `False`
  (TXG-CFG-017), a `cudagraph_mode` or compilation mode other than an explicit NONE
  (TXG-CFG-018), and neither `--num-gpu-blocks-override` nor a positive `--kv-cache-memory-bytes`
  (TXG-CFG-019), each named with its remedy, until core's profiling and extensible KV caches
  support a model-state capture over state pages and the qualification covers the other settings
  (Future roadmap 8); a core whose `ModelState` lacks `state_group_layouts` (TXG-CFG-016); and a
  population above core's admission limit (`--max-num-active-seqs`, or `max_num_seqs` when it is
  unset), which no step can reach (TXG-CFG-020); and a `--num-gpu-blocks-override` below the
  admission limit plus one, which core's scheduler would refuse only after both captures
  (TXG-CFG-005);
- prefix caching: `enable_prefix_caching = True` in the resolved configuration. An unset command-line
  option arrives already `False`: `EngineArgs` resolves it from the model, which reports prefix
  caching as unsupported when it declares persistent state. A configuration built directly with
  `True` is refused by core's own check before this rule runs (Core's checks first, below);
- speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator,
  hierarchical sparse attention, fault tolerance (`parallel_config.enable_fault_tolerance`;
  `--fault-tolerance-config` turns it on, and the refusal names that cause) and elastic expert
  parallelism (`parallel_config.enable_elastic_ep`, whose reconfiguration reruns the worker's
  warmup and so would call the serving capture again over live state pages; A second serving
  capture, below). Core refuses several of these settings itself, before the configuration hook
  (Core's checks first, below). Moving execution preparation into core's profiling step does not
  make sleep mode or the cumem allocator usable: both stay refused, so there is no wake-up
  re-preparation;
- any tensor, pipeline, data, decode-context or prefill-context parallel size other than 1 (a
  setting the build does not have counts as 1);
- a runner configuration other than the two of Runner configurations, below.

The model state's construction re-checks these rules against the final engine configuration, where
an unset `cudagraph_mode` or compilation mode is not accepted.

On a core that runs native burst, its own configuration check, run at model construction, applies
the same rule for a setting the build does not define: a parallel size counts as 1 and expert
parallelism as off.

**Core's checks first.** While core builds the engine configuration (`VllmConfig.__post_init__`),
its persistent-state check (`_verify_persistent_state_config`) runs before the configuration hook
(`try_verify_and_update_config`) and raises `ValueError` with the message `<option> is not
supported for this model` for the first setting it finds of: `enable_prefix_caching`; a KV
connector (`kv_transfer_config`); an EC connector (`ec_transfer_config`); speculative decoding
(`speculative_config`); a pipeline, data or tensor parallel size above 1; multi-dimensional rotary
positions (M-RoPE), which this model does not use; fault tolerance's `request_failure_policy`
`recompute`; and, because the model declines synthetic warmup (below), `enable_fault_tolerance`
and `enable_elastic_ep`. For these settings the operator sees core's message, which names one
setting; the configuration hook does not run. The plugin keeps its own rules for them (REG-ENG-002
to REG-ENG-004), which list every failed check when the hook is called directly on such a
configuration. Core's check does not cover decode-context and
prefill-context parallelism, dual-batch overlap, sleep mode, the cumem allocator or hierarchical
sparse attention; for them the configuration hook's refusal is the operator's message.

### Synthetic warmup

The model declines core's synthetic warmup (`persistent_state_accepts_synthetic_warmup = False`,
Supported tasks). Core's `warmup_kernels` adds warmup requests through the model state's
`add_request`, each scheduling two tokens with non-greedy sampling. The worker runs them whenever
JIT warmup or the extensible KV cache is on, and runs the extensible KV cache's sizing warmup, also
through the model state, whenever that cache is on. The extensible KV cache defaults on for CUDA,
so both would run under `--enforce-eager` too, where JIT warmup is off. The model state records
every request `add_request` delivers and refuses a real row that schedules more than one token, so
a warmup request would fail startup; recognizing warmup requests would need a contract that names
them, which core does not have. With the declaration core runs neither warmup, and the profiling
run is the only startup step that runs the model's transaction; the profile transaction at its
maximum shape (Execution memory) is the model's transient peak there. Core's `kernel_warmup` dummy
runs, which add no request through the model state, still run (Execution memory, Repeated dummy
steps). The cost is that the sampler's and other JIT-compiled kernels compile on the first serving
steps; that latency is unmeasured.

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
mixed-batch full graphs it cannot reduce, it refuses. (A piecewise mode under any compilation mode but `VLLM_COMPILE`, with breakable graphs off,
loses its piecewise part already when the engine configuration is built, because no compiled
region exists to split: PIECEWISE becomes NONE and FULL_AND_PIECEWISE becomes FULL_DECODE_ONLY.)
The model class carries no
`@support_torch_compile`, so no compilation mode compiles any of it except `STOCK_TORCH_COMPILE`,
under which the runner compiles the whole model with `fullgraph=True`.

Breakable CUDA graphs and the compilation mode interact through core's configuration: core sets
`use_breakable_cudagraph` from `VLLM_USE_BREAKABLE_CUDAGRAPH` when it is set, and otherwise from
automatic rules that cannot select this model (its architecture is not on core's default list, and
batch invariance enables breakable graphs only for bf16); when it enables them, and when
`TORCH_COMPILE_DISABLE=1` is set, it sets the compilation mode to NONE.

Two runner configurations are accepted:
- `--enforce-eager`. Core normalizes `cudagraph_mode` to `NONE` and runs none of the three
  graph-capture hooks; the model captures its own segment
  graphs on the first profiling dummy step (Execution memory). Every segment arm is accepted.
- **The hook configuration**: `enforce_eager` off, on CUDA. Core captures every graph the model
  runs through the model-state capture hook, and none is captured on the profiling dummy step.
  Its conditions, each with the remedy the refusal names:
  - **graph mode**: `cudagraph_mode` is not FULL. Core cannot reduce FULL and would refuse it
    itself only after the weights load (remedy `-cc.cudagraph_mode none`, or leave the option
    unset). NONE, PIECEWISE, FULL_DECODE_ONLY and FULL_AND_PIECEWISE all resolve to NONE for
    this model, subject to the next condition.
  - **breakable piecewise**: `cudagraph_mode` is not PIECEWISE while breakable graphs are on.
    Breakable graphs give core a piecewise capture path without compilation, so PIECEWISE would
    survive resolution and core would capture the forward as one breakable segment (remedy
    `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, or `-cc.cudagraph_mode none`). With breakable graphs on, the
    compilation mode is NONE, so FULL_AND_PIECEWISE resolves to NONE and is accepted.
  - **compilation mode**: not `STOCK_TORCH_COMPILE` as the mode the runner will see. A requested
    `STOCK_TORCH_COMPILE` is accepted when core will set the mode to NONE (breakable graphs on, or
    `TORCH_COMPILE_DISABLE=1`), and refused otherwise (remedy `-cc.mode none`, or leave the option
    unset). NONE, `DYNAMO_TRACE_ONCE` and `VLLM_COMPILE` compile nothing in the model.
  - **platform**: CUDA (remedy `--enforce-eager`).
  - **hook-captured graphs**: at least one graph the hook captures, the `eager-graphed` encoder,
    the `dense-graphed` decoder, or transaction graphs (remedy: select the `eager-graphed` encoder
    or the `dense-graphed` decoder, or `--enforce-eager`); and no graph it does not capture: the compiled `dense-graphed` encoder
    arm and CHUNK-bucket graphs (`chunk_bucket_graph_populations`) are refused with the remedy
    `--enforce-eager`. The eager arms alone capture nothing, so they need `--enforce-eager`.

`enforce_eager` is not core's default, so the hook configuration is the configuration a serve
command without runner options selects. A refusal names each unmet condition with its remedy;
every remedy that names `--enforce-eager` also names removing `transaction_graph_populations`
when it is set (the option is refused under `--enforce-eager`).

While transaction graphs are on, the hook configuration narrows to explicit
`-cc.cudagraph_mode none -cc.mode none` (TXG-CFG-018): every other accepted mode, an unset one
included, is refused, and the remedies of the graph-mode, breakable-piecewise and
compilation-mode conditions name only `-cc.cudagraph_mode none` and `-cc.mode none`. The
`transaction-graph` design owns that condition; this design's conditions still apply beside it.

One function in `engine_flags` holds these conditions. Three stages apply it, each to what core
has settled by then:

| Stage | Runs | Modes it sees | What it judges |
| --- | --- | --- | --- |
| Configuration hook (`verify_and_update_config`) | in the API server, while core builds the engine configuration, before core resolves scheduling, eager mode, breakable graphs and the mode defaults | the requested `cudagraph_mode` and compilation mode, either possibly unset | every condition; an unset mode passes; breakable graphs are predicted from `VLLM_USE_BREAKABLE_CUDAGRAPH`, and the compilation mode core will set from breakable graphs and `TORCH_COMPILE_DISABLE`, as core will apply them |
| Model and model-state construction | in the worker, after core filled every default and normalized the configuration; model construction first | both modes set; breakable graphs resolved | every condition again; an unset mode is refused; a `STOCK_TORCH_COMPILE` still present here is refused, since core would have set it to NONE if it were going to |
| The model state's `initialize_kv_cache` | after core resolves the runner mode, before any capture, in the profiling and the serving KV-cache initialization alike (also under a budget fixed through `--kv-cache-memory-bytes`) | the resolved `cudagraph_mode` | that it is NONE; otherwise startup fails naming the resolved mode |

So an unset mode whose optimization-level default is PIECEWISE (`-O1`) under breakable graphs is
refused at model construction, where the default is known; the configuration hook does not
duplicate core's table of defaults. Behind the three stages, every path by which core would
capture the forward passes through the model state before the forward runs, and the model state
refuses each: `prepare_dummy_inputs` (core's capture inputs) and `prepare_attn` for a capture or
a FULL step raise, naming the mode and stating that the model supports no runner graph.

While `enforce_eager` is not set, on its first `initialize_kv_cache` call in the process the model
state logs once that core's runner captures no graph over the model's forward and that the model
captures its own graphs through the hook, naming the `cudagraph_mode` and compilation mode its
construction saw and the resolved `cudagraph_mode`; core's own log names the mode it reduced.
`prepare_execution_memory` and the profiling dummy step run before core resolves the mode, so
neither depends on the resolved mode.

A mode that is not NONE after core's configuration build (FULL_DECODE_ONLY; PIECEWISE and
FULL_AND_PIECEWISE under `VLLM_COMPILE`; FULL_AND_PIECEWISE with breakable graphs on), and that
core therefore reduces only when it initializes the KV cache, is not identical to an explicit NONE at startup: the
engine configuration sizes core's graph capture sizes from the requested mode, and core's
profiling KV-cache initialization, with it the state group of the profiling capture,
holds `min(max_num_seqs, the largest capture size) or 1` pages, where an explicit NONE, or a
mode reduced while the configuration is built, has no capture sizes and so holds one. A set
`num_gpu_blocks_override` replaces that count, so the profiling KV cache then holds the override's
block count, the serving page count (`may_override_num_blocks` in
`vllm/v1/core/kv_cache_planning.py`). The model's graphs and its serving state pages do not
change; the profiling capture's transient memory does (Future roadmap 3). Transaction-graph cells
capture over `N` pages beside the null page, which is one reason the restricted startup requires
an explicit block count or KV budget.

### Qualification of the runner configurations

Accepting a configuration and qualifying it are separate. Acceptance is decided at startup by the
rule above. Qualification has two parts.

The CPU tier shows, for every accepted combination of `cudagraph_mode`, compilation mode and
breakable setting, that core's own resolution, run on the normalized configuration with the
model's reported support and with piecewise capture available exactly when breakable graphs are
on (the model has no compiled submodule), yields NONE, and
that the model's graph inventory (its segment-graph entry keys: encoder CHUNK counts and decoder
tiers) does not depend on either mode: the model decides what it captures from the served
configuration alone.

The GPU tier runs on an A40 at every chunk size the model configuration allows (80, 160, 320, 560
and 1120 ms; one per server, fixed at startup), on the realtime workload: the first 200
LibriSpeech test-clean utterances in the harness's fixed order through `/v1/realtime`, paced at
real time, 32 concurrent sessions, with the same segment arms, `--max-num-seqs`,
`--max-model-len`, dtype and budget setting in every server of a chunk size. Its reference is
`--enforce-eager --no-async-scheduling`. Each server's startup record shows core's runner graph
count (zero), the model's graph inventory and the modes its construction saw. The comparisons:
- explicit NONE (`-cc.cudagraph_mode none -cc.mode none --no-async-scheduling`) and a serve
  command whose only scheduling or runner flag is `--no-async-scheduling` (FULL_AND_PIECEWISE
  requested under `VLLM_COMPILE`, the defaults) produce the reference's final transcripts;
- asynchronous scheduling, with no runner flag and under `--enforce-eager`, produces the
  reference's final transcripts;
- under `VLLM_BATCH_INVARIANT=1`, one server with no runner flag and `--max-num-seqs` of at least
  32 gives each utterance the same final transcript at 32 concurrent sessions, at 8, and, for the
  first 20 utterances, at 1. At float32 core enables no breakable graphs for batch invariance, so
  the runner conditions are unchanged under it.
Outside batch-invariant mode the comparisons presume, as every earlier tier observed at float32,
that transcripts do not change with batch composition; a mismatch there fails the tier and is
investigated against the batch-invariant run. The plugin's `gpu` test tier, which covers stream
and whole-file aborts and steps with only finished requests, also runs under asynchronous
scheduling. A configuration that fails its tier is refused with
its reason and remedy, and its gap becomes a Future roadmap item, until a fix passes.

### Core hooks

The plugin needs #67583's model-state hooks in every configuration: `prepare_execution_memory`,
`needs_cudagraph_capture`, `capture_cudagraphs`, `clear_cudagraphs`, `post_warmup` and `shutdown`
(`engine_flags.MODEL_STATE_GRAPH_HOOKS`). `verify_and_update_config` refuses a core whose
`ModelState` does not define each of them, naming every missing hook and #67583, and the
model-adapter dependency check (below) lists them, so a worker on such a core also fails before
any weights load. Under `--enforce-eager` core still calls `prepare_execution_memory`,
`post_warmup` and `shutdown`.

While transaction graphs are on, the model also needs core's state-group layout accessor,
`ModelState.state_group_layouts`, through which their cells reach the state pages (the
`transaction-graph` design, Rules of the capture call). Model construction checks it as a
conditional dependency (Model-adapter dependencies), and the configuration checks refuse the
option on a core without it (TXG-CFG-016).

### README serve command

The README's serve command is the default configuration: the default segment arms and the
checkpoint's default lookahead (3, which is 320 ms, on the published card), so it sets no `num_lookahead_tokens` and no
`supported_num_lookahead_tokens`. It shows `--dtype float32`, `--config-format
nemotron_asr_streaming`, `--tokenizer-mode nemotron_asr_streaming`, one of the two runner
configurations, chosen from those that acceptance tier 2 has met with the default arms (until the
hook configuration meets it, `--enforce-eager`) with the runner and scheduling flags tier 2 was met
with (`--no-async-scheduling` until a tier is met with asynchronous scheduling, which core
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
and its largest transaction needs transient memory beyond that. Core sizes the KV cache, whose one
group is the model's state group, from the memory profile of its profiling run, and binds the
group's pages to the state layer (`MambaBase.bind_kv_cache`) only after it allocates them; the
model state checks that binding in `initialize_kv_cache` (State-slot access). #67583 divides model
execution memory between two hooks, and the model keeps them apart: long-lived working memory
outside any graph belongs to `prepare_execution_memory`, which every profiled budget counts; graphs
captured through `capture_cudagraphs` are counted only by core's graph-memory estimate. Under
`--enforce-eager`, where core runs no capture hook, the model captures its segment graphs on core's
profiling dummy step, inside the profiled budget.

The model prepares in three places, by runner configuration:

| Where | `--enforce-eager` | Hook configuration |
| --- | --- | --- |
| `prepare_execution_memory()` | Allocates long-lived working memory and warms uncaptured execution (static encoder execution, compilation of a compiled arm). Captures nothing. | The same. |
| First profiling dummy step | Captures the segment graphs, verifies the inventory, then, with every graph retained, runs the profile transaction at its maximum shape. | Verifies the uncaptured inventory and runs the profile transaction once at its maximum shape over the arms' uncaptured paths (the `eager-graphed` binding's uncaptured transition and the dense decoder's uncaptured decode, at that shape's row count). Captures nothing. |
| `capture_cudagraphs()` | Not called. | Captures the segment graphs (below). |

The profile transaction's maximum shape is `max_num_seqs` rows at the server's geometry and each
configured CHUNK-bucket graph population; transaction graphs add no shape to it. It sizes the option-off transaction's transient use and its use of core's
workspace, which core locks after the serving capture.

- **Step classification.** Core gives the model state no record of why it runs a step, so the
  model state's `prepare_inputs` classifies each step by the requests it holds at that moment,
  after deciding any pending removal (State-slot access): a step with a row whose request the
  model state does not hold, because it never recorded the request or has retired it, is a
  *dummy step* (core's dummy batches carry fresh request IDs), every other step is a *real step*,
  and a dummy step for which core called
  no `prepare_attn` since the previous `prepare_inputs` is a *profiling dummy step* (core's
  profiling run builds no attention metadata). A dummy step's projection reads no page
  (`no_page_io`).
- **One owner per stage.** The model's forward on the first profiling dummy step is the only place
  that runs the profile transaction, and under `--enforce-eager` the only place that captures. It
  returns the dummy output core expects. The forward runs no forward-time profile of an
  unprepared encoder: none can run once the inventory is sealed.
- **Dummy output.** Core indexes the profiling step's forward output by its logits indices, computes
  logits and runs the sampler on it. The dummy output matches the input embeddings' token count,
  width, dtype and device, and its carrier column holds a valid decision ID (zeros), so dummy sampling
  needs no state; dummy sampling reads no state page.
- **Repeated dummy steps.** Core runs one profiling run per worker, but its kernel warm-up
  (`kernel_warmup`) may run further dummy steps after the state pages are bound (FlashInfer
  autotuning on recent GPUs), with real token IDs and possibly attention metadata, several times.
  Whether or not the model state classifies such a step as a profiling dummy step, the forward
  returns the dummy output with no capture, no profile transaction and no page access; a later profiling dummy step also checks the inventory
  (under the hook configuration, segment graphs not yet captured by the serving capture are
  accepted as pending).
- **Other dummy steps.** On a dummy step the model reads no page, no recorded request and nothing
  left from an earlier real step. Core's dummy batches carry only fresh request IDs, so no row of
  a dummy step names a request the model state holds.
- **No state page needed.** Preparation, capture and profiling use model-owned scratch state built
  from the model's state layout (`StateLayout`, built from the checkpoint's state manifest):
  segment graphs copy state into and out of their own static buffers, and the gather from and
  scatter to the pages run outside any segment graph, so neither preparation, the profile
  transaction nor segment-graph capture reads the state layer's views, a page or its address, in
  either runner configuration. Transaction-graph cells reach the pages from inside their graphs,
  through `state_group_layouts`; the `transaction-graph` design owns that access.
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
  and cannot apply, because the model declares a state layer. When the model declines synthetic
  warmup, core runs no sizing warmup for an extensible KV cache either (Synthetic warmup).
- **State pages.** Core sizes the state group's block count from the KV-cache budget as for any
  `MambaSpec` group, unless the operator overrides the block count. A request holds one block,
  because the block size is `max_model_len`. The scheduler fails construction unless the group
  holds a page for every request that can be active (`max_num_active_seqs`, by default
  `max_num_seqs`) besides core's null block; whether the budget yields that many pages at the
  README's settings is unmeasured. Core builds the scheduler after the model's graph captures, so
  a block count that is too small fails only after them; while transaction graphs are on, the
  plugin refuses such an override in the configuration hook and such a budget at the
  transaction-graph part's pool access (TXG-CFG-005, TXG-CAP-008).
- **Graph runtime.** Graph capture uses core's graph-capture context, forward context and the
  platform's static graph wrapper. Under the hook configuration every capture runs on the `stream`
  core passes, inside `vllm.compilation.cudagraph_pool.capture_pool` with the `graph_pool` core
  passes; the model creates no stream and no graph pool of its own. The target core has
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
- **Transaction graphs on or off.** Whether transaction graphs are on is decided only by the
  `transaction-graph` design's `transaction_graphs_on` (TXG-CFG-014), which every check, hook and
  dependency gate here calls, so a malformed value such as `{}` is on, and refused, everywhere.
- **Memory diagnostics.** The target core reads no list of the options that enlarge the model's
  execution memory or graphs, so core's own failures of profiling, capture and capacity planning
  name none of them. The model's capture line and capture-failure message (Segment-graph memory)
  name the arms, the geometry, `max_num_seqs` and the remedies (Future roadmap 12).

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
| `capture_cudagraphs(profile_only=True)` | Runs only when core's graph-memory estimate runs, over core's minimal profiling KV cache and into the estimate's throwaway pool. Captures the segment graphs for the server's geometry, then the transaction-graph part; with transaction graphs off, it ends with the charged margin. Logs the capture line. |
| `clear_cudagraphs()` | Releases every segment graph and its buffers, after the transaction-graph part, and, with transaction graphs off, the margin the profiling capture created (with them on, the transaction-graph part releases it); keeps the profiling figures as host integers. Core calls it also when the profiling capture raised or was never reached, so it tolerates nothing captured. Core also clears every registered static graph wrapper after the estimate, so no profiling entry survives either way. The model state also drops the state layer's views (State-slot access, View lifetime), whether or not anything was captured. |
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
pages is not supported and that the recovery is an engine restart, and naming
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
  under a profiled budget, also a lower `--gpu-memory-utilization`, or, when
  `--num-gpu-blocks-override` fixes the block count (and with it the state pages), a smaller block
  count that still holds a page for every request that can be active and, while transaction graphs
  are on, for the largest population. The serving-over-profiling
  failure names both call totals, `r` and the excess, with the same remedies. Whatever the
  failure, the propagated error holds no reference to the tensors or graphs the capture created:
  the cleanup clears the local variables of the finished frames in its traceback and in the
  tracebacks of its cause and context, keeping the messages and traceback lines, so that a
  retained error does not keep the discarded memory alive.

## Model-state interface

Every `ModelState` method the model state overrides keeps the base signature at the target core,
because core's model runner calls them positionally; a test binds each override against the base
signature. In particular the profiling path reaches the forward through
`dummy_inputs_embeds(num_tokens, input_ids=None)`. The model state returns its pre-allocated
embeddings buffer sliced to `num_tokens` in both cases: when core passes randomized `input_ids`
the contents stay unused, because the model's dummy forward reads no embedding values and returns
the dummy output.

Core gives the model state no step record of its own; the model state rebuilds each step from
these overrides, which the runner calls on the worker:

| Method | What the model state does |
| --- | --- |
| `add_request(req_index, new_req_data)` | Decides a pending removal, records the request (its prompt, slot, admission generation and `NewRequestData`) and takes a file request's samples (State-slot access; the `transcription` design). |
| `remove_request(req_id)` | Decides a pending removal, then retires a file request at once or records a stream request as the one pending removal, which the model state's next call decides (State-slot access). |
| `initialize_kv_cache(kv_cache_config, block_tables)` | Checks that the one KV-cache group holds exactly the state layer and that core bound its pages, then the resolved runner mode (REG-ENG-017). |
| `prepare_inputs(input_batch, req_states)` | Decides a pending removal, classifies the step (Execution memory), fails at planning a real row that schedules more than one token (Error boundary, below), binds the step's rows to their slots and plans the step once. |
| `prepare_attn(...)` | Decides a pending removal, builds no attention metadata and returns `{}`; refuses a capture, a FULL step and a micro-batch (REG-ENG-029). |
| `postprocess_state(...)` | Waits for the step's commit event, collects the step's commit status, applies file queue counts, and records the step's failed requests (Row failure, below). |
| `take_failed_requests()` | Returns the requests `postprocess_state` recorded as failed and clears the record. |

The six graph hooks, `get_additional_cg_support`, `prepare_dummy_inputs` and
`get_supported_generation_tasks` are described under Execution memory, Engine settings and
Supported tasks.

**Row failure.** A step can fail some of its rows and complete the others. A row fails when its
collected commit status resolves to a nonzero value (the `session` design, Row status in the
model), when the model state failed it at planning (the `transaction-graph` design, TXG-FAIL-008,
with transaction graphs on or off; Error boundary, below), and when it is a file row with no
accepted queue count (the `transcription` design, Queue counts). `postprocess_state` records each
such request and raises no error for it. The runner asks once per step, after `postprocess_state`, through
`take_failed_requests`, which returns the recorded request IDs and clears the record; it returns
`None` when none is recorded, and a dummy step records none. The IDs are the worker's request
IDs, which are the IDs core's scheduler holds. Core's scheduler discards those rows' sampled
tokens and finishes each such request alone with finish reason `error`, publishing a terminal
output for a state stream, and ignores an ID it no longer holds. The failed request's record,
file buffer and progress are retired like those of any finished request, from core's
`remove_request` (State-slot access). The channel adds no host synchronization:
`postprocess_state` already waits for the step's commit event before it reads the commit status
(Engine settings). The plugin's unit tests check the return-and-clear contract; the `gpu` test
tier checks that a returned request ends with finish reason `error` while the step's other
requests continue.

**Error boundary.** The model state fails a request alone for every fault that the request's own
prompt or multimodal items can cause, because a direct engine client can send any of them and a
raise would end every stream on the server: a row that schedules more than one token, a scheduled
window that crosses the end of the recorded prompt (State-slot access, Recorded prompt), a unit
at another geometry (the `session` design, Geometry check in the model), and a malformed or
foreign-geometry file item (the `transcription` design). Each such row fails at planning through
the planning-failed set: it stays in the row plan at its scheduled position as a non-CHUNK row
that the transaction rejects (the `transaction-graph` design, Option and configuration), and its
request ends through `take_failed_requests`. A multi-token row's tokens are not classified; the
step's other rows read and write at their own token positions (the `transaction-graph` design,
Option and configuration), so the multi-token row fails its request alone. The model state raises
for a violation of core's contract with the model state, which means that core and the model
disagree about a request's state, so that no outcome confined to one request is safe: a
`NewRequestData` that does not carry exactly one block, a slot outside the bound pages, and an
`add_request` that neither starts nor continues a request (State-slot access). It also raises for
the configuration and lifecycle refusals of Engine settings and Execution memory. A planning
fault that no list here names takes its class by the same test: it fails the request when the
request's own prompt or multimodal items can cause it, and raises otherwise (REG-STATE-015).

A stream CHUNK row's envelope header is resolved from the request's multimodal feature that
overlaps the row's scheduled token (`resolve_row_envelope_header` in `nemotron_asr/plan.py`).
Core lets a direct engine client bypass the plugin's processor, so the faults of that resolution
fall in both classes (REG-STATE-015):

| Fault | Class | Why |
| --- | --- | --- |
| a non-CHUNK row that a feature overlaps | fail the request | the request's prompt places a feature at a non-carrier token |
| a CHUNK row that no feature, or more than one, overlaps | fail the request | the request's prompt and items disagree on where the units are |
| an overlapping feature whose modality is not audio | fail the request | the request sent another modality |
| an overlapping feature with no kwargs payload or no envelope payload | fail the request | the request's item carries no envelope |
| an envelope smaller than its header | fail the request | the request's item is truncated |
| a feature without position metadata | raise | core's multimodal processing places every feature it delivers |
| a feature of non-positive length | raise | core never delivers an empty placeholder range |
| a non-CHUNK row with a scheduled encoder input | raise | core schedules encoder input only for a token that a feature covers |
| a CHUNK row with more than one scheduled encoder input | raise | a one-token window overlaps at most one scheduled input |
| a scheduled encoder input that names a feature other than the overlapping one | raise | core's scheduler and the request's positions disagree |
| an envelope payload that is not host-resident | raise | where the payload lives when the model state reads it is decided by core's multimodal delivery, not by the request |

A row that matches both a fail-the-request fault and a raise fault fails its request: the client
caused it.

**Exceptions.** Core ends the engine when a model-state method or the forward raises, so every
raise is worker-fatal and no later step of the worker runs. `postprocess_state` records a step's
failed requests only after the step's collection completes; if it raises, it first drops the
step's plan and records no failed request for that step, so `take_failed_requests` never returns
a partial set. Execution preparation and capture discard their partial state before they raise
(Execution memory; Segment-graph memory, Capture failure).

## State-slot access

The model's state is one page per request in one KV-cache group. The state layer
(`NemotronStateLayer`, a `MambaBase` layer registered under its prefix in the static forward
context) declares a `MambaSpec` whose states are the state layout's descriptors in manifest order,
with `state_alignment` 1 (the manifest publishes its padding explicitly), `block_size` equal to
`max_model_len` (so a request holds one block) and `mamba_type` `PERSISTENT_STATE`, and returns
`MambaManager.layer_config(spec, recomputable=False)` as its KV-cache configuration. The spec's
page size must equal the layout's page size. Core refuses a group its tokens cannot rebuild
unless the group's spec is a `MambaSpec` of `mamba_type` `PERSISTENT_STATE`, and it resolves that
type to its persistent-state attention backend, whose metadata the model never builds. Core
allocates the request's block when it first schedules the request, keeps it while a stream waits
for its next input, frees it when the request finishes, and never preempts the request. It binds
the group's memory through `MambaBase.bind_kv_cache`, one view per state with one row per block,
and binds it again when the extensible KV cache grows, so the layer rebuilds its views after every
binding (View lifetime, below). The extensible KV cache keeps every block at a fixed offset in its reservation, and it
grows once, while the engine initializes, before the worker serves; its only other growth is in
the synthetic warmup that the model declines. A rebinding therefore rebuilds the views over the
same addresses, and no page moves while a request holds it. While transaction graphs are on, the
restricted startup turns the extensible KV cache off, and the state layer reports any binding
after the serving capture to the transaction-graph binding, which checks it against the layout
its cells captured (TXG-LIFE-009).

- **Slot.** A request's slot is its one block, `block_ids[0][0]` of the `NewRequestData` core
  delivers to `add_request`, whose `block_ids` must hold exactly one group of exactly one block;
  block 0 is core's null block, so a valid slot lies between 1 and the bound block count minus 1.
  The model state pairs the slot with an admission generation from its own counter, new for every
  `add_request` that does not continue its request (`SlotHandle`). The
  `transcription` design records a file's host buffer with that pair.
- **Fresh page.** A row is fresh when its `num_computed_tokens` is 0; core delivers that count
  exactly on the request's first step, the step on which its page is new. Core initializes a new
  page before the step's forward (`StatePageInitializer`, writing each state's entry of the spec's
  `initial_values`, zeros because the layer sets none). The model's forward then initializes each
  fresh row's page through the layer's `initialize_pages` before its transaction: it zeroes every
  byte and writes each descriptor's non-zero initializer (the label book's blank label). Both
  writes run on the current stream before any read of the page, whether or not the row later
  fails at planning or the transaction rejects it, an all-failed step included: the body finds a
  per-row rejection only on the device, and a fresh page must never expose an earlier request's
  state. A rejected row's pages are then preserved as initialized (the `transaction-graph` design,
  Fresh rows). The layout has no bytes outside
  its states, so core's zeroing and the layer's agree on every byte except the blank labels
  (Future roadmap 11).
- **Retirement and turn continuation.** For each input after a stream's first, core calls
  `remove_request` and then `add_request` for the same request, with the compacted prompt and
  `num_computed_tokens` 1 (Recorded prompt, below). For a finished request it calls
  `remove_request` alone, also on a step with no forward work, and before it adds any request it
  calls `remove_request` for that request's ID. A file request (File requests, below) is one
  request that core never continues, so `remove_request` retires it within the call, dropping its
  record, admission generation, file buffer, progress and failed-file mark; an idle engine keeps
  none of them. A stream request's `remove_request` may precede its next turn, so the model state
  holds that request as its one pending removal and decides the removal at its next model-state
  call: the next call of `add_request`, `remove_request`, `prepare_attn` or `prepare_inputs`. An
  `add_request` of the same request, with `num_computed_tokens` above 0 and the same slot,
  continues the request: it keeps the request's admission generation and replaces its recorded
  prompt and `NewRequestData`. Every other such call retires the pending request first, dropping
  its record and admission generation. A finished stream's record therefore stays until the next
  model-state call; it holds no host buffer, and core has already freed its page. A
  `remove_request` for an ID the model state does not hold only decides an earlier pending
  removal. An `add_request` for a live request (one held with no removal since its latest
  `add_request`) raises, naming the request, whatever its `num_computed_tokens`, 0 included; so
  does an `add_request` with `num_computed_tokens` above 0 that continues no request.
- **Views per step.** During a real step, the model reads state only through the layer's views
  as bound at that step (`component_pools`, read when the step's `StateBatch` is built), for the
  slots of the step's rows, and refuses a slot outside the bound pages before it reads any. The
  one other page it reads is the null page (block 0), from which a fresh row gathers its zero
  initial state (`transaction-graph`, TXG-MASK-001); it writes no null page (TXG-MASK-007).
- **View lifetime.** The layer holds its page view and its per-state views (`component_pools`)
  by reference: they are views of the storage core bound, never copies, so their addresses are
  the bound pages' addresses, which the transaction graphs' cells bake in. It keeps them in one
  cache keyed on the identity of the bound `kv_cache` object and drops the cache when core calls
  `bind_kv_cache`, when a read finds another object bound, and whenever core calls the model
  state's `clear_cudagraphs`; the next read rebuilds it from the bound `kv_cache`. A read of a
  current cache costs one identity comparison and no rebuild, and the model state reads the views
  a few times per step. The `clear_cudagraphs` drop removes the plugin's own references to core's
  profiling KV cache. When `clear_cudagraphs` returns, the plugin's objects hold that cache only
  through the references core installed: the layer's `kv_cache` and the model state's
  `_state_group_layouts`. Core's teardown of the cache, which follows the hook
  (`_teardown_profiling_state` in `vllm/v1/worker/gpu/cudagraph_utils.py`), clears both: it sets
  `_state_group_layouts` to an empty tuple, resets each layer's `kv_cache`
  (`clear_layer_kv_caches` in `vllm/v1/worker/utils.py`), then runs `gc.collect` and
  `empty_cache`. After that teardown no object of the plugin keeps the cache allocated
  (REG-STATE-018). Nothing reads the layer between that teardown and the serving binding, so an
  identity check alone would keep the profiling KV cache allocated until core rebinds the layer,
  after it has allocated the serving KV cache: peak and resident memory would exceed the budget by
  the profiling KV cache while both exist. The only model-state call core makes in that interval
  is `get_additional_cg_support()` (`initialize_kv_cache` in
  `vllm/v1/worker/gpu/model_runner.py`), a capability query that returns a constant (REG-ENG-016)
  and is neither a release point nor a measurement point, so the installed-engine check
  (REG-STATE-019) observes the release through a `StorageWeakRef` taken at the profiling binding,
  which must have expired by the serving binding. The serving
  block count would not change, because core sizes it from a memory reading
  taken before its graph-memory estimate. Core's `ModelState.clear_cudagraphs` contract releases
  whatever the model state kept from the profiling KV cache, and core calls it while that cache is
  still bound, after a successful and after a failed profiling capture, so the drop rests on an
  existing core contract and needs no core change. The hook covers every case in which a profiling
  KV cache exists. Core binds one only when its graph-memory estimate runs past its early returns
  (the estimate enabled, the extensible KV cache off, and a graph to capture), which for this
  plugin means the hook configuration under `--num-gpu-blocks-override` or with
  `enable_extensible_kv_cache` false. There the model state captures through the hook, since the
  hook configuration selects a hook-captured graph (REG-ENG-025), so core calls
  `clear_cudagraphs` before its teardown. Under `--enforce-eager`, with the estimate disabled,
  with the extensible KV cache on, or under `--kv-cache-memory-bytes`, core binds no profiling KV
  cache, and the drop releases nothing. Core also calls `clear_cudagraphs` before a wake
  recapture, with the serving KV cache still bound; sleep mode is refused (Engine settings), and
  the next read would rebuild the cache from the bound `kv_cache`.
- **Commit check.** Before committing a step's state, the model checks that the commit's row order
  and slots equal the step's bindings, and writes no state if they differ. The check compares no
  admission generation, because steps are serial for the model: core calls `add_request` and
  `remove_request` only at the start of a step, and the worker runs a step's `postprocess_state`
  before it starts the next (Engine settings), so no request is retired or admitted between a
  step's planning and its commit, and the rows a step commits are the incarnations it planned.
  Admission generations guard the host records that outlive a step: a file's buffer and queue
  count (the `transcription` design) and the model's session registry, which resolves a status
  report whose admission generation is not its request's current one as a failure, accepting no
  queue count and publishing no progress from it.
- **Projection.** The model's projection of the views into named state tensors
  (`project_nemotron_state_pools`) takes the views and the model's `StateLayout` and reads nothing
  else.
- **Recorded prompt.** Core sends no token IDs for scheduled cached requests unless pipeline
  parallelism is on, which the plugin refuses. The model state therefore records each request's
  prompt (`prefill_token_ids`) at its latest `add_request` and reads a row's scheduled token from
  that record at the row's `num_computed_tokens`. A stream's first input arrives with its prompt
  from position zero. Every KV-cache group of this model is non-recomputable, so core compacts a
  stream's history: when it folds a later input into the stream, it drops the earlier tokens and
  the last sampled token (the park), keeps one frontier token followed by the new input as the
  prompt, and sets `num_computed_tokens` to 1. The recorded prompt is then that compacted prompt,
  and the row reads the new input from position 1. A position past the recorded prompt holds a
  token the sampler produced on the device; the host does not read it (that would synchronize
  with the device), and no sampled token is a CHUNK carrier, so the row is planned as a non-CHUNK
  row. A scheduled window that crosses the end of the recorded prompt fails the row at planning
  (Model-state interface, Error boundary). The record is dropped when the request is retired.
  For a file request this classifies only the scheduled token's origin: the file step rule (`transcription`, FILE-ROW-002) decides whether
  the row executes as CHUNK, before transaction classification.
- **File requests.** When `add_request` delivers a request whose scheduled token is the file
  carrier, the request is a *file request*, and the model state takes the file's samples and
  header from the `NewRequestData` and host data only (the `transcription` design, Request-owned
  host buffer). Every other request is a *stream request*.

### Endpoint policy

A request carries an endpoint policy in its sampling parameters: `extra_args["endpoint_policy"]`, a
mapping with `mode` (`"disabled"` or `"greedy_blank"`), `threshold_frames` and `residue_frames`
(non-negative integers). Core passes a request's sampling parameters to the model state with the
request (`NewRequestData.sampling_params`), so the model reads the policy when core delivers the
request and again at each stream turn core delivers as a new request. A request without the key
runs with endpointing disabled. A malformed policy (an unknown mode, or a count that is not a
non-negative integer) is the request's own fault: it fails that request alone (REG-STATE-015). The
plugin's own paths set no policy (the session factory, `buffer_realtime_audio` and the file prompt;
SESS-EP-001), so only a direct engine client sets one.

## Supported tasks

The model class declares `supports_realtime`, `SupportsTranscription` with
`supports_transcription_only = False`, and `SupportsPersistentState` with three settings that
core reads from the class:
- **`persistent_state_reads_positions = False`.** The forward discards `positions` and reads every
  other value in the token-index space of the request's recorded prompt. Core compacts this
  model's history whatever the setting, because its only KV-cache group is non-recomputable; with
  the setting false, compaction adds no position offset. True would add every dropped token to the
  request's position offset, which core counts against `max_model_len`, so a long session would
  end with a length cap for no gain.
- **`persistent_state_accepts_synthetic_warmup = False`.** The model state cannot serve core's
  warmup requests (Engine settings, Synthetic warmup).
- **`persistent_state_resumable_stops = frozenset({"eos"})`.** The park token is the tokenizer's
  only EOS and ends every unit, so a stream resumes only at a stop-free EOS. A unit that reaches
  its `max_tokens` without parking has broken the model's per-unit emission bound
  (`realtime_max_tokens`), and ends the stream; making that stop resumable would fold the next
  input into a stream whose last unit never parked.

Core would also infer `generate` from the presence of `compute_logits`, so the model's model
state overrides `get_supported_generation_tasks()` to return exactly
`("transcription", "realtime")`. A frontend requiring `realtime` loads; one that needs
complete-file transcription checks for `transcription`.

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
| `vllm.v1.kv_cache_interface` | `KVCacheConfig`, `KVCacheLayerConfig`, `MambaSpec` |
| `vllm.model_executor.layers.mamba.abstract` | `MambaBase` |
| `vllm.v1.core.single_type_kv_cache_manager` | `MambaManager`, with the `recomputable` parameter of its `layer_config` |
| `vllm.v1.attention.backends.registry` | `MambaAttentionBackendEnum`, with its member `PERSISTENT_STATE` |
| `vllm.model_executor.models.interfaces` | `SupportsPersistentState` |
| `vllm.v1.worker.gpu.input_batch` | `InputBatch` |
| `vllm.v1.worker.gpu.model_states.interface` | `ModelState`, with its hooks `prepare_execution_memory`, `needs_cudagraph_capture`, `capture_cudagraphs`, `clear_cudagraphs`, `post_warmup` and `shutdown`, its `get_additional_cg_support`, its `take_failed_requests` and, conditionally, its property `state_group_layouts` |
| `vllm.v1.worker.gpu.states` | `RequestState` |
| `vllm.v1.worker.utils` | `AttentionGroup`; conditionally, the attributes `StateGroupLayout.layers` and `StateComponentLayout.tensor` of the layouts the state-group layout accessor returns, read without importing either class |
| `vllm.multimodal` | `MULTIMODAL_REGISTRY` |
| `vllm.v1.attention.backends.utils` | `NULL_BLOCK_ID` |
| `vllm.v1.worker.gpu.sample.output` | `SamplerOutput` |
| `vllm.distributed.parallel_state` | `graph_capture` |
| `vllm.forward_context` | `BatchDescriptor`, `set_forward_context` |
| `vllm.platforms` | `current_platform` (its static graph wrapper class) |
| `vllm.utils.import_utils` | `resolve_obj_by_qualname` |
| `vllm.compilation.cudagraph_pool` | `capture_pool` |
| `vllm.envs` | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` |
| `vllm.utils.torch_utils` | `direct_register_custom_op` |
| `vllm.model_executor.determinism.batch_invariant` | the batch-invariant matmul used by the batch-invariant mode |

Three groups of names are conditional. The graph-runtime rows (`graph_capture`, `BatchDescriptor`,
`set_forward_context`, `current_platform`, `resolve_obj_by_qualname`, `capture_pool` and
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`) are imported when graphed execution is configured; the
batch-invariant row when that mode is; and the transaction-graph names (`state_group_layouts` and
the two layout attributes) are checked at model construction while transaction graphs are on,
where a missing one raises naming it (REG-018). The import-time check covers the rest, the six
hooks of `ModelState`, its `get_additional_cg_support` and `take_failed_requests`, the
`recomputable` parameter and the `PERSISTENT_STATE` member included. A property such as
`state_group_layouts` counts as defined; the check does not require it to be callable. vLLM's model-registration, config
parser, tokenizer and renderer registries, model configuration hooks (`VerifyAndUpdateConfig`,
`MODELS_CONFIG_MAP`), speech-to-text configuration, multimodal processing, `SupportsTranscription`,
logging and Hub file helpers, which out-of-tree models use, are outside this table, as is the public
client surface that the session and `register()` use (`AsyncLLM`, `StreamingInput`, `SamplingParams`,
`RequestOutputKind`, and `StreamAdmissionError` and `EngineDeadError` from
`vllm.v1.engine.exceptions`; see the HLD's Engine surface constraint). Adding a dependency means
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
| Where a request carries its endpoint policy | `SamplingParams.extra_args["endpoint_policy"]` | A request field core does not carry to the model state (`additional_information`); a plugin-owned side channel | Core's `NewRequestData` carries the sampling parameters to the model state, and `extra_args` is the sampling parameters' generic channel for model-specific options; a field core does not populate cannot reach the model |
| How the card becomes a served config | A registered config parser selected by `--config-format nemotron_asr_streaming` | `AutoConfig` registration for `nemotron3_5_asr`; `--hf-overrides` alone | Registering the card's `model_type` would claim the offline plugin's configuration; overrides apply after the configuration is built, too late for translation |
| Where launch options are resolved | Post-override validation of the final configuration; the parser reads only the checkpoint | Reading `hf_overrides` inside the parser; a plugin-specific environment variable | Core gives the parser only flat overrides, applies dictionary overrides after it, and also calls it with none when deriving a generation configuration |
| Control tokens in the tokenizer | A registered tokenizer mode that appends them as special tokens, with park as the tokenizer's EOS | Publishing a modified tokenizer; leaving them unknown | Core detokenizes every token; unknown IDs break detokenization and unmarked ones appear in text; the card must stay usable unmodified |
| Context ceiling | Translation sets `max_position_embeddings` to 2^24 | Requiring `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | The model has no positional limit; long files need a context far above the card's 32,768 |
| Where model execution memory is prepared | Long-lived working memory in `prepare_execution_memory`; the profile transaction on core's first profiling dummy step (classified in `prepare_inputs`); segment graphs on that step under `--enforce-eager` and in `capture_cudagraphs` under the hook configuration | `initialize_kv_cache` or first serving step: graphs created after core sized the KV cache, so their memory is unbudgeted. Capturing in `prepare_execution_memory`: the hook's contract forbids capture there. Capturing on the profiling step under the hook configuration too: core's estimate would not see those graphs as graph memory, and the profiling and serving captures would differ | Each stage runs where #67583 budgets it. A failed preparation ends worker startup and the engine is restarted: partial preparation is not safely retryable in the worker, and no serving state outlives the failure |
| Served geometry | One per server, fixed at startup: `num_lookahead_tokens` when given, else the card's `default_num_lookahead_tokens`; any of the five implemented lookaheads; `supported_num_lookahead_tokens` refused | A served set with a default (every geometry warmed and captured, multiplying graph memory and startup time); requiring the override (ignores what the checkpoint declares); ignoring `supported_num_lookahead_tokens` silently (an old serve command would start at a geometry its operator did not choose); only the card's four lookaheads (drops the tested 160 ms geometry) | One geometry keeps graph memory and capture time to one set of segment graphs, and matches how a Hugging Face Transformers model fixes its lookahead |
| Default segment arms | `eager-graphed` encoder and `dense-graphed` decoder, declared at translation; overrides kept | The eager arms by default; the compiled `dense-graphed` encoder | The pair measured fastest with equal transcripts (HLD, Key Design Decisions); the eager arms stay selectable as the reference |
| Arms under the hook configuration | The `eager-graphed` encoder and `dense-graphed` decoder (and transaction graphs); the compiled `dense-graphed` encoder and CHUNK-bucket graphs refused there | Capture every graphed arm in the hook; capture the unsupported ones on the profiling step | Capture of the compiled encoder and of CHUNK-bucket graphs through the hook is not yet specified or qualified, and the HLD forbids profiling-step capture in the hook configuration; both stay available under `--enforce-eager`. The refusal holds until their hook capture is specified and measured (Future roadmap 4); nothing technical prevents it |
| Segment-graph memory under the hook | Core's estimate through the profiling capture, a flat charged margin per call and an aggregate serving-over-profiling check, the same procedure as the transaction-graph cells | A per-graph comparison; a warning only | Core's charge is aggregate and allocator rounding arises per segment over the call; a warning would serve with uncharged memory where core estimated |
| Graphed arms without static graphs | A stand-in wrapper that runs each entry eagerly, keeping the graphed arms' inventory and serving path | Refusing graphed arms off CUDA; resolving the default arms to the eager arms there | The card's default arms are graphed, so a refusal would fail every non-CUDA run of the default configuration, and silently switching arms would leave the graphed staging and checks untested off CUDA; the stand-in keeps one path whose outputs equal the eager arms' |
| Core's runner graphs over the forward | None: the model state reports `AttentionCGSupport.NEVER`, core reduces every graphed mode it can to NONE, and the plugin refuses FULL and a piecewise mode under breakable graphs | Refuse every graphed mode; register the stateful segment operations as splitting ops; breakable graphs with host planning inside break operations; core full graphs with the model running eagerly during core's capture | Refusing would make every serve command without runner options name `-cc.cudagraph_mode none` while core can already reduce the mode itself, as it does for encoder-only model states. Splitting ops split a compiled graph, and the forward is not traceable; the code between its stateful segments is index uploads and a projection. Inside a breakable graph the whole forward would be one break operation, so core would capture nothing of value until planning and staging move out of the forward (Future roadmap 5). Core's full graphs replay without calling the forward, so its host planning would not run. The model's own graphs, captured through the hook, already key on its step shapes |
| Compilation modes | Accept NONE, `DYNAMO_TRACE_ONCE` and `VLLM_COMPILE`, which compile nothing in an undecorated model, and a requested `STOCK_TORCH_COMPILE` that core sets to NONE; refuse `STOCK_TORCH_COMPILE` that would reach the runner | Require compilation off; accept every mode | The model has no `CustomOp` subclass, whose dispatch the compilation mode's `custom_ops` selects, so a compilation mode changes no operator dispatch in the model; `STOCK_TORCH_COMPILE` compiles the whole model with `fullgraph=True`, which a forward with host reads cannot pass |
| Asynchronous scheduling | Accepted while transaction graphs are off; refused while they are on (TXG-CFG-017) | Refuse it always; accept it with transaction graphs too | Core's per-request barrier for state models runs each stream one step at a time, which is the property the model needs; the model reads commit status in `postprocess_state`, before the next step executes. Transaction graphs' baseline `B_hook` and their qualification run synchronously, and the worker's call order under asynchronous scheduling is unverified with cells |
| Core hooks | Required in every configuration; a core without them is refused | Detect them and fall back to profiling-step capture | One core API to test (HLD, Key Design Decisions, Core API) |
| Load-time checks | The state-layer surface in `register()` (the six model-state hooks, `take_failed_requests`, the `recomputable` parameter, `PERSISTENT_STATE` and `SupportsPersistentState`), none of which needs a model class module; every adapter symbol at model import | All checks in `register()`; only the hooks and `recomputable` in `register()`; a check of client calls in `register()` | `register()` and the session factory do not import the model class modules, so loading the plugin stays cheap, and a core that lacks any part of the state layer fails at plugin load, before model configuration; core's speech-to-text serving imports the model class in the API server when it starts, so importing it initializes no device and loads no weights. The target core admits a stream without a client call to check, and the session factory checks its own calls |
| Whether the forward reads positions | `persistent_state_reads_positions = False`, declared | Core's default, True | The forward discards `positions`. Core compacts this model's history either way, because its only KV-cache group is non-recomputable; with True, compaction adds every dropped token to the request's position offset, which core counts against `max_model_len`, so a long session would end with a length cap for no gain |
| Core's synthetic warmup | Declined: `persistent_state_accepts_synthetic_warmup = False` | Core's default, True; recognizing warmup requests in the model state and serving them | Warmup requests schedule two tokens with non-greedy sampling, which the model state refuses, and they run by default on CUDA, where the extensible KV cache defaults on, even under `--enforce-eager`. No contract names a warmup request, so the model state could recognize one only by guessing. The cost is JIT compilation on the first serving steps |
| Resumable stops | `persistent_state_resumable_stops = frozenset({"eos"})`, declared | `{"eos", "max_tokens"}`; `{"max_tokens"}` alone or an empty set | The park token is the only EOS and ends every unit. A resumable `max_tokens` stop would hide a unit that never parked and leave the session waiting for a park; without `"eos"`, every park would end the stream after one unit |
| State group type | `MambaSpec` with `mamba_type` `PERSISTENT_STATE` | `SHORT_CONV`, the short-convolution type with the same KV-cache shape handling; another Mamba type | Core refuses a group its tokens cannot rebuild unless its spec is a `MambaSpec` of `mamba_type` `PERSISTENT_STATE`. The model builds no attention metadata, so the backend the type selects changes nothing else for it |
| Per-request failure | `postprocess_state` records the step's failed rows and `take_failed_requests` returns them, so core finishes each failed request alone with finish reason `error` | Raise from `postprocess_state`; retry or roll back the row in the model | A raise ends the engine, and with it every stream on the server, for one row's failure. A retry or rollback cannot recover a row: a stream row's status is final, and a file row's position has already advanced (the `transcription` design). Core's channel needs no host synchronization beyond the wait `postprocess_state` already makes |
| Faults a request's inputs cause | Fail the request alone at planning, through the planning-failed set and `take_failed_requests`; raise only for a violation of core's contract with the model state; a fault no list names takes its class by whether a request's input can cause it | Raise for every fault; fail a contract violation per request too; raise for every unlisted fault | A direct engine client can send a multi-token prompt, a unit at another geometry or a malformed file item, and a raise would end every stream on the server for one client's input. A contract violation means core and the model disagree about a request's state, so no outcome confined to that request is safe. Raising for every unlisted fault would end the server for an input the lists missed; the class test needs no list to be complete |
| Observing the release of the profiling KV cache | A `StorageWeakRef` taken at the profiling binding, expired at the serving binding (REG-STATE-019), in an arm under `--num-gpu-blocks-override` with transaction graphs on under the restricted startup and an arm under a profiled budget with them off | Device and allocator memory readings after core's teardown; the override arm with transaction graphs off | The only model-state call between core's teardown and the serving KV-cache allocation is `get_additional_cg_support()`, a capability query that is neither a release point nor a measurement point, so the plugin has no measurement point there without a core change; a reading at the serving binding, which follows the allocation, mixes both caches. With transaction graphs on, the override arm also exercises core's state-group layout and the profiling cells, which read the profiling KV cache; the profiled-budget arm runs with them off because TXG-CFG-019 refuses that startup otherwise |
| Rows that schedule more than one token | Fail the row's request alone at planning (REG-STATE-014), through the planning-failed set and `take_failed_requests`; the step's other rows run at their own token positions | Sample the row's last token and drop the others; run the row's tokens in order as several stream units, as vLLM's and SGLang's Mamba prefill scan a multi-token row from the request's state | Each token of this model's stream is one unit of audio, and the design keeps per-step work and per-request state constant: one unit per request per step, so a step's cost does not depend on what one client sends. A multi-token row would let one request's work grow within a step. Sampling the last token and dropping the rest discards audio and returns a wrong transcript with no error. Running the tokens as several units costs `k` dependent encoder and decoder passes for a `k`-unit row, which every co-batched row waits for; it needs the intermediate units' labels to pass through the model's own output queue, since core samples one position per row; and the transaction-graph cells and their per-unit checks (TXG-TXN-010, TXG-EXEC-007) would address units, not rows. Plugin-packed input never schedules a multi-token row. A consumer that needs multi-token input uses a design that re-prefills or scans history, such as Mamba prefill or bounded sessions. Running the tokens as several units is Future roadmap 14 |
| State-layer view lifetime | Views held by reference in a cache keyed on the bound `kv_cache` object, dropped at a binding, at an identity mismatch and in `clear_cudagraphs` (State-slot access, View lifetime) | Derive the views on every read; hold weak references to the bound tensors; keep the views as plain attributes replaced at each binding | Deriving on every read needs no hook but costs about 2.7 µs per read, and weak references about 3.8 µs, against about 0.02 µs for the identity comparison (CPU measurements). Plain attributes hold the profiling KV cache until the serving binding, after core has allocated the serving KV cache |
| Envelope-header faults | Each fault in one of REG-STATE-015's two classes, by whether a request's input can cause it (Model-state interface, Error boundary) | Raise for every envelope fault; fail the request for every one | Core lets a direct engine client bypass the processor, so a malformed item reaches planning, and a raise would end every stream on the server. A fault that no request input can cause means core and the model disagree about the step, which no per-request outcome makes safe |
| Retirement of a finished request | A file request within `remove_request`; a stream request at the next model-state call | Every request at the next model-state call; every request within `remove_request` | A file is one request that core never continues, so nothing after its removal reads its buffer, and a deferred retirement would keep the last file's buffer (64,000 bytes per second of audio) on an idle engine. A stream's `remove_request` may precede its next turn's `add_request`, which keeps the admission generation, so retiring it at once would lose the generation a continuing turn needs; the deferred stream record holds no host buffer |
| Settings core refuses before the configuration hook | Core's message, which names one setting (Engine settings, Core's checks first); the plugin keeps its own rules for them | Ask core to defer its check to the hook; drop the plugin's rules for those settings | Core's check runs while it builds the engine configuration, before the hook, and its message names the setting to remove, which is the remedy. The plugin's rules still list every failed check when the hook is called directly, and keep the plugin's refusals independent of core's list |
| `generate` task | Not advertised (model-state override) | Advertise for compatibility | The model emits transcript labels, not text generation |

## References

- vLLM config parsers: `vllm/transformers_utils/config.py` (`register_config_parser`, the parser
  call and override application), `vllm/transformers_utils/config_parser_base.py`.
- Model configuration hooks: `vllm/model_executor/models/config.py` (`MODELS_CONFIG_MAP`).
- Supported tasks, model-state hooks and `take_failed_requests`:
  `vllm/v1/worker/gpu/model_states/interface.py` (hooks from vllm#67583); graph-pool routing:
  `vllm/compilation/cudagraph_pool.py`.
- The target core's state layer (`pdtgct/vllm` at `ddf6db2ab8`): the state-group layout accessor
  `ModelState.state_group_layouts` in `vllm/v1/worker/gpu/model_states/interface.py`, with
  `StateGroupLayout` and `StateComponentLayout` in `vllm/v1/worker/utils.py`; the profiling KV
  cache's size in `vllm/v1/worker/gpu/cudagraph_utils.py` and the block-count override in
  `vllm/v1/core/kv_cache_planning.py`; `SupportsPersistentState` in
  `vllm/model_executor/models/interfaces.py`; the state-group check
  `verify_persistent_state_groups` in `vllm/v1/core/kv_cache_utils.py`; `MambaSpec` and its
  `initial_values` in `vllm/v1/kv_cache_interface.py`; `StatePageInitializer` in
  `vllm/v1/worker/utils.py`; the synthetic-warmup decision in `vllm/v1/worker/gpu_worker.py`;
  compaction and the failed-request path in `vllm/v1/core/sched/scheduler.py`; the
  persistent-state configuration check `_verify_persistent_state_config` in
  `vllm/config/vllm.py`; the request add and remove order and `take_failed_requests` call in
  `vllm/v1/worker/gpu/model_runner.py`; fixed block offsets and the one growth of the extensible
  KV cache in `vllm/v1/worker/extensible_kv_cache.py` and `vllm/v1/engine/core.py`.
- Plugin loading: `vllm/plugins/__init__.py`.
- Card translation: `nemotron_asr/configuration_nemotron_asr.py`; geometries:
  `nemotron_asr/manifests.py` and `lookahead.py`.

## Future roadmap

Each item names what is deferred, the trigger that reopens it and the shape of the change.

1. **Version gate.** A version check for clean vLLM releases. Trigger: RFC 1's state layer
   ships in a release. Shape: `register()` checks the release version; until then the core check
   is the gate.
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
   compilation and warm-up in `prepare_execution_memory`, profiling capture over core's minimal
   profiling KV cache, release in `clear_cudagraphs`, serving capture into core's graph pool, the
   same memory accounting and serving-over-profiling check; the hook-configuration refusal and the arm table
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
8. **Transaction graphs beyond the restricted startup.** While transaction graphs are on, the
   plugin accepts only synchronous scheduling, an explicit `cudagraph_mode` NONE with compilation
   mode NONE, and an explicit block count (`--num-gpu-blocks-override`) or KV budget
   (`--kv-cache-memory-bytes`) (Engine settings). The block-count condition exists because core's
   minimal profiling KV cache holds `min(max_num_seqs, max_cudagraph_capture_size) or 1` blocks,
   one under an explicit NONE, too few for a capture over `N` state pages, and because the
   extensible KV cache backs only its first blocks when the serving capture runs and grows without
   refreshing `state_group_layouts`. Triggers: core sizes its profiling KV cache for a model state
   that captures over state pages, and backs the state group's pages before the model-state
   capture or refreshes the layouts when the extensible KV cache grows (the condition on block
   count or budget then drops); the transaction-graph qualification at the reduced modes, whose
   profiling KV caches hold more state pages, and with asynchronous scheduling (those conditions
   then drop). Shape: the `transaction-graph` design's Future roadmap; here, REG-ENG-001,
   REG-ENG-011 and REG-ENG-026 lose their transaction-graph exceptions and REG-ENG-005 its
   narrowed remedies as each condition drops.
9. **Native burst on the state layer, with asynchronous scheduling.** Native burst is refused on
   this core: its history check requires a turn with `num_computed_tokens` 0 whose prompt is its
   whole history, and core delivers each later input compacted, with `num_computed_tokens` 1
   (State-slot access, Recorded prompt). Trigger: a native-burst history rule for compacted
   turns is specified. Shape: the history check reads the compacted prompt; decide whether the
   read-ahead sampler composes with core's per-request barrier; refuse the pair until a GPU tier
   shows equal transcripts.
10. **Overlap under asynchronous scheduling.** The commit sink waits for each step's commit event
    in `postprocess_state`, so asynchronous scheduling overlaps little for this model. Trigger: a
    latency A/B that shows the wait on the critical path. Shape: read the commit status one step
    later, from a ring of pinned buffers.
11. **Page initial values in core.** The layer's `initialize_pages` writes each fresh page after
    core's `StatePageInitializer` has zeroed it. Every descriptor's initializer is zero or one
    scalar (the blank label), and the layout has no bytes outside its states, so the spec's
    `initial_values`, one scalar per state, can carry the layout's initializers. Trigger: the next
    change to the state layer, or a measurement that shows the second write on a fresh row's first
    step. Shape: the layer builds `initial_values` from the layout's descriptors; `initialize_pages`
    and its call in the forward go; REG-STATE-010 names core's initialization; a test shows equal
    pages.
12. **Memory contributors in core's failures.** The target core reads no list of the options that
    enlarge the model's memory (Execution memory, Memory diagnostics). Trigger: core accepts such
    a list from a model. Shape: the model passes `max_num_seqs` and the two segment arms when
    graphed execution is configured, `chunk_bucket_graph_populations` when it is non-empty and
    `transaction_graph_populations` while transaction graphs are on; core names them in a failure
    of profiling, capture or capacity planning.
13. **Session factory on the target core.** The session factory is not available on the target
    core (HLD, Approach, section 2). Trigger and shape: the HLD's and the `session` design's Future
    roadmap. Registration's part: `register()` keeps checking no client call, and the
    `StreamAdmissionError` import stays on `vllm.v1.engine.exceptions`.
14. **Multi-unit rows.** A row that schedules `k > 1` tokens fails its request (REG-STATE-014).
    The alternative, not built: run the row's tokens in order as `k` stream units, as vLLM and
    SGLang scan a multi-token prefill row from the request's state and write its final state once.
    Trigger: a client or frontend needs to submit several units in one input, such as a batch
    upload into a live stream or a catch-up after a stall. Shape and costs: `k` dependent encoder
    and decoder passes per step for that row, which every co-batched row waits for, because the
    cache-aware encoder needs chunk `t`'s cache for chunk `t + 1` and the plugin has no
    variable-length kernel; the labels of the intermediate units pass through the model's own
    output queue within its capacity and emission bounds, since core samples one position per row;
    each unit keeps its own header checks and partial-failure semantics inside the row; and the
    transaction graphs address units (TXG-TXN-010, TXG-EXEC-007 per unit) or keep falling back
    with `multi-token` (the `transaction-graph` design, Future roadmap).
