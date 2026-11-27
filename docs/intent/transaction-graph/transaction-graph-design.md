---
parent: high-level-design
prefix: TXG
---

# Transaction graph

## Context and Design Philosophy

A streaming step of this model is one model-defined transaction over resident state slots
(Transaction structure). The encoder and decoder replay their own segment graphs, yet the step
stays bound by host dispatch around them: per-operation dispatch, kernel launches and
caching-allocator work. This component replays the rest of the device body and its scatters from
CUDA graphs, split by row kind: a step's CHUNK rows replay a **chunk cell** captured at exactly
their count, and its other rows (REPLAY, park echo, FLUSH) an **echo cell** captured
at exactly their count. A step that its cells do not cover runs wholly as the option-off
transaction, which is the reference, and is counted.

The option is `transaction_graph_populations`, off by default. The cells are captured through
vllm#67583's model-state capture hook (`capture_cudagraphs`), into core's graph pool, on core's
capture stream, while core's runner captures no graph over the forward (its resolved
`cudagraph_mode` is NONE). The model keeps
no graph pool and no memory admission of its own. On the target core the option starts only
under a restricted startup: synchronous scheduling and an explicit NONE runner and compilation
mode (Configuration checks).

**Interface.** The rules of this design use the target core's state layer, `pdtgct/vllm` at
`cbf13fed4a` (the `registration` design, State-slot access): the state layer's views, which core
binds through `MambaBase.bind_kv_cache` and the layer exposes as `component_pools`, views of the
bound storage that the layer drops when core rebinds and in `clear_cudagraphs` (REG-STATE-018);
the model state's step classification (REG-MEM-001) and its record of each request's slot
(REG-STATE-002); the model state's failure channel `take_failed_requests` (REG-MS-003,
REG-MS-004); and core's state-group layout accessor, `ModelState.state_group_layouts`, which core
sets after it binds a KV cache and before `initialize_kv_cache` and `capture_cudagraphs`, keeps
unchanged until it releases that cache (its growth rebinds the layer's views over the same pages
with no more pages than the layout's *page count*, the first dimension of its tensors), and
clears on release (`vllm/v1/worker/gpu/model_states/interface.py`); at every capture core
guarantees memory behind block 0, the only page it guarantees backed, and under the extensible
KV cache other pages may have no memory until it backs every page it can assign, before the
first request. Core keeps a held request's page at its block ID and address until the request's
terminal finish, so a cell may address a page by block ID. A cell therefore addresses only
block 0 at capture and reaches each held page by the block ID the step arena carries.

**Evidence.** The quantitative figures in this design are unpublished project measurements; the
tier 2 and tier 5 results published with the plugin replace them. A whole-transaction prototype
on an earlier in-tree copy of this model, on an A40 at 80 ms, FP32 with TF32 off, with the
`eager-graphed` encoder and `dense-graphed` decoder, native burst off and homogeneous continuing
CHUNK steps only, measured:

| Population | Option-off step | Replayed step | Saving |
|-----------:|----------------:|--------------:|-------:|
| 1  | 22.6 ms | 16.2 ms | 6.4 ms |
| 31 | 46.0 ms | 40.0 ms | 6.0 ms |
| 63 | 70.8 ms | 65.6 ms | 5.2 ms |

Outputs and pages were bitwise equal to the option-off path, with zero fallbacks. The figures show
that the saving exists on homogeneous steps; they do not predict its size on split steps, at
160 ms or in this plugin.

### Lean

**Exact counts before coverage.** A cell replays only rows whose kind, count and signature match
it exactly; any other step runs as the option-off transaction and is counted. Coverage gaps close
by capturing cells, never by loosening the match or padding rows into a larger cell. The decoder
already pads its rows to a tier, but identically in the cell and in the option-off transaction,
because the tier is a function of the CHUNK count. *Opposite:* pad rows to population buckets and
accept a broader exactness proof, which buys coverage at the cost of an oracle that the option-off
transaction cannot check row for row.

Bitwise equality with the option-off transaction of the same row plan is an acceptance
requirement for every covered step. It is not a lean, because nothing trades against it.

## Option and configuration

### The option

`transaction_graph_populations` is a key of the served configuration, given as a flat
`--hf-overrides` key (or stored by an already-translated configuration). Its value is a list of
distinct step populations that the deployment expects. An absent key or exactly an empty list is
off. Any other value turns the option on and is validated (Configuration checks) before `N` is
computed, so a malformed value such as `63` or `null` is refused, never silently off. One
function, `transaction_graphs_on`, in the module of the configuration checks, implements this
predicate; `registration`'s checks, hooks, dependency gate and memory contributors call it too,
so no component reads a value such as `{}` as off by its own test.
The largest entry, `N`, fixes the cell inventory: a chunk cell and an echo cell at every count
from 1 to `N`. The other entries change no capture; they are validated and recorded in the
configuration identity, so `[63]` and `[1, 31, 63]` capture the same cells. Example (80 ms):

```json
{"num_lookahead_tokens": 0, "transaction_graph_populations": [64]}
```

A **population** is the number of real rows in a step: stream rows and whole-file rows counted
together. Slots with no row in the step (parked, idle or never-fed sessions) do not count. A
step's **CHUNK count** `K` is its number of CHUNK rows; its **echo count** is `E = P − K`.

A step's **row plan** is every row the runner scheduled for model execution, at its scheduled
position, with its binding and host-produced vectors. Core lays a step's tokens on one axis: row
`i` owns the **token positions** `query_start_loc[i]` to `query_start_loc[i+1] − 1`, and core
samples each row at its last position. Row `i` reads only the runner's inputs at its own token
positions and writes only the runner-facing output there, which has one row per scheduled token
in the runner's order, so no input or output crosses between rows. Row `i`'s **step position** is
its index `i` in the row plan, from 0 to `P − 1`; the step arena, the step static inputs and the
step outputs address rows by it. Every unit the plugin packs is one token (`session`, Packer),
so on plugin-packed input every step is a **one-token step**, in which row `i`'s one token
position is `i`, equal to its step position; in any other step the two can differ. Only a direct
engine client can schedule a **multi-token row**
(a prompt or streaming input of several tokens); planning fails it (`registration`,
REG-STATE-014), and the step's other rows still read and write at their own positions. The
**row-failure path** fails a row at planning: a multi-token row (`registration`, REG-STATE-014),
a foreign-geometry row (`session`, SESS-MODEL-001 and SESS-MODEL-002; `transcription`,
FILE-PROMPT-007), a continuing file row whose queue count is not known (FILE-QC-010), or a row
with another fault that its request's own inputs cause (`registration`, REG-STATE-008 and
REG-STATE-015).
Such a **planning-failed row** stays in the row plan at its position, bound as a non-CHUNK row with a planned-rejection bit that the body
turns into a nonzero status whatever the row's kind, freshness or token; it counts in `P` and `E`,
never in `K`. Its output rows are the rejected-row output (the projection row the transaction
writes for any rejected row), one at each of its token positions, its masked scatter leaves its pages untouched, and the model state
records its request at planning and fails it in `postprocess_state` whether or not a status is
collected for it. A step whose every row failed at planning (an **all-failed step**) is a bypass
outcome, not an empty step: the model state first checks the failed latch (a latched worker
raises, as for any step that schedules a row), then runs no transaction, returns the
rejected-row output for each row and fails the requests in `postprocess_state`. It replays no
cell, reserves nothing, counts no fallback and adds one `(0, P)` entry to the `(K, E)` histogram.
The bypass makes no verification check: post-warmup verification protects cell replay, which the
bypass never reaches, so an unlatched worker takes it whether or not verification has passed.
Coverage, classification, the arena upload and the reservation apply only to steps that reach
transaction execution; in a mixed step the planning-failed rows count in `E`, and the inventory stays `2N`
cells.

### Configuration checks

With the option on, the configuration must satisfy every condition below. A failure of a
transaction-graph-specific condition (Populations, Admission limit, Block count, Geometry, Native
burst, Segment arms, `enforce_eager` in Runner, State layer, Scheduling and Runner modes) is a
configuration error that names `transaction_graph_populations`, the condition, the rejected
value and the remedy. The shared runner and platform conditions are reported as `registration` states them
(REG-ENG-005, REG-ENG-006). Scheduling and Runner modes form the **restricted startup**.

| Condition | Rule |
| --- | --- |
| Populations | A list of `int` values (not `bool`), distinct, each in `1 … min(max_num_seqs, max_num_batched_tokens)`. A limit core has not yet resolved when the configuration hook runs bounds nothing there; model-state construction applies the full bound. |
| Admission limit | Every entry is at most the number of requests core's scheduler admits into RUNNING: `max_num_active_seqs` when set (`--max-num-active-seqs`), otherwise `max_num_seqs`. Core never schedules a step with more rows than that, so a larger entry declares a population that can never run and cells that can never replay. Remedies: a largest population of at most the limit, or a larger `--max-num-active-seqs` (at most `max_num_seqs`) or none. Bounded as Populations is when the configuration hook runs before the limit is resolved. |
| Block count | Under `num_gpu_blocks_override`, `override − 1` (the non-null pages) is at least the admission limit, and so at least every entry. Core's scheduler refuses to start when the cache holds fewer state pages (`num_blocks − 1`, at one block per request) than requests that can be active, but core builds it after both captures, so an override of `N + 1` with `N` below the limit would capture `2N` cells twice and then fail. Remedies: a `--num-gpu-blocks-override` of at least the admission limit plus one, or a lower admission limit with the largest population at most it. Before the limit is resolved, the configuration hook compares the override with the entries only. Under every other KV-cache sizing (a fixed `kv_cache_memory_bytes`, a profiled budget, or the extensible KV cache) the page count is known only at the serving capture, where the pool-access check makes the same comparison with the layout's page count (Rules of the capture call); under the extensible KV cache that count is the reservation capacity, and core's scheduler, built after the extension, refuses a final count too small for the requests that can be active, naming a lower `--max-num-seqs` or a larger cache (`vllm/v1/core/sched/scheduler.py:155-170`). |
| Geometry | The server's one geometry, as `registration` resolves it at startup (the `num_lookahead_tokens` override, else the checkpoint's default lookahead), is lookahead 0 or 1 (80 or 160 ms). Lookahead 3 (320 ms), the published checkpoint's default, is refused with the remedy `num_lookahead_tokens` 0 or 1. When `served_lookahead` raises, the condition carries its message. |
| Native burst | `experimental_native_burst` is false or absent. |
| Segment arms | The arms the model will run are the default arms: `encoder_execution_arm` `eager-graphed` and `decode_dispatch_arm` `dense-graphed`, configured or written by card translation (REG-CARD-008). An unset key resolves to an eager arm (`eager` encoder, `compact-eager` decoder) and is refused. Every other arm is refused as well, the `dense-graphed` encoder included, which `registration` also refuses under the hook configuration (REG-ENG-005); a non-empty `chunk_bucket_graph_populations` is refused there by `registration`, not by this function. |
| Runner | `enforce_eager` off. The hook configuration's `cudagraph_mode`, compilation-mode and platform conditions are `registration`'s (the `registration` design, Runner configurations), which it validates (REG-ENG-005 in the configuration hook, REG-ENG-006 at model-state construction); this function does not repeat those conditions. |
| State layer | The running core's `ModelState` defines `state_group_layouts`, as the target core does. The cells bake in state-page addresses, which the model reads from that accessor at capture (Rules of the capture call). Refused on a core without it (remedy: remove `transaction_graph_populations`). |
| Scheduling | `async_scheduling` is `False` (remedy `--no-async-scheduling`). An unset value is refused, because the target core resolves it to asynchronous scheduling. `B_hook` and the qualification run synchronously, and the worker's call order under asynchronous scheduling is unverified with cells. |
| Runner modes | `cudagraph_mode` and the compilation mode are NONE: requested explicitly in the configuration hook (an unset mode is refused), and NONE as core's configuration build left them at construction (remedy `-cc.cudagraph_mode none -cc.mode none`). The qualification covers explicit NONE only. A mode that core reduces to NONE only when it initializes the KV cache leaves the runner capturing nothing, as explicit NONE does, but no transaction-graph run has shown its served equality (Future roadmap). This condition narrows the hook configuration's; it does not replace it. |

**`--enforce-eager`.** Core runs no capture hook under `--enforce-eager`, so the option on
with `--enforce-eager` is refused, naming both. Once the option is the default, the default
resolves to off there with one INFO line naming the reason, and an explicit value (the key
present in the final configuration) is still refused.

**Core without the hooks.** `registration` refuses a core that lacks #67583's model-state hooks
(HLD Constraints, Engine); this component assumes them.

### Where the checks run

The checks are one function, `transaction_graph_config_errors`, in a module that imports no vLLM;
it returns the violated conditions and changes nothing. Whether the running core's `ModelState`
defines `state_group_layouts` (a property counts) is a capability that its callers determine
through `engine_flags` and pass in. `registration`'s post-override validation
includes its result in the single `ValueError` that lists every failed check (REG-VAL-003), so a
misconfiguration fails in the API server before any worker starts. The model state's construction
runs it again against the final engine configuration, because core resolves `cudagraph_mode` and
the compilation mode after the configuration hook.

### Option-off baseline

Every comparison in this design ("the option-off transaction", "option-off run") is against
**baseline `B_hook`**: the same engine configuration with `transaction_graph_populations` absent.
That is the hook configuration with `-cc.cudagraph_mode none`, `-cc.mode none` and
`--no-async-scheduling` (no `--enforce-eager`), with
the default segment arms captured through the hook (`eager-graphed` encoder at every CHUNK count
from 1 to `max_num_seqs`, `dense-graphed` decoder at its tiers), no CHUNK-bucket graphs, native
burst off, and the same geometry, batching limits, dtype, device and KV-cache sizing (core's
extensible KV cache, a profiled budget with it off, the same `--num-gpu-blocks-override`, or the
same `--kv-cache-memory-bytes`), so the option
changes exactly one setting. `B_hook` is defined by these runner flags, not by the README's default
serve command, which may run under `--enforce-eager` until the hook configuration meets tier 2.

## Transaction structure

The transaction has three phases with one launch boundary:

1. **Prepare** (host only). Structural preflight of the whole row plan, decode resolution for the
   CHUNK bucket, CHUNK bucket formation, per-step batch statistics and every host-only check. It
   produces every int64 and bool vector the body reads, from CPU plan authority. It reads no
   device memory and touches no resident page.
2. **Body** (device only). Gathers of every row's state (a fresh row's zero-initial state from the
   null page), the fresh book-row and file-row selects, the device role and protocol checks,
   frontend, encoder, decode, projection, row status, progress, remaining queue counts and the
   complete scatter plan, validated before any scatter. The body reads host inputs only as device
   tensors given to it; it stages nothing itself.
3. **Commit.** One composite reservation, the prevalidated predicated scatters, then the no-fail
   stage of status, progress and remaining queue counts. This is the option-off order; a covered
   step reserves before its cell launches, the cells run the body and the scatters, and the stage
   follows the last launch (Execute order).

The option-off transaction runs the phases in order over the whole row plan, uploading the host
vectors through the model's host staging. The graph path runs phase 1 over the whole plan, makes
the reservation, replays phase 2 with the scatters once per kind present, and runs the commit
stage once. Both run the same body code, with no host branch on the step's composition: a cell is
a capture of the body on a one-kind sub-plan.

## Cells, classification and the split

### Cells

A cell is `(geometry, kind, count)`. `geometry` is the index of the server's one geometry (0 for
80 ms, 1 for 160 ms; for these two the index equals the lookahead); a later release that offers
more geometries adds cells under their own index without changing the key.

| Kind | Rows it replays |
| --- | --- |
| `chunk` | CHUNK rows, fresh and continuing alike: regular chunks, final tails (including an empty final tail), silent chunks, a stream's first chunk, a file's first step |
| `echo` | every other row: REPLAY (label drain), park echo and FLUSH; a forced-EOU candidate is never replayed (Endpoint arguments, below) |

The model captures one chunk cell and one echo cell at every count from 1 to `N`: `2N` cells.

**Endpoint arguments.** The forward gives every transaction it runs on a real step the endpoint
history and book pools and the EOU token ID, with the option on or off, so the transaction always
runs with `endpoint_enabled` true, as in `B_hook`. Every cell is captured that way and runs the
*policy-free endpoint bookkeeping* exactly as the option-off transaction does: each CHUNK row's
endpoint-book `armed` and `has_output` flags follow its observed output, and each fresh row's
endpoint pages are initialized and stored, whether or not the row carries an endpoint policy.
Two row classes stay outside the cells and fall back with reason `signature`:
- **Endpoint-policy rows.** A row whose request sets an endpoint policy (`extra_args["endpoint_policy"]`, REG-STATE-020);
  a row plan has *endpointing* on when one of its rows does. Policy evaluation is elementwise in
  the body, but no plugin path sets a policy (`session`, Endpointing; `transcription`,
  FILE-ENG-002), so no qualification run exercises it.
- **Forced-EOU candidates** (field `forced_eou`). The body forces EOU for a non-CHUNK row whose
  scheduled token is the EOU token and whose book has no pending echo, an empty queue and an
  unfinished stream, also when the row carries no policy. The pending echo, queue and finished state live on the
  device, which classification does not read, so the host marks every non-CHUNK row whose
  scheduled token it records as the EOU token, a superset of the rows the body forces. A position
  past a request's recorded prompt holds a sampled token, which the host does not read; a
  policy-free row samples the EOU token only as the decision of a forced EOU, which arms its
  echo, so that token returns as a REPLAY row, never on the forced-EOU path. A row with a policy
  falls back by its policy whatever its token.

Each of these fields is part of the cell signature and false at capture (Signature). Only a direct
engine stream client produces either class; such a step runs as the option-off transaction, where
a policy-free row on the forced-EOU path is forced exactly as in `B_hook`. The eager
sub-plan identity (Split equivalence) covers forced-EOU rows with endpointing on and off.

### Coverage and classification

`classify` reads the CPU row plan and returns the cells the step needs, or the reason the step
has none. A scheduled step with at least one real row, in neither the memory-profile nor the
native-burst mode, with `K` CHUNK rows and `E` other rows is **covered** when it is a one-token
step, `K = 0` or the chunk
cell at `K` holds a graph, `E = 0` or the echo cell at `E` holds a graph, and each needed cell's
signature matches the step. An empty step or an ineligible mode is never covered. The total population need not be a declared entry: a 33-row step split 30/3 is
covered when `N ≥ 30`. A step that is not covered runs wholly as the option-off transaction and is
counted under `(reason, kind, count)`:

| Reason | Meaning |
| --- | --- |
| `multi-token` | the step holds a multi-token row, which planning failed; in an eligible mode, assigned before every other reason. The cells, the step arena and the static inputs address rows by step position, which equals the token position only in a one-token step, so the option-off transaction places the rows by their token positions instead. Only a direct engine client produces such a step |
| `chunk-uncovered` | `K > N` |
| `echo-uncovered` | `E > N` |
| `signature` | a needed cell's structural inputs differ from the captured ones (named kind and field); a step whose needed cells classification found falls back here with field `endpoint_policy` when a row carries an endpoint policy and with field `forced_eou` when a row is a forced-EOU candidate (Endpoint arguments), which only a direct engine stream client can cause; an earlier reason takes precedence |
| `multi-geometry`, `uncaptured`, `empty` | a CHUNK row of another geometry, a needed cell with no graph, or no real rows; unreachable from the production planner: it binds every foreign-geometry row, a direct engine client's included, as a planning-failed non-CHUNK row; startup captures every cell; and neither a cleanup-only step nor a step whose every row failed at planning reaches the transaction. `multi-geometry` and `empty` are reachable only through a row plan built outside the production planner. Multi-geometry classification is defensive and never authorizes execution at another geometry: the declined step runs option-off at the server's geometry, whose device geometry status rejects the row |
| `memory-profile`, `native-burst` | modes never covered; both unreachable: the profile transaction runs on a profiling dummy step, which never calls `execute`, and native burst is refused while the option is on |

`execute` counts each declined step once; the caller that then runs the option-off transaction
adds no count, and a dummy step, which never calls `execute`, is never counted.

On plugin-packed input, which carries no endpoint policy, schedules no EOU token and packs one
token per unit, no step falls back with `multi-token`, and the signature is constant after startup, so a nonzero `signature` count there is a defect to investigate, not a
tolerated rate. A direct engine stream client that sets an endpoint policy or schedules the EOU
token causes an expected `signature` fallback. The first `signature`
fallback per `(kind, field)` logs one WARNING naming the field.

With `N = min(P_max, admission limit)`, the largest `N` the configuration checks accept, every
reachable split of a one-token step that reaches transaction execution (not an all-failed step)
and whose signatures match is covered, because no step holds more rows than that `N`. On
plugin-packed input that is every split; a step with a multi-token row still falls back with
`multi-token`. A smaller `N`
trades coverage of large steps for graph memory (Fit analysis).

### Step arena, static inputs and step outputs

The binding owns, outside the graph pool, buffers that every cell reads or writes at fixed
addresses. `P_max` is the largest step population, `min(max_num_seqs, max_num_batched_tokens)`.
- **Step arena.** One device buffer with a chunk section and an echo section. Each section holds,
  for `N` rows, every int64 and bool vector its kind's body reads: the row's position in the step,
  its composed block index, `is_fresh` and the fresh row's initial book row, `is_file`, geometry,
  admitted and prior prompt rows, per-row cadence frames, the planned-rejection bit and, in the
  chunk section, the endpoint mode, threshold and residue frames that the endpoint bookkeeping
  reads (Endpoint arguments). A cell at
  count `k` reads the first `k` rows of its section. Two pinned host buffers
  alternate per step, each fenced by an event so that it is rewritten only after its previous
  upload completed. Each step packs both sections and uploads the whole arena in one
  host-to-device copy.
- **Step static inputs.** `ids_static[P_max]` and `embeds_static[P_max, H]`, `H` being the
  carrier width. Before the launches, two device copies write the step's `input_ids` and
  `inputs_embeds` into their first `P` rows (a covered step is a one-token step, so its `P` rows
  are its `P` tokens); each cell gathers its own rows by the arena's row
  positions.
- **Step outputs.** `rows_step[P_max, H]`, `status_step[P_max]`, `progress_step[P_max, 3]` and
  `remaining_step[P_max]`. Each cell writes its rows' projection rows, status, progress and
  remaining queue counts at the rows' positions in the step (an in-graph `index_copy_` by the
  arena's row positions), so every output is in row order when the launches finish.

Sharing these buffers is safe because the runner stream serializes every upload, copy, replay and
stage, and the two launches of one step touch disjoint row positions and disjoint pages.

### Execute order

`execute` runs in a fixed order:

1. the failed latch (Failure containment);
2. that post-warmup verification has passed in the worker;
3. the process constants (Process constants);
4. classification from the CPU plan, including the mode reasons;
5. phase 1 over the whole row plan, which also builds both arena sections;
6. the signature of each needed cell;
7. each needed cell's graph object;
8. arena pack and upload; static-input copies;
9. the escaped row tensor, `torch.empty((P, H))`, owned by the caller;
10. the composite reservation, after re-reading the rows' `component_pools` (REG-STATE-004);
11. the chunk cell's launch when `K > 0`, then the echo cell's launch when `E > 0`;
12. the copy of `rows_step[:P]` into the escaped tensor, the commit stage and the counters.

A call declined at step 4 or 6 increments its fallback counter and returns for the option-off
transaction. A call before verification passed raises at step 2, naming the lifecycle defect and
running neither path, without latching the binding; whether the worker survives that error is the
environment's choice, which the plugin does not guarantee. An exception at step 1, 3 or 7 is
worker-fatal; at step 4, 5, 6 or 8 to 10 it follows the option-off transaction's structural-failure
contract, cancelling any reservation held, except that a cancel that itself raises latches the
binding and is worker-fatal; from step 11 on it is worker-fatal. All
structural validation, both signatures, every allocation and the reservation therefore precede
the first launch.

### Output order, decoder tier and reservation

- **Row order.** The chunk section lists the CHUNK rows in ascending step position, which is the
  order of the option-off transaction's CHUNK bucket, so the encoder and decoder see the same
  rows in the same order. The echo section lists the other rows in ascending step position. Each
  cell writes its outputs back to the rows' step positions (Step outputs); the escaped tensor and
  the staged status are in row order.
- **Decoder tier.** The option-off transaction selects the decoder tier from the CHUNK count
  alone (the smallest captured tier at least `K`). A chunk cell is keyed by `K`, so it bakes in the
  tier the option-off transaction uses for any step with `K` CHUNK rows, with the same zero padding
  of rows `K` to the tier.
- **Composite reservation.** One reservation covers all `P` rows: the plan's bindings in row
  order, the views `progress_step[:P]` and `remaining_step[:P]`, and no capture records. It is
  made before the first launch and staged once after the last, with `status_step[:P]`; the sink's
  single in-flight reservation is never split per kind.
- **Reservation disposition.** Every reservation, a covered step's or the option-off
  transaction's, is resolved by exactly one successful stage or one cancel. The stage is no-fail
  in that it validates and allocates nothing, but a fault in its copies or its event can still
  raise. It publishes the reservation to the sink's collection only after it has issued its
  copies and recorded its completion event, and the ticket counts as resolved only once the stage
  returns. A stage that
  raises before publication therefore leaves the reservation held and cancellable, and the failure
  path's cancel releases it; a cancel after a successful stage is a no-op. Treating a stage that
  raised as resolved would leave the sink's one in-flight reservation held, and every later
  reserve would fail. The one unresolved case is a cancel that itself raises: the sink's state is then unknown,
  so the failure is worker-fatal and no later step runs on that worker.

## Split equivalence

**Claim.** For a row plan `R` with CHUNK positions `C` and other positions `O`, the option-off
transaction `T(R)` and the merge of `T(R|C)` and `T(R|O)` agree bit for bit at every row: the
projection row, the status, the progress, the remaining queue count and every resident page of the
row's slot. `R|S` is the sub-plan of the rows `S` in ascending order, over the same pools.

**Proof sketch.**
1. Phase 1 runs once on `R`. Its structural checks (indices non-null, live, in range and unique
   across all rows) hold for every subset, so the sub-plans need no check of their own. Its other
   host checks are per row, or a maximum over rows (carrier width) that the binding evaluates on
   `R`.
2. Every operation of the body's prologue (state gathers, the fresh and file selects, the role,
   echo, protocol and book-invariant checks, role assignment) is row-local: a gather by the row's
   own block index or, for a fresh row's zero-initial state, of the null page, which no
   transaction stores to (Null page); an elementwise operation along the row axis; or a reduction
   along a non-row axis (the queue slots). Restricting the rows therefore commutes with the
   prologue.
3. `T(R)` forms one CHUNK bucket, the rows `C` in ascending order at count `K`, and resolves its
   decoder from `(geometry, K)`, one ready bucket and the tier for `K`. `T(R|C)` forms the same
   bucket with the same resolver inputs. The frontend, encoder and decoder therefore receive
   identical tensors of identical shape and run the same kernels; no bucket output depends on a
   row outside the bucket. `T(R|O)` forms no bucket.
4. Projection, the projection-invariant check, endpoint observation and forced EOU are row-local,
   given the bucket outputs indexed by each CHUNK row's position.
5. Each predicated scatter writes a row's own page under its own status. Pages are unique across
   `R`, so the scatters of `C` and `O` touch disjoint pages, and their order does not matter.
6. Status, progress and remaining counts are row-local, and the commit stage takes them in row
   order.

A cell's replay equals `T` of its sub-plan (Capture procedure and Qualification), so a split
step equals `T(R)`. The argument fails if the body gains a cross-row operation (a reduction over
rows or a host branch on the step's composition), if the decoder tier is keyed by anything but
`K`, or if two rows could share a page; the tests below detect each.

**Tests.**
- *Sub-plan identity, eager.* `T(R)` equals the merge of `T(R|C)` and `T(R|O)` in every output
  and every page, at 80 and 160 ms with endpointing on and off, over: every accepted plan of 1 to
  4 rows with every assignment of each row's kind, fresh bit and file bit (4,680 plans, so CHUNK
  rows interleave with other rows in every order); every plan of 1 or 2 rows in which one row
  carries each rejection status class reachable for that row's full configuration (kind, fresh bit,
  file bit); and 500 random plans per geometry and endpointing setting with `P` uniform on 1 to 16,
  each row's kind and bits drawn independently and its status accepted with probability 3/4, else
  drawn from the classes reachable for its configuration, by a generator seeded from the
  preregistered seed list, a failing plan being reported with its seed. A rejected row comes from a
  fixture whose step reports exactly the intended status. Combinations that no fixture can produce
  are excluded and recorded: for example, a fresh CHUNK row cannot have an undrained queue (its
  initial book's queue is empty), and a fresh non-CHUNK row cannot have a pending-echo mismatch (it
  has no armed echo). Runs without graphs, on CUDA and on CPU.
- *Cell equals sub-plan.* At every count from 1 to `N`, for both kinds, a replay equals `T` of the
  same sub-plan with block indices, row orders, prompts and masks that differ from the capture
  rows.
- *Split equals whole.* On mixed steps, the two launches equal `T(R)` with the escaped rows,
  staged status, progress and counts compared in row order.
- *Decoder tier.* For `K` at each tier boundary and one past it, the chunk cell's decoder tier
  equals the tier `T` selects for `K` CHUNK rows.
- *Fresh rows.* With the recycled pages of fresh rows filled with sentinel patterns (NaN, ±Inf,
  random bits) as the transaction starts, every output and status equals the run with zeroed
  pages. Written pages are compared across the two runs for accepted rows only; a rejected row's
  pages keep their different initial bytes, so each run's are compared with its own snapshot
  taken as the transaction starts. In a served step that snapshot follows the fresh-page
  initialization (Fresh rows), so the test shows only that the transaction does not rely on it.
- *Mask identities.* With all-false fresh and file masks, the masked body equals the stream-only
  transaction of the same continuing rows (Mask identities).

## Freshness and file masks

### Fresh rows

A fresh row is a request's first step (`num_computed_tokens` 0): a stream's first chunk or a
file's first step. Its state starts from plan authority: zero encoder, frontend, predictor,
replay-queue and endpoint state, and a book row holding the blank last label, the admitted prompt
and the geometry.

Before the transaction, the model initializes every fresh row's page (`registration`,
REG-STATE-010), whether or not the row later fails at planning or the transaction rejects it, so
that a fresh page never exposes an earlier request's state. Preservation starts after that
initialization: a rejected row's pages keep their contents as the transaction found them, and on
an all-failed step, which runs no transaction, as the initialization left them (TXG-EQ-006,
TXG-MASK-006, TXG-FAIL-001, TXG-FAIL-008). The transaction does not rely on the initialization
for its outputs.

The body reads the zero state from the null page. Once per step it computes each row's
initial-state index, `where(is_fresh, null_block_id, block)`, and gathers every zero-initial state
tensor by that index (the CHUNK bucket by its rows' entries), so a fresh row reads the null page
and a continuing row its own page. The book row is the one state whose initial value is not zero:
the body gathers it by the row's own page and selects `where(is_fresh, initial, gathered)` as that
gather's first consumer. A fresh row's outputs, status and stores therefore never depend on the
content of its recycled page, and the cost of freshness is one select on the index and one on the
book row, not a select over every gathered state tensor. Every scatter names the row's own page,
the block core allocated to the row's request (its slot, REG-STATE-002). Fresh and continuing CHUNK rows
stay in one bucket, as in the option-off transaction, so freshness never changes the encoder's
batch dimension.

### Null page

The null page is the page of block `null_block_id` (vLLM's `NULL_BLOCK_ID`, block 0) in every
component pool. Core allocates the KV cache zeroed and allocates the null block to no request, so
the null page starts as zero bytes. The model keeps it that way: structural preflight refuses the null block ID
for a real row, the row plan's CUDA-graph padding entries never enter the body, a predicated
scatter stores only rows with a zero status and only to their own pages, and capture rows and
the scatter warm-up's row name block 0 and are rejected. No transaction, cell replay or capture
therefore stores to it (TXG-MASK-007), which makes it the zero initial state fresh rows gather.
Block 0 is also the only page core guarantees backed at every capture (Capture procedure). An
extension that remaps or defragments the KV cache maps fresh zeroed blocks before any request
holds a page (`vllm/v1/worker/extensible_kv_cache.py:195-218,531-534`), so block 0 keeps zero
bytes across it.
That read-only gather is the one access outside the rows' own pages that `registration`'s state
access rule permits (REG-STATE-003). A cell bakes the null block ID into its gather, so the ID is
part of the cell's signature.

### Whole-file rows

A whole-file request (`transcription`) is a fully buffered stream on its own state slot. Each of
its steps is one row of an existing stream role, so file rows share the stream cells:

| File step | Row role | Cell kind |
| --- | --- | --- |
| First step (chunk or tail) | CHUNK, fresh | `chunk` |
| Regular chunk, final tail (possibly zero samples) | CHUNK | `chunk` |
| Drain (emit the next queued label) | REPLAY | `echo` |
| End (emit EOS) | REPLAY when the file's last emitted token was a label, FLUSH when it was the continuation token; both emit park, the file request's EOS | `echo` |

A file row differs from a stream row of the same role only in a few selects, each an elementwise
choice on the row's `is_file` bit:
- **Role check.** A file row's scheduled token is the token it emitted last (the file carrier on
  its first step, then a label or the continuation token), never the stream carrier. A file row in
  any role whose token is the stream carrier is a role mismatch; its CHUNK row is also one if no
  echo is pending and its token is neither the file carrier nor the continuation token.
- **Armed echo on a CHUNK row.** A file CHUNK row follows a drained queue whose last label's echo
  may still be armed. The echo is validated as a REPLAY row's is, and the chunk then replaces it;
  the pending echo does not count as an undrained queue for a file CHUNK row.
- **Continuation emission.** A file CHUNK row that advanced audio but queued no label emits the
  continuation token where a stream row would emit park.
- **Decision legality.** The continuation token is a legal decision only on a file row.

The body always evaluates these selects and always receives the file-carrier and continuation
token IDs, which are configuration constants. A cell covers any mixture of file and stream rows of
its kind, and a split step covers file rows of both kinds.

### Mask identities

With `f` the file bit and `c` the CHUNK bit, an all-false file mask reduces each select to the
stream rule, term by term:
- role check: `(~f) & (c != (id == carrier))` is `c != (id == carrier)`, and the file term
  `f & …` is false;
- echo check: `(~c) | f` is `~c`;
- undrained queue: `c & (((~f) & pending) | (remaining > 0))` is `c & (pending | (remaining > 0))`;
- empty-burst decision: `where(f, continuation, park)` is `park`, in the adapter and in the
  expected decision of the projection-invariant check;
- decision legality: `legal | (f & (decision == continuation))` is `legal`.

An all-false fresh mask reduces the initial-state index `where(is_fresh, null_block_id, block)` to
`block`, so every gather reads the row's own page, and the book select
`where(is_fresh, initial, gathered)` to `gathered`. Each term is an integer or boolean elementwise
operation, so each identity is exact.

### Exactness obligations for masked rows

For each case below, a covered step's outputs, row statuses, progress, remaining counts and
resident pages equal the option-off transaction of the same row plan bit for bit: a stream's first
chunk among continuing rows; a file's first step; a regular chunk; a silent chunk (no label:
continuation emitted); a final tail with samples; an empty final tail (zero samples, `final_tail`
set); a zero-sample file (first step is the empty tail, then end); a drain step; the end step as
REPLAY and as FLUSH; an armed echo on a file CHUNK row; and split steps whose CHUNK and echo rows
each mix file and stream rows. A served step holds no fresh row that is not CHUNK for the protocol
checks to judge: such a request fails at planning (`registration`, REG-STATE-015), so its row
reaches the transaction only as a planned rejection (TXG-COV-010). The enumerated plans still cover
every kind, fresh bit and file bit (Split equivalence, Tests).

### File samples

A file row's envelope (header and one chunk's samples) is written into its carrier row of
`inputs_embeds` before the transaction, by its own host-to-device copy (`transcription`). It
stays outside the arena: samples are per-row float data, not structure; `inputs_embeds` reaches
the cells through the step static inputs; and packing samples into the int64 arena would size it
by population times chunk width. Both legs of every comparison pay the same copy, so neither
exactness nor the step rule depends on it.

## Queue counts

The body writes every row's remaining queue count after the step, `QUEUE_LEN − QUEUE_HEAD` of its
proposed book as int32, into `remaining_step` at the row's position, with the option on or off.
The composite reservation passes `remaining_step[:P]` as the commit plan's queue-count field. The
cells require only that the field is a fixed-address device view that `reserve` checks by shape
and dtype and `stage` reads without allocating, as it treats `progress`. The transport from the
sink to file planning, its collection seam and its application belong to `transcription`
(Cascade obligations).

## Capture

### Flat capture

A chunk cell is captured with the nested encoder and decoder bindings running eagerly inside the
outer capture, so the graph records their kernels directly, not nested graph launches. While the
outer capture is active, every nested launch site sees the eager mode through a context variable,
and host-to-device staging inside that scope raises, because a captured copy would replay a frozen
pinned source. The eager paths use the nested bindings' storage and padding, so the flat capture
computes what the nested graph replays compute.

### Signature

A cell's signature covers every structural input its body bakes in: kind, count, geometry, control
token IDs, `endpoint_enabled`, the evidence-capture flag (the transaction's `capture` argument),
the emission adapter's callable identity, the decode, encoder-transition and bucket-transition
callable identities (chunk cells), resident pool layouts (address, shape, strides, dtype), the null
page's block ID (Null page), lazily bound encoder buffers (chunk cells), the carrier width, and the
layouts of the step arena, static inputs and step outputs. It also covers the two row classes the
cells exclude (Endpoint arguments): whether any of the cell's rows carries an endpoint policy
(`endpoint_policy`) and whether any is a forced-EOU candidate (`forced_eou`), both false at
capture. For the resident pool layouts, a step's view matches the cell's entry when it *fits*
that entry: the same address, page shape (every dimension after the first), strides and dtype, and
a first dimension no larger. The extensible KV cache's growth rebinds the views over the same
pages with no more pages than the cell captured, which fit; a view with more pages, another
address or other strides does not. The host computes it before the upload.

Callables compare by `(id(__self__), id(__func__))` for a bound method and `id` otherwise. The
dense decode binding caches one callable per graph key and the encoder binding one transition per
count for their lifetime, so an ID is stable and never reused; a resolver that rebuilt closures
per call would need another identity mechanism.

### Host control flow inside the body

The body's host control flow depends only on the cell key and the signature: `endpoint_enabled`,
the kind (a chunk cell runs one bucket over all its rows, an echo cell none), the count, the
geometry's lookahead and hop (chunk cells only), decode output widths, and the evidence-capture
flag (always off on the replay path); `endpoint_enabled` is true on every step (Endpoint
arguments). Any further host-dependent branch joins the signature or the cell key before its cell
is captured. Per-row protocol state (first chunk, final tail, sample
count, fresh and file bits, per-row cadence) is device data. Host-side checks inside the body
(result-structure validation, nested invocation counters) are frozen under replay, so serving
reads of nested counters are deltas from a baseline taken after startup.

### Capture procedure and the replay-cell oracle

Capturing one cell runs, on core's capture stream, an eager reference of the body on capture rows
that the transaction rejects (so no page is stored), three warm-ups with the nested bindings
running eagerly (Flat capture), the capture, and one validation replay. The cell is published only
if the replay's outputs equal the reference's bit for bit and, in every component pool, the null
page equals bit for bit the call's snapshot of it: a host copy of block 0 of every component pool,
taken once, at the start of the call's transaction-graph part (before the scatter warm-up). Because the comparison is with that snapshot, a store by the scatter warm-up, the
reference or a warm-up fails the check as well as a store by the replay. The pages are compared
before the outputs: fresh capture rows gather the null page, so a store to it also changes the
outputs, and the check reports the store. A cell at count `k` has `k` capture rows, every one naming
block 0, the null page, the only page core guarantees backed at every capture; the pages of held
requests may then have no memory, so no part of a capture call reads or writes any page but
block 0. The capture rows' plan passes the structural preflight in its capture-only mode, which
requires every row to name the null block ID and the live set to be empty, and waives, for those
rows, exactly three row checks: the refusal of the null block ID for a real row, of a block ID
named by two rows, and of a block ID outside the live set. It keeps every other check, among them
the bounds of each row's block ID against the pool and the validation of the live set, which an
empty live set passes. A scheduled step's plan never takes that mode: `prepare_model_rows`
receives it only from a capture call's cell programs, and `advance_model_rows` and `execute` do
not pass it. A CHUNK capture row carries a zero envelope header (`ROW_STATUS_ENVELOPE`); an echo capture
row carries the stream carrier token (`ROW_STATUS_ROLE_MISMATCH`). Capture rows carry the server's
geometry and alternate their fresh and file bits, so the self-check exercises both arms of every
select whose arms differ when every row names block 0 (the initial book row against the gathered
one, the file selects). The initial-state index's two arms both name block 0 there, and a cell
that baked a page address instead of reading the arena's block IDs would read and write block 0
on every replay, which the self-check cannot see either. Both defects are left to the replay
oracles: TXG-EQ-003 replays the same captured cell, at every count, once on a sub-plan of fresh
rows whose recycled pages hold sentinel patterns and once on a sub-plan of continuing rows whose
pages hold nonzero state distinct from the null page, and, at every count of 2 or more, on a
sub-plan that mixes the two; TXG-QUAL-003 compares every qualification step, pages included, with
the option-off transaction. Each capture row's rejection depends on its header or token alone, never on the content
of the page it gathers; the eager reference launches the nested segment graphs while the warm-ups and capture run
them eagerly, so it also compares the two nested paths. The self-check covers rejected rows only;
exactness for accepted rows rests on Split equivalence and Qualification, at both geometries,
including every case of Exactness obligations for masked rows and Lookahead rows at 160 ms.

Before the first cell of a capture call, a scatter warm-up launches, for each component pool and
each projected resident-pool view a transaction passes to the kernel, one row with a rejected
status that names block 0; the kernel's status predicate returns before it reads the page ID or
any resident address.

Cells are captured largest first: chunk cells from `N` down to 1, then echo cells from `N` down
to 1. Memory that a larger cell's capture allocated and freed is then available to the smaller
cells captured after it on the same stream and pool.

### Lookahead rows at 160 ms

At 160 ms the lookahead `L` is 1 and a regular chunk is 16 mel frames of 2,560 samples. Each
row's frontend target is `(8L + 1) + seq · 16` for chunk sequence `seq`; a final tail ignores its
target, and a final tail of 2,560 samples or more is rejected with `ROW_STATUS_FINAL_OVERSIZE`.
A first chunk takes no mel-tail prefix and drops nothing; a continuing row takes the nine-frame
prefix and drops the configured pre-encode outputs; each row's encoder length is its own, clamped
at zero, and zero when the row produced no new frames. Decode reads at most the first `L + 1`
encoder frames. These are per-row device values within one chunk cell. Replay equals the
option-off transaction for: a one-chunk session (fresh and final in one row); a final tail among
regular continuing rows; a final chunk followed by REPLAY drains and a FLUSH row, as a direct
engine client schedules it; and an oversized
final tail, whose row is rejected while the other rows commit and whose request then fails alone
(`session`, SESS-MODEL-003). These rules are the acoustic transition's 160 ms behaviour with the
option off; this component states them because its cells must reproduce them, and they add no
requirement on the transition.

### Process constants

The batch-invariant mode, the per-operation FP32 precision settings
(`torch.backends.cuda.matmul.fp32_precision`, `torch.backends.cudnn.conv.fp32_precision`,
`torch.backends.cudnn.rnn.fp32_precision`) and `CUBLAS_WORKSPACE_CONFIG` are constant for the
binding's lifetime. `execute` compares them with their capture-time values at step 3 of Execute
order, before classification, so the check precedes every return, fallbacks included. A change is
worker-fatal: every graph was captured under the old settings, so a fallback would be as stale.
The check reads only the per-operation settings, which also reflect the legacy
`torch.set_float32_matmul_precision` setter, because the legacy getters raise once a
per-operation value is set.

## Capture lifecycle on the hook

This component owns the transaction-graph part of #67583's hooks; `registration` owns the
segment graphs' part, which runs first in each capture call after this component's entry checks
(Rules of the capture call).

**Profiling and serving cells.** The profiling capture creates *profiling cells* over the pages
of core's temporary profiling KV cache (the *profiling pools*); they exist only until
`clear_cudagraphs` and no step ever replays them. The serving capture creates *serving cells* over
the serving pools; they are the only cells a scheduled step replays, and only after post-warmup
verification has passed. The one replay outside a step is each capture's validation replay of
rejected capture rows, inside its own `capture_cudagraphs` call (Capture procedure and the
replay-cell oracle).

| Hook | Transaction-graph behaviour |
| --- | --- |
| `needs_cudagraph_capture()` | True exactly when the platform is CUDA and the configuration selects a hook-captured graph: the `eager-graphed` encoder, the `dense-graphed` decoder or the option on (REG-MEM-019). Configuration only, the same value for the model state's lifetime. Under `--enforce-eager` the runner does not ask. |
| `prepare_execution_memory()` | Called once before the profiling forward pass, before core binds any state page, in both runner configurations. This component allocates nothing there: its buffers belong to the captures, which core's estimate charges when it runs and core's reading of free memory after the captures charges under the extensible KV cache. |
| profiling forward pass | Runs under the hook configuration too, as the profiling dummy step that the model state's `prepare_inputs` classifies (REG-MEM-001); `registration` runs its profile transaction there, on the first profiling dummy step only (REG-MEM-016), over model-owned storage, at the maximum shape. It is a dummy step, so it never calls `execute` and is never counted as a fallback. No cell is captured or replayed. |
| core's KV-cache binding and `initialize_kv_cache` | Core binds the state layer's views (`bind_kv_cache`) and sets `state_group_layouts` before `initialize_kv_cache`: first for its profiling KV cache, when its estimate runs, then for the serving KV cache. This component records nothing there; each capture call reads the layout itself (Rules of the capture call). Core clears the layout when it releases the profiling KV cache, after `clear_cudagraphs`. Under the extensible KV cache core binds the serving views again after `post_warmup` and before the first request, with no more pages, over the same base and strides (Repeated bindings). A binding after the serving capture follows Repeated bindings. |
| `capture_cudagraphs(profile_only=True)` | Runs only when core's estimate runs (a profiled budget or a fixed block count, with the extensible KV cache off). The profiling capture: segment graphs, the scatter warm-up, every cell, then the charged margin (Graph memory), over block 0 of the profiling pools, which hold the override's block count when it is set, otherwise `min(max_num_seqs, max_cudagraph_capture_size) or 1` blocks (one block under the explicit NONE the restricted startup requires), into the pool core passes (the estimate's throwaway pool). A fixed `kv_cache_memory_bytes` runs no profiling capture. |
| `clear_cudagraphs()` | Releases every graph, buffer, signature, recorded layout and the margin the profiling capture created, this component's part before the segment graphs. It keeps the capture record, whose profiling totals are host integers. Core calls it also when the profiling capture raised or was never reached, so it tolerates nothing captured. In the same call the model state drops the state layer's views (`registration`, REG-STATE-018), so when the call returns the plugin's objects hold the profiling KV cache only through the references core installed and clears in its teardown (the layer's `kv_cache` and the model state's `_state_group_layouts`), and after that teardown nothing of the plugin's keeps it allocated. |
| `capture_cudagraphs(profile_only=False)` | Runs whenever `needs_cudagraph_capture()` is true and `enforce_eager` is off. Its entry checks run first, in this order: Ordering, A second serving capture, `registration`'s segment-graph entry check (REG-MEM-024), then whether a profiling capture was required (When core's estimate does not run). The serving capture: the same procedure, without the margin, over the serving pools, into core's global graph pool; then the memory comparison. Records the plain-value pool layout for post-warmup verification. |
| `post_warmup()` | Called once per run of the worker's warmup; at startup after the serving capture and before the worker serves. Post-warmup verification. Allocates no device memory and replays nothing. |
| scheduled steps | `execute` per step (Execute order). |
| dummy steps | Classified by `prepare_inputs` (REG-MEM-001); never replay and never call `execute`. |
| `shutdown()` | Called at the start of the runner's shutdown, also after a failed startup: releases every cell and binding-owned buffer, before the segment graphs, and logs the counter summary. Tolerates a model state whose initialization did not complete and a binding that never captured. A killed worker, or one whose construction failed, never calls it. |

Under `--enforce-eager`, where the option is refused, core calls none of the three capture hooks,
and the transaction-graph parts of the other hooks do nothing.

### Rules of the capture call

- **Pool access.** Cells reach the pools only through the state-group layout that
  `ModelState.state_group_layouts` returns inside the `capture_cudagraphs` call. The
  transaction-graph part starts by checking it: exactly one layout, holding exactly the state
  layer; each of the layer's component tensors the layer's bound view of the same name and
  position (core builds the layout from the views it bound, `get_state_group_layouts` in
  `vllm/v1/worker/utils.py`); and, in the serving capture only, that the layout's page count is
  at least the admission limit plus one, the count core's scheduler requires once it is built
  after the captures, which is at least `N + 1`. The profiling capture makes no page-count check:
  its cells need block 0 alone, and its page count is not the serving count. A failed check is a
  failure of the pool-access phase (Capture failure messages). The block-count part can fail
  under every KV-cache sizing except a fixed block count, which the Block count condition has
  already checked; under the extensible KV cache the layout's page count is the reservation
  capacity, so a final count below the limit is left to core's scheduler. The layout grants no slot ownership and authorizes
  no store to any page, the null page included; capture, warm-ups and the self-check run only
  rejected rows and write only binding-owned memory. The call keeps no layout object and no tensor
  taken from it after it returns, only plain values (address, shape, strides, dtype per
  component, and in the serving capture the page count).
- **Stream and pool.** Every capture in the call runs on the `stream` core passes, inside
  `vllm.compilation.cudagraph_pool.capture_pool` with the `graph_pool` core passes. Each cell owns
  a `torch.cuda.CUDAGraph`; the model creates no stream and no pool of its own.
- **Device work.** Before the call returns or raises, it synchronizes the capture stream, so
  every eager reference, warm-up, capture and validation replay it started has finished.
- **Cleanup on failure.** When any failure occurs in the call after its entry checks pass, in the
  segment-graph part, in this component's part or in the aggregate memory comparison, the call
  measures what it retained so far, then, before propagating, discards every object the call
  created (cells, binding-owned buffers and the margin first, then segment graphs and their
  buffers), synchronizes
  the capture stream and propagates the original error with its diagnostics (Capture failure
  messages; `registration`'s Capture failure). Nothing the call created then remains, independent of
  a later `clear_cudagraphs` or `shutdown`. The propagated error holds no reference to the call's
  tensors or graphs: the cleanup, which receives the error, clears the local variables of the
  finished frames in its traceback and in the tracebacks of its cause and context, keeping the
  messages and traceback lines as diagnostics. Otherwise a retained error would keep the
  discarded memory alive.
- **Inventory.** With the option off the call captures no cell; with the option on it captures
  `2N` cells or raises.
- **Ordering.** A serving capture while profiling cells exist, or a profiling capture after a
  serving capture was attempted, whether or not it succeeded, is a lifecycle defect and raises. A
  serving capture is *attempted* once a serving call has passed every entry check below, the
  required-capture decision included.
- **Entry checks.** The ordering checks (Ordering, A second serving capture), then `registration`'s
  segment-graph entry check (REG-MEM-024) and, in the serving capture, the decision of When core's
  estimate does not run, in that order, are made at the call's entry, before the segment-graph part
  captures anything, so a refusal there captures nothing and its error
  names the option.
- **Workspace.** The option allocates nothing from core's workspace, which core locks after the
  serving capture; `registration`'s profile transaction sizes the option-off transaction's use.

### Warm-up domains

Core enables JIT monitoring (Triton compilation and autotuning, among others) after the startup
warmup. A scheduled step, replayed or option-off, compiles nothing those monitors observe, for
these warmed domains:
- **Masked-scatter kernel.** The plugin's one Triton kernel specializes on each resident-pool
  argument's device, dtype, page size, strides and launch configuration and on whether its pointer
  is 16-byte aligned; the row count is its grid. The scatter receives only the pools' fixed views:
  the component pools and the projected resident-pool views over them, whose addresses, and so
  whose alignment, are the same at warm-up, capture and serving; after the extensible KV cache
  grows, a fallback step passes views of no more pages over the same addresses, which select the
  same kernel specialization, because the kernel specializes on the page size, the page stride,
  the dtype, the device and the pointer alignment, not on the page count (`state_scatter.py:121-144`).
  The scatter warm-up launches
  every component pool and every projected view a transaction passes, and each cell's eager
  reference launches the views its kind's body passes, so every argument a scheduled step passes,
  covered or fallback, was launched before core enables JIT monitoring. The scatter's record of
  warmed arguments includes each argument's alignment, so a view at an alignment not yet launched
  is never taken as warmed. Padding the views to 16-byte alignment is a measured optimization, not
  an obligation (Deferred 10).
- **Bodies at every count.** Each cell's eager reference and warm-ups run its kind's body at its
  count, so both kinds are warmed at every count from 1 to `N`.
- **Segment graphs.** The encoder at every CHUNK count to `max_num_seqs` and the decoder at its
  tiers (`registration`).

The claim excludes first-use costs the monitors do not observe, such as lazy CUDA module loading.
A test runs each reachable fallback class (`chunk-uncovered`, `echo-uncovered`, and `signature`
with the fields `endpoint_policy` and `forced_eou` among its cases) after startup under
`jit_monitor_mode` `error` and requires no monitored compilation, and compares the scatter's
host-side arguments with those launched at startup by address, page shape, strides and dtype,
with a first dimension no larger.

### A second serving capture

An elastic expert-parallel reconfiguration reruns the worker's warmup, which calls the serving
capture again over the live pools while streams hold slots. A serving capture after a serving
capture was attempted (Rules of the capture call), whether or not the attempt succeeded, raises at
its entry, naming the option and stating that recapture over live pools is not supported; the
recovery is an engine restart. The refusal is keyed on the attempt, not on cells existing,
because a failed attempt discards its cells and segment graphs, and a retry would then capture a
second time in the worker. Like the other entry refusals, it changes nothing: the binding's latch
and any existing graphs stay as they were until `shutdown`, and core treats the propagated error
as worker-fatal. `registration` also refuses elastic expert parallelism, so the path is
unreachable in a supported configuration.

### Repeated bindings

Core binds the state layer's views again when its extensible KV cache grows: once, after
post-warmup verification and before the first request, over the same pages with the final page
count, which is at most the layout's page count; for a layout that is not block-compact it binds
nothing (`vllm/v1/worker/extensible_kv_cache.py:489-578`). When core binds the views after a
serving capture was attempted, the state layer reports them to the binding. Views that fit the
layout the serving capture recorded (the same address, page shape, strides and dtype per
component, and a first dimension at most the recorded page count) change nothing: the cells
address held pages by block ID, every block ID the scheduler assigns is below the bound count,
and the bound count is at most the captured extent. Views that do not fit latch the binding and
are worker-fatal, because the cells bake in the captured addresses. The comparison uses the plain
values the serving capture recorded; the binding does not read the accessor outside a capture
call. `registration` refuses sleep mode, whose wake path recaptures the model state's graphs, a
second serving capture that the entry checks would refuse. The profiling KV cache's binding takes
no part in the comparison. A row's slot is checked on the host against the bound views' first
dimension (`registration`, REG-STATE-003), never against the layout's page count, because a
cell's in-graph gathers carry no bound below the captured page count; for a layout that is not
block-compact the bound views keep the reservation's page count, and core's scheduler, which
assigns only block IDs below the final count, is then the only bound. That reliance is an
assumption on core's contract with a narrow reach: the persistent-state backend declares no
supported layouts, so core takes its default preference, whose first entry, LBNHC, is
block-compact (`vllm/v1/attention/backends/utils.py:226-235,426`); only `VLLM_KV_CACHE_LAYOUT` set
to LHBNC or BHLNC selects a layout that the extension leaves at the reservation's page count. The
model state holds the runner's live `kv_cache_config`, whose `num_blocks` the extension updates
in place (`vllm/v1/worker/gpu/model_runner.py:717-718,899`;
`vllm/v1/worker/extensible_kv_cache.py:535`).

## Graph memory

### Measurement

**Retained bytes** of a span are the drop in driver free memory (`torch.cuda.mem_get_info`)
across it, each reading taken after synchronizing the capture stream and releasing the allocator's
unused cached blocks. `registration` owns the one implementation of this measurement, of the
charged margin's allocation and retained charge, of the required-capture predicate and of the
aggregate comparison (REG-MEM-027); this component calls it for its spans and implements none of
them itself. Each capture call records:
- the **call total**: the whole `capture_cudagraphs` call, segment graphs, cells and (profiling
  capture) the margin;
- the **transaction total**: the span from the scatter warm-up to the last cell, excluding the
  margin;
- per kind, the span of its cells;
- per cell, its incremental growth in capture order;
- the call's **transient peak**: the allocator's peak reserved bytes over the call, with peak
  statistics reset at its start, less the reserved bytes at its start. The reset is process-wide;
  core reads peak statistics only inside its `memory_profiling` window, which has exited when
  either capture hook runs;
- the page count of the call's state-group layout (the profiling KV cache's, and the serving KV
  cache's).

The transient peak is incremental allocator memory. It is a different quantity from a device's
total memory in use (`nvidia-smi` `memory.used`, weights and every other process allocation
included), which Fit analysis reports from measurement.

Per-cell figures are diagnostic: under shared-pool reuse a cell's growth depends on the cells
captured before it, and no rule compares cells. The record also holds `PYTORCH_CUDA_ALLOC_CONF`
(including `expandable_segments`), `CUBLAS_WORKSPACE_CONFIG` and the PyTorch, CUDA and driver
versions. The figures only a profiling capture produces (its totals, `m`, `r`, the readings
around the margin and the absorbed rounding) are marked not applicable, never zero, when none ran
(TXG-MEM-006, TXG-OBS-003). Under the extensible KV cache the record also holds the final bound
count, the first dimension of the views reported at the extension (TXG-LIFE-009), as a field
separate from the capture layout's page count.

### When core's estimate runs

Core subtracts from the budget what its profiling capture window retains, read when the hook
returns. Core runs its estimate under a profiled budget or a fixed block count with the
extensible KV cache off. Its profiling KV cache holds the override's block count when one is set,
otherwise `min(max_num_seqs, max_cudagraph_capture_size) or 1` blocks, one block under the
explicit NONE the restricted startup requires. The profiling capture runs the serving capture's
procedure on the same cells over block 0 of those pools; a cell's allocations depend on its rows,
not on the pools' page count, so the comparison of the two calls does not depend on that count.
The profiling capture then allocates a **charged margin**: one device
buffer of `m` bytes, held until `clear_cudagraphs`. The model reads free memory immediately before
and after allocating it, under the convention of Measurement (capture stream synchronized,
unused cached blocks released); the drop is the margin's **retained charge** `r`, which allocator
rounding can make differ from `m`. The margin is live at core's reading, so core charges `r`.

The margin covers the rounding by which a serving call can retain more than an identical
profiling call. That rounding arises per allocator segment over the whole call, segment graphs
included, not per cell: the caching allocator serves requests from segments of its small and large
size classes, and under largest-first capture on one pool a smaller cell may allocate no segment.
The margin is therefore one flat amount per call. Its starting value, 64 MiB, is an empirical
allowance for a few large-class segments, negligible against the A40's capacity; the first A40
measurement replaces it with a value derived from the measured excess (Deferred 1).

After the serving capture, startup fails when the serving call total exceeds the profiling call
total, which includes `r` (Capture failure messages). The comparison is aggregate because core's
charge is aggregate. The profiling totals survive `clear_cudagraphs` as plain numbers in the
capture record, beside `m`, both raw free-memory readings around the margin, `r` and the
absorbed rounding `serving − (profiling − r)`.

Core's estimate charges what a capture retains, not its transient peak. The serving capture runs
after the state pages and KV cache are allocated, so its peak must fit beside the memory resident
at that capture.

### When core's estimate does not run

When the serving capture begins, the model decides from the worker's effective cache
configuration whether a profiling capture was required. Core can change both inputs after the
model state is constructed: a persisted startup plan sets `kv_cache_memory_bytes` during memory
determination, and the engine can disable the extensible KV cache. A profiling capture is
**required** (the predicate of REG-MEM-027) exactly when the effective `kv_cache_memory_bytes` is
unset or 0 (core's own truthiness test), the estimate is enabled (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` not 0) and
the effective extensible-KV-cache setting is off; the option already requires CUDA. An absence
has one of three reasons, checked in this order: the extensible KV cache, under which core runs
no estimate and sizes the KV cache from the free memory it reads after the captures and
`post_warmup` (`vllm/v1/worker/gpu_worker.py:1308-1321`), so the graphs are charged by that
reading, whatever the budget or estimate setting (a startup plan sets `kv_cache_memory_bytes`
with the extensible KV cache left on, `vllm/v1/worker/gpu_worker.py:742`,
`vllm/v1/worker/startup_plan.py:146-175`, and core still measures); with the extensible KV cache
off, a fixed budget (`--kv-cache-memory-bytes` or a startup plan); or, with it off, the estimate
disabled under a profiled budget or a fixed block count. The capture record holds those effective
values and the resulting reason. Tests cover an applied startup plan with the extensible KV cache
off (an expected absence, logged as a fixed budget) and with it on (a measured absence).
- **Measured absence** (the effective extensible KV cache on). The serving capture logs one INFO
  line: core sizes the KV cache from the memory left after the captures, which includes the
  transaction graphs and segment graphs; the call, transaction and per-kind totals.
- **Expected absence** (not required, the extensible KV cache off). The serving capture logs one WARNING: the transaction
  graphs and segment graphs were not charged to the budget; the reason (the fixed budget value
  and whether a startup plan set it, or the estimate's setting); the call, transaction and
  per-kind totals; and that the budget must leave room for them.
- **Unexpected absence** (required, but no profiling capture ran in the worker before the serving
  capture). Startup fails at the serving capture's entry, before any graph is captured, naming the
  configuration that should have run the estimate.

### Capture failure messages

This section covers a failure in a capture call's transaction-graph part, after the entry checks
pass and before the aggregate comparison. A failure in the segment-graph part carries
`registration`'s message (REG-MEM-017); entry-check refusals carry their own messages (Rules of the
capture call; When core's estimate does not run); the serving-over-profiling failure has its own
message, below. Whatever the part, the call's cleanup is the whole-call rule (Rules of the capture
call).

A failure in the transaction-graph part raises an error chained from the original exception. It names the option,
its value and `N`; the capture (profiling, before the state pages are planned, or serving); the
phase that failed (pool access, binding-buffer allocation, scatter warm-up, a cell's eager
reference, warm-ups, capture or validation replay, or the margin allocation); the failing cell
(geometry label, kind, count) only when the phase belongs to a cell; the totals retained so far
and the transient peak; the KV-cache sizing (the extensible KV cache, with its page count; a
block count fixed at the given `num_gpu_blocks_override`, with the estimate enabled or disabled; a
budget fixed at the given `kv_cache_memory_bytes`; or a profiled budget with the extensible KV
cache off, with the estimate enabled or disabled); the allocator settings; and the remedy for its
case:
- pool access with too few pages, in the serving capture: a block count or budget that holds at
  least the admission limit plus one state pages (a larger `--kv-cache-memory-bytes`, a larger
  `--num-gpu-blocks-override`, or a larger `--gpu-memory-utilization` under a profiled budget or
  the extensible KV cache), or a lower admission limit with the largest population at most
  it; never a smaller budget, which would hold fewer pages still;
- pool access that finds core's state-group layout contract broken (a layout count other than
  one, a layout that does not hold exactly the state layer, or a component tensor that is not the
  layer's bound view): no remedy, since no configuration repairs core's contract; the error names
  the broken condition;
- profiling capture, in a later phase: a smaller largest population, a smaller `max_num_seqs`,
  or the option off;
- serving capture, fixed block count, in a later phase: a profiling-capture remedy; or a smaller
  `--num-gpu-blocks-override` that still holds at least the admission limit plus one blocks, which
  helps only when the allocated state pages leave too little device memory for the capture's
  transient peak;
- serving capture, fixed budget, in a later phase: a `--kv-cache-memory-bytes` smaller by at
  least the projected remainder (per kind with at least one cell captured in the call, the mean
  growth of those cells times its cells not yet captured, stated as an estimate; for a kind with
  no captured cell the estimate is stated as unavailable), or a profiling-capture remedy;
- serving capture under a profiled budget or the extensible KV cache, in a later phase: a
  profiling-capture remedy, or, under a profiled budget, a lower `--gpu-memory-utilization`, which
  helps only when the allocated state pages leave too little device memory for the capture's
  transient peak, which core's estimate does not charge.

The serving-over-profiling failure names both call totals, `m`, `r`, the excess and the
allocator settings, and gives as remedies a fixed `--kv-cache-memory-bytes` that leaves room for
the reported serving total (the logged uncharged path), a smaller largest population, or the
option off.

### Temporary profiling pool

Core allocates its profiling KV cache before the profiling capture: the override's block count
when it is set, otherwise one block under the explicit NONE the restricted startup requires. When the device cannot hold it,
the allocation's own exception fails startup and core calls neither `capture_cudagraphs` nor
`clear_cudagraphs`. The target core names no model option in that failure, in a capture failure or
in a planned-capacity failure; the model's own capture line and capture-failure messages name the
option (the `registration` design, Future roadmap 12).

### No growth after capture

Every buffer the option uses is allocated during the serving capture; from `execute` entry
through the commit stage a covered step allocates only the caller-owned escaped row tensor,
before its first launch. Tests check that, over those spans, replayed steps leave the allocator's
reserved and allocated bytes unchanged apart from those tensors, so the serving thread runs no
memory reading, device or stream synchronize, garbage collection or cache release for the
option. A file row's envelope preparation (File samples) runs before `execute` and allocates
its device chunk; the tests measure it separately, identically in both comparison arms. The one host wait it makes is on a pinned host arena's event, which waits only for
that arena's previous upload (Step arena).

## Post-warmup verification

`post_warmup()` runs after the serving capture and before the worker serves; an exception from it
fails startup. The model checks, in order, against the values recorded at the serving capture,
using only plain values and host state:

1. whether the state layer's currently bound views fit the layout the serving capture recorded
   (Repeated bindings), and the pools' device against the encoder parameters' device; core runs
   `post_warmup` before the extension (`vllm/v1/worker/gpu_worker.py:1275`,
   `vllm/v1/engine/core.py:424`), so the views are then the captured ones, and the fit rule makes
   verification and the binding check one comparison;
2. the projected resident layout (per transaction role: component pool, byte offset, shape,
   strides, dtype), which catches a projection that swaps two same-shaped roles;
3. that all `2N` cells hold their graphs, and the callable identities;
4. the address and size of every buffer outside the cells that the graphs read or write (step
   arena, static inputs and outputs, model parameters and buffers, nested bindings' scratch);
5. the process constants.

A mismatch latches the binding as failed and raises, so startup fails. Verification replays no
graph (it runs outside any step, and each cell's capture already ran its validation replay),
allocates no device memory and does not enlarge the
workspace. Its one allocation is a bounded host shadow page of the state layout, over which check
2 re-runs the model's projection to obtain the current role table.

## Failure containment

**Per-row rejections are not step failures.** A rejected row has a nonzero status, its pages stay
as the transaction found them (a fresh row's as initialized before the transaction, Fresh rows),
and the other rows commit. Under
replay as under the option-off transaction, its status flows through the sink's collection, and
`postprocess_state` then fails its request alone, returning it through the model state's
`take_failed_requests` (the `registration` design, Row failure), as the owning design states:
`session` for stream rows (SESS-MODEL-003), `transcription` for file rows (FILE-QC-012). No row status is recoverable. Such a step neither raises nor latches. A
planning-failed row (Option and configuration) is not such a rejection: its request fails from
the planning-failed set, independently of collection.

**Before the first launch**, work touches only binding-owned buffers and host bookkeeping; a
failure there (Execute order) raises with resident state at the baseline and a reserved ticket
cancelled, and does not latch the binding, except the worker-fatal process-constant and
graph-object failures: the arena and static inputs are rewritten in full on every step, so nothing
outlives the failure. Not latching does not promise that the engine survives the propagated
exception. If cancelling the held reservation itself raises, the sink's state is no longer known,
so the binding latches and the failure propagates as worker-fatal. A delayed device fault that
surfaces during a pre-launch copy is not a pre-launch failure; it takes the path of Delayed device
faults, below.

**From the first launch on**, any failure is worker-fatal: a failed launch, a failure between the
launches, the escaped-row copy or the commit stage. The binding latches, attempts to cancel the
reservation (the worker is not reused, so the outcome is irrelevant; a stage that raised before
publication leaves the reservation cancellable) and propagates the exception.
No launch is retried and no half of a split step falls back, because the first launch may already
have scattered. Core's engine loop reports the engine dead (`EngineDeadError` to clients).

**Delayed device faults.** An asynchronous device fault surfaces at a later synchronizing call,
normally the sink's collection, after the stage. It propagates as worker-fatal, without retry,
fallback or attribution to a cell. At the collection it neither latches the binding nor cancels
the reservation, which the successful stage already resolved; core's engine loop then reports the
engine dead.

**Recovery.** The latch lives in one worker process: every later step there that schedules a row
raises "transaction graphs failed on this worker; a fresh engine is required", an all-failed
step included, because the model state checks the latch before its bypass; only a cleanup-only
step completes, and an engine restart builds a new
worker that captures again. Pages a failed step touched stay as it left them until core allocates
the block to a new request, whose first step initializes the page (REG-STATE-010). Qualification tests this path end to end.

**Output lifetime.** The escaped row tensor survives later steps. The step outputs are valid
until the next step's launches; the commit stage and the escaped-row copy consume them first.

**Invariants.** No slot is freed or reassigned between the reservation and the sink's
collection, which the worker places before the next step's planning: it runs the step's
`postprocess_state`, where the sink is collected, before it executes the next step (the option
runs with synchronous scheduling, Configuration checks). A request abort
is not ticket cancellation: a launched step completes its stage and collection, and core
suppresses the aborted output, frees the request's page only after the step, and reuses a page
only after its request's terminal finish (the `registration` design, State-slot access). Cells may share graph-pool memory for
intermediates, and nested segment-binding scratch is shared with the option-off transaction; no
intermediate is live across a replay, and each call writes the scratch fully before reading it.

## Observability

- **Counters**, host-only and never synchronizing: captured cells; replays per cell; covered
  steps by shape (one kind or split); fallbacks per `(reason, kind, count)`; the `(K, E)`
  histogram over every step that reaches transaction execution, covered or not, plus one
  `(0, P)` entry per all-failed step; prologue host time. With the option off the reader
  returns an empty mapping.
- **Reader.** `transaction_graph_counters(worker_or_model)` returns the counters and the capture
  record. It takes a model, runner, worker or engine core in-process, and runs on the worker
  through `AsyncLLM.collective_rpc` under a multiprocess engine; vLLM serializes a function sent
  that way only with `VLLM_ALLOW_INSECURE_SERIALIZATION=1` set in the environment the engine
  starts in (`vllm/v1/serial_utils.py`). It walks runner and worker
  attributes (a harness convenience, not a core contract) and raises if it cannot reach the
  model, so an unreachable model is never mistaken for the option being off.
- **Capture record** (per worker): the configuration identity, the figures and settings of
  Measurement for both captures, the effective cache values that decided whether a profiling
  capture was required, whether it ran (and why not), the budget setting, the margin's figures
  (When core's estimate runs), and capture and verification seconds.
- **Log lines**, the production record under a multiprocess engine: one INFO line per capture
  (cells, `N`, call and transaction totals, transient peak, capture time); the WARNING lines of
  Graph memory and of the first `signature` fallback; one INFO line when post-warmup verification
  passes; one INFO line at shutdown with replays, split steps and fallbacks by reason.

## Fit analysis

Startup memory under the option has two constraints: the retained memory, which reduces the KV
budget, and each capture's transient peak, which must fit beside everything resident at that
capture. The measured reference is `B_hook`'s segment graphs, on an A40 (45 GiB usable) under
`--enforce-eager`, where the model captures them once, itself (unpublished project measurement).
The hook configuration captures them twice, profiling and serving, when core's estimate runs,
and once otherwise; it is unmeasured. The peak column is the sampled whole-device `memory.used` maximum during startup, weights included, not
the incremental transient peak of Measurement:

| Geometry | `max_num_seqs` | Retained (core's charge) | Peak device memory in use | Startup |
| --- | ---: | --- | ---: | ---: |
| 320 ms | 32 | 1.06 GiB (1.34 GiB) | 7.0 GiB | about 41 s |
| 320 ms | 64 | 2.31 GiB (2.84 GiB) | 18.3 GiB | about 78 s |
| 320 ms | 128 | 17.37 GiB (18.49 GiB) | 44.4 GiB | about 182 s |
| 160 ms | 64 | 1.77 GiB (2.38 GiB) | 17.6 GiB | about 66 s |
| 80 ms | 64 | 1.59 GiB (2.24 GiB) | 16.8 GiB | about 65 s |

The eager arms start in about 25 s. Device memory in use at ready was 3.1 to 5.6 GiB at 32 and 64
rows, so this model's state pages do not fill the budget core reports. The decoder's tiers were
0.17 GiB at 64 rows, measured at 320 ms only; taking that figure at every geometry, the encoder's
share at 64 rows is about 2.1, 1.6 and 1.4 GiB at 320, 160 and 80 ms. Growth from 64 to 128 rows
is 7.5×, and the 128-row peak leaves no headroom on the card. The startup peak, not the retained
memory, binds startup, and core's estimate does not charge it.

The option adds, beside those graphs:

| Part | Size | Status |
| --- | --- | --- |
| Chunk cells, counts 1 to `N` | each repeats its count's frontend and encoder intermediates beside the segment graphs kept for fallback; the encoder share above (about 1.4 GiB at 80 ms and 1.6 GiB at 160 ms, 64 rows) is a reference scale | unmeasured; cross-cell reuse under largest-first capture unknown |
| Echo cells, counts 1 to `N` | integer and boolean work over the queue and book rows; expected small | unmeasured |
| Charged margin `m` (profiling capture only) | 64 MiB starting value | design value |
| Binding-owned buffers | step arena (two sections of `N` rows of int64 and bool vectors), two pinned host arenas, static inputs and step outputs of `P_max` rows of width `H` | allocated in each capture, inside its call total, and released with the profiling capture |

The prototype's nine homogeneous cells retained about 1.96 GB at 80 ms with no cross-cell reuse.
The profiling capture's transient peak competes with the weights and the temporary profiling
pool, the serving capture's with the weights, the state pages and the KV cache. Whether the A40
holds all of it at `N` = 64 is open. The prototype's capture-only time, about 0.25 s per cell,
puts `2N` cells captured twice at minutes for `N` = 64, beside the segment graphs' two captures,
against `VLLM_ENGINE_READY_TIMEOUT_S` (600 s by default).

**Measurement plan.** The first A40 run at 80 and 160 ms records, for both captures, the call,
transaction and per-kind totals, the transient peak, `m`, `r`, the absorbed rounding and the
capture time, for `N` = 16, 32 and 64 at `max_num_seqs` 64.

**Choosing `N`.** Run the served `Q` workload of Qualification with
`N = min(P_max, admission limit)`, the largest `N` the configuration checks accept, when it
fits, otherwise the largest `N` that fits, and read the `(K, E)` histogram from the counters.
For tier 5, declare as `N` the smallest count of at least 63 that fits and for which steps with
`K ≤ N` and `E ≤ N` make up at least 99 % of served steps; if none fits, tier 5 is not claimed. A
deployment may choose a smaller `N` by the same 99 % rule; it is reported separately as a
deployment choice and carries no tier 5 claim. `docs/evidence.md` records the histogram, the chosen `N`
and its memory.
"Fits" means both captures and their peaks succeed under `--num-gpu-blocks-override` set to
`max_num_seqs + 1` with core's estimate enabled. A step outside the inventory executes `B_hook` after classification.
The step rule's population 63 presumes `N ≥ 63`, which is why the tier 5 inventory has that floor.

## Calibration

Admission decides how many streams an engine admits from how much step time each admitted stream
adds. This component defines that measurement for the option; how admission turns the figure into
a stream count belongs to its consumer (Open Questions).

- **U(1)** is the step time one admitted stream adds at the server's geometry: the median host
  time, from the transaction call to the completion of its device work, of a one-row continuing
  CHUNK step on an otherwise idle engine.
- **The calibration run** is a preregistered microbenchmark in a fresh process at the
  configuration being calibrated, using the step rule's procedure with a one-row plan: 2 warm-up
  and 21 counted steps per arm, option on (the chunk cell at 1) and off. U(1) and the ladder are
  measured at core's default sizing as well as under the step rule's fixed override, each in its
  own record; a calibration applies only at the sizing in its identity (TXG-CAL-001 to
  TXG-CAL-003). Its frozen record holds
  the audio (utterance ID and chunk index), the snapshot rule, the plan and the procedure. The run
  also records the step-time ladder `t(P)` on and off from the step rule's compositions.
- **Served replay fraction.** The fraction of steps that replayed in the served `Q` run at
  concurrency `N` (Qualification), on an engine of the same identity.
- **Output.** One record: the configuration identity, U(1) on and off, the ladder, the served
  replay fraction, the device and the date. The identity covers the checkpoint revision, plugin
  version, core build, device model, dtype, the process constants (Process constants), geometry,
  segment arms, scheduling mode, the KV-cache sizing, `max_num_seqs`, `max_num_batched_tokens`
  and the option's value.
  A consumer applies the option-on figures only to an engine with the same identity and only when
  the served replay fraction measured at that identity is 1; a fraction from another scheduling
  or precision configuration does not qualify. Otherwise it uses the option-off figures.

## Qualification

The HLD's tier 5 is met separately at 80 ms and at 160 ms, on an A40 at FP32, on the stage 1
build, under the restricted startup and core's default KV-cache sizing (the extensible KV cache).
Every option-on/off comparison is against `B_hook`. Runs that need core's graph-memory estimate
(the step rule, the fit choice of `N`, the estimate check below) use a fixed
`--num-gpu-blocks-override` with the estimate enabled. The record of every run, option on and
`B_hook`, holds the KV-cache sizing, each capture layout's page count and, under the extensible
KV cache, the final bound count read after the extension; that the option changes exactly one
setting (Option-off baseline) refers to the runner flags, and the engines' final counts are
reported, not equated. The qualification driver assigns slots only below the final committed
count (TXG-QUAL-026) and copies only the pages it can assign; it digests the other committed pages
(TXG-QUAL-003).

### Workloads

- **Qualification workload `Q`.** Independently paced streams at the server's geometry, driven at
  the transaction level, 2,000 steps at each of the concurrencies 1, `⌈N/2⌉` and `N`. Clips are
  LibriSpeech test-clean utterances in a seeded permutation; each stream slot plays the next one,
  a fresh stream replacing a retired one. A stream starts at its predecessor's retirement time
  (0 for the first) plus a delay uniform on `[0, c)`, `c` being the chunk duration, from the
  seeded generator, and offers each chunk when its audio would have arrived. The clock advances
  by each step's measured option-on duration and, when no stream has a schedulable unit, jumps to
  the next arrival. Each step holds every stream with a schedulable unit (a chunk, a label drain,
  an echo, a FLUSH), so mixed steps arise from the pacing. The driver runs every step replayed and
  as the option-off transaction on a detached copy of the state layer's bound views
  (`component_pools`), taken before the step, and compares outputs, statuses, progress, remaining
  counts and every page the copy holds. The copy holds pages 0 to the admission limit: `Q` and `M`
  hold at most `N` requests, `N` is at most the admission limit (TXG-CFG-020), and the adapter
  assigns the lowest free block, so no request holds a block above the limit. Core's scheduler,
  built after the extension, refuses a final committed count below the admission limit plus one
  at one state block per request (`vllm/v1/core/sched/scheduler.py:160-171`), so these pages are
  committed. The other committed pages, from the admission limit plus one to the final committed
  count, are digested in place before the run's first step and after its last: each view, one
  bounded chunk at a time, has its bytes copied to the host and hashed with SHA-256 together with
  the view's identity and the chunk's offset, so any changed bit changes the digest. A changed
  digest fails the run as a page mismatch does. The comparisons read both sides in place, in the
  same bounded chunks, as same-size integer views of the strided pool views, so neither the
  digest nor a comparison needs workspace beyond one chunk. The reference runs the step's row plan
  with the copy's page count as its resident pool extent (`num_pool_blocks`), the bound the
  transaction checks every pool and block index against, so a row naming a block above the limit
  fails the reference. The replayed step runs on
  the serving pools, whose addresses its graphs hold; the copy is test-only storage that the serving
  rules (REG-STATE-003, TXG-CAP-013) do not govern, and the reference never touches the serving
  pools. The driver runs on the worker of a hook-captured
  process through a test-only adapter over the serving pools, which drives the model state's step
  lifecycle as core's runner does: per step it calls `remove_request` for each request that
  finished, in ascending request-ID order; then, for each request the step delivers as
  `NewRequestData` in `scheduled_new_reqs` (a new request, or a stream's next turn), in that
  list's order, `remove_request` for that request immediately followed by its `add_request` with
  that `NewRequestData`; then `prepare_inputs`. Removal and addition interleave per request, as in
  core's runner: in the order "remove A, remove B, add A" the model state would retire A at B's
  removal (REG-STATE-011) and then refuse A's continuing turn (REG-STATE-012). A cached request
  (a file's later step, or a label drain within a stream's turn) gets neither call: core updates
  it with no model-state call, and a removal would retire a running file at once (REG-STATE-016),
  after which its addition would raise (REG-STATE-012). It executes on the runner stream, then
  calls `postprocess_state`, which waits for the step's commit event, and `take_failed_requests`
  before the next step. The adapter assigns each request, at its first step, the lowest free
  block above block 0 and below the final committed block count, keeps it
  until the request's terminal finish and reuses it only after that, as core's allocator does, and
  delivers a stream's next turn as core does (its compacted prompt, `num_computed_tokens` 1). No
  replay, restore or store runs outside a step run this way. The lifecycle calls, the pool copy,
  the digests and the sink's collection run outside any timer.
- **Mixed workload `M`.** `N` in-flight requests at the transaction level: `max(1, ⌊N/4⌋)` file
  slots and the rest `Q`'s streams, with populations counting stream and file rows together. Each
  file slot takes the next file when its file retires, until every file has completed and at
  least 2,000 steps have run; a file is admitted only while every stream slot is occupied, so its
  steps share steps with stream rows. The files are, in order: one with leading and trailing
  silence, one whose length is a whole number of chunks (empty final tail), a zero-sample file,
  one shorter than one chunk, then test-clean utterances from the same permutation.
- **Minimum case coverage.** The record counts steps by case. `Q` at concurrency `N` needs at
  least 200 split steps and 50 steps with a fresh CHUNK row among continuing CHUNK rows; `M`
  needs at least 100 split steps containing a file row and at least one step of each case of
  Exactness obligations for masked rows. A run lacking a required case is void: it is reported, and the next seed of the
  preregistered seed list runs; an exhausted seed list leaves qualification incomplete. Voiding
  covers missing coverage only: a mismatch or fallback in a void run is still a failure.
- **Transaction-level pacing.** The driver omits engine overhead (scheduling, input preparation,
  output processing), so its steps can batch differently from served steps. Its histogram shows
  exactness coverage only; the served `(K, E)` histogram is the coverage evidence.
- **Constructed schedules.** Aligned phases, filler rows, enumerated compositions and the step
  rule's recorded plans are constructed schedules. Exactness tests use them freely; preregistered
  microbenchmarks (the step rule, calibration) use them and claim
  only the step time of the compositions they construct. No rate, coverage or capacity claim
  rests on a constructed schedule; served coverage and capacity claims rest on free-running runs.
- **Uncovered steps.** Steps with `K > N`, steps with `E > N`, a forced signature mismatch and a
  multi-token row among one-token rows, each with its expected fallback count and outputs equal to
  the option-off transaction's. The run holds `N + 4` requests (the `K = N + 1` step's, the
  endpoint-policy request and the multi-token step's two), so its copy follows `Q`'s rule with
  pages 0 to `N + 4`, and the other committed pages are digested. One function of a mode's most
  requests gives both modes' extents and copy figures. A final committed count below `N + 5`
  fails the run before its first step, naming both counts (TXG-QUAL-007).
- **Served runs.** An option-on engine completes two served runs on plugin-packed input (every
  unit packed by the plugin's packer, every file prompt built by `get_generation_prompt`, no
  endpoint policy), each paced by the wall clock: `Q`'s streams (concurrency `N`, the same
  manifest, seed and offsets) on core's `/v1/realtime`; and `N` requests in flight as `M` defines,
  its streams on `/v1/realtime` and its files on the transcription route. Each completes every
  request with zero fallbacks and reports its counters, including the `(K, E)` histogram and the
  replay fraction. Under load, two runs of one engine can differ: wall-clock pacing changes which
  steps a request's rows share, and the FP32 kernels are not batch-invariant, so two `B_hook` runs
  can disagree on a near-tie. Transcript agreement with at least two `B_hook` runs, and between
  them, is therefore reported, not gated; every disagreement is investigated, and an unresolved
  suspected execution defect blocks qualification. A file transcribed alone on an idle server has a
  fixed row plan, so that comparison is exact and gated. Bitwise exactness under load rests on the
  transaction-level comparison of identical inputs, initial state and ordered row plans
  (TXG-QUAL-003, TXG-QUAL-006).

### Exactness

On `Q` and `M`, every covered step, of one kind or split, equals the option-off transaction of the
same row plan bit for bit; the transcripts equal the option-off run's; and the run records zero
fallbacks of any kind. Both served runs also record zero fallbacks. Every zero-fallback claim is
scoped to plugin-packed input: a direct engine client can send a foreign-geometry row, which
planning fails, an endpoint policy or a scheduled EOU token, which falls back, or a multi-token
row, which planning fails and whose step falls back.

Under the extensible KV cache the detached copy is a PyTorch allocation under the cap core sets
after sizing the cache, `memory_reserved` plus what stays free after the extension
(`vllm/v1/worker/extensible_kv_cache.py:684-699`). The run reserves a fixed headroom through a
`--gpu-memory-utilization` below 1.0, never a fixed budget, which would turn the extensible KV
cache off. The headroom is preregistered before the run from quantities known then: the copy
figure, the copy's page count (the admission limit plus one for `Q` and `M`, `N + 5` for the
uncovered steps) times the bound views' bytes per page (the state manifest's
page, which core's state page equals), plus a stated margin. The margin covers core's reserve for
memory outside the torch allocator (`KV_CACHE_NON_TORCH_RESERVE_BYTES`,
`vllm/v1/worker/extensible_kv_cache.py:71`), the memory in use at startup, which the utilization
fraction of total memory does not exclude (`vllm/v1/worker/utils.py:906`), the digest's and the
comparison's one-chunk workspaces, and the copy's slack over the bytes it copies. Each term counts
what PyTorch's CUDA caching allocator reserves: an allocation of at most 1 MiB takes a 2 MiB
segment, one below 10 MiB a 20 MiB segment, and a larger one is rounded up to a multiple of
2 MiB. The copy makes one allocation per distinct storage of the bound views, one for the state
layer's single storage; a copy that makes more fails the run, because the margin states one. The
startup allowance is a stated figure of 1 GiB, preregistered with the margin. A run that compares
with a detached copy under the extensible KV cache without a preregistered headroom is refused
before the engine starts. The run records the bytes each copy copies and fails
when they exceed the copy figure. The record also holds core's logged torch memory limit and
`torch.cuda.memory_reserved` and `max_memory_reserved` before and after the copy, with
`nvidia-smi` and `mem_get_info` as context only; served figures come from runs without the
headroom (TXG-QUAL-029).

### Step rule

The step rule is a preregistered microbenchmark at the transaction level, one fresh process per
population, on test-clean audio. The process captures the deployed inventory, the `2N` cells of the
declared `N`, through the hook under a fixed block count with core's estimate enabled, on the
A40; a tier-5 pass at a population
requires that inventory to start. A run with a reduced inventory is diagnostic only, ineligible for
tier 5, and compares against a `B_hook` process of the same `max_num_seqs` and segment arms.

- **Constructed plans.** For each population and composition the driver runs a preregistered
  prefix that steps `P` streams individually, freezing each at its target state: a CHUNK-row
  stream has no armed echo, zero remaining queue and a regular next chunk; a drain-row stream has
  an armed echo and at least one remaining label. It then snapshots the slots' pages and records
  one row plan: the `K` CHUNK rows and `E` REPLAY label drains, the drains at evenly spaced step
  positions among the CHUNK rows. Every leg starts from that snapshot and runs that plan on every
  step. Before each step the driver restores the pages by an in-place copy into the same pool
  memory (the cells bake in its addresses), on the runner stream, touching only the slots of that
  step's rows, which the adapter of `Q` assigned; the lifecycle calls, the restore and
  the sink's collection run outside the timer. Every row is accepted in both arms; a
  rejected row voids the leg.
- **One-kind group:** continuing CHUNK steps (`E` = 0) at populations 1, 31 and 63. One-kind echo
  steps are measured and reported, not gated.
- **Split group:** at population 31, compositions `(K, E)` = (27, 4) and (16, 15); at population
  63, (55, 8) and (32, 31). The first composition at each population approximates a CHUNK-heavy
  step; the second is the balanced case, the most work for the second launch. FLUSH rows end their
  streams and cannot recur in a recorded plan; their step times are reported separately from the
  free-running `Q` runs, by composition, and not gated.
- Per population and composition: two blocks, each with three legs in order OFF1, ON, OFF2; each
  leg 2 warm-up and 21 counted steps; ON replays the cells, OFF runs the option-off transaction of
  the same plan.
- The clock is the host time from the transaction call to `torch.cuda.synchronize()` after it.
- For block `b`, `Δ_b = min(median OFF1_b, median OFF2_b) − median ON_b` and
  `drift_b = |median OFF1_b − median OFF2_b|`; the option is material for a population and
  composition exactly when `Δ_b ≥ max(5 ms, drift_b)` in both blocks.
- **Aggregation.** The one-kind group passes exactly when the option is material at 1, 31 and 63.
  The split group passes exactly when it is material for all four compositions. Each group reports its own result;
  no result is pooled across compositions or groups.

### Feed-to-park rule (`U_api`)

The single-stream admission demand is measured through `AsyncLLM` in a multiprocess engine, on
the client path the target core offers: `AsyncLLM.generate` given an input generator that yields
`StreamingInput` units the plugin's packer builds, which is the path core's `/v1/realtime` takes.
The timer starts before the generator yields one unit (a fresh stream's first chunk, not final)
and stops when that unit's parked output arrives, the output whose new token IDs end with the
park token. Each timed repeat reaches exactly one park with no error output; for option-on legs
the counter reader shows one more replay of the chunk cell at 1, one more replay of the echo cell
at 1 per later step, and no fallback. The stream's admission, which core completes before it
reads the generator, its close or abort and its terminal output are outside the timer. There are
five brackets of counted legs OFF1, ON, OFF2; every leg is a fresh engine process and follows a
washout leg of its own arm. Within a leg, repeat 0 is discarded and repeats 1 to 3 are counted;
`U` and `M` are their maximum and median.
- **Rule 1 (per bracket):** in all five brackets,
  `G = min(U_OFF1, U_OFF2) − U_ON ≥ T = max(5 ms, |U_OFF2 − U_OFF1|)`.
- **Rule 2 (pooled):** the same formula on each arm's maximum over its 15 counted samples.
- **Rule 3 (medians):** in all five brackets,
  `Z = min(M_OFF1, M_OFF2) − M_ON − max(5 ms, |M_OFF2 − M_OFF1|) > 0`.
- **Rule 4 (overall):** the gain is established exactly when rules 1, 2 and 3 all pass.

### Preregistration

Each qualification run freezes, in a dated record before its first counted sample: the scripts
and clock; `Q`'s and `M`'s parameters (concurrencies, seed list, clip and file manifests, offset
rule, run lengths, file admission, idle-clock rule, minimum case coverage); the step rule's
prefix, snapshot rule, plans, compositions, aggregation and adapter identity; `U_api`'s
procedure, input generator and rules; U(1)'s audio, plan and procedure; and the exactness runs'
headroom: the run's maximum request count (the admission limit for `Q` and `M`, `N + 4` for
the uncovered steps), bytes per page, copy figure and margin (TXG-QUAL-029). A change after that
voids the run.

### Memory, refusals and recovery

- Each capture's call, transaction, per-kind and per-cell figures and its transient peak are
  measured and reported.
- An installed-engine check under a fixed block count with the estimate enabled shows that core's
  logged graph-memory estimate includes the profiling capture (it exceeds `B_hook`'s estimate by at
  least the profiling transaction total, both estimates including one charged margin, within the
  run-to-run spread of two `B_hook` startups, which the check reports) and that the serving call
  total is at most the profiling call total. The difference between the KV budgets core computes
  for the two engines is reported, not bounded, because transient peaks also differ.
- Under a fixed budget the startup log carries the uncharged-memory WARNING with the figures; with
  the profiling capture suppressed where the configuration requires it, startup fails.
- An installed-engine check under the extensible KV cache shows the serving capture with only
  core's minimum blocks backed, the extension's binding accepted without a latch, the
  measured-memory INFO line (When core's estimate does not run), steps after the extension
  replaying cells with no `signature` fallback, and transcripts equal to `B_hook`'s; a second arm
  under a profiled budget with the extensible KV cache off shows the profiling capture over a
  one-block profiling KV cache; a third arm under a fixed `--num-gpu-blocks-override` shows both
  captures over block 0 passing. Block-0 capture has no GPU evidence under any sizing until these
  arms run. The round that runs these arms also runs `registration`'s REG-STATE-019 arm with
  transaction graphs on and its REG-ENG-032 arm with transaction graphs on under the hook
  configuration.
- Each refusal refuses its configuration: the option under `--enforce-eager`, at a geometry other
  than 80 or 160 ms, with native burst at 160 ms (at 80 ms the lookahead check refuses it first),
  every other row of Configuration checks (each unmet restricted-startup condition with its
  remedy, a population above the admission limit, and a core without `state_group_layouts`), and
  a core without #67583's hooks
  (`registration`).
- **Recovery.** A test-only fault injected after the first launch of a split step latches the
  binding, the engine reports dead (`EngineDeadError` to the client), and a fresh engine process
  with the same configuration captures, passes post-warmup verification and serves a stream whose
  transcript equals `B_hook`'s.
- The TLA+ system model checks without violations (System model).

Tier 4 (mixed load) is met and claimed separately with the option on.

## System model

The capture lifecycle on the hook and the split step are modelled in TLA+ and checked with TLC,
in `models/transaction-graph/`: one specification and configuration file per checked model, each
with a header stating the TLC version and what each invariant means. Every change to this
lifecycle (a hook's behaviour, an ordering rule, a failure path) updates the model in the same
change. It models one worker with a small `N` and bounded steps:
- **Actions:** every hook call of the Capture lifecycle table, in core's startup order, with
  core's binding of the profiling and of the serving KV cache's views, the state-group layout set
  at each binding and cleared when core releases the profiling KV cache after `clear_cudagraphs`,
  a binding after the serving capture with views that fit the recorded layout (an equal one, and
  the extensible KV cache's growth with no more pages) and with views that do not fit (Repeated
  bindings),
  the pool-access check able to fail, each
  cell capture able to fail, the memory comparison able to overrun, `clear_cudagraphs` also after
  a failed or skipped capture, a second serving capture attempt after a successful and after a
  failed serving capture, and `shutdown` from any state; scheduled steps (one-kind replay, split
  replay, counted fallback, pre-launch failure, a failing cancel of the held reservation before
  the first launch, failure at the first launch, between the launches or at the stage, a stage
  that raises before and one that raises after publication, and a declined step's option-off
  reservation, scatter and stage each able to fail, as one nondeterministic choice per declined
  step); a step before post-warmup verification, after whose propagated error the engine may die;
  all-failed bypass steps on an unlatched worker, before and after verification, and on a latched
  worker; cleanup-only steps; dummy steps; and, as nondeterministic choices, the KV-cache sizing
  (the extensible KV cache, a profiled budget with it off, a fixed block count, a fixed budget),
  whether a profiling capture is required (none under the extensible KV cache, a startup plan's
  budget there included; with it off, including a budget fixed by a startup plan after
  construction) and whether it runs.
- **Invariants:** no replay outside a scheduled step after a passed verification, other than a
  capture's validation replay of rejected rows inside its own capture call; no step replay over
  profiling pools; nothing this component derived from the profiling pools (cells, buffers,
  signatures, recorded layouts), the margin included, retained after `clear_cudagraphs`, and the
  state-group layout read only inside a capture call; at most one serving capture; a serving capture with no profiling capture
  before it is reported as measured under the extensible KV cache, as uncharged when the
  extensible KV cache is off and none was required, or fails startup; a serving
  overrun fails startup; each reservation, a covered step's or a declined step's option-off one,
  is resolved by exactly one successful stage or one cancel (a stage that raised before
  publication leaves it cancellable, a cancel after a successful stage is a no-op) and is left
  unresolved after a worker-fatal failure only if its cancellation failed; a step whose first launch ran never falls back and never stages after a
  failure; after a latch, no step that schedules a row completes, an all-failed bypass step
  included (a cleanup-only step may); an all-failed bypass step replays no cell, counts no
  fallback and holds no reservation;
  after any failure in a capture call, nothing that call created remains captured, segment graphs
  included; after `shutdown`, nothing
  remains captured.
- **Liveness under fairness:** under cooperative teardown (a worker that runs the runner's
  shutdown), every latched or failed worker reaches `shutdown`; and every startup whose
  construction succeeded and that is neither shut down cooperatively nor killed before it serves
  eventually serves or fails with a reported error. Core gives no shutdown-hook guarantee for a
  killed worker or a failed construction, and the model claims none.

The checked models carry the target core's lifecycle above. `CaptureLifecycle.tla` models core's
binding of the profiling and serving KV caches with the state-group layout set and cleared, the
profiling cache released after `clear_cudagraphs` on both the success and the failure path, the
pool-access check of the capture call (a layout contract break, and too few pages in the serving
capture under every sizing but a fixed block count), the extension under the extensible KV cache
after `post_warmup`, which rebinds the views while the accessor's layout stays unchanged and
which serving waits for, accepted, and a binding whose views do not fit, latched. Its invariants include that the layout is read only inside a capture call, for that call's own KV
cache, and held by no object between calls. `SplitStep.tla` classifies dummy steps as
`prepare_inputs` does, accepts a binding of no more pages between steps and latches a binding
whose views do not fit. Per-request failure delivery
through `take_failed_requests` is outside both models.

## User documents

This component owns the option's sections of the user manual (`docs/manual.md`) and of
`docs/evidence.md`. The manual's section gives a serve command that runs the hook configuration
(no `--enforce-eager`) under the restricted startup (`-cc.cudagraph_mode none -cc.mode none
--no-async-scheduling`) with core's default KV-cache sizing, the default segment arms, `num_lookahead_tokens` 0 or 1 and `transaction_graph_populations` in
`--hf-overrides`, a `--max-model-len` that covers the geometry (`registration`, REG-ENG-010), at
the `--max-num-seqs` and `N` that the tier 5 run verified to fit the A40. It states that a fixed
budget must leave room for the graphs, which the startup log reports, and that under core's
default sizing the KV cache is sized after the captures, from the memory they leave; the limits (one geometry per server, 80 or 160 ms; `N` bounded by `max_num_seqs` and the
admission limit `--max-num-active-seqs`; a fixed block count or budget, when set, that holds at
least the admission limit plus one state pages; steps beyond `N` CHUNK or other rows, and steps with a
multi-token row, run option-off and are counted; refused under `--enforce-eager`,
with native burst, and outside the restricted startup, each setting with its reason); that the server needs a frontend release that opens streams by locale only;
and the tier each claim met, with its build. Its troubleshooting text states that under core's
default KV-cache sizing, a state page count below the admission limit plus one is refused by
core's scheduler after the captures with "Lower max_num_seqs or enlarge the cache", even when
`--max-num-active-seqs` sets the limit; the startup log's measured graph-memory line, logged
before it, gives the graph totals, and the remedies are a lower admission limit, a smaller
largest population, a larger `--gpu-memory-utilization`, or transaction graphs off. The evidence
document's section states the measured
graph memory, transient peaks and capture time with the build, device, geometry and `N`, and the
`(K, E)` histogram `N` was chosen from.

## Permitted dependencies

The option uses these internal dependencies of the model adapter. `registration`'s table and
import-time check own the ones the option-off model also uses: `ModelState.post_warmup` is an
unconditional row (REG-010), and `capture_pool` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`
are conditional on graphed execution (REG-014), because the default segment arms use them with the
option off. `ModelState.state_group_layouts` is the one dependency only the option has: a
conditional row of `registration`'s table, which model construction checks while the option is on
and model import does not (REG-018); on a core without it the configuration checks refuse the
option (TXG-CFG-016).

| Module | Names | Use by this component |
| --- | --- | --- |
| `vllm.v1.worker.gpu.model_states.interface` | `ModelState.state_group_layouts` (a property of a permitted class) | the pools of each capture call (Rules of the capture call); conditional on the option (REG-018) |
| `vllm.v1.worker.utils` | the attributes `StateGroupLayout.layers` and `StateComponentLayout.tensor` of the layouts the accessor returns, read without importing either class | the pool-access check and the capture rows' pools (REG-018) |
| `vllm.compilation.cudagraph_pool` | `capture_pool` | captures into the pool core passes (REG-014) |
| `vllm.v1.worker.gpu.model_states.interface` | `ModelState.post_warmup` (a method of a permitted class) | post-warmup verification (REG-010) |
| `vllm.envs` | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | whether a profiling capture is required (REG-014) |

`torch.cuda.CUDAGraph`, `torch.cuda.graph` and `torch.cuda.mem_get_info` are PyTorch, not vLLM.
The counter reader uses `AsyncLLM.collective_rpc`, part of the public client surface.

## Cascade obligations

Each sibling owner applies its own; this list does not change their documents.

- **`registration`:** default arms (REG-CARD-008); one startup geometry (REG-LA-001 to
  REG-LA-007); an explicit `--max-num-seqs` (REG-ENG-007); hook capture of the graphed encoder and
  decoder, replacing the refusal in `hook_captured_arm` (a code change); REG-MEM-002 and
  REG-MEM-004 for segment-graph capture, release, rebuild, failure handling and the aggregate
  serving-over-profiling rule with the option off; REG-VAL-003 (Where the checks run); REG-ENG-003
  (A second serving capture); the six-hook core check, with `post_warmup` in
  `MODEL_STATE_GRAPH_HOOKS`; REG-010, REG-014 and REG-018 specifying Permitted dependencies, with
  `state_group_layouts` a conditional row; `transaction_graphs_on` for every on/off decision
  (TXG-CFG-014); the one measurement implementation both parts call (REG-MEM-027); the restricted
  startup's exceptions to the accepted scheduling and runner modes (REG-ENG-001, REG-ENG-011,
  REG-ENG-026) and remedies (REG-ENG-005); the state layer's report of every binding after the
  serving capture, which the binding accepts when the views fit (TXG-LIFE-009); the state layer's view lifetime, by reference to the bound
  storage and dropped in `clear_cudagraphs` (REG-STATE-018); and the failure at planning of a
  multi-token row, with every other row at its own token positions (REG-STATE-014).
- **`session`:** SESS-FAC-004, SESS-OPEN-001/002, SESS-MODEL-002 and SESS-RT-001: the factory's
  `open` takes only a locale, every session runs at the server's one startup geometry, and a
  frontend neither passes nor discovers a cadence; with a seam-conformance test. SESS-MODEL-001
  and SESS-MODEL-002 fail a foreign-geometry row at planning, outside the row plan, and
  SESS-MODEL-003 fails a stream request whose collected row status is nonzero.
- **`transcription`:** file steps as stream-role rows (FILE-ROW-002, FILE-ROW-003 and FILE-ROW-010 with the token
  IDs always present, the end step as REPLAY or FLUSH); the stream-carrier role mismatch of a file
  row in any role (FILE-ROW-014); FILE-PROMPT-002 and one geometry helper;
  no request-chosen cadence remains as deferred work. It also owns the **queue-count transport**, for every row with the option
  on or off, replacing its synchronous read of the label book and the host-side decrement. It
  defines: a typed `CommitPlan` field, a device int32 tensor of one count per row that `reserve`
  validates like `progress` (this component fills it from `remaining_step[:P]`); a preallocated
  pinned host buffer that `stage` copies into without blocking or allocating; the count on
  `SessionStatusReport`, set only for a clean row whose stage-time lease held; an amended
  collection seam, because `postprocess_state` today receives only the status dictionary and
  failed-request set of `collect_commit_status`, which consumes the reports: an amended return
  value or a consume-once accessor that yields clean, lease-validated counts with request ID and
  admission generation, independent of `progress_key`; application only after collection and only
  to the file progress of the same request and generation; a fresh file's count of 0 before its
  first step; and failure of a continuing file row that has no count, before planning. Its tests
  cover a rejected DRAIN row keeping the prior count, a fresh file's first count, zero remaining,
  a stale generation, request-ID reuse, and the full sink → model → model-state path.
- **Repository instructions:** `AGENTS.md` lists `TXG` among the owned prefixes.

## Default-on gate

Turning the option on by default needs the maintainers' approval and, against a declared
baseline: cells, exactness and latency evidence at 320, 560 and 1120 ms (the published checkpoint's
default is 320 ms); served fallback rates and the `(K, E)` histogram under admitted, staggered sessions
and whole-file load; A40 fit, transient peaks and startup time of both captures at each
geometry, with the resident capacity they cost accepted explicitly; the counter reader in routine
use under a multiprocess engine; U(1) calibrated with the option on and used by admission; the
`--enforce-eager` resolution of Configuration checks; the restricted startup lifted (Future
roadmap), because a default-on option cannot require settings that the default serve command does
not set; and a frontend release that opens streams by locale only.

## Implementation notes

- **Modules.** The binding, cells, step arena, classification, signature and counters live in
  `nemotron_asr/transaction_graph.py`; the configuration checks live in a module importing no
  vLLM.
- **Phase split.** `advance.advance_model_rows` is composed of prepare, body and commit functions
  that take the same arguments in total, so existing callers keep one entry point. The body has no
  branch on `has_file`, no variable-length fresh index or initialization and no per-call staging
  of fresh positions.
- **Nested eager paths.** The `eager-graphed` encoder binding exposes, at each count, an uncaptured
  transition over its own storage, as the dense decoder's `uncaptured_decode_fn` does.
- **Counters across engines.** Module-level state (batch statistics) is reset when a binding is
  built, so a second engine in one process starts clean.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Steps that mix CHUNK and other rows | Split by kind: a chunk cell at `K` and an echo cell at `E`, two launches under one reservation | One cell per exact `(P, K)`; padding the other rows into a P-wide chunk cell; running mixed steps option-off; native burst | Under an illustrative assumption (each row CHUNK with probability 0.9, independently), 57 %, 81 % and 97 % of steps at 8, 16 and 32 rows are mixed, and running them option-off would leave those steps uncovered; the served mix is unmeasured. `(P, K)` cells need `P + 1` cells per population (about 594 for 1 to 33). Padding changes the encoder's batch dimension, so bitwise equality is unlikely, and each padded row costs encoder work. Native burst has its own exactness programme and is out of the first release. The split keeps each half at the option-off shapes and costs one extra launch |
| Cell range | Both kinds at every count 1 to `N` | Cells only at declared populations | Served populations drift step to step, so a sparse list rarely matches a real `K` or `E`; a contiguous range makes coverage a single bound |
| Option value | A list whose largest entry fixes `N` | A single maximum | The HLD defines the option as declared step populations; the list keeps that surface while the cell range depends only on its maximum |
| Freshness | A per-row select inside the chunk cell | A cell dimension (`chunk-fresh`, `chunk-continuing`); separate launches for fresh and continuing rows | A dimension triples cells and sends every step with both to fallback; separating fresh from continuing rows changes the encoder's batch dimension and so its numerics |
| Fresh-row state | Gather the zero-initial state tensors by an initial-state index that names the null page for a fresh row, and select the initial book row after the book's gather | A select of the initial state after the gather of every state tensor; the variable-length continuing index; a masked-load kernel that skips fresh rows' loads; a select of the book row only, relying on core's initialization of a newly allocated page | A select after every gather reads and writes every gathered state tensor once more per step (three per encoder layer and eight more), a cost that grows with the population; the null page is zero and never stored to, so the index gives the same values with one small select. The variable-length index is data-dependent and cannot be captured; a masked-load kernel adds a JIT specialization for no change in values; the book-row-only select rests on an unverified coincidence of the fresh flag with core's initialization of a newly allocated page, which the null page does not need |
| Launch boundary | Validation, signatures, allocations and the reservation before the first launch; worker-fatal after it | A reservation per launch; option-off fallback of the second half after a failed first launch | The first launch may already have scattered; a second-half fallback would commit a step neither path produced, and two reservations break the sink's one-in-flight rule |
| Reservation disposition | Exactly one successful stage or one cancel resolves a reservation, a covered step's or the option-off transaction's; a stage that raises before publication leaves it cancellable; a cancel after a successful stage is a no-op | Treating a called stage as the disposition whether or not it returns; limiting the rule and its model check to covered steps | A stage that raised and counted as the disposition would leave the sink's one in-flight reservation held, so every later reserve fails; every declined step uses the option-off reservation, so the rule and its check must cover it too |
| Host inputs | One step arena with a section per kind, one upload, double-buffered pinned host side | One arena per cell (two uploads); per-call staging; captured host-to-device copies | One upload per step; captured copies replay a frozen source; per-call staging adds launches |
| Static inputs and outputs | Shared step-level buffers (Step arena, static inputs and step outputs) | Per-cell buffers with a host-side permutation; reordering the plan CHUNK-first | One buffer set; outputs stay in core's row order with no extra kernels |
| Capture order | Largest first, chunk cells then echo cells, on core's one stream and pool | Smallest first; interleaved kinds; a stream per cell | Freed graph-pool blocks are reusable only on the stream that allocated them, and a larger capture's freed intermediates fit a smaller one |
| Capture rows' page | Every capture row names block 0 | Pages `1 … k`; model-owned scratch pages; asking core to back `N + 1` pages before the model-state capture | Core guarantees memory only behind block 0 at every capture, so pages `1 … k` may have no memory. Model-owned scratch pages, which core permits, would need a per-step copy of every row's state between the scratch and its page. Core keeps its capture rule because backing pages before capture conflicts with its warmup releasing blocks before capture. The capture rows are rejected, so naming one page `k` times stores nothing, and a predication defect stores to block 0, where the self-check catches it. With every capture row on block 0 the self-check does not separate the initial-state index's arms or detect a baked page address; TXG-EQ-003's fresh and continuing replays of the same cell at every count (sentinel fresh pages, nonzero continuing pages) and its mixed-freshness replays at every count of 2 or more, and TXG-QUAL-003's per-step comparison, detect both |
| Preflight of capture rows | A capture-only mode of the structural preflight, passed only by a capture call's cell programs through `prepare_model_rows`, that requires every row to name the null block ID and an empty live set, and waives exactly three row checks (the null block ID for a real row, a block ID named twice, a block ID outside the live set) while keeping the bounds and the live-set validation | Drop those refusals for every plan; a second preflight function for capture rows; distinct block IDs below the page count | Dropping the refusals removes the protection real plans need. A second function would duplicate every other structural check. Distinct IDs would read unbacked pages. Requiring the null block ID in every row keeps the mode from naming a live page; `advance_model_rows`'s interface (TXG-TXN-007) is unchanged |
| Masked-scatter alignment | Every resident-pool argument a step passes after startup fits one launched by the serving capture's scatter warm-up or a cell's eager reference (address, page shape, strides, dtype, and a first dimension no larger); the warm-up launches the component pools and the projected views, and its record includes alignment | Padding the state layout so that every view is 16-byte aligned; an aligned base pointer plus an element offset in the Triton kernel | The pools' views have fixed addresses, so warm-up, capture and serving share one alignment specialization whatever a view's offset; padding every component would change the model's state layout and break the projection that reads adjacent scalar fields (the replay book, the frontend counters) as one matrix, for no correctness gain; changing the kernel's addressing would change the option-off transaction's kernel without measured need. Padding remains a measured optimization (Deferred 10). The kernel does not specialize on the page count, so the extension's views, of no more pages, select the warmed specialization |
| Graph-memory accounting | Core's accounting: its estimate through the profiling capture when it runs, or, under the extensible KV cache, its reading of free memory after the serving capture; no model pool | A model-owned admission inventory and graph pool | Core's estimate, or its reading after capture, already measures the same graphs before it sizes the state pages; a model admission needs its own fit rule and release check |
| Profiling capture content | The serving capture's procedure and cells over the profiling pools, plus the charged margin | A synthetic upper bound; extrapolation from one cell | The same procedure gives the same allocation sequence; extrapolation is unsound because cells do not share memory predictably |
| Serving against profiling | Aggregate call totals; startup fails when the serving total exceeds the profiling total, which includes one flat charged margin | A per-cell margin; a per-cell inequality; a warning only; an uncharged tolerance | Rounding arises per allocator segment over the whole call, so a per-cell margin overcharges at large `N` and undercharges at small `N`; per-cell growth is order-dependent under reuse; a warning serves with uncharged memory where core estimated; an uncharged tolerance hides real overruns. The margin is memory core charges, so the strict comparison still bounds serving memory |
| Missing profiling capture | Required from the effective cache configuration at the serving capture; logged as measured whenever the effective extensible KV cache is on, otherwise as uncharged | Treat the extensible KV cache as uncharged; require the estimate by turning the extensible KV cache off | Under the extensible KV cache core reads free memory after the captures and sizes the cache from it, so "uncharged" would misreport the budget; turning the cache off re-imposes a startup setting the default serve command lacks |
| Startup on the target core | A restricted startup of synchronous scheduling and an explicit NONE runner and compilation mode, under every KV-cache sizing core offers; every other startup refused, naming each unmet condition and its remedy | Also require a `--num-gpu-blocks-override` or a positive `--kv-cache-memory-bytes`; capture the cells over model-owned scratch pages; accept every startup | Core guarantees memory behind block 0 at every capture and keeps the layout's base, strides and page count until release, so a capture over block 0 is valid under every sizing and needs no setting a serve command without KV-cache options lacks. Core permits a capture over model-owned scratch pages, but every step would then copy each row's state between the scratch and its page. Asynchronous scheduling and the reduced runner modes have no transaction-graph evidence |
| Pool source at capture | Core's accessor `ModelState.state_group_layouts`, checked against the state layer's bound views, keeping only plain values | The state layer's own views (`component_pools`), which hold the same tensors | The accessor is core's contract that a held page keeps its block ID and address until terminal finish, and core sets it only while a KV cache is bound. The layer holds its views by reference to the bound storage and drops them when core rebinds, when the bound object changes and in `clear_cudagraphs` (`registration`, REG-STATE-018), so both name the same tensors at a capture; the identity check keeps the two in agreement |
| A binding after the serving capture | Accept views that fit the recorded layout; latch on any other | Exact equality; reading the accessor at each binding; ignoring later bindings | Exact equality latches at the extensible KV cache's growth, which binds no more pages over the same base, and fewer when the final count is below the capacity. The accessor is read only inside capture calls (TXG-CAP-008), and core keeps the layout fixed, so a read would return the recorded values. Ignoring bindings would miss a pool moved to another address |
| Option-off body regression | Tier 5 rests on the option-on/off comparisons of one body (TXG-QUAL-002 to TXG-QUAL-006); no check compares the option-off step time with a commit from before the phase split | A preregistered non-regression check of the option-off step time against the newest commit whose `advance_model_rows` is not composed of prepare, body and commit functions; rebasing that commit onto the target core | That commit runs only on the state-pool core, which the plugin refuses (REG-002), so the check cannot run on the target core. Rebasing it would time a body that no release ships on a core it never ran on. Its result on the state-pool core stays in `docs/evidence.md` as a historical record |
| Feed-to-park client path | `AsyncLLM.generate` with an input generator of packer-built `StreamingInput` units, in a multiprocess engine | The session factory's `feed`; an `LLM.generate` arm | The factory is not available on the target core (HLD, Approach, section 2). An input generator is the client path the target core offers and the one core's `/v1/realtime` uses. `LLM.generate` takes complete prompts only, so an `LLM` arm can time only a whole one-unit request, not a feed into an open stream (Future roadmap) |
| Steps with a multi-token row | The step falls back with reason `multi-token`; the option-off transaction places every row by its token positions | Cells and static inputs addressed by token position; failing every request of the step | Planning fails the multi-token row, and only a direct engine client produces one. Token-position addressing in the cells would size the static inputs by `max_num_batched_tokens` and change the arena and every cell for a step that plugin-packed input never produces. Failing the whole step would fail requests that did nothing wrong |
| Populations above the admission limit | Refused (TXG-CFG-020) | Accepted; `N` clamped to the limit | Core schedules at most the admission limit's rows per step, so cells above it are captured, charged and never replayed. Clamping would capture an inventory other than the one declared, without telling the operator |
| Block count | At least the admission limit plus one: refused in the configuration hook under `num_gpu_blocks_override` (TXG-CFG-005), and checked against the layout's page count at the serving capture's pool access under every other sizing (TXG-CAP-008); no check at the profiling capture | At least `N + 1`; a check at both captures; a check of the final count after the extension; leave every check to core's scheduler | Core builds its scheduler after the captures, so a count between `N + 1` and the limit would capture `2N` cells, twice when core's estimate runs, and then fail with core's message. The profiling cache holds one block under a profiled budget with the extensible KV cache off and the explicit NONE the restricted startup requires (core's default sizing runs no profiling capture), and its count is not the serving count, so a check there would refuse every profiled startup. Under the extensible KV cache the final count is known only after the one capture; the state layer's binding report at the extension could check it, but core's scheduler refuses a final count too small itself, with a message that names the remedy (`vllm/v1/core/sched/scheduler.py:166-170`), and only one capture has run by then |
| Pool-access remedy | For too few blocks, a larger block count or budget, or a lower admission limit (TXG-MEM-017); for a break of core's state-group layout contract, no configuration remedy (TXG-CAP-008) | TXG-MEM-013's smaller `--kv-cache-memory-bytes`; the remedies of TXG-MEM-011 to TXG-MEM-013 for every pool-access failure | A block-count failure means the budget holds too few state pages, and a smaller budget holds fewer still. TXG-MEM-013 answers a serving capture that ran out of device memory after pool access. A layout count other than one, or a component tensor that is not the layer's bound view, is a break of core's contract that no setting changes, so a configuration remedy would direct the operator to the wrong cause |
| Pages of rejected fresh rows | Initialized before the transaction whether or not the row is rejected; preservation starts after initialization (Fresh rows) | Initialize only the rows that commit; preserve a rejected fresh row's recycled contents | The body finds a per-row rejection on the device, after the initialization has run, so initializing only committed rows would need every rejection on the host before the step. Preserving recycled contents would leave an earlier request's state in a page that the rejected request holds |
| Exactness copy extent | The detached copy holds block 0 and one page for each request the run can hold, the only pages the adapter's lowest-free assignment gives its requests: pages 0 to the admission limit for `Q` and `M` (TXG-QUAL-003), 0 to `N + 4` for the uncovered steps (TXG-QUAL-007), from one function of the mode's most requests; every other committed page is digested in place before the run and after it, and a changed digest fails the run; a final committed count below the extent fails the run before its first step; the headroom is the extent's pages plus a stated margin (TXG-QUAL-029) | Copy every committed page, up to the final committed count, under a headroom of the whole committed cache | Under the extensible KV cache the committed count is sized from the memory the headroom leaves, so a whole-cache copy needs a headroom near half the device: the KV cache under test would shrink far below serving's, and the copy's figure would depend on the count it shrinks. The written pages are only the extent's pages, a figure known before the run; a run whose final count falls below the extent fails, because a whole-cache copy in its place would bring back the shrunken cache. A digest still catches a write to an unassigned committed page, which a narrower copy alone would miss |
| Qualification adapter's call order | Core's runner order: finished requests removed first, then each request delivered as `NewRequestData` in `scheduled_new_reqs` (a new request or a stream's next turn) removed and at once added; a cached request gets neither call (TXG-QUAL-026) | All removals, then all additions; a removal and an addition for every scheduled request | With all removals first, the next request's removal retires a continuing request's pending removal (REG-STATE-011), and the model state then refuses that request's continuing turn (REG-STATE-012), so the adapter would not drive the model state as core does. Core makes no model-state call for a cached request; a removal there would retire a running file at once (REG-STATE-016), and the addition after it would raise (REG-STATE-012), failing every file at its second step |
| Option-off baseline | `B_hook`: the default segment arms under the hook configuration, no CHUNK-bucket graphs | The eager encoder; the `--enforce-eager` graphed configuration | `B_hook` is the default at the same geometry, so the option is the only difference; the eager encoder would credit the option with the encoder graphs' gain |
| CHUNK-bucket graphs | Refused under the option | Accept the binding inside or beside the cells | The binding graphs an already-gathered bucket, which a chunk cell captures in full; it is experimental, covers three counts at 160 ms, and would only change fallback steps |
| Endpoint arguments | The forward passes the endpoint pools and EOU token on every step, option on or off; cells run with `endpoint_enabled` true and the policy-free endpoint bookkeeping; a step with an endpoint-policy row or a forced-EOU candidate, marked conservatively on the host, falls back by `signature` | Cells with `endpoint_enabled` false and the endpoint arguments passed only for a step with a policy row; forcing EOU only for a row that carries a policy; cells that also cover policy rows and forced EOU | Passing the arguments only for policy steps makes the option change the transaction: policy-free rows' endpoint pages would differ from `B_hook`'s, and a policy-free row that schedules the EOU token would be rejected instead of forced, against TXG-EQ-001. Forcing EOU only for policy rows changes `B_hook` itself, a session-protocol decision outside this component. Policy evaluation and forced EOU arise only from a direct engine client and have no qualification evidence, so they wait for cells of their own (Deferred 6) |
| File rows | Rows of stream roles in the same cells, chosen by a per-row file mask | Separate file cells; leave file rows uncovered | Separate cells cost memory for work the streams already do; uncovered file rows would make every step a file shares run option-off |
| File samples | Outside the arena (File samples) | Pack samples into the arena | See File samples |
| Planning-failed rows | Kept in the row plan at their scheduled positions as non-CHUNK rows with a planned-rejection bit, counted in `P` and `E`; failed from a planning-failed set in `postprocess_state`; a step whose every row failed runs no transaction | Exclude them from the row plan and compact the surviving rows | The model state already keeps a failed file row in its step plan and binds a foreign-geometry unit as non-CHUNK, so the retained row is the existing model. It keeps every input, binding and output at the runner's position, with no remap and no restoration of output cardinality; the masked scatter leaves the pages untouched; an explicit bit, not a role-check coincidence, rejects fresh rows too; and the host set delivers the failure even when no status is collected. Exclusion would need a position map on both sides of the transaction and an out-of-band output for each excluded row |
| Queue-count ownership | The body writes each row's count; `transcription` owns the transport to file planning | This component owns the transport | The transport runs with the option off and replaces `transcription`'s synchronous read; owning it here would make `transcription` depend on this component, which the HLD's component rule forbids |
| Configuration checks | One function, two callers (Where the checks run) | Model construction only | Fails in the API server before workers start |
| `--enforce-eager` | Explicit request refused; once default-on, the default resolves off with a log line | Silently off; always refused | An explicit request is never ignored; refusing a default would break every eager deployment |
| Second serving capture | Refused at entry once a serving capture was attempted, whether or not it succeeded; worker-fatal, with the latch and existing graphs unchanged until shutdown; recovery by restart | Recapture over live pools; refusal only while serving cells or segment graphs exist; latching the binding at the refusal | Recapture would bake new addresses while streams hold slots, with no estimate for the memory. A failed attempt discards its cells and segment graphs, so a refusal keyed on their existence would admit a retry. The refusal changes nothing, like the other entry refusals; latching there would break that uniformity for no gain, since core treats the error as worker-fatal |
| All-failed bypass and verification | The bypass checks the latch only; an unlatched all-failed step takes it whether or not post-warmup verification passed | Requiring verification before the bypass, as `execute` does | Verification protects cell replay; the bypass replays no cell and reserves nothing, so a guard would make the option raise where the option-off path succeeds |
| Failure recovery | Worker-fatal latch; engine restart; tested end to end | Roll back pages; retry option-off after a launch error | The launches scatter, so partial mutation is unknowable; rollback needs hot-path snapshots of every touched page |
| Signature mismatch | Count, warn once per field, run option-off | Recapture on demand; raise | Recapture needs uncharged memory and stalls; the option-off transaction is the exact reference |
| The pools field of a cell's signature | A step's view matches the cell's entry when it fits it | Exact equality; dropping the first dimension | Exact equality sends every step after an extension that shrinks the page count to a `signature` fallback. Dropping the first dimension would accept a view with more pages than the cell captured, which the captured bounds do not cover |
| Process constants | Worker-fatal on change | Part of the signature | Every cell and the segment graphs were captured under the same settings, so a fallback would be stale too |
| Post-warmup check | Plain-value layout, graph presence, identities, buffer addresses, constants; no replay and no device allocation, the projected layout re-derived over one host shadow page | A validation replay; a first-step check | `post_warmup` runs outside any step, and each cell's capture already ran its validation replay; a first-step check lets the engine report ready before a mismatch is found |
| Qualification workload | Independently paced streams and files at the transaction level, replayed and option-off per step, with preregistered parameters and minimum case coverage, plus served runs | Phase-aligned schedules with filler rows | Aligned schedules produce steps serving never produces and hide mixed steps; the transaction-level driver keeps the page-by-page comparison; minimum coverage stops a run from passing without the cases it must exercise |
| Step-rule plans | One recorded plan per composition, replayed from one restored snapshot | Timing free-running steps; a plan sequence advanced across steps | Free-running steps differ between legs, so medians would compare different work; an advancing sequence cannot sustain 23 steps of label drains or FLUSH rows on fixed slots |
| Step-rule inventory | The deployed inventory under a fixed block count with core's estimate enabled; a reduced inventory is diagnostic only | Capture only the cells the legs replay | A pass on an inventory that cannot start says nothing about the deployed engine; the inventory changes memory, capture order and pool reuse |
| Step rule groups | One-kind CHUNK steps at 1, 31, 63 and split steps at 31, 63 with preregistered compositions, each group passing on its own | One pooled rule over all steps | Pooling could hide a regression of the second launch behind one-kind gains |
| Calibration identity | Includes the process constants, scheduling and batching limits, the KV-cache sizing, the option's value and a served replay fraction measured at that identity | Model and device only | An option-on figure would misstate capacity on an engine whose steps fall back or that schedules or computes differently |

## Open Questions & Future Decisions

### Deferred
1. **Margin size.** Whether the flat 64 MiB margin absorbs the serving rounding at the deployed
   inventory; the first A40 measurement of `r` and the absorbed rounding sets it.
2. **Fit, reuse and `N`.** The A40 fit of `B_hook`'s segment graphs, `2N` cells and both transient
   peaks at 80 and 160 ms, the cross-cell reuse of largest-first capture, and the `N` the served
   histogram selects (Fit analysis).
3. **Startup time.** Whether to refuse a declaration whose projected capture time for both
   captures exceeds part of `VLLM_ENGINE_READY_TIMEOUT_S`.
4. **Other encoder arms under the option.** Whether to accept the `eager` encoder arm, which drops
   the encoder graphs' memory at the cost of slower fallback steps and a different baseline, and
   the `dense-graphed` encoder, with its own flat-capture and exactness evidence.
5. **More geometries.** Cells at 320, 560 and 1120 ms, each with exactness and latency evidence.
6. **Endpoint policies and forced EOU.** Cells run the policy-free endpoint bookkeeping, and a
   step with an endpoint-policy row or a forced-EOU candidate is a `signature` fallback. Whether
   such steps get cells, with exactness evidence for policy evaluation and forced EOU, once a
   policy binding exists.
7. **Admission's use of U(1).** Which component turns U(1) and the ladder into a stream count.
8. **Lazily bound encoder buffers.** Whether a long stream rebinds them after warm-up, causing a
   permanent `signature` fallback; to be probed on long streams at both geometries.
9. **Second-launch cost.** The echo launch's cost at small `E` against the echo rows' option-off
   work, from the step rule's split group.
10. **Scatter alignment padding.** Whether padding the resident-pool views to 16-byte alignment
    speeds the masked scatter enough to justify changing the state layout; a measured
    optimization, not a correctness requirement (Warm-up domains).

## References

- Core hooks: vllm#67583. The target core, `pdtgct/vllm` at `cbf13fed4a`:
  `vllm/v1/worker/gpu/model_states/interface.py` (the hooks and `state_group_layouts`);
  `vllm/v1/worker/utils.py` (`StateGroupLayout`, `StateComponentLayout`,
  `get_state_group_layouts`); `vllm/v1/worker/gpu/model_runner.py` (the layout set after each
  KV-cache binding); `vllm/v1/worker/gpu/cudagraph_utils.py` (the graph-memory estimate, the
  profiling KV cache's size and its teardown); `vllm/v1/core/kv_cache_planning.py`
  (`may_override_num_blocks`); `vllm/v1/worker/extensible_kv_cache.py`; `vllm/config/vllm.py`
  (asynchronous scheduling and extensible KV cache resolution);
  `vllm/compilation/cudagraph_pool.py`; `vllm/utils/jit_monitor.py`.
- This repository, under `nemotron_asr/`: `advance.py`, `commit_sink.py`, `model_state_v2.py`,
  `nemotron_asr.py`, `decode_graph.py`, `encoder_execution.py`, `state_scatter.py`,
  `file_rows.py`; and `vllm_nemotron_asr_streaming/engine_flags.py`.
- Sibling designs: `registration`, `transcription`, `session`.

## Future roadmap

- **Transaction graphs beyond the restricted startup.** On the target core the option needs
  synchronous scheduling and an explicit NONE runner and compilation mode (Configuration checks).
  Triggers: for Runner modes, a transaction-graph run with the option on under each reduced mode
  that core resolves to NONE (an unset mode, FULL_DECODE_ONLY, PIECEWISE and FULL_AND_PIECEWISE
  under `VLLM_COMPILE`), with exactness and served transcripts equal to `B_hook`'s; for
  Scheduling, the qualification under asynchronous scheduling. Shape: each condition is dropped
  when its trigger fires, and `B_hook` and the qualification specs take the new settings. The
  `registration` design's Future roadmap item 8 holds the same item.
- **Calibration across KV-cache sizings.** A calibration applies only at its own sizing. Trigger:
  the U(1) and ladder of the default sizing and of the step rule's fixed override agree within the
  preregistered noise. Shape: a preregistered transfer rule in TXG-CAL-003, and TXG-DEF-002
  applies a record across the two sizings.
- **Sleep mode with transaction graphs.** Core keeps the layout across sleep and wake and
  recaptures model-state graphs once after a full wake with no state held. Trigger:
  `registration` accepts sleep mode. Shape: `clear_cudagraphs` releases the serving cells, and the
  entry check of A second serving capture admits that one recapture; the cells capture over
  block 0 as at startup.
- **More geometries, encoder arms, endpoint policies.** Deferred items 4 to 6 above. Trigger: each
  item's exactness and latency evidence at the geometry or arm. Shape: new cells with their own
  warm-up domains and qualification rows.
- **Runner-captured graphs.** Cells replayed inside core's breakable graphs. Trigger and shape:
  the `registration` design, Future roadmap items 5 and 6.
- **An offline arm of `U_api`.** At `cbf13fed4a`, `LLM.generate` takes complete prompts only, with
  no streaming input, so an `LLM` arm can time only a whole one-unit request (admission, the unit,
  the park and the request's finish), not the feed into an open stream that TXG-QUAL-016 times;
  it would be a second measurement with a rule of its own, so it is not an arm of TXG-QUAL-016.
  Trigger: `LLM` accepts streaming input, or a deployment needs the offline single-request
  latency. Shape: a secondary arm with its own preregistered rule, reported beside `U_api` and not
  gating tier 5.
- **A measured startup allowance.** The exactness runs' headroom margin carries a stated 1 GiB
  for the memory in use at startup (TXG-QUAL-029). Trigger: the A40 round records the device
  memory in use before the engine starts and the torch memory limit core logs. Shape: the
  allowance becomes that round's measured figure plus a stated spread, preregistered in its place.
- **Multi-unit rows.** A row that carries several units, run in order as several stream units
  (the `registration` design, Future roadmap 14). Here: the cells, the step arena and the static
  inputs address units, not rows, gathered by `query_start_loc`, or such steps keep falling back
  with `multi-token`; and each unit gets its own per-unit checks and commit (TXG-TXN-010,
  TXG-EXEC-007).
