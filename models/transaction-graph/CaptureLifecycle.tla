\* CaptureLifecycle: the transaction-graph capture lifecycle on #67583's
\* model-state hooks, for one worker with transaction graphs on.
\*
\* Checked with TLC2 Version 2.19 of 08 August 2024 (tla2tools.jar 1.7.4,
\* SHA-1 bee4a54f3ee3d4afc347c3240ec2d9e93b075104), configuration
\* CaptureLifecycle.cfg: N = 3 (six cells per capture), MaxSteps = 2,
\* MaxMem = 2. Run ./check.sh with TLA2TOOLS_JAR naming that jar.
\*
\* Sources: docs/intent/transaction-graph/transaction-graph-design.md
\* (Capture lifecycle on the hook, Rules of the capture call, Graph memory,
\* Post-warmup verification, System model) and transaction-graph-specs.md
\* (TXG-LIFE-*, TXG-CAP-*, TXG-MEM-*, TXG-VER-*, TXG-MODEL-*);
\* registration-specs.md REG-MEM-015 to REG-MEM-027 for the segment-graph
\* part of each hook call. SplitStep.tla refines the scheduled step.
\*
\* WHAT IS MODELLED (TXG-MODEL-001, TXG-MODEL-002)
\*  Every hook call in core's startup order: construction (or a failed
\*  construction), needs_cudagraph_capture, prepare_execution_memory (after
\*  an optional startup plan that fixes the budget), the profiling forward
\*  pass (a dummy step), then core's estimate: skipped, its temporary
\*  profiling pool failing, or initialize_state_slots with profiling
\*  bindings followed by capture_cudagraphs(profile_only=True) (or a core
\*  failure before that call is reached) and clear_cudagraphs, also after a
\*  failed or skipped capture; initialize_state_slots with serving bindings;
\*  capture_cudagraphs(profile_only=False); post_warmup; serving with
\*  scheduled steps, dummy steps and cleanup-only steps; a scheduled step
\*  before post-warmup verification, after whose propagated error the
\*  engine dies or survives (the environment's choice, TXG-EXEC-015);
\*  all-failed bypass steps after the serving capture on an unlatched
\*  worker, before and after verification, behind the latch check
\*  (TXG-FAIL-010; here a latch fails the worker at once, so SplitStep.tla
\*  checks the latched refusal); a repeated initialize_state_slots with an
\*  equal or a changed layout; a later serving capture call after a
\*  successful serving capture (an elastic reconfiguration) and after a
\*  failed or refused one (ServingCaptureRetry), which core never makes but
\*  the model does, so that TXG-MODEL-009 holds by the entry refusal and not
\*  by core's order; shutdown from any state between hook calls; and a
\*  killed worker. Each capture call runs its entry checks, then the
\*  segment graphs, the binding buffers, the scatter warm-up, the 2N cells
\*  (chunk N..1 then echo N..1, each with its validation replay on rejected
\*  capture rows), then the margin (profiling) or the aggregate memory
\*  comparison (serving), then synchronizes and returns; every phase can
\*  fail, and a failure runs the whole-call cleanup (transaction-graph part,
\*  then segment graphs, then synchronize) before it propagates. Whether a
\*  profiling capture is required (effective budget fixed, possibly by a
\*  startup plan after construction; estimate setting; extensible KV cache,
\*  possibly disabled after construction) and whether one runs are
\*  independent nondeterministic choices. Call totals are abstract units in
\*  0..MaxMem chosen nondeterministically.
\*
\* ENTRY CHECKS (TXG-LIFE-006, TXG-LIFE-007, TXG-LIFE-008, REG-MEM-024,
\* TXG-MEM-009)
\*  A serving call tests, in TXG-LIFE-006's order: profiling cells exist
\*  (TXG-LIFE-007); a serving capture was attempted (TXG-LIFE-008);
\*  profiling segment graphs exist (REG-MEM-024, whose attempted clause
\*  TXG-LIFE-008 already decided); a required profiling capture did not run
\*  (TXG-MEM-009). A profiling call tests a serving capture attempted
\*  (TXG-LIFE-007), then profiling segment graphs exist (REG-MEM-024). A
\*  serving call is attempted (servAttempted) only once it has passed every
\*  one of its checks. The defects these checks detect (profiling state not
\*  released, a serving capture already attempted, a required profiling
\*  capture missing) are mutually exclusive for one call: at most one holds
\*  in any state outside a capture call (EntryChecksExclusive). The order
\*  therefore never decides whether a call is refused; it decides only
\*  which of the two checks that test the same defect names it (TXG-LIFE-007
\*  before REG-MEM-024 for profiling state, TXG-LIFE-008 before REG-MEM-024
\*  for an attempt). No reachable state holds two defects, so the model has
\*  no multi-defect precedence witness; one would need an action outside
\*  core's order and this model (a serving capture with the profiling
\*  capture not cleared).
\*
\* WHAT IS NOT MODELLED
\*  CUDA semantics, bytes, signatures, page contents, bitwise equivalence;
\*  the option off (TXG-LIFE-011); a profiling capture after a serving
\*  capture, which core's order never makes (its entry check is modelled);
\*  references the propagated error holds (TXG-LIFE-012's last clause);
\*  the step's execute order, reservations and latch interplay, which
\*  SplitStep.tla models. A capture call and a hook's two release halves are
\*  atomic with respect to other hook calls; only a kill interrupts them.
\*
\* INVARIANTS (CaptureLifecycle.cfg)
\*  TypeOK: every variable stays in its domain.
\*  NoReplayOutsideStep (TXG-MODEL-003, TXG-LIFE-013): every cell replay is
\*    either a scheduled step's replay after post-warmup verification passed,
\*    outside any capture call, or a capture's validation replay of rejected
\*    capture rows inside its own capture_cudagraphs call.
\*  NoStepReplayOverProfilingPools (TXG-MODEL-007): no scheduled step
\*    replays a cell captured over the profiling pools; the profiling
\*    capture's validation replays are not step replays.
\*  NothingProfilingAfterClear (TXG-MODEL-008, TXG-LIFE-005, REG-MEM-021):
\*    once clear_cudagraphs has returned, no object the profiling capture
\*    created (segment graph, cell, binding buffer or margin) is retained.
\*  AtMostOneServingCapture (TXG-MODEL-009, TXG-LIFE-008, REG-MEM-024): at
\*    most one serving capture call passes its entry checks and captures
\*    graphs in the worker; every later serving call, after a successful or
\*    a failed serving capture, is refused at its entry.
\*  AttemptedOnlyAfterEntryChecks (TXG-LIFE-007, TXG-LIFE-008,
\*    REG-MEM-024): the worker records a serving capture as attempted
\*    exactly when a serving call has passed every entry check; a refused
\*    call is not an attempt.
\*  EntryChecksExclusive (TXG-LIFE-006, TXG-LIFE-007, TXG-LIFE-008,
\*    REG-MEM-024, TXG-MEM-009): between hook calls at most one of the
\*    defects the entry checks detect holds (ENTRY CHECKS above).
\*  UnprofiledServingCapture (TXG-MODEL-010, TXG-MEM-007 to TXG-MEM-009):
\*    a serving capture with no completed profiling capture before it either
\*    succeeded with the uncharged WARNING because none was required, or
\*    startup failed; when one was required, the call captured nothing.
\*  OverrunFailsStartup (TXG-MODEL-011, TXG-MEM-005, REG-MEM-027): a serving
\*    capture whose call total exceeds the profiling call total never
\*    succeeds, and startup fails once its call has returned.
\*  NothingLeftByFailedCall (TXG-MODEL-015, TXG-LIFE-012, REG-MEM-017):
\*    after a capture call raised (at its entry or after it), no segment
\*    graph, cell, binding buffer or margin that call created remains.
\*  NothingAfterShutdown (TXG-MODEL-016, TXG-LIFE-010, REG-MEM-026): after
\*    shutdown nothing remains captured.
\*  FailureIsReported (TXG-MODEL-004, TXG-MEM-010): a failed worker or a
\*    failed startup always carries a reported error.
\*  CallReturnsSynchronized (TXG-CAP-010, TXG-LIFE-012): no capture call
\*    returns or raises with device work it started still pending.
\*  SegmentsOutliveCells (TXG-LIFE-014, REG-MEM-020): a call's cells,
\*    buffers or margin exist only while its segment graphs exist, so
\*    segment graphs are captured first and released last.
\*  NothingOutsideCaptures (TXG-LIFE-001, TXG-LIFE-002): no object exists
\*    unless the capture call that creates it has run; construction,
\*    prepare_execution_memory, the profiling forward pass and dummy steps
\*    create nothing.
\*  ServingInventoryComplete (TXG-CAP-011, TXG-LIFE-006, TXG-VER-001): a
\*    successful serving capture holds its segment graphs, its buffers and
\*    exactly the 2N serving cells, and only a profiling call has a margin.
\*  VerifiedBeforeServing (TXG-VER-002, TXG-EXEC-015): the worker serves
\*    only after post_warmup verification passed.
\*
\* PROPERTIES (CaptureLifecycle.cfg), under the fairness in Spec: WF on
\* startup progress (every hook call, capture-call phase and the second half
\* of a shutdown) and WF on the shutdown of a failed or latched worker, the
\* cooperative-teardown assumption (TXG-MODEL-004). A healthy serving worker
\* is never forced to shut down; no property is claimed for a killed worker
\* or a failed construction.
\*  StartupSettles (TXG-MODEL-004): every startup whose construction
\*    succeeded and that is neither shut down cooperatively nor killed
\*    before it serves eventually serves or fails with a reported error; no
\*    startup stalls.
\*  FailedWorkerShutsDown (TXG-MODEL-004, TXG-LIFE-010): every latched or
\*    failed worker reaches shutdown, unless it is killed.
\*  LatchIsStable (TXG-FAIL-003): the failed latch is never cleared.
\*  RecaptureRefusalChangesNothing (TXG-LIFE-008, TXG-MODEL-009,
\*    REG-MEM-024; an action property): refusing a later serving call is
\*    worker-fatal and leaves the failed latch and every captured object
\*    unchanged, and after the refusal only shutdown changes them.
---------------------------- MODULE CaptureLifecycle ----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    N,          \* largest entry of transaction_graph_populations
    MaxSteps,   \* bound on scheduled steps after the worker serves
    MaxMem      \* abstract units for the capture-call totals

ASSUME N \in Nat \ {0} /\ MaxSteps \in Nat /\ MaxMem \in Nat

-------------------------------------------------------------------------------
(* Cells, capture calls and captured objects                               *)

Kinds == {"chunk", "echo"}
CellKeys == Kinds \X (1..N)
NoKey == <<"none", 0>>

\* TXG-CAP-007: chunk cells from N down to 1, then echo cells from N down to 1.
CaptureKey(i) == IF i <= N THEN <<"chunk", N + 1 - i>> ELSE <<"echo", 2 * N + 1 - i>>

\* "prof": capture_cudagraphs(profile_only=True); "serv": the serving capture;
\* "serv2": a later serving capture call, after the serving capture succeeded
\* (over the live pools) or failed or was refused (a retry).
CallKinds == {"prof", "serv", "serv2"}
ServingCalls == {"serv", "serv2"}

Seg(c) == <<c, "seg", NoKey>>        \* the call's segment graphs (registration)
Buf(c) == <<c, "buf", NoKey>>        \* binding-owned buffers (arena, statics, outputs)
Margin(c) == <<c, "margin", NoKey>>  \* the charged margin (TXG-MEM-004)
Cell(c, k) == <<c, "cell", k>>       \* one cell, keyed (kind, count)

Objects ==
    {Seg(c) : c \in CallKinds} \cup {Buf(c) : c \in CallKinds}
    \cup {Margin(c) : c \in CallKinds} \cup {Cell(c, k) : c \in CallKinds, k \in CellKeys}

TxgParts == {"buf", "margin", "cell"}
OfCall(S, c) == {o \in S : o[1] = c}
TxgOf(S, c) == {o \in S : o[1] = c /\ o[2] \in TxgParts}
ServingCells == {Cell("serv", k) : k \in CellKeys}

\* The pools whose cells a scheduled step may look up (TXG-COV-002).
ReplayablePools == ServingCalls

\* A replay record: who replayed, over which call's pools, whether
\* verification had passed, whether inside a capture call, and whether the
\* replayed rows were rejected capture rows.
ReplayRecs ==
    [ctx : {"step", "validation"}, pool : CallKinds, verified : BOOLEAN,
     inCall : BOOLEAN, rejected : BOOLEAN]

NoCall == [kind |-> "none", pc |-> "none", i |-> 0]
CallPCs == {"none", "seg", "buf", "scatter", "cell", "margin", "compare", "sync",
            "cleanup-txg", "cleanup-seg", "cleanup-sync"}

\* Step shapes for the coarse scheduled step: K CHUNK rows, E other rows.
Shapes == {s \in (0..N + 1) \X (0..N + 1) : 1 <= s[1] + s[2] /\ s[1] + s[2] <= N + 1}
Needed(s) ==
    (IF s[1] > 0 THEN {<<"chunk", s[1]>>} ELSE {}) \cup (IF s[2] > 0 THEN {<<"echo", s[2]>>} ELSE {})

-------------------------------------------------------------------------------
(* Phases in core's startup order                                          *)

Phases == {
    "constructing", "construction-failed", "constructed", "asked", "prepared",
    "forwarded", "prof-bound", "clearing", "clearing-seg", "estimated",
    "serv-bound", "captured", "serving", "failed", "shutting-down",
    "shutdown", "killed"}
Terminal == {"construction-failed", "shutdown", "killed"}
PreServing == {"constructed", "asked", "prepared", "forwarded", "prof-bound",
               "clearing", "estimated", "serv-bound"}
\* Between hook calls a cooperative shutdown can start in any of these.
Shutdownable == Phases \ (Terminal \cup {"constructing", "clearing-seg", "shutting-down"})

Errors == {
    "none", "profiling-pool", "core-capture", "profiling-after-serving",
    "profiling-cells-exist", "recapture-over-live-pools",
    "profiling-segment-graphs-exist",
    "profiling-capture-required", "segment-capture", "buffer-allocation",
    "scatter-warmup", "cell-capture", "margin", "serving-overrun",
    "verification", "layout-changed", "execute-before-verification",
    "step-after-launch"}

VARIABLES
    phase,          \* lifecycle phase
    call,           \* the capture call in progress, or NoCall
    captured,       \* objects currently captured or allocated
    replays,        \* replay records seen so far
    bindings,       \* state-slot bindings recorded: "none", "prof", "serv"
    budgetFixed,    \* effective kv_cache_memory_bytes set
    extensible,     \* effective extensible KV cache on
    estimateOn,     \* VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS not 0
    profAttempted,  \* capture_cudagraphs(profile_only=True) was called
    profRan,        \* a profiling capture completed in the worker
    servAttempted,  \* a serving call passed its entry checks (TXG-LIFE-007)
    servCapturing,  \* serving calls that passed their entry checks
    servOk,         \* the serving capture returned successfully
    requiredAtServ, \* TXG-MEM-007 decision: "undecided", "yes", "no"
    profTotal,      \* profiling call total, margin charge included (host integer)
    servTotal,      \* serving call total
    servMeasured,   \* the aggregate comparison read servTotal
    uncharged,      \* the uncharged-memory WARNING was logged (TXG-MEM-008)
    failedCalls,    \* calls that raised after passing their entry checks
    refusedCalls,   \* calls refused at their entry
    cleared,        \* clear_cudagraphs has returned
    pendingFailure, \* the profiling estimate failed; clear runs, then it propagates
    verified,       \* post-warmup verification passed
    latched,        \* the binding's failed latch
    devicePending,  \* device work started by the current call not yet synchronized
    outcome,        \* startup outcome: "pending", "served", "failed"
    error,          \* the first reported error
    steps,          \* scheduled steps run while serving
    dummySeen,      \* a dummy or cleanup-only step ran
    bypassUnverified \* an all-failed bypass ran before verification passed

vars == <<phase, call, captured, replays, bindings, budgetFixed, extensible,
          estimateOn, profAttempted, profRan, servAttempted, servCapturing, servOk,
          requiredAtServ, profTotal, servTotal, servMeasured, uncharged,
          failedCalls, refusedCalls, cleared, pendingFailure, verified, latched,
          devicePending, outcome, error, steps, dummySeen, bypassUnverified>>

cfgVars == <<budgetFixed, extensible, estimateOn>>
memVars == <<profAttempted, profRan, servAttempted, servCapturing, servOk,
             requiredAtServ, profTotal, servTotal, servMeasured, uncharged>>
histVars == <<failedCalls, refusedCalls, cleared, pendingFailure>>

TypeOK ==
    /\ phase \in Phases
    /\ call.kind \in CallKinds \cup {"none"} /\ call.pc \in CallPCs /\ call.i \in 0..2 * N
    /\ captured \subseteq Objects
    /\ replays \subseteq ReplayRecs
    /\ bindings \in {"none", "prof", "serv"}
    /\ budgetFixed \in BOOLEAN /\ extensible \in BOOLEAN /\ estimateOn \in BOOLEAN
    /\ profAttempted \in BOOLEAN /\ profRan \in BOOLEAN /\ servAttempted \in BOOLEAN
    /\ servCapturing \in 0..2 /\ servOk \in BOOLEAN
    /\ requiredAtServ \in {"undecided", "yes", "no"}
    /\ profTotal \in 0..MaxMem /\ servTotal \in 0..MaxMem
    /\ servMeasured \in BOOLEAN /\ uncharged \in BOOLEAN
    /\ failedCalls \subseteq CallKinds /\ refusedCalls \subseteq CallKinds
    /\ cleared \in BOOLEAN /\ pendingFailure \in BOOLEAN
    /\ verified \in BOOLEAN /\ latched \in BOOLEAN /\ devicePending \in BOOLEAN
    /\ outcome \in {"pending", "served", "failed"}
    /\ error \in Errors
    /\ steps \in 0..MaxSteps
    /\ dummySeen \in BOOLEAN
    /\ bypassUnverified \in BOOLEAN

Init ==
    /\ phase = "constructing"
    /\ call = NoCall
    /\ captured = {}
    /\ replays = {}
    /\ bindings = "none"
    /\ budgetFixed \in BOOLEAN
    /\ extensible \in BOOLEAN
    /\ estimateOn \in BOOLEAN
    /\ profAttempted = FALSE /\ profRan = FALSE /\ servAttempted = FALSE
    /\ servCapturing = 0 /\ servOk = FALSE /\ requiredAtServ = "undecided"
    /\ profTotal = 0 /\ servTotal = 0 /\ servMeasured = FALSE /\ uncharged = FALSE
    /\ failedCalls = {} /\ refusedCalls = {} /\ cleared = FALSE /\ pendingFailure = FALSE
    /\ verified = FALSE /\ latched = FALSE /\ devicePending = FALSE
    /\ outcome = "pending" /\ error = "none" /\ steps = 0 /\ dummySeen = FALSE
    /\ bypassUnverified = FALSE

\* The required-capture predicate of REG-MEM-027 on the effective values.
Required == ~budgetFixed /\ estimateOn /\ ~extensible

\* The serving inventory a scheduled step classifies against (TXG-COV-002).
Inventory == {o \in captured : o[2] = "cell" /\ o[1] \in ReplayablePools}

FirstError(e) == IF error = "none" THEN e ELSE error
StartupFails == IF outcome = "pending" THEN "failed" ELSE outcome

-------------------------------------------------------------------------------
(* Construction and the memory determination                               *)

Construct ==
    /\ phase = "constructing"
    /\ phase' = "constructed"
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* No model state exists, so core calls no hook; no property is claimed.
ConstructionFails ==
    /\ phase = "constructing"
    /\ phase' = "construction-failed"
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* REG-MEM-019: configuration only, true with the option on, the same on
\* every call; nothing is allocated or captured.
NeedsCudagraphCapture ==
    /\ phase = "constructed"
    /\ phase' = "asked"
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* A persisted startup plan may fix the budget first (a budget fixed after
\* construction); then prepare_execution_memory, where the transaction-graph
\* part allocates nothing (TXG-LIFE-001).
PrepareExecutionMemory ==
    /\ phase = "asked"
    /\ \E plan \in BOOLEAN : budgetFixed' = (budgetFixed \/ plan)
    /\ phase' = "prepared"
    /\ UNCHANGED <<call, captured, replays, bindings, extensible, estimateOn,
                   memVars, histVars, verified, latched, devicePending, outcome,
                   error, steps, dummySeen,
                   bypassUnverified>>

\* The profiling forward pass is a dummy step (prepare_step with
\* is_profile=True): no execute, no cell captured or replayed (TXG-LIFE-002).
ProfilingForwardPass ==
    /\ phase = "prepared"
    /\ phase' = "forwarded"
    /\ dummySeen' = TRUE
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps,
                   bypassUnverified>>

\* Core's estimate does not run: no profiling capture, no clear_cudagraphs.
EstimateSkipped ==
    /\ phase = "forwarded"
    /\ phase' = "estimated"
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* Core's temporary profiling pool does not fit: neither capture hook runs.
ProfilingPoolFails ==
    /\ phase = "forwarded"
    /\ phase' = "failed"
    /\ error' = FirstError("profiling-pool")
    /\ outcome' = StartupFails
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, steps, dummySeen,
                   bypassUnverified>>

\* initialize_state_slots with the profiling bindings (TXG-LIFE-003).
InitializeStateSlotsProfiling ==
    /\ phase = "forwarded"
    /\ bindings' = "prof"
    /\ phase' = "prof-bound"
    /\ UNCHANGED <<call, captured, replays, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* Core fails before it reaches the model state's capture: core still calls
\* clear_cudagraphs, which must tolerate nothing captured (a skipped capture).
ProfilingCaptureNotReached ==
    /\ phase = "prof-bound"
    /\ phase' = "clearing"
    /\ pendingFailure' = TRUE
    /\ error' = FirstError("core-capture")
    /\ outcome' = StartupFails
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars,
                   failedCalls, refusedCalls, cleared, verified, latched,
                   devicePending, steps, dummySeen,
                   bypassUnverified>>

\* The engine disables the extensible KV cache after construction.
DisableExtensibleCache ==
    /\ phase \in PreServing /\ call.kind = "none" /\ extensible
    /\ extensible' = FALSE
    /\ UNCHANGED <<phase, call, captured, replays, bindings, budgetFixed, estimateOn,
                   memVars, histVars, verified, latched, devicePending, outcome,
                   error, steps, dummySeen,
                   bypassUnverified>>

-------------------------------------------------------------------------------
(* capture_cudagraphs: entry checks                                        *)

\* TXG-LIFE-007 and REG-MEM-024: a profiling capture after a serving capture
\* was attempted, whether or not it succeeded, raises at its entry, as does
\* one while profiling segment graphs exist (REG-MEM-024).
ProfEntryRefusal ==
    IF servAttempted THEN "profiling-after-serving"
    ELSE IF Seg("prof") \in captured THEN "profiling-segment-graphs-exist"
    ELSE "none"

CaptureCudagraphsProfiling ==
    /\ phase = "prof-bound" /\ call.kind = "none" /\ bindings = "prof"
    /\ profAttempted' = TRUE
    /\ IF ProfEntryRefusal = "none"
       THEN /\ call' = [kind |-> "prof", pc |-> "seg", i |-> 0]
            /\ UNCHANGED <<phase, refusedCalls, pendingFailure, error, outcome>>
       ELSE /\ phase' = "clearing"
            /\ refusedCalls' = refusedCalls \cup {"prof"}
            /\ pendingFailure' = TRUE
            /\ error' = FirstError(ProfEntryRefusal)
            /\ outcome' = StartupFails
            /\ UNCHANGED call
    /\ UNCHANGED <<captured, replays, bindings, cfgVars, profRan, servAttempted,
                   servCapturing, servOk, requiredAtServ, profTotal, servTotal,
                   servMeasured, uncharged, failedCalls, cleared, verified, latched,
                   devicePending, steps, dummySeen,
                   bypassUnverified>>

\* The serving capture's entry checks, in order (TXG-LIFE-006): profiling
\* cells exist (TXG-LIFE-007); a serving capture was attempted, whether or
\* not it succeeded (TXG-LIFE-008); profiling segment graphs exist
\* (REG-MEM-024); then the required-capture decision (TXG-MEM-007) and the
\* unexpected absence of a profiling capture (TXG-MEM-009). Each refusal
\* raises at the call's entry and captures nothing.
ServEntryRefusal ==
    IF \E k \in CellKeys : Cell("prof", k) \in captured THEN "profiling-cells-exist"
    ELSE IF servAttempted THEN "recapture-over-live-pools"
    ELSE IF Seg("prof") \in captured THEN "profiling-segment-graphs-exist"
    ELSE IF Required /\ ~profRan THEN "profiling-capture-required"
    ELSE "none"
OrderingRefusal ==
    ServEntryRefusal \in {"profiling-cells-exist", "recapture-over-live-pools",
                          "profiling-segment-graphs-exist"}

\* A serving call. It counts as attempted (TXG-LIFE-007) only once it has
\* passed every entry check. A refusal is worker-fatal and leaves the latch
\* and every captured object unchanged (TXG-LIFE-008).
ServingEntry(c) ==
    /\ call.kind = "none" /\ bindings = "serv"
    /\ servAttempted' = (servAttempted \/ ServEntryRefusal = "none")
    /\ requiredAtServ' =
           IF OrderingRefusal THEN requiredAtServ
           ELSE IF Required THEN "yes" ELSE "no"
    /\ IF ServEntryRefusal = "none"
       THEN /\ call' = [kind |-> c, pc |-> "seg", i |-> 0]
            /\ servCapturing' = servCapturing + 1
            /\ UNCHANGED <<phase, refusedCalls, error, outcome>>
       ELSE /\ phase' = "failed"
            /\ refusedCalls' = refusedCalls \cup {c}
            /\ error' = FirstError(ServEntryRefusal)
            /\ outcome' = StartupFails
            /\ UNCHANGED <<call, servCapturing>>
    /\ UNCHANGED <<captured, replays, bindings, cfgVars, profAttempted, profRan,
                   servOk, profTotal, servTotal, servMeasured, uncharged,
                   failedCalls, cleared, pendingFailure, verified, latched,
                   devicePending, steps, dummySeen,
                   bypassUnverified>>

\* capture_cudagraphs(profile_only=False) at startup.
CaptureCudagraphsServing == phase = "serv-bound" /\ ServingEntry("serv")

\* One later serving call is modelled; it is refused at its entry.
LaterServingCall ==
    /\ "serv2" \notin failedCalls \cup refusedCalls
    /\ ServingEntry("serv2")

\* A second serving capture after the serving capture succeeded (an elastic
\* expert-parallel reconfiguration rerunning the warmup, before or after
\* post_warmup, or on a worker that failed later): refused at its entry
\* (TXG-LIFE-008); the recovery is an engine restart, so the worker fails.
SecondServingCapture ==
    /\ servOk /\ phase \in {"captured", "serving", "failed"}
    /\ LaterServingCall

\* A retry after the serving capture failed or was refused, which core never
\* makes (a failed serving capture fails startup): a retry after an attempted
\* call is refused by TXG-LIFE-008, and one after a refused call by the check
\* that refused it, so TXG-MODEL-009 holds by refusal.
ServingCaptureRetry ==
    /\ ~servOk /\ "serv" \in failedCalls \cup refusedCalls /\ phase = "failed"
    /\ LaterServingCall

-------------------------------------------------------------------------------
(* capture_cudagraphs: the call body (both captures)                       *)

C == call.kind
InCall(pc) == call.kind /= "none" /\ call.pc = pc
GoTo(pc) == call' = [call EXCEPT !.pc = pc]

\* A failure anywhere after the entry checks: start the whole-call cleanup.
FailTo(e) == /\ call' = [call EXCEPT !.pc = "cleanup-txg", !.i = 0]
             /\ error' = FirstError(e)

ValidationReplay ==
    [ctx |-> "validation", pool |-> C, verified |-> verified, inCall |-> TRUE,
     rejected |-> TRUE]

\* Registration's segment graphs, before any cell (REG-MEM-020, REG-MEM-022).
SegmentGraphsCaptured ==
    /\ InCall("seg")
    /\ captured' = captured \cup {Seg(C)}
    /\ devicePending' = TRUE
    /\ GoTo("buf")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* A segment-graph capture fails after creating some of them (REG-MEM-017).
SegmentGraphCaptureFails ==
    /\ InCall("seg")
    /\ captured' = captured \cup {Seg(C)}
    /\ devicePending' = TRUE
    /\ FailTo("segment-capture")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, steps, dummySeen,
                   bypassUnverified>>

BindingBuffersAllocated ==
    /\ InCall("buf")
    /\ captured' = captured \cup {Buf(C)}
    /\ devicePending' = TRUE
    /\ GoTo("scatter")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, error, steps, dummySeen,
                   bypassUnverified>>

BindingBufferAllocationFails ==
    /\ InCall("buf")
    /\ captured' = captured \cup {Buf(C)}
    /\ devicePending' = TRUE
    /\ FailTo("buffer-allocation")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, steps, dummySeen,
                   bypassUnverified>>

\* TXG-CAP-006: one rejected scatter row per resident pool.
ScatterWarmup ==
    /\ InCall("scatter")
    /\ devicePending' = TRUE
    /\ call' = [call EXCEPT !.pc = "cell", !.i = 1]
    /\ UNCHANGED <<phase, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, error, steps, dummySeen,
                   bypassUnverified>>

ScatterWarmupFails ==
    /\ InCall("scatter")
    /\ devicePending' = TRUE
    /\ FailTo("scatter-warmup")
    /\ UNCHANGED <<phase, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, steps, dummySeen,
                   bypassUnverified>>

\* One cell (TXG-CAP-004): eager reference, three warm-ups, the capture and
\* one validation replay on rejected capture rows; published only if that
\* replay equals the reference.
CellCaptured ==
    /\ InCall("cell")
    /\ captured' = captured \cup {Cell(C, CaptureKey(call.i))}
    /\ replays' = replays \cup {ValidationReplay}
    /\ devicePending' = TRUE
    /\ call' = IF call.i < 2 * N
               THEN [call EXCEPT !.i = @ + 1]
               ELSE [call EXCEPT !.pc = IF C = "prof" THEN "margin" ELSE "compare", !.i = 0]
    /\ UNCHANGED <<phase, bindings, cfgVars, memVars, histVars, verified, latched,
                   outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* Any phase of one cell fails; at worst its graph exists and its validation
\* replay ran and mismatched.
CellCaptureFails ==
    /\ InCall("cell")
    /\ captured' = captured \cup {Cell(C, CaptureKey(call.i))}
    /\ replays' = replays \cup {ValidationReplay}
    /\ devicePending' = TRUE
    /\ FailTo("cell-capture")
    /\ UNCHANGED <<phase, bindings, cfgVars, memVars, histVars, verified, latched,
                   outcome, steps, dummySeen,
                   bypassUnverified>>

\* TXG-MEM-004: the profiling capture ends with the charged margin; its
\* call total (margin charge r included) is kept as a host integer.
MarginAllocated ==
    /\ InCall("margin")
    /\ captured' = captured \cup {Margin(C)}
    /\ \E t \in 0..MaxMem : profTotal' = t
    /\ devicePending' = TRUE
    /\ GoTo("sync")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, profAttempted, profRan,
                   servAttempted, servCapturing, servOk, requiredAtServ, servTotal,
                   servMeasured, uncharged, histVars, verified, latched, outcome,
                   error, steps, dummySeen,
                   bypassUnverified>>

MarginAllocationFails ==
    /\ InCall("margin")
    /\ devicePending' = TRUE
    /\ FailTo("margin")
    /\ UNCHANGED <<phase, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, outcome, steps, dummySeen,
                   bypassUnverified>>

\* TXG-MEM-005: when a profiling capture ran, the serving call total may not
\* exceed the profiling call total; an overrun is a failure of the call
\* (TXG-LIFE-012). With no profiling capture (none was required, or the
\* entry would have refused), the WARNING of TXG-MEM-008 is logged.
MemoryComparison ==
    /\ InCall("compare")
    /\ \E t \in 0..MaxMem :
         /\ servTotal' = t
         /\ servMeasured' = TRUE
         /\ IF profRan /\ t > profTotal
            THEN /\ FailTo("serving-overrun")
                 /\ UNCHANGED uncharged
            ELSE /\ GoTo("sync")
                 /\ uncharged' = ~profRan
                 /\ UNCHANGED error
    /\ UNCHANGED <<phase, captured, replays, bindings, cfgVars, profAttempted,
                   profRan, servAttempted, servCapturing, servOk, requiredAtServ,
                   profTotal, histVars, verified, latched, devicePending, outcome,
                   steps, dummySeen,
                   bypassUnverified>>

\* TXG-CAP-010: synchronize the capture stream, then return. The serving
\* capture records the plain-value pool layout for post_warmup.
CaptureCallReturns ==
    /\ InCall("sync")
    /\ devicePending' = FALSE
    /\ call' = NoCall
    /\ phase' = CASE C = "prof" -> "clearing"
                  [] C = "serv" -> "captured"
                  [] OTHER -> phase
    /\ profRan' = (profRan \/ C = "prof")
    /\ servOk' = (servOk \/ C = "serv")
    /\ UNCHANGED <<captured, replays, bindings, cfgVars, profAttempted,
                   servAttempted, servCapturing, requiredAtServ, profTotal,
                   servTotal, servMeasured, uncharged, histVars, verified, latched,
                   outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* TXG-LIFE-012, TXG-LIFE-014: discard the transaction-graph part this call
\* created (cells, buffers, margin) first ...
FailureCleanupTxg ==
    /\ InCall("cleanup-txg")
    /\ captured' = captured \ TxgOf(captured, C)
    /\ GoTo("cleanup-seg")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* ... then its segment graphs (REG-MEM-017) ...
FailureCleanupSegment ==
    /\ InCall("cleanup-seg")
    /\ captured' = captured \ {Seg(C)}
    /\ GoTo("cleanup-sync")
    /\ UNCHANGED <<phase, replays, bindings, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* ... then synchronize and propagate the original error. A failed profiling
\* capture still gets core's clear_cudagraphs before startup fails.
FailureCleanupRaises ==
    /\ InCall("cleanup-sync")
    /\ devicePending' = FALSE
    /\ call' = NoCall
    /\ failedCalls' = failedCalls \cup {C}
    /\ IF C = "prof"
       THEN /\ phase' = "clearing" /\ pendingFailure' = TRUE
       ELSE /\ phase' = "failed" /\ UNCHANGED pendingFailure
    /\ outcome' = StartupFails
    /\ UNCHANGED <<captured, replays, bindings, cfgVars, memVars, refusedCalls,
                   cleared, verified, latched, error, steps, dummySeen,
                   bypassUnverified>>

CallProgress ==
    \/ SegmentGraphsCaptured \/ SegmentGraphCaptureFails
    \/ BindingBuffersAllocated \/ BindingBufferAllocationFails
    \/ ScatterWarmup \/ ScatterWarmupFails
    \/ CellCaptured \/ CellCaptureFails
    \/ MarginAllocated \/ MarginAllocationFails
    \/ MemoryComparison \/ CaptureCallReturns
    \/ FailureCleanupTxg \/ FailureCleanupSegment \/ FailureCleanupRaises

-------------------------------------------------------------------------------
(* clear_cudagraphs, serving bindings and post_warmup                      *)

\* TXG-LIFE-005: release the profiling cells, buffers and margin, keeping the
\* profiling figures as host integers; tolerates nothing captured.
ClearCudagraphsTxg ==
    /\ phase = "clearing" /\ call.kind = "none"
    /\ captured' = captured \ TxgOf(captured, "prof")
    /\ phase' = "clearing-seg"
    /\ UNCHANGED <<call, replays, bindings, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* REG-MEM-021: then the profiling segment graphs. A pending estimate failure
\* propagates now and startup fails.
ClearCudagraphsSegment ==
    /\ phase = "clearing-seg"
    /\ captured' = captured \ {Seg("prof")}
    /\ cleared' = TRUE
    /\ phase' = IF pendingFailure THEN "failed" ELSE "estimated"
    /\ UNCHANGED <<call, replays, bindings, cfgVars, memVars, failedCalls,
                   refusedCalls, pendingFailure, verified, latched, devicePending,
                   outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* initialize_state_slots with the serving bindings (TXG-LIFE-003).
InitializeStateSlotsServing ==
    /\ phase = "estimated"
    /\ bindings' = "serv"
    /\ phase' = "serv-bound"
    /\ UNCHANGED <<call, captured, replays, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* TXG-VER-001: segment-graph check, then layout, projected roles, all 2N
\* cells holding their graphs, external buffers and process constants; no
\* replay, no allocation (TXG-VER-003).
PostWarmupPasses ==
    /\ phase = "captured" /\ ~latched
    /\ ServingCells \subseteq captured
    /\ verified' = TRUE
    /\ phase' = "serving"
    /\ outcome' = IF outcome = "pending" THEN "served" ELSE outcome
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   latched, devicePending, error, steps, dummySeen,
                   bypassUnverified>>

\* TXG-VER-002: a mismatch latches the binding and raises; startup fails.
PostWarmupMismatch ==
    /\ phase = "captured" /\ ~latched
    /\ latched' = TRUE
    /\ phase' = "failed"
    /\ error' = FirstError("verification")
    /\ outcome' = StartupFails
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, devicePending, steps, dummySeen,
                   bypassUnverified>>

\* TXG-LIFE-009: a repeated initialize_state_slots after the serving
\* capture; an equal layout changes nothing ...
ReinitializeStateSlotsSame ==
    /\ phase \in {"captured", "serving"} /\ call.kind = "none"
    /\ UNCHANGED vars

\* ... a different layout latches and raises (worker-fatal).
ReinitializeStateSlotsChanged ==
    /\ phase \in {"captured", "serving"} /\ call.kind = "none"
    /\ latched' = TRUE
    /\ phase' = "failed"
    /\ error' = FirstError("layout-changed")
    /\ outcome' = StartupFails
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, devicePending, steps, dummySeen,
                   bypassUnverified>>

-------------------------------------------------------------------------------
(* Steps (coarse; SplitStep.tla refines execute)                           *)

StepReplays(s) ==
    {[ctx |-> "step", pool |-> o[1], verified |-> verified, inCall |-> FALSE,
      rejected |-> FALSE] : o \in {x \in Inventory : x[3] \in Needed(s)}}

Covered(s) ==
    /\ s[1] <= N /\ s[2] <= N
    /\ \A k \in Needed(s) : \E o \in Inventory : o[3] = k

\* A scheduled step on a serving worker: covered steps replay their serving
\* cells (one kind, or split); others, and signature mismatches, are counted
\* fallbacks run as the option-off transaction; a failure from the first
\* launch on latches the binding (worker-fatal).
ScheduledStep ==
    /\ phase = "serving" /\ call.kind = "none" /\ ~latched /\ steps < MaxSteps
    /\ steps' = steps + 1
    /\ \E s \in Shapes, sigOk \in BOOLEAN, fatal \in BOOLEAN :
         IF Covered(s) /\ sigOk
         THEN /\ replays' = replays \cup StepReplays(s)
              /\ IF fatal
                 THEN /\ latched' = TRUE
                      /\ phase' = "failed"
                      /\ error' = FirstError("step-after-launch")
                 ELSE UNCHANGED <<latched, phase, error>>
         ELSE UNCHANGED <<replays, latched, phase, error>>
    /\ UNCHANGED <<call, captured, bindings, cfgVars, memVars, histVars, verified,
                   devicePending, outcome, dummySeen,
                   bypassUnverified>>

\* TXG-EXEC-015: execute before post-warmup verification raises, replays
\* nothing, runs no option-off transaction and does not latch. Whether the
\* engine survives the propagated error is the environment's choice: it dies
\* (startup fails) or survives (nothing changed, so that branch is a
\* stuttering step).
PrematureScheduledStep ==
    /\ phase \in {"serv-bound", "captured"} /\ call.kind = "none" /\ ~verified
    /\ \E dies \in BOOLEAN :
         IF dies
         THEN /\ phase' = "failed"
              /\ error' = FirstError("execute-before-verification")
              /\ outcome' = StartupFails
         ELSE UNCHANGED <<phase, error, outcome>>
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, steps, dummySeen,
                   bypassUnverified>>

\* TXG-FAIL-009, TXG-FAIL-010: a real step whose every row failed at
\* planning, after the serving capture. The step entry checks the latch
\* first; a latched worker raises TXG-FAIL-003's error (here a latch has
\* already failed the worker, so SplitStep.tla checks that branch).
\* Otherwise the step takes the bypass whether or not post-warmup
\* verification passed: no transaction, no replay, no reservation.
AllFailedStep ==
    /\ phase \in {"captured", "serving"} /\ call.kind = "none"
    /\ IF latched
       THEN UNCHANGED vars
       ELSE /\ bypassUnverified' = (bypassUnverified \/ ~verified)
            /\ UNCHANGED <<phase, call, captured, replays, bindings, cfgVars,
                           memVars, histVars, verified, latched, devicePending,
                           outcome, error, steps, dummySeen>>

\* TXG-EXEC-012: a dummy step (warmup or idle) never calls execute.
DummyStep ==
    /\ phase \in {"serv-bound", "captured", "serving"} /\ call.kind = "none"
    /\ dummySeen' = TRUE
    /\ UNCHANGED <<phase, call, captured, replays, bindings, cfgVars, memVars,
                   histVars, verified, latched, devicePending, outcome, error, steps,
                   bypassUnverified>>

\* REG-MEM-014: a cleanup-only step schedules no row and runs no transaction.
CleanupOnlyStep ==
    /\ phase = "serving" /\ call.kind = "none"
    /\ dummySeen' = TRUE
    /\ UNCHANGED <<phase, call, captured, replays, bindings, cfgVars, memVars,
                   histVars, verified, latched, devicePending, outcome, error, steps,
                   bypassUnverified>>

-------------------------------------------------------------------------------
(* shutdown and kill                                                       *)

\* TXG-LIFE-010, TXG-LIFE-014: the runner's shutdown releases every cell,
\* binding buffer and margin first, from any state between hook calls ...
ShutdownTxg ==
    /\ phase \in Shutdownable /\ call.kind = "none"
    /\ captured' = {o \in captured : o[2] \notin TxgParts}
    /\ phase' = "shutting-down"
    /\ UNCHANGED <<call, replays, bindings, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* ... then every segment graph (REG-MEM-026).
ShutdownSegment ==
    /\ phase = "shutting-down"
    /\ captured' = {}
    /\ phase' = "shutdown"
    /\ UNCHANGED <<call, replays, bindings, cfgVars, memVars, histVars, verified,
                   latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

\* A killed worker runs no further hook, not even shutdown; no property is
\* claimed for it.
Kill ==
    /\ phase \notin Terminal
    /\ phase' = "killed"
    /\ UNCHANGED <<call, captured, replays, bindings, cfgVars, memVars, histVars,
                   verified, latched, devicePending, outcome, error, steps, dummySeen,
                   bypassUnverified>>

Done == phase \in Terminal /\ UNCHANGED vars

-------------------------------------------------------------------------------

StartupProgress ==
    \/ Construct \/ ConstructionFails \/ NeedsCudagraphCapture
    \/ PrepareExecutionMemory \/ ProfilingForwardPass
    \/ EstimateSkipped \/ ProfilingPoolFails \/ InitializeStateSlotsProfiling
    \/ ProfilingCaptureNotReached \/ CaptureCudagraphsProfiling
    \/ CallProgress \/ ClearCudagraphsTxg \/ ClearCudagraphsSegment
    \/ InitializeStateSlotsServing \/ CaptureCudagraphsServing
    \/ PostWarmupPasses \/ PostWarmupMismatch \/ ShutdownSegment

\* Cooperative teardown: a failed or latched worker's runner shuts down.
FailedTeardown == (phase = "failed" \/ latched) /\ ShutdownTxg

Next ==
    \/ StartupProgress
    \/ DisableExtensibleCache
    \/ SecondServingCapture \/ ServingCaptureRetry
    \/ ReinitializeStateSlotsSame \/ ReinitializeStateSlotsChanged
    \/ ScheduledStep \/ PrematureScheduledStep \/ AllFailedStep
    \/ DummyStep \/ CleanupOnlyStep
    \/ ShutdownTxg \/ Kill \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(StartupProgress) /\ WF_vars(FailedTeardown)

-------------------------------------------------------------------------------
(* Invariants                                                              *)

NoReplayOutsideStep ==
    \A r \in replays :
        \/ r.ctx = "step" /\ r.verified /\ ~r.inCall
        \/ r.ctx = "validation" /\ r.inCall /\ r.rejected

NoStepReplayOverProfilingPools ==
    \A r \in replays : r.ctx = "step" => r.pool /= "prof"

NothingProfilingAfterClear == cleared => OfCall(captured, "prof") = {}

AtMostOneServingCapture == servCapturing <= 1

AttemptedOnlyAfterEntryChecks == servAttempted = (servCapturing >= 1)

\* The defects the entry checks detect (ENTRY CHECKS in the header).
ProfilingStateLeft == \E o \in captured : o[1] = "prof"
RequiredCaptureMissing == Required /\ ~profRan
EntryChecksExclusive ==
    call.kind = "none" =>
        Cardinality({d \in {"prof-left", "attempted", "required"} :
                       \/ d = "prof-left" /\ ProfilingStateLeft
                       \/ d = "attempted" /\ servAttempted
                       \/ d = "required" /\ RequiredCaptureMissing}) <= 1

UnprofiledServingCapture ==
    /\ (servOk /\ ~profRan) => (requiredAtServ = "no" /\ uncharged)
    /\ (requiredAtServ = "yes" /\ ~profRan) => (servCapturing = 0 /\ ~servOk)
    /\ (servAttempted /\ ~profRan /\ call.kind = "none")
           => (servOk \/ outcome = "failed")

OverrunFailsStartup ==
    /\ (servOk /\ profRan) => servTotal <= profTotal
    /\ (servMeasured /\ profRan /\ servTotal > profTotal)
           => (~servOk /\ (call.kind = "none" => outcome = "failed"))

NothingLeftByFailedCall ==
    \A c \in failedCalls \cup refusedCalls : OfCall(captured, c) = {}

NothingAfterShutdown == phase = "shutdown" => captured = {}

FailureIsReported == (phase = "failed" \/ outcome = "failed") => error /= "none"

CallReturnsSynchronized == call.kind = "none" => ~devicePending

SegmentsOutliveCells == \A c \in CallKinds : TxgOf(captured, c) /= {} => Seg(c) \in captured

NothingOutsideCaptures ==
    \A o \in captured :
        /\ o[1] = "prof" => profAttempted
        /\ o[1] \in ServingCalls => servCapturing > 0

ServingInventoryComplete ==
    /\ (servOk /\ phase \in {"captured", "serving"})
           => OfCall(captured, "serv") = ServingCells \cup {Seg("serv"), Buf("serv")}
    /\ \A c \in ServingCalls : Margin(c) \notin captured

VerifiedBeforeServing == phase = "serving" => verified

-------------------------------------------------------------------------------
(* Temporal properties                                                     *)

StartupSettles ==
    <>(outcome = "served"
       \/ (outcome = "failed" /\ error /= "none")
       \/ phase \in {"shutting-down", "shutdown", "killed", "construction-failed"})

FailedWorkerShutsDown == (phase = "failed" \/ latched) ~> (phase \in {"shutdown", "killed"})

LatchIsStable == [][latched => latched']_vars

RecaptureRefusalChangesNothing ==
    [][/\ ("serv2" \notin refusedCalls /\ "serv2" \in refusedCalls')
            => (phase' = "failed" /\ latched' = latched /\ captured' = captured)
       /\ ("serv2" \in refusedCalls /\ phase' \notin {"shutting-down", "shutdown"})
            => (latched' = latched /\ captured' = captured)]_vars

===============================================================================
