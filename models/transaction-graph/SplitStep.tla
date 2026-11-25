\* SplitStep: the scheduled steps of one worker after its serving capture,
\* with execute's order, the split step's two launches, the composite
\* commit reservation, the failed latch and the all-failed bypass.
\*
\* Checked with TLC2 Version 2.19 of 08 August 2024 (tla2tools.jar 1.7.4,
\* SHA-1 bee4a54f3ee3d4afc347c3240ec2d9e93b075104), configuration
\* SplitStep.cfg: N = 2, Pmax = 3 (so K > N and E > N are reachable),
\* MaxSteps = 4. Run ./check.sh with TLA2TOOLS_JAR naming that jar.
\*
\* Sources: docs/intent/transaction-graph/transaction-graph-design.md
\* (Execute order; Output order, decoder tier and reservation; Failure
\* containment; System model) and transaction-graph-specs.md (TXG-COV-*,
\* TXG-EXEC-*, TXG-FAIL-*, TXG-MODEL-*). CaptureLifecycle.tla models the
\* hooks that create, verify and release the cells.
\*
\* WHAT IS MODELLED (TXG-MODEL-002)
\*  The worker starts after a successful serving capture holding all 2N
\*  serving cells, before post_warmup, which passes or latches. Each step is
\*  a dummy step (as the model state's prepare_inputs classifies it,
\*  REG-MEM-001), a cleanup-only step (no row scheduled), or a real step
\*  that schedules rows with K CHUNK rows and E
\*  other rows. A real step enters the model state's step entry, which
\*  checks the failed latch first (TXG-FAIL-010): latched, it raises;
\*  otherwise an all-failed step (every row failed at planning) takes the
\*  bypass and runs no transaction (TXG-FAIL-009); otherwise execute runs,
\*  in order, the verification check, the process constants,
\*  classification, the prepare phase, the signature comparison, the graph
\*  objects, the arena upload with the static-input copies and the escaped
\*  row allocation, the composite reservation, the first launch (the chunk
\*  cell when K > 0, else the echo cell), the second launch of a split step
\*  (the echo cell), and the escaped-row copy with the commit stage and the
\*  counters. Each can fail as the design states: a counted fallback at
\*  classification or signature runs the option-off transaction; a
\*  pre-launch exception propagates without latching, cancelling any
\*  reservation held, except that a cancel of the held reservation that
\*  itself raises latches and is worker-fatal (TXG-EXEC-003); a changed
\*  process constant, a missing graph object, a failure at the first launch,
\*  between the launches or at the stage, and an exception after a
\*  successful stage (in the counters) are worker-fatal (latch, attempt to
\*  cancel, which can itself fail, then propagate). A stage that raises
\*  does so before or after it publishes the reservation to the sink's
\*  collection; either way it is not a successful stage and leaves the
\*  reservation cancellable, while a cancel after a successful stage is a
\*  no-op (TXG-EXEC-010). A declined step's option-off transaction makes its
\*  own reservation, then the scatters, then its stage, and takes one
\*  nondeterministic outcome per step: it completes, or its reservation,
\*  its scatter or its stage raises, the stage before or after it publishes
\*  the reservation. After a scatter or stage failure its failure path
\*  cancels the reservation and the exception propagates without latching,
\*  unless that cancel itself raises: the reservation then stays unresolved
\*  and the cancel's failure propagates as worker-fatal (TXG-TXN-008), so
\*  the worker fails, with its binding's latch unchanged (the option-off
\*  transaction holds no binding to latch), and runs no later step. A
\*  request may be aborted after the first launch.
\*  Between steps: core's binding of the state layer's views (bind_kv_cache)
\*  with a changed state-group layout latches (TXG-LIFE-009; an equal one
\*  changes nothing, see CaptureLifecycle.tla); a delayed device fault
\*  surfaces at the sink's collection,
\*  which neither latches nor cancels the staged reservation
\*  (TXG-FAIL-004); the engine dies, while serving or after a step raised
\*  before post_warmup (whether the engine survives a propagated error is
\*  the environment's choice, TXG-EXEC-015); the worker is shut down from
\*  any state between steps, or killed.
\*
\* WHAT IS NOT MODELLED
\*  CUDA semantics, row contents, page contents, signatures, exact fallback
\*  reasons beyond their class, counters other than the per-step fallback
\*  increments and histogram entries, the option-off transaction's resident
\*  pages after a failed scatter, each request's failure delivery through
\*  take_failed_requests (TXG-FAIL-008), the sink's own refusal of a second
\*  reservation (the model stops later steps by the latch or by the failed
\*  worker, never by that refusal), concurrency between steps (synchronous
\*  scheduling serializes them), and the capture lifecycle
\*  (CaptureLifecycle.tla).
\*
\* INVARIANTS (SplitStep.cfg). A check of one step's record applies to the
\* step in progress and to each finished step in the state where it finishes
\* (the variable last holds that record until the next step starts); the
\* log keeps a summary of every finished step for the checks across steps.
\*  TypeOK: every variable stays in its domain.
\*  ReservationDisposition (TXG-MODEL-012, TXG-EXEC-010, TXG-EXEC-003,
\*    TXG-FAIL-002, TXG-TXN-008): each commit reservation, a covered step's
\*    or a declined step's option-off one, is resolved at most once in
\*    total, by a successful stage or by a cancel (a stage that raised,
\*    before or after publication, is not a resolution, and a cancel after
\*    a successful stage is a no-op); on every nonfatal path it is resolved
\*    exactly once; a reservation whose cancel raised stays unresolved, and
\*    that failure is worker-fatal; after a worker-fatal failure it is
\*    unresolved only if its cancellation failed; and no later step that
\*    schedules a row then completes.
\*  FirstLaunchNeverFallsBack (TXG-MODEL-013, TXG-FAIL-002): a step whose
\*    first launch ran never falls back, never runs the option-off
\*    transaction and, after a failure, never stages (only a failure after
\*    a successful stage leaves it staged).
\*  LatchStopsRowSteps (TXG-MODEL-014, TXG-FAIL-003, TXG-FAIL-010): a step
\*    that schedules a row and enters on a latched worker, an all-failed
\*    step included, never completes, launches nothing and runs no option-off
\*    transaction; an all-failed bypass step replays no cell, increments no
\*    fallback counter, holds no reservation and adds exactly one histogram
\*    entry (TXG-COV-013). Cleanup-only steps are not constrained and may
\*    complete on a latched worker (REG-MEM-014).
\*  UnlatchedBypassCompletes (TXG-FAIL-009, TXG-FAIL-010, TXG-MODEL-014): an
\*    all-failed step on an unlatched worker takes the bypass and completes,
\*    whether or not post-warmup verification has passed; the verification
\*    check of TXG-EXEC-015 applies only to execute.
\*  StepReplayOnlyAfterVerification (TXG-MODEL-003, TXG-EXEC-015,
\*    TXG-LIFE-013): a step launches a cell only after post-warmup
\*    verification passed; dummy, cleanup-only and bypass steps and
\*    post_warmup itself launch nothing.
\*  FallbackCountedOnce (TXG-COV-003, TXG-EXEC-002, TXG-COV-011,
\*    TXG-EXEC-012): a declined step increments its fallback counter exactly
\*    once, holds no reservation from execute, launches no cell and runs the
\*    option-off transaction; dummy, cleanup-only and bypass steps increment
\*    none.
\*  PreLaunchFailureContained (TXG-EXEC-003, TXG-EXEC-015, TXG-TXN-008): a
\*    nonfatal failure, and a failure of a declined step's option-off
\*    transaction, launches nothing and sets no latch; it leaves no
\*    reservation unresolved, except an option-off scatter or stage failure
\*    whose cancel raised (the only such failure whose cancel can raise).
\*  WorkerFatalFailuresLatch (TXG-FAIL-002, TXG-EXEC-003, TXG-EXEC-013,
\*    TXG-CAP-016, TXG-TXN-008): a changed process constant, a missing graph
\*    object, a failed cancel of the held reservation before the first
\*    launch and every failure from the first launch on latch the binding,
\*    and nothing else does; a delayed device fault is worker-fatal without
\*    being attributed to a cell (TXG-FAIL-004); and a declined step's
\*    option-off failure whose cancel raised is worker-fatal without
\*    latching: the worker fails, its latch as it was at the step's entry.
\*  ExactCellsReplayed (TXG-COV-009, TXG-EXEC-001, TXG-CELL-002): a step
\*    launches only the chunk cell at exactly K and the echo cell at exactly
\*    E, the chunk cell first, never a cell for a count above N.
\*  AbortDoesNotCancel (TXG-FAIL-005): a request aborted after the first
\*    launch does not cancel the reservation, and the step stages it.
\*
\* PROPERTIES (SplitStep.cfg), under the fairness in Spec: WF on the
\* progress of the step in progress, WF on post_warmup, and WF on the
\* shutdown of a latched or failed worker (cooperative teardown,
\* TXG-MODEL-004). A healthy worker is never forced to shut down; no property
\* is claimed for a killed worker.
\*  FailedWorkerShutsDown (TXG-MODEL-004): every latched or failed worker
\*    reaches shutdown, unless it is killed.
\*  LatchIsStable (TXG-FAIL-003): the failed latch is never cleared.
\*  DelayedFaultContained (TXG-FAIL-004; an action property): a delayed
\*    device fault that surfaces at the sink's collection fails the worker
\*    and neither latches the binding nor cancels, or otherwise changes, the
\*    staged reservation of the step it follows.
------------------------------- MODULE SplitStep -------------------------------
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    N,          \* largest entry of transaction_graph_populations
    Pmax,       \* largest step population, min(max_num_seqs, max_num_batched_tokens)
    MaxSteps    \* bound on steps of every type

ASSUME N \in Nat \ {0} /\ Pmax \in Nat /\ Pmax > N /\ MaxSteps \in Nat \ {0}

Kinds == {"chunk", "echo"}
CellKeys == Kinds \X (1..N)

\* (K, E): CHUNK rows and other rows, 1 <= K + E <= Pmax. Planning-failed
\* rows count in E (TXG-COV-010).
Shapes == {s \in (0..Pmax) \X (0..Pmax) : 1 <= s[1] + s[2] /\ s[1] + s[2] <= Pmax}

Types == {"none", "rows", "cleanup", "dummy"}
Failures == {
    "none",
    "latched",                                  \* TXG-FAIL-003 refusal
    "unverified",                               \* TXG-EXEC-015
    "classify", "prepare", "signature",         \* TXG-EXEC-003, nonfatal
    "upload", "reserve", "reserve-held",        \* TXG-EXEC-003, nonfatal
    "prelaunch-cancel",                         \* TXG-EXEC-003, worker-fatal
    "constants", "graph",                       \* TXG-CAP-016, TXG-EXEC-013
    "async-fault",                              \* TXG-FAIL-004
    "launch1", "launch2", "stage",              \* TXG-FAIL-002
    "after-stage",                              \* TXG-EXEC-001 step 12
    "off-reserve", "off-scatter", "off-stage"}  \* TXG-TXN-008, TXG-EXEC-010
NonfatalFailures == {"unverified", "classify", "prepare", "signature", "upload",
                     "reserve", "reserve-held"}
OptionOffFailures == {"off-reserve", "off-scatter", "off-stage"}
LatchingFailures == {"prelaunch-cancel", "constants", "graph", "launch1",
                     "launch2", "stage", "after-stage"}
FatalFailures == LatchingFailures \cup {"async-fault"}

\* One step's record.
Recs == [
    type : Types,
    allFailed : BOOLEAN,       \* every row failed at planning
    latchedAtEntry : BOOLEAN,
    verifiedAtEntry : BOOLEAN,
    bypass : BOOLEAN,          \* took the all-failed bypass
    launches : 0..2,           \* cell launches started
    fallbacks : 0..2,          \* fallback-counter increments
    optionOff : BOOLEAN,       \* ran the option-off transaction
    made : BOOLEAN,            \* a commit reservation was made
    staged : 0..2,             \* successful stages (the stage returned)
    published : BOOLEAN,       \* a stage published it to the sink's collection
    cancelled : 0..2,          \* cancels that released it
    cancelFailed : BOOLEAN,    \* a cancellation attempt failed
    failure : Failures,
    latches : BOOLEAN,         \* this step set the failed latch
    aborted : BOOLEAN,         \* a request was aborted after the first launch
    histogram : 0..2,          \* (K, E) or (0, P) histogram entries added
    outcome : {"running", "completed", "raised"}]

NewRec(t, af, l, v) == [
    type |-> t, allFailed |-> af, latchedAtEntry |-> l, verifiedAtEntry |-> v,
    bypass |-> FALSE, launches |-> 0, fallbacks |-> 0, optionOff |-> FALSE,
    made |-> FALSE, staged |-> 0, published |-> FALSE, cancelled |-> 0,
    cancelFailed |-> FALSE,
    failure |-> "none", latches |-> FALSE, aborted |-> FALSE, histogram |-> 0,
    outcome |-> "running"]
IdleRec == NewRec("none", FALSE, FALSE, FALSE)

\* A step's failure is worker-fatal: a fatal failure, or any failure whose
\* cancel raised (TXG-EXEC-010), a declined step's option-off one included
\* (TXG-TXN-008).
Fatal(r) == r.failure \in FatalFailures \/ r.cancelFailed

\* What later steps and checks need of a finished step.
Sums == [type : Types, fatal : BOOLEAN, replayed : BOOLEAN,
         outcome : {"running", "completed", "raised"}]
Summary(r) == [type |-> r.type, fatal |-> Fatal(r),
               replayed |-> r.launches > 0, outcome |-> r.outcome]

PCs == {"idle", "verify", "constants", "classify", "prepare", "signature",
        "graph", "upload", "reserve", "launch1", "launch2", "stage", "cancel",
        "option-off"}

Phases == {"captured", "serving", "failed", "shutdown", "killed"}

VARIABLES
    phase,      \* "captured" (before post_warmup), "serving", "failed", ...
    verified,   \* post-warmup verification passed
    latched,    \* the binding's failed latch
    error,      \* the first reported worker error
    pc,         \* execute's program counter for the step in progress
    shape,      \* (K, E) of the step in progress
    launched,   \* cells whose launch the step in progress started
    rec,        \* the record of the step in progress
    log,        \* the summaries of finished steps
    last,       \* the record of the last finished step, until the next starts
    fault       \* where a delayed device fault surfaced: "none", "upload", "collection"

vars == <<phase, verified, latched, error, pc, shape, launched, rec, log, last, fault>>
stepVars == <<pc, shape, launched, rec, log, last>>

TypeOK ==
    /\ phase \in Phases
    /\ verified \in BOOLEAN /\ latched \in BOOLEAN
    /\ error \in {"none", "verification", "layout-changed", "async-fault",
                  "engine-dead", "step", "cancel-failed"}
    /\ pc \in PCs
    /\ shape \in Shapes \cup {<<0, 0>>}
    /\ launched \subseteq CellKeys
    /\ rec \in Recs
    /\ log \in Seq(Sums) /\ Len(log) <= MaxSteps
    /\ last \in Recs
    /\ fault \in {"none", "upload", "collection"}

Init ==
    /\ phase = "captured"
    /\ verified = FALSE
    /\ latched = FALSE
    /\ error = "none"
    /\ pc = "idle"
    /\ shape = <<0, 0>>
    /\ launched = {}
    /\ rec = IdleRec
    /\ log = <<>>
    /\ last = IdleRec
    /\ fault = "none"

FirstError(e) == IF error = "none" THEN e ELSE error

-------------------------------------------------------------------------------
(* Starting and finishing a step                                           *)

CanStart == phase \in {"captured", "serving"} /\ pc = "idle" /\ Len(log) < MaxSteps

Finish(r) ==
    /\ log' = Append(log, Summary(r))
    /\ last' = r
    /\ pc' = "idle"
    /\ shape' = <<0, 0>>
    /\ launched' = {}
    /\ rec' = IdleRec

Entry(t, af) == NewRec(t, af, latched, verified)

\* The worker state a step leaves alone, unless it sets the latch.
Keep == UNCHANGED <<phase, verified, latched, error, fault>>
KeepButLatch == UNCHANGED <<phase, verified, error, fault>>

\* TXG-EXEC-012: a dummy step, as the model state's prepare_inputs classifies
\* it (REG-MEM-001), never calls execute and replays nothing.
DummyStep ==
    /\ CanStart
    /\ Finish([Entry("dummy", FALSE) EXCEPT !.outcome = "completed"])
    /\ Keep

\* REG-MEM-014: a cleanup-only step schedules no row and completes, latched
\* or not.
CleanupOnlyStep ==
    /\ CanStart
    /\ Finish([Entry("cleanup", FALSE) EXCEPT !.outcome = "completed"])
    /\ Keep

\* TXG-FAIL-010: the step entry checks the latch before the bypass, so a
\* latched worker raises TXG-FAIL-003's error for every step that schedules
\* a row, an all-failed step included.
LatchedStepRaises ==
    /\ CanStart /\ latched
    /\ \E af \in BOOLEAN :
         Finish([Entry("rows", af) EXCEPT !.failure = "latched", !.outcome = "raised"])
    /\ Keep

\* TXG-FAIL-009, TXG-FAIL-010, TXG-COV-013: on an unlatched worker the
\* all-failed bypass runs no transaction, whether or not post-warmup
\* verification has passed (before post_warmup only a step that arrives too
\* early produces it), returns the rejected-row output and adds one (0, P)
\* histogram entry; postprocess_state fails its requests through
\* take_failed_requests (TXG-FAIL-008, not modelled).
AllFailedBypass ==
    /\ CanStart /\ ~latched
    /\ Finish([Entry("rows", TRUE) EXCEPT !.bypass = TRUE, !.histogram = 1,
                                          !.outcome = "completed"])
    /\ Keep

\* A step with at least one row that did not fail at planning calls execute.
BeginExecute ==
    /\ CanStart /\ ~latched
    /\ \E s \in Shapes :
         /\ shape' = s
         /\ rec' = Entry("rows", FALSE)
         /\ pc' = "verify"
    /\ last' = IdleRec
    /\ UNCHANGED <<launched, log>>
    /\ Keep

StartStep ==
    \/ DummyStep \/ CleanupOnlyStep \/ LatchedStepRaises \/ AllFailedBypass
    \/ BeginExecute

-------------------------------------------------------------------------------
(* execute (Execute order)                                                 *)

K == shape[1]
E == shape[2]
IsSplit == K > 0 /\ E > 0
FirstCell == IF K > 0 THEN <<"chunk", K>> ELSE <<"echo", E>>
EchoCell == <<"echo", E>>

Go(p) == /\ pc' = p /\ UNCHANGED <<shape, launched, rec, log, last>> /\ Keep

\* A nonfatal pre-launch exception: propagated, not latched.
Raise(f) ==
    /\ Finish([rec EXCEPT !.failure = f, !.outcome = "raised"])
    /\ Keep

\* A worker-fatal failure before any reservation exists: latch and raise.
LatchAndRaise(f) ==
    /\ latched' = TRUE
    /\ Finish([rec EXCEPT !.failure = f, !.latches = TRUE, !.outcome = "raised"])
    /\ KeepButLatch

\* Step 2 (TXG-EXEC-015): before verification passed, raise naming the
\* lifecycle defect; no latch, no replay, no option-off transaction.
VerificationCheckFails == pc = "verify" /\ ~verified /\ Raise("unverified")
VerificationCheckPasses == pc = "verify" /\ verified /\ Go("constants")

\* Step 3 (TXG-CAP-016): a changed process constant is worker-fatal.
ProcessConstantsChanged == pc = "constants" /\ LatchAndRaise("constants")
ProcessConstantsEqual == pc = "constants" /\ Go("classify")

\* Step 4 (TXG-COV-003, TXG-COV-004): K > N or E > N is a counted fallback.
ClassifyFallback ==
    /\ pc = "classify" /\ (K > N \/ E > N)
    /\ pc' = "option-off"
    /\ rec' = [rec EXCEPT !.fallbacks = @ + 1, !.histogram = @ + 1]
    /\ UNCHANGED <<shape, launched, log, last>>
    /\ Keep
ClassifyRaises == pc = "classify" /\ Raise("classify")
ClassifyCovered == pc = "classify" /\ K <= N /\ E <= N /\ Go("prepare")

\* Step 5: the prepare phase over the whole row plan.
PrepareRaises == pc = "prepare" /\ Raise("prepare")
PrepareDone == pc = "prepare" /\ Go("signature")

\* Step 6 (TXG-COV-005, TXG-COV-012): a signature mismatch is a counted
\* fallback.
SignatureFallback ==
    /\ pc = "signature"
    /\ pc' = "option-off"
    /\ rec' = [rec EXCEPT !.fallbacks = @ + 1, !.histogram = @ + 1]
    /\ UNCHANGED <<shape, launched, log, last>>
    /\ Keep
SignatureRaises == pc = "signature" /\ Raise("signature")
SignatureMatches == pc = "signature" /\ Go("graph")

\* Step 7 (TXG-EXEC-013): a needed cell's graph object is gone: worker-fatal.
GraphObjectMissing == pc = "graph" /\ LatchAndRaise("graph")
GraphObjectsHeld == pc = "graph" /\ Go("upload")

\* Steps 8 and 9: arena pack and upload, static-input copies, escaped row
\* tensor. A delayed device fault of an earlier replayed step can surface
\* here; it is worker-fatal, not a pre-launch failure (TXG-FAIL-004).
UploadRaises == pc = "upload" /\ Raise("upload")
DelayedFaultAtUpload ==
    /\ pc = "upload"
    /\ \E i \in 1..Len(log) : log[i].replayed /\ log[i].outcome = "completed"
    /\ phase' = "failed"
    /\ error' = FirstError("async-fault")
    /\ fault' = "upload"
    /\ Finish([rec EXCEPT !.failure = "async-fault", !.outcome = "raised"])
    /\ UNCHANGED <<verified, latched>>
UploadDone == pc = "upload" /\ Go("reserve")

\* Step 10 (TXG-EXEC-010, TXG-EXEC-003): one composite reservation for all P
\* rows. It can raise before taking the ticket, or after taking it (a
\* reservation without a callable stage), when the held ticket is
\* cancelled; a cancel that itself raises latches and is worker-fatal.
ReserveRaises == pc = "reserve" /\ Raise("reserve")
ReserveRaisesHolding ==
    /\ pc = "reserve"
    /\ Finish([rec EXCEPT !.made = TRUE, !.cancelled = @ + 1,
                          !.failure = "reserve-held", !.outcome = "raised"])
    /\ Keep
PreLaunchCancelFails ==
    /\ pc = "reserve"
    /\ latched' = TRUE
    /\ Finish([rec EXCEPT !.made = TRUE, !.cancelFailed = TRUE, !.latches = TRUE,
                          !.failure = "prelaunch-cancel", !.outcome = "raised"])
    /\ KeepButLatch
Reserved ==
    /\ pc = "reserve"
    /\ pc' = "launch1"
    /\ rec' = [rec EXCEPT !.made = TRUE]
    /\ UNCHANGED <<shape, launched, log, last>>
    /\ Keep

\* Step 11: the chunk cell's launch when K > 0, else the echo cell's. A
\* request may be aborted once the first launch ran (TXG-FAIL-005).
FirstLaunch ==
    /\ pc = "launch1"
    /\ launched' = {FirstCell}
    /\ \E ab \in BOOLEAN : rec' = [rec EXCEPT !.launches = 1, !.aborted = ab]
    /\ pc' = IF IsSplit THEN "launch2" ELSE "stage"
    /\ UNCHANGED <<shape, log, last>>
    /\ Keep

\* TXG-FAIL-002: from the first launch on, a failure latches the binding,
\* then the step attempts to cancel and propagates; no launch is retried and
\* no kind falls back.
FirstLaunchFails ==
    /\ pc = "launch1"
    /\ launched' = {FirstCell}
    /\ latched' = TRUE
    /\ rec' = [rec EXCEPT !.launches = 1, !.failure = "launch1", !.latches = TRUE]
    /\ pc' = "cancel"
    /\ UNCHANGED <<shape, log, last>>
    /\ KeepButLatch

\* The split step's second launch: the echo cell.
SplitSecondLaunch ==
    /\ pc = "launch2"
    /\ launched' = launched \cup {EchoCell}
    /\ rec' = [rec EXCEPT !.launches = 2]
    /\ pc' = "stage"
    /\ UNCHANGED <<shape, log, last>>
    /\ Keep

\* A failure between the launches, or at the echo launch itself.
SplitSecondLaunchFails ==
    /\ pc = "launch2"
    /\ \E atLaunch \in BOOLEAN :
         /\ launched' = IF atLaunch THEN launched \cup {EchoCell} ELSE launched
         /\ rec' = [rec EXCEPT !.launches = IF atLaunch THEN 2 ELSE 1,
                               !.failure = "launch2", !.latches = TRUE]
    /\ latched' = TRUE
    /\ pc' = "cancel"
    /\ UNCHANGED <<shape, log, last>>
    /\ KeepButLatch

\* Step 12: the escaped-row copy, the commit stage (once, all P rows) and
\* the counters, including the (K, E) histogram entry. The stage publishes
\* the reservation to the sink's collection and, once it returns, has
\* resolved it.
StageCommits ==
    /\ pc = "stage"
    /\ Finish([rec EXCEPT !.staged = @ + 1, !.published = TRUE,
                          !.histogram = @ + 1, !.outcome = "completed"])
    /\ Keep

\* The escaped-row copy or the stage raises, the stage before or after it
\* published the reservation. Not a successful stage: the reservation stays
\* cancellable (TXG-EXEC-010).
StageRaises ==
    /\ pc = "stage"
    /\ latched' = TRUE
    /\ \E pub \in BOOLEAN :
         rec' = [rec EXCEPT !.published = pub, !.failure = "stage", !.latches = TRUE]
    /\ pc' = "cancel"
    /\ UNCHANGED <<shape, launched, log, last>>
    /\ KeepButLatch

\* The stage succeeded and the counters after it raise: worker-fatal; the
\* failure path's cancel then finds the reservation resolved.
AfterStageFails ==
    /\ pc = "stage"
    /\ latched' = TRUE
    /\ rec' = [rec EXCEPT !.staged = @ + 1, !.published = TRUE,
                          !.failure = "after-stage", !.latches = TRUE]
    /\ pc' = "cancel"
    /\ UNCHANGED <<shape, launched, log, last>>
    /\ KeepButLatch

\* The cancellation attempt after a worker-fatal failure. After a successful
\* stage it is a no-op; otherwise it releases the reservation, published or
\* not, or itself fails; the worker is not reused, so the outcome is
\* irrelevant to the engine.
CancelAfterFatal ==
    /\ pc = "cancel"
    /\ IF rec.staged > 0
       THEN Finish([rec EXCEPT !.outcome = "raised"])
       ELSE \E ok \in BOOLEAN :
              Finish([rec EXCEPT !.cancelled = IF ok THEN @ + 1 ELSE @,
                                 !.cancelFailed = ~ok, !.outcome = "raised"])
    /\ Keep

\* TXG-COV-003, TXG-TXN-008: the declined step runs as the option-off
\* transaction (the caller adds no fallback increment): its reservation,
\* the scatters, then its stage, with one nondeterministic outcome
\* <<failure, published, cancel raises>> per step. It completes, staging
\* its reservation once; its reservation raises, holding nothing; or its
\* scatter or its stage raises, the stage before or after it publishes the
\* reservation (not a successful stage either way, TXG-EXEC-010), and its
\* failure path cancels the reservation. The exception propagates without
\* latching, unless that cancel raises (OptionOffCancelRaises).
OptionOffOutcomes ==
    {<<"none", TRUE, FALSE>>, <<"off-reserve", FALSE, FALSE>>}
    \cup ({"off-scatter"} \X {FALSE} \X BOOLEAN)
    \cup ({"off-stage"} \X BOOLEAN \X BOOLEAN)

\* TXG-TXN-008, TXG-EXEC-010: the failure path's cancel itself raises. The
\* reservation stays unresolved and the cancel's failure, with the original
\* failure as its context, propagates as worker-fatal: the worker fails and
\* runs no later step. The option-off transaction holds no binding, so the
\* latch is left as it was.
OptionOffCancelRaises ==
    /\ phase' = "failed"
    /\ error' = FirstError("cancel-failed")
    /\ UNCHANGED <<verified, latched, fault>>

OptionOffTransaction ==
    /\ pc = "option-off"
    /\ \E o \in OptionOffOutcomes :
         LET f == o[1]
             cancelRaises == o[3]
         IN /\ Finish(CASE f = "none" ->
                             [rec EXCEPT !.optionOff = TRUE, !.made = TRUE, !.staged = @ + 1,
                                         !.published = TRUE, !.outcome = "completed"]
                        [] f = "off-reserve" ->
                             [rec EXCEPT !.optionOff = TRUE, !.failure = f, !.outcome = "raised"]
                        [] OTHER ->
                             [rec EXCEPT !.optionOff = TRUE, !.made = TRUE, !.published = o[2],
                                         !.cancelled = IF cancelRaises THEN @ ELSE @ + 1,
                                         !.cancelFailed = cancelRaises,
                                         !.failure = f, !.outcome = "raised"])
            /\ IF cancelRaises THEN OptionOffCancelRaises ELSE Keep

StepProgress ==
    \/ VerificationCheckFails \/ VerificationCheckPasses
    \/ ProcessConstantsChanged \/ ProcessConstantsEqual
    \/ ClassifyFallback \/ ClassifyRaises \/ ClassifyCovered
    \/ PrepareRaises \/ PrepareDone
    \/ SignatureFallback \/ SignatureRaises \/ SignatureMatches
    \/ GraphObjectMissing \/ GraphObjectsHeld
    \/ UploadRaises \/ DelayedFaultAtUpload \/ UploadDone
    \/ ReserveRaises \/ ReserveRaisesHolding \/ PreLaunchCancelFails \/ Reserved
    \/ FirstLaunch \/ FirstLaunchFails
    \/ SplitSecondLaunch \/ SplitSecondLaunchFails
    \/ StageCommits \/ StageRaises \/ AfterStageFails \/ CancelAfterFatal
    \/ OptionOffTransaction

-------------------------------------------------------------------------------
(* Lifecycle events between steps                                          *)

Between == pc = "idle"

\* TXG-VER-001, TXG-VER-002: post_warmup passes, or latches and fails startup.
PostWarmupPasses ==
    /\ phase = "captured" /\ Between /\ ~latched
    /\ verified' = TRUE
    /\ phase' = "serving"
    /\ UNCHANGED <<latched, error, fault, stepVars>>
PostWarmupMismatch ==
    /\ phase = "captured" /\ Between /\ ~latched
    /\ latched' = TRUE
    /\ phase' = "failed"
    /\ error' = FirstError("verification")
    /\ UNCHANGED <<verified, fault, stepVars>>

\* TXG-LIFE-009: core binds the state layer's views (bind_kv_cache) after the
\* serving capture with a state-group layout whose plain values differ from
\* the recorded one: worker-fatal; later steps see the latch until the
\* worker is torn down.
BindKvCacheChangedLayout ==
    /\ phase \in {"captured", "serving"} /\ Between
    /\ latched' = TRUE
    /\ error' = FirstError("layout-changed")
    /\ UNCHANGED <<phase, verified, fault, stepVars>>

\* TXG-FAIL-004: a delayed device fault of the last replayed step surfaces at
\* the sink's collection, after its successful stage: worker-fatal, not
\* retried or attributed; it neither latches the binding nor cancels the
\* staged reservation.
DelayedFaultAtCollection ==
    /\ phase = "serving" /\ Between
    /\ last.launches > 0 /\ last.outcome = "completed"
    /\ phase' = "failed"
    /\ error' = FirstError("async-fault")
    /\ fault' = "collection"
    /\ UNCHANGED <<verified, latched, stepVars>>

\* The engine dies, after a propagated step exception or of another cause:
\* while serving, or before post_warmup once a step has raised (whether the
\* worker survives a premature step's error is the environment's choice,
\* TXG-EXEC-015).
LastStepRaised == Len(log) > 0 /\ log[Len(log)].outcome = "raised"
EngineDies ==
    /\ Between
    /\ \/ phase = "serving"
       \/ phase = "captured" /\ LastStepRaised
    /\ phase' = "failed"
    /\ error' = FirstError(IF latched THEN "step" ELSE "engine-dead")
    /\ UNCHANGED <<verified, latched, fault, stepVars>>

\* The runner's shutdown, from any state between steps (TXG-LIFE-010).
Shutdown ==
    /\ phase \in {"captured", "serving", "failed"} /\ Between
    /\ phase' = "shutdown"
    /\ UNCHANGED <<verified, latched, error, fault, stepVars>>

\* A killed worker runs nothing more; no property is claimed for it.
Kill ==
    /\ phase \notin {"shutdown", "killed"}
    /\ phase' = "killed"
    /\ UNCHANGED <<verified, latched, error, fault, stepVars>>

Done == phase \in {"shutdown", "killed"} /\ UNCHANGED vars

PostWarmup == PostWarmupPasses \/ PostWarmupMismatch

\* Cooperative teardown: a latched or failed worker's runner shuts down.
FailedTeardown == (latched \/ phase = "failed") /\ Shutdown

Next ==
    \/ StartStep \/ StepProgress \/ PostWarmup
    \/ BindKvCacheChangedLayout \/ DelayedFaultAtCollection \/ EngineDies
    \/ Shutdown \/ Kill \/ Done

Spec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(StepProgress) /\ WF_vars(PostWarmup) /\ WF_vars(FailedTeardown)

-------------------------------------------------------------------------------
(* Invariants                                                              *)

Finished == {last}
AllRecs == Finished \cup (IF pc = "idle" THEN {} ELSE {rec})
Resolutions(r) == r.staged + r.cancelled

ReservationDisposition ==
    /\ \A r \in AllRecs : Resolutions(r) <= 1 /\ (Resolutions(r) > 0 => r.made)
    /\ \A r \in Finished :
         /\ r.made =>
               /\ ~Fatal(r) => Resolutions(r) = 1
               /\ (Fatal(r) /\ Resolutions(r) = 0) => r.cancelFailed
         /\ r.cancelFailed => (r.made /\ Resolutions(r) = 0)
    /\ \A i, j \in 1..Len(log) :
         (i < j /\ log[i].fatal /\ log[j].type = "rows")
            => log[j].outcome /= "completed"

FirstLaunchNeverFallsBack ==
    \A r \in AllRecs :
        r.launches >= 1 =>
            /\ r.fallbacks = 0
            /\ ~r.optionOff
            /\ r.failure \notin {"none", "after-stage"} => r.staged = 0

LatchStopsRowSteps ==
    /\ \A r \in AllRecs :
         (r.type = "rows" /\ r.latchedAtEntry)
            => /\ r.outcome /= "completed"
               /\ r.launches = 0
               /\ ~r.optionOff
    /\ \A r \in AllRecs :
         r.bypass =>
            /\ r.allFailed /\ ~r.latchedAtEntry
            /\ r.launches = 0 /\ r.fallbacks = 0 /\ ~r.made /\ ~r.optionOff
            /\ r.histogram = 1

UnlatchedBypassCompletes ==
    \A r \in Finished :
        (r.type = "rows" /\ r.allFailed /\ ~r.latchedAtEntry)
            => (r.bypass /\ r.outcome = "completed")

StepReplayOnlyAfterVerification ==
    /\ \A r \in AllRecs : r.launches >= 1 => (r.type = "rows" /\ r.verifiedAtEntry)
    /\ launched /= {} => verified

FallbackCountedOnce ==
    \A r \in AllRecs :
        /\ r.fallbacks <= 1
        /\ (r.type /= "rows" \/ r.bypass) => r.fallbacks = 0
        /\ r.optionOff => r.fallbacks = 1
        /\ (r.fallbacks = 1 /\ r.outcome = "completed")
               => (r.optionOff /\ r.launches = 0 /\ r.staged = 1)

PreLaunchFailureContained ==
    \A r \in AllRecs :
        r.failure \in NonfatalFailures \cup OptionOffFailures \cup {"latched"} =>
            /\ r.launches = 0
            /\ ~r.latches
            /\ (r.made /\ ~r.cancelFailed) => r.cancelled = 1
            /\ r.cancelFailed => (r.optionOff /\ r.failure \in {"off-scatter", "off-stage"})

WorkerFatalFailuresLatch ==
    /\ \A r \in AllRecs :
         /\ (r.failure \in LatchingFailures /\ r.outcome = "raised") => r.latches
         /\ r.latches => r.failure \in LatchingFailures
         /\ r.failure = "prelaunch-cancel" => (r.launches = 0 /\ r.cancelFailed)
    /\ \A r \in Finished : r.failure = "async-fault" => phase \in {"failed", "shutdown", "killed"}
    /\ fault /= "none" => phase \in {"failed", "shutdown", "killed"}
    /\ \A r \in Finished :
         (r.optionOff /\ r.cancelFailed)
            => /\ ~r.latches /\ latched = r.latchedAtEntry
               /\ phase \in {"failed", "shutdown", "killed"}

ExactCellsReplayed ==
    /\ \A key \in launched :
         \/ key[1] = "chunk" /\ key[2] = K
         \/ key[1] = "echo" /\ key[2] = E
    /\ launched /= {} => K <= N /\ E <= N
    /\ (K > 0 /\ launched /= {}) => <<"chunk", K>> \in launched
    /\ Cardinality(launched) = rec.launches

AbortDoesNotCancel ==
    \A r \in AllRecs :
        (r.aborted /\ r.failure = "none")
            => (r.cancelled = 0 /\ (r.outcome = "completed" => r.staged = 1))

-------------------------------------------------------------------------------
(* Temporal properties                                                     *)

FailedWorkerShutsDown == (latched \/ phase = "failed") ~> (phase \in {"shutdown", "killed"})

LatchIsStable == [][latched => latched']_vars

DelayedFaultContained ==
    [][(fault = "none" /\ fault' = "collection")
         => /\ phase' = "failed"
            /\ latched' = latched
            /\ log' = log /\ last' = last
            /\ last.staged = 1 /\ last.cancelled = 0]_vars

===============================================================================
