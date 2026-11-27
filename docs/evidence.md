# Evidence

This document records every acceptance claim for `vllm-nemotron-asr-streaming`. An acceptance
claim states that a build met one tier of the acceptance criteria. Each claim names the tier, the
build and the source revisions. Claims from earlier builds are marked historical.

The [high-level design](high-level-design.md) (Success Metrics) defines the five tiers:

1. CPU conformance: unit and contract tests, including the frontend's conformance suite, pass
   without a GPU.
2. Installed-engine cycle: on the target build, a stream runs from plugin load to final transcript
   and slot retirement, for one session and for concurrent sessions.
3. Equivalence: on public clips from 20 s to over 400 s, whole-file transcripts equal the streaming
   transcript.
4. Mixed load: the latency goal holds in unprofiled serving measurements.
5. Transaction graphs: the opt-in option meets its exactness, speed and memory criteria, separately
   at 80 ms and at 160 ms, on the A40 at FP32, on stage 1.

The raw evidence is in the public harness repository `pdtgct/nemotron-omni-port`, under the
experiment directories named below. The [README](../README.md#expected-performance) summarizes the
figures, and the [user manual](manual.md) explains the settings they depend on.

## Contents

- [Builds](#builds)
- [Tier 2 on the state-pool build](#tier-2-on-the-state-pool-build)
- [Unpublished measurement under `--enforce-eager`](#unpublished-measurement-under---enforce-eager)
- [Served accuracy on the target core](#served-accuracy-on-the-target-core)
- [Light round on the parent of the target core](#light-round-on-the-parent-of-the-target-core)
- [Block-0 capture under every KV-cache sizing](#block-0-capture-under-every-kv-cache-sizing)
- [Stream-only servers at `--max-model-len 144`](#stream-only-servers-at---max-model-len-144)
- [Transaction graphs, tier 5](#transaction-graphs-tier-5)
- [Owed](#owed)
- [Historical entries](#historical-entries)
- [Future roadmap](#future-roadmap)

## Builds

- Stage 1 (target core): `pdtgct/vllm` `pstate-impl-20261122` @ `9503e3b878`, the persistent-state
  layer with the #67583 model-state hooks. No tier has been met on it yet (Owed). The served
  accuracy round below ran on its ancestor `cbf13fed4a`; the light round below ran on `de0b0bccdb`,
  the parent of `cbf13fed4a`, which differs only in core's persistent-state detokenizer. The core
  at `9503e3b878` keeps chunked prefill off for the model, which does not declare
  `persistent_state_chunked_prefill`, and refuses an explicit `enable_chunked_prefill`.
- State-pool build (historical; the baseline for retiring the state pool, with plugin `v0.1.0b2`):
  `pdtgct/vllm` `hook-build-20261114b` @ `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee`
  (with #67583), precompiled wheel `191e5d5df2` (cu132), one NVIDIA A40 at FP32, card revision
  `ea30d66d`, frontend v0.4.0b3. Evidence: `pdtgct/nemotron-omni-port`
  `experiments/2026-11-15-plugin-tier2-hook-build/`, `experiments/2026-11-15-plugin-h1-txg-tier5-a/`
  and `experiments/2026-11-15-plugin-h1-txg-tier5-b/`.
- Stage 2 has not been measured with this release; frontend v0.4.0b3 passed CI on CPU only.

## Tier 2 on the state-pool build

- Partly achieved (state-pool build `9c40a6ffd6`, plugin `8166d85`, hook configuration, default
  segment arms, transaction graphs off), tier 2 (installed-engine cycle) at 320 ms: with
  `--max-num-seqs 64` and `--max-model-len 77000`, 5,559 LibriSpeech `/v1/realtime` sessions at
  concurrency 32 all ended `stop`, with WER 3.57 % on test-clean and 7.99 % on test-other. One final,
  for `2414-128291-0020` (test-other, 1.25 s, a known empty final), was empty, and the WER counts it
  as deletions. 640 test-clean sessions at concurrency 64 all reached a final transcript (WER
  2.89 %). The server logged no preemption and no error. GPU tier at plugin `adfb24b`: 173 of 175
  tests passed, 2 skipped, 0 failed.

The serving run's command was the README's serve command with `--enforce-eager` replaced by
`-cc.cudagraph_mode none -cc.mode none` (the hook configuration), at the checkpoint's default
lookahead, 3 (320 ms), with the default segment arms. The client sent audio at real-time pace.
Figures from the same run (state-pool build `9c40a6ffd6`, plugin `8166d85`, one A40):

| Figure | Value |
|---|---|
| `vllm serve` ready after launch | 130.6 s |
| Segment graphs retained, profiling capture | 1,544 MiB, captured in 42.6 s |
| Segment graphs retained, serving capture | 716 MiB, captured in 41.4 s |
| Core's graph pool | 0.70 GiB, against its 1.57 GiB estimate |
| Device memory in use (`nvidia-smi`), peak | 16,445 MiB, on the memory-profiling step |
| Available KV memory | 35.74 GiB; the KV budget admits 65 requests at 77,000 tokens |
| Final latency, concurrency 32, p50 / p99 | 206 / 430 ms (test-clean), 198 / 417 ms (test-other) |
| Final latency, concurrency 64, p50 / p99 | 234 / 499 ms (first 640 test-clean utterances) |

Final latency runs from the client's final commit to `transcription.done`. With
`--max-num-seqs 64`, the server started, both segment-graph captures fit, and 64 concurrent sessions
completed. The largest sampled step held 38 rows. Paced arrivals spread the chunks, so no 64-row
step was observed. Tier 2 figures under `--enforce-eager` and at lookaheads other than 320 ms are
owed.

## Unpublished measurement under `--enforce-eager`

These figures carry no acceptance tier. They come from an unpublished A40 measurement at 320 ms
under `--enforce-eager`, the runner configuration of the README's serve command. They are the source
of that command's interim `--max-num-seqs 64`.

| `--max-num-seqs` | Segment graphs retained | Device memory in use during capture, weights included | Startup |
|---|---|---|---|
| 32 | about 1.1 GiB | 7.0 GiB | about 41 s |
| 64 | about 2.3 GiB | 18.3 GiB | about 78 s |
| 128 | about 17.4 GiB | 44.4 GiB of the card's 45 GiB | about 182 s |

With the eager arms, startup took about 25 s. At 64 rows, 160 ms retained 1.8 GiB and 80 ms
retained 1.6 GiB.

## Served accuracy on the target core

No tier is met by this round. It records served word-error rate on stage 1.

- Build: `pdtgct/vllm` `pstate-impl-20261122` @ `cbf13fed4a`, with precompiled wheel `842f74fa8d`
  (cu132); plugin `feat/core-graphs-20261123` @ `0951319`; card revision `ea30d66d`; one NVIDIA
  A40 at FP32; 2026-11-26. Evidence: `pdtgct/nemotron-omni-port`
  `experiments/2026-11-26-plugin-wer-cbf13fed4a/`.
- 200 paced LibriSpeech test-clean utterances at 32 sessions on `/v1/realtime`, 0 failed in every
  run. Word-error rate under `--enforce-eager`:

  | Chunk size (ms) | 80 | 160 | 320 | 560 | 1120 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | WER (%) | 3.15 | 2.63 | 2.55 | 2.70 | 2.44 |

- At 320 ms with no runner flag, WER is 2.55% and all 200 transcripts equal those under
  `--enforce-eager`.
- One unprofiled run per setting with no repeats: no latency claim rests on this round.

## Light round on the parent of the target core

No tier is met by this round. It ran on `de0b0bccdb`, the parent of stage 1, which differs only in
core's persistent-state detokenizer, and records measurements that the owed tiers build on.

- Build: `pdtgct/vllm` `pstate-impl-20261122` @ `de0b0bccdb`, with precompiled wheel `842f74fa8d`
  (cu132), torch 2.14.0+cu132; plugin `feat/core-graphs-20261123` @ `2a55139`; card revision
  `ea30d66d`; one NVIDIA A40 at FP32; 2026-11-26. Evidence: `pdtgct/nemotron-omni-port`
  `experiments/2026-11-25-plugin-core-graphs-a40/`.
- Served transcripts, 200 paced LibriSpeech test-clean utterances at 32 sessions on `/v1/realtime`,
  compared by utterance id, are equal in every comparison:
  - the hook configuration (`-cc.cudagraph_mode none -cc.mode none`) and no runner flag, each
    against `--enforce-eager`, at 80, 160, 320, 560 and 1120 ms;
  - asynchronous scheduling, with and without `--enforce-eager`, against synchronous
    `--enforce-eager` at 160 and 320 ms;
  - under `VLLM_BATCH_INVARIANT=1`, 8 sessions and 1 session (20 utterances) against 32 sessions at
    320 ms;
  - transaction graphs on (`transaction_graph_populations=[63]`) against off at 80 and 160 ms under
    the restricted startup (`--num-gpu-blocks-override 65`, NONE/NONE, synchronous scheduling).
- The state-layer memory check (REG-STATE-019) passed in both arms.
- Served accuracy is not valid on this build. Word-error rate is 47 to 100%, rising as the chunk
  size falls, in every arm alike: finals lose the space before a word that starts a turn. The cause
  is in core's detokenizer for persistent-state streams, which starts each turn's decoding from the
  turn's prompt; the fix belongs to core. No accuracy, latency or block-count figure is claimed.
- Plugin GPU tests: registration 8 of 8 after one test correction; transcription 26 passed, 10
  failed; transaction graphs 113 passed, 3 failed (the exhaustive and random-plan equivalence cases
  were not run). Every failure is in the tests' own setup: the ten transcription tests build their
  stream reference through the session factory, which is not available on stage 1; the three
  transaction-graph tests start an engine without the restricted startup, or pass a Python function
  to a multiprocess engine.

## Block-0 capture under every KV-cache sizing

No tier is met by this round. It records the GPU checks of transaction graphs whose capture rows
sit on block 0, under each KV-cache sizing core offers, on stage 1.

- Build: `pdtgct/vllm` `pstate-impl-20261122` @ `9503e3b878`, with precompiled wheel `842f74fa8d`
  (cu132); plugin `feat/core-graphs-20261123` @ `21b5981`, with the test-support fixes and the
  capture record's plain torch version that `37115a1` and `3a0a046` carry; one NVIDIA A40 at FP32;
  2026-11-27. Evidence: `pdtgct/nemotron-omni-port` `experiments/2026-11-27-plugin-followup-b-a40/`
  and `experiments/2026-11-27-plugin-hlc-a40/`.
- TXG-QUAL-028 at 80 and 160 ms, under the extensible KV cache, a profiled budget with
  `--no-enable-extensible-kv-cache` and a fixed `--num-gpu-blocks-override`: one stream served alone
  and each of TXG-QUAL-004's files transcribed alone replayed cells with no fallback of reason
  `signature` and produced transcripts equal to `B_hook`'s under the same sizing.
- TXG-QUAL-029: the exactness runs under the extensible KV cache, `Q` and `M` and the fallback steps,
  compared equal, with every committed page outside the detached copy unchanged, inside the
  preregistered headroom.
- TXG-QUAL-022 (the uncharged-memory WARNING under a fixed `--kv-cache-memory-bytes`), TXG-CAP-018
  and TXG-EQ-003 on CUDA, the rest of the transaction-graph GPU tests, REG-ENG-032's and
  REG-STATE-019's arms with transaction graphs on, and the transcription GPU tests passed.
- Served transcripts at 32 sessions with transaction graphs on under the default sizing were not
  measured in this round; no served throughput, latency or tier 5 claim rests on it.

## Stream-only servers at `--max-model-len 144`

No tier is met by this round. A server that serves only streams runs at `--max-model-len 144`, the
smallest context that holds one stream unit, and so keeps core's default token budget of 2048.

- Build: as [Block-0 capture under every KV-cache sizing](#block-0-capture-under-every-kv-cache-sizing),
  at plugin `21b5981`. Evidence: `experiments/2026-11-27-plugin-followup-b-a40/`.
- 200 paced LibriSpeech test-clean utterances at 32 sessions on `/v1/realtime` under
  `--enforce-eager`: word-error rate 3.15% at 80 ms, 2.55% at 320 ms and 2.44% at 1120 ms, every
  transcript equal to the same server's at the full `--max-model-len`, and every request finished
  with `stop`, none length-capped. At 320 ms with no runner flag, all 200 transcripts equal those
  under `--enforce-eager`.
- The startup equals the full-length server's at a token budget of 2048: the same available KV
  cache memory, block count, encoder cache and graph memory. At `--max-model-len 77000` core raises
  the token budget to the context length, which takes about 14 GiB of available KV cache memory on
  the A40 (`experiments/2026-11-26-plugin-budget-9503e3b878/`).
- One unprofiled run per setting with no repeats: no latency claim rests on this round.

## Transaction graphs, tier 5

- Not met (state-pool build `9c40a6ffd6`; transaction level, body check and the revised TXG-QUAL-008 at
  plugin `adfb24b`, forced fallbacks and step rule at plugin `a19621e`, the first served runs,
  inventory and `U_api` at plugin `8166d85`), tier 5 for transaction graphs at 80 ms and at 160 ms:
  the step rule and `U_api` fail at both lookaheads, and the body check fails at population 1.

The tier 5 run used one NVIDIA A40 at FP32, the state-pool build `hook-build-20261114b`
(`pdtgct/vllm` @ `9c40a6ffd66ce4019ebcc6a0efac126bdbb4f6ee`), card revision `ea30d66d`, `N = 63`
and `--max-num-seqs 64`, at 80 ms and at 160 ms. Three plugin revisions ran its parts:

- `adfb24b`, whose production code equals this release's: the transaction-level checks, the capture
  figures, the body check and the revised served comparison of TXG-QUAL-008.
- `8166d85`: the first served runs, the inventory and `U_api`. This revision predates the change in
  `adfb24b` to how fresh rows take their zero initial state.
- `a19621e`, with `8166d85`'s production code: the forced fallbacks and the step rule.

Tier 5 is not met at either lookahead.

- Passed: the inventory (TXG-QUAL-025); every transaction-level step compared bit for bit at both
  lookaheads (TXG-QUAL-003); `Q` and `M` at 80 ms and at 160 ms (TXG-QUAL-002 to -006); forced
  fallbacks (TXG-QUAL-007); the revised served comparison (TXG-QUAL-008) at both lookaheads; the
  capture figures (TXG-QUAL-019).
- Failed: the step rule (TXG-QUAL-010 to -014), both groups at both lookaheads; `U_api`
  (TXG-QUAL-016, -017) at both lookaheads; the body check (TXG-QUAL-015, since retired; historical)
  at population 1, one block per lookahead. The first served `M` run at 80 ms (plugin `8166d85`)
  failed its original file criterion, equality with one `B_hook` run (123 of 124 files);
  TXG-QUAL-008 as revised replaced that criterion (see the note after the served-run table).
- Not run: tier 4 with the option on, on any build.

`N` = 63 and `--max-num-seqs 64` are the values the tier 5 run verified to fit at 80 ms and at
160 ms: the `N = 63` engines captured 126 cells per capture, the KV budget admitted 65 requests, and
the engines served the tier 5 `Q` and `M` workloads with no fallback. The inventory runs, at `N` =
64, served 512 streams at concurrency 64 in steps of up to 64 rows.

#### Step rule

`Δ` = min(median OFF1, median OFF2) − median ON, in ms, per block of 21 counted steps per leg; a
composition is material when `Δ ≥ max(5 ms, |median OFF1 − median OFF2|)` in both blocks
(TXG-QUAL-013). A group passes only when every one of its compositions is material.

| Group | P | (K, E) | 80 ms Δ, block 1 / 2 | Material | 160 ms Δ, block 1 / 2 | Material |
|---|---|---|---|---|---|---|
| one kind | 1 | (1, 0) | 6.85 / 6.72 | yes | 6.82 / 6.83 | yes |
| one kind | 31 | (31, 0) | 5.66 / 6.11 | yes | 6.00 / 5.89 | yes |
| one kind | 63 | (63, 0) | 4.38 / 4.30 | no | 4.28 / 4.26 | no |
| split | 31 | (27, 4) | 5.47 / 5.32 | yes | 5.04 / 4.92 | no |
| split | 31 | (16, 15) | 4.78 / 4.72 | no | 4.64 / 5.03 | no |
| split | 63 | (55, 8) | 3.75 / 6.42 | no | 4.10 / 3.58 | no |
| split | 63 | (32, 31) | 5.07 / 5.00 | yes | 5.25 / 5.01 | yes |

Echo-only steps, reported and not gated, gained 4.42 to 4.89 ms. U(1), the (1, 0) composition's
block 1 median (TXG-CAL-001): 15.06 ms on and 21.97 ms off at 80 ms, 17.46 ms on and 24.29 ms off at
160 ms.

#### Body check

Body check (TXG-QUAL-015, since retired; this record is historical): the option-off step time of
this release (`adfb24b`) against plugin `4cc1381`, both under `--enforce-eager` with transaction
graphs off, in two blocks of legs OLD1, NEW, OLD2 per population. Medians of 21 counted steps (ms);
a block passes when NEW − min(OLD1, OLD2) ≤ max(1 ms, |OLD1 − OLD2|). These are engine step times,
not served latencies.

| ms | P | Block 1: OLD1 / NEW / OLD2 | Block 2: OLD1 / NEW / OLD2 | NEW − mean(OLD), block 1 / 2 |
|---|---|---|---|---|
| 80 | 1 | 29.59 / 26.20 / 30.41 | 25.80 / 26.85 / 25.08 (**fail**) | −3.80 / +1.41 |
| 80 | 31 | 48.18 / 47.95 / 48.98 | 48.35 / 48.55 / 47.74 | −0.62 / +0.51 |
| 80 | 63 | 73.37 / 72.79 / 72.43 | 72.78 / 73.28 / 73.19 | −0.12 / +0.29 |
| 160 | 1 | 26.17 / 28.68 / 27.79 (**fail**) | 26.23 / 25.85 / 27.30 | +1.69 / −0.92 |
| 160 | 31 | 50.82 / 50.69 / 51.15 | 51.14 / 51.67 / 51.54 | −0.29 / +0.33 |
| 160 | 63 | 78.72 / 78.66 / 79.12 | 78.50 / 78.34 / 78.76 | −0.25 / −0.29 |

The check is not met: at population 1 one block per lookahead fails (80 ms: NEW − min(OLD) 1.77 ms
against a bound of 1.00 ms; 160 ms: 2.50 ms against 1.62 ms), while in the other block of the same
lookahead NEW is faster than both OLD legs. At populations 31 and 63 every block passes. In an
ungated bridge block (option off, `--enforce-eager` legs around one leg in the hook
configuration), the hook configuration was within the eager legs' drift at 160 ms and 0.7 ms (P 1)
and 0.9 ms (P 63) above it at 80 ms.

#### `U_api`

`U_api` (TXG-QUAL-016, -017; plugin `8166d85`), five brackets of OFF1, ON, OFF2 per lookahead at
`N = 63`: not met at either lookahead. The per-bracket rule failed in three brackets at 80 ms and in
two at 160 ms; over the brackets, the counted median was 28.49 ms on and 37.45 ms off at 80 ms, and
30.59 ms on and 38.68 ms off at 160 ms.

#### Exactness and served runs

| Exactness and served runs | 80 ms | 160 ms |
|---|---|---|
| Transaction-level `Q` (concurrency 1, 32 and 63, 2,000 steps each) and `M`, plugin `adfb24b`: steps covered and equal to the option-off transaction in outputs, status, progress, queue counts and every page | 8,151 of 8,151 (`M` 2,151); 0 fallbacks | 8,158 of 8,158 (`M` 2,158); 0 fallbacks |
| `B_hook` replay of that run's schedule: requests with equal label tokens | 2,036 of 2,036 | 1,700 of 1,700 |
| Forced fallbacks (`K` > `N`, `E` > `N`, two signature mismatches; plugin `a19621e`): each counted once, outputs equal to the option-off transaction's | pass | pass |
| Served `Q` (plugin `8166d85`), 512 streams at concurrency 63: transcripts equal to `B_hook`'s | 512 of 512; 2,251 steps, all replayed | 512 of 512; 1,978 steps, all replayed |
| Served `M` (plugin `8166d85`), 48 stream slots and 15 file slots, 512 streams and 124 files: transcripts equal to `B_hook`'s | streams 512 of 512, files 123 of 124; 2,221 steps, all replayed | streams 512 of 512, files 124 of 124; 2,494 steps, all replayed |

The 80 ms file `1284-1180-0011` differs from the `B_hook` run by one comma ("I am, my dear, and all
strangers …" against "I am, my dear and all strangers …"). A second, nonqualifying `B_hook` run
differed from the first on that file and equalled the option-on run on all 124 files. This points
to `B_hook`'s own run-to-run variation under wall-clock pacing. Revised TXG-QUAL-008 comparison (two
`B_hook` runs, and each file alone; plugin `adfb24b`): pass at both lookaheads. Every request
completed, the option-on engine replayed every step with no fallback, and every file sent alone to an
idle server equalled both `B_hook` runs (124 of 124). Under load the option-on transcripts equalled
both `B_hook` runs on every stream and file at 160 ms and on all but `1284-1180-0011` at 80 ms, where
the option-on run kept the comma and both `B_hook` runs dropped it; alone, all three dropped it.

#### Graph memory and capture time

The table gives graph memory on the A40 at `N = 63`, per capture, in MiB unless stated. The call
total covers the whole capture call: segment graphs, cells and, in the profiling capture, a 64 MiB
margin. The captures ran at plugin `adfb24b` on the state-pool build `9c40a6ffd6`. Available KV memory
and ready time come from the served runs at plugin `8166d85`, whose profiling transaction totals
were 2,462 MiB (80 ms) and 2,702 MiB (160 ms).

| | 80 ms | 160 ms |
|---|---|---|
| Profiling capture: call / transaction total (chunk / echo) | 3,832 / 2,410 (2,144 / 266) | 4,148 / 2,676 (2,410 / 266) |
| Serving capture: call / transaction total (chunk / echo) | 986 / 422 (422 / 0) | 1,042 / 428 (428 / 0) |
| Transient peak, each capture | 2,104 | 2,154 |
| Capture time, each capture | 69.5 s and 70.6 s | 73.1 s and 74.0 s |
| Available KV memory, option on (`B_hook`) | 33.63 GiB (36.03 GiB) | 33.31 GiB (35.95 GiB) |
| `vllm serve` ready after launch (`B_hook`) | 220.8 s (100.6 s) | 179.4 s (110.4 s) |

#### The `(K, E)` histogram and the choice of `N`

The `(K, E)` histogram from which `N` was chosen comes from the served `Q` run on an `N` = 64 engine
(512 streams at concurrency 64). The three 160 ms steps outside `N` = 63 are (64, 0); an `N` = 63
engine runs them without transaction graphs (`chunk-uncovered`).

| | 80 ms | 160 ms |
|---|---|---|
| Served steps (split / one-kind CHUNK / echo only) | 2,654 (2,262 / 270 / 122) | 1,927 (1,554 / 182 / 191) |
| Largest `K` / `E` | 60 / 33 | 64 / 40 |
| Fraction covered at `N` = 32 / 40 / 48 / 63 | 0.8873 / 0.9717 / 0.9962 / 1.0000 | 0.9725 / 0.9974 / 0.9979 / 0.9984 |

Tier 5 chooses `N` as the smallest count of at least 63 that fits and for which steps with `K ≤ N`
and `E ≤ N` make up at least 99 % of served steps. At `N` = 63 they make up 100 % at 80 ms and
99.84 % at 160 ms. The same rule below 63 carries no tier 5 claim. It gives a deployment `N` of 44 at 80 ms
(99.10 %; serving capture 832 MiB call total, 268 MiB transaction total) and of 36 at 160 ms
(99.07 %; 794 and 180 MiB).

## Owed

- Tiers 1 to 4 on the target core at the current plugin, at every chunk size (80, 160, 320, 560 and
  1120 ms), under `--enforce-eager` and the hook configuration, with and without asynchronous
  scheduling and batch invariance.
- Served latency on the target core, with repeats (Served accuracy on the target core gives one
  unprofiled run per chunk size).
- Tier 5 for transaction graphs at 80 and 160 ms on the target core, under the restricted startup;
  served equality is measured above.
- Served transcripts at 32 sessions with transaction graphs on under core's default KV-cache sizing,
  against `B_hook`, at 80 and 160 ms.

## Historical entries

Every entry below is historical. The state-pool entries ran on a state-pool build without #67583, at
plugin revisions before the default graphed arms and the one-lookahead server. The other entries ran
on the earlier integration build of RFC 1 (the superseded #65544 API) and RFC 2. None is a claim
about the current plugin on the target core or on stage 2.

- Historical state-pool build (without #67583): `pdtgct/vllm` `stage1-build-20261113` @ `b7e7a20c26`
  (upstream `main` `cf8fce503c` + vllm#67375 + vllm#67151), precompiled wheel `191e5d5df2` (cu132),
  one NVIDIA A40. Evidence: `pdtgct/nemotron-omni-port`
  `experiments/2026-11-13-two-stage-stage1-a40/`.
- Achieved (historical, state-pool build without #67583, plugin `5acd9d2`), tier 1 (CPU conformance):
  1203 passed, 43 skipped.
- Achieved (historical, state-pool build without #67583; serving at plugin `a850916`, GPU tier at
  `5acd9d2`, which changes only tests since), tier 2 (installed-engine cycle): the GPU tier passed
  (35 passed, 1 skipped), and 5559 LibriSpeech `/v1/realtime` sessions at concurrency 32 each
  reached a final transcript with none failed. Tiers 3 and 4 were not run on that build.

- Historical build: the merge of RFC 1 `persistent-state-primitive-v2` @ `c384f178cd` and RFC 2
  `endpoint-plugin-lifecycle` @ `6b1ce05706` on `pdtgct/vllm` (branch `rfc12-integration-v2` @
  `c7cf80ac45`).
- Achieved (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 1 (CPU
  conformance): the CPU-tier tests passed, including the request guards of the transcription route
  (run against a core `SpeechToTextParams` that carries the two fields of vllm#67151). The
  frontend-free file route was not run at tier 2 (installed engine).
- Achieved (historical, integration build `c7cf80ac45`, plugin `4e4fd6f` for the GPU tier; the
  unmodified-client runs record no plugin revision), tier 2 (installed-engine cycle), on one A40
  with the Riva frontend at `782dec8`: this repository's GPU tier passed, and unmodified Riva clients
  completed streaming, `Recognize`, Realtime WebSocket and file-route requests, for one session and
  for eight concurrent sessions.
- Partly achieved (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 3
  (equivalence), same A40: on eight public clips from 20 s to 428 s, the core transcription route
  (sent with `language`) and `Recognize` returned identical transcripts, and the whole-file
  transcript equaled a session at the 1120 ms file cadence. The comparison with the offline plugin
  has not been run on these clips.
- Not met (historical, integration build `c7cf80ac45`, plugin revision not recorded), tier 4 (mixed
  load): with one 293 s file request in flight, eight concurrent streams' p95 final latency rose
  from 0.55 s to 3.64 s, with no preemptions. Run file traffic on a separate server, or keep files
  short, until this is fixed.
- These are exploratory measurements on one GPU, not a qualification run.

## Future roadmap

Owed lists the records due on the target core now. These records wait on a trigger:

- **Stage 2 records.** Trigger: the session factory runs on core's automatic admission
  ([high-level design, Future roadmap](high-level-design.md#future-roadmap)). Shape: tier 2 and
  tier 3 entries for the frontend's routes on a recorded stage 2 commit.
- **Latency parity with NVIDIA's NIM.** Trigger: tier 4 met on stage 1 with transaction graphs off
  and on, and a NIM that serves this checkpoint available. Shape: an entry comparing final latency
  and time to the first transcript on one GPU, naming both builds.
- **Transaction graphs at 320, 560 and 1120 ms.** Trigger: their cells are specified (the
  `transaction-graph` design, Deferred 5). Shape: a tier 5 entry per geometry.
- **The state-pool comparison.** Trigger: tier 2 and tier 5 met on the target core. Shape: an
  entry that compares them with the state-pool build's records above, by which RFC 1 retires the
  state pool.
