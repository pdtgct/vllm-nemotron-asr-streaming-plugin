# Status

The implementation follows the specs in `docs/intent/` (registration, session, transcription).
The CPU tiers pass; the GPU tier has not been run.

## First GPU round (A40)

Install the plugin into the engine's environment (`pip install -e .`) so the engine-core process
loads its entry point, and serve with the flags in the README.

1. `pytest -m "gpu and not inprocess_engine"` with `NEMOTRON_CARD_DIR` set.
2. `pytest -m inprocess_engine` in its own invocation (the in-process engine freezes `vllm.envs`
   for the rest of the run): one chunk per step at the staging, device-copy and encoder boundaries.
3. `NEMOTRON_TEST_AUDIO` naming a speech WAV, for file/stream label equivalence (FILE-ROW-007).
4. Check in particular, in this order: a file request's first schedule (keep-on-CPU fields and
   `file_header` through `mm_processor_kwargs`), the label-queue read after an advance, a file whose
   chunks emit no labels (continuation token through the emission checks), a file shorter than
   one chunk, a mixed stream-and-file step (group order), a malformed file request failing only its
   row, and `stream=true` transcription (FILE-ENG-006).
5. The served-set enforcement tests for direct clients (SESS-MODEL-001/002) and the
   transcription-server test (FILE-ENG-006) need a 48 GB card.

## Open items

- Replace the per-step label-queue read with the remaining count carried in the device's per-step
  status, and stage file headers and chunks through pinned buffers (see the transcription design's
  implementation notes).
- Deferred: a minimum-vLLM-version check in `register()` once releases include RFC 1 and RFC 2
  (REG-011).
