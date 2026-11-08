# vllm-nemotron-asr-streaming

Out-of-tree plugin for cache-aware streaming Nemotron ASR on vLLM core. It serves the published
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
card unmodified, for streaming sessions and for complete-file transcription. It targets a vLLM
build with RFC 1 (the persistent-state primitive and stream admission, `AsyncLLM.generate(...,
stream_admission=True)` and `wait_stream_retired`) and RFC 2 (the endpoint plugin lifecycle); a
released vLLM does not provide the symbols it needs, and `register()` fails with a message saying so.

The design is in `docs/high-level-design.md` and `docs/intent/`; module names are in
`docs/implementation-layout.md`.

## What is registered

Distribution `vllm-nemotron-asr-streaming`, import package `vllm_nemotron_asr_streaming`. Two entry
points:

```toml
[project.entry-points."vllm.general_plugins"]
nemotron_asr_streaming = "vllm_nemotron_asr_streaming:register"

[project.entry-points."vllm_riva_frontend.session_factories"]
nemotron_asr_streaming = "vllm_nemotron_asr_streaming.session:create_session_factory"
```

`register()` checks the client calls it uses (`AsyncLLM.generate(stream_admission=...)` and
`AsyncLLM.wait_stream_retired`), then registers, without importing the model or touching a device:

- the architecture `Nemotron3_5AsrStreamingForRNNT` (a lazy reference) and its model-configuration hook;
- the config format `nemotron_asr_streaming`, which translates the Hub card;
- the tokenizer mode `nemotron_asr_streaming` (and the renderer of the same name), which makes the
  control tokens special and the park token the EOS.

The Hub card names the architecture `Nemotron3_5AsrForRNNT`, which is the offline plugin's key. This
plugin never writes that key or its configuration hook, so both plugins can be installed together.
When `VLLM_PLUGINS` is an allowlist, include `nemotron_asr_streaming`.

## Serve

<!-- @spec REG-ENG-007 -->
```bash
vllm serve nvidia/nemotron-3.5-asr-streaming-0.6b \
    --dtype float32 \
    --enforce-eager \
    --no-async-scheduling \
    --no-enable-prefix-caching \
    --config-format nemotron_asr_streaming \
    --tokenizer-mode nemotron_asr_streaming \
    --hf-overrides '{"supported_num_lookahead_tokens": [6, 13], "num_lookahead_tokens": 13}'
```

- The persistent-state model needs synchronous scheduling, no prefix caching, a single device and an
  eager outer runner (it runs its own captured graphs inside the step). Other settings it refuses
  (speculative decoding, KV transfer, dual-batch overlap, sleep mode, the cumem allocator, hierarchical
  sparse attention, any parallel size other than 1) fail at startup with a message naming each.
- `num_lookahead_tokens` is required: the card has no default the plugin accepts. It sets the cadence for
  complete files and for `/v1/realtime`, and must be in the served set.
- `supported_num_lookahead_tokens` is the served set. It defaults to the card's own set (0, 3, 6, 13) and
  accepts any of the implemented right contexts 0, 1, 3, 6 and 13. The reference deployment serves
  `[6, 13]` with default 13.
- Pass `--generation-config vllm` if the checkpoint's generation configuration or an
  `--override-generation-config` sets a token limit; the startup error says when.

Support boundary:

- Core's `/v1/realtime` is served directly, with no frontend. A unit core rejects ends that stream, as
  core's contract says.
- Complete-file transcription through core's `/v1/audio/transcriptions` is supported only with the Riva
  frontend's guards (the file-size cap, and refusal of beam search and client completion limits). Without
  that frontend it is unsupported.
- The session factory, found through the second entry point, is for frontends.

## Evidence

- Target engine: the merge of RFC 1 `persistent-state-primitive-v2` @ `574efbd8f6` and RFC 2
  `endpoint-plugin-lifecycle` @ `6b1ce05706` on `pdtgct/vllm` (local integration commit `5194f13ebc`, not
  yet published).
- Achieved: CPU-tier tests (all passing; see `TODO.md` for the GPU round). The GPU tier has not been run.
- Pending: installed-engine runs on an A40, file/stream equivalence, and mixed load.

## Install and test

```bash
pip install -e '.[test]'
pytest
```

Tests are tiered: unmarked tests need only this package; `engine` tests need the RFC 1 + RFC 2 vLLM build
on `PYTHONPATH`; `gpu` tests need an installed engine on CUDA; `card` tests need `NEMOTRON_CARD_DIR` set to
a local copy of the published card, which is never committed.
