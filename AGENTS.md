# AGENTS.md

Out-of-tree vLLM plugin for the cache-aware streaming Nemotron 3.5 ASR model. Start with
[docs/high-level-design.md](docs/high-level-design.md); component designs and their EARS specs
live under `docs/intent/<component>/`.

## Working rules

- Cite only public sources (vLLM, NVIDIA NeMo, Hugging Face model cards, published papers, public
  NVIDIA documentation). Never add proprietary material.
- Sign commits and add a DCO line: `Signed-off-by: Name <email>`.
- Session code uses only `AsyncLLM` / `EngineClient` public and RFC-proposed calls; the model
  adapter's permitted internal dependencies are listed in the `registration` design.
- README claims name the acceptance tier they meet (see the HLD's Success Metrics).

## LID

- Mode: Full
- Version: 1.3.0
