# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` entry point.

``register`` checks that the vLLM build carries the core changes the plugin
needs (``_require_client_path``), then registers the
streaming architecture, its model-configuration hook, the card parser and the
tokenizer mode. It does not import the model module, initialize a device or
load weights, so a process that has not initialized CUDA can still load the
plugin. The model adapter's own dependencies are checked when the model class
module is imported (``engine_flags.check_adapter_dependencies``).
"""

from __future__ import annotations

import inspect


def _require_client_path() -> None:
    """Fail plugin load on a vLLM build the PR 6 proof of concept cannot run
    on: one without vllm#67583's model-state hooks or without pdtgct/vllm#6's
    non-recomputable state layer (``MambaManager.layer_config(...,
    recomputable=...)``). Stream admission is not required: PR 6 has none,
    and the session factory that uses it is not supported on this build."""
    from vllm_nemotron_asr_streaming.engine_flags import missing_graph_hooks

    missing = [f"ModelState.{hook}" for hook in missing_graph_hooks()]
    try:
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager

        layer_config = getattr(MambaManager, "layer_config", None)
        has_flag = layer_config is not None and (
            "recomputable" in inspect.signature(layer_config).parameters
        )
    except (ImportError, TypeError, ValueError):
        has_flag = False
    if not has_flag:
        missing.append("MambaManager.layer_config(recomputable=...)")
    if missing:
        raise RuntimeError(
            "vllm-nemotron-asr-streaming (PR 6 proof of concept) requires a "
            "vLLM build with pdtgct/vllm#6's state layer and vllm#67583's "
            "model-state hooks; this build lacks " + ", ".join(missing)
        )


# @spec REG-002, REG-003, REG-004, REG-005, REG-006, REG-007, REG-008, REG-009
def register() -> None:
    """Register the streaming architecture with vLLM.

    Safe to call more than once in one process. Never writes the offline
    architecture key or its configuration hook.
    """
    _require_client_path()

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.transformers_utils import config as transformers_config

    from .card_parser import NemotronAsrStreamingConfigParser
    from .identity import (
        ARCHITECTURE,
        CONFIG_FORMAT,
        LAZY_MODEL,
        OFFLINE_ARCHITECTURE,
        TOKENIZER_MODE,
    )
    from .streaming_config import Nemotron3_5AsrStreamingConfig

    if ARCHITECTURE == OFFLINE_ARCHITECTURE:
        raise RuntimeError(
            f"refusing to register the offline architecture {OFFLINE_ARCHITECTURE!r}"
        )
    offline_model_before = ModelRegistry.models.get(OFFLINE_ARCHITECTURE)
    offline_config_before = MODELS_CONFIG_MAP.get(OFFLINE_ARCHITECTURE)

    module_name, _, class_name = LAZY_MODEL.partition(":")
    registered = ModelRegistry.models.get(ARCHITECTURE)
    if (
        getattr(registered, "module_name", None) != module_name
        or getattr(registered, "class_name", None) != class_name
    ):
        ModelRegistry.register_model(ARCHITECTURE, LAZY_MODEL)
    MODELS_CONFIG_MAP[ARCHITECTURE] = Nemotron3_5AsrStreamingConfig

    parsers = transformers_config._CONFIG_FORMAT_TO_CONFIG_PARSER
    if parsers.get(CONFIG_FORMAT) is not NemotronAsrStreamingConfigParser:
        transformers_config.register_config_parser(CONFIG_FORMAT)(
            NemotronAsrStreamingConfigParser
        )

    tokenizer_target = (
        "vllm_nemotron_asr_streaming.tokenizer_mode",
        "NemotronAsrStreamingTokenizer",
    )
    if TokenizerRegistry.tokenizers.get(TOKENIZER_MODE) != tokenizer_target:
        TokenizerRegistry.register(TOKENIZER_MODE, *tokenizer_target)

    # In this core build the tokenizer mode also selects the renderer; the
    # standard HF renderer serves token-id prompts with multimodal data.
    renderer_target = ("vllm.renderers.hf", "HfRenderer")
    if RENDERER_REGISTRY.renderers.get(TOKENIZER_MODE) != renderer_target:
        RENDERER_REGISTRY.register(TOKENIZER_MODE, *renderer_target)

    if ModelRegistry.models.get(OFFLINE_ARCHITECTURE) != offline_model_before:
        raise RuntimeError(
            f"streaming registration changed the offline architecture "
            f"{OFFLINE_ARCHITECTURE!r}"
        )
    if MODELS_CONFIG_MAP.get(OFFLINE_ARCHITECTURE) != offline_config_before:
        raise RuntimeError(
            f"streaming registration changed the offline config hook "
            f"{OFFLINE_ARCHITECTURE!r}"
        )
    if ModelRegistry.models.get(ARCHITECTURE) is None:
        raise RuntimeError(f"{ARCHITECTURE!r} was not registered")
