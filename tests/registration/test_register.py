# SPDX-License-Identifier: Apache-2.0
"""``register()``, the entry point and import-time behaviour (REG-001 to REG-013).

Tests that need an unregistered process or a clean ``sys.modules`` run a fresh
interpreter (``run_fresh``).
"""

from __future__ import annotations

import importlib
import json
import tomllib

import pytest

ARCH = "Nemotron3_5AsrStreamingForRNNT"
OFFLINE = "Nemotron3_5AsrForRNNT"
LAZY = "vllm_nemotron_asr_streaming.model:Nemotron3_5AsrStreamingForRNNT"

#: Registry snapshot used by the fresh-process snippets.
_SNAPSHOT = """
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.transformers_utils import config as tcfg
from vllm.tokenizers.registry import TokenizerRegistry

def snap():
    return dict(
        models={k: repr(v) for k, v in ModelRegistry.models.items()},
        cfg={k: repr(v) for k, v in MODELS_CONFIG_MAP.items()},
        parsers={k: repr(v) for k, v in tcfg._CONFIG_FORMAT_TO_CONFIG_PARSER.items()},
        tokenizers={k: repr(v) for k, v in TokenizerRegistry.tokenizers.items()},
    )
"""


# @spec REG-001
def test_single_general_plugin_entry_point(repo_root):
    project = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert project["project"]["name"] == "vllm-nemotron-asr-streaming"
    group = project["project"]["entry-points"]["vllm.general_plugins"]
    assert group == {"nemotron_asr_streaming": "vllm_nemotron_asr_streaming:register"}
    module_name, _, attr = group["nemotron_asr_streaming"].partition(":")
    target = getattr(importlib.import_module(module_name), attr)
    assert callable(target)
    assert target.__name__ == "register"


@pytest.mark.engine
@pytest.mark.parametrize(
    "missing",
    [
        ["stream_admission"],
        ["wait_stream_retired"],
        ["stream_admission", "wait_stream_retired"],
    ],
    ids=["no-stream_admission", "no-wait_stream_retired", "neither"],
)
# @spec REG-002
def test_missing_client_calls_raise_and_change_no_registry(run_fresh, missing):
    # Core surface: AsyncLLM.generate(..., stream_admission=...) and
    # AsyncLLM.wait_stream_retired (vllm/v1/engine/async_llm.py).
    code = (
        _SNAPSHOT
        + """
import vllm_nemotron_asr_streaming as plugin
from vllm.v1.engine.async_llm import AsyncLLM
MISSING = json.loads('__MISSING__')
if "stream_admission" in MISSING:
    async def generate(self, prompt, sampling_params, request_id):
        yield None
    AsyncLLM.generate = generate
if "wait_stream_retired" in MISSING:
    del AsyncLLM.wait_stream_retired
before = snap()
raised = message = None
try:
    plugin.register()
except Exception as exc:
    raised, message = type(exc).__name__, str(exc)
emit(raised=raised, message=message, unchanged=before == snap(), after=snap())
""".replace("__MISSING__", json.dumps(missing))
    )
    out = run_fresh(code)
    assert out["raised"], "register() did not raise on an engine without the calls"
    for name in missing:
        assert name in out["message"]
    assert "RFC 1" in out["message"] and "RFC 2" in out["message"]
    assert out["unchanged"]
    after = out["after"]
    assert ARCH not in after["models"]
    assert ARCH not in after["cfg"]
    assert "nemotron_asr_streaming" not in after["parsers"]
    assert "nemotron_asr_streaming" not in after["tokenizers"]


@pytest.mark.engine
# @spec REG-003
def test_architecture_registered_as_lazy_reference(registered):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.registry import _LazyRegisteredModel

    entry = ModelRegistry.models[ARCH]
    assert isinstance(entry, _LazyRegisteredModel)
    assert f"{entry.module_name}:{entry.class_name}" == LAZY


@pytest.mark.engine
# @spec REG-003, REG-012
def test_register_and_session_import_leave_model_modules_unimported(run_fresh):
    out = run_fresh(
        """
import vllm_nemotron_asr_streaming as plugin
plugin.register()
import vllm_nemotron_asr_streaming.session
names = (
    "vllm_nemotron_asr_streaming.model",
    "vllm_nemotron_asr_streaming.nemotron_asr.nemotron_asr",
)
emit(imported=[n for n in names if n in sys.modules])
"""
    )
    assert out["imported"] == []


@pytest.mark.engine
# @spec REG-004
def test_models_config_map_entry_is_the_plugin_hook(registered):
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        VerifyAndUpdateConfig,
    )

    from vllm_nemotron_asr_streaming.streaming_config import (
        Nemotron3_5AsrStreamingConfig,
    )

    assert MODELS_CONFIG_MAP[ARCH] is Nemotron3_5AsrStreamingConfig
    assert issubclass(Nemotron3_5AsrStreamingConfig, VerifyAndUpdateConfig)


@pytest.mark.engine
# @spec REG-005
def test_card_parser_registered_under_its_format_name(registered):
    from vllm.transformers_utils.config import get_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase
    from vllm_nemotron_asr_streaming.card_parser import (
        NemotronAsrStreamingConfigParser,
    )

    parser = get_config_parser("nemotron_asr_streaming")
    assert isinstance(parser, NemotronAsrStreamingConfigParser)
    assert isinstance(parser, ConfigParserBase)


@pytest.mark.engine
# @spec REG-006
def test_tokenizer_mode_registered(registered):
    from vllm.tokenizers.registry import TokenizerRegistry

    assert TokenizerRegistry.tokenizers["nemotron_asr_streaming"] == (
        "vllm_nemotron_asr_streaming.tokenizer_mode",
        "NemotronAsrStreamingTokenizer",
    )
    cls = TokenizerRegistry.load_tokenizer_cls("nemotron_asr_streaming")
    assert callable(cls.from_pretrained)


_OFFLINE_CASES = {
    "models-absent": "models_absent",
    "models-changed": "models_changed",
    "config-map-absent": "cfg_absent",
    "config-map-changed": "cfg_changed",
}

_OFFLINE_SNIPPET = (
    _SNAPSHOT
    + """
import vllm_nemotron_asr_streaming as plugin
from vllm.model_executor.models.registry import _LazyRegisteredModel

OFFLINE = "Nemotron3_5AsrForRNNT"
CASE = "__CASE__"

class SentinelA: pass
class SentinelB: pass

if CASE in ("models_changed", "clean"):
    ModelRegistry.models[OFFLINE] = _LazyRegisteredModel("other.module", "A")
if CASE in ("cfg_changed", "clean"):
    MODELS_CONFIG_MAP[OFFLINE] = SentinelA

original = ModelRegistry.register_model
def wrapped(model_arch, model_cls):
    original(model_arch, model_cls)
    if CASE.startswith("models"):
        ModelRegistry.models[OFFLINE] = _LazyRegisteredModel("other.module", "B")
    if CASE.startswith("cfg"):
        MODELS_CONFIG_MAP[OFFLINE] = SentinelB
ModelRegistry.register_model = wrapped if CASE != "clean" else original

before = snap()
raised = None
try:
    plugin.register()
except Exception as exc:
    raised = type(exc).__name__
after = snap()
emit(
    raised=raised,
    offline_models_same=before["models"].get(OFFLINE) == after["models"].get(OFFLINE),
    offline_cfg_same=before["cfg"].get(OFFLINE) == after["cfg"].get(OFFLINE),
    registered="Nemotron3_5AsrStreamingForRNNT" in after["models"],
)
"""
)


@pytest.mark.engine
@pytest.mark.parametrize("case", list(_OFFLINE_CASES), ids=list(_OFFLINE_CASES))
# @spec REG-007
def test_offline_key_change_during_register_raises(run_fresh, case):
    out = run_fresh(_OFFLINE_SNIPPET.replace("__CASE__", _OFFLINE_CASES[case]))
    assert out["raised"], f"register() accepted a changed offline entry ({case})"


@pytest.mark.engine
# @spec REG-007
def test_register_leaves_existing_offline_entries_alone(run_fresh):
    out = run_fresh(_OFFLINE_SNIPPET.replace("__CASE__", "clean"))
    assert out["raised"] is None
    assert out["offline_models_same"] and out["offline_cfg_same"]
    assert out["registered"]


@pytest.mark.engine
# @spec REG-008
def test_register_is_idempotent(registered):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.transformers_utils.config import _CONFIG_FORMAT_TO_CONFIG_PARSER

    from vllm_nemotron_asr_streaming import register

    def state():
        return (
            ModelRegistry.models[ARCH],
            MODELS_CONFIG_MAP[ARCH],
            _CONFIG_FORMAT_TO_CONFIG_PARSER["nemotron_asr_streaming"],
            TokenizerRegistry.tokenizers["nemotron_asr_streaming"],
        )

    first = state()
    register()
    register()
    assert state() == first
    assert state()[1] is first[1] and state()[2] is first[2]


@pytest.mark.engine
# @spec REG-009
def test_register_touches_no_device_and_loads_no_weights(run_fresh):
    out = run_fresh(
        """
import vllm
import vllm.model_executor.models
import torch, safetensors, safetensors.torch
import vllm_nemotron_asr_streaming as plugin

def boom(*a, **k):
    raise AssertionError("device or weights touched")

torch.cuda._lazy_init = boom
torch.cuda.init = boom
torch.load = boom
safetensors.safe_open = boom
safetensors.torch.load_file = boom
plugin.register()
emit(cuda_initialized=torch.cuda.is_initialized())
"""
    )
    assert out["cuda_initialized"] is False


@pytest.mark.engine
# @spec REG-010
def test_missing_adapter_dependencies_named_on_model_import(run_fresh):
    out = run_fresh(
        """
import importlib
for module in (
    "vllm.v1.worker.gpu.state_slots",
    "vllm.v1.worker.utils",
    "vllm.v1.kv_cache_interface",
    "vllm.v1.worker.gpu.states",
):
    importlib.import_module(module)
delattr(sys.modules["vllm.v1.worker.gpu.state_slots"], "StateSlotRows")
delattr(sys.modules["vllm.v1.worker.utils"], "AttentionGroup")
delattr(sys.modules["vllm.v1.kv_cache_interface"], "StateSlotHandle")
sys.modules["vllm.v1.worker.gpu.states"] = None
raised = message = None
try:
    importlib.import_module("vllm_nemotron_asr_streaming.model")
except Exception as exc:
    raised, message = type(exc).__name__, str(exc)
emit(raised=raised, message=message)
"""
    )
    assert out["raised"], "model import accepted missing adapter dependencies"
    for needle in (
        "StateSlotRows",
        "vllm.v1.worker.gpu.state_slots",
        "AttentionGroup",
        "vllm.v1.worker.utils",
        "StateSlotHandle",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.worker.gpu.states",
    ):
        assert needle in out["message"], f"{needle!r} not named in the error"


@pytest.mark.engine
# @spec REG-013
def test_model_import_initializes_no_device_and_loads_no_weights(run_fresh):
    out = run_fresh(
        """
import vllm
import torch, safetensors, safetensors.torch

def boom(*a, **k):
    raise AssertionError("device or weights touched")

torch.cuda._lazy_init = boom
torch.cuda.init = boom
torch.load = boom
safetensors.safe_open = boom
safetensors.torch.load_file = boom
import vllm_nemotron_asr_streaming.model as model
emit(
    cuda_initialized=torch.cuda.is_initialized(),
    has_class=hasattr(model, "Nemotron3_5AsrStreamingForRNNT"),
)
"""
    )
    assert out["has_class"]
    assert out["cuda_initialized"] is False
