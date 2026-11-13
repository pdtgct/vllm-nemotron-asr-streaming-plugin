# SPDX-License-Identifier: Apache-2.0
"""The model adapter's vLLM dependencies (REG-010, REG-014, REG-015, REG-016).

The permitted set is the registration design's dependency table
(``docs/intent/registration/registration-design.md``, "Model-adapter
dependencies"); the tests read it from there, so the design and the checks
cannot drift apart. Conditional rows are the graph runtime (REG-014) and the
batch-invariant module (REG-015); every other row is unconditional (REG-010).

Absence of a dependency is simulated in a fresh interpreter: the plugin's
modules are imported once (so core's own modules are loaded and keep the
names they bound), then the plugin's modules are dropped from
``sys.modules``, the names are removed from core's modules, and the plugin
is imported again. Only the plugin's own imports can then notice.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

DESIGN = Path("docs/intent/registration/registration-design.md")
PACKAGE = "vllm_nemotron_asr_streaming"

#: REG-014: the conditional graph-runtime dependencies.
GRAPH_RUNTIME = (
    ("vllm.distributed.parallel_state", "graph_capture"),
    ("vllm.forward_context", "BatchDescriptor"),
    ("vllm.forward_context", "set_forward_context"),
    ("vllm.platforms", "current_platform"),
    ("vllm.utils.import_utils", "resolve_obj_by_qualname"),
)

#: REG-015: the conditional batch-invariant dependency (a whole module).
BATCH_INVARIANT = "vllm.model_executor.determinism.batch_invariant"

#: The interfaces the design excludes from the table ("vLLM's
#: model-registration, config parser, tokenizer and renderer registries, model
#: configuration hooks, speech-to-text configuration, multimodal processing,
#: ``SupportsTranscription``, logging and Hub file helpers"), as the modules
#: that provide them. A module ending in ``.`` is a prefix.
EXCLUDED_MODULES = {
    # model registration
    "vllm.model_executor.models": {"ModelRegistry"},
    # config parser registry
    "vllm.transformers_utils": {"config"},
    "vllm.transformers_utils.config": None,
    "vllm.transformers_utils.config_parser_base": {"ConfigParserBase"},
    # tokenizer registry and the tokenizer type it registers
    "vllm.tokenizers.registry": {"TokenizerRegistry"},
    "vllm.tokenizers.hf": {"CachedHfTokenizer", "get_cached_tokenizer"},
    # renderer registry
    "vllm.renderers.registry": {"RENDERER_REGISTRY"},
    # model configuration hooks
    "vllm.model_executor.models.config": {"VerifyAndUpdateConfig", "MODELS_CONFIG_MAP"},
    # speech-to-text configuration
    "vllm.config.speech_to_text": {"SpeechToTextConfig"},
    # multimodal processing
    "vllm.multimodal.": None,
    "vllm.config.multimodal": {"MultiModalDummyOptions"},
    "vllm.inputs": {"MultiModalDataDict"},
    # SupportsTranscription
    "vllm.model_executor.models.interfaces": {"SupportsTranscription"},
    # logging
    "vllm.logger": {"init_logger"},
    # Hub file helpers
    "vllm.transformers_utils.repo_utils": {"get_hf_file_to_dict"},
}

#: The public client surface (``AsyncLLM`` / ``EngineClient`` and the types
#: they exchange), which the session and ``register()``'s client-path check
#: use. It is outside the model adapter, so these files may use it.
CLIENT_SURFACE_FILES = ("plugin.py", "session/")
CLIENT_SURFACE = {
    "vllm.v1.engine.async_llm": {"AsyncLLM"},
    "vllm.engine.protocol": {"StreamingInput"},
    "vllm.sampling_params": {"RequestOutputKind", "SamplingParams"},
    "vllm.v1.engine": {"StreamAdmissionError"},
    "vllm.v1.engine.exceptions": {"EngineDeadError"},
}


def design_table(repo_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Return the design table's rows as ``(module, names)``; ``names`` is
    empty for a row that permits the whole module."""
    text = (repo_root / DESIGN).read_text()
    section = text.split("## Model-adapter dependencies", 1)[1].split("\n## ", 1)[0]
    rows: list[tuple[str, tuple[str, ...]]] = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or not cells[0].startswith("`"):
            continue
        module = re.findall(r"`([^`]+)`", cells[0])[0]
        names = tuple(n for n in re.findall(r"`([^`]+)`", cells[1]) if n.isidentifier())
        rows.append((module, names))
    assert rows, "the design has no dependency table"
    return rows


def unconditional_rows(repo_root: Path) -> list[tuple[str, str]]:
    pairs = []
    for module, names in design_table(repo_root):
        if module == BATCH_INVARIANT:
            continue
        for name in names:
            if (module, name) not in GRAPH_RUNTIME:
                pairs.append((module, name))
    return pairs


def test_design_table_has_the_conditional_rows_of_reg_014_and_reg_015(repo_root):
    rows = design_table(repo_root)
    pairs = {(m, n) for m, names in rows for n in names}
    assert set(GRAPH_RUNTIME) <= pairs
    assert (BATCH_INVARIANT, ()) in rows


# --------------------------------------------------------------------------
# REG-016: static scan
# --------------------------------------------------------------------------


def _vllm_imports(path: Path) -> list[tuple[int, str, str | None]]:
    """``(line, module, name)`` for every ``vllm`` import in a file; ``name``
    is ``None`` for ``import vllm.x``. Includes imports inside functions."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "vllm" or node.module.startswith("vllm."):
                found.extend((node.lineno, node.module, a.name) for a in node.names)
        elif isinstance(node, ast.Import):
            found.extend(
                (node.lineno, a.name, None)
                for a in node.names
                if a.name == "vllm" or a.name.startswith("vllm.")
            )
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None))
            == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).startswith("vllm.")
        ):
            found.append((node.lineno, node.args[0].value, None))
    return found


def _listed(table: dict[str, set[str] | None], module: str, name: str | None) -> bool:
    for listed, names in table.items():
        if listed.endswith(".") and (module + ".").startswith(listed):
            return True
        if module == listed and (names is None or name in names):
            return True
        # ``from vllm.a import b`` of the listed module ``vllm.a.b``
        if name is not None and f"{module}.{name}" == listed and names is None:
            return True
    return False


def _resolve(module: str, name: str | None):
    target = importlib.import_module(module)
    if name is None:
        return target
    if not hasattr(target, name):
        return importlib.import_module(f"{module}.{name}")
    return getattr(target, name)


@pytest.mark.engine
# @spec REG-016
def test_package_uses_no_vllm_dependency_outside_the_design_table(repo_root):
    """Every ``vllm`` import in the package is a table entry (the same object,
    so a re-export of a table name counts as that name), a whole-module row,
    an interface the design excludes from the table, or, in the session and
    ``register()``'s client-path check, the public client surface."""
    rows = design_table(repo_root)
    permitted_objects = []
    whole_modules: dict[str, set[str] | None] = {}
    for module, names in rows:
        if not names:
            whole_modules[module] = None
            continue
        for name in names:
            try:
                permitted_objects.append(_resolve(module, name))
            except (ImportError, AttributeError):
                pass  # a table entry missing at this core is REG-010's concern
    root = repo_root / PACKAGE
    outside: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        client = relative.startswith(CLIENT_SURFACE_FILES)
        for line, module, name in _vllm_imports(path):
            if _listed(EXCLUDED_MODULES, module, name) or _listed(
                whole_modules, module, name
            ):
                continue
            if client and _listed(CLIENT_SURFACE, module, name):
                continue
            try:
                obj = _resolve(module, name)
            except (ImportError, AttributeError):
                obj = None
            if obj is not None and any(obj is p for p in permitted_objects):
                continue
            outside.append(
                f"{relative}:{line}: {module}" + (f".{name}" if name else "")
            )
    assert not outside, (
        "vLLM dependencies outside the registration design's table:\n"
        + "\n".join(outside)
    )


# --------------------------------------------------------------------------
# Fresh-process helpers
# --------------------------------------------------------------------------

#: Imports the plugin's model module once, then lets a snippet remove names
#: from core and import it again. ``remove(module, name)`` removes an
#: attribute (``name=None`` blocks a whole module).
_REIMPORT = """
import importlib, sys, os
import vllm_nemotron_asr_streaming.model  # load core's modules once

def remove(module, name=None):
    if name is None:
        parent, _, child = module.rpartition(".")
        sys.modules[module] = None
        if parent in sys.modules and hasattr(sys.modules[parent], child):
            delattr(sys.modules[parent], child)
        return
    target = importlib.import_module(module)
    if module == "vllm.platforms" and name == "current_platform":
        original = target.__getattr__
        def lacking(attr, original=original):
            if attr == "current_platform":
                raise AttributeError(attr)
            return original(attr)
        target.__getattr__ = lacking
        target.__dict__.pop("current_platform", None)
        return
    if hasattr(target, name):
        delattr(target, name)

def forget_plugin():
    for key in [k for k in sys.modules if k.startswith("vllm_nemotron_asr_streaming")]:
        del sys.modules[key]

def import_model():
    try:
        importlib.import_module("vllm_nemotron_asr_streaming.model")
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return None, None
"""

#: Builds the model on CPU without weights, as core's loader constructs it.
#: ``configure`` runs before any name is removed (it loads core's
#: configuration and engine modules); ``construct`` runs after.
_CONSTRUCT = """
import tempfile
from pathlib import Path

def configure(overrides=None):
    from support.card import write_card
    from vllm.config import (
        CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig,
    )
    from vllm_nemotron_asr_streaming import register

    register()
    card = write_card(Path(tempfile.mkdtemp()) / "card")
    hf_overrides = {
        "num_lookahead_tokens": 6,
        "supported_num_lookahead_tokens": [0, 1, 3, 6, 13],
    }
    hf_overrides.update(overrides or {})
    model_config = ModelConfig(
        model=str(card),
        config_format="nemotron_asr_streaming",
        hf_overrides=hf_overrides,
        enforce_eager=True,
        dtype="float32",
        max_model_len=2048,
    )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=False, max_num_seqs=2
        ),
    )

def construct(vllm_config):
    from vllm.config.vllm import set_current_vllm_config
    from vllm_nemotron_asr_streaming.model import Nemotron3_5AsrStreamingForRNNT

    try:
        with set_current_vllm_config(vllm_config):
            Nemotron3_5AsrStreamingForRNNT(vllm_config=vllm_config, prefix="nemotron")
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return None, None
"""


def _tests_path_env(repo_root: Path) -> dict[str, str]:
    import os

    return {
        "PYTHONPATH": os.pathsep.join(
            [str(repo_root / "tests"), str(repo_root), os.environ.get("PYTHONPATH", "")]
        )
    }


# --------------------------------------------------------------------------
# REG-010: unconditional rows checked at model import
# --------------------------------------------------------------------------


@pytest.mark.engine
# @spec REG-010
def test_every_missing_unconditional_dependency_is_named_on_model_import(
    run_fresh, repo_root
):
    pairs = unconditional_rows(repo_root)
    out = run_fresh(
        _REIMPORT
        + f"""
PAIRS = {json.dumps(pairs)}
forget_plugin()
for module, name in PAIRS:
    remove(module, name)
raised, message = import_model()
emit(raised=raised, message=message)
"""
    )
    assert out["raised"], "model import accepted missing adapter dependencies"
    missing = [
        f"{module}.{name}"
        for module, name in pairs
        if module not in out["message"] or name not in out["message"]
    ]
    assert not missing, f"not named in the import error: {missing}"


@pytest.mark.engine
@pytest.mark.parametrize(
    "pair",
    [
        ("vllm.v1.kv_cache_interface", "PersistentStateSpec"),
        ("vllm.v1.worker.gpu.sample.output", "SamplerOutput"),
        ("vllm.utils.torch_utils", "direct_register_custom_op"),
    ],
    ids=lambda p: p[1],
)
# @spec REG-010
def test_one_missing_unconditional_dependency_fails_model_import_naming_it(
    run_fresh, pair
):
    module, name = pair
    out = run_fresh(
        _REIMPORT
        + f"""
forget_plugin()
remove({module!r}, {name!r})
raised, message = import_model()
emit(raised=raised, message=message)
"""
    )
    assert out["raised"], f"model import accepted a missing {module}.{name}"
    assert module in out["message"] and name in out["message"], out["message"]


# --------------------------------------------------------------------------
# REG-014: graph-runtime rows
# --------------------------------------------------------------------------


@pytest.mark.engine
# @spec REG-014
def test_missing_graph_runtime_does_not_prevent_model_import(run_fresh):
    out = run_fresh(
        _REIMPORT
        + f"""
forget_plugin()
for module, name in {json.dumps(GRAPH_RUNTIME)}:
    remove(module, name)
raised, message = import_model()
emit(raised=raised, message=message)
"""
    )
    assert out["raised"] is None, out["message"]


@pytest.mark.engine
# @spec REG-014
def test_missing_graph_runtime_does_not_prevent_eager_model_construction(
    run_fresh, repo_root
):
    """The card default (encoder ``eager``, decoder ``dense-eager``, no CHUNK
    populations) configures no graphed execution."""
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
vllm_config = configure()
for module, name in {json.dumps(GRAPH_RUNTIME)}:
    remove(module, name)
raised, message = construct(vllm_config)
emit(raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["raised"] is None, out["message"]


@pytest.mark.engine
@pytest.mark.parametrize("pair", GRAPH_RUNTIME, ids=lambda p: p[1])
# @spec REG-014
def test_missing_graph_runtime_fails_graphed_construction_naming_it(
    run_fresh, repo_root, pair
):
    module, name = pair
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
vllm_config = configure({{"decode_dispatch_arm": "dense-graphed"}})
remove({module!r}, {name!r})
raised, message = construct(vllm_config)
emit(raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["raised"], "graphed construction accepted a missing graph runtime"
    assert name in out["message"], out["message"]


# --------------------------------------------------------------------------
# REG-015: batch-invariant row
# --------------------------------------------------------------------------


@pytest.mark.engine
# @spec REG-015
def test_missing_batch_invariant_module_does_not_prevent_import_or_construction(
    run_fresh, repo_root
):
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
os.environ.pop("VLLM_BATCH_INVARIANT", None)
vllm_config = configure()
remove({BATCH_INVARIANT!r})
forget_plugin()
imported = import_model()
raised, message = construct(vllm_config)
emit(imported=imported, raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["imported"] == [None, None], out["imported"]
    assert out["raised"] is None, out["message"]


@pytest.mark.engine
# @spec REG-015
def test_missing_batch_invariant_module_fails_construction_when_configured(
    run_fresh, repo_root
):
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
import vllm.envs as envs
os.environ["VLLM_BATCH_INVARIANT"] = "1"
if hasattr(envs, "disable_envs_cache"):
    envs.disable_envs_cache()
vllm_config = configure()
remove({BATCH_INVARIANT!r})
raised, message = construct(vllm_config)
emit(raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["raised"], "construction accepted a missing batch-invariant module"
    assert BATCH_INVARIANT in out["message"] or "batch_invariant" in out["message"], (
        out["message"]
    )
