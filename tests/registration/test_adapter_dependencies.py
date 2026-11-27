# SPDX-License-Identifier: Apache-2.0
"""The model adapter's vLLM dependencies (REG-010, REG-014, REG-015, REG-016,
REG-018).

The permitted set is the registration design's dependency table
(``docs/intent/registration/registration-design.md``, "Model-adapter
dependencies"); the tests read it from there, so the design and the checks
cannot drift apart. Conditional rows are the graph runtime (REG-014), the
batch-invariant module (REG-015) and the transaction-graph members
(REG-018); every other row is unconditional (REG-010).

A table entry is a name of its module, or a dotted member path: the
``ModelState`` hooks and members are ``ModelState.<member>`` (the
transaction-graph row is the property ``ModelState.state_group_layouts``), the
``PERSISTENT_STATE`` member is ``MambaAttentionBackendEnum.PERSISTENT_STATE``,
the ``recomputable`` parameter is ``MambaManager.layer_config.recomputable``,
and the layout attributes read through the accessor are
``StateGroupLayout.layers`` and ``StateComponentLayout.tensor``.
``engine_flags.ADAPTER_DEPENDENCIES`` lists the unconditional entries in that
form.

Absence of a dependency is simulated in a fresh interpreter: the plugin's
modules are imported once (so core's own modules are loaded and keep the
names they bound), then the plugin's modules are dropped from
``sys.modules``, the names are removed from core's modules (a lazily resolved
module attribute, such as ``vllm.envs``' variables, is made to raise; a class
member is deleted; an enum member is removed by standing in an enum without
it; an argument is removed by standing in a callable whose signature lacks
it), and the plugin is imported again. Only the plugin's
own imports can then notice.
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
    ("vllm.compilation.cudagraph_pool", "capture_pool"),
    ("vllm.envs", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
)

#: REG-015: the conditional batch-invariant dependency (a whole module).
BATCH_INVARIANT = "vllm.model_executor.determinism.batch_invariant"

#: REG-010: the members and the argument the unconditional rows name.
HOOK_MODULE = "vllm.v1.worker.gpu.model_states.interface"
HOOKS = (
    "prepare_execution_memory",
    "needs_cudagraph_capture",
    "capture_cudagraphs",
    "clear_cudagraphs",
    "post_warmup",
    "shutdown",
)
#: The other ``ModelState`` members the unconditional row names.
MODEL_STATE_MEMBERS = ("get_additional_cg_support", "take_failed_requests")
ARGUMENT = (
    "vllm.v1.core.single_type_kv_cache_manager",
    "MambaManager.layer_config.recomputable",
)
PERSISTENT_STATE = (
    "vllm.v1.attention.backends.registry",
    "MambaAttentionBackendEnum.PERSISTENT_STATE",
)
SUPPORTS_PERSISTENT_STATE = (
    "vllm.model_executor.models.interfaces",
    "SupportsPersistentState",
)

#: REG-018: the conditional transaction-graph dependency, the accessor
#: property, checked at model construction while transaction graphs are on.
TRANSACTION_GRAPH = ((HOOK_MODULE, "ModelState.state_group_layouts"),)
#: REG-018: the attributes of the layouts the accessor returns, read without
#: importing their classes (so no import or class member can be removed).
LAYOUT_ATTRIBUTES = (
    ("vllm.v1.worker.utils", "StateGroupLayout.layers"),
    ("vllm.v1.worker.utils", "StateComponentLayout.tensor"),
)

#: The interfaces the design excludes from the table ("vLLM's
#: model-registration, config parser, tokenizer and renderer registries, model
#: configuration hooks, speech-to-text configuration, multimodal processing,
#: ``SupportsTranscription``, logging and Hub file helpers"), narrowed to the
#: modules and names the package uses under each.
EXCLUDED = {
    # model registration
    "vllm.model_executor.models": {"ModelRegistry"},
    # config parser registry
    "vllm.transformers_utils.config": {
        "register_config_parser",
        "_CONFIG_FORMAT_TO_CONFIG_PARSER",
    },
    "vllm.transformers_utils.config_parser_base": {"ConfigParserBase"},
    # tokenizer registry, and the tokenizer type registered in it
    "vllm.tokenizers.registry": {"TokenizerRegistry"},
    "vllm.tokenizers.hf": {"CachedHfTokenizer", "get_cached_tokenizer"},
    # renderer registry, and the standard renderer registered in it
    "vllm.renderers.registry": {"RENDERER_REGISTRY"},
    "vllm.renderers.hf": {"HfRenderer"},
    # model configuration hooks
    "vllm.model_executor.models.config": {"VerifyAndUpdateConfig", "MODELS_CONFIG_MAP"},
    # speech-to-text configuration
    "vllm.config.speech_to_text": {"SpeechToTextConfig"},
    # multimodal processing
    "vllm.config.multimodal": {"MultiModalDummyOptions"},
    "vllm.inputs": {"MultiModalDataDict"},
    "vllm.multimodal.inputs": {"MultiModalFieldConfig", "MultiModalKwargsItems"},
    "vllm.multimodal.parse": {"MultiModalDataItems"},
    "vllm.multimodal.processing": {
        "BaseProcessingInfo",
        "ProcessorInputs",
        "PromptUpdate",
        "TimingContext",
    },
    "vllm.multimodal.processing.dummy_inputs": {"BaseDummyInputsBuilder"},
    "vllm.multimodal.processing.processor": {
        "BaseMultiModalProcessor",
        "MultiModalProcessingResult",
        "PlaceholderFeaturesInfo",
    },
    # SupportsTranscription
    "vllm.model_executor.models.interfaces": {"SupportsTranscription"},
    # logging
    "vllm.logger": {"init_logger"},
    # Hub file helpers
    "vllm.transformers_utils.repo_utils": {"get_hf_file_to_dict"},
}

#: The public client surface the session and ``register()`` use (design:
#: ``AsyncLLM``, ``StreamingInput``, ``SamplingParams``, ``RequestOutputKind``,
#: and ``StreamAdmissionError`` and ``EngineDeadError`` from
#: ``vllm.v1.engine.exceptions``); outside the adapter, so only these files
#: may use it.
CLIENT_SURFACE_FILES = ("plugin.py", "session/")
CLIENT_SURFACE = {
    "vllm.v1.engine.async_llm": {"AsyncLLM"},
    "vllm.engine.protocol": {"StreamingInput"},
    "vllm.sampling_params": {"RequestOutputKind", "SamplingParams"},
    "vllm.v1.engine.exceptions": {"StreamAdmissionError", "EngineDeadError"},
}


def _cell_names(cell: str) -> tuple[str, ...]:
    """Backticked identifiers (dotted paths allowed) of a table cell. Names
    after "with its" are members of the name before it (``ModelState``, with
    its hooks ...; ``MambaAttentionBackendEnum``, with its member ...), and
    "with the `X` parameter of its `Y`" names the parameter ``Owner.Y.X``."""

    def names(text: str) -> list[str]:
        return [
            n
            for n in re.findall(r"`([^`]+)`", text)
            if all(part.isidentifier() for part in n.split("."))
        ]

    parameter = re.search(r"with the `(\w+)` parameter of its `(\w+)`", cell)
    if parameter is not None:
        owners = names(cell[: parameter.start()])
        assert owners, f"a parameter without an owner in: {cell}"
        argument, method = parameter.groups()
        return (*owners, f"{owners[-1]}.{method}.{argument}")
    head, _, members = cell.partition("with its")
    owners = names(head)
    result = list(owners)
    if members:
        assert owners, f"members without an owner in: {cell}"
        owner_part = members.split(")", 1)
        member_names = names(owner_part[0])
        result.extend(f"{owners[-1]}.{m}" for m in member_names)
        if len(owner_part) > 1:
            result.extend(names(owner_part[1]))
    return tuple(result)


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
        rows.append((module, _cell_names(cells[1])))
    assert rows, "the design has no dependency table"
    return rows


CONDITIONAL = set(GRAPH_RUNTIME) | set(TRANSACTION_GRAPH) | set(LAYOUT_ATTRIBUTES)


def unconditional_rows(repo_root: Path) -> list[tuple[str, str]]:
    pairs = []
    for module, names in design_table(repo_root):
        if module == BATCH_INVARIANT:
            continue
        for name in names:
            if (module, name) not in CONDITIONAL:
                pairs.append((module, name))
    return pairs


def test_design_table_has_the_rows_the_tests_assume(repo_root):
    """Self-check of the table reading: the conditional rows of REG-014,
    REG-015 and REG-018, and the hooks and argument of REG-010."""
    rows = design_table(repo_root)
    pairs = {(m, n) for m, names in rows for n in names}
    assert set(GRAPH_RUNTIME) <= pairs
    assert set(TRANSACTION_GRAPH) <= pairs
    assert set(LAYOUT_ATTRIBUTES) <= pairs
    assert (BATCH_INVARIANT, ()) in rows
    assert {(HOOK_MODULE, f"ModelState.{h}") for h in HOOKS} <= pairs
    assert {(HOOK_MODULE, f"ModelState.{m}") for m in MODEL_STATE_MEMBERS} <= pairs
    assert {ARGUMENT, PERSISTENT_STATE, SUPPORTS_PERSISTENT_STATE} <= pairs
    assert all(names for module, names in rows if module != BATCH_INVARIANT), rows


# --------------------------------------------------------------------------
# REG-016: static scan
# --------------------------------------------------------------------------


def _is_module(path: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(path) is not None
    except (ImportError, ValueError):
        return False


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


def _literal_strings(node: ast.AST) -> list[str]:
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return []
    if isinstance(value, dict):
        value = list(value)
    if isinstance(value, str):
        return [value]
    return (
        [v for v in value if isinstance(v, str)]
        if isinstance(value, (list, tuple, set))
        else []
    )


class Scanner:
    """Every ``vllm`` dependency of one source file, as ``(line, module,
    name)``; ``name`` is ``None`` for a module used as a whole.

    Covers ``import`` / ``from ... import`` (aliases included), members
    reached through a module alias (``alias.member``), ``importlib.
    import_module`` and ``__import__`` (aliased or not; a non-constant
    argument is resolved over the module-level literal table its loop
    variable iterates, and is reported as unresolved otherwise), and string
    constants naming an importable ``vllm`` module (as an import target,
    with the identifier that follows it in a tuple, or the names of a dict
    value). Also: ``__import__`` reached through ``builtins`` (imported or
    aliased), importer functions rebound by assignment (``load =
    importlib.import_module``), ``getattr(alias, "X")`` on a module alias, and
    every string constant ``vllm(.name)+`` whose longest importable prefix is
    a ``vllm`` submodule, read as ``(module, remainder)`` (so qualnames such
    as ``resolve_obj_by_qualname`` targets and ``sys.modules`` keys count; a
    label like ``"vllm.matmul_kernel_persistent"``, with no submodule prefix,
    does not)."""

    def __init__(self, source: str) -> None:
        self.tree = ast.parse(source)
        self.found: list[tuple[int, str, str | None]] = []
        self.aliases: dict[str, str] = {}  # local name -> vllm module path
        self.importers: set[str] = {"__import__"}
        self.importlib_aliases: set[str] = set()
        self.builtins_aliases: set[str] = set()
        self.tables = {
            target.id: node.value
            for node in self.tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name) and node.value is not None
        }

    def scan(self) -> list[tuple[int, str, str | None]]:
        for node in ast.walk(self.tree):
            self._imports(node)
        for _ in range(2):  # rebinding chains of length two
            for node in ast.walk(self.tree):
                self._rebinds(node)
        for node in ast.walk(self.tree):
            self._uses(node)
        return self.found

    def _is_importer(self, func: list[str]) -> bool:
        return (
            (len(func) == 1 and func[0] in self.importers)
            or (
                len(func) == 2
                and func[0] in self.importlib_aliases
                and func[1] == "import_module"
            )
            or (
                len(func) == 2
                and func[0] in self.builtins_aliases
                and func[1] == "__import__"
            )
        )

    def _rebinds(self, node: ast.AST) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = _dotted(node.value) or []
            name = node.targets[0].id
            if value and self._is_importer(value):
                self.importers.add(name)
            elif len(value) == 1 and value[0] in self.aliases:
                self.aliases[name] = self.aliases[value[0]]
            elif len(value) == 1 and value[0] in self.importlib_aliases:
                self.importlib_aliases.add(name)

    def _add(self, line: int, module: str, name: str | None) -> None:
        if name is not None and _is_module(f"{module}.{name}"):
            module, name = f"{module}.{name}", None
        self.found.append((line, module, name))

    def _imports(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    self.importlib_aliases.add(alias.asname or "importlib")
                if alias.name == "builtins":
                    self.builtins_aliases.add(alias.asname or "builtins")
                if alias.name == "vllm" or alias.name.startswith("vllm."):
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                    else:
                        self.aliases["vllm"] = "vllm"
                    self.found.append((node.lineno, alias.name, None))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        self.importers.add(alias.asname or alias.name)
            if node.module == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        self.importers.add(alias.asname or alias.name)
            if node.module == "vllm" or node.module.startswith("vllm."):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if _is_module(f"{node.module}.{alias.name}"):
                        self.aliases[local] = f"{node.module}.{alias.name}"
                        self.found.append(
                            (node.lineno, f"{node.module}.{alias.name}", None)
                        )
                    else:
                        self.found.append((node.lineno, node.module, alias.name))

    def _uses(self, node: ast.AST) -> None:
        if isinstance(node, ast.Attribute) and not isinstance(
            getattr(node, "_parent_attr", None), ast.Attribute
        ):
            parts = _dotted(node)
            if parts and parts[0] in self.aliases:
                path = self.aliases[parts[0]]
                for member in parts[1:]:
                    if _is_module(f"{path}.{member}"):
                        path = f"{path}.{member}"
                        continue
                    self.found.append((node.lineno, path, member))
                    break
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"vllm(\.\w+)+", node.value)
        ):
            parts = node.value.split(".")
            for k in range(len(parts), 1, -1):
                prefix = ".".join(parts[:k])
                if _is_module(prefix):
                    rest = parts[k] if k < len(parts) else None
                    self.found.append((node.lineno, prefix, rest))
                    break
        if isinstance(node, ast.Call):
            func = _dotted(node.func) or []
            if (
                func == ["getattr"]
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in self.aliases
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                self._add(
                    node.lineno, self.aliases[node.args[0].id], node.args[1].value
                )
            if self._is_importer(func) and node.args:
                for target in self._import_targets(node.args[0]):
                    if target is None:
                        self.found.append(
                            (node.lineno, "<unresolved dynamic import>", None)
                        )
                    elif target == "vllm" or target.startswith("vllm."):
                        self.found.append((node.lineno, target, None))
        if isinstance(node, ast.Tuple):
            values = [
                e.value if isinstance(e, ast.Constant) else None for e in node.elts
            ]
            for index, value in enumerate(values):
                if (
                    isinstance(value, str)
                    and value.startswith("vllm.")
                    and _is_module(value)
                ):
                    following = values[index + 1] if index + 1 < len(values) else None
                    name = (
                        following
                        if isinstance(following, str) and following.isidentifier()
                        else None
                    )
                    self.found.append((node.lineno, value, name))
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.startswith("vllm.")
                    and _is_module(key.value)
                ):
                    names = [n for n in _literal_strings(value) if n.isidentifier()]
                    for name in names or [None]:
                        self.found.append((key.lineno, key.value, name))

    def _import_targets(self, arg: ast.AST) -> list[str | None]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return [arg.value]
        if isinstance(arg, ast.Name):
            for node in ast.walk(self.tree):
                if not isinstance(node, ast.For | ast.comprehension):
                    continue
                target = node.target
                names = (
                    [target]
                    if isinstance(target, ast.Name)
                    else list(target.elts)
                    if isinstance(target, ast.Tuple)
                    else []
                )
                if (
                    not names
                    or not isinstance(names[0], ast.Name)
                    or names[0].id != arg.id
                ):
                    continue
                source = node.iter
                if isinstance(source, ast.Call) and isinstance(
                    source.func, ast.Attribute
                ):
                    source = source.func.value
                if isinstance(source, ast.Name) and source.id in self.tables:
                    strings = _literal_strings(self.tables[source.id])
                    if strings:
                        return list(strings)
        return [None]


def scan_package(root: Path) -> list[tuple[str, int, str, str | None]]:
    found = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for line, module, name in Scanner(path.read_text()).scan():
            found.append((relative, line, module, name))
    return found


def _listed(table: dict[str, set[str] | None], module: str, name: str | None) -> bool:
    names = table.get(module, False)
    if names is False:
        return False
    return names is None or name in names


def _resolve(module: str, name: str | None):
    target = importlib.import_module(module)
    if name is None:
        return target
    for part in name.split("."):
        target = getattr(target, part)
    return target


def outside_table(repo_root: Path, root: Path) -> list[str]:
    """Dependencies of the package under ``root`` that are neither a table
    entry (by identity, so a re-export of a table name counts as that name),
    a whole-module row, an excluded interface, nor (in the client files) the
    public client surface. A module imported only to reach its members is
    judged by the members used through it."""
    permitted_objects: list = []
    whole_modules: dict[str, set[str] | None] = {}
    for module, names in design_table(repo_root):
        if not names:
            whole_modules[module] = None
        for name in names:
            if (module, name) == ARGUMENT or (module, name) in LAYOUT_ATTRIBUTES:
                continue  # an argument or instance attribute, not an export
            try:
                permitted_objects.append(_resolve(module, name))
            except (ImportError, AttributeError):
                pass  # missing at this core: REG-010's concern
    member_modules = set()
    found = scan_package(root)
    for _, _, module, name in found:
        if name is not None:
            member_modules.add(module)
    outside = []
    for relative, line, module, name in found:
        client = relative.startswith(CLIENT_SURFACE_FILES)
        if name is None and module in member_modules and module not in whole_modules:
            continue  # an alias import; its member uses are judged instead
        if _listed(EXCLUDED, module, name) or _listed(whole_modules, module, name):
            continue
        if client and _listed(CLIENT_SURFACE, module, name):
            continue
        try:
            obj = _resolve(module, name)
        except (ImportError, AttributeError):
            obj = None
        if (
            obj is not None
            and name is not None
            and any(obj is p for p in permitted_objects)
        ):
            continue
        outside.append(f"{relative}:{line}: {module}" + (f".{name}" if name else ""))
    return outside


@pytest.mark.engine
# @spec REG-016
def test_package_uses_no_vllm_dependency_outside_the_design_table(repo_root):
    outside = outside_table(repo_root, repo_root / PACKAGE)
    assert not outside, (
        "vLLM dependencies outside the registration design's table:\n"
        + "\n".join(outside)
    )


@pytest.mark.engine
# @spec REG-010, REG-016
def test_import_time_check_lists_exactly_the_unconditional_rows(repo_root):
    from vllm_nemotron_asr_streaming.engine_flags import ADAPTER_DEPENDENCIES

    expected: dict[str, set[str]] = {}
    for module, name in unconditional_rows(repo_root):
        expected.setdefault(module, set()).add(name)
    actual = {m: set(names) for m, names in ADAPTER_DEPENDENCIES.items()}
    assert actual == expected


SELF_TEST_CASES = {
    "allowed-aliased": (
        "import vllm.v1.worker.gpu.states as states\n"
        "from vllm.v1.kv_cache_interface import MambaSpec as Spec\n"
        "def f():\n    return states.RequestState, Spec\n",
        False,
    ),
    "unlisted-direct": ("from vllm.v1.outputs import ModelRunnerOutput\n", True),
    "unlisted-aliased": (
        "import vllm.v1.outputs as out\ndef f():\n    return out.ModelRunnerOutput\n",
        True,
    ),
    "unlisted-dynamic": (
        "import importlib as il\n"
        "def f():\n    return il.import_module('vllm.v1.outputs')\n",
        True,
    ),
    "unlisted-dunder-import": (
        "def f():\n    return __import__('vllm.v1.outputs')\n",
        True,
    ),
    "unresolved-dynamic": (
        "from importlib import import_module as load\n"
        "def f(name):\n    return load(name)\n",
        True,
    ),
    "string-target": ("TARGET = ('vllm.v1.outputs', 'ModelRunnerOutput')\n", True),
    "builtins-import-alias": (
        "from builtins import __import__ as load\n"
        "def f():\n    return load('vllm.v1.outputs')\n",
        True,
    ),
    "builtins-module-import": (
        "import builtins\n"
        "def f():\n    return builtins.__import__('vllm.v1.outputs')\n",
        True,
    ),
    "rebound-import-module": (
        "import importlib\nload = importlib.import_module\n"
        "def f():\n    return load('vllm.v1.outputs')\n",
        True,
    ),
    "qualname-string": (
        "from vllm.utils.import_utils import resolve_obj_by_qualname\n"
        "def f():\n"
        "    return resolve_obj_by_qualname('vllm.v1.outputs.ModelRunnerOutput')\n",
        True,
    ),
    "sys-modules-key": (
        "import sys\ndef f():\n    return sys.modules['vllm.v1.outputs']\n",
        True,
    ),
    "getattr-on-alias": (
        "import vllm.v1.outputs as out\n"
        "def f():\n    return getattr(out, 'ModelRunnerOutput')\n",
        True,
    ),
    "kernel-label-string": (
        "KERNEL = 'vllm.matmul_kernel_persistent'\n",
        False,
    ),
    "table-driven-dynamic": (
        "import importlib\n"
        "TABLE = {'vllm.v1.outputs': ('ModelRunnerOutput',)}\n"
        "def f():\n"
        "    for module_name, names in TABLE.items():\n"
        "        importlib.import_module(module_name)\n",
        True,
    ),
}


@pytest.mark.engine
@pytest.mark.parametrize("case", list(SELF_TEST_CASES))
# @spec REG-016
def test_scanner_flags_exactly_the_unlisted_dependencies(repo_root, tmp_path, case):
    source, flagged = SELF_TEST_CASES[case]
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(source)
    outside = outside_table(repo_root, tmp_path / "pkg")
    assert bool(outside) is flagged, outside


# --------------------------------------------------------------------------
# Fresh-process helpers
# --------------------------------------------------------------------------

#: Imports the plugin's model module once, then lets a snippet remove names
#: from core and import it again. ``remove(module, name)`` removes an
#: attribute (``name=None`` blocks a whole module).
_REIMPORT = """
import importlib, inspect, sys, os
import vllm_nemotron_asr_streaming.model  # load core's modules once

def _make_lacking(target, name):
    # A lazily resolved module attribute (PEP 562 ``__getattr__``, as
    # ``vllm.platforms.current_platform`` and ``vllm.envs`` variables are):
    # make the module's ``__getattr__`` raise for it.
    original = target.__dict__.get("__getattr__")
    def lacking(attr, original=original):
        if attr == name:
            raise AttributeError(attr)
        if original is None:
            raise AttributeError(attr)
        return original(attr)
    target.__getattr__ = lacking
    target.__dict__.pop(name, None)

def _strip_argument(target, class_name, argument):
    # Stand in a class whose constructor lacks ``argument`` (a core that
    # predates it); every other behaviour is the original class's.
    original = getattr(target, class_name, None)
    if original is None:
        return
    signature = inspect.signature(original)
    kept = [p for p in signature.parameters.values() if p.name != argument]
    def __init__(self, *args, **kwargs):
        if argument in kwargs:
            raise TypeError(f"unexpected keyword argument {argument!r}")
        self.__dict__["_inner"] = original(*args, **kwargs)
    def __getattr__(self, attr):
        if attr == argument:
            raise AttributeError(attr)
        return getattr(self.__dict__["_inner"], attr)
    stand_in = type(class_name, (), {"__init__": __init__, "__getattr__": __getattr__})
    stand_in.__signature__ = signature.replace(parameters=kept)
    stand_in.__module__ = original.__module__
    setattr(target, class_name, stand_in)

def remove(module, name=None):
    if name is None:
        parent, _, child = module.rpartition(".")
        sys.modules[module] = None
        if parent in sys.modules and hasattr(sys.modules[parent], child):
            delattr(sys.modules[parent], child)
        return
    target = importlib.import_module(module)
    if name.count(".") == 2:
        # ``Owner.method.argument``: stand in the method without the argument.
        class_name, method, argument = name.split(".")
        owner = getattr(target, class_name, None)
        bound = getattr(owner, method, None)
        if bound is None:
            return
        signature = inspect.signature(bound)
        kept = [p for p in signature.parameters.values() if p.name != argument]
        def without(cls, *args, **kwargs):
            if argument in kwargs:
                raise TypeError(f"unexpected keyword argument {argument!r}")
            return bound(*args, **kwargs)
        without.__signature__ = signature.replace(parameters=kept)
        setattr(owner, method, classmethod(without))
        return
    if "." in name:
        class_name, member = name.split(".", 1)
        owner = getattr(target, class_name, None)
        if owner is None:
            return
        import enum
        if isinstance(owner, enum.EnumMeta) and member in owner.__members__:
            stand_in = enum.Enum(
                owner.__name__,
                {m.name: m.value for m in owner if m.name != member},
            )
            stand_in.__module__ = owner.__module__
            setattr(target, class_name, stand_in)
            return
        if member in vars(owner) and callable(vars(owner)[member]):
            delattr(owner, member)
        elif member in inspect.signature(owner).parameters:
            _strip_argument(target, class_name, member)
        elif hasattr(owner, member):
            delattr(owner, member)
        return
    if name in target.__dict__:
        delattr(target, name)
    elif hasattr(target, name):
        _make_lacking(target, name)

def forget_plugin():
    for key in [k for k in sys.modules if k.startswith("vllm_nemotron_asr_streaming")]:
        del sys.modules[key]

def import_model():
    try:
        importlib.import_module("vllm_nemotron_asr_streaming.model")
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return None, None

def removed(module, name):
    # Whether ``remove`` took effect (a self-check of the helper).
    try:
        target = importlib.import_module(module)
    except ImportError:
        return True
    obj = target
    parts = name.split(".")
    for index, part in enumerate(parts):
        if index == len(parts) - 1 and index > 0:
            if part in inspect.signature(obj).parameters:
                return False
        if not hasattr(obj, part):
            return True
        obj = getattr(obj, part)
    return False
"""

#: Builds the model on CPU without weights, as core's loader constructs it.
#: ``configure`` runs before any name is removed (it loads core's
#: configuration and engine modules); ``construct`` runs after, with the
#: ``state_name`` blocker isolated. The base configuration selects the eager
#: arms (no graphed execution); ``overrides`` select others.
_CONSTRUCT = """
import tempfile
from pathlib import Path

EAGER = {"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"}

def configure(overrides=None, after=None):
    from support.card import write_card
    from vllm.config import (
        CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig,
    )
    from vllm_nemotron_asr_streaming import register

    register()
    card = write_card(Path(tempfile.mkdtemp()) / "card")
    hf_overrides = {"num_lookahead_tokens": 0, **EAGER}
    hf_overrides.update(overrides or {})
    model_config = ModelConfig(
        model=str(card),
        config_format="nemotron_asr_streaming",
        hf_overrides=hf_overrides,
        enforce_eager=True,
        dtype="float32",
        max_model_len=2048,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=False,
            max_num_seqs=2,
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=model_config.max_model_len,
            enable_chunked_prefill=False,
        ),
    )
    # Settings the engine hook would refuse under ``--enforce-eager`` (the
    # transaction-graph option) are applied after it ran, so construction
    # reaches the dependency gate.
    for key, value in (after or {}).items():
        setattr(vllm_config.model_config.hf_config, key, value)
    return vllm_config

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
        if module not in out["message"] or name.split(".")[-1] not in out["message"]
    ]
    assert not missing, f"not named in the import error: {missing}"


@pytest.mark.engine
# @spec REG-010, REG-014, REG-018
def test_the_removal_helper_removes_every_table_entry(run_fresh, repo_root):
    """Self-check: ``remove`` takes effect for every entry it is asked to
    remove, lazily resolved module attributes, class members and the
    argument included, so the import tests fail or pass for the dependency
    under test."""
    pairs = (
        unconditional_rows(repo_root) + list(GRAPH_RUNTIME) + list(TRANSACTION_GRAPH)
    )
    out = run_fresh(
        _REIMPORT
        + f"""
results = {{}}
for module, name in {json.dumps(pairs)}:
    remove(module, name)
    results[module + ":" + name] = removed(module, name)
emit(results=results)
"""
    )
    assert all(out["results"].values()), [k for k, v in out["results"].items() if not v]


@pytest.mark.engine
@pytest.mark.parametrize(
    "pair",
    [
        ("vllm.v1.kv_cache_interface", "MambaSpec"),
        ("vllm.v1.worker.gpu.sample.output", "SamplerOutput"),
        ("vllm.utils.torch_utils", "direct_register_custom_op"),
        *((HOOK_MODULE, f"ModelState.{hook}") for hook in HOOKS),
        *((HOOK_MODULE, f"ModelState.{member}") for member in MODEL_STATE_MEMBERS),
        ARGUMENT,
        PERSISTENT_STATE,
        SUPPORTS_PERSISTENT_STATE,
    ],
    ids=lambda p: p[1],
)
# @spec REG-010
def test_one_missing_unconditional_dependency_fails_model_import_naming_it(
    run_fresh, pair
):
    module, name = pair
    leaf = name.split(".")[-1]
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
    assert module in out["message"] and leaf in out["message"], out["message"]


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
    """The eager arms (encoder ``eager``, decoder ``dense-eager``, no CHUNK
    populations) configure no graphed execution."""
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


#: Configurations that configure graphed execution (registration design,
#: Segment arms): the default arms, each graphed arm alone, CHUNK buckets.
GRAPHED_CONFIGURATIONS = {
    "default arms": {
        "encoder_execution_arm": "eager-graphed",
        "decode_dispatch_arm": "dense-graphed",
    },
    "graphed decoder": {"decode_dispatch_arm": "dense-graphed"},
    "graphed encoder": {"encoder_execution_arm": "eager-graphed"},
}


@pytest.mark.engine
@pytest.mark.parametrize("configuration", list(GRAPHED_CONFIGURATIONS))
@pytest.mark.parametrize("pair", GRAPH_RUNTIME, ids=lambda p: p[1])
# @spec REG-014
def test_missing_graph_runtime_fails_graphed_construction_naming_it(
    run_fresh, repo_root, pair, configuration
):
    module, name = pair
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
vllm_config = configure({GRAPHED_CONFIGURATIONS[configuration]!r})
remove({module!r}, {name!r})
raised, message = construct(vllm_config)
emit(raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["raised"], "graphed construction accepted a missing graph runtime"
    assert name in out["message"], out["message"]


# --------------------------------------------------------------------------
# REG-018: the state-group layout accessor
# --------------------------------------------------------------------------


@pytest.mark.engine
@pytest.mark.parametrize(
    "after", [None, {"transaction_graph_populations": []}], ids=["absent", "empty"]
)
# @spec REG-018
def test_a_missing_accessor_prevents_neither_import_nor_construction_while_off(
    run_fresh, repo_root, after
):
    """Transaction graphs off (TXG-CFG-001): model import does not check
    ``ModelState.state_group_layouts``, and the default arms construct
    without it."""
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
vllm_config = configure({GRAPHED_CONFIGURATIONS["default arms"]!r}, after={after!r})
for module, name in {json.dumps(TRANSACTION_GRAPH)}:
    remove(module, name)
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
@pytest.mark.parametrize("value", [[2], {}], ids=["list", "malformed-on"])
# @spec REG-018
def test_a_missing_accessor_fails_construction_while_on_naming_it(
    run_fresh, repo_root, value
):
    """Transaction graphs on (TXG-CFG-001; ``{}`` is on): construction
    raises naming ``ModelState.state_group_layouts``."""
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
vllm_config = configure(
    {GRAPHED_CONFIGURATIONS["default arms"]!r},
    after={{"transaction_graph_populations": {value!r}}},
)
# The option runs under the hook configuration only; leave the dependency
# gate as the first thing construction can refuse.
vllm_config.model_config.enforce_eager = False
for module, name in {json.dumps(TRANSACTION_GRAPH)}:
    remove(module, name)
raised, message = construct(vllm_config)
emit(raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["raised"], "construction accepted a missing state-group layout accessor"
    assert "ModelState.state_group_layouts" in out["message"], out["message"]


@pytest.mark.engine
# @spec REG-018
def test_the_accessor_counts_as_defined_as_a_property(run_fresh, repo_root):
    """On the target core ``state_group_layouts`` is a property, not a
    callable; with transaction graphs on, construction does not refuse it as
    missing."""
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
from vllm.v1.worker.gpu.model_states.interface import ModelState
is_property = isinstance(vars(ModelState)["state_group_layouts"], property)
vllm_config = configure(
    {GRAPHED_CONFIGURATIONS["default arms"]!r},
    after={{"transaction_graph_populations": [2]}},
)
vllm_config.model_config.enforce_eager = False
raised, message = construct(vllm_config)
emit(is_property=is_property, raised=raised, message=message)
""",
        env=_tests_path_env(repo_root),
    )
    assert out["is_property"]
    assert "state_group_layouts" not in (out["message"] or ""), out["message"]


# --------------------------------------------------------------------------
# REG-015: batch-invariant row
# --------------------------------------------------------------------------


@pytest.mark.engine
@pytest.mark.parametrize("setting", [None, "0"], ids=["unset", "0"])
# @spec REG-015
def test_missing_batch_invariant_module_does_not_prevent_import_or_construction(
    run_fresh, repo_root, setting
):
    """Batch-invariant mode is not configured: ``VLLM_BATCH_INVARIANT`` unset
    or ``0``."""
    out = run_fresh(
        _REIMPORT
        + _CONSTRUCT
        + f"""
import vllm.envs as envs
os.environ.pop("VLLM_BATCH_INVARIANT", None)
if {setting!r} is not None:
    os.environ["VLLM_BATCH_INVARIANT"] = {setting!r}
if hasattr(envs, "disable_envs_cache"):
    envs.disable_envs_cache()
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
