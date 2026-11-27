# SPDX-License-Identifier: Apache-2.0
"""Builders and record accessors for the capture, lifecycle, memory,
verification and failure tests (TXG-CAP, -LIFE, -MEM, -VER, -FAIL).

Two ways to hold a model state on CPU (the CUDA environment is
``support.txg_gpu.build_option_on``):

* ``build_off``: the real model and model state on CPU with transaction
  graphs off, as ``tests/registration/test_execution_memory.py`` builds them
  (``support.model_harness``), for the specs that hold with the option on or
  off (TXG-FAIL-001, -007, -008, -009).
* ``build_on_cpu``: the same real classes on CPU with transaction graphs on,
  through a *stated fake-capture seam*: core's CUDA surface is
  ``support.fake_cuda.FakeCuda`` (graphs, streams, memory readings,
  ``capture_pool``, the CUDA platform claim), segment graphs are
  ``model_harness.FakeGraphRuntime``. A fake cell's replay runs nothing, so
  these environments test ordering, inventory, accounting, messages and
  latches, never a covered step's outputs (those are ``gpu`` tests).
  The option is set on the final configuration after core's configuration
  hooks ran (they refuse a CPU platform and ``--enforce-eager``), as an
  already-translated configuration stores it (TXG-CFG-001); the runner is
  then core's hook configuration (``enforce_eager`` off, ``cudagraph_mode``
  NONE, compilation off).
* ``on_cpu`` pairs an option-on environment with its
  ``support.core_order.CoreOrder`` (``core_of``, a :class:`TxgCoreOrder`),
  which calls the model state's hooks in core's order, one call at a time.

Every environment built here follows core's state-group layout accessor
(:func:`follow_state_group_layouts`): when core binds the state layer's
views it sets ``ModelState._state_group_layouts`` from them before the model
state's ``initialize_kv_cache``, and core's release of the profiling KV cache
(after ``clear_cudagraphs``) resets it to ``()``. Transaction graphs read the
state pools only through that accessor during a capture call (TXG-CAP-008,
TXG-LIFE-003).

Failed requests reach core through the model state's
``take_failed_requests`` (REG-MS-003, REG-MS-004, TXG-FAIL-008):
:class:`FailSpy` records every set it returns.

The reader, the binding, the counter names, the capture-record keys
(``txg_api.Record``) and the fallback key fields
(``txg_api.FALLBACK_KEY_FIELDS``) are ``support.txg_api``'s.
"""

from __future__ import annotations

import gc
import importlib
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from support import model_harness as h
from support import txg_api
from support.core_order import CoreOrder
from support.fake_cuda import MARGIN_BYTES, FakeCuda, FakeStream
from support.txg_api import Sizing, sizing_settings

OPTION = "transaction_graph_populations"
LOOKAHEADS = (0, 1, 3, 6, 13)
#: The default segment arms, which transaction graphs require (TXG-CFG-008).
DEFAULT_ARMS = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
}
FRESH_ENGINE = re.compile(r"fresh engine", re.IGNORECASE)
#: The page count of every serving state pool a CPU environment's
#: ``CoreOrder`` binds: the reservation capacity under the extensible KV
#: cache, and ``num_gpu_blocks_override`` under a fixed block count.
CAPACITY = 8


# ---------------------------------------------------------------------------
# The reader and the record


def counters(obj: Any) -> Any:
    """``transaction_graph_counters(obj)`` (TXG-OBS-002) through the shared
    interface module ``support.txg_api``."""
    return txg_api.counters(obj)


def counter(env: Any, name: str) -> Any:
    """One counter of TXG-OBS-001, named as ``support.txg_api.COUNTER_KEYS``
    names it."""
    assert name in txg_api.COUNTER_KEYS, name
    return counters(env.model)["counters"][name]


def record(env: Any) -> dict[str, Any]:
    return counters(env.model)[txg_api.Record.RECORD]


def cell_order(env: Any, capture: str) -> list[tuple[str, int]]:
    cells = record(env)[capture][txg_api.Record.CELLS]
    return [(c["kind"], int(c["count"])) for c in cells]


def largest_first(n: int) -> list[tuple[str, int]]:
    """TXG-CAP-007: chunk cells from ``N`` down to 1, then echo cells."""
    return [("chunk", k) for k in range(n, 0, -1)] + [
        ("echo", k) for k in range(n, 0, -1)
    ]


# ---------------------------------------------------------------------------
# Messages


def chain(error: BaseException) -> list[BaseException]:
    """The error and every exception it was raised from or during."""
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def text(error: BaseException) -> str:
    """Every message in the error's chain, and any notes core added."""
    parts = []
    for item in chain(error):
        parts.append(str(item))
        parts.extend(getattr(item, "__notes__", ()) or ())
    return "\n".join(parts)


def mentions_bytes(message: str, value: int) -> bool:
    """Whether ``message`` states ``value`` bytes: as an integer (with or
    without thousands separators) or in KiB/MiB/GiB with up to two decimals."""
    candidates = {str(value), f"{value:,}", f"{value:_}"}
    for unit, scale in (("KiB", 1 << 10), ("MiB", 1 << 20), ("GiB", 1 << 30)):
        amount = value / scale
        renderings = [f"{amount:.1f}", f"{amount:.2f}"]
        if amount == int(amount):
            renderings.append(f"{int(amount)}")
        for rendering in renderings:
            candidates.update({f"{rendering} {unit}", f"{rendering}{unit}"})
    return any(re.search(rf"(?<![\d.]){re.escape(c)}", message) for c in candidates)


# ---------------------------------------------------------------------------
# Rows


def geometry_of(right_context: int) -> int:
    return LOOKAHEADS.index(right_context)


def stream_unit(
    env: Any,
    req_id: str,
    *,
    geometry: int | None = None,
    sequence: int = 0,
    final: bool = False,
    valid_samples: int | None = None,
) -> Any:
    """A stream unit at the server's geometry (or ``geometry``), its header
    in ``manifests.ENVELOPE_HEADER_FIELDS`` order
    (``model_harness.stream_envelope``). ``final`` marks the final
    tail, whose valid samples default to half a chunk (a valid tail is
    shorter than a full chunk); a continuing unit carries a full chunk.
    The header is checked field by field before the unit is used."""
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        CADENCES,
        ENVELOPE_HEADER_FIELDS,
        RAW_SAMPLES_PER_CHUNK,
    )

    geometry = env.geometry if geometry is None else geometry
    width = RAW_SAMPLES_PER_CHUNK[tuple(CADENCES)[geometry]]
    if valid_samples is None:
        valid_samples = width // 2 if final else width
    row = h.stream_envelope(
        geometry=geometry,
        sequence=sequence,
        final=final,
        valid_samples=valid_samples,
    )
    header = dict(zip(ENVELOPE_HEADER_FIELDS, (int(v) for v in row[:7]), strict=True))
    assert header["geometry_id"] == geometry, header
    assert header["final_tail"] == int(final), header
    assert header["chunk_sequence"] == sequence, header
    assert header["valid_samples"] == valid_samples, header
    if final:
        assert valid_samples < width, "a valid final tail is shorter than a chunk"
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=row,
        identifier=f"uuid-{req_id}-{sequence}",
    )


def file_item(
    env: Any,
    req_id: str,
    *,
    n_samples: int = 4000,
    geometry: int | None = None,
    header_samples: int | None = None,
) -> Any:
    """A whole-file request; ``geometry`` other than the server's makes a
    planning failure (FILE-PROMPT-007), ``header_samples`` other than
    ``n_samples`` a malformed item."""
    geometry = env.geometry if geometry is None else geometry
    samples = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
    return h.new_request(
        req_id,
        int(env.config.file_carrier_token_id),
        samples=samples * 0.1,
        header=[geometry, 0, n_samples if header_samples is None else header_samples],
    )


def foreign_geometry(env: Any) -> int:
    return (env.geometry + 1) % len(LOOKAHEADS)


def with_endpoint_policy(request: Any, *, threshold_frames: int = 2) -> Any:
    """``request`` carrying an endpoint policy in its sampling parameters'
    ``extra_args`` (REG-STATE-020), as only a direct engine stream client sets
    one (SESS-EP-001): its row then carries endpoint mode 1 in the row plan
    (TXG-COV-012)."""
    extra_args = dict(request.sampling_params.extra_args or {})
    extra_args["endpoint_policy"] = {
        "mode": "greedy_blank",
        "threshold_frames": threshold_frames,
        "residue_frames": 0,
    }
    return h.with_extra_args(request, extra_args)


def eou_request(env: Any, req_id: str) -> Any:
    """A new request with no audio whose scheduled token is the EOU token, as
    only a direct engine stream client schedules it, for a stream the worker
    already serves (the planner refuses a non-CHUNK row of a request it never
    saw a CHUNK of): a non-CHUNK row that is a forced-EOU candidate
    (TXG-COV-012)."""
    return h.new_request(req_id, int(env.config.eou_token_id))


def fallback_reasons(env: Any) -> dict[tuple[Any, ...], int]:
    """The ``fallbacks`` counter, keyed ``(reason, kind, count)``."""
    return dict(counter(env, "fallbacks"))


def run_fallback_classes(env: Any, core: Any, *, first_slot: int = 1) -> list[str]:
    """After startup at ``N = 2`` and ``max_num_seqs >= 3``, run one covered
    step and one step of every reachable fallback class (TXG-CAP-013,
    TXG-CAP-014): ``chunk-uncovered``, ``echo-uncovered`` and ``signature``
    with the fields ``endpoint_policy`` and ``forced_eou``. Every request
    takes a slot of its own, from ``first_slot`` (five slots). Returns the
    classes run; fails unless the counters show each one."""
    assert env.n == 2, "the scenario is written for N = 2"
    slots: dict[str, int] = {}

    def step(*rows: Any, cached: tuple[str, ...] = ()) -> None:
        for row in rows:
            slots.setdefault(row.req_id, first_slot + len(slots))
        ids = [r.req_id for r in rows] + list(cached)
        core.step(*rows, cached=cached, handles={r: slots[r] for r in ids})

    streams = tuple(f"cap13-k{i}" for i in range(3))
    before = fallback_reasons(env)
    step(stream_unit(env, "cap13-covered"))
    assert fallback_reasons(env) == before, "the covered step fell back"
    # K = N + 1, then the same requests with no new audio: E = N + 1.
    step(*[stream_unit(env, r) for r in streams])
    step(cached=streams)
    step(with_endpoint_policy(stream_unit(env, "cap13-policy")))
    step(eou_request(env, streams[0]))
    after = fallback_reasons(env)
    grown = {key for key, value in after.items() if value != before.get(key, 0)}
    expected = {
        ("chunk-uncovered", txg_api.CHUNK, 3),
        ("echo-uncovered", txg_api.ECHO, 3),
        ("signature", txg_api.CHUNK, 1),
        ("signature", txg_api.ECHO, 1),
    }
    assert grown == expected, (grown, after)
    return ["chunk-uncovered", "echo-uncovered", "endpoint_policy", "forced_eou"]


# ---------------------------------------------------------------------------
# Core's state-group layout accessor (TXG-CAP-008, TXG-LIFE-003)


def state_layer_of(env_or_runner: Any) -> Any:
    """The model's state layer (``NemotronStateLayer``)."""
    model = getattr(env_or_runner, "model", env_or_runner)
    return model._persistent_state_layer


def layouts_from_bound_views(runner: Any) -> tuple[Any, ...]:
    """What core's ``get_state_group_layouts`` builds from the views it just
    bound (``vllm/v1/worker/utils.py``): one ``StateGroupLayout`` for the one
    non-recomputable group, holding the state layer's bound ``kv_cache``
    views in the layer's spec order. Built with core's own function over a
    ``KVCacheConfig`` double of the fields it reads."""
    from vllm.v1.worker.utils import get_state_group_layouts

    layer = state_layer_of(runner)
    group = SimpleNamespace(
        manager_config=SimpleNamespace(recomputable=False),
        kv_cache_specs={layer.prefix: None},
    )
    return get_state_group_layouts(
        SimpleNamespace(kv_cache_groups=[group]),
        runner.vllm_config.compilation_config.static_forward_context,
    )


def follow_state_group_layouts(runner: Any) -> Any:
    """Make ``runner``'s pool bindings set ``ModelState._state_group_layouts``
    as core's model runner does: from the views it just bound, before the
    model state's ``initialize_kv_cache`` (``model_runner.py``,
    ``initialize_kv_cache``). Returns ``runner``. Idempotent with a harness
    that sets them itself: both set the same layouts."""
    state = runner.state
    original = state.initialize_kv_cache
    if getattr(original, "_txg_follows_layouts", False):
        return runner

    def initialize_kv_cache(*args: Any, **kwargs: Any) -> Any:
        state._state_group_layouts = layouts_from_bound_views(runner)
        return original(*args, **kwargs)

    initialize_kv_cache._txg_follows_layouts = True  # type: ignore[attr-defined]
    state.initialize_kv_cache = initialize_kv_cache
    return runner


def release_profiling_layouts(state: Any) -> None:
    """Core's release of the profiling KV cache after ``clear_cudagraphs``
    (``cudagraph_utils.py``, ``_teardown_profiling_state``): the accessor
    returns ``()`` until the serving cache is bound."""
    state._state_group_layouts = ()


def component_layouts(state: Any) -> list[tuple[str, tuple[Any, ...]]]:
    """The plain values of the accessor's current layouts, per state layer
    and component in order: ``(layer, ((address, shape, strides, dtype),
    ...))``. Empty when core has bound no KV cache."""
    out = []
    for layout in state.state_group_layouts:
        for name, components in layout.layers.items():
            out.append(
                (
                    name,
                    tuple(pool_layout_of(component.tensor) for component in components),
                )
            )
    return out


TRANSACTION_GRAPH_SOURCE = "nemotron_asr/transaction_graph.py"


def _called_from(source: str) -> bool:
    import sys

    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename.replace("\\", "/").endswith(source):
            return True
        frame = frame.f_back
    return False


class AccessorSpy:
    """Records every read of ``ModelState.state_group_layouts`` on ``state``
    (the property, shadowed on the model state's class for the test): whether
    it happened inside a ``capture_cudagraphs`` call, whether the
    transaction-graph module (``transaction_graph.py``) made it, how many
    layouts it returned, and for each component tensor a weak reference to
    its storage (``storages``) and the storage's span (``spans``,
    :func:`storage_span`) (TXG-CAP-008, TXG-LIFE-003). The storage, not the
    tensor object: a slice the plugin kept outlives the component and keeps
    the storage. Core's layout types take no weak reference (slotted
    dataclasses), so what holds a layout object is checked by reachability
    instead (:func:`plugin_holds`)."""

    def __init__(self, monkeypatch: Any, state: Any) -> None:
        from torch.multiprocessing.reductions import StorageWeakRef

        self.reads: list[dict[str, Any]] = []
        self.inside = False
        spy = self

        def state_group_layouts(self: Any) -> Any:
            layouts = self._state_group_layouts
            tensors = [
                component.tensor
                for layout in layouts
                for components in layout.layers.values()
                for component in components
            ]
            spy.reads.append(
                {
                    "inside": spy.inside,
                    "txg": _called_from(TRANSACTION_GRAPH_SOURCE),
                    "layouts": len(layouts),
                    "storages": [StorageWeakRef(t.untyped_storage()) for t in tensors],
                    "spans": [
                        (f"component tensor {i}", span)
                        for i, t in enumerate(tensors)
                        if (span := storage_span(t)) is not None
                    ],
                }
            )
            return layouts

        monkeypatch.setattr(
            type(state), "state_group_layouts", property(state_group_layouts)
        )
        capture = state.capture_cudagraphs

        def capture_cudagraphs(*args: Any, **kwargs: Any) -> Any:
            spy.inside = True
            try:
                return capture(*args, **kwargs)
            finally:
                spy.inside = False

        state.capture_cudagraphs = capture_cudagraphs

    def txg_reads(self, *, inside: bool) -> list[dict[str, Any]]:
        """The transaction-graph module's reads inside (or outside) a
        capture call."""
        return [r for r in self.reads if r["txg"] and r["inside"] is inside]


def accessor_objects(state: Any) -> list[tuple[str, Any]]:
    """What ``ModelState.state_group_layouts`` returns now, object by
    object: each layout, each component layout and each component tensor,
    labelled. Core installs these (``_state_group_layouts``) and keeps them
    until it binds or releases the KV cache again, so right after a capture
    call they are the objects the call's reads returned."""
    layouts = state._state_group_layouts
    # ``()`` is shared by every empty tuple, so only a nonempty one is a target.
    objects: list[tuple[str, Any]] = [("layout tuple", layouts)] if layouts else []
    for index, layout in enumerate(layouts):
        objects.append((f"layout {index}", layout))
        for name, components in layout.layers.items():
            for position, component in enumerate(components):
                label = f"layout {index} {name}[{position}]"
                objects.append((f"{label} component", component))
                objects.append((f"{label} tensor", component.tensor))
    return objects


#: The model state's attribute that core installs the layouts in.
CORE_ATTRIBUTE = "_state_group_layouts"

#: Modules whose objects are the harness's or pytest's, never the plugin's.
_HARNESS_MODULES = ("support.", "test_", "tests.", "_pytest", "pytest", "pluggy")

#: The plugin's package: its module namespaces and classes are walked as the
#: plugin's (module-level state such as ``advance._batch_stats_slot``).
PLUGIN_PACKAGE = "vllm_nemotron_asr_streaming"


def _harness_object(obj: Any) -> bool:
    import types

    owner = obj if isinstance(obj, (type, types.FunctionType)) else type(obj)
    return (getattr(owner, "__module__", None) or "").startswith(_HARNESS_MODULES)


def _plugin_name(name: str) -> bool:
    return name == PLUGIN_PACKAGE or name.startswith(PLUGIN_PACKAGE + ".")


def _referents(obj: Any) -> list[Any]:
    """What ``obj`` holds, for :func:`plugin_holds`: a function's closure
    cells and defaults (not its module globals), a bound method's object
    and function, a plugin class's attributes (no other class's, no
    module's), and otherwise the garbage collector's referents."""
    import types

    if isinstance(obj, type):
        return list(vars(obj).values()) if _plugin_name(obj.__module__) else []
    if isinstance(obj, types.ModuleType):
        return []
    if isinstance(obj, types.FunctionType):
        held = [c.cell_contents for c in obj.__closure__ or () if _filled(c)]
        held += list(obj.__defaults__ or ())
        held += list((obj.__kwdefaults__ or {}).values())
        return held
    if isinstance(obj, types.MethodType):
        return [obj.__self__, obj.__func__]
    return list(gc.get_referents(obj))


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return False
    return True


def storage_span(tensor: torch.Tensor) -> tuple[str, int, int] | None:
    """The allocation ``tensor`` views, as ``(device, start, end)`` byte
    addresses of its untyped storage: every view, slice, ``detach`` or
    reshape-as-view of a tensor shares its span, whichever tensor PyTorch
    records as its ``_base``. ``None`` for a tensor with no addressable
    storage (a meta tensor, an empty allocation)."""
    try:
        storage = tensor.untyped_storage()
        start, size = storage.data_ptr(), storage.nbytes()
    except (RuntimeError, NotImplementedError, TypeError):
        return None
    if not start or not size:
        return None
    return str(tensor.device), start, start + size


def _overlapping(
    span: tuple[str, int, int] | None, spans: list[tuple[str, tuple[str, int, int]]]
) -> str | None:
    if span is None:
        return None
    device, start, end = span
    for label, (other_device, other_start, other_end) in spans:
        if device == other_device and start < other_end and other_start < end:
            return label
    return None


def layer_permitted(layer: Any) -> list[Any]:
    """What the state layer may hold of core's storage (REG-STATE-018): its
    bound ``kv_cache`` and the objects its ``pages`` and ``component_pools``
    return (the page view, the views mapping and each view), read only while
    a ``kv_cache`` is bound. Reading them may build the layer's view cache,
    which is the layer's own."""
    bound = getattr(layer, "kv_cache", None)
    permitted: list[Any] = [] if bound is None else [bound]
    if bound is None or isinstance(bound, torch.Tensor) or not len(bound):
        return permitted
    for name in ("pages", "component_pools"):
        try:
            value = getattr(layer, name)
        except (AttributeError, RuntimeError):
            continue
        permitted.append(value)
        if isinstance(value, dict):
            permitted.extend(value.values())
    return permitted


def plugin_holds(
    env: Any,
    targets: list[tuple[str, Any]],
    *,
    spans: list[tuple[str, tuple[str, int, int]]] = (),
    limit: int = 3_000_000,
) -> list[str]:
    """Every path through which the plugin's objects hold one of
    ``targets`` (``[(label, object)]``) or any tensor that shares storage
    with a target tensor (its untyped storage overlaps the target's,
    :func:`storage_span`: a view, slice, ``detach`` or flattened view of it,
    whatever its ``_base``), or with one of ``spans`` (``[(label, span)]``,
    for a tensor already dropped). The walk covers every object the model,
    its model state and the plugin's module namespaces (with its classes'
    attributes) reach, except core's attribute on the model state
    (``_state_group_layouts``; another reference to the same objects
    counts), other modules' namespaces, and the harness's and pytest's
    objects (spies, fakes, the runner). It walks the state layer too: there,
    and in what the layer holds, only the objects :func:`layer_permitted`
    lists are the layer's (its bound ``kv_cache`` and its view cache,
    REG-STATE-018), so anything else on the layer that shares a pool's
    storage (a pool slice transaction-graph code stored there) is found; the
    same objects held anywhere else are the plugin's and are found too.
    Empty when the plugin holds none (TXG-CAP-008)."""
    import sys
    from collections import deque

    wanted = {id(obj): label for label, obj in targets}
    storages = list(spans)
    for label, obj in targets:
        if isinstance(obj, torch.Tensor) and (span := storage_span(obj)):
            storages.append((label, span))
    state = env.state
    layer = h.state_layer_of(env.model)
    permitted = {id(obj) for obj in layer_permitted(layer)}
    modules = [(n, m) for n, m in list(sys.modules.items()) if m is not None]
    namespaces = {id(vars(m)) for _, m in modules if hasattr(m, "__dict__")}
    # An object reached inside the layer (where its permitted objects are
    # skipped) is walked again when reached outside it, so that the layer's
    # permission never hides what the plugin holds elsewhere; one already
    # walked outside needs no walk inside.
    seen: dict[bool, set[int]] = {False: set(namespaces), True: set()}
    parent: dict[tuple[int, bool], tuple[tuple[int, bool] | None, str]] = {}
    queue: deque[tuple[Any, bool]] = deque()
    roots: list[tuple[str, Any]] = [("model", env.model), ("model state", state)]
    roots += [
        (f"module {n}", vars(m))
        for n, m in sorted(modules, key=lambda item: item[0])
        if _plugin_name(n) and hasattr(m, "__dict__")
    ]
    for name, root in roots:
        seen[False].add(id(root))
        parent[(id(root), False)] = (None, name)
        queue.append((root, False))

    def path(node: tuple[int, bool]) -> str:
        names = []
        current: tuple[int, bool] | None = node
        while current is not None:
            current, name = parent[current]
            names.append(name)
        return " -> ".join(reversed(names))

    found: list[str] = []
    reported: set[int] = set()

    def report(key: int, line: str) -> None:
        if key not in reported:
            reported.add(key)
            found.append(line)

    while queue:
        obj, inside = queue.popleft()
        if obj is state:
            # Core's attribute, not the plugin's: skip that one edge only.
            children = [v for k, v in vars(state).items() if k != CORE_ATTRIBUTE]
        elif obj is layer:
            inside = True
            children = (
                list(vars(layer).values())
                if hasattr(layer, "__dict__")
                else _referents(layer)
            )
        else:
            children = _referents(obj)
        node = (id(obj), inside if obj is not layer else False)
        for child in children:
            key = id(child)
            if inside and key in permitted:
                continue
            if key in seen[False] or key in seen[inside]:
                continue
            seen[inside].add(key)
            parent[(key, inside)] = (node, type(child).__name__)
            if key in wanted:
                report(key, f"{path((key, inside))}: {wanted[key]}")
                continue
            if isinstance(child, torch.Tensor):
                shared = _overlapping(storage_span(child), storages)
                if shared is not None:
                    report(key, f"{path((key, inside))}: shares storage with {shared}")
                continue
            if _harness_object(child):
                continue
            queue.append((child, inside))
            if len(seen[False]) + len(seen[True]) > limit:
                raise AssertionError(f"plugin_holds walked more than {limit} objects")
    return found


def make_layouts(*groups: dict[str, tuple[Any, ...]]) -> tuple[Any, ...]:
    """State-group layouts in core's types: one per ``{layer: (tensor,
    ...)}`` mapping, numbered from 0."""
    from types import MappingProxyType

    from vllm.v1.worker.utils import StateComponentLayout, StateGroupLayout

    return tuple(
        StateGroupLayout(
            group_id,
            MappingProxyType(
                {
                    name: tuple(StateComponentLayout(t) for t in tensors)
                    for name, tensors in layers.items()
                }
            ),
        )
        for group_id, layers in enumerate(groups)
    )


def bound_views(env: Any) -> tuple[Any, ...]:
    """The state layer's bound per-state views, in its spec order (what
    core's layout holds)."""
    return tuple(state_layer_of(env).kv_cache)


def kv_cache_sizing(vllm_config: Any) -> str:
    """The KV-cache sizing (``txg_api.Sizing``) a configuration's effective
    cache values select, the extensible KV cache checked first: a startup
    plan's ``kv_cache_memory_bytes`` set with the extensible KV cache on
    leaves it on (TXG-MEM-018)."""
    cache = vllm_config.cache_config
    if getattr(cache, "enable_extensible_kv_cache", False):
        return Sizing.EXTENSIBLE
    if getattr(cache, "kv_cache_memory_bytes", None):
        return Sizing.BUDGET
    if getattr(cache, "num_gpu_blocks_override", None) is not None:
        return Sizing.BLOCK_COUNT
    return Sizing.PROFILED


def estimate_runs(vllm_config: Any) -> bool:
    """Whether core runs its graph-memory estimate, and with it the
    profiling capture (``vllm/v1/worker/gpu_worker.py``): the extensible KV
    cache off, no fixed budget (``kv_cache_memory_bytes`` unset or 0) and
    ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`` not 0. A test's own reading
    of core, independent of the plugin's predicate."""
    import vllm.envs as envs

    cache = vllm_config.cache_config
    return (
        not getattr(cache, "enable_extensible_kv_cache", False)
        and not getattr(cache, "kv_cache_memory_bytes", None)
        and bool(envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS)
    )


def narrow_pool(bindings: Any, pages: int) -> Any:
    """The first ``pages`` pages of a state pool, as a pool of its own over
    the same storage (``base`` is the whole pool): what core binds when it
    binds views of fewer pages than its allocation holds."""
    pool = h.core_bindings(bindings)
    base = getattr(pool, "base", pool)
    assert 0 < pages <= base.capacity, (pages, base.capacity)
    return SimpleNamespace(
        layout=base.layout, capacity=pages, tensor=base.tensor[:pages], base=base
    )


class TxgCoreOrder(CoreOrder):
    """``CoreOrder`` whose bindings follow core's state-group layout accessor
    (:func:`follow_state_group_layouts`) and whose ``clear`` is followed by
    core's release of the profiling KV cache
    (:func:`release_profiling_layouts`), under the configuration's KV-cache
    sizing:

    * the profiling KV cache holds ``num_gpu_blocks_override`` blocks when
      that is set, otherwise ``min(max_num_seqs, max_cudagraph_capture_size)
      or 1`` blocks, one under the explicit NONE runner mode
      (``vllm/v1/worker/gpu/cudagraph_utils.py``; TXG-LIFE-004);
    * ``startup`` runs the profiling capture only when core's estimate runs
      (:func:`estimate_runs`), and under the extensible KV cache follows
      ``post_warmup`` with core's extension (:meth:`extend`).
    """

    def __init__(self, runner: Any, **kwargs: Any) -> None:
        super().__init__(follow_state_group_layouts(runner), **kwargs)
        #: The page count of the views :meth:`extend` bound, once it ran.
        self.final_count: int | None = None

    @property
    def vllm_config(self) -> Any:
        return self.runner.vllm_config

    def estimate_runs(self) -> bool:
        return estimate_runs(self.vllm_config)

    def extensible(self) -> bool:
        return kv_cache_sizing(self.vllm_config) == Sizing.EXTENSIBLE

    def profiling_capacity(self) -> int:
        """The block count of core's profiling KV cache."""
        cache = self.vllm_config.cache_config
        override = getattr(cache, "num_gpu_blocks_override", None)
        if override is not None:
            return int(override)
        sizes = getattr(
            self.vllm_config.compilation_config, "max_cudagraph_capture_size", None
        )
        seqs = int(self.vllm_config.scheduler_config.max_num_seqs)
        return min(seqs, int(sizes or 0)) or 1

    def bindings(self, capacity: int | None = None) -> Any:
        """A fresh state pool of ``capacity`` pages (the serving capacity
        unless given)."""
        bindings = h.real_bindings(
            h.model_spec(self.runner.model),
            capacity=self.capacity if capacity is None else capacity,
            device=self.device.type,
        )
        return h.BindingsSpy(bindings) if self.spy else bindings

    def bind_profiling(self, bindings: Any = None) -> Any:
        """Core's profiling KV cache, of :meth:`profiling_capacity` blocks
        unless ``bindings`` are given."""
        if bindings is None:
            bindings = self.bindings(self.profiling_capacity())
        return super().bind_profiling(bindings)

    def extend(self, final: int | None = None) -> Any:
        """Core's extension of the extensible KV cache after
        ``post_warmup`` and before the first request
        (``vllm/v1/worker/extensible_kv_cache.py``, ``extend_kv_cache``):
        ``bind_kv_cache`` alone over the same storage with ``final`` pages
        (the serving pool's whole allocation unless given), with no
        ``initialize_kv_cache`` and no new state-group layout. ``final`` above
        the bound views' page count binds views of more pages than the
        serving capture read (a pool from :func:`narrow_pool`). Returns the
        bindings, now the serving ones."""
        pool = h.core_bindings(self.serving_bindings)
        base = getattr(pool, "base", pool)
        final = base.capacity if final is None else final
        extended = narrow_pool(base, final)
        h.state_layer_of(self.runner.model).bind_kv_cache(extended.tensor)
        config = getattr(self.state, "kv_cache_config", None)
        if config is not None and hasattr(config, "num_blocks"):
            config.num_blocks = final
        self.runner.bindings = self.serving_bindings = extended
        self.final_count = final
        return extended

    def startup(
        self,
        *,
        profiling: bool | None = None,
        warmup: bool = False,
        verify: bool = True,
        extend: bool | None = None,
    ) -> Any:
        """``CoreOrder.startup`` under the configuration's sizing:
        ``profiling`` defaults to whether core's estimate runs, and
        ``extend`` to whether the extensible KV cache is on and ``verify``
        ran ``post_warmup``."""
        if profiling is None:
            profiling = self.estimate_runs()
        bindings = super().startup(profiling=profiling, warmup=warmup, verify=verify)
        if extend is None:
            extend = verify and self.extensible()
        if extend:
            bindings = self.extend()
        return bindings

    def clear(self) -> None:
        super().clear()
        release_profiling_layouts(self.state)


# ---------------------------------------------------------------------------
# Builders


def _common_spies(monkeypatch: Any, events: h.Events) -> SimpleNamespace:
    return SimpleNamespace(
        profile=h.ProfileSpy(events).install(monkeypatch),
        advance=h.AdvanceSpy(events).install(monkeypatch),
        graphs=h.FakeGraphRuntime(events).install(monkeypatch),
    )


def _card(tmp_path: Any) -> Any:
    return h.write_runnable_card(tmp_path / f"card-{len(list(tmp_path.iterdir()))}")


def build_model(vllm_config: Any) -> Any:
    """The model with seeded initial weights: the harness loads no
    checkpoint, so every environment of a test (an option-on arm and its
    ``B_hook`` arm, two processes' worth of history) gets the same weights."""
    torch.manual_seed(0)
    return h.build_model(vllm_config)


def build_off(
    monkeypatch: Any,
    tmp_path: Any,
    *,
    right_context: int = 0,
    max_num_seqs: int = 2,
    overrides: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Transaction graphs off, ``--enforce-eager`` (the harness default)."""
    events = h.Events()
    spies = _common_spies(monkeypatch, events)
    vllm_config = h.build_vllm_config(
        _card(tmp_path),
        right_context=right_context,
        max_num_seqs=max_num_seqs,
        overrides=overrides,
    )
    model = build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    return SimpleNamespace(
        vllm_config=vllm_config,
        model=model,
        state=state,
        config=model.config,
        runner=follow_state_group_layouts(h.Runner(vllm_config, model, state)),
        events=events,
        geometry=geometry_of(right_context),
        max_num_seqs=max_num_seqs,
        n=0,
        **vars(spies),
    )


def _hook_configuration(vllm_config: Any, populations: Any) -> None:
    """The option on the final configuration, core's hook runner."""
    from vllm.config import CompilationMode, CUDAGraphMode

    setattr(vllm_config.model_config.hf_config, OPTION, populations)
    vllm_config.model_config.enforce_eager = False
    compilation = vllm_config.compilation_config
    compilation.cudagraph_mode = CUDAGraphMode.NONE
    compilation.mode = CompilationMode.NONE


def build_on_cpu(
    monkeypatch: Any,
    tmp_path: Any,
    *,
    n: int = 2,
    populations: Any = None,
    right_context: int = 0,
    max_num_seqs: int = 2,
    max_num_active_seqs: int | None = None,
    sizing: str | None = None,
    kv_cache_memory_bytes: int | None = None,
    num_gpu_blocks_override: int | None = None,
    capacity: int = CAPACITY,
) -> SimpleNamespace:
    """Transaction graphs on (``N = n``) through the fake-capture seam, under
    the restricted startup (TXG-CFG-017, TXG-CFG-018: synchronous
    scheduling, an explicit NONE runner and compilation mode) and the
    KV-cache sizing ``sizing`` (``txg_api.Sizing``). The sizing is core's
    default, the extensible KV cache, unless ``kv_cache_memory_bytes`` fixes
    a budget, ``num_gpu_blocks_override`` a block count, or ``sizing`` names
    another; a test sets a fixed setting only where its spec names one.
    Under :attr:`Sizing.BLOCK_COUNT` the override is
    ``num_gpu_blocks_override``, ``capacity`` unless given.

    Every serving state pool the environment's ``CoreOrder`` binds holds
    ``capacity`` pages (the reservation capacity under the extensible KV
    cache); its profiling KV cache holds the override's blocks under a fixed
    block count, otherwise one block (:class:`TxgCoreOrder`, TXG-LIFE-004).
    ``max_num_active_seqs`` sets the admission limit below
    ``max_num_seqs``."""
    if sizing is None:
        sizing = (
            Sizing.BUDGET
            if kv_cache_memory_bytes
            else Sizing.BLOCK_COUNT
            if num_gpu_blocks_override is not None
            else Sizing.EXTENSIBLE
        )
    blocks = budget = None
    if sizing == Sizing.BLOCK_COUNT:
        blocks = num_gpu_blocks_override
        if blocks is None:
            blocks = capacity
    elif num_gpu_blocks_override is not None:
        raise ValueError(f"num_gpu_blocks_override under the sizing {sizing!r}")
    if sizing == Sizing.BUDGET:
        budget = kv_cache_memory_bytes
    elif kv_cache_memory_bytes:
        raise ValueError(f"kv_cache_memory_bytes under the sizing {sizing!r}")
    events = h.Events()
    spies = _common_spies(monkeypatch, events)
    vllm_config = h.build_vllm_config(
        _card(tmp_path),
        right_context=right_context,
        max_num_seqs=max_num_seqs,
        overrides=dict(DEFAULT_ARMS),
    )
    _hook_configuration(vllm_config, [n] if populations is None else populations)
    set_cache(vllm_config, **sizing_settings(sizing, blocks=blocks, budget=budget))
    set_scheduler(vllm_config, max_num_active_seqs=max_num_active_seqs)
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    _set_env_attr(monkeypatch, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", True)
    fake = FakeCuda(events).install(monkeypatch)
    scatter = ScatterSpy(events).install(monkeypatch)
    model = build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    return SimpleNamespace(
        vllm_config=vllm_config,
        model=model,
        state=state,
        config=model.config,
        runner=follow_state_group_layouts(h.Runner(vllm_config, model, state)),
        events=events,
        fake=fake,
        scatter=scatter,
        geometry=geometry_of(right_context),
        max_num_seqs=max_num_seqs,
        admission_limit=max_num_seqs
        if max_num_active_seqs is None
        else max_num_active_seqs,
        capacity=capacity,
        sizing=sizing,
        n=n,
        capture_stream=fake.new_stream(name="core-capture"),
        **vars(spies),
    )


def core_of(env: Any) -> TxgCoreOrder:
    """The ``CoreOrder`` of a CPU environment: its runner, its capture
    stream."""
    return TxgCoreOrder(
        env.runner,
        capture_stream=getattr(env, "capture_stream", None),
        capacity=getattr(env, "capacity", CAPACITY),
    )


def on_cpu(monkeypatch: Any, tmp_path: Any, **kwargs: Any) -> SimpleNamespace:
    """An option-on CPU environment (``build_on_cpu``) and its ``CoreOrder``."""
    env = build_on_cpu(monkeypatch, tmp_path, **kwargs)
    return SimpleNamespace(env=env, core=core_of(env))


def set_cache(vllm_config: Any, **values: Any) -> None:
    """Set effective cache values as core does after construction (a startup
    plan's ``kv_cache_memory_bytes``; the engine disabling the extensible KV
    cache), without re-running the configuration's validators. The CPU
    platform resolves the extensible KV cache off, so a CPU environment sets
    its sizing here."""
    cache = vllm_config.cache_config
    for key, value in values.items():
        try:
            setattr(cache, key, value)
        except Exception:  # noqa: BLE001 - validated dataclass
            object.__setattr__(cache, key, value)


def set_scheduler(vllm_config: Any, **values: Any) -> None:
    """Set scheduler values (``max_num_active_seqs``) on a built
    configuration, as ``set_cache`` does."""
    scheduler = vllm_config.scheduler_config
    for key, value in values.items():
        try:
            setattr(scheduler, key, value)
        except Exception:  # noqa: BLE001 - validated dataclass
            object.__setattr__(scheduler, key, value)


def _set_env_attr(monkeypatch: Any, name: str, value: Any) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, name, value, raising=False)


def set_estimate(monkeypatch: Any, enabled: bool) -> None:
    monkeypatch.setenv(
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1" if enabled else "0"
    )
    _set_env_attr(monkeypatch, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", enabled)


# ---------------------------------------------------------------------------
# Spies


def is_scatter_warmup(call: dict[str, Any]) -> bool:
    """Whether a ``ScatterSpy`` entry has the shape of a transaction-graph
    scatter warm-up launch: one executed row with a rejected (nonzero)
    status (TXG-CAP-006, TXG-LIFE-004), whatever page it names, so that a
    test locates the warm-up by its shape and checks its page (block 0)
    where its spec states it. The default encoder's one-row profile
    transaction commits its row (status 0). A count-1 cell's eager
    reference, warm-ups and capture also launch one rejected row, so a
    caller that needs the warm-up alone bounds its window before the first
    cell. The kernel-specialization warm-up (``warmup_masked_page_scatter``,
    logged as ``"warmup"``) is not one."""
    return (
        call["fn"] == "execute"
        and call["rows"] == 1
        and all(s != 0 for s in call["status"])
    )


def pool_layout_of(pool: Any) -> tuple[Any, ...]:
    """A scatter pool argument as TXG-CAP-013 compares it, from host values
    alone: ``(address, shape, strides, dtype)``."""
    return (pool.data_ptr(), tuple(pool.shape), tuple(pool.stride()), pool.dtype)


def within(tensor: Any, pool: Any) -> bool:
    """Whether ``pool``'s address (an integer, or a tensor's) lies inside
    ``tensor``'s storage."""
    address = pool if isinstance(pool, int) else pool.data_ptr()
    start = tensor.data_ptr()
    return start <= address < start + tensor.numel() * tensor.element_size()


class ScatterSpy:
    """Logs every masked-scatter launch (the plugin's one Triton kernel,
    ``state_scatter``) wherever the module's executors are bound by name.

    Every entry holds the pool argument's host-side layout: ``pool`` (its
    address), ``layout`` (``(address, shape, strides, dtype)``, what
    TXG-CAP-013 compares) and ``rows``. The page IDs (``pages``) and statuses
    (``status``) are read only from host tensors: reading a device tensor's
    values is a synchronization, which fails inside a graph capture, so for a
    device launch both are ``None``. CPU executes the CPU loop; a CUDA
    warm-up on a CPU pool is logged and skipped (the kernel cannot run
    there)."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []
        #: Raise from the ``n``-th transaction-graph scatter warm-up launch
        #: (``is_scatter_warmup``), TXG-MEM-010's warm-up phase.
        self.fail_warmup: int | None = None
        self.warmups = 0

    def _log(self, name: str, pool: Any, block_ids: Any, row_status: Any) -> None:
        host = block_ids.device.type == "cpu" and row_status.device.type == "cpu"
        entry = {
            "fn": name,
            "pool": pool.data_ptr(),
            "layout": pool_layout_of(pool),
            "rows": int(block_ids.numel()),
            "pages": (
                [int(v) for v in block_ids.reshape(-1).tolist()] if host else None
            ),
            "status": (
                [int(v) for v in row_status.reshape(-1).tolist()] if host else None
            ),
        }
        self.calls.append(entry)
        self.events.append(("scatter", entry))

    def install(self, monkeypatch: Any) -> ScatterSpy:
        from vllm_nemotron_asr_streaming.nemotron_asr import state_scatter

        spy = self
        execute = state_scatter.execute_masked_page_scatter_
        warmup = state_scatter.warmup_masked_page_scatter

        def execute_spy(pool, scratch, block_ids, row_status):
            spy._log("execute", pool, block_ids, row_status)
            if is_scatter_warmup(spy.calls[-1]):
                number, spy.warmups = spy.warmups, spy.warmups + 1
                if spy.fail_warmup == number:
                    from support.fake_cuda import InjectedFault

                    raise InjectedFault("injected fault in the scatter warm-up")
            return execute(pool, scratch, block_ids, row_status)

        def warmup_spy(pool, scratch):
            ids = torch.zeros(1, dtype=torch.int64)
            spy._log("warmup", pool, ids, torch.ones(1, dtype=torch.int32))
            if pool.device.type == "cuda":
                return warmup(pool, scratch)
            return None

        targets = [state_scatter]
        for module_name in (
            "vllm_nemotron_asr_streaming.nemotron_asr.advance",
            "vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph",
        ):
            try:
                targets.append(importlib.import_module(module_name))
            except ImportError:
                continue
        for module in targets:
            if hasattr(module, "execute_masked_page_scatter_"):
                monkeypatch.setattr(module, "execute_masked_page_scatter_", execute_spy)
            if hasattr(module, "warmup_masked_page_scatter"):
                monkeypatch.setattr(module, "warmup_masked_page_scatter", warmup_spy)
        return self


class SinkSpy:
    """Counts commit reservations, stages, cancellations and collections of
    the model's ``BoundedCommitSink``."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def install(self, monkeypatch: Any) -> SinkSpy:
        from vllm_nemotron_asr_streaming.nemotron_asr import commit_sink

        spy = self
        for cls, name in (
            (commit_sink.BoundedCommitSink, "reserve"),
            (commit_sink.BoundedCommitSink, "collect"),
            (commit_sink.CommitTicket, "stage"),
            (commit_sink.CommitTicket, "cancel"),
        ):
            original = getattr(cls, name)

            def wrapper(self, *args, __original=original, __name=name, **kwargs):
                spy.log.append(__name)
                return __original(self, *args, **kwargs)

            monkeypatch.setattr(cls, name, wrapper)
        return self


class FailSpy:
    """Records every request set the model state's ``take_failed_requests``
    returns (REG-MS-003, REG-MS-004, TXG-FAIL-008): core's runner calls it
    after each step's ``postprocess_state``, and each call returns the
    requests recorded since the previous one and clears the record.

    Wraps the method on the model state instance, so a harness that calls it
    after each step and a test that calls it (``failed`` takes once more,
    draining what the harness left) see the same sets."""

    def __init__(self, state: Any) -> None:
        self.calls: list[set[str]] = []
        self._state = state
        original = state.take_failed_requests

        def take_failed_requests() -> Any:
            result = original()
            if result:
                self.calls.append(set(result))
            return result

        state.take_failed_requests = take_failed_requests

    def take(self) -> Any:
        """One ``take_failed_requests`` call, as core makes after a step."""
        return self._state.take_failed_requests()

    @property
    def failed(self) -> set[str]:
        """Every request returned so far, after one more drain."""
        self.take()
        return set().union(*self.calls) if self.calls else set()


def _after_initialization(env: Any, take: Any) -> None:
    """Call ``take()`` now, and again right after the state layer's next
    fresh-page initialization (``initialize_pages``, REG-STATE-010), which
    runs before the step's transaction reads or writes a page."""
    take()
    layer = state_layer_of(env)
    original = layer.initialize_pages

    def initialize_pages(slots: Any) -> Any:
        result = original(slots)
        take()
        layer.initialize_pages = original
        return result

    layer.initialize_pages = initialize_pages


def slots_at_step_start(env: Any, slots: Any) -> dict[int, torch.Tensor]:
    """The pages of ``slots`` as the next step's transaction finds them: the
    pages now, replaced by the state layer's fresh-page initialization
    (REG-STATE-010) once the next step runs it. A rejected row's pages must
    equal these (TXG-FAIL-001, TXG-FAIL-008, TXG-MASK-006): preservation
    starts after initialization."""
    taken: dict[int, torch.Tensor] = {}

    def take() -> None:
        for slot in slots:
            taken[slot] = h.slot_bytes(env.runner.bindings, slot).clone()

    _after_initialization(env, take)
    return taken


def pool_at_step_start(env: Any) -> dict[str, torch.Tensor]:
    """The whole state pool as the next step's transaction finds it (the
    pool now, replaced after the step's fresh-page initialization,
    REG-STATE-010), under key ``"pool"``. A step that fails before it writes
    a page must leave the pool equal to this (TXG-EXEC-003)."""
    taken: dict[str, torch.Tensor] = {}

    def take() -> None:
        taken["pool"] = h.pool_tensor(env.runner.bindings).clone()

    _after_initialization(env, take)
    return taken


def silence_collection(env: Any, req_ids: set[str]) -> None:
    """Make the sink's collection report nothing for ``req_ids`` (no
    status, no failure, no queue count) to the model state, so a test can
    tell failure delivered from the planning-failed set from failure
    delivered by collection (TXG-FAIL-008). Works on the ``CommitCollection``
    the model returns (FILE-QC-005), through ``Runner.collect_hook``."""

    def without(collection: Any) -> Any:
        statuses, failed, counts = h.collection_parts(collection)
        if not hasattr(collection, "queue_counts"):
            raise AssertionError(
                "collect_commit_status returned no CommitCollection (FILE-QC-005)"
            )
        return h.replace_collection(
            collection,
            statuses={k: v for k, v in statuses.items() if k not in req_ids},
            failed=set(failed) - req_ids,
            queue_counts=type(collection.queue_counts)(
                c for c in counts if getattr(c, "request_id", None) not in req_ids
            ),
        )

    env.runner.collect_hook = without


def fake_graph_count(env: Any) -> int:
    gc.collect()
    return len(env.fake.live_graphs())


def captures_since(env: Any, mark: int) -> list:
    """Cell and segment-graph captures logged since ``mark``."""
    return [
        e for e in env.events[mark:] if e[0] in ("cuda.graph.capture_begin", "capture")
    ]


def replays_since(env: Any, mark: int) -> int:
    return sum(1 for e in env.events[mark:] if e[0] == "cuda.graph.replay")


def stream_of(env: Any) -> FakeStream:
    return env.capture_stream


__all__ = [
    "MARGIN_BYTES",
    "OPTION",
    "FailSpy",
    "SinkSpy",
    "build_off",
    "build_on_cpu",
    "core_of",
]


# ---------------------------------------------------------------------------
# The ``execute`` seam (gpu no-growth tests)


def wrap_execute(monkeypatch: Any, model: Any, around: Any) -> None:
    """Wrap the ``execute`` (TXG-EXEC-001) of the model's transaction-graph
    binding (``support.txg_api.binding_of``) so a test can measure exactly the
    span from ``execute`` entry to its return, commit staging included.
    ``around(call)`` receives a zero-argument callable running the original
    and returns its result."""
    from support import txg_api

    binding = txg_api.binding_of(model)
    assert binding is not None, "transaction graphs are off"
    original = binding.execute

    def execute(*args: Any, **kwargs: Any) -> Any:
        return around(lambda: original(*args, **kwargs))

    monkeypatch.setattr(binding, "execute", execute)


# ---------------------------------------------------------------------------
# Phase and publication spies (TXG-CAP-004, TXG-CAP-005)


def spy_phases(monkeypatch: Any, env: Any, *names: str) -> None:
    """Log each call of the named phase functions of ``advance``
    (``support.txg_api.PHASES``) as ``("phase.<name>", plan)`` in
    ``env.events``; ``plan`` is the call's ``RowPlan`` argument (the
    positional or keyword argument with an ``is_file`` attribute)."""
    from support import txg_api

    module = txg_api.advance_module()
    for name in names:
        original = txg_api.phase_function(name)

        def spy(*args: Any, __original=original, __name=name, **kwargs: Any) -> Any:
            plan = next(
                (a for a in (*args, *kwargs.values()) if hasattr(a, "is_file")), None
            )
            env.events.append((f"phase.{__name}", plan))
            return __original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)


def row_bits(plan: Any) -> tuple[list[bool], list[bool]]:
    """Per-row fresh and file bits of a position-indexed ``RowPlan``
    (TXG-TXN-010): ``is_fresh`` and ``is_file``."""
    fresh = getattr(plan, "is_fresh", None)
    if fresh is None:
        raise AssertionError("RowPlan has no per-row is_fresh vector (TXG-TXN-010)")
    return [bool(v) for v in fresh.tolist()], [bool(v) for v in plan.is_file.tolist()]


# ---------------------------------------------------------------------------
# Block accesses (TXG-CAP-018)

#: Calls that read no tensor data: metadata, and views (classified after the
#: call by aliasing, below).
_METADATA = frozenset(
    {
        "__get__",
        "data_ptr",
        "size",
        "stride",
        "dim",
        "ndimension",
        "numel",
        "nelement",
        "element_size",
        "storage_offset",
        "untyped_storage",
        "is_contiguous",
        "__len__",
        "get_device",
        "is_pinned",
        "is_floating_point",
        "is_complex",
        "_is_view",
    }
)
#: In-place calls: the result aliases the input, and the call writes it.
_IN_PLACE = frozenset({"__setitem__", "copy_", "index_put_", "fill_", "zero_"})
#: Calls that address the first argument's dimension ``dim`` by an index
#: tensor: ``(position of dim, position of the index)``, as positional
#: arguments after the tensor.
_INDEXED = {
    "index_select": (0, 1),
    "index_copy_": (0, 1),
    "index_copy": (0, 1),
    "index_add_": (0, 1),
    "index_add": (0, 1),
    "index_fill_": (0, 1),
    "gather": (0, 1),
    "scatter_": (0, 1),
    "scatter": (0, 1),
}
#: The plugin's masked page scatter, a custom op on CUDA
#: (``torch.ops.vllm``, ``state_scatter``): for each row whose status is 0
#: it stores that row's page at the row's block ID; a row with any other
#: status returns before it reads its block ID or the pool. Its arguments are
#: ``(pool, scratch, block_ids, row_status)``.
_MASKED_SCATTER = "nemotron_asr_masked_page_scatter_"


class _Unread(Exception):
    """An index or status tensor a CUDA capture computed and has not yet
    written (``BlockAccesses``)."""


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _tensors(item)]
    return []


class BlockAccesses(torch.overrides.TorchFunctionMode):
    """Records which blocks of one state pool every torch call in its
    context reads or writes (TXG-CAP-018): a CPU reading of "addresses only
    block 0", over the pool's backing storage.

    ``pool`` is the backing tensor core allocated (``[blocks, 1, 1, page]``
    bytes; ``model_harness.StatePool.tensor``); every view the plugin makes
    of it (the state layer's component pools, the projected resident-pool
    views, slices of either) shares its storage. A call that creates a view
    or reads metadata addresses nothing; a call that indexes a pool view
    along its first dimension by an index tensor addresses the blocks of the
    rows it names; the plugin's masked page scatter op addresses the block
    IDs of its rows whose status is 0, the rows it stores; any other call
    that takes a pool view (a copy, a comparison, an elementwise operation,
    a reduction) addresses every block the view spans. ``accesses`` lists
    ``(call name, sorted blocks)``; it holds no tensor.

    Only torch calls are recorded: a CUDA graph's replay
    (``CUDAGraph.replay``) and a Triton kernel launched directly, not
    through a torch op, address pages unrecorded. While a CUDA stream
    captures, an index or status tensor on the device cannot be read without
    ending the capture, and a tensor the capture itself computes (from the
    graph's private memory pool, so at another address than in the
    warm-ups) holds no value until a replay writes it. The reading then
    takes the values it last read from the same tensor (the capture's own
    warm-ups); with none, it keeps the call and its arguments and reads them
    when the context exits, after the call's validation replay has written
    them. Holding the tensor keeps the capture from reusing its memory, so
    the value read is the one the replayed graph indexed by. An index value
    outside the view's first dimension charges the whole view.
    """

    def __init__(self, pool: torch.Tensor) -> None:
        super().__init__()
        self._storage = pool.untyped_storage().data_ptr()
        self._base = pool.data_ptr()
        self._page = int(pool.stride(0)) * pool.element_size()
        self._blocks = int(pool.shape[0])
        self.accesses: list[tuple[str, tuple[int, ...]]] = []
        #: The values last read from each index or status tensor, by its
        #: device, address, shape and dtype.
        self._read: dict[tuple[Any, ...], list[int]] = {}
        #: Calls recorded while a capture could not read their index or
        #: status values: ``(position in accesses, name, args, kwargs,
        #: pooled)``, read when the context exits.
        self._pending: list[tuple[int, str, Any, Any, list[torch.Tensor]]] = []

    def __exit__(self, *exc: Any) -> Any:
        out = super().__exit__(*exc)
        self._resolve()
        return out

    def _resolve(self) -> None:
        """Read the calls a capture kept, now that a replay has written
        their index and status tensors."""
        pending, self._pending = self._pending, []
        if pending and torch.cuda.is_available():
            torch.cuda.synchronize()
        for position, name, args, kwargs, pooled in pending:
            blocks = self._classify(name, args, kwargs, pooled)
            self.accesses[position] = (name, tuple(sorted(blocks)))

    def _unreadable(self, index: torch.Tensor) -> bool:
        """Whether ``index``'s values cannot be read now: it is on the
        device while a CUDA stream captures."""
        return index.is_cuda and torch.cuda.is_current_stream_capturing()

    def _aliases(self, tensor: torch.Tensor) -> bool:
        try:
            return tensor.untyped_storage().data_ptr() == self._storage
        except Exception:  # noqa: BLE001 - a tensor without storage
            return False

    def _span(self, tensor: torch.Tensor, start: int | None = None) -> set[int]:
        """The blocks a view spans (from byte ``start`` when given)."""
        if tensor.numel() == 0:
            return set()
        size = tensor.element_size()
        first = tensor.data_ptr() - self._base if start is None else start
        low = high = 0
        for extent, stride in zip(tensor.shape, tensor.stride(), strict=True):
            reach = (extent - 1) * stride * size
            low, high = low + min(0, reach), high + max(0, reach)
        return set(
            range(
                (first + low) // self._page, (first + high + size - 1) // self._page + 1
            )
        )

    def _values(self, index: Any) -> list[int]:
        """The row indices ``index`` holds (a boolean mask's true
        positions), or, while a CUDA stream captures, the values last read
        from the same device tensor (``_Unread`` when there are none)."""
        if isinstance(index, torch.Tensor):
            key = (index.device, index.data_ptr(), tuple(index.shape), index.dtype)
            if self._unreadable(index):
                if key not in self._read:
                    raise _Unread
                return self._read[key]
            values = index.reshape(-1)
        else:
            key, values = None, torch.as_tensor(index).reshape(-1)
        if values.dtype == torch.bool:
            values = values.nonzero().reshape(-1)
        out = [int(v) for v in values.tolist()]
        if key is not None:
            self._read[key] = out
        return out

    def _rows(self, tensor: torch.Tensor, index: Any) -> set[int]:
        """The blocks of the rows ``index`` names along the view's first
        dimension (every block the view spans when the values cannot be
        read)."""
        values = self._values(index)
        rows = tensor[0] if tensor.dim() > 1 else tensor[:1]
        first = tensor.data_ptr() - self._base
        step = int(tensor.stride(0)) * tensor.element_size()
        blocks: set[int] = set()
        for value in set(values):
            value = value + tensor.shape[0] if value < 0 else value
            if not 0 <= value < tensor.shape[0]:
                return self._span(tensor)
            blocks |= self._span(rows, start=first + value * step)
        return blocks

    def _scattered(
        self, pool: torch.Tensor, rows: int, block_ids: Any, status: Any
    ) -> set[int]:
        """The blocks the masked page scatter stores over its ``rows`` rows:
        the block IDs of the rows whose status is 0."""
        ids, statuses = self._values(block_ids), self._values(status)
        stored = [i for i, s in zip(ids[:rows], statuses[:rows]) if s == 0]
        return self._rows(pool, stored) if stored else set()

    def _key_rows(self, tensor: torch.Tensor, key: Any) -> set[int] | None:
        """The rows an indexing key names by an index tensor in the first
        dimension; ``None`` for a basic key (a view)."""
        head = key[0] if isinstance(key, tuple) and key else key
        if isinstance(head, torch.Tensor) or (
            isinstance(head, list) and head and not isinstance(head[0], slice)
        ):
            return self._rows(tensor, head)
        return None

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        result = func(*args, **kwargs)
        if name in _METADATA:
            return result
        pooled = [t for t in _tensors((args, kwargs)) if self._aliases(t)]
        if not pooled:
            return result
        if (
            name.split(".")[0] != _MASKED_SCATTER
            and name not in _IN_PLACE
            and isinstance(result, torch.Tensor)
            and self._aliases(result)
        ):
            return result  # a view, or a no-op such as ``cpu()`` on CPU
        try:
            blocks = self._classify(name, args, kwargs, pooled)
        except _Unread:
            self._pending.append((len(self.accesses), name, args, kwargs, pooled))
            self.accesses.append((name, ()))
            return result
        self.accesses.append((name, tuple(sorted(blocks))))
        return result

    def _classify(
        self, name: str, args: Any, kwargs: Any, pooled: list[torch.Tensor]
    ) -> set[int]:
        """The blocks one recorded call addresses (``_Unread`` while a
        capture cannot read its index or status values)."""
        if (
            name.split(".")[0] == _MASKED_SCATTER
            and len(args) >= 4
            and isinstance(args[0], torch.Tensor)
            and self._aliases(args[0])
        ):
            pool, scratch, block_ids, status = args[:4]
            blocks = self._scattered(pool, int(scratch.shape[0]), block_ids, status)
            for tensor in pooled:
                if tensor is not pool:
                    blocks |= self._span(tensor)
            return blocks
        blocks: set[int] = set()
        target = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if target is not None and self._aliases(target):
            if name in ("__getitem__", "__setitem__"):
                rows = self._key_rows(target, args[1])
                if rows is None and name == "__setitem__":
                    rows = self._span(target[args[1]])
                if rows is not None:
                    blocks |= rows
                    pooled = [t for t in pooled if t is not target]
            elif name in ("index_put_", "index_put"):
                blocks |= self._key_rows(target, tuple(args[1])) or self._span(target)
                pooled = [t for t in pooled if t is not target]
            elif name in _INDEXED:
                dim = args[1] if len(args) > 1 else kwargs.get("dim")
                index = args[2] if len(args) > 2 else kwargs.get("index")
                if dim == 0 and index is not None:
                    blocks |= self._rows(target, index)
                    pooled = [t for t in pooled if t is not target]
        if name == "embedding" and len(args) > 1 and self._aliases(args[1]):
            blocks |= self._rows(args[1], args[0])
            pooled = [t for t in pooled if t is not args[1]]
        for tensor in pooled:
            blocks |= self._span(tensor)
        return blocks

    def blocks(self) -> set[int]:
        """Every block any recorded call addressed."""
        return {block for _, touched in self.accesses for block in touched}

    def outside(self, allowed: set[int]) -> list[tuple[str, tuple[int, ...]]]:
        """The recorded calls that addressed a block outside ``allowed``."""
        return [
            (name, touched)
            for name, touched in self.accesses
            if not set(touched) <= allowed
        ]
