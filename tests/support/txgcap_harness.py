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
  ``support.core_order.CoreOrder`` (``core_of``), which calls the model
  state's hooks in core's order, one call at a time.

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

OPTION = "transaction_graph_populations"
LOOKAHEADS = (0, 1, 3, 6, 13)
#: The default segment arms, which transaction graphs require (TXG-CFG-008).
DEFAULT_ARMS = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
}
FRESH_ENGINE = re.compile(r"fresh engine", re.IGNORECASE)


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


# ---------------------------------------------------------------------------
# Builders


def _common_spies(monkeypatch: Any, events: h.Events) -> SimpleNamespace:
    h.spec_without_state_name(monkeypatch)
    h.legacy_state_api(monkeypatch)
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
        runner=h.Runner(vllm_config, model, state),
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
    kv_cache_memory_bytes: int | None = None,
    extensible_kv_cache: bool | None = False,
) -> SimpleNamespace:
    """Transaction graphs on (``N = n``) through the fake-capture seam."""
    events = h.Events()
    spies = _common_spies(monkeypatch, events)
    vllm_config = h.build_vllm_config(
        _card(tmp_path),
        right_context=right_context,
        max_num_seqs=max_num_seqs,
        overrides=dict(DEFAULT_ARMS),
    )
    _hook_configuration(vllm_config, [n] if populations is None else populations)
    set_cache(vllm_config, kv_cache_memory_bytes=kv_cache_memory_bytes)
    set_cache(vllm_config, enable_extensible_kv_cache=extensible_kv_cache)
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
        runner=h.Runner(vllm_config, model, state),
        events=events,
        fake=fake,
        scatter=scatter,
        geometry=geometry_of(right_context),
        max_num_seqs=max_num_seqs,
        n=n,
        capture_stream=fake.new_stream(name="core-capture"),
        **vars(spies),
    )


def core_of(env: Any) -> CoreOrder:
    """The ``CoreOrder`` of a CPU environment: its runner, its capture
    stream."""
    return CoreOrder(env.runner, capture_stream=getattr(env, "capture_stream", None))


def on_cpu(monkeypatch: Any, tmp_path: Any, **kwargs: Any) -> SimpleNamespace:
    """An option-on CPU environment (``build_on_cpu``) and its ``CoreOrder``."""
    env = build_on_cpu(monkeypatch, tmp_path, **kwargs)
    return SimpleNamespace(env=env, core=core_of(env))


def set_cache(vllm_config: Any, **values: Any) -> None:
    """Set effective cache values as core does after construction (a startup
    plan's ``kv_cache_memory_bytes``; the engine disabling the extensible KV
    cache), without re-running the configuration's validators."""
    cache = vllm_config.cache_config
    for key, value in values.items():
        try:
            setattr(cache, key, value)
        except Exception:  # noqa: BLE001 - validated dataclass
            object.__setattr__(cache, key, value)


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
    scatter warm-up launch: one row naming page 1 with a rejected (nonzero)
    status (TXG-CAP-006, TXG-LIFE-004). The default encoder's one-row
    profile transaction also names page 1, of model-owned scratch, but
    commits its row (status 0). A count-1 cell's eager reference, warm-ups
    and capture also launch one rejected row naming page 1, so a caller that
    needs the warm-up alone bounds its window before the first cell."""
    return (
        call["rows"] == 1
        and call["pages"] == [1]
        and all(s != 0 for s in call["status"])
    )


class ScatterSpy:
    """Logs every masked-scatter launch (the plugin's one Triton kernel,
    ``state_scatter``) with its pool, rows, page IDs and statuses, wherever
    the module's executors are bound by name. CPU executes the CPU loop; a
    CUDA warm-up on a CPU pool is logged and skipped (the kernel cannot run
    there)."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []
        #: Raise from the ``n``-th transaction-graph scatter warm-up launch
        #: (``is_scatter_warmup``), TXG-MEM-010's warm-up phase.
        self.fail_warmup: int | None = None
        self.warmups = 0

    def _log(self, name: str, pool: Any, block_ids: Any, row_status: Any) -> None:
        entry = {
            "fn": name,
            "pool": pool.data_ptr(),
            "rows": int(block_ids.numel()),
            "pages": [int(v) for v in block_ids.reshape(-1).tolist()],
            "status": [int(v) for v in row_status.reshape(-1).tolist()],
        }
        self.calls.append(entry)
        self.events.append(("scatter", entry))

    def install(self, monkeypatch: Any) -> ScatterSpy:
        from vllm_nemotron_asr_streaming.nemotron_asr import state_scatter

        spy = self
        execute = state_scatter._execute_masked_page_scatter_
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
            if hasattr(module, "_execute_masked_page_scatter_"):
                monkeypatch.setattr(
                    module, "_execute_masked_page_scatter_", execute_spy
                )
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
    """Records every ``state_slots.fail_requests`` call (TXG-FAIL-008)."""

    def __init__(self, bindings: Any) -> None:
        self.calls: list[set[str]] = []
        core = h.core_bindings(bindings)
        original = core.fail_requests

        def fail_requests(request_ids):
            self.calls.append(set(request_ids))
            return original(request_ids)

        object.__setattr__(core, "fail_requests", fail_requests)

    @property
    def failed(self) -> set[str]:
        return set().union(*self.calls) if self.calls else set()


def slots_at_step_start(bindings: Any, slots: Any) -> dict[int, torch.Tensor]:
    """The pages of ``slots`` as core's slot initialization leaves them at
    the start of the next step (after ``activate_requests``), filled when
    that step runs. A rejected row's pages must equal these (TXG-FAIL-001,
    TXG-FAIL-008)."""
    core = h.core_bindings(bindings)
    original = core.activate_requests
    taken: dict[int, torch.Tensor] = {}

    def activate_requests(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        for slot in slots:
            taken[slot] = h.slot_bytes(bindings, slot).clone()
        object.__setattr__(core, "activate_requests", original)
        return result

    object.__setattr__(core, "activate_requests", activate_requests)
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
