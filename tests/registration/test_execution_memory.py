# SPDX-License-Identifier: Apache-2.0
"""Execution memory, the model-state interface and state-slot access
(REG-MEM-*, REG-MS-*, REG-STATE-*).

The model and its model state are the real classes on CPU, driven by
``support.model_harness.Runner`` (see its docstring for what is real, what is
omitted and which seams the doubles use): ``prepare_execution_memory`` once
before the first profiling step (the runner's ``profile_run`` calls it, as
core's does), a profiling run, later dummy steps (kernel warm-up, with
attention metadata), and complete real steps (``add_request`` through
``postprocess_state`` and ``take_failed_requests``) whose row transaction and
commit are the plugin's own. The
capture hooks are called through ``support.core_order.CoreOrder``.

Execution arms (``ARMS``), each at one server geometry: the eager arms
(``eager`` encoder, ``dense-eager`` decoder, the reference), ``compiled-static``,
a graphed decoder alone (``dense-graphed``), the card's default arms
(``eager-graphed`` encoder and ``dense-graphed`` decoder) and CHUNK
populations. CPU cannot capture graphs, so graphed arms run with
``FakeGraphRuntime``: core's forward context and batch descriptors, and a
wrapper with core's entry contract whose capture is logged and can be made to
fail or to store no graph.

The hook configuration (REG-MEM-015 to REG-MEM-027) runs on the same CPU
harness with ``enforce_eager`` off, vLLM's current platform reporting CUDA
(``is_cuda``), a ``support.fake_cuda.FakeStream`` for the ``stream`` core
passes, and the memory queries replaced by ``support.fake_cuda.FakeCuda``'s
(``install_memory``; free memory falls by ``per_segment_graph`` bytes per live
captured segment graph), so the measurement, comparison and failure rules are
exercised without a device. The GPU tests repeat the memory rules against
real free memory.

Expected graph inventories are derived from the configuration alone
(``expected_entries``), never from an observed run.

Model-internal seams these tests read (beyond the harness's): the model
state's step record (``_dummy_run`` / ``_is_profile``, REG-MEM-001, no public
accessor), the model's state layer (``_persistent_state_layer``, its
``pages`` and ``component_pools``), the
profile scratch (``build_profile_invocation(...).storage.spec``), the encoder
inventory's ``ready`` property (REG-MEM-003 inventory test). The GPU test
reads no model attribute: it finds graph entries by instrumenting
core's ``CUDAGraphWrapper.__call__``.

The weightless state-layer and bookkeeping tests (REG-STATE-*, REG-MS-003 to
REG-MS-006) are ``test_state_layer.py``; the ones here need the forward.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import inspect
import logging
import re
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support import model_harness as h
from support.core_order import PROFILING_POOL, SERVING_POOL, CoreOrder
from support.fake_cuda import MARGIN_BYTES, FakeCuda

from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import SAMPLED_TOKEN

pytestmark = pytest.mark.engine

LOOKAHEADS = (0, 1, 3, 6, 13)
CADENCES = ("80ms", "160ms", "320ms", "560ms", "1120ms")
#: The synthetic card's default lookahead.
CARD_DEFAULT = 3

EAGER_ARMS = {"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-eager"}

#: name → (configuration overrides, server lookahead or ``None`` for the
#: card default, max_num_seqs).
ARMS = {
    "eager": (dict(EAGER_ARMS), 6, 2),
    "compiled-static": (
        {
            "encoder_execution_arm": "compiled-static",
            "decode_dispatch_arm": "dense-eager",
        },
        0,
        1,
    ),
    "dense-graphed": (
        {"encoder_execution_arm": "eager", "decode_dispatch_arm": "dense-graphed"},
        6,
        2,
    ),
    # The card's default arms (REG-CARD-008) at the card's default geometry,
    # given explicitly so these tests do not depend on translation or on the
    # no-override resolution (tested in test_card_parser and
    # test_lookahead_validation).
    "default": (
        {
            "encoder_execution_arm": "eager-graphed",
            "decode_dispatch_arm": "dense-graphed",
        },
        CARD_DEFAULT,
        2,
    ),
    # CHUNK buckets accept populations 1, 31 and 63 only; with two
    # sequences the maximum shape (2 rows) differs from the CHUNK cell (1 row).
    "chunk": (
        {
            "decode_dispatch_arm": "dense-graphed",
            "encoder_execution_arm": "eager-graphed",
            "chunk_bucket_graph_populations": [1],
        },
        1,
        2,
    ),
}
GRAPHED = ("dense-graphed", "default", "chunk")
#: Arms the hook configuration accepts (REG-ENG-005): at least one
#: hook-captured graph, no compiled encoder and no CHUNK buckets.
HOOK_ARMS = ("dense-graphed", "default")


def geometry_of(right_context: int) -> int:
    return LOOKAHEADS.index(right_context)


def batch_tiers(maximum: int) -> list[int]:
    """The decoder's batch tiers for ``max_num_seqs`` (powers of two below the
    maximum, then the maximum)."""
    tiers, value = [], 1
    while value < maximum:
        tiers.append(value)
        value *= 2
    return [*tiers, maximum]


def expected_entries(env) -> int:
    """Graph entries the configuration requires, all at the server's one
    geometry: each batch tier of a graphed decoder; each CHUNK count of a
    graphed encoder, except the cells a CHUNK bucket owns; each CHUNK cell.
    An arm the overrides do not set is the card's default arm."""
    overrides = env.overrides
    encoder = overrides.get("encoder_execution_arm", "eager-graphed")
    decoder = overrides.get("decode_dispatch_arm", "dense-graphed")
    count = 0
    if decoder == "dense-graphed":
        count += len(env.served_geometries) * len(batch_tiers(env.max_num_seqs))
    if encoder in ("dense-graphed", "eager-graphed"):
        # One entry per CHUNK count 1..max_num_seqs. On the CHUNK arm
        # (max_num_seqs=2) those counts coincide with the decoder's batch
        # tiers (1, 2) only because max_num_seqs is 2; the two terms are
        # counted separately.
        cells = {
            (g, n)
            for g in env.served_geometries
            for n in range(1, env.max_num_seqs + 1)
        }
        count += len(cells - set(env.chunk_cells))
    return count + len(env.chunk_cells)


#: Free memory and the bytes each live captured segment graph holds.
DEVICE_BYTES = 1 << 40
PER_GRAPH = 3 << 20


@pytest.fixture
def build(monkeypatch, tmp_path):
    """``build(arm=..., **config)`` → the runner and its spies."""

    def make(arm="eager", overrides=None, before_state=None, **config):
        events = h.Events()
        profile = h.ProfileSpy(events).install(monkeypatch)
        advance = h.AdvanceSpy(events).install(monkeypatch)
        graphs = h.FakeGraphRuntime(events).install(monkeypatch)
        fake = FakeCuda(
            total=DEVICE_BYTES,
            free=DEVICE_BYTES,
            segment_graphs=graphs,
            per_segment_graph=PER_GRAPH,
        ).install_memory(monkeypatch)
        arm_overrides, right_context, max_num_seqs = ARMS[arm]
        merged = dict(arm_overrides) | dict(overrides or {})
        settings = {"right_context": right_context, "max_num_seqs": max_num_seqs}
        settings.update(config)
        card = h.write_runnable_card(tmp_path / f"card-{len(list(tmp_path.iterdir()))}")
        vllm_config = h.build_vllm_config(card, overrides=merged, **settings)
        model = h.build_model(vllm_config)
        if before_state is not None:
            before_state()
        state = h.build_model_state(vllm_config, model)
        served = (
            CARD_DEFAULT
            if settings["right_context"] is None
            else settings["right_context"]
        )
        geometry = geometry_of(served)
        runner = h.Runner(vllm_config, model, state)
        return SimpleNamespace(
            arm=arm,
            vllm_config=vllm_config,
            model=model,
            state=state,
            config=model.config,
            runner=runner,
            core=CoreOrder(runner),
            events=events,
            profile=profile,
            advance=advance,
            graphs=graphs,
            fake=fake,
            max_num_seqs=settings["max_num_seqs"],
            right_context=served,
            max_geometry=geometry,
            served_geometries=[geometry],
            overrides=merged,
            chunk_cells=[
                (geometry, n) for n in merged.get("chunk_bucket_graph_populations", [])
            ],
        )

    return make


def pool(env, *, spy: bool = True, capacity: int = 8):
    bindings = h.real_bindings(h.model_spec(env.model), capacity=capacity)
    return env.runner.bind_pool(h.BindingsSpy(bindings) if spy else bindings)


def trapped_pool(env):
    """Bind a pool, then let the model reach its state layer, and the model
    state its state-group layouts (which hold the same views), only through
    an armed ``support.state_layer.ViewTrap``: any read of the layer's views
    or pages, or use of the layouts, fails and is recorded (REG-MEM-005,
    REG-MEM-009, REG-MEM-012, REG-MEM-013). ``trap._target`` is the bound
    pool."""
    from support import state_layer as sl

    bindings = pool(env, spy=False)
    trap = sl.trap_views(env.model, h.core_bindings(bindings), env.state)
    trap.armed = True
    return trap


def stream(
    env,
    req_id: str,
    *,
    geometry: int | None = None,
    sequence: int = 0,
    final: bool = False,
    valid_samples: int | None = None,
):
    """One stream unit: a full, non-final chunk by default. A final tail
    (``final=True``) carries fewer samples than a full chunk (half of one
    unless ``valid_samples`` says otherwise)."""
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        CADENCES as MANIFEST_CADENCES,
    )
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        RAW_SAMPLES_PER_CHUNK,
    )

    geometry = env.max_geometry if geometry is None else geometry
    if final and valid_samples is None:
        width = RAW_SAMPLES_PER_CHUNK[tuple(MANIFEST_CADENCES)[geometry]]
        valid_samples = width // 2
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=h.stream_envelope(
            env.config,
            geometry=geometry,
            sequence=sequence,
            final=final,
            valid_samples=valid_samples,
        ),
        # each unit is its own multimodal item (its own encoder cache entry)
        identifier=f"uuid-{req_id}-{sequence}",
    )


def file(env, req_id: str, n_samples: int = 4000):
    samples = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
    return h.new_request(
        req_id,
        int(env.config.file_carrier_token_id),
        samples=samples * 0.1,
        header=[env.max_geometry, 0, n_samples],
    )


def rows_of(env, kind: str):
    rows, handles = [], {}
    if kind in ("stream", "mixed"):
        rows.append(stream(env, "s1"))
        handles["s1"] = 1
    if kind in ("file", "mixed"):
        rows.append(file(env, "f1"))
        handles["f1"] = 2
    return h.scheduler_output(new=tuple(rows), handles=handles)


def assert_dummy_output(output, embeds):
    """REG-MEM-010: the input embeddings' token count, width, dtype and
    device, with a valid decision ID (zero) in the carrier column."""
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    assert isinstance(output, torch.Tensor)
    assert output.shape == embeds.shape
    assert output.dtype == embeds.dtype
    assert output.device == embeds.device
    carrier = read_decision_carrier(output)
    assert torch.equal(carrier, torch.zeros_like(carrier))


def max_shapes(env) -> set[tuple[int, int]]:
    return {(env.max_num_seqs, env.max_geometry), *((n, g) for g, n in env.chunk_cells)}


def profiled_shapes(env, calls=None) -> set[tuple[int, int]]:
    return {
        (c.get("num_rows"), c.get("geometry_id"))
        for c in (env.profile.calls if calls is None else calls)
    }


def shape_of(kwargs) -> tuple:
    return (kwargs.get("num_rows"), kwargs.get("geometry_id"))


def is_final(kwargs) -> bool:
    """Whether a call of the profile function is the final, readiness-gated
    profile transaction rather than a warm-up invocation. Seam: the
    preparation path passes ``ready_domain=True`` to
    ``run_persistent_state_profile`` for the final transaction (which then
    runs against the sealed, ready inventory: ``profile_ready_cell``), while
    warm-up (``warmup_static_encoder_execution``) and the retired
    forward-time profile do not."""
    return bool(kwargs.get("ready_domain"))


def skip_evidence(env) -> bool:
    """A selected entry was left without a graph: recorded at skip time, and
    its wrapper is either gone (discarded) or still holds no graph."""
    graphs = env.graphs
    return (
        graphs.skipped is not None
        and graphs.skip_stored_nothing
        and graphs.skipped_entry_has_graph() is not True
    )


def assert_bindings_valid(spy, output, since: int = 0):
    """Every row binding of the step (from index ``since``; recorded by the
    harness at ``prepare_inputs``) names the model's layer, requests of this
    step in the step's order, and the slots core gave them: a new request's
    ``block_ids``."""
    order = [r.req_id for r in output.scheduled_new_reqs]
    order += list(output.scheduled_cached_reqs.req_ids)
    slots = {r.req_id: r.block_ids[0][0] for r in output.scheduled_new_reqs}
    for layer, ids, handles, _ in spy.bind_calls[since:]:
        assert layer == h.STATE_LAYER
        assert list(ids) == [r for r in order if r in ids], (ids, order)
        assert all(
            handle.slot == slots[r] for r, handle in zip(ids, handles) if r in slots
        )


# ===========================================================================
# Model-state interface
# ===========================================================================


def _calls_accepted_by(signature: inspect.Signature):
    """Calls a signature accepts, as ``(args, kwargs)`` of sentinels: required
    only; every parameter (positionals positionally); every parameter by
    keyword where the kind allows it."""
    required, every, keyword = ([], {}), ([], {}), ([], {})
    for name, param in list(signature.parameters.items())[1:]:  # drop self
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        value = object()
        is_required = param.default is param.empty
        if param.kind is param.KEYWORD_ONLY:
            every[1][name] = value
            keyword[1][name] = value
            if is_required:
                required[1][name] = value
            continue
        every[0].append(value)
        if param.kind is param.POSITIONAL_ONLY:
            keyword[0].append(value)
        else:
            keyword[1][name] = value
        if is_required:
            required[0].append(value)
    return [required, every, keyword]


# @spec REG-MS-001
def test_every_model_state_override_accepts_every_call_of_the_base_method():
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        NemotronASRModelState,
    )

    mismatched = []
    for name, member in vars(NemotronASRModelState).items():
        base = getattr(ModelState, name, None)
        if name.startswith("__") and name != "__init__":
            continue
        if not (inspect.isfunction(member) and inspect.isfunction(base)):
            continue
        override, original = inspect.signature(member), inspect.signature(base)
        for args, kwargs in _calls_accepted_by(original):
            try:
                override.bind(None, *args, **kwargs)
            except TypeError as error:
                mismatched.append(f"{name}: {error}")
        has_var = any(
            p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in override.parameters.values()
        )
        for pname, param in original.parameters.items():
            mine = override.parameters.get(pname)
            if mine is None:
                if not has_var:
                    mismatched.append(f"{name}: parameter {pname} missing")
                continue
            if mine.kind != param.kind:
                mismatched.append(f"{name}: kind of {pname} differs")
            if mine.default != param.default:
                mismatched.append(f"{name}: default of {pname} differs")
    assert not mismatched, "\n".join(mismatched)


@pytest.mark.parametrize("with_ids", [False, True], ids=["input_ids=None", "tensor"])
# @spec REG-MS-002
def test_dummy_inputs_embeds_returns_the_preallocated_buffer_slice(build, with_ids):
    env = build()
    buffer = env.state.encoder_runner.inputs_embeds
    num_tokens = 5
    input_ids = (
        torch.randint(0, int(env.config.vocab_size), (num_tokens,))
        if with_ids
        else None
    )
    embeds = env.state.dummy_inputs_embeds(num_tokens, input_ids)
    assert embeds.shape == (num_tokens, buffer.shape[1])
    assert embeds.data_ptr() == buffer.data_ptr()


# @spec REG-MS-002
def test_dummy_inputs_embeds_is_called_as_core_calls_it_on_warm_up(build):
    env = build()
    _, output, embeds = env.runner.profile_run(randomize_inputs=True, attn=True)
    assert embeds.data_ptr() == env.state.encoder_runner.inputs_embeds.data_ptr()
    assert output.shape == embeds.shape


# ===========================================================================
# Execution memory: profiling and dummy steps
# ===========================================================================


def is_profiling_record(state) -> bool:
    """Whether the model state classifies the current step as a profiling
    dummy step. There is no public accessor; this reads the model state's
    record (seam: today's ``_dummy_run`` and ``_is_profile``)."""
    return bool(state._dummy_run) and bool(state._is_profile)


@pytest.mark.parametrize("earlier", [None, "profiling"], ids=["fresh", "after"])
@pytest.mark.parametrize("kind", ["profiling", "dummy"])
# @spec REG-MEM-001
def test_prepare_inputs_records_the_step_kind_and_prepares_nothing(
    build, earlier, kind
):
    """``prepare_inputs`` classifies a dummy batch (no ``prepare_attn``
    since the last ``prepare_inputs``: the profiling dummy step), replacing
    an earlier record, and prepares no execution (no profile, no capture)."""
    env = build()
    pool(env)
    batch = h.dummy_batch(env.state, 2, 2)
    if earlier is not None:
        env.state.prepare_inputs(batch, env.runner.req_states)
        assert is_profiling_record(env.state)
    if kind == "dummy":
        env.runner._attn(batch)
    env.state.prepare_inputs(batch, env.runner.req_states)
    assert is_profiling_record(env.state) is (kind == "profiling")
    assert env.profile.calls == [] and env.graphs.captures == []


@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-002
def test_first_profiling_forward_prepares_in_order_at_maximum_shape(build, arm):
    """Warm-up and capture of the whole configured inventory, then every
    required profile (maximum shape and each CHUNK cell) with every expected
    graph retained, then the dummy output."""
    env = build(arm=arm)
    expected = expected_entries(env)
    retained: dict[tuple, list[int]] = {}

    def record(kwargs):
        if is_final(kwargs):
            retained.setdefault(shape_of(kwargs), []).append(len(env.graphs.captured()))

    env.profile.before = record
    _, output, embeds = env.runner.profile_run(core_embeds=False)
    env.events.append(("returned", None))
    assert_dummy_output(output, embeds)
    assert len(env.graphs.captures) == expected, (len(env.graphs.captures), expected)
    assert len(env.graphs.captured()) == expected
    kinds = env.events.kinds()
    last_capture = max((i for i, k in enumerate(kinds) if k == "capture"), default=-1)
    returned = kinds.index("returned")
    # Warm-up invocations of the profile function may run anywhere. Every
    # final transaction runs after capture is complete (the last capture),
    # with every expected graph retained, at a required shape, and each
    # required shape has one; none runs before capture is complete.
    finals = [
        (i, shape_of(info))
        for i, (k, info) in enumerate(env.events)
        if k == "profile" and is_final(info)
    ]
    for index, shape in finals:
        assert index > last_capture, (
            f"a final profile transaction at {shape} ran before capture completed"
        )
        assert index < returned
        assert shape in max_shapes(env), f"a final transaction at {shape}"
    assert {shape for _, shape in finals} == max_shapes(env), finals
    for shape, counts in retained.items():
        assert all(n == expected for n in counts), (
            f"a final transaction at {shape} ran without every expected graph"
        )


@pytest.mark.parametrize("arm", GRAPHED)
# @spec REG-MEM-002
def test_first_profiling_forward_refuses_an_entry_left_without_a_graph(build, arm):
    """Each expected graph entry (counted from the configuration), in turn, is
    left without a graph in its wrapper during capture; preparation fails."""
    probe = build(arm=arm)
    expected = expected_entries(probe)
    assert expected > 0
    for index in range(expected):
        env = build(arm=arm)
        env.graphs.skip_capture = index
        error = h.call_with(env.runner.profile_run, core_embeds=False)
        assert len(env.graphs.captures) > index, (
            f"capture {index} was never attempted (no preparation)"
        )
        assert skip_evidence(env), f"entry {index} was not left without a graph"
        assert error is not None, f"preparation accepted entry {index} without a graph"
        # Readiness gates the final transaction (the ordering rule above): no
        # final transaction may run with an entry missing. Warm-up
        # invocations are allowed.
        assert not [
            info for k, info in env.events if k == "profile" and is_final(info)
        ], f"a final profile transaction ran although entry {index} had no graph"


# @spec REG-MEM-002, REG-MEM-011
def test_first_profiling_forward_leaves_the_worker_ready_to_serve(build):
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    env.runner.real_step(rows_of(env, "stream"))
    assert env.advance.calls, "the prepared worker did not execute the step"


@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-003
def test_later_profiling_steps_after_the_pool_exists_check_and_return(build, arm):
    """E5: kernel warm-up runs further profiling dummy steps after the pool
    exists, with random token IDs and attention metadata, several times."""
    env = build(arm=arm)
    env.runner.profile_run(core_embeds=False)
    profiles, captures = len(env.profile.calls), len(env.graphs.captures)
    trap = trapped_pool(env)
    for _ in range(3):
        batch, output, embeds = env.runner.profile_run(
            randomize_inputs=True, attn=True, core_embeds=False
        )
        assert_dummy_output(output, embeds)
    assert len(env.profile.calls) == profiles, "a later profiling step profiled"
    assert len(env.graphs.captures) == captures, "a later profiling step captured"
    assert trap.reads == []


@pytest.mark.parametrize("arm", GRAPHED)
# @spec REG-MEM-003
def test_later_profiling_step_refuses_a_wrapper_entry_without_a_graph(build, arm):
    env = build(arm=arm)
    env.runner.profile_run(core_embeds=False)
    captured = env.graphs.captured()
    assert captured
    for wrapper, descriptor in captured:
        entry = wrapper.concrete_cudagraph_entries[descriptor]
        graph, entry.cudagraph = entry.cudagraph, None
        attempts = len(env.graphs.captures)
        error = h.call_with(env.runner.profile_run, core_embeds=False)
        assert error is not None, f"later step accepted {descriptor} without a graph"
        assert len(env.graphs.captures) == attempts, "a later step captured"
        entry.cudagraph = graph


# @spec REG-MEM-003
def test_later_profiling_step_refuses_an_inventory_that_is_no_longer_ready(
    build, monkeypatch
):
    from vllm_nemotron_asr_streaming.nemotron_asr.encoder_execution import (
        ResolvedEncoderExecution,
    )

    env = build(arm="compiled-static")
    env.runner.profile_run(core_embeds=False)
    monkeypatch.setattr(ResolvedEncoderExecution, "ready", property(lambda s: False))
    profiles = len(env.profile.calls)
    error = h.call_with(env.runner.profile_run, core_embeds=False)
    assert error is not None, "a later profiling step skipped the readiness check"
    assert len(env.profile.calls) == profiles


def _inject(env, where: str) -> None:
    """Fail preparation part-way, by condition rather than position: the first
    eager warm-up made after one has completed; the first capture made after
    one has completed; a selected entry that never holds a graph (readiness);
    the first final profile transaction made once every expected graph is
    captured."""
    if where == "warm-up":
        env.graphs.fail_eager_after = 1
    elif where == "capture":
        env.graphs.fail_capture_after = 1
    elif where == "readiness":
        env.graphs.skip_capture = 1
    elif where == "profile":
        expected = expected_entries(env)

        def fail(kwargs):
            if (
                not env.profile_failed
                and is_final(kwargs)
                and len(env.graphs.captured()) >= expected
            ):
                env.profile_failed = True
                raise RuntimeError("profile transaction failed")

        env.profile_failed = False
        env.profile.before = fail


def _reached(env, where: str) -> bool:
    """The injection point was reached after demonstrable partial
    preparation."""
    graphs = env.graphs
    if where == "warm-up":
        return graphs.failed_eager is not None and graphs.eager_completed >= 1
    if where == "capture":
        return graphs.failed_capture is not None and graphs.captures_completed >= 1
    if where == "readiness":
        return graphs.captures_completed >= 1 and skip_evidence(env)
    return env.profile_failed


def _attempts(env) -> int:
    return env.graphs.eager_calls + len(env.graphs.captures) + len(env.profile.calls)


def _release(error: BaseException | None) -> None:
    """Clear every traceback reachable from an exception through both
    ``__cause__`` and ``__context__``."""
    stack, seen = [error], set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        stack.extend((current.__cause__, current.__context__))


@pytest.mark.parametrize("arm", GRAPHED)
@pytest.mark.parametrize("where", ["warm-up", "capture", "readiness", "profile"])
# @spec REG-MEM-004
def test_failed_preparation_discards_its_state_and_stays_failed(build, arm, where):
    env = build(arm=arm)
    _inject(env, where)
    first = h.call_with(env.runner.profile_run, core_embeds=False)
    assert _reached(env, where), f"the {where} injection point was not reached"
    assert first is not None, f"preparation passed despite a {where} failure"
    _release(first)
    del first
    env.graphs.disarm()
    env.profile.fail = None
    env.profile.before = None
    gc.collect()
    assert env.graphs.live_graphs() == 0, (
        "a graph of the failed preparation is still referenced"
    )
    attempts = _attempts(env)
    for _ in range(2):
        later = h.call_with(env.runner.profile_run, core_embeds=False)
        assert later is not None, "a profiling step after a failure passed"
        _release(later)
    assert _attempts(env) == attempts, "a failed worker prepared again"


# @spec REG-MEM-004
def test_failed_preparation_on_the_eager_arms_stays_failed(build):
    env = build()
    env.profile.fail = RuntimeError("profile transaction failed")
    assert h.call_with(env.runner.profile_run, core_embeds=False) is not None
    calls = len(env.profile.calls)
    assert calls >= 1
    env.profile.fail = None
    assert h.call_with(env.runner.profile_run, core_embeds=False) is not None
    assert len(env.profile.calls) == calls


# @spec REG-MEM-004, REG-MEM-011
def test_failed_preparation_refuses_real_steps(build):
    env = build()
    env.profile.fail = RuntimeError("profile transaction failed")
    h.call_with(env.runner.profile_run, core_embeds=False)
    pool(env)
    error = h.call_with(env.runner.real_step, rows_of(env, "stream"))
    assert error is not None
    assert env.advance.calls == []


@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-005
def test_preparation_reads_no_state_slot_bindings_or_pool(build, arm):
    env = build(arm=arm)
    trap = trapped_pool(env)
    pool_bytes = h.pool_tensor(trap._target).clone()
    env.runner.profile_run(core_embeds=False)
    finals = [c for c in env.profile.calls if is_final(c)]
    assert max_shapes(env) <= profiled_shapes(env, finals)
    assert trap.reads == []
    assert torch.equal(pool_bytes, h.pool_tensor(trap._target))


# @spec REG-MEM-005
def test_profile_scratch_is_built_from_the_models_state_spec(build):
    from vllm_nemotron_asr_streaming.nemotron_asr.profile_execution import (
        build_profile_invocation,
    )

    env = build()
    invocation = build_profile_invocation(
        env.config, num_rows=2, device=torch.device("cpu"), geometry_id=3
    )

    def layout(spec):
        return spec.page_size_bytes, [
            (d.name, d.shape, d.dtype, d.offset_bytes) for d in spec.descriptors
        ]

    assert layout(invocation.storage.spec) == layout(h.model_spec(env.model))


# @spec REG-MEM-008
def test_first_profiling_step_runs_no_forward_time_profile(build):
    """A profiling step with fewer dummy rows than ``max_num_seqs``: no
    profile is sized to the dummy batch."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(num_tokens=1, core_embeds=False)
    assert profiled_shapes(env) == max_shapes(env), env.profile.calls
    assert all(is_final(c) for c in env.profile.calls), env.profile.calls


@pytest.mark.parametrize("prepared", [False, True], ids=["before", "after"])
# @spec REG-MEM-009, REG-MEM-010
def test_non_profiling_dummy_step_returns_the_dummy_output_only(build, prepared):
    env = build(arm="dense-graphed")
    if prepared:
        env.runner.profile_run(core_embeds=False)
    trap = trapped_pool(env)
    profiles, captures = len(env.profile.calls), len(env.graphs.captures)
    _, output, embeds = env.runner.dummy_step(
        is_profile=False, randomize_inputs=True, attn=True, core_embeds=False
    )
    assert len(env.profile.calls) == profiles
    assert len(env.graphs.captures) == captures
    assert_dummy_output(output, embeds)
    assert trap.reads == []


@pytest.mark.parametrize("step", ["first-profiling", "later-profiling", "dummy"])
# @spec REG-MEM-010
def test_dummy_output_matches_the_input_embeddings(build, step):
    env = build()
    if step != "first-profiling":
        env.runner.profile_run(core_embeds=False)
    _, output, embeds = env.runner.dummy_step(
        is_profile=step != "dummy", core_embeds=False
    )
    assert_dummy_output(output, embeds)


@pytest.mark.parametrize("pool_exists", [False, True], ids=["no-pool", "pool"])
@pytest.mark.parametrize("is_profile", [False, True], ids=["dummy", "profiling"])
# @spec REG-MEM-012
def test_dummy_prepare_inputs_after_a_real_step_reads_nothing_retained(
    build, pool_exists, is_profile
):
    """E15: core calls ``prepare_inputs`` on dummy steps after real steps;
    nothing of the previous real step's rows reaches the dummy step."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    h.drive_step(env.state, rows_of(env, "mixed"))
    trap = trapped_pool(env) if pool_exists else None
    if not is_profile:
        # core builds attention metadata for every dummy step but the
        # memory profile's
        env.runner._attn(h.dummy_batch(env.state, 2, 2))
    batch = h.dummy_batch(env.state, 2, 2)
    kwargs = env.state.prepare_inputs(batch, env.runner.req_states)
    assert "'s1'" not in repr(kwargs) and "'f1'" not in repr(kwargs)
    if trap is not None:
        assert trap.reads == []


# @spec REG-MEM-013, REG-MEM-010
def test_dummy_sampling_reads_no_state_slots(build):
    """E13: core indexes the profiling output by ``logits_indices`` and runs
    ``_dummy_sampler_run`` (the model state's ``dummy_sampler_run``, then two
    sampler passes with ``add_request`` / ``apply_staged_writes`` between)
    through the sampler the model state installs (none: native burst, the
    only custom sampler, is refused on the state layer, REG-ENG-019)."""
    env = build()
    batch, output, embeds = env.runner.profile_run(core_embeds=False)
    assert_dummy_output(output, embeds)
    trap = trapped_pool(env)
    base = h.RecordingSampler()
    hooks = env.state.custom_sampler(base)
    assert hooks is None
    sampler = base
    result = env.runner.dummy_sampler_run(sampler, output[batch.logits_indices])
    if result is not None:
        for sampled in result:
            assert sampled.sampled_token_ids.shape[0] == len(batch.logits_indices)
    assert "add_request" in base.calls and "apply_staged_writes" in base.calls
    assert trap.reads == []


# ===========================================================================
# Execution memory: real steps
# ===========================================================================


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-MEM-011
def test_real_step_before_preparation_is_refused_without_preparing(build, kind):
    """E14: no real step that runs model execution is served before
    preparation succeeded, whatever its rows."""
    env = build()
    spy = pool(env)
    output = rows_of(env, kind)
    error = h.call_with(env.runner.real_step, output)
    assert error is not None, "a real step ran before preparation"
    assert env.profile.calls == [] and env.graphs.captures == [], "prepared lazily"
    assert env.advance.calls == [], "the row transaction ran"
    assert_bindings_valid(spy, output)


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-MEM-011
def test_first_real_step_after_preparation_runs(build, kind):
    env = build()
    env.runner.profile_run(core_embeds=False)
    profiles = len(env.profile.calls)
    spy = pool(env)
    output = rows_of(env, kind)
    env.runner.real_step(output)
    assert env.advance.calls, "the step was not executed"
    assert len(env.profile.calls) == profiles, "a real step prepared again"
    assert spy.bind_calls
    assert_bindings_valid(spy, output)


def cached_step(env, cached, computed, *, new=(), handles):
    """A step as core builds it without pipeline parallelism: cached requests
    carry no token slices; ``computed`` is each one's ``num_computed_tokens``."""
    return h.scheduler_output(
        new=tuple(new), cached=tuple(cached), handles=handles, computed=computed
    )


def assert_commits_clean(env, *, steps: int):
    """Every row of every real step so far committed: status zero, none failed."""
    log = env.runner.commit_log
    assert len(log) == steps, log
    for statuses, failed in log:
        assert not failed, log
        assert all(status == 0 for status in statuses.values()), log


def drain_to_park(
    env, ids, handles, *, watch: str = "s1", limit: int = 300, start: int = 1
):
    """Cached steps from ``num_computed_tokens`` ``start`` (a first turn's
    second position by default), each fed the token the model emitted on the
    previous one, until the ``watch`` row's decision is the park token (a legal
    park); returns the next ``num_computed_tokens``."""
    eos = int(env.config.eos_token_id)
    position = start
    while env.runner._sampled[watch] != eos:
        assert position < limit, "the turn never reached the park token"
        out = cached_step(env, ids, {r: position for r in ids}, handles=handles)
        env.runner.real_step(out)
        position += 1
    return position


# @spec REG-STATE-008
@pytest.mark.parametrize("kind", ["stream", "file", "mixed"])
def test_consecutive_cached_steps_run_on_core_scheduler_output(build, kind):
    """Core sends no ``new_token_ids`` for cached requests (no pipeline
    parallelism): every step after a request's first still runs, the host
    reading its prompt token or marking a sampled one."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    first = rows_of(env, kind)
    assert first.scheduled_cached_reqs.new_token_ids == []
    env.runner.real_step(first)
    ids = [r.req_id for r in first.scheduled_new_reqs]
    handles = {r: (1 if r == "s1" else 2) for r in ids}
    for step_no in range(1, 4):
        out = cached_step(env, ids, {r: step_no for r in ids}, handles=handles)
        assert out.scheduled_cached_reqs.new_token_ids == []
        env.runner.real_step(out)
        # a token the sampler produced on the device, never a carrier
        assert env.state._scheduled_tokens == {r: (SAMPLED_TOKEN,) for r in ids}
        assert_commits_clean(env, steps=step_no + 1)


# @spec REG-STATE-008
def test_session_turn_arrives_as_a_new_request_between_cached_steps(build):
    """A later stream unit is a new request delivery (core resets its computed
    tokens) once the preceding turn drained to EOS; it is a CHUNK again while a
    file row beside it keeps decoding."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    handles = {"s1": 1, "f1": 2}
    env.runner.real_step(
        h.scheduler_output(
            new=(stream(env, "s1"), file(env, "f1", 60000)),
            handles=handles,
        )
    )
    # The turn's own sampled tokens are fed back until the stream row parks.
    position = drain_to_park(env, ["s1", "f1"], handles)
    steps = position
    assert_commits_clean(env, steps=steps)
    file_before = dataclasses.replace(env.state._file_progress["f1"])
    env.runner.real_step(
        cached_step(
            env,
            ["f1"],
            {"f1": position},
            new=[stream(env, "s1", sequence=1)],
            handles=handles,
        )
    )
    chunk = int(env.config.audio_chunk_token_id)
    assert env.state._scheduled_tokens == {"s1": (chunk,), "f1": (SAMPLED_TOKEN,)}
    assert_commits_clean(env, steps=steps + 1)
    assert "f1" not in env.state._failed_files
    assert env.state._file_progress["f1"] != file_before, "the file row stalled"


def row_batch(req_ids, computed, *, scheduled: int = 1):
    """The ``InputBatch`` fields the model state reads for host tokens."""
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
        num_scheduled_tokens=np.full(len(req_ids), scheduled, dtype=np.int32),
    )


def prompted(req_id: str, prompt: list[int]):
    return dataclasses.replace(
        h.new_request(req_id, prompt[0]),
        prompt_token_ids=list(prompt),
        prefill_token_ids=list(prompt),
    )


# @spec REG-STATE-008
def test_cached_row_reads_a_real_prompt_token(build):
    """A two-token prompt delivered with one token scheduled: the cached step at
    one computed token takes the prompt's second token, not a sampled marker."""
    env = build()
    first, second = 7, 9
    assert {first, second}.isdisjoint(
        {int(env.config.file_carrier_token_id), int(env.config.audio_chunk_token_id)}
    )
    new = h.scheduler_output(
        new=(prompted("r1", [first, second]),), encoder_inputs={}, handles={"r1": 1}
    )
    h.drive_step(env.state, new)
    assert env.state._scheduled_tokens == {"r1": (first,)}
    assert env.state._row_tokens(row_batch(["r1"], [1])) == [(second,)]


# @spec REG-STATE-008
def test_cached_window_crossing_the_end_of_the_prompt_is_refused(build):
    env = build()
    new = h.scheduler_output(
        new=(prompted("r1", [7, 9]),), encoder_inputs={}, handles={"r1": 1}
    )
    h.drive_step(env.state, new)
    error = h.call_with(env.state._row_tokens, row_batch(["r1"], [1], scheduled=2))
    assert isinstance(error, ValueError) and "r1" in str(error), error


@pytest.mark.parametrize("prepared", [False, True], ids=["before", "after"])
# @spec REG-MEM-014
def test_cleanup_only_real_step_completes(build, prepared):
    """A step that only finishes a request (core removes it, then returns
    before any forward work) completes whether or not preparation ran."""
    env = build()
    if prepared:
        env.runner.profile_run(core_embeds=False)
    profiles = len(env.profile.calls)
    pool(env)
    env.state.add_request(0, h.with_slot(stream(env, "gone"), 3))
    output = h.scheduler_output(finished=("gone",))
    assert output.total_num_scheduled_tokens == 0
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert len(env.profile.calls) == profiles and env.advance.calls == []


# ===========================================================================
# State-slot access
# ===========================================================================


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-002, REG-STATE-007
def test_slots_come_only_from_the_delivered_block_ids(build, kind):
    """Over a complete step: each row's slot is its ``NewRequestData``'s one
    block, and the model reads nothing of core's pool object (only the
    state layer's views)."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    output = rows_of(env, kind)
    env.runner.real_step(output)
    assert spy.reads == [], spy.reads
    assert spy.bind_calls
    assert_bindings_valid(spy, output)


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-003
def test_rows_are_bound_with_the_steps_order_and_slots(build, kind):
    # three requests in the mixed case: three sequences
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    new = (
        (stream(env, "s1"), file(env, "f1"), stream(env, "s2"))
        if kind == "mixed"
        else tuple(rows_of(env, kind).scheduled_new_reqs)
    )
    slots = {"s1": 3, "f1": 1, "s2": 2}
    output = h.scheduler_output(
        new=new, handles={r.req_id: slots[r.req_id] for r in new}
    )
    env.runner.real_step(output)
    assert spy.reads == [], spy.reads
    assert spy.bind_calls
    assert_bindings_valid(spy, output)
    bound = {r for call in spy.bind_calls for r in call[1]}
    assert bound == set(output.num_scheduled_tokens)


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-005
def test_the_step_projects_the_layers_views_and_the_models_layout(
    build, monkeypatch, kind
):
    """The projection of a real step receives the state layer's views as
    bound at that step and the model's ``StateLayout``, and nothing else."""
    from vllm_nemotron_asr_streaming.nemotron_asr import model_state_v2

    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    calls = []
    real = model_state_v2.project_nemotron_state_pools

    def spy(views, layout, **kwargs):
        calls.append((views, layout, dict(kwargs)))
        return real(views, layout, **kwargs)

    monkeypatch.setattr(model_state_v2, "project_nemotron_state_pools", spy)
    env.runner.real_step(rows_of(env, kind))
    layer = env.model._persistent_state_layer
    assert calls, "the step projected no views"
    for views, layout, kwargs in calls:
        assert views is layer.component_pools or dict(views) == dict(
            layer.component_pools
        )
        assert layout is h.model_spec(env.model)
        assert set(kwargs) <= {"n_layers"}, kwargs


def _commit(env, kind: str):
    """A complete step through the production commit path. The pool is
    snapshotted inside the row transaction's entry (after the forward
    initialized the fresh pages, before the transaction runs)."""
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    output = rows_of(env, kind)
    slots = [1, 2] if kind == "mixed" else [1 if kind == "stream" else 2]
    seen: dict = {}

    def before(kwargs):
        seen["pool"] = h.pool_tensor(spy).clone()

    env.advance.before = before
    error = h.call_with(env.runner.real_step, output)
    return spy, seen, slots, error


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-003, REG-STATE-004, REG-STATE-006
def test_commit_writes_the_bound_slots_and_no_null_page(build, kind):
    """A step commits state into each bound slot and no other, the null
    page (block 0) included."""
    env = build()
    spy, seen, slots, error = _commit(env, kind)
    assert error is None, repr(error)
    assert "pool" in seen, "the row transaction did not run"
    before, after = seen["pool"], h.pool_tensor(spy)
    page = h.model_spec(env.model).page_size_bytes
    for slot in range(len(after) // page):
        changed = not torch.equal(
            before[slot * page : (slot + 1) * page],
            after[slot * page : (slot + 1) * page],
        )
        assert changed is (slot in slots), f"slot {slot}: written={changed}"


# @spec REG-STATE-006, FILE-BUF-004
def test_file_rows_on_recycled_slots_and_mixed_batches(build):
    """E7: a file whose slot is recycled to a new file (a new admission
    generation), sharing its step with a stream, binds and commits like
    stream rows."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    first = h.scheduler_output(new=(file(env, "f1"),), handles={"f1": 1})
    env.runner.real_step(first)
    after_first = h.slot_bytes(spy, 1).clone()
    recycled = h.scheduler_output(
        new=(file(env, "f2", 6000), stream(env, "s1")),
        finished=("f1",),
        handles={"f2": 1, "s1": 2},
    )
    since = len(spy.bind_calls)
    env.runner.real_step(recycled)
    assert_bindings_valid(spy, recycled, since)
    bound = {r for call in spy.bind_calls[since:] for r in call[1]}
    assert bound == {"f2", "s1"}
    assert not torch.equal(after_first, h.slot_bytes(spy, 1))


# ---------------------------------------------------------------------------
# Row failure through take_failed_requests (REG-MS-003, REG-MS-004,
# SESS-MODEL-003) and fresh pages of failed rows (REG-STATE-010)


def _foreign_geometry(env) -> int:
    return (env.max_geometry + 1) % len(LOOKAHEADS)


def _assert_initialized(env, slot: int) -> None:
    """Every state of ``slot``'s page holds its initializer: zero, or the
    book's blank label."""
    layer = env.model._persistent_state_layer
    blank = int(env.config.num_asr_labels)
    for descriptor in h.model_spec(env.model).descriptors:
        row = layer.component_pools[descriptor.name][slot]
        expected = 0 if descriptor.zero_initialized else blank
        assert row.eq(expected).all(), (slot, descriptor.name)


# @spec SESS-MODEL-003, REG-MS-003, REG-MS-004
def test_a_nonzero_collected_status_fails_that_request_alone(build):
    """The model's collection reports one row nonzero: ``postprocess_state``
    raises nothing, ``take_failed_requests`` returns that request once, and
    the step's other row committed."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)

    def fail_s1(result):
        statuses = dict(result.statuses) | {"s1": 1}
        return h.replace_collection(
            result, statuses=statuses, failed=set(result.failed) | {"s1"}
        )

    env.runner.collect_hook = fail_s1
    output = h.scheduler_output(
        new=(stream(env, "s1"), stream(env, "s2")), handles={"s1": 1, "s2": 2}
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"s1"}]
    statuses, failed = env.runner.commit_log[-1]
    assert statuses.get("s2") == 0 and "s2" not in failed
    assert env.state.take_failed_requests() is None, "the record was not cleared"


# @spec SESS-MODEL-001, SESS-MODEL-002, REG-STATE-010, TXG-FAIL-009
def test_an_all_failed_step_initializes_its_fresh_page_and_fails_alone(build):
    """A first unit at another geometry fails its row at planning; the step
    is all-failed, yet the fresh page is initialized (no earlier request's
    state shows), and the request ends through ``take_failed_requests``."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    env.model._persistent_state_layer.pages[3].fill_(-1)
    output = h.scheduler_output(
        new=(stream(env, "s1", geometry=_foreign_geometry(env)),), handles={"s1": 3}
    )
    error = h.call_with(env.runner.real_step, output)
    _assert_initialized(env, 3)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"s1"}]


# @spec SESS-MODEL-002, REG-STATE-010
def test_a_failed_fresh_row_beside_a_clean_one_keeps_its_initialized_page(build):
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    env.model._persistent_state_layer.pages[3].fill_(-1)
    output = h.scheduler_output(
        new=(stream(env, "bad", geometry=_foreign_geometry(env)), stream(env, "s1")),
        handles={"bad": 3, "s1": 1},
    )
    error = h.call_with(env.runner.real_step, output)
    _assert_initialized(env, 3)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


#: A label token: neither a carrier nor a control token.
LABEL = 7


def _prompted(request, prompt):
    """``request`` with the prompt ``prompt``; its features are unchanged."""
    return dataclasses.replace(
        request, prompt_token_ids=list(prompt), prefill_token_ids=list(prompt)
    )


def _unit_at(env, req_id: str, offset: int):
    """A regular unit's feature at ``offset``, with its own identifier."""
    return _envelope_feature(
        env, req_id, offset=offset, identifier=f"uuid-{req_id}-at-{offset}"
    )


def _carrier_request(env, req_id: str, *, token=None, features=()):
    """A stream request with the given scheduled token (the stream carrier
    by default) and multimodal features."""
    request = h.new_request(
        req_id, int(env.config.audio_chunk_token_id) if token is None else token
    )
    return dataclasses.replace(request, mm_features=list(features))


def _envelope_feature(
    env, req_id: str, *, envelope=None, offset: int = 0, identifier=None, **changes
):
    """The multimodal feature of a regular unit at ``offset``; ``envelope``
    replaces its envelope row, ``identifier`` its encoder-cache identity, and
    ``changes`` any other field (``modality``, ``data``, ``mm_position``)."""
    template = stream(env, req_id)
    (feature,) = template.mm_features
    if envelope is not None:
        from vllm.multimodal.inputs import (
            MultiModalBatchedField,
            MultiModalFieldElem,
            MultiModalKwargsItem,
        )

        field = MultiModalBatchedField(keep_on_cpu=True)
        feature = dataclasses.replace(
            feature,
            data=MultiModalKwargsItem({"audio": MultiModalFieldElem(envelope, field)}),
        )
    if offset:
        from vllm.multimodal.inputs import PlaceholderRange

        feature = dataclasses.replace(
            feature, mm_position=PlaceholderRange(offset=offset, length=1)
        )
    if identifier is not None:
        feature = dataclasses.replace(feature, identifier=identifier)
    return dataclasses.replace(feature, **changes)


def _second_feature(env, req_id: str):
    """Another unit's feature over the same token (its own cache identity)."""
    return _envelope_feature(env, req_id, identifier=f"uuid-{req_id}-second")


def _cached(env, feature):
    """``feature`` with its encoder output already in core's encoder cache
    (keyed by its identifier), as core leaves a feature it encoded at an
    earlier step: the output is the model's own encoding of the feature's
    envelope, so core's scheduler does not schedule it again."""
    (output,) = env.model.embed_multimodal(audio=[feature.data["audio"].data])
    env.state.encoder_cache.encoder_outputs[feature.identifier] = output
    return feature


def _payload_without_an_envelope():
    """A kwargs payload whose only field is not the envelope (``audio``)."""
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFieldElem,
        MultiModalKwargsItem,
    )

    field = MultiModalBatchedField(keep_on_cpu=True)
    return MultiModalKwargsItem(
        {"features": MultiModalFieldElem(torch.zeros(20), field)}
    )


#: REG-STATE-015's request-caused envelope-header faults, delivered as core
#: delivers them: name -> (request builder, scheduled encoder inputs). Core's
#: scheduler schedules every overlapping feature it has not cached, so two
#: overlapping features arrive with both inputs scheduled, or with one when
#: the other is cached; both are the request's fault. The model state's
#: ``prepare_inputs_embeds`` runs core's ``execute_mm_encoder`` and
#: ``gather_mm_embeddings``, so it screens these rows before it calls them.
_REQUEST_FAULTS = {
    "CHUNK row without a feature": (
        lambda env: _carrier_request(env, "bad"),
        {},
    ),
    "CHUNK row two features overlap, both scheduled": (
        lambda env: _carrier_request(
            env,
            "bad",
            features=[_envelope_feature(env, "bad"), _second_feature(env, "bad")],
        ),
        {"bad": [0, 1]},
    ),
    "CHUNK row two features overlap, one cached": (
        lambda env: _carrier_request(
            env,
            "bad",
            features=[
                _envelope_feature(env, "bad"),
                _cached(env, _second_feature(env, "bad")),
            ],
        ),
        {"bad": [0]},
    ),
    "non-CHUNK row a feature overlaps": (
        lambda env: _carrier_request(
            env, "bad", token=7, features=[_envelope_feature(env, "bad")]
        ),
        {"bad": [0]},
    ),
    "overlapping feature not audio": (
        lambda env: _carrier_request(
            env, "bad", features=[_envelope_feature(env, "bad", modality="image")]
        ),
        {"bad": [0]},
    ),
    "overlapping feature without a kwargs payload": (
        lambda env: _carrier_request(
            env, "bad", features=[_envelope_feature(env, "bad", data=None)]
        ),
        {"bad": [0]},
    ),
    "overlapping feature without an envelope payload": (
        lambda env: _carrier_request(
            env,
            "bad",
            features=[
                _envelope_feature(env, "bad", data=_payload_without_an_envelope())
            ],
        ),
        {"bad": [0]},
    ),
    "envelope smaller than its header": (
        lambda env: _carrier_request(
            env,
            "bad",
            features=[_envelope_feature(env, "bad", envelope=torch.zeros(3))],
        ),
        {"bad": [0]},
    ),
}


def _step_with_a_clean_row(env, bad, encoder_inputs):
    """``bad`` (slot 3) and a clean stream unit ``s1`` (slot 1) in one real
    step; returns the error it raised, or ``None``."""
    output = h.scheduler_output(
        new=(bad, stream(env, "s1")),
        handles={bad.req_id: 3, "s1": 1},
        encoder_inputs={**encoder_inputs, "s1": [0]},
    )
    return h.call_with(env.runner.real_step, output)


@pytest.mark.parametrize("fault", list(_REQUEST_FAULTS))
# @spec REG-STATE-015
def test_a_request_caused_envelope_fault_fails_that_request_alone(build, fault):
    """A direct engine client can send each of these: the row fails at
    planning, before core's encoder path sees it, its request alone ends
    through ``take_failed_requests``, and the step's other row commits."""
    make, encoder_inputs = _REQUEST_FAULTS[fault]
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    error = _step_with_a_clean_row(env, make(env), encoder_inputs)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


@pytest.mark.parametrize("fault", list(_REQUEST_FAULTS))
# @spec REG-STATE-015
def test_envelope_resolution_classifies_each_request_caused_fault(build, fault):
    """Each of ``_REQUEST_FAULTS`` is also a request not shaped as the
    processor builds it, so on the served path the shape rule fails its row
    before envelope resolution judges it, and the test above checks only
    that the row fails alone. Resolution still judges every row
    (``resolve_row_envelope_header``): called directly on the row core
    schedules, with the encoder inputs core schedules for it, each fault is
    the request's (``RequestRowFault``), never core's contract fault."""
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
        RequestRowFault,
        resolve_row_envelope_header,
    )

    make, encoder_inputs = _REQUEST_FAULTS[fault]
    env = build()
    bad = make(env)
    assert _shape_fault(env, bad) is not None
    with pytest.raises(RequestRowFault):
        resolve_row_envelope_header(
            scheduled_token_id=int(bad.prefill_token_ids[0]),
            placeholder_id=int(env.config.audio_chunk_token_id),
            num_computed_tokens=0,
            mm_features=bad.mm_features,
            scheduled_encoder_input_ids=encoder_inputs.get("bad", ()),
            carrier_width=int(env.config.hidden_size),
        )


def _raised_in(error: BaseException) -> str:
    """``"plugin"`` or ``"core"``: whose code raised ``error``, by the
    innermost frame of its traceback in either package (frames of torch or
    another library are skipped); ``"other"`` for neither."""
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        path = frame.filename.replace("\\", "/")
        if "/vllm_nemotron_asr_streaming/" in path:
            return "plugin"
        if "/vllm/" in path:
            return "core"
    return "other"


#: The plugin's error for each contract fault of a stream row's envelope
#: header: what the message names. The class is checked with ``isinstance``
#: (``ValueError`` or a subclass), the plugin's convention; no spec names it.
_PLUGIN_CONTRACT_ERRORS = {
    "feature without position metadata": r"(?i)position",
    "non-CHUNK row with a scheduled encoder input": (
        r"(?i)non-CHUNK row[^.]*scheduled encoder input"
    ),
    "CHUNK row with more than one scheduled encoder input": (
        r"(?i)(more than one|at most one)[^.]*encoder input"
    ),
    "scheduled encoder input naming another feature": (
        r"(?i)encoder input[^.]*(same overlapping|other) feature"
    ),
    "envelope payload not host-resident": r"(?i)host",
}

#: Where core's own multimodal plumbing meets a contract fault before the
#: model state's planning can (the CPU harness runs core's encoder cache,
#: ``execute_mm_encoder`` and ``gather_mm_embeddings``): core's error class
#: and message, raised in core's code. A meta tensor stands for a device
#: payload on CPU; core's batching of the step's encoder inputs, or its
#: host-to-device copy, cannot read it.
_CORE_CONTRACT_ERRORS = {
    "feature without position metadata": (
        (AttributeError, r"'NoneType' object has no attribute 'offset'"),
    ),
    "scheduled encoder input naming another feature": (
        (RuntimeError, r"Encoder cache miss for uuid-bad"),
    ),
    "envelope payload not host-resident": (
        (RuntimeError, r"not on the expected device meta"),
        (NotImplementedError, r"Cannot copy out of meta tensor"),
    ),
}


def _far_feature(env, req_id: str):
    """A feature that does not overlap the row's token (position 5)."""
    return _envelope_feature(env, req_id, offset=5, identifier=f"uuid-{req_id}-far")


def _later_input(env, request):
    """``request`` as the next input of the stream ``bad``, whose first unit
    a step before has run: the runner folds it into the stream as core does
    (the frontier token, then the input, ``num_computed_tokens`` 1). A
    non-CHUNK row of a request shaped as the processor builds it is a
    continuing stream's row: a fresh stream starts with a CHUNK row
    (REG-STATE-015)."""
    env.runner.real_step(
        h.scheduler_output(new=(stream(env, "bad"),), handles={"bad": 3})
    )
    return request


#: Faults of a stream row's envelope header that core's scheduling or
#: multimodal plumbing produces and no request input can (REG-STATE-015):
#: name -> (request builder, scheduled encoder inputs). Each request is
#: otherwise shaped as the processor builds it, or the request's fault would
#: decide the class. Core's msgpack decoder refuses a feature whose position
#: is null; a feature's other placement faults are the request's
#: (``_UNPLACED_FEATURES``).
_CONTRACT_FAULTS = {
    "feature without position metadata": (
        lambda env: _carrier_request(
            env, "bad", features=[_envelope_feature(env, "bad", mm_position=None)]
        ),
        [0],
    ),
    "non-CHUNK row with a scheduled encoder input": (
        # a label row, then a unit whose feature does not overlap the row
        lambda env: _later_input(
            env,
            _prompted(
                _carrier_request(env, "bad", features=[_unit_at(env, "bad", 1)]),
                [LABEL, int(env.config.audio_chunk_token_id)],
            ),
        ),
        [0],
    ),
    "CHUNK row with more than one scheduled encoder input": (
        lambda env: _carrier_request(
            env, "bad", features=[_envelope_feature(env, "bad")]
        ),
        [0, 0],
    ),
    "scheduled encoder input naming another feature": (
        # two units, at the prompt's first and last positions
        lambda env: _prompted(
            _carrier_request(
                env,
                "bad",
                features=[_envelope_feature(env, "bad"), _far_feature(env, "bad")],
            ),
            [int(env.config.audio_chunk_token_id), *[LABEL] * 4]
            + [int(env.config.audio_chunk_token_id)],
        ),
        [1],
    ),
    "envelope payload not host-resident": (
        lambda env: _carrier_request(
            env,
            "bad",
            features=[
                _envelope_feature(
                    env,
                    "bad",
                    envelope=torch.zeros(h.stream_envelope().shape, device="meta"),
                )
            ],
        ),
        [0],
    ),
}


@pytest.mark.parametrize("fault", list(_CONTRACT_FAULTS))
# @spec REG-STATE-015
def test_a_contract_fault_of_the_envelope_header_raises(build, fault):
    """No request input can cause these, so core and the model disagree
    about the step: the step raises, naming the fault, and records no failed
    request. The raise is the plugin's ``ValueError`` naming the fault, or,
    where core's own multimodal plumbing meets the fault first, core's error
    raised in core's code."""
    make, encoder_inputs = _CONTRACT_FAULTS[fault]
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    error = _step_with_a_clean_row(env, make(env), {"bad": encoder_inputs})
    assert error is not None, f"the step accepted a {fault}"
    message = str(error)
    if _raised_in(error) == "core" and fault in _CORE_CONTRACT_ERRORS:
        expected = _CORE_CONTRACT_ERRORS[fault]
    else:
        assert _raised_in(error) == "plugin", (fault, _raised_in(error), repr(error))
        expected = ((ValueError, _PLUGIN_CONTRACT_ERRORS[fault]),)
    assert any(
        isinstance(error, cls) and re.search(pattern, message)
        for cls, pattern in expected
    ), (fault, repr(error))
    assert not any(env.runner.failed_log), (
        "a contract fault was recorded as a row failure"
    )


@pytest.mark.parametrize("step", ["stream-step", "with-a-file-row"])
@pytest.mark.parametrize(
    "stamp", [float("nan"), float("inf")], ids=["nan-stamp", "inf-stamp"]
)
# @spec REG-STATE-015
def test_an_unlisted_fault_a_request_can_cause_fails_that_request_alone(
    build, stamp, step
):
    """REG-STATE-015's default class: no list names an envelope header whose
    admission stamp is not a number, and a direct engine client can send one,
    so its row fails alone (here with the step's clean stream row, and with a
    file row whose planning reads every stream row's header)."""
    # Three rows (bad, s1 and, with a file row, f1): core never schedules more
    # rows than max_num_seqs.
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    envelope = h.stream_envelope(geometry=env.max_geometry)
    envelope[6] = stamp  # admission_ms_mod, the header's last slot
    bad = _carrier_request(
        env, "bad", features=[_envelope_feature(env, "bad", envelope=envelope)]
    )
    new, handles = [bad, stream(env, "s1")], {"bad": 3, "s1": 1}
    if step == "with-a-file-row":
        new.append(file(env, "f1"))
        handles["f1"] = 2
    error = h.call_with(
        env.runner.real_step, h.scheduler_output(new=tuple(new), handles=handles)
    )
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


def _audio_payload(envelope: torch.Tensor):
    """A kwargs payload whose one field is the envelope ``envelope``."""
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFieldElem,
        MultiModalKwargsItem,
    )

    field = MultiModalBatchedField(keep_on_cpu=True)
    return MultiModalKwargsItem({"audio": MultiModalFieldElem(envelope, field)})


def _wider_than_the_carrier(env) -> torch.Tensor:
    """A stream envelope with a valid header and a full chunk, padded to one
    value more than the carrier row holds (``hidden_size``)."""
    valid = h.stream_envelope(geometry=env.max_geometry)
    envelope = torch.zeros(int(env.config.hidden_size) + 1)
    envelope[: valid.shape[0]] = valid
    return envelope


# @spec REG-STATE-015
def test_an_envelope_wider_than_the_carrier_fails_that_request_alone(build):
    """REG-STATE-015 lists an envelope wider than the carrier: a direct engine
    client can send a stream envelope whose header is valid and whose payload
    is wider than the carrier row. Screening checks the whole envelope, not
    only its header, so the row fails at planning before core's encoder path
    meets it, and the step's clean row commits."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    bad = _carrier_request(
        env,
        "bad",
        features=[_envelope_feature(env, "bad", envelope=_wider_than_the_carrier(env))],
    )
    error = _step_with_a_clean_row(env, bad, {"bad": [0]})
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


def _file_carrier_request(env, req_id: str, data):
    """A file-carrier request whose one feature carries the kwargs payload
    ``data`` instead of a file's samples and header."""
    template = file(env, req_id)
    (feature,) = template.mm_features
    return dataclasses.replace(
        template, mm_features=[dataclasses.replace(feature, data=data)]
    )


#: Malformed items of a file request (FILE-BUF-001) that core's encoder path
#: would raise for if it ran them: name -> payload builder.
_MALFORMED_FILE_ITEMS = {
    "item with neither samples nor a file header": (
        lambda env: _payload_without_an_envelope()
    ),
    "item carrying an envelope wider than the carrier": (
        lambda env: _audio_payload(_wider_than_the_carrier(env))
    ),
}


@pytest.mark.parametrize("item", list(_MALFORMED_FILE_ITEMS))
# @spec REG-STATE-015, FILE-BUF-001
def test_a_malformed_file_request_fails_alone_before_cores_encoder_path(build, item):
    """A file request whose item the model state could not take at
    ``add_request`` is a known failure before the step's encoder path runs:
    screening drops its item from core's encoder and its row from core's
    gather, so beside a clean stream row the step does not raise, the file
    request alone fails and the stream row commits."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    bad = _file_carrier_request(env, "bad", _MALFORMED_FILE_ITEMS[item](env))
    error = _step_with_a_clean_row(env, bad, {"bad": [0]})
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


@pytest.mark.parametrize("beside", ["stream row", "file row"])
# @spec REG-STATE-015, FILE-BUF-001
def test_a_file_item_the_encoder_refuses_fails_that_file_request_alone(build, beside):
    """A direct engine client can send a file-carrier request whose item
    carries valid samples and a valid header under another modality. The
    model's encoder refuses that item (``encoder_item_fault``), so screening
    drops its encoder input; ``add_request`` refuses it by the same rule, so
    its row also leaves core's gather. Beside a healthy stream or file
    request (each with its own identifier) the step does not raise, the bad
    file request alone fails and the other request commits."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    template = file(env, "bad")
    (feature,) = template.mm_features
    bad = dataclasses.replace(
        template, mm_features=[dataclasses.replace(feature, modality="image")]
    )
    good = stream(env, "s1") if beside == "stream row" else file(env, "f1")
    assert feature.identifier != good.mm_features[0].identifier
    output = h.scheduler_output(
        new=(bad, good),
        handles={"bad": 3, good.req_id: 1},
        encoder_inputs={"bad": [0], good.req_id: [0]},
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get(good.req_id) == 0


def _with_item(request, item):
    """``request`` whose one feature carries the kwargs item ``item``."""
    (feature,) = request.mm_features
    return dataclasses.replace(
        request, mm_features=[dataclasses.replace(feature, data=item)]
    )


def _field(kind: str, size: int, *, keep_on_cpu: bool = True):
    """A field of class ``kind`` (``batched``, ``flat`` or ``shared``) over a
    one-item payload of ``size`` values: core's msgpack decoder rebuilds the
    field class a direct engine client names (``serial_utils``)."""
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFlatField,
        MultiModalSharedField,
    )

    if kind == "flat":
        return MultiModalFlatField(slices=[slice(0, size)], keep_on_cpu=keep_on_cpu)
    if kind == "shared":
        return MultiModalSharedField(batch_size=1, keep_on_cpu=keep_on_cpu)
    return MultiModalBatchedField(keep_on_cpu=keep_on_cpu)


def _with_field(request, name: str, change):
    """``request`` whose item has its field ``name`` rebuilt by ``change``, a
    function of the field's payload and field that returns the new ones, as
    a direct engine client can send it: core's decoder keeps each tensor's
    dtype and shape as sent."""
    from vllm.multimodal.inputs import MultiModalFieldElem, MultiModalKwargsItem

    (feature,) = request.mm_features
    item = dict(feature.data)
    item[name] = MultiModalFieldElem(*change(item[name].data, item[name].field))
    return _with_item(request, MultiModalKwargsItem(item))


#: Items not shaped as the model's processor builds them (REG-STATE-015,
#: FILE-PROMPT-003): name -> (the kind of request that carries it, the field
#: changed, the change). Each was accepted by the item rule: core's encoder
#: path or the plugin's header conversion raised for the whole step, the
#: stacked batch failed a healthy neighbour (a ``float16`` envelope stacked
#: first rounds every envelope of its group to ``float16``), or the item ran
#: as if well formed.
_UNSHAPED_ITEMS = {
    "envelope in a flat field": (
        "stream",
        "audio",
        lambda data, field: (data, _field("flat", data.shape[0])),
    ),
    "envelope in a shared field": (
        "stream",
        "audio",
        lambda data, field: (data, _field("shared", data.shape[0])),
    ),
    "int64 envelope": (
        "stream",
        "audio",
        lambda data, field: (data.round().to(torch.int64), field),
    ),
    "float16 envelope": (
        "stream",
        "audio",
        lambda data, field: (data.to(torch.float16), field),
    ),
    "complex envelope": (
        "stream",
        "audio",
        lambda data, field: (data.to(torch.complex64), field),
    ),
    "two-dimensional envelope": (
        "stream",
        "audio",
        lambda data, field: (data.unsqueeze(0), field),
    ),
    "file header in a flat field": (
        "file",
        "file_header",
        lambda data, field: (data, _field("flat", 3)),
    ),
    "file header in a shared field": (
        "file",
        "file_header",
        lambda data, field: (data, _field("shared", 3)),
    ),
    "float32 file header": (
        "file",
        "file_header",
        lambda data, field: (data.to(torch.float32), field),
    ),
    "samples not kept on the host": (
        "file",
        "samples",
        lambda data, field: (data, _field("batched", 0, keep_on_cpu=False)),
    ),
}


@pytest.mark.parametrize("order", ["bad first", "bad last", "alone"])
@pytest.mark.parametrize("case", list(_UNSHAPED_ITEMS))
# @spec REG-STATE-015, FILE-PROMPT-003
def test_an_item_not_shaped_as_the_processor_builds_it_fails_alone(build, case, order):
    """Core's msgpack decoder rebuilds each field's class from the factory
    name the sender gives and each tensor with the dtype it was sent with, so
    a direct engine client can send an item the model's processor never
    builds. Core's encoder path batches consecutive items with the same
    fields and field classes, stacking same-shape tensors into the first
    one's dtype, so the bad item sits next to a healthy item of its kind
    (first or last; or alone in its step) beside a healthy stream and a
    healthy file, each with its own identifier. The step does not raise, the
    bad request alone fails through ``take_failed_requests``, the healthy
    requests commit, and the healthy stream's encoder output is its own
    envelope's encoding."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    kind, name, change = _UNSHAPED_ITEMS[case]
    template = stream(env, "bad") if kind == "stream" else file(env, "bad")
    bad = _with_field(template, name, change)
    healthy_stream, healthy_file = stream(env, "s1"), file(env, "f1")
    same, other = (
        (healthy_stream, healthy_file)
        if kind == "stream"
        else (healthy_file, healthy_stream)
    )
    rows = {
        "bad first": (bad, same, other),
        "bad last": (other, same, bad),
        "alone": (bad,),
    }[order]
    assert len({r.mm_features[0].identifier for r in rows}) == len(rows)
    output = h.scheduler_output(
        new=rows,
        handles={"bad": 3, "s1": 1, "f1": 2},
        encoder_inputs={r.req_id: [0] for r in rows},
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    if order == "alone":
        return
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0 and statuses.get("f1") == 0, statuses
    (feature,) = healthy_stream.mm_features
    (own,) = env.model.embed_multimodal(audio=[feature.data["audio"].data])
    cached = env.state.encoder_cache.encoder_outputs[feature.identifier]
    assert torch.equal(cached, own), "a neighbour's item changed s1's encoder input"


def _processor_item(env, *, envelope=None, samples=None):
    """The item the model's processor builds for a stream unit's envelope
    (``envelope``, as the session packer builds it) or a complete file's
    ``samples`` at the server's geometry, delivered as core delivers it to the
    engine core: through core's msgpack encoder and decoder."""
    from types import SimpleNamespace

    from vllm.multimodal.inputs import MultiModalKwargsItem, MultiModalKwargsItems
    from vllm.multimodal.parse import MultiModalDataParser
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
        NemotronASRMultiModalProcessor,
    )

    # the processor's field building reads only the served configuration
    processor = object.__new__(NemotronASRMultiModalProcessor)
    processor.info = SimpleNamespace(get_hf_config=lambda: env.config)
    audio = envelope if samples is None else samples
    kwargs = (
        {} if samples is None else {"file_header": [env.max_geometry, 0, len(samples)]}
    )
    items = MultiModalDataParser(target_sr=16000).parse_mm_data({"audio": audio})
    batch = processor._apply_hf_processor_main(items, kwargs)
    fields = processor._get_mm_fields_config(batch, kwargs)
    (item,) = MultiModalKwargsItems.from_hf_inputs(batch, fields)["audio"]
    return MsgpackDecoder(MultiModalKwargsItem).decode(MsgpackEncoder().encode(item))


# @spec REG-STATE-015, FILE-PROMPT-003
def test_items_the_processor_builds_pass_the_item_rule_and_commit(build):
    """Positive control of the item rule (``encoder_item_fault``): a stream
    unit's item and a complete file's item, as the model's processor builds
    them and core delivers them, pass the rule and commit in one step."""
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import encoder_item_fault

    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    s1, f1 = stream(env, "s1"), file(env, "f1")
    envelope = s1.mm_features[0].data["audio"].data.numpy()
    samples = f1.mm_features[0].data["samples"].data.numpy()
    s1 = _with_item(s1, _processor_item(env, envelope=envelope))
    f1 = _with_item(f1, _processor_item(env, samples=samples))
    for request in (s1, f1):
        (feature,) = request.mm_features
        fault = encoder_item_fault(feature, carrier_width=int(env.config.hidden_size))
        assert fault is None, (request.req_id, fault)
    output = h.scheduler_output(new=(s1, f1), handles={"s1": 1, "f1": 2})
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [None]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0 and statuses.get("f1") == 0, statuses


def _through_core(request):
    """``request`` as core delivers a direct engine client's request to the
    worker, with the encoder inputs core schedules for its first token: an
    ``EngineCoreRequest`` through core's msgpack encoder and decoder (which
    keep a placeholder range's ``offset``, ``length`` and ``is_embed`` as
    sent), core's ``Request``, core's encoder-input scheduling
    (``Scheduler._try_schedule_encoder_inputs`` over an empty encoder cache)
    and core's ``NewRequestData``. Core's scheduling must schedule the row's
    one token: a feature it cannot schedule never reaches the worker."""
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    sent = EngineCoreRequest(
        request_id=request.req_id,
        prompt_token_ids=np.array(request.prompt_token_ids, dtype=np.int32),
        mm_features=list(request.mm_features),
        sampling_params=request.sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    received = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(sent))
    core_request = Request.from_engine_core_request(received, None)
    # the attributes the scheduling routine reads, as core's scheduler sets
    # them for this model (no encoder-decoder, no EC connector)
    scheduler = object.__new__(Scheduler)
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)
    scheduler.encoder_cache_manager = EncoderCacheManager(1 << 20)
    scheduler.ec_connector = None
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None)
    )
    scheduled, num_new_tokens, *_ = scheduler._try_schedule_encoder_inputs(
        core_request, 0, 1, 1 << 20
    )
    assert num_new_tokens == 1, "core's scheduler does not schedule the row"
    prompt = [int(token) for token in core_request.prompt_token_ids]
    delivered = NewRequestData.from_request(core_request, request.block_ids, prompt)
    return delivered, list(scheduled)


def _placed(*, offset: int = 0, length: int = 1, is_embed=None):
    """A change that places a request's one feature at ``offset`` over
    ``length`` positions with the embedding mask ``is_embed``. The processor
    places a unit on its one carrier position with no mask (``offset`` 0,
    ``length`` 1, ``is_embed`` ``None``)."""
    from vllm.multimodal.inputs import PlaceholderRange

    def change(request):
        (feature,) = request.mm_features
        placement = PlaceholderRange(offset=offset, length=length, is_embed=is_embed)
        moved = dataclasses.replace(feature, mm_position=placement)
        return dataclasses.replace(request, mm_features=[moved])

    return change


def _out_of_offset_order(request):
    """``request`` whose features are a well-formed unit at a later position
    (its own identifier) followed by its own unit at the first token. Core
    keeps a request's features in the order sent, and its window search
    assumes offset order, so it schedules and gathers both for the first
    token."""
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.multimodal.utils import get_mm_features_in_window

    (feature,) = request.mm_features
    later = dataclasses.replace(
        feature,
        identifier=f"{feature.identifier}-later",
        mm_position=PlaceholderRange(offset=5, length=1),
    )
    features = [later, feature]
    assert get_mm_features_in_window(features, start=0, end=1) == (0, 2)
    return dataclasses.replace(request, mm_features=features)


#: Features not placed as the model's processor places them (REG-STATE-015):
#: name -> (the kind of request that carries it, the change). Each is
#: well formed in its item, survives core's msgpack decoding, and is
#: scheduled by core's encoder-input scheduling (``_through_core``). Each was
#: accepted: core's gather raised for the whole step (an integer or float
#: mask, which core ors into its boolean placement mask; a feature that
#: starts before the prompt, whose gathered rows do not match its marked
#: positions), the plugin raised a contract fault for the whole step (a
#: feature of no position; features out of offset order, for which core
#: schedules two inputs for one token), or the request ran as if placed by
#: the processor.
_UNPLACED_FEATURES = {
    "int64 embedding mask": ("stream", _placed(is_embed=torch.tensor([1]))),
    "float32 embedding mask": ("stream", _placed(is_embed=torch.tensor([1.0]))),
    "boolean mask of its one position": (
        "stream",
        _placed(is_embed=torch.tensor([True])),
    ),
    "boolean mask of no position": ("stream", _placed(is_embed=torch.tensor([False]))),
    "feature of no position": ("stream", _placed(length=0)),
    "feature of two positions": ("stream", _placed(length=2)),
    "feature that starts before the prompt": ("stream", _placed(offset=-1, length=2)),
    "features out of offset order": ("stream", _out_of_offset_order),
    "file feature with an int64 embedding mask": (
        "file",
        _placed(is_embed=torch.tensor([1])),
    ),
    "file feature of no position": ("file", _placed(length=0)),
    "file feature of two positions": ("file", _placed(length=2)),
    "file feature off its carrier token": ("file", _placed(offset=1)),
}


@pytest.mark.parametrize("order", ["bad first", "bad last", "alone"])
@pytest.mark.parametrize("case", list(_UNPLACED_FEATURES))
# @spec REG-STATE-015
def test_a_feature_not_placed_as_the_processor_places_it_fails_alone(
    build, case, order
):
    """A feature's placement (its placeholder range's ``offset``, ``length``
    and ``is_embed``) is the request's: core's msgpack decoder keeps it as a
    direct engine client sends it, and core's scheduling, gather and the
    plugin's planning read it. A request whose feature is not placed as the
    processor places it, delivered and scheduled as core delivers and
    schedules it, sits first or last beside a healthy stream and a healthy
    file (each its own identifier), or alone in its step. The step does not
    raise, the bad request alone fails through ``take_failed_requests``, the
    healthy requests commit, and the healthy stream's encoder output is its
    own envelope's encoding."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    kind, change = _UNPLACED_FEATURES[case]
    template = stream(env, "bad") if kind == "stream" else file(env, "bad")
    delivered = {
        "bad": _through_core(change(template)),
        "s1": _through_core(stream(env, "s1")),
        "f1": _through_core(file(env, "f1")),
    }
    names = {
        "bad first": ("bad", "s1", "f1"),
        "bad last": ("f1", "s1", "bad"),
        "alone": ("bad",),
    }[order]
    rows = tuple(delivered[name][0] for name in names)
    identifiers = [f.identifier for r in rows for f in r.mm_features]
    assert len(set(identifiers)) == len(identifiers)
    scheduled = {name: delivered[name][1] for name in names}
    output = h.scheduler_output(
        new=rows,
        handles={"bad": 3, "s1": 1, "f1": 2},
        encoder_inputs={name: inputs for name, inputs in scheduled.items() if inputs},
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    if order == "alone":
        return
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0 and statuses.get("f1") == 0, statuses
    (feature,) = delivered["s1"][0].mm_features
    (own,) = env.model.embed_multimodal(audio=[feature.data["audio"].data])
    cached = env.state.encoder_cache.encoder_outputs[feature.identifier]
    assert torch.equal(cached, own), "a neighbour's feature changed s1's input"


def _processor_request(env, req_id: str, *, envelope=None, samples=None):
    """The request core's input processing builds from the model's processor
    for a stream unit's envelope (as the session packer builds it) or a
    complete file's ``samples``: the processor's prompt, its item
    (``_processor_item``) and its placeholder as the processor places it
    (``_maybe_apply_prompt_updates``), turned into a placeholder range as core
    turns it (``PlaceholderFeaturesInfo.to_range``)."""
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.parse import MultiModalDataParser

    from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
        NemotronASRMultiModalProcessor,
    )

    processor = object.__new__(NemotronASRMultiModalProcessor)
    processor.info = SimpleNamespace(get_hf_config=lambda: env.config)
    carrier = (
        env.config.audio_chunk_token_id
        if samples is None
        else env.config.file_carrier_token_id
    )
    audio = envelope if samples is None else samples
    items = MultiModalDataParser(target_sr=16000).parse_mm_data({"audio": audio})
    prompt, placeholders = processor._maybe_apply_prompt_updates(
        items, SimpleNamespace(prompt_ids=[int(carrier)])
    )
    (placeholder,) = placeholders["audio"]
    template = h.new_request(req_id, int(carrier))
    feature = MultiModalFeatureSpec(
        data=_processor_item(env, envelope=envelope, samples=samples),
        modality="audio",
        identifier=f"uuid-{req_id}",
        mm_position=placeholder.to_range(),
        mm_hash=f"uuid-{req_id}",
    )
    return dataclasses.replace(
        template,
        prompt_token_ids=list(prompt),
        prefill_token_ids=list(prompt),
        mm_features=[feature],
    )


@pytest.mark.parametrize("right_context", LOOKAHEADS)
# @spec REG-STATE-015, FILE-PROMPT-003
def test_features_the_processor_builds_pass_and_commit_at_every_chunk_size(
    build, right_context
):
    """Positive control of the feature and item rules at every chunk size: a
    regular unit and a final tail as the session packer builds them, and a
    complete file, each as the model's processor builds its request and core
    delivers and schedules it (``_through_core``), pass both rules; the
    regular unit and the file commit in one step."""
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import (
        encoder_item_fault,
        feature_placement_fault,
    )
    from vllm_nemotron_asr_streaming.session.packer import Packer

    env = build(right_context=right_context)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    packer = Packer(
        right_context=right_context,
        geometry_index=geometry_of(right_context),
        locale_row=0,
        carrier_token_id=int(env.config.audio_chunk_token_id),
    )
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(1280 * (right_context + 1) + 300).astype(np.float32)
    (unit,) = packer.append(audio * 0.1)
    tail = packer.close()
    samples = rng.standard_normal(5000).astype(np.float32) * 0.1
    requests = {
        "s1": _processor_request(env, "s1", envelope=unit.row),
        "s2": _processor_request(env, "s2", envelope=tail.row),
        "f1": _processor_request(env, "f1", samples=samples),
    }
    delivered = {name: _through_core(r) for name, r in requests.items()}
    width = int(env.config.hidden_size)
    for name, (request, inputs) in delivered.items():
        assert inputs == [0], (name, inputs)
        (feature,) = request.mm_features
        assert feature_placement_fault(request.mm_features) is None, name
        assert encoder_item_fault(feature, carrier_width=width) is None, name
    output = h.scheduler_output(
        new=(delivered["s1"][0], delivered["f1"][0]),
        handles={"s1": 1, "f1": 2},
        encoder_inputs={"s1": [0], "f1": [0]},
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.runner.failed_log == [None]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0 and statuses.get("f1") == 0, statuses


@pytest.mark.parametrize("kind", ["stream", "file"])
@pytest.mark.parametrize("rows", [1, 2], ids=["one row", "two rows"])
# @spec REG-STATE-021
def test_the_processor_refuses_audio_that_is_not_one_dimensional(build, kind, rows):
    """Core's audio parser keeps the rank of an array given in a list
    (``{"audio": [array]}``), so the model's processor can receive a
    two-dimensional audio item, whose item the item rule refuses. The
    processor refuses it at input processing with a ``ValueError`` naming
    its shape, so the client gets an input-processing error and no request
    reaches the engine; a file's header counts the array's rows, which is
    all the header check compares."""
    from vllm.multimodal.parse import MultiModalDataParser

    from vllm_nemotron_asr_streaming.nemotron_asr.processor import (
        NemotronASRMultiModalProcessor,
    )

    env = build()
    processor = object.__new__(NemotronASRMultiModalProcessor)
    processor.info = SimpleNamespace(get_hf_config=lambda: env.config)
    audio = np.zeros((rows, 300), dtype=np.float32)
    kwargs = {} if kind == "stream" else {"file_header": [env.max_geometry, 0, rows]}
    items = MultiModalDataParser(target_sr=16000).parse_mm_data({"audio": [audio]})
    with pytest.raises(ValueError, match=re.escape(str((rows, 300)))):
        processor._apply_hf_processor_main(items, kwargs)


#: Requests whose prompt is not shaped as the model's processor and core's
#: session turns build it (REG-STATE-015), each with well-formed items and
#: placements: name -> request builder. The processor and the session
#: packer build a stream unit as the one stream-carrier token with its
#: unit's feature on it, and a complete file as the one file-carrier token
#: with the file's feature on it; core puts the frontier token before a later
#: turn's input. So a fresh stream request starts with a CHUNK row, every
#: carrier token of a stream has one unit's feature and every feature sits on
#: one, and a file's prompt is its carrier alone. A direct engine client can
#: send each of these. On the head before this rule, the fresh non-CHUNK rows
#: (any token but the stream carrier, alone or before a unit) raised for the
#: whole step at registry binding ("continuing row ... is not registered"),
#: and the others ran as if the processor had built them.
_UNSHAPED_REQUESTS = {
    "fresh label row": lambda env: h.new_request("bad", LABEL),
    "fresh park row": lambda env: h.new_request("bad", int(env.config.eos_token_id)),
    "fresh EOU row": lambda env: h.new_request("bad", int(env.config.eou_token_id)),
    "fresh continuation row": lambda env: h.new_request(
        "bad", int(env.config.continuation_token_id)
    ),
    "label row before a unit": lambda env: _prompted(
        _carrier_request(env, "bad", features=[_unit_at(env, "bad", 1)]),
        [LABEL, int(env.config.audio_chunk_token_id)],
    ),
    "stream carrier without a feature": lambda env: _prompted(
        stream(env, "bad"), [int(env.config.audio_chunk_token_id)] * 2
    ),
    "feature on a label token": lambda env: _prompted(
        _carrier_request(
            env,
            "bad",
            features=[_envelope_feature(env, "bad"), _unit_at(env, "bad", 1)],
        ),
        [int(env.config.audio_chunk_token_id), LABEL],
    ),
    "file carrier in a stream": lambda env: _prompted(
        stream(env, "bad"),
        [int(env.config.audio_chunk_token_id), int(env.config.file_carrier_token_id)],
    ),
    "file prompt longer than its carrier": lambda env: _prompted(
        file(env, "bad"), [int(env.config.file_carrier_token_id), LABEL]
    ),
}


def _mixed_step(env, delivered, order: str):
    """One real step of the delivered ``bad`` request (slot 3) beside the
    healthy ``s1`` stream (slot 1) and ``f1`` file (slot 2) (``delivered``
    maps each name to its request and scheduled encoder inputs; slots and
    inputs follow each request's own ID), ``bad`` first, last or alone; each
    feature has its own identifier. Returns the error the step raised."""
    names = {
        "bad first": ("bad", "s1", "f1"),
        "bad last": ("f1", "s1", "bad"),
        "alone": ("bad",),
    }[order]
    rows = tuple(delivered[name][0] for name in names)
    identifiers = [f.identifier for r in rows for f in r.mm_features]
    assert len(set(identifiers)) == len(identifiers)
    slots = {"bad": 3, "s1": 1, "f1": 2}
    output = h.scheduler_output(
        new=rows,
        handles={delivered[name][0].req_id: slots[name] for name in names},
        encoder_inputs={
            delivered[name][0].req_id: delivered[name][1]
            for name in names
            if delivered[name][1]
        },
    )
    return h.call_with(env.runner.real_step, output)


def _assert_healthy_rows_committed(env, s1) -> None:
    """``s1`` and ``f1`` committed, and ``s1``'s encoder output is its own
    envelope's encoding."""
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0 and statuses.get("f1") == 0, statuses
    (feature,) = s1.mm_features
    (own,) = env.model.embed_multimodal(audio=[feature.data["audio"].data])
    cached = env.state.encoder_cache.encoder_outputs[feature.identifier]
    assert torch.equal(cached, own), "a neighbour's request changed s1's input"


@pytest.mark.parametrize("order", ["bad first", "bad last", "alone"])
@pytest.mark.parametrize("case", list(_UNSHAPED_REQUESTS))
# @spec REG-STATE-015
def test_a_request_not_shaped_as_the_processor_builds_it_fails_alone(
    build, case, order
):
    """A request's prompt is the client's: core's msgpack decoder, its
    encoder-input scheduling and its ``NewRequestData`` keep the prompt
    tokens and their relation to the features as a direct engine client
    sends them. A request whose prompt is not shaped as the processor builds
    it, delivered and scheduled as core delivers and schedules it, sits first
    or last beside a healthy stream and a healthy file, or alone in its step.
    The step does not raise, the bad request alone fails through
    ``take_failed_requests`` on the step that delivers it, the healthy
    requests commit, and the healthy stream's encoder output is its own
    envelope's encoding."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    delivered = {
        "bad": _through_core(_UNSHAPED_REQUESTS[case](env)),
        "s1": _through_core(stream(env, "s1")),
        "f1": _through_core(file(env, "f1")),
    }
    error = _mixed_step(env, delivered, order)
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    if order != "alone":
        _assert_healthy_rows_committed(env, delivered["s1"][0])


def _with_identifier(request, identifier: str):
    """``request`` whose one feature has the encoder-cache identifier
    ``identifier``."""
    (feature,) = request.mm_features
    return dataclasses.replace(
        request, mm_features=[dataclasses.replace(feature, identifier=identifier)]
    )


#: What a direct engine client sends as the next input of a file request it
#: opened as a stream: a stream unit, or another complete file.
_FILE_CONTINUATIONS = {
    "stream unit": lambda env: stream(env, "bad", sequence=1),
    "file prompt": lambda env: _with_identifier(file(env, "bad"), "uuid-bad-again"),
}


@pytest.mark.parametrize("order", ["bad first", "bad last", "alone"])
@pytest.mark.parametrize("given", list(_FILE_CONTINUATIONS))
# @spec REG-STATE-012, REG-STATE-015, REG-STATE-016
def test_core_continuing_a_file_request_fails_that_request_alone(build, given, order):
    """The model's file prompt is one request that core never continues, but
    a direct engine client can open a stream whose first input is a file
    prompt: the file parks at its end, and core folds the client's next input
    into it (``remove_request``, then ``add_request`` with the compacted
    prompt and ``num_computed_tokens`` 1 on the same slot). The model state
    has retired the file at that ``remove_request``. The step does not raise:
    the continued request alone fails at planning, and a healthy stream and a
    healthy file beside it commit. On the head before this rule the
    ``add_request`` raised for the whole step ("continues no live
    request")."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    env.runner.real_step(
        h.scheduler_output(new=(file(env, "bad"),), handles={"bad": 3})
    )
    drain_to_park(env, ["bad"], {"bad": 3}, watch="bad")
    assert env.runner.failed_log[-1] is None
    delivered = {
        "bad": (_FILE_CONTINUATIONS[given](env), [0]),
        "s1": (stream(env, "s1"), [0]),
        "f1": (file(env, "f1"), [0]),
    }
    before = len(env.runner.failed_log)
    error = _mixed_step(env, delivered, order)
    assert error is None, repr(error)
    assert env.runner.failed_log[before:] == [{"bad"}]
    if order != "alone":
        _assert_healthy_rows_committed(env, delivered["s1"][0])


def _received(request, *, resumable: bool = False):
    """Core's ``Request`` for ``request`` as a direct engine client sends it:
    an ``EngineCoreRequest`` through core's msgpack encoder and decoder."""
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    sent = EngineCoreRequest(
        request_id=request.req_id,
        prompt_token_ids=np.array(request.prompt_token_ids, dtype=np.int32),
        mm_features=list(request.mm_features),
        sampling_params=request.sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )
    received = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(sent))
    return Request.from_engine_core_request(received, None)


def _core_folded_turn(env, first, second):
    """``second``, the next input of the parked stream whose first input was
    ``first``, as core's scheduler folds it into the stream
    (``Scheduler._update_request_as_session``, which compacts the stream's
    history to its frontier token followed by the new input, shifts the new
    input's features past the frontier and sets ``num_computed_tokens`` to
    1), with the encoder inputs core schedules for its next token and the
    ``NewRequestData`` core delivers. The stream's history is the tokens the
    harness fed it, then the park token it sampled last."""
    from collections import deque

    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus, StreamingUpdate

    session = _received(first, resumable=True)
    fed = env.runner._fed[first.req_id]
    park = env.runner._sampled[first.req_id]
    assert park == int(env.config.eos_token_id), "the stream has not parked"
    session.append_output_token_ids([*fed[1:], park])
    session.num_computed_tokens = session.num_tokens - 1
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.streaming_queue = deque()
    # the attributes the fold and the scheduling routine read, as core's
    # scheduler sets them for this model (a non-recomputable state group
    # whose history core compacts, positions not read)
    scheduler = object.__new__(Scheduler)
    scheduler.has_unrecomputable_state = True
    scheduler.compacts_state_history = True
    scheduler.state_reads_positions = False
    scheduler.max_model_len = int(env.vllm_config.model_config.max_model_len)
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.hash_block_size = 16
    scheduler.num_waiting_for_streaming_input = 1
    scheduler.resumed_streaming_sessions = set()
    scheduler.dropped_streaming_updates = []
    scheduler.log_stats = False
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=False)
    scheduler.encoder_cache_manager = EncoderCacheManager(1 << 20)
    scheduler.ec_connector = None
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None)
    )
    update = StreamingUpdate.from_request(_received(second, resumable=True))
    assert scheduler._update_request_as_session(session, update)
    scheduled, num_new_tokens, *_ = scheduler._try_schedule_encoder_inputs(
        session, session.num_computed_tokens, 1, 1 << 20
    )
    assert num_new_tokens == 1, "core's scheduler does not schedule the row"
    delivered = NewRequestData.from_request(
        session, second.block_ids, list(session.prefill_token_ids)
    )
    return delivered, list(scheduled)


def _shape_fault(env, request) -> str | None:
    """The request-shape rule's verdict on a delivered request."""
    from vllm_nemotron_asr_streaming.nemotron_asr.plan import request_shape_fault

    return request_shape_fault(
        request.prefill_token_ids,
        request.mm_features,
        num_computed_tokens=int(request.num_computed_tokens),
        chunk_token_id=int(env.config.audio_chunk_token_id),
        file_carrier_id=int(env.config.file_carrier_token_id),
        carrier_width=int(env.config.hidden_size),
    )


@pytest.mark.parametrize("right_context", LOOKAHEADS)
# @spec REG-STATE-015, REG-STATE-008, FILE-PROMPT-003
def test_requests_the_processor_and_core_build_pass_the_shape_rule_and_commit(
    build, right_context
):
    """Positive control of the request-shape rule at every chunk size: two
    streams' first units and a complete file, each as the model's processor
    builds its request from what the session packer and the file prompt send
    and core delivers and schedules it, pass the rule and commit; then one
    stream's next regular unit and the other's final tail, each folded into
    its parked stream by core's own scheduler (the frontier token, then the
    unit's carrier with its feature, ``num_computed_tokens`` 1), pass the
    rule and commit, with no row failed."""
    from vllm_nemotron_asr_streaming.session.packer import Packer

    env = build(right_context=right_context, max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    rng = np.random.default_rng(2)

    def packed(units: int):
        """A session packer's rows for ``units`` regular units of audio and
        300 samples more: the regular units and the final tail."""
        packer = Packer(
            right_context=right_context,
            geometry_index=geometry_of(right_context),
            locale_row=0,
            carrier_token_id=int(env.config.audio_chunk_token_id),
        )
        audio = rng.standard_normal(units * 1280 * (right_context + 1) + 300) * 0.1
        regular = packer.append(audio.astype(np.float32))
        return [unit.row for unit in regular] + [packer.close().row]

    rows = {"s1": packed(2)[:2], "s2": packed(1)}
    samples = (rng.standard_normal(5000) * 0.1).astype(np.float32)
    handles = {"s1": 1, "s2": 2, "f1": 3}
    first = {
        name: _processor_request(env, name, envelope=rows[name][0])
        for name in ("s1", "s2")
    }
    turns = {name: _through_core(request) for name, request in first.items()}
    turns["f1"] = _through_core(_processor_request(env, "f1", samples=samples))
    # s1's next regular unit and s2's final tail
    later = {
        name: _processor_request(env, name, envelope=rows[name][1])
        for name in ("s1", "s2")
    }

    def run(step, label):
        for name, (request, inputs) in step.items():
            assert inputs == [0], (label, name, inputs)
            assert _shape_fault(env, request) is None, (label, name)
        output = h.scheduler_output(
            new=tuple(request for request, _ in step.values()),
            handles=handles,
            encoder_inputs={name: inputs for name, (_, inputs) in step.items()},
        )
        error = h.call_with(env.runner.real_step, output)
        assert error is None, (label, repr(error))
        assert env.runner.failed_log[-1] is None, (label, env.runner.failed_log)
        statuses, _ = env.runner.commit_log[-1]
        assert all(statuses.get(name) == 0 for name in step), (label, statuses)

    run(turns, "first inputs")
    for name in later:
        drain_to_park(env, [name], handles, watch=name)
    folded = {
        name: _core_folded_turn(
            env, first[name], _with_identifier(request, f"uuid-{name}-1")
        )
        for name, request in later.items()
    }
    chunk = int(env.config.audio_chunk_token_id)
    for request, _ in folded.values():
        assert request.num_computed_tokens == 1
        assert list(request.prefill_token_ids)[1:] == [chunk]
        assert [f.mm_position.offset for f in request.mm_features] == [1]
    run(folded, "folded turns")


#: A request ID core accepts that a truth test would mistake for no ID.
EMPTY_ID = ""


@pytest.mark.parametrize("order", ["bad first", "bad last", "alone"])
@pytest.mark.parametrize("kind", ["stream", "file"])
# @spec REG-STATE-022
def test_a_request_id_is_any_string_core_accepts(build, kind, order):
    """Core's ``add_request`` checks only that a request ID is a string, and
    its msgpack decoder, encoder-input scheduling and ``NewRequestData`` keep
    the ID a direct engine client sends, the empty string included. A stream
    unit or a complete file with the empty ID, otherwise as the processor
    builds it, sits first or last beside a healthy stream and a healthy file,
    or alone in its step. The step does not raise, no request fails, every
    row commits, and the empty-ID stream's encoder output is its own
    envelope's encoding. On the head before this rule the transaction's
    preflight raised for the whole step ("plan.request_ids must be nonempty
    strings")."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    made = stream(env, EMPTY_ID) if kind == "stream" else file(env, EMPTY_ID)
    delivered = {
        "bad": _through_core(made),
        "s1": _through_core(stream(env, "s1")),
        "f1": _through_core(file(env, "f1")),
    }
    assert delivered["bad"][0].req_id == EMPTY_ID
    error = _mixed_step(env, delivered, order)
    assert error is None, repr(error)
    assert env.runner.failed_log == [None]
    statuses, failed = env.runner.commit_log[-1]
    assert not failed and statuses.get(EMPTY_ID) == 0, statuses
    if order != "alone":
        _assert_healthy_rows_committed(env, delivered["s1"][0])
    if kind == "stream":
        (feature,) = delivered["bad"][0].mm_features
        (own,) = env.model.embed_multimodal(audio=[feature.data["audio"].data])
        cached = env.state.encoder_cache.encoder_outputs[feature.identifier]
        assert torch.equal(cached, own)


# @spec REG-STATE-022, REG-STATE-011, REG-STATE-016
def test_an_empty_request_id_lives_through_turns_retirement_and_reuse(build):
    """The empty ID through a request's whole life: a stream's first unit,
    its park, its next unit folded into it by core's own scheduler (the
    pending removal and the continuing ``add_request``), its park and its
    finish; then a complete file under the same ID on the same slot, which
    core may reuse once the stream has finished, to its end. Every step
    commits with no row failed, and the stream keeps its admission
    generation across the turn while the file gets a new one."""
    env = build(max_num_seqs=3)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    handles = {EMPTY_ID: 1}
    first = stream(env, EMPTY_ID)
    env.runner.real_step(h.scheduler_output(new=(first,), handles=handles))
    generation = env.state._requests[EMPTY_ID].handle.generation
    drain_to_park(env, [EMPTY_ID], handles, watch=EMPTY_ID)
    turn, inputs = _core_folded_turn(
        env, first, stream(env, EMPTY_ID, sequence=1, final=True)
    )
    assert turn.req_id == EMPTY_ID and int(turn.num_computed_tokens) == 1
    output = h.scheduler_output(
        new=(turn,), handles=handles, encoder_inputs={EMPTY_ID: inputs}
    )
    error = h.call_with(env.runner.real_step, output)
    assert error is None, repr(error)
    assert env.state._requests[EMPTY_ID].handle.generation == generation
    # the folded turn's row is at position 1; its drains follow it
    drain_to_park(env, [EMPTY_ID], handles, watch=EMPTY_ID, start=2)
    env.runner.real_step(h.scheduler_output(finished=(EMPTY_ID,)))
    # held pending removal until the model state's next call (REG-STATE-011)
    assert env.state._pending_removal == EMPTY_ID
    later = file(env, EMPTY_ID)
    error = h.call_with(
        env.runner.real_step, h.scheduler_output(new=(later,), handles=handles)
    )
    assert error is None, repr(error)
    assert env.state._requests[EMPTY_ID].handle.generation > generation
    drain_to_park(env, [EMPTY_ID], handles, watch=EMPTY_ID)
    assert all(failed is None for failed in env.runner.failed_log)
    assert_commits_clean(env, steps=len(env.runner.commit_log))


def _masked(feature):
    """``feature`` placed with an ``int64`` embedding mask over its one
    position."""
    from vllm.multimodal.inputs import PlaceholderRange

    placement = PlaceholderRange(offset=0, length=1, is_embed=torch.tensor([1]))
    return dataclasses.replace(feature, mm_position=placement)


#: Faults of a row's prompt structure or of its feature's placement, whose
#: own item is valid (REG-STATE-015): name -> (request builder over the
#: shared feature, the row's scheduled encoder inputs).
_STRUCTURE_FAULTS = {
    "non-CHUNK row a feature overlaps": (
        lambda env, shared: _carrier_request(env, "bad", token=7, features=[shared]),
        [0],
    ),
    "CHUNK row two features overlap": (
        lambda env, shared: _carrier_request(
            env, "bad", features=[shared, _second_feature(env, "bad")]
        ),
        [0, 1],
    ),
    "feature with an int64 embedding mask": (
        lambda env, shared: _carrier_request(env, "bad", features=[_masked(shared)]),
        [0],
    ),
}


@pytest.mark.parametrize("when", ["same step", "later step"])
@pytest.mark.parametrize("fault", list(_STRUCTURE_FAULTS))
# @spec REG-STATE-015
def test_a_structure_fault_keeps_the_encoder_output_a_shared_item_needs(
    build, fault, when
):
    """Core's scheduler encodes an identifier once, for the first request it
    schedules, and treats it as cached for every other request whose feature
    has that identifier, in the same step or a later one. A row that fails
    for a prompt-structure fault, or for its feature's placement (which is
    the request's, not the item's), still runs the encoder inputs whose own
    item is valid, so another request with the same item gathers its encoder
    output: the faulty request alone fails and the other one commits."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    shared = _envelope_feature(env, "good", identifier="uuid-shared")
    make, inputs = _STRUCTURE_FAULTS[fault]
    bad = make(env, shared)
    good = _carrier_request(env, "good", features=[shared])
    if when == "same step":
        steps = [
            h.scheduler_output(
                new=(bad, good),
                handles={"bad": 3, "good": 1},
                encoder_inputs={"bad": inputs},
            )
        ]
    else:
        steps = [
            h.scheduler_output(
                new=(bad,), handles={"bad": 3}, encoder_inputs={"bad": inputs}
            ),
            h.scheduler_output(
                new=(good,),
                finished=("bad",),
                handles={"good": 1},
                encoder_inputs={},
            ),
        ]
    for output in steps:
        error = h.call_with(env.runner.real_step, output)
        assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}] + [None] * (len(steps) - 1)
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("good") == 0


# ---------------------------------------------------------------------------
# REG-STATE-020: the endpoint policy, from the sampling parameters' extra_args

#: A well-formed endpoint policy with distinct counts.
_POLICY = {"mode": "greedy_blank", "threshold_frames": 2, "residue_frames": 1}


def _endpoint_row(env, req_id: str) -> tuple[int, int, int]:
    """The endpoint mode, threshold and residue of ``req_id``'s row in the
    step's row plan (the row transaction's plan)."""
    plan = env.advance.calls[-1]
    row = plan.request_ids.index(req_id)
    return (
        int(plan.endpoint_mode[row]),
        int(plan.endpoint_threshold_frames[row]),
        int(plan.endpoint_residue_frames[row]),
    )


# @spec REG-STATE-020
def test_the_endpoint_policy_is_read_from_the_sampling_extra_args(build):
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    policy = h.with_extra_args(stream(env, "p"), {"endpoint_policy": dict(_POLICY)})
    error = _step_with_a_clean_row(env, policy, {"p": [0]})
    assert error is None, repr(error)
    assert env.runner.failed_log == [None]
    assert _endpoint_row(env, "p") == (1, 2, 1)
    assert _endpoint_row(env, "s1") == (0, 0, 0)


def _without_the_key(env, carrier: str):
    request = stream(env, "p")
    if carrier == "extra_args without the key":
        return h.with_extra_args(request, {"other_option": 1})
    if carrier == "a policy only in additional_information":
        # The request field an earlier design read; core carries no such field
        # to the model state, so a policy there is never read.
        request.additional_information = {"endpoint_policy": dict(_POLICY)}
    return request


@pytest.mark.parametrize(
    "carrier",
    [
        "no extra_args",
        "extra_args without the key",
        "a policy only in additional_information",
    ],
)
# @spec REG-STATE-020
def test_a_request_without_the_policy_key_runs_with_endpointing_disabled(
    build, carrier
):
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    error = _step_with_a_clean_row(env, _without_the_key(env, carrier), {"p": [0]})
    assert error is None, repr(error)
    assert env.runner.failed_log == [None]
    assert _endpoint_row(env, "p") == (0, 0, 0)


#: Endpoint policies a direct engine client can send that are not a mapping
#: of a known mode and non-negative integer counts.
_MALFORMED_POLICIES = {
    "unknown mode": {"mode": "eager", "threshold_frames": 2, "residue_frames": 0},
    "no mode": {"threshold_frames": 2, "residue_frames": 0},
    "negative threshold": {
        "mode": "greedy_blank",
        "threshold_frames": -1,
        "residue_frames": 0,
    },
    "fractional residue": {
        "mode": "greedy_blank",
        "threshold_frames": 2,
        "residue_frames": 0.5,
    },
    "string threshold": {
        "mode": "greedy_blank",
        "threshold_frames": "2",
        "residue_frames": 0,
    },
    "boolean threshold": {
        "mode": "greedy_blank",
        "threshold_frames": True,
        "residue_frames": 0,
    },
    "not a mapping": "greedy_blank",
}


@pytest.mark.parametrize("row", ["stream", "file"])
@pytest.mark.parametrize("policy", list(_MALFORMED_POLICIES))
# @spec REG-STATE-020, REG-STATE-015
def test_a_malformed_endpoint_policy_fails_that_request_alone(build, policy, row):
    """On a stream row and on a file row (which runs with endpointing
    disabled, FILE-ENG-002, but whose policy is still the request's): the
    row fails at planning, the step does not raise, and the step's clean
    stream row commits."""
    env = build()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    bad = stream(env, "bad") if row == "stream" else file(env, "bad")
    bad = h.with_extra_args(bad, {"endpoint_policy": _MALFORMED_POLICIES[policy]})
    error = _step_with_a_clean_row(env, bad, {"bad": [0]})
    assert error is None, repr(error)
    assert env.runner.failed_log == [{"bad"}]
    statuses, _ = env.runner.commit_log[-1]
    assert statuses.get("s1") == 0


# ===========================================================================
# Graph runtime
# ===========================================================================


def _package_sources(repo_root):
    root = repo_root / "vllm_nemotron_asr_streaming"
    return {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*.py")}


# @spec REG-MEM-007
def test_package_neither_imports_nor_calls_a_capture_enabled_switch(repo_root):
    found = []
    for name, source in _package_sources(repo_root).items():
        for node in ast.walk(ast.parse(source)):
            text = None
            if isinstance(node, ast.ImportFrom) and node.module:
                text = node.module + " " + " ".join(a.name for a in node.names)
            elif isinstance(node, (ast.Name, ast.Attribute)):
                text = node.id if isinstance(node, ast.Name) else node.attr
            if text and re.search(r"captur\w*_enabled|compilation\.monitor", text):
                found.append(f"{name}:{node.lineno}: {text}")
    assert not found, "\n".join(found)


# @spec REG-MEM-007
def test_graph_runtime_uses_cores_capture_context_forward_context_and_wrapper(
    monkeypatch,
):
    import vllm.distributed.parallel_state as parallel_state
    import vllm.forward_context as forward_context
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import (
        platform_graph_runtime,
    )

    captured = []
    monkeypatch.setattr(
        parallel_state, "graph_capture", lambda **kw: captured.append(kw) or "ctx"
    )
    runtime = platform_graph_runtime()
    assert runtime.forward_context is forward_context.set_forward_context
    assert runtime.wrapper_factory is resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls()
    )
    assert runtime.capture_context(torch.device("cpu")) == "ctx"
    assert captured == [{"device": torch.device("cpu")}]


# ===========================================================================
# GPU: live capture
# ===========================================================================


def _prepare_graphed(env) -> None:
    """Prepare through the first profiling step (REG-MEM-002)."""
    env.runner.profile_run(core_embeds=False)


@pytest.mark.parametrize("arm", ["dense-graphed", "default"])
# @spec REG-MEM-006
def test_serving_refuses_a_graphed_step_whose_wrapper_entry_lost_its_graph_cpu(
    build, arm
):
    """CPU, ``FakeGraphRuntime``: a good real step replays graphs; one entry
    that step replayed loses its graph; the identical step is refused with no
    capture, no wrapper invocation and no state write; with the graph restored
    the identical step succeeds again (which attributes the refusal to the
    missing graph). No exception type or message is pinned, and the check may
    sit inside the row transaction."""
    env = build(arm=arm)
    _prepare_graphed(env)
    spy = pool(env)

    def step():
        # the session's second unit, delivered once the first drained to park
        return h.scheduler_output(
            new=(stream(env, "s1", sequence=1),), handles={"s1": 1}
        )

    replays = len(env.graphs.replays)
    env.runner.real_step(
        h.scheduler_output(new=(stream(env, "s1"),), handles={"s1": 1})
    )
    new = env.graphs.replays[replays:]
    assert new, "the prepared step replayed no graph"
    wrapper_ref, descriptor = new[-1]
    drain_to_park(env, ["s1"], {"s1": 1})
    entry = wrapper_ref().concrete_cudagraph_entries[descriptor]
    graph, entry.cudagraph = entry.cudagraph, None
    captures, replays = len(env.graphs.captures), len(env.graphs.replays)
    before = h.pool_tensor(spy).clone()
    error = h.call_with(env.runner.real_step, step())
    assert error is not None, "the step ran without its graph"
    assert len(env.graphs.captures) == captures, "serving attempted a capture"
    assert len(env.graphs.replays) == replays, "a wrapper was invoked"
    assert entry.cudagraph is None
    assert torch.equal(before, h.pool_tensor(spy)), "the refused step wrote state"
    entry.cudagraph = graph
    env.runner.real_step(step())
    assert len(env.graphs.replays) > replays, "the restored step did not replay"
    # every step that committed (the refused one raised before committing) was clean
    assert env.runner.commit_log, "no step committed"
    assert all(not failed for _, failed in env.runner.commit_log)
    assert all(v == 0 for st, _ in env.runner.commit_log for v in st.values())


@pytest.mark.gpu
# @spec REG-MEM-002, REG-MEM-006
def test_serving_refuses_a_graphed_step_whose_wrapper_entry_lost_its_graph(
    monkeypatch, tmp_path, single_process_dist
):
    """CUDA, core's own graph runtime and static graph wrapper, the card's
    default arms under ``--enforce-eager``: as the CPU test. Wrapper
    invocations are counted by instrumenting core's
    ``CUDAGraphWrapper.__call__`` (no model-internal attribute is read)."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import get_forward_context

    calls: list = []
    original = CUDAGraphWrapper.__call__

    def instrumented(self, *args, **kwargs):
        context = get_forward_context()
        descriptor = context.batch_descriptor
        entry = self.concrete_cudagraph_entries.get(descriptor)
        capturing = entry is None or entry.cudagraph is None
        if context.cudagraph_runtime_mode == self.runtime_mode:
            calls.append((self, descriptor, "capture" if capturing else "replay"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CUDAGraphWrapper, "__call__", instrumented)
    card = h.write_runnable_card(tmp_path / "card")
    vllm_config = h.build_vllm_config(card, right_context=6, device="cuda")
    model = h.build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    runner = h.Runner(vllm_config, model, state)
    runner.profile_run(core_embeds=False)
    assert any(kind == "capture" for _, _, kind in calls), "nothing was captured"
    bindings = runner.bind_pool(h.real_bindings(h.model_spec(model), device="cuda"))
    # the helpers of the CPU test, over this runner
    env = SimpleNamespace(
        runner=runner, config=model.config, max_geometry=3, model=model
    )

    def step():
        # the session's second unit, delivered once the first drained to park
        return h.scheduler_output(
            new=(stream(env, "s1", sequence=1),), handles={"s1": 1}
        )

    mark = len(calls)
    runner.real_step(h.scheduler_output(new=(stream(env, "s1"),), handles={"s1": 1}))
    replayed = [(w, d) for w, d, kind in calls[mark:] if kind == "replay"]
    assert replayed, "the prepared serving step replayed no graph"
    wrapper, descriptor = replayed[-1]
    drain_to_park(env, ["s1"], {"s1": 1})
    entry = wrapper.concrete_cudagraph_entries[descriptor]
    graph, entry.cudagraph = entry.cudagraph, None
    mark = len(calls)
    before = h.pool_tensor(bindings).clone()
    error = h.call_with(runner.real_step, step())
    assert error is not None, "the step ran without its graph"
    assert calls[mark:] == [], "a wrapper was invoked (capture or replay)"
    assert entry.cudagraph is None
    assert torch.equal(before, h.pool_tensor(bindings)), "the refused step wrote state"
    entry.cudagraph = graph
    runner.real_step(step())
    assert any(kind == "replay" for _, _, kind in calls[mark:]), (
        "the restored step did not replay"
    )
    # every step that committed (the refused one raised before committing) was clean
    assert runner.commit_log, "no step committed"
    assert all(not failed for _, failed in runner.commit_log), runner.commit_log
    assert all(v == 0 for st, _ in runner.commit_log for v in st.values()), (
        runner.commit_log
    )


# ===========================================================================
# Runner configurations and core's model-state hooks
# ===========================================================================
#
# Core calls, in every runner configuration: ``prepare_execution_memory`` once
# before the profiling forward (``Runner.profile_run``), the profiling forward,
# ``post_warmup`` before the worker serves and ``shutdown`` last. Under the
# hook configuration only: ``needs_cudagraph_capture``, then
# ``capture_cudagraphs(profile_only=True)`` for the graph-memory estimate (over
# core's temporary profiling pools, into a throwaway pool), ``clear_cudagraphs``,
# the state pool, and ``capture_cudagraphs(profile_only=False)`` with the global
# pool and the ``graph_capture`` stream. These tests call the hooks in that
# order on the model's real model state (``support.core_order.CoreOrder``).


@pytest.fixture
def hook(build, monkeypatch, platform):
    """``hook(arm, ...)``: ``build`` under the hook configuration
    (``enforce_eager`` off, a CUDA platform from model-state construction on)
    with a profiled budget: ``kv_cache_memory_bytes`` unset, core's
    graph-memory estimate on, the extensible KV cache off."""

    def make(arm="default", *, kv_cache_memory_bytes=None, estimate=True, **kwargs):
        env = build(arm=arm, before_state=lambda: platform(True), **kwargs)
        env.vllm_config.model_config.enforce_eager = False
        cache = env.vllm_config.cache_config
        cache.kv_cache_memory_bytes = kv_cache_memory_bytes
        cache.enable_extensible_kv_cache = False
        monkeypatch.setenv(
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1" if estimate else "0"
        )
        env.streams = []
        return env

    return make


def _stream(env, name: str):
    stream = env.fake.new_stream(name=name)
    env.streams.append(stream)
    return stream


def _capture(env, *, profile_only, stream=None):
    """One capture call, with core's pool and a named stream of its own."""
    stream = stream or _stream(env, "profiling" if profile_only else "serving")
    env.core.capture(profile_only=profile_only, stream=stream)
    return stream


def _wrapper_pools(env) -> set:
    return {w.graph_pool for w, _ in env.graphs.captured()}


def _finals(env) -> list[tuple]:
    return [shape_of(c) for c in env.profile.calls if is_final(c)]


def _startup(env, *, serve=True):
    """Core's startup under the hook configuration up to (and with ``serve``,
    including) the serving capture, each capture on a named stream."""
    core = env.core
    core.prepare()
    core.bind_profiling()
    _capture(env, profile_only=True)
    core.clear()
    if serve:
        pool(env)
        _capture(env, profile_only=False)
    return env


def _causes(error: BaseException | None) -> list[BaseException]:
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    return chain


# ---------------------------------------------------------------------------
# REG-MEM-019: needs_cudagraph_capture
# ---------------------------------------------------------------------------


def _transaction_graph_check_owner(repo_root):
    """The module that defines ``transaction_graph_config_errors`` (``None``
    before it exists)."""
    import importlib

    root = repo_root / "vllm_nemotron_asr_streaming"
    for path in sorted(root.rglob("*.py")):
        if "def transaction_graph_config_errors" in path.read_text():
            name = ".".join(path.relative_to(repo_root).with_suffix("").parts)
            return importlib.import_module(name)
    return None


#: served-configuration fields → whether a hook-captured graph is selected.
_PREDICATE = {
    "default arms": ({}, True),
    "graphed encoder only": ({"decode_dispatch_arm": "dense-eager"}, True),
    "graphed decoder only": ({"encoder_execution_arm": "eager"}, True),
    "eager arms": (dict(EAGER_ARMS), False),
    "compiled-static": (
        {
            "encoder_execution_arm": "compiled-static",
            "decode_dispatch_arm": "compact-eager",
        },
        False,
    ),
    "compiled graphed encoder alone": (
        {
            "encoder_execution_arm": "dense-graphed",
            "decode_dispatch_arm": "dense-eager",
        },
        False,
    ),
    "chunk buckets on the eager arms": (
        {**EAGER_ARMS, "chunk_bucket_graph_populations": [1]},
        False,
    ),
    "transaction graphs on the eager arms": (
        {**EAGER_ARMS, "transaction_graph_populations": [2]},
        True,
    ),
    "malformed transaction graphs (on)": (
        {**EAGER_ARMS, "transaction_graph_populations": {}},
        True,
    ),
    "empty transaction graphs (off)": (
        {**EAGER_ARMS, "transaction_graph_populations": []},
        False,
    ),
}


@pytest.mark.parametrize("cuda", [True, False], ids=["cuda", "not-cuda"])
@pytest.mark.parametrize("case", list(_PREDICATE))
# @spec REG-MEM-019
def test_needs_cudagraph_capture_is_the_hook_predicate(
    make_model_config,
    make_vllm_config,
    card_dir,
    build_model_state,
    platform,
    monkeypatch,
    repo_root,
    case,
    cuda,
):
    fields, selects = _PREDICATE[case]
    vllm_config = make_vllm_config(make_model_config(card_dir, lookahead=0))
    for key, value in fields.items():
        setattr(vllm_config.model_config.hf_config, key, value)
    if "transaction_graph_populations" in fields:
        # The option's own checks (TXG-CFG-008, TXG-CFG-009) refuse these arms
        # and this runner at construction; they are the transaction-graph
        # component's specs, isolated here.
        owner = _transaction_graph_check_owner(repo_root)
        if owner is not None:
            monkeypatch.setattr(
                owner, "transaction_graph_config_errors", lambda *a, **k: []
            )
    platform(cuda)
    state = build_model_state(vllm_config)
    answer = state.needs_cudagraph_capture()
    assert answer is (selects and cuda)
    # The same value for the model state's lifetime: configuration and
    # platform read once.
    platform(not cuda)
    vllm_config.model_config.hf_config.decode_dispatch_arm = (
        "dense-eager" if answer else "dense-graphed"
    )
    assert state.needs_cudagraph_capture() is answer
    assert state.needs_cudagraph_capture() is answer


# ---------------------------------------------------------------------------
# REG-MEM-018: prepare_execution_memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", ["enforce-eager", "hook"])
@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-018
def test_prepare_execution_memory_captures_nothing_and_runs_no_profile(
    build, hook, runner, arm
):
    if runner == "hook" and arm not in HOOK_ARMS:
        pytest.skip("the hook configuration refuses this arm (REG-ENG-005)")
    env = hook(arm) if runner == "hook" else build(arm=arm)
    env.runner.prepare_execution_memory()
    assert env.graphs.captures == [], "prepare_execution_memory captured"
    assert _finals(env) == [], "prepare_execution_memory ran the profile transaction"


# @spec REG-MEM-018, REG-MEM-002
def test_enforce_eager_captures_on_the_first_profiling_step_after_preparation(build):
    """``--enforce-eager``: core calls no capture hook; preparation captures
    nothing and the first profiling step captures every expected graph."""
    env = build(arm="default")
    env.runner.prepare_execution_memory()
    assert env.graphs.captures == []
    env.runner.profile_run(core_embeds=False)
    assert len(env.graphs.captures) == expected_entries(env)
    assert len(env.graphs.captured()) == expected_entries(env)


# ---------------------------------------------------------------------------
# REG-MEM-016: the profiling step under the hook configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", HOOK_ARMS)
# @spec REG-MEM-016, REG-MEM-010
def test_hook_profiling_step_profiles_once_uncaptured(hook, arm):
    env = hook(arm)
    _, output, embeds = env.runner.profile_run(core_embeds=False)
    assert_dummy_output(output, embeds)
    assert env.graphs.captures == [], "the profiling step captured"
    assert _finals(env) == [(env.max_num_seqs, env.max_geometry)], env.profile.calls


@pytest.mark.parametrize("arm", HOOK_ARMS)
# @spec REG-MEM-003, REG-MEM-016
def test_hook_later_profiling_steps_accept_pending_segment_graphs(hook, arm):
    """Before the serving capture, later profiling steps verify readiness with
    the segment graphs pending, and neither capture nor profile again."""
    env = hook(arm)
    env.runner.profile_run(core_embeds=False)
    profiles = len(env.profile.calls)
    trap = trapped_pool(env)
    for _ in range(2):
        _, output, embeds = env.runner.profile_run(
            randomize_inputs=True, attn=True, core_embeds=False
        )
        assert_dummy_output(output, embeds)
    assert len(env.profile.calls) == profiles
    assert env.graphs.captures == []
    assert trap.reads == []


# ---------------------------------------------------------------------------
# REG-MEM-020, REG-MEM-021, REG-MEM-022: the capture hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", HOOK_ARMS)
# @spec REG-MEM-020, REG-MEM-021, REG-MEM-007
def test_profiling_capture_uses_cores_pool_and_stream_and_clear_releases_it(hook, arm):
    env = hook(arm)
    expected = expected_entries(env)
    assert env.state.needs_cudagraph_capture() is True
    env.runner.profile_run(core_embeds=False)
    streams_before = len(env.graphs.capture_streams)
    stream = _capture(env, profile_only=True)
    assert len(env.graphs.captures) == expected
    assert len(env.graphs.captured()) == expected
    assert _wrapper_pools(env) == {PROFILING_POOL}
    used = env.graphs.capture_streams[streams_before:]
    assert used and {id(s) for s in used} == {id(stream)}, (
        "a capture ran on a stream core did not pass"
    )
    env.state.clear_cudagraphs()
    gc.collect()
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0
    env.state.clear_cudagraphs()  # nothing left: tolerated


@pytest.mark.parametrize("arm", HOOK_ARMS)
# @spec REG-MEM-022, REG-MEM-011, REG-MEM-007
def test_serving_capture_rebuilds_into_the_global_pool_and_serves(hook, arm):
    env = hook(arm)
    expected = expected_entries(env)
    _startup(env, serve=False)
    spy = pool(env)
    # A real step before the serving capture is refused.
    assert h.call_with(env.runner.real_step, rows_of(env, "stream")) is not None
    assert env.advance.calls == []
    streams_before = len(env.graphs.capture_streams)
    capture_stream = _capture(env, profile_only=False)
    assert len(env.graphs.captures) == 2 * expected
    assert len(env.graphs.captured()) == expected
    assert _wrapper_pools(env) == {SERVING_POOL}
    used = env.graphs.capture_streams[streams_before:]
    assert used and {id(s) for s in used} == {id(capture_stream)}
    replays = len(env.graphs.replays)
    output = h.scheduler_output(new=(stream(env, "s1"),), handles={"s1": 1})
    env.runner.real_step(output)
    assert env.advance.calls, "the step did not run"
    assert len(env.graphs.replays) > replays, "the step replayed no graph"
    assert spy.bind_calls


# @spec REG-MEM-022, REG-MEM-027
def test_serving_capture_within_the_profiling_total_passes(hook):
    env = hook()
    _startup(env)
    assert len(env.graphs.captured()) == expected_entries(env)


def _amounts(value: int) -> list[str]:
    """Renderings of a byte count an error may use."""
    mib = value / (1 << 20)
    return [
        str(value),
        f"{value:,}",
        f"{mib:.0f} MiB",
        f"{mib:.1f} MiB",
        f"{mib:.2f} MiB",
        f"{mib:g} MiB",
    ]


def _names_amount(message: str, value: int) -> bool:
    return any(text in message for text in _amounts(value))


REMEDY_WORDS = (
    "--max-num-seqs",
    "encoder_execution_arm",
    "decode_dispatch_arm",
    "dense-eager",
    "chunk_bucket_graph_populations",
    "transaction_graph_populations",
    "--enforce-eager",
)


# @spec REG-MEM-022, REG-MEM-027, REG-MEM-017
def test_serving_capture_over_the_profiling_total_fails_startup(hook):
    """The serving call retains 1 MiB more per graph than the profiling call
    (whose margin charge is zero here): startup fails naming both totals, the
    margin's charge, the excess and the remedies, and the call's segment
    graphs are discarded."""
    env = hook()
    expected = expected_entries(env)
    per_graph = env.fake.per_segment_graph
    _startup(env, serve=False)
    pool(env)
    env.fake.per_segment_graph = per_graph + (1 << 20)
    stream = _stream(env, "serving")
    error = h.call_with(_capture, env, profile_only=False, stream=stream)
    assert error is not None, "a serving capture above the profiling total passed"
    message = str(error)
    assert _names_amount(message, expected * per_graph), message
    assert _names_amount(message, expected * env.fake.per_segment_graph), message
    assert _names_amount(message, expected << 20), message  # the excess
    assert "margin" in message.lower(), message
    for word in (*REMEDY_WORDS, "--gpu-memory-utilization"):
        assert word in message, (word, message)
    _release(error)
    gc.collect()
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0
    assert ("cuda.stream.synchronize", "serving") in env.fake.events, (
        "the capture stream was not synchronized"
    )


# ---------------------------------------------------------------------------
# REG-MEM-023: no profiling capture
# ---------------------------------------------------------------------------


# @spec REG-MEM-023, REG-MEM-027
def test_a_required_but_absent_profiling_capture_fails_at_entry(hook):
    env = hook()
    env.runner.profile_run(core_embeds=False)
    pool(env)
    error = h.call_with(_capture, env, profile_only=False)
    assert error is not None, "serving capture accepted a missing graph-memory estimate"
    assert env.graphs.captures == [], "the refused call captured"
    message = str(error)
    assert (
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" in message
        or "kv_cache_memory_bytes" in message
    ), message


#: Startups that require no profiling capture with the extensible KV cache
#: off: the segment graphs are charged nowhere (``hook`` keyword arguments,
#: then the reason the WARNING names).
_UNCHARGED = {
    "fixed budget": ({"kv_cache_memory_bytes": 1 << 30}, "kv_cache_memory_bytes"),
    "estimate off": ({"estimate": False}, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
}

#: Startups under the effective extensible KV cache, which core sizes from the
#: device memory left after the captures whatever the budget or estimate
#: setting: a profiled budget, the estimate disabled, and a budget that a
#: startup plan sets after construction with the extensible KV cache left on.
_MEASURED = {
    "extensible KV cache": {},
    "extensible, estimate off": {"estimate": False},
    "extensible, startup-plan budget": {"kv_cache_memory_bytes": 1 << 30},
}


def _plugin_records(captured_logs, level):
    return [
        r
        for r in captured_logs.records
        if r.levelno == level and r.name.startswith("vllm_nemotron_asr_streaming")
    ]


@pytest.mark.parametrize("case", list(_UNCHARGED))
# @spec REG-MEM-023, REG-MEM-027
def test_an_unrequired_profiling_capture_warns_once_after_capture(
    hook, captured_logs, case
):
    """With the extensible KV cache off and either a fixed budget or the
    estimate disabled, one WARNING says that the segment graphs were not
    charged to the budget, with the reason and their retained graph
    memory."""
    settings, reason = _UNCHARGED[case]
    env = hook(**settings)
    env.runner.profile_run(core_embeds=False)
    pool(env)
    _capture(env, profile_only=False)
    expected = expected_entries(env)
    assert len(env.graphs.captured()) == expected
    warnings = [
        r
        for r in _plugin_records(captured_logs, logging.WARNING)
        if re.search(r"budget|charg", r.getMessage(), re.I)
    ]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    message = warnings[0].getMessage()
    assert _names_amount(message, expected * env.fake.per_segment_graph), message
    assert reason.lower() in message.lower(), message


@pytest.mark.parametrize("case", list(_MEASURED))
# @spec REG-MEM-023, REG-MEM-027
def test_an_unrequired_profiling_capture_under_the_extensible_kv_cache_is_measured(
    hook, captured_logs, case
):
    """The effective extensible KV cache is checked before the budget and the
    estimate setting: whatever they are, one INFO line says that core sizes
    the KV cache from the device memory left after the captures, with the
    reason and the segment graphs' retained graph memory, and no WARNING
    says that they were not charged."""
    env = hook(**_MEASURED[case])
    env.vllm_config.cache_config.enable_extensible_kv_cache = True
    env.runner.profile_run(core_embeds=False)
    pool(env)
    _capture(env, profile_only=False)
    expected = expected_entries(env)
    assert len(env.graphs.captured()) == expected
    uncharged = [
        r.getMessage()
        for r in _plugin_records(captured_logs, logging.WARNING)
        if re.search(r"not charged|budget must|charg", r.getMessage(), re.I)
    ]
    assert uncharged == [], uncharged
    measured = [
        r.getMessage()
        for r in _plugin_records(captured_logs, logging.INFO)
        if re.search(r"(?i)after the captures", r.getMessage())
    ]
    assert len(measured) == 1, measured
    message = measured[0]
    assert re.search(r"(?i)extensible", message), message
    assert re.search(r"(?i)sizes? the kv cache|kv cache[^.]{0,40}siz", message), message
    assert _names_amount(message, expected * env.fake.per_segment_graph), message


# ---------------------------------------------------------------------------
# REG-MEM-024: ordering and a second serving capture
# ---------------------------------------------------------------------------


# @spec REG-MEM-024
def test_a_second_serving_capture_is_refused_at_entry(hook):
    env = _startup(hook())
    captures = len(env.graphs.captures)
    error = h.call_with(_capture, env, profile_only=False)
    assert error is not None
    message = str(error).lower()
    assert "recapture" in message and "not supported" in message, message
    assert "restart" in message, message
    assert len(env.graphs.captures) == captures, "the refused call captured"
    assert len(env.graphs.captured()) == expected_entries(env), "graphs dropped"


@pytest.mark.parametrize("serving", ["succeeded", "failed"])
# @spec REG-MEM-024
def test_a_profiling_capture_after_a_serving_capture_is_refused(hook, serving):
    env = hook()
    _startup(env, serve=False)
    pool(env)
    if serving == "failed":
        env.graphs.fail_capture_after = 1
        assert h.call_with(_capture, env, profile_only=False) is not None
        env.graphs.disarm()
    else:
        _capture(env, profile_only=False)
    captures = len(env.graphs.captures)
    assert h.call_with(_capture, env, profile_only=True) is not None
    assert len(env.graphs.captures) == captures


# @spec REG-MEM-024
def test_a_serving_capture_while_profiling_graphs_exist_is_refused(hook):
    env = hook()
    env.runner.profile_run(core_embeds=False)
    _capture(env, profile_only=True)
    captures = len(env.graphs.captures)
    pool(env)
    assert h.call_with(_capture, env, profile_only=False) is not None
    assert len(env.graphs.captures) == captures
    assert _wrapper_pools(env) == {PROFILING_POOL}


# ---------------------------------------------------------------------------
# REG-MEM-025, REG-MEM-026: post_warmup and shutdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", ["enforce-eager", "hook"])
# @spec REG-MEM-025
def test_post_warmup_verifies_every_segment_graph_without_replaying(
    build, hook, runner
):
    env = hook() if runner == "hook" else build(arm="default")
    if runner == "hook":
        _startup(env)
    else:
        env.runner.profile_run(core_embeds=False)
    captures, replays = len(env.graphs.captures), len(env.graphs.replays)
    reads = env.fake.names().count("cuda.mem_get_info")
    env.state.post_warmup()
    assert len(env.graphs.captures) == captures and len(env.graphs.replays) == replays
    assert env.fake.names().count("cuda.mem_get_info") == reads
    captured = env.graphs.captured()
    assert len(captured) == expected_entries(env)
    for wrapper, descriptor in captured:
        entry = wrapper.concrete_cudagraph_entries[descriptor]
        graph, entry.cudagraph = entry.cudagraph, None
        error = h.call_with(env.state.post_warmup)
        assert error is not None, f"post_warmup accepted {descriptor} without a graph"
        assert len(env.graphs.captures) == captures, "post_warmup captured"
        entry.cudagraph = graph
    env.state.post_warmup()


# @spec REG-MEM-025
def test_post_warmup_before_the_serving_capture_is_refused(hook):
    env = hook()
    _startup(env, serve=False)
    assert h.call_with(env.state.post_warmup) is not None


@pytest.mark.parametrize("runner", ["enforce-eager", "hook"])
# @spec REG-MEM-026
def test_shutdown_releases_every_segment_graph(build, hook, runner):
    env = hook() if runner == "hook" else build(arm="default")
    if runner == "hook":
        _startup(env)
    else:
        env.runner.profile_run(core_embeds=False)
    assert env.graphs.captured()
    env.state.shutdown()
    gc.collect()
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0


@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-021, REG-MEM-026
def test_clear_and_shutdown_tolerate_nothing_captured(build, arm):
    env = build(arm=arm)
    env.state.clear_cudagraphs()
    env.state.shutdown()


# @spec REG-MEM-026
def test_shutdown_tolerates_a_model_state_whose_initialization_did_not_complete():
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        NemotronASRModelState,
    )

    NemotronASRModelState.__new__(NemotronASRModelState).shutdown()


# ---------------------------------------------------------------------------
# REG-MEM-005 under the hook configuration
# ---------------------------------------------------------------------------


# @spec REG-MEM-005
def test_hook_preparation_profile_and_capture_read_no_state_slots(hook):
    env = hook()
    trap = trapped_pool(env)
    pool_bytes = h.pool_tensor(trap._target).clone()
    env.runner.profile_run(core_embeds=False)  # preparation, then the profile
    _capture(env, profile_only=True)
    env.state.clear_cudagraphs()
    _capture(env, profile_only=False)
    assert trap.reads == []
    assert torch.equal(pool_bytes, h.pool_tensor(trap._target))


# ---------------------------------------------------------------------------
# REG-MEM-015: the capture line
# ---------------------------------------------------------------------------


def _capture_lines(captured_logs, env) -> list[str]:
    decoder = env.overrides.get("decode_dispatch_arm", "dense-graphed")
    return [
        r.getMessage()
        for r in captured_logs.records
        if r.levelno == logging.INFO
        and r.name.startswith("vllm_nemotron_asr_streaming")
        and decoder in r.getMessage()
        and "max_num_seqs" in r.getMessage()
    ]


#: The information each capture line names, as patterns any of whose
#: renderings will do: the runner configuration and the capture.
RUNNERS = {"enforce-eager": r"eager", "hook": r"hook"}
CAPTURES = {
    "profiling dummy step": (r"profil", r"dummy"),
    "profiling capture": (r"profil",),
    "serving capture": (r"serv",),
}


def _assert_capture_line(line: str, env, *, runner: str, capture: str) -> None:
    encoder = env.overrides.get("encoder_execution_arm", "eager-graphed")
    assert encoder in line, line
    assert re.search(RUNNERS[runner], line, re.I), line
    for pattern in CAPTURES[capture]:
        assert re.search(pattern, line, re.I), (capture, line)
    rc = env.right_context
    assert CADENCES[LOOKAHEADS.index(rc)] in line or re.search(
        rf"lookahead\D{{0,10}}{rc}\b", line
    ), line
    assert re.search(rf"max_num_seqs\D{{0,10}}{env.max_num_seqs}\b", line), line
    assert re.search(r"\d(\.\d+)?\s*(ms|s|sec|seconds?)\b", line), (
        f"no capture time: {line}"
    )


@pytest.mark.parametrize("arm", GRAPHED)
# @spec REG-MEM-015
def test_the_profiling_dummy_step_capture_logs_one_line(build, captured_logs, arm):
    env = build(arm=arm)
    env.runner.profile_run(core_embeds=False)
    lines = _capture_lines(captured_logs, env)
    assert len(lines) == 1, lines
    _assert_capture_line(
        lines[0], env, runner="enforce-eager", capture="profiling dummy step"
    )
    assert _names_amount(
        lines[0], expected_entries(env) * env.fake.per_segment_graph
    ) or (re.search(r"\d+(\.\d+)?\s*(B|KiB|MiB|GiB)\b", lines[0])), lines[0]


# @spec REG-MEM-015, REG-MEM-027
def test_each_hook_capture_logs_one_line_with_its_retained_memory(hook, captured_logs):
    env = hook()
    expected = expected_entries(env)
    env.runner.profile_run(core_embeds=False)
    assert _capture_lines(captured_logs, env) == [], (
        "the profiling step logged a capture"
    )
    _capture(env, profile_only=True)
    (profiling,) = _capture_lines(captured_logs, env)
    _assert_capture_line(profiling, env, runner="hook", capture="profiling capture")
    assert _names_amount(profiling, expected * env.fake.per_segment_graph), profiling
    env.state.clear_cudagraphs()
    pool(env)
    _capture(env, profile_only=False)
    lines = _capture_lines(captured_logs, env)
    assert len(lines) == 2, lines
    _assert_capture_line(lines[1], env, runner="hook", capture="serving capture")
    assert _names_amount(lines[1], expected * env.fake.per_segment_graph), lines[1]


# ---------------------------------------------------------------------------
# REG-MEM-027: the measurement
# ---------------------------------------------------------------------------

#: The allocator peak statistics' resets, as ``FakeCuda`` logs them.
PEAK_RESETS = ("cuda.reset_peak_memory_stats", "cuda.reset_max_memory_allocated")


@pytest.mark.parametrize("profile_only", [True, False], ids=["profiling", "serving"])
# @spec REG-MEM-027
def test_each_free_memory_reading_follows_a_sync_and_a_cache_release(
    hook, profile_only
):
    """Over a capture call, every reading of free memory is preceded, since
    the previous reading, by a synchronization of the capture stream (or the
    device) and a release of the allocator's cached blocks; no allocator peak
    statistic is reset."""
    env = hook()
    env.runner.profile_run(core_embeds=False)
    if not profile_only:
        _capture(env, profile_only=True)
        env.state.clear_cudagraphs()
        pool(env)
    start = len(env.fake.events)
    name = "profiling" if profile_only else "serving"
    _capture(env, profile_only=profile_only, stream=_stream(env, name))
    log = env.fake.events[start:]
    reads = [i for i, (kind, _) in enumerate(log) if kind == "cuda.mem_get_info"]
    assert len(reads) >= 2, f"the call was not measured across: {log}"
    previous = -1
    for index in reads:
        window = log[previous + 1 : index]
        assert any(
            k == "cuda.synchronize" or (k == "cuda.stream.synchronize" and d == name)
            for k, d in window
        ), f"a reading without a synchronization before it: {log}"
        assert any(k == "cuda.empty_cache" for k, _ in window), (
            f"a reading without a cache release before it: {log}"
        )
        previous = index
    assert not set(PEAK_RESETS) & {k for k, _ in log}, log


# @spec REG-MEM-027
def test_the_enforce_eager_capture_resets_no_peak_statistic(build):
    """Under ``--enforce-eager`` the capture runs inside core's
    memory-profiling window, which reads the peak statistics."""
    env = build(arm="default")
    env.runner.profile_run(core_embeds=False)
    assert not set(PEAK_RESETS) & set(env.fake.names()), env.fake.events


# ---------------------------------------------------------------------------
# REG-MEM-017: capture failure
# ---------------------------------------------------------------------------


def _assert_failure_message(error, env, *, extra=()) -> None:
    message = str(error)
    chain = _causes(error)
    assert any("capture" in str(c) and "failed" in str(c) for c in chain[1:]), (
        "the error is not chained from the original failure"
    )
    encoder = env.overrides.get("encoder_execution_arm", "eager-graphed")
    decoder = env.overrides.get("decode_dispatch_arm", "dense-graphed")
    assert encoder in message and decoder in message, message
    rc = env.right_context
    assert CADENCES[LOOKAHEADS.index(rc)] in message or re.search(
        rf"lookahead\D{{0,10}}{rc}\b", message
    ), message
    assert re.search(rf"max_num_seqs\D{{0,10}}{env.max_num_seqs}\b", message), message
    assert re.search(r"(chunk|tier|count|bucket)", message, re.I), (
        f"the failing graph is not named: {message}"
    )
    for word in (*REMEDY_WORDS, *extra):
        assert word in message, (word, message)


@pytest.mark.parametrize("arm", GRAPHED)
# @spec REG-MEM-017, REG-MEM-004
def test_a_profiling_dummy_step_capture_failure_names_the_capture_and_remedies(
    build, arm
):
    env = build(arm=arm)
    env.graphs.fail_capture_after = 1
    error = h.call_with(env.runner.profile_run, core_embeds=False)
    assert error is not None
    _assert_failure_message(error, env)
    env.graphs.disarm()
    gc.collect()
    # The propagated error, its cause and context included, is still held
    # here with its tracebacks: it keeps no graph of the failed capture alive.
    assert error.__traceback__ is not None
    assert env.graphs.live_graphs() == 0, (
        "the retained error keeps a graph of the failed capture alive"
    )
    _release(error)


#: A lower ``--gpu-memory-utilization`` offered as a remedy.
_LOWER_UTILIZATION = re.compile(
    r"(?i)\b(lower|smaller)\b[^;.]{0,24}--gpu-memory-utilization"
)


@pytest.mark.parametrize(
    ("profile_only", "budget", "extra"),
    [
        (True, None, ()),
        (False, None, ("--gpu-memory-utilization",)),
        (False, 1 << 30, ("--kv-cache-memory-bytes",)),
    ],
    ids=["profiling", "serving-profiled-budget", "serving-fixed-budget"],
)
# @spec REG-MEM-017
def test_a_hook_capture_failure_discards_the_call_and_names_the_remedies(
    hook, profile_only, budget, extra
):
    env = hook(kv_cache_memory_bytes=budget)
    env.runner.profile_run(core_embeds=False)
    if not profile_only:
        if budget is None:
            _capture(env, profile_only=True)
            env.state.clear_cudagraphs()
        pool(env)
    env.graphs.fail_capture_after = len(env.graphs.captures) + 1
    env.graphs.captures_completed = len(env.graphs.captures)
    name = "profiling" if profile_only else "serving"
    error = h.call_with(
        _capture, env, profile_only=profile_only, stream=_stream(env, name)
    )
    assert error is not None
    _assert_failure_message(error, env, extra=extra)
    env.graphs.disarm()
    gc.collect()
    # The propagated error, its cause and context included, is still held
    # here with its tracebacks: it keeps no graph of the failed call alive.
    assert error.__traceback__ is not None
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0, (
        "the retained error keeps a graph of the failed call alive"
    )
    assert ("cuda.stream.synchronize", name) in env.fake.events, (
        "the capture stream was not synchronized"
    )
    _release(error)


# @spec REG-MEM-017
def test_a_serving_capture_failure_under_the_extensible_kv_cache_names_its_sizing(
    hook,
):
    """Under the extensible KV cache the serving capture precedes core's
    sizing of the KV cache, so no lower ``--gpu-memory-utilization`` is a
    remedy; the budget setting the error names is the extensible KV cache,
    as the sizing function that REG-MEM-023 and TXG-MEM-019 share names it.
    No profiling capture runs (core runs no estimate)."""
    env = hook()
    env.vllm_config.cache_config.enable_extensible_kv_cache = True
    env.runner.profile_run(core_embeds=False)
    pool(env)
    env.graphs.fail_capture_after = len(env.graphs.captures) + 1
    env.graphs.captures_completed = len(env.graphs.captures)
    error = h.call_with(
        _capture, env, profile_only=False, stream=_stream(env, "serving")
    )
    assert error is not None
    _assert_failure_message(error, env)
    message = str(error)
    assert re.search(r"(?i)extensible", message), message
    assert not _LOWER_UTILIZATION.search(message), message
    env.graphs.disarm()
    gc.collect()
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0
    _release(error)


# @spec REG-MEM-017
def test_a_serving_capture_failure_under_a_fixed_block_count_names_a_smaller_one(
    hook,
):
    """Under ``--num-gpu-blocks-override`` the remedies also name a smaller
    block count that still holds, besides the null block, a page for every
    request that can be active (TXG-CFG-005): at least the admission limit
    (``max_num_seqs`` here, transaction graphs off) plus one. The budget
    setting the error names is the fixed block count, so no lower
    ``--gpu-memory-utilization`` is a remedy."""
    from support.txg_api import SIZING_NAMES, Sizing

    env = hook(sizing=Sizing.BLOCK_COUNT, blocks=9)
    env.runner.profile_run(core_embeds=False)
    _capture(env, profile_only=True)
    env.state.clear_cudagraphs()
    pool(env)
    env.graphs.fail_capture_after = len(env.graphs.captures) + 1
    env.graphs.captures_completed = len(env.graphs.captures)
    error = h.call_with(
        _capture, env, profile_only=False, stream=_stream(env, "serving")
    )
    assert error is not None
    _assert_failure_message(error, env)
    message = str(error)
    assert re.search(SIZING_NAMES[Sizing.BLOCK_COUNT], message), message
    remedy = re.search(
        r"(?i)\bsmaller\b[^;]{0,40}--num-gpu-blocks-override[^;]*", message
    )
    assert remedy is not None, message
    assert re.search(rf"\b{env.max_num_seqs + 1}\b", remedy.group(0)), message
    assert re.search(r"(?i)null block", remedy.group(0)), message
    assert not _LOWER_UTILIZATION.search(message), message
    env.graphs.disarm()
    gc.collect()
    assert env.graphs.captured() == [] and env.graphs.live_graphs() == 0
    _release(error)


# ===========================================================================
# GPU: the hook configuration on CUDA
# ===========================================================================


@pytest.mark.gpu
# @spec REG-MEM-016, REG-MEM-020, REG-MEM-021, REG-MEM-022, REG-MEM-025
# @spec REG-MEM-026, REG-MEM-027
def test_hook_lifecycle_on_cuda(monkeypatch, tmp_path, single_process_dist):
    """CUDA, core's graph runtime, the card's default arms under the hook
    configuration: the profiling capture retains the segment graphs and the
    charged margin (``MARGIN_BYTES``), ``clear_cudagraphs`` returns them, the
    serving capture passes the aggregate comparison, a real step replays,
    ``post_warmup`` passes, ``shutdown`` releases the serving graphs."""
    card = h.write_runnable_card(tmp_path / "card")
    # The hook configuration from construction on, so the engine hook and
    # the model state's re-check see it.
    vllm_config = h.build_vllm_config(
        card, right_context=6, device="cuda", enforce_eager=False
    )
    vllm_config.cache_config.enable_extensible_kv_cache = False
    model = h.build_model(vllm_config)
    state = h.build_model_state(vllm_config, model)
    runner = h.Runner(vllm_config, model, state)
    core = CoreOrder(runner, device="cuda", spy=False)
    assert core.needs_capture()
    core.prepare()

    def free() -> int:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0]

    core.bind_profiling()
    free_before = free()
    core.profiling_capture()
    retained = free_before - free()
    assert retained >= MARGIN_BYTES, f"the profiling capture retained {retained} bytes"
    core.clear()
    assert free() >= free_before - (8 << 20), "the profiling capture was not released"
    core.bind_serving()
    free_before_serving = free()
    core.serving_capture()
    serving_retained = free_before_serving - free()
    # The serving call holds the same segment graphs without the margin
    # (allocator rounding may differ by less than the margin absorbs).
    assert retained - serving_retained >= MARGIN_BYTES - (16 << 20), (
        retained,
        serving_retained,
    )
    core.post_warmup()
    env = SimpleNamespace(
        runner=runner, config=model.config, max_geometry=3, model=model
    )
    runner.real_step(h.scheduler_output(new=(stream(env, "s1"),), handles={"s1": 1}))
    assert runner.commit_log and not runner.commit_log[-1][1]
    serving_free = free()
    core.shutdown()
    assert free() > serving_free, "shutdown released no graph memory"
