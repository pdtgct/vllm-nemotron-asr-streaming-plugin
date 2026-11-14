# SPDX-License-Identifier: Apache-2.0
"""Execution memory, the model-state interface and state-slot access
(REG-MEM-*, REG-MS-*, REG-STATE-*).

The model and its model state are the real classes on CPU, driven in core's
order by ``support.model_harness.Runner`` (see its docstring for what is real,
what is omitted and which seams the doubles use): a profiling run, later
dummy steps (kernel warm-up, with attention metadata), and complete real
steps (``prepare_step`` through ``finish_step``) whose row transaction and
commit are the plugin's own.

Execution arms (``ARMS``): the card default (``eager``), ``compiled-static``,
a graphed decoder (``dense-graphed``) and CHUNK populations. CPU cannot
capture graphs, so graphed arms run with ``FakeGraphRuntime``: core's forward
context and batch descriptors, and a wrapper with core's entry contract whose
capture is logged and can be made to fail or to store no graph.

Expected graph inventories are derived from the configuration alone
(``expected_entries``), never from an observed run.

Model-internal seams these tests read (beyond the harness's): the model
state's step record (``_dummy_run`` / ``_is_profile``, REG-MEM-001, no public
accessor), the model's state-layer spec (``_persistent_state_layer``), the
profile scratch (``build_profile_invocation(...).storage.spec``), the encoder
inventory's ``ready`` property (REG-MEM-003 inventory test). The GPU test
reads no model attribute: it finds graph entries by instrumenting
core's ``CUDAGraphWrapper.__call__``.

Migration blockers are isolated per test (``legacy`` argument of ``build``):
tests of REG-MEM-* run with the removed state-slot calls re-added so that
REG-STATE-002..005 do not mask them; the REG-STATE contract tests do not.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import inspect
import re
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support import model_harness as h

from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import SAMPLED_TOKEN

pytestmark = pytest.mark.engine

LOOKAHEADS = (0, 1, 3, 6, 13)
#: Members of core's ``StateSlotBindings`` the model may use during a step:
#: ``bind_rows`` (REG-STATE-003) and ``fail_requests`` (core's channel for
#: model-reported row failures, used by ``postprocess_state``).
PERMITTED_BINDINGS = {"bind_rows", "fail_requests"}
PERMITTED_ROWS = {"component_pools", "slot_indices_cpu"}
STALE = "Stale state step access"

#: name → (configuration overrides, right context, max_num_seqs). Served sets
#: are narrowed where the arm's warm-up cost on CPU grows with them.
ARMS = {
    "eager": ({}, 6, 2),
    "compiled-static": (
        {
            "encoder_execution_arm": "compiled-static",
            "supported_num_lookahead_tokens": [0],
        },
        0,
        1,
    ),
    "dense-graphed": (
        {
            "decode_dispatch_arm": "dense-graphed",
            "supported_num_lookahead_tokens": [0, 6],
        },
        6,
        2,
    ),
    # CHUNK buckets accept populations 1, 31 and 63 only; with two
    # sequences the maximum shape (2 rows) differs from the CHUNK cell (1 row).
    "chunk": (
        {
            "decode_dispatch_arm": "dense-graphed",
            "encoder_execution_arm": "eager-graphed",
            "chunk_bucket_graph_populations": [1],
            "supported_num_lookahead_tokens": [1],
        },
        1,
        2,
    ),
}
GRAPHED = ("dense-graphed", "chunk")


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
    """Graph entries the configuration requires: each served geometry and
    batch tier of a graphed decoder; each served geometry and population of a
    graphed encoder, except the cells a CHUNK bucket owns; each CHUNK cell."""
    overrides = env.overrides
    count = 0
    if overrides.get("decode_dispatch_arm") == "dense-graphed":
        count += len(env.served_geometries) * len(batch_tiers(env.max_num_seqs))
    if overrides.get("encoder_execution_arm") in ("dense-graphed", "eager-graphed"):
        # Without population bucketing a graphed encoder has one entry per
        # population 1..max_num_seqs. On the CHUNK arm (max_num_seqs=2) those
        # populations coincide with the decoder's batch tiers (1, 2) only
        # because max_num_seqs is 2; the two terms are counted separately.
        cells = {
            (g, n)
            for g in env.served_geometries
            for n in range(1, env.max_num_seqs + 1)
        }
        count += len(cells - set(env.chunk_cells))
    return count + len(env.chunk_cells)


@pytest.fixture
def build(monkeypatch, tmp_path):
    """``build(arm=..., legacy=..., **config)`` → the runner and its spies.

    ``legacy``: ``"all"`` re-adds every removed state-slot call, ``"rows"``
    all but ``validate_current``, ``"handle"`` only ``get_handle``, ``None``
    nothing."""

    def make(arm="eager", legacy="all", overrides=None, **config):
        h.spec_without_state_name(monkeypatch)
        if legacy is not None:
            h.legacy_state_api(
                monkeypatch,
                rows_layout=legacy in ("all", "rows"),
                validate_current=legacy == "all",
            )
        events = h.Events()
        profile = h.ProfileSpy(events).install(monkeypatch)
        advance = h.AdvanceSpy(events).install(monkeypatch)
        graphs = h.FakeGraphRuntime(events).install(monkeypatch)
        arm_overrides, right_context, max_num_seqs = ARMS[arm]
        merged = dict(arm_overrides) | dict(overrides or {})
        settings = {"right_context": right_context, "max_num_seqs": max_num_seqs}
        settings.update(config)
        card = h.write_runnable_card(tmp_path / f"card-{len(list(tmp_path.iterdir()))}")
        vllm_config = h.build_vllm_config(card, overrides=merged, **settings)
        model = h.build_model(vllm_config)
        state = h.build_model_state(vllm_config, model)
        served = merged.get("supported_num_lookahead_tokens", list(LOOKAHEADS))
        return SimpleNamespace(
            arm=arm,
            vllm_config=vllm_config,
            model=model,
            state=state,
            config=model.config,
            runner=h.Runner(vllm_config, model, state),
            events=events,
            profile=profile,
            advance=advance,
            graphs=graphs,
            max_num_seqs=settings["max_num_seqs"],
            max_geometry=geometry_of(max(served)),
            served_geometries=sorted(geometry_of(r) for r in served),
            overrides=merged,
            chunk_cells=[
                (geometry_of(r), n)
                for r in served
                for n in merged.get("chunk_bucket_graph_populations", [])
            ],
        )

    return make


def pool(env, *, spy: bool = True, capacity: int = 8):
    bindings = h.real_bindings(h.model_spec(env.model), capacity=capacity)
    return env.runner.bind_pool(h.BindingsSpy(bindings) if spy else bindings)


def trapped_pool(env):
    trap = h.Trap(h.real_bindings(h.model_spec(env.model)), "state_slots")
    env.runner.bind_pool(trap)
    trap.armed = True
    return trap


def stream(
    env,
    req_id: str,
    *,
    geometry: int | None = None,
    sequence: int = 0,
    final: bool = True,
):
    geometry = env.max_geometry if geometry is None else geometry
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=h.stream_envelope(
            env.config, geometry=geometry, sequence=sequence, final=final
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
    """Every ``bind_rows`` call of the step (from index ``since``) names the
    model's layer, requests of this step in the step's order, and their
    scheduler-output handles."""
    order = [r.req_id for r in output.scheduled_new_reqs]
    order += list(output.scheduled_cached_reqs.req_ids)
    for layer, ids, handles, _ in spy.bind_calls[since:]:
        assert layer == h.STATE_LAYER
        assert list(ids) == [r for r in order if r in ids], (ids, order)
        assert handles == tuple(
            output.state_slot_handles[r][h.STATE_LAYER] for r in ids
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


COMBINATIONS = [(True, True), (True, False), (False, False), (False, True)]


def _prepare(env, dummy_run: bool, is_profile: bool):
    from vllm.v1.core.sched.output import SchedulerOutput

    output = SchedulerOutput.make_empty() if dummy_run else h.scheduler_output()
    return h.call_with(
        env.state.prepare_step, output, dummy_run=dummy_run, is_profile=is_profile
    )


@pytest.mark.parametrize("earlier", [None, (True, True)], ids=["fresh", "after"])
@pytest.mark.parametrize("dummy_run,is_profile", COMBINATIONS)
# @spec REG-MEM-001
def test_prepare_step_records_whether_the_step_is_a_profiling_dummy_step(
    build, earlier, dummy_run, is_profile
):
    """Every combination is recorded, and replaces an earlier profiling
    record; only ``dummy_run`` and ``is_profile`` both true is profiling."""
    env = build()
    pool(env)
    if earlier is not None:
        assert _prepare(env, *earlier) is None
        assert is_profiling_record(env.state)
    error = _prepare(env, dummy_run, is_profile)
    assert error is None, repr(error)
    assert is_profiling_record(env.state) is (dummy_run and is_profile)
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
def test_failed_preparation_on_the_card_default_stays_failed(build):
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
    pool_bytes = trap._target.views[h.STATE_LAYER]._tensor.clone()
    env.runner.profile_run(core_embeds=False)
    finals = [c for c in env.profile.calls if is_final(c)]
    assert max_shapes(env) <= profiled_shapes(env, finals)
    assert trap.reads == []
    assert torch.equal(pool_bytes, trap._target.views[h.STATE_LAYER]._tensor)


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
    """E15: core calls ``prepare_inputs`` on dummy steps after ``prepare_step``
    returned early; the previous real step's scheduler output is stale."""
    from vllm.v1.core.sched.output import SchedulerOutput

    env = build()
    env.runner.profile_run(core_embeds=False)
    real = h.Trap(rows_of(env, "mixed"), "scheduler_output")
    env.state.prepare_step(real, dummy_run=False, is_profile=False)
    real.armed = True
    trap = trapped_pool(env) if pool_exists else None
    env.state.prepare_step(
        SchedulerOutput.make_empty(), dummy_run=True, is_profile=is_profile
    )
    batch = h.dummy_batch(env.state, 2, 2)
    kwargs = env.state.prepare_inputs(batch, env.runner.req_states)
    assert "'s1'" not in repr(kwargs) and "'f1'" not in repr(kwargs)
    if trap is not None:
        assert trap.reads == []


@pytest.mark.parametrize("native_burst", [False, True], ids=["default", "native"])
# @spec REG-MEM-013, REG-MEM-010
def test_dummy_sampling_reads_no_state_slots(build, native_burst):
    """E13: core indexes the profiling output by ``logits_indices`` and runs
    ``_dummy_sampler_run`` (the model state's ``dummy_sampler_run``, then two
    sampler passes with ``add_request`` / ``apply_staged_writes`` between)
    through the sampler the model state installs."""
    overrides = {"experimental_native_burst": True} if native_burst else {}
    env = build(right_context=1 if native_burst else 6, overrides=overrides)
    batch, output, embeds = env.runner.profile_run(core_embeds=False)
    assert_dummy_output(output, embeds)
    trap = trapped_pool(env)
    base = h.RecordingSampler()
    hooks = env.state.custom_sampler(base)
    assert (hooks is not None) is native_burst
    sampler = hooks[0] if hooks is not None else base
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


def drain_to_park(env, ids, handles, *, watch: str = "s1", limit: int = 300):
    """Cached steps, each fed the token the model emitted on the previous one,
    until the ``watch`` row's decision is the park token (a legal park); returns
    the next ``num_computed_tokens``."""
    eos = int(env.config.eos_token_id)
    position = 1
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
            new=(stream(env, "s1", final=False), file(env, "f1", 60000)),
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
    new = h.scheduler_output(new=(prompted("r1", [first, second]),), encoder_inputs={})
    env.state.prepare_step(new, dummy_run=False, is_profile=False)
    assert env.state._scheduled_tokens == {"r1": (first,)}
    cached = cached_step(env, ["r1"], {"r1": 1}, handles={"r1": 1})
    env.state.prepare_step(cached, dummy_run=False, is_profile=False)
    assert env.state._scheduled_tokens == {"r1": (second,)}


# @spec REG-STATE-008
def test_cached_window_crossing_the_end_of_the_prompt_is_refused(build):
    env = build()
    new = h.scheduler_output(new=(prompted("r1", [7, 9]),), encoder_inputs={})
    env.state.prepare_step(new, dummy_run=False, is_profile=False)
    cached = cached_step(env, ["r1"], {"r1": 1}, handles={"r1": 1})
    cached.num_scheduled_tokens["r1"] = 2
    cached.total_num_scheduled_tokens = 2
    error = h.call_with(
        env.state.prepare_step, cached, dummy_run=False, is_profile=False
    )
    assert isinstance(error, ValueError) and "r1" in str(error), error


# @spec REG-STATE-008
def test_cached_request_without_a_recorded_prompt_is_refused(build):
    env = build()
    out = cached_step(env, ["ghost"], {"ghost": 1}, handles={"ghost": 1})
    error = h.call_with(env.state.prepare_step, out, dummy_run=False, is_profile=False)
    assert isinstance(error, ValueError) and "ghost" in str(error)


@pytest.mark.parametrize("prepared", [False, True], ids=["before", "after"])
# @spec REG-MEM-014
def test_cleanup_only_real_step_completes(build, prepared):
    env = build()
    if prepared:
        env.runner.profile_run(core_embeds=False)
    profiles = len(env.profile.calls)
    spy = pool(env)
    # A request that held slot 3 in an earlier step and has now finished.
    h.core_bindings(spy).views[h.STATE_LAYER].bind("gone", h.handle(3))
    output = h.scheduler_output(finished=("gone",), cleanup={"gone": 3})
    assert output.total_num_scheduled_tokens == 0
    _, completion = env.runner.real_step(output)
    assert completion is not None
    assert len(env.profile.calls) == profiles and env.advance.calls == []


# ===========================================================================
# State-slot access
# ===========================================================================


@pytest.fixture
def served_hf_config(tmp_path):
    from vllm.config import ModelConfig

    from vllm_nemotron_asr_streaming import register

    register()
    card = h.write_runnable_card(tmp_path / "card")
    return ModelConfig(
        model=str(card),
        config_format="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": 6},
        enforce_eager=True,
        dtype="float32",
    ).hf_config


# @spec REG-STATE-001
def test_state_spec_is_layout_only(monkeypatch, served_hf_config):
    """No shim here: the builder runs against the target core's class."""
    from vllm.v1.kv_cache_interface import PersistentStateSpec

    from vllm_nemotron_asr_streaming.nemotron_asr import state_profile

    seen: list[tuple[tuple, dict]] = []

    def spy(*args, **kwargs):
        seen.append((args, dict(kwargs)))
        return PersistentStateSpec(*args, **kwargs)

    monkeypatch.setattr(state_profile, "PersistentStateSpec", spy)
    spec = state_profile.build_nemotron_persistent_state_spec(served_hf_config)
    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args == ()
    assert set(kwargs) == {"descriptors", "page_size_bytes", "block_size"}
    assert kwargs["block_size"] == 1
    assert isinstance(spec, PersistentStateSpec)


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-002, REG-STATE-007
def test_handles_come_only_from_the_scheduler_output(build, kind):
    """Over a complete step: ``prepare_step`` reads no bindings member, and
    no handle is looked up from the bindings at any point."""
    env = build(legacy=None)
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    h.call_with(env.runner.real_step, rows_of(env, kind))
    assert [r for r in spy.reads if r[1] == "prepare_step"] == []
    assert [c for c in spy.bind_calls if c[3] == "prepare_step"] == []
    assert "get_handle" not in [r[0] for r in spy.reads], spy.reads


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-003
def test_rows_are_bound_with_the_steps_order_and_scheduler_handles(build, kind):
    """Contract test, no shim: over a complete step the bindings are used only
    through ``bind_rows`` (and ``fail_requests``), rows only through
    ``component_pools`` and ``slot_indices_cpu``."""
    # three requests in the mixed case: three sequences
    env = build(legacy=None, max_num_seqs=3)
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
    assert {r[0] for r in spy.reads} <= PERMITTED_BINDINGS, spy.reads
    assert spy.bind_calls
    assert_bindings_valid(spy, output)
    bound = {r for call in spy.bind_calls for r in call[1]}
    assert bound == set(output.num_scheduled_tokens)
    assert set(spy.row_reads) <= PERMITTED_ROWS, spy.row_reads


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-005
def test_pool_projection_uses_only_component_pools_and_the_models_spec(build, kind):
    """``get_handle`` re-added (REG-STATE-002 isolated); nothing else."""
    env = build(legacy="handle")
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    h.call_with(env.runner.real_step, rows_of(env, kind))
    assert spy.row_reads, "no rows were read"
    assert set(spy.row_reads) <= PERMITTED_ROWS, spy.row_reads


def _commit(env, kind: str, *, stale: bool):
    """A complete step through the production commit path. The pool is
    snapshotted inside the row transaction's entry (after core's
    ``begin_step`` bound the slots, which zeroes and initializes new pages,
    and before the transaction runs). With ``stale``, core then begins
    another step, so the commit-time re-read is refused."""
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    output = rows_of(env, kind)
    slots = [1, 2] if kind == "mixed" else [1 if kind == "stream" else 2]
    seen: dict = {}

    def before(kwargs):
        seen["pool"] = h.pool_tensor(spy).clone()
        seen["reads"] = len(spy.row_reads)
        if stale:
            h.core_bindings(spy).begin_step(h.scheduler_output(step_seq=99), None)

    env.advance.before = before
    error = h.call_with(env.runner.real_step, output)
    return spy, seen, slots, error


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-004, REG-STATE-006
def test_commit_writes_the_bound_slots(build, kind):
    """Control: a live step commits state into each bound slot and no other."""
    env = build(legacy="rows")
    spy, seen, slots, error = _commit(env, kind, stale=False)
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


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-004, REG-STATE-006
def test_stale_commit_rereads_the_pools_and_writes_nothing(build, kind):
    env = build(legacy="rows")
    spy, seen, _, error = _commit(env, kind, stale=True)
    assert "pool" in seen, "the row transaction did not run"
    assert "component_pools" in spy.row_reads[seen["reads"] :], (
        "the commit did not re-read component_pools"
    )
    assert isinstance(error, ValueError) and STALE in str(error), repr(error)
    assert torch.equal(seen["pool"], h.pool_tensor(spy)), "a stale commit wrote state"


# @spec REG-STATE-006
def test_file_rows_on_recycled_slots_and_mixed_batches(build):
    """E7: a file whose slot is recycled to a new file (higher generation),
    sharing its step with a stream, binds and commits like stream rows."""
    env = build(legacy="rows")
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    first = h.scheduler_output(new=(file(env, "f1"),), handles={"f1": (1, 1)})
    env.runner.real_step(first)
    after_first = h.slot_bytes(spy, 1).clone()
    recycled = h.scheduler_output(
        new=(file(env, "f2", 6000), stream(env, "s1")),
        finished=("f1",),
        handles={"f2": (1, 2), "s1": 2},
        step_seq=2,
    )
    since = len(spy.bind_calls)
    env.runner.real_step(recycled)
    assert_bindings_valid(spy, recycled, since)
    bound = {r for call in spy.bind_calls[since:] for r in call[1]}
    assert bound == {"f2", "s1"}
    assert set(spy.row_reads) <= PERMITTED_ROWS
    assert not torch.equal(after_first, h.slot_bytes(spy, 1))


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


# @spec REG-MEM-006
def test_serving_refuses_a_graphed_step_whose_wrapper_entry_lost_its_graph_cpu(build):
    """CPU, ``FakeGraphRuntime``: a good real step replays graphs; one entry
    that step replayed loses its graph; the identical step is refused with no
    capture, no wrapper invocation and no state write; with the graph restored
    the identical step succeeds again (which attributes the refusal to the
    missing graph). No exception type or message is pinned, and the check may
    sit inside the row transaction."""
    env = build(arm="dense-graphed")
    _prepare_graphed(env)
    spy = pool(env)

    def step():
        # the session's second unit, delivered once the first drained to park
        return h.scheduler_output(
            new=(stream(env, "s1", sequence=1),), handles={"s1": 1}
        )

    replays = len(env.graphs.replays)
    env.runner.real_step(
        h.scheduler_output(new=(stream(env, "s1", final=False),), handles={"s1": 1})
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
    assert all(not failed for _, failed in env.runner.commit_log)
    assert all(v == 0 for st, _ in env.runner.commit_log for v in st.values())


@pytest.mark.gpu
# @spec REG-MEM-002, REG-MEM-006
def test_serving_refuses_a_graphed_step_whose_wrapper_entry_lost_its_graph(
    monkeypatch, tmp_path, single_process_dist
):
    """CUDA, core's own graph runtime and static graph wrapper: as the CPU
    test. Wrapper invocations are counted by instrumenting core's
    ``CUDAGraphWrapper.__call__`` (no model-internal attribute is read)."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import get_forward_context

    h.spec_without_state_name(monkeypatch)
    h.legacy_state_api(monkeypatch)
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
    vllm_config = h.build_vllm_config(
        card,
        overrides={
            "decode_dispatch_arm": "dense-graphed",
            "supported_num_lookahead_tokens": [6],
        },
        device="cuda",
    )
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
    runner.real_step(
        h.scheduler_output(new=(stream(env, "s1", final=False),), handles={"s1": 1})
    )
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
