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

Readiness reads are logged by wrapping the inventories' readiness surfaces
(encoder ``ready``, decoder ``captured_keys``, CHUNK ``ready``).

Migration blockers are isolated per test (``legacy`` argument of ``build``):
tests of REG-MEM-* run with the removed state-slot calls re-added so that
REG-STATE-002..005 do not mask them; the REG-STATE contract tests do not.
"""

from __future__ import annotations

import ast
import inspect
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from support import model_harness as h

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
    "chunk": (
        {
            "decode_dispatch_arm": "dense-graphed",
            "encoder_execution_arm": "eager-graphed",
            "chunk_bucket_graph_populations": [1],
            "supported_num_lookahead_tokens": [1],
        },
        1,
        1,
    ),
}
GRAPHED = ("dense-graphed", "chunk")


def geometry_of(right_context: int) -> int:
    return LOOKAHEADS.index(right_context)


def _log_reads(monkeypatch, events, cls, name, label):
    original = getattr(cls, name)

    def read(self):
        events.append(("ready", label))
        return original.fget(self)

    monkeypatch.setattr(cls, name, property(read))


@pytest.fixture
def build(monkeypatch, tmp_path):
    """``build(arm=..., legacy=..., **config)`` → the runner and its spies.

    ``legacy``: ``"all"`` re-adds every removed state-slot call, ``"rows"``
    all but ``validate_current``, ``"handle"`` only ``get_handle``, ``None``
    nothing."""

    def make(arm="eager", legacy="all", overrides=None, **config):
        from vllm_nemotron_asr_streaming.nemotron_asr.chunk_bucket_graph import (
            ExactChunkGraphBinding,
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.decode_graph import (
            DenseGraphBinding,
        )
        from vllm_nemotron_asr_streaming.nemotron_asr.encoder_execution import (
            ResolvedEncoderExecution,
        )

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
        _log_reads(monkeypatch, events, ResolvedEncoderExecution, "ready", "encoder")
        _log_reads(monkeypatch, events, DenseGraphBinding, "captured_keys", "decoder")
        _log_reads(monkeypatch, events, ExactChunkGraphBinding, "ready", "chunk")
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
            chunk_cells=[
                (1, n) for n in merged.get("chunk_bucket_graph_populations", [])
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


def stream(env, req_id: str, *, geometry: int | None = None, sequence: int = 0):
    geometry = env.max_geometry if geometry is None else geometry
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=h.stream_envelope(env.config, geometry=geometry, sequence=sequence),
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


def recorded(state) -> tuple[bool, bool]:
    """The model state's record of the current step: ``(dummy_run,
    is_profile)`` (current attribute names ``_dummy_run`` / ``_is_profile``)."""
    return bool(state._dummy_run), bool(state._is_profile)


# @spec REG-MEM-001
def test_prepare_step_records_whether_the_step_is_a_profiling_dummy_step(build):
    from vllm.v1.core.sched.output import SchedulerOutput

    env = build()
    pool(env)
    state = env.state
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=True, is_profile=True)
    assert recorded(state) == (True, True)
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=True, is_profile=False)
    assert recorded(state) == (True, False), "the profiling record was not replaced"
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=True, is_profile=True)
    state.prepare_step(
        h.scheduler_output(finished=(), cleanup={}), dummy_run=False, is_profile=False
    )
    assert recorded(state) == (False, False), "a real step kept the profiling record"
    assert env.profile.calls == [] and env.graphs.captures == []


@pytest.mark.parametrize("arm", list(ARMS))
# @spec REG-MEM-002
def test_first_profiling_forward_prepares_in_order_at_maximum_shape(build, arm):
    env = build(arm=arm)
    retained = []
    env.profile.before = lambda kw: retained.append(
        (len(env.graphs.captured()), len(env.graphs.captures))
    )
    _, output, embeds = env.runner.profile_run(core_embeds=False)
    env.events.append(("returned", None))
    assert_dummy_output(output, embeds)
    kinds = env.events.kinds()
    assert max_shapes(env) <= profiled_shapes(env), env.profile.calls
    last_capture = max((i for i, k in enumerate(kinds) if k == "capture"), default=-1)
    readiness = [i for i, k in enumerate(kinds) if k == "ready" and i > last_capture]
    assert readiness, "no readiness check after warm-up and capture"
    final = [
        i
        for i, (k, info) in enumerate(env.events)
        if k == "profile"
        and (info.get("num_rows"), info.get("geometry_id")) in max_shapes(env)
        and i > readiness[0]
    ]
    assert final, "no maximum-shape profile transaction after the readiness check"
    assert final[-1] < kinds.index("returned")
    if arm in GRAPHED:
        assert env.graphs.captured(), "graphed arm captured nothing"
        assert all(have == tried for have, tried in retained[-len(final) :]), (
            "a captured graph was not retained during the profile transaction"
        )


@pytest.mark.parametrize("arm", GRAPHED)
# @spec REG-MEM-002
def test_first_profiling_forward_refuses_an_entry_left_without_a_graph(build, arm):
    """Each expected graph entry, in turn, is left without a graph in its
    wrapper during capture; preparation must notice and fail."""
    clean = build(arm=arm)
    clean.runner.profile_run(core_embeds=False)
    attempts = len(clean.graphs.captures)
    assert attempts, "the first profiling step captured nothing (no preparation)"
    for index in range(attempts):
        env = build(arm=arm)
        env.graphs.skip_capture = index
        error = h.call_with(env.runner.profile_run, core_embeds=False)
        assert error is not None, f"preparation accepted entry {index} without a graph"


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
        before = len(env.events)
        batch, output, embeds = env.runner.profile_run(
            randomize_inputs=True, attn=True, core_embeds=False
        )
        assert "ready" in h.Events(env.events[before:]).kinds(), (
            "a later profiling step did not verify readiness"
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
    if where == "warm-up":
        env.graphs.fail_eager = RuntimeError("warm-up failed")
    elif where == "capture":
        env.graphs.fail_capture = 1
    elif where == "readiness":
        env.graphs.skip_capture = 1
    elif where == "profile":
        env.profile.fail = RuntimeError("profile transaction failed")


@pytest.mark.parametrize("where", ["warm-up", "capture", "readiness", "profile"])
# @spec REG-MEM-004
def test_failed_preparation_discards_its_state_and_stays_failed(build, where):
    env = build(arm="dense-graphed")
    _inject(env, where)
    first = h.call_with(env.runner.profile_run, core_embeds=False)
    assert first is not None, f"preparation passed despite a {where} failure"
    attempted = len(env.graphs.captures) + len(env.profile.calls)
    assert attempted, "the first profiling step attempted no preparation"
    binding = env.model._decode_graph_binding
    assert binding.captured_keys == (), "partial execution state was kept"
    for attr in ("fail_eager", "fail_capture", "skip_capture"):
        setattr(env.graphs, attr, None)
    env.profile.fail = None
    for _ in range(2):
        later = h.call_with(env.runner.profile_run, core_embeds=False)
        assert later is not None, "a profiling step after a failure passed"
    assert len(env.graphs.captures) + len(env.profile.calls) == attempted, (
        "a failed worker prepared again"
    )


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
    assert max_shapes(env) <= profiled_shapes(env)
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
    env = build(legacy=None)
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
    """A complete step through the production commit path; with ``stale``,
    core begins another step after the rows were bound and before the row
    transaction runs, so the commit-time re-read is refused."""
    env.runner.profile_run(core_embeds=False)
    spy = pool(env)
    output = rows_of(env, kind)
    slots = [1, 2] if kind == "mixed" else [1 if kind == "stream" else 2]
    before = h.pool_tensor(spy).clone()
    reads = []
    if stale:

        def make_stale(kwargs):
            reads.append(len(spy.row_reads))
            h.core_bindings(spy).begin_step(h.scheduler_output(step_seq=99), None)

        env.advance.before = make_stale
    error = h.call_with(env.runner.real_step, output)
    return spy, before, slots, error, reads


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-004, REG-STATE-006
def test_commit_writes_the_bound_slots(build, kind):
    """Control: a live step commits state into exactly its bound slots."""
    env = build(legacy="rows")
    spy, before, slots, error, _ = _commit(env, kind, stale=False)
    assert error is None, repr(error)
    after = h.pool_tensor(spy)
    page = h.model_spec(env.model).page_size_bytes
    for slot in range(len(after) // page):
        changed = not torch.equal(
            before[slot * page : (slot + 1) * page],
            after[slot * page : (slot + 1) * page],
        )
        assert changed is (slot in slots), f"slot {slot}"


@pytest.mark.parametrize("kind", ["file", "stream", "mixed"])
# @spec REG-STATE-004, REG-STATE-006
def test_stale_commit_rereads_the_pools_and_writes_nothing(build, kind):
    env = build(legacy="rows")
    spy, before, _, error, reads = _commit(env, kind, stale=True)
    assert reads, "the row transaction did not run"
    assert "component_pools" in spy.row_reads[reads[0] :], (
        "the commit did not re-read component_pools"
    )
    assert isinstance(error, ValueError) and STALE in str(error), repr(error)
    assert torch.equal(before, h.pool_tensor(spy)), "a stale commit wrote state"


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


@pytest.mark.gpu
# @spec REG-MEM-002, REG-MEM-006
def test_serving_refuses_a_graphed_step_whose_wrapper_entry_lost_its_graph(
    monkeypatch, tmp_path
):
    """On CUDA with the dense-graphed decoder and core's own graph runtime:
    preparation captures every entry; a prepared serving step succeeds; then
    exactly the entry the next step needs loses its graph, and that step is
    refused before the row transaction runs and without capturing.

    The refusal is the model's own check, so the error must not be one of
    the generic failures an unguarded replay or lookup would raise."""
    h.spec_without_state_name(monkeypatch)
    h.legacy_state_api(monkeypatch)
    advance = h.AdvanceSpy().install(monkeypatch)
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
    binding = model._decode_graph_binding
    entries = list(binding._entries.values())
    assert entries and all(
        e.wrapper.concrete_cudagraph_entries[e.descriptor].cudagraph is not None
        for e in entries
    )
    runner.bind_pool(h.real_bindings(h.model_spec(model), device="cuda"))

    def unit(sequence):
        return h.new_request(
            "s1",
            int(model.config.audio_chunk_token_id),
            audio=h.stream_envelope(model.config, geometry=3, sequence=sequence),
        )

    runner.real_step(h.scheduler_output(new=(unit(0),), handles={"s1": 1}))
    executed = len(advance.calls)
    assert executed, "the prepared serving step did not run"
    target = next(e for e in entries if e.tier == 1 and e.geometry == 3)
    wrapper_entry = target.wrapper.concrete_cudagraph_entries[target.descriptor]
    wrapper_entry.cudagraph = None
    before = dict(target.wrapper.concrete_cudagraph_entries)
    error = h.call_with(
        runner.real_step,
        h.scheduler_output(new=(unit(1),), handles={"s1": 1}, step_seq=2),
    )
    assert error is not None, "the step ran without its graph"
    assert not isinstance(error, (AttributeError, TypeError, KeyError, IndexError)), (
        repr(error)
    )
    assert len(advance.calls) == executed, "the row transaction ran"
    assert wrapper_entry.cudagraph is None, "serving captured a graph"
    assert dict(target.wrapper.concrete_cudagraph_entries) == before
