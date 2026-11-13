# SPDX-License-Identifier: Apache-2.0
"""Execution memory, the model-state interface and state-slot access
(REG-MEM-*, REG-MS-*, REG-STATE-*).

The model and its model state are the real classes on CPU, driven in the
order core's model runner uses (``support.model_harness``): a profiling run
(``profile_run`` → ``_dummy_run(is_profile=True)``), later dummy steps (kernel
warm-up), and real steps (``execute_model``: ``prepare_step``, core's
``begin_step`` / ``activate_requests``, ``add_request``, ``prepare_inputs``,
the forward). The pool is core's ``StateSlotBindings`` over CPU memory.

The profile transaction and the row transaction cannot run the synthetic
card's encoder on CPU; the profile transaction is recorded
(``ProfileRecorder``) and the row transaction stubbed (``stub_advance``), so
these tests check which preparation and state access the model performs,
not its numerics.
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
READINESS = re.compile(r"(?i)prepar|ready")


def geometry_of(right_context: int) -> int:
    return LOOKAHEADS.index(right_context)


@pytest.fixture
def build(monkeypatch, card_dir):
    """Factory: ``build(**config)`` → namespace with the real model, model
    state, configuration and the profile recorder."""

    def make(*, install_recorder: bool = True, **config):
        h.spec_without_state_name(monkeypatch)
        recorder = h.ProfileRecorder()
        if install_recorder:
            recorder.install(monkeypatch)
        vllm_config = h.build_vllm_config(card_dir, **config)
        model = h.build_model(vllm_config)
        state = h.build_model_state(vllm_config, model)
        return SimpleNamespace(
            vllm_config=vllm_config,
            model=model,
            state=state,
            config=model.config,
            recorder=recorder,
        )

    return make


def pool(env, *, capacity: int = 8, spy: bool = True):
    """Core's bindings over a CPU pool, bound to the model state as
    ``initialize_state_slots`` binds them (after the profiling run)."""
    bindings = h.real_bindings(h.model_spec(env.model), capacity=capacity)
    if spy:
        bindings = h.BindingsSpy(bindings)
    env.state.initialize_state_slots(bindings)
    if spy:
        bindings.reads.clear()
    return bindings


def trapped_pool(env):
    """Bindings that fail on any access once armed (after binding)."""
    trap = h.Trap(h.real_bindings(h.model_spec(env.model)), "state_slots")
    env.state.initialize_state_slots(trap)
    trap.reads.clear()
    trap.armed = True
    return trap


def stream(env, req_id: str, *, geometry: int = 3):
    return h.new_request(
        req_id,
        int(env.config.audio_chunk_token_id),
        audio=h.stream_envelope(env.config, geometry=geometry),
    )


def file(env, req_id: str, n_samples: int = 4000, *, geometry: int = 3):
    samples = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
    return h.new_request(
        req_id,
        int(env.config.file_carrier_token_id),
        samples=samples,
        header=[geometry, 0, n_samples],
    )


def assert_dummy_output(output, embeds):
    """REG-MEM-010: the input embeddings' token count, width, dtype and
    device, with a valid decision ID (zero) in the carrier column."""
    from vllm_nemotron_asr_streaming.nemotron_asr.rnnt import read_decision_carrier

    assert isinstance(output, torch.Tensor)
    assert output.shape == embeds.shape
    assert output.dtype == embeds.dtype
    assert output.device == embeds.device
    assert torch.equal(
        read_decision_carrier(output), torch.zeros_like(read_decision_carrier(output))
    )


# ===========================================================================
# Model-state interface
# ===========================================================================


def _calls_accepted_by(signature: inspect.Signature):
    """The calls a signature accepts, as ``(args, kwargs)`` of sentinels:
    required arguments only, and every argument (positionals positionally,
    keyword-only by name)."""
    required_args, all_args, required_kw, all_kw = [], [], {}, {}
    for name, param in list(signature.parameters.items())[1:]:  # drop self
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.kind is param.KEYWORD_ONLY:
            all_kw[name] = object()
            if param.default is param.empty:
                required_kw[name] = object()
            continue
        all_args.append(object())
        if param.default is param.empty:
            required_args.append(object())
    return [(required_args, required_kw), (all_args, all_kw)]


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
        for pname, param in original.parameters.items():
            mine = override.parameters.get(pname)
            if mine is not None and mine.default != param.default:
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
def test_dummy_inputs_embeds_is_called_as_core_calls_it_on_a_profiling_run(build):
    """The profiling path reaches the forward through
    ``dummy_inputs_embeds(num_tokens, input_ids)`` (positionally)."""
    env = build()
    _, output, embeds = h.profile_run(env.state, env.model, randomize_inputs=True)
    assert embeds.data_ptr() == env.state.encoder_runner.inputs_embeds.data_ptr()
    assert output.shape == embeds.shape


# ===========================================================================
# Execution memory: profiling and dummy steps
# ===========================================================================


# @spec REG-MEM-001
def test_profiling_prepare_step_records_the_step_and_prepares_nothing(build):
    from vllm.v1.core.sched.output import SchedulerOutput

    env = build()
    env.state.prepare_step(
        SchedulerOutput.make_empty(), dummy_run=True, is_profile=True
    )
    env.state.prepare_step(
        SchedulerOutput.make_empty(), dummy_run=True, is_profile=True
    )
    assert env.recorder.calls == []


# @spec REG-MEM-001
def test_a_later_non_profiling_prepare_step_replaces_the_profiling_record(build):
    """A profiling ``prepare_step`` followed by a non-profiling dummy one: the
    forward of the second step does not prepare."""
    from vllm.v1.core.sched.output import SchedulerOutput

    env = build()
    env.state.prepare_step(
        SchedulerOutput.make_empty(), dummy_run=True, is_profile=True
    )
    _, output, embeds = h.dummy_step(
        env.state, env.model, is_profile=False, core_embeds=False
    )
    assert env.recorder.calls == []
    assert_dummy_output(output, embeds)


@pytest.mark.parametrize(
    "served,default",
    [((0, 1, 3, 6, 13), 6), ((0, 3), 3)],
    ids=["all-five", "narrowed"],
)
# @spec REG-MEM-002
def test_first_profiling_forward_prepares_and_runs_the_profile_at_maximum_shape(
    build, served, default
):
    env = build(
        right_context=default,
        overrides={"supported_num_lookahead_tokens": list(served)},
    )
    _, output, embeds = h.profile_run(env.state, env.model, core_embeds=False)
    assert len(env.recorder.calls) == 1, env.recorder.calls
    call = env.recorder.calls[0]
    assert call.get("num_rows") == env.vllm_config.scheduler_config.max_num_seqs
    assert call.get("geometry_id") == geometry_of(max(served))
    assert_dummy_output(output, embeds)


# @spec REG-MEM-002, REG-MEM-011
def test_first_profiling_forward_leaves_the_worker_ready_to_serve(build, monkeypatch):
    env = build()
    advanced = h.stub_advance(monkeypatch)
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    output = h.scheduler_output(
        new=(stream(env, "s1"),), handles={"s1": 1}, encoder_inputs={"s1": [0]}
    )
    h.real_step(env.state, env.model, bindings, output)
    assert len(advanced) == 1


# @spec REG-MEM-003
def test_later_profiling_steps_after_the_pool_exists_check_and_return(build):
    """E5: kernel warm-up may run further profiling dummy steps after the pool
    exists, with random token IDs, several times."""
    env = build()
    h.profile_run(env.state, env.model, core_embeds=False)
    assert len(env.recorder.calls) == 1
    trap = trapped_pool(env)
    for _ in range(3):
        batch, output, embeds = h.profile_run(
            env.state, env.model, randomize_inputs=True, core_embeds=False
        )
        assert isinstance(batch.input_ids, torch.Tensor)
        assert_dummy_output(output, embeds)
    assert len(env.recorder.calls) == 1, "a later profiling step profiled again"
    assert trap.reads == []


# @spec REG-MEM-003
def test_later_profiling_step_verifies_readiness(build, monkeypatch):
    """A later profiling step checks the inventory: if it is no longer ready,
    the step raises rather than returning a dummy output."""
    env = build()
    h.profile_run(env.state, env.model, core_embeds=False)
    execution = env.model._encoder_execution
    monkeypatch.setattr(type(execution), "ready", property(lambda self: False))
    error = h.call_with(h.profile_run, env.state, env.model, core_embeds=False)
    assert error is not None, "a later profiling step skipped the readiness check"
    assert len(env.recorder.calls) == 1


# @spec REG-MEM-004
def test_failed_preparation_raises_and_later_profiling_steps_raise_without_preparing(
    build, monkeypatch
):
    env = build()

    def failing(model, **kwargs):
        env.recorder.calls.append(kwargs)
        raise RuntimeError("profile transaction failed")

    from vllm_nemotron_asr_streaming.nemotron_asr import profile_execution

    monkeypatch.setattr(profile_execution, "run_persistent_state_profile", failing)
    first = h.call_with(h.profile_run, env.state, env.model, core_embeds=False)
    assert first is not None
    attempts = len(env.recorder.calls)
    assert attempts >= 1, "the first profiling step did not attempt preparation"
    second = h.call_with(h.profile_run, env.state, env.model, core_embeds=False)
    assert second is not None, "a profiling step after a failed preparation passed"
    assert len(env.recorder.calls) == attempts, "a failed worker prepared again"


# @spec REG-MEM-004, REG-MEM-011
def test_failed_preparation_refuses_real_steps(build, monkeypatch):
    env = build()
    advanced = h.stub_advance(monkeypatch)
    from vllm_nemotron_asr_streaming.nemotron_asr import profile_execution

    def failing(model, **kwargs):
        raise RuntimeError("profile transaction failed")

    monkeypatch.setattr(profile_execution, "run_persistent_state_profile", failing)
    h.call_with(h.profile_run, env.state, env.model, core_embeds=False)
    bindings = pool(env)
    output = h.scheduler_output(
        new=(stream(env, "s1"),), handles={"s1": 1}, encoder_inputs={"s1": [0]}
    )
    error = h.call_with(h.real_step, env.state, env.model, bindings, output)
    assert error is not None and READINESS.search(str(error)), error
    assert advanced == []


# @spec REG-MEM-005
def test_preparation_reads_no_state_slot_bindings(build):
    """Core binds state slots after its profiling run; even when bindings
    exist, preparation does not read them."""
    env = build()
    trap = trapped_pool(env)
    h.profile_run(env.state, env.model, core_embeds=False)
    assert len(env.recorder.calls) == 1
    assert trap.reads == []


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
    """A profiling step with fewer dummy rows than ``max_num_seqs``: the only
    profile is preparation's maximum-shape transaction, never one sized to
    the dummy batch."""
    env = build()
    h.profile_run(env.state, env.model, num_tokens=2, core_embeds=False)
    max_rows = env.vllm_config.scheduler_config.max_num_seqs
    assert [c.get("num_rows") for c in env.recorder.calls] == [max_rows]


@pytest.mark.parametrize("prepared", [False, True], ids=["before", "after"])
# @spec REG-MEM-009, REG-MEM-010
def test_non_profiling_dummy_step_returns_the_dummy_output_only(build, prepared):
    env = build()
    if prepared:
        h.profile_run(env.state, env.model, core_embeds=False)
        trap = trapped_pool(env)
    calls = len(env.recorder.calls)
    _, output, embeds = h.dummy_step(
        env.state, env.model, is_profile=False, randomize_inputs=True, core_embeds=False
    )
    assert len(env.recorder.calls) == calls
    assert_dummy_output(output, embeds)
    if prepared:
        assert trap.reads == []


@pytest.mark.parametrize("step", ["first-profiling", "later-profiling", "dummy"])
# @spec REG-MEM-010
def test_dummy_output_matches_the_input_embeddings(build, step):
    env = build()
    if step != "first-profiling":
        h.profile_run(env.state, env.model, core_embeds=False)
    _, output, embeds = h.dummy_step(
        env.state, env.model, is_profile=step != "dummy", core_embeds=False
    )
    assert_dummy_output(output, embeds)


@pytest.mark.parametrize("pool_exists", [False, True], ids=["no-pool", "pool"])
@pytest.mark.parametrize("is_profile", [False, True], ids=["dummy", "profiling"])
# @spec REG-MEM-012
def test_dummy_prepare_inputs_after_a_real_step_reads_nothing_retained(
    build, monkeypatch, pool_exists, is_profile
):
    """E15: core calls ``prepare_inputs`` on dummy steps after ``prepare_step``
    returned early; the previous real step's scheduler output is stale."""
    env = build()
    h.profile_run(env.state, env.model, core_embeds=False)
    real = h.Trap(
        h.scheduler_output(
            new=(stream(env, "s1"), file(env, "f1")),
            handles={"s1": 1, "f1": 2},
            encoder_inputs={"s1": [0]},
        ),
        "scheduler_output",
    )
    env.state.prepare_step(real, dummy_run=False, is_profile=False)
    real.armed = True
    trap = trapped_pool(env) if pool_exists else None
    from vllm.v1.core.sched.output import SchedulerOutput

    env.state.prepare_step(
        SchedulerOutput.make_empty(), dummy_run=True, is_profile=is_profile
    )
    batch = h.dummy_batch(env.state, 4, 4)
    kwargs = env.state.prepare_inputs(batch, h.req_states(list(batch.req_ids)))
    assert "'s1'" not in repr(kwargs) and "'f1'" not in repr(kwargs)
    if trap is not None:
        assert trap.reads == []


# @spec REG-MEM-013, REG-MEM-010
def test_native_burst_dummy_sampling_reads_no_state_slots(build):
    """E13: core indexes the profiling output by ``logits_indices``, computes
    logits and samples them through the model's sampler hook."""
    env = build(right_context=1, overrides={"experimental_native_burst": True})
    batch, output, embeds = h.profile_run(env.state, env.model, core_embeds=False)
    assert_dummy_output(output, embeds)
    trap = trapped_pool(env)
    base = h.Trap(SimpleNamespace(), "base_sampler")
    base.armed = True
    hooks = env.state.custom_sampler(base)
    assert hooks is not None, "native burst installs no sampler wrapper"
    sampler, _ = hooks
    hidden = output[batch.logits_indices]
    logits = env.model.compute_logits(hidden)
    for _ in range(2):  # core samples twice after warm-up
        sampled = sampler(
            logits, h.dummy_batch(env.state, hidden.shape[0], hidden.shape[0])
        )
        assert sampled.sampled_token_ids.shape[0] == hidden.shape[0]
    assert trap.reads == []


# @spec REG-MEM-013, REG-MEM-010
def test_profiling_chain_computes_logits_without_state(build):
    """E13 without native burst: the dummy output's logits select the valid
    decision ID in every row, reading no state slots."""
    env = build()
    trap = trapped_pool(env)
    batch, output, embeds = h.profile_run(env.state, env.model, core_embeds=False)
    hidden = output[batch.logits_indices]
    logits = env.model.compute_logits(hidden)
    assert logits.shape[0] == hidden.shape[0]
    assert torch.all(logits.argmax(dim=-1) == 0)
    assert env.state.custom_sampler(object()) is None
    assert trap.reads == []


# ===========================================================================
# Execution memory: real steps
# ===========================================================================


def _first_real_step(env, kind: str):
    rows, handles, encoder_inputs = [], {}, {}
    if kind in ("stream", "both"):
        rows.append(stream(env, "s1"))
        handles["s1"] = 1
        encoder_inputs["s1"] = [0]
    if kind in ("file", "both"):
        rows.append(file(env, "f1"))
        handles["f1"] = 2
    return h.scheduler_output(
        new=tuple(rows), handles=handles, encoder_inputs=encoder_inputs
    )


@pytest.mark.parametrize("kind", ["file", "stream", "both"])
# @spec REG-MEM-011
def test_real_step_before_preparation_is_refused_without_preparing(
    build, monkeypatch, kind
):
    """E14: no real step that runs model execution is served before
    preparation succeeded, whatever its rows."""
    env = build()
    advanced = h.stub_advance(monkeypatch)
    bindings = pool(env)
    error = h.call_with(
        h.real_step, env.state, env.model, bindings, _first_real_step(env, kind)
    )
    assert error is not None and READINESS.search(str(error)), error
    assert env.recorder.calls == [], "a real step prepared lazily"
    assert advanced == []


@pytest.mark.parametrize("kind", ["file", "stream", "both"])
# @spec REG-MEM-011, REG-STATE-003, REG-STATE-006
def test_first_real_step_after_preparation_runs(build, monkeypatch, kind):
    env = build()
    advanced = h.stub_advance(monkeypatch)
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    output = _first_real_step(env, kind)
    h.real_step(env.state, env.model, bindings, output)
    assert len(advanced) == 1
    assert len(env.recorder.calls) == 1, "a real step prepared again"
    assert [c[0] for c in bindings.bind_calls] == [h.STATE_LAYER]


@pytest.mark.parametrize("prepared", [False, True], ids=["before", "after"])
# @spec REG-MEM-014
def test_cleanup_only_real_step_completes(build, monkeypatch, prepared):
    env = build()
    h.stub_advance(monkeypatch)
    if prepared:
        h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    # A request that held slot 3 in an earlier step and has now finished.
    bindings._bindings.views[h.STATE_LAYER].bind("gone", h.handle(3))
    output = h.scheduler_output(finished=("gone",), cleanup={"gone": 3})
    assert output.total_num_scheduled_tokens == 0
    h.real_step(env.state, env.model, bindings, output)
    assert len(env.recorder.calls) == (1 if prepared else 0)


# ===========================================================================
# State-slot access
# ===========================================================================


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


@pytest.fixture
def served_hf_config(card_dir):
    from vllm.config import ModelConfig

    from vllm_nemotron_asr_streaming import register

    register()
    return ModelConfig(
        model=str(card_dir),
        config_format="nemotron_asr_streaming",
        hf_overrides={"num_lookahead_tokens": 6},
        enforce_eager=True,
        dtype="float32",
    ).hf_config


@pytest.mark.parametrize("kind", ["file", "stream", "both"])
# @spec REG-STATE-002, REG-STATE-007
def test_prepare_step_reads_only_the_scheduler_output(build, kind):
    env = build()
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    env.state.prepare_step(
        _first_real_step(env, kind), dummy_run=False, is_profile=False
    )
    assert bindings.reads == []
    assert bindings.bind_calls == []


# @spec REG-STATE-003, REG-STATE-005
def test_rows_are_bound_with_the_steps_order_and_scheduler_handles(build, monkeypatch):
    env = build()
    h.stub_advance(monkeypatch)
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    output = h.scheduler_output(
        new=(stream(env, "s1"), file(env, "f1"), stream(env, "s2")),
        handles={"s1": 3, "f1": 1, "s2": 2},
        encoder_inputs={"s1": [0], "s2": [0]},
    )
    h.real_step(env.state, env.model, bindings, output)
    assert bindings.reads == [], "rows were read outside bind_rows"
    assert len(bindings.bind_calls) == 1
    layer, ids, handles, _phase = bindings.bind_calls[0]
    assert layer == h.STATE_LAYER
    assert ids == ("s1", "f1", "s2")
    assert handles == tuple(output.state_slot_handles[r][h.STATE_LAYER] for r in ids)
    assert set(bindings.row_reads) <= {"component_pools", "slot_indices_cpu"}, (
        bindings.row_reads
    )


# @spec REG-STATE-004
def test_commit_rereads_the_pools_and_writes_nothing_when_that_fails(
    build, monkeypatch
):
    """The row transaction's commit sink re-reads ``component_pools`` before
    it writes; here the step is made stale first (core began another step),
    so the re-read raises and no state may be written."""
    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    env = build()
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    core = bindings._bindings
    tensor = core.views[h.STATE_LAYER]._tensor
    outcome: dict = {}

    def advance_model_rows(core_model, input_ids, inputs_embeds, plan, **kwargs):
        before = tensor.clone()
        reads = len(bindings.row_reads)
        core.begin_step(h.scheduler_output(step_seq=99), None)
        sink = kwargs["commit_sink"]
        outcome["error"] = h.call_with(sink.reserve, plan)
        outcome["reread"] = bindings.row_reads[reads:].count("component_pools")
        outcome["unchanged"] = torch.equal(before, tensor)
        return torch.zeros_like(inputs_embeds)

    monkeypatch.setattr(advance, "advance_model_rows", advance_model_rows)
    output = h.scheduler_output(
        new=(stream(env, "s1"),), handles={"s1": 1}, encoder_inputs={"s1": [0]}
    )
    h.real_step(env.state, env.model, bindings, output)
    assert outcome["reread"] >= 1, "the commit did not re-read component_pools"
    assert outcome["error"] is not None, "a stale commit was accepted"
    assert outcome["unchanged"]


# @spec REG-STATE-006
def test_file_rows_on_recycled_slots_and_mixed_batches(build, monkeypatch):
    """E7: a file whose slot is recycled to a new file (higher generation),
    and a file sharing its step with a stream, bind like stream rows."""
    env = build()
    h.stub_advance(monkeypatch)
    h.profile_run(env.state, env.model, core_embeds=False)
    bindings = pool(env)
    first = h.scheduler_output(new=(file(env, "f1"),), handles={"f1": (1, 1)})
    h.real_step(env.state, env.model, bindings, first)
    recycled = h.scheduler_output(
        new=(file(env, "f2"), stream(env, "s1")),
        finished=("f1",),
        handles={"f2": (1, 2), "s1": 2},
        encoder_inputs={"s1": [0]},
        step_seq=2,
    )
    h.real_step(env.state, env.model, bindings, recycled)
    assert [c[1] for c in bindings.bind_calls] == [("f1",), ("f2", "s1")]
    assert bindings.bind_calls[1][2] == (h.handle(1, 2), h.handle(2, 1))
    assert set(bindings.row_reads) <= {"component_pools", "slot_indices_cpu"}
    assert bindings.reads == []


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
        tree = ast.parse(source)
        for node in ast.walk(tree):
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
def test_graphed_preparation_captures_and_serving_refuses_a_cleared_wrapper(
    monkeypatch, card_dir
):
    """On CUDA with the dense-graphed decoder: the first profiling step
    captures every (served geometry, batch tier) entry in its graph wrapper;
    if a wrapper's entry is then emptied, a real step that would use it
    raises without capturing again."""
    from vllm.config import DeviceConfig

    h.spec_without_state_name(monkeypatch)
    vllm_config = h.build_vllm_config(
        card_dir, overrides={"decode_dispatch_arm": "dense-graphed"}
    )
    vllm_config.device_config = DeviceConfig(device="cuda")
    with torch.device("cuda"):
        model = h.build_model(vllm_config).to("cuda")
        state = h.build_model_state(vllm_config, model)
        h.profile_run(state, model, core_embeds=False)
    binding = model._decode_graph_binding
    assert binding is not None and binding.captured_keys
    wrappers = [entry.wrapper for entry in binding._entries.values()]
    assert all(
        any(e.cudagraph is not None for e in w.concrete_cudagraph_entries.values())
        for w in wrappers
    )
    for wrapper in wrappers:
        for entry in wrapper.concrete_cudagraph_entries.values():
            entry.cudagraph = None
    bindings = h.real_bindings(h.model_spec(model))
    state.initialize_state_slots(bindings)
    output = h.scheduler_output(
        new=(
            h.new_request(
                "s1",
                int(model.config.audio_chunk_token_id),
                audio=h.stream_envelope(model.config).cuda(),
            ),
        ),
        handles={"s1": 1},
        encoder_inputs={"s1": [0]},
    )
    error = h.call_with(h.real_step, state, model, bindings, output)
    assert error is not None
    assert all(
        e.cudagraph is None
        for w in wrappers
        for e in w.concrete_cudagraph_entries.values()
    ), "serving captured a graph"
