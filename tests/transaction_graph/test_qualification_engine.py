# SPDX-License-Identifier: Apache-2.0
"""Tier 5 pass/fail checks that run on an A40 (``gpu``).

Every engine runs the restricted startup (``txgcap_qual.restricted_startup``:
``--no-async-scheduling`` and ``-cc.cudagraph_mode none -cc.mode none``) with
core's graph-memory estimate enabled, under core's default KV-cache sizing
(the extensible KV cache), unless a check names another sizing
(``txgcap_qual.kv_cache_sizing``); ``B_hook`` is the same settings with
``transaction_graph_populations`` absent, its KV-cache sizing included
(TXG-QUAL-001). The checks that need core's graph-memory estimate run under a
fixed ``--num-gpu-blocks-override`` of the admission limit plus one
(``txgcap_qual.estimate_startup``).

Installed-engine checks (``card``: the published checkpoint named by
``NEMOTRON_CARD_DIR``): startup refusals, each naming its remedy
(TXG-QUAL-023), and startup accepted under every KV-cache sizing; the
uncharged WARNING under a fixed ``--kv-cache-memory-bytes`` and the
required-but-suppressed profiling capture (TXG-QUAL-022); core's
graph-memory estimate against ``B_hook`` under the fixed block count
(TXG-QUAL-021); the block-0 capture under each of the three sizings with one
stream and each file served alone on the client path, through a
multiprocess engine, equal to ``B_hook``'s under the same sizing
(TXG-QUAL-028, through the qualification driver's ``alone`` mode, on
LibriSpeech test-clean named by ``TXG_QUAL_CORPUS``); an
exactness run's test-only headroom and its allocator record (TXG-QUAL-029);
and one ``U_api`` leg of each arm through ``AsyncLLM`` (TXG-QUAL-016; the
qualifying brackets are preregistered runs of the same driver mode). Each
engine runs in a fresh child process with its engine core and worker in that
process (``support.engine_child``: ``LLMEngine`` with
``VLLM_ENABLE_V1_MULTIPROCESSING=0``; ``AsyncLLM`` always runs the engine core
out of process in the stage 1 build), so a check reads the worker's log
records and figures, patches core, and sees the worker's own startup error.
The engines start, one at a time, before any test of the session runs
(marker ``installed_engine``), so the device memory each finds free does not
depend on which in-process CUDA tests ran before it.

Harness checks (the runnable synthetic card, ``support.txgcap_harness``):
the expected fallbacks of ``K > N`` and ``E > N`` steps with outputs equal to
``B_hook``'s, and of a multi-token row among one-token rows through the
qualification adapter (TXG-QUAL-007), and recovery after a worker-fatal fault
(TXG-QUAL-024). The served halves of TXG-QUAL-007 (a forced signature
mismatch) and TXG-QUAL-024 (``EngineDeadError`` at the client) are
qualification-run obligations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from support import model_harness as h
from support import txg_api
from support import txg_conditions as c
from support import txgcap_harness as t
from support import txgcap_qual as q
from support.txg_api import Record, Sizing

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

N = 64


def engine_settings(
    sizing: str = Sizing.EXTENSIBLE,
    *,
    blocks: int | None = None,
    budget: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """The ``AsyncEngineArgs`` keywords of the hook configuration under the
    restricted startup with the option at ``N`` and the KV-cache sizing
    ``sizing``, core's default unless named (under a fixed block count,
    ``blocks`` or the admission limit plus one; under a fixed budget,
    ``budget``; ``hf`` entries and settings set to ``...`` are removed: an
    unset setting)."""
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    hf = {"num_lookahead_tokens": 0, t.OPTION: [N]}
    hf.update(overrides.pop("hf", {}))
    hf = {k: v for k, v in hf.items() if v is not ...}
    settings = dict(
        model=card,
        config_format="nemotron_asr_streaming",
        tokenizer_mode="nemotron_asr_streaming",
        dtype="float32",
        enforce_eager=False,
        max_model_len=4096,
        max_num_seqs=N,
        hf_overrides=hf,
        **q.restricted_startup(),
        **q.kv_cache_sizing(
            sizing,
            blocks=(blocks or N + 1) if sizing == Sizing.BLOCK_COUNT else None,
            budget=budget,
        ),
    )
    settings.update(overrides)
    return {k: v for k, v in settings.items() if v is not ...}


def engine_args(**overrides: Any) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs

    return AsyncEngineArgs(**engine_settings(**overrides))


def create_config(args: Any) -> Any:
    from vllm_nemotron_asr_streaming import register

    register()
    return args.create_engine_config()


# ---------------------------------------------------------------------------
# TXG-QUAL-023: refusals


ARMS = re.compile(r"eager-graphed|dense-graphed")
EAGER = re.compile(r"--enforce-eager")

#: case -> (settings, the remedy patterns its refusal names).
REFUSALS: dict[str, tuple[dict[str, Any], tuple[re.Pattern[str], ...]]] = {
    "enforce-eager": (dict(enforce_eager=True), (EAGER, c.REMOVAL)),
    "320-ms": (dict(hf={"num_lookahead_tokens": 3}), (c.LOOKAHEAD,)),
    # At lookahead 1, where native burst is otherwise valid: at lookahead 0
    # the model configuration hook refuses native burst first (REG-LA-007),
    # before the configuration hook evaluates the transaction-graph checks
    # (REG-VAL-008); each hook lists its own failed checks (REG-VAL-003).
    "native-burst": (
        dict(hf={"experimental_native_burst": True, "num_lookahead_tokens": 1}),
        (),
    ),
    "not-a-list": (dict(hf={t.OPTION: N}), ()),
    "bool": (dict(hf={t.OPTION: [True]}), ()),
    "duplicate": (dict(hf={t.OPTION: [N, N]}), ()),
    "zero": (dict(hf={t.OPTION: [0]}), ()),
    "above-p-max": (dict(hf={t.OPTION: [N + 1]}), ()),
    # TXG-CFG-005: fewer than the admission limit plus one blocks.
    "block-override-small": (
        dict(sizing=Sizing.BLOCK_COUNT, blocks=8, hf={t.OPTION: [8]}),
        (c.BLOCKS_OVERRIDE,),
    ),
    "block-override-n": (
        dict(sizing=Sizing.BLOCK_COUNT, blocks=N),
        (c.BLOCKS_OVERRIDE,),
    ),
    "eager-encoder": (dict(hf={"encoder_execution_arm": "eager"}), (ARMS,)),
    "dense-graphed-encoder": (
        dict(hf={"encoder_execution_arm": "dense-graphed"}),
        (ARMS,),
    ),
    "eager-decoder": (dict(hf={"decode_dispatch_arm": "compact-eager"}), (ARMS,)),
    # The restricted startup (TXG-CFG-017 and TXG-CFG-018).
    "async-scheduling": (dict(async_scheduling=True), (c.NO_ASYNC_SCHEDULING,)),
    "async-scheduling-unset": (dict(async_scheduling=...), (c.NO_ASYNC_SCHEDULING,)),
    **{
        f"cudagraph-mode-{mode.lower()}": (
            dict(compilation_config={"cudagraph_mode": mode, "mode": 0}),
            (c.RUNNER_MODES_NONE,),
        )
        for mode in ("PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    },
    "cudagraph-mode-unset": (
        dict(compilation_config={"mode": 0}),
        (c.RUNNER_MODES_NONE,),
    ),
    **{
        f"compilation-mode-{value}": (
            dict(compilation_config={"cudagraph_mode": "NONE", "mode": value}),
            (c.RUNNER_MODES_NONE,),
        )
        for value in (1, 2, 3)
    },
    "compilation-mode-unset": (
        dict(compilation_config={"cudagraph_mode": "NONE"}),
        (c.RUNNER_MODES_NONE,),
    ),
    # TXG-CFG-020: a population above the admission limit
    # --max-num-active-seqs sets.
    "above-admission-limit": (
        dict(max_num_active_seqs=N // 2),
        (c.ACTIVE_SEQS,),
    ),
}

#: Every KV-cache sizing core offers, each a valid startup with the option on
#: under the restricted startup (TXG-QUAL-001; no setting of the sizing is a
#: condition of TXG-QUAL-023's refusals).
SIZINGS = {
    "extensible": dict(sizing=Sizing.EXTENSIBLE),
    "profiled": dict(sizing=Sizing.PROFILED),
    "block-count": dict(sizing=Sizing.BLOCK_COUNT),
    "budget": dict(sizing=Sizing.BUDGET, budget=8 << 30),
}


def refusal(case: str, settings: dict[str, Any]) -> str:
    error = h.call_with(create_config, engine_args(**settings))
    assert error is not None, f"{case} was accepted"
    return t.text(error)


@pytest.mark.card
@pytest.mark.parametrize("sizing", list(SIZINGS))
# @spec TXG-QUAL-001, TXG-QUAL-023
def test_the_valid_hook_configuration_starts_configuration(sizing):
    """The option under the restricted startup configures under every
    KV-cache sizing: core's default (no KV-cache option), a profiled budget,
    a fixed block count and a fixed budget."""
    create_config(engine_args(**SIZINGS[sizing]))


@pytest.mark.card
@pytest.mark.parametrize("case", list(REFUSALS))
# @spec TXG-QUAL-023
def test_startup_is_refused_for_each_unsupported_condition(case):
    settings, remedies = REFUSALS[case]
    message = refusal(case, dict(settings))
    conditions = [part for part in re.split(r"[;\n]", message) if t.OPTION in part]
    assert conditions, message
    for remedy in remedies:
        assert c.found(remedy, message), (case, remedy.pattern, message)
    if case == "native-burst":
        # The transaction-graph check's own refusal, not another one that
        # names the option: one aggregated condition names both.
        assert any("experimental_native_burst" in part for part in conditions), message


@pytest.mark.card
# @spec TXG-QUAL-023
def test_startup_is_refused_on_a_core_without_the_state_group_layouts(monkeypatch):
    """TXG-CFG-016: the refusal names core's accessor and removing the
    option."""
    from vllm.v1.worker.gpu.model_states import interface

    monkeypatch.delattr(interface.ModelState, "state_group_layouts")
    message = refusal("no state_group_layouts", {})
    assert any(
        c.is_state_layer_condition(part) for part in re.split(r"[;\n]", message)
    ) or c.is_state_layer_condition(message), message


@pytest.mark.card
# @spec TXG-QUAL-023
def test_startup_is_refused_on_a_core_without_the_model_state_hooks(monkeypatch):
    from vllm.v1.worker.gpu.model_states import interface

    monkeypatch.delattr(interface.ModelState, "post_warmup")
    error = h.call_with(create_config, engine_args())
    assert error is not None, "a core without post_warmup was accepted"


# ---------------------------------------------------------------------------
# Installed engine, one child process per engine (``support.engine_child``)


def _run(settings: dict[str, Any], *, patch=None, reads=()) -> dict[str, Any]:
    """An engine run with core's graph-memory estimate enabled."""
    return {
        "mode": "llm",
        "settings": settings,
        "patch": patch,
        "reads": list(reads),
        "env": q.estimate_enabled(),
    }


#: The engine runs of this module's checks, by name. The conftest starts the
#: selected ones before any test runs (``support.engine_child.prestart``).
#: ``fixed-budget`` holds a fixed KV budget (TXG-QUAL-022); the runs that
#: need core's graph-memory estimate (the required-but-suppressed profiling
#: capture, TXG-QUAL-022; the estimate check, TXG-QUAL-021) hold a fixed
#: block count of the admission limit plus one.
ENGINE_RUNS = {
    "fixed-budget": lambda: _run(
        engine_settings(Sizing.BUDGET, budget=8 << 30),
        reads=("record",),
    ),
    "suppressed-profiling": lambda: _run(
        engine_settings(Sizing.BLOCK_COUNT), patch="suppress-profiling"
    ),
    "b-hook-1": lambda: _run(
        engine_settings(Sizing.BLOCK_COUNT, hf={t.OPTION: ...}), reads=("estimate",)
    ),
    "b-hook-2": lambda: _run(
        engine_settings(Sizing.BLOCK_COUNT, hf={t.OPTION: ...}), reads=("estimate",)
    ),
    "option-on": lambda: _run(
        engine_settings(Sizing.BLOCK_COUNT), reads=("estimate", "record")
    ),
}


def engine(name: str, tmp_path) -> dict[str, Any]:
    from support.engine_child import reply

    return reply(name, ENGINE_RUNS[name], tmp_path)


def started(reply):
    assert reply["ok"], (reply.get("error") or {}).get("traceback", reply)
    return reply


@pytest.mark.card
@pytest.mark.installed_engine("fixed-budget")
# @spec TXG-QUAL-022, TXG-QUAL-001
def test_a_fixed_budget_logs_the_uncharged_warning_with_its_figures(tmp_path):
    reply = started(engine("fixed-budget", tmp_path))
    warnings = [
        message
        for level, message in reply["records"]
        if level == "WARNING" and "not charged" in message
    ]
    assert len(warnings) == 1, reply["records"]
    serving = reply["reads"]["record"][Record.SERVING]
    for value in (serving[Record.CALL_TOTAL], serving[Record.TXN_TOTAL]):
        assert t.mentions_bytes(warnings[0], value), (value, warnings[0])


@pytest.mark.card
@pytest.mark.installed_engine("suppressed-profiling")
# @spec TXG-QUAL-022
def test_a_suppressed_required_profiling_capture_fails_startup(tmp_path):
    """Under a fixed block count with core's estimate enabled the profiling
    capture is required (TXG-MEM-007); suppressing it fails startup."""
    reply = engine("suppressed-profiling", tmp_path)
    assert not reply["ok"], "startup succeeded without the required estimate"
    error = reply["error"]
    assert t.OPTION in "\n".join(error["messages"]), error["traceback"]


@pytest.mark.card
@pytest.mark.installed_engine("b-hook-1", "b-hook-2", "option-on")
# @spec TXG-QUAL-021, TXG-QUAL-001
def test_cores_estimate_charges_the_profiling_transaction_total(tmp_path, capsys):
    baseline = [
        started(engine(name, tmp_path))["reads"]["estimate"]
        for name in ("b-hook-1", "b-hook-2")
    ]
    reply = started(engine("option-on", tmp_path))
    estimate, kv = reply["reads"]["estimate"]
    record = reply["reads"]["record"]
    spread = abs(baseline[0][0] - baseline[1][0])
    transaction = record[Record.PROFILING][Record.TXN_TOTAL]
    assert estimate - max(b[0] for b in baseline) >= transaction - spread, (
        estimate,
        baseline,
        transaction,
        spread,
    )
    assert (
        record[Record.SERVING][Record.CALL_TOTAL]
        <= record[Record.PROFILING][Record.CALL_TOTAL]
    )
    with capsys.disabled():
        print(
            f"\nTXG-QUAL-021: estimate {estimate} B_hook {baseline} spread {spread} "
            f"transaction {transaction} KV budget difference "
            f"{kv - min(b[1] for b in baseline)}"
        )


# ---------------------------------------------------------------------------
# Block-0 capture under each KV-cache sizing (TXG-QUAL-028) and the exactness
# runs' headroom (TXG-QUAL-029), through the qualification driver


def run_driver(spec: dict[str, Any], tmp_path: Path, name: str) -> dict[str, Any]:
    """One qualification driver process (``support.txg_qual_driver``) for
    ``spec``; returns its reply."""
    path, out = tmp_path / f"{name}-spec.json", tmp_path / f"{name}-out.json"
    path.write_text(json.dumps(spec))
    tests = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tests), env.get("PYTHONPATH", "")) if p
    )
    subprocess.run(
        [sys.executable, "-m", "support.txg_qual_driver", str(path), str(out)],
        cwd=tests,
        env=env,
        timeout=3600,
        check=False,
    )
    return json.loads(out.read_text())


#: TXG-QUAL-028's server: ``N`` and ``max_num_seqs`` (the check concerns the
#: capture rule and served equality, which no ``N`` changes; a small ``N``
#: keeps the twelve engine processes of the check short).
ALONE_N = 8


def driver_spec(lookahead: int, sizing: str, *, option: bool) -> dict[str, Any]:
    """The driver's ``alone`` mode on LibriSpeech test-clean
    (``TXG_QUAL_CORPUS``, the directory holding ``test-clean``), whose
    utterances TXG-QUAL-004's files take."""
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    corpus = os.environ.get("TXG_QUAL_CORPUS")
    if not corpus:
        pytest.skip(
            "TXG_QUAL_CORPUS not set: TXG-QUAL-028 serves TXG-QUAL-004's files, "
            "which take LibriSpeech test-clean utterances"
        )
    hf = {"num_lookahead_tokens": lookahead}
    if option:
        hf[t.OPTION] = [ALONE_N]
    spec = {
        "mode": "alone",
        "card": card,
        "max_model_len": 8192,
        "max_num_seqs": ALONE_N,
        "lookahead": lookahead,
        "hf_overrides": hf,
        "kv_cache_sizing": sizing,
        "seed": 20261126,
        "alone_utterances": 2,
        "corpus": corpus,
    }
    return spec


def _fits(view: list[Any], recorded: list[Any]) -> bool:
    """TXG-LIFE-009's fit: the same address, page shape, strides and dtype,
    and a first dimension at most the recorded one."""
    address, shape, strides, dtype = view
    r_address, r_shape, r_strides, r_dtype = recorded
    return (
        address == r_address
        and shape[1:] == r_shape[1:]
        and strides == r_strides
        and dtype == r_dtype
        and shape[0] <= r_shape[0]
    )


@pytest.mark.card
@pytest.mark.parametrize(
    "sizing", [Sizing.EXTENSIBLE, Sizing.PROFILED, Sizing.BLOCK_COUNT]
)
@pytest.mark.parametrize("lookahead", [0, 1], ids=["80ms", "160ms"])
# @spec TXG-QUAL-028, TXG-QUAL-001, TXG-CAP-005, TXG-LIFE-009, TXG-MEM-018
def test_block_0_capture_serves_alone_equal_to_b_hook_under_each_sizing(
    tmp_path, lookahead, sizing
):
    """Under core's default sizing the serving capture runs while the
    extensible KV cache has committed only its minimum blocks, core's
    extension binds views that fit the recorded layout and leaves the binding
    unlatched, and the measured-memory INFO line appears; under a profiled
    budget the profiling capture runs over a one-block profiling KV cache;
    under a fixed block count both captures pass. Under each sizing one
    stream, served alone through the client path of core's ``/v1/realtime``
    as TXG-QUAL-024 serves it, and each of TXG-QUAL-004's files, each alone
    on the idle server as core's transcription serving submits it, completes
    with no failed row, replays cells with no ``signature`` fallback, each
    item checked on its own, and produces ``B_hook``'s tokens under the same
    sizing (``txgcap_qual.alone_problems``)."""
    on = run_driver(driver_spec(lookahead, sizing, option=True), tmp_path, "on")
    assert on["ok"], on.get("traceback")
    off = run_driver(driver_spec(lookahead, sizing, option=False), tmp_path, "off")
    assert off["ok"], off.get("traceback")
    for reply in (on, off):
        record = reply["identity"]
        assert q.missing_run_record_fields(record) == [], record
        assert record["kv_cache_sizing"] == sizing
    probes = on["probes"]
    captures = probes["captures"]
    record = on["capture_record"][Record.RECORD]
    # Every capture row names block 0 (TXG-CAP-005): the capture-only plans
    # of each capture that ran, ``2N`` cell programs each.
    rows = probes["capture_rows"]
    for name in captures:
        assert len(rows.get(name, ())) >= 2 * ALONE_N, (name, rows)
        assert all(blocks == [0] for blocks in rows[name]), (name, rows[name])
    if sizing == Sizing.EXTENSIBLE:
        assert "profiling" not in captures, "a profiling capture ran"
        assert captures["serving"]["num_committed_blocks"] in (1, 2), captures
        extension = probes["extension"]
        assert extension is not None and "error" not in extension, extension
        assert extension["latched"] is False, extension
        recorded = captures["serving"]["views"]
        assert set(extension["views"]) == set(recorded)
        for name, view in extension["views"].items():
            assert _fits(view, recorded[name]), (name, view, recorded[name])
        info = [
            message
            for level, message in probes["records"]
            if level == "INFO" and re.search(r"(?i)after the captures", message)
        ]
        assert len(info) == 1, probes["records"]
        assert not [m for level, m in probes["records"] if "not charged" in m], (
            "the measured startup logged the uncharged WARNING"
        )
        assert record[Record.FINAL_BOUND_COUNT] == on["identity"]["final_bound_count"]
    elif sizing == Sizing.PROFILED:
        assert record[Record.PROFILING][Record.PAGE_COUNT] == 1, record
        assert "profiling" in captures and "serving" in captures
    else:
        assert set(captures) == {"profiling", "serving"}
        assert record[Record.PROFILING][Record.PAGE_COUNT] == ALONE_N + 1
    items = [item["name"] for item in on["result"]["items"]]
    assert items[0] == "alone-stream" and len(items) == 1 + 4 + 2, items
    assert q.alone_problems(on["result"], off["result"]) == []


@pytest.mark.card
@pytest.mark.parametrize("mode", ["qm", "fallback"])
# @spec TXG-QUAL-029, TXG-QUAL-003, TXG-QUAL-007
def test_an_exactness_run_leaves_its_headroom_and_records_the_allocator(tmp_path, mode):
    """A short exactness run under the extensible KV cache, ``Q``/``M``
    (``qm``) and the uncovered steps (``fallback``), each with a headroom
    preregistered before the run from the mode's most requests, the card's
    bytes per page and the stated margin, left through
    ``--gpu-memory-utilization`` below 1.0: the detached copies hold block 0
    and one page per request and copy no more than the preregistered figure,
    the other committed pages keep their digests, the record holds core's
    logged torch memory limit and the allocator readings around the copy and
    passes TXG-QUAL-029's checks, and the run is no source of a served
    figure."""
    from support import txg_qual_driver as d

    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    n = 4
    spec = {
        "mode": mode,
        "card": card,
        "max_model_len": 82513,
        "max_num_seqs": n,
        "n": n,
        "lookahead": 0,
        "hf_overrides": {"num_lookahead_tokens": 0, t.OPTION: [n]},
        "kv_cache_sizing": Sizing.EXTENSIBLE,
        "compare": True,
        "seed": 3,
        "concurrencies": [n],
        "steps": 20,
        "m_steps": 20,
        "synthetic_lengths": [24000] * 16,
    }
    # Preregistered (TXG-QUAL-018) before the run: block 0 and one page per
    # request the mode holds (the admission limit for ``qm``, ``N + 4`` for
    # ``fallback``) of the card's state page, plus the stated margin.
    spec["headroom"] = q.preregistered_headroom(
        q.spec_max_requests(spec), d.card_page_bytes(card)
    )
    headroom = spec["headroom"]
    reply = run_driver(spec, tmp_path, f"exactness-{mode}")
    assert reply["ok"], reply.get("traceback")
    assert q.headroom_problems(reply) == [], reply["exactness_memory"]
    memory = reply["exactness_memory"]
    assert memory["torch_memory_limit"] > 0
    assert memory["copies"] > 0
    assert memory["copy_bytes"] <= headroom["copy_bytes"], memory
    assert memory["copy_allocations"] <= q.COPY_ALLOCATIONS, memory
    assert q.served_figure_runs([reply]) == []
    result = reply["result"]
    assert result["untouched_problems"] == [], result["untouched_problems"]
    if mode == "qm":
        assert result["mismatches"] == 0, result
    else:
        assert result["passed"], result


# ---------------------------------------------------------------------------
# Harness checks on the runnable synthetic card


def b_hook(on_cuda, **kwargs):
    """``B_hook``: the same CUDA configuration with the option absent."""
    return on_cuda(**kwargs).option_off()


def uncovered_schedule(env):
    ids = [f"s{i}" for i in range(env.n + 1)]
    handles = {r: i + 1 for i, r in enumerate(ids)}
    return [
        dict(new=tuple(t.stream_unit(env, r) for r in ids), cached=(), handles=handles),
        dict(new=(), cached=tuple(ids), handles=handles),
    ]


def run_steps(env, schedule):
    env.core.startup()
    outputs = []
    for item in schedule(env):
        hidden, _ = env.core.step(
            *item["new"], cached=item["cached"], handles=item["handles"]
        )
        outputs.append(hidden.clone())
    return outputs, h.pool_tensor(env.runner.bindings).clone()


# @spec TXG-QUAL-007
def test_uncovered_steps_fall_back_once_with_option_off_outputs(on_cuda):
    on = on_cuda(n=2, max_num_seqs=3)
    on_out, on_pool = run_steps(on, uncovered_schedule)
    fallbacks = t.counter(on, "fallbacks")
    for reason, kind in (("chunk-uncovered", "chunk"), ("echo-uncovered", "echo")):
        key = txg_api.fallback_key(reason=reason, kind=kind, count=3)
        assert fallbacks.get(key) == 1, fallbacks
    off_out, off_pool = run_steps(
        b_hook(on_cuda, n=2, max_num_seqs=3), uncovered_schedule
    )
    for a, b in zip(on_out, off_out, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(on_pool, off_pool)


# @spec TXG-QUAL-024
def test_a_fresh_process_recovers_after_a_worker_fatal_fault(on_cuda, monkeypatch):
    env = on_cuda(n=2)
    core = env.core
    core.startup()
    original = torch.cuda.CUDAGraph.replay
    launches = []

    def replay(self):
        original(self)
        launches.append(1)
        if len(launches) == 1:
            raise RuntimeError("test-only fault after the first launch")

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay)
    split = (t.stream_unit(env, "s1"), t.file_item(env, "f1", header_samples=1))
    assert h.call_with(core.step, *split) is not None
    later = h.call_with(core.step, t.stream_unit(env, "s2"), handles={"s2": 3})
    assert later is not None and t.FRESH_ENGINE.search(t.text(later))
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", original)
    env.state.shutdown()

    def one_stream(env):
        return [dict(new=(t.stream_unit(env, "r1"),), cached=(), handles={"r1": 1})]

    fresh_out, _ = run_steps(on_cuda(n=2), one_stream)
    off_out, _ = run_steps(b_hook(on_cuda, n=2), one_stream)
    assert torch.equal(fresh_out[0], off_out[0])


# @spec TXG-QUAL-007
def test_a_multi_token_row_among_one_token_rows_falls_back_once(on_cuda):
    """A direct client's two-token request beside a stream's chunk, through
    the qualification adapter: the step falls back once with reason
    ``multi-token``, planning fails the multi-token row alone, and the
    option-off transaction equals its reference on a detached pool copy."""
    from support import txg_qual_driver as d
    from support.txg_qual_adapter import QualAdapter, TransactionMonitor

    env = on_cuda(n=2, max_num_seqs=3)
    env.core.startup()
    adapter = QualAdapter(vllm_config=env.vllm_config, model=env.model, state=env.state)
    monitor = TransactionMonitor(env.model, compare=True).install()
    requests = d.Requests(env.model, lookahead=env.lookahead)
    blocks = d.Blocks(adapter.capacity)
    before = dict(t.counter(env, "fallbacks"))
    try:
        multi = requests.multi_token("multi-token", requests.label_tokens(2))
        monitor.begin_step()
        result = adapter.step(
            new=[t.stream_unit(env, "one-token"), multi],
            blocks=blocks.of(["one-token", "multi-token"]),
            tokens={"multi-token": 2},
        )
    finally:
        monitor.uninstall()
    after = dict(t.counter(env, "fallbacks"))
    grown = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    assert [k[0] for k in grown] == [d.MULTI_TOKEN] and sum(grown.values()) == 1, grown
    assert result.failed == {"multi-token"}, result.failed
    calls = {call.kind: call for call in monitor.calls}
    assert calls["execute"].covered is False
    assert calls["off"].compared and not calls["off"].problems, calls["off"].problems


@pytest.mark.card
@pytest.mark.parametrize("arm", ["off", "on"])
# @spec TXG-QUAL-016
def test_a_u_api_leg_parks_each_repeat_once_with_the_counted_replays(arm, tmp_path):
    """One ``U_api`` leg in a fresh multiprocess engine (the driver's
    ``uapi`` mode): ``AsyncLLM.generate`` with an input generator of the
    plugin packer's ``StreamingInput`` units, repeats 0 to 3, each timed to
    its unit's parked output and reaching exactly one park with no error; an
    ON leg replays the chunk cell at 1 once and the echo cell at 1 once per
    later step, with no fallback. The audio here is seeded noise; the
    qualifying brackets run test-clean under their frozen record."""
    card = os.environ.get("NEMOTRON_CARD_DIR")
    if not card:
        pytest.skip("NEMOTRON_CARD_DIR not set")
    hf = {"num_lookahead_tokens": 0}
    if arm == "on":
        hf[t.OPTION] = [4]
    spec = {
        "mode": "uapi",
        "arm": arm,
        "leg": "ON" if arm == "on" else "OFF1",
        "card": card,
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "lookahead": 0,
        "hf_overrides": hf,
        "seed": 0,
        "synthetic_lengths": [16000] * q.U_API_REPEATS,
        "engine_extra": {"gpu_memory_utilization": 0.5},
    }
    path, out = tmp_path / "spec.json", tmp_path / "out.json"
    path.write_text(json.dumps(spec))
    tests = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tests), env.get("PYTHONPATH", "")) if p
    )
    subprocess.run(
        [sys.executable, "-m", "support.txg_qual_driver", str(path), str(out)],
        cwd=tests,
        env=env,
        timeout=1800,
        check=False,
    )
    reply = json.loads(out.read_text())
    assert reply["ok"], reply.get("traceback")
    leg = reply["result"]
    assert leg["problems"] == [], leg
    assert all(r["ms"] > 0 for r in leg["repeats"]), leg
