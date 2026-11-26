# SPDX-License-Identifier: Apache-2.0
"""Pass/fail rules and record checks of the transaction-graph tier 5
qualification and calibration (TXG-QUAL-*, TXG-CAL-001 to TXG-CAL-003).

These are the measurement harness's decision arithmetic, written once so the
preregistered run and its CPU tests use the same code: the restricted startup
every qualifying engine runs under and the KV-cache sizing beside it
(TXG-QUAL-001, -010, -021, -025), the step rule (TXG-QUAL-013, -014), the
``U_api`` procedure and gain (TXG-QUAL-016, -017), the served checks of
requests alone (TXG-QUAL-028), the void rule (TXG-QUAL-005), the frozen
preregistration (TXG-QUAL-018), the choice of ``N`` (TXG-QUAL-025, -027),
the exactness runs' test-only headroom (TXG-QUAL-029), and the required
contents of the qualification and calibration records (TXG-QUAL-001, -019,
-020, TXG-CAL-001 to -003). Times are milliseconds.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from support.txg_api import NOT_APPLICABLE, Sizing, sizing_arguments

#: Step-rule legs: 2 warm-up and 21 counted steps (TXG-QUAL-013).
WARMUP, COUNTED = 2, 21
MATERIAL_MS = 5.0
#: The step rule's populations and split compositions (TXG-QUAL-012).
ONE_KIND = (1, 31, 63)
SPLITS = ((31, 27, 4), (31, 16, 15), (63, 55, 8), (63, 32, 31))
TIER5_FLOOR = 63
COVERAGE = 0.99

# ---------------------------------------------------------------------------
# The restricted startup of every qualifying engine, and its KV-cache sizing

#: Core's graph-memory estimate (enabled for the step rule, the installed
#: engine checks and the fit of TXG-QUAL-025).
ESTIMATE_ENV = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"


def restricted_startup() -> dict[str, Any]:
    """The ``AsyncEngineArgs`` keywords of the restricted startup
    (TXG-CFG-017, TXG-CFG-018): synchronous scheduling and an explicit NONE
    ``cudagraph_mode`` and compilation mode. It names no KV-cache setting:
    the sizing is :func:`kv_cache_sizing`'s, core's default (the extensible
    KV cache) unless a spec names a fixed block count or budget
    (TXG-QUAL-001). ``B_hook`` keeps these settings and the run's sizing;
    only ``transaction_graph_populations`` differs."""
    return {
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "compilation_config": {"cudagraph_mode": "NONE", "mode": 0},
    }


def kv_cache_sizing(
    sizing: str = Sizing.EXTENSIBLE,
    *,
    blocks: int | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """The ``AsyncEngineArgs`` keywords of a KV-cache sizing
    (``txg_api.Sizing``; ``txg_api.sizing_arguments``): none for core's
    default, the extensible KV cache, which a serve command without KV-cache
    options selects; ``enable_extensible_kv_cache`` false for a profiled
    budget; ``num_gpu_blocks_override`` or ``kv_cache_memory_bytes`` for a
    fixed block count or budget, each of which turns the extensible KV cache
    off."""
    return sizing_arguments(sizing, blocks=blocks, budget=budget)


def estimate_startup(
    max_num_seqs: int, *, max_num_active_seqs: int | None = None
) -> dict[str, Any]:
    """The restricted startup under the fixed block count of the checks that
    need core's graph-memory estimate (the step rule, the estimate check and
    the first A40 run; TXG-QUAL-010, -020, -021): ``--num-gpu-blocks-override``
    the admission limit plus one, with the estimate enabled
    (:func:`estimate_enabled`). A fixed block count gives the option-on and
    option-off engines the same page count."""
    limit = max_num_seqs if max_num_active_seqs is None else max_num_active_seqs
    return {
        **restricted_startup(),
        **kv_cache_sizing(Sizing.BLOCK_COUNT, blocks=int(limit) + 1),
    }


def estimate_enabled() -> dict[str, str]:
    """The environment that enables core's graph-memory estimate."""
    return {ESTIMATE_ENV: "1"}


def fit_startup(max_num_seqs: int) -> dict[str, Any]:
    """TXG-QUAL-025's "fits": both captures and their transient peaks
    succeed under ``--num-gpu-blocks-override`` ``max_num_seqs + 1`` with
    core's graph-memory estimate enabled (:func:`estimate_enabled`)."""
    return {
        **restricted_startup(),
        **kv_cache_sizing(Sizing.BLOCK_COUNT, blocks=int(max_num_seqs) + 1),
    }


def served_run_n(
    p_max: int, admission_limit: int, fits: Mapping[int, bool]
) -> int | None:
    """TXG-QUAL-025: the ``N`` of the served ``Q`` run that chooses the
    inventory: ``min(P_max, admission limit)``, the largest ``N`` that
    TXG-CFG-004 and TXG-CFG-020 accept, when it fits, otherwise the largest
    ``N`` below it that fits (``None`` when none does)."""
    top = min(int(p_max), int(admission_limit))
    if fits.get(top):
        return top
    fitting = [n for n, ok in fits.items() if ok and n <= top]
    return max(fitting) if fitting else None


def counted(samples: Sequence[float]) -> Sequence[float]:
    """The counted samples of a leg recorded with its warm-ups."""
    if len(samples) == WARMUP + COUNTED:
        return samples[WARMUP:]
    if len(samples) != COUNTED:
        raise ValueError(f"a leg has {len(samples)} samples, not {COUNTED} counted")
    return samples


@dataclass(frozen=True)
class Block:
    """One block of legs OFF1, ON, OFF2."""

    first: Sequence[float]
    middle: Sequence[float]
    last: Sequence[float]

    def medians(self) -> tuple[float, float, float]:
        return tuple(
            statistics.median(counted(x)) for x in (self.first, self.middle, self.last)
        )


def block_material(block: Block) -> bool:
    """TXG-QUAL-013 for one block: ``Δ_b ≥ max(5 ms, drift_b)``."""
    off1, on, off2 = block.medians()
    delta = min(off1, off2) - on
    drift = abs(off1 - off2)
    return delta >= max(MATERIAL_MS, drift)


def material(blocks: Sequence[Block]) -> bool:
    """TXG-QUAL-013: material if and only if both blocks are."""
    if len(blocks) != 2:
        raise ValueError("the step rule runs exactly two blocks")
    return all(block_material(b) for b in blocks)


def one_kind_group(results: Mapping[int, bool]) -> bool:
    """TXG-QUAL-014: material at populations 1, 31 and 63, no pooling."""
    if set(results) != set(ONE_KIND):
        raise ValueError(f"one-kind results must cover {ONE_KIND}")
    return all(results[p] for p in ONE_KIND)


def split_group(results: Mapping[tuple[int, int, int], bool]) -> bool:
    """TXG-QUAL-014: material for all four split compositions."""
    if set(results) != set(SPLITS):
        raise ValueError(f"split results must cover {SPLITS}")
    return all(results[c] for c in SPLITS)


@dataclass(frozen=True)
class Bracket:
    """One ``U_api`` bracket: each leg's repeats 0..3 (repeat 0 discarded)."""

    off1: Sequence[float]
    on: Sequence[float]
    off2: Sequence[float]

    @staticmethod
    def _counted(leg: Sequence[float]) -> Sequence[float]:
        if len(leg) != 4:
            raise ValueError("a U_api leg has repeats 0 to 3")
        return leg[1:]

    def u(self) -> tuple[float, float, float]:
        return tuple(max(self._counted(x)) for x in (self.off1, self.on, self.off2))

    def m(self) -> tuple[float, float, float]:
        return tuple(
            statistics.median(self._counted(x)) for x in (self.off1, self.on, self.off2)
        )


#: ``U_api`` (TXG-QUAL-016): five brackets of legs OFF1, ON, OFF2, each leg
#: repeats 0 to 3 (repeat 0 discarded).
U_API_BRACKETS = 5
U_API_LEGS = ("OFF1", "ON", "OFF2")
U_API_REPEATS = 4


def u_api_schedule() -> list[dict[str, Any]]:
    """TXG-QUAL-016's processes in order: five brackets of legs OFF1, ON,
    OFF2, each leg a fresh engine process that follows a washout leg of its
    own arm (``washout``: run, then discarded)."""
    out = []
    for bracket in range(U_API_BRACKETS):
        for leg in U_API_LEGS:
            arm = "on" if leg == "ON" else "off"
            for washout in (True, False):
                out.append(
                    {"bracket": bracket, "leg": leg, "arm": arm, "washout": washout}
                )
    return out


def u_api_leg_problems(leg: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-016's checks of one recorded leg (``arm`` ``"on"`` or
    ``"off"``; ``repeats``, each with ``ms``, ``parks``, ``errors``,
    ``tokens`` (the unit's output tokens, the park included) and, for an ON
    leg, the counter ``deltas`` over the repeat: ``chunk1`` and ``echo1``
    (replays of the cells at count 1), ``other_replays`` and ``fallbacks``):
    repeats 0 to 3, each reaching exactly one park with no error output, and
    for an ON leg one more chunk-cell replay at 1, one more echo-cell replay
    at 1 per later step and no fallback."""
    problems: list[str] = []
    repeats = list(leg.get("repeats") or ())
    if len(repeats) != U_API_REPEATS:
        problems.append(f"{len(repeats)} repeats, not {U_API_REPEATS}")
    for index, repeat in enumerate(repeats):
        if repeat.get("parks") != 1:
            problems.append(f"repeat {index}: {repeat.get('parks')} parks, not 1")
        if repeat.get("errors"):
            problems.append(f"repeat {index}: an error output")
        if leg.get("arm") != "on":
            continue
        deltas = repeat.get("deltas") or {}
        later = int(repeat.get("tokens", 0)) - 1
        expected = {"chunk1": 1, "echo1": later, "other_replays": 0, "fallbacks": 0}
        for name, value in expected.items():
            if deltas.get(name) != value:
                problems.append(
                    f"repeat {index}: {name} {deltas.get(name)!r}, not {value}"
                )
    return problems


def u_api_bracket(legs: Sequence[Mapping[str, Any]]) -> Bracket:
    """A bracket from its three counted legs in order OFF1, ON, OFF2."""
    if [leg.get("leg") for leg in legs] != list(U_API_LEGS):
        raise ValueError(f"a bracket's legs are {U_API_LEGS} in order")
    return Bracket(*([r["ms"] for r in leg["repeats"]] for leg in legs))


def _gain(off1: float, on: float, off2: float) -> bool:
    return min(off1, off2) - on >= max(MATERIAL_MS, abs(off2 - off1))


def u_api_gain(brackets: Sequence[Bracket]) -> bool:
    """TXG-QUAL-017: the three conditions, over exactly five brackets."""
    if len(brackets) != 5:
        raise ValueError("U_api runs five brackets")
    per_bracket = all(_gain(*b.u()) for b in brackets)
    arm_max = [
        max(v for b in brackets for v in Bracket._counted(getattr(b, arm)))
        for arm in ("off1", "on", "off2")
    ]
    pooled = _gain(*arm_max)
    medians = all(
        min(m1, m2) - mon - max(MATERIAL_MS, abs(m2 - m1)) > 0
        for m1, mon, m2 in (b.m() for b in brackets)
    )
    return per_bracket and pooled and medians


#: TXG-QUAL-005 minimum case coverage.
MIN_Q_SPLIT = 200
MIN_Q_FRESH_AMONG_CONTINUING = 50
MIN_M_SPLIT_WITH_FILE = 100
#: The cases of TXG-EQ-004, each of which a run of ``M`` must contain.
EQ004_CASES = (
    "stream-first-chunk-among-continuing",
    "file-first-step",
    "regular-chunk",
    "silent-chunk",
    "final-tail-with-samples",
    "empty-final-tail",
    "zero-sample-file",
    "drain-step",
    "end-step-replay",
    "end-step-flush",
    "armed-echo-on-file-chunk",
    "split-mixed-file-and-stream",
)


def run_void(counts: Mapping[str, Any]) -> bool:
    """TXG-QUAL-005: whether a run's case coverage voids it."""
    cases = set(counts.get("m_cases", ()))
    return (
        counts["q_split_at_n"] < MIN_Q_SPLIT
        or counts["q_fresh_among_continuing_at_n"] < MIN_Q_FRESH_AMONG_CONTINUING
        or counts["m_split_with_file"] < MIN_M_SPLIT_WITH_FILE
        or not set(EQ004_CASES) <= cases
    )


def qualification_outcome(
    runs: Iterable[Mapping[str, Any]], seeds: Sequence[int]
) -> tuple[str, list[int]]:
    """TXG-QUAL-005 over the preregistered seed list: ``("fail", seeds)``
    when any run, void or not, has a mismatch or fallback; ``("pass",
    seeds)`` at the first non-void clean run; ``("incomplete", seeds)``
    when every seed ran void. Runs are in seed order."""
    used = []
    for run, seed in zip(runs, seeds, strict=False):
        used.append(seed)
        if run["mismatches"] or run["fallbacks"]:
            return "fail", used
        if not run_void(run["counts"]):
            return "pass", used
    return "incomplete", used


def frozen_before_counting(record: Mapping[str, Any], run: Mapping[str, Any]) -> bool:
    """TXG-QUAL-018: the dated record exists before the first counted sample
    and every frozen parameter equals the run's."""
    if record.get("frozen_at") is None or run.get("first_counted_at") is None:
        return False
    if record["frozen_at"] >= run["first_counted_at"]:
        return False
    frozen = record.get("parameters", {})
    return (
        bool(frozen)
        and all(run["parameters"].get(k) == v for k, v in frozen.items())
        and (set(run["parameters"]) == set(frozen))
    )


#: What a qualification run freezes before its first counted sample
#: (TXG-QUAL-018).
FROZEN_PARAMETERS = (
    "scripts",
    "clock",
    "concurrencies",
    "seed_list",
    "clip_manifest",
    "file_manifest",
    "offset_rule",
    "run_lengths",
    "file_admission",
    "idle_clock_rule",
    "minimum_case_coverage",
    "step_rule_prefix",
    "snapshot_rule",
    "plans",
    "compositions",
    "aggregation",
    "adapter_identity",
    "u_api_procedure",
    "u_api_input_generator",
    "u_api_rules",
    "u1_audio",
    "u1_plan",
    "u1_procedure",
    "exactness_headroom",
    "exactness_headroom_margin",
)


def missing_frozen_parameters(record: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-018: the frozen parameters the dated record lacks."""
    frozen = record.get("parameters") or {}
    return [name for name in FROZEN_PARAMETERS if name not in frozen]


def coverage(histogram: Mapping[tuple[int, int], int], n: int) -> float:
    total = sum(histogram.values())
    if not total:
        return 0.0
    covered = sum(c for (k, e), c in histogram.items() if k <= n and e <= n)
    return covered / total


def tier5_inventory(
    histogram: Mapping[tuple[int, int], int], fits: Mapping[int, bool]
) -> int | None:
    """TXG-QUAL-025: the smallest fitting ``N ≥ 63`` whose steps with
    ``K ≤ N`` and ``E ≤ N`` are at least 99 % of served steps; ``None``
    (no tier 5 claim) when none qualifies."""
    for n in sorted(n for n, ok in fits.items() if ok and n >= TIER5_FLOOR):
        if coverage(histogram, n) >= COVERAGE:
            return n
    return None


def deployment_choice(
    histogram: Mapping[tuple[int, int], int], fits: Mapping[int, bool]
) -> dict[str, Any] | None:
    """TXG-QUAL-027: a deployment ``N`` below 63 by the same rule, reported
    with its histogram, carrying no tier 5 claim."""
    for n in sorted(n for n, ok in fits.items() if ok and n < TIER5_FLOOR):
        if coverage(histogram, n) >= COVERAGE:
            return {"n": n, "histogram": dict(histogram), "tier5": False}
    return None


# ---------------------------------------------------------------------------
# Record contents

CAPTURE_FIGURES = (
    "call_total",
    "transaction_total",
    "kind_totals",
    "cells",
    "transient_peak",
)
FIRST_RUN_FIGURES = (
    "call_total",
    "transaction_total",
    "kind_totals",
    "transient_peak",
    "capture_seconds",
)
FIRST_RUN_TOP = ("margin_bytes", "margin_charge", "absorbed_rounding")
CALIBRATION_IDENTITY = (
    "checkpoint_revision",
    "plugin_version",
    "core_build",
    "device_model",
    "dtype",
    "process_constants",
    "geometry",
    "segment_arms",
    "scheduling_mode",
    "kv_cache_sizing",
    "max_num_seqs",
    "max_num_batched_tokens",
    "transaction_graph_populations",
)
#: The KV-cache sizings a calibration runs under, each giving its own record
#: (TXG-CAL-001, TXG-CAL-002): core's default and the step rule's fixed
#: ``--num-gpu-blocks-override``. No record applies at another sizing.
CALIBRATION_SIZINGS = (Sizing.EXTENSIBLE, Sizing.BLOCK_COUNT)
CALIBRATION_FIELDS = (
    "audio",
    "snapshot_rule",
    "plan",
    "procedure",
    "u1_on",
    "u1_off",
    "ladder",
    "served_replay_fraction",
    "device",
    "date",
    "identity",
)


def missing_capture_figures(record: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-019: each capture's call, transaction, per-kind and per-cell
    figures and transient peak. Under core's default KV-cache sizing no
    profiling capture runs and the record marks its figures not applicable
    (TXG-MEM-006, TXG-QUAL-001); the serving capture always runs."""
    missing = []
    for capture in ("profiling", "serving"):
        figures = record.get(capture)
        if capture == "profiling" and figures == NOT_APPLICABLE:
            continue
        figures = figures or {}
        missing += [f"{capture}.{k}" for k in CAPTURE_FIGURES if k not in figures]
        for cell in figures.get("cells", ()):
            if "growth" not in cell:
                missing.append(f"{capture}.cells[].growth")
                break
    return missing


def missing_first_run_figures(
    runs: Mapping[tuple[int, int], Mapping[str, Any]],
) -> list[str]:
    """TXG-QUAL-020: for 80 and 160 ms (lookahead 0, 1) and N = 16, 32, 64
    at max_num_seqs 64, both captures' totals, peak and time, and m, r and
    the absorbed rounding."""
    missing = []
    for lookahead in (0, 1):
        for n in (16, 32, 64):
            run = runs.get((lookahead, n))
            if run is None:
                missing.append(f"run lookahead={lookahead} N={n}")
                continue
            if run.get("max_num_seqs") != 64:
                missing.append(f"run lookahead={lookahead} N={n}: max_num_seqs")
            if run.get("kv_cache_sizing") != Sizing.BLOCK_COUNT:
                missing.append(
                    f"run lookahead={lookahead} N={n}: a fixed "
                    "--num-gpu-blocks-override"
                )
            if run.get("estimate_enabled") is not True:
                missing.append(
                    f"run lookahead={lookahead} N={n}: core's graph-memory estimate"
                )
            for capture in ("profiling", "serving"):
                figures = run.get(capture) or {}
                missing += [
                    f"lookahead={lookahead} N={n} {capture}.{k}"
                    for k in FIRST_RUN_FIGURES
                    if k not in figures
                ]
            missing += [
                f"lookahead={lookahead} N={n} {k}"
                for k in FIRST_RUN_TOP
                if k not in run
            ]
    return missing


def missing_calibration_fields(record: Mapping[str, Any]) -> list[str]:
    """TXG-CAL-003: the frozen calibration record's contents."""
    missing = [k for k in CALIBRATION_FIELDS if k not in record]
    audio = record.get("audio") or {}
    missing += [f"audio.{k}" for k in ("utterance_id", "chunk_index") if k not in audio]
    identity = record.get("identity") or {}
    missing += [f"identity.{k}" for k in CALIBRATION_IDENTITY if k not in identity]
    return missing


def missing_calibration_records(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """TXG-CAL-001, TXG-CAL-002: one complete record per KV-cache sizing of
    :data:`CALIBRATION_SIZINGS`, each naming its sizing in its identity and
    holding U(1) and the ladder on and off."""
    by_sizing: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        sizing = (record.get("identity") or {}).get("kv_cache_sizing")
        by_sizing.setdefault(sizing, []).append(record)
    missing = []
    for sizing in CALIBRATION_SIZINGS:
        found = by_sizing.pop(sizing, [])
        if len(found) != 1:
            missing.append(f"{len(found)} records under {sizing}, not 1")
            continue
        record = found[0]
        missing += [f"{sizing}: {k}" for k in missing_calibration_fields(record)]
        ladder = record.get("ladder") or {}
        missing += [
            f"{sizing}: ladder.{arm}" for arm in ("on", "off") if arm not in ladder
        ]
    missing += [f"a record under {sizing!r}" for sizing in by_sizing]
    return missing


# ---------------------------------------------------------------------------
# The record of every run (TXG-QUAL-001)

#: What the record of every qualification run, option on and ``B_hook``
#: alike, holds: its KV-cache sizing and each capture layout's page count;
#: under the extensible KV cache also the state group's final bound count,
#: read by the driver's instrumentation after core's extension.
RUN_RECORD_FIELDS = ("kv_cache_sizing", "capture_page_counts")


def missing_run_record_fields(identity: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-001: the fields a run's record (the driver's ``identity``)
    lacks. The final bound count is a number under the extensible KV cache
    and is not required otherwise."""
    missing = [k for k in RUN_RECORD_FIELDS if identity.get(k) is None]
    counts = identity.get("capture_page_counts") or {}
    if "serving" not in counts:
        missing.append("capture_page_counts.serving")
    if identity.get("kv_cache_sizing") == Sizing.EXTENSIBLE and not isinstance(
        identity.get("final_bound_count"), int
    ):
        missing.append("final_bound_count")
    return missing


# ---------------------------------------------------------------------------
# Requests served alone (TXG-QUAL-028)

#: The fallback reason that plugin-packed input never causes (TXG-QUAL-028).
SIGNATURE = "signature"


def alone_problems(on: Mapping[str, Any], off: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-028's served checks of the driver's ``alone`` results of an
    option-on run and its ``B_hook`` run under the same KV-cache sizing:
    the same items in the same order; every item of both runs completed
    (finish reason ``stop``, the normal close) with no failed row and no
    client error; each option-on item replayed at least one cell and had no
    fallback of reason ``signature``; no ``B_hook`` item replayed a cell;
    and each item's token IDs equal ``B_hook``'s. Every check is per item."""
    problems = []
    on_items = list(on.get("items") or [])
    off_items = list(off.get("items") or [])
    if not on_items:
        problems.append("the option-on run served no item")
    names = [item.get("name") for item in on_items]
    if names != [item.get("name") for item in off_items]:
        problems.append(
            f"the runs served different items: {names} and "
            f"{[item.get('name') for item in off_items]}"
        )
    for arm, items in (("option-on", on_items), ("B_hook", off_items)):
        for item in items:
            name = item.get("name")
            if item.get("error"):
                problems.append(f"{arm} {name}: the client received {item['error']}")
            if item.get("finish_reason") != "stop":
                problems.append(
                    f"{arm} {name} finished with {item.get('finish_reason')!r}, "
                    "not 'stop'"
                )
            failed = item.get("failed_rows")
            if failed != 0:
                problems.append(f"{arm} {name} has failed rows: {failed!r}")
    for item in on_items:
        name = item.get("name")
        replays = item.get("replays")
        if not isinstance(replays, int) or replays < 1:
            problems.append(f"option-on {name} replayed no cell ({replays!r})")
        signature = (item.get("fallbacks") or {}).get(SIGNATURE, 0)
        if signature:
            problems.append(
                f"option-on {name} fell back {signature} times with reason "
                f"{SIGNATURE!r}"
            )
    for item in off_items:
        if item.get("replays"):
            problems.append(f"B_hook {item.get('name')} replayed a cell")
    for a, b in zip(on_items, off_items, strict=False):
        if a.get("token_ids") != b.get("token_ids"):
            problems.append(
                f"{a.get('name')}: the option-on tokens differ from B_hook's"
            )
    return problems


# ---------------------------------------------------------------------------
# The exactness runs' test-only headroom (TXG-QUAL-029)

#: The allocator figures the record of an exactness run under the
#: extensible KV cache holds: the torch memory limit core logs when it sizes
#: the cache, and ``torch.cuda.memory_reserved`` and ``max_memory_reserved``
#: before and after the detached copy.
HEADROOM_FIGURES = (
    "torch_memory_limit",
    "memory_reserved_before",
    "max_memory_reserved_before",
    "memory_reserved_after",
    "max_memory_reserved_after",
)


def headroom_utilization(
    copy_bytes: int, margin_bytes: int, device_total_bytes: int
) -> float:
    """The ``--gpu-memory-utilization`` that leaves the preregistered
    headroom (the detached copy's bytes plus a stated margin) outside what
    core may use: below 1.0, and never a fixed ``--kv-cache-memory-bytes``,
    which would turn the extensible KV cache off."""
    headroom = int(copy_bytes) + int(margin_bytes)
    if headroom <= 0 or headroom >= int(device_total_bytes):
        raise ValueError(
            f"a headroom of {headroom} bytes does not fit a device of "
            f"{device_total_bytes} bytes"
        )
    return (int(device_total_bytes) - headroom) / int(device_total_bytes)


def _byte_count(value: Any) -> bool:
    """Whether ``value`` is a byte count: an integer, not a ``bool``, at
    least zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def headroom_problems(reply: Mapping[str, Any]) -> list[str]:
    """TXG-QUAL-029's checks of an exactness run's reply (the driver's
    ``spec``, ``identity`` and ``exactness_memory``). Each allocator figure
    and the copy's bytes must be a byte count; a missing figure, ``None`` or
    a non-integer value is a problem. The device-level context
    (``nvidia-smi``, ``torch.cuda.mem_get_info``) is recorded and read by no
    check."""
    problems = []
    spec = reply.get("spec") or {}
    identity = reply.get("identity") or {}
    settings = identity.get("settings") or {}
    headroom = spec.get("headroom") or {}
    if identity.get("kv_cache_sizing") != Sizing.EXTENSIBLE:
        problems.append("the run is not under the extensible KV cache")
    if "kv_cache_memory_bytes" in settings:
        problems.append("the headroom came from a fixed --kv-cache-memory-bytes")
    utilization = settings.get("gpu_memory_utilization")
    if not isinstance(utilization, float) or not 0.0 < utilization < 1.0:
        problems.append(f"--gpu-memory-utilization {utilization!r} is not below 1.0")
    for name in ("copy_bytes", "margin_bytes"):
        if not isinstance(headroom.get(name), int):
            problems.append(f"the preregistered headroom has no {name}")
    memory = reply.get("exactness_memory") or {}
    problems += [
        f"the record has no {name}"
        for name in HEADROOM_FIGURES
        if not _byte_count(memory.get(name))
    ]
    copied = memory.get("copy_bytes")
    if not _byte_count(copied):
        problems.append("the record has no copy_bytes")
    elif (
        isinstance(headroom.get("copy_bytes"), int) and copied > headroom["copy_bytes"]
    ):
        problems.append(
            f"the detached copy took {copied} bytes, more than the "
            f"preregistered {headroom['copy_bytes']}"
        )
    return problems


def served_figure_runs(replies: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The runs a served memory, capacity or latency figure may come from:
    those without the exactness runs' headroom (TXG-QUAL-029)."""
    return [r for r in replies if not (r.get("spec") or {}).get("headroom")]
