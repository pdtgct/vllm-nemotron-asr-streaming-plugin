# SPDX-License-Identifier: Apache-2.0
"""Pass/fail rules and record checks of the transaction-graph tier 5
qualification and calibration (TXG-QUAL-*, TXG-CAL-003).

These are the measurement harness's decision arithmetic, written once so the
preregistered run and its CPU tests use the same code: the restricted startup
every qualifying engine runs under (TXG-QUAL-001, -010, -021, -025), the step
rule (TXG-QUAL-013, -014), the ``U_api`` procedure and gain (TXG-QUAL-016,
-017), the void rule (TXG-QUAL-005), the frozen preregistration
(TXG-QUAL-018), the choice of ``N`` (TXG-QUAL-025, -027), and the required
contents of the qualification and calibration records (TXG-QUAL-019, -020,
TXG-CAL-003). Times are milliseconds.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Step-rule legs: 2 warm-up and 21 counted steps (TXG-QUAL-013).
WARMUP, COUNTED = 2, 21
MATERIAL_MS = 5.0
#: The step rule's populations and split compositions (TXG-QUAL-012).
ONE_KIND = (1, 31, 63)
SPLITS = ((31, 27, 4), (31, 16, 15), (63, 55, 8), (63, 32, 31))
TIER5_FLOOR = 63
COVERAGE = 0.99

# ---------------------------------------------------------------------------
# The restricted startup of every qualifying engine

#: Core's graph-memory estimate (enabled for the step rule, the installed
#: engine checks and the fit of TXG-QUAL-025).
ESTIMATE_ENV = "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"


def restricted_startup(
    max_num_seqs: int, *, max_num_active_seqs: int | None = None
) -> dict[str, Any]:
    """The ``AsyncEngineArgs`` keywords of the restricted startup
    (TXG-CFG-017 to TXG-CFG-019) under the fixed block count the
    qualification uses (TXG-QUAL-001, -010, -021): synchronous scheduling,
    ``cudagraph_mode`` and compilation mode NONE, and
    ``--num-gpu-blocks-override`` the admission limit plus one. ``B_hook``
    keeps these settings, the block count included; only
    ``transaction_graph_populations`` differs."""
    limit = max_num_seqs if max_num_active_seqs is None else max_num_active_seqs
    return {
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "compilation_config": {"cudagraph_mode": "NONE", "mode": 0},
        "num_gpu_blocks_override": int(limit) + 1,
    }


def estimate_enabled() -> dict[str, str]:
    """The environment that enables core's graph-memory estimate."""
    return {ESTIMATE_ENV: "1"}


def fit_startup(max_num_seqs: int) -> dict[str, Any]:
    """TXG-QUAL-025's "fits": both captures and their transient peaks
    succeed under ``--num-gpu-blocks-override`` ``max_num_seqs + 1`` with
    core's graph-memory estimate enabled (:func:`estimate_enabled`)."""
    return restricted_startup(max_num_seqs)


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
    "max_num_seqs",
    "max_num_batched_tokens",
    "transaction_graph_populations",
)
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
    figures and transient peak."""
    missing = []
    for capture in ("profiling", "serving"):
        figures = record.get(capture) or {}
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
