# SPDX-License-Identifier: Apache-2.0
"""The order of the step arena's sections (TXG-EXEC-009), on CPU row plans.
The arena's device contents and upload are in ``test_exec_gpu.py``."""

from __future__ import annotations

import pytest
from support import txg_api as api


# @spec TXG-EXEC-009
@pytest.mark.parametrize(
    "spec",
    ["c", "o", "cocoo", "oocc", "ccxoc", "xxc", "cccooo", "ococococ"],
)
def test_sections_list_each_kind_in_ascending_step_position(plan_builder, spec):
    plan = plan_builder(spec)
    chunk, echo = api.arena_rows(plan)
    expected_chunk = [i for i, ch in enumerate(spec) if ch == "c"]
    expected_echo = [i for i, ch in enumerate(spec) if ch != "c"]
    assert chunk == expected_chunk
    assert echo == expected_echo


# @spec TXG-EXEC-009, TXG-COV-010
def test_planning_failed_rows_are_listed_in_the_echo_section(plan_builder):
    chunk, echo = api.arena_rows(plan_builder("xcx"))
    assert chunk == [1] and echo == [0, 2]
