# SPDX-License-Identifier: Apache-2.0
"""Warm-up domains (TXG-CAP-014): after startup no scheduled step, replayed
or falling back with a reachable reason, compiles or autotunes anything
core's JIT monitors observe under ``jit_monitor_mode`` ``error``.

The monitor is process-global once activated, so the scenario runs in a
fresh interpreter. It runs a covered step and one step of every reachable
fallback class: ``chunk-uncovered``, ``echo-uncovered`` and ``signature`` with
the fields ``endpoint_policy`` (a request that sets an endpoint policy) and
``forced_eou`` (a request that schedules the EOU token), as a direct engine
stream client produces them (``txgcap_harness.run_fallback_classes``)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.engine]

TESTS = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import pytest
    import torch
    from support import txg_gpu as g
    from support import txgcap_harness as t
    from support.distributed import single_process_distributed

    tmp = Path(sys.argv[1])
    with pytest.MonkeyPatch.context() as mp, single_process_distributed(
        tmp / "dist", "nccl"
    ):
        env = g.build_option_on(
            tmp, n=2, max_num_seqs=3, capacity=12, startup=False, monkeypatch=mp
        )
        core = env.core
        core.startup()
        from vllm.utils.jit_monitor import activate

        activate(mode="error")
        classes = t.run_fallback_classes(env, core)
        assert classes == [
            "chunk-uncovered",
            "echo-uncovered",
            "endpoint_policy",
            "forced_eou",
        ], classes
        torch.cuda.synchronize()
    print("OK")
    """
)


# @spec TXG-CAP-014
def test_no_monitored_compilation_after_startup(tmp_path):
    script = tmp_path / "jit_scenario.py"
    script.write_text(SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), *filter(None, [env.get("PYTHONPATH")])]
    )
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-8000:]
    )
