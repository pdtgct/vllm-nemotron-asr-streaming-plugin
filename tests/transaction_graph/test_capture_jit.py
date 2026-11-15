# SPDX-License-Identifier: Apache-2.0
"""Warm-up domains (TXG-CAP-014): after startup no scheduled step, replayed
or falling back with a reachable reason, compiles or autotunes anything
core's JIT monitors observe under ``jit_monitor_mode`` ``error``.

The monitor is process-global once activated, so the scenario runs in a
fresh interpreter. The ``signature`` fallback class needs a step carrying an
endpoint policy, which only a direct engine stream client sets; that case is
a qualification-run obligation (TXG-QUAL-007) until the envelope field that
carries it is exposed to the harness."""

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
    from support import txg_api as api
    from support import txg_gpu as g
    from support import txgcap_harness as t
    from support.distributed import single_process_distributed

    tmp = Path(sys.argv[1])
    with pytest.MonkeyPatch.context() as mp, single_process_distributed(
        tmp / "dist", "nccl"
    ):
        env = g.build_option_on(
            tmp, n=2, max_num_seqs=3, startup=False, monkeypatch=mp
        )
        core = env.core
        core.startup()
        from vllm.utils.jit_monitor import activate

        activate(mode="error")
        ids = [f"s{i}" for i in range(3)]
        handles = {r: i + 1 for i, r in enumerate(ids)}
        # covered: one CHUNK row
        core.step(t.stream_unit(env, "c0"), handles={"c0": 4})
        # chunk-uncovered: K = N + 1
        core.step(*[t.stream_unit(env, r) for r in ids], handles=handles)
        # echo-uncovered: E = N + 1 (the same requests, no new audio)
        core.step(cached=tuple(ids), handles=handles)
        fallbacks = t.counter(env, "fallbacks")
        reasons = {
            dict(zip(api.FALLBACK_KEY_FIELDS, k, strict=True))["reason"]
            for k in fallbacks
        }
        assert {"chunk-uncovered", "echo-uncovered"} <= reasons, fallbacks
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
