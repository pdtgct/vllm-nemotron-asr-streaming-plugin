# SPDX-License-Identifier: Apache-2.0
"""Run one test alone in a fresh pytest process (``ran_in_child``).

The device-memory checks of the capture lifecycle read what the allocator and
the driver hold before and after a call. In a long tier the pytest process
also holds what earlier in-process CUDA tests left behind: objects that a
closure held outside Python's reachable graph keeps alive (model parameters,
transaction arguments), which a later call's inputs can join. Those are not
the call's, and a test cannot release them. The check therefore runs in a
child pytest process of its own, which holds nothing else; the parent waits,
so its own device memory is constant while the child reads.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

#: Set in the child to the node ID it runs.
ISOLATED = "TXG_ISOLATED_TEST"


def ran_in_child(request: Any, *, timeout: float = 1200.0) -> bool:
    """In the parent: run ``request``'s test alone in a child pytest process,
    assert that it passed, and return ``True`` (the parent's test then
    returns). In the child: return ``False`` (the test body runs)."""
    nodeid = request.node.nodeid
    if os.environ.get(ISOLATED) == nodeid:
        return False
    env = dict(os.environ)
    env[ISOLATED] = nodeid
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    done = subprocess.run(
        [*command, "-rfE", nodeid],
        cwd=str(request.config.rootpath),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = done.stdout[-8000:] + done.stderr[-2000:]
    assert done.returncode == 0, f"the isolated run failed:\n{output}"
    assert "1 passed" in done.stdout, f"the isolated run did not pass:\n{output}"
    return True
