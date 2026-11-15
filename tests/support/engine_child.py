# SPDX-License-Identifier: Apache-2.0
"""Installed engines in fresh child processes, for the installed-engine checks
(``test_qualification_engine``, ``test_observability_gpu``).

Two things keep these checks out of the pytest process:

* ``AsyncLLM`` always runs its engine core in another process in the stage 1
  build, whatever ``VLLM_ENABLE_V1_MULTIPROCESSING`` says, so a check that
  must see the worker's log records, patch core or read the worker cannot use
  it. ``LLMEngine`` with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` builds the
  engine core and its uniprocess worker in the calling process (mode
  ``"llm"``). A check of the multiprocess path itself uses ``AsyncLLM``
  (mode ``"async"``).
* Each engine asks core for a fixed fraction of the device's total memory,
  which core refuses when less is free. The in-process CUDA tests of the
  tier leave device memory allocated in the pytest process (about 250 MB per
  model built), so an engine started late in the tier finds too little. A
  check therefore names its engine runs with the marker
  ``installed_engine(name, ...)``; the tier's conftest starts every selected
  run, one at a time, when collection finishes and before any test runs
  (:func:`prestart`), and the test reads the cached reply (:func:`reply`),
  starting the run itself only when nothing started it.

The child reads a JSON request and writes a JSON reply:

* request: ``mode``; ``settings`` (``AsyncEngineArgs`` keywords); ``patch``
  (``None`` or ``"suppress-profiling"``: core's graph-memory estimate,
  ``GPUModelRunner.profile_cudagraph_memory``, returns 0 without running the
  profiling capture); ``reads`` (names in :data:`READS`, each run on the
  worker through ``collective_rpc``); ``env`` (environment variables the
  child sets before it imports vLLM).
* reply: ``ok``; ``error`` (every message in the failure's chain, its types
  and its formatted traceback) when the engine did not start; ``reads``;
  ``records`` (every record the ``vllm_nemotron_asr_streaming`` loggers
  emitted at WARNING or above in the child, as ``[levelname, message]``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

PLUGIN_LOGGER = "vllm_nemotron_asr_streaming"
TESTS = Path(__file__).resolve().parents[1]
MARKER = "installed_engine"


def _record(worker: Any) -> Any:
    from support import txg_api

    return txg_api.counters(worker)["capture_record"]


def _counters(worker: Any) -> Any:
    from support import txg_api

    return txg_api.graph_module().transaction_graph_counters(worker)


def _estimate(worker: Any) -> Any:
    return [
        int(getattr(worker, "cudagraph_memory_estimate", 0)),
        int(getattr(worker, "available_kv_cache_memory_bytes", 0)),
    ]


#: What the child can read on the worker.
READS = {"record": _record, "estimate": _estimate, "counters": _counters}


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.items: list[list[str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.items.append([record.levelname, record.getMessage()])


def _plain(value: Any) -> Any:
    """``value`` as JSON: mapping keys as strings, sequences as lists, other
    leaves as themselves or their ``repr``."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _start_llm(request: dict[str, Any], reply: dict[str, Any]) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_engine_args(AsyncEngineArgs(**request["settings"]))
    reply["ok"] = True
    for name in request.get("reads", ()):
        reply["reads"][name] = engine.collective_rpc(READS[name])[0]
    return engine.engine_core.shutdown


def _start_async(request: dict[str, Any], reply: dict[str, Any]) -> Any:
    import asyncio

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def run() -> None:
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**request["settings"]))
        try:
            reply["ok"] = True
            for name in request.get("reads", ()):
                reply["reads"][name] = await llm.collective_rpc(READS[name])
        finally:
            llm.shutdown()

    asyncio.run(run())
    return None


def _child(request_path: str, reply_path: str) -> int:
    request = json.loads(Path(request_path).read_text())
    os.environ.update(request.get("env") or {})
    if request.get("mode", "llm") == "llm":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    records = _Records()
    plugin_logger = logging.getLogger(PLUGIN_LOGGER)
    plugin_logger.addHandler(records)
    if plugin_logger.getEffectiveLevel() > logging.WARNING:
        plugin_logger.setLevel(logging.WARNING)
    reply: dict[str, Any] = {"ok": False, "reads": {}}
    stop = None
    try:
        from vllm_nemotron_asr_streaming import register

        register()
        if request.get("patch") == "suppress-profiling":
            from vllm.v1.worker.gpu import model_runner

            model_runner.GPUModelRunner.profile_cudagraph_memory = lambda self: 0
        start = _start_llm if request.get("mode", "llm") == "llm" else _start_async
        try:
            stop = start(request, reply)
        except Exception as error:  # noqa: BLE001 - reported to the parent
            chain = _chain(error)
            reply["ok"] = False
            reply["error"] = {
                "messages": [str(item) for item in chain],
                "types": [type(item).__name__ for item in chain],
                "traceback": traceback.format_exc(),
            }
    finally:
        reply["records"] = records.items
        Path(reply_path).write_text(json.dumps(_plain(reply)))
        if stop is not None:
            stop()
    return 0


def run_engine(
    directory: Path,
    request: dict[str, Any],
    *,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Start one engine for ``request`` (see the module docstring) in a child
    process and return its reply. The pytest process first releases its
    allocator's cached blocks, so the child's engine sees what the device
    has free."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    directory.mkdir(parents=True, exist_ok=True)
    index = len(list(directory.glob("engine-request-*.json")))
    request_path = directory / f"engine-request-{index}.json"
    reply_path = directory / f"engine-reply-{index}.json"
    log = directory / f"engine-{index}.log"
    request_path.write_text(json.dumps(request))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TESTS), *filter(None, [env.get("PYTHONPATH")])]
    )
    with log.open("w") as output:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "support.engine_child",
                str(request_path),
                str(reply_path),
            ],
            env=env,
            cwd=str(TESTS),
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if not reply_path.exists():
        tail = log.read_text()[-4000:]
        raise AssertionError(
            f"the engine child wrote no reply (exit {done.returncode}):\n{tail}"
        )
    result = json.loads(reply_path.read_text())
    result["returncode"] = done.returncode
    result["log"] = str(log)
    return result


#: Replies of the runs :func:`prestart` started, by run name.
_REPLIES: dict[str, dict[str, Any]] = {}


def prestart(runs: Iterable[tuple[str, Callable[[], dict[str, Any]]]]) -> None:
    """Start each ``(name, request factory)`` run once, one at a time, and
    keep its reply for :func:`reply`. A factory that raises (a skipped
    check) starts nothing."""
    directory = Path(tempfile.mkdtemp(prefix="installed-engines-"))
    for name, make in runs:
        if name in _REPLIES:
            continue
        try:
            request = make()
        except BaseException:  # noqa: BLE001 - the test reports it itself
            continue
        _REPLIES[name] = run_engine(directory, request)


def reply(name: str, make: Callable[[], dict[str, Any]], directory: Path) -> dict:
    """The reply of run ``name``: started at collection's end, or now."""
    if name not in _REPLIES:
        _REPLIES[name] = run_engine(directory, make())
    return _REPLIES[name]


if __name__ == "__main__":
    sys.exit(_child(sys.argv[1], sys.argv[2]))
