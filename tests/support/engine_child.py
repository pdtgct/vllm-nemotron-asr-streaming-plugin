# SPDX-License-Identifier: Apache-2.0
"""Installed engines in fresh child processes, for the installed-engine checks
(``test_qualification_engine``, ``test_observability_gpu``,
``tests/registration/test_startup_gpu``).

Two things keep these checks out of the pytest process:

* ``AsyncLLM`` always runs its engine core in another process in the stage 1
  build, whatever ``VLLM_ENABLE_V1_MULTIPROCESSING`` says, so a check that
  must see the worker's log records, patch core or read the worker cannot use
  it. ``LLMEngine`` with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` builds the
  engine core and its uniprocess worker in the calling process (mode
  ``"llm"``). A check of the multiprocess path itself uses ``AsyncLLM``
  (mode ``"async"``).
* Each engine asks core for a fixed fraction of the device's total memory,
  which core refuses when less is free. While the tier runs, the pytest
  process holds device memory of its own (its CUDA context, the caching
  allocator's segments, library workspaces, session-scoped fixtures), so the
  memory an engine finds free would depend on which in-process CUDA tests
  ran before it. A check therefore names its engine runs with the marker
  ``installed_engine(name, ...)``; the tier's conftest starts every selected
  run, one at a time, when collection finishes and before any test runs
  (:func:`prestart`), and the test reads the cached reply (:func:`reply`),
  starting the run itself only when nothing started it.

The child reads a JSON request and writes a JSON reply:

* request: ``mode``; ``settings`` (``AsyncEngineArgs`` keywords); ``patch``
  (``None``; ``"suppress-profiling"``: core's graph-memory estimate,
  ``GPUModelRunner.profile_cudagraph_memory``, returns 0 without running the
  profiling capture; or ``"record-add-request"``: the model state's
  ``add_request`` records each request ID it receives, which the reply
  returns as ``reads["add_requests"]``; or ``"profiling-kv-cache-ref"``: the
  state layer takes a ``StorageWeakRef`` of the storage of the first KV cache
  core binds to it (the profiling KV cache under core's graph-memory
  estimate), and the reply's ``reads["profiling_kv_cache_expired"]`` records,
  at each later ``bind_kv_cache``, whether that reference had expired; or
  ``"inject-failed-requests"``: the model state records the worker's request
  IDs as ``reads["added"]``, those core retired through ``remove_request``
  as ``reads["retired"]`` and each real step's rows as
  ``reads["step_rows"]``, and its ``take_failed_requests`` adds, on every
  real step after a request retired, the first retired ID, and, once, on the
  first real step that holds a request whose ID starts with
  ``inject["live"]`` beside another, that request's ID; ``reads["injected"]``
  records both; the patches need mode ``"llm"``, where the worker runs in
  the child); ``reads`` (names in :data:`READS`, each run on the worker
  through ``collective_rpc``); ``phases`` (mode ``"llm"``: after the reads,
  complete-file requests run phase by phase, each phase's requests added
  together and stepped until none is unfinished; a request is ``{"id",
  "seconds", "seed"}``, seeded noise at 16 kHz through the model's file
  prompt; ``reads["finished"]`` lists ``[request ID, finish reason]`` per
  terminal output); ``inject`` (see the patch); ``env`` (environment
  variables the child sets before it imports vLLM).
* reply: ``ok``; ``error`` (every message in the failure's chain, its types
  and its formatted traceback) when the engine did not start or a phase
  raised; ``reads``; ``records`` (every record the
  ``vllm_nemotron_asr_streaming`` loggers emitted at WARNING or above in the
  child, as ``[levelname, message]``).
"""

from __future__ import annotations

import importlib
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


def _estimate(worker: Any) -> Any:
    return [
        int(getattr(worker, "cudagraph_memory_estimate", 0)),
        int(getattr(worker, "available_kv_cache_memory_bytes", 0)),
    ]


#: What the child can read on the worker: a function of the worker, or
#: ``"module:function"``, imported in the child and sent itself. ``counters``
#: is the plugin's own reader, ``transaction_graph_counters`` (TXG-OBS-008).
READS: dict[str, Callable[[Any], Any] | str] = {
    "record": _record,
    "estimate": _estimate,
    "counters": (
        "vllm_nemotron_asr_streaming.nemotron_asr.transaction_graph"
        ":transaction_graph_counters"
    ),
}


def _reader(name: str) -> Callable[[Any], Any]:
    """The function :data:`READS` names for ``name``."""
    read = READS[name]
    if isinstance(read, str):
        module, _, function = read.partition(":")
        return getattr(importlib.import_module(module), function)
    return read


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
        reply["reads"][name] = engine.collective_rpc(_reader(name))[0]
    _run_phases(engine, request.get("phases") or (), reply)
    return engine.engine_core.shutdown


def _file_prompt(model_config: Any, spec: dict[str, Any]) -> Any:
    """The model's prompt for a complete file of ``spec["seconds"]`` of
    seeded noise at 16 kHz (``build_file_prompt``, as the transcription
    route builds it)."""
    from types import SimpleNamespace

    import numpy as np

    from vllm_nemotron_asr_streaming.transcription import build_file_prompt

    rng = np.random.default_rng(int(spec.get("seed", 0)))
    samples = rng.standard_normal(int(float(spec["seconds"]) * 16000)) * 0.01
    return build_file_prompt(
        SimpleNamespace(
            task_type="transcribe",
            request_prompt="",
            hotwords=None,
            to_language=None,
            model_config=model_config,
            extra_args=None,
            language=None,
            audio=samples.astype(np.float32),
        )
    )


def _run_phases(engine: Any, phases: Any, reply: dict[str, Any]) -> None:
    """See ``phases`` in the module docstring."""
    if not phases:
        return
    from vllm.sampling_params import SamplingParams

    model_config = engine.vllm_config.model_config
    # core's transcription route: max_model_len minus the one-token prompt
    params = SamplingParams(temperature=0.0, max_tokens=model_config.max_model_len - 1)
    finished: list[list[Any]] = []
    reply["reads"]["finished"] = finished
    names: dict[str, str] = {}

    def name_of(request_id: str) -> str:
        # The output's ID is the one given, or the engine's (which starts
        # with it); either way the request's own name is recorded.
        if request_id in names:
            return names[request_id]
        given = [n for n in names.values() if request_id.startswith(n)]
        return given[0] if given else request_id

    for phase in phases:
        for spec in phase:
            used = engine.add_request(
                spec["id"], _file_prompt(model_config, spec), params
            )
            names[str(used or spec["id"])] = spec["id"]
            names[spec["id"]] = spec["id"]
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    reason = output.outputs[0].finish_reason
                    finished.append([name_of(str(output.request_id)), reason])


def _start_async(request: dict[str, Any], reply: dict[str, Any]) -> Any:
    import asyncio

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    async def run() -> None:
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**request["settings"]))
        try:
            reply["ok"] = True
            for name in request.get("reads", ()):
                reply["reads"][name] = await llm.collective_rpc(_reader(name))
        finally:
            llm.shutdown()

    asyncio.run(run())
    return None


def _record_add_request(reply: dict[str, Any]) -> None:
    """Record every request ID the model state's ``add_request`` receives in
    ``reply["reads"]["add_requests"]`` (the worker runs in this process)."""
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        NemotronASRModelState,
    )

    received: list[str] = []
    reply["reads"]["add_requests"] = received
    original = NemotronASRModelState.add_request

    def add_request(self: Any, req_index: int, new_req_data: Any) -> None:
        received.append(str(new_req_data.req_id))
        original(self, req_index, new_req_data)

    NemotronASRModelState.add_request = add_request


def _profiling_kv_cache_ref(reply: dict[str, Any]) -> None:
    """See ``"profiling-kv-cache-ref"`` in the module docstring
    (REG-STATE-019)."""
    from torch.multiprocessing.reductions import StorageWeakRef

    from vllm_nemotron_asr_streaming.nemotron_asr.state_layer import (
        NemotronStateLayer,
    )

    refs: list[Any] = []
    expired: list[bool] = []
    reply["reads"]["profiling_kv_cache_expired"] = expired
    original = NemotronStateLayer.bind_kv_cache

    def bind_kv_cache(self: Any, kv_cache: Any) -> None:
        # core's teardown of the profiling KV cache ran gc.collect already
        if refs:
            expired.append(bool(refs[0].expired()))
        else:
            refs.append(StorageWeakRef(kv_cache.untyped_storage()))
        original(self, kv_cache)

    NemotronStateLayer.bind_kv_cache = bind_kv_cache


def _inject_failed_requests(reply: dict[str, Any], request: dict[str, Any]) -> None:
    """See ``"inject-failed-requests"`` in the module docstring
    (REG-MS-005)."""
    from vllm_nemotron_asr_streaming.nemotron_asr.model_state_v2 import (
        NemotronASRModelState,
    )

    live = (request.get("inject") or {}).get("live")
    added: list[str] = []
    retired: list[str] = []
    steps: list[list[str]] = []
    injected: dict[str, list[Any]] = {"retired": [], "live": []}
    reply["reads"].update(
        added=added, retired=retired, step_rows=steps, injected=injected
    )
    current: dict[str, Any] = {"rows": None}
    cls = NemotronASRModelState
    add, remove = cls.add_request, cls.remove_request
    prepare, take = cls.prepare_inputs, cls.take_failed_requests

    def add_request(self: Any, req_index: int, new_req_data: Any) -> None:
        added.append(str(new_req_data.req_id))
        add(self, req_index, new_req_data)

    def remove_request(self: Any, req_id: str) -> None:
        # core also removes a new request before it adds it: not a retirement
        if req_id in added and req_id not in retired:
            retired.append(req_id)
        remove(self, req_id)

    def prepare_inputs(self: Any, input_batch: Any, req_states: Any) -> Any:
        rows = [str(r) for r in input_batch.req_ids]
        real = bool(rows) and all(r in added and r not in retired for r in rows)
        current["rows"] = rows if real else None
        if real:
            steps.append(rows)
        return prepare(self, input_batch, req_states)

    def take_failed_requests(self: Any) -> Any:
        result = take(self)
        rows, current["rows"] = current["rows"], None
        if rows is None:
            return result
        failed = set(result or ())
        if retired:
            failed.add(retired[0])
            injected["retired"].append(retired[0])
        victims = [r for r in rows if live and r.startswith(live)]
        if victims and len(rows) > 1 and not injected["live"]:
            failed.add(victims[0])
            injected["live"].append({"id": victims[0], "step": rows})
        return failed or result

    cls.add_request = add_request
    cls.remove_request = remove_request
    cls.prepare_inputs = prepare_inputs
    cls.take_failed_requests = take_failed_requests


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
        if request.get("patch") == "record-add-request":
            _record_add_request(reply)
        if request.get("patch") == "profiling-kv-cache-ref":
            _profiling_kv_cache_ref(reply)
        if request.get("patch") == "inject-failed-requests":
            _inject_failed_requests(reply, request)
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
    if done.returncode != 0:
        # The child writes its reply before it stops the engine, so a failed
        # shutdown or a crash shows only in its exit status. An expected
        # startup refusal is a reply with ``ok`` false and exit status 0.
        tail = log.read_text()[-4000:]
        raise AssertionError(
            f"the engine child exited with status {done.returncode} after "
            f"writing its reply:\n{tail}"
        )
    result = json.loads(reply_path.read_text())
    result["returncode"] = done.returncode
    result["log"] = str(log)
    return result


#: Replies of the runs :func:`prestart` started, by run name.
_REPLIES: dict[str, dict[str, Any]] = {}
#: Failures of the runs :func:`prestart` started, by run name.
_FAILURES: dict[str, Exception] = {}


def prestart(runs: Iterable[tuple[str, Callable[[], dict[str, Any]]]]) -> None:
    """Start each ``(name, request factory)`` run once, one at a time, and
    keep its reply for :func:`reply`. A factory that fails or skips starts
    nothing; the test reports that when it calls the factory itself. A run
    that fails (a timeout, no reply, a nonzero exit status) is kept as that
    run's failure, which :func:`reply` raises, so only the tests that read
    the run fail and the session continues."""
    import pytest

    directory = Path(tempfile.mkdtemp(prefix="installed-engines-"))
    for name, make in runs:
        if name in _REPLIES or name in _FAILURES:
            continue
        try:
            request = make()
        except (Exception, pytest.skip.Exception):
            continue
        try:
            _REPLIES[name] = run_engine(directory, request)
        except Exception as error:  # noqa: BLE001 - raised by reply()
            _FAILURES[name] = error


def reply(name: str, make: Callable[[], dict[str, Any]], directory: Path) -> dict:
    """The reply of run ``name``: started at collection's end, or now. Raises
    if the run started at collection's end failed."""
    if name in _FAILURES:
        raise AssertionError(
            f"engine run {name!r}, started before the tests, failed"
        ) from _FAILURES[name]
    if name not in _REPLIES:
        _REPLIES[name] = run_engine(directory, make())
    return _REPLIES[name]


if __name__ == "__main__":
    sys.exit(_child(sys.argv[1], sys.argv[2]))
