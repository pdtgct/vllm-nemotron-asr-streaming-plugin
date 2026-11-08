# SPDX-License-Identifier: Apache-2.0
"""Session factory over vLLM's admitted-stream client path.

A session is one ``generate(..., stream_admission=True)`` task on the engine
client, owned by the factory's session table from before it starts until its
slot has retired, or until the factory is torn down. The factory and its leases
satisfy the ``SessionFactory`` and ``SessionLease`` protocols structurally; this
package never imports a frontend.

Importing this module imports vLLM, which ``vllm_nemotron_asr_streaming.session``
does not do until a factory is created.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import StreamAdmissionError
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_nemotron_asr_streaming.identity import ARCHITECTURE
from vllm_nemotron_asr_streaming.locales import default_locale
from vllm_nemotron_asr_streaming.lookahead import (
    GEOMETRIES,
    served_lookaheads,
)

from .errors import SessionFailed, SessionRefused
from .packer import Packer, PackerCapacityError, Unit, wall_clock_ms

logger = logging.getLogger(__name__)

#: Retained transcript limit per session, in UTF-8 bytes.
TRANSCRIPT_LIMIT_BYTES = 1 << 20

OPEN, FLUSHED, FAILED, ABORTED, RELEASED = (
    "open",
    "flushed",
    "failed",
    "aborted",
    "released",
)
#: Calls allowed per state. ``finish`` in ``released`` and ``release`` in
#: ``released`` return without effect; ``finish`` in ``aborted`` awaits the
#: cancelled task's end.
ALLOWED: dict[str, frozenset[str]] = {
    OPEN: frozenset({"feed", "update_locale", "flush", "abort"}),
    FLUSHED: frozenset({"finish", "release"}),
    FAILED: frozenset({"abort", "finish", "release"}),
    ABORTED: frozenset({"release", "finish"}),
    RELEASED: frozenset({"release", "finish"}),
}

PENDING, ACCEPTED, REFUSED = "pending", "accepted", "refused"


# @spec SESS-ENG-001
def _sampling_params() -> SamplingParams:
    from vllm_nemotron_asr_streaming.nemotron_asr.manifests import (
        REALTIME_MAX_TOKENS,
    )

    return SamplingParams(
        n=1,
        temperature=0,
        max_tokens=REALTIME_MAX_TOKENS,
        output_kind=RequestOutputKind.DELTA,
        skip_special_tokens=True,
    )


def _retrieve(task: asyncio.Task) -> None:
    """Done-callback: retrieve a task's exception so it is never reported
    as unretrieved."""
    if not task.cancelled():
        task.exception()


# @spec SESS-FAC-005
class SessionLease:
    """One session: the lease a frontend holds, and the factory's record."""

    def __init__(
        self,
        factory: StreamingSessionFactory,
        caller_id: str,
        packer: Packer,
        locale: str,
    ) -> None:
        self._factory = factory
        self.caller_id = caller_id
        #: Admission state of the factory's record (SESS-OWN-001).
        self.admission = PENDING
        self.state = OPEN
        self.failure: str | None = None
        self.task: asyncio.Task | None = None
        self.started = False
        self.cleanup_task: asyncio.Task | None = None
        self.locale = locale
        self._packer = packer
        self._loop = asyncio.get_running_loop()
        self._admitted: asyncio.Future[bool] = self._loop.create_future()
        self._units: asyncio.Queue[Unit | None] = asyncio.Queue()
        self._yielded = 0
        self._position = -1
        self._outcome: asyncio.Future[tuple[str, str]] | None = None
        self._busy = False
        self._active = False
        self._input_closed = False
        self._text = ""
        self._text_bytes = 0
        self._finished = False
        self._input_exhausted = False
        #: The terminal output was observed after the input generator ended.
        self._finished_after_close = False
        self._finish_reason: str | None = None

    # ------------------------------------------------------------ the task

    # @spec SESS-ENG-008
    async def _inputs(self):
        """The input generator: one unit at a time, only on request."""
        self.admission = ACCEPTED
        if not self._admitted.done():
            self._admitted.set_result(True)
        while True:
            unit = await self._units.get()
            if unit is None:
                self._input_exhausted = True
                return
            self._position = self._yielded
            self._yielded += 1
            yield StreamingInput(prompt=unit.prompt)

    async def _run(self) -> None:
        self.started = True
        try:
            async for output in self._factory._engine.generate(
                self._inputs(),
                _sampling_params(),
                self.caller_id,
                stream_admission=True,
            ):
                self._on_output(output)
        except StreamAdmissionError:
            self.admission = REFUSED
            self._factory._drop(self)
            raise

    def _on_task_done(self, task: asyncio.Task) -> None:
        _retrieve(task)
        if not self._admitted.done():
            self._admitted.set_result(False)
        self._on_stream_end()

    # @spec SESS-TXT-003
    def _on_output(self, output: Any) -> None:
        rejection = getattr(output, "stream_rejection", None)
        if rejection is not None:
            if rejection.input_index == self._position:
                self._resolve(("rejected", rejection.reason))
            return
        completion = output.outputs[0] if output.outputs else None
        if completion is not None:
            if completion.text and not self._append_text(completion.text):
                return
            ids = completion.token_ids
            if ids and ids[-1] == self._factory._park_token_id:
                self._resolve(("parked", ""))
        if output.finished:
            self._finished = True
            self._finished_after_close = self._input_exhausted
            self._finish_reason = completion.finish_reason if completion else None

    def _resolve(self, outcome: tuple[str, str]) -> None:
        if self._outcome is not None and not self._outcome.done():
            self._outcome.set_result(outcome)

    # @spec SESS-TXT-001, SESS-TXT-004
    def _append_text(self, text: str) -> bool:
        size = len(text.encode())
        if self._text_bytes + size > TRANSCRIPT_LIMIT_BYTES:
            self._fail("output_capacity")
            self._resolve(("overflow", ""))
            return False
        self._text_bytes += size
        self._text += text
        return True

    # ------------------------------------------------------------- failure

    # @spec SESS-LEASE-004, SESS-LEASE-005, SESS-LEASE-007
    def _fail(self, reason: str) -> None:
        if self.state == OPEN:
            self.state = FAILED
            self.failure = reason

    def _failed(self, reason: str) -> SessionFailed:
        self._fail(reason)
        return SessionFailed(reason)

    def _end_reason(self, *, idle: bool) -> str:
        task = self.task
        assert task is not None and task.done()
        if task.cancelled():
            return "cancelled"
        error = task.exception()
        if isinstance(error, EngineDeadError):
            return "engine_dead"
        if error is None and idle and self._finish_reason == "abort":
            return "expired"
        return "abnormal_end"

    # @spec SESS-EXP-001
    def _on_stream_end(self) -> None:
        """The task ended: fail an idle open lease; a running call reports it."""
        if self.state == OPEN and not self._active and not self._input_closed:
            self._fail(self._end_reason(idle=True))

    # -------------------------------------------------------- call checking

    @contextlib.contextmanager
    # @spec SESS-LEASE-001, SESS-LEASE-002, SESS-LEASE-003, SESS-LEASE-006
    def _call(self, name: str) -> Iterator[None]:
        """Serialization, closed factory, then state (SESS-LEASE-006)."""
        if self._busy:
            raise RuntimeError(f"a call is already running on session {self.caller_id}")
        if self._factory.closed:
            raise SessionFailed("cancelled", "the session factory is closed")
        if name not in ALLOWED[self.state]:
            if self.state == FAILED:
                raise SessionFailed(self.failure or "abnormal_end")
            raise SessionFailed("state", f"{name} is not allowed in state {self.state}")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False

    @contextlib.contextmanager
    def _io(self) -> Iterator[None]:
        """A feed or flush: a cancellation breaks the lease (SESS-LEASE-004)."""
        self._active = True
        try:
            yield
        except asyncio.CancelledError:
            self._fail("cancelled")
            raise
        finally:
            self._active = False
            assert self.task is not None
            if self.task.done():
                self._on_stream_end()

    # --------------------------------------------------------- unit sending

    # @spec SESS-ENG-002, SESS-ENG-003, SESS-ENG-004, SESS-ENG-005
    # @spec SESS-ENG-006, SESS-ENG-007, SESS-ENG-009
    async def _send(self, unit: Unit) -> None:
        """Send one unit and await its outcome, resending as the rejection
        reason allows."""
        task = self.task
        assert task is not None
        pause = self._factory._backoff_initial_s
        cache_misses = 0
        while True:
            if task.done():
                raise self._failed(self._end_reason(idle=False))
            self._outcome = self._loop.create_future()
            self._units.put_nowait(unit)
            await asyncio.wait(
                {self._outcome, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if not self._outcome.done():
                raise self._failed(self._end_reason(idle=False))
            kind, reason = self._outcome.result()
            self._outcome = None
            if kind == "parked":
                return
            if kind == "overflow":
                raise SessionFailed("output_capacity")
            if reason == "engine_paused":
                await asyncio.sleep(pause)
                pause = min(pause * 2, self._factory._backoff_max_s)
            elif reason == "mm_cache_miss" and cache_misses == 0:
                cache_misses = 1
            else:
                raise self._failed("rejected_input")

    # ----------------------------------------------------------- lease API

    # @spec SESS-IN-001, SESS-IN-002, SESS-IN-003, SESS-IN-004
    async def feed(self, samples: np.ndarray, *, on_accepted: Callable[[int], Any]):
        with self._call("feed"):
            if (
                not isinstance(samples, np.ndarray)
                or samples.dtype != np.float32
                or samples.ndim != 1
                or not np.isfinite(samples).all()
            ):
                raise ValueError(
                    "samples must be a finite one-dimensional float32 array"
                )
            with self._io():
                try:
                    units = self._packer.append(samples)
                except PackerCapacityError as error:
                    self._fail("input_capacity")
                    raise SessionFailed("input_capacity", str(error)) from error
                on_accepted(len(samples))
                texts = []
                for unit in units:
                    before = self._text_bytes
                    await self._send(unit)
                    if self._text_bytes != before:
                        texts.append(self._text)
                return texts

    # @spec SESS-LOC-001, SESS-LOC-002, SESS-LOC-003
    async def update_locale(self, locale: str) -> None:
        with self._call("update_locale"):
            row = self._factory._locale_table.get(locale)
            if row is None:
                raise SessionRefused("locale", f"locale {locale!r} is not served")
            self._packer.set_locale_row(row)
            self.locale = locale

    # @spec SESS-FLUSH-001, SESS-FLUSH-002, SESS-FLUSH-003
    async def flush(self) -> str:
        with self._call("flush"), self._io():
            try:
                tail = self._packer.close()
            except PackerCapacityError as error:
                self._fail("input_capacity")
                raise SessionFailed("input_capacity", str(error)) from error
            await self._send(tail)
            self._input_closed = True
            self._units.put_nowait(None)
            task = self.task
            assert task is not None
            await asyncio.wait({task})
            if (
                not task.cancelled()
                and task.exception() is None
                and self._finished
                and self._finished_after_close
                and self._finish_reason == "stop"
            ):
                self.state = FLUSHED
                return self._text
            raise self._failed(self._end_reason(idle=False))

    # @spec SESS-FIN-001
    async def finish(self) -> None:
        with self._call("finish"):
            if self.state == RELEASED:
                return
            # In ``aborted`` the cancelled task may still be ending.
            assert self.task is not None
            await asyncio.wait({self.task})

    # @spec SESS-ABORT-001, SESS-ABORT-002
    async def abort(self) -> None:
        with self._call("abort"):
            assert self.task is not None
            self._input_closed = True
            self.task.cancel()
            # The cancellation stands even if this call is cancelled while it
            # waits, so the lease is aborted before the wait and `release` stays
            # allowed; `release` keeps the task owned until it has ended.
            self.state = ABORTED
            await asyncio.wait({self.task})

    # @spec SESS-REL-001, SESS-REL-002, SESS-REL-003
    async def release(self) -> None:
        with self._call("release"):
            if self.state == RELEASED:
                return
            factory = self._factory
            try:
                await factory._engine.wait_stream_retired(self.caller_id)
            except asyncio.CancelledError:
                factory._start_cleanup(self)
                raise
            except EngineDeadError:
                factory._hold_until_ended(self)
                if self.failure == "engine_dead":
                    self.state = RELEASED
                    return
                raise SessionFailed("engine_dead") from None
            except (StreamAdmissionError, ValueError):
                pass
            except Exception:
                factory._hold_until_ended(self)
                raise
            if self.task is not None and not self.task.done():
                # Keep the generate task owned until it has ended.
                factory._start_cleanup(self)
            else:
                factory._drop(self)
            self.state = RELEASED

    # @spec SESS-TEAR-001
    def teardown(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()


class StreamingSessionFactory:
    """Opens sessions on an engine client and owns them until retirement."""

    # @spec SESS-FAC-004, SESS-OWN-001, SESS-OWN-005
    def __init__(
        self,
        engine_client: Any,
        hf_config: Any,
        *,
        backoff_initial_s: float = 0.05,
        backoff_max_s: float = 1.0,
    ) -> None:
        self._engine = engine_client
        self._hf = hf_config
        self._backoff_initial_s = backoff_initial_s
        self._backoff_max_s = backoff_max_s
        self._locale_table: dict[str, int] = dict(hf_config.prompt_dictionary)
        self._park_token_id: int = hf_config.eos_token_id
        self._carrier_token_id: int = hf_config.audio_chunk_token_id
        self._geometry_by_label = {
            label: (right, index)
            for right, label, index in GEOMETRIES
            if right in set(served_lookaheads(hf_config))
        }
        self.served_cadences: frozenset[str] = frozenset(self._geometry_by_label)
        self.locales: frozenset[str] = frozenset(self._locale_table)
        self.default_locale: str | None = default_locale(self._locale_table)
        self.sessions: dict[str, SessionLease] = {}
        self.closed = False
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._faults: list[BaseException] = []

    # @spec SESS-OPEN-001, SESS-OPEN-002, SESS-OPEN-003, SESS-OPEN-004
    # @spec SESS-OPEN-005, SESS-OPEN-006, SESS-OWN-006
    async def open(self, *, cadence: str, locale: str) -> SessionLease:
        if self.closed:
            raise SessionFailed("cancelled", "the session factory is closed")
        if cadence not in self._geometry_by_label:
            raise SessionRefused("cadence", f"cadence {cadence!r} is not served")
        if locale not in self._locale_table:
            raise SessionRefused("locale", f"locale {locale!r} is not served")
        right_context, geometry_index = self._geometry_by_label[cadence]
        caller_id = str(uuid.uuid4())
        lease = SessionLease(
            self,
            caller_id,
            Packer(
                right_context=right_context,
                geometry_index=geometry_index,
                locale_row=self._locale_table[locale],
                carrier_token_id=self._carrier_token_id,
                clock_ms=wall_clock_ms,
            ),
            locale,
        )
        self.sessions.setdefault(caller_id, lease)
        task = asyncio.ensure_future(lease._run())
        lease.task = task
        task.add_done_callback(lease._on_task_done)
        try:
            await asyncio.wait({lease._admitted})
        except asyncio.CancelledError:
            task.cancel()
            self._abandon(lease)
            raise
        if lease._admitted.result():
            return lease
        # The task ended before the input generator was first read.
        if lease.admission == REFUSED:
            error = task.exception()
            assert isinstance(error, StreamAdmissionError)
            raise SessionRefused(error.result.reason, str(error)) from error
        self._abandon(lease)
        reason = lease._end_reason(idle=False)
        raise SessionFailed(reason, f"the stream ended before admission: {reason}")

    def _drop(self, lease: SessionLease) -> None:
        if self.sessions.get(lease.caller_id) is lease:
            del self.sessions[lease.caller_id]

    # @spec SESS-ABORT-002
    def _hold_until_ended(self, lease: SessionLease) -> None:
        """Remove the record now if the task has ended; otherwise keep the record
        and a tracked task until the generate task ends, so ``teardown_all`` still
        finds it."""
        task = lease.task
        if task is None or task.done() or self.closed:
            self._drop(lease)
            return

        async def hold() -> None:
            await asyncio.wait({task})
            self._drop(lease)

        holder = asyncio.ensure_future(hold())
        self._cleanup_tasks.add(holder)
        holder.add_done_callback(self._cleanup_tasks.discard)
        holder.add_done_callback(_retrieve)

    def _abandon(self, lease: SessionLease) -> None:
        """The owner is gone: cancel the task and hand the session to cleanup,
        unless the task never started or the factory is closed."""
        if lease.task is not None and not lease.task.done():
            lease.task.cancel()
        if self.closed or not lease.started:
            self._drop(lease)
        else:
            self._start_cleanup(lease)

    # @spec SESS-OWN-002, SESS-OWN-003
    def _start_cleanup(self, lease: SessionLease) -> None:
        if lease.admission == REFUSED:
            self._drop(lease)
            return
        if self.closed or lease.cleanup_task is not None:
            return
        task = asyncio.ensure_future(self._cleanup(lease))
        lease.cleanup_task = task
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        task.add_done_callback(_retrieve)

    # @spec SESS-OWN-002, SESS-OWN-008
    async def _cleanup(self, lease: SessionLease) -> None:
        try:
            await self._engine.wait_stream_retired(lease.caller_id)
        except (StreamAdmissionError, ValueError):
            pass
        except Exception as error:
            logger.warning(
                "retirement wait for session %s failed: %r", lease.caller_id, error
            )
            self._faults.append(error)
        task = lease.task
        if task is not None and not task.done():
            # Ownership lasts until the generate task has ended too.
            await asyncio.wait({task})
        self._drop(lease)

    # @spec SESS-OWN-004
    async def drain(self, deadline: float) -> list[BaseException]:
        pending = [t for t in self._cleanup_tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
        faults, self._faults = self._faults, []
        return faults

    # @spec SESS-OWN-005, SESS-OWN-006, SESS-OWN-007
    def teardown_all(self) -> set[asyncio.Task]:
        self.closed = True
        tasks = set(self._cleanup_tasks)
        for lease in self.sessions.values():
            if lease.task is not None:
                tasks.add(lease.task)
        for task in tasks:
            task.cancel()
            if task.done():
                _retrieve(task)
            else:
                task.add_done_callback(_retrieve)
        return tasks


# @spec SESS-FAC-002, SESS-FAC-003
def create_session_factory(engine_client: Any, model_config: Any):
    """The factory for the served model, or ``None`` for another model."""
    architecture = (
        getattr(model_config, "architecture", None)
        or (list(model_config.architectures)[0])
    )
    if architecture != ARCHITECTURE:
        return None
    generate = getattr(engine_client, "generate", None)
    if not callable(generate):
        raise TypeError("the engine client has no generate()")
    if "stream_admission" not in inspect.signature(generate).parameters:
        raise TypeError(
            "the engine client's generate() does not accept stream_admission"
        )
    if not callable(getattr(engine_client, "wait_stream_retired", None)):
        raise TypeError("the engine client has no wait_stream_retired()")
    return StreamingSessionFactory(engine_client, model_config.hf_config)
