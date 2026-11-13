# SPDX-License-Identifier: Apache-2.0
"""An in-process fake of the engine client path a streaming session uses.

It models ``AsyncLLM.generate(..., stream_admission=True)`` and
``AsyncLLM.wait_stream_retired`` of the RFC 1 + RFC 2 build, using the real
vLLM types (``RequestOutput``, ``CompletionOutput``, ``StreamingInput``,
``StreamAdmissionError``, ``StreamRejection``, ``EngineDeadError``). Core line
numbers below are from ``vllm/v1/engine/async_llm.py`` (``AL``),
``vllm/v1/engine/output_processor.py`` (``OP``) and
``vllm/v1/engine/__init__.py`` (``EN``) of that build.

A test scripts the engine with *outcomes*:

* ``Park(label_ids, text)``: the unit's labels, then the park token that ends
  the unit (a ``stop`` finish on a request that is not finished: a chunk
  boundary, OP ``_is_chunk_boundary``).
* ``Text(label_ids, text)``: labels and text without the park token.
* ``Reject(reason)``: a rejection output; ``input_index`` is the 0-based yield
  position of the input (OP ``_process_stream_rejection``: ``update_seq - 1``).
* ``Finish(finish_reason, text)``: the request ends (``finished=True``).
* ``Die()``: the engine dies; ``generate`` raises ``EngineDeadError``.
* ``Fail(exception)``: ``generate`` raises that exception.
* ``Silent()``: no output (the unit stays outstanding).

``outcomes`` is consumed one entry per input read, in yield order; an entry
may be a list of outcomes emitted in order. ``default`` applies when the list
is exhausted. ``close_outcome`` is what the close (generator exhaustion)
produces.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from support.card import BLANK, CARD_LOOKAHEADS, LOCALES

ROW_WIDTH = 17_927

#: The label count a fake engine and a fake served configuration use when a
#: test does not name one. The session tests run once per layout (see
#: ``tests/session/conftest.py``) so a session that hard-codes the card's or
#: the default fixture's control-token IDs fails under the other layout.
DEFAULT_BLANK = BLANK


# ---------------------------------------------------------------- outcomes


@dataclass
class Park:
    label_ids: tuple[int, ...] = ()
    text: str = ""
    delay: float = 0.0
    #: Emit the labels and the park token as two outputs, not one.
    split: bool = False


@dataclass
class Text:
    """Label tokens and delta text of a unit, without the park token."""

    label_ids: tuple[int, ...] = ()
    text: str = ""
    delay: float = 0.0


@dataclass
class Reject:
    reason: str
    delay: float = 0.0
    #: Override the reported input_index (default: the yield position).
    input_index: int | None = None


@dataclass
class Finish:
    finish_reason: str = "stop"
    text: str = ""
    delay: float = 0.0


@dataclass
class Die:
    delay: float = 0.0


@dataclass
class Fail:
    exception: BaseException
    delay: float = 0.0


@dataclass
class Silent:
    pass


@dataclass
class Delay:
    """A retirement result: wait ``seconds``, then return."""

    seconds: float


# --------------------------------------------------------------- recording


@dataclass
class GenerateCall:
    request_id: str
    prompt: Any
    sampling_params: Any
    stream_admission: bool
    kwargs: dict[str, Any]


@dataclass
class FakeRequest:
    request_id: str
    inputs: list[Any] = field(default_factory=list)
    reserved: bool = True
    ended: bool = False
    #: The input generator was exhausted: the close was sent (AL 897-902).
    input_exhausted: bool = False
    retired: bool = False
    cancelled_by_client: bool = False
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    ended_event: asyncio.Event = field(default_factory=asyncio.Event)
    #: Admission reply, output completion and retirement are independent.
    reply_arrived: asyncio.Event = field(default_factory=asyncio.Event)
    slot_free: asyncio.Event = field(default_factory=asyncio.Event)
    reply_task: asyncio.Task | None = None
    input_task: asyncio.Task | None = None


class FakeEngine:
    """The fake engine client. See the module docstring."""

    def __init__(
        self,
        *,
        blank: int | None = None,
        retire_mode: str = "auto",
        outcomes: list[Any] | None = None,
        default: Any = None,
        close_outcome: Any = None,
        admission: Any = None,
        admission_delay: float = 0.0,
        first_read_delay: float = 0.0,
        retire: list[Any] | None = None,
        cancel_raises: BaseException | None = None,
    ) -> None:
        blank = DEFAULT_BLANK if blank is None else blank
        self.blank = blank
        #: "auto": the slot retires once the stream has ended and the
        #: admission reply has arrived. "manual": only ``retire_now()`` frees it
        #: (core releases the reservation through the retirement callback
        #: after the worker fence, not when output ends: core_client.py
        #: ~1075, state_scheduler.py ~279).
        self.retire_mode = retire_mode
        self._dead = asyncio.Event()
        self.park_id = blank + 1
        self.outcomes = list(outcomes or [])
        self.default = default if default is not None else Park()
        self.close_outcome = close_outcome if close_outcome is not None else Finish()
        #: None (accept), a StreamAdmissionReason string (refuse), "hang",
        #: "end" (generate ends without reading), or an exception instance.
        self.admission = admission
        self.admission_delay = admission_delay
        self.first_read_delay = first_read_delay
        #: One entry per wait_stream_retired call: None (return), an exception
        #: instance (raise), a Delay, "hang"; absent entries mean "natural".
        self.retire = list(retire or [])
        self.cancel_raises = cancel_raises
        self.errored = False

        self.generate_calls: list[GenerateCall] = []
        self.requests: dict[str, FakeRequest] = {}
        self.inputs: list[Any] = []
        self.sampling_params: list[Any] = []
        self.aborted: list[str] = []
        self.wait_calls: list[str] = []
        self.events: list[tuple] = []
        self.unexpected: list[str] = []
        #: Called with the GenerateCall when the generate body starts running.
        self.on_generate_started: Callable[[GenerateCall], None] | None = None
        self._hang = asyncio.Event()  # never set

    # ------------------------------------------------------------ plumbing

    def __getattr__(self, name: str) -> Any:
        # Only generate / wait_stream_retired are part of the client path the
        # session is allowed to use (no engine_core, add_request, abort ...).
        if name.startswith("__") or name in {"_hang"}:
            raise AttributeError(name)
        self.unexpected.append(name)
        raise AttributeError(name)

    def only(self) -> FakeRequest:
        assert len(self.requests) == 1, list(self.requests)
        return next(iter(self.requests.values()))

    def _log(self, *event: Any) -> None:
        self.events.append(event)

    # ------------------------------------------------------------- outputs

    def _output(
        self,
        request_id: str,
        *,
        text: str = "",
        token_ids: tuple[int, ...] = (),
        finish_reason: str | None = None,
        finished: bool = False,
        rejection: Any = None,
    ):
        from vllm.outputs import CompletionOutput, RequestOutput

        completion = CompletionOutput(
            index=0,
            text=text,
            token_ids=list(token_ids),
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=finish_reason,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[completion],
            finished=finished,
            stream_rejection=rejection,
        )

    def _outputs_for(self, req: FakeRequest, position: int, outcome: Any) -> list:
        rid = req.request_id
        if isinstance(outcome, Park):
            ids = tuple(outcome.label_ids)
            if outcome.split:
                return [
                    self._output(rid, text=outcome.text, token_ids=ids),
                    self._output(rid, token_ids=(self.park_id,), finish_reason="stop"),
                ]
            # OP 1010-1031: a unit ends at a chunk boundary (finish_reason on
            # a request that has not finished); the park token ends it.
            return [
                self._output(
                    rid,
                    text=outcome.text,
                    token_ids=ids + (self.park_id,),
                    finish_reason="stop",
                )
            ]
        if isinstance(outcome, Text):
            return [
                self._output(rid, text=outcome.text, token_ids=tuple(outcome.label_ids))
            ]
        if isinstance(outcome, Reject):
            from vllm.outputs import StreamRejection

            index = position if outcome.input_index is None else outcome.input_index
            # OP 1029: StreamRejection(update_seq - 1, ...) == yield position.
            return [self._output(rid, rejection=StreamRejection(index, outcome.reason))]
        if isinstance(outcome, Finish):
            return [
                self._output(
                    rid,
                    text=outcome.text,
                    finish_reason=outcome.finish_reason,
                    finished=True,
                )
            ]
        raise AssertionError(f"not an output outcome: {outcome!r}")

    async def _emit(self, req: FakeRequest, position: int, outcome: Any) -> None:
        items = outcome if isinstance(outcome, list) else [outcome]
        for item in items:
            if isinstance(item, Silent):
                continue
            await asyncio.sleep(getattr(item, "delay", 0.0))
            # OP 140-175 first_end_wins: nothing after the first end counts.
            if req.ended:
                return
            if isinstance(item, Die):
                self.kill()
                return
            if isinstance(item, Fail):
                self._end(req, item.exception)
                return
            for out in self._outputs_for(req, position, item):
                self._log("emit", req.request_id, position, type(item).__name__)
                req.queue.put_nowait(out)
                if out.finished:
                    self._end(req, None, queue_done=True)

    def _end(
        self,
        req: FakeRequest,
        error: BaseException | None,
        *,
        queue_done: bool = False,
    ) -> None:
        if error is not None and not req.ended:
            req.queue.put_nowait(error)
        if req.ended:
            return
        req.ended = True
        req.ended_event.set()
        self._log("ended", req.request_id)
        self._maybe_free(req)

    def _maybe_free(self, req: FakeRequest) -> None:
        if (
            self.retire_mode == "auto"
            and req.ended
            and req.reply_arrived.is_set()
            and not self.errored
            and not req.slot_free.is_set()
        ):
            req.slot_free.set()
            req.reserved = False  # AL 800-803 ``release`` callback.

    def retire_now(self, request_id: str | None = None) -> None:
        """Retire a stream explicitly (``retire_mode="manual"``)."""
        req = self.requests[request_id] if request_id else self.only()
        req.slot_free.set()
        req.reserved = False
        self._log("retire", req.request_id)

    # ------------------------------------------------------------- control

    def end_request(self, outcome: Any, request_id: str | None = None) -> None:
        """End a live request from the engine side (deadline, operator abort,
        error). ``Finish("abort")`` is an idle abort."""
        req = self.requests[request_id] if request_id else self.only()
        self._log("engine_end", req.request_id)
        if isinstance(outcome, Fail):
            self._end(req, outcome.exception)
            return
        for out in self._outputs_for(req, -1, outcome):
            req.queue.put_nowait(out)
        self._end(req, None)

    def abort_idle(self, request_id: str | None = None) -> None:
        self.end_request(Finish("abort"), request_id)

    def kill(self) -> None:
        """The engine dies (AL 478: ``if self.errored: raise EngineDeadError``)."""
        from vllm.v1.engine.exceptions import EngineDeadError

        self.errored = True
        self._dead.set()
        for req in self.requests.values():
            if not req.ended:
                self._end(req, EngineDeadError())

    # ------------------------------------------------------------ generate

    async def generate(
        self,
        prompt: AsyncGenerator,
        sampling_params: Any,
        request_id: str,
        *,
        stream_admission: bool = False,
        **kwargs: Any,
    ):
        from vllm.v1.engine import StreamAdmissionError, StreamAdmissionResult
        from vllm.v1.engine.exceptions import EngineDeadError

        call = GenerateCall(
            request_id, prompt, sampling_params, stream_admission, dict(kwargs)
        )
        self.generate_calls.append(call)
        self.sampling_params.append(sampling_params)
        if self.on_generate_started is not None:
            self.on_generate_started(call)
        self._log("generate", request_id)
        req: FakeRequest | None = None
        try:
            # AL 475-480 / 928-935: invalid sampling parameters are refused
            # before admission.
            if stream_admission:
                _validate_sampling_params(sampling_params)
            if self.errored:  # AL 478
                raise EngineDeadError()
            if not stream_admission:
                raise AssertionError("session must use stream_admission=True")
            # AL 786-790: a reserved caller ID is refused.
            earlier = self.requests.get(request_id)
            if earlier is not None and earlier.reserved:
                raise StreamAdmissionError(
                    _refusal(StreamAdmissionResult, request_id, "request_id_in_use")
                )
            req = FakeRequest(request_id)
            self.requests[request_id] = req

            async def reply(req=req):
                await asyncio.sleep(self.admission_delay)
                if self.admission != "hang":
                    req.reply_arrived.set()
                    self._maybe_free(req)

            req.reply_task = asyncio.ensure_future(reply())
            # The reply is independent of this call: a cancelled generate
            # still has to wait for it before the slot can retire.
            await asyncio.shield(req.reply_arrived.wait())
            admission = self.admission
            if admission == "end":
                self._end(req, None)
                return
            if isinstance(admission, BaseException):
                self._end(req, None)
                raise admission
            if isinstance(admission, str):
                # AL 861-863: ``raise StreamAdmissionError(result)``.
                self._end(req, None)
                raise StreamAdmissionError(
                    _refusal(StreamAdmissionResult, request_id, admission)
                )
            self._log("admitted", request_id)
            req.input_task = asyncio.ensure_future(self._handle_inputs(req, prompt))
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # AL 1070-1078: a cancelled generate aborts the request.
            if req is not None and not req.ended:
                self.aborted.append(request_id)
                self._log("aborted", request_id)
                if self.admission == "hang":
                    # An abort releases a hung admission.
                    req.reply_arrived.set()
                self._end(req, None)
            if self.cancel_raises is not None:
                raise self.cancel_raises from None
            raise
        finally:
            if req is not None:
                req.ended_event.set()
                if req.input_task is not None and not req.input_task.done():
                    req.input_task.cancel()

    async def _handle_inputs(self, req: FakeRequest, input_stream: Any) -> None:
        """Mirrors AL 866-910 ``handle_inputs``."""
        from vllm.engine.protocol import StreamingInput

        await asyncio.sleep(self.first_read_delay)
        position = 0
        try:
            async for chunk in input_stream:
                # AL 869: stop once the session has ended.
                if req.ended:
                    return
                # AL 877-895: process_inputs / extract_prompt_components run
                # on every input; a failure aborts the session.
                if not isinstance(chunk, StreamingInput):
                    raise TypeError("input is not a StreamingInput")
                _validate_prompt(chunk.prompt)
                if chunk.sampling_params:
                    _validate_sampling_params(chunk.sampling_params)
                req.inputs.append(chunk)
                self.inputs.append(chunk)
                self._log("read", req.request_id, position, time.monotonic())
                if position < len(self.outcomes):
                    outcome = self.outcomes[position]
                else:
                    outcome = self.default
                asyncio.ensure_future(self._emit(req, position, outcome))
                position += 1
            if not req.ended:
                # AL 897-902: the close, numbered update_seq + 1.
                req.input_exhausted = True
                self._log("close", req.request_id)
                close = self.close_outcome
                if isinstance(close, Reject):
                    # OP 1020-1023: a rejected close aborts the session.
                    close = Finish("abort", delay=close.delay)
                asyncio.ensure_future(self._emit(req, position, close))
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception as error:
            # AL 908-925: the input task aborts the session; generate raises
            # the cause.
            if not req.ended:
                self.aborted.append(req.request_id)
                self._end(req, error)

    # ------------------------------------------------------ retirement wait

    async def wait_stream_retired(self, request_id: str) -> None:
        """Mirrors AL 937-956."""
        from vllm.v1.engine import StreamAdmissionError, StreamAdmissionResult
        from vllm.v1.engine.exceptions import EngineDeadError

        self.wait_calls.append(request_id)
        self._log("wait", request_id)
        req = self.requests.get(request_id)
        if req is None:
            raise ValueError(f"No stream was admitted under {request_id!r}")
        script = self.retire.pop(0) if self.retire else "natural"
        if isinstance(script, BaseException):
            raise script
        if isinstance(script, Delay):
            await asyncio.sleep(script.seconds)
            return
        if script == "hang":
            await self._hang.wait()
        if script is None:
            return
        # Natural: core checks for death before waiting and again after
        # waking (core_client.py ~1095-1112), so a death after the slot
        # freed still fails the wait.
        if self.errored:
            raise EngineDeadError()
        if not req.slot_free.is_set():
            waiters = [
                asyncio.ensure_future(req.slot_free.wait()),
                asyncio.ensure_future(self._dead.wait()),
            ]
            try:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waiters:
                    w.cancel()
        if self.errored:
            raise EngineDeadError()
        if req.retired:
            # The retirement is no longer tracked (AL 951-956).
            raise ValueError(f"Stream {request_id!r} already retired")
        if self.admission is not None and isinstance(self.admission, str):
            if self.admission not in ("hang", "end"):
                raise StreamAdmissionError(
                    _refusal(StreamAdmissionResult, request_id, self.admission)
                )
        req.retired = True
        self._log("retired", request_id)


# ------------------------------------------------------ admission results


def refusal(request_id: str, reason: str) -> Any:
    """A refused admission as the target core builds it: the three-field
    ``StreamAdmissionResult(accepted, request_id, reason)`` (EN), imported
    from core so that a schema change there fails these doubles."""
    from vllm.v1.engine import StreamAdmissionResult

    return _refusal(StreamAdmissionResult, request_id, reason)


def _refusal(result_type: Any, request_id: str, reason: str) -> Any:
    return result_type(accepted=False, request_id=request_id, reason=reason)


# -------------------------------------------------------- unit extraction


def _validate_sampling_params(params: Any) -> None:
    """AL 966-979 ``_validate_admitted_stream_sampling_params``."""
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    if (
        not isinstance(params, SamplingParams)
        or params.n > 1
        or params.output_kind == RequestOutputKind.FINAL_ONLY
        or params.stop
    ):
        raise VLLMValidationError(
            "Input streaming not currently supported for pooling models, "
            "n > 1, request_kind = FINAL_ONLY or with stop strings."
        )
    if params.structured_outputs is not None:
        raise VLLMValidationError(
            "Stream admission does not support structured outputs."
        )


def _validate_prompt(prompt: Any) -> None:
    """The one prompt shape the model's processor parses: a token prompt of
    one carrier token plus ``multi_modal_data["audio"]``, one float32 row of
    width 17,927 (the processor reads the ``audio`` field; with
    ``process_inputs(..., resumable=True)`` and ``extract_prompt_components``
    an unparseable prompt raises here, as in core)."""
    from vllm.exceptions import VLLMValidationError

    def bad(why: str):
        raise VLLMValidationError(f"unparseable streaming prompt: {why}")

    if not isinstance(prompt, dict):
        bad("not a dict")
    ids = prompt.get("prompt_token_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 1
        or not all(isinstance(t, int) for t in ids)
    ):
        bad("prompt_token_ids must be a list with one int")
    mm = prompt.get("multi_modal_data")
    if not isinstance(mm, dict) or set(mm) != {"audio"}:
        bad("multi_modal_data must hold exactly 'audio'")
    row = mm["audio"]
    if hasattr(row, "detach"):
        row = row.detach().cpu().numpy()
    if (
        not isinstance(row, np.ndarray)
        or row.dtype != np.float32
        or row.shape != (ROW_WIDTH,)
    ):
        bad(f"audio must be one float32 row of width {ROW_WIDTH}")


def unit_of(streaming_input: Any) -> SimpleNamespace:
    """Read a sent unit from the one sanctioned location:
    ``prompt["prompt_token_ids"]`` and ``prompt["multi_modal_data"]["audio"]``.
    ``header`` is the first 7 values, ``samples`` the ``valid_samples`` that
    follow."""
    prompt = streaming_input.prompt
    _validate_prompt(prompt)
    audio = prompt["multi_modal_data"]["audio"]
    row = np.asarray(
        audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
    ).reshape(-1)
    header = [int(v) for v in row[:7]]
    valid = header[1]
    return SimpleNamespace(
        token_ids=list(prompt["prompt_token_ids"]),
        row=audio,
        header=header,
        version=header[0],
        valid_samples=valid,
        geometry_id=header[2],
        final_tail=header[3],
        prompt_index=header[4],
        chunk_sequence=header[5],
        admission_ms_mod=header[6],
        samples=row[7 : 7 + valid].copy(),
    )


def _walk(value: Any, found: list[np.ndarray], tokens: list[list[int]]) -> None:
    """Collect arrays and token lists anywhere in ``value`` (used only for
    equality fingerprints, never to locate a unit's row)."""
    if isinstance(value, np.ndarray):
        found.append(value)
        return
    if hasattr(value, "detach") and hasattr(value, "shape"):
        try:
            found.append(value.detach().cpu().numpy())
        except Exception:
            pass
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            if key == "prompt_token_ids" and isinstance(sub, (list, tuple)):
                tokens.append([int(t) for t in sub])
            _walk(sub, found, tokens)
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            _walk(sub, found, tokens)


def fingerprint(value: Any) -> Any:
    """A comparable canonical form of a prompt (arrays become bytes)."""
    found: list[np.ndarray] = []
    tokens: list[list[int]] = []
    _walk(value, found, tokens)
    return (
        tuple(tuple(t) for t in tokens),
        tuple(np.asarray(a).tobytes() for a in found),
    )


def contains_key(value: Any, needle: str) -> bool:
    """Whether any dict key (recursively) contains ``needle``, case-folded."""
    if isinstance(value, dict):
        return any(
            needle in str(k).lower() or contains_key(v, needle)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_key(v, needle) for v in value)
    if hasattr(value, "__dict__") and not isinstance(value, type | np.ndarray):
        return any(
            needle in str(k).lower() or contains_key(v, needle)
            for k, v in vars(value).items()
        )
    return False


# ------------------------------------------------------------ configuration


def make_hf_config(
    *,
    locales: dict[str, int] | None = None,
    lookaheads: list[int] | None = None,
    default_lookahead: int = 3,
    blank: int | None = None,
    architecture: str = "Nemotron3_5AsrStreamingForRNNT",
    supported: list[int] | None = None,
) -> SimpleNamespace:
    """A served (translated) configuration, shaped as the registration design
    describes it: ``num_asr_labels`` is ``blank``, the locale table is
    ``prompt_dictionary``, the card's set is ``card_num_lookahead_tokens`` and
    ``supported_num_lookahead_tokens`` overrides it when present."""
    blank = DEFAULT_BLANK if blank is None else blank
    cfg = SimpleNamespace(
        architectures=[architecture],
        num_asr_labels=blank,
        blank_token_id=blank,
        eos_token_id=blank + 1,  # park
        audio_chunk_token_id=blank + 2,  # stream chunk carrier
        eou_token_id=blank + 3,
        file_carrier_token_id=blank + 4,
        continuation_token_id=blank + 5,
        vocab_size=blank + 6,
        num_prompts=16,
        prompt_dictionary=dict(LOCALES if locales is None else locales),
        card_num_lookahead_tokens=list(
            CARD_LOOKAHEADS if lookaheads is None else lookaheads
        ),
        num_lookahead_tokens=default_lookahead,
        att_context_right=default_lookahead,
    )
    if supported is not None:
        cfg.supported_num_lookahead_tokens = list(supported)
    return cfg


def make_model_config(hf_config: SimpleNamespace | None = None) -> SimpleNamespace:
    hf = hf_config or make_hf_config()
    return SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        architectures=list(hf.architectures),
        architecture=hf.architectures[0],
    )


def run(coro: Any, timeout: float = 10.0) -> Any:
    """Run a coroutine to completion on a fresh loop, with an overall bound."""

    async def bounded():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(bounded())


def new_caller_id() -> str:
    return str(uuid.uuid4())


async def wait_until(predicate: Callable[[], Any], timeout: float = 2.0) -> None:
    """Yield to the loop until ``predicate()`` is truthy."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)
