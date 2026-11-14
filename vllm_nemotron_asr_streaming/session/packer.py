# SPDX-License-Identifier: Apache-2.0
"""The audio packer: 16 kHz samples to the model's input units.

One packer serves a session or a core realtime stream, and is fixed to one
geometry. Importing this module needs only numpy. ``pack_realtime`` is the
shared driver for core's ``/v1/realtime`` route; it imports the plugin's
lookahead and locale tables when it runs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Samples per 80 ms input frame at 16 kHz.
FRAME_SAMPLES = 1280
#: Header values: version, valid_samples, geometry_id, final_tail,
#: prompt_index, chunk_sequence, admission_ms_mod.
HEADER_WIDTH = 7
#: The widest regular unit (right context 13): 1280 * 14 samples.
MAX_UNIT_SAMPLES = FRAME_SAMPLES * 14
ROW_WIDTH = HEADER_WIDTH + MAX_UNIT_SAMPLES
ENVELOPE_VERSION = 2
#: Every header value is an integer below this, so it is exact in float32.
HEADER_LIMIT = 1 << 24


class PackerCapacityError(Exception):
    """A unit would need a ``chunk_sequence`` of 2**24 or more."""


@dataclass(frozen=True)
class Unit:
    """One input unit: a one-token prompt and one audio row."""

    prompt_token_ids: list[int]
    row: np.ndarray
    chunk_sequence: int
    final_tail: bool

    @property
    def prompt(self) -> dict[str, Any]:
        """The token prompt core's input processing parses."""
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "multi_modal_data": {"audio": self.row},
        }


def wall_clock_ms() -> int:
    return int(time.time() * 1000)


class Packer:
    """Turns appended samples into units of ``1280 * (right_context + 1)``."""

    # @spec SESS-PACK-001, SESS-PACK-002, SESS-PACK-003, SESS-PACK-004
    # @spec SESS-PACK-005, SESS-PACK-006, SESS-PACK-007, SESS-PACK-008
    def __init__(
        self,
        *,
        right_context: int,
        geometry_index: int,
        locale_row: int,
        carrier_token_id: int,
        clock_ms: Callable[[], int] = wall_clock_ms,
        first_chunk_sequence: int = 0,
    ) -> None:
        if right_context < 0:
            raise ValueError("right_context must be non-negative")
        self._unit_samples = FRAME_SAMPLES * (right_context + 1)
        if self._unit_samples > MAX_UNIT_SAMPLES:
            raise ValueError(f"right_context {right_context} exceeds the row width")
        self._geometry_index = int(geometry_index)
        self._locale_row = int(locale_row)
        self._carrier = int(carrier_token_id)
        self._clock_ms = clock_ms
        self._next_sequence = int(first_chunk_sequence)
        self._buffer = np.empty(0, dtype=np.float32)
        self._closed = False

    @property
    def buffered(self) -> int:
        """Samples appended but not yet in an emitted unit."""
        return int(self._buffer.shape[0])

    def set_locale_row(self, row: int) -> None:
        """Units packed after this call carry ``row`` as ``prompt_index``."""
        self._locale_row = int(row)

    def _stamp(self) -> int:
        return int(self._clock_ms()) % HEADER_LIMIT

    def _unit(
        self, samples: np.ndarray, *, final_tail: bool, sequence: int, stamp: int
    ) -> Unit:
        row = np.zeros(ROW_WIDTH, dtype=np.float32)
        row[:HEADER_WIDTH] = (
            ENVELOPE_VERSION,
            samples.shape[0],
            self._geometry_index,
            1 if final_tail else 0,
            self._locale_row,
            sequence,
            stamp,
        )
        row[HEADER_WIDTH : HEADER_WIDTH + samples.shape[0]] = samples
        return Unit([self._carrier], row, sequence, final_tail)

    def append(self, samples: np.ndarray) -> list[Unit]:
        """Add samples; return the regular units they complete, in order."""
        if self._closed:
            raise RuntimeError("the packer is closed")
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        merged = np.concatenate((self._buffer, data)) if self._buffer.size else data
        count = merged.shape[0] // self._unit_samples
        if count == 0:
            self._buffer = merged.copy()
            return []
        if self._next_sequence + count > HEADER_LIMIT:
            raise PackerCapacityError(
                "chunk_sequence space exhausted: a unit would need a sequence "
                f"of {HEADER_LIMIT} or more"
            )
        stamp = self._stamp()
        units = []
        for i in range(count):
            lo = i * self._unit_samples
            units.append(
                self._unit(
                    merged[lo : lo + self._unit_samples],
                    final_tail=False,
                    sequence=self._next_sequence + i,
                    stamp=stamp,
                )
            )
        self._next_sequence += count
        self._buffer = merged[count * self._unit_samples :].copy()
        return units

    def close(self) -> Unit:
        """Return the final tail (the remaining samples, possibly none)."""
        if self._closed:
            raise RuntimeError("the packer is closed")
        if self._next_sequence >= HEADER_LIMIT:
            raise PackerCapacityError(
                "chunk_sequence space exhausted: the final tail would need a "
                f"sequence of {HEADER_LIMIT} or more"
            )
        tail = self._unit(
            self._buffer,
            final_tail=True,
            sequence=self._next_sequence,
            stamp=self._stamp(),
        )
        self._next_sequence += 1
        self._buffer = np.empty(0, dtype=np.float32)
        self._closed = True
        return tail


async def _drain(queue: asyncio.Queue) -> None:
    while True:
        await queue.get()


# @spec SESS-RT-001, SESS-RT-002, SESS-RT-003, SESS-PACK-008, SESS-EP-001
async def pack_realtime(
    audio_stream: AsyncIterator[Any],
    input_stream: asyncio.Queue,
    hf_config: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Pack ``audio_stream`` for core's realtime route.

    Yields one prompt per unit at the served default lookahead and the default
    locale, then the final tail when ``audio_stream`` ends. ``input_stream`` is
    drained, unused, for the life of the stream. A ``PackerCapacityError``
    propagates, so the stream never ends with a normal close.
    """
    from vllm_nemotron_asr_streaming.locales import default_locale
    from vllm_nemotron_asr_streaming.lookahead import GEOMETRIES, default_lookahead

    table = hf_config.prompt_dictionary
    locale = default_locale(table)
    if locale is None:
        raise ValueError(
            "the locale table has several locales and no 'auto'; the realtime "
            f"route cannot choose one: {sorted(table)}"
        )
    right_context = default_lookahead(hf_config)
    geometry = {r: gi for r, _label, gi in GEOMETRIES}[right_context]
    packer = Packer(
        right_context=right_context,
        geometry_index=geometry,
        locale_row=table[locale],
        carrier_token_id=hf_config.audio_chunk_token_id,
    )
    drain = asyncio.ensure_future(_drain(input_stream))
    drain.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        async for chunk in audio_stream:
            for unit in packer.append(chunk):
                yield unit.prompt
        yield packer.close().prompt
    finally:
        drain.cancel()
