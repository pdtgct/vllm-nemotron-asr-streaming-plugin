# SPDX-License-Identifier: Apache-2.0
"""Packer specs (SESS-PACK-*): pure CPU, injected clock."""

from __future__ import annotations

import numpy as np
import pytest
from support.card import BLANK

WIDTH = 17_927
CARRIER = BLANK + 2

#: (right_context, geometry_index) in the implemented order.
GEOMS = [(0, 0), (1, 1), (3, 2), (6, 3), (13, 4)]


class Clock:
    def __init__(self, now: int = 1_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def make_packer(r=0, g=0, row=3, clock=None, **extra):
    from vllm_nemotron_asr_streaming.session.packer import Packer

    return Packer(
        right_context=r,
        geometry_index=g,
        locale_row=row,
        carrier_token_id=CARRIER,
        clock_ms=clock or Clock(),
        **extra,
    )


def ramp(n: int, start: int = 0) -> np.ndarray:
    return (np.arange(start, start + n, dtype=np.float32) + 1.0) / 1e6


def samples_of(unit) -> np.ndarray:
    return unit.row[7 : 7 + int(unit.row[1])]


def seg(r: int) -> int:
    return 1280 * (r + 1)


# @spec SESS-PACK-001
@pytest.mark.parametrize("r,g", GEOMS)
def test_unit_envelope(r, g):
    packer = make_packer(r, g, row=5)
    audio = ramp(seg(r) + 100)
    (unit,) = packer.append(audio)
    assert unit.prompt_token_ids == [CARRIER]
    assert unit.row.dtype == np.float32
    assert unit.row.shape == (WIDTH,)
    assert [int(v) for v in unit.row[:7]][:6] == [2, seg(r), g, 0, 5, 0]
    np.testing.assert_array_equal(unit.row[7 : 7 + seg(r)], audio[: seg(r)])
    assert not unit.row[7 + seg(r) :].any(), "zero padded"
    assert unit.final_tail is False or unit.final_tail == 0
    tail = packer.close()
    assert tail.row.shape == (WIDTH,) and tail.prompt_token_ids == [CARRIER]
    assert int(tail.row[1]) == 100 and int(tail.row[3]) == 1
    np.testing.assert_array_equal(tail.row[7:107], audio[seg(r) :])
    assert not tail.row[107:].any()


# @spec SESS-PACK-001
def test_unit_fields_chunk_sequence_and_final_tail():
    packer = make_packer(0, 0)
    (unit,) = packer.append(ramp(1280))
    assert unit.chunk_sequence == 0 and not unit.final_tail
    tail = packer.close()
    assert tail.chunk_sequence == 1 and tail.final_tail


# @spec SESS-PACK-002
@pytest.mark.parametrize("r,g", GEOMS)
def test_regular_units_carry_exactly_s_g_samples_consecutively(r, g):
    packer = make_packer(r, g)
    audio = ramp(seg(r) * 3 + 77)
    units = packer.append(audio)
    assert len(units) == 3
    assert all(int(u.row[1]) == seg(r) for u in units)
    joined = np.concatenate([samples_of(u) for u in units])
    np.testing.assert_array_equal(joined, audio[: seg(r) * 3])
    assert packer.buffered == 77


# @spec SESS-PACK-002
def test_segmentation_independent_of_append_sizes():
    audio = ramp(5000)
    whole = make_packer(0, 0).append(audio)
    pieces = []
    packer = make_packer(0, 0)
    for lo in range(0, 5000, 337):
        pieces += packer.append(audio[lo : lo + 337])
    assert len(pieces) == len(whole) == 3
    for a, b in zip(whole, pieces, strict=True):
        np.testing.assert_array_equal(samples_of(a), samples_of(b))
        assert a.chunk_sequence == b.chunk_sequence


# @spec SESS-PACK-002
def test_buffered_counts_incomplete_samples():
    packer = make_packer(0, 0)
    assert packer.buffered == 0
    assert packer.append(ramp(1000)) == []
    assert packer.buffered == 1000
    assert len(packer.append(ramp(500))) == 1
    assert packer.buffered == 220


# @spec SESS-PACK-003
def test_chunk_sequence_counts_from_zero():
    packer = make_packer(0, 0)
    units = packer.append(ramp(1280 * 2))
    units += packer.append(ramp(1280))
    assert [u.chunk_sequence for u in units] == [0, 1, 2]
    assert [int(u.row[5]) for u in units] == [0, 1, 2]
    tail = packer.close()
    assert tail.chunk_sequence == 3 and int(tail.row[5]) == 3


# @spec SESS-PACK-004
@pytest.mark.parametrize("r,g", GEOMS)
def test_close_emits_one_final_tail_with_remainder(r, g):
    packer = make_packer(r, g)
    audio = ramp(seg(r) + 123)
    packer.append(audio)
    tail = packer.close()
    assert tail.final_tail and int(tail.row[3]) == 1
    assert int(tail.row[1]) == 123 < seg(r)
    np.testing.assert_array_equal(samples_of(tail), audio[seg(r) :])


# @spec SESS-PACK-004
def test_final_tail_may_be_empty_and_is_last():
    packer = make_packer(0, 0)
    assert len(packer.append(ramp(1280))) == 1
    tail = packer.close()
    assert tail.final_tail and int(tail.row[1]) == 0 and tail.chunk_sequence == 1
    assert not tail.row[7:].any()
    try:
        extra = packer.append(ramp(1280))
    except Exception:
        extra = []  # a closed packer may refuse more audio
    assert extra == [], "no unit after the final tail"


# @spec SESS-PACK-004
def test_empty_stream_closes_with_empty_tail():
    tail = make_packer(3, 2).close()
    assert tail.final_tail and int(tail.row[1]) == 0 and tail.chunk_sequence == 0


# @spec SESS-PACK-005
@pytest.mark.parametrize("r,g", GEOMS)
def test_geometry_id_is_index_in_implemented_order(r, g):
    from vllm_nemotron_asr_streaming.lookahead import GEOMETRIES

    assert (r, g) in [(rc, gi) for rc, _label, gi in GEOMETRIES]
    packer = make_packer(r, g)
    (unit,) = packer.append(ramp(seg(r)))
    assert int(unit.row[2]) == g
    assert int(packer.close().row[2]) == g


# @spec SESS-PACK-005
def test_prompt_index_is_locale_row_when_packed():
    packer = make_packer(0, 0, row=4)
    (first,) = packer.append(ramp(1280))
    assert int(first.row[4]) == 4
    # Samples buffered before the change are packed under the new row.
    assert packer.append(ramp(700)) == []
    packer.set_locale_row(9)
    (second,) = packer.append(ramp(580))
    assert int(second.row[4]) == 9
    assert int(first.row[4]) == 4, "an emitted unit is not rewritten"
    packer.set_locale_row(2)
    assert int(packer.close().row[4]) == 2


# @spec SESS-PACK-006
def test_admission_stamp_is_time_unit_became_complete():
    clock = Clock(5_000)
    packer = make_packer(0, 0, clock=clock)
    clock.now = 6_000
    assert packer.append(ramp(1000)) == []
    clock.now = 7_000
    (unit,) = packer.append(ramp(300))
    assert int(unit.row[6]) == 7_000
    clock.now = 8_000
    units = packer.append(ramp(2560))
    assert [int(u.row[6]) for u in units] == [8_000, 8_000]
    clock.now = 9_000
    assert int(packer.close().row[6]) == 9_000
    assert int(unit.row[6]) == 7_000, "stamp is fixed once complete"


# @spec SESS-PACK-006
def test_admission_stamp_is_epoch_ms_modulo_2_24():
    clock = Clock(1_780_000_000_123)
    packer = make_packer(0, 0, clock=clock)
    (unit,) = packer.append(ramp(1280))
    assert int(unit.row[6]) == 1_780_000_000_123 % (1 << 24)
    clock.now = (1 << 24) * 3 + 5
    assert int(packer.close().row[6]) == 5


# @spec SESS-PACK-007
def test_header_values_are_nonnegative_integers_below_2_24():
    clock = Clock(1_780_000_000_999)
    packer = make_packer(13, 4, row=15, clock=clock)
    units = packer.append(ramp(seg(13) * 2 + 5)) + [packer.close()]
    for unit in units:
        header = unit.row[:7]
        assert (header >= 0).all() and (header < (1 << 24)).all()
        assert (header == np.floor(header)).all()
        assert header.astype(np.int64).astype(np.float32).tolist() == header.tolist()


# @spec SESS-PACK-008
def test_capacity_error_instead_of_wrapping_sequence():
    from vllm_nemotron_asr_streaming.session import PackerCapacityError

    # Layout addition: a way to start the sequence near its limit, since
    # reaching 2**24 units with real audio takes 21e9 samples.
    packer = make_packer(0, 0, first_chunk_sequence=(1 << 24) - 2)
    (last_ok,) = packer.append(ramp(1280))
    assert last_ok.chunk_sequence == (1 << 24) - 2
    (last_regular,) = packer.append(ramp(1280))
    assert last_regular.chunk_sequence == (1 << 24) - 1
    with pytest.raises(PackerCapacityError):
        packer.append(ramp(1280))


# @spec SESS-PACK-008
def test_capacity_error_on_final_tail():
    from vllm_nemotron_asr_streaming.session import PackerCapacityError

    packer = make_packer(0, 0, first_chunk_sequence=(1 << 24) - 1)
    (regular,) = packer.append(ramp(1280))
    assert regular.chunk_sequence == (1 << 24) - 1
    with pytest.raises(PackerCapacityError):
        packer.close()
