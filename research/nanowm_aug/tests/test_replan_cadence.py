"""`closed_loop_rollout.chunk_slices` / `validate_cadence`: the --replan-every-rounds bookkeeping, offline."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from nwaug.wm.closed_loop_rollout import chunk_slices, validate_cadence, window_segment  # noqa: E402

FI, NEED, PAST, H_MAX = 5, 20, 10, 8


def test_r1_reproduces_the_original_slices_for_every_round():
    chunk = np.arange(16)
    for r in range(H_MAX):
        sample_now, off, look, ex = chunk_slices(r, 1, FI, NEED, PAST)
        assert sample_now and off == 0
        # original code: `chunks[b, : need - len(past)]` and `chunks[:, :fi]`
        assert np.array_equal(chunk[look], chunk[: NEED - PAST])
        assert np.array_equal(chunk[ex], chunk[:FI])
        assert look == slice(0, 10) and ex == slice(0, 5)


def test_r2_samples_on_even_rounds_only_and_offsets_odd_rounds():
    for r in range(H_MAX):
        sample_now, off, look, ex = chunk_slices(r, 2, FI, NEED, PAST)
        assert sample_now == (r % 2 == 0)
        assert off == (0 if r % 2 == 0 else 5)
        assert look == slice(off, off + 10)
        assert ex == slice(off, off + 5)
    assert chunk_slices(1, 2, FI, NEED, PAST)[2] == slice(5, 15)  # only [5, 10) reaches NanoWM


@pytest.mark.parametrize("every", [1, 2, 4])
def test_executed_slices_tile_the_stream_without_gap_or_overlap(every):
    covered = []
    for r in range(H_MAX):
        _s, _off, _look, ex = chunk_slices(r, every, FI, NEED, PAST)
        c = r // every  # chunk number; each chunk spans every*fi executed steps
        covered += [c * every * FI + i for i in range(ex.start, ex.stop)]
    assert covered == list(range(H_MAX * FI))  # in order, each stream step exactly once
    # and for R=2 the two executed slices of one chunk tile [0:10] of that chunk
    if every == 2:
        a, b = chunk_slices(0, 2, FI, NEED, PAST)[3], chunk_slices(1, 2, FI, NEED, PAST)[3]
        assert (a.start, a.stop, b.start, b.stop) == (0, 5, 5, 10)


def test_window_segment_is_unchanged_when_the_chunk_is_long_enough():
    past, chunk = np.arange(PAST * 2.0).reshape(PAST, 2), 100 + np.arange(32.0).reshape(16, 2)
    for r in range(4):
        look = chunk_slices(r, 2, FI, NEED, PAST)[2]
        seg = window_segment(past, chunk, look, NEED, FI)
        assert np.array_equal(seg, np.concatenate([past, chunk[look]], 0)[:NEED])


def test_window_segment_pads_only_the_dropped_group():
    past, chunk = np.zeros((PAST, 2)), 1 + np.arange(30.0).reshape(15, 2)
    look = chunk_slices(2, 3, FI, NEED, PAST)[2]  # R=3, off=10: chunk[10:20], horizon 15
    seg = window_segment(past, chunk, look, NEED, FI)
    assert seg.shape == (NEED, 2)
    assert np.array_equal(seg[PAST : NEED - FI], chunk[10:15])  # the executed slice is intact
    assert (seg[NEED - FI :] == chunk[14]).all()  # padding sits in the group NanoWM drops
    with pytest.raises(AssertionError):  # the executed slice itself must exist
        window_segment(past, chunk[:12], look, NEED, FI)


def test_h_max_must_be_a_multiple_of_r():
    validate_cadence(8, 1)
    validate_cadence(8, 2)
    validate_cadence(8, 4)
    for h, every in ((8, 3), (7, 2), (8, 0)):
        with pytest.raises(AssertionError):
            validate_cadence(h, every)
