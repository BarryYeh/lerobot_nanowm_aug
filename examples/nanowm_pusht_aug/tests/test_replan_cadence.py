"""`closed_loop_rollout.chunk_slices` / `validate_cadence`: the --replan-every-rounds bookkeeping, offline."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from closed_loop_rollout import chunk_slices, validate_cadence  # noqa: E402

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
    assert chunk_slices(1, 2, FI, NEED, PAST)[2] == slice(5, 15)  # needs horizon >= 15 (ckpt has 16)


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


def test_h_max_must_be_a_multiple_of_r():
    validate_cadence(8, 1)
    validate_cadence(8, 2)
    validate_cadence(8, 4)
    for h, every in ((8, 3), (7, 2), (8, 0)):
        with pytest.raises(AssertionError):
            validate_cadence(h, every)
