"""`collect_value_states.make_starts` with `band.n == 0` (no band keys, shape-consistent concatenation)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from collect_value_states import make_starts  # noqa: E402

CFG = {"stage1": {"band": {"n": 0}, "reset": {"seed": 730000}}}  # NO band.seed / band.min_wall_gap


def test_zero_band_does_not_read_band_keys_and_returns_n_by_5():
    starts, source = make_starts(CFG, 0, 3)
    assert starts.shape == (3, 5) and starts.dtype == np.float64
    assert source.tolist() == ["reset"] * 3
    assert np.isfinite(starts).all()


def test_reset_starts_follow_the_namespace_seed():
    a, _ = make_starts(CFG, 0, 2)
    b, _ = make_starts({"stage1": {"band": {"n": 0}, "reset": {"seed": 730001}}}, 0, 1)
    assert np.allclose(a[1], b[0])  # episode i uses env.reset(seed=reset.seed + i)


def test_empty_everything_is_a_0_by_5_array():
    starts, source = make_starts(CFG, 0, 0)
    assert starts.shape == (0, 5) and len(source) == 0
