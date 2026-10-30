"""`nwaug.datagen.selection`: the stalled-coverage rule and water-filling."""

from __future__ import annotations

import numpy as np
import pytest

from nwaug.datagen.selection import qualifying, water_fill

STEPS = np.arange(0, 301, 5)
RULE = {"t_min": 15, "cov_lo": 0.5, "cov_hi": 0.8, "stall": 0.005}


def _est(points: dict[int, float]) -> np.ndarray:
    est = np.zeros(len(STEPS))
    for g, v in points.items():
        est[g] = v
    return est


def test_qualifying_needs_band_stall_and_an_open_episode():
    est = _est({8: 0.6, 9: 0.6, 10: 0.6, 11: 0.7, 12: 0.7})  # g 10 stalled (0.6 - 0.6), g 11 rising
    rng = np.random.default_rng(0)
    ok = qualifying(STEPS, est, -1, per_episode=10, gap=0, rng=rng, **RULE)
    assert 10 in ok and 11 not in ok and 12 not in ok  # 12: 0.7 - est[10] = 0.1 > stall
    assert qualifying(STEPS, est, 50, per_episode=10, gap=0, rng=rng, **RULE) == []  # ended at t=50


def test_qualifying_respects_per_episode_and_gap():
    est = _est(dict.fromkeys(range(3, 40), 0.6))
    ok = qualifying(STEPS, est, -1, per_episode=3, gap=20, rng=np.random.default_rng(1), **RULE)
    assert len(ok) == 3 and all(abs(STEPS[a] - STEPS[b]) >= 20 for a in ok for b in ok if a != b)


def test_water_fill_caps_and_spreads():
    q = water_fill({1: 2, 2: 10, 3: 10}, 12, np.random.default_rng(0))
    assert q[1] == 2 and q[2] + q[3] == 10 and abs(q[2] - q[3]) <= 1


def test_water_fill_refuses_more_than_capacity():
    with pytest.raises(SystemExit):
        water_fill({1: 2, 2: 3}, 6, np.random.default_rng(0))
