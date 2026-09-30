"""Pure-logic tests for datagen.coverage_from_frame (size 224 only)."""

from __future__ import annotations

import numpy as np
from datagen.coverage_from_frame import (
    T_FILL_RGB,
    block_mask,
    calibrate_coverage,
    coverage_from_frame,
    goal_mask,
)


def test_blank_frame_zero_coverage() -> None:
    blank = np.zeros((224, 224, 3), np.uint8)
    d = coverage_from_frame(blank)
    assert d["coverage"] == 0.0
    assert block_mask(blank).sum() == 0


def test_painted_goal_near_unit_coverage() -> None:
    frame = np.zeros((224, 224, 3), np.uint8)
    gm = goal_mask(224)
    frame[gm.astype(bool)] = np.array(T_FILL_RGB, np.uint8)
    d = coverage_from_frame(frame)
    assert d["coverage"] > 0.95


def test_calibrate_coverage_monotone_and_clipped() -> None:
    xs = np.linspace(0.0, 1.0, 101)
    ys = np.array([calibrate_coverage(float(x)) for x in xs])
    assert np.all(ys >= 0.0) and np.all(ys <= 1.0)
    assert np.all(np.diff(ys) >= -1e-15)
