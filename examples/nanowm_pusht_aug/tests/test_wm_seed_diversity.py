"""eval/wm_seed_diversity.py: replayed windows obey the closed-loop alignment contract; metric helpers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("gym_pusht")
import wm_seed_diversity as wsd  # noqa: E402
from datagen import action_convert as ac  # noqa: E402
from datagen.coverage_from_frame import T_FILL_RGB, T_OUTLINE_RGB  # noqa: E402


def test_replay_windows_match_closed_loop_formula() -> None:
    """With an agent that follows the PD model exactly, the exact-rel replay window of round r equals the
    closed loop's `abs_targets_to_rel(stream[r*fi : r*fi+nf*fi], apos[r*fi], avel[r*fi])` window."""
    rng = np.random.default_rng(0)
    R = 8
    T = wsd.stream_len(R)
    assert T == (R - 1) * wsd.FI + wsd.NF * wsd.FI
    acts = 256 + rng.normal(0, 40, (T, 2))
    _, apos, avel = ac.abs_targets_to_rel(acts, np.array([250.0, 260.0]), np.array([1.0, -2.0]))
    win = wsd.replay_windows(acts, apos, R)
    assert win.shape == (R, wsd.NF, 2 * wsd.FI)
    need = wsd.NF * wsd.FI
    for r in range(R):
        k0 = r * wsd.FI
        rel, _, _ = ac.abs_targets_to_rel(acts[k0 : k0 + need], apos[k0], avel[k0])
        np.testing.assert_allclose(win[r], ac.rel_to_nanowm(rel), rtol=1e-5, atol=1e-5)
    with pytest.raises(AssertionError):
        wsd.replay_windows(acts[:-1], apos[:-1], R)


def test_stat_helpers() -> None:
    assert wsd.in_range(np.array([1.0, 3.0, np.nan]), 2.0) == 1.0
    assert wsd.in_range(np.array([1.0, 3.0]), 4.0) == 0.0
    assert np.isnan(wsd.in_range(np.array([1.0]), 1.0))
    assert wsd.circ_std_deg(np.array([0.3, 0.3, 0.3])) == pytest.approx(0.0, abs=1e-4)
    # wrap-around: angles just either side of +-pi are close, not 360 deg apart
    assert wsd.circ_std_deg(np.array([np.pi - 0.01, -np.pi + 0.01])) < 1.0
    m, lo, hi = wsd.boot_ci(np.array([1.0, 2.0, 3.0, np.nan]))
    assert m == pytest.approx(2.0) and lo <= m <= hi


def _block_frame() -> np.ndarray:
    img = np.full((224, 224, 3), 255, np.uint8)
    cv2.rectangle(img, (70, 90), (150, 106), T_FILL_RGB, -1)
    cv2.rectangle(img, (102, 106), (118, 160), T_FILL_RGB, -1)
    cv2.rectangle(img, (70, 90), (150, 106), T_OUTLINE_RGB, 1)
    return img


def test_sharpness_orders_blur() -> None:
    crisp = _block_frame()
    blurry = cv2.GaussianBlur(crisp, (0, 0), 2.0)
    s_c, s_b = wsd.sharpness(crisp), wsd.sharpness(blurry)
    assert np.isfinite(s_c).all() and np.isfinite(s_b).all()
    assert s_b[0] < s_c[0] and s_b[1] < s_c[1], (s_c, s_b)  # laplacian variance drops
    assert s_b[2] > s_c[2], (s_c, s_b)  # edge width grows
    assert np.isnan(wsd.sharpness(np.full((224, 224, 3), 255, np.uint8))[1])  # no block -> nan
