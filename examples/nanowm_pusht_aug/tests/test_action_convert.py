"""Pure-logic tests for datagen.action_convert."""

from __future__ import annotations

import numpy as np
from datagen import action_convert as ac


def test_abs_rel_abs_roundtrip() -> None:
    rng = np.random.default_rng(0)
    pos0 = np.array([256.0, 300.0])
    vel0 = np.zeros(2)
    abs_t = pos0 + rng.normal(0, 40, size=(20, 2))
    rel, agent_pos, _ = ac.abs_targets_to_rel(abs_t, pos0, vel0)
    back = ac.rel_to_abs_targets(rel, agent_pos)
    assert np.abs(back - abs_t).max() < 1e-9


def test_rel_to_nanowm_pads_and_shape() -> None:
    short = np.array([[10.0, -5.0], [20.0, 0.0], [30.0, 5.0]])
    win = ac.rel_to_nanowm(short)
    assert win.shape == (4, 10)
    # last row held for padding to num_frames*frame_interval == 20
    rel_back = ac.nanowm_to_rel(win)
    assert len(rel_back) == 20
    assert np.allclose(rel_back[2], short[2])
    assert np.allclose(rel_back[3], short[2])


def test_nanowm_window_roundtrip() -> None:
    rng = np.random.default_rng(1)
    rel = rng.normal(0, 20, size=(20, 2))
    win = ac.rel_to_nanowm(rel)
    assert np.allclose(ac.nanowm_to_rel(win), rel)


def test_pd_step_fixed_point_at_rest() -> None:
    pos = np.array([100.0, 200.0])
    vel = np.zeros(2)
    new_pos, new_vel = ac.pd_step(pos, vel, pos)
    assert np.allclose(new_pos, pos)
    assert np.allclose(new_vel, vel)


def test_abs_targets_to_rel_agent_pos0() -> None:
    pos0 = np.array([12.0, 34.0])
    vel0 = np.zeros(2)
    abs_t = np.array([[50.0, 60.0], [70.0, 80.0]])
    rel, agent_pos, _ = ac.abs_targets_to_rel(abs_t, pos0, vel0)
    assert np.array_equal(agent_pos[0], pos0)
    assert np.allclose(rel[0], abs_t[0] - pos0)
