"""Chunk-selection rule of rollout_batch's perturbation (critic/labels.py::chunks_active), offline."""

from __future__ import annotations

import numpy as np
import pytest
from critic.labels import chunks_active

EP = 300
P = np.array([0, 10, 30, 100, 300, 1000])
SIGMA = np.full(len(P), 0.1)
ALIVE = np.zeros(len(P), bool)


def test_point_selects_only_p_equal_t():
    for t in range(0, EP, 10):
        assert chunks_active(t, ALIVE, SIGMA, P, "point") == [int(i) for i in np.nonzero(t == P)[0]]
    assert chunks_active(0, ALIVE, SIGMA, P, "point") == [0]
    assert chunks_active(10, ALIVE, SIGMA, P, "point") == [1]
    assert chunks_active(20, ALIVE, SIGMA, P, "point") == []


def test_prefix_selects_all_t_less_than_p():
    for t in range(0, EP, 10):
        assert chunks_active(t, ALIVE, SIGMA, P, "prefix") == [int(i) for i in np.nonzero(t < P)[0]]
    assert chunks_active(0, ALIVE, SIGMA, P, "prefix") == [1, 2, 3, 4, 5]  # p=0 never selected
    assert chunks_active(10, ALIVE, SIGMA, P, "prefix") == [2, 3, 4, 5]
    assert chunks_active(30, ALIVE, SIGMA, P, "prefix") == [3, 4, 5]


def test_prefix_p_zero_selects_nothing():
    p0 = np.zeros(4, np.int64)
    for t in range(0, EP, 10):
        assert chunks_active(t, np.zeros(4, bool), np.full(4, 0.1), p0, "prefix") == []


def test_prefix_p_at_least_episode_steps_selects_every_boundary():
    p = np.array([EP, EP + 10])
    for t in range(0, EP, 10):  # every replan boundary of a 300-step episode
        assert chunks_active(t, np.zeros(2, bool), np.full(2, 0.1), p, "prefix") == [0, 1]


def test_total_noised_chunk_count():
    # p = 30 -> boundaries 0, 10, 20 -> 3 chunks; p=300 -> 30 chunks
    for p, n in ((0, 0), (10, 1), (30, 3), (300, 30)):
        cnt = sum(
            len(chunks_active(t, np.zeros(1, bool), np.array([0.1]), np.array([p]), "prefix"))
            for t in range(0, EP, 10)
        )
        assert cnt == n


@pytest.mark.parametrize("mode", ["point", "prefix"])
def test_done_and_zero_sigma_never_selected(mode):
    p = np.array([100, 100, 100])
    done = np.array([True, False, False])
    sigma = np.array([0.1, 0.1, 0.0])
    t = 100 if mode == "point" else 50
    assert chunks_active(t, done, sigma, p, mode) == [1]


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        chunks_active(0, ALIVE, SIGMA, P, "suffix")
