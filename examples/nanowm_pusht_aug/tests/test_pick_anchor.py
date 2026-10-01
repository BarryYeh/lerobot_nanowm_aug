"""`labels.pick_anchor` (one anchor per source episode) and `labels.draw_perturbations` switches."""

from __future__ import annotations

import numpy as np
import pytest
from critic.labels import QUERY_KEEPS, draw_perturbations, keep_query, pick_anchor

TS = list(range(10, 101, 10))


def test_always_consumes_exactly_one_draw():
    for kw in (
        {"success_step": -1, "wall_artifact": False, "perturb_p": 300, "sigma": 0.1},
        {
            "success_step": 5,
            "wall_artifact": False,
            "perturb_p": 300,
            "sigma": 0.1,
        },  # rejected: succeeded early
        {"success_step": -1, "wall_artifact": True, "perturb_p": 300, "sigma": 0.1},  # rejected: wall
    ):
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        pick_anchor(t_choices=TS, rng=a, **kw)
        b.choice(np.asarray(TS))
        assert a.integers(1 << 30) == b.integers(1 << 30)


def test_returns_t_from_choices_and_is_deterministic():
    out = [pick_anchor(-1, False, TS, 300, 0.1, np.random.default_rng(3), "all") for _ in range(2)]
    assert out[0] == out[1] and out[0] in TS


def test_rejects_wall_artifact():
    assert pick_anchor(-1, True, TS, 300, 0.1, np.random.default_rng(0), "all") is None


def test_rejects_episode_already_successful_by_t_but_keeps_later_success():
    rng_t = int(np.random.default_rng(1).choice(np.asarray(TS)))
    for s, expect_none in ((rng_t - 1, True), (rng_t, True), (rng_t + 1, False), (-1, False)):
        out = pick_anchor(s, False, TS, 300, 0.1, np.random.default_rng(1), "all")
        assert (out is None) == expect_none, (s, rng_t, out)


def test_honours_prefix_rule_for_after_switch():
    rng_t = int(np.random.default_rng(2).choice(np.asarray(TS)))
    # p > t: the query is inside the noised prefix -> dropped; p <= t or sigma == 0 -> kept
    assert pick_anchor(-1, False, TS, rng_t + 10, 0.1, np.random.default_rng(2)) is None
    assert pick_anchor(-1, False, TS, rng_t, 0.1, np.random.default_rng(2)) == rng_t
    assert pick_anchor(-1, False, TS, rng_t + 10, 0.0, np.random.default_rng(2)) == rng_t


def test_query_keep_all_drops_nothing_even_with_p_equal_episode_steps():
    # keep_query is NOT vacuous for p = episode_steps (it would reject every anchor) -> rule is explicit
    assert not keep_query(50, 300, 0.1)
    for seed in range(5):
        assert pick_anchor(-1, False, TS, 300, 0.1, np.random.default_rng(seed), "after_switch") is None
        assert pick_anchor(-1, False, TS, 300, 0.1, np.random.default_rng(seed), "all") in TS
    with pytest.raises(ValueError):
        pick_anchor(-1, False, TS, 300, 0.1, np.random.default_rng(0), "nope")


def test_draw_perturbations_switch_all_vs_random():
    pr = {"sigmas": [0.1], "probs": [1.0], "mode": "prefix", "seed": 5}
    s_all, p_all = draw_perturbations({**pr, "switch": "all"}, 50, 300)
    s_rnd, p_rnd = draw_perturbations({**pr, "switch": "random"}, 50, 300)
    assert (p_all == 300).all() and (s_all == 0.1).all()
    assert set(p_rnd.tolist()) <= set(range(0, 101, 10)) and len(set(p_rnd.tolist())) > 3
    assert np.array_equal(s_all, s_rnd)  # the rng stream (hence sigma) is identical for both switches
    # default switch == random, legacy configs (no mode / switch) keep their old draws
    s_leg, p_leg = draw_perturbations(
        {"sigmas": [0.0, 0.5, 1.0], "probs": [0.4, 0.3, 0.3], "seed": 0}, 40, 310
    )
    rng = np.random.default_rng(0)
    assert np.array_equal(s_leg, rng.choice(np.array([0.0, 0.5, 1.0]), size=40, p=[0.4, 0.3, 0.3]))
    assert np.array_equal(p_leg, rng.choice(np.arange(0, 101, 10), size=40))


def test_draw_perturbations_rejects_bad_config():
    with pytest.raises(ValueError):
        draw_perturbations(
            {"sigmas": [0.1], "probs": [1.0], "seed": 0, "switch": "all", "mode": "point"}, 3, 300
        )
    with pytest.raises(ValueError):
        draw_perturbations({"sigmas": [0.1], "probs": [1.0], "seed": 0, "switch": "nope"}, 3, 300)
    assert set(QUERY_KEEPS) == {"all", "after_switch"}
