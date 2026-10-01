"""Lead-in save / load / replay (O1): a replayed lead-in reproduces the anchor state bit for bit."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from closed_loop_rollout import (  # noqa: E402
    check_lead_in_replay,
    load_lead_in,
    run_lead_in,
    save_lead_in,
)
from nwaug.pusht_state import read_body  # noqa: E402

H, FI = 3, 5
LEAD = (H - 1) * FI


def _envs():
    kw = {"max_episode_steps": LEAD + 40}
    e96 = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96, observation_height=96, **kw
    )
    e224 = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels", observation_width=224, observation_height=224, **kw
    )
    return e96, e224


def _states(n=2):
    e = gym.make("gym_pusht/PushT-v0")
    out = []
    for i in range(n):
        e.reset(seed=500 + i)
        out.append(read_body(e.unwrapped))
    e.close()
    return np.stack(out)


class FakePolicyActions:
    """Stand-in for the diffusion policy: random targets drawn from its OWN rng (the 'torch RNG')."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, i, t, pix):
        assert pix.shape == (96, 96, 3)
        self.calls += 1
        return self.rng.uniform(100, 400, size=2)


@pytest.fixture()
def first_run(tmp_path):
    states = _states()
    e96, e224 = _envs()
    pol = FakePolicyActions(0)
    res = run_lead_in(e96, e224, states, pol, hist=H, fi=FI, lead=LEAD)
    e96.close()
    e224.close()
    assert pol.calls == len(states) * LEAD
    path = tmp_path / "lead_in.npz"
    save_lead_in(path, res, states, hist=H, fi=FI, lead_na=10, torch_seed=740000)
    return states, res, path


def test_saved_file_contents_round_trip(first_run):
    states, res, path = first_run
    d = load_lead_in(path, states, hist=H, fi=FI)
    assert np.array_equal(d["lead_acts"], res["lead_acts"])
    assert np.array_equal(d["anchor_state"], res["anchor_state"])
    assert np.array_equal(d["agent_state0"][:, :2], res["lead_pos"][:, LEAD])
    assert np.array_equal(d["agent_state0"][:, 2:], res["lead_vel"][:, LEAD])
    assert d["seed_states"].shape == (2, 5) and d["ctx_sha1"].shape == (2,)
    assert (int(d["fi"]), int(d["hist"]), int(d["lead_na"]), int(d["torch_seed"])) == (FI, H, 10, 740000)


def test_replay_yields_identical_anchor_state_without_calling_the_policy(first_run):
    states, res, path = first_run
    saved = load_lead_in(path, states, hist=H, fi=FI)

    def forbidden(i, t, pix):
        raise AssertionError("the policy must not be called when replaying a lead-in")

    del forbidden  # replay act_fn is the saved actions only
    e96, e224 = _envs()
    res2 = run_lead_in(
        e96, e224, states, lambda i, t, pix: saved["lead_acts"][i, t], hist=H, fi=FI, lead=LEAD
    )
    e96.close()
    e224.close()
    check_lead_in_replay(saved, res2)
    assert np.array_equal(res2["anchor_state"], res["anchor_state"])  # bit-identical
    assert np.array_equal(res2["ctx"], res["ctx"])
    assert np.array_equal(res2["last96"], res["last96"])


def test_replay_check_detects_a_different_lead_in(first_run):
    states, _res, path = first_run
    saved = load_lead_in(path, states, hist=H, fi=FI)
    e96, e224 = _envs()
    other = run_lead_in(
        e96, e224, states, FakePolicyActions(1), hist=H, fi=FI, lead=LEAD
    )  # other "torch seed"
    e96.close()
    e224.close()
    with pytest.raises(AssertionError):
        check_lead_in_replay(saved, other)


def test_load_rejects_other_seeds_or_geometry(first_run):
    states, _res, path = first_run
    with pytest.raises(AssertionError):
        load_lead_in(path, states[::-1].copy(), hist=H, fi=FI)
    with pytest.raises(AssertionError):
        load_lead_in(path, states, hist=H, fi=4)
    with pytest.raises(AssertionError):
        load_lead_in(path, states[:1], hist=H, fi=FI)
