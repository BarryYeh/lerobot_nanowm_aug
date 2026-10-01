"""`rollout_batch` per-step coverage + perturb_mode, driven by a fake policy (no ckpt / GPU).

The fake policy holds a constant absolute target, so the agent sits still: an env whose block starts
on the goal pose succeeds on step 1 (terminates), an env whose block is far from the goal never does.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from critic.labels import forward_fill_cov  # noqa: E402
from ood_failure_map import sim_state  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

from lerobot.utils.constants import ACTION  # noqa: E402

AGENT = (60.0, 60.0)
TARGET = torch.tensor([60.0, 60.0])


class FakePolicy:
    """Minimal surface `rollout_batch` uses: config.n_action_steps, reset(), _queues[ACTION], select_action."""

    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=0)
        self._queues = {}

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def select_action(self, batch):
        q = self._queues[ACTION]
        if len(q) == 0:
            n = batch["observation.image"].shape[0]
            for _ in range(self.config.n_action_steps):
                q.append(TARGET.expand(n, 2).clone())
        return q.popleft()


def _ident(x):
    return x


def _goal_start() -> np.ndarray:
    """CoG start state with the block exactly on the goal pose (coverage ~ 1)."""
    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.agent.position = AGENT
    u.block.angle = float(u.goal_pose[2])  # angle first: rotating about the CoM would shift the origin
    u.block.position = (float(u.goal_pose[0]), float(u.goal_pose[1]))
    u.space.step(u.dt)
    assert u._get_coverage() > 0.99
    st = sim_state(u)
    env.close()
    return st


def _far_start(x: float = 400.0, y: float = 400.0) -> np.ndarray:
    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.agent.position = AGENT
    u.block.position = (x, y)
    u.block.angle = 0.0
    u.space.step(u.dt)
    st = sim_state(u)
    env.close()
    return st


def _run(starts, steps, **kw):
    kw.setdefault("frame_cutoff_step", 0)
    return rollout_batch(
        FakePolicy(),
        _ident,
        _ident,
        np.asarray(starts),
        device="cpu",
        na=10,
        episode_steps=steps,
        seed=0,
        batch_start=0,
        lockstep224=False,
        **kw,
    )


def test_forward_fill_cov_pure():
    cov = np.array([[0.1, 0.2, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.0, 0.0]], np.float32)
    out = forward_fill_cov(cov, np.array([1, 3, 0]))
    np.testing.assert_allclose(out, [[0.1, 0.2, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]])
    assert out.dtype == cov.dtype
    assert (cov[0, 2:] == 0).all()  # input untouched


def test_cov_per_step_shape_success_and_forward_fill():
    steps = 40
    res = _run([_goal_start(), _far_start()], steps)
    cov = res["cov_per_step"]
    ss = res["success_step"]
    assert cov.shape == (2, steps + 1) and cov.dtype == np.float32
    assert ss[0] == 1 and ss[1] == -1
    assert cov[0, 0] > 0.95  # start-state coverage (index 0) is recorded
    assert cov[0, ss[0]] > 0.95
    assert (cov[0, ss[0] :] == cov[0, ss[0]]).all()  # forward-filled after termination, not zeros
    assert cov[0].min() > 0.95  # no zeros anywhere on the terminated env
    assert cov[1].max() < 0.5  # the far block never covers the goal
    assert res["perturb_mode"] == "point"
    assert res["grid_steps"].tolist() == [0]  # frame grid decoupled from tracking


def test_cov_per_step_early_break_done_all_is_filled():
    # every env succeeds on step 1 -> `done.all()` breaks the loop at t=0; nothing may stay zero
    steps = 30
    res = _run([_goal_start(), _goal_start()], steps)
    cov = res["cov_per_step"]
    assert cov.shape == (2, steps + 1)
    assert (res["success_step"] == 1).all()
    assert (cov > 0.95).all()
    for i in range(2):
        assert cov[i, res["success_step"][i]] > 0.95


def test_cov_per_step_not_limited_by_frame_cutoff():
    res = _run([_far_start()], 25, frame_cutoff_step=10)
    assert res["cov_per_step"].shape == (1, 26)
    assert res["grid_steps"].tolist() == [0, 5, 10]  # grid still follows frame_cutoff_step


class CountingRng:
    """inject_noise only calls rng.standard_normal once per noised chunk -> counts noised chunks."""

    def __init__(self):
        self._rng = np.random.default_rng(0)
        self.calls = 0

    def standard_normal(self, size):
        self.calls += 1
        return self._rng.standard_normal(size)


@pytest.mark.parametrize(
    "mode,p,expected",
    [
        ("point", [0, 10, 30, 1000], [1, 1, 1, 0]),
        ("prefix", [0, 10, 30, 1000], [0, 1, 3, 6]),  # 6 = every boundary of a 60-step episode
    ],
)
def test_perturb_mode_noised_chunk_count(mode, p, expected):
    starts = [_far_start(300 + 10 * k, 400) for k in range(len(p))]
    for i in range(len(p)):  # one env at a time so each CountingRng.calls is attributable
        rng = CountingRng()
        res = _run(
            [starts[i]],
            60,
            sigma=np.array([0.5]),
            perturb_p=np.array([p[i]]),
            noise_hold=1,
            noise_rng=rng,
            perturb_mode=mode,
        )
        assert rng.calls == expected[i], (mode, p[i], rng.calls)
        assert res["perturb_mode"] == mode


def test_clean_chunk_hook_sees_unnoised_chunks():
    seen = []

    def hook(t, idx, chunks, envs):
        seen.append((t, list(idx), chunks.copy()))

    _run(
        [_far_start()],
        30,
        sigma=np.array([0.5]),
        perturb_p=np.array([1000]),
        noise_hold=1,
        noise_rng=np.random.default_rng(1),
        perturb_mode="prefix",
        chunk_hook=hook,
    )
    assert [s[0] for s in seen] == [0, 10, 20]
    for _t, idx, chunks in seen:
        assert idx == [0] and chunks.shape == (1, 10, 2)
        np.testing.assert_allclose(chunks, np.broadcast_to(TARGET.numpy(), (1, 10, 2)))


def test_bad_perturb_mode_raises():
    with pytest.raises(ValueError):
        _run([_far_start()], 5, perturb_mode="suffix")
