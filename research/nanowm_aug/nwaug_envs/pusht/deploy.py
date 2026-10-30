"""Deploy an image-only policy in PushT from env.reset(seed) (lerobot venv only: imports gym_pusht and torch).

Each episode runs a 96px policy env and a 224px camera env in lockstep (same seed, same actions); the CoG of both is
checked on the 5-step grid. Shared by imagined_replay (near-miss starts, branches) and early_anchors (anchor episodes).

Conventions: F(t) = 224px frame after t executed steps (F(0) = right after reset); a[t] = the action executed at
step t, taking F(t) to F(t+1); success = number of executed steps when info["is_success"].
"""

from __future__ import annotations

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from nwaug_envs.pusht.state import read_cog

FI = 5  # NanoWM frame interval == grid stride of the recorded frames
EPISODE_STEPS = 300


def make_envs(n: int) -> tuple[list, list]:
    kw = {"max_episode_steps": EPISODE_STEPS + 10}
    e96 = [
        gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            observation_width=96,
            observation_height=96,
            **kw,
        )
        for _ in range(n)
    ]
    e224 = [
        gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=224, observation_height=224, **kw)
        for _ in range(n)
    ]
    return e96, e224


def close(*env_lists) -> None:
    for envs in env_lists:
        for e in envs:
            e.close()


def policy_act(policy, pre, post, obs96: list, device: str) -> np.ndarray:
    img = torch.from_numpy(np.stack(obs96)).permute(0, 3, 1, 2).float().div(255.0).to(device)
    with torch.no_grad():
        return post(policy.select_action(pre({"observation.image": img}))).cpu().numpy().astype(np.float64)


def reset_policy(policy, torch_seed: int, na: int) -> None:
    torch.manual_seed(torch_seed)
    policy.config.n_action_steps = na
    policy.reset()


def source_batch(policy, pre, post, env_seeds: list[int], device: str, torch_seed: int, na: int) -> dict:
    """Deploy from env.reset(seed); record every action, the 224px grid frames and the CoG on the grid."""
    nb = len(env_seeds)
    e96, e224 = make_envs(nb)
    obs96 = []
    for i, s in enumerate(env_seeds):
        o, _ = e96[i].reset(seed=int(s))
        e224[i].reset(seed=int(s))
        obs96.append(o["pixels"])
    steps = np.arange(0, EPISODE_STEPS + 1, FI)
    frames = np.zeros((nb, len(steps), 224, 224, 3), np.uint8)
    cog = np.zeros((nb, len(steps), 5))
    acts = np.zeros((nb, EPISODE_STEPS, 2))
    success = np.full(nb, -1, np.int64)
    end = np.full(nb, EPISODE_STEPS, np.int64)
    done = np.zeros(nb, bool)
    for i in range(nb):
        frames[i, 0] = e224[i].unwrapped.get_obs()
        cog[i, 0] = read_cog(e96[i].unwrapped)
    reset_policy(policy, torch_seed, na)
    for t in range(EPISODE_STEPS):
        act = policy_act(policy, pre, post, obs96, device)
        for i in range(nb):
            if done[i]:
                continue
            o, _r, term, trunc, info = e96[i].step(act[i])
            e224[i].step(act[i])
            obs96[i] = o["pixels"]
            acts[i, t] = act[i]
            if (t + 1) % FI == 0:
                g = (t + 1) // FI
                frames[i, g] = e224[i].unwrapped.get_obs()
                cog[i, g] = read_cog(e96[i].unwrapped)
                assert np.allclose(cog[i, g], read_cog(e224[i].unwrapped), atol=1e-9), (
                    f"lockstep drift {i} t={t + 1}"
                )
            if term or trunc:
                done[i], end[i] = True, t + 1
                if bool(info.get("is_success", False)):
                    success[i] = t + 1
        if done.all():
            break
    close(e96, e224)
    return {"steps": steps, "frames": frames, "cog": cog, "acts": acts, "success": success, "end": end}
