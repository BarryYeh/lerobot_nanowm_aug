#!/usr/bin/env python
"""Batched base-policy rollout in gym_pusht from CoG start states -- the reusable core shared by
the stage-1 value-state collector (collect_value_states.py) and, later, the stage-2 imagined-branch
labeler (label_imagined_branches.py).

Follows eval/band_eval.py:77-112 for the batched loop shape (env list, `set_state`, batched
`select_action`, success via `info["is_success"]`, wall check via `block_wall_gap`) and
eval/closed_loop_rollout.py:183-246 for the lockstep 96px/224px env pair and its EXACT grid-frame
capture timing: a 224px frame is captured with `u224.get_obs()` at the top of every 5-step
boundary, BEFORE that boundary's action is computed or stepped. Grid index 0 is therefore the
frame right after `set_state`, with no action executed yet (so it carries no red action marker --
gym_pusht only draws that marker on an explicit `.render()` call, never on `get_obs()`); grid index
`g` (g>0) is captured after exactly `g*frame_grid_stride` real steps, using the SAME action that
just produced that state, matching closed_loop_rollout.py's own context frames bit for bit.

State convention: every state this module takes or returns is CoG-based (5,): [agent x, agent y,
CoG x, CoG y, theta] -- see `set_state` / `sim_state` in ood_failure_map.py. Body-origin
states are never used here.

Not a script -- imported by critic/collect_value_states.py:
    from sim_rollout import rollout_batch
"""

from __future__ import annotations

import sys
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

_EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(_EVAL))
from closed_loop_rollout import inject_noise  # noqa: E402
from ood_failure_map import block_wall_gap, set_state, sim_state  # noqa: E402

from lerobot.utils.constants import ACTION  # noqa: E402


def rollout_batch(
    policy,
    pre,
    post,
    cog_starts: np.ndarray,
    *,
    device: str,
    na: int,
    episode_steps: int,
    seed: int,
    batch_start: int,
    sigma: np.ndarray | None = None,
    perturb_p: np.ndarray | None = None,
    max_z: float = 3.0,
    noise_hold: int = 5,
    noise_rng: np.random.Generator | None = None,
    lockstep224: bool = True,
    frame_grid_stride: int = 5,
    frame_cutoff_step: int = 110,
    wall_pen_px: float = 10.0,
    prepared_envs: list | None = None,
) -> dict:
    """Roll `nb = len(cog_starts)` envs in lockstep from CoG start states, batched per step.

    Perturbation (optional, one chunk per env): at the replan boundary `t == perturb_p[i]` (a
    multiple of `na`), env i's next `na` actions -- which the base policy has already committed to
    in its action queue for this chunk -- are replaced by `inject_noise` applied to that exact
    chunk, using the sim's own agent pos/vel at the boundary. No other step of the rollout is
    touched: the returned action for every other step, and everything the policy itself computes
    (its future replanning, which conditions on the resulting -- now perturbed -- image), is
    exactly the unmodified base policy. This is implemented by overriding only the NUMPY action
    actually sent to `env.step` for envs/steps in the perturbed window; `policy._queues[ACTION]`
    is read (via `post`) to recover the chunk's un-executed steps in real pixel units, but is never
    written back to, so the policy's internal state and later behaviour are provably unaffected.

    Prepared envs (optional, used by the stage-2 labeler): if `prepared_envs` is given (a list of
    gym_pusht 96px `pixels_agent_pos` envs, one per row of `cog_starts`, ALREADY reset and advanced to
    the state to continue from), no env is created, reset or `set_state`d -- the rollout simply
    continues from wherever each env is. `cog_starts` is then unused beyond its length (pass the
    envs' current CoG states). Requires `lockstep224=False`; the caller keeps ownership of the envs
    (they are not closed here), and each env's own TimeLimit must cover `episode_steps` more steps.
    `success_step` counts steps from this call's start. Grid entries are then recorded at
    `frame_cutoff_step`-limited steps relative to the continuation start (grid index 0 = the
    prepared state).

    Returns a dict with, for every env `i` in `range(nb)`:
      success_step   (nb,) int64      first step index (1-based) where `info["is_success"]`, else -1
      wall_artifact  (nb,) bool       True iff min block_wall_gap over the whole rollout < -wall_pen_px
      wall_gap_min   (nb,) float32
      grid_steps     (G,)  int64      the recorded step indices, 0, frame_grid_stride, ..., frame_cutoff_step
      grid_cog       (nb,G,5)         CoG states on the grid
      grid_agent_pos (nb,G,2)
      grid_agent_vel (nb,G,2)
      grid_coverage  (nb,G) float32   sim coverage on the grid
      grid_frames    (nb,G,224,224,3) uint8, or None if lockstep224=False
      queue_maxlen   int              policy._queues[ACTION].maxlen after the na override (must == na)
      lockstep_checks int             number of (env, grid step) pairs where 96px/224px sim_state
                                       was compared and found equal within 1e-9 (only if lockstep224)
    """
    cog_starts = np.asarray(cog_starts, np.float64)
    nb = len(cog_starts)
    if sigma is None:
        sigma = np.zeros(nb)
    if perturb_p is None:
        perturb_p = np.zeros(nb, dtype=np.int64)
    if noise_rng is None:
        noise_rng = np.random.default_rng(0)
    sigma = np.asarray(sigma, np.float64)
    perturb_p = np.asarray(perturb_p, np.int64)

    max_ep_steps = episode_steps + 10  # margin so gym's own TimeLimit never truncates us early
    prepared = prepared_envs is not None
    if prepared:
        assert not lockstep224, "prepared_envs requires lockstep224=False"
        assert len(prepared_envs) == nb, (len(prepared_envs), nb)
        envs96 = list(prepared_envs)
    else:
        envs96 = [
            gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                observation_width=96,
                observation_height=96,
                max_episode_steps=max_ep_steps,
            )
            for _ in range(nb)
        ]
    envs224 = None
    if lockstep224:
        envs224 = [
            gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels",
                observation_width=224,
                observation_height=224,
                max_episode_steps=max_ep_steps,
            )
            for _ in range(nb)
        ]

    obs96 = []
    for i in range(nb):
        if prepared:
            obs96.append(envs96[i].unwrapped.get_obs()["pixels"])
            continue
        envs96[i].reset(seed=0)
        o = set_state(envs96[i].unwrapped, cog_starts[i])
        obs96.append(o["pixels"])
        if lockstep224:
            envs224[i].reset(seed=0)
            set_state(envs224[i].unwrapped, cog_starts[i])

    grid_steps = list(range(0, frame_cutoff_step + 1, frame_grid_stride))
    G = len(grid_steps)
    grid_idx_of_step = {s: gi for gi, s in enumerate(grid_steps)}

    grid_cog = np.zeros((nb, G, 5), np.float64)
    grid_agent_pos = np.zeros((nb, G, 2), np.float64)
    grid_agent_vel = np.zeros((nb, G, 2), np.float64)
    grid_coverage = np.zeros((nb, G), np.float32)
    grid_frames = np.zeros((nb, G, 224, 224, 3), np.uint8) if lockstep224 else None

    wall_min = np.full(nb, np.inf, np.float64)
    success_step = np.full(nb, -1, np.int64)
    done = np.zeros(nb, bool)
    lockstep_checks = 0

    # grid index 0: right after set_state, before any action -- no red action marker possible.
    for i in range(nb):
        u = envs96[i].unwrapped
        st = sim_state(u)
        grid_cog[i, 0] = st
        grid_agent_pos[i, 0] = np.asarray(u.agent.position)
        grid_agent_vel[i, 0] = np.asarray(u.agent.velocity)
        grid_coverage[i, 0] = u._get_coverage()
        wall_min[i] = min(wall_min[i], float(block_wall_gap(st[None])[0]))
        if lockstep224:
            grid_frames[i, 0] = np.asarray(envs224[i].unwrapped.get_obs())
            st224 = sim_state(envs224[i].unwrapped)
            assert np.allclose(st, st224, atol=1e-9), f"lockstep drift env {i} at grid step 0"
            lockstep_checks += 1

    torch.manual_seed(seed + batch_start)
    policy.config.n_action_steps = na
    policy.reset()
    queue_maxlen = policy._queues[ACTION].maxlen
    assert queue_maxlen == na, f"expected action-queue maxlen {na}, got {queue_maxlen}"

    pending: dict[int, tuple[int, np.ndarray]] = {}  # env i -> (boundary step p, noised chunk (na,2))

    for t in range(episode_steps):
        img = torch.from_numpy(np.stack(obs96)).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.no_grad():
            raw = policy.select_action(pre({"observation.image": img}))
            act = post(raw).cpu().numpy().astype(np.float64)

        if t % na == 0:
            active = [i for i in range(nb) if (not done[i]) and sigma[i] > 0 and perturb_p[i] == t]
            if active:
                queue = policy._queues[ACTION]  # na-1 entries left: chunk steps 1..na-1, all envs
                if len(queue) > 0:
                    with torch.no_grad():
                        future_real = post(torch.stack(list(queue), dim=0)).cpu().numpy().astype(np.float64)
                else:
                    future_real = np.zeros((0, nb, 2))
                for i in active:
                    fut = future_real[:, i] if future_real.shape[0] else np.zeros((0, 2))
                    chunk = np.concatenate([act[i : i + 1], fut], axis=0)
                    if len(chunk) < na:  # defensive: hold the last row if the queue was short
                        chunk = np.concatenate(
                            [chunk, np.repeat(chunk[-1:], na - len(chunk), axis=0)], axis=0
                        )
                    u = envs96[i].unwrapped
                    pos_i = np.asarray(u.agent.position, np.float64)[None]
                    vel_i = np.asarray(u.agent.velocity, np.float64)[None]
                    noised = inject_noise(
                        chunk[None], pos_i, vel_i, float(sigma[i]), max_z, noise_hold, noise_rng
                    )
                    pending[i] = (t, noised[0])

        for i, (p0, noised) in pending.items():
            if not done[i] and p0 <= t < p0 + na:
                act[i] = noised[t - p0]

        for i in range(nb):
            if done[i]:
                continue
            o, _r, term, trunc, info = envs96[i].step(act[i])
            obs96[i] = o["pixels"]
            u = envs96[i].unwrapped
            if lockstep224 and (t + 1) <= frame_cutoff_step:
                envs224[i].step(act[i])
            st = sim_state(u)
            wall_min[i] = min(wall_min[i], float(block_wall_gap(st[None])[0]))
            if (t + 1) in grid_idx_of_step:
                gi = grid_idx_of_step[t + 1]
                grid_cog[i, gi] = st
                grid_agent_pos[i, gi] = np.asarray(u.agent.position)
                grid_agent_vel[i, gi] = np.asarray(u.agent.velocity)
                grid_coverage[i, gi] = info["coverage"]
                if lockstep224 and (t + 1) <= frame_cutoff_step:
                    grid_frames[i, gi] = np.asarray(envs224[i].unwrapped.get_obs())
                    st224 = sim_state(envs224[i].unwrapped)
                    assert np.allclose(st, st224, atol=1e-9), f"lockstep drift env {i} at grid step {t + 1}"
                    lockstep_checks += 1
            if term or trunc:
                done[i] = True
                if bool(info.get("is_success", False)):
                    success_step[i] = t + 1

        for i in list(pending):
            if t == pending[i][0] + na - 1:
                del pending[i]
        if done.all():
            break

    if not prepared:
        for e in envs96:
            e.close()
    if lockstep224:
        for e in envs224:
            e.close()

    return {
        "success_step": success_step,
        "wall_artifact": wall_min < -wall_pen_px,
        "wall_gap_min": wall_min.astype(np.float32),
        "grid_steps": np.asarray(grid_steps, np.int64),
        "grid_cog": grid_cog,
        "grid_agent_pos": grid_agent_pos,
        "grid_agent_vel": grid_agent_vel,
        "grid_coverage": grid_coverage,
        "grid_frames": grid_frames,
        "queue_maxlen": queue_maxlen,
        "lockstep_checks": lockstep_checks,
    }
