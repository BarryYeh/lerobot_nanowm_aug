#!/usr/bin/env python
"""Record policy rollouts in gym_pusht as CUPID's test set (Stage 2): one row per policy QUERY.

A thin wrapper around lerobot-eval's own `rollout()` (same env, processors, seeding: episode i of
batch b uses seed `rollout.seed + b*batch_size + i`), with two hooks on the diffusion model:
  * generate_actions  -> the normalized image the policy was actually conditioned on
  * conditional_sample -> the FULL normalized horizon (16 steps). generate_actions slices it down to
                          n_action_steps, but the influence gradient needs the training-shaped chunk
Both are already in the normalized space the gradient code works in, so no re-normalization.

Queries happen for the whole vector batch at steps 0, na, 2na, ...; a query is kept for env i only
if it happened before that env's episode ended (all envs keep stepping until the last is done).

Output <out_dir>/test_rollouts.npz:
  images (Q,C,H,W) f32, actions (Q,horizon,A) f32, episode (Q,), query_step (Q,)
  ep_seed, ep_success, ep_max_reward, ep_len, ep_n_queries  (per episode)

    uv run --extra training python examples/nanowm_pusht_aug/eval/record_policy_rollouts.py \
        --config configs/cupid_filter_untrimmed464.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.random_utils import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attribution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_failure_map import block_wall_gap, set_state, sim_state  # noqa: E402
from policy_io import load_config, load_policy  # noqa: E402


def start_from_states(env, cur: dict) -> None:
    """Make the vector env start every episode from cur["states"] (CoG-based, one per sub-env)
    instead of gym_pusht's random reset, and log each sub-env's sim state after every step.

    lerobot's rollout() still calls env.reset(seed=...) itself; this wraps that call, places each
    sub-env with set_state() (angle before position, body origin from the CoG) and returns the
    observation of the PLACED state -- the one the policy's first query must see."""
    orig_reset, orig_step = env.reset, env.step

    def reset(*a, **kw):
        _, info = orig_reset(*a, **kw)
        subs = [e.unwrapped for e in env.envs]
        per = [set_state(u, st) for u, st in zip(subs, cur["states"], strict=True)]
        cur["traj"] = [[sim_state(u) for u in subs]]
        return {k: np.stack([np.asarray(o[k]) for o in per]) for k in per[0]}, info

    def step(action):
        out = orig_step(action)
        cur["traj"].append([sim_state(e.unwrapped) for e in env.envs])
        return out

    env.reset, env.step = reset, step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    r = cfg["rollout"]
    assert r["seed"] >= cfg["eval"]["seed"] + cfg["eval"]["n_episodes"] or (
        r["seed"] + r["n_episodes"] <= cfg["eval"]["seed"]
    ), "rollout seeds overlap the final-eval seeds: selection would see eval episodes"
    assert r["n_action_steps"] == cfg["eval"]["n_action_steps"], (
        "test queries must sit where eval queries sit"
    )

    torch.backends.cudnn.benchmark = True  # as lerobot-eval
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(r["seed"])
    policy, pre, post = load_policy(cfg["ckpt"])
    policy.config.n_action_steps = r["n_action_steps"]
    env_cfg = PushtEnv()
    env = next(
        iter(next(iter(make_env(env_cfg, n_envs=r["batch_size"], use_async_envs=False).values())).values())
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    # Optional: start from given states (CUPID add-one-in tests band recovery). Episodes whose block
    # is pushed through a wall (a sim artifact) are dropped, so their labels never reach the scores.
    starts, cur, wall_pen = None, {}, None
    if r.get("start_states"):
        starts = np.load(Path(cfg["out_dir"]) / r["start_states"])["cog_states"]
        assert len(starts) >= r["n_episodes"], f"{len(starts)} start states < n_episodes {r['n_episodes']}"
        wall_pen = cfg["topband"]["wall_pen_px"]
        start_from_states(env, cur)
        print(
            f"starting from {r['start_states']} ({len(starts)} states); wall artifact > {wall_pen} px dropped"
        )

    queries: list[tuple[np.ndarray, np.ndarray]] = []
    model = policy.diffusion
    orig_gen, orig_sample = model.generate_actions, model.conditional_sample
    pending = {}

    def generate_actions(batch, noise=None):
        pending["img"] = batch[OBS_IMAGES][:, -1, 0].float().cpu().numpy()  # (B,C,H,W), n_obs=1, 1 cam
        out = orig_gen(batch, noise)
        queries.append((pending.pop("img"), pending.pop("act")))
        return out

    def conditional_sample(*a, **kw):
        full = orig_sample(*a, **kw)
        pending["act"] = full.float().cpu().numpy()  # (B,horizon,A) before the n_action_steps slice
        return full

    model.generate_actions, model.conditional_sample = generate_actions, conditional_sample

    na, bs = r["n_action_steps"], r["batch_size"]
    imgs, acts, ep_of, qstep = [], [], [], []
    ep_seed, ep_succ, ep_maxr, ep_len, ep_nq, ep_start = [], [], [], [], [], []
    n_artifact = 0
    for b in range((r["n_episodes"] + bs - 1) // bs):
        seeds = list(range(r["seed"] + b * bs, r["seed"] + (b + 1) * bs))
        if starts is not None:
            idx = np.arange(b * bs, (b + 1) * bs) % len(starts)  # the last batch may wrap; not kept
            cur["states"] = starts[idx]
        queries.clear()
        with torch.no_grad():
            data = rollout(env, policy, env_pre, env_post, pre, post, seeds=seeds)
        done = data["done"].numpy()
        traj = np.asarray(cur["traj"]) if starts is not None else None  # (steps+1, bs, 5)
        for i in range(bs):
            ep = b * bs + i
            if ep >= r["n_episodes"]:
                break
            last = int(done[i].argmax()) if done[i].any() else done.shape[1] - 1
            if traj is not None and block_wall_gap(traj[: last + 2, i]).min() < -wall_pen:
                n_artifact += 1
                continue
            kept = [k for k in range(len(queries)) if k * na <= last]
            for k in kept:
                imgs.append(queries[k][0][i])
                acts.append(queries[k][1][i])
                ep_of.append(len(ep_succ))  # contiguous over KEPT episodes
                qstep.append(k * na)
            ep_start.append(ep)
            ep_seed.append(seeds[i])
            ep_succ.append(bool(data["success"][i, : last + 1].any()))
            ep_maxr.append(float(data["reward"][i, : last + 1].max()))
            ep_len.append(last + 1)
            ep_nq.append(len(kept))
        print(
            f"batch {b}: {sum(ep_succ[-bs:])}/{min(bs, len(ep_succ) - b * bs)} success, {len(imgs)} queries so far"
        )

    out = Path(cfg["out_dir"]) / "test_rollouts.npz"
    np.savez(
        out,
        images=np.stack(imgs).astype(np.float32),
        actions=np.stack(acts).astype(np.float32),
        episode=np.array(ep_of),
        query_step=np.array(qstep),
        ep_seed=np.array(ep_seed),
        ep_success=np.array(ep_succ),
        ep_max_reward=np.array(ep_maxr),
        ep_len=np.array(ep_len),
        ep_n_queries=np.array(ep_nq),
        ep_start_index=np.array(ep_start),
        n_wall_artifact=np.int64(n_artifact),
    )
    s, nq = np.array(ep_succ), np.array(ep_nq)
    if starts is not None:
        print(f"dropped {n_artifact} wall-artifact episodes of {n_artifact + len(s)}")
    print(
        f"{len(s)} episodes, success {s.mean():.1%} ({s.sum()} succ / {(~s).sum()} fail); "
        f"{len(imgs)} queries, mean/episode succ {nq[s].mean():.1f} vs fail {nq[~s].mean():.1f} -> {out}"
    )


if __name__ == "__main__":
    main()
