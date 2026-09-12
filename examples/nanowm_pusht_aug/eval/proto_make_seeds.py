#!/usr/bin/env python
"""Seed generator for the closed-loop rollout experiment (Task 2b, lerobot venv).

Rolls the baseline image-only n_obs=1 policy in gym_pusht at native 10 Hz, records per
step the full sim state (agent xy, block xyθ) + a 224 render, then samples NEAR-GOAL
states as closed-loop seeds: for each episode take the window ending at the first step
where coverage >= --cov-enter (or the last window if it never gets there) and sample
--per-episode states from it. This scopes the closed-loop experiment to endgame action
refinement, which is the regime a short (<= 8 NanoWM step) strict-terminal accept can reach.

    uv run --extra training python examples/nanowm_pusht_aug/eval/proto_make_seeds.py \
        --ckpt outputs/train/pusht_imgonly/obs1_s100000/checkpoints/100000/pretrained_model \
        --episodes 60 --n-seeds 200 --out examples/nanowm_pusht_aug/cl/seeds.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--per-episode", type=int, default=5)
    p.add_argument("--cov-enter", type=float, default=0.5, help="'near goal' = coverage first crosses this")
    p.add_argument(
        "--lookback", type=int, default=45, help="sample seeds from [enter-lookback, enter] (10 Hz steps)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/cl/seeds.npz"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device

    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        observation_width=96,
        observation_height=96,
        visualization_width=224,
        visualization_height=224,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps + 5,
    )
    u = env.unwrapped

    seeds_state, seeds_frame, seeds_meta = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed * 10_000 + ep)
        policy.reset()
        states, frames, covs = [], [], []
        for _t in range(args.max_steps):
            state5 = np.array(
                [*u.agent.position, *u.block.position, u.block.angle % (2 * np.pi)], dtype=np.float64
            )
            states.append(state5)
            frames.append(env.render().astype(np.uint8))  # 224 visualization render
            covs.append(float(u._get_coverage()))

            img = (
                torch.from_numpy(np.asarray(obs["pixels"])).permute(2, 0, 1).float().div(255.0)[None].to(dev)
            )
            with torch.no_grad():
                a = post(policy.select_action(pre({"observation.image": img})))
            obs, _r, term, trunc, _i = env.step(a[0].cpu().numpy().astype(np.float64))
            if term or trunc:
                break
        covs = np.array(covs)
        n = len(states)
        enter = int(np.argmax(covs >= args.cov_enter)) if (covs >= args.cov_enter).any() else n - 1
        lo = max(0, enter - args.lookback)
        idx = rng.choice(
            np.arange(lo, min(enter + 1, n)), size=min(args.per_episode, enter + 1 - lo), replace=False
        )
        for k in idx:
            seeds_state.append(states[k])
            seeds_frame.append(frames[k])
            seeds_meta.append((ep, int(k), float(covs[k]), float(covs[-1])))
        print(
            f"ep {ep:3d}: {n:3d} steps  final_cov={covs[-1]:.2f}  enter@{enter}  +{len(idx)} seeds  (total {len(seeds_state)})"
        )
        if len(seeds_state) >= args.n_seeds:
            break
    env.close()

    sel = slice(0, args.n_seeds)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=np.asarray(seeds_state, np.float64)[sel],
        frames_224=np.asarray(seeds_frame, np.uint8)[sel],
        meta=np.asarray(seeds_meta, np.float64)[sel],
    )
    m = np.asarray(seeds_meta, np.float64)[sel]
    print(
        f"\nwrote {out}  ({len(m)} seeds)  seed-cov mean={m[:, 2].mean():.2f}  "
        f"src-ep-final-cov mean={m[:, 3].mean():.2f}"
    )


if __name__ == "__main__":
    main()
