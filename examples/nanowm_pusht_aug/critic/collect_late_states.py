#!/usr/bin/env python
"""Last-mile experiment, step 1: collect the base policy's STUCK late states by deploying it (no noise).

Episodes start from env.reset(seed=seed+i) and run the base policy (n_action_steps 10) for up to 300 steps with the
224px camera recorded every 5 steps. A camera frame qualifies when, judged ONLY from the image (coverage_from_frame,
calibrated): t >= t_min, estimate in [cov_lo, cov_hi), the estimate rose by no more than `stall` over the previous 10
steps, and the episode has not ended (an ended episode is visible to a deployed robot). At most `per_episode` starts per
episode, >= `gap` steps apart. Each start is saved as the simulator state 10 steps EARLIER, so that closed_loop_rollout's
standard 10-step policy lead-in rebuilds real context frames and arrives at (a fresh policy sample of) the stuck moment.

Output (nwaug.pusht_state.save_seeds): states (body), cog_states, src_episode, src_step (the qualifying step),
est_cov, plus the seed / rule in provenance_json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pymunk
import torch

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent))
from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402
from make_topband_seeds import cog_to_body  # noqa: E402
from nwaug.pusht_state import save_seeds  # noqa: E402
from ood_failure_map import sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

FI = 5


def qualifying(steps, est, success_step, *, t_min, cov_lo, cov_hi, stall, per_episode, gap, rng) -> list[int]:
    """Grid indices g (each >= 2) of one episode that qualify, at most per_episode, >= gap steps apart (random order)."""
    end = success_step if success_step > 0 else 10**9
    ok = [
        g
        for g in range(2, len(steps))
        if steps[g] >= t_min and steps[g] < end and cov_lo <= est[g] < cov_hi and est[g] - est[g - 2] <= stall
    ]
    picked = []
    for g in rng.permutation(ok) if ok else []:
        if all(abs(int(steps[g]) - int(steps[p])) >= gap for p in picked):
            picked.append(int(g))
        if len(picked) >= per_episode:
            break
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=940000)
    ap.add_argument("--max-episodes", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--t-min", type=int, default=100)
    ap.add_argument("--cov-lo", type=float, default=0.85)
    ap.add_argument("--cov-hi", type=float, default=0.945)
    ap.add_argument("--stall", type=float, default=0.005)
    ap.add_argument("--per-episode", type=int, default=3)
    ap.add_argument("--gap", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(a.ckpt, device=dev)
    rng = np.random.default_rng(a.seed + 1)
    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    cog_local = pymunk.Vec2d(*env.unwrapped.block.center_of_gravity)
    got, n_ep, t0 = [], 0, time.time()
    while len(got) < a.n:
        if n_ep >= a.max_episodes:
            raise SystemExit(f"only {len(got)}/{a.n} stuck starts after {n_ep} episodes")
        starts = []
        for i in range(n_ep, n_ep + a.batch):
            env.reset(seed=a.seed + i)
            starts.append(sim_state(env.unwrapped))
        res = rollout_batch(
            policy,
            pre,
            post,
            np.asarray(starts),
            device=dev,
            na=10,
            episode_steps=300,
            seed=a.seed,
            batch_start=n_ep,
            lockstep224=True,
            frame_grid_stride=FI,
            frame_cutoff_step=300,
        )
        steps, frames, gcog = res["grid_steps"], res["grid_frames"], res["grid_cog"]
        for j in range(a.batch):
            ss = int(res["success_step"][j])
            last = len(steps) if ss < 0 else int(np.searchsorted(steps, ss))
            est = np.full(len(steps), np.nan)
            for g in range(max(0, int(np.searchsorted(steps, a.t_min)) - 2), last):
                est[g] = coverage_from_frame(frames[j, g])["coverage_cal"]
            for g in qualifying(
                steps,
                np.nan_to_num(est, nan=-1.0),
                ss,
                t_min=a.t_min,
                cov_lo=a.cov_lo,
                cov_hi=a.cov_hi,
                stall=a.stall,
                per_episode=a.per_episode,
                gap=a.gap,
                rng=rng,
            ):
                if len(got) < a.n:
                    got.append(
                        {
                            "cog": gcog[j, g - 2].copy(),
                            "ep": n_ep + j,
                            "step": int(steps[g]),
                            "est": float(est[g]),
                        }
                    )
        n_ep += a.batch
        print(f"  {n_ep} episodes: {len(got)}/{a.n} stuck starts ({time.time() - t0:.0f}s)", flush=True)
    env.close()
    cog = np.stack([g["cog"] for g in got])
    body = cog_to_body(cog, cog_local)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_seeds(
        a.out,
        body=body,
        cog=cog,
        src_episode=np.array([g["ep"] for g in got], np.int64),
        src_step=np.array([g["step"] for g in got], np.int64),
        est_cov=np.array([g["est"] for g in got], np.float32),
        provenance_json=json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}),
    )
    print(
        f"{len(got)} stuck starts from {n_ep} episodes; est cov median {np.median([g['est'] for g in got]):.3f}, "
        f"step median {np.median([g['step'] for g in got]):.0f} -> {a.out}"
    )


if __name__ == "__main__":
    main()
