#!/usr/bin/env python
"""Stage-2 final eval: policy rollouts from FIXED band starts (not resampled), success-only output.

Reuses ood_failure_map.py's set_state/sim_state/block_wall_gap. Unlike stage_sweep (which
resamples sample_topband_starts fresh from a seed), this reads a saved states npz so every arm in
the CUPID-vs-Random comparison faces the EXACT same band starts, in the same order -- required for
the paired test in paired_report.py. Episodes where the block is pushed through a wall (a sim
artifact, see the OOD study) are dropped from both the numerator and denominator.

    uv run --extra training python examples/nanowm_pusht_aug/eval/band_eval.py \
        --ckpt outputs/train/cl_aug/select_cupid_k500_d8_s1000/checkpoints/last/pretrained_model \
        --states outputs/cupid/select_topband/seeds_eval.npz --n 400 --n-action-steps 10 \
        --wall-pen-px 10 --out outputs/cupid/select_topband/eval/select_cupid_k500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL))
from ood_failure_map import block_wall_gap, set_state, sim_state  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True, help="npz with cog_states (N,5)")
    p.add_argument("--n", type=int, default=None, help="default: all states in the file")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-action-steps", type=int, default=0, help="0 = keep the checkpoint's value")
    p.add_argument("--wall-pen-px", type=float, default=10.0)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seeds the diffusion policy's DDPM sampling noise "
        "per batch, so a fixed states file gives REPRODUCIBLE outcomes across arms",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True, help="output dir; writes successes.json")
    args = p.parse_args()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dev = args.device
    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    if args.n_action_steps:
        policy.config.n_action_steps = args.n_action_steps
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )

    starts = np.load(args.states)["cog_states"]
    n = args.n or len(starts)
    assert n <= len(starts), f"--n {n} > {len(starts)} states in {args.states}"
    B = args.batch
    envs = [
        gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96, observation_height=96
        )
        for _ in range(B)
    ]
    success, wall_artifact, t0 = [], [], time.time()
    for b0 in range(0, n, B):
        nb = min(B, n - b0)
        obs, traj = [], [[] for _ in range(nb)]
        for i in range(nb):
            envs[i].reset(seed=0)
            o = set_state(envs[i].unwrapped, starts[b0 + i])
            obs.append(o["pixels"])
            traj[i].append(sim_state(envs[i].unwrapped))
        torch.manual_seed(args.seed + b0)
        policy.reset()
        done = np.zeros(nb, bool)
        succ = np.zeros(nb, bool)
        for _t in range(args.max_steps):
            img = torch.from_numpy(np.stack(obs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)
            with torch.no_grad():
                act = post(policy.select_action(pre({"observation.image": img}))).cpu().numpy()
            for i in range(nb):
                if done[i]:
                    continue
                o, _r, term, trunc, info = envs[i].step(act[i].astype(np.float64))
                obs[i] = o["pixels"]
                traj[i].append(sim_state(envs[i].unwrapped))
                if term or trunc:
                    done[i], succ[i] = True, bool(info.get("is_success", term))
            if done.all():
                break
        for i in range(nb):
            gap = block_wall_gap(np.asarray(traj[i])).min()
            wall_artifact.append(bool(gap < -args.wall_pen_px))
        success += succ.tolist()
        el = time.time() - t0
        kept = np.asarray(success)[~np.asarray(wall_artifact)]
        print(
            f"  {b0 + nb}/{n}  success so far {kept.mean() if len(kept) else float('nan'):.3f}  "
            f"({el / (b0 + nb):.2f} s/ep, eta {el / (b0 + nb) * (n - b0 - nb) / 60:.1f} min)"
        )
    for e in envs:
        e.close()

    success, wall_artifact = np.asarray(success), np.asarray(wall_artifact)
    kept = success[~wall_artifact]
    args.out.mkdir(parents=True, exist_ok=True)
    import json

    (args.out / "successes.json").write_text(
        json.dumps(
            {
                "successes": kept.tolist(),
                "n_wall_artifact": int(wall_artifact.sum()),
                "n_total": n,
                # full per-start arrays, so paired_report can align arms on the SAME start states
                "success_all": success.tolist(),
                "wall_artifact": wall_artifact.tolist(),
            }
        )
    )
    print(
        f"\n{args.out.name}: success {kept.mean():.3f} over {len(kept)} episodes "
        f"({wall_artifact.sum()} wall-artifact dropped of {n}) -> {args.out / 'successes.json'}"
    )


if __name__ == "__main__":
    main()
