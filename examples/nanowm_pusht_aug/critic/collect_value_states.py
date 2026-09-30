#!/usr/bin/env python
"""Stage 1 of the value/reward-model plan: base-policy rollouts from top-wall-band and random-reset
starts, producing supervised (state, coverage, success-within-B) targets on the 5-step frame grid.

V label: y = 1{any is_success in (t, t+B]}, with B=200 steps and the base policy at n_action_steps=10
from t. Episodes run the full 310 steps regardless of success, so every query window t in
{10,...,110} is fully observed (t+B <= 310). RM label: sim `info["coverage"]` at t. Every sample's
`frame_steps` = [t-10, t-5, t] (a 5-step-grid triple spanning 10 real steps), matching the H=3,
fi=5 latent stack `imagined_rollout.py encode` will later build from `frames_224`.

Perturbation: with prob 0.4/0.3/0.3 each episode gets sigma 0/0.5/1.0 injected into ONE action
chunk at a random replan boundary p in {0,10,...,100} (see sim_rollout.inject_noise). Queries at
t < p+10 would label a value that mixes in a non-base action, so they are dropped for sigma>0
episodes; sigma=0 episodes emit every t (p is irrelevant there). Episodes whose block gets pushed
through a wall (sim artifact, block_wall_gap < -wall_pen_px anywhere in the rollout) keep their
episode-level record but emit no queries.

Already-succeeded queries: if the episode's first success step s equals the query step t exactly
(coverage>0.95 already reached there), the query is kept with y=True and pre_success=True (a
strictly earlier success would have made the query not "exist" at all -- see the s < t skip
below). pre_success rows should be excluded from gate metrics downstream, but are kept for
training.

Writes out_dir/stage1/shard_XXX.npz (out_dir/smoke/stage1/shard_000.npz with --smoke), one shard
per `stage1.shard_episodes` episodes (savez_compressed).

    uv run --extra training python examples/nanowm_pusht_aug/critic/collect_value_states.py \
        --config configs/value_model_topband.yaml --smoke
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

_CRITIC = Path(__file__).resolve().parent
_EVAL = _CRITIC.parent / "eval"
_ATTR = _CRITIC.parent / "attribution"
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_EVAL))
sys.path.insert(0, str(_ATTR))
from config_io import load_config  # noqa: E402
from ood_failure_map import sample_topband_starts, sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

QUERY_TS = list(range(10, 111, 10))  # {10, 20, ..., 110}, every replan boundary after the lead-in


def make_starts(cfg: dict, n_band: int, n_reset: int) -> tuple[np.ndarray, np.ndarray]:
    """(cog_starts (n_band+n_reset,5), source labels) -- band starts first, then random resets.

    Reset-source starts use gym_pusht's OWN reset distribution (`env.reset(seed=...)`), then read
    back as a CoG state via `sim_state` so the exact same physical placement can be reproduced with
    `set_state` in both the 96px and 224px envs later (set_state and gym_pusht's native reset use
    different internal conventions; only the CoG-state readback is portable between them)."""
    s1 = cfg["stage1"]
    band = sample_topband_starts(n_band, seed=s1["band"]["seed"], min_wall_gap=s1["band"]["min_wall_gap"])
    reset_env = gym.make("gym_pusht/PushT-v0")
    reset_starts = []
    for i in range(n_reset):
        reset_env.reset(seed=s1["reset"]["seed"] + i)
        reset_starts.append(sim_state(reset_env.unwrapped))
    reset_env.close()
    reset_starts = np.asarray(reset_starts, np.float64) if n_reset else np.zeros((0, 5))
    starts = np.concatenate([band, reset_starts], axis=0)
    source = np.array(["band"] * n_band + ["reset"] * n_reset)
    return starts, source


def draw_perturbations(cfg: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(sigma (n,), perturb_p (n,)) drawn once, in episode order, from a single rng seeded by
    `stage1.perturb.seed` -- independent of how episodes are later chunked into batches/shards."""
    pr = cfg["stage1"]["perturb"]
    rng = np.random.default_rng(pr["seed"])
    sigmas = np.asarray(pr["sigmas"], np.float64)
    probs = np.asarray(pr["probs"], np.float64)
    sigma = rng.choice(sigmas, size=n, p=probs)
    p_choices = np.arange(0, 101, 10)
    perturb_p = rng.choice(p_choices, size=n)
    return sigma, perturb_p


def build_queries(
    ep_ids: np.ndarray,
    source: np.ndarray,
    sigma: np.ndarray,
    perturb_p: np.ndarray,
    success_step: np.ndarray,
    wall_artifact: np.ndarray,
    grid_cog: np.ndarray,
    grid_agent_pos: np.ndarray,
    grid_agent_vel: np.ndarray,
    grid_coverage: np.ndarray,
    b_steps: int,
) -> dict:
    """Per-query rows for one shard's worth of episodes. `ep_ids` are the GLOBAL episode indices
    (band starts first, then reset starts, 0-based over the whole stage-1 run) stored per query so
    training-time grouping (by episode, for the 80/20 split) is unambiguous across shard files."""
    rows = {
        k: []
        for k in (
            "episode",
            "t",
            "frame_steps",
            "agent_state",
            "cog_state",
            "coverage",
            "y",
            "pre_success",
            "source",
        )
    }
    for e in range(len(ep_ids)):
        if wall_artifact[e]:
            continue
        s = int(success_step[e])
        for t in QUERY_TS:
            if s != -1 and s < t:  # env not terminated before t is required to "exist"
                continue
            if sigma[e] > 0 and t < perturb_p[e] + 10:
                continue
            gi = t // 5
            pre_success = bool(s != -1 and s == t)  # success reached exactly at this query step
            y = True if pre_success else bool(s != -1 and t < s <= t + b_steps)
            rows["episode"].append(int(ep_ids[e]))
            rows["t"].append(t)
            rows["frame_steps"].append([t - 10, t - 5, t])
            rows["agent_state"].append(np.concatenate([grid_agent_pos[e, gi], grid_agent_vel[e, gi]]))
            rows["cog_state"].append(grid_cog[e, gi])
            rows["coverage"].append(float(grid_coverage[e, gi]))
            rows["y"].append(y)
            rows["pre_success"].append(pre_success)
            rows["source"].append(str(source[e]))
    return {
        "episode": np.asarray(rows["episode"], np.int64),
        "t": np.asarray(rows["t"], np.int64),
        "frame_steps": np.asarray(rows["frame_steps"], np.int64).reshape(-1, 3),
        "agent_state": np.asarray(rows["agent_state"], np.float64).reshape(-1, 4),
        "cog_state": np.asarray(rows["cog_state"], np.float64).reshape(-1, 5),
        "coverage": np.asarray(rows["coverage"], np.float32),
        "y": np.asarray(rows["y"], bool),
        "pre_success": np.asarray(rows["pre_success"], bool),
        "source": np.asarray(rows["source"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        total = cfg["smoke"]["stage1_episodes"]
        ratio_band = s1["band"]["n"] / (s1["band"]["n"] + s1["reset"]["n"])
        n_band = round(total * ratio_band)
        n_reset = total - n_band
        shard_episodes = total
        out_root = Path(cfg["out_dir"]) / "smoke" / "stage1"
    else:
        n_band, n_reset = s1["band"]["n"], s1["reset"]["n"]
        shard_episodes = s1["shard_episodes"]
        out_root = Path(cfg["out_dir"]) / "stage1"
    out_root.mkdir(parents=True, exist_ok=True)
    n_total = n_band + n_reset
    print(f"stage1 collect: {n_band} band + {n_reset} reset = {n_total} episodes -> {out_root}")

    starts, source = make_starts(cfg, n_band, n_reset)
    sigma, perturb_p = draw_perturbations(cfg, n_total)

    ckpt = Path(cfg["ckpt"][cfg["ckpt_key"]])
    policy, pre, post = load_policy(ckpt, device=device)

    batch = s1["batch"]
    na = cfg["label"]["na"]
    episode_steps = s1["episode_steps"]
    B = cfg["label"]["B"]
    wall_pen_px = cfg["label"]["wall_pen_px"]
    noise_rng = np.random.default_rng(s1["perturb"]["seed"] + 1)  # separate stream from sigma/p draws

    t0 = time.time()
    n_done = 0
    for shard_i, shard0 in enumerate(range(0, n_total, shard_episodes)):
        shard_n = min(shard_episodes, n_total - shard0)
        ep_ids = np.arange(shard0, shard0 + shard_n)
        s_success = np.full(shard_n, -1, np.int64)
        s_wall = np.zeros(shard_n, bool)
        s_grid_cog = None
        s_grid_pos = None
        s_grid_vel = None
        s_grid_cov = None
        s_grid_frames = None
        s_grid_steps = None
        s_perturb_p = perturb_p[shard0 : shard0 + shard_n]
        s_sigma = sigma[shard0 : shard0 + shard_n]
        s_source = source[shard0 : shard0 + shard_n]
        s_start_cog = starts[shard0 : shard0 + shard_n]

        lockstep_checks_total = 0
        for b0 in range(0, shard_n, batch):
            nb = min(batch, shard_n - b0)
            gi = slice(shard0 + b0, shard0 + b0 + nb)
            out = rollout_batch(
                policy,
                pre,
                post,
                starts[gi],
                device=device,
                na=na,
                episode_steps=episode_steps,
                seed=0,
                batch_start=shard0 + b0,
                sigma=sigma[gi],
                perturb_p=perturb_p[gi],
                max_z=s1["perturb"]["max_z"],
                noise_hold=s1["perturb"]["noise_hold"],
                noise_rng=noise_rng,
                lockstep224=True,
                frame_grid_stride=5,
                frame_cutoff_step=110,
                wall_pen_px=wall_pen_px,
            )
            assert out["queue_maxlen"] == na, out["queue_maxlen"]
            lockstep_checks_total += out["lockstep_checks"]
            if s_grid_cog is None:
                G = len(out["grid_steps"])
                s_grid_steps = out["grid_steps"]
                s_grid_cog = np.zeros((shard_n, G, 5), np.float64)
                s_grid_pos = np.zeros((shard_n, G, 2), np.float64)
                s_grid_vel = np.zeros((shard_n, G, 2), np.float64)
                s_grid_cov = np.zeros((shard_n, G), np.float32)
                s_grid_frames = np.zeros((shard_n, G, 224, 224, 3), np.uint8)
            s_success[b0 : b0 + nb] = out["success_step"]
            s_wall[b0 : b0 + nb] = out["wall_artifact"]
            s_grid_cog[b0 : b0 + nb] = out["grid_cog"]
            s_grid_pos[b0 : b0 + nb] = out["grid_agent_pos"]
            s_grid_vel[b0 : b0 + nb] = out["grid_agent_vel"]
            s_grid_cov[b0 : b0 + nb] = out["grid_coverage"]
            s_grid_frames[b0 : b0 + nb] = out["grid_frames"]
            n_done += nb
            el = time.time() - t0
            print(
                f"  shard {shard_i} {b0 + nb}/{shard_n}  (total {n_done}/{n_total}, "
                f"{el / n_done:.2f} s/ep, eta {el / n_done * (n_total - n_done) / 60:.1f} min)"
            )

        queries = build_queries(
            ep_ids,
            s_source,
            s_sigma,
            s_perturb_p,
            s_success,
            s_wall,
            s_grid_cog,
            s_grid_pos,
            s_grid_vel,
            s_grid_cov,
            B,
        )
        shard_path = out_root / f"shard_{shard_i:03d}.npz"
        np.savez_compressed(
            shard_path,
            # per-episode
            frames_224=s_grid_frames,
            cog_states=s_grid_cog,
            grid_steps=s_grid_steps,
            perturb_p=s_perturb_p,
            sigma=s_sigma,
            wall_artifact=s_wall,
            source=s_source,
            success_step=s_success,
            start_cog_state=s_start_cog,
            episode=ep_ids,
            state_convention="cog",
            # per-query
            q_episode=queries["episode"],
            q_t=queries["t"],
            q_frame_steps=queries["frame_steps"],
            q_agent_state=queries["agent_state"],
            q_cog_state=queries["cog_state"],
            q_coverage=queries["coverage"],
            q_y=queries["y"],
            q_pre_success=queries["pre_success"],
            q_source=queries["source"],
        )
        n_q = len(queries["t"])
        pos_rate = queries["y"].mean() if n_q else float("nan")
        print(
            f"wrote {shard_path}: {shard_n} episodes, {n_q} queries, y positive rate {pos_rate:.3f}, "
            f"wall-artifact {s_wall.sum()}/{shard_n}, lockstep checks {lockstep_checks_total}"
        )

    el = time.time() - t0
    print(f"\nstage1 collect done: {n_total} episodes in {el:.1f}s ({el / n_total:.2f} s/ep) -> {out_root}")


if __name__ == "__main__":
    main()
