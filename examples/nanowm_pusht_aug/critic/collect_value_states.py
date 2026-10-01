#!/usr/bin/env python
"""Stage 1 of the value/reward-model plan: base-policy rollouts from top-wall-band and random-reset
starts, producing supervised (state, coverage, success-within-B) targets on the 5-step frame grid.

V label: y = 1{first success step s with t <= s <= t+B}, with B=200 steps and the base policy at
n_action_steps=10 from t. An episode stops at its first success (gym_pusht terminates on success) and
runs at most `stage1.episode_steps` steps (legacy topband config 310, SAILOR v1 300), so every query
window t+B <= episode_steps is fully observed. RM label: sim `info["coverage"]` at t. Every sample's
`frame_steps` = [t-10, t-5, t] (a 5-step-grid triple spanning 10 real steps), matching the H=3,
fi=5 latent stack `imagined_rollout.py encode` will later build from `frames_224`.

Perturbation (`stage1.perturb`, see `critic/labels.py::draw_perturbations`):
  * legacy (no `mode` key): with prob 0.4/0.3/0.3 each episode gets sigma 0/0.5/1.0 injected into ONE
    action chunk at a random replan boundary p in {0,10,...,100}; queries at t < p+10 are dropped.
  * SAILOR v1 (`mode: prefix`): every chunk whose start is < p is noised (`switch: random` -> p drawn
    from {0,10,...,100}; `switch: all` -> p = episode_steps, i.e. the whole episode). `stage1.query_keep`
    = `all` keeps every query, `after_switch` keeps t >= p (`labels.keep_query`).
Episodes whose block gets pushed through a wall (sim artifact, block_wall_gap < -wall_pen_px anywhere
in the rollout) keep their episode-level record but emit no queries.

Already-succeeded queries: if the episode's first success step s equals the query step t exactly
(coverage>0.95 already reached there), the query is kept with y=True and pre_success=True (a
strictly earlier success would have made the query not "exist" at all -- see the s < t skip
below). pre_success rows should be excluded from gate metrics downstream, but are kept for
training.

Per-query label fields (D1 / D3 of the SAILOR v1 pre-registration, all via `critic/labels.py`):
  q_T_succ   absolute first-success step (-1 = never)      q_T        steps from t to success (-1 never, may exceed B)
  q_cov_end  coverage at t + min(T, B) (t+B when T<0/T>B)  q_v_target gamma**T * 1[0<=T<=B]   (beta = 0)
  q_perturb_p, q_noised = (sigma>0) & (perturb_p>0), q_t.
Shards also carry `cov_per_step` (E, episode_steps+1), `perturb_mode` and `provenance_json`.

Writes out_dir/stage1/shard_XXX.npz (out_dir/smoke/stage1/shard_000.npz with --smoke), one shard
per `stage1.shard_episodes` episodes (savez_compressed); a same-name stale `latents_XXX.npz` next to
the shard is deleted when the shard is (re)written.

    uv run --extra training python examples/nanowm_pusht_aug/critic/collect_value_states.py \
        --config configs/sailor_v1_reset.yaml --smoke
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
import torch

_CRITIC = Path(__file__).resolve().parent
_EVAL = _CRITIC.parent / "eval"
_ATTR = _CRITIC.parent / "attribution"
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_EVAL))
sys.path.insert(0, str(_ATTR))
sys.path.insert(0, str(_CRITIC.parent))
from config_io import load_config  # noqa: E402
from datagen.provenance import provenance_entry  # noqa: E402
from ood_failure_map import sample_topband_starts, sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

from critic import labels  # noqa: E402

QUERY_TS = list(range(10, 111, 10))  # DEFAULT {10, ..., 110} when `stage1.query_ts` is absent
DEFAULT_GAMMA = 0.995
QUERY_KEEP_LEGACY = "legacy"  # no `stage1.query_keep` key: the old `t < p + 10` drop for sigma > 0


def make_starts(cfg: dict, n_band: int, n_reset: int) -> tuple[np.ndarray, np.ndarray]:
    """(cog_starts (n_band+n_reset,5), source labels) -- band starts first, then random resets.

    Reset-source starts use gym_pusht's OWN reset distribution (`env.reset(seed=...)`), then read
    back as a CoG state via `sim_state` so the exact same physical placement can be reproduced with
    `set_state` in both the 96px and 224px envs later (set_state and gym_pusht's native reset use
    different internal conventions; only the CoG-state readback is portable between them).

    `n_band == 0` short-circuits: `band.seed` / `band.min_wall_gap` are never read and the band part
    is an empty `(0, 5)` array (so the concatenation below is shape-consistent)."""
    s1 = cfg["stage1"]
    if n_band > 0:
        band = sample_topband_starts(n_band, seed=s1["band"]["seed"], min_wall_gap=s1["band"]["min_wall_gap"])
    else:
        band = np.zeros((0, 5))
    reset_starts = []
    if n_reset:
        reset_env = gym.make("gym_pusht/PushT-v0")
        for i in range(n_reset):
            reset_env.reset(seed=s1["reset"]["seed"] + i)
            reset_starts.append(sim_state(reset_env.unwrapped))
        reset_env.close()
    reset_starts = np.asarray(reset_starts, np.float64) if n_reset else np.zeros((0, 5))
    starts = np.concatenate([np.asarray(band, np.float64).reshape(-1, 5), reset_starts], axis=0)
    source = np.array(["band"] * n_band + ["reset"] * n_reset)
    return starts, source


def draw_perturbations(cfg: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(sigma (n,), perturb_p (n,)) for stage 1: `labels.draw_perturbations` on `stage1.perturb`
    (+ `stage1.episode_steps`, which `switch: all` uses as the p value)."""
    return labels.draw_perturbations(cfg["stage1"]["perturb"], n, cfg["stage1"]["episode_steps"])


def _keep(query_keep: str, t: int, p: int, sigma: float) -> bool:
    if query_keep == "all":
        return True
    if query_keep == "after_switch":
        return bool(labels.keep_query(t, p, sigma))
    if query_keep == QUERY_KEEP_LEGACY:
        return not (sigma > 0 and t < p + 10)
    raise ValueError(f"query_keep must be one of {labels.QUERY_KEEPS}, got {query_keep!r}")


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
    *,
    cov_per_step: np.ndarray,
    query_ts=None,
    query_keep: str = QUERY_KEEP_LEGACY,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    """Per-query rows for one shard's worth of episodes. `ep_ids` are the GLOBAL episode indices
    (band starts first, then reset starts, 0-based over the whole stage-1 run) stored per query so
    training-time grouping (by episode, for the 80/20 split) is unambiguous across shard files.

    `cov_per_step` (E, L) is the per-step coverage (index s = after s executed steps, forward-filled
    after termination); `cov_end` is read at `labels.cov_end_index(t, T, B)`. A success later than
    the window (`T > B`) gives y = 0 / v_target = 0 with `q_T` keeping the true T and `cov_end` read
    at t + B. `query_keep`: `all` | `after_switch` | (absent -> legacy `t < p + 10` drop)."""
    query_ts = QUERY_TS if query_ts is None else [int(t) for t in query_ts]
    cov_per_step = np.asarray(cov_per_step)
    L = cov_per_step.shape[1]
    keys = (
        "episode",
        "t",
        "frame_steps",
        "agent_state",
        "cog_state",
        "coverage",
        "y",
        "pre_success",
        "source",
        "T_succ",
        "T",
        "cov_end",
        "perturb_p",
        "noised",
        "sigma",
    )
    rows = {k: [] for k in keys}
    for e in range(len(ep_ids)):
        if wall_artifact[e]:
            continue
        s = int(success_step[e])
        for t in query_ts:
            T = int(labels.success_T(s, t))
            if T == labels.NOT_EXISTING:  # env terminated before t: the query does not exist
                continue
            if not _keep(query_keep, t, int(perturb_p[e]), float(sigma[e])):
                continue
            gi = t // 5
            pre_success = T == 0  # success reached exactly at this query step
            y = 0 <= T <= b_steps
            ci = int(labels.cov_end_index(t, T, b_steps))
            assert ci < L, f"cov_end index {ci} outside cov_per_step ({L}): episode_steps < t + B"
            rows["episode"].append(int(ep_ids[e]))
            rows["t"].append(t)
            rows["frame_steps"].append([t - 10, t - 5, t])
            rows["agent_state"].append(np.concatenate([grid_agent_pos[e, gi], grid_agent_vel[e, gi]]))
            rows["cog_state"].append(grid_cog[e, gi])
            rows["coverage"].append(float(grid_coverage[e, gi]))
            rows["y"].append(y)
            rows["pre_success"].append(pre_success)
            rows["source"].append(str(source[e]))
            rows["T_succ"].append(s)
            rows["T"].append(T)
            rows["cov_end"].append(float(cov_per_step[e, ci]))
            rows["perturb_p"].append(int(perturb_p[e]))
            rows["noised"].append(bool(sigma[e] > 0 and perturb_p[e] > 0))
            rows["sigma"].append(float(sigma[e]))
    T_arr = np.asarray(rows["T"], np.int64)
    cov_end = np.asarray(rows["cov_end"], np.float32)
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
        "T_succ": np.asarray(rows["T_succ"], np.int64),
        "T": T_arr,
        "cov_end": cov_end,
        "v_target": np.asarray(labels.v_target(T_arr, cov_end, b_steps, gamma, 0.0), np.float32),
        "perturb_p": np.asarray(rows["perturb_p"], np.int64),
        "noised": np.asarray(rows["noised"], bool),
        "sigma": np.asarray(rows["sigma"], np.float32),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    pr = s1["perturb"]
    perturb_mode = pr.get("mode", "point")
    query_ts = [int(t) for t in s1.get("query_ts", QUERY_TS)]
    query_keep = s1.get("query_keep", QUERY_KEEP_LEGACY)
    if query_keep != QUERY_KEEP_LEGACY and query_keep not in labels.QUERY_KEEPS:
        raise SystemExit(f"stage1.query_keep must be one of {labels.QUERY_KEEPS}, got {query_keep!r}")
    gamma = float(cfg["label"].get("gamma", DEFAULT_GAMMA))
    frame_cutoff = int(s1.get("frame_cutoff_step", max(query_ts)))  # 100 for query_ts <= 100
    assert frame_cutoff >= max(query_ts), (frame_cutoff, max(query_ts))

    n_band_cfg, n_reset_cfg = s1["band"]["n"], s1["reset"]["n"]
    if args.smoke:
        total = cfg["smoke"]["stage1_episodes"]
        n_cfg = n_band_cfg + n_reset_cfg
        ratio_band = n_band_cfg / n_cfg if n_cfg else 0.0
        n_band = round(total * ratio_band)
        n_reset = total - n_band
        shard_episodes = total
        out_root = Path(cfg["out_dir"]) / "smoke" / "stage1"
    else:
        n_band, n_reset = n_band_cfg, n_reset_cfg
        shard_episodes = s1["shard_episodes"]
        out_root = Path(cfg["out_dir"]) / "stage1"
    out_root.mkdir(parents=True, exist_ok=True)
    n_total = n_band + n_reset
    print(
        f"stage1 collect: {n_band} band + {n_reset} reset = {n_total} episodes -> {out_root} "
        f"(perturb mode={perturb_mode} switch={pr.get('switch', 'random')} query_keep={query_keep})"
    )

    starts, source = make_starts(cfg, n_band, n_reset)
    sigma, perturb_p = draw_perturbations(cfg, n_total)

    ckpt = Path(cfg["ckpt"][cfg["ckpt_key"]])
    policy, pre, post = load_policy(ckpt, device=device)

    batch = s1["batch"]
    na = cfg["label"]["na"]
    episode_steps = s1["episode_steps"]
    B = cfg["label"]["B"]
    wall_pen_px = cfg["label"]["wall_pen_px"]
    noise_rng = np.random.default_rng(pr["seed"] + 1)  # separate stream from sigma/p draws
    # policy torch seed (rollout_batch: manual_seed(seed + batch_start)); legacy configs: 0. The SAILOR
    # config sets `stage1.policy_seed` to its stage-1 reset namespace (730000), asserted in its test.
    base_seed = int(s1.get("policy_seed", 0))
    prov = provenance_entry(args, stage="collect", config=str(args.config.resolve()))

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
        s_cov_per_step = np.zeros((shard_n, episode_steps + 1), np.float32)
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
                seed=base_seed,
                batch_start=shard0 + b0,
                sigma=sigma[gi],
                perturb_p=perturb_p[gi],
                max_z=pr["max_z"],
                noise_hold=pr["noise_hold"],
                noise_rng=noise_rng,
                lockstep224=True,
                frame_grid_stride=5,
                frame_cutoff_step=frame_cutoff,
                wall_pen_px=wall_pen_px,
                perturb_mode=perturb_mode,
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
            s_cov_per_step[b0 : b0 + nb] = out["cov_per_step"]
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
            cov_per_step=s_cov_per_step,
            query_ts=query_ts,
            query_keep=query_keep,
            gamma=gamma,
        )
        shard_path = out_root / f"shard_{shard_i:03d}.npz"
        # N18b: a same-name latents file now belongs to a DIFFERENT shard content -- drop it
        stale = out_root / f"latents_{shard_i:03d}.npz"
        if stale.exists():
            stale.unlink()
            print(f"removed stale {stale}")
        np.savez_compressed(
            shard_path,
            # per-episode
            frames_224=s_grid_frames,
            cog_states=s_grid_cog,
            grid_steps=s_grid_steps,
            cov_per_step=s_cov_per_step,
            perturb_p=s_perturb_p,
            sigma=s_sigma,
            wall_artifact=s_wall,
            source=s_source,
            success_step=s_success,
            start_cog_state=s_start_cog,
            episode=ep_ids,
            state_convention="cog",
            perturb_mode=perturb_mode,
            query_keep=query_keep,
            episode_steps=np.int64(episode_steps),
            provenance_json=json.dumps(prov),
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
            q_T_succ=queries["T_succ"],
            q_T=queries["T"],
            q_cov_end=queries["cov_end"],
            q_v_target=queries["v_target"],
            q_perturb_p=queries["perturb_p"],
            q_noised=queries["noised"],
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
