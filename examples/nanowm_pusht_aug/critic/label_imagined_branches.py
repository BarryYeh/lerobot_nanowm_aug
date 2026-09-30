#!/usr/bin/env python
"""Stage 2 of the value/reward-model plan: EXACT-replay V labels for NanoWM-imagined branches.

Input: every branch npz that `eval/closed_loop_rollout.py --save-latents --accept-cov 0` wrote under
<stage2 root>/sigma*/branches/ (one per (anchor seed, candidate)). Each carries the NanoWM latents of
the whole rollout, the real lead-in actions, the executed action targets and the PD-tracked agent
state at every round boundary.

For every branch and every query round r in `stage2.query_rounds` (r=0 once per seed, not once per
branch: at r=0 all three frames are the real context, only the VAE sample noise differs) the label
is the outcome of the BASE policy from exactly the state the imagined branch is in after r rounds:

  1. Re-create the seed state the way the driver does (state convention: BODY origin, the same
     `seed_state` the driver placed): reset(seed=9000+seed_index), agent position, block.angle,
     block.position, zero block velocity, one `space.step`. Replay `lead_acts` (10 real actions) and
     ASSERT the sim equals the saved `anchor_state` (body origin, same form) within 1e-6 and the agent
     pos/vel equals `agent_state_hist[0]` within 1e-6 px. This proves the replay reproduces the run
     that produced the context frames.
  2. Replay `exec_abs_targets[: r*5]` and ASSERT agent pos/vel == `agent_state_hist[r]` within 1e-6 px
     (the driver's PD tracking of the agent equals the sim's). If the env terminated (success) during
     the replay the row is pre_success=True, y=True and no continuation is run.
  3. Otherwise continue with the base policy (n_action_steps=10, fresh `policy.reset()`) for B=200
     steps; y = any `info["is_success"]`. Continuations are batched across rows through
     `sim_rollout.rollout_batch(prepared_envs=...)`, so the rollout loop is not duplicated.
  4. wall_artifact = min `block_wall_gap` (CoG states) over the replay AND the continuation < -10 px.
     The row is kept and flagged; the gate/training code excludes flagged rows per the label rules.

This is NOT proto_filter_roc.py's replay, which starts from `anchor_state` with zero velocity; here
the agent's true velocity at the anchor and at every boundary is reproduced by construction.

Row fields (labels.npz, one row per (branch, r) with the r=0 exception above):
  latent_stack (N,3,4,32,32) f16   latents[r : r+3]; index r+2 (newest) is the frame of query time r*5
  frame_steps  (N,3) int64         [r*5-10, r*5-5, r*5] relative to the anchor (spacing 5, asserted)
  agent_state  (N,4)  f64          agent_state_hist[r] = [x, y, vx, vy]
  cov          (N,)   f32          RM label: worker cov of the imagined frame (r>=1: coverage[r-1]);
                                   r=0: SIM coverage at the anchor (the frame is real)
  cov_source   (N,)   str          "imagined" | "sim"
  sim_cov      (N,)   f32          sim coverage after the replay (diagnostic; == cov at r=0)
  y, pre_success, wall_artifact (N,) bool;  wall_gap_min (N,) f32
  sigma (N,) f32 (branch inject_sigma), seed_id (N,) i64 (row in seeds_stage2.npz), cand (N,) i64,
  branch (N,) str (file name, sigma dir prefixed), r (N,) i64
  replay_cog_state (N,5) f64       CoG state [agent xy, CoG xy, theta] after the replay (oracle input)
Stored scalars: state_convention_replay_cog_state="cog", B, na, check counters.

    uv run --extra training python examples/nanowm_pusht_aug/critic/label_imagined_branches.py \
        --config configs/value_model_topband.yaml [--smoke]
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
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402
from ood_failure_map import block_wall_gap, sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

FI = 5  # sim steps per NanoWM round
LEAD = 10  # real lead-in steps (driver: (H-1)*fi)
TOL = 1e-6


def body_state(u) -> np.ndarray:
    """Body-origin sim state in the SAME form as the driver's `anchor_state`."""
    return np.array([*u.agent.position, u.block.position[0], u.block.position[1], u.block.angle], np.float64)


def agent_pv(u) -> np.ndarray:
    return np.array([*u.agent.position, *u.agent.velocity], np.float64)


def place_seed(env, seed_index: int, st: np.ndarray) -> None:
    """The driver's seed placement, verbatim (closed_loop_rollout.py lead-in, BODY-origin state)."""
    u = env.unwrapped
    env.reset(seed=9000 + seed_index)
    u.agent.position = (float(st[0]), float(st[1]))
    u.block.angle = float(st[4])
    u.block.position = (float(st[2]), float(st[3]))
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


def replay_row(env, br: dict, r: int, name: str, stats: dict) -> dict:
    """Exact replay of one branch to round r on `env`. Asserts as documented in the module doc."""
    u = env.unwrapped
    place_seed(env, int(br["seed_index"]), br["seed_state"])
    wall_min = float(block_wall_gap(sim_state(u)[None])[0])
    for t in range(LEAD):
        env.step(br["lead_acts"][t])
        wall_min = min(wall_min, float(block_wall_gap(sim_state(u)[None])[0]))
    err_anchor = float(np.abs(body_state(u) - br["anchor_state"]).max())
    assert err_anchor <= TOL, f"{name} r={r}: replayed anchor differs from anchor_state by {err_anchor:.3e}"
    err0 = float(np.abs(agent_pv(u) - br["agent_state_hist"][0]).max())
    assert err0 <= TOL, f"{name} r={r}: agent state at anchor differs from agent_state_hist[0] by {err0:.3e}"
    stats["anchor_checks"] += 1
    stats["max_anchor_err"] = max(stats["max_anchor_err"], err_anchor)
    stats["agent_checks"] += 1
    stats["max_agent_err"] = max(stats["max_agent_err"], err0)
    cov_anchor = float(u._get_coverage())

    terminated, n_done = False, 0
    for t in range(r * FI):
        _o, _r, term, _tr, _info = env.step(br["exec_abs_targets"][t])
        n_done += 1
        wall_min = min(wall_min, float(block_wall_gap(sim_state(u)[None])[0]))
        if term:
            terminated = True
            break
    if r > 0 and n_done == r * FI:  # the whole replay ran (success on its last step still qualifies)
        errr = float(np.abs(agent_pv(u) - br["agent_state_hist"][r]).max())
        assert errr <= TOL, f"{name} r={r}: agent state differs from agent_state_hist[{r}] by {errr:.3e}"
        stats["agent_checks"] += 1
        stats["max_agent_err"] = max(stats["max_agent_err"], errr)
    return {
        "terminated": terminated,
        "wall_min": wall_min,
        "sim_cov": float(u._get_coverage()),
        "cov_anchor": cov_anchor,
        "cog": sim_state(u),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch", type=int, default=100, help="rows whose envs are alive/continued together")
    args = p.parse_args()
    cfg = load_config(args.config)
    s2, lab = cfg["stage2"], cfg["label"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["out_dir"]) / ("smoke/stage2" if args.smoke else "stage2")
    rounds = [int(r) for r in s2["query_rounds"]]
    B, na, wall_pen = int(lab["B"]), int(lab["na"]), float(lab["wall_pen_px"])
    h_max = int(s2["h_max"])
    assert max(rounds) <= h_max and rounds[0] == 0, rounds

    branch_files = []
    for d in sorted(root.glob("sigma*/branches")):
        branch_files += sorted(d.glob("seed*_cand*.npz"))
    assert branch_files, f"no branch npz under {root}/sigma*/branches"
    print(f"label: {len(branch_files)} branches, rounds {rounds}, B={B}, na={na} <- {root}")

    seeds_path = Path(cfg["out_dir"]) / "seeds_stage2.npz"
    seeds_body = np.load(seeds_path)["states"] if seeds_path.exists() else None

    # ---- build the job list: (branch idx, r); r=0 only for the first branch of each seed ----
    branches, jobs, seen_seed = [], [], set()
    for f in branch_files:
        d = np.load(f)
        # skip the big per-branch image arrays (`frames`, `anchor_frame`): ~1.4 MB/branch, unused here
        br = {k: d[k] for k in d.files if k not in ("frames", "anchor_frame")}
        br["name"] = f"{f.parent.parent.name}/{f.name}"
        br["path"] = f
        assert "latents" in br, f"{f} has no latents: rerun closed_loop_rollout.py with --save-latents"
        assert br["latents"].shape == (3 + h_max, 4, 32, 32), br["latents"].shape
        assert br["agent_state_hist"].shape == (h_max + 1, 4), br["agent_state_hist"].shape
        assert br["exec_abs_targets"].shape == (h_max * FI, 2)
        assert br["lead_acts"].shape == (LEAD, 2)
        sid = int(br["seed_index"])
        if seeds_body is not None:  # the branch was really generated from this seeds_stage2 row
            assert np.allclose(br["seed_state"], seeds_body[sid], atol=1e-9), f"{f}: seed_state != seeds row"
        bi = len(branches)
        branches.append(br)
        for r in rounds:
            if r == 0 and sid in seen_seed:
                continue
            jobs.append((bi, r))
        seen_seed.add(sid)
    n_rows = len(jobs)
    print(f"label: {n_rows} rows ({len(seen_seed)} seeds)")

    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    max_steps = LEAD + h_max * FI + B + 60
    out = {
        k: []
        for k in [
            "latent_stack",
            "frame_steps",
            "agent_state",
            "cov",
            "cov_source",
            "sim_cov",
            "y",
            "pre_success",
            "wall_artifact",
            "wall_gap_min",
            "sigma",
            "seed_id",
            "cand",
            "branch",
            "r",
            "replay_cog_state",
        ]
    }
    stats = {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    t0, t_replay, t_cont = time.time(), 0.0, 0.0

    for c0 in range(0, n_rows, args.batch):
        chunk = jobs[c0 : c0 + args.batch]
        envs = [
            gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                observation_width=96,
                observation_height=96,
                max_episode_steps=max_steps,
            )
            for _ in chunk
        ]
        ta = time.time()
        rp = [
            replay_row(e, branches[bi], r, branches[bi]["name"], stats)
            for e, (bi, r) in zip(envs, chunk, strict=True)
        ]
        t_replay += time.time() - ta

        alive = [i for i, x in enumerate(rp) if not x["terminated"]]
        y = np.ones(len(chunk), bool)
        wall_gap = np.array([x["wall_min"] for x in rp], np.float64)
        if alive:
            ta = time.time()
            res = rollout_batch(
                policy,
                pre,
                post,
                np.stack([rp[i]["cog"] for i in alive]),
                device=device,
                na=na,
                episode_steps=B,
                seed=0,
                batch_start=c0,
                lockstep224=False,
                wall_pen_px=wall_pen,
                prepared_envs=[envs[i] for i in alive],
            )
            assert res["queue_maxlen"] == na, res["queue_maxlen"]
            t_cont += time.time() - ta
            for j, i in enumerate(alive):
                s = int(res["success_step"][j])
                y[i] = s != -1 and s <= B
                wall_gap[i] = min(wall_gap[i], float(res["wall_gap_min"][j]))
        for e in envs:
            e.close()

        for i, (bi, r) in enumerate(chunk):
            br = branches[bi]
            out["latent_stack"].append(br["latents"][r : r + 3])
            out["frame_steps"].append([r * FI - 10, r * FI - 5, r * FI])
            out["agent_state"].append(br["agent_state_hist"][r])
            out["cov"].append(float(br["coverage"][r - 1]) if r >= 1 else rp[i]["cov_anchor"])
            out["cov_source"].append("imagined" if r >= 1 else "sim")
            out["sim_cov"].append(rp[i]["sim_cov"])
            out["y"].append(bool(y[i]))
            out["pre_success"].append(bool(rp[i]["terminated"]))
            out["wall_artifact"].append(bool(wall_gap[i] < -wall_pen))
            out["wall_gap_min"].append(float(wall_gap[i]))
            out["sigma"].append(float(br["inject_sigma"]))
            out["seed_id"].append(int(br["seed_index"]))
            out["cand"].append(int(br["path"].stem.split("_cand")[1]))
            out["branch"].append(br["name"])
            out["r"].append(r)
            out["replay_cog_state"].append(rp[i]["cog"])
        el = time.time() - t0
        print(
            f"  rows {c0 + len(chunk)}/{n_rows}  y={np.mean(out['y']):.3f}  "
            f"{el / (c0 + len(chunk)):.2f} s/row  (replay {t_replay:.0f}s, continuation {t_cont:.0f}s)",
            flush=True,
        )

    fs = np.asarray(out["frame_steps"], np.int64)
    assert (np.diff(fs, axis=1) == FI).all(), "frame_steps spacing != 5"

    # ---- stored cov vs the worker's own frames_{r}.npz (only possible while the work dir survives) ----
    cov_cmp, cov_max = 0, 0.0
    by_name = {b["name"]: b for b in branches}
    rr, cov_arr = np.asarray(out["r"]), np.asarray(out["cov"], np.float32)
    for i, name in enumerate(out["branch"]):
        r = int(rr[i])
        if r == 0:
            continue
        sig_dir = name.split("/")[0]
        wf = root / sig_dir / "work" / f"frames_{r}.npz"
        if not wf.exists():
            continue
        br = by_name[name]
        k = int(br["meta"][0])
        b_idx = int(br["seed_index"]) * k + int(out["cand"][i])
        w = np.load(wf)["cov"][b_idx]
        cov_max = max(cov_max, abs(float(w) - float(cov_arr[i])))
        cov_cmp += 1

    dst = root / "labels.npz"
    np.savez_compressed(
        dst,
        latent_stack=np.stack(out["latent_stack"]).astype(np.float16),
        frame_steps=fs,
        agent_state=np.asarray(out["agent_state"], np.float64),
        cov=cov_arr,
        cov_source=np.asarray(out["cov_source"]),
        sim_cov=np.asarray(out["sim_cov"], np.float32),
        y=np.asarray(out["y"], bool),
        pre_success=np.asarray(out["pre_success"], bool),
        wall_artifact=np.asarray(out["wall_artifact"], bool),
        wall_gap_min=np.asarray(out["wall_gap_min"], np.float32),
        sigma=np.asarray(out["sigma"], np.float32),
        seed_id=np.asarray(out["seed_id"], np.int64),
        cand=np.asarray(out["cand"], np.int64),
        branch=np.asarray(out["branch"]),
        r=rr.astype(np.int64),
        replay_cog_state=np.asarray(out["replay_cog_state"], np.float64),
        state_convention_replay_cog_state="cog",
        B=np.int64(B),
        na=np.int64(na),
    )
    y_, pre_, wa_ = (np.asarray(out[k], bool) for k in ("y", "pre_success", "wall_artifact"))
    print(
        f"\nasserts passed: anchor_state {stats['anchor_checks']} (max err {stats['max_anchor_err']:.2e}), "
        f"agent_state_hist {stats['agent_checks']} (max err {stats['max_agent_err']:.2e} px)"
    )
    print(f"stored cov vs worker frames_r.npz cov: {cov_cmp} compared, max |diff| {cov_max:.2e}")
    print(f"rows {n_rows}: y {y_.mean():.3f}, pre_success {pre_.sum()}, wall_artifact {wa_.sum()}")
    for r in rounds:
        m = rr == r
        print(f"  r={r}: n={m.sum():4d} y={y_[m].mean():.3f} pre_success={pre_[m].sum()} wall={wa_[m].sum()}")
    print(f"wrote {dst} in {time.time() - t0:.1f}s ({(time.time() - t0) / n_rows:.2f} s/row)")


if __name__ == "__main__":
    main()
