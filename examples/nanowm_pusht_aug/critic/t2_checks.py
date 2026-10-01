#!/usr/bin/env python
"""SAILOR v1 phase T2: simulator checks that settle the stage-1 noise design (pre-registration §6).

Two subcommands (split so the heavy simulation needs only the lerobot venv, while the report needs
only numpy + matplotlib and can run in any venv that has them, e.g. the NanoWM venv):

  sim     base policy, na=10, 300-step episodes, N reset starts `env.reset(seed=760000+i)`:
            arm clean : sigma = 0
            arm noise : sigma = 0.1, noise_hold = 1, perturb_mode = "prefix", perturb_p = episode_steps
                        (whole episode noised), max_z = 3
          the two arms are PAIRED (same starts, same policy seed). Also captures the clean chunks the
          clean arm produces at replan boundaries t <= 100 (via rollout_batch's `chunk_hook`) and
          measures how large 0.1 sigma noise is in pixels (T2(c)). Writes episodes.npz and t2c_pixels.json.
  report  T2(a) paired success rate + decision, T2(b) failure timing / class balance, PNGs,
          t2a_success.json, t2b_timing.json, T2_SUMMARY.md (reads episodes.npz + t2c_pixels.json).

    uv run --locked python examples/nanowm_pusht_aug/critic/t2_checks.py sim
    third_party/nano-world-model/.venv/bin/python examples/nanowm_pusht_aug/critic/t2_checks.py report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_CRITIC = Path(__file__).resolve().parent
_PROJ = _CRITIC.parent
_REPO = _PROJ.parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "attribution"))
sys.path.insert(0, str(_PROJ / "eval"))
sys.path.insert(0, str(_CRITIC))
from datagen.provenance import provenance_entry  # noqa: E402

from critic.labels import NOT_EXISTING, success_T, v_target  # noqa: E402

DEFAULT_OUT = Path("~/datastorage/nanowm_pusht_aug/value_model/sailor_v1/t2_checks").expanduser()
COMMON_CFG = _REPO / "configs" / "nanowm_pusht_common.yaml"
EPISODE_STEPS = 300
NA = 10
B = 200
GAMMA = 0.995
QUERY_TS = list(range(10, 101, 10))
SIGMA_NOISE = 0.1
MAX_Z = 3.0
NOISE_HOLD = 1


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    return o


def write_json(path: Path, obj: dict, args) -> None:
    obj = dict(obj)
    obj["provenance"] = provenance_entry(args, stage=path.name)
    path.write_text(json.dumps(_jsonable(obj), indent=2))
    print(f"wrote {path}")


# ----------------------------------------------------------------------------------------------
# sim
# ----------------------------------------------------------------------------------------------


def reset_starts(seed0: int, n: int) -> np.ndarray:
    """Same construction as collect_value_states.make_starts' reset branch: gym_pusht's own reset
    distribution via `env.reset(seed=seed0+i)`, read back as a CoG state with `sim_state`."""
    import gym_pusht  # noqa: F401
    import gymnasium as gym
    from ood_failure_map import sim_state

    env = gym.make("gym_pusht/PushT-v0")
    out = []
    for i in range(n):
        env.reset(seed=seed0 + i)
        out.append(sim_state(env.unwrapped))
    env.close()
    return np.asarray(out, np.float64)


def run_arm(name, policy, pre, post, starts, *, device, seed, batch, sigma_val, noise_seed, capture):
    from sim_rollout import rollout_batch

    n = len(starts)
    sigma = np.full(n, sigma_val, np.float64)
    perturb_p = np.full(n, EPISODE_STEPS, np.int64)  # prefix mode: p >= episode_steps == whole episode
    noise_rng = np.random.default_rng(noise_seed)
    ss = np.full(n, -1, np.int64)
    wall = np.zeros(n, bool)
    cov = np.zeros((n, EPISODE_STEPS + 1), np.float32)
    cap = {k: [] for k in ("ep", "t", "chunk", "pos", "vel", "state")}
    from ood_failure_map import sim_state

    t0 = time.time()
    for b0 in range(0, n, batch):
        nb = min(batch, n - b0)
        sl = slice(b0, b0 + nb)

        def hook(t, idx, chunks, envs, b0=b0):
            if not capture or t > 100:
                return
            for j, i in enumerate(idx):
                u = envs[i].unwrapped
                cap["ep"].append(b0 + i)
                cap["t"].append(t)
                cap["chunk"].append(chunks[j])
                cap["pos"].append(np.asarray(u.agent.position, np.float64))
                cap["vel"].append(np.asarray(u.agent.velocity, np.float64))
                cap["state"].append(sim_state(u))

        out = rollout_batch(
            policy,
            pre,
            post,
            starts[sl],
            device=device,
            na=NA,
            episode_steps=EPISODE_STEPS,
            seed=seed,
            batch_start=b0,
            sigma=sigma[sl],
            perturb_p=perturb_p[sl],
            max_z=MAX_Z,
            noise_hold=NOISE_HOLD,
            noise_rng=noise_rng,
            lockstep224=False,
            frame_grid_stride=5,
            frame_cutoff_step=0,
            perturb_mode="prefix",
            chunk_hook=hook if capture else None,
        )
        assert out["queue_maxlen"] == NA and out["perturb_mode"] == "prefix"
        assert out["cov_per_step"].shape == (nb, EPISODE_STEPS + 1)
        ss[sl] = out["success_step"]
        wall[sl] = out["wall_artifact"]
        cov[sl] = out["cov_per_step"]
        el = time.time() - t0
        done = b0 + nb
        print(
            f"[{name}] {done}/{n}  success so far {(ss[:done] != -1).mean():.3f}  "
            f"{el / done:.2f} s/ep  eta {el / done * (n - done) / 60:.1f} min",
            flush=True,
        )
    arrays = {"ss": ss, "wall": wall, "cov": cov}
    if capture:
        arrays["cap"] = {
            "ep": np.asarray(cap["ep"], np.int64),
            "t": np.asarray(cap["t"], np.int64),
            "chunk": np.asarray(cap["chunk"], np.float64),
            "pos": np.asarray(cap["pos"], np.float64),
            "vel": np.asarray(cap["vel"], np.float64),
            "state": np.asarray(cap["state"], np.float64),
        }
    return arrays


def level0_decomposition(chunks, pos, vel):
    """Mirror of `inject_noise` at level 0 that also records WHY a step changed: `capped` (the
    max_z cap shrank the relative action) and `clipped` (the AGENT_BOX clip moved the target)."""
    from closed_loop_rollout import AGENT_BOX, SIGMA_PX
    from datagen import action_convert as ac

    n, hz, _ = chunks.shape
    out = chunks.copy()
    capped = np.zeros((n, hz), bool)
    clipped = np.zeros((n, hz), bool)
    for b in range(n):
        pp, vv = pos[b].copy(), vel[b].copy()
        for t in range(hz):
            rel = chunks[b, t] - pp
            z = float(np.linalg.norm(rel / SIGMA_PX))
            scale = min(1.0, MAX_Z / max(z, 1e-9))
            capped[b, t] = scale < 1.0
            rel = rel * scale
            raw = pp + rel
            out[b, t] = np.clip(raw, *AGENT_BOX)
            clipped[b, t] = bool(np.any(out[b, t] != raw))
            pp, vv = ac.pd_step(pp, vv, out[b, t])
    return out, capped, clipped


def pd_track(pos, vel, chunks):
    """Agent positions after each of the chunk's steps under the kinematic PD model (n,hz,2)."""
    from datagen import action_convert as ac

    n, hz, _ = chunks.shape
    traj = np.zeros((n, hz, 2))
    for b in range(n):
        p, v = pos[b].copy(), vel[b].copy()
        for t in range(hz):
            p, v = ac.pd_step(p, v, chunks[b, t])
            traj[b, t] = p
    return traj


def _q(x):
    x = np.asarray(x, np.float64).ravel()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


def chunk_stats(chunks, pos, vel, seed):
    """All T2(c) numbers for one set of clean chunks (n, NA, 2)."""
    from closed_loop_rollout import inject_noise

    rng = np.random.default_rng(seed)
    n = len(chunks)
    lvl0, capped, clipped = level0_decomposition(chunks, pos, vel)
    lvl0_ref = inject_noise(chunks, pos, vel, 0.0, MAX_Z, NOISE_HOLD, np.random.default_rng(seed + 1))
    assert np.allclose(lvl0, lvl0_ref, atol=1e-9), "mirror of inject_noise(level=0) diverged"
    noised = inject_noise(chunks, pos, vel, SIGMA_NOISE, MAX_Z, NOISE_HOLD, rng)

    d_clean = np.linalg.norm(noised - chunks, axis=-1)  # vs the raw policy chunk
    d_pure = np.linalg.norm(noised - lvl0, axis=-1)  # noise only (cap / clip effect cancelled)
    d_lvl0 = np.linalg.norm(lvl0 - chunks, axis=-1)
    changed = d_lvl0 > 1e-6

    traj_clean = pd_track(pos, vel, chunks)
    traj_noised = pd_track(pos, vel, noised)
    traj_lvl0 = pd_track(pos, vel, lvl0)
    dev = np.linalg.norm(traj_noised - traj_clean, axis=-1)  # (n, hz) realised agent deviation
    dev_pure = np.linalg.norm(traj_noised - traj_lvl0, axis=-1)
    dev0 = np.linalg.norm(traj_lvl0 - traj_clean, axis=-1)
    return {
        "n_chunks": n,
        "target_dev_px__noised_vs_raw": _q(d_clean),
        "target_dev_px__noised_vs_raw__per_step_mean": d_clean.mean(0),
        "target_dev_px__noised_vs_level0": _q(d_pure),
        "agent_dev_px_pd__noised_vs_raw__all_steps": _q(dev),
        "agent_dev_px_pd__noised_vs_raw__final_step": _q(dev[:, -1]),
        "agent_dev_px_pd__noised_vs_raw__per_step_mean": dev.mean(0),
        "agent_dev_px_pd__noised_vs_level0__final_step": _q(dev_pure[:, -1]),
        "level0": {
            "frac_steps_changed": float(changed.mean()),
            "frac_chunks_any_step_changed": float(changed.any(1).mean()),
            "frac_steps_capped_by_max_z": float(capped.mean()),
            "frac_steps_clipped_by_agent_box": float(clipped.mean()),
            "frac_chunks_any_capped": float(capped.any(1).mean()),
            "frac_chunks_any_clipped": float(clipped.any(1).mean()),
            "change_px_among_changed_steps": _q(d_lvl0[changed]) if changed.any() else None,
            "agent_dev_px_pd__level0_vs_raw__final_step": _q(dev0[:, -1]),
        },
        "_arrays": {"d_clean": d_clean, "dev_final": dev[:, -1]},
    }


def sim_replay(states, chunks, noised_seed):
    """Replay clean vs noised chunk in the REAL simulator from each captured CoG state (set_state zeroes
    nothing about the block but the agent starts with whatever velocity reset leaves; both arms share
    it). Returns per-chunk agent-position and block-CoG deviation after the 10 executed steps."""
    import gym_pusht  # noqa: F401
    import gymnasium as gym
    from closed_loop_rollout import inject_noise
    from ood_failure_map import set_state, sim_state

    env = gym.make("gym_pusht/PushT-v0")
    env.reset(seed=0)
    u = env.unwrapped
    pos = states[:, 0:2]
    # agent velocity at the boundary is not in `states`; use 0 for both the noise tracker and the replay
    noised = inject_noise(
        chunks, pos, np.zeros_like(pos), SIGMA_NOISE, MAX_Z, NOISE_HOLD, np.random.default_rng(noised_seed)
    )
    agent_dev, block_dev = [], []
    for k in range(len(chunks)):
        end = []
        for ch in (chunks[k], noised[k]):
            env.reset(seed=0)
            set_state(u, states[k])
            for a in ch:
                env.step(a)
            end.append((np.asarray(u.agent.position, np.float64), sim_state(u)[2:4]))
        agent_dev.append(np.linalg.norm(end[0][0] - end[1][0]))
        block_dev.append(np.linalg.norm(end[0][1] - end[1][1]))
    env.close()
    return np.asarray(agent_dev), np.asarray(block_dev)


def cmd_sim(args) -> None:
    import torch
    from config_io import load_config
    from policy_io import load_policy

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(COMMON_CFG, out_dir_copy=False)
    ns = int(cfg["seed_namespace"]["sailor_v1_t2_checks"])
    assert ns == 760000, ns
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(cfg["ckpt"]["base"])
    if not ckpt.is_absolute():
        ckpt = _REPO / ckpt
    n = args.n
    print(f"T2 sim: n={n} batch={args.batch} seed={ns} ckpt={ckpt.resolve()} -> {out_dir}", flush=True)

    policy, pre, post = load_policy(ckpt, device=device)
    starts = reset_starts(ns, n)

    t0 = time.time()
    clean = run_arm(
        "clean", policy, pre, post, starts, device=device, seed=ns, batch=args.batch,
        sigma_val=0.0, noise_seed=ns + 1, capture=True,
    )  # fmt: skip
    np.savez_compressed(
        out_dir / "arm_clean.npz", ss=clean["ss"], wall=clean["wall"], cov=clean["cov"], starts=starts,
        **{f"cap_{k}": v for k, v in clean["cap"].items()},
    )  # fmt: skip
    noise = run_arm(
        "noise", policy, pre, post, starts, device=device, seed=ns, batch=args.batch,
        sigma_val=SIGMA_NOISE, noise_seed=ns + 1, capture=False,
    )  # fmt: skip
    np.savez_compressed(
        out_dir / "episodes.npz",
        starts=starts, seeds=ns + np.arange(n),
        success_step_clean=clean["ss"], success_step_noise=noise["ss"],
        wall_artifact_clean=clean["wall"], wall_artifact_noise=noise["wall"],
        cov_per_step_clean=clean["cov"], cov_per_step_noise=noise["cov"],
        cap_ep=clean["cap"]["ep"], cap_t=clean["cap"]["t"], cap_chunk=clean["cap"]["chunk"],
        cap_pos=clean["cap"]["pos"], cap_vel=clean["cap"]["vel"], cap_state=clean["cap"]["state"],
    )  # fmt: skip
    (out_dir / "arm_clean.npz").unlink()
    print(f"simulation done in {(time.time() - t0) / 60:.1f} min -> {out_dir / 'episodes.npz'}", flush=True)

    # ---- T2(c) ----
    from closed_loop_rollout import AGENT_BOX, SIGMA_PX

    cap = clean["cap"]
    m = len(cap["ep"])
    perm = np.random.default_rng(ns + 3).permutation(m)
    sel = np.sort(perm[: min(200, m)])
    full = chunk_stats(cap["chunk"], cap["pos"], cap["vel"], seed=ns + 4)
    samp = chunk_stats(cap["chunk"][sel], cap["pos"][sel], cap["vel"][sel], seed=ns + 4)
    agent_dev, block_dev = sim_replay(cap["state"][sel], cap["chunk"][sel], ns + 5)
    arrays = {
        "d_clean_all": full["_arrays"]["d_clean"].ravel(),
        "dev_final_all": full["_arrays"]["dev_final"],
        "sim_agent_dev": agent_dev,
        "sim_block_dev": block_dev,
    }
    np.savez_compressed(out_dir / "t2c_arrays.npz", **arrays)
    for d in (full, samp):
        d.pop("_arrays")
    write_json(
        out_dir / "t2c_pixels.json",
        {
            "SIGMA_PX": SIGMA_PX,
            "sigma_units": "level * SIGMA_PX * z, z ~ N(0,I) per step (noise_hold=1); level=0.1",
            "level": SIGMA_NOISE,
            "noise_hold": NOISE_HOLD,
            "max_z": MAX_Z,
            "AGENT_BOX": AGENT_BOX,
            "chunk_source": "clean chunks captured by rollout_batch(chunk_hook) in the clean arm at "
            "replan boundaries t in {0,10,...,100} (same na=10 chunks inject_noise sees in stage 1)",
            "n_captured": m,
            "sample200_indices_seed": ns + 3,
            "sample200": samp,
            "all_captured": full,
            "sim_replay_sample200": {
                "note": "clean vs 0.1-sigma-noised chunk replayed in gym_pusht from the captured CoG state "
                "(agent velocity not restored: 0 in both); deviation after the 10 executed steps",
                "agent_pos_dev_px": _q(agent_dev),
                "block_cog_dev_px": _q(block_dev),
            },
        },
        args,
    )


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * p))


def success_stats(ss):
    s = ss[ss != -1]
    return {
        "n_success": int(len(s)),
        "mean_success_step": float(s.mean()) if len(s) else None,
        "median_success_step": float(np.median(s)) if len(s) else None,
        "p10_p90_success_step": [float(np.percentile(s, 10)), float(np.percentile(s, 90))]
        if len(s)
        else None,
    }


def class_balance(ss, wall, ts=QUERY_TS):
    rows = {}
    for t in ts:
        T = success_T(ss, t)
        alive = (T != NOT_EXISTING) & ~wall
        y = (T >= 0) & (T <= B)
        n_alive = int(alive.sum())
        rows[str(t)] = {
            "n_alive": n_alive,
            "n_pre_success": int(((T == 0) & alive).sum()),
            "n_y1": int((y & alive).sum()),
            "frac_y1": float((y & alive).sum() / n_alive) if n_alive else None,
            "n_success_after_window": int(((T > B) & alive).sum()),
            "mean_v_target_beta0": float(v_target(T[alive], 0.0, B, GAMMA).mean()) if n_alive else None,
        }
    return rows


def cmd_report(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out).expanduser()
    d = np.load(out / "episodes.npz")
    ssc, ssn = d["success_step_clean"], d["success_step_noise"]
    wc, wn = d["wall_artifact_clean"], d["wall_artifact_noise"]
    covc, covn = d["cov_per_step_clean"], d["cov_per_step_noise"]
    n = len(ssc)
    assert covc.shape == (n, EPISODE_STEPS + 1) and covn.shape == covc.shape
    for ss_, cov_ in ((ssc, covc), (ssn, covn)):  # audit: success steps carry coverage > 0.95, filled after
        ok = np.nonzero(ss_ != -1)[0]
        assert (cov_[ok, ss_[ok]] > 0.95).all() and (cov_[ok, -1] > 0.95).all()

    # ------------------------------------------------ (a)
    sc, sn = (ssc != -1).astype(np.float64), (ssn != -1).astype(np.float64)
    delta = sn - sc
    rng = np.random.default_rng(760002)
    idx = rng.integers(0, n, size=(10000, n))
    bd = delta[idx].mean(1) * 100
    bc, bn = sc[idx].mean(1) * 100, sn[idx].mean(1) * 100
    both = int(((sc == 1) & (sn == 1)).sum())
    neither = int(((sc == 0) & (sn == 0)).sum())
    clean_only = int(((sc == 1) & (sn == 0)).sum())
    noise_only = int(((sc == 0) & (sn == 1)).sum())
    d_pp = float(delta.mean() * 100)
    lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
    backup = abs(d_pp) <= 2.0
    decision = (
        "BACKUP adopted (|delta| <= 2 pp): noise the whole episode, keep every query, stage-2 continuation also noised"
        if backup
        else "PREFIX rule stands (|delta| > 2 pp): noise only chunks before p, keep queries with t >= p"
    )
    edges = np.arange(0, 301, 10)
    hc, _ = np.histogram(ssc[ssc != -1], bins=edges)
    hn, _ = np.histogram(ssn[ssn != -1], bins=edges)
    a = {
        "n_episodes": n,
        "arms": {
            "clean": "sigma=0",
            "noise": f"sigma={SIGMA_NOISE}, noise_hold={NOISE_HOLD}, perturb_mode=prefix, perturb_p=episode_steps (whole episode), max_z={MAX_Z}",
        },
        "success_rate_clean": float(sc.mean()),
        "success_rate_noise": float(sn.mean()),
        "success_rate_ci95_clean_pp": [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))],
        "success_rate_ci95_noise_pp": [float(np.percentile(bn, 2.5)), float(np.percentile(bn, 97.5))],
        "paired_diff_noise_minus_clean_pp": d_pp,
        "paired_diff_ci95_pp": [lo, hi],
        "bootstrap": {"resamples": 10000, "unit": "episode (paired)", "seed": 760002},
        "discordant": {
            "both_success": both,
            "neither": neither,
            "clean_only_success": clean_only,
            "noise_only_success": noise_only,
            "mcnemar_exact_two_sided_p": mcnemar_exact_p(clean_only, noise_only),
        },
        "decision_rule": "|delta| <= 2 pp -> backup (noise everywhere); else prefix rule (pre-reg §6 T2(a))",
        "abs_delta_pp": abs(d_pp),
        "backup_adopted": bool(backup),
        "decision": decision,
        "ci_entirely_within_pm2pp": bool(lo >= -2.0 and hi <= 2.0),
        "success_step_clean": success_stats(ssc),
        "success_step_noise": success_stats(ssn),
        "success_step_hist_edges": edges,
        "success_step_hist_clean": hc,
        "success_step_hist_noise": hn,
        "wall_artifact_count": {"clean": int(wc.sum()), "noise": int(wn.sum())},
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 4
    ax[0].bar(edges[:-1] - w / 2 + 5, hc, width=w, label=f"clean (n={int(sc.sum())})")
    ax[0].bar(edges[:-1] + w / 2 + 5, hn, width=w, label=f"noise 0.1σ (n={int(sn.sum())})")
    ax[0].set_xlabel("success_step (10-step bins)")
    ax[0].set_ylabel("episodes")
    ax[0].legend()
    ax[0].set_title("success-step distribution")
    ax[1].hist(bd, bins=60)
    ax[1].axvline(d_pp, color="k")
    ax[1].axvspan(-2, 2, color="g", alpha=0.15, label="±2 pp")
    ax[1].set_xlabel("paired Δ success (noise − clean), pp")
    ax[1].set_title(f"bootstrap Δ = {d_pp:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "t2a_success_step_hist.png", dpi=130)
    plt.close(fig)
    write_json(out / "t2a_success.json", a, args)

    # ------------------------------------------------ (b)
    fail = ssc == -1
    dc = np.diff(covc.astype(np.float64), axis=1)  # dc[:, s-1] = cov[s] - cov[s-1]
    inc = dc > 0.01
    last_inc = np.where(
        inc.any(1), EPISODE_STEPS - np.argmax(inc[:, ::-1], axis=1), 0
    )  # step index s of last increase
    argmax_step = np.argmax(covc, axis=1)
    final_cov = covc[:, -1]
    fdec = last_inc[fail]
    nf = int(fail.sum())
    frac_late = float((fdec > 100).mean()) if nf else None
    es = np.arange(0, 301, 10)
    cb_clean = class_balance(ssc, wc)
    cb_noise = class_balance(ssn, wn)
    cb_clean_all = class_balance(ssc, np.zeros(n, bool))
    succ = ssc != -1
    b = {
        "n_episodes": n,
        "n_failed_clean": nf,
        "failed_decision_time": {
            "definition": "last step s with cov[s]-cov[s-1] > 0.01 (0 if coverage never rose by >0.01 in one step)",
            "mean": float(fdec.mean()) if nf else None,
            "median": float(np.median(fdec)) if nf else None,
            "frac_gt_100": frac_late,
            "frac_gt_200": float((fdec > 200).mean()) if nf else None,
            "flag_ge_20pct_after_t100": bool(frac_late is not None and frac_late >= 0.20),
            "hist_edges": es,
            "hist": np.histogram(fdec, bins=es)[0] if nf else None,
        },
        "failed_argmax_cov_step": {
            "mean": float(argmax_step[fail].mean()) if nf else None,
            "median": float(np.median(argmax_step[fail])) if nf else None,
            "frac_gt_100": float((argmax_step[fail] > 100).mean()) if nf else None,
        },
        "failed_final_coverage": {
            "mean": float(final_cov[fail].mean()) if nf else None,
            "median": float(np.median(final_cov[fail])) if nf else None,
            "p10_p90": [float(np.percentile(final_cov[fail], 10)), float(np.percentile(final_cov[fail], 90))]
            if nf
            else None,
        },
        "success_timing_clean": {
            "frac_of_all_episodes_success_step_le_100": float((succ & (ssc <= 100)).mean()),
            "frac_of_all_episodes_success_step_le_200": float((succ & (ssc <= 200)).mean()),
            "frac_of_all_episodes_success_step_le_300": float((succ & (ssc <= 300)).mean()),
            "frac_of_successes_step_le_100": float((ssc[succ] <= 100).mean()) if succ.any() else None,
            "frac_of_successes_step_le_200": float((ssc[succ] <= 200).mean()) if succ.any() else None,
        },
        "class_balance_clean_arm_wall_excluded": cb_clean,
        "class_balance_clean_arm_incl_wall": cb_clean_all,
        "class_balance_noise_arm_wall_excluded__informative_only": cb_noise,
        "class_balance_definition": "among episodes not yet terminated at t (success_step == -1 or >= t), "
        "non-wall-artifact: y=1 iff t <= success_step <= t+200 (pre_success success_step==t counts as y=1). "
        "The noise-arm block is NOT a stage-1 label (its continuation is noised).",
    }
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].hist(fdec, bins=es)
    ax[0].axvline(100, color="r")
    ax[0].set_title(f"failed eps: decision time (n={nf}, >100: {frac_late:.1%})" if nf else "no failures")
    ax[1].hist(argmax_step[fail], bins=es)
    ax[1].set_title("failed eps: step of max coverage")
    ax[2].hist(final_cov[fail], bins=30)
    ax[2].set_title("failed eps: final coverage")
    ts = QUERY_TS
    ax[3].plot(ts, [cb_clean[str(t)]["frac_y1"] for t in ts], "o-", label="clean arm (stage-1 labels)")
    ax[3].plot(ts, [cb_noise[str(t)]["frac_y1"] for t in ts], "s--", label="noise arm (informative)")
    ax[3].set_ylim(0, 1)
    ax[3].set_xlabel("query t")
    ax[3].set_ylabel("frac y=1")
    ax[3].set_title("class balance per t")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "t2b_failure_timing.png", dpi=130)
    plt.close(fig)
    write_json(out / "t2b_timing.json", b, args)

    # (c) figure from saved arrays
    c = json.loads((out / "t2c_pixels.json").read_text())
    ca = np.load(out / "t2c_arrays.npz")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].hist(ca["d_clean_all"], bins=60)
    ax[0].set_title("|noised − clean| target, px/step (all captured)")
    ax[1].hist(ca["dev_final_all"], bins=60)
    ax[1].set_title("agent deviation after 10 steps (PD model), px")
    ax[2].hist(ca["sim_agent_dev"], bins=30, alpha=0.7, label="agent")
    ax[2].hist(ca["sim_block_dev"], bins=30, alpha=0.7, label="block CoG")
    ax[2].legend()
    ax[2].set_title("real-sim replay, 200 chunks, px after 10 steps")
    fig.tight_layout()
    fig.savefig(out / "t2c_pixels.png", dpi=130)
    plt.close(fig)

    write_summary(out, a, b, c)


def write_summary(out: Path, a: dict, b: dict, c: dict) -> None:
    S = c["sample200"]
    F = c["all_captured"]
    sim = c["sim_replay_sample200"]

    def q(x):
        return f"mean {x['mean']:.2f} / median {x['median']:.2f} / p95 {x['p95']:.2f}"

    L0 = F["level0"]
    lines = [
        "# SAILOR v1 — T2 simulator checks",
        "",
        f"- episodes per arm: {a['n_episodes']} (paired starts `env.reset(seed=760000+i)`, policy seed 760000, na=10, 300 steps)",
        f"- results dir: `{out}`",
        "",
        "## (a) noise vs clean success rate",
        "",
        "| arm | success rate | 95% CI (bootstrap, pp) | n success | mean success_step | median |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| clean (σ=0) | {a['success_rate_clean']:.3%} | [{a['success_rate_ci95_clean_pp'][0]:.1f}, {a['success_rate_ci95_clean_pp'][1]:.1f}] | {a['success_step_clean']['n_success']} | {a['success_step_clean']['mean_success_step']:.1f} | {a['success_step_clean']['median_success_step']:.1f} |",
        f"| noise (σ=0.1, hold 1, whole episode) | {a['success_rate_noise']:.3%} | [{a['success_rate_ci95_noise_pp'][0]:.1f}, {a['success_rate_ci95_noise_pp'][1]:.1f}] | {a['success_step_noise']['n_success']} | {a['success_step_noise']['mean_success_step']:.1f} | {a['success_step_noise']['median_success_step']:.1f} |",
        "",
        f"- paired Δ (noise − clean) = **{a['paired_diff_noise_minus_clean_pp']:+.2f} pp**, 95% CI [{a['paired_diff_ci95_pp'][0]:+.2f}, {a['paired_diff_ci95_pp'][1]:+.2f}] pp (10000 paired episode bootstraps)",
        f"- discordant pairs: clean-only success {a['discordant']['clean_only_success']}, noise-only success {a['discordant']['noise_only_success']} "
        f"(both {a['discordant']['both_success']}, neither {a['discordant']['neither']}); McNemar exact two-sided p = {a['discordant']['mcnemar_exact_two_sided_p']:.3g}",
        f"- wall artifacts: clean {a['wall_artifact_count']['clean']}, noise {a['wall_artifact_count']['noise']}",
        f"- **decision (pre-reg §6, |Δ| ≤ 2 pp → backup): |Δ| = {a['abs_delta_pp']:.2f} pp → {a['decision']}**",
        f"- CI entirely within ±2 pp: {a['ci_entirely_within_pm2pp']}",
        "",
        "![success step](t2a_success_step_hist.png)",
        "",
        "## (b) failure timing (clean arm)",
        "",
        f"- failed episodes: {b['n_failed_clean']} / {b['n_episodes']}",
        f"- decision time (last step with a >0.01 coverage jump): mean {b['failed_decision_time']['mean']:.1f}, median {b['failed_decision_time']['median']:.1f}; "
        f"**fraction > 100 = {b['failed_decision_time']['frac_gt_100']:.1%}** (> 200: {b['failed_decision_time']['frac_gt_200']:.1%}) → "
        f"**{'FLAG (≥ 20%): note the query-range limitation in the revision log' if b['failed_decision_time']['flag_ge_20pct_after_t100'] else 'no flag (< 20%)'}**",
        f"- step of max coverage (failed): mean {b['failed_argmax_cov_step']['mean']:.1f}, median {b['failed_argmax_cov_step']['median']:.1f}, frac > 100 = {b['failed_argmax_cov_step']['frac_gt_100']:.1%}",
        f"- final coverage (failed): mean {b['failed_final_coverage']['mean']:.3f}, median {b['failed_final_coverage']['median']:.3f}",
        f"- success timing (share of ALL clean episodes): ≤100: {b['success_timing_clean']['frac_of_all_episodes_success_step_le_100']:.1%}, "
        f"≤200: {b['success_timing_clean']['frac_of_all_episodes_success_step_le_200']:.1%}, ≤300: {b['success_timing_clean']['frac_of_all_episodes_success_step_le_300']:.1%}",
        "",
        "Expected class balance per query t (clean arm; episodes not yet terminated at t, wall artifacts excluded; y=1 ⇔ success within 200 steps after t):",
        "",
        "| t | n alive | n y=1 | frac y=1 | pre_success | success after window (T>200) | mean v_target(β=0) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in QUERY_TS:
        r = b["class_balance_clean_arm_wall_excluded"][str(t)]
        lines.append(
            f"| {t} | {r['n_alive']} | {r['n_y1']} | {r['frac_y1']:.3f} | {r['n_pre_success']} | {r['n_success_after_window']} | {r['mean_v_target_beta0']:.3f} |"
        )
    lines += [
        "",
        "![failure timing](t2b_failure_timing.png)",
        "",
        "## (c) pixel offset of 0.1σ noise",
        "",
        f"- `SIGMA_PX` = {np.round(np.asarray(c['SIGMA_PX']), 3).tolist()} px/step (level 0.1 → std {0.1 * np.asarray(c['SIGMA_PX'])[0]:.2f} / {0.1 * np.asarray(c['SIGMA_PX'])[1]:.2f} px per step per axis); `max_z`={c['max_z']}, `AGENT_BOX`={c['AGENT_BOX']}",
        f"- chunks: clean chunks captured from the clean arm at boundaries t ∈ {{0,…,100}} (n={c['n_captured']}); 200-chunk random sample shown first, all captured second",
        f"- per-step |noised − clean target| (px): sample200 {q(S['target_dev_px__noised_vs_raw'])}; all {q(F['target_dev_px__noised_vs_raw'])}",
        f"- same, noise only (vs level-0 output): sample200 {q(S['target_dev_px__noised_vs_level0'])}",
        f"- realised agent deviation after PD tracking of the 10 executed steps (px, kinematic PD model with real pos/vel): final step sample200 {q(S['agent_dev_px_pd__noised_vs_raw__final_step'])}; all {q(F['agent_dev_px_pd__noised_vs_raw__final_step'])}; over all 10 steps (all): {q(F['agent_dev_px_pd__noised_vs_raw__all_steps'])}",
        f"- real-sim replay (200 chunks, clean vs noised, px after 10 steps): agent {q(sim['agent_pos_dev_px'])}; block CoG {q(sim['block_cog_dev_px'])}",
        f"- **cap/clip at level 0 (N15.3)**, all captured chunks: steps changed by `inject_noise(level=0)` = {L0['frac_steps_changed']:.2%} "
        f"(capped by max_z {L0['frac_steps_capped_by_max_z']:.2%}, clipped by AGENT_BOX {L0['frac_steps_clipped_by_agent_box']:.2%}); "
        f"chunks with ≥1 changed step = {L0['frac_chunks_any_step_changed']:.2%}; change among changed steps: "
        + (q(L0["change_px_among_changed_steps"]) if L0["change_px_among_changed_steps"] else "n/a")
        + f"; agent deviation at step 10 from level-0 vs raw: {q(L0['agent_dev_px_pd__level0_vs_raw__final_step'])}",
        f"- sample200 level 0: steps changed {S['level0']['frac_steps_changed']:.2%}, chunks any {S['level0']['frac_chunks_any_step_changed']:.2%}",
        "",
        "![pixels](t2c_pixels.png)",
        "",
    ]
    (out / "T2_SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {out / 'T2_SUMMARY.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sim", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(DEFAULT_OUT))
    sub.choices["sim"].add_argument("--n", type=int, default=1000)
    sub.choices["sim"].add_argument("--batch", type=int, default=50)
    args = p.parse_args()
    {"sim": cmd_sim, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
