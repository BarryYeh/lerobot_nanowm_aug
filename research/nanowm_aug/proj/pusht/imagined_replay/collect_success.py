#!/usr/bin/env python
"""imagined_replay step 2 (and the step-0 pilot): near-miss starts -> K base-policy rollouts each -> keep successes.

  starts   Deploy the base policy (no noise, na 10) from env.reset(seed) and find near-miss starts t_nm with the
           last-mile rule (`nwaug.datagen.selection.qualifying`: image-only coverage on the 5-step grid, t >= t_min,
           estimate in [cov_lo, cov_hi), stalled, episode not ended) plus t_nm <= t_max. Each start keeps the episode seed and the
           executed actions a[0:t_nm], so a branch rebuilds the exact state by replay (no state placement).
  branch   For each start: K envs replay a[0:t_nm] (bit-identical, checked), then the base policy continues with a
           fresh action queue. Every sim step from t_nm-19 on is recorded (224px frame + action). A success at T
           gives t_start = T - 5*ceil((T - t_nm)/5) (the first 5-step back-off from T that is <= t_nm) and the pairs
           s = t_start+1 .. T-5; the saved trajectory holds frames F(t_start-15 .. T-5) and actions a[t_start-15 .. T-1].
           Sharded with --part/--parts (one process per GPU).
  summary  Pilot statistics and the number of starts needed for a target dose.

Conventions: F(t) = 224px frame after t executed steps (F(0) = right after reset); a[t] = the action executed at
step t, taking F(t) to F(t+1); T = success_step = number of executed steps when info["is_success"].

    uv run --extra training python research/nanowm_aug/proj/pusht/imagined_replay/collect_success.py starts --out <dir> --n 20
    CUDA_VISIBLE_DEVICES=6 uv run --extra training python .../collect_success.py branch --out <dir> --k 16 --part 0 --parts 2
    uv run --extra training python .../collect_success.py summary --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from ir_common import BASE_CKPT, BRANCH_SEED, CTX, NA, SOURCE_SEED, load_starts, start

from nwaug.attribution.policy_io import load_policy
from nwaug.datagen.selection import qualifying
from nwaug_envs.pusht.coverage import coverage_from_frame
from nwaug_envs.pusht.deploy import (
    EPISODE_STEPS,
    FI,
    close,
    make_envs,
    policy_act,
    reset_policy,
    source_batch,
)
from nwaug_envs.pusht.state import read_cog

REC_BACK = CTX + FI - 1  # t_start - 15 >= t_nm - 19
SAMPLEABLE_REAL = 24002  # sampleable real frames at drop_n_last_frames 8 (method1 criteria.md)


# ---------------------------------------------------------------------------------------------------------- starts
def cmd_starts(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "starts.npz"
    if dest.exists():
        raise SystemExit(f"{dest} exists; delete it to recollect")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(a.ckpt, device=dev)
    rng = np.random.default_rng(a.seed + 1)
    got, n_ep, t0 = [], 0, time.time()
    n_success_src = 0
    while len(got) < a.n:
        if n_ep >= a.max_episodes:
            raise SystemExit(f"only {len(got)}/{a.n} near-miss starts after {n_ep} episodes")
        seeds = list(range(a.seed + n_ep, a.seed + n_ep + a.batch))
        res = source_batch(policy, pre, post, seeds, dev, torch_seed=a.seed + n_ep, na=NA)
        steps = res["steps"]
        for j, s in enumerate(seeds):
            ss = int(res["success"][j])
            n_success_src += ss > 0
            last = (
                int(res["end"][j]) // FI + 1
            )  # grid points that exist (the episode had not ended before them)
            est = np.full(len(steps), -1.0)
            # grid points past t_max stay at -1, so they never qualify
            for g in range(max(0, int(np.searchsorted(steps, a.t_min)) - 2), min(last, a.t_max // FI + 1)):
                est[g] = coverage_from_frame(res["frames"][j, g])["coverage_cal"]
            for g in qualifying(
                steps,
                est,
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
                    t_nm = int(steps[g])
                    got.append(
                        {
                            "env_seed": s,
                            "t_nm": t_nm,
                            "est": float(est[g]),
                            "cog": res["cog"][j, g].copy(),
                            "prefix": res["acts"][j, :t_nm].copy(),
                            "src_success": ss,
                        }
                    )
        n_ep += a.batch
        print(f"  {n_ep} episodes: {len(got)}/{a.n} near-miss starts ({time.time() - t0:.0f}s)", flush=True)
    prov = {
        "rule": {
            k: getattr(a, k) for k in ("t_min", "t_max", "cov_lo", "cov_hi", "stall", "per_episode", "gap")
        },
        "seed": a.seed,
        "ckpt": str(a.ckpt),
        "na": NA,
        "episode_steps": EPISODE_STEPS,
        "source_episodes": n_ep,
        "source_success": int(n_success_src),
    }
    t_nm = np.array([g["t_nm"] for g in got], np.int64)
    prefix = np.zeros((len(got), int(t_nm.max()), 2))
    for i, g in enumerate(got):
        prefix[i, : g["t_nm"]] = g["prefix"]
    np.savez(
        dest,
        env_seed=np.array([g["env_seed"] for g in got], np.int64),
        t_nm=t_nm,
        est_cov=np.array([g["est"] for g in got]),
        cog_at_t_nm=np.stack([g["cog"] for g in got]),
        prefix_actions=prefix,  # row i valid for [:t_nm[i]]
        src_success=np.array([g["src_success"] for g in got], np.int64),
        provenance_json=json.dumps(prov),
    )
    print(f"wrote {dest}: {len(got)} starts from {len({g['env_seed'] for g in got})} of {n_ep} episodes")


# ---------------------------------------------------------------------------------------------------------- branch
def back_off(T: int, t_nm: int) -> int:  # noqa: N803
    """First 5-step back-off from T that is <= t_nm."""
    return T - FI * math.ceil((T - t_nm) / FI)


def branch_start(policy, pre, post, st: dict, k: int, device: str, torch_seed: int) -> list[dict]:
    t_nm, prefix = st["t_nm"], st["prefix"]
    rec0 = t_nm - REC_BACK
    e96, e224 = make_envs(k)
    for i in range(k):
        e96[i].reset(seed=st["env_seed"])
        e224[i].reset(seed=st["env_seed"])
    frames = [[] for _ in range(k)]  # F(rec0), F(rec0+1), ...
    acts = np.zeros((k, EPISODE_STEPS, 2))
    obs96 = [None] * k

    def step(i: int, t: int, a: np.ndarray):
        if t >= rec0:
            frames[i].append(np.asarray(e224[i].unwrapped.get_obs()))
        o, _r, term, trunc, info = e96[i].step(a)
        e224[i].step(a)
        obs96[i] = o["pixels"]
        acts[i, t] = a
        return term or trunc, bool(info.get("is_success", False))

    for t in range(t_nm):
        for i in range(k):
            ended, _ = step(i, t, prefix[t])
            assert not ended, f"start env_seed={st['env_seed']}: replay ended at t={t + 1} < t_nm={t_nm}"
    for i in range(k):
        c = read_cog(e96[i].unwrapped)
        assert np.allclose(c, st["cog"], atol=1e-9), (
            f"replay drift env_seed={st['env_seed']}: {c} vs {st['cog']}"
        )
        assert np.allclose(c, read_cog(e224[i].unwrapped), atol=1e-9), "lockstep drift at t_nm"

    reset_policy(policy, torch_seed, NA)
    done = np.zeros(k, bool)
    T = np.full(k, -1, np.int64)
    for t in range(t_nm, EPISODE_STEPS):
        act = policy_act(policy, pre, post, obs96, device)
        for i in range(k):
            if done[i]:
                continue
            ended, ok = step(i, t, act[i])
            if ended:
                done[i] = True
                if ok:
                    T[i] = t + 1
        if done.all():
            break
    close(e96, e224)

    out = []
    for i in range(k):
        r = {"branch": i, "success": bool(T[i] > 0), "T": int(T[i])}
        if T[i] > 0:
            ts = back_off(int(T[i]), t_nm)
            r["t_start"] = ts
            r["n_pairs"] = max(0, int(T[i]) - FI - ts)
            if r["n_pairs"] > 0:
                lo = ts - CTX
                hi = int(T[i]) - FI  # last frame kept
                r["frames224"] = np.stack(frames[i][lo - rec0 : hi - rec0 + 1])
                r["frame_steps"] = np.arange(lo, hi + 1)
                r["actions"] = acts[i, lo : int(T[i])].copy()
                r["action_steps"] = np.arange(lo, int(T[i]))
        out.append(r)
    return out


def cmd_branch(a) -> None:
    starts = load_starts(a.out / "starts.npz")
    mine = [i for i in range(len(starts["t_nm"])) if i % a.parts == a.part]
    traj_dir = a.out / "traj"
    traj_dir.mkdir(parents=True, exist_ok=True)
    log_dir = a.out / "branch_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(a.ckpt, device=dev)
    t0 = time.time()
    for n, si in enumerate(mine):
        rec_path = log_dir / f"start{si:04d}.json"
        if rec_path.exists():  # resumable: one finished start = one log file
            continue
        st = start(starts, si)
        torch_seed = a.branch_seed + si
        res = branch_start(policy, pre, post, st, a.k, dev, torch_seed)
        rows = []
        for r in res:
            if "frames224" in r:
                np.savez_compressed(
                    traj_dir / f"start{si:04d}_b{r['branch']:02d}.npz",
                    frames224=r["frames224"],
                    frame_steps=r["frame_steps"],
                    actions=r["actions"],
                    action_steps=r["action_steps"],
                    t_nm=st["t_nm"],
                    T=r["T"],
                    t_start=r["t_start"],
                    env_seed=st["env_seed"],
                    start_id=si,
                    branch=r["branch"],
                    torch_seed=torch_seed,
                )
            rows.append({k: v for k, v in r.items() if k in ("branch", "success", "T", "t_start", "n_pairs")})
        rec = {
            "start_id": si,
            "env_seed": st["env_seed"],
            "t_nm": st["t_nm"],
            "k": a.k,
            "torch_seed": torch_seed,
        }
        rec["branches"] = rows
        rec_path.write_text(json.dumps(rec, indent=1))
        n_ok = sum(r["success"] for r in rows)
        print(
            f"  [part {a.part}] {n + 1}/{len(mine)} start {si}: {n_ok}/{a.k} success, "
            f"pairs {sum(r.get('n_pairs', 0) for r in rows)} ({time.time() - t0:.0f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------------------- summary
def cmd_summary(a) -> None:
    recs = [json.loads(p.read_text()) for p in sorted((a.out / "branch_log").glob("start*.json"))]
    d = np.load(a.out / "starts.npz")
    prov = json.loads(str(d["provenance_json"]))
    k = recs[0]["k"]
    rate = np.array([sum(b["success"] for b in r["branches"]) / r["k"] for r in recs])
    succ = [b for r in recs for b in r["branches"] if b["success"]]
    dur = np.array([b["T"] - r["t_nm"] for r in recs for b in r["branches"] if b["success"]])
    pairs = np.array([b["n_pairs"] for b in succ])
    pairs_per_start = np.array([sum(b.get("n_pairs", 0) for b in r["branches"]) for r in recs])
    q = lambda x: [float(v) for v in np.percentile(x, [10, 25, 50, 75, 90])] if len(x) else []  # noqa: E731
    summ = {
        "n_starts": len(recs),
        "k": k,
        "source_episodes": prov["source_episodes"],
        "source_success": prov["source_success"],
        "distinct_source_episodes": len({int(s) for s in d["env_seed"]}),
        "success_rate_mean": float(rate.mean()),
        "success_rate_per_start": [float(x) for x in rate],
        "starts_with_zero_success": int((rate == 0).sum()),
        "T_minus_t_nm_p10_25_50_75_90": q(dur),
        "successes": len(succ),
        "successes_with_pairs": int((pairs > 0).sum()),
        "pairs_per_traj_p10_25_50_75_90": q(pairs[pairs > 0]),
        "pairs_per_traj_mean": float(pairs[pairs > 0].mean()) if (pairs > 0).any() else 0.0,
        "pairs_per_start_mean": float(pairs_per_start.mean()),
        "pairs_total": int(pairs.sum()),
        "fidelity_filter_pass_rate": None,  # needs replay_imagine.py (step 3)
    }
    dose = {}
    for p in (0.005, 0.01, 0.02, 0.05):
        need = round(SAMPLEABLE_REAL * p / (1 - p))
        dose[f"{p:.1%}"] = {
            "pairs": need,
            "starts_at_k": math.ceil(need / max(summ["pairs_per_start_mean"], 1e-9)),
        }
    summ["dose_vs_24002_real_frames"] = dose
    (a.out / "pilot_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("starts")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--ckpt", type=Path, default=Path(BASE_CKPT))
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=SOURCE_SEED)
    s.add_argument("--batch", type=int, default=50)
    s.add_argument("--max-episodes", type=int, default=2000)
    s.add_argument("--t-min", type=int, default=100)
    # a pair needs T - t_nm >= 6 within the 300-step episode, so a start past 290 can never yield one
    s.add_argument("--t-max", type=int, default=290)
    s.add_argument("--cov-lo", type=float, default=0.85)
    s.add_argument("--cov-hi", type=float, default=0.945)
    s.add_argument("--stall", type=float, default=0.005)
    s.add_argument("--per-episode", type=int, default=3)
    s.add_argument("--gap", type=int, default=20)
    b = sub.add_parser("branch")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--ckpt", type=Path, default=Path(BASE_CKPT))
    b.add_argument("--k", type=int, default=16)
    b.add_argument("--part", type=int, default=0)
    b.add_argument("--parts", type=int, default=1)
    b.add_argument("--branch-seed", type=int, default=BRANCH_SEED)
    m = sub.add_parser("summary")
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"starts": cmd_starts, "branch": cmd_branch, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
