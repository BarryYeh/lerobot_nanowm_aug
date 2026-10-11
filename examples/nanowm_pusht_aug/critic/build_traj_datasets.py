#!/usr/bin/env python
"""Full-trajectory fine-tune data from the replicated branches (arm F) and its self-imitation control (arm S).

For each replicated branch (sailor_expand selection.json) the anchor is rebuilt with the labeler's replay (asserted),
then, `--reps` times:
  F  the branch's R*5 executed actions replayed, then the base policy (no injected noise) until success
  S  the base policy (no injected noise) from the anchor
Every simulator step is logged (native 96px frame before the step, the executed absolute-target action). A run is
kept only if it succeeds within label.B steps of the anchor; the episode is anchor -> success step. Per branch the
same number of runs k = min(--k, #F successes, #S successes) is drawn at random from each arm's successes, so the two
datasets hold the same episode count per anchor (S episodes are longer: S carries more frames, stated up front).
Both use the base policy's torch seed per batch; F and S rows are separate runs (not paired by noise).

Output: LeRobot datasets <repo-f> and <repo-s> (same features as datagen/pack_closed_loop.py, fps 10) and
<out>/traj_datasets.json (counts, lengths).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent))
from config_io import load_config  # noqa: E402
from datagen.pack_closed_loop import FEATURES  # noqa: E402
from expand_select import branch_path  # noqa: E402
from label_imagined_branches import FI, LEAD, replay_row  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

BATCH = 100


def roll(cfg, policy, pre, post, jobs, arm, R, reps, seed, device) -> list[list]:
    """-> per job, a list of successful episodes [(frames (T,96,96,3) u8, actions (T,2) f32)]."""
    B = int(cfg["label"]["B"])
    off = R * FI if arm == "F" else 0
    rows = [(j, k) for j in range(len(jobs)) for k in range(reps)]
    out = [[] for _ in jobs]
    stats = {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    t0 = time.time()
    for c0 in range(0, len(rows), BATCH):
        chunk = rows[c0 : c0 + BATCH]
        envs = [
            gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                observation_width=96,
                observation_height=96,
                max_episode_steps=LEAD + R * FI + B + 60,
            )
            for _ in chunk
        ]
        logs = []
        for e in envs:
            log = []
            orig = e.step

            def rec(a, _orig=orig, _log=log):
                res = _orig(a)
                _log.append(
                    (
                        np.array(a, np.float32),
                        np.asarray(res[0]["pixels"], np.uint8),
                        bool(res[4].get("is_success", False)),
                    )
                )
                return res

            e.step = rec
            logs.append(log)
        rp = [
            replay_row(e, jobs[j]["br"], R if arm == "F" else 0, jobs[j]["name"], stats)
            for e, (j, _k) in zip(envs, chunk, strict=True)
        ]
        alive = [i for i, x in enumerate(rp) if not x["terminated"]]
        if alive:
            rollout_batch(
                policy,
                pre,
                post,
                np.stack([rp[i]["cog"] for i in alive]),
                device=device,
                na=int(cfg["label"]["na"]),
                episode_steps=B - off,
                seed=seed,
                batch_start=c0,
                lockstep224=False,
                wall_pen_px=float(cfg["label"]["wall_pen_px"]),
                prepared_envs=[envs[i] for i in alive],
            )
        for i, (j, _k) in enumerate(chunk):
            log = logs[i]
            assert len(log) >= LEAD, (len(log), LEAD)
            steps = log[LEAD:]  # after the lead-in: step t takes action a_t from frame obs_t
            succ = next((t + 1 for t, (_a, _o, s) in enumerate(steps) if s), -1)
            if succ < 0 or succ > B:
                continue
            frames = [log[LEAD - 1][1]] + [o for _a, o, _s in steps[: succ - 1]]
            acts = [a for a, _o, _s in steps[:succ]]
            out[j].append((np.stack(frames), np.stack(acts)))
        for e in envs:
            e.close()
        print(f"  {arm}: {min(c0 + BATCH, len(rows))}/{len(rows)} runs ({time.time() - t0:.0f}s)", flush=True)
    return out


def write(repo: str, episodes: list) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = HF_LEROBOT_HOME / repo
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(
        repo_id=repo, fps=10, features=FEATURES, root=root, robot_type="pusht_sim", use_videos=True
    )
    n = 0
    for frames, acts in episodes:
        for f, a in zip(frames, acts, strict=True):
            ds.add_frame({"observation.image": f, "action": a.astype(np.float32), "task": "pusht"})
            n += 1
        ds.save_episode()
    ds.finalize()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sailor_expand.yaml"))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--repo-f", default="local/pusht_traj_f")
    ap.add_argument("--repo-s", default="local/pusht_traj_s")
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    base = Path(cfg["out_dir"]).expanduser()
    out = base / "traj"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = int(max(cfg["stage2"]["query_rounds"]))
    sel = [r for r in json.loads((base / "selection.json").read_text())["rows"] if r["replicated"]]
    jobs = []
    for r in sel:
        z = np.load(branch_path(base / "stage2", r["branch"]))
        jobs.append(
            {
                "name": r["branch"],
                "seed": r["seed"],
                "br": {k: z[k] for k in z.files if k not in ("frames", "anchor_frame", "latents")},
            }
        )
    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    F = roll(cfg, policy, pre, post, jobs, "F", R, a.reps, a.seed, device)
    S = roll(cfg, policy, pre, post, jobs, "S", R, a.reps, a.seed, device)
    rng = np.random.default_rng(a.seed)
    epf, eps, per = [], [], []
    for j, job in enumerate(jobs):
        k = min(a.k, len(F[j]), len(S[j]))
        pf = [F[j][i] for i in sorted(rng.choice(len(F[j]), k, replace=False))] if k else []
        ps = [S[j][i] for i in sorted(rng.choice(len(S[j]), k, replace=False))] if k else []
        epf += pf
        eps += ps
        per.append(
            {
                "branch": job["name"],
                "anchor": job["seed"],
                "f_success": len(F[j]),
                "s_success": len(S[j]),
                "k": k,
                "f_len": [len(x[1]) for x in pf],
                "s_len": [len(x[1]) for x in ps],
            }
        )
    nf, ns = write(a.repo_f, epf), write(a.repo_s, eps)
    summ = {
        "reps": a.reps,
        "k": a.k,
        "seed": a.seed,
        "repo_f": a.repo_f,
        "repo_s": a.repo_s,
        "episodes": len(epf),
        "frames_f": nf,
        "frames_s": ns,
        "median_len_f": float(np.median([len(x[1]) for x in epf])),
        "median_len_s": float(np.median([len(x[1]) for x in eps])),
        "f_success_rate": float(np.mean([p["f_success"] for p in per]) / a.reps),
        "s_success_rate": float(np.mean([p["s_success"] for p in per]) / a.reps),
        "per_branch": per,
    }
    (out / "traj_datasets.json").write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k != "per_branch"})


if __name__ == "__main__":
    main()
