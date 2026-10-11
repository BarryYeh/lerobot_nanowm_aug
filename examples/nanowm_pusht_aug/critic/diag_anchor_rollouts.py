#!/usr/bin/env python
"""Diagnostic: did the fine-tuned policies pick up the collected behaviour AT THE ANCHORS IT CAME FROM?

For every anchor with a replicated branch (sailor_expand selection.json), the exact anchor state is rebuilt with the
labeler's replay (place_seed + lead-in, asserted against anchor_state / agent_state_hist[0]) and each policy is rolled
`--reps` times for label.B steps with the label continuation noise (cont_perturb). Arms:
  base, ctrl, img, real, img_x10, real_x10   the policy from the anchor
  branch                                      the anchor's first replicated branch replayed (R*5 steps), then base
All arms share the torch seed per batch and a freshly seeded noise rng, so they are paired by (anchor, rep).
Success time is counted from the anchor (branch: R*5 + continuation) and only within label.B of the anchor.

Outputs (<out>): anchor_rollouts.npz (success steps per arm), anchor_rollouts.json + .md (per-arm summary, paired
deltas vs ctrl / base over anchors), and side-by-side videos for the --videos anchors with the largest replicated
speed-up: [imagined branch | branch in sim | img policy | img_x10 policy | ctrl policy], each tile the rep whose
success time is closest to that arm's median at that anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from scipy.stats import wilcoxon

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent))
from config_io import load_config  # noqa: E402
from expand_select import branch_path  # noqa: E402
from label_imagined_branches import FI, LEAD, place_seed, replay_row  # noqa: E402
from ood_failure_map import sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402
from topscore_video import GREEN, GREY, RED, WHITE, tile  # noqa: E402

BATCH = 100
ARM_TAGS = {
    "ctrl": "pusht_imgonly",
    "img": "pusht_expand_img_plus_real",
    "real": "pusht_expand_real_plus_real",
    "img_x10": "pusht_expand_img_x10_plus_real",
    "real_x10": "pusht_expand_real_x10_plus_real",
    "traj_f": "pusht_traj_f_plus_real",
    "traj_s": "pusht_traj_s_plus_real",
}
VIDEO_ARMS = ("img", "img_x10", "ctrl")


def ckpt_of(cfg: dict, arm: str) -> Path:
    if arm in ("base", "branch"):
        return Path(cfg["ckpt"][cfg["ckpt_key"]])
    ft = cfg["expand"]["finetune"]
    base = Path(cfg["out_dir"]).expanduser() / "finetune" / "train"
    return (
        base
        / f"{ARM_TAGS[arm]}_d{ft['drop']}_s{ft['seed']}"
        / "checkpoints"
        / f"{int(ft['steps']):06d}"
        / "pretrained_model"
    )


def load_branch(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files if k not in ("frames", "anchor_frame", "latents")}


def run_arm(cfg, arm, jobs, R, reps, seed, device, log_rows: set[int]) -> tuple[np.ndarray, dict]:
    """jobs: list of (anchor_id, branch dict). Returns success steps (n_anchor, reps) and logs for row indices."""
    lab = cfg["label"]
    B, cp = int(lab["B"]), lab["cont_perturb"]
    policy, pre, post = load_policy(ckpt_of(cfg, arm), device=device)
    noise_rng = np.random.default_rng(seed + 1)  # the same stream for every arm -> paired
    rows = [(a, k) for a in range(len(jobs)) for k in range(reps)]
    succ = np.full(len(rows), -1, np.int64)
    logs, stats = {}, {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    r = R if arm == "branch" else 0
    off = R * FI if arm == "branch" else 0
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
        rp = [
            replay_row(e, jobs[a][1], r, jobs[a][1]["name"], stats)
            for e, (a, _k) in zip(envs, chunk, strict=True)
        ]
        alive = [i for i, x in enumerate(rp) if not x["terminated"]]
        for i, x in enumerate(rp):
            if x["terminated"]:
                succ[c0 + i] = off  # success inside the replay (deterministic)
        for i in alive:
            if c0 + i in log_rows:
                logs[c0 + i] = {"acts": [], "states": []}
                e, orig = envs[i], envs[i].step

                def rec(act, _e=e, _orig=orig, _log=logs[c0 + i]):
                    res = _orig(act)
                    _log["acts"].append(np.array(act, np.float64))
                    _log["states"].append(sim_state(_e.unwrapped))
                    return res

                e.step = rec
        if alive:
            res = rollout_batch(
                policy,
                pre,
                post,
                np.stack([rp[i]["cog"] for i in alive]),
                device=device,
                na=int(lab["na"]),
                episode_steps=B,
                seed=seed,
                batch_start=c0,
                sigma=np.full(len(alive), float(cp["sigma"])),
                perturb_p=np.full(len(alive), B, np.int64),
                perturb_mode="prefix",
                max_z=float(cp["max_z"]),
                noise_hold=int(cp["noise_hold"]),
                noise_rng=noise_rng,
                lockstep224=False,
                wall_pen_px=float(lab["wall_pen_px"]),
                prepared_envs=[envs[i] for i in alive],
            )
            for j, i in enumerate(alive):
                s = int(res["success_step"][j])
                succ[c0 + i] = off + s if s > 0 else -1
        for e in envs:
            e.close()
    succ[succ > B] = -1  # same window as the anchor (B steps from the anchor)
    return succ.reshape(len(jobs), reps), logs


def render(br: dict, prefix: np.ndarray, cont: dict) -> tuple[np.ndarray, np.ndarray]:
    """224px frames from the anchor: `prefix` actions (branch replay or none), then the logged continuation."""
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        observation_width=224,
        observation_height=224,
        max_episode_steps=10_000,
    )
    u = env.unwrapped
    place_seed(env, int(br["seed_index"]), br["seed_state"])
    for a in br["lead_acts"]:
        env.step(a)
    frames, cov = [np.asarray(u.get_obs())], [float(u._get_coverage())]
    done = False
    for a in prefix:
        _o, _r, term, _tr, info = env.step(a)
        frames.append(np.asarray(u.get_obs()))
        cov.append(float(info["coverage"]))
        if term:
            done = True
            break
    if not done:
        for a, st in zip(cont["acts"], cont["states"], strict=True):
            _o, _r, term, _tr, info = env.step(a)
            err = float(np.abs(sim_state(u) - st).max())
            assert err <= 1e-6, f"224px replay drifted from the logged rollout by {err:.2e}"
            frames.append(np.asarray(u.get_obs()))
            cov.append(float(info["coverage"]))
            if term and info.get("is_success", False):
                break
    env.close()
    return np.stack(frames), np.asarray(cov)


def tile_at(frames, cov, t, title, succ):
    ti = min(t, len(frames) - 1)
    if succ > 0 and t >= succ:
        return tile(frames[ti], title, f"SUCCESS @ {succ} (cov {cov[ti]:.2f})", GREEN, GREEN)
    if succ < 0 and t >= len(frames) - 1:
        return tile(frames[ti], title, f"FAIL in {len(frames) - 1} steps", RED, RED)
    return tile(frames[ti], title, f"t={ti} cov={cov[ti]:.2f}", WHITE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sailor_expand.yaml"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=920000)
    ap.add_argument(
        "--arms", nargs="+", default=["base", "ctrl", "img", "real", "img_x10", "real_x10", "branch"]
    )
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument(
        "--video-arms", nargs="+", default=list(VIDEO_ARMS), help="policy columns after the branch"
    )
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    base = Path(cfg["out_dir"]).expanduser()
    out = a.out or base / "diag_anchor"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = int(max(cfg["stage2"]["query_rounds"]))
    sel = [r for r in json.loads((base / "selection.json").read_text())["rows"] if r["replicated"]]
    first = {}
    for (
        r
    ) in sel:  # the anchor's first replicated branch (strongest discovery q first, as candidates are ordered)
        first.setdefault(r["seed"], r)
    anchors = sorted(first)
    jobs = []
    for s in anchors:
        br = load_branch(branch_path(base / "stage2", first[s]["branch"]))
        br["name"] = first[s]["branch"]
        jobs.append((s, br))
    speedup = np.array([first[s]["rep"]["t0"] - first[s]["rep"]["t1"] for s in anchors])
    vid_idx = list(np.argsort(-speedup)[: a.videos])
    log_rows = {ai * a.reps + k for ai in vid_idx for k in range(a.reps)}
    print(
        f"{len(anchors)} anchors x {a.reps} reps, arms {a.arms}, videos at anchors {[anchors[i] for i in vid_idx]}"
    )

    res, logs, t0 = {}, {}, time.time()
    for arm in a.arms:
        res[arm], logs[arm] = run_arm(cfg, arm, jobs, R, a.reps, a.seed, device, log_rows)
        ss = res[arm]
        print(
            f"  {arm:9s} success {np.mean(ss > 0):.3f}  median steps {np.median(ss[ss > 0]):.0f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    np.savez(out / "anchor_rollouts.npz", anchors=np.array(anchors), **res)

    def per_anchor(ss):
        p = (ss > 0).mean(1)
        t = np.array([np.median(x[x > 0]) if (x > 0).any() else np.nan for x in ss])
        return p, t

    summ = {"anchors": anchors, "reps": a.reps, "seed": a.seed, "arms": {}, "paired": {}}
    pa = {arm: per_anchor(res[arm]) for arm in res}
    for arm, (p, t) in pa.items():
        summ["arms"][arm] = {"success": float(p.mean()), "median_steps": float(np.nanmedian(t))}
    lines = [
        f"# Policies from the {len(anchors)} anchors of the replicated branches ({a.reps} reps each)",
        "",
        "| arm | success | median of per-anchor median steps |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {v['success']:.3f} | {v['median_steps']:.0f} |" for k, v in summ["arms"].items()]
    lines += [
        "",
        "| comparison | mean Δ success (over anchors) | median Δ steps | Wilcoxon p (steps) | anchors faster / slower |",
        "|---|---|---|---|---|",
    ]
    pairs = [
        (x, "ctrl") for x in ("img", "real", "img_x10", "real_x10", "traj_f", "traj_s", "branch", "base")
    ]
    for x, y in pairs + [("ctrl", "base"), ("traj_f", "traj_s")]:
        if x not in pa or y not in pa:
            continue
        dp = pa[x][0] - pa[y][0]
        dt = pa[x][1] - pa[y][1]
        m = ~np.isnan(dt)
        pv = float(wilcoxon(dt[m]).pvalue) if m.sum() > 2 and (dt[m] != 0).any() else float("nan")
        summ["paired"][f"{x}-{y}"] = {
            "dp": float(dp.mean()),
            "dt_median": float(np.median(dt[m])),
            "wilcoxon_p": pv,
            "faster": int((dt[m] < 0).sum()),
            "slower": int((dt[m] > 0).sum()),
        }
        lines.append(
            f"| {x} − {y} | {dp.mean():+.3f} | {np.median(dt[m]):+.1f} | {pv:.3g} | "
            f"{int((dt[m] < 0).sum())} / {int((dt[m] > 0).sum())} |"
        )
    (out / "anchor_rollouts.json").write_text(json.dumps(summ, indent=1, default=float))
    (out / "anchor_rollouts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- videos
    for ai in vid_idx:
        s, br = jobs[ai]
        z = np.load(branch_path(base / "stage2", br["name"]))
        tiles = []
        img = z["frames"]  # imagined, one per FI steps
        tiles.append(("img", None))
        for arm in ("branch", *a.video_arms):
            if arm not in res:
                continue
            ss = res[arm][ai]
            ok = ss[ss > 0]
            target = np.median(ok) if len(ok) else -1
            k = int(np.argmin(np.abs(ss - target))) if len(ok) else 0
            row = ai * a.reps + k
            prefix = br["exec_abs_targets"][: R * FI] if arm == "branch" else np.zeros((0, 2))
            cont = logs[arm].get(row, {"acts": [], "states": []})
            frames, cov = render(br, prefix, cont)
            tiles.append((arm, (frames, cov, int(ss[k]))))
        T = max(len(v[0]) for _k, v in tiles if v is not None) + 10
        title = {
            "branch": "BRANCH in sim (+base)",
            "img": "IMG policy (A x1)",
            "img_x10": "IMG policy (A x10)",
            "ctrl": "CTRL policy (C)",
            "real": "REAL policy (B x1)",
            "real_x10": "REAL policy (B x10)",
            "base": "BASE policy",
            "traj_f": "TRAJ-F policy (branch traj)",
            "traj_s": "TRAJ-S policy (self-imit.)",
        }
        vid = []
        for t in range(T):
            row_tiles = []
            for arm, v in tiles:
                if v is None:
                    kk = min(t // FI, len(img))
                    f = z["anchor_frame"] if kk == 0 else img[kk - 1]
                    if t > len(img) * FI:
                        row_tiles.append(
                            tile(
                                (img[-1] * 0.45 + 140).astype(np.uint8),
                                "IMAGINED branch",
                                f"imagination ended @ t={len(img) * FI}",
                                GREY,
                            )
                        )
                    else:
                        row_tiles.append(tile(f, "IMAGINED branch", f"t={t} round {kk}/{len(img)}", WHITE))
                else:
                    row_tiles.append(tile_at(v[0], v[1], t, title[arm], v[2]))
            frame = np.concatenate(row_tiles, axis=1)
            head = np.zeros((22, frame.shape[1], 3), np.uint8)
            cv2.putText(
                head,
                f"anchor {s}  branch {br['name']}  (each tile: the rep closest to that arm's median)",
                (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                WHITE,
                1,
                cv2.LINE_AA,
            )
            vid.append(np.concatenate([head, frame], axis=0))
        mp4 = out / f"anchor{s:04d}_{br['name'].replace('/', '_').replace('.npz', '')}.mp4"
        imageio.mimwrite(mp4, vid, fps=10, codec="libx264")
        print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
