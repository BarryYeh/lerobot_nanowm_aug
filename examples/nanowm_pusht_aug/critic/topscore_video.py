#!/usr/bin/env python
"""Video: the highest-V imagined branches of the SAILOR v1 fine-tune pool next to their REAL rollouts.

Pool = `finetune_pool.py select` output (`<pool>/branch_scores.csv`, one row per test-anchor branch,
scored at the final query round r=8). The top `--top` branches by that score are shown, each as a pair
of tiles: left = the NanoWM imagined frames (`frames`, one per 5-step round, held between rounds; t=0
is the real anchor frame), right = the real simulator.

The real side is the EXACT label run of `label_imagined_branches.py` (row r=8 of `test_stage2/labels.npz`):
the label loop is re-run batch by batch with the same job order, batch size, torch seed and
continuation-noise rng (the outcome depends on batch composition and on the rng draws of every earlier
batch, so the run stops after the last batch that holds a shown row, not earlier). For the shown rows
the 96px env's `step` is wrapped to log every executed action; those actions are then replayed in a
224px env (lead-in -> 40 replayed imagined-branch actions -> base-policy continuation) and every frame
is captured. Asserted: every re-run row's success_step equals labels.npz, and the 224px replay
reproduces the logged states within 1e-6.

    CUDA_VISIBLE_DEVICES=0 uv run --extra training python examples/nanowm_pusht_aug/critic/topscore_video.py \
        --config configs/sailor_v1_reset.yaml --top 6
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent))
from config_io import load_config  # noqa: E402
from finetune_pool import pool_dir  # noqa: E402
from label_imagined_branches import FI, LEAD, place_seed, replay_row  # noqa: E402
from ood_failure_map import sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

HEADER = 44
ROW_HEADER = 22
GREEN, RED, WHITE, GREY = (80, 220, 80), (235, 80, 80), (255, 255, 255), (170, 170, 170)


def top_branches(pool: Path, k: int) -> list[dict]:
    rows = list(csv.DictReader(open(pool / "branch_scores.csv")))
    key = "score_final" if "score_final" in rows[0] else "score_r8"  # older select() wrote *_r8 columns
    rows.sort(key=lambda r: -float(r[key]))
    return [
        {
            "anchor": int(r["anchor"]),
            "sigma": float(r["sigma"]),
            "cand": int(r["cand"]),
            "score": float(r[key]),
        }
        for r in rows[:k]
    ]


def picked_branches(pool: Path, names: list[str]) -> list[dict]:
    """`--branches sigma0.0/seed0038_cand1.npz ...`: these branches in this order; V score from the pool if present."""
    scores = {}
    if (pool / "branch_scores.csv").exists():
        for r in csv.DictReader(open(pool / "branch_scores.csv")):
            key = "score_final" if "score_final" in r else "score_r8"
            scores[(int(r["anchor"]), round(float(r["sigma"]), 4), int(r["cand"]))] = float(r[key])
    out = []
    for n in names:
        sg, rest = n.split("/")
        a, c = int(rest[4:8]), int(rest.split("_cand")[1].split(".")[0])
        sigma = float(sg[5:])
        out.append(
            {
                "anchor": a,
                "sigma": sigma,
                "cand": c,
                "score": scores.get((a, round(sigma, 4), c), float("nan")),
            }
        )
    return out


def label_jobs(root: Path, rounds: list[int]) -> tuple[list[dict], list[tuple[int, int]]]:
    """The labeler's branch list and (branch index, r) job order (label_imagined_branches.main, no subset)."""
    files = []
    for d in sorted(root.glob("sigma*/branches")):
        files += sorted(d.glob("seed*_cand*.npz"))
    branches = []
    for f in files:
        z = np.load(f)
        br = {k: z[k] for k in z.files if k not in ("frames", "anchor_frame", "latents")}
        br["name"], br["path"] = f"{f.parent.parent.name}/{f.name}", f
        branches.append(br)
    keyed = sorted(
        (int(b["seed_index"]), float(b["inject_sigma"]), int(b["path"].stem.split("_cand")[1]), r, bi)
        for bi, b in enumerate(branches)
        for r in rounds
    )
    jobs, seen = [], set()
    for sid, _s, _c, r, bi in keyed:
        if r == 0 and sid in seen:
            continue
        jobs.append((bi, r))
        seen.add(sid)
    return branches, jobs


def rerun_labels(
    cfg,
    root,
    branches,
    jobs,
    targets: set[int],
    device,
    labels_name: str = "labels.npz",
    cont_seed: int | None = None,
) -> dict[int, dict]:
    """Re-run the label loop up to the last batch holding a target job; log the targets' continuation actions.
    `labels_name` / `cont_seed` select a re-label pass (label_imagined_branches --out-name / --cont-seed)."""
    lab = cfg["label"]
    B, na, batch_n = int(lab["B"]), int(lab["na"]), int(lab["batch"])
    cont_seed = int(lab["cont_seed"]) if cont_seed is None else int(cont_seed)
    cp = lab.get("cont_perturb")
    max_steps = LEAD + int(cfg["stage2"]["h_max"]) * FI + B + 60
    ref = np.load(root / labels_name)
    ref_key = {(str(b), int(r)): i for i, (b, r) in enumerate(zip(ref["branch"], ref["r"], strict=True))}
    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    cont_noise_rng = np.random.default_rng(cont_seed + 1)
    stats = {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    last = max(targets)
    out, n_cmp, t0 = {}, 0, time.time()
    for c0 in range(0, last + 1, batch_n):
        chunk = jobs[c0 : c0 + batch_n]
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
        rp = [
            replay_row(e, branches[bi], r, branches[bi]["name"], stats)
            for e, (bi, r) in zip(envs, chunk, strict=True)
        ]
        alive = [i for i, x in enumerate(rp) if not x["terminated"]]
        logs = {}
        for i in alive:
            if c0 + i in targets:
                logs[i] = {"acts": [], "states": []}
                e, orig = envs[i], envs[i].step

                def rec(a, _e=e, _orig=orig, _log=logs[i]):
                    res = _orig(a)
                    _log["acts"].append(np.array(a, np.float64))
                    _log["states"].append(sim_state(_e.unwrapped))
                    return res

                e.step = rec
        succ = np.zeros(len(chunk), np.int64)
        if alive:
            kw = {}
            if cp is not None:
                kw = {
                    "sigma": np.full(len(alive), float(cp["sigma"])),
                    "perturb_p": np.full(len(alive), B, np.int64),
                    "perturb_mode": "prefix",
                    "max_z": float(cp["max_z"]),
                    "noise_hold": int(cp["noise_hold"]),
                    "noise_rng": cont_noise_rng,
                }
            res = rollout_batch(
                policy,
                pre,
                post,
                np.stack([rp[i]["cog"] for i in alive]),
                device=device,
                na=na,
                episode_steps=B,
                seed=cont_seed,
                batch_start=c0,
                lockstep224=False,
                wall_pen_px=float(lab["wall_pen_px"]),
                prepared_envs=[envs[i] for i in alive],
                **kw,
            )
            for j, i in enumerate(alive):
                succ[i] = int(res["success_step"][j])
        for e in envs:
            e.close()
        for i, (bi, r) in enumerate(chunk):
            ri = ref_key[(branches[bi]["name"], r)]
            assert int(ref["success_step"][ri]) == int(succ[i]), (
                f"re-run differs from labels.npz at job {c0 + i} ({branches[bi]['name']} r={r}): "
                f"{int(succ[i])} vs {int(ref['success_step'][ri])}"
            )
            n_cmp += 1
            if c0 + i in targets:
                out[c0 + i] = {
                    "success_step": int(succ[i]),
                    "pre_success": bool(rp[i]["terminated"]),
                    "label_y": bool(ref["y"][ri]),
                    **logs.get(i, {"acts": [], "states": []}),
                }
        print(
            f"  batch {c0 // batch_n + 1}/{last // batch_n + 1}: {n_cmp} rows match labels.npz "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    return out


def render_real(br: dict, cont: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """224px frames from the anchor (t=0) on: 40 replayed branch actions, then the logged continuation.
    Returns frames, real coverage per frame and the success step counted from the anchor (-1 = none)."""
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
    success = -1
    for t, a in enumerate(br["exec_abs_targets"]):
        _o, _r, term, _tr, info = env.step(a)
        frames.append(np.asarray(u.get_obs()))
        cov.append(float(info["coverage"]))
        if term:
            success = t + 1
            break
    if success < 0:
        n_img = len(br["exec_abs_targets"])
        for a, st in zip(cont["acts"], cont["states"], strict=True):
            _o, _r, term, _tr, info = env.step(a)
            err = float(np.abs(sim_state(u) - st).max())
            assert err <= 1e-6, f"224px replay drifted from the logged continuation by {err:.2e}"
            frames.append(np.asarray(u.get_obs()))
            cov.append(float(info["coverage"]))
            if term and info.get("is_success", False):
                success = len(frames) - 1
                break
        if cont["success_step"] > 0:
            assert success == n_img + cont["success_step"], (success, n_img, cont["success_step"])
    env.close()
    return np.stack(frames), np.asarray(cov), success


def put(img, text, y, color=WHITE, scale=0.42, x=4):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def tile(frame, title, line2, color, border=None):
    img = np.zeros((frame.shape[0] + HEADER, frame.shape[1], 3), np.uint8)
    img[HEADER:] = frame
    put(img, title, 16)
    put(img, line2, 36, color)
    if border is not None:
        cv2.rectangle(img, (1, HEADER + 1), (img.shape[1] - 2, img.shape[0] - 2), border, 3)
    return img


def pair_frame(item: dict, t: int) -> np.ndarray:
    img_frames, real, cov, succ, n_img = (
        item["img"],
        item["real"],
        item["cov"],
        item["success"],
        item["n_img"],
    )
    # imagined: t=0 real anchor frame; round k (state after 5k steps) held from t=5k; held after the last round
    k = min(t // FI, len(img_frames))
    if k == 0:
        left = tile(item["anchor_frame"], "IMAGINED (NanoWM)", "t=0 anchor (real context)", WHITE)
    elif t <= n_img:
        left = tile(
            img_frames[k - 1],
            "IMAGINED (NanoWM)",
            f"t={t} round {k}/{len(img_frames)} img cov={item['img_cov'][k - 1]:.2f}",
            WHITE,
        )
    else:
        left = tile(
            (img_frames[-1] * 0.45 + 140).astype(np.uint8),
            "IMAGINED (NanoWM)",
            f"imagination ended @ t={n_img}",
            GREY,
        )
    ti = min(t, len(real) - 1)
    done = succ > 0 and t >= succ
    phase = "replay img acts" if ti <= n_img else "base policy"
    if done:
        line2, color, border = f"SUCCESS @ step {succ} (cov {cov[ti]:.2f})", GREEN, GREEN
    elif t >= len(real) - 1 and succ < 0:
        line2, color, border = f"FAIL: none in {len(real) - 1} steps (cov {cov[ti]:.2f})", RED, RED
    else:
        line2, color, border = f"t={ti} cov={cov[ti]:.2f} | {phase}", WHITE, None
    right = tile(real[ti], "REAL (gym_pusht sim)", line2, color, border)
    row = np.concatenate([left, right], axis=1)
    head = np.zeros((ROW_HEADER, row.shape[1], 3), np.uint8)
    put(head, item["caption"], 15, item["caption_color"], 0.45)
    return np.concatenate([head, row], axis=0)


def grid(items, t, cols):
    rows = [
        np.concatenate([pair_frame(it, t) for it in items[i : i + cols]], axis=1)
        for i in range(0, len(items), cols)
    ]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(rows, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--top", type=int, default=6)
    p.add_argument(
        "--branches",
        nargs="+",
        default=None,
        help="show these branches (sigma0.0/seed0038_cand1.npz ...) instead of the top-V ones",
    )
    p.add_argument("--labels-name", default="labels.npz", help="test_stage2/<name> the real side reproduces")
    p.add_argument(
        "--cont-seed", type=int, default=None, help="continuation seed of that pass (default label.cont_seed)"
    )
    p.add_argument("--prefix", default="top", help="output file prefix")
    p.add_argument("--cols", type=int, default=2, help="branch pairs per grid row")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=_CRITIC.parent / "output" / "sailor_v1" / "topscore_video")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["out_dir"]).expanduser() / "test_stage2"
    q_final = int(max(cfg["stage2"]["query_rounds"]))

    tops = (
        picked_branches(pool_dir(cfg), args.branches)
        if args.branches
        else top_branches(pool_dir(cfg), args.top)
    )
    branches, jobs = label_jobs(root, [int(r) for r in cfg["stage2"]["query_rounds"]])
    name_to_bi = {b["name"]: bi for bi, b in enumerate(branches)}
    job_of = {(bi, r): j for j, (bi, r) in enumerate(jobs)}
    for t in tops:
        t["name"] = f"sigma{t['sigma']}/seed{t['anchor']:04d}_cand{t['cand']}.npz"
        t["job"] = job_of[(name_to_bi[t["name"]], q_final)]
    print("top branches:", [(t["name"], round(t["score"], 3), t["job"]) for t in tops])
    tag = f"{args.prefix}_{Path(args.labels_name).stem}" if args.branches else f"top{args.top}"
    cache = (
        out_dir / f"rerun_{tag}.pkl"
    )  # the label re-run (~30 min for the full v1 test set); delete to recompute
    if cache.exists():
        conts = pickle.loads(cache.read_bytes())
        assert {t["job"] for t in tops} <= set(conts), f"{cache} lacks some top jobs; delete it"
    else:
        conts = rerun_labels(
            cfg, root, branches, jobs, {t["job"] for t in tops}, device, args.labels_name, args.cont_seed
        )
        cache.write_bytes(pickle.dumps(conts))

    items, summary = [], []
    for rank, t in enumerate(tops, 1):
        br = branches[name_to_bi[t["name"]]]
        z = np.load(br["path"])
        real, cov, succ = render_real(br, conts[t["job"]])
        n_img = len(br["exec_abs_targets"])
        c = conts[t["job"]]
        outcome = f"SUCCESS@{succ}" if succ > 0 else "FAIL"
        if c["pre_success"]:
            outcome += " (in replay)"
        caption = f"#{rank} a{t['anchor']} s{t['sigma']} c{t['cand']} V={t['score']:.3f} {outcome}"
        if args.labels_name != "labels.npz":
            caption += f" [{Path(args.labels_name).stem}]"
        items.append(
            {
                "img": z["frames"],
                "img_cov": z["coverage"],
                "anchor_frame": z["anchor_frame"],
                "real": real,
                "cov": cov,
                "success": succ,
                "n_img": n_img,
                "caption": caption,
                "caption_color": GREEN if succ > 0 else RED,
            }
        )
        summary.append(
            {
                "rank": rank,
                "branch": t["name"],
                "anchor": t["anchor"],
                "sigma": t["sigma"],
                "cand": t["cand"],
                "v_score_r8": t["score"],
                "success": succ > 0,
                "success_step_from_anchor": succ,
                "imagined_steps": n_img,
                "pre_success": c["pre_success"],
                "label_success_step_after_replay": c["success_step"],
                "label_y": c["label_y"],
                "real_cov_end": float(cov[-1]),
                "imagined_cov_end": float(z["coverage"][-1]),
            }
        )
        print(summary[-1])
        T = max(len(real), n_img + 1) + 10
        vid = [pair_frame(items[-1], t_) for t_ in range(T)]
        imageio.mimwrite(
            out_dir / f"{args.prefix}{rank}_anchor{t['anchor']}_sigma{t['sigma']}_cand{t['cand']}.mp4",
            vid,
            fps=args.fps,
            codec="libx264",
        )

    T = max(max(len(it["real"]), it["n_img"] + 1) for it in items) + 10
    mp4 = out_dir / f"{tag}_imagined_vs_real.mp4"
    imageio.mimwrite(mp4, [grid(items, t, args.cols) for t in range(T)], fps=args.fps, codec="libx264")
    sheet_ts = [0, 20, n_img, 80, 140, T - 1]
    sheet = np.concatenate([grid(items, t, args.cols) for t in sheet_ts], axis=0)
    imageio.imwrite(
        out_dir / f"{tag}_sheet.png", cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
    )
    (out_dir / ("summary.json" if not args.branches else f"summary_{tag}.json")).write_text(
        json.dumps(summary, indent=1)
    )
    print(f"wrote {mp4}\nwrote {out_dir}")


if __name__ == "__main__":
    main()
