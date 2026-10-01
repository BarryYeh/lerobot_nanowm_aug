#!/usr/bin/env python
"""Diagnostic video: base-policy rollouts under the stage-1 perturbation vs the same rollouts clean.

Starts are the first `--n` stage-1 reset starts (`env.reset(seed=stage1.reset.seed + i)`). The top
row uses `stage1.perturb` (sigma, noise_hold, max_z, mode prefix over the whole episode); the bottom
row is identical with sigma=0. Both runs use the same policy torch seed and batch, so each column's
diffusion-sampling noise is the same in both rows and the only difference is the injected action noise.

    uv run --extra training python examples/nanowm_pusht_aug/critic/perturb_video.py \
        --config configs/sailor_v1_reset.yaml --n 4
"""

from __future__ import annotations

import argparse
import sys
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
from ood_failure_map import sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

HEADER = 34


def reset_starts(seed0: int, n: int) -> np.ndarray:
    env = gym.make("gym_pusht/PushT-v0")
    starts = []
    for i in range(n):
        env.reset(seed=seed0 + i)
        starts.append(sim_state(env.unwrapped))
    env.close()
    return np.asarray(starts, np.float64)


def run(policy, pre, post, starts, sigma, cfg, device, seed) -> dict:
    s1, pr = cfg["stage1"], cfg["stage1"]["perturb"]
    steps = int(s1["episode_steps"])
    n = len(starts)
    return rollout_batch(
        policy,
        pre,
        post,
        starts,
        device=device,
        na=int(cfg["label"]["na"]),
        episode_steps=steps,
        seed=seed,
        batch_start=0,
        sigma=np.full(n, sigma),
        perturb_p=np.full(n, steps, np.int64),
        max_z=float(pr["max_z"]),
        noise_hold=int(pr["noise_hold"]),
        noise_rng=np.random.default_rng(int(pr["seed"]) + 1),
        lockstep224=True,
        frame_grid_stride=1,
        frame_cutoff_step=steps,
        wall_pen_px=float(cfg["label"]["wall_pen_px"]),
        perturb_mode="prefix",
    )


def fill_frames(out: dict) -> np.ndarray:
    """Frames after an env terminated are never captured; hold that env's last captured frame."""
    frames = out["grid_frames"].copy()
    for i in range(frames.shape[0]):
        s = int(out["success_step"][i])
        last = s if s > 0 else frames.shape[1] - 1
        frames[i, last + 1 :] = frames[i, last]
    return frames


def tile(frame: np.ndarray, label: str, t: int, cov: float, success_step: int) -> np.ndarray:
    img = np.zeros((frame.shape[0] + HEADER, frame.shape[1], 3), np.uint8)
    img[HEADER:] = frame
    done = 0 < success_step <= t
    color = (80, 220, 80) if done else (255, 255, 255)
    cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    status = f"SUCCESS @{success_step}" if done else f"t={t} cov={cov:.2f}"
    cv2.putText(img, status, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img


def compose(rows: list[tuple[str, dict, np.ndarray]], t: int, ids: list[int]) -> np.ndarray:
    lines = []
    for name, out, frames in rows:
        tiles = [
            tile(
                frames[i, t],
                f"{name} | reset {ids[i]}",
                t,
                float(out["cov_per_step"][i, t]),
                int(out["success_step"][i]),
            )
            for i in range(frames.shape[0])
        ]
        lines.append(np.concatenate(tiles, axis=1))
    return np.concatenate(lines, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--n", type=int, default=4, help="number of stage-1 reset starts (columns)")
    p.add_argument("--first", type=int, default=0, help="first reset index (seed offset)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <out_dir>/videos")
    args = p.parse_args()
    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or Path(cfg["out_dir"]) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed0 = int(s1["reset"]["seed"]) + args.first
    ids = list(range(seed0, seed0 + args.n))
    starts = reset_starts(seed0, args.n)
    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    torch_seed = int(s1.get("policy_seed", 0))
    sig = float(s1["perturb"]["sigmas"][0])

    noised = run(policy, pre, post, starts, sig, cfg, device, torch_seed)
    clean = run(policy, pre, post, starts, 0.0, cfg, device, torch_seed)
    rows = [
        (f"noised {sig}sigma", noised, fill_frames(noised)),
        ("clean", clean, fill_frames(clean)),
    ]
    T = rows[0][2].shape[1]
    video = [compose(rows, t, ids) for t in range(T)]
    tag = f"perturb_s{sig}_reset{ids[0]}-{ids[-1]}"
    mp4 = out_dir / f"{tag}.mp4"
    imageio.mimwrite(mp4, video, fps=args.fps, codec="libx264")
    sheet_ts = [0, 50, 100, 150, 200, 250, T - 1]
    sheet = np.concatenate([compose(rows, t, ids) for t in sheet_ts], axis=0)
    png = out_dir / f"{tag}_sheet.png"
    imageio.imwrite(png, cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2)))

    for name, out, _ in rows:
        print(
            f"{name}: success_step={out['success_step'].tolist()} final_cov={out['cov_per_step'][:, -1].round(3).tolist()}"
        )
    gap = np.linalg.norm(noised["grid_agent_pos"] - clean["grid_agent_pos"], axis=-1)
    print(f"agent |noised - clean| px at t=10/50/100: {gap[:, [10, 50, 100]].round(1).tolist()}")
    print(f"wrote {mp4}\nwrote {png}")


if __name__ == "__main__":
    main()
