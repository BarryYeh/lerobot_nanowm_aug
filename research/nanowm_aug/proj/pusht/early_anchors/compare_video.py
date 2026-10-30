#!/usr/bin/env python
"""early_anchors: one evaluation episode played side by side by the fine-tuned policies of three runs.

Panels: the Img arm (fine-tune seed --ft-seed; or, with --base, the un-fine-tuned base policy) of the original 40k base,
the 40k_cos base and the 100k base. Frames are
the videos the original 2,000-episode eval already rendered (lerobot-eval renders episodes 0..9 of batch 0, i.e. eval
seeds 100000..100009), so each panel is exactly the scored episode. Per panel the step counter runs until the episode
ends (success at coverage > 0.95, or 300 steps) and then holds the last frame; the outcome (success step, and whether
coverage reached 0.9 from eval_info.json max_rewards) is that eval's own record. The eval_steps09 re-runs are NOT used:
GPU nondeterminism makes a re-run's trajectory diverge from the original episode.

    uv run python research/nanowm_aug/proj/pusht/early_anchors/compare_video.py --episode 0 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

D = Path.home() / "datastorage/nanowm_pusht_aug"
RUNS = [  # (label, run dir)
    ("Original 40k (lr 6.6e-5)", "early_anchors"),
    ("40k_cos (lr -> 0 at 40k)", "early_anchors_base40K_cos"),
    ("100k (lr -> 0 at 100k)", "early_anchors_base100k"),
]
S, M, HEAD, FOOT = 384, 16, 64, 76
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, FG, MUTED = (255, 255, 255), (61, 33, 20), (104, 85, 74)  # BGR
GREEN, RED, BLUE = (40, 140, 40), (40, 40, 200), (219, 111, 42)


def text(img, s, x, y, scale, color, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=0, help="eval episode index 0..9 (seed 100000 + index)")
    ap.add_argument("--ft-seed", type=int, default=1000)
    ap.add_argument(
        "--base", action="store_true", help="the un-fine-tuned base policies instead of the Img arms"
    )
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=int, default=20, help="extra frames on the final composite")
    ap.add_argument("--out", type=Path, default=D / "early_anchors_base40K_cos/video")
    a = ap.parse_args()
    arm = "base" if a.base else f"img_s{a.ft_seed}"
    panels = []
    for label, run in RUNS:
        ev = D / run / "finetune/eval" / arm
        frames = imageio.mimread(ev / f"videos/pusht_0/eval_episode_{a.episode}.mp4", memtest=False)
        z = np.load(ev / "success_steps.npz")
        max_r = json.loads((ev / "eval_info.json").read_text())["per_task"][0]["metrics"]["max_rewards"][
            a.episode
        ]
        succ, ep_len = int(z["success_step"][a.episode]), int(z["ep_len"][a.episode])
        # the rendered video holds the reset frame plus one frame per step, up to the step the episode ended
        assert len(frames) in (ep_len, ep_len + 1), (run, len(frames), ep_len)
        panels.append(
            {"label": label, "frames": frames, "c9": max_r >= 0.9 / 0.95, "succ": succ, "ep_len": ep_len}
        )
    T = max(len(p["frames"]) for p in panels) + a.hold
    W, H = 3 * S + 4 * M, HEAD + S + FOOT
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"compare_{arm}_ep{a.episode}.mp4"
    # browser-friendly H.264 / yuv420p, faststart
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(a.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for t in range(T):
        img = np.full((H, W, 3), BG, np.uint8)
        for i, p in enumerate(panels):
            x0 = M + i * (S + M)
            k = min(t, len(p["frames"]) - 1)
            img[HEAD : HEAD + S, x0 : x0 + S] = cv2.cvtColor(
                np.asarray(p["frames"][k])[..., :3], cv2.COLOR_RGB2BGR
            )
            text(img, p["label"], x0, 40, 0.62, FG, 2)
            step = min(t, p["ep_len"])
            text(img, f"step {step}", x0, HEAD + S + 28, 0.6, MUTED, 1)
            if step >= p["ep_len"]:
                ok = p["succ"] > 0
                msg = f"> 0.95 at step {p['succ']}" if ok else "no > 0.95 in 300 steps"
                text(img, msg, x0 + 130, HEAD + S + 28, 0.6, GREEN if ok else RED, 2)
                msg = "reached cov >= 0.9" if p["c9"] else "never reached cov >= 0.9"
                text(img, msg, x0, HEAD + S + 60, 0.6, BLUE if p["c9"] else RED, 2)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    assert ff.wait() == 0
    print(out, [(p["label"], p["c9"], p["succ"], p["ep_len"]) for p in panels])


if __name__ == "__main__":
    main()
