#!/usr/bin/env python
"""Denoise pusht_noise -> a 185-demo image-only fps-10 LeRobotDataset.

BC should not learn the injected per-step action noise. pusht_noise = 185 base pushes
x 101 noised replays. Per group (unique init) `--method minres` (default) picks the
replay whose rel-action sequence is closest to the group's per-step median -- that
trajectory is ~noise-free (mean |residual| ~0.1 px/step) AND uses the authentic archive
frames NanoWM was trained on. `--method median_rerender` instead takes the median
state+action trajectory and re-renders each frame in gym_pusht.

Output: local/pushtnoise_clean185_imgonly (185 episodes; observation.image 96x96 video;
action = absolute agent target (2,); fps 10; robot_type pusht_sim). With --dump-videos,
also writes 3 verification mp4s to examples/nanowm_pusht_aug/diag/ for manual sign-off.

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py \
        pushtnoise-denoise --overwrite --dump-videos
"""

from __future__ import annotations

import argparse
import math
import pickle  # nosec B403 -- seq_lengths.pkl ships inside the DINO-WM pusht_noise archive the user downloads themselves; not untrusted input
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset_converter import ConverterConfig, LeRobotDatasetConverter

DEFAULT_ARCHIVE = Path("~/Downloads/bmw48-osfstorage-datasets-archive/pusht_noise").expanduser()
WIN = 512
FEATURES = {
    "observation.image": {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": (2,), "names": {"axes": ["target_x", "target_y"]}},
}

# ---- raster true coverage (block T at pose  vs  goal T at (256,256,pi/4)) ----
_S, _L, _R = 30.0, 4, 512
_CB = np.array([(-_L * _S / 2, _S), (_L * _S / 2, _S), (_L * _S / 2, 0.0), (-_L * _S / 2, 0.0)])
_ST = np.array([(-_S / 2, _S), (-_S / 2, _L * _S), (_S / 2, _L * _S), (_S / 2, _S)])


def _tee_mask(cx, cy, th):
    c, s = math.cos(th), math.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    m = np.zeros((_R, _R), np.uint8)
    for q in (_CB, _ST):
        cv2.fillConvexPoly(m, ((q @ Rm.T) + [cx, cy]).astype(np.int32), 1)
    return m


_GM = _tee_mask(256, 256, math.pi / 4)
_GA = int(_GM.sum())


def true_cov(bx, by, ang):
    return float((_tee_mask(bx, by, ang) & _GM).sum() / _GA)


def w2px(xy, size):
    # pusht_noise states.pth / abs_actions.pth use image-style coords (y down, no flip).
    return int(xy[0] / WIN * size), int(xy[1] / WIN * size)


def decode_mp4(path, size=None):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if size:
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        out.append(fr)
    cap.release()
    return np.asarray(out, np.uint8)


def load_split(archive: Path, split: str):
    # weights_only=True: these .pth files hold plain tensors, nothing to unpickle.
    st = torch.load(archive / split / "states.pth", weights_only=True).float().numpy()
    ab = torch.load(archive / split / "abs_actions.pth", weights_only=True).float().numpy()
    rl = torch.load(archive / split / "rel_actions.pth", weights_only=True).float().numpy()
    with open(archive / split / "seq_lengths.pkl", "rb") as f:
        sl = np.array([int(x) for x in pickle.load(f)])  # nosec B301 -- see the import note
    return st, ab, rl, sl


def select_minres(rl, sl, group):
    """group: indices into the split (all share an init). Return (best_idx, Lc)."""
    Lc = int(sl[group].min())
    A = rl[group, :Lc]
    med = np.median(A, axis=0)
    res = np.linalg.norm(A - med[None], axis=2).mean(1)
    return int(group[int(np.argmin(res))]), Lc


class PushtNoiseDenoiseConverter(LeRobotDatasetConverter):
    name = "pushtnoise-denoise"
    DEFAULT_REPO_ID = "local/pushtnoise_clean185_imgonly"

    def __init__(self, config, archive, method, max_groups, dump_videos, video_dir):
        super().__init__(config)
        self.archive = archive
        self.method = method
        self.max_groups = max_groups
        self.dump_videos = dump_videos
        self.video_dir = video_dir
        self.end_cov: list[float] = []
        self.chosen: list[int] = []
        self._env = self._u = None

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
        p.add_argument("--method", choices=["minres", "median_rerender"], default="minres")
        p.add_argument("--max-groups", type=int, default=None, help="smoke test")
        p.add_argument("--dump-videos", action="store_true")
        p.add_argument("--video-dir", type=Path, default=Path("examples/nanowm_pusht_aug/diag"))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PushtNoiseDenoiseConverter:
        cfg = ConverterConfig(
            repo_id=args.repo_id,
            fps=10,
            features=FEATURES,
            root=args.root,
            robot_type="pusht_sim",
            overwrite=args.overwrite,
            image_writer_threads=args.image_writer_threads,
            extra={"method": args.method},
        )
        return cls(
            cfg,
            archive=args.archive,
            method=args.method,
            max_groups=args.max_groups,
            dump_videos=args.dump_videos,
            video_dir=args.video_dir,
        )

    def load_source(self) -> None:
        self.st, self.ab, self.rl, self.sl = load_split(self.archive, "train")
        inits = np.round(self.st[:, 0, :5], 1)
        _, inv = np.unique(inits, axis=0, return_inverse=True)
        self.groups = [np.nonzero(inv == g)[0] for g in range(int(inv.max()) + 1)]
        if self.max_groups:
            self.groups = self.groups[: self.max_groups]
        print(f"{len(self.groups)} groups (x{len(self.groups[0])} replays)  method={self.method}")
        if self.method == "median_rerender":
            import gym_pusht  # noqa: F401
            import gymnasium as gym

            self._env = gym.make(
                "gym_pusht/PushT-v0", obs_type="pixels", observation_width=224, observation_height=224
            )
            self._u = self._env.unwrapped

    def iter_episodes(self):
        for gi, group in enumerate(self.groups):
            yield self._group_frames(gi, group)

    def _group_frames(self, gi: int, group):
        if self.method == "minres":
            best, Lc = select_minres(self.rl, self.sl, group)
            frames96 = decode_mp4(self.archive / "train" / "obses" / f"episode_{best:03d}.mp4", size=96)[:Lc]
            acts = self.ab[best, :Lc].astype(np.float32)
            states = self.st[best, :Lc]
            self.chosen.append(best)
        else:  # median_rerender
            Lc = int(self.sl[group].min())
            states = np.median(self.st[group, :Lc], axis=0)
            ang = self.st[group, :Lc, 4]
            states[:, 4] = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))  # circular mean
            acts = np.median(self.ab[group, :Lc], axis=0).astype(np.float32)
            frames96 = np.zeros((Lc, 96, 96, 3), np.uint8)
            for t in range(Lc):
                self._env.reset(seed=gi)
                self._u.agent.position = tuple(float(x) for x in states[t, :2])
                self._u.block.angle = float(states[t, 4])
                self._u.block.position = tuple(float(x) for x in states[t, 2:4])
                self._u.block.velocity = (0.0, 0.0)
                self._u.block.angular_velocity = 0.0
                self._u.space.step(self._u.dt)
                frames96[t] = cv2.resize(
                    np.asarray(self._u.get_obs()), (96, 96), interpolation=cv2.INTER_AREA
                )
            self.chosen.append(-1)

        self.end_cov.append(true_cov(states[-1, 2], states[-1, 3], states[-1, 4]))
        for t in range(len(frames96)):
            yield {"observation.image": frames96[t], "action": acts[t]}

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:
        if (episode_index + 1) % 40 == 0:
            print(
                f"  {episode_index + 1}/{len(self.groups)}  running end-cov mean={np.mean(self.end_cov):.3f}"
            )

    def post_run(self) -> None:
        if self._env is not None:
            self._env.close()
        if self.dump_videos:
            _dump_videos(self.video_dir, self.archive, self.groups, self.st, self.ab, self.rl, self.sl)

    def summarize(self) -> str:
        ec = np.array(self.end_cov)
        return (
            f"done -> {self.config.root}  ({self.n_episodes} episodes)\n"
            f"ending coverage: mean={ec.mean():.3f} median={np.median(ec):.3f} "
            f"frac>=0.6={(ec >= 0.6).mean():.2f} frac>=0.7={(ec >= 0.7).mean():.2f}"
        )


def _overlay(fr224, agent_xy, target_xy, cov, t):
    im = fr224.copy()
    a, tg = w2px(agent_xy, 224), w2px(target_xy, 224)
    cv2.arrowedLine(im, a, tg, (255, 0, 0), 2, tipLength=0.25)
    cv2.circle(im, a, 3, (255, 0, 255), -1)
    cv2.putText(
        im, f"t={t:3d}  cov={cov:.2f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    return im


def _dump_videos(vdir: Path, archive: Path, groups, st, ab, rl, sl):
    import imageio.v2 as imageio

    vdir.mkdir(parents=True, exist_ok=True)
    obses = archive / "train" / "obses"
    picks = [0, len(groups) // 2, len(groups) - 1]
    for tag, gi in zip("ABC", picks, strict=True):
        group = groups[gi]
        best, Lc = select_minres(rl, sl, group)
        arch224 = decode_mp4(obses / f"episode_{best:03d}.mp4")[:Lc]
        states, acts = st[best, :Lc], ab[best, :Lc]
        vid = [
            _overlay(
                arch224[t], states[t, :2], acts[t], true_cov(states[t, 2], states[t, 3], states[t, 4]), t
            )
            for t in range(Lc)
        ]
        imageio.mimwrite(vdir / f"check_ep_{tag}.mp4", vid, fps=10, codec="libx264")
        if tag == "A":
            store96 = decode_mp4(obses / f"episode_{best:03d}.mp4", size=96)[:Lc]
            sbs = []
            for t in range(Lc):
                r = cv2.resize(store96[t], (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.putText(arch224[t], "archive 224", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(r, "training 96->224", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                sbs.append(np.hstack([arch224[t], r]))
            imageio.mimwrite(vdir / "check_ep_A_sidebyside.mp4", sbs, fps=10, codec="libx264")
        print(
            f"  check_ep_{tag}.mp4  group {gi} (archive ep {best}, len {Lc}, end cov "
            f"{true_cov(states[-1, 2], states[-1, 3], states[-1, 4]):.2f})"
        )
    print(f"videos -> {vdir}")


if __name__ == "__main__":
    PushtNoiseDenoiseConverter.main()
