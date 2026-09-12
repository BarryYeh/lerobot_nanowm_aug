#!/usr/bin/env python
"""Convert the DINO-WM ``pusht_noise`` archive (osf.io/bmw48) to a LeRobotDataset.

``pusht_noise`` is the exact PushT data NanoWM was trained on, so training a policy on
it puts the policy, the NanoWM context frames and the NanoWM imagined rollouts in one
visual distribution.

Frame rate: the NanoWM checkpoint runs at ``frame_interval=5`` (1 NanoWM step = 5 sim
steps). ``--frame-interval 5`` sub-samples the dense mp4s with the same stride -> fps 2,
matched to NanoWM; ``--frame-interval 1`` keeps native fps 10.

Coarse action at step ``k`` (covering sim-steps ``t0 .. t0+stride-1``):
  * ``end``            (default) ``abs_actions[t0+stride-1]`` -- target the demonstrator
                        was heading to; best match for a held-target controller at eval.
  * ``start``          ``abs_actions[t0]``
  * ``next_agent_pos`` ``states[t0+stride, :2]`` -- where the agent actually is next.

Source layout (per split dir ``train/`` and ``val/``):
    states.pth        (N, T_pad, 5)   [agent_x, agent_y, block_x, block_y, block_angle]
    abs_actions.pth   (N, T_pad, 2)   absolute agent target position
    velocities.pth    (N, T_pad, 2)   agent velocity
    seq_lengths.pkl   list[int]       real (unpadded) length of each episode
    obses/episode_XXX.mp4             224x224 RGB, one frame per sim-step

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py \
        pusht-noise --splits train val --frame-interval 5 --overwrite
"""

from __future__ import annotations

import argparse
import pickle  # nosec B403 -- seq_lengths.pkl ships inside the DINO-WM pusht_noise archive the user downloads themselves; not untrusted input
from pathlib import Path

import av
import numpy as np
import torch

from .dataset_converter import ConverterConfig, LeRobotDatasetConverter

SIM_FPS = 10  # PushT control_hz (native mp4 rate)
IMAGE_HW = (224, 224)
DEFAULT_ARCHIVE = Path("~/Downloads/bmw48-osfstorage-datasets-archive/pusht_noise").expanduser()

FEATURES = {
    "observation.image": {"dtype": "video", "shape": (*IMAGE_HW, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": {"axes": ["agent_x", "agent_y"]}},
    "action": {"dtype": "float32", "shape": (2,), "names": {"axes": ["target_x", "target_y"]}},
    "observation.pose": {
        "dtype": "float32",
        "shape": (5,),
        "names": {"axes": ["agent_x", "agent_y", "block_x", "block_y", "block_angle"]},
    },
    "observation.velocity": {"dtype": "float32", "shape": (2,), "names": {"axes": ["vel_x", "vel_y"]}},
}
MINIMAL_KEYS = {"observation.image", "observation.state", "action"}
IMAGE_ONLY_KEYS = {"observation.image", "action"}


def decode_mp4(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames) if frames else np.empty((0, *IMAGE_HW, 3), dtype=np.uint8)


def coarse_action(abs_actions, states, t0, stride, length, mode):
    if mode == "start":
        return abs_actions[t0]
    if mode == "end":
        return abs_actions[min(t0 + stride - 1, length - 1)]
    if mode == "next_agent_pos":
        return states[min(t0 + stride, length - 1), :2]
    raise ValueError(mode)


class PushtNoiseToLerobotConverter(LeRobotDatasetConverter):
    name = "pusht-noise"
    DEFAULT_REPO_ID = ""  # derived from --frame-interval / --drop-state when left blank

    def __init__(self, config, archive, splits, stride, action_mode, max_episodes, keep_keys):
        super().__init__(config)
        self.archive = archive
        self.splits = splits
        self.stride = stride
        self.action_mode = action_mode
        self.max_episodes = max_episodes
        self.keep_keys = keep_keys

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="dir with train/ and val/")
        p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
        p.add_argument(
            "--frame-interval",
            type=int,
            default=5,
            help="sub-sample stride in sim-steps; must divide 10 (5 -> fps 2, 1 -> fps 10)",
        )
        p.add_argument("--action-at", default="end", choices=["end", "start", "next_agent_pos"])
        p.add_argument("--max-episodes", type=int, default=None, help="cap episodes per split (smoke test)")
        p.add_argument("--minimal", action="store_true", help="store only image+state+action")
        p.add_argument(
            "--drop-state",
            action="store_true",
            help="store only image+action (repo_id gets an '_imgonly' suffix)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PushtNoiseToLerobotConverter:
        if SIM_FPS % args.frame_interval != 0:
            raise ValueError(f"--frame-interval must divide {SIM_FPS}")
        fps = SIM_FPS // args.frame_interval

        repo_id = args.repo_id or f"local/pusht_noise_224_fi{args.frame_interval}"
        if args.drop_state and not args.repo_id:
            repo_id += "_imgonly"

        if args.drop_state:
            keep = IMAGE_ONLY_KEYS
        elif args.minimal:
            keep = MINIMAL_KEYS
        else:
            keep = set(FEATURES)
        features = {k: v for k, v in FEATURES.items() if k in keep}

        for split in args.splits:
            if not (args.archive / split).is_dir():
                raise FileNotFoundError(args.archive / split)

        cfg = ConverterConfig(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=args.root,
            robot_type="pusht_sim",
            overwrite=args.overwrite,
            image_writer_threads=args.image_writer_threads,
            extra={"frame_interval": args.frame_interval, "action_at": args.action_at, "splits": args.splits},
        )
        return cls(
            cfg,
            archive=args.archive,
            splits=args.splits,
            stride=args.frame_interval,
            action_mode=args.action_at,
            max_episodes=args.max_episodes,
            keep_keys=keep,
        )

    def load_source(self) -> None:
        print(
            f"frame_interval={self.stride}  fps={self.config.fps}  action_at={self.action_mode}  "
            f"-> {self.config.repo_id}"
        )

    def iter_episodes(self):
        for split in self.splits:
            print(f"[{split}]")
            yield from self._split_episodes(self.archive / split)

    def _split_episodes(self, split_dir: Path):
        # weights_only=True: these .pth files hold plain tensors, nothing to unpickle.
        states_all = torch.load(split_dir / "states.pth", weights_only=True).to(torch.float32).numpy()
        actions_all = torch.load(split_dir / "abs_actions.pth", weights_only=True).to(torch.float32).numpy()
        vel_all = torch.load(split_dir / "velocities.pth", weights_only=True).to(torch.float32).numpy()
        with open(split_dir / "seq_lengths.pkl", "rb") as f:
            seq_lengths = [int(x) for x in pickle.load(f)]  # nosec B301 -- see the import note

        n_eps = len(seq_lengths)
        if self.max_episodes is not None:
            n_eps = min(n_eps, self.max_episodes)

        for ep in range(n_eps):
            mp4 = split_dir / "obses" / f"episode_{ep:03d}.mp4"
            if not mp4.exists():
                print(f"  [skip] ep {ep}: missing {mp4.name}")
                continue
            frames = decode_mp4(mp4)
            length = min(len(frames), seq_lengths[ep])
            if abs(len(frames) - seq_lengths[ep]) > 2:
                print(
                    f"  [warn] ep {ep}: {len(frames)} mp4 frames vs seq_length {seq_lengths[ep]} -> {length}"
                )
            if length == 0:
                print(f"  [skip] ep {ep}: no usable frames")
                continue
            if frames.shape[1:3] != IMAGE_HW:
                raise ValueError(f"ep {ep}: frame size {frames.shape[1:3]} != {IMAGE_HW}")
            yield self._frames(frames, states_all[ep], actions_all[ep], vel_all[ep], length)

    def _frames(self, frames, st, act, vel, length):
        for t0 in range(0, length, self.stride):
            full = {
                "observation.image": frames[t0],
                "observation.state": st[t0, :2],
                "action": coarse_action(act, st, t0, self.stride, length, self.action_mode),
                "observation.pose": st[t0, :5],
                "observation.velocity": vel[t0],
            }
            yield {k: v for k, v in full.items() if k in self.keep_keys}

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:
        if (episode_index + 1) % 100 == 0:
            print(f"  {episode_index + 1} episodes written")


if __name__ == "__main__":
    PushtNoiseToLerobotConverter.main()
