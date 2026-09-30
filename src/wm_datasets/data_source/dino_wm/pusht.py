"""
PushT data source (DINO-WM format) aligned with original implementation.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange

import decord
from decord import VideoReader

from ..base import DataSource, TrajectoryData

decord.bridge.set_bridge("torch")


class PushTDataSource(DataSource):
    """
    PushT dataset loader adapted from https://github.com/gaoyuezhou/dino_wm.

    Expected directory structure:
        data_path/
        ├── states.pth
        ├── rel_actions.pth or abs_actions.pth
        ├── seq_lengths.pkl (or .pth)
        ├── velocities.pth (optional)
        ├── shapes.pkl (optional)
        └── obses/
            ├── episode_000.mp4
            └── ...
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        use_relative_actions: bool = True,
        action_scale: float = 100.0,
        with_velocity: bool = True,
    ):
        self.data_path = Path(data_path)
        self.use_relative_actions = use_relative_actions
        self.action_scale = action_scale
        self.with_velocity = with_velocity

        print(f"Loading PushT trajectories from {self.data_path}...")
        self.states = torch.load(self.data_path / "states.pth").float()

        if use_relative_actions:
            actions_file = self.data_path / "rel_actions.pth"
        else:
            actions_file = self.data_path / "abs_actions.pth"
        self.actions = torch.load(actions_file).float()
        if action_scale != 1.0:
            self.actions = self.actions / action_scale

        seq_lengths_pth = self.data_path / "seq_lengths.pth"
        seq_lengths_pkl = self.data_path / "seq_lengths.pkl"
        if seq_lengths_pth.exists():
            seq_lengths = torch.load(seq_lengths_pth)
        elif seq_lengths_pkl.exists():
            with open(seq_lengths_pkl, "rb") as f:
                seq_lengths = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Missing seq_lengths for PushT: {seq_lengths_pth} or {seq_lengths_pkl}"
            )
        if isinstance(seq_lengths, list):
            self.seq_lengths = torch.tensor(seq_lengths)
        else:
            self.seq_lengths = seq_lengths

        # Shapes: default to 'T' if file missing
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                self.shapes = pickle.load(f)
        else:
            self.shapes = ["T"] * len(self.states)

        # Optionally append velocities to states
        self.velocities = None
        if with_velocity:
            velocities_file = self.data_path / "velocities.pth"
            if velocities_file.exists():
                self.velocities = torch.load(velocities_file).float()
                self.states = torch.cat([self.states, self.velocities], dim=-1)

        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.shapes = self.shapes[:n]
        if self.velocities is not None:
            self.velocities = self.velocities[:n]

        self.num_trajectories = n
        self._action_dim = self.actions.shape[-1]
        self._state_dim = self.states.shape[-1]

        print(f"Loaded {n} PushT trajectories from {self.data_path}")
        print(f"  State dim: {self._state_dim}, Action dim: {self._action_dim}")

    def load_trajectory(self, index: int) -> TrajectoryData:
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        seq_len = self.seq_lengths[index].item()
        meta: Dict = {"shape": self.shapes[index]}
        if self.velocities is not None:
            meta["velocity"] = self.velocities[index]

        return TrajectoryData(
            states=self.states[index],
            actions=self.actions[index],
            seq_length=seq_len,
            meta=meta,
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1
    ) -> torch.Tensor:
        obs_dir = self.data_path / "obses"
        video_file = obs_dir / f"episode_{index:03d}.mp4"
        if not video_file.exists():
            raise FileNotFoundError(
                f"Visual frames not found: {video_file}\n"
                f"Trajectory {index} exists in metadata but has no corresponding video file."
            )

        reader = VideoReader(str(video_file), num_threads=1)
        frame_indices = list(range(start, end, step))
        frames = reader.get_batch(frame_indices)

        if not isinstance(frames, torch.Tensor):
            if hasattr(frames, "asnumpy"):
                frames = frames.asnumpy()
            frames = torch.from_numpy(frames)

        frames = frames.float() / 255.0
        frames = rearrange(frames, "T H W C -> T C H W")
        return frames

    def get_num_trajectories(self) -> int:
        return self.num_trajectories

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim
