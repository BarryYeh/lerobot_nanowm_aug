"""
Deformable environment data source (rope/granular), aligned with original implementation.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
from einops import rearrange

from ..base import DataSource, TrajectoryData


class DeformableEnvDataSource(DataSource):
    """
    Deformable dataset loader adapted from https://github.com/gaoyuezhou/dino_wm.

    Expected directory structure:
        data_path/
        └── object_name/
            ├── states.pth   # [N, T, P, D]
            ├── actions.pth  # [N, T, action_dim]
            ├── 000000/
            │   └── obses.pth  # [T, H, W, C]
            └── 000001/
                └── obses.pth
    """

    def __init__(
        self,
        data_path: str,
        object_name: str,
        n_rollout: Optional[int] = None,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path) / object_name
        self.object_name = object_name
        self.action_scale = action_scale

        print(f"Loading deformable trajectories from {self.data_path}...")
        states = torch.load(self.data_path / "states.pth").float()
        # [N, T, P, D] -> [N, T, P*D]
        self.states = rearrange(states, "N T P D -> N T (P D)")

        self.actions = torch.load(self.data_path / "actions.pth").float()
        if action_scale != 1.0:
            self.actions = self.actions / action_scale

        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]

        self.num_trajectories = n
        self._action_dim = self.actions.shape[-1]
        self._state_dim = self.states.shape[-1]

        # Deformable datasets typically have fixed length
        self.seq_lengths = torch.tensor([self.states.shape[1]] * len(self.states))

        print(f"Loaded {n} deformable trajectories from {self.data_path}")
        print(f"  State dim: {self._state_dim}, Action dim: {self._action_dim}")

    def load_trajectory(self, index: int) -> TrajectoryData:
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        seq_len = self.seq_lengths[index].item()
        meta: Dict = {}

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
        obs_dir = self.data_path / f"{index:06d}"
        obs_file = obs_dir / "obses.pth"
        if not obs_file.exists():
            raise FileNotFoundError(
                f"Visual frames not found: {obs_file}\n"
                f"Trajectory {index} exists in metadata but has no corresponding video file."
            )

        full_video = torch.load(obs_file, weights_only=False, map_location="cpu")
        frames = full_video[start:end:step].contiguous()
        # [T, H, W, C] -> [T, C, H, W], normalize to [0, 1]
        frames = rearrange(frames, "T H W C -> T C H W").float() / 255.0
        return frames

    def get_num_trajectories(self) -> int:
        return self.num_trajectories

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim
