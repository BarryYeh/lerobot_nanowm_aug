"""
CSGO dataset data source for HDF5-based gameplay recordings.
"""

import h5py
import torch
from pathlib import Path
from typing import Dict, Optional

from ..base import DataSource, TrajectoryData


class CSGODataSource(DataSource):
    """
    Data source for Counter-Strike: Global Offensive gameplay dataset.

    Expected directory structure:
        data_path/
        ├── 1-200/
        │   ├── hdf5_dm_july2021_1.hdf5
        │   ├── hdf5_dm_july2021_2.hdf5
        │   └── ...
        ├── 201-400/
        │   └── ...
        └── ...

    Each HDF5 file contains:
        - frame_{i}_x: Visual frame at timestep i [H, W, C] uint8
        - frame_{i}_y: Action vector at timestep i [51] float32
        - 1000 timesteps per file (1 minute at 3 FPS)

    Action space (51-dimensional):
        - 11 keyboard keys (one-hot/multi-hot)
        - 2 mouse clicks (one-hot)
        - 23 mouse_x bins (one-hot)
        - 15 mouse_y bins (one-hot)

    Args:
        data_path: Path to dataset root directory
        n_rollout: Limit number of episodes to load (None = load all)
        val_file_list: Optional path to validation file list (not implemented)
        use_auxiliary_state: Extract health/ammo as state (not implemented)
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        file_list: Optional[str] = None,
        use_auxiliary_state: bool = False,
    ):
        self.data_path = Path(data_path)
        self.use_auxiliary_state = use_auxiliary_state
        self.file_list = file_list

        if use_auxiliary_state:
            raise NotImplementedError("use_auxiliary_state not yet implemented for CSGO dataset")

        # If file_list provided, directly construct paths (skip directory scan)
        if file_list is not None:
            file_list_path = Path(file_list)
            # Resolve relative paths against hydra's original cwd (the invocation dir),
            # matching how world_model_dataset.py treats dataset paths.
            if not file_list_path.is_absolute():
                try:
                    from hydra.utils import get_original_cwd  # type: ignore

                    file_list_path = Path(get_original_cwd()) / file_list_path
                except Exception:
                    # Outside a Hydra run (e.g. unit tests) — fall back to CWD.
                    file_list_path = Path.cwd() / file_list_path
            print(f"Using explicit file list from: {file_list_path}")
            if not file_list_path.exists():
                raise FileNotFoundError(f"File list not found: {file_list_path}")

            with open(file_list_path, "r") as f:
                specified_files = [line.strip() for line in f if line.strip()]
            self.file_list = str(file_list_path)

            hdf5_files = []
            for filename in specified_files:
                # Extract file number: hdf5_dm_july2021_N.hdf5 -> N
                file_num = int(filename.split("_")[-1].replace(".hdf5", ""))
                # Subdirectory: 1-200, 201-400, etc.
                subdir_start = ((file_num - 1) // 200) * 200 + 1
                subdir_end = subdir_start + 199
                subdir_name = f"{subdir_start}-{subdir_end}"

                file_path = self.data_path / subdir_name / filename
                if file_path.exists():
                    hdf5_files.append(file_path)
                else:
                    # Handle the last folder naming (5401-5500) used in some datasets
                    if 5401 <= file_num <= 5500:
                        alt_subdir_name = "5401-5500"
                        alt_path = self.data_path / alt_subdir_name / filename
                        if alt_path.exists():
                            hdf5_files.append(alt_path)
                            continue
                    print(f"Warning: File {filename} not found at {file_path}, skipping")

            hdf5_files = sorted(hdf5_files, key=lambda x: int(x.stem.split("_")[-1]))
            print(f"Loaded {len(hdf5_files)} files from list (no directory scan)")
        else:
            # Fallback: scan directory if no file_list provided
            print(f"Discovering HDF5 files in {self.data_path}...")
            hdf5_files = sorted(
                self.data_path.rglob("*.hdf5"),
                key=lambda x: int(x.stem.split("_")[-1])
            )
            if len(hdf5_files) == 0:
                raise FileNotFoundError(f"No HDF5 files found in {self.data_path}")

        # Create file mapping: relative_path -> absolute_path
        self._file_paths = [
            (f"{f.parent.name}/{f.name}", f) for f in hdf5_files
        ]

        # Limit number of episodes if requested
        if n_rollout is not None:
            self._file_paths = self._file_paths[:n_rollout]

        self.num_trajectories = len(self._file_paths)
        self._episode_length = 1000  # Fixed length per HDF5 file

        # Action cache (lazy loading)
        self._action_cache: Dict[int, torch.Tensor] = {}

        print(f"Loaded {self.num_trajectories} CSGO episodes")
        print(f"  Action dim: 51, State dim: 0 (pure vision)")
        print(f"  Episode length: {self._episode_length} frames")

    def load_trajectory(self, index: int) -> TrajectoryData:
        """
        Load trajectory metadata (states and actions).

        For CSGO, we have no explicit state (pure vision), so states are zeros.
        Actions are loaded from HDF5 and cached in memory.

        Returns:
            TrajectoryData with:
                - states: [1000, 0] tensor (empty)
                - actions: [1000, 51] tensor
                - seq_length: 1000
                - meta: {"episode_id": str}
        """
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        # Load actions (with caching)
        if index not in self._action_cache:
            relative_path, absolute_path = self._file_paths[index]

            with h5py.File(absolute_path, "r") as f:
                # Load all actions as numpy array first, then convert to tensor
                import numpy as np
                actions_np = np.array([f[f"frame_{i}_y"][:] for i in range(self._episode_length)])
                actions = torch.from_numpy(actions_np).float()

            self._action_cache[index] = actions

        actions = self._action_cache[index]

        # Create dummy states (state_dim = 0)
        states = torch.zeros(self._episode_length, 0, dtype=torch.float32)

        relative_path, _ = self._file_paths[index]
        meta = {"episode_id": relative_path}

        return TrajectoryData(
            states=states,
            actions=actions,
            seq_length=self._episode_length,
            meta=meta,
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1
    ) -> torch.Tensor:
        """
        Load visual frames from HDF5 file.

        Preprocessing pipeline:
            1. Load frame from HDF5: [H, W, C] uint8
            2. Flip channel dimension and permute to [C, H, W]
            3. Normalize to [0, 1] range

        Note: WorldModelDataset will automatically resize frames to the target
        image_size, so we don't need to do any resizing here.

        Args:
            index: Episode index
            start: Start frame (inclusive)
            end: End frame (exclusive)
            step: Frame step size

        Returns:
            Frames tensor [T, C, H, W], normalized to [0, 1], float32
        """
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        _, absolute_path = self._file_paths[index]

        # Open HDF5 file and load frames (not kept open for thread safety)
        with h5py.File(absolute_path, "r") as f:
            frame_indices = range(start, end, step)
            frames = []

            for i in frame_indices:
                if i >= self._episode_length:
                    raise ValueError(
                        f"Frame index {i} exceeds episode length {self._episode_length}"
                    )

                # Load frame: [H, W, C] uint8
                frame = torch.tensor(f[f"frame_{i}_x"][:])

                # Flip channel dimension and permute to [C, H, W]
                frame = frame.flip(2).permute(2, 0, 1)

                frames.append(frame)

        # Stack frames: [T, C, H, W]
        frames = torch.stack(frames)

        # Convert to float and normalize to [0, 1]
        frames = frames.float() / 255.0

        return frames

    def get_num_trajectories(self) -> int:
        """Return total number of trajectories."""
        return self.num_trajectories

    def get_seq_length(self, index: int) -> int:
        """Return sequence length without loading actions."""
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")
        return self._episode_length

    @property
    def action_dim(self) -> int:
        """CSGO action dimension (51)."""
        return 51

    @property
    def state_dim(self) -> int:
        """CSGO state dimension (0 - pure vision)."""
        return 0
