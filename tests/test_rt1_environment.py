"""A real local LeRobot video dataset, without Hub access or system FFmpeg."""

import importlib.util

import numpy as np
import pytest
import torch


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="RT-1 extra not installed")
def test_lerobot_record_and_load_video(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from wm_datasets.data_source.factory import create_data_source

    root = tmp_path / "local-dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/nanowm-smoke", root=root, fps=4, video_backend="pyav",
        features={
            "observation.images.image": {
                "dtype": "video", "shape": (3, 64, 64),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
        },
    )
    for frame in range(4):
        dataset.add_frame({
            "observation.images.image": np.full((64, 64, 3), 30 + frame * 40, dtype=np.uint8),
            "observation.state": np.array([frame, frame + 1, frame + 2], dtype=np.float32),
            "action": np.array([frame, -frame], dtype=np.float32),
        }, task="environment smoke")
    dataset.save_episode()
    source = create_data_source("rt1", "local/nanowm-smoke", root=str(root))
    assert source.dataset.video_backend == "pyav"
    assert source.get_num_trajectories() == 1
    assert source.action_dim == 2 and source.state_dim == 3
    frames = source.load_visual_frames(0, 0, 3)
    assert frames.shape == (3, 3, 64, 64)
    assert torch.isfinite(frames).all() and 0 <= frames.min() <= frames.max() <= 1
    assert frames[0].mean() < frames[1].mean() < frames[2].mean()
