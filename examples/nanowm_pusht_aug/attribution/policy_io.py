"""Shared loading for the attribution scripts: config, policy + preprocessor, and the training set
rebuilt exactly as lerobot-train saw it, so every scored index is a sample the policy trained on."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

# same directory; the attribution scripts put it on sys.path before importing either module
from config_io import load_config  # noqa: F401 -- re-exported so scripts can import it from here
from torch.utils.data import default_collate

from lerobot.datasets import EpisodeAwareSampler, LeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_IMAGE


def load_policy(ckpt: str | Path, device: str = "cuda"):
    policy = DiffusionPolicy.from_pretrained(str(ckpt)).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def load_train_dataset(repo_id: str, policy_config) -> LeRobotDataset:
    """No image_transforms: augmentation is handled (or deliberately not) by the gradient code."""
    root = HF_LEROBOT_HOME / repo_id
    meta_ds = LeRobotDataset(repo_id, root=root)
    delta = resolve_delta_timestamps(policy_config, meta_ds.meta)
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta)


def trained_indices(ds: LeRobotDataset, drop_n_last_frames: int) -> np.ndarray:
    """Every index EpisodeAwareSampler can emit -- the exact sample set the checkpoint trained on."""
    sampler = EpisodeAwareSampler(
        ds.meta.episodes["dataset_from_index"],
        ds.meta.episodes["dataset_to_index"],
        drop_n_last_frames=drop_n_last_frames,
        shuffle=False,
        absolute_to_relative_idx=ds.absolute_to_relative_idx,
    )
    return np.fromiter(iter(sampler), dtype=np.int64)


def normalized_batch(pre, items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Dataset items -> (images (B,C,H,W), actions (B,horizon,A)), both normalized by the
    checkpoint's own (real-pinned) stats."""
    batch = pre(copy.copy(default_collate(items)))
    return batch[OBS_IMAGE], batch[ACTION]
