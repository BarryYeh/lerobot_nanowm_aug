#!/usr/bin/env python
"""Image-only copy of lerobot/pusht (drop observation.state).

The diffusion policy factory fills `input_features` from the dataset only when it is
unset (factory.py), so a dataset carrying no `observation.state` yields an image-only
policy with no code change. fps 10 is kept native. Every frame is decoded from the
source av1 video, visited in global index order so the decode stays sequential.

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py \
        pusht-imgonly --overwrite
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .dataset_converter import ConverterConfig, LeRobotDatasetConverter

SRC_REPO = "lerobot/pusht"
FEATURES = {
    "observation.image": {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": (2,), "names": {"axes": ["x", "y"]}},
}


class PushtImgOnlyConverter(LeRobotDatasetConverter):
    name = "pusht-imgonly"
    DEFAULT_REPO_ID = "local/pusht_imgonly"

    def __init__(self, config: ConverterConfig, src_repo: str = SRC_REPO, max_episodes: int | None = None):
        super().__init__(config)
        self.src_repo = src_repo
        self.max_episodes = max_episodes

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--src-repo", default=SRC_REPO)
        p.add_argument("--max-episodes", type=int, default=None, help="smoke-test with a few episodes")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PushtImgOnlyConverter:
        fps = LeRobotDataset(args.src_repo).meta.fps
        cfg = ConverterConfig(
            repo_id=args.repo_id,
            fps=fps,
            features=FEATURES,
            root=args.root,
            robot_type="pusht_sim",
            overwrite=args.overwrite,
            image_writer_threads=args.image_writer_threads,
            extra={"src_repo": args.src_repo},
        )
        return cls(cfg, src_repo=args.src_repo, max_episodes=args.max_episodes)

    def load_source(self) -> None:
        self.src = LeRobotDataset(self.src_repo)
        self.ep_idx = np.asarray(self.src.hf_dataset["episode_index"])
        self.n_src = int(self.ep_idx.max()) + 1
        if self.max_episodes:
            self.n_src = min(self.n_src, self.max_episodes)
        n_frames = int((self.ep_idx < self.n_src).sum())
        print(f"source: {self.src_repo}  fps={self.src.meta.fps}  episodes={self.n_src}  frames={n_frames}")

    def iter_episodes(self):
        for ep in range(self.n_src):
            yield self._frames(ep)

    def _frames(self, ep: int):
        for i in np.nonzero(self.ep_idx == ep)[0]:
            item = self.src[int(i)]
            yield {
                "observation.image": (item["observation.image"].permute(1, 2, 0).numpy() * 255.0).astype(
                    np.uint8
                ),
                "action": item["action"].numpy().astype(np.float32),
            }

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:
        if (episode_index + 1) % 50 == 0:
            print(f"  {episode_index + 1}/{self.n_src} episodes")


if __name__ == "__main__":
    PushtImgOnlyConverter.main()
