#!/usr/bin/env python
"""Pack NanoWM imagined-rollout .npz (from rollout_imagine.py) into a LeRobotDataset.

Stage 2 of 2 (lerobot venv). Output schema matches the pusht_noise image-only sets so
the two mix in one dataloader. Accepts a single .npz or a directory of shard_*.npz.

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py \
        imagined examples/nanowm_pusht_aug/imagined/run1 --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset_converter import ConverterConfig, LeRobotDatasetConverter

FEATURES = {
    "observation.image": {"dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": (2,), "names": {"axes": ["target_x", "target_y"]}},
}


class ImaginedRolloutPacker(LeRobotDatasetConverter):
    name = "imagined"
    DEFAULT_REPO_ID = "local/pusht_noise_224_imgonly_nanoWM_fi5"

    def __init__(self, config: ConverterConfig, src: Path):
        super().__init__(config)
        self.src = src

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("src", type=Path, help="a shard_*.npz, imagined_rollouts.npz, or a dir of shards")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ImaginedRolloutPacker:
        fps, _ = _peek_fps(args.src)
        cfg = ConverterConfig(
            repo_id=args.repo_id,
            fps=fps,
            features=FEATURES,
            root=args.root,
            robot_type="pusht_sim_nanowm",
            overwrite=args.overwrite,
            image_writer_threads=args.image_writer_threads,
            extra={"src": str(args.src)},
        )
        return cls(cfg, src=args.src)

    def load_source(self) -> None:
        self.npzs, self.meta = _list_shards(self.src)
        img_parts, act_parts, len_parts = [], [], []
        for n in self.npzs:
            d = np.load(n, allow_pickle=True)
            img_parts.append(d["images"])
            act_parts.append(d["actions"])
            len_parts.append(d["ep_lens"])
        self.imgs = np.concatenate(img_parts)
        self.acts = np.concatenate(act_parts)
        self.ep_lens = np.concatenate(len_parts)
        assert self.imgs.shape[1:] == (224, 224, 3), self.imgs.shape
        assert int(self.ep_lens.sum()) == len(self.imgs) == len(self.acts)
        print(
            f"{len(self.npzs)} shard(s): {len(self.ep_lens)} episodes, {len(self.imgs)} frames, "
            f"fps={self.config.fps}  (meta: {self.meta})"
        )

    def iter_episodes(self):
        c = 0
        for n in self.ep_lens:
            yield [{"observation.image": self.imgs[c + i], "action": self.acts[c + i]} for i in range(int(n))]
            c += int(n)

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:
        if (episode_index + 1) % 200 == 0:
            print(f"  {episode_index + 1}/{len(self.ep_lens)}")


def _list_shards(src: Path):
    if src.is_dir():
        npzs = sorted(src.glob("shard_*.npz")) or sorted(src.glob("*.npz"))
        meta_path = src / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        npzs, meta = [src], {}
    if not npzs:
        raise FileNotFoundError(f"no .npz under {src}")
    if not meta:
        for n in npzs:
            d = np.load(n, allow_pickle=True)
            if "meta" in d:
                meta = json.loads(str(d["meta"]))
                break
    return npzs, meta


def _peek_fps(src: Path) -> tuple[int, dict]:
    _, meta = _list_shards(src)
    return int(meta.get("fps", 2)), meta


if __name__ == "__main__":
    ImaginedRolloutPacker.main()
