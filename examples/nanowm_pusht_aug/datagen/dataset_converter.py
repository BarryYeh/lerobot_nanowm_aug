#!/usr/bin/env python
"""Shared base for the scenario converters that emit a LeRobotDataset.

Every converter in this package (pusht_noise -> lerobot, image-only copy, denoise,
imagined-rollout packing) follows the same lifecycle:

    resolve root -> LeRobotDataset.create -> per episode { add_frame* ; save_episode } -> finalize

This class owns that lifecycle. A subclass supplies only:

  * ``name`` / ``DEFAULT_REPO_ID``  -- registry key + default output repo id
  * ``add_cli_args(parser)``        -- extra CLI flags                     (optional)
  * ``from_args(args) -> instance`` -- build a configured converter from parsed args
  * ``load_source()``              -- open archives / source dataset        (optional)
  * ``iter_episodes()``           -- yield one frame-iterable per episode
  * ``on_episode_end(idx, n)``     -- progress / stat accumulation          (optional)
  * ``post_run()`` / ``summarize()`` -- verification pass / final log line   (optional)

Subclasses register themselves by setting ``name``; ``run_converter.py`` dispatches on it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_TASK = "Push the T-shaped block to the target."


@dataclass
class ConverterConfig:
    """The kwargs `LeRobotDataset.create` needs, plus root/overwrite handling."""

    repo_id: str
    fps: int
    features: dict
    root: Path | None = None  # default: $HF_LEROBOT_HOME/<repo_id>
    robot_type: str = "pusht_sim"
    task: str = DEFAULT_TASK
    use_videos: bool = True
    image_writer_threads: int = 4
    overwrite: bool = False
    extra: dict = field(default_factory=dict)  # free-form provenance kept in the log


class LeRobotDatasetConverter(ABC):
    name: str = ""  # registry key for run_converter.py
    DEFAULT_REPO_ID: str = ""
    registry: dict[str, type[LeRobotDatasetConverter]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", ""):
            LeRobotDatasetConverter.registry[cls.name] = cls

    @classmethod
    def help_text(cls) -> str:
        """Class docstring if set, else the module docstring (where the usage lives)."""
        return cls.__doc__ or getattr(sys.modules.get(cls.__module__), "__doc__", "") or ""

    def __init__(self, config: ConverterConfig) -> None:
        self.config = config
        self.dataset: LeRobotDataset | None = None
        self.n_episodes = 0
        self.n_frames = 0

    # ------------------------------------------------------------------ lifecycle
    def run(self) -> LeRobotDataset:
        self._resolve_root()
        self.load_source()
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            features=self.config.features,
            root=self.config.root,
            robot_type=self.config.robot_type,
            use_videos=self.config.use_videos,
            image_writer_threads=self.config.image_writer_threads,
        )
        for episode in self.iter_episodes():
            n = 0
            for frame in episode:
                frame = dict(frame)
                frame.setdefault("task", self.config.task)
                self.dataset.add_frame(frame)
                n += 1
            self.dataset.save_episode()
            self.on_episode_end(self.n_episodes, n)
            self.n_episodes += 1
            self.n_frames += n
        self.dataset.finalize()
        self.post_run()
        print(self.summarize())
        return self.dataset

    def _resolve_root(self) -> None:
        root = Path(self.config.root or (HF_LEROBOT_HOME / self.config.repo_id)).expanduser()
        if root.exists():
            if not self.config.overwrite:
                raise FileExistsError(f"{root} exists; pass --overwrite")
            shutil.rmtree(root)
        self.config.root = root

    # ------------------------------------------------------------ subclass hooks
    @abstractmethod
    def iter_episodes(self) -> Iterator[Iterable[Mapping]]:
        """Yield one iterable of per-frame `add_frame` payloads per episode.

        `task` is injected by `run()` when a frame omits it.
        """

    def load_source(self) -> None:  # noqa: B027 -- opt-in hook, deliberately not abstract
        """Open source archives / datasets. Runs after root is resolved, before create()."""

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:  # noqa: B027 -- opt-in hook, deliberately not abstract
        """Progress logging / stat accumulation for the episode just written."""

    def post_run(self) -> None:  # noqa: B027 -- opt-in hook, deliberately not abstract
        """Optional verification pass after finalize() (e.g. dump check videos)."""

    def summarize(self) -> str:
        return (
            f"done -> {self.config.root}  "
            f"({self.n_episodes} episodes, {self.n_frames} frames, fps {self.config.fps}, "
            f"robot_type {self.config.robot_type})"
        )

    # -------------------------------------------------------------------- CLI glue
    @staticmethod  # noqa: B027 -- opt-in hook, deliberately not abstract
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Register converter-specific flags. Base flags are added by `base_parser`."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> LeRobotDatasetConverter:
        """Build a fully configured converter instance from parsed args."""

    @classmethod
    def base_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog=f"run_converter.py {cls.name}" if cls.name else cls.__name__,
            description=cls.help_text(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        p.add_argument("--repo-id", default=cls.DEFAULT_REPO_ID)
        p.add_argument("--root", type=Path, default=None, help="default: $HF_LEROBOT_HOME/<repo-id>")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--image-writer-threads", type=int, default=4)
        cls.add_cli_args(p)
        return p

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        args = cls.base_parser().parse_args(argv)
        cls.from_args(args).run()
