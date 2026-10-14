#!/usr/bin/env python
"""Original PushT demos -> SD-VAE latent dataset for the latent base policy (base v2).

Same 206 episodes / 25650 frames / fps 10 / actions as `local/pusht_imgonly` (the v1 base data), but
the observation is `observation.latent` (4,32,32) float32: each frame's full sim state is placed in
gym_pusht, rendered at NanoWM's 224px, and SD-VAE encoded exactly like NanoWM's context frames
(`nwaug.sdvae.encode_224`, posterior mode, TF32 off).

Source: diffusion_policy's pusht_cchi_v7_replay.zarr (the dataset lerobot/pusht was converted from;
lerobot/pusht itself drops the block pose, so it cannot be re-rendered). The zarr is converted to npz
once, in a throwaway env (zarr is not a lerobot dependency):

    curl -L -o ~/Downloads/diffusion_policy_pusht/pusht.zip \\
        https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip   # sha256 63d52a11...
    cd ~/Downloads/diffusion_policy_pusht && unzip pusht.zip
    uv run --no-project --with "zarr<3" --with numpy python -c "import zarr, numpy as np; \\
        z = zarr.open('pusht/pusht_cchi_v7_replay.zarr', 'r'); \\
        np.savez('pusht_cchi_v7_replay.npz', episode_ends=z['meta/episode_ends'][:], \\
                 **{k: z['data'][k][:] for k in z['data'].keys()})"

    uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py \\
        pusht-latent --overwrite

State convention (checked, not assumed): the zarr `state` is body-origin
[agent_x, agent_y, block_x, block_y, angle] and must be placed angle-first (`place_body`). Every
frame is ALSO rendered at 96px and compared with the zarr's own 96px image; the run aborts if the
mean abs error of any frame exceeds `--max-render-mae` (placing position-first gives ~2.5/255, the
correct order ~0.08/255). After finalize, the actions are checked against lerobot/pusht and the
`observation.latent` stats in meta/stats.json are rewritten per channel, shape (4,1,1) like image
stats, so MEAN_STD normalization is per VAE channel rather than per latent pixel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset_converter import ConverterConfig, LeRobotDatasetConverter

SRC_NPZ = Path("~/Downloads/diffusion_policy_pusht/pusht_cchi_v7_replay.npz")
FEATURES = {
    "observation.latent": {"dtype": "float32", "shape": (4, 32, 32), "names": ["channel", "height", "width"]},
    "action": {"dtype": "float32", "shape": (2,), "names": {"axes": ["x", "y"]}},
}
FPS = 10  # lerobot/pusht / diffusion_policy pusht control rate
QUANTILES = (1, 10, 50, 90, 99)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class PushtLatentConverter(LeRobotDatasetConverter):
    name = "pusht-latent"
    DEFAULT_REPO_ID = "local/pusht_latent"

    def __init__(
        self,
        config: ConverterConfig,
        src_npz: Path,
        device: str,
        max_episodes: int | None,
        max_render_mae: float,
        check_repo: str | None,
    ):
        super().__init__(config)
        self.src_npz = src_npz
        self.device = device
        self.max_episodes = max_episodes
        self.max_render_mae = max_render_mae
        self.check_repo = check_repo
        self.render_mae: list[float] = []
        self.latents: list[np.ndarray] = []

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--src-npz", type=Path, default=SRC_NPZ)
        p.add_argument("--device", default="cuda")
        p.add_argument("--max-episodes", type=int, default=None, help="smoke-test with a few episodes")
        p.add_argument("--max-render-mae", type=float, default=0.5, help="per-frame 96px re-render MAE bound")
        p.add_argument(
            "--check-repo", default="lerobot/pusht", help="actions must match this dataset ('' = skip)"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PushtLatentConverter:
        src = args.src_npz.expanduser()
        cfg = ConverterConfig(
            repo_id=args.repo_id,
            fps=FPS,
            features=FEATURES,
            root=args.root,
            robot_type="pusht_sim",
            use_videos=False,
            overwrite=args.overwrite,
            image_writer_threads=args.image_writer_threads,
            extra={"src_npz": str(src)},
        )
        return cls(cfg, src, args.device, args.max_episodes, args.max_render_mae, args.check_repo or None)

    def load_source(self) -> None:
        import gym_pusht  # noqa: F401
        import gymnasium as gym
        from nwaug.sdvae import RENDER_SIZE, load_sdvae

        d = np.load(self.src_npz)
        self.state, self.action, self.img96 = d["state"], d["action"], d["img"]
        ends = d["episode_ends"].astype(np.int64)
        self.bounds = list(zip(np.r_[0, ends[:-1]], ends, strict=True))
        if self.max_episodes:
            self.bounds = self.bounds[: self.max_episodes]
        mk = {"obs_type": "pixels", "render_mode": "rgb_array"}
        self.env224 = gym.make(
            "gym_pusht/PushT-v0", observation_width=RENDER_SIZE, observation_height=RENDER_SIZE, **mk
        )
        self.env96 = gym.make("gym_pusht/PushT-v0", observation_width=96, observation_height=96, **mk)
        self.env224.reset(seed=0)
        self.env96.reset(seed=0)
        self.vae = load_sdvae(self.device)
        print(
            f"source: {self.src_npz} sha256={_sha256(self.src_npz)[:16]}  episodes={len(self.bounds)}  "
            f"frames={self.bounds[-1][1] - self.bounds[0][0]}  vae={self.vae.config._name_or_path}"
        )

    def _render(self, env, s5) -> np.ndarray:
        from nwaug.pusht_state import place_body

        u = env.unwrapped
        place_body(u, s5, zero_agent_vel=True)
        return u._render(visualize=False)

    def iter_episodes(self):
        for b in self.bounds:
            yield self._frames(*b)

    def _frames(self, i0: int, i1: int):
        from nwaug.sdvae import encode_224

        f224 = np.stack([self._render(self.env224, self.state[i]) for i in range(i0, i1)])
        for i in range(i0, i1):
            mae = float(
                np.abs(self._render(self.env96, self.state[i]).astype(np.float32) - self.img96[i]).mean()
            )
            if mae > self.max_render_mae:
                raise RuntimeError(f"frame {i}: 96px re-render MAE {mae:.3f} > {self.max_render_mae}")
            self.render_mae.append(mae)
        lat = encode_224(self.vae, f224, sample=False).cpu().numpy().astype(np.float32)
        self.latents.append(lat)
        for k, i in enumerate(range(i0, i1)):
            yield {"observation.latent": lat[k], "action": self.action[i].astype(np.float32)}

    def on_episode_end(self, episode_index: int, n_frames: int) -> None:
        if (episode_index + 1) % 25 == 0:
            print(
                f"  {episode_index + 1}/{len(self.bounds)} episodes  render MAE max {max(self.render_mae):.3f}"
            )

    def post_run(self) -> None:
        n = self.bounds[-1][1] - self.bounds[0][0]
        if self.check_repo:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ref = np.asarray(LeRobotDataset(self.check_repo).hf_dataset["action"], np.float32)[:n]
            err = np.abs(ref - self.action[:n]).max()
            if err > 1e-4:
                raise RuntimeError(f"actions differ from {self.check_repo}: max |d| = {err}")
            print(f"actions match {self.check_repo} (max |d| = {err:.1e})")
        # per-channel latent stats, (4,1,1) like image stats
        lat = np.concatenate(self.latents)  # (N,4,32,32)
        c = np.moveaxis(lat, 1, -1).reshape(-1, 4).astype(np.float64)
        per_ch = {
            "min": c.min(0),
            "max": c.max(0),
            "mean": c.mean(0),
            "std": c.std(0),
            **{f"q{q:02d}": np.percentile(c, q, axis=0) for q in QUANTILES},
        }
        stats_path = Path(self.config.root) / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text())
        stats["observation.latent"] = {k: v.reshape(4, 1, 1).tolist() for k, v in per_ch.items()}
        stats["observation.latent"]["count"] = [int(len(lat))]
        stats_path.write_text(json.dumps(stats, indent=4))
        self.latent_mean, self.latent_std = per_ch["mean"], per_ch["std"]

    def summarize(self) -> str:
        mae = np.asarray(self.render_mae)
        return (
            super().summarize()
            + f"\n  96px re-render MAE mean {mae.mean():.3f} max {mae.max():.3f} (/255)"
            + f"\n  latent per-channel mean {np.round(self.latent_mean, 4).tolist()} std {np.round(self.latent_std, 4).tolist()}"
        )


if __name__ == "__main__":
    PushtLatentConverter.main()
