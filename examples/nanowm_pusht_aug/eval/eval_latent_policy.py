#!/usr/bin/env python
"""lerobot-eval for the SD-VAE latent base policy (base v2): gym_pusht renders at 224px, and every
observation is SD-VAE encoded (`nwaug.sdvae.encode_224`, the same path the training data took) into
`observation.latent` before the policy's own preprocessor.

Only two things in `lerobot.scripts.lerobot_eval` are swapped, everything else (rollout, seeding --
episode i of batch b uses seed start_seed + b*batch + i --, success = gym_pusht `is_success`,
eval_info.json) is stock, so results pair by episode order with `proto_eval_pusht.sh` runs of the
v1 RGB base:

  * `make_env`                      -> the same SyncVectorEnv / AutoresetMode.SAME_STEP construction as
                                       `EnvConfig.create_envs`, plus observation_width/height=224
                                       (the stock PushtEnv config never passes them, so gym_pusht
                                       renders its default 96px)
  * `make_env_pre_post_processors`  -> env preprocessor = [SdVaeEncodeStep]
  * `make_policy`                   -> adds observation.image -> observation.latent to `rename_map`
                                       for its env-vs-policy feature-name check only (the env still
                                       emits `pixels`; the policy preprocessor's rename step then
                                       finds no `observation.image` left and is a no-op)

Takes the same CLI as lerobot-eval:

    uv run --extra training python examples/nanowm_pusht_aug/eval/eval_latent_policy.py \\
        --policy.path=<ckpt> --policy.device=cuda --policy.use_amp=false --policy.n_action_steps=10 \\
        --env.type=pusht --eval.n_episodes=1000 --eval.batch_size=50 --eval.use_async_envs=false \\
        --seed=100000 --output_dir=<out>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nwaug.sdvae import LATENT_SHAPE, RENDER_SIZE, encode_224, load_sdvae  # noqa: E402

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.processor import ObservationProcessorStep, PolicyProcessorPipeline  # noqa: E402
from lerobot.scripts import lerobot_eval as le  # noqa: E402
from lerobot.utils.constants import OBS_IMAGE  # noqa: E402

OBS_LATENT = "observation.latent"


@dataclass
class SdVaeEncodeStep(ObservationProcessorStep):
    """`observation.image` (B,3,224,224) float in [0,1] -> `observation.latent` (B,4,32,32).

    `preprocess_observation` made the image as uint8/255, so `round(x*255)` recovers the render
    exactly and `encode_224` sees the same uint8 frames (same NHWC layout) the dataset converter
    rendered.
    """

    device: str = "cuda"
    _vae: object = field(default=None, init=False, repr=False)

    def observation(self, observation):
        if self._vae is None:
            self._vae = load_sdvae(self.device)
        obs = dict(observation)
        img = obs.pop(OBS_IMAGE)
        if tuple(img.shape[1:]) != (3, RENDER_SIZE, RENDER_SIZE):
            raise ValueError(f"expected (B,3,{RENDER_SIZE},{RENDER_SIZE}) renders, got {tuple(img.shape)}")
        # contiguous NHWC, like the numpy renders the converter encoded: a strided view takes other
        # interpolate / conv kernels and moves the latent by up to 3e-4
        u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        obs[OBS_LATENT] = encode_224(self._vae, u8, sample=False)
        return obs

    def transform_features(self, features):
        obs = dict(features[PipelineFeatureType.OBSERVATION])
        obs.pop(OBS_IMAGE, None)
        obs[OBS_LATENT] = PolicyFeature(type=FeatureType.VISUAL, shape=LATENT_SHAPE)
        return {**features, PipelineFeatureType.OBSERVATION: obs}


def _make_env_224(cfg, n_envs: int = 1, use_async_envs: bool = False, **_):
    if cfg.type != "pusht":
        raise ValueError(f"eval_latent_policy only supports --env.type=pusht, got {cfg.type}")
    import gym_pusht  # noqa: F401

    kwargs = {**cfg.gym_kwargs, "observation_width": RENDER_SIZE, "observation_height": RENDER_SIZE}
    env_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv

    def _make_one():
        return gym.make(cfg.gym_id, disable_env_checker=cfg.disable_env_checker, **kwargs)

    from gymnasium.vector import AutoresetMode

    vec = env_cls([_make_one for _ in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
    return {cfg.type: {0: vec}}


def _make_env_processors(env_cfg, policy_cfg):
    return (
        PolicyProcessorPipeline(steps=[SdVaeEncodeStep(device=str(policy_cfg.device))]),
        PolicyProcessorPipeline(steps=[]),
    )


_orig_make_policy = le.make_policy


def _make_policy(cfg, env_cfg=None, rename_map=None, **kwargs):
    return _orig_make_policy(
        cfg, env_cfg=env_cfg, rename_map={OBS_IMAGE: OBS_LATENT, **(rename_map or {})}, **kwargs
    )


le.make_env = _make_env_224
le.make_env_pre_post_processors = _make_env_processors
le.make_policy = _make_policy

if __name__ == "__main__":
    le.main()
