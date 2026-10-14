"""Base v2 (SD-VAE latent policy): render convention, eval-time encode path, latent_conv backbone."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gym_pusht")

from nwaug.sdvae import VAE_DIR  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

LATENT_FT = {"observation.latent": PolicyFeature(type=FeatureType.VISUAL, shape=(4, 32, 32))}
ACTION_FT = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}


def _cfg(**kw) -> DiffusionConfig:
    base = {
        "vision_backbone": "latent_conv",
        "pretrained_backbone_weights": None,
        "crop_shape": (28, 28),
        "n_obs_steps": 1,
        "horizon": 16,
        "n_action_steps": 8,
        "drop_n_last_frames": 8,
        "down_dims": (32, 64, 128),
        "input_features": LATENT_FT,
        "output_features": ACTION_FT,
        "device": "cpu",
    }
    return DiffusionConfig(**{**base, **kw})


def test_latent_conv_rejects_pretrained_weights() -> None:
    with pytest.raises(ValueError, match="trained from scratch"):
        _cfg(pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1")


def test_latent_conv_policy_trains_and_acts() -> None:
    torch.manual_seed(0)
    p = DiffusionPolicy(_cfg())
    enc = p.diffusion.rgb_encoder[0]
    assert enc.backbone(torch.zeros(1, 4, 28, 28)).shape == (1, 512, 14, 14)
    batch = {
        "observation.latent": torch.randn(2, 1, 4, 32, 32),
        "action": torch.randn(2, 16, 2),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    loss, _ = p.forward(batch)
    assert torch.isfinite(loss)
    p.eval()
    p.reset()
    assert p.select_action({"observation.latent": torch.randn(2, 4, 32, 32)}).shape == (2, 2)


def test_placement_order_changes_render() -> None:
    """The two placement orders render visibly different frames, so the order matters.
    convert_pusht_latent checks angle-first against every original 96px frame (MAE <= 0.35/255 over
    all 25650; position-first gives ~2.5/255)."""
    import gymnasium as gym
    from nwaug.pusht_state import place_body

    s5 = np.array([222.0, 97.0, 222.99382, 381.59903, 3.0079994])  # pusht_cchi_v7_replay state[0]
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=96, observation_height=96)
    env.reset(seed=0)
    u = env.unwrapped
    place_body(u, s5, zero_agent_vel=True)
    a = u._render(visualize=False).astype(np.float32)
    # position-first (gym_pusht's legacy _set_state order) from a fresh reset rotates the T about its
    # CoG after the origin is placed, shifting it by the rotated CoG offset
    env.reset(seed=0)
    u.agent.position = (float(s5[0]), float(s5[1]))
    u.block.position = (float(s5[2]), float(s5[3]))
    u.block.angle = float(s5[4])
    u.space.step(u.dt)
    b = u._render(visualize=False).astype(np.float32)
    assert np.abs(a - b).mean() > 1.0


@pytest.mark.skipif(not (VAE_DIR / "config.json").exists(), reason="SD-VAE weights not present")
def test_eval_encode_step_matches_dataset_encode() -> None:
    """eval: env uint8 -> preprocess_observation (/255, CHW) -> SdVaeEncodeStep must give exactly the
    latent the dataset converter wrote (encode_224 on the same uint8 render)."""
    import gymnasium as gym
    from eval_latent_policy import SdVaeEncodeStep
    from nwaug.sdvae import encode_224, load_sdvae

    from lerobot.envs.utils import preprocess_observation

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=224, observation_height=224)
    frames = np.stack([env.reset(seed=s)[0] for s in (1, 2)])  # (2,224,224,3) uint8
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = encode_224(load_sdvae(dev), frames, sample=False)
    obs = preprocess_observation({"pixels": frames})
    out = SdVaeEncodeStep(device=dev).observation(obs)
    assert "observation.image" not in out
    torch.testing.assert_close(out["observation.latent"], ref, rtol=0, atol=1e-5)
