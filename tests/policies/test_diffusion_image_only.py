"""Image-only (no observation.state) DiffusionPolicy coverage."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from tests.utils import require_cuda


def _image_batch(b: int = 2, device: str = "cpu", with_time: bool = False) -> dict[str, torch.Tensor]:
    img = torch.rand(b, 1, 3, 96, 96, device=device) if with_time else torch.rand(b, 3, 96, 96, device=device)
    return {OBS_IMAGE: img}


def _image_only_cfg(**overrides) -> DiffusionConfig:
    cfg = DiffusionConfig(
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        n_obs_steps=1,
        horizon=16,
        n_action_steps=8,
        crop_shape=(84, 84),
        pretrained_backbone_weights=None,
        down_dims=(32, 64, 128),
        num_inference_steps=2,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _train_batch(b: int = 2, device: str = "cpu", with_state: bool = False) -> dict[str, torch.Tensor]:
    batch = _image_batch(b, device=device, with_time=True)
    batch[ACTION] = torch.randn(b, 16, 2, device=device)
    batch["action_is_pad"] = torch.zeros(b, 16, dtype=torch.bool, device=device)
    if with_state:
        batch[OBS_STATE] = torch.randn(b, 1, 2, device=device)
    return batch


def test_image_only_config_has_no_state() -> None:
    cfg = _image_only_cfg()
    assert cfg.robot_state_feature is None
    cfg.validate_features()


def test_reset_has_no_state_queue() -> None:
    from lerobot.utils.constants import OBS_IMAGES

    policy = DiffusionPolicy(_image_only_cfg())
    policy.reset()
    assert OBS_STATE not in policy._queues
    assert OBS_IMAGES in policy._queues
    assert ACTION in policy._queues


def test_forward_backward_finite_grads_cpu() -> None:
    policy = DiffusionPolicy(_image_only_cfg())
    policy.train()
    batch = _train_batch()
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    enc_grad = next(p for p in policy.diffusion.rgb_encoder.parameters() if p.grad is not None)
    unet_grad = next(p for p in policy.diffusion.unet.parameters() if p.grad is not None)
    assert torch.isfinite(enc_grad.grad).all()
    assert torch.isfinite(unet_grad.grad).all()


def test_select_action_queue_drain_cpu() -> None:
    policy = DiffusionPolicy(_image_only_cfg())
    policy.eval()
    policy.reset()
    b = 2
    for _ in range(3):
        action = policy.select_action(_image_batch(b))
        assert action.shape == (b, 2)
    # after 8 actions the queue empties and a new chunk is generated; 3 calls leave 5
    assert len(policy._queues[ACTION]) == 5


def test_predict_action_chunk_shape_cpu() -> None:
    policy = DiffusionPolicy(_image_only_cfg())
    policy.eval()
    policy.reset()
    # populate queues via select_action path, or call with empty queues + batch
    chunk = policy.predict_action_chunk(_image_batch(2))
    assert chunk.shape == (2, 8, 2)


def test_backward_compat_with_state() -> None:
    cfg = _image_only_cfg()
    cfg.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }
    assert cfg.robot_state_feature is not None
    policy = DiffusionPolicy(cfg)
    policy.reset()
    assert OBS_STATE in policy._queues
    batch = _train_batch(with_state=True)
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


def test_save_load_roundtrip(tmp_path) -> None:
    policy = DiffusionPolicy(_image_only_cfg())
    policy.eval()
    before = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    policy.save_pretrained(tmp_path)
    loaded = DiffusionPolicy.from_pretrained(tmp_path)
    after = loaded.state_dict()
    assert set(before) == set(after)
    for k in before:
        assert torch.equal(before[k], after[k].cpu())


@require_cuda
@pytest.mark.parametrize(
    "fn",
    [
        "forward_backward",
        "select_action",
        "predict_chunk",
    ],
)
def test_image_only_cuda(fn: str) -> None:
    device = "cuda"
    policy = DiffusionPolicy(_image_only_cfg(device=device)).to(device)
    if fn == "forward_backward":
        policy.train()
        loss, _ = policy.forward(_train_batch(device=device))
        assert torch.isfinite(loss)
        loss.backward()
    elif fn == "select_action":
        policy.eval()
        policy.reset()
        a = policy.select_action(_image_batch(2, device=device))
        assert a.shape == (2, 2)
        assert a.device.type == "cuda"
    else:
        policy.eval()
        policy.reset()
        chunk = policy.predict_action_chunk(_image_batch(2, device=device))
        assert chunk.shape == (2, 8, 2)
