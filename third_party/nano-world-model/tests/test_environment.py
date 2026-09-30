"""Small real operations that expose dependency and binary API incompatibilities."""

import torch
from diffusers import AutoencoderKL
from models.nanowm import NanoWM


def test_vae_and_world_model_backward():
    torch.set_num_threads(2)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, block_out_channels=(32,),
        down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        latent_channels=4, norm_num_groups=32, sample_size=8,
    )
    with torch.no_grad():
        latents = vae.encode(torch.randn(2, 3, 8, 8)).latent_dist.sample()
        assert torch.isfinite(vae.decode(latents).sample).all()
    model = NanoWM(input_size=8, patch_size=2, in_channels=4, hidden_size=64,
                   depth=2, num_heads=4, num_frames=2, extras=1,
                   use_action=True, action_dim=2, action_injection_type="additive")
    from diffusion import create_diffusion
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100,
                                 noise_schedule="squaredcos_cap_v2", pred_name="v",
                                 snr_gamma=5.0, zero_terminal_snr=True)
    losses = diffusion.training_losses(
        model, latents.unsqueeze(0), torch.tensor([[1, 50]]),
        {"y": None, "action": torch.randn(1, 2, 2)},
    )
    loss = losses["loss"].mean()
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
