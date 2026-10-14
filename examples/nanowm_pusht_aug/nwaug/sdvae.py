"""SD-VAE encode of gym_pusht 224px renders -> NanoWM-space latents (4,32,32), for the latent policy (base v2).

ONLY torch + diffusers imports: this file is loaded in the lerobot venv (dataset conversion, sim eval)
and must stay importable in the nanowm venv too, so the two sides can be checked for parity.

The preprocessing is byte-for-byte `imagined_rollout.encode_context_frames` (NanoWM's own path for
real context frames):

    uint8 (224,224,3) -> /255 -> CHW -> bilinear stretch to (256,256), align_corners=False
    -> *2-1 -> vae.encode (fp32, autocast off) -> latent_dist.{sample,mode}() * scaling_factor

NanoWM draws `sample()`; `mode()` (the posterior mean) is the deterministic version. Which one a
caller uses is explicit (`sample=`), never a default. On PushT renders the scaled posterior std is
~2e-5 (max 2e-4) against a latent std of ~0.4, so the two are interchangeable in practice.

TF32 (cuDNN convs and CUDA matmuls) is switched OFF for the encode and restored after: with cuDNN
TF32 on (PyTorch's default for convs; lerobot-eval also turns matmul TF32 on) the same frames move
by up to 4e-2; with it off this path matches NanoWM's own encode in the
nanowm venv to max 3e-4 / mean 2e-6.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# hf stabilityai/sd-vae-ft-mse @ 31f26fde, pinned in configs/nanowm_pusht_common.yaml
VAE_DIR = Path(os.environ.get("VAE_MODEL_PATH", "~/models/sd-vae-ft-mse")).expanduser()
RENDER_SIZE = 224  # gym_pusht observation size NanoWM was trained on
VAE_IMG_SIZE = 256  # NanoWM ckpt model.image_size
LATENT_SHAPE = (4, 32, 32)


def load_sdvae(device: str | torch.device, path: str | Path | None = None):
    """AutoencoderKL in fp32 eval mode, no grad."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        str(Path(path).expanduser() if path else VAE_DIR), torch_dtype=torch.float32
    )
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


def preprocess_224(frames: np.ndarray | torch.Tensor, device: str | torch.device) -> torch.Tensor:
    """(N,224,224,3) uint8 -> (N,3,256,256) float32 in [-1,1] on `device`."""
    f = torch.as_tensor(frames)
    if f.dtype != torch.uint8 or f.ndim != 4 or tuple(f.shape[1:]) != (RENDER_SIZE, RENDER_SIZE, 3):
        raise ValueError(f"expected (N,{RENDER_SIZE},{RENDER_SIZE},3) uint8, got {tuple(f.shape)} {f.dtype}")
    f = f.to(device).float().permute(0, 3, 1, 2).div(255.0)
    f = F.interpolate(f, size=(VAE_IMG_SIZE, VAE_IMG_SIZE), mode="bilinear", align_corners=False)
    return f * 2.0 - 1.0


@torch.no_grad()
def encode_224(
    vae,
    frames: np.ndarray | torch.Tensor,
    *,
    sample: bool,
    batch: int = 64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(N,224,224,3) uint8 -> (N,4,32,32) float32 scaled latents on the VAE's device."""
    dev = vae.device
    out = []
    prev_tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for i in range(0, len(frames), batch):
            x = preprocess_224(frames[i : i + batch], dev)
            with torch.autocast(
                device_type=dev.type if dev.type in ("cuda", "cpu") else "cuda", enabled=False
            ):
                dist = vae.encode(x).latent_dist
                z = dist.sample(generator=generator) if sample else dist.mode()
            out.append(z * vae.config.scaling_factor)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return torch.cat(out, dim=0)
