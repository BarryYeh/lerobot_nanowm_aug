"""Per-sample gradients of CUPID's model-output function for lerobot's DiffusionPolicy.

Port of CUPID's DiffusionHybridImageFunctionalModelOutput (codes/cupid/diffusion_policy/
data_attribution/modelout_functions.py). Differences from CUPID, none of which change the quantity:
  * no inpainting condition mask: lerobot conditions only through global_cond, so every action slot
    is in the loss -- auto-padded slots included, because the checkpoint trained with
    do_mask_loss_for_padding=False.
  * normalization lives in the preprocessor, outside the policy: inputs here are ALREADY normalized.
  * the crop happens here, outside vmap (torchvision RandomCrop calls .item(), which vmap rejects).
    The encoder then receives an 84x84 image, where its eval-mode CenterCrop(84) is the identity.
    Test samples always use the center crop (what the policy saw in the rollout); train/pool samples
    use `train_crop` = center (CUPID's choice) or random (one fresh crop per diffusion timestep).
Approximations inherited from CUPID, which scores in policy.eval(): BatchNorm (use_group_norm=False)
uses running stats instead of batch stats, and photometric augmentation is off.

The T diffusion timesteps of one sample run as ONE inner batch inside the vmapped function, so its
gradient is already the timestep mean (linearity) and no (B, P) accumulator is needed.

T0 fail-fast (equivalence against compute_loss, sketch sanity, micro-batch benchmark):
    uv run --extra training python examples/nanowm_pusht_aug/attribution/diffusion_influence_loss.py \
        --config configs/cupid_filter_untrimmed464.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grad_sketch import CountSketch  # noqa: E402
from policy_io import load_config, load_policy, load_train_dataset, trained_indices  # noqa: E402

ENC = "rgb_encoder.0."
UNET = "unet."


class DiffusionInfluence:
    def __init__(self, policy, param_prefixes: list[str], loss_fn: str, num_timesteps: int):
        cfg = policy.config
        assert cfg.n_obs_steps == 1 and len(cfg.image_features) == 1, "port assumes 1 camera, 1 obs step"
        assert cfg.robot_state_feature is None and cfg.env_state_feature is None, "port assumes image-only"
        assert loss_fn in ("square", "ddpm")
        self.model = policy.diffusion.eval()
        self.enc = self.model.rgb_encoder[0]
        self.loss_fn = loss_fn
        self.num_timesteps = num_timesteps
        self.crop = tuple(cfg.crop_shape)
        self.epsilon = cfg.prediction_type == "epsilon"
        self.n_train_t = self.model.noise_scheduler.config.num_train_timesteps
        acp = self.model.noise_scheduler.alphas_cumprod.to(next(self.model.parameters()).device)
        self.sqrt_acp, self.sqrt_1m_acp = acp.sqrt(), (1 - acp).sqrt()
        self.params = {
            n: p.detach()
            for n, p in self.model.named_parameters()
            if any(n.startswith(x) for x in param_prefixes)
        }
        self.shapes = {n: p.shape for n, p in self.params.items()}
        self.dim = sum(p.numel() for p in self.params.values())

    def _loss(self, params, img, action, t, eps):
        """One sample. img (C,h,w) or (T,C,h,w) already cropped; action (H,A); t (T,); eps (T,H,A)."""
        enc_p = {k[len(ENC) :]: v for k, v in params.items() if k.startswith(ENC)}
        unet_p = {k[len(UNET) :]: v for k, v in params.items() if k.startswith(UNET)}
        T = t.shape[0]
        feat = functional_call(self.enc, enc_p, (img if img.dim() == 4 else img[None],))
        cond = feat.expand(T, -1)
        x = action.expand(T, *action.shape)
        noisy = self.sqrt_acp[t][:, None, None] * x + self.sqrt_1m_acp[t][:, None, None] * eps
        pred = functional_call(self.model.unet, unet_p, (noisy, t), {"global_cond": cond})
        if self.loss_fn == "square":  # CUPID default: target-free
            return (pred**2).mean()
        return ((pred - (eps if self.epsilon else x)) ** 2).mean()

    def _crop(self, images: torch.Tensor, mode: str, rng: np.random.Generator) -> torch.Tensor:
        h, w = self.crop
        H, W = images.shape[-2:]
        if mode == "center":
            i, j = int(round((H - h) / 2.0)), int(round((W - w) / 2.0))  # torchvision center_crop
            return images[..., i : i + h, j : j + w]
        # random: independent crop per (sample, timestep), as training would see across steps
        ii = rng.integers(0, H - h + 1, size=(images.shape[0], self.num_timesteps))
        jj = rng.integers(0, W - w + 1, size=(images.shape[0], self.num_timesteps))
        return torch.stack(
            [
                torch.stack([images[b, :, i : i + h, j : j + w] for i, j in zip(ii[b], jj[b], strict=True)])
                for b in range(images.shape[0])
            ]
        )

    def per_sample_grads(
        self, images, actions, crop: str, gen: torch.Generator, rng: np.random.Generator, t=None, eps=None
    ) -> dict[str, torch.Tensor]:
        """images (B,C,H,W), actions (B,H,A), both normalized -> {name: (B, *shape)}."""
        B, T = actions.shape[0], self.num_timesteps
        dev = actions.device
        if t is None:
            t = torch.randint(0, self.n_train_t, (B, T), generator=gen, device=dev)
        if eps is None:
            eps = torch.randn((B, T, *actions.shape[1:]), generator=gen, device=dev)
        imgs = self._crop(images, crop, rng)
        return vmap(grad(self._loss), in_dims=(None, 0, 0, 0, 0), randomness="error")(
            self.params, imgs, actions, t, eps
        )


# ----------------------------------------------------------------------------------------------
# T0 fail-fast
# ----------------------------------------------------------------------------------------------


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def _flat(d: dict[str, torch.Tensor], names) -> torch.Tensor:
    return torch.cat([d[n].reshape(d[n].shape[0], -1) for n in names], 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bench-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = p.parse_args()
    cfg = load_config(args.config, out_dir_copy=False)
    g = cfg["grad"]
    dev = "cuda"

    policy, pre, _ = load_policy(cfg["ckpt"], dev)
    ds = load_train_dataset(cfg["train_repo_id"], policy.config)
    idx = trained_indices(ds, cfg["drop_n_last_frames"])
    pick = [int(idx[0]), int(idx[-1])]  # one real sample, one imagined pair
    batch = pre(torch.utils.data.default_collate([ds[i] for i in pick]))
    imgs, acts = batch["observation.image"], batch["action"]
    B = len(pick)

    # ---- 1+2: loss and gradient equivalence against lerobot's own compute_loss ----
    # TF32 off: per-sample and batched convs pick different kernels, and TF32 rounding alone gives
    # ~6e-3 relative grad error (cos 0.99998) -- real but far below the sketch's ~0.5%. Exactness of
    # the PORT is tested without it; featurization then runs with grad.allow_tf32.
    set_tf32(False)
    ref = DiffusionInfluence(policy, g["param_prefixes"], "ddpm", num_timesteps=1)
    names = list(ref.params)
    torch.manual_seed(123)
    ref_loss = policy.forward(dict(batch))[0]
    live = dict(policy.diffusion.named_parameters())
    ref_grad = torch.autograd.grad(ref_loss, [live[n] for n in names])
    ref_grad = torch.cat([x.reshape(-1) for x in ref_grad])
    torch.manual_seed(123)  # replay compute_loss's RNG order: randn(eps) then randint(t)
    eps = torch.randn(acts.shape, device=dev)
    t = torch.randint(0, ref.n_train_t, (B,), device=dev).long()
    crop_imgs = ref._crop(imgs, "center", None)
    with torch.no_grad():
        mine = torch.stack(
            [ref._loss(ref.params, crop_imgs[b], acts[b], t[b : b + 1], eps[b : b + 1]) for b in range(B)]
        ).mean()
    psg = ref.per_sample_grads(imgs, acts, "center", None, None, t=t[:, None], eps=eps[:, None])
    mine_grad = _flat(psg, names).mean(0)
    del psg
    loss_rel = abs(mine.item() - ref_loss.item()) / abs(ref_loss.item())
    grad_rel = ((mine_grad - ref_grad).norm() / ref_grad.norm()).item()
    print(f"[1] loss  ref={ref_loss.item():.6f} mine={mine.item():.6f} rel_err={loss_rel:.2e}")
    print(f"[2] grad  |ref|={ref_grad.norm().item():.4e} rel_err={grad_rel:.2e}  (P={ref.dim:,})")
    del ref_grad, mine_grad
    ok = loss_rel < 1e-5 and grad_rel < 1e-4
    set_tf32(g["allow_tf32"])

    # ---- 3: sketch preserves inner products of real per-sample gradients ----
    infl = DiffusionInfluence(policy, g["param_prefixes"], g["loss_fn"], g["num_timesteps"])
    sk = CountSketch(infl.shapes, cfg["sketch"]["proj_dim"], cfg["sketch"]["seed"], dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)
    full = infl.per_sample_grads(imgs, acts, "center", gen, rng)
    x = _flat(full, names)
    phi = sk.project(full)
    ip, ip_s = (x[0] @ x[1]).item(), (phi[0] @ phi[1]).item()
    n0, n0_s = x[0].norm().item(), phi[0].norm().item()
    cos, cos_s = ip / (n0 * x[1].norm().item()), ip_s / (n0_s * phi[1].norm().item())
    print(
        f"[3] sketch |g0| {n0:.4e} -> {n0_s:.4e} ({n0_s / n0 - 1:+.2%}); cos(g0,g1) {cos:+.4f} -> {cos_s:+.4f}"
    )
    del x, phi

    # ---- 4: micro-batch benchmark (square loss, config timesteps) ----
    print(
        f"[4] benchmark: loss={g['loss_fn']} T={g['num_timesteps']} crop={g['train_crop']} tf32={g['allow_tf32']}"
    )
    many = pre(torch.utils.data.default_collate([ds[int(i)] for i in idx[:: len(idx) // 8][:8]]))
    for mb in args.bench_batches:
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _rep in range(2):  # first pass = warmup
                t0 = time.perf_counter()
                sk.project(
                    infl.per_sample_grads(
                        many["observation.image"][:mb], many["action"][:mb], g["train_crop"], gen, rng
                    )
                )
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / mb
            print(
                f"    B={mb}: {dt * 1e3:7.1f} ms/sample   peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GB"
            )
        except torch.OutOfMemoryError:
            print(f"    B={mb}: OOM")
            break
    n_real = len(idx) - 464 if cfg["xtx"]["n_real"] == "all" else cfg["xtx"]["n_real"]
    print(f"    -> featurize ~{n_real + 464 + 4000:,} samples (real + pool + ~4000 test) at the best B above")
    print("T0 PASS" if ok else "T0 FAIL: functional loss does not match compute_loss")


if __name__ == "__main__":
    main()
