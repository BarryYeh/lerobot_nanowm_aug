#!/usr/bin/env python
"""Generate NanoWM imagined-rollout episodes -> raw .npz (stage 1 of 2).

Runs in the `nanowm` conda env (diffusers 0.24.0 / torch 2.1.2 — incompatible with
the lerobot venv). Stage 2 = datagen/run_converter.py imagined, run in the lerobot venv,
turns the .npz into a LeRobotDataset.

Prototype question: does NanoWM (SD-VAE, nanowm-b2-dino-wm-pusht-100k) imagination add
diversity that raises a data-poor PushT diffusion policy's coverage?

Method (branch source = the 185 pusht_noise demo corridors, in-distribution for NanoWM):
  1. Pick a pusht_noise train trajectory + start frame.
  2. Take `history_length` real context frames.
  3. Perturb the real rel-action sequence with Gaussian noise (the diversity source).
  4. Autoregressive NanoWM rollout with the perturbed actions -> future SD-VAE latents.
  5. VAE-decode -> RGB, resize 256 -> 224.
  6. PD-replay the kinematic, block-independent agent from the real start state driven
     by the same perturbed actions -> exact agent_pos at every 10 Hz step; the absolute-
     target action label is `agent_pos[t0+4] + rel_px[t0+4]` (`--action-at end`, matching
     datagen/run_converter.py pusht-noise).
  7. Keep only generated frames -> one imagined episode (fps 2, 224x224, image-only).

Output: <out>/imagined_rollouts.npz with
    images   (N, 224, 224, 3) uint8
    actions  (N, 2) float32           absolute target position, [0,512]-ish
    ep_lens  (E,) int32               frames per episode (sum == N)
    meta     dict (json string): config used

Run:
    export DATASET_DIR=~/Downloads/bmw48-osfstorage-datasets-archive
    ~/miniconda3/envs/nanowm/bin/python examples/nanowm_pusht_aug/rollout_imagine.py \
        --num-samples 300 --out examples/nanowm_pusht_aug/imagined/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

NANOWM_ROOT = Path("~/repos/nano-world-model").expanduser()
sys.path.insert(0, str(NANOWM_ROOT / "src"))
sys.path.insert(0, str(NANOWM_ROOT / "src" / "sample"))
os.environ.setdefault("DATASET_DIR", str(Path("~/Downloads/bmw48-osfstorage-datasets-archive").expanduser()))

from diffusion import create_diffusion  # noqa: E402
from diffusion.df_sample import dfot_sample  # noqa: E402
from latent_codecs import (  # noqa: E402
    get_model_latent_channels,
    get_model_latent_size,
    load_autoencoder_kl,
    resolve_latent_codec_config,
)
from models import get_models  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sampling_utils import decode_latents, encode_frames, resize_frames  # noqa: E402
from utils.nanowm_utils import find_model  # noqa: E402
from wm_datasets import create_train_val_datasets  # noqa: E402

IMAGE_HW = (224, 224)
KP, KV, DT, SUBSTEPS = 100.0, 20.0, 0.01, 10  # PushT agent PD: sim_hz=100, control_hz=10
ACTION_SCALE = 100.0


def pd_replay_agent(pos0: np.ndarray, vel0: np.ndarray, rel_px: np.ndarray) -> np.ndarray:
    """Kinematic agent replay. rel_px (T,2) = pixel displacement command per 10 Hz step.
    Returns agent_pos after each 10 Hz step, (T,2)."""
    pos, vel = pos0.astype(np.float64).copy(), vel0.astype(np.float64).copy()
    out = np.empty_like(rel_px, dtype=np.float64)
    for t in range(len(rel_px)):
        target = pos + rel_px[t]
        for _ in range(SUBSTEPS):
            acc = KP * (target - pos) + KV * (-vel)
            vel = vel + acc * DT
            pos = pos + vel * DT
        out[t] = pos
    return out


def load_nanowm(device: str):
    cfg = OmegaConf.load(NANOWM_ROOT / "ckpt" / "config.yaml")
    cfg.model.latent_size = get_model_latent_size(cfg)
    model = get_models(cfg).to(device)
    sd = find_model(str(NANOWM_ROOT / "ckpt" / "model.pt"))
    sd = sd.get("model", sd)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    codec_cfg = resolve_latent_codec_config(cfg)
    assert codec_cfg.has_decoder, "need SD-VAE (decoder) checkpoint"
    vae = load_autoencoder_kl(codec_cfg.model_path).to(device).eval()

    diffusion = create_diffusion(
        timestep_respacing=str(cfg.model.num_sampling_steps),
        noise_schedule=cfg.experiment.diffusion.noise_schedule,
        pred_name=cfg.experiment.diffusion.pred_name,
        diffusion_steps=cfg.experiment.diffusion.diffusion_steps,
        snr_gamma=cfg.experiment.diffusion.snr_gamma,
        zero_terminal_snr=cfg.experiment.diffusion.zero_terminal_snr,
    )
    return cfg, model, vae, diffusion


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument(
        "--rollout-length",
        type=int,
        default=20,
        help="total NanoWM frames (context+gen). *fi must fit pusht_noise ep length ~109",
    )
    p.add_argument("--history-length", type=int, default=3)
    p.add_argument("--action-noise-std", type=float, default=10.0, help="Gaussian sigma on rel actions, px")
    p.add_argument("--num-sampling-steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--decode-chunk", type=int, default=8, help="frames per VAE decode call (memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/imagined/run"))
    p.add_argument("--shard-every", type=int, default=100, help="flush a shard_*.npz every N episodes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
    torch.set_grad_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg, model, vae, diffusion = load_nanowm(dev)
    Cz, Sz = get_model_latent_channels(cfg), get_model_latent_size(cfg)
    nf = cfg.model.num_frames
    fi = int(cfg.dataset.frame_interval)  # 5
    img_size = int(cfg.model.image_size)  # 256
    vae_prec = cfg.experiment.infra.vae_precision
    H, G = args.history_length, args.rollout_length
    n_gen = G - H
    assert n_gen > 0

    train_ds, val_ds = create_train_val_datasets(
        dataset_name=cfg.dataset.name,
        num_frames=nf,
        frame_interval=fi,
        image_size=img_size,
        **OmegaConf.to_container(cfg.dataset.loader, resolve=True),
    )
    # branch from the 185 training corridors (in-distribution for both policy and NanoWM)
    branch_ds = (
        train_ds
        if train_ds.data_source.get_num_trajectories() >= val_ds.data_source.get_num_trajectories()
        else val_ds
    )
    src = branch_ds.data_source
    a_mean = branch_ds._raw_action_mean.to(dev)
    a_std = branch_ds._raw_action_std.to(dev)
    n_traj = src.get_num_trajectories()
    need = G * fi
    print(f"branch source: {n_traj} trajectories, need {need} sim-steps/branch")

    run_meta = {
        "num_samples": args.num_samples,
        "rollout_length": G,
        "history_length": H,
        "n_gen": n_gen,
        "action_noise_std_px": args.action_noise_std,
        "num_sampling_steps": args.num_sampling_steps,
        "seed": args.seed,
        "frame_interval": fi,
        "fps": 2,
        "image_hw": list(IMAGE_HW),
        "action_at": "end",
    }
    (args.out / "meta.json").write_text(json.dumps(run_meta, indent=2))

    all_imgs, all_acts, ep_lens = [], [], []
    shard_idx, done = 0, 0

    def flush_shard():
        nonlocal shard_idx, all_imgs, all_acts, ep_lens
        if not ep_lens:
            return
        path = args.out / f"shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            path,
            images=np.asarray(all_imgs, dtype=np.uint8),
            actions=np.asarray(all_acts, dtype=np.float32),
            ep_lens=np.asarray(ep_lens, dtype=np.int32),
        )
        print(
            f"  [shard {shard_idx:04d}] {len(ep_lens)} eps, {len(all_imgs)} frames -> {path.name}", flush=True
        )
        shard_idx += 1
        all_imgs, all_acts, ep_lens = [], [], []

    print(
        f"NanoWM imagined rollout: {args.num_samples} eps, {n_gen} gen frames each, "
        f"noise_std={args.action_noise_std}px, steps={args.num_sampling_steps}",
        flush=True,
    )

    for b0 in range(0, args.num_samples, args.batch_size):
        bs = min(args.batch_size, args.num_samples - b0)
        ctx_vis, raw_act_norm, rel_px_pert, pos0, vel0 = [], [], [], [], []
        for _ in range(bs):
            for _try in range(200):
                ti = int(rng.integers(n_traj))
                L = src.get_seq_length(ti)
                if need <= L:
                    break
            sf = int(rng.integers(0, max(1, L - need + 1)))
            frames = src.load_visual_frames(ti, sf, sf + need, step=fi)  # (G,C,H,W) in [0,1]
            if frames.shape[0] < G:  # short traj: pad with last frame
                frames = torch.cat([frames, frames[-1:].repeat(G - frames.shape[0], 1, 1, 1)], dim=0)
            frames = resize_frames(frames, (img_size, img_size), "stretch") * 2.0 - 1.0
            traj = src.load_trajectory(ti)
            rel = traj.actions[sf : sf + need].clone().numpy()  # = rel_actions / ACTION_SCALE
            if rel.shape[0] < need:  # pad actions for short traj
                rel = np.concatenate([rel, np.repeat(rel[-1:], need - rel.shape[0], axis=0)], axis=0)
            rel_p = rel + rng.normal(0.0, args.action_noise_std / ACTION_SCALE, size=rel.shape)
            st = traj.states[sf].numpy()  # [ax, ay, bx, by, ang, vx, vy]
            ctx_vis.append(frames)
            raw_act_norm.append((torch.from_numpy(rel_p).float().to(dev) - a_mean) / a_std)
            rel_px_pert.append(rel_p * ACTION_SCALE)
            pos0.append(st[:2])
            vel0.append(st[5:7] if st.shape[0] >= 7 else np.zeros(2))

        gt_ctx = torch.stack([f[:H] for f in ctx_vis]).to(dev)  # only context frames -> VAE
        acts = torch.stack(raw_act_norm)
        gen_lat = encode_frames(vae, gt_ctx, dev, vae_precision=vae_prec)

        for t in range(H, G):
            ctx = gen_lat[:, -H:]
            s = (t - H) * fi
            win = acts[:, s : s + nf * fi]
            if win.shape[1] < nf * fi:
                win = torch.cat([win, win[:, -1:].repeat(1, nf * fi - win.shape[1], 1)], dim=1)
            win = win.reshape(bs, nf, fi * 2)
            pred = dfot_sample(
                diffusion=diffusion,
                model=model.forward,
                shape=(bs, nf, Cz, Sz, Sz),
                context=ctx,
                n_context_frames=H,
                scheduling_mode=cfg.model.scheduling_mode,
                num_sampling_steps=args.num_sampling_steps,
                model_kwargs={"y": None, "use_fp16": False, "action": win},
                device=dev,
                progress=False,
                eta=0.0,
                clip_denoised=False,
                n_generate_frames=1,
                history_stabilization_level=cfg.experiment.diffusion.history_stabilization_level,
            )
            gen_lat = torch.cat([gen_lat, pred[:, H : H + 1]], dim=1)

        lat_gen = gen_lat[:, H:].reshape(bs * n_gen, Cz, Sz, Sz)
        rgb_chunks = []
        for c0 in range(0, lat_gen.shape[0], args.decode_chunk):
            ch = decode_latents(vae, lat_gen[c0 : c0 + args.decode_chunk][:, None], vae_precision=vae_prec)[
                :, 0
            ]
            ch = torch.nn.functional.interpolate(ch, size=IMAGE_HW, mode="bilinear", align_corners=False)
            rgb_chunks.append((ch.clamp(0, 1) * 255).to(torch.uint8).cpu())
        rgb = torch.cat(rgb_chunks).reshape(bs, n_gen, 3, *IMAGE_HW).permute(0, 1, 3, 4, 2).numpy()

        for i in range(bs):
            ap = pd_replay_agent(pos0[i], vel0[i], rel_px_pert[i])
            for j in range(n_gen):
                idx = min((H + j) * fi + fi - 1, need - 1)
                all_imgs.append(rgb[i, j])
                all_acts.append((ap[idx] + rel_px_pert[i][idx]).astype(np.float32))
            ep_lens.append(n_gen)
            done += 1
        print(f"  {done}/{args.num_samples} episodes", flush=True)
        if len(ep_lens) >= args.shard_every:
            flush_shard()

    flush_shard()
    print(f"\nDone. {done} episodes across {shard_idx} shard(s) -> {args.out}")


if __name__ == "__main__":
    main()
