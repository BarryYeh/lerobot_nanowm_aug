#!/usr/bin/env python
"""NanoWM imagined-rollout generation -- both modes live here.

Runs in the `nanowm` conda env (diffusers 0.24.0 / torch 2.1.2, incompatible with the lerobot
venv), so this file must never import lerobot. NanoWM + SD-VAE loading is shared by both modes
(`load_nanowm`), which is why they sit in one file.

CLOSED LOOP (`closed-loop`, the current direction) -- a persistent worker driven by
`eval/closed_loop_rollout.py` in the lerobot venv. The policy re-plans on each imagined frame, so
the driver owns the policy and this side owns the world model; they exchange numbered .npz files
in a work dir:

    driver writes  work/seeds_expanded.npz   (frames_224: (bs,H,224,224,3) uint8)   -- once
    driver writes  work/chunks_<r>.npz       (win: (bs,nf,fi*2) f32, live: (bs,) bool)
    worker writes  work/frames_<r+1>.npz     (frames: (bs,224,224,3) u8, cov: (bs,) f32)
    driver writes  work/stop                 -> worker exits

One round generates exactly ONE frame per branch (`n_generate_frames=1`) and the H-frame context
window slides, so from round H onwards the context is entirely self-generated.

OPEN LOOP (`open-loop`) -- no policy in the loop: branch off a real pusht_noise trajectory and
roll NanoWM forward under the demonstrator's own actions plus Gaussian noise. Kept for world-model
fidelity checks and as the fps-2 data path (stage 2 = `datagen/run_converter.py imagined`).

  1. Pick a pusht_noise train trajectory + start frame.
  2. Take `history_length` real context frames.
  3. Perturb the real rel-action sequence with Gaussian noise (the diversity source).
  4. Autoregressive NanoWM rollout with the perturbed actions -> future SD-VAE latents.
  5. VAE-decode -> RGB, resize 256 -> 224.
  6. PD-replay the kinematic, block-independent agent from the real start state driven by the same
     perturbed actions -> exact agent_pos at every 10 Hz step; the absolute-target action label is
     `agent_pos[t0+4] + rel_px[t0+4]` (`--action-at end`, matching run_converter.py pusht-noise).
  7. Keep only generated frames -> one imagined episode (fps 2, 224x224, image-only).

Output: <out>/shard_*.npz with images (N,224,224,3) u8, actions (N,2) f32, ep_lens (E,) i32,
plus <out>/meta.json.

Run:
    NW=~/miniconda3/envs/nanowm/bin/python
    P=examples/nanowm_pusht_aug

    # closed loop: normally spawned by eval/closed_loop_rollout.py, not by hand
    $NW $P/imagined_rollout.py closed-loop --work $P/cl/work --rounds 8 --num-sampling-steps 30

    export DATASET_DIR=~/Downloads/bmw48-osfstorage-datasets-archive
    $NW $P/imagined_rollout.py open-loop --num-samples 300 --out $P/imagined/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Wire every search path BEFORE the imports below: isort sorts the import block
# alphabetically, so `from datagen...` must not be relied on to run after `from diffusion...`.
_HERE = Path(__file__).resolve().parent
NANOWM_ROOT = Path("~/repos/nano-world-model").expanduser()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(NANOWM_ROOT / "src"))
sys.path.insert(0, str(NANOWM_ROOT / "src" / "sample"))
os.environ.setdefault("DATASET_DIR", str(Path("~/Downloads/bmw48-osfstorage-datasets-archive").expanduser()))

from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402
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


def _wait(path: Path, timeout: float = 1200.0) -> bool:
    t0 = time.time()
    while not path.exists():
        if (path.parent / "stop").exists():
            return False
        if time.time() - t0 > timeout:
            raise TimeoutError(f"worker timed out waiting for {path}")
        time.sleep(0.3)
    return True


def _save_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def closed_loop(args) -> None:
    """Persistent worker: advance every branch one imagined frame per round, driven by the
    lerobot-venv driver through .npz files in `args.work`."""
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    dev = args.device

    cfg, model, vae, diffusion = load_nanowm(dev)
    Cz, Sz = get_model_latent_channels(cfg), get_model_latent_size(cfg)
    nf = cfg.model.num_frames
    img_size = int(cfg.model.image_size)
    vae_prec = cfg.experiment.infra.vae_precision
    H = args.history
    print(f"[wm] loaded. Cz={Cz} Sz={Sz} nf={nf} img={img_size} H={H} rounds={args.rounds}", flush=True)

    if not _wait(work / "seeds_expanded.npz"):
        print("[wm] stop before seeds", flush=True)
        return
    seed_frames = np.load(work / "seeds_expanded.npz")["frames_224"]  # (bs,224,224,3) or (bs,H,224,224,3) u8
    multi = seed_frames.ndim == 5
    bs = len(seed_frames)
    if multi:  # (bs,Hs,224,224,3) real context frames
        Hs = seed_frames.shape[1]
        f = torch.from_numpy(seed_frames).float().permute(0, 1, 4, 2, 3).div(255.0)  # (bs,Hs,3,224,224)
        f = resize_frames(f.reshape(bs * Hs, 3, 224, 224), (img_size, img_size), "stretch")
        f = (f * 2.0 - 1.0).reshape(bs, Hs, 3, img_size, img_size)
        lat_parts = []
        for i in range(0, bs, args.wm_batch):
            lat_parts.append(
                encode_frames(vae, f[i : i + args.wm_batch].to(dev), dev, vae_precision=vae_prec).cpu()
            )
        gen_lat = torch.cat(lat_parts, dim=0).to(dev)  # (bs,Hs,Cz,Sz,Sz)
        if Hs < H:
            raise SystemExit(
                f"[wm] got {Hs} context frames, need {H}. Padding by repeating frame 0 would "
                "fabricate a STATIC past -- the original alignment bug, which made the real "
                "actions land ~1.0 s early (open-loop tracking corr 0.958 -> 0.347). The driver "
                f"must send {H} frames spaced fi sim-steps apart. See diag/ROOT_CAUSE_BRIEF.md."
            )
    else:
        raise SystemExit(
            "[wm] seeds_expanded.npz holds a single frame per branch (bs,224,224,3). Replicating "
            f"it to {H} context slots fabricates a static past -- the original alignment bug. "
            "Send (bs,H,224,224,3) real frames fi sim-steps apart instead."
        )
    print(f"[wm] seeded bs={bs}  multi_context={multi}", flush=True)

    for r in range(args.rounds):
        if not _wait(work / f"chunks_{r}.npz"):
            print(f"[wm] stop at round {r}", flush=True)
            return
        d = np.load(work / f"chunks_{r}.npz")
        win = torch.from_numpy(d["win"].astype(np.float32)).to(dev)  # (bs,nf,fi*2)

        # sub-batch the NanoWM forward over branches (GPU memory)
        parts = []
        for i in range(0, bs, args.wm_batch):
            j = min(i + args.wm_batch, bs)
            pred = dfot_sample(
                diffusion=diffusion,
                model=model.forward,
                shape=(j - i, nf, Cz, Sz, Sz),
                context=gen_lat[i:j, -H:],
                n_context_frames=H,
                scheduling_mode=cfg.model.scheduling_mode,
                num_sampling_steps=args.num_sampling_steps,
                model_kwargs={"y": None, "use_fp16": False, "action": win[i:j]},
                device=dev,
                progress=False,
                eta=0.0,
                clip_denoised=False,
                n_generate_frames=1,
                history_stabilization_level=cfg.experiment.diffusion.history_stabilization_level,
            )
            parts.append(pred[:, H : H + 1])
        new_lat = torch.cat(parts, dim=0)  # (bs,1,Cz,Sz,Sz)
        gen_lat = torch.cat([gen_lat, new_lat], dim=1)

        rgb_parts = []
        for c0 in range(0, bs, args.decode_chunk):
            ch = decode_latents(vae, new_lat[c0 : c0 + args.decode_chunk], vae_precision=vae_prec)[:, 0]
            ch = torch.nn.functional.interpolate(ch, size=IMAGE_HW, mode="bilinear", align_corners=False)
            rgb_parts.append((ch.clamp(0, 1) * 255).to(torch.uint8).cpu())
        rgb = torch.cat(rgb_parts).permute(0, 2, 3, 1).numpy()  # (bs,224,224,3) u8

        _cf = [coverage_from_frame(rgb[b]) for b in range(bs)]
        cov = np.array([c["coverage_cal"] for c in _cf], np.float32)
        cov_raw = np.array([c["coverage"] for c in _cf], np.float32)
        _save_atomic(work / f"frames_{r + 1}.npz", frames=rgb, cov=cov, cov_raw=cov_raw)
        print(f"[wm] round {r} -> frames_{r + 1}  cov mean={cov.mean():.3f} max={cov.max():.3f}", flush=True)

    print("[wm] done", flush=True)


def open_loop(args) -> None:
    """Branch off real pusht_noise trajectories under perturbed demonstrator actions."""

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = p.add_subparsers(dest="mode", required=True)

    cl = modes.add_parser("closed-loop", help="persistent worker for eval/closed_loop_rollout.py")
    cl.add_argument("--work", type=Path, required=True)
    cl.add_argument("--rounds", type=int, default=8, help="H_max NanoWM steps")
    cl.add_argument("--history", type=int, default=3, help="n_context_frames fed to NanoWM")
    cl.add_argument("--num-sampling-steps", type=int, default=30)
    cl.add_argument("--wm-batch", type=int, default=24, help="branches per NanoWM forward (GPU memory)")
    cl.add_argument("--decode-chunk", type=int, default=16)
    cl.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cl.set_defaults(fn=closed_loop)

    ol = modes.add_parser("open-loop", help="branch off real trajectories with perturbed actions")
    ol.add_argument("--num-samples", type=int, default=2000)
    ol.add_argument(
        "--rollout-length",
        type=int,
        default=20,
        help="total NanoWM frames (context+gen). *fi must fit pusht_noise ep length ~109",
    )
    ol.add_argument("--history-length", type=int, default=3)
    ol.add_argument("--action-noise-std", type=float, default=10.0, help="Gaussian sigma on rel actions, px")
    ol.add_argument("--num-sampling-steps", type=int, default=40)
    ol.add_argument("--batch-size", type=int, default=2)
    ol.add_argument("--decode-chunk", type=int, default=8, help="frames per VAE decode call (memory)")
    ol.add_argument("--seed", type=int, default=0)
    ol.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/imagined/run"))
    ol.add_argument("--shard-every", type=int, default=100, help="flush a shard_*.npz every N episodes")
    ol.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ol.set_defaults(fn=open_loop)

    args = p.parse_args()
    torch.set_grad_enabled(False)
    args.fn(args)


if __name__ == "__main__":
    main()
