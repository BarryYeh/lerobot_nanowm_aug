#!/usr/bin/env python
"""Persistent NanoWM worker for the closed-loop rollout (Task 3, nanowm conda env).

Loads NanoWM + SD-VAE ONCE, then advances a batch of branches one 0.5 s frame per round,
synchronised with the lerobot-venv driver (proto_closed_loop_rollout.py) through numbered
.npz files in a work dir:

    driver writes  work/seeds_expanded.npz   (frames_224: (bs,224,224,3) uint8)   -- once
    driver writes  work/chunks_<r>.npz       (win: (bs,4,10) f32, live: (bs,) bool)
    worker writes  work/frames_<r+1>.npz     (frames: (bs,224,224,3) u8, cov: (bs,) f32)
    driver writes  work/stop                 -> worker exits

Run:
    ~/miniconda3/envs/nanowm/bin/python examples/nanowm_pusht_aug/proto_cl_wm_worker.py \
        --work examples/nanowm_pusht_aug/cl/work --rounds 8 --num-sampling-steps 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Wire both search paths BEFORE the imports below, and do it here rather than relying on
# `rollout_imagine` doing it as a side effect: isort orders the import block alphabetically,
# so `from rollout_imagine import ...` cannot be trusted to run before `from diffusion ...`.
_HERE = Path(__file__).parent
NANOWM_ROOT = Path("~/repos/nano-world-model").expanduser()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(NANOWM_ROOT / "src"))
sys.path.insert(0, str(NANOWM_ROOT / "src" / "sample"))
os.environ.setdefault("DATASET_DIR", str(Path("~/Downloads/bmw48-osfstorage-datasets-archive").expanduser()))

from datagen.coverage_from_frame import coverage_from_frame  # noqa: E402
from diffusion.df_sample import dfot_sample  # noqa: E402
from latent_codecs import get_model_latent_channels, get_model_latent_size  # noqa: E402
from rollout_imagine import load_nanowm  # noqa: E402
from sampling_utils import decode_latents, encode_frames, resize_frames  # noqa: E402

IMAGE_HW = (224, 224)


def _wait(path: Path, timeout: float = 1200.0) -> bool:
    t0 = time.time()
    while not path.exists():
        if (_HERE / path.parent.name / "stop").exists() or (path.parent / "stop").exists():
            return False
        if time.time() - t0 > timeout:
            raise TimeoutError(f"worker timed out waiting for {path}")
        time.sleep(0.3)
    return True


def _save_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=8, help="H_max NanoWM steps")
    p.add_argument("--history", type=int, default=3, help="n_context_frames fed to NanoWM")
    p.add_argument("--num-sampling-steps", type=int, default=30)
    p.add_argument("--wm-batch", type=int, default=24, help="branches per NanoWM forward (GPU memory)")
    p.add_argument("--decode-chunk", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.set_grad_enabled(False)

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
            gen_lat = torch.cat([gen_lat[:, :1].repeat(1, H - Hs, 1, 1, 1), gen_lat], dim=1)
    else:
        f = torch.from_numpy(seed_frames).float().permute(0, 3, 1, 2).div(255.0)
        f = resize_frames(f, (img_size, img_size), "stretch") * 2.0 - 1.0
        lat_parts = []
        for i in range(0, bs, args.wm_batch):
            lat_parts.append(
                encode_frames(vae, f[i : i + args.wm_batch, None].to(dev), dev, vae_precision=vae_prec).cpu()
            )
        gen_lat = torch.cat(lat_parts, dim=0).to(dev).repeat(1, H, 1, 1, 1)  # (bs,H,Cz,Sz,Sz)
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


if __name__ == "__main__":
    main()
