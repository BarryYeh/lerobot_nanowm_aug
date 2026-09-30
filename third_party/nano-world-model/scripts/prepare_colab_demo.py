#!/usr/bin/env python3
"""Fetch the official Point Maze demo without expanding the entire 30 GB dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from safetensors.torch import load_file

MODEL_ID = "knightnemo/nanowm-b2-dino-wm-point-maze-30k"
MODEL_REVISION = "b84303b7151cbd1836841c3a87bf095e1015b150"
VAE_ID = "stabilityai/sd-vae-ft-mse"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
DATA_URL = "https://osf.io/download/vr5gy/?view_only=a56a296ce3b24cceaf408383a175ce28"
DATA_SHA256 = "6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validation_episodes(lengths, count, rollout_length=20, frame_interval=5):
    """Match the checkpoint's full-data 90/10 split, in its original order."""
    order = np.random.RandomState(42).permutation(len(lengths))
    ids = [int(i) for i in order[int(len(order) * 0.9):]
           if int(lengths[i]) >= rollout_length * frame_interval][:count]
    if len(ids) != count:
        raise ValueError(f"Only {len(ids)} validation episodes support this rollout")
    return ids


def extract_member(archive, name, target):
    """Only explicitly selected members; never extract arbitrary archive paths."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == archive.getinfo(name).file_size:
        return
    temporary = target.with_name(target.name + ".partial")
    with archive.open(name) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output)
    temporary.replace(target)


def prepare(root, num_samples):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "point_maze.zip"
    if not archive.exists() or sha256(archive) != DATA_SHA256:
        temporary = root / "point_maze.zip.partial"
        print("Downloading official Point Maze archive (718 MB)...", flush=True)
        with requests.get(DATA_URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    output.write(chunk)
        if sha256(temporary) != DATA_SHA256:
            raise RuntimeError("Point Maze archive checksum mismatch; rerun preparation")
        temporary.replace(archive)

    data = root / "data" / "point_maze"
    with zipfile.ZipFile(archive) as z:
        # Keep ALL metadata, so normalization and train/val membership stay identical.
        for name in ("states.pth", "actions.pth", "seq_lengths.pth"):
            extract_member(z, "point_maze/" + name, data / name)
        lengths = torch.load(data / "seq_lengths.pth", weights_only=True)
        episodes = validation_episodes(lengths, num_samples)
        for episode in episodes:
            name = f"obses/episode_{episode:03d}.pth"
            extract_member(z, "point_maze/" + name, data / name)
    print(f"Extracted observations only for validation episodes {episodes}.", flush=True)

    ckpt = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                allow_patterns=["config.yaml", "model.safetensors"],
                                local_dir=root / "checkpoint"))
    vae = snapshot_download(VAE_ID, revision=VAE_REVISION,
                           allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
                           local_dir=root / "vae")
    # The current production loader consumes a state dict via torch.load.
    # Convert the trusted, revision-pinned safetensors once, atomically.
    model_digest = sha256(ckpt / "model.safetensors")
    pt = ckpt / "model.pt"
    stamp = ckpt / "model.pt.source-sha256"
    if not pt.exists() or not stamp.exists() or stamp.read_text().strip() != model_digest:
        temporary = ckpt / "model.pt.partial"
        torch.save(load_file(str(ckpt / "model.safetensors")), temporary)
        temporary.replace(pt)
        stamp.write_text(model_digest + "\n")

    conf = OmegaConf.load(ckpt / "config.yaml")
    conf.dataset_dir = str(data.parent)
    conf.vae_model_path = str(Path(vae).resolve())
    # Training images are unused for inference; stats still use all train metadata.
    conf.dataset.loader.train_slice_mode = "random"
    config_path = root / "demo-config.yaml"
    OmegaConf.save(conf, config_path)
    provenance = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                  "model_sha256": model_digest, "vae_id": VAE_ID,
                  "vae_revision": VAE_REVISION, "dataset_url": DATA_URL,
                  "dataset_sha256": DATA_SHA256, "episodes": episodes,
                  "config": str(config_path), "checkpoint": str(pt),
                  "uv_lock_sha256": sha256(Path(__file__).resolve().parents[1] / "uv.lock")}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("num-samples must be positive")
    prepare(args.root, args.num_samples)
