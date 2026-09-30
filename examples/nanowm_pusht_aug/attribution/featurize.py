"""TRAK featurization (CUPID stage 3.1): projected per-sample gradients for the three sample sets.

  real  -- real training samples of theta (all, or an `xtx.n_real` subsample); only enter XTX
  pool  -- the imagined pairs, one sample each (sidecar -> first frame of episode real_episodes+i);
           they enter XTX too, because in setting B theta WAS trained on them
  test  -- every policy query of the Stage-2 rollouts (always center crop: what the policy saw)

Real/pool indices are taken from EpisodeAwareSampler(drop_n_last_frames), so a scored sample is
exactly one the checkpoint could have drawn. `--replicate k` re-runs with sketch/grad seeds + k into
features_r{k}/ -- only for the rank-stability diagnostic in score_pairs.py.

    uv run --extra training python examples/nanowm_pusht_aug/attribution/featurize.py \
        --config configs/cupid_filter_untrimmed464.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion_influence_loss import DiffusionInfluence, set_tf32  # noqa: E402
from grad_sketch import CountSketch  # noqa: E402
from policy_io import load_config, load_policy, load_train_dataset, trained_indices  # noqa: E402


def featurize(name, batches, n, infl, sk, crop, gen, rng) -> np.ndarray:
    """-> (n_sketches, n, proj_dim). One gradient pass feeds every sketch."""
    out, done, t0 = np.zeros((sk.n_sketches, n, sk.proj_dim), np.float32), 0, time.time()
    for imgs, acts in batches:
        phi = sk.project(infl.per_sample_grads(imgs, acts, crop, gen, rng))  # (k, B, d)
        b = phi.shape[1]
        out[:, done : done + b] = phi.cpu().numpy()
        done += b
        if done in (b, n) or done % 2000 < b:
            el = time.time() - t0
            print(
                f"  {name}: {done}/{n}  {el / 60:5.1f} min  eta {el / done * (n - done) / 60:5.1f} min",
                flush=True,
            )
    assert done == n
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--replicate", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    g, k = cfg["grad"], args.replicate
    out_dir = Path(cfg["out_dir"]) / ("features" if k == 0 else f"features_r{k}")
    out_dir.mkdir(parents=True, exist_ok=True)
    set_tf32(g["allow_tf32"])
    dev = "cuda"

    policy, pre, _ = load_policy(cfg["ckpt"], dev)
    ds = load_train_dataset(cfg["train_repo_id"], policy.config)
    idx = trained_indices(ds, cfg["drop_n_last_frames"])
    ep_of = np.asarray(ds.hf_dataset["episode_index"])  # absolute index -> episode
    ep_of_idx = ep_of[idx]

    # pool: the one sample of each imagined episode, in sidecar order
    pairs = json.loads(Path(cfg["pool_index"]).read_text())["pairs"]
    if cfg.get("pool_repo_id"):
        # add-one-in: the pool is its own dataset that theta never trained on (a holdout)
        pool_ds = load_train_dataset(cfg["pool_repo_id"], policy.config)
        assert pool_ds.meta.total_episodes == len(pairs), "pool episodes != sidecar rows"
        pstarts = np.asarray(pool_ds.meta.episodes["dataset_from_index"])
        pool_idx = np.array([pstarts[r["episode_index"]] for r in pairs])
    else:
        # leave-one-out: the pool is the imagined tail of the dataset theta was trained on
        pool_ds = ds
        starts = np.asarray(ds.meta.episodes["dataset_from_index"])
        pool_idx = np.array([starts[cfg["real_episodes"] + r["episode_index"]] for r in pairs])
        assert np.isin(pool_idx, idx).all(), "a pool sample is not one the sampler can emit"
        assert (ep_of_idx >= cfg["real_episodes"]).sum() == len(pairs), "imagined sample count != sidecar"

    real_all = idx[ep_of_idx < cfg["real_episodes"]]
    n_real = cfg["xtx"]["n_real"]
    real_idx = (
        real_all
        if n_real == "all"
        else np.sort(np.random.default_rng(cfg["xtx"]["seed"]).choice(real_all, n_real, replace=False))
    )

    infl = DiffusionInfluence(policy, g["param_prefixes"], g["loss_fn"], g["num_timesteps"])
    sk = CountSketch(
        infl.shapes,
        cfg["sketch"]["proj_dim"],
        cfg["sketch"]["seed"] + 100 * k,
        dev,
        n_sketches=cfg["sketch"].get("n_sketches", 1),
    )
    gen = torch.Generator(device=dev).manual_seed(g["seed"] + k)
    rng = np.random.default_rng(g["seed"] + k)
    mb = g["micro_batch"]
    print(
        f"P={infl.dim:,}  real={len(real_idx)}  pool={len(pool_idx)}  T={g['num_timesteps']}  "
        f"B={mb}  k={sk.n_sketches}  d={sk.proj_dim}  -> {out_dir}",
        flush=True,
    )

    def dataset_batches(indices, source=ds):
        loader = DataLoader(Subset(source, indices.tolist()), batch_size=mb, num_workers=4, shuffle=False)
        for b in loader:
            b = pre(b)
            yield b["observation.image"], b["action"]

    test = np.load(Path(cfg["out_dir"]) / "test_rollouts.npz")

    def test_batches():
        for s in range(0, len(test["images"]), mb):
            yield (
                torch.from_numpy(test["images"][s : s + mb]).to(dev),
                torch.from_numpy(test["actions"][s : s + mb]).to(dev),
            )

    crop = g["train_crop"]

    def run(name: str, batches, n: int, crop_mode: str) -> None:
        """Resume: a set whose file already has the expected shape is kept, not recomputed."""
        f = out_dir / f"phi_{name}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[:2] == (sk.n_sketches, n):
            print(f"  {name}: {f.name} exists with the expected shape -- skipped", flush=True)
            return
        np.save(f, featurize(name, batches, n, infl, sk, crop_mode, gen, rng))

    run("pool", dataset_batches(pool_idx, pool_ds), len(pool_idx), crop)
    run("test", test_batches(), len(test["images"]), "center")
    run("real", dataset_batches(real_idx), len(real_idx), crop)
    np.save(out_dir / "real_idx.npy", real_idx)
    np.save(out_dir / "pool_idx.npy", pool_idx)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
