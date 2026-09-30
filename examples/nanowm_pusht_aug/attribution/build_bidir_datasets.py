"""Stage 4 inputs for the bidirectional CUPID test: real data PLUS k pairs from ONE pool
(`pool_dataset`), chosen five different ways -- LOO scores, AOI scores, and three random draws --
so every arm differs only in WHICH pairs were added, never in the estimator that built the dataset.

  local/pusht_plus_verified464_loo_cupid_k{k}     -- top-k by the LOO direction's scores
  local/pusht_plus_verified464_aoi_cupid_k{k}      -- top-k by the AOI direction's scores
  local/pusht_plus_verified464_random_s{seed}_k{k} -- one arm per select.random_seeds entry

Each direction config (bidir.directions.{loo,aoi}) already ran score_pairs.py against the SAME
pool, so this script only needs their pair_scores.csv -- keyed by `pair_episode` (the pool's own
episode index), NOT `dataset_episode` (LOO's has a +real_episodes offset because its pool sits
inside the trained-on dataset; AOI's does not).

    uv run --extra training python examples/nanowm_pusht_aug/attribution/build_bidir_datasets.py \
        --config configs/cupid_bidir_verified464.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from lerobot.utils.constants import HF_LEROBOT_HOME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_selected_datasets import build_real_plus_pairs  # noqa: E402
from config_io import load_config  # noqa: E402


def arm_names(cfg: dict) -> list[str]:
    """Every arm tag -- shared with run_cupid_bidir.sh via --list."""
    return ["loo_cupid", "aoi_cupid"] + [f"random_s{s}" for s in cfg["select"]["random_seeds"]]


def arm_repo_ids(cfg: dict) -> dict[str, str]:
    """arm tag -> repo_id, e.g. local/pusht_plus_verified464_loo_cupid_k116."""
    prefix = cfg["select"].get("name_prefix", "local/pusht_plus_verified464")
    k = cfg["select"]["k"]
    return {a: f"{prefix}_{a}_k{k}" for a in arm_names(cfg)}


def load_pair_scores(csv_path: Path) -> dict[int, float]:
    """pair_episode -> score, from one direction's pair_scores.csv."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {int(r["pair_episode"]): float(r["score"]) for r in rows}


def rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def mean_pairwise_cosine(x: np.ndarray) -> float:
    """Mean cosine similarity over every unordered pair of rows of x (n, d)."""
    norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = norm @ norm.T
    iu = np.triu_indices(len(x), k=1)
    return float(sim[iu].mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--list", action="store_true", help="print the arm repo ids and exit")
    args = p.parse_args()
    cfg = load_config(args.config, out_dir_copy=False)
    repo_of = arm_repo_ids(cfg)
    if args.list:
        print("\n".join(repo_of[a] for a in arm_names(cfg)))
        return

    loo_path, aoi_path = cfg["bidir"]["directions"]["loo"], cfg["bidir"]["directions"]["aoi"]
    loo_cfg = load_config(loo_path, out_dir_copy=False)
    aoi_cfg = load_config(aoi_path, out_dir_copy=False)
    loo_scores = load_pair_scores(Path(loo_cfg["out_dir"]) / "pair_scores.csv")
    aoi_scores = load_pair_scores(Path(aoi_cfg["out_dir"]) / "pair_scores.csv")

    sidecar = json.loads(Path(cfg["pool_index"]).read_text())["pairs"]
    n_expected = len(sidecar)
    assert set(loo_scores) == set(aoi_scores), "loo and aoi direction configs scored different pair sets"
    assert len(loo_scores) == n_expected, (
        f"{len(loo_scores)} scored pairs != {n_expected} sidecar rows in {cfg['pool_index']}"
    )
    pair_eps = np.array(sorted(loo_scores))  # == np.arange(n_expected): sidecar row i == pool episode i

    k = cfg["select"]["k"]
    assert k <= len(pair_eps), f"select.k={k} > pool size {len(pair_eps)}"
    loo_arr = np.array([loo_scores[i] for i in pair_eps])
    aoi_arr = np.array([aoi_scores[i] for i in pair_eps])
    chosen = {
        "loo_cupid": pair_eps[np.argsort(-loo_arr)[:k]],
        "aoi_cupid": pair_eps[np.argsort(-aoi_arr)[:k]],
    }
    for seed in cfg["select"]["random_seeds"]:
        chosen[f"random_s{seed}"] = pair_eps[np.random.default_rng(seed).permutation(len(pair_eps))[:k]]

    real_repo = cfg["datasets"]["real"]
    pool_repo = cfg["pool_dataset"]
    real_stats = json.loads((HF_LEROBOT_HOME / real_repo / "meta/stats.json").read_text())

    manifest = {"arms": {}}
    for arm in arm_names(cfg):
        repo_id = repo_of[arm]
        keep = sorted(int(e) for e in chosen[arm])
        n_eps = build_real_plus_pairs(real_repo, pool_repo, keep, repo_id, real_stats)
        assert n_eps == cfg["real_episodes"] + k, f"{repo_id}: {n_eps} episodes, expected real + {k}"
        manifest["arms"][arm] = {"repo_id": repo_id, "k": k, "pair_episodes": keep}
        print(f"{repo_id}: real + {k} {arm} pairs -> {n_eps} episodes, stats pinned to {real_repo}")

    manifest["loo_aoi_topk_overlap"] = len(
        set(chosen["loo_cupid"].tolist()) & set(chosen["aoi_cupid"].tolist())
    )
    manifest["spearman_loo_vs_aoi_score"] = spearman(loo_arr, aoi_arr)
    seeds = cfg["select"]["random_seeds"]
    manifest["random_pairwise_overlap"] = {
        f"random_s{a}_vs_random_s{b}": len(
            set(chosen[f"random_s{a}"].tolist()) & set(chosen[f"random_s{b}"].tolist())
        )
        for i, a in enumerate(seeds)
        for b in seeds[i + 1 :]
    }

    phi_path = Path(aoi_cfg["out_dir"]) / "features" / "phi_pool.npy"
    cosine = {}
    if phi_path.exists():
        phi = np.load(phi_path, mmap_mode="r")
        phi = np.asarray(phi[0] if phi.ndim == 3 else phi)  # sketch 0, or the single 2-D sketch
        for arm in arm_names(cfg):
            cosine[arm] = mean_pairwise_cosine(phi[chosen[arm]])
    else:
        warnings.warn(f"{phi_path} not found -- within_set_mean_pairwise_cosine left null", stacklevel=2)
        cosine = dict.fromkeys(arm_names(cfg))
    manifest["within_set_mean_pairwise_cosine"] = cosine

    out_dir = Path(cfg["bidir"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bidir_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {out_dir / 'bidir_manifest.json'}")


if __name__ == "__main__":
    main()
