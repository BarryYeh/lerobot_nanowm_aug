"""Stage 4 inputs for add-one-in: real data PLUS k pool pairs, chosen by CUPID score or at random.

  <real>+<pool>_cupid_k{k}   -- real + the k HIGHEST-scoring pool pairs
  <real>+<pool>_random_k{k}  -- real + k pool pairs drawn at random (fixed seed)
Both are real-first aggregates (as run_dose464.sh builds its arms) with the image/action stats
re-pinned to the real-only set, so normalization is identical to every other arm and the only
difference between the two is WHICH pool pairs were added.

    uv run --extra training python examples/nanowm_pusht_aug/attribution/build_selected_datasets.py \
        --config configs/cupid_select_topband.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.constants import HF_LEROBOT_HOME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_io import load_config  # noqa: E402

PINNED = ("observation.image", "action")


def arm_repo_ids(cfg: dict) -> list[tuple[str, str]]:
    """(method, repo_id) for every arm -- shared with run_cupid_select.sh via --list."""
    k = cfg["select"]["k"]
    stem = cfg["pool_repo_id"].removeprefix("local/pusht_cl_") + cfg["select"].get("name_suffix", "")
    return [
        (m, f"local/pusht_plus_{stem}_{m}_k{k}") for m in cfg["select"].get("methods", ["cupid", "random"])
    ]


def build_real_plus_pairs(
    real_repo: str, pool_repo: str, keep_pair_eps: list[int], repo_id: str, real_stats: dict
) -> int:
    """Build <repo_id> = real_repo (all episodes) + pool_repo's episodes in keep_pair_eps, real FIRST
    (as run_dose464.sh builds its arms), with observation.image/action stats re-pinned to
    real_stats so normalization is identical across every arm built this way.

    Returns the aggregated dataset's total_episodes.
    """
    keep = set(keep_pair_eps)
    sub_repo = f"{pool_repo}__sub_{repo_id.removeprefix('local/')}"
    for r in (sub_repo, repo_id):
        if (HF_LEROBOT_HOME / r).exists():
            shutil.rmtree(HF_LEROBOT_HOME / r)
    pool = LeRobotDataset(pool_repo, root=HF_LEROBOT_HOME / pool_repo)
    drop = sorted(set(range(pool.meta.total_episodes)) - keep)
    if drop:
        delete_episodes(pool, episode_indices=drop, output_dir=HF_LEROBOT_HOME / sub_repo, repo_id=sub_repo)
    else:  # keep every pair: delete_episodes refuses an empty drop list, aggregate the pool itself
        sub_repo = pool_repo
    aggregate_datasets(
        repo_ids=[real_repo, sub_repo],
        aggr_repo_id=repo_id,
        roots=[HF_LEROBOT_HOME / real_repo, HF_LEROBOT_HOME / sub_repo],
        aggr_root=HF_LEROBOT_HOME / repo_id,
    )
    sp = HF_LEROBOT_HOME / repo_id / "meta/stats.json"
    stats = json.loads(sp.read_text())
    for key in PINNED:
        stats[key] = real_stats[key]
    sp.write_text(json.dumps(stats))
    return json.loads((HF_LEROBOT_HOME / repo_id / "meta/info.json").read_text())["total_episodes"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--list", action="store_true", help="print the arm repo ids and exit")
    args = p.parse_args()
    cfg = load_config(args.config, out_dir_copy=False)
    if args.list:
        print("\n".join(r for _, r in arm_repo_ids(cfg)))
        return

    out, k = Path(cfg["out_dir"]), cfg["select"]["k"]
    with open(out / "pair_scores.csv") as f:
        rows = list(csv.DictReader(f))
    ep = np.array([int(r["dataset_episode"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    assert k <= len(ep), f"select.k={k} > pool size {len(ep)}"
    chosen = {
        "cupid": ep[np.argsort(-score)[:k]],
        "random": ep[np.random.default_rng(cfg["select"]["random_seed"]).permutation(len(ep))[:k]],
    }
    real_repo, pool_repo = cfg["train_repo_id"], cfg["pool_repo_id"]
    real_stats = json.loads((HF_LEROBOT_HOME / real_repo / "meta/stats.json").read_text())

    manifest = {}
    for method, repo_id in arm_repo_ids(cfg):
        keep = set(chosen[method].tolist())
        n_eps = build_real_plus_pairs(real_repo, pool_repo, sorted(keep), repo_id, real_stats)
        assert n_eps == cfg["real_episodes"] + k, f"{repo_id}: {n_eps} episodes, expected real + {k}"
        manifest[repo_id] = {"method": method, "k": k, "added_pool_episodes": sorted(keep)}
        print(f"{repo_id}: real + {k} {method} pairs -> {n_eps} episodes, stats pinned to {real_repo}")
    if "cupid" in chosen and "random" in chosen:
        overlap = len(set(chosen["cupid"].tolist()) & set(chosen["random"].tolist()))
        print(f"cupid/random overlap: {overlap}/{k} (random expectation {k * k / len(ep):.0f})")
    (out / "selected_datasets.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
