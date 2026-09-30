"""Stage 4 inputs: drop imagined pairs from the trained-on dataset, by CUPID score and at random.

For each fraction f in filter.remove_fracs, two datasets with the SAME number of pairs removed:
  <train_repo_id>_cupid_r{f}   -- the round(f * n_pool) lowest-scoring pairs removed
  <train_repo_id>_random_r{f}  -- the same count removed from one fixed random permutation
Both families are nested across f, so a trend over f is not confounded by resampling.
Real episodes are never touched: the only difference from Control is which imagined pairs remain.

delete_episodes recomputes stats from the remaining episodes; they are re-pinned to the real-only
set (as run_dose464.sh does) so normalization is identical across every arm.

    uv run --extra training python examples/nanowm_pusht_aug/attribution/build_filtered_datasets.py \
        --config configs/cupid_filter_untrimmed464.yaml
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
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.constants import HF_LEROBOT_HOME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_io import load_config  # noqa: E402

REAL_REPO = "local/pusht_imgonly"
PINNED = ("observation.image", "action")


def arm_repo_ids(cfg: dict) -> list[tuple[str, str, float]]:
    """(method, repo_id, frac) for every arm -- shared with run_cupid_filter.sh via --list."""
    return [
        (m, f"{cfg['train_repo_id']}_{m}_r{round(f * 100)}", f)
        for f in cfg["filter"]["remove_fracs"]
        for m in ("cupid", "random")
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--list", action="store_true", help="print the arm repo ids and exit")
    args = p.parse_args()
    cfg = load_config(args.config, out_dir_copy=False)
    if args.list:
        print("\n".join(r for _, r, _ in arm_repo_ids(cfg)))
        return

    out = Path(cfg["out_dir"])
    with open(out / "pair_scores.csv") as f:
        rows = list(csv.DictReader(f))
    ep = np.array([int(r["dataset_episode"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    by_score = ep[np.argsort(score)]  # ascending: worst first
    by_random = ep[np.random.default_rng(cfg["filter"]["random_seed"]).permutation(len(ep))]
    real_stats = json.loads((HF_LEROBOT_HOME / REAL_REPO / "meta/stats.json").read_text())
    src = LeRobotDataset(cfg["train_repo_id"], root=HF_LEROBOT_HOME / cfg["train_repo_id"])

    manifest = {}
    for method, repo_id, frac in arm_repo_ids(cfg):
        n = round(frac * len(ep))
        drop = sorted((by_score if method == "cupid" else by_random)[:n].tolist())
        root = HF_LEROBOT_HOME / repo_id
        if root.exists():
            shutil.rmtree(root)
        new = delete_episodes(src, episode_indices=drop, output_dir=root, repo_id=repo_id)
        sp = root / "meta/stats.json"
        stats = json.loads(sp.read_text())
        for k in PINNED:
            stats[k] = real_stats[k]
        sp.write_text(json.dumps(stats))
        assert new.meta.total_episodes == src.meta.total_episodes - n
        manifest[repo_id] = {
            "method": method,
            "remove_frac": frac,
            "n_removed": n,
            "removed_dataset_episodes": drop,
        }
        print(
            f"{repo_id}: removed {n} pairs -> {new.meta.total_episodes} episodes, stats pinned to {REAL_REPO}"
        )
    (out / "filtered_datasets.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
