#!/usr/bin/env python
"""SAILOR v1 end-to-end fine-tune comparison (sailor_v1_finetune.md): arm V (V-selected) vs arm R (random).

Pool = the T7 test-anchor imagined branches (<out_dir>/test_stage2/sigma*/branches), never seen by V.

    select   score every branch with the trained ValueEnsemble at the final query round (SAILOR z_end score) and r=2, with
             beta = the beta the T7 report selected on val; drop anchors G6 drops; pick one branch per
             anchor for arm V (argmax, tie -> (sigma, cand) ascending) and arm R (uniform, seed 0);
             write selection.json / branch_scores.csv and a merged symlink dir of the union of branches.
    build    (after pack_closed_loop.py packed that merged dir) real + selected pairs, real FIRST, stats
             re-pinned to the real-only set, one dataset per arm.

    uv run --extra training python examples/nanowm_pusht_aug/critic/finetune_pool.py select --config configs/sailor_v1_reset.yaml
    uv run --extra training python examples/nanowm_pusht_aug/critic/finetune_pool.py build  --config configs/sailor_v1_reset.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402

ARMS = ("V", "R")
RANDOM_SEED = 0
Q_EARLY = 2  # the final query round is max(stage2.query_rounds) (8 in sailor_v1, 6 in sailor_v1_h6)
N_SIGMA_CAND = 8  # 2 sigma x 4 cand


def branch_name(sigma: float, seed_id: int, cand: int) -> str:
    """Name the packer sees (run_cupid_select.sh `merge`: <sigma dir>_<branch file>)."""
    return f"sigma{sigma}_seed{seed_id:04d}_cand{cand}.npz"


def pool_dir(cfg: dict) -> Path:
    """`finetune.pool_dir` in the config, else <out_dir>/../finetune/pool (the sailor_v1 location)."""
    pd = cfg.get("finetune", {}).get("pool_dir")
    return Path(pd).expanduser() if pd else Path(cfg["out_dir"]).expanduser().parent / "finetune" / "pool"


def select(cfg: dict) -> None:
    q_final = int(max(cfg["stage2"]["query_rounds"]))
    import torch
    from gate_tables import candidate_table
    from train_value_model import load_stage2_rows, predict_rows
    from value_model import OracleEnsemble, ValueEnsemble

    out = Path(cfg["out_dir"]).expanduser()
    t2 = out / "test_stage2"
    rep = json.loads((out / "gate_report.json").read_text())
    beta = float(rep["G6"]["selected_beta"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_stage2_rows(t2 / "labels.npz", cfg["stage2"]["query_rounds"])
    ve = ValueEnsemble.load(out / "model" / "value_ensemble.pt", device=dev)
    oe = OracleEnsemble.load(out / "model" / "oracle_ensemble.pt", device=dev)
    d = predict_rows(ve, oe, rows, np.arange(len(rows["y"])), dev)
    d["sigma"] = np.round(d["sigma"].astype(np.float64), 4)

    # anchors G6 keeps: exactly 8 candidates at r=2, none wall_artifact (same function the report uses)
    tab = candidate_table(d, N_SIGMA_CAND, Q_EARLY)
    keep = [int(a) for a in tab["anchor"]]
    print(f"beta={beta}  G6-usable anchors {len(keep)}  wall-excluded {tab['n_wall_excluded']}  "
          f"missing-excluded {tab['n_missing_excluded']}")

    cols = ("anchor", "sigma", "cand", "score_final", "score_r2", "y_final", "y_r2", "pre_success_final", "wall_final", "v_final", "rm_final")
    table, per_anchor = [], {}
    for a in keep:
        cand_rows = []
        for sg in sorted(set(d["sigma"][d["seed_id"] == a].tolist())):
            for c in sorted(set(d["cand"][(d["seed_id"] == a) & (d["sigma"] == sg)].tolist())):
                m = (d["seed_id"] == a) & (d["sigma"] == sg) & (d["cand"] == c)
                rfin, r2 = np.flatnonzero(m & (d["r"] == q_final)), np.flatnonzero(m & (d["r"] == Q_EARLY))
                assert len(rfin) == 1 and len(r2) == 1, (a, sg, c, len(rfin), len(r2))
                ifin, i2 = rfin[0], r2[0]
                s = lambda i: float(d["v_mean_all"][i] + beta * d["rm_mean_all"][i])  # noqa: E731
                cand_rows.append(
                    dict(anchor=a, sigma=float(sg), cand=int(c), score_final=s(ifin), score_r2=s(i2),
                         y_final=float(d["y"][ifin] | d["pre_success"][ifin]), y_r2=float(d["y"][i2] | d["pre_success"][i2]),
                         pre_success_final=bool(d["pre_success"][ifin]), wall_final=bool(d["wall_artifact"][ifin]),
                         v_final=float(d["v_mean_all"][ifin]), rm_final=float(d["rm_mean_all"][ifin]))
                )
        assert len(cand_rows) == N_SIGMA_CAND
        per_anchor[a] = cand_rows  # already (sigma, cand) ascending
        table += cand_rows

    rng = np.random.default_rng(RANDOM_SEED)
    sel = {"V": {}, "R": {}}
    for a in keep:  # ascending anchor order
        cr = per_anchor[a]
        sel["V"][a] = int(np.argmax([c["score_final"] for c in cr]))  # ties -> first = lowest (sigma, cand)
        sel["R"][a] = int(rng.integers(0, N_SIGMA_CAND))
    # sanity (labels only): real y at the final round of the selected candidates vs all candidates
    qual = {}
    for arm in ARMS:
        yfin = np.array([per_anchor[a][sel[arm][a]]["y_final"] for a in keep])
        y2 = np.array([per_anchor[a][sel[arm][a]]["y_r2"] for a in keep])
        qual[arm] = {"mean_y_final": float(yfin.mean()), "mean_y_r2": float(y2.mean()), "n": len(yfin)}
    allm = np.array([[c["y_final"] for c in per_anchor[a]] for a in keep])
    qual["all_candidates_mean_y_final"] = float(allm.mean())
    qual["hindsight_best_mean_y_final"] = float(allm.max(1).mean())
    qual["n_informative_final"] = int(((allm.sum(1) > 0) & (allm.sum(1) < N_SIGMA_CAND)).sum())
    qual["pre_success_selected_final"] = {
        arm: int(sum(per_anchor[a][sel[arm][a]]["pre_success_final"] for a in keep)) for arm in ARMS
    }
    qual["overlap_V_R"] = int(sum(sel["V"][a] == sel["R"][a] for a in keep))
    chosen = {arm: [(a, per_anchor[a][sel[arm][a]]) for a in keep] for arm in ARMS}
    arm_json = {
        arm: [branch_name(c["sigma"], a, c["cand"]) for a, c in chosen[arm]] for arm in ARMS
    }
    pd = pool_dir(cfg)
    pd.mkdir(parents=True, exist_ok=True)
    with open(pd / "branch_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    (pd / "selection.json").write_text(
        json.dumps({"beta": beta, "anchors": keep, "random_seed": RANDOM_SEED, "arms": arm_json,
                    "quality": qual, "n_g6_wall_excluded": tab["n_wall_excluded"],
                    "n_g6_missing_excluded": tab["n_missing_excluded"]}, indent=1)
    )
    merged = pd / "merged"
    if merged.exists():
        for p in merged.iterdir():
            p.unlink()
    merged.mkdir(parents=True, exist_ok=True)
    union = sorted(set(arm_json["V"]) | set(arm_json["R"]))
    for name in union:
        sg, rest = name.split("_", 1)  # "sigma0.1", "seed0003_cand2.npz"
        src = t2 / sg / "branches" / rest
        assert src.exists(), src
        z = np.load(src)
        assert int(z["seed_index"]) == int(rest[4:8]), (src, int(z["seed_index"]))  # seed_id == seed_index
        (merged / name).symlink_to(src.resolve())
    print(json.dumps(qual, indent=1))
    print(f"selection -> {pd / 'selection.json'}; merged {len(union)} branches -> {merged}")


def build(cfg: dict, pool_repo: str, real_repo: str, prefix: str) -> None:
    from build_selected_datasets import build_real_plus_pairs
    from lerobot.utils.constants import HF_LEROBOT_HOME

    pd = pool_dir(cfg)
    sel = json.loads((pd / "selection.json").read_text())
    pidx = json.loads((HF_LEROBOT_HOME / pool_repo / "meta/pair_index.json").read_text())["pairs"]
    real_stats = json.loads((HF_LEROBOT_HOME / real_repo / "meta/stats.json").read_text())
    n_real = json.loads((HF_LEROBOT_HOME / real_repo / "meta/info.json").read_text())["total_episodes"]
    manifest = {}
    for arm in ARMS:
        names = set(sel["arms"][arm])
        eps = sorted(p["episode_index"] for p in pidx if p["branch"] in names)
        repo = f"local/{prefix}_{arm.lower()}"
        n = build_real_plus_pairs(real_repo, pool_repo, eps, repo, real_stats)
        assert n == n_real + len(eps), (repo, n, n_real, len(eps))
        manifest[repo] = {"arm": arm, "n_branches": len(names), "n_pairs": len(eps),
                          "n_anchor_pairs": sum(p["source"] == "anchor" for p in pidx if p["episode_index"] in set(eps)),
                          "total_episodes": n}
        print(repo, manifest[repo])
    (pd / "datasets.json").write_text(json.dumps(manifest, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["select", "build"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pool-repo", default="local/pusht_sailor_v1_testpool")
    p.add_argument("--real-repo", default="local/pusht_imgonly")
    p.add_argument("--prefix", default="pusht_plus_sailorv1")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "select":
        select(cfg)
    else:
        build(cfg, a.pool_repo, a.real_repo, a.prefix)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
