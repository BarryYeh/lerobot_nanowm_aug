"""TRAK scoring + CUPID aggregation (stage 3.2): one performance-influence score per imagined pair.

  score(i, q) = phi_i^T (X^T X + lambda I)^-1 phi_q          X = [phi_real; phi_pool] (all trained-on)
  pair score  = sum_e w_e * agg_{q in e} score(i, q)         e = test rollout episode
                binary:           w = +1 success / -1 failure   (CUPID "net" = succ - fail)
                centered_max_cov: w = max_cov_e - mean(max_cov)  (graded; ablation)
                agg = sum (CUPID sum_of_sum) or mean (removes the episode-length bias)
High score = the pair pushes theta toward what succeeded and away from what failed. Filtering drops
the LOWEST scores.

Diagnostics written next to the scores:
  * score vs imagined round j   -- does CUPID mostly see world-model drift (later j = more drift)?
  * score by source / injection -- anchor (real frame) vs imagined pairs, per noise level (add-one-in)
  * score vs oracle             -- AUROC against real-sim replay success; utility != realism, so a
                                   moderate value is the interesting case, ~0.5 AND unstable = broken
  * pool vs real                -- where imagined pairs fall in the real samples' score distribution
  * replicate stability         -- Spearman between score.replicates (other sketch + timestep seeds);
                                   the reported pair score is their mean

    uv run --extra training --with matplotlib python examples/nanowm_pusht_aug/attribution/score_pairs.py \
        --config configs/cupid_filter_untrimmed464.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_io import load_config  # noqa: E402


def rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    """Mann-Whitney: P(score of a positive > score of a negative). None if labels are a single class
    (AUROC is undefined -- avoids a division by zero)."""
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rank(score) + 1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def episode_scores(feat_dir: Path, test: dict, cfg: dict) -> dict:
    """Per-episode influence, averaged over the independent sketches.

    Each sketch lives in its own random subspace, so its SCORE is computed separately and only the
    scores are averaged -- averaging features across sketches would be meaningless. 2-D feature
    files (written before n_sketches existed) are read as a single sketch.
    """
    raw = {s: np.load(feat_dir / f"phi_{s}.npy", mmap_mode="r") for s in ("real", "pool", "test")}
    n_sk = raw["real"].shape[0] if raw["real"].ndim == 3 else 1
    ep, n_ep = test["episode"], len(test["ep_success"])
    within = cfg["score"]["within_rollout"]
    counts = np.bincount(ep, minlength=n_ep)
    lam = cfg["xtx"]["lambda_reg"]
    pool = real = 0.0
    conds = []

    for m in range(n_sk):
        phi_real, phi_pool, phi_test = (
            np.asarray(raw[s][m] if raw[s].ndim == 3 else raw[s], dtype=np.float64)
            for s in ("real", "pool", "test")
        )
        # leave-one-out: theta trained on the pool too, so it is part of the Hessian proxy;
        # add-one-in (xtx.include_pool: false): the pool is a holdout, scored but outside XtX
        X = np.vstack([phi_real, phi_pool]) if cfg["xtx"].get("include_pool", True) else phi_real
        xtx = X.T @ X
        conds.append(float(np.linalg.cond(xtx)))
        # lambda_rel: ridge as a fraction of XtX's largest eigenvalue (a holdout pool has large
        # leverage outside the real gradient span; lambda = 0 lets that dominate the score)
        lam_m = lam + cfg["xtx"].get("lambda_rel", 0.0) * np.linalg.eigvalsh(xtx)[-1]
        K = np.linalg.solve(xtx + lam_m * np.eye(len(xtx)), phi_test.T)  # (d, Q)

        def per_episode(phi: np.ndarray, k_mat: np.ndarray = K) -> np.ndarray:
            S = phi @ k_mat  # (N, Q)
            agg = np.zeros((len(phi), n_ep))
            np.add.at(agg.T, ep, S.T)  # sum over the queries of each episode
            return agg / counts if within == "mean" else agg

        pool = pool + per_episode(phi_pool) / n_sk
        real = real + per_episode(phi_real) / n_sk

    return {"pool": pool, "real": real, "cond": float(np.mean(conds)), "n_xtx": len(X), "n_sketches": n_sk}


def weights(test: dict, mode: str) -> np.ndarray:
    succ = test["ep_success"].astype(bool)
    if mode == "binary":
        return np.where(succ, 1.0, -1.0)
    if mode == "centered_max_cov":
        r = test["ep_max_reward"].astype(np.float64)
        return r - r.mean()
    raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config, out_dir_copy=False)
    out = Path(cfg["out_dir"])
    test = dict(np.load(out / "test_rollouts.npz"))
    succ = test["ep_success"].astype(bool)
    w = weights(test, cfg["score"]["rollout_weight"])

    # Final scores average every replicate listed in score.replicates (independent sketch + timestep
    # seeds); the first two are also compared with each other as the rank-stability gate.
    reps = [r for r in cfg["score"].get("replicates", ["features"]) if (out / r / "phi_real.npy").exists()]
    eps = {r: episode_scores(out / r, test, cfg) for r in reps}
    main_ep = eps[reps[0]]
    rep_pair = {r: e["pool"] @ w for r, e in eps.items()}
    pair = np.mean([rep_pair[r] for r in reps], 0)
    real = np.mean([e["real"] @ w for e in eps.values()], 0)
    pool_ep = np.mean([e["pool"] for e in eps.values()], 0)
    succ_part = pool_ep[:, succ].sum(1)
    fail_part = -pool_ep[:, ~succ].sum(1)

    # ---- metadata: sidecar (+ oracle when the pool has a simulator replay) ----
    side = json.loads(Path(cfg["pool_index"]).read_text())
    pairs = side["pairs"]
    j = np.array([r["j"] for r in pairs])
    source = np.array([r.get("source", "imagined") for r in pairs])
    inject = np.array([float(r.get("inject_sigma", 0.0)) for r in pairs])
    ep_offset = 0 if cfg.get("pool_repo_id") else cfg["real_episodes"]  # separate pool vs imagined tail
    o_succ = o_peak = None
    if cfg.get("oracle_npz"):
        branches = sorted(p.name for p in Path(side["accepted"]).glob("*.npz"))
        row = {b: i for i, b in enumerate(branches)}
        orc = np.load(cfg["oracle_npz"])
        assert len(orc["success"]) == len(branches), "oracle rows != branches in the sidecar's accepted dir"
        o_succ = np.array([bool(orc["success"][row[r["branch"]]]) for r in pairs])
        o_peak = np.array([float(orc["R_peak"][row[r["branch"]]]) for r in pairs])

    def by(groups: np.ndarray) -> dict:
        return {
            str(v): [
                float(pair[groups == v].mean()),
                float(pair[groups == v].std() / np.sqrt((groups == v).sum())),
                int((groups == v).sum()),
            ]  # fmt: skip
            for v in np.unique(groups)
        }

    # ---- diagnostics ----
    summary = {
        "n_pool": len(pair),
        "n_real": len(real),
        "n_test_episodes": len(succ),
        "test_success_rate": float(succ.mean()),
        "n_test_queries": int(len(test["episode"])),
        "mean_queries_succ_vs_fail": [
            float(test["ep_n_queries"][succ].mean()),
            float(test["ep_n_queries"][~succ].mean()),
        ],
        "xtx_cond": float(main_ep["cond"]),
        "n_xtx": int(main_ep["n_xtx"]),
        "score_by_j": by(j),
        "spearman_score_vs_j": spearman(pair, j),
        "score_by_source": by(source),
        "score_by_inject_sigma": by(inject),
        "pool_mean_vs_real_mean": [float(pair.mean()), float(real.mean())],
        "pool_frac_below_real_p10": float((pair < np.percentile(real, 10)).mean()),
        "pool_frac_above_real_p90": float((pair > np.percentile(real, 90)).mean()),
    }
    if o_succ is not None:
        summary["oracle_success_rate_in_pool"] = float(o_succ.mean())
        summary["auroc_score_vs_oracle_success"] = auroc(pair, o_succ)
        summary["spearman_score_vs_oracle_peak"] = spearman(pair, o_peak)
    k_sel = cfg.get("select", {}).get("k")
    if k_sel:  # add-one-in: what the top-k CUPID would add is made of
        top = np.argsort(-pair)[:k_sel]
        summary["top_k_composition"] = {
            "k": int(k_sel),
            "source": {str(v): int((source[top] == v).sum()) for v in np.unique(source)},
            "inject_sigma": {str(v): int((inject[top] == v).sum()) for v in np.unique(inject)},
        }
    summary["replicates_averaged"] = reps
    for r in reps[1:]:
        summary[f"spearman_{reps[0]}_vs_{r}"] = spearman(rep_pair[reps[0]], rep_pair[r])

    with open(out / "pair_scores.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair_episode", "dataset_episode", "branch", "j", "source", "inject_sigma", "score",
                     "score_succ", "score_fail", "oracle_success", "oracle_peak"])  # fmt: skip
        for i, r in enumerate(pairs):
            wr.writerow([r["episode_index"], ep_offset + r["episode_index"], r["branch"], r["j"], source[i],
                         inject[i], pair[i], succ_part[i], fail_part[i],
                         "" if o_succ is None else int(o_succ[i]), "" if o_peak is None else o_peak[i]])  # fmt: skip
    np.save(out / "real_sample_scores.npy", real)
    (out / "score_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].boxplot([pair[j == v] for v in np.unique(j)], tick_labels=[str(v) for v in np.unique(j)])
    ax[0].set(
        xlabel="imagined round j",
        ylabel="pair score",
        title=f"score vs j (rho={summary['spearman_score_vs_j']:+.2f})",
    )
    if o_succ is not None:
        ax[1].hist([pair[o_succ], pair[~o_succ]], bins=40, label=["oracle success", "oracle fail"], alpha=0.7)
        auc = summary["auroc_score_vs_oracle_success"]
        ax[1].set(xlabel="pair score", title=f"vs oracle (AUROC={'n/a' if auc is None else f'{auc:.2f}'})")
        ax[1].legend()
    else:
        grp = np.array([f"{a}\n{b:g}sigma" for a, b in zip(source, inject, strict=True)])
        keys = sorted(np.unique(grp))
        ax[1].boxplot([pair[grp == g] for g in keys], tick_labels=keys)
        ax[1].set(ylabel="pair score", title="score by source x injection")
    bins = np.linspace(
        min(pair.min(), np.percentile(real, 0.5)), max(pair.max(), np.percentile(real, 99.5)), 60
    )
    ax[2].hist(real, bins=bins, density=True, alpha=0.5, label="real samples")
    ax[2].hist(pair, bins=bins, density=True, alpha=0.5, label="imagined pairs")
    ax[2].set(xlabel="score", title="pool vs real")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "score_diagnostics.png", dpi=110)
    print(f"-> {out / 'score_diagnostics.png'}")


if __name__ == "__main__":
    main()
