#!/usr/bin/env python
"""Compare two eval_baseline.py result JSONs (a vs a+aug): paired coverage stats + CDF.

Validation question: does the NanoWM imagined data raise the policy's PushT coverage?
Primary signal = per-episode max coverage (continuous; the baseline is ~0% success so
binary success rate has no resolution). Stats are non-parametric (bounded, skewed):
Wilcoxon signed-rank on the paired delta + bootstrap CI, not a t-test.

Pairing: episodes are matched by (init_id, rollout_id). eval_baseline.py uses a
deterministic per-episode seed, so `a` and `a+aug` saw identical reset + diffusion
noise -> a true paired comparison, which removes init-state variance.

Run (matplotlib is not in the venv -> pull it in with --with):
    uv run --with matplotlib python examples/nanowm_pusht_aug/eval/compare_eval.py \
        outputs/eval/nanowm_aug/a_valinit.json outputs/eval/nanowm_aug/aug_valinit.json \
        --out outputs/eval/nanowm_aug/compare_valinit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

THRESHOLDS = (0.5, 0.7, 0.9, 0.95)


def load(path: Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    by_key = {(e["init_id"], e["rollout_id"]): e for e in d["episodes"]}
    return d, by_key


def dist_row(name: str, x: np.ndarray) -> str:
    return (
        f"  {name:9s} n={len(x):3d}  mean={x.mean():.3f}  median={np.median(x):.3f}  "
        f"p25={np.percentile(x, 25):.3f}  p75={np.percentile(x, 75):.3f}  p90={np.percentile(x, 90):.3f}"
    )


def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a_json", type=Path, help="baseline (a) result")
    p.add_argument("b_json", type=Path, help="augmented (a+aug) result")
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="a+aug")
    p.add_argument("--out", type=Path, help="prefix for <out>_report.txt / <out>_cdf.png")
    args = p.parse_args()

    da, ka = load(args.a_json)
    db, kb = load(args.b_json)
    if da["distribution"] != db["distribution"]:
        raise SystemExit(f"distribution mismatch: {da['distribution']} vs {db['distribution']}")

    keys = sorted(set(ka) & set(kb))
    if len(keys) < len(ka) or len(keys) < len(kb):
        print(f"[warn] pairing on intersection: {len(keys)} of a={len(ka)} b={len(kb)}")

    a_mx = np.array([ka[k]["max_cov"] for k in keys])
    b_mx = np.array([kb[k]["max_cov"] for k in keys])
    a_fn = np.array([ka[k]["final_cov"] for k in keys])
    b_fn = np.array([kb[k]["final_cov"] for k in keys])
    thresh = da["success_threshold"]
    d = b_mx - a_mx  # paired delta on max coverage

    L = []
    L.append(f"distribution: {da['distribution']}   paired episodes: {len(keys)}")
    L.append(f"  {args.label_a}: {da['config']}")
    L.append(f"  {args.label_b}: {db['config']}")
    L.append("\n== max_cov distribution ==")
    L.append(dist_row(args.label_a, a_mx))
    L.append(dist_row(args.label_b, b_mx))
    L.append("\n== final_cov distribution ==")
    L.append(dist_row(args.label_a, a_fn))
    L.append(dist_row(args.label_b, b_fn))

    L.append("\n== frac(max_cov >= t) ==")
    L.append(f"  {'t':>6} | {args.label_a:>8} | {args.label_b:>8} | delta")
    for t in (*THRESHOLDS, thresh):
        fa, fb = (a_mx >= t).mean(), (b_mx >= t).mean()
        L.append(f"  {t:>6.2f} | {fa:>8.3f} | {fb:>8.3f} | {fb - fa:+.3f}")

    L.append("\n== paired delta (max_cov: b - a) ==")
    lo, hi = bootstrap_ci(d)
    L.append(f"  mean d   = {d.mean():+.4f}   bootstrap 95% CI = [{lo:+.4f}, {hi:+.4f}]")
    L.append(f"  median d = {np.median(d):+.4f}")
    nz = d[d != 0]
    if len(nz) >= 1:
        w_two = stats.wilcoxon(nz, alternative="two-sided")
        w_gt = stats.wilcoxon(nz, alternative="greater")
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        rbc = (n_pos - n_neg) / len(nz)  # rank-biserial-ish sign summary
        L.append(
            f"  Wilcoxon (n_nonzero={len(nz)}): two-sided p={w_two.pvalue:.4g}   "
            f"one-sided (b>a) p={w_gt.pvalue:.4g}"
        )
        L.append(f"  wins b/a/tie = {n_pos}/{n_neg}/{int((d == 0).sum())}   sign-summary={rbc:+.3f}")
    else:
        L.append("  (all deltas zero)")

    verdict = "HELPS" if lo > 0 else ("HURTS" if hi < 0 else "INCONCLUSIVE")
    L.append("\n== verdict (val-init a-priori: mean-d 95% CI > 0 AND frac>=0.7 up) ==")
    L.append(f"  mean-d CI excludes 0 and > 0 : {lo > 0}")
    L.append(f"  frac(max_cov>=0.7) up        : {(b_mx >= 0.7).mean() > (a_mx >= 0.7).mean()}")
    L.append(f"  -> {verdict}")

    report = "\n".join(L)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.out}_report.txt").write_text(report + "\n")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for x, lab, c in [(a_mx, args.label_a, "C0"), (b_mx, args.label_b, "C1")]:
                xs = np.sort(x)
                ax[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab, color=c, lw=2)
            ax[0].set(
                xlabel="max coverage",
                ylabel="empirical CDF",
                title=f"max_cov CDF ({da['distribution']}, n={len(keys)})",
            )
            ax[0].legend()
            ax[0].grid(alpha=0.3)

            nbins = 25 if np.ptp(d) > 1e-9 else 1
            ax[1].hist(d, bins=nbins, color="C2", alpha=0.8)
            ax[1].axvline(0, color="k", lw=1)
            ax[1].axvline(d.mean(), color="C3", lw=2, label=f"mean {d.mean():+.3f}")
            ax[1].set(
                xlabel=f"paired delta max_cov ({args.label_b} - {args.label_a})",
                ylabel="episodes",
                title="paired delta",
            )
            ax[1].legend()
            ax[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{args.out}_cdf.png", dpi=110)
            print(f"\nwrote {args.out}_report.txt  {args.out}_cdf.png")
        except ImportError:
            print(
                f"\nwrote {args.out}_report.txt  (matplotlib missing -> no plot; rerun with `uv run --with matplotlib`)"
            )


if __name__ == "__main__":
    main()
