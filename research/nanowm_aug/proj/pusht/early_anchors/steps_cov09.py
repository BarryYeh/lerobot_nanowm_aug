#!/usr/bin/env python
"""early_anchors: steps to reach sim coverage >= 0.9, from the re-run evals in <run>/finetune/eval_steps09/.

First checks that each re-run reproduces the original eval (<run>/finetune/eval/<arm>/: same success flags), then, per
comparison X - Y, takes the episodes where BOTH arms reach 0.9 (per seed) and reports the median steps of each and the
mean per-episode step difference, pooled over fine-tune seeds; 95% CI from a paired bootstrap over episode indices
(10,000 draws, rng 0; an episode outside a seed's both-reach set contributes nothing to that seed).

    uv run python research/nanowm_aug/proj/pusht/early_anchors/steps_cov09.py <run-dir> [<run-dir> ...]

Writes <run>/finetune/eval_steps09/steps_cov09.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ea_common import boot_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001])
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    S = a.seeds
    for run in a.runs:
        E, orig = run / "finetune/eval_steps09", run / "finetune/eval"
        multi = [arm for arm in ("img", "real", "ctrl") if all((E / f"{arm}_s{s}").is_dir() for s in S)]
        R, match = {}, {}
        for name in ["base"] + [f"{m}_s{s}" for m in multi for s in S]:
            z = np.load(E / name / "success_steps.npz")
            match[name] = float(
                (z["success"] == np.load(orig / name / "success_steps.npz")["success"]).mean()
            )
            R[name] = z["cov09_step"]
        key = lambda arm, s: "base" if arm == "base" else f"{arm}_s{s}"  # noqa: E731
        n = len(R["base"])
        idx = boot_idx(n, a.draws)
        arms = {
            arm: {
                "reach09": float(np.mean([(R[key(arm, s)] > 0).mean() for s in S])),
                "median_steps": float(
                    np.median(np.concatenate([R[key(arm, s)][R[key(arm, s)] > 0] for s in S]))
                ),
            }
            for arm in ["base", *multi]
        }
        comps = [("img", "base"), ("img", "real"), ("real", "base")]
        if "ctrl" in multi:
            comps += [("ctrl", "base"), ("img", "ctrl")]
        pairs = {}
        for x, y in comps:
            num, cnt, med = np.zeros(n), np.zeros(n), []
            for s in S:
                X, Y = R[key(x, s)], R[key(y, s)]
                both = (X > 0) & (Y > 0)
                num += np.where(both, X - Y, 0)
                cnt += both
                med.append([float(np.median(X[both])), float(np.median(Y[both]))])
            boot = num[idx].sum(1) / cnt[idx].sum(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pairs[f"{x}-{y}"] = {
                "mean_diff": float(num.sum() / cnt.sum()),
                "ci95": [float(lo), float(hi)],
                "median_both": med,
                "n_both": int(cnt.sum()),
            }
        out = {"reproduces_original": match, "arms": arms, "pairs": pairs}
        (E / "steps_cov09.json").write_text(json.dumps(out, indent=1))
        lines = [
            f"# {run.name}: steps to sim coverage >= 0.9 ({n} eps, seeds {S})",
            "",
            "Re-run reproduces original success flags: "
            + ", ".join(f"{k} {v * 100:.2f}%" for k, v in match.items()),
            "",
            "| arm | reach 0.9 | median steps (reached) |",
            "|---|---|---|",
        ]
        lines += [f"| {k} | {v['reach09'] * 100:.1f}% | {v['median_steps']:.0f} |" for k, v in arms.items()]
        lines += [
            "",
            "| comparison | episodes both reach | median X vs Y (s1000; s1001) | mean diff (steps) | 95% CI |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {k.replace('-', ' - ')} | {v['n_both']} | "
            + "; ".join(f"{m[0]:.0f} vs {m[1]:.0f}" for m in v["median_both"])
            + f" | {v['mean_diff']:+.1f} | [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] |"
            for k, v in pairs.items()
        ]
        (E / "steps_cov09.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), "\n")


if __name__ == "__main__":
    main()
