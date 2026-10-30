#!/usr/bin/env python
"""early_anchors step 7 report, judged by <out_dir>/criteria.md.

<eval-dir>/{base, img_s<seed>, real_s<seed>}/ each hold eval_info.json + success_steps.npz on the same episodes
(seed 100000 + i). For arms X, Y the pooled difference is the mean over fine-tune seeds of (rate X_s - rate Y_s), the
base being the same checkpoint for every seed; the 95% CI comes from a paired bootstrap that resamples EPISODE indices
once per draw and applies them to every seed and arm (10,000 draws). Exact McNemar per seed is reported only.

  P1  pooled Img - base >= +3 pp AND CI lower bound > 0
  P2  pooled Img - Real CI lower bound > -3 pp (non-inferiority)

Writes <eval-dir>/report.json and report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ea_common import boot_idx, mcnemar, pooled_diff

BAR, MARGIN = 0.03, 0.03
NEAR = 0.9 / 0.95  # max reward of a near miss: coverage >= 0.9 (gym_pusht reward = coverage / 0.95)


def load(d: Path) -> dict:
    """One eval dir: per-episode success (> 0.95), max reward and success step."""
    m = json.loads((d / "eval_info.json").read_text())["per_task"][0]["metrics"]
    z = np.load(d / "success_steps.npz")
    s = np.asarray(m["successes"], bool)
    assert np.array_equal(s, z["success"]), d
    return {"succ": s, "max_r": np.asarray(m["max_rewards"], float), "step": z["success_step"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001])
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    R = {("base", s): load(a.eval_dir / "base") for s in a.seeds}
    for arm in ("img", "real"):
        for s in a.seeds:
            R[arm, s] = load(a.eval_dir / f"{arm}_s{s}")
    n = len(R["base", a.seeds[0]]["succ"])
    assert all(len(v["succ"]) == n for v in R.values())
    idx = boot_idx(n, a.draws)
    out = {"n_episodes": n, "seeds": a.seeds, "arms": {}, "pairs": {}}
    for arm in ("base", "img", "real"):
        ss = a.seeds[:1] if arm == "base" else a.seeds
        out["arms"][arm] = {
            "per_seed": [float(R[arm, s]["succ"].mean()) for s in ss],
            "near_miss_failures": [
                int((~R[arm, s]["succ"] & (R[arm, s]["max_r"] >= NEAR)).sum()) for s in ss
            ],
        }
        out["arms"][arm]["pooled"] = float(np.mean(out["arms"][arm]["per_seed"]))
    for x, y in (("img", "base"), ("img", "real"), ("real", "base")):
        both = [R[x, s]["succ"] & R[y, s]["succ"] for s in a.seeds]
        out["pairs"][f"{x}-{y}"] = {
            **pooled_diff([R[x, s]["succ"] for s in a.seeds], [R[y, s]["succ"] for s in a.seeds], idx),
            "mcnemar_p": [mcnemar(R[x, s]["succ"], R[y, s]["succ"]) for s in a.seeds],
            "median_steps_both_solved": [
                [float(np.median(R[x, s]["step"][b])), float(np.median(R[y, s]["step"][b]))]
                for s, b in zip(a.seeds, both, strict=True)
            ],
        }
    p1, p2 = out["pairs"]["img-base"], out["pairs"]["img-real"]
    out["P1"] = bool(p1["pooled"] >= BAR and p1["ci95"][0] > 0)
    out["P2"] = bool(p2["ci95"][0] > -MARGIN)
    (a.eval_dir / "report.json").write_text(json.dumps(out, indent=1))

    S = " | ".join(f"seed {s}" for s in a.seeds)
    lines = [
        f"# early_anchors fine-tune ({n} paired episodes, seeds {a.seeds})",
        "",
        f"| arm | {S} | pooled | near-miss failures (max cov >= 0.9) |",
        "|---|" + "---|" * len(a.seeds) + "---|---|",
    ]
    for arm, v in out["arms"].items():
        rates = v["per_seed"] * (len(a.seeds) if arm == "base" else 1)
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{r * 100:.1f}%" for r in rates)
            + f" | {v['pooled'] * 100:.1f}% | {v['near_miss_failures']} |"
        )
    lines += [
        "",
        f"| comparison | {S} | pooled (pp) | 95% CI (pp) | McNemar p | median steps both solved |",
        "|---|" + "---|" * len(a.seeds) + "---|---|---|---|",
    ]
    for k, v in out["pairs"].items():
        lines.append(
            f"| {k.replace('-', ' - ')} | "
            + " | ".join(f"{d * 100:+.1f}" for d in v["per_seed"])
            + f" | {v['pooled'] * 100:+.2f} | [{v['ci95'][0] * 100:+.2f}, {v['ci95'][1] * 100:+.2f}] | "
            + ", ".join(f"{p:.3g}" for p in v["mcnemar_p"])
            + " | "
            + ", ".join(f"{m[0]:.0f} vs {m[1]:.0f}" for m in v["median_steps_both_solved"])
            + " |"
        )
    lines += [
        "",
        f"P1 (Img - base >= +3 pp and CI lower bound > 0): {'MET' if out['P1'] else 'NOT MET'}",
        f"P2 (Img - Real CI lower bound > -3 pp): {'MET' if out['P2'] else 'NOT MET'}",
    ]
    (a.eval_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
