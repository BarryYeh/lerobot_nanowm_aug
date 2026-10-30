#!/usr/bin/env python
"""early_anchors: re-score every arm with success = sim coverage >= 0.9 (the rollout filter / fidelity criterion)
instead of the eval's > 0.95, from the saved evals; no new simulation.

Exact, not an approximation: an episode's trajectory up to the first step with coverage >= 0.9 does not depend on the
termination threshold (0.95 only ends episodes later), so "reached 0.9 within 300 steps" is max reward >= 0.9 / 0.95
(gym_pusht reward = coverage / 0.95) in eval_info.json. Steps-to-0.9 are not saved, so no median-steps column.

Statistics as report.py: pooled difference = mean over fine-tune seeds of (rate X_s - rate Y_s), a single-checkpoint
arm (base, refs) reused for every seed; 95% CI from a paired bootstrap over episode indices (10,000 draws, rng 0);
exact McNemar per seed. P1 / P2 use report.py's bars on the re-scored success.

    uv run python research/nanowm_aug/proj/pusht/early_anchors/report_cov09.py <eval-dir> [--refs base100k ...]

Arms read: base, {img,real,ctrl}_s<seed> (ctrl only if present), and each --refs subdir. Writes
<eval-dir>/report_cov09.{json,md}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ea_common import boot_idx, mcnemar, pooled_diff
from report import BAR, MARGIN, NEAR, load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001])
    ap.add_argument(
        "--refs", nargs="*", default=[], help="extra single-checkpoint eval subdirs to compare base with"
    )
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    E, S = a.eval_dir, a.seeds

    def hit(d: Path) -> dict:
        r = load(d)
        return {"cov09": r["max_r"] >= NEAR, "succ095": r["succ"]}

    single = ["base", *a.refs]
    multi = [arm for arm in ("img", "real", "ctrl") if all((E / f"{arm}_s{s}").is_dir() for s in S)]
    R = {}
    for arm in single:
        h = hit(E / arm)
        for s in S:
            R[arm, s] = h
    for arm in multi:
        for s in S:
            R[arm, s] = hit(E / f"{arm}_s{s}")
    n = len(R["base", S[0]]["cov09"])
    assert all(len(v["cov09"]) == n for v in R.values())
    idx = boot_idx(n, a.draws)

    out = {"n_episodes": n, "seeds": S, "success": "sim coverage >= 0.9", "arms": {}, "pairs": {}}
    for arm in single + multi:
        ss = S[:1] if arm in single else S
        out["arms"][arm] = {
            "per_seed": [float(R[arm, s]["cov09"].mean()) for s in ss],
            "per_seed_cov095": [float(R[arm, s]["succ095"].mean()) for s in ss],
        }
        out["arms"][arm]["pooled"] = float(np.mean(out["arms"][arm]["per_seed"]))
        out["arms"][arm]["pooled_cov095"] = float(np.mean(out["arms"][arm]["per_seed_cov095"]))
    comps = [("img", "base"), ("img", "real"), ("real", "base")]
    if "ctrl" in multi:
        comps += [("ctrl", "base"), ("img", "ctrl"), ("real", "ctrl")]
    comps += [(x, ref) for ref in a.refs for x in ("base", *multi)]
    for x, y in comps:
        out["pairs"][f"{x}-{y}"] = {
            **pooled_diff([R[x, s]["cov09"] for s in S], [R[y, s]["cov09"] for s in S], idx),
            "mcnemar_p": [mcnemar(R[x, s]["cov09"], R[y, s]["cov09"]) for s in S],
        }
    p1, p2 = out["pairs"]["img-base"], out["pairs"]["img-real"]
    out["P1"] = bool(p1["pooled"] >= BAR and p1["ci95"][0] > 0)
    out["P2"] = bool(p2["ci95"][0] > -MARGIN)
    (E / "report_cov09.json").write_text(json.dumps(out, indent=1))

    H = " | ".join(f"seed {s}" for s in S)
    lines = [
        f"# early_anchors re-scored with success = sim coverage >= 0.9 ({n} paired episodes, seeds {S})",
        "",
        f"| arm | {H} | pooled | pooled at > 0.95 (report.md) |",
        "|---|" + "---|" * len(S) + "---|---|",
    ]
    for arm, v in out["arms"].items():
        rates = v["per_seed"] * (len(S) if arm in single else 1)
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{r * 100:.1f}%" for r in rates)
            + f" | {v['pooled'] * 100:.1f}% | {v['pooled_cov095'] * 100:.1f}% |"
        )
    lines += [
        "",
        f"| comparison | {H} | pooled (pp) | 95% CI (pp) | McNemar p |",
        "|---|" + "---|" * len(S) + "---|---|---|",
    ]
    for k, v in out["pairs"].items():
        lines.append(
            f"| {k.replace('-', ' - ')} | "
            + " | ".join(f"{d * 100:+.1f}" for d in v["per_seed"])
            + f" | {v['pooled'] * 100:+.2f} | [{v['ci95'][0] * 100:+.2f}, {v['ci95'][1] * 100:+.2f}] | "
            + ", ".join(f"{p:.3g}" for p in v["mcnemar_p"])
            + " |"
        )
    lines += [
        "",
        f"P1 (Img - base >= +3 pp and CI lower bound > 0): {'MET' if out['P1'] else 'NOT MET'}",
        f"P2 (Img - Real CI lower bound > -3 pp): {'MET' if out['P2'] else 'NOT MET'}",
    ]
    (E / "report_cov09.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
