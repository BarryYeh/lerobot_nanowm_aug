#!/usr/bin/env python
"""Pooled multi-seed comparison for the last-mile experiment (criterion: lastmile/criteria.md, addendum 1).

Each <eval-dir>/<arm>_s<seed>/eval_info.json holds the same episodes (lerobot-eval seed 100000 + i). For arms X, Y the
pooled difference is the mean over seeds of (success rate X_s - success rate Y_s); its 95% CI comes from a paired
bootstrap that resamples EPISODE indices once per draw and applies them to every seed and arm.
Writes <eval-dir>/pooled.json / pooled.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path) -> tuple[np.ndarray, np.ndarray]:
    m = json.loads((d / "eval_info.json").read_text())["per_task"][0]["metrics"]
    return np.asarray(m["successes"], bool), np.asarray(m["max_rewards"], float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--arms", nargs="+", default=["W", "R", "ctrl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    ap.add_argument("--pairs", nargs="+", default=["W:R", "W:ctrl", "R:ctrl"])
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--bar", type=float, default=0.03, help="primary effect-size bar for the first pair")
    a = ap.parse_args()
    S, MR = {}, {}
    for arm in a.arms:
        for s in a.seeds:
            S[arm, s], MR[arm, s] = load(a.eval_dir / f"{arm}_s{s}")
    n = len(S[a.arms[0], a.seeds[0]])
    assert all(len(v) == n for v in S.values()), {k: len(v) for k, v in S.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(a.draws, n))
    out = {"n_episodes": n, "seeds": a.seeds, "arms": {}, "pairs": {}}
    for arm in a.arms:
        rates = [float(S[arm, s].mean()) for s in a.seeds]
        near = [int(((~S[arm, s]) & (MR[arm, s] >= 0.9 / 0.95)).sum()) for s in a.seeds]
        out["arms"][arm] = {
            "per_seed": rates,
            "pooled": float(np.mean(rates)),
            "near_miss_failures_per_seed": near,
        }
    for p in a.pairs:
        x, y = p.split(":")
        per = [float(S[x, s].mean() - S[y, s].mean()) for s in a.seeds]
        dsum = sum((S[x, s].astype(float) - S[y, s].astype(float)) for s in a.seeds) / len(
            a.seeds
        )  # per episode
        boot = dsum[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["pairs"][p] = {
            "per_seed": per,
            "pooled": float(dsum.mean()),
            "ci95": [float(lo), float(hi)],
            "seeds_positive": int(sum(d > 0 for d in per)),
        }
    first = a.pairs[0]
    pr = out["pairs"][first]
    out["primary"] = {"pair": first, "bar": a.bar, "met": bool(pr["pooled"] >= a.bar and pr["ci95"][0] > 0)}
    (a.eval_dir / "pooled.json").write_text(json.dumps(out, indent=1))
    lines = [
        f"# Pooled over seeds {a.seeds} ({n} paired episodes each)",
        "",
        "| arm | " + " | ".join(f"seed {s}" for s in a.seeds) + " | pooled | near-miss failures per seed |",
        "|---|" + "---|" * len(a.seeds) + "---|---|",
    ]
    for arm, v in out["arms"].items():
        lines.append(
            f"| {arm} | "
            + " | ".join(f"{r * 100:.1f}%" for r in v["per_seed"])
            + f" | {v['pooled'] * 100:.1f}% | {v['near_miss_failures_per_seed']} |"
        )
    lines += [
        "",
        "| comparison | "
        + " | ".join(f"seed {s}" for s in a.seeds)
        + " | pooled (pp) | 95% CI (pp) | seeds > 0 |",
        "|---|" + "---|" * len(a.seeds) + "---|---|---|",
    ]
    for p, v in out["pairs"].items():
        lines.append(
            f"| {p.replace(':', ' - ')} | "
            + " | ".join(f"{d * 100:+.1f}" for d in v["per_seed"])
            + f" | {v['pooled'] * 100:+.2f} | [{v['ci95'][0] * 100:+.2f}, {v['ci95'][1] * 100:+.2f}] | "
            f"{v['seeds_positive']}/{len(a.seeds)} |"
        )
    lines += [
        "",
        f"PRIMARY ({first} >= +{a.bar * 100:.0f} pp and CI lower bound > 0): "
        f"{'MET' if out['primary']['met'] else 'NOT MET'}",
    ]
    (a.eval_dir / "pooled.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
