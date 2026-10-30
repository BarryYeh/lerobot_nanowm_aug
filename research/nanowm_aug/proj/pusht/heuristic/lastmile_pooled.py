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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # research/nanowm_aug
from nwaug.attribution.paired_report import boot_idx, pooled_diff  # noqa: E402

NEAR = (
    0.9 / 0.95
)  # gym_pusht reward = clip(coverage / 0.95, 0, 1): max reward >= NEAR <=> max coverage >= 0.9


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
    ap.add_argument(
        "--margin",
        type=float,
        default=None,
        help="non-inferiority instead: primary met when the first pair's CI lower bound > -margin",
    )
    a = ap.parse_args()
    S, MR = {}, {}
    for arm in a.arms:
        for s in a.seeds:
            S[arm, s], MR[arm, s] = load(a.eval_dir / f"{arm}_s{s}")
    n = len(S[a.arms[0], a.seeds[0]])
    assert all(len(v) == n for v in S.values()), {k: len(v) for k, v in S.items()}
    idx = boot_idx(n, a.draws)
    out = {"n_episodes": n, "seeds": a.seeds, "arms": {}, "pairs": {}}
    for arm in a.arms:
        rates = [float(S[arm, s].mean()) for s in a.seeds]
        near = [int(((~S[arm, s]) & (MR[arm, s] >= NEAR)).sum()) for s in a.seeds]
        out["arms"][arm] = {
            "per_seed": rates,
            "pooled": float(np.mean(rates)),
            "near_miss_failures_per_seed": near,
        }
    for p in a.pairs:
        x, y = p.split(":")
        r = pooled_diff([S[x, s] for s in a.seeds], [S[y, s] for s in a.seeds], idx)
        out["pairs"][p] = {**r, "seeds_positive": int(sum(d > 0 for d in r["per_seed"]))}
    first = a.pairs[0]
    pr = out["pairs"][first]
    if a.margin is None:
        out["primary"] = {
            "pair": first,
            "bar": a.bar,
            "met": bool(pr["pooled"] >= a.bar and pr["ci95"][0] > 0),
        }
        rule = f"{first} >= +{a.bar * 100:.0f} pp and CI lower bound > 0"
    else:
        out["primary"] = {"pair": first, "margin": a.margin, "met": bool(pr["ci95"][0] > -a.margin)}
        rule = f"non-inferiority: {first} CI lower bound > -{a.margin * 100:g} pp"
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
        f"PRIMARY ({rule}): {'MET' if out['primary']['met'] else 'NOT MET'}",
    ]
    (a.eval_dir / "pooled.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
