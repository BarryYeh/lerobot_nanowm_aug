#!/usr/bin/env python
"""early_anchors: compare the fine-tune arms with a reference checkpoint evaluated on the same episodes.

Secondary analysis, outside criteria.md: report.py's P1 / P2 stay judged against base 040000. Here every arm
(base 040000, img, real) is set against <eval-dir>/<ref>/ (e.g. base100k = base_s100000/checkpoints/100000), with
report.py's statistics: pooled difference = mean over fine-tune seeds of (rate X_s - rate ref), 95% CI from a paired
bootstrap over episode indices (10,000 draws, rng 0), exact McNemar per seed.

    uv run python research/nanowm_aug/proj/pusht/early_anchors/compare_ref.py <eval-dir> --ref base100k

Writes <eval-dir>/compare_<ref>.json and compare_<ref>.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ea_common import boot_idx, mcnemar, pooled_diff
from report import load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--ref", required=True, help="eval subdir of the reference checkpoint")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001])
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args()
    ref = load(a.eval_dir / a.ref)
    arms = {"base": {s: load(a.eval_dir / "base") for s in a.seeds[:1]}}
    for arm in ("img", "real"):
        arms[arm] = {s: load(a.eval_dir / f"{arm}_s{s}") for s in a.seeds}
    n = len(ref["succ"])
    assert all(len(v["succ"]) == n for d in arms.values() for v in d.values())
    idx = boot_idx(n, a.draws)
    out = {"n_episodes": n, "ref": a.ref, "ref_rate": float(ref["succ"].mean()), "pairs": {}}
    for arm, runs in arms.items():
        out["pairs"][f"{arm}-{a.ref}"] = {
            "seeds": list(runs),
            "rate": [float(r["succ"].mean()) for r in runs.values()],
            **pooled_diff([r["succ"] for r in runs.values()], [ref["succ"]] * len(runs), idx),
            "mcnemar_p": [mcnemar(r["succ"], ref["succ"]) for r in runs.values()],
        }
    (a.eval_dir / f"compare_{a.ref}.json").write_text(json.dumps(out, indent=1))
    lines = [
        f"# early_anchors arms vs {a.ref} ({n} paired episodes; {a.ref} = {out['ref_rate'] * 100:.1f}%)",
        "",
        "| comparison | rate per seed | diff per seed (pp) | pooled (pp) | 95% CI (pp) | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in out["pairs"].items():
        lines.append(
            f"| {k.replace('-', ' - ')} | "
            + ", ".join(f"{r * 100:.1f}%" for r in v["rate"])
            + " | "
            + ", ".join(f"{d * 100:+.1f}" for d in v["per_seed"])
            + f" | {v['pooled'] * 100:+.2f} | [{v['ci95'][0] * 100:+.2f}, {v['ci95'][1] * 100:+.2f}] | "
            + ", ".join(f"{p:.3g}" for p in v["mcnemar_p"])
            + " |"
        )
    (a.eval_dir / f"compare_{a.ref}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
