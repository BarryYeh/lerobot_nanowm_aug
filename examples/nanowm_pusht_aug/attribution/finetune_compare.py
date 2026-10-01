#!/usr/bin/env python
"""Paired success table for base / arm V / arm R (lerobot-eval eval_info.json, same start seeds per arm).

    uv run python examples/nanowm_pusht_aug/attribution/finetune_compare.py \
        ~/datastorage/nanowm_pusht_aug/finetune/eval --arms base v r
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_report import load_successes, mcnemar_exact, paired_bootstrap  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_dir", type=Path)
    p.add_argument("--arms", nargs="+", default=["base", "v", "r"])
    a = p.parse_args()
    s = {t: load_successes(a.eval_dir.expanduser() / t) for t in a.arms}
    lines = ["| arm | n | successes | success rate | 95% CI (Wilson) |", "|---|---|---|---|---|"]
    res = {"arms": {}, "pairs": {}}
    for t, v in s.items():
        lo, hi = wilson(int(v.sum()), len(v))
        lines.append(f"| {t} | {len(v)} | {int(v.sum())} | {v.mean() * 100:.1f}% | [{lo * 100:.1f}, {hi * 100:.1f}] |")
        res["arms"][t] = {"n": len(v), "k": int(v.sum()), "rate": float(v.mean()), "ci95": [lo, hi]}
    lines += ["", "| comparison | diff (pp) | paired bootstrap 95% CI (pp) | only-A / only-B | McNemar p |", "|---|---|---|---|---|"]
    pairs = [("v", "r"), ("v", "base"), ("r", "base")]
    for x, y in pairs:
        if x not in s or y not in s:
            continue
        va, vb = s[x], s[y]
        assert len(va) == len(vb)
        lo, hi = paired_bootstrap(va, vb)
        pv = mcnemar_exact(va, vb)
        d = (va.mean() - vb.mean()) * 100
        ob = (int((va & ~vb).sum()), int((~va & vb).sum()))
        lines.append(f"| {x} - {y} | {d:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] | {ob[0]} / {ob[1]} | {pv:.3f} |")
        res["pairs"][f"{x}-{y}"] = {"diff": float(d / 100), "ci95": [lo, hi], "only_a": ob[0], "only_b": ob[1], "mcnemar_p": pv}
    txt = "\n".join(lines)
    print(txt)
    (a.eval_dir.expanduser() / "comparison.md").write_text(txt + "\n")
    (a.eval_dir.expanduser() / "comparison.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
