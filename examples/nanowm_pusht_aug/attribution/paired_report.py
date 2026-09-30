"""Success-rate table + PAIRED comparisons for arms evaluated on the same episode seeds.

Every arm is evaluated with the same start seeds (episode i faces the same initial state in every
arm), so arm-vs-arm differences are paired. Independent binomial CIs throw that away; here each
comparison uses the per-episode outcomes:
  * exact McNemar test on the discordant episodes
  * paired bootstrap 95% CI of the success-rate difference
Comparisons: every `*_cupid_*` arm vs its `*_random_*` twin, and every arm vs `--ref`.

Reads <out>/eval/<tag>/eval_info.json (lerobot-eval) or <out>/eval/<tag>/successes.json
({"successes": [...]}, written by the band evaluation).

    python examples/nanowm_pusht_aug/attribution/paired_report.py <out_dir> --ref a_real_only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_artifact_mask(d: Path) -> np.ndarray | None:
    """Per-start wall-artifact flags (band_eval), aligned with the start-state order; else None."""
    f = d / "successes.json"
    if f.exists():
        m = json.loads(f.read_text()).get("wall_artifact")
        return None if m is None else np.asarray(m, bool)
    return None


def load_successes(d: Path) -> np.ndarray | None:
    if (d / "successes.json").exists():
        j = json.loads((d / "successes.json").read_text())
        # prefer the full per-start array so arms stay aligned; artifacts are masked per comparison
        return np.asarray(j.get("success_all", j["successes"]), bool)
    if (d / "eval_info.json").exists():
        info = json.loads((d / "eval_info.json").read_text())
        return np.asarray([s for t in info["per_task"] for s in t["metrics"]["successes"]], bool)
    return None


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value: binomial test on the discordant pairs."""
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap 95% CI of the mean of an arbitrary per-episode float difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def combined_mask(tags: list[str], art: dict[str, np.ndarray | None]) -> np.ndarray | None:
    """OR of every listed arm's own wall-artifact mask (None if none of them have one)."""
    masks = [art[t] for t in tags if art.get(t) is not None]
    if not masks:
        return None
    out = masks[0].copy()
    for m in masks[1:]:
        out = out | m
    return out


def comparison_row(
    a_tag: str, b_tag: str, arms: dict[str, np.ndarray], art: dict[str, np.ndarray | None]
) -> str:
    """One `| A vs B | diff | CI | McNemar p | discordant |` markdown row. Masking matches the
    original single-pair rule exactly (both sides need their own artifact mask, not just one)."""
    if a_tag not in arms or b_tag not in arms:
        return f"| {a_tag} vs {b_tag} | arm not found | - | - | - |"
    a, b = arms[a_tag], arms[b_tag]
    if len(a) == len(b) and art.get(a_tag) is not None and art.get(b_tag) is not None:
        keep = ~(art[a_tag] | art[b_tag])  # drop a start if EITHER arm hit the wall artifact there
        a, b = a[keep], b[keep]
    if len(a) != len(b):
        return f"| {a_tag} vs {b_tag} | n mismatch {len(a)} vs {len(b)} | - | - | - |"
    lo, hi = paired_bootstrap(a, b)
    return (
        f"| {a_tag} vs {b_tag} | {a.mean() - b.mean():+.1%} | [{lo:+.1%}, {hi:+.1%}] | "
        f"{mcnemar_exact(a, b):.3f} | {int((a & ~b).sum())} / {int((~a & b).sum())} |"
    )


def random_baseline_tables(
    arms: dict[str, np.ndarray], art: dict[str, np.ndarray | None], prefix: str
) -> list[str]:
    """Non-random arm vs the mean of every `prefix`-tagged random arm (per-episode diff, so it stays
    paired), plus a table of the random seeds' pairwise differences to show their own spread."""
    rand_tags = sorted(t for t in arms if prefix in t)
    if len(rand_tags) < 2:
        print(f"--random-prefix {prefix!r} matched {len(rand_tags)} arm(s); need >= 2 for a pooled baseline")
        return []
    lines = [
        "",
        "| arm vs pooled Random | mean diff | paired 95% CI | McNemar p (min-max vs each seed) | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tag in (t for t in arms if t not in rand_tags):
        involved = [tag, *rand_tags]
        if len({len(arms[t]) for t in involved}) != 1:
            lines.append(f"| {tag} | n mismatch across {involved} | - | - | - |")
            continue
        mask = combined_mask(involved, art)
        a = arms[tag][~mask] if mask is not None else arms[tag]
        rs = [arms[t][~mask] if mask is not None else arms[t] for t in rand_tags]
        diff = a.astype(np.float64) - np.mean([r.astype(np.float64) for r in rs], axis=0)
        lo, hi = bootstrap_mean_ci(diff)
        ps = [mcnemar_exact(a, r) for r in rs]
        lines.append(
            f"| {tag} | {diff.mean():+.1%} | [{lo:+.1%}, {hi:+.1%}] | {min(ps):.3f}-{max(ps):.3f} | {len(a)} |"
        )
    lines += [
        "",
        "| random seed pair (A vs B) | A - B | paired 95% CI | McNemar p | discordant (A only / B only) |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        comparison_row(a_tag, b_tag, arms, art)
        for i, a_tag in enumerate(rand_tags)
        for b_tag in rand_tags[i + 1 :]
    ]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--eval-subdir", default="eval", help="per-arm results live in <out_dir>/<this>/<tag>/")
    p.add_argument("--ref", default=None, help="reference arm tag every other arm is compared with")
    p.add_argument("--report", default="report.md")
    p.add_argument(
        "--random-prefix",
        default=None,
        help="arms whose tag contains this form a pooled Random baseline for a paired arm-vs-Random table",
    )
    p.add_argument(
        "--pair", action="append", default=[], metavar="A:B", help="extra explicit comparison; repeatable"
    )
    args = p.parse_args()

    root = args.out_dir / args.eval_subdir
    arms = {d.name: s for d in sorted(root.iterdir()) if d.is_dir() and (s := load_successes(d)) is not None}
    art = {t: load_artifact_mask(root / t) for t in arms}
    lines = ["| arm | success | 95% CI (binomial) | n |", "| --- | --- | --- | --- |"]
    for tag, s in arms.items():
        if art[tag] is not None:
            s = s[~art[tag]]  # each arm's own artifact-free episodes
        m, n = s.mean(), len(s)
        lines.append(f"| {tag} | {m:.1%} | +/-{1.96 * math.sqrt(m * (1 - m) / n):.1%} | {n} |")

    comps = [(t, t.replace("_cupid_", "_random_")) for t in arms if "_cupid_" in t]
    if args.ref in arms:
        comps += [(t, args.ref) for t in arms if t != args.ref]
    lines += ["", "| comparison (A vs B) | A - B | paired 95% CI | McNemar p | discordant (A only / B only) |",
              "| --- | --- | --- | --- | --- |"]  # fmt: skip
    lines += [comparison_row(a_tag, b_tag, arms, art) for a_tag, b_tag in comps if b_tag in arms]

    if args.random_prefix:
        lines += random_baseline_tables(arms, art, args.random_prefix)

    if args.pair:
        lines += ["", "| extra comparison (A vs B) | A - B | paired 95% CI | McNemar p | discordant (A only / B only) |",
                  "| --- | --- | --- | --- | --- |"]  # fmt: skip
        for spec in args.pair:
            a_tag, _, b_tag = spec.partition(":")
            lines.append(comparison_row(a_tag, b_tag, arms, art))

    text = "\n".join(lines) + "\n"
    (args.out_dir / args.report).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
