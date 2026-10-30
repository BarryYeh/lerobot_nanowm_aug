#!/usr/bin/env python
"""early_anchors steps 1 and 3: deploy a policy from env.reset(seed), find each episode's early anchor, plot coverage.

  source   Deploy the policy (no noise, na = --na, default eval.n_action_steps) from env.reset(seed + i), i < n,
           and record the executed actions plus the image coverage estimate (coverage_from_frame()["coverage_cal"])
           on the 5-step grid. Each episode's anchor is the FIRST grid point passing the `anchor:` rule of
           configs/early_anchors.yaml (see `first_anchor`). Writes <out>/episodes.npz and <out>/summary.json (anchor
           counts and the anchor -> termination step distribution, also as closed-loop iterations
           ceil(steps / (f * fi)) for each f).
  plot     Per-episode coverage-vs-timestep panels (success / failure, anchor region, anchor, termination) and an
           overview page (all curves + anchor-time and anchor->termination histograms).

The source episodes come from `nwaug_envs.pusht.deploy.source_batch` (the env pair, lockstep check and policy reset
imagined_replay uses too).

    CUDA_VISIBLE_DEVICES=6 uv run --extra training python research/nanowm_aug/proj/pusht/early_anchors/anchors.py \
        source --stage define --out ~/datastorage/nanowm_pusht_aug/early_anchors/define
    uv run --extra training python .../anchors.py source --stage source --ckpt <base> --na 10 --out <dir>/source_na10
    uv run --extra training python .../anchors.py plot --out ~/datastorage/nanowm_pusht_aug/early_anchors/define
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from ea_common import CFG, FI

from nwaug.attribution.config_io import load_config
from nwaug.datagen.selection import qualifying


def first_anchor(steps: np.ndarray, est: np.ndarray, success_step: int, rule: dict) -> int:
    """Grid index of the episode's first anchor, or -1. Grid points past t_max are masked out before `qualifying`."""
    masked = np.where(steps <= rule["t_max"], est, -1.0)
    ok = qualifying(
        steps,
        masked,
        success_step,
        t_min=rule["t_min"],
        cov_lo=rule["cov_lo"],
        cov_hi=rule["cov_hi"],
        stall=rule["stall"],
        per_episode=len(steps),
        gap=0,
        rng=np.random.default_rng(0),
    )
    return min(ok) if ok else -1


def summarize(d: dict, fs: list[int]) -> dict:
    has = d["anchor_t"] >= 0
    to_end = (d["end"] - d["anchor_t"])[has]
    succ_from_anchor = d["success"][has] > 0

    def stats(x: np.ndarray) -> dict:
        if len(x) == 0:
            return {}
        q = np.percentile(x, [10, 25, 50, 75, 90])
        return {"mean": float(x.mean()), "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4]}

    out = {
        "episodes": int(len(d["env_seed"])),
        "success_rate": float((d["success"] > 0).mean()),
        "with_anchor": int(has.sum()),
        "anchor_t": stats(d["anchor_t"][has].astype(float)),
        "anchor_to_termination_steps": {
            "all": stats(to_end.astype(float)),
            "success": stats(to_end[succ_from_anchor].astype(float)),
            "failure": stats(to_end[~succ_from_anchor].astype(float)),
        },
        "success_rate_given_anchor": float(succ_from_anchor.mean()) if has.any() else None,
        "iterations": {},
    }
    for f in fs:
        it = np.ceil(to_end / (f * FI))
        out["iterations"][f"f{f}"] = {
            "all": stats(it),
            "success": stats(it[succ_from_anchor]),
        }
    return out


def cmd_source(a) -> None:
    import torch

    from nwaug.attribution.policy_io import load_policy
    from nwaug_envs.pusht.coverage import coverage_from_frame
    from nwaug_envs.pusht.deploy import EPISODE_STEPS, source_batch

    cfg = load_config(CFG)
    na = a.na or cfg["eval"]["n_action_steps"]
    rule, stage = cfg["anchor"], cfg[a.stage]
    ckpt = Path(a.ckpt) if a.ckpt else Path(cfg["ckpt"][stage["ckpt_key"]])
    n = a.n or stage["n"]
    seed0 = stage["seed"]
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "episodes.npz"
    if dest.exists():
        raise SystemExit(f"{dest} exists; delete it to recollect")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(ckpt, device=dev)
    rows, t0 = [], time.time()
    for b0 in range(0, n, a.batch):
        seeds = list(range(seed0 + b0, seed0 + min(n, b0 + a.batch)))
        res = source_batch(policy, pre, post, seeds, dev, torch_seed=seed0 + b0, na=na)
        steps = res["steps"]
        for j, s in enumerate(seeds):
            last = int(res["end"][j]) // FI + 1  # grid points that exist
            est = np.full(len(steps), -1.0)
            for g in range(last):
                est[g] = coverage_from_frame(res["frames"][j, g])["coverage_cal"]
            ss = int(res["success"][j])
            g = first_anchor(steps, est, ss, rule)
            rows.append(
                {
                    "env_seed": s,
                    "est": est,
                    "acts": res["acts"][j],
                    "success": ss,
                    "end": int(res["end"][j]),
                    "anchor_g": g,
                    "anchor_t": int(steps[g]) if g >= 0 else -1,
                }
            )
        print(f"  {len(rows)}/{n} episodes ({time.time() - t0:.0f}s)", flush=True)
    d = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    prov = {
        "stage": a.stage,
        "ckpt": str(ckpt),
        "seed": seed0,
        "rule": rule,
        "na": na,
        "episode_steps": EPISODE_STEPS,
    }
    np.savez(dest, steps=np.arange(0, EPISODE_STEPS + 1, FI), provenance_json=json.dumps(prov), **d)
    summ = {"provenance": prov, **summarize(d, cfg["rollout"]["f"])}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))


def _region(ax, rule) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (rule["t_min"], rule["cov_lo"]),
            rule["t_max"] - rule["t_min"],
            rule["cov_hi"] - rule["cov_lo"],
            color="#e8e6df",
            lw=0,
            zorder=0,
        )
    )


def _panel(ax, steps, est, success, end, anchor_t, rule, thr_filter, thr_eval) -> None:
    _region(ax, rule)
    ax.axhline(thr_eval, color="#8a8984", lw=0.6, ls=":")
    ax.axhline(thr_filter, color="#8a8984", lw=0.6, ls="--")
    ok = est >= 0
    ok_style = {"color": "#2a78d6", "lw": 1.6} if success > 0 else {"color": "#8a8984", "lw": 1.4, "ls": "--"}
    ax.plot(steps[ok], est[ok], **ok_style)
    if success > 0:
        ax.axvline(end, color="#2a78d6", lw=0.6, ls=":")
    if anchor_t >= 0:
        g = anchor_t // FI
        ax.plot([anchor_t], [est[g]], "o", ms=7, color="#eb6834", mec="white", mew=1.5, zorder=5)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.02)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def cmd_plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = load_config(CFG, out_dir_copy=False)
    d = dict(np.load(a.out / "episodes.npz"))
    prov = json.loads(str(d["provenance_json"]))
    rule = prov["rule"]
    thr_f, thr_e = cfg["filter"]["success_cov"], cfg["pusht"]["success_threshold"]
    steps = d["steps"]
    n = len(d["env_seed"])
    legend = [
        Line2D([], [], color="#2a78d6", lw=1.6, label="success episode"),
        Line2D([], [], color="#8a8984", lw=1.4, ls="--", label="failure episode"),
        Line2D([], [], marker="o", ls="", ms=7, color="#eb6834", mec="white", label="early anchor"),
        matplotlib.patches.Patch(color="#e8e6df", label="anchor region"),
        Line2D([], [], color="#8a8984", lw=0.6, ls="--", label=f"filter cov {thr_f}"),
        Line2D([], [], color="#8a8984", lw=0.6, ls=":", label=f"eval cov {thr_e}"),
    ]
    per_page, cols = a.per_page, 5
    rows = math.ceil(per_page / cols)
    for p0 in range(0, n, per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.3 * rows), sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            i = p0 + k
            if i >= n:
                ax.axis("off")
                continue
            _panel(ax, steps, d["est"][i], d["success"][i], d["end"][i], d["anchor_t"][i], rule, thr_f, thr_e)
            tag = f"success @{d['success'][i]}" if d["success"][i] > 0 else "failure"
            anc = f", anchor t={d['anchor_t'][i]}" if d["anchor_t"][i] >= 0 else ", no anchor"
            ax.set_title(f"seed {d['env_seed'][i]}: {tag}{anc}", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("timestep", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("image coverage", fontsize=8)
        fig.legend(handles=legend, loc="upper center", ncol=6, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        dest = a.out / f"episodes_{p0:04d}.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        print(f"wrote {dest}")

    has = d["anchor_t"] >= 0
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axes[0]
    _region(ax, rule)
    for i in range(n):
        ok = d["est"][i] >= 0
        succ = d["success"][i] > 0
        ax.plot(steps[ok], d["est"][i][ok], color="#2a78d6" if succ else "#8a8984", lw=0.6, alpha=0.35)
    ax.scatter(d["anchor_t"][has], d["est"][has, d["anchor_g"][has]], s=14, color="#eb6834", zorder=5)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("timestep")
    ax.set_ylabel("image coverage")
    ax.set_title(f"{n} episodes, success {(d['success'] > 0).mean():.0%}, anchors {has.sum()}", fontsize=10)
    axes[1].hist(d["anchor_t"][has], bins=np.arange(0, rule["t_max"] + 10, 10), color="#eb6834", rwidth=0.9)
    axes[1].set_xlabel("anchor timestep")
    axes[1].set_ylabel("episodes")
    axes[1].set_title("anchor time", fontsize=10)
    to_end = (d["end"] - d["anchor_t"])[has]
    succ = d["success"][has] > 0
    bins = np.arange(0, 310, 10)
    axes[2].hist(
        [to_end[succ], to_end[~succ]],
        bins=bins,
        stacked=True,
        color=["#2a78d6", "#b4b2a9"],
        label=["ends in success", "hits the 300-step limit"],
        rwidth=0.9,
    )
    axes[2].set_xlabel("steps from anchor to termination")
    axes[2].set_ylabel("anchors")
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_title(
        f"median {np.median(to_end):.0f} steps (success only {np.median(to_end[succ]) if succ.any() else float('nan'):.0f})",
        fontsize=10,
    )
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    dest = a.out / "overview.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source")
    s.add_argument("--stage", choices=["define", "source", "heldout"], required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--ckpt", default=None, help="policy checkpoint (default: cfg.ckpt[<stage>.ckpt_key])")
    s.add_argument("--n", type=int, default=None, help="episodes (default: <stage>.n)")
    s.add_argument("--na", type=int, default=None, help="n_action_steps (default: eval.n_action_steps)")
    s.add_argument("--batch", type=int, default=50)
    p = sub.add_parser("plot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-page", type=int, default=20)
    a = ap.parse_args()
    {"source": cmd_source, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
