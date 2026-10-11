#!/usr/bin/env python
"""SAILOR expand S5: pick the imagined branches a SIMULATOR check shows to beat the base policy from their anchor.

Input: K re-label passes of one stage-2 root (label_imagined_branches.py --cont-seed s --out-name ...), query
rounds [0, R]: row r=0 = the base policy from the anchor, row r=R = the branch's first R*5 executed actions
replayed in the simulator, then the base policy. Success time from the anchor: r=0 -> success_step,
r=R -> R*5 + success_step (a success inside the replay is deterministic and counts as R*5). Both sides count a
success only within `select.cap_steps` of the anchor, so neither side gets more time than the other.

  discover   per branch: success time (Mann-Whitney, two-sided) and success rate (Fisher) vs its anchor, BH-FDR over
             all branches; class rescue / more_reliable (rate up) / faster (rate n.s., time down) / slower /
             worse_rate / no_clear_diff. Candidates = rescue | more_reliable | faster, at most `per_anchor` per
             anchor with DISTINCT cand (one cand across sigmas is often the same action sequence), strongest first.
             Writes discovery.json and a replication root (symlinks + a resolved config) for fresh seeds.
  finalize   one-sided test of each candidate's own claim on the replication passes (faster: time less;
             rescue / more_reliable: rate greater), p < replicate.alpha -> selected/ symlinks + selection.json.
             Stops (exit 1) if fewer than select.min_selected pass -- the criterion is never loosened.
  build      real + packed pairs datasets for fine-tune arms A (imagined frames) and B (real frames).

The value model plays no part here (expand_score_v.py only scores).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import fisher_exact, mannwhitneyu

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402

FI = 5
IMPROVED = ("rescue", "more_reliable", "faster")


# ---------------------------------------------------------------------------------------------- pure helpers
def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN -> 1)."""
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    if len(p) == 0:
        return p
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def classify(p0: float, dp: float, q_rate: float, q_time: float, t0: float, t1: float, alpha: float) -> str:
    """Discovery class of one branch (q = BH-adjusted; NaN times = no successes on that side)."""
    if q_rate < alpha:
        if dp > 0:
            return "rescue" if p0 < 0.5 else "more_reliable"
        return "worse_rate"
    if q_time < alpha and not (np.isnan(t0) or np.isnan(t1)):
        return "faster" if t1 < t0 else "slower"
    return "no_clear_diff"


def pick_per_anchor(rows: list[dict], per_anchor: int) -> list[dict]:
    """Improved branches, at most `per_anchor` per anchor with distinct cand, strongest (smallest q) first."""
    out, by_anchor = [], {}
    for r in rows:
        if r["cls"] in IMPROVED:
            by_anchor.setdefault(r["seed"], []).append(r)
    for seed in sorted(by_anchor):
        cands = sorted(by_anchor[seed], key=lambda r: (min(r["q_rate"], r["q_time"]), r["branch"]))
        used = set()
        for r in cands:
            if len(used) >= per_anchor:
                break
            if r["cand"] in used:
                continue
            used.add(r["cand"])
            out.append(r)
    return out


def collect(files: list[Path], cap: int) -> tuple[int, dict, dict]:
    """Per anchor / per branch lists of success time from the anchor (None = no success within cap)."""
    anc, br, r_final = {}, {}, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        rs = set(np.unique(d["r"]).tolist()) - {0}
        assert len(rs) == 1, f"{f}: expected query rounds [0, R], got {sorted(rs | {0})}"
        R = rs.pop()
        assert r_final in (None, R), (r_final, R)
        r_final = R
        for i in range(len(d["y"])):
            ok = bool(d["y"][i] | d["pre_success"][i])
            if int(d["r"][i]) == 0:
                t, rec = int(d["success_step"][i]), anc.setdefault(int(d["seed_id"][i]), [])
            else:
                t = R * FI if d["pre_success"][i] else R * FI + int(d["success_step"][i])
                rec = br.setdefault(
                    str(d["branch"][i]),
                    {
                        "seed": int(d["seed_id"][i]),
                        "cand": int(d["cand"][i]),
                        "sigma": float(d["sigma"][i]),
                        "pre": False,
                        "t": [],
                    },
                )
                rec["pre"] |= bool(d["pre_success"][i])
                rec = rec["t"]
            rec.append(t if ok and t <= cap else None)
    return r_final, anc, br


def branch_path(root: Path, name: str) -> Path:
    """Label `branch` field "sigma0.1/seed0003_cand2.npz" -> <root>/sigma0.1/branches/seed0003_cand2.npz."""
    sg, f = name.split("/")
    return root / sg / "branches" / f


def compare(anchor_t: list, branch_t: list, alt: str = "two-sided") -> dict:
    ta, tb = [t for t in anchor_t if t is not None], [t for t in branch_t if t is not None]
    k0, k1, n0, n1 = len(ta), len(tb), len(anchor_t), len(branch_t)
    rate_alt = {"two-sided": "two-sided", "less": "less", "greater": "greater"}[alt]
    # fisher on [[branch succ, fail], [anchor succ, fail]]: "greater" = branch rate higher
    p_rate = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative=rate_alt)[1]
    p_time = mannwhitneyu(tb, ta, alternative=alt).pvalue if k0 >= 3 and k1 >= 3 else np.nan
    return {
        "p0": k0 / n0,
        "p1": k1 / n1,
        "dp": k1 / n1 - k0 / n0,
        "p_rate": float(p_rate),
        "p_time": float(p_time),
        "t0": float(np.median(ta)) if ta else np.nan,
        "t1": float(np.median(tb)) if tb else np.nan,
        "n0": n0,
        "n1": n1,
    }


# ---------------------------------------------------------------------------------------------- stages
def base_dir(cfg: dict, smoke: bool) -> Path:
    return Path(cfg["out_dir"]).expanduser() / ("smoke" if smoke else "")


def discover(cfg: dict, smoke: bool, root: Path | None, glob: str, out_json: Path | None) -> list[dict]:
    sel = cfg["expand"]["select"]
    base = base_dir(cfg, smoke)
    root = root or base / "stage2"
    files = sorted(root.glob(glob))
    assert files, f"no {glob} under {root}"
    R, anc, br = collect(files, int(sel["cap_steps"]))
    rows = []
    for name, b in sorted(br.items()):
        rows.append(
            {
                "branch": name,
                "seed": b["seed"],
                "cand": b["cand"],
                "sigma": b["sigma"],
                "pre": b["pre"],
                **compare(anc[b["seed"]], b["t"]),
            }
        )
    q_rate, q_time = bh([r["p_rate"] for r in rows]), bh([r["p_time"] for r in rows])
    alpha = float(sel["fdr_q"])
    for r, qr, qt in zip(rows, q_rate, q_time, strict=True):
        r["q_rate"], r["q_time"] = float(qr), float(qt)
        r["cls"] = classify(r["p0"], r["dp"], qr, qt, r["t0"], r["t1"], alpha)
    counts = {
        c: sum(r["cls"] == c for r in rows) for c in (*IMPROVED, "no_clear_diff", "slower", "worse_rate")
    }
    print(
        f"discovery: {len(files)} passes, replay {R * FI} steps, {len(anc)} anchors, {len(rows)} branches: {counts}"
    )
    out_json = out_json or base / "discovery.json"
    out_json.write_text(
        json.dumps(
            {"passes": [f.name for f in files], "replay_steps": R * FI, "counts": counts, "rows": rows},
            indent=1,
            default=float,
        )
    )
    return rows


def prepare_replication(cfg: dict, smoke: bool, cfg_path: Path) -> None:
    base = base_dir(cfg, smoke)
    disc = json.loads((base / "discovery.json").read_text())["rows"]
    cands = pick_per_anchor(disc, int(cfg["expand"]["select"]["per_anchor"]))
    if smoke and not cands:  # smoke has 2 anchors: force 2 rows through so every later stage is exercised
        cands = [{**r, "cls": "faster", "smoke_forced": True} for r in disc[:2]]
        print("smoke: no real candidates, forcing", [r["branch"] for r in cands])
    rep = base / "replicate"
    (rep / "stage2").mkdir(parents=True, exist_ok=True)
    for r in cands:
        src = branch_path(base / "stage2", r["branch"]).resolve()
        dst = branch_path(rep / "stage2", r["branch"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)
    seeds = rep / "seeds_stage2.npz"
    if not seeds.exists():
        seeds.symlink_to((base / "seeds_stage2.npz").resolve())
    rc = load_config(cfg_path, out_dir_copy=False)
    rc["out_dir"] = str(rep)
    (rep / "config_replicate.yaml").write_text(
        "# resolved copy of "
        + str(cfg_path)
        + " with out_dir = this dir\n"
        + yaml.safe_dump(rc, sort_keys=False)
    )
    (base / "candidates.json").write_text(json.dumps(cands, indent=1, default=float))
    print(f"{len(cands)} candidates over {len({r['seed'] for r in cands})} anchors -> {rep}")


def finalize(cfg: dict, smoke: bool) -> None:
    base = base_dir(cfg, smoke)
    ex = cfg["expand"]
    cands = json.loads((base / "candidates.json").read_text())
    files = sorted((base / "replicate" / "stage2").glob("labels_rep_*.npz"))
    assert files, "no replication passes"
    _R, anc, br = collect(files, int(ex["select"]["cap_steps"]))
    alpha = float(ex["replicate"]["alpha"])
    out = []
    for c in cands:
        if c["cls"] == "faster":
            rep = compare(anc[c["seed"]], br[c["branch"]]["t"], alt="less")
            p = rep["p_time"]
        else:  # rescue / more_reliable: the claim is a higher success rate
            rep = compare(anc[c["seed"]], br[c["branch"]]["t"], alt="greater")
            p = rep["p_rate"]
        out.append({**c, "rep": rep, "rep_p": float(p), "replicated": bool(p < alpha)})
    ok = [r for r in out if r["replicated"]]
    if smoke and not ok:  # see prepare_replication: plumbing test only, never in a real run
        ok = [{**r, "smoke_forced": True} for r in out]
    sel_dir = base / "selected"
    sel_dir.mkdir(exist_ok=True)
    for f in sel_dir.iterdir():
        f.unlink()
    for r in ok:  # flat dir for pack_closed_loop: <sigma dir>_<branch file>, like finetune_pool's merged/
        (sel_dir / r["branch"].replace("/", "_")).symlink_to(
            branch_path(base / "stage2", r["branch"]).resolve()
        )
    min_sel = int(ex["smoke"]["min_selected"] if smoke else ex["select"]["min_selected"])
    summary = {
        "n_candidates": len(out),
        "n_selected": len(ok),
        "n_anchors": len({r["seed"] for r in ok}),
        "by_class": {c: sum(r["cls"] == c for r in ok) for c in IMPROVED},
        "by_sigma": {str(s): sum(r["sigma"] == s for r in ok) for s in sorted({r["sigma"] for r in out})},
        "passes": [f.name for f in files],
        "alpha": alpha,
        "rows": out,
    }
    (base / "selection.json").write_text(json.dumps(summary, indent=1, default=float))
    print(
        f"replicated {len(ok)}/{len(out)} over {summary['n_anchors']} anchors {summary['by_class']} -> {sel_dir}"
    )
    if len(ok) < min_sel:
        raise SystemExit(f"only {len(ok)} branches replicated (< {min_sel}); stopping before fine-tune")


def build(cfg: dict, smoke: bool) -> None:
    from build_selected_datasets import build_real_plus_pairs

    from lerobot.utils.constants import HF_LEROBOT_HOME

    ft = cfg["expand"]["finetune"]
    real = ft["real_repo"]
    real_stats = json.loads((HF_LEROBOT_HOME / real / "meta/stats.json").read_text())
    manifest = {}
    for arm, pool in (("img", ft["img_repo"]), ("real", ft["real_frames_repo"])):
        pool = pool + ("_smoke" if smoke else "")
        n_pool = json.loads((HF_LEROBOT_HOME / pool / "meta/info.json").read_text())["total_episodes"]
        repo = f"{pool}_plus_real"
        n = build_real_plus_pairs(real, pool, list(range(n_pool)), repo, real_stats)
        manifest[arm] = {"pool_repo": pool, "repo": repo, "n_pairs": n_pool, "total_episodes": n}
        print(arm, manifest[arm])
    a, b = manifest["img"]["n_pairs"], manifest["real"]["n_pairs"]
    assert a == b, f"arms A / B must hold the same pairs, got {a} vs {b}"
    (base_dir(cfg, smoke) / "datasets.json").write_text(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--root", type=Path, default=None, help="stage-2 root (default <out_dir>/stage2)")
    d.add_argument("--glob", default="labels_v*.npz")
    d.add_argument("--out-json", type=Path, default=None)
    d.add_argument(
        "--no-replication", action="store_true", help="only write discovery.json (consistency checks)"
    )
    sub.add_parser("finalize")
    sub.add_parser("build")
    a = ap.parse_args()
    cfg = load_config(a.config, out_dir_copy=False)
    if a.cmd == "discover":
        discover(cfg, a.smoke, a.root, a.glob, a.out_json)
        if not a.no_replication:
            prepare_replication(cfg, a.smoke, a.config)
    elif a.cmd == "finalize":
        finalize(cfg, a.smoke)
    else:
        build(cfg, a.smoke)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
