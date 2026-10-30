#!/usr/bin/env python
"""early_anchors_select Step 1: pick the best of k noised closed-loop branches per anchor (W) vs a random one (R).

Input: a `rollout.py run --k 4 --inject-sigma 0.1` output (f = 2: one policy query, i.e. one noised chunk, every 2
rounds = 10 steps; 20 rounds = 100 steps). Conventions: round r yields frame t + 5(r+1); query m (m = 0, 1, ...) sees
the frame at t + 10m (m = 0: the REAL anchor frame, m >= 1: imagined frame 2m-1) and executes exec_acts[10m : 10m+10].

  select    peak p = argmax over rounds of the branch's imagined coverage. W = the branch with the highest peak cov
            per anchor (ties -> lowest cand), R = a uniformly random branch (rng wpick.select_seed).
            Experience = queries m = 0 .. p_W // 2 with p_W = W's peak (10m < 5(p_W+1): the actions that lead up to
            W's peak frame), taken from W's branch for arm W and from R's branch for arm R: both arms share the same
            (anchor, m) set, so they differ only in which branch. Above the dose, the (anchor, m) set is water-filled
            over anchors (m drawn uniformly inside an anchor) once and applied to both arms.
  simcheck  Diagnostic gate, never selects or labels: every branch is replayed in sim from the exact anchor state up
            to its peak (exec_acts[0 : 5(p+1)]), then the base policy continues (select_check.na = 10) to sim
            coverage >= 0.9 or the 300-step cap; m continuations (torch seed wpick.sim_seed + 1000 * m + part). Restricted-mean steps-to-0.9 (cap = 300 - t). Gate: W - mean over the k
            branches <= -5 steps AND paired-bootstrap (anchors) 95% CI upper bound < 0.
  pack      local/pusht_<tag>_{W,R}: one ep_len-9 episode per pair, row 0 image = the frame of query m (96px),
            rows 0..8 = exec_acts[10m : 10m+9] (the noised, executed actions) clipped to the action range.
  build     local/pusht_<tag>_{W,R}_natural = local/pusht_imgonly + the pairs once, stats pinned to the real data.
  report    Fine-tune evals (eval_success_steps.py: cov09_step) of W, R, ctrl x seeds and the base, same 2000
            episodes. Restricted-mean steps-to-0.9 (never -> 300), pooled over seeds, paired bootstrap over episodes.

    uv run --extra training python research/nanowm_aug/proj/pusht/early_anchors/wpick.py select --rollout <r>
    CUDA_VISIBLE_DEVICES=0 uv run --extra training python .../wpick.py simcheck --rollout <r> --m 2
    uv run --extra training python .../wpick.py pack --rollout <r> --tag <tag> --overwrite
    uv run --extra training python .../wpick.py build --tag <tag>
    uv run python .../wpick.py report --eval-dir <d> --seeds 1000 1001
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ea_common import (
    CFG,
    EP_LEN,
    EPISODE_STEPS,
    FI,
    add_pair,
    boot_diff,
    boot_idx,
    build_mix,
    concat_parts,
    create_pair_repo,
    load_rollout,
    load_source,
    n_episodes,
    part_files,
    real_sampleable,
    restricted_steps,
    save_part,
)

from nwaug.attribution.config_io import load_config
from nwaug.datagen.selection import water_fill

ARMS = ("W", "R")


def picks(d: dict, k: int, select_seed: int) -> dict:
    """Per anchor: its k branch indices (by cand), W and R. Deterministic given the rollout."""
    peak = d["cov"].argmax(1)
    pcov = d["cov"].max(1)
    rng = np.random.default_rng(select_seed)
    out = {}
    for s in np.unique(d["src_idx"]):
        bs = np.flatnonzero(d["src_idx"] == s)
        bs = bs[np.argsort(d["cand"][bs])]
        assert len(bs) == k, (s, len(bs))
        out[int(s)] = {"branches": bs, "W": int(bs[np.argmax(pcov[bs])]), "R": int(bs[rng.integers(k)])}
    return {"anchors": out, "peak": peak, "pcov": pcov}


# --------------------------------------------------------------------------------------------------------- select
def cmd_select(a) -> None:
    cfg = load_config(CFG)
    seed = cfg["wpick"]["select_seed"]
    prov, d, _ = load_rollout(a.rollout, keys=("src_idx", "cand", "cov"))
    assert prov["f"] == 2 and prov.get("inject_sigma", 0) > 0, "expects the noised f=2 rollout"
    pk = picks(d, prov["k"], seed)
    peak, pcov = pk["peak"], pk["pcov"]
    cand = {
        s: list(range(int(peak[v["W"]]) // 2 + 1)) for s, v in pk["anchors"].items()
    }  # shared (anchor, m)
    tot = sum(len(m) for m in cand.values())
    n = min(a.n or round(cfg["pack"]["dose"] * real_sampleable()), tot)
    rng = np.random.default_rng(seed + 1)
    cap = {s: len(m) for s, m in cand.items()}
    quota = water_fill(cap, n, rng) if n < tot else cap
    keep = {s: sorted(rng.choice(cand[s], size=quota[s], replace=False).tolist()) for s in sorted(cand)}
    rows = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for s in sorted(keep):
            b = pk["anchors"][s][arm]
            for m in keep[s]:
                rows[arm].append(
                    {
                        "anchor": s,
                        "branch": b,
                        "cand": int(d["cand"][b]),
                        "part": int(d["part"][b]),
                        "row": int(d["row"][b]),
                        "m": int(m),
                        "peak": int(peak[b]),
                        "peak_cov": round(float(pcov[b]), 4),
                    }
                )
    A = pk["anchors"]
    meta = {
        "rollout": str(a.rollout),
        "anchors": len(A),
        "n_per_arm": n,
        "available": tot,
        "R_m_beyond_own_peak": int(sum(r["m"] > r["peak"] // 2 for r in rows["R"])),
        "W_eq_R": int(sum(v["W"] == v["R"] for v in A.values())),
        "peak_cov_mean": {arm: float(np.mean([pcov[v[arm]] for v in A.values()])) for arm in ARMS},
        "peak_round_median": {arm: float(np.median([peak[v[arm]] for v in A.values()])) for arm in ARMS},
        "m_hist": {arm: np.bincount([r["m"] for r in rows[arm]], minlength=10).tolist() for arm in ARMS},
        "pairs": rows,
    }
    out = a.rollout / "wpick"
    out.mkdir(exist_ok=True)
    (out / "selection.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "pairs"}, indent=1))


# ------------------------------------------------------------------------------------------------------- simcheck
def cmd_simcheck(a) -> None:
    import torch
    from select_check import run_batch

    from nwaug.attribution.policy_io import load_policy

    cfg = load_config(CFG)
    sim_seed, na, thr = cfg["wpick"]["sim_seed"], cfg["select_check"]["na"], cfg["filter"]["success_cov"]
    prov, d, _ = load_rollout(a.rollout)
    src = load_source(Path(prov["source"]))
    pk = picks(d, prov["k"], cfg["wpick"]["select_seed"])
    anchors = sorted(pk["anchors"])[: a.n_anchors] if a.n_anchors else sorted(pk["anchors"])
    out = a.rollout / "wpick" / "simcheck"
    (out / "parts").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(Path(prov["ckpt"]), device=dev)
    tic = time.time()
    for p, i0 in enumerate(range(0, len(anchors), a.batch_anchors)):
        dest = out / "parts" / f"part{p}.npz"
        if dest.exists():
            continue
        rows = np.concatenate([pk["anchors"][s]["branches"] for s in anchors[i0 : i0 + a.batch_anchors]])
        n_exec = FI * (pk["peak"][rows] + 1)
        reps = [
            run_batch(policy, pre, post, src, d, rows, n_exec, sim_seed + 1000 * r + p, dev, na, thr)
            for r in range(a.m)
        ]
        save_part(dest, row=rows, cap=reps[0]["cap"], hit=np.stack([r["hit"] for r in reps], 1))
        print(
            f"simcheck part {p}: anchors {min(len(anchors), i0 + a.batch_anchors)}/{len(anchors)} "
            f"({time.time() - tic:.0f}s)",
            flush=True,
        )
    z = concat_parts(part_files(out), keys=("row", "cap", "hit"))
    steps = dict(zip(z["row"].tolist(), restricted_steps(z["hit"], z["cap"][:, None]).mean(1), strict=True))
    reach = dict(zip(z["row"].tolist(), (z["hit"] >= 0).mean(1), strict=True))
    done = [s for s in anchors if all(int(b) in steps for b in pk["anchors"][s]["branches"])]
    per = {
        k: np.array([f(pk["anchors"][s]) for s in done])
        for k, f in {
            "W": lambda v: steps[v["W"]],
            "R": lambda v: steps[v["R"]],
            "Rbar": lambda v: np.mean([steps[int(b)] for b in v["branches"]]),
            "best": lambda v: min(steps[int(b)] for b in v["branches"]),
            "W_reach": lambda v: reach[v["W"]],
            "Rbar_reach": lambda v: np.mean([reach[int(b)] for b in v["branches"]]),
        }.items()
    }
    idx = boot_idx(len(done))

    def diff(x, y):
        return boot_diff(per[x] - per[y], idx)

    res = {
        "anchors": len(done),
        "m": a.m,
        "mean": {k: float(v.mean()) for k, v in per.items()},
        "W-Rbar": diff("W", "Rbar"),
        "W-R": diff("W", "R"),
        "best-Rbar": diff("best", "Rbar"),
        "W-Rbar_reach": diff("W_reach", "Rbar_reach"),
    }
    g = res["W-Rbar"]
    res["gate_pass"] = bool(g["mean"] <= -5 and g["ci95"][1] < 0)
    (out / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ----------------------------------------------------------------------------------------------------------- pack
def cmd_pack(a) -> None:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    sel = json.loads((a.rollout / "wpick" / "selection.json").read_text())["pairs"]
    _, _, parts = load_rollout(a.rollout, keys=("cov",))
    cache = {}
    for arm in ARMS:
        repo = f"local/pusht_{a.tag}_{arm}"
        ds = create_pair_repo(repo, a.overwrite)
        index = []
        for e, r in enumerate(sel[arm]):
            if r["part"] not in cache:
                z = np.load(parts[r["part"]])
                cache[r["part"]] = {k: z[k] for k in ("frames96", "exec_acts", "anchor_frame96")}
            z, m = cache[r["part"]], r["m"]
            frame = z["anchor_frame96"][r["row"]] if m == 0 else z["frames96"][r["row"], 2 * m - 1]
            add_pair(ds, frame, z["exec_acts"][r["row"], 10 * m : 10 * m + EP_LEN])
            index.append({"episode_index": e, **r})
        ds.finalize()
        (HF_LEROBOT_HOME / repo / "meta" / "pair_index.json").write_text(
            json.dumps({"arm": arm, "pairs": index}, indent=1)
        )
        print(f"packed {len(index)} pairs into {repo}")


# ---------------------------------------------------------------------------------------------------------- build
def cmd_build(a) -> None:
    for arm in ARMS:
        pool, repo = f"local/pusht_{a.tag}_{arm}", f"local/pusht_{a.tag}_{arm}_natural"
        build_mix(pool, repo, rep=1)
        n_pool = n_episodes(pool)
        print(
            f"{repo}: {n_pool} pairs + real; pair share of samples {n_pool / (n_pool + real_sampleable()):.1%}"
        )


# --------------------------------------------------------------------------------------------------------- report
def cmd_report(a) -> None:
    """Two thresholds on the same evals: cov 0.9 (cov09_step, the pre-registered metric) and cov 0.95 (the eval's own
    success, success_step). Per threshold: restricted-mean steps (never -> 300) and the rate of reaching it."""
    E, S = a.eval_dir, a.seeds
    names = {"base": dict.fromkeys(S, "base")}
    for arm in ("W", "R", "ctrl"):
        if all((E / f"{arm}_s{s}").is_dir() for s in S):
            names[arm] = {s: f"{arm}_s{s}" for s in S}
    z = {nm: np.load(E / nm / "success_steps.npz") for v in names.values() for nm in v.values()}
    n = len(z["base"]["cov09_step"])
    idx = boot_idx(n)
    md = [
        "# early_anchors_select Step 1: steps to coverage 0.9 and 0.95",
        "",
        f"{n} paired episodes, seeds {S}. Restricted mean: never reaching the threshold counts as {EPISODE_STEPS}.",
        "Coverage 0.9 is the pre-registered metric; 0.95 (the eval's success) was added after the result.",
    ]
    res = {}
    for thr, key in (("0.9", "cov09_step"), ("0.95", "success_step")):

        def get(nm, key=key):
            c = z[nm][key]
            return np.where(c > 0, c, EPISODE_STEPS).astype(float), (c > 0).astype(float)

        arms = {k: {s: get(v[s]) for s in S} for k, v in names.items()}
        r = res[thr] = {"arms": {}, "pairs": {}}
        md += ["", f"## coverage >= {thr}", "", f"| arm | steps-to-{thr} | reach {thr} |", "|---|---|---|"]
        for k, v in arms.items():
            st, rc = np.mean([v[s][0].mean() for s in S]), np.mean([v[s][1].mean() for s in S])
            r["arms"][k] = {"steps": float(st), "reach": float(rc)}
            md.append(f"| {k} | {st:.1f} | {rc:.3f} |")
        md += ["", "| comparison | steps (pooled) | 95% CI | reach (pp) | 95% CI |", "|---|---|---|---|---|"]
        for x, y in [("W", "R"), ("W", "ctrl"), ("R", "ctrl"), ("W", "base"), ("ctrl", "base")]:
            if x not in arms or y not in arms:
                continue
            row = [f"{x} - {y}"]
            for q in (0, 1):
                dd = np.mean([arms[x][s][q] - arms[y][s][q] for s in S], 0)
                lo, hi = np.percentile(dd[idx].mean(1), [2.5, 97.5])
                sc = 100 if q else 1
                r["pairs"].setdefault(f"{x}-{y}", {})["steps" if q == 0 else "reach"] = {
                    "mean": float(dd.mean() * sc),
                    "ci95": [float(lo * sc), float(hi * sc)],
                }
                row += [f"{dd.mean() * sc:+.2f}", f"[{lo * sc:+.2f}, {hi * sc:+.2f}]"]
            md.append("| " + " | ".join(row) + " |")
    (E / "report_wpick.json").write_text(json.dumps(res, indent=2))
    (E / "report_wpick.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--rollout", type=Path, required=True)
    s.add_argument("--n", type=int, default=None, help="pairs per arm (default: pack.dose x real sampleable)")
    c = sub.add_parser("simcheck")
    c.add_argument("--rollout", type=Path, required=True)
    c.add_argument("--m", type=int, default=2, help="sim continuations per branch")
    c.add_argument("--batch-anchors", type=int, default=64)
    c.add_argument("--n-anchors", type=int, default=None, help="first N anchors only (smoke tests)")
    p = sub.add_parser("pack")
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--tag", required=True)
    r = sub.add_parser("report")
    r.add_argument("--eval-dir", type=Path, required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    a = ap.parse_args()
    {
        "select": cmd_select,
        "simcheck": cmd_simcheck,
        "pack": cmd_pack,
        "build": cmd_build,
        "report": cmd_report,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
