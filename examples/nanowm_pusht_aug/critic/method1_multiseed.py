#!/usr/bin/env python
"""Method 1 offline check: NanoWM seed spread on the last-mile candidates (DIAGNOSTIC ONLY, no training data).

Each accepted start's 12 candidates are re-imagined with exactly the worker inputs of the original last-mile run
(<lastmile>/imag/sigma*/work: seeds_expanded.npz = 3 real context frames, chunks_{0,1}.npz = the action windows the
worker conditioned on), replicated S times in one worker batch with --seed (every replica draws other noise rows,
i.e. S WM seeds; same mechanism as eval/wm_seed_diversity.py). Only the WM noise changes.

  run      one sigma (optionally one part of its rows) -> <out>/multiseed/ms_sigma<X>_p<k>.npz: sid, cand, cov (n,S,2)
  analyze  joins the parts with <out>/sim_truth.json (simulator replay of every candidate, 1 and 2 rounds) and writes
           <out>/multiseed/multiseed.json / .md: seed spread, success-call precision/recall and W true success for
           single-draw vs seed-mean vs seed-lower-quartile scores, spread vs error.

    CUDA_VISIBLE_DEVICES=6 uv run --extra training python examples/nanowm_pusht_aug/critic/method1_multiseed.py run \
        --out <method1 dir> --sigma 0.1 --part 0 --nparts 1
    uv run --extra training python examples/nanowm_pusht_aug/critic/method1_multiseed.py analyze --out <method1 dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent), str(_HERE.parent / "eval")]

LASTMILE = Path.home() / "datastorage/nanowm_pusht_aug/lastmile"
SIGMAS = ("0.0", "0.1", "0.25")
THR = 0.95


def accepted_starts() -> list[int]:
    sel = json.loads((LASTMILE / "sel/selection.json").read_text())
    return sorted(r["start"] for r in sel["rows"] if r["accepted"])


def run(a) -> None:
    from wm_fidelity_ood import run_worker

    w = LASTMILE / "imag" / f"sigma{a.sigma}" / "work"
    res = np.load(w / "cl_result.npz")
    acc = set(accepted_starts())
    rows = np.flatnonzero([int(s) in acc for s in res["sid"]])
    rows = rows[a.part :: a.nparts]
    if a.limit:
        rows = rows[: a.limit]
    ctx = np.load(w / "seeds_expanded.npz")["frames_224"][rows]
    win = np.stack([np.load(w / f"chunks_{r}.npz")["win"][rows] for r in (0, 1)], 1)  # (n, 2, NF, 2FI)
    S = a.n_seeds
    work = a.out / "multiseed" / f"work_sigma{a.sigma}_p{a.part}"
    print(f"sigma {a.sigma} part {a.part}/{a.nparts}: {len(rows)} candidates x {S} seeds")
    run_worker(
        work,
        np.repeat(ctx, S, axis=0),  # candidate-major: row i*S + s
        np.repeat(win, S, axis=0),
        SimpleNamespace(num_sampling_steps=a.num_sampling_steps, wm_batch=a.wm_batch, wm_seed=a.wm_seed),
    )
    cov = np.stack([np.load(work / f"frames_{r}.npz")["cov"] for r in (1, 2)], 1).reshape(len(rows), S, 2)
    np.savez(
        a.out / "multiseed" / f"ms_sigma{a.sigma}_p{a.part}.npz",
        sid=res["sid"][rows],
        cand=res["cand"][rows],
        cov=cov.astype(np.float32),
        n_seeds=np.int64(S),
        wm_seed=np.int64(a.wm_seed),
    )
    for f in work.glob("frames_*.npz"):  # frames are large and not needed after the coverage is read
        f.unlink()


def analyze(a) -> None:
    truth = {}
    for r in json.loads((a.out / "sim_truth.json").read_text())["rows"]:
        sg, rest = r["name"].split("_", 1)  # sigma0.1 / seed0001_cand1.npz
        truth[sg[5:], int(rest[4:8]), int(rest.split("cand")[1][:-4])] = r
    keys, ms = [], []
    for sg in SIGMAS:
        for f in sorted((a.out / "multiseed").glob(f"ms_sigma{sg}_p*.npz")):
            z = np.load(f)
            for s, c, v in zip(z["sid"], z["cand"], z["cov"], strict=True):
                keys.append((sg, int(s), int(c)))
                ms.append(v)
    assert len(keys) == len(truth) == len(set(keys)), (len(keys), len(truth))
    ms = np.stack(ms)  # (N, S, 2)
    T = [truth[k] for k in keys]
    y = np.array([t["term2"] for t in T])  # true success within the 10 executed steps
    single = np.array([max(t["imag"]) for t in T])  # the original last-mile score (one WM draw)
    sim = np.array([[t["sim1"], t["sim2"]] for t in T])
    live = np.array([[not t["term1"], not t["term2"]] for t in T])  # sim value is AT that step only if still running
    start = np.array([k[1] for k in keys])
    W_orig = np.array([t["W"] for t in T])
    scores = {
        "single draw (current)": single,
        "seed mean": ms.mean(1).max(1),
        "seed lower quartile": np.percentile(ms, 25, axis=1).max(1),
        "share of seeds >= 0.95": (ms.max(2) >= THR).mean(1),
    }

    def call(score: np.ndarray, thr: float) -> dict:
        pred = score >= thr
        tp = int((pred & y).sum())
        return {
            "precision": tp / max(int(pred.sum()), 1),
            "recall": tp / max(int(y.sum()), 1),
            "called": float(pred.mean()),
        }

    def w_success(score: np.ndarray) -> float:
        ok = []
        for s in np.unique(start):
            i = np.flatnonzero(start == s)
            ok.append(y[i[np.argmax(score[i])]])  # ties -> first in (sigma, cand) order, as select_lastmile
        return float(np.mean(ok))

    rep = {"n_candidates": len(keys), "n_seeds": int(ms.shape[1]), "true_success_rate": float(y.mean())}
    sd = ms.std(1)  # (N, 2)
    rep["seed_sd"] = {
        f"round{r + 1}": {"mean": float(sd[:, r].mean()), "median": float(np.median(sd[:, r]))} for r in (0, 1)
    }
    rep["calls"] = {}
    for name, sc in scores.items():
        thr = 0.5 if name.startswith("share") else THR
        rep["calls"][name] = {**call(sc, thr), "thr": thr, "W_true_success": w_success(sc)}
    rep["W_true_success_reference"] = {
        "original W": float(y[W_orig].mean()),
        "R (random)": float(y[np.array([t["R"] for t in T])].mean()),
    }
    from scipy.stats import spearmanr

    spread_err = {}
    for r in (0, 1):
        m = live[:, r]
        err = np.abs(ms[m, :, r].mean(1) - sim[m, r])
        spread_err[f"round{r + 1}"] = {
            "n": int(m.sum()),
            "spearman_sd_vs_abs_err": float(spearmanr(sd[m, r], err).statistic),
            "abs_err_seed_mean": float(err.mean()),
            "abs_err_single_draw": float(np.abs(np.array([t["imag"][r] for t in T])[m] - sim[m, r]).mean()),
        }
    rep["spread_vs_error"] = spread_err
    (a.out / "multiseed" / "multiseed.json").write_text(json.dumps(rep, indent=1))
    lines = [
        f"# NanoWM seed spread on {len(keys)} last-mile candidates ({ms.shape[1]} seeds each)",
        "",
        f"True success within 10 steps: {y.mean():.3f}. Seed sd of coverage: round 1 mean "
        f"{rep['seed_sd']['round1']['mean']:.3f}, round 2 mean {rep['seed_sd']['round2']['mean']:.3f}.",
        "",
        "| score | threshold | precision | recall | share called | W true success |",
        "|---|---|---|---|---|---|",
    ]
    for name, v in rep["calls"].items():
        lines.append(
            f"| {name} | {v['thr']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['called']:.2f} | "
            f"{v['W_true_success']:.3f} |"
        )
    ref = rep["W_true_success_reference"]
    lines += [
        "",
        f"Reference: original W {ref['original W']:.3f}, R {ref['R (random)']:.3f}.",
        "",
        "| round | n | Spearman(seed sd, abs err) | abs err, seed mean | abs err, single draw |",
        "|---|---|---|---|---|",
    ]
    for r, v in spread_err.items():
        lines.append(
            f"| {r} | {v['n']} | {v['spearman_sd_vs_abs_err']:.2f} | {v['abs_err_seed_mean']:.3f} | "
            f"{v['abs_err_single_draw']:.3f} |"
        )
    (a.out / "multiseed" / "multiseed.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--out", type=Path, required=True, help="the method-1 dir")
    ap.add_argument("--sigma", choices=SIGMAS)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="SMOKE ONLY: first N candidates of the part")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--wm-seed", type=int, default=0)
    ap.add_argument("--num-sampling-steps", type=int, default=30)
    ap.add_argument("--wm-batch", type=int, default=24)
    a = ap.parse_args()
    (a.out / "multiseed").mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        assert a.sigma is not None
        run(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
