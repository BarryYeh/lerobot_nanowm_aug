#!/usr/bin/env python
"""early_anchors_select Step 0: can NanoWM rank the first chunks of the same early anchor? (criteria.md in --out)

  run     For every branch of an existing `rollout.py run` (k candidates per anchor), the simulator rebuilds the exact
          anchor state (env.reset(env_seed) + a[0:t]), executes the branch's first chunk exec_acts[0 : f*fi] (sampled on
          the REAL anchor frame), then the base policy continues closed loop (select_check.na = 10, fresh queue) up to
          the episode's 300-step cap. m continuations per candidate (torch seed select_check.seed + 1000 * m + part).
          Records the sim coverage right after the chunk and the first step (from the anchor) with sim coverage >=
          filter.success_cov (0.9). Resumable per batch of anchors.
  report  W = argmax imagined coverage after the chunk (round f-1) per anchor, R = mean over all k candidates.
          Paired bootstrap over anchors. Writes <out>/report.{json,md}.

    CUDA_VISIBLE_DEVICES=0 uv run --extra training python research/nanowm_aug/proj/pusht/early_anchors/select_check.py \
        run --rollout ~/datastorage/nanowm_pusht_aug/early_anchors/rollout_na10base/f2_n10 --out <o> --m 2
    uv run python .../select_check.py report --out <o>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ea_common import (
    CFG,
    EPISODE_STEPS,
    FI,
    boot_diff,
    boot_idx,
    concat_parts,
    load_rollout,
    load_source,
    make_env,
    part_files,
    replay_to_anchor,
    restricted_steps,
    save_part,
)

from nwaug.attribution.config_io import load_config


def run_batch(
    policy, pre, post, src, d, rows: np.ndarray, n_exec, torch_seed: int, dev: str, na: int, thr: float
) -> dict:
    """Per row: exact anchor replay, exec_acts[0 : n_exec] (an int or one per row), then the base policy (na actions
    per query) until sim coverage >= thr or the 300-step episode cap. A hit inside the replayed actions ends that
    row there."""
    from nwaug_envs.pusht.deploy import policy_act, reset_policy

    n = len(rows)
    n_exec = np.broadcast_to(np.asarray(n_exec, np.int64), (n,))
    envs = [make_env("pixels_agent_pos", 96, EPISODE_STEPS + 10) for _ in range(n)]
    obs, t0 = [], np.zeros(n, np.int64)
    cov_anchor, cov_chunk = np.zeros(n), np.zeros(n)
    hit = np.full(n, -1, np.int64)  # steps from the anchor to sim coverage >= thr, -1 if never within the cap
    for j, b in enumerate(rows):
        i = d["src_idx"][b]
        e = envs[j]
        o, info = replay_to_anchor(e, src, i)
        t0[j], cov_anchor[j] = int(src["anchor_t"][i]), info["coverage"]
        for s in range(int(n_exec[j])):
            o, _, _, _, info = e.step(d["exec_acts"][b, s])
            if info["coverage"] >= thr:
                hit[j] = s + 1
                break
        cov_chunk[j] = info["coverage"]
        obs.append(o["pixels"])
    done = hit >= 0
    reset_policy(policy, torch_seed, na)
    steps = t0 + n_exec
    while not done.all():
        act = policy_act(policy, pre, post, obs, dev)
        for j in range(n):
            if done[j]:
                continue
            o, _, term, trunc, info = envs[j].step(act[j])
            obs[j] = o["pixels"]
            steps[j] += 1
            if info["coverage"] >= thr:
                hit[j], done[j] = steps[j] - t0[j], True
            elif term or trunc or steps[j] >= EPISODE_STEPS:
                done[j] = True
    for e in envs:
        e.close()
    return {"cov_anchor": cov_anchor, "cov_chunk": cov_chunk, "hit": hit, "cap": EPISODE_STEPS - t0}


def cmd_run(a) -> None:
    import torch

    from nwaug.attribution.policy_io import load_policy

    cfg = load_config(CFG)
    seed, na, thr = cfg["select_check"]["seed"], cfg["select_check"]["na"], cfg["filter"]["success_cov"]
    prov, d, _ = load_rollout(a.rollout)
    src = load_source(Path(prov["source"]))
    chunk = prov["f"] * FI
    anchors = np.unique(d["src_idx"])
    if a.n_anchors:
        anchors = anchors[: a.n_anchors]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "parts").mkdir(exist_ok=True)
    meta = {
        "rollout": str(a.rollout),
        "ckpt": prov["ckpt"],
        "k": prov["k"],
        "f": prov["f"],
        "chunk": chunk,
        "m": a.m,
        "anchors": int(len(anchors)),
        "seed": seed,
        "thr": thr,
        "na": na,
    }
    (a.out / "provenance.json").write_text(json.dumps(meta, indent=2))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(Path(prov["ckpt"]), device=dev)
    tic = time.time()
    for p, i0 in enumerate(range(0, len(anchors), a.batch_anchors)):
        dest = a.out / "parts" / f"part{p}.npz"
        if dest.exists():
            continue
        rows = np.flatnonzero(np.isin(d["src_idx"], anchors[i0 : i0 + a.batch_anchors]))
        reps = [
            run_batch(policy, pre, post, src, d, rows, chunk, seed + 1000 * rep + p, dev, na, thr)
            for rep in range(a.m)
        ]
        res = {
            "row": rows,
            "src_idx": d["src_idx"][rows],
            "cand": d["cand"][rows],
            "img_cov": d["cov"][rows],
            "cov_anchor": reps[0]["cov_anchor"],
            "cov_chunk": reps[0]["cov_chunk"],
            "cap": reps[0]["cap"],
            "hit": np.stack([r["hit"] for r in reps], 1),  # (rows, m)
        }
        for r in reps[
            1:
        ]:  # every continuation starts from the same state (coverage readout jitters at ~1e-16)
            assert np.allclose(r["cov_chunk"], res["cov_chunk"], rtol=0, atol=1e-9), "chunk replay drift"
        save_part(dest, **res)
        print(
            f"part {p}: anchors {min(len(anchors), i0 + a.batch_anchors)}/{len(anchors)} "
            f"({time.time() - tic:.0f}s)",
            flush=True,
        )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_report(a) -> None:
    meta = json.loads((a.out / "provenance.json").read_text())
    z = concat_parts(part_files(a.out))
    f = meta["f"]
    rmst = restricted_steps(z["hit"], z["cap"][:, None]).mean(1)  # restricted mean steps, avg over m
    reach = (z["hit"] >= 0).mean(1)
    dcov_sim = z["cov_chunk"] - z["cov_anchor"]
    score = z["img_cov"][
        :, f - 1
    ]  # imagined coverage after the first chunk; anchor term is shared per anchor
    A = np.unique(z["src_idx"])
    per = {
        k: np.zeros(len(A))
        for k in (
            "W_steps",
            "R_steps",
            "O_steps",
            "W_reach",
            "R_reach",
            "W_dcov",
            "R_dcov",
            "O_dcov",
            "Wsim_steps",
            "rho",
            "W20_steps",
            "Wmax_steps",
        )
    }
    score20 = z["img_cov"][:, 2 * f - 1]  # diagnostic: imagined coverage after 2 queries (20 steps at f=2)
    scoremax = z["img_cov"].max(1)  # diagnostic: max imagined coverage over the whole branch
    for ai, s in enumerate(A):
        r = np.flatnonzero(z["src_idx"] == s)
        r = r[np.argsort(z["cand"][r])]
        w = r[np.argmax(score[r])]
        per["W_steps"][ai], per["R_steps"][ai] = rmst[w], rmst[r].mean()
        per["O_steps"][ai] = rmst[r].min()  # oracle by steps (optimistic: picks on continuation noise too)
        per["W_reach"][ai], per["R_reach"][ai] = reach[w], reach[r].mean()
        per["W_dcov"][ai], per["R_dcov"][ai], per["O_dcov"][ai] = (
            dcov_sim[w],
            dcov_sim[r].mean(),
            dcov_sim[r].max(),
        )
        per["Wsim_steps"][ai] = rmst[r[np.argmax(dcov_sim[r])]]  # pick by the TRUE chunk Delta cov
        per["rho"][ai] = spearman(score[r], dcov_sim[r])
        per["W20_steps"][ai] = rmst[r[np.argmax(score20[r])]]
        per["Wmax_steps"][ai] = rmst[r[np.argmax(scoremax[r])]]
    idx = boot_idx(len(A))

    def demean(x, g):  # remove the per-anchor mean so the correlation is within anchors only
        return x - np.array([x[g == q].mean() for q in g])

    def diff(x, y):
        return boot_diff(per[x] - per[y], idx)

    res = {
        "meta": meta,
        "anchors": int(len(A)),
        "candidates": int(len(z["row"])),
        "mean": {k: float(np.nanmean(v)) for k, v in per.items()},
        "W-R_steps": diff("W_steps", "R_steps"),
        "W-R_reach": diff("W_reach", "R_reach"),
        "W-R_dcov": diff("W_dcov", "R_dcov"),
        "Wsim-R_steps": diff("Wsim_steps", "R_steps"),
        "O-R_steps": diff("O_steps", "R_steps"),
        "W20-R_steps": diff("W20_steps", "R_steps"),
        "Wmax-R_steps": diff("Wmax_steps", "R_steps"),
        "rho_median": float(np.nanmedian(per["rho"])),
        "corr_within": float(np.corrcoef(demean(score, z["src_idx"]), demean(dcov_sim, z["src_idx"]))[0, 1]),
    }
    g = res["W-R_steps"]
    res["gate_pass"] = bool(g["mean"] <= -5 and g["ci95"][1] < 0)
    (a.out / "report.json").write_text(json.dumps(res, indent=2))
    m = res["mean"]

    def row(name, k, unit=""):
        q = res[k]
        return f"| {name} | {q['mean']:+.3f}{unit} | [{q['ci95'][0]:+.3f}, {q['ci95'][1]:+.3f}] |"

    md = [
        f"# Step 0 select check: {a.out.name}",
        "",
        f"{len(A)} anchors x {meta['k']} candidates, chunk {meta['chunk']} steps, m = {meta['m']} continuations.",
        "",
        "| arm | steps-to-0.9 (restricted mean) | reach 0.9 | sim Delta cov after chunk |",
        "|---|---|---|---|",
        f"| W (imagined argmax) | {m['W_steps']:.1f} | {m['W_reach']:.3f} | {m['W_dcov']:+.4f} |",
        f"| R (mean of all k) | {m['R_steps']:.1f} | {m['R_reach']:.3f} | {m['R_dcov']:+.4f} |",
        f"| pick by true chunk Delta cov | {m['Wsim_steps']:.1f} | | {m['O_dcov']:+.4f} |",
        f"| oracle by steps | {m['O_steps']:.1f} | | |",
        f"| diag: imagined argmax at 20 steps | {m['W20_steps']:.1f} | | |",
        f"| diag: imagined argmax of branch max | {m['Wmax_steps']:.1f} | | |",
        "",
        "| comparison | mean | 95% CI |",
        "|---|---|---|",
        row("W - R steps (PRIMARY)", "W-R_steps"),
        row("W - R reach 0.9", "W-R_reach"),
        row("W - R sim Delta cov", "W-R_dcov"),
        row("true-Delta-cov pick - R steps", "Wsim-R_steps"),
        row("oracle - R steps", "O-R_steps"),
        row("diag: 20-step pick - R steps", "W20-R_steps"),
        row("diag: branch-max pick - R steps", "Wmax-R_steps"),
        "",
        f"Within-anchor Spearman(imagined, sim Delta cov) median {res['rho_median']:.3f}; "
        f"within-anchor pooled corr {res['corr_within']:.3f}.",
        "",
        f"GATE (W - R steps <= -5 and CI upper < 0): **{'PASS' if res['gate_pass'] else 'FAIL'}**",
    ]
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rollout", type=Path, required=True, help="rollout.py run output dir")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--m", type=int, default=2, help="sim continuations per candidate")
    r.add_argument("--batch-anchors", type=int, default=16)
    r.add_argument("--n-anchors", type=int, default=None, help="first N anchors only (smoke tests)")
    rp = sub.add_parser("report")
    rp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
