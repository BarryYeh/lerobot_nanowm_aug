#!/usr/bin/env python
"""early_anchors steps 4-5: closed-loop NanoWM imagination from the early anchors, then a sim-replay fidelity check.

  run       From each anchor of a `anchors.py source` run, the policy runs closed-loop inside NanoWM.
            Context: the real frames F(t-10), F(t-5), F(t) and the real actions a[t-10 : t], all from an exact
            simulator replay of the source episode (env.reset(env_seed) + a[0 : t]); the agent position and
            velocity before each of those steps come from the same replay.
            One NanoWM round = fi (5) sim steps. A policy query happens every f rounds and executes f * fi
            actions of its chunk (`chunk_slices` with R = f); the first query sees the real frame F(t), every
            later one the branch's own imagined frame. n_iter queries -> n_iter * f rounds.
            --inject-sigma > 0 adds local action noise (`closed_loop_rollout.inject_noise`, NanoWM sigma units, z
            redrawn every --noise-hold steps, capped at --max-z) to EVERY sampled chunk, before it is executed and
            imagined; exec_acts then hold the noised actions. 0 (default) = the original, bit-identical rollout.
            The NanoWM call, the window alignment and the PD agent track are those of
            nwaug/wm/closed_loop_rollout.py (see its ALIGNMENT CONTRACT); NanoWM runs in its own venv as the
            `imagined_rollout.py closed-loop` worker. Anchors are processed in parts of --part-anchors, one worker
            per part; a finished part leaves <out>/parts/part<i>.npz (resumable).
  fidelity  Replays the executed actions of --n branches in the simulator from the same anchor state and compares
            the imagined frames with the real ones on the same grid: image coverage gap per round, and precision /
            recall of the imagined success against the real image coverage and the simulator's own coverage. All
            three share one success criterion, coverage >= filter.success_cov (0.9) at any round; the eval's 0.95
            (pusht.success_threshold) is not used here.
            Writes <out>/fidelity.npz, <out>/fidelity.json and <out>/fidelity.png.

    CUDA_VISIBLE_DEVICES=6 uv run --extra training python research/nanowm_aug/proj/pusht/early_anchors/rollout.py \
        run --source ~/datastorage/nanowm_pusht_aug/early_anchors/source_na10 --n-iter 20 --f 1 --out <o>
    uv run --extra training python .../rollout.py fidelity --out <o> --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from ea_common import (
    CFG,
    FI,
    LEAD,
    NF,
    RES,
    H,
    first_hit,
    load_rollout,
    load_source,
    make_env,
    replay_to_anchor,
    save_part,
)

from nwaug.attribution.config_io import load_config

WORKER = RES / "nwaug" / "wm" / "imagined_rollout.py"


def anchor_context(env96, env224, seed: int, acts: np.ndarray, t: int) -> dict:
    """Exact replay of a[0 : t] from env.reset(seed): context frames F(t-10), F(t-5), F(t) (224px), the 96px
    policy frame at t, and the agent position / velocity BEFORE each step t-10 .. t (index LEAD = at t)."""
    for e in (env96, env224):
        e.reset(seed=int(seed))
    u96 = env96.unwrapped
    ctx = np.zeros((H, 224, 224, 3), np.uint8)
    pos = np.zeros((LEAD + 1, 2))
    vel = np.zeros((LEAD + 1, 2))
    for s in range(t + 1):
        if s >= t - LEAD:
            i = s - (t - LEAD)
            pos[i], vel[i] = np.asarray(u96.agent.position), np.asarray(u96.agent.velocity)
            if i % FI == 0:
                ctx[i // FI] = np.asarray(env224.unwrapped.get_obs())
        if s == t:
            break
        for e in (env96, env224):
            e.step(acts[s])
    return {"ctx": ctx, "f96": np.asarray(u96.get_obs()["pixels"]), "pos": pos, "vel": vel}


# ------------------------------------------------------------------------------------------------------------ run
def run_part(policy, pre, post, src: dict, idx: np.ndarray, a, part: int, dev: str) -> dict:
    import torch

    from nwaug.datagen.pack_closed_loop import to96
    from nwaug.wm.closed_loop_rollout import (
        NANOWM_PY,
        _save_atomic,
        _wait,
        chunk_slices,
        inject_noise,
        sample_chunks,
        window_segment,
    )
    from nwaug_envs.pusht import action_convert as ac

    k, rounds, R = a.k, a.n_iter * a.f, a.f
    n = len(idx)
    bs = n * k
    env96 = make_env("pixels_agent_pos", 96, 400)
    env224 = make_env("pixels", 224, 400)
    reps = [
        anchor_context(env96, env224, src["env_seed"][i], src["acts"][i], src["anchor_t"][i]) for i in idx
    ]
    env96.close()
    env224.close()

    stream = np.repeat(
        np.stack([src["acts"][i, src["anchor_t"][i] - LEAD : src["anchor_t"][i]] for i in idx]), k, 0
    )
    apos = np.repeat(np.stack([r["pos"][:LEAD] for r in reps]), k, 0)  # exact, from the replay
    avel = np.repeat(np.stack([r["vel"][:LEAD] for r in reps]), k, 0)
    pos = np.repeat(np.stack([r["pos"][LEAD] for r in reps]), k, 0)
    vel = np.repeat(np.stack([r["vel"][LEAD] for r in reps]), k, 0)
    cur96 = np.repeat(np.stack([r["f96"] for r in reps]), k, 0)

    work = a.out / "work" / f"part{part}"
    work.mkdir(parents=True, exist_ok=True)
    for f in work.glob("*"):
        f.unlink()
    _save_atomic(work / "seeds_expanded.npz", frames_224=np.repeat(np.stack([r["ctx"] for r in reps]), k, 0))
    wlog = open(work / "wm_worker.log", "w")  # noqa: SIM115 -- subprocess stdout, closed below
    proc = subprocess.Popen(
        [
            str(NANOWM_PY),
            str(WORKER),
            "closed-loop",
            "--work",
            str(work),
            "--rounds",
            str(rounds),
            "--history",
            str(H),
            "--num-sampling-steps",
            str(a.num_sampling_steps),
            "--wm-batch",
            str(a.wm_batch),
            "--seed",
            str(a.wm_seed + part),
        ],
        stdout=wlog,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    cov = np.zeros((bs, rounds), np.float32)
    frames96 = np.zeros((bs, rounds, 96, 96, 3), np.uint8)
    exec_acts = np.zeros((bs, rounds * FI, 2))
    need = NF * FI
    noise_rng = np.random.default_rng(a.noise_seed + part)
    try:
        for r in range(rounds):
            sample_now, _off, look_sl, ex_sl = chunk_slices(r, R, FI, need, LEAD)
            if sample_now:
                if r == 0:
                    torch.manual_seed(a.branch_seed + part)
                chunks = sample_chunks(
                    policy, pre, post, cur96[::k] if r == 0 else cur96, k if r == 0 else 1, dev
                )
                if a.inject_sigma > 0:  # pos / vel = the agent state at this query (tracked by the PD model)
                    chunks = inject_noise(chunks, pos, vel, a.inject_sigma, a.max_z, a.noise_hold, noise_rng)
            wins = np.zeros((bs, NF, FI * 2), np.float32)
            k0 = r * FI
            for b in range(bs):
                seg = window_segment(stream[b, k0 : k0 + LEAD], chunks[b], look_sl, need, FI)
                rel, _, _ = ac.abs_targets_to_rel(seg, apos[b, k0], avel[b, k0])
                wins[b] = ac.rel_to_nanowm(rel)
            _save_atomic(work / f"chunks_{r}.npz", win=wins, live=np.ones(bs, bool))
            _wait(work / f"frames_{r + 1}.npz", proc=proc)
            fr = np.load(work / f"frames_{r + 1}.npz")
            cov[:, r] = fr["cov"]
            frames96[:, r] = np.stack([to96(x) for x in fr["frames"]])
            ex = chunks[:, ex_sl]
            exec_acts[:, r * FI : (r + 1) * FI] = ex
            newpos, newvel = np.zeros((bs, FI, 2)), np.zeros((bs, FI, 2))
            for b in range(bs):
                for t in range(FI):
                    newpos[b, t], newvel[b, t] = pos[b], vel[b]
                    pos[b], vel[b] = ac.pd_step(pos[b], vel[b], ex[b, t])
            stream = np.concatenate([stream, ex], 1)
            apos = np.concatenate([apos, newpos], 1)
            avel = np.concatenate([avel, newvel], 1)
            cur96 = frames96[:, r]
            print(f"  part {part} round {r}: cov mean {cov[:, r].mean():.3f}", flush=True)
        (work / "stop").touch()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            (work / "stop").touch()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        wlog.close()
    return {
        "src_idx": np.repeat(idx, k),
        "cand": np.tile(np.arange(k), n),
        "cov": cov,
        "frames96": frames96,
        "exec_acts": exec_acts,
        "anchor_frame96": np.repeat(np.stack([r["f96"] for r in reps]), k, 0),
    }


def cmd_run(a) -> None:
    import torch

    from nwaug.attribution.policy_io import load_policy

    cfg = load_config(CFG)
    ro = cfg["rollout"]
    a.k = a.k or ro["k"]
    a.branch_seed = ro["branch_seed"] if a.branch_seed is None else a.branch_seed
    a.wm_seed = ro["wm_seed"] if a.wm_seed is None else a.wm_seed
    srcf = np.load(a.source / "episodes.npz")
    src = {k: srcf[k] for k in ("env_seed", "acts", "anchor_t", "success", "end")}
    sprov = json.loads(str(srcf["provenance_json"]))
    ckpt = Path(a.ckpt or sprov["ckpt"])
    anchors = np.flatnonzero(src["anchor_t"] >= 0)
    if a.n_anchors:
        anchors = anchors[: a.n_anchors]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "parts").mkdir(exist_ok=True)
    prov = {
        "source": str(a.source),
        "source_na": sprov["na"],
        "ckpt": str(ckpt),
        "n_iter": a.n_iter,
        "f": a.f,
        "rounds": a.n_iter * a.f,
        "k": a.k,
        "anchors": int(len(anchors)),
        "branch_seed": a.branch_seed,
        "wm_seed": a.wm_seed,
        "num_sampling_steps": a.num_sampling_steps,
        "success_cov": cfg["filter"]["success_cov"],
    }
    if a.inject_sigma > 0:
        prov.update(
            inject_sigma=a.inject_sigma, max_z=a.max_z, noise_hold=a.noise_hold, noise_seed=a.noise_seed
        )
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load_policy(ckpt, device=dev)
    t0 = time.time()
    for p, i0 in enumerate(range(0, len(anchors), a.part_anchors)):
        dest = a.out / "parts" / f"part{p}.npz"
        if dest.exists():
            continue
        res = run_part(policy, pre, post, src, anchors[i0 : i0 + a.part_anchors], a, p, dev)
        save_part(dest, **res)
        done = min(len(anchors), i0 + a.part_anchors)
        acc = (res["cov"] >= prov["success_cov"]).any(1).mean()
        print(
            f"part {p}: anchors {done}/{len(anchors)}, imagined success {acc:.1%} ({time.time() - t0:.0f}s)"
        )
    print(json.dumps(summarize_run(a.out), indent=2))


def summarize_run(out: Path) -> dict:
    prov, d, _ = load_rollout(out)
    r_hit = first_hit(d["cov"], prov["success_cov"])
    ok = r_hit >= 0
    per_anchor = np.array([ok[d["src_idx"] == s].any() for s in np.unique(d["src_idx"])])
    return {
        "branches": int(len(ok)),
        "imagined_success": float(ok.mean()),
        "anchors_with_success": f"{int(per_anchor.sum())}/{len(per_anchor)}",
        "success_round_median": float(np.median(r_hit[ok]) + 1) if ok.any() else None,
        "cov_last_mean": float(d["cov"][:, -1].mean()),
    }


# ------------------------------------------------------------------------------------------------------- fidelity
def cmd_fidelity(a) -> None:
    from nwaug_envs.pusht.coverage import coverage_from_frame

    prov, d, _ = load_rollout(a.out)
    src = load_source(Path(prov["source"]))
    thr_img = thr_sim = prov["success_cov"]  # imagined and real share one success criterion
    rounds = prov["rounds"]
    rng = np.random.default_rng(prov["branch_seed"])
    pick = np.sort(rng.choice(len(d["cov"]), size=min(a.n, len(d["cov"])), replace=False))
    env224 = make_env("pixels", 224, 300 + rounds * FI + 10)
    real_cov = np.zeros((len(pick), rounds), np.float32)  # image coverage of the real frame, same grid
    sim_cov = np.zeros((len(pick), rounds), np.float32)  # the simulator's own coverage
    t0 = time.time()
    for j, b in enumerate(pick):
        replay_to_anchor(env224, src, d["src_idx"][b])
        for r in range(rounds):
            for s in range(FI):
                _, _, _, _, info = env224.step(d["exec_acts"][b, r * FI + s])
            real_cov[j, r] = coverage_from_frame(np.asarray(env224.unwrapped.get_obs()))["coverage_cal"]
            sim_cov[j, r] = info["coverage"]
        if (j + 1) % 50 == 0:
            print(f"  replayed {j + 1}/{len(pick)} ({time.time() - t0:.0f}s)", flush=True)
    env224.close()
    img_cov = d["cov"][pick]
    np.savez(a.out / "fidelity.npz", pick=pick, img_cov=img_cov, real_cov=real_cov, sim_cov=sim_cov)
    summ = fidelity_summary(img_cov, real_cov, sim_cov, thr_img, thr_sim)
    summ = {"n": int(len(pick)), "thr_img": thr_img, "thr_sim": thr_sim, **summ}
    (a.out / "fidelity.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    plot_fidelity(a.out / "fidelity.png", img_cov, real_cov, sim_cov, thr_img, thr_sim, prov)


def fidelity_summary(img_cov, real_cov, sim_cov, thr_img, thr_sim) -> dict:
    gap = np.abs(img_cov - real_cov)
    img_hit = first_hit(img_cov, thr_img)
    real_ok = (real_cov >= thr_img).any(1)
    sim_ok = (sim_cov >= thr_sim).any(1)
    img_ok = img_hit >= 0
    rows = np.flatnonzero(img_ok)
    at_hit = real_cov[rows, img_hit[rows]]  # real image coverage at the round the imagination succeeded

    def ratio(num, den):
        return None if den == 0 else round(float(num) / float(den), 4)

    return {
        "imagined_success": ratio(img_ok.sum(), len(img_ok)),
        "real_success_img_cov": ratio(real_ok.sum(), len(real_ok)),
        "sim_success": ratio(sim_ok.sum(), len(sim_ok)),
        "precision_vs_real_img_cov": ratio((img_ok & real_ok).sum(), img_ok.sum()),
        "precision_vs_sim_success": ratio((img_ok & sim_ok).sum(), img_ok.sum()),
        "precision_at_hit_round": ratio((at_hit >= thr_img).sum(), len(rows)),
        "recall_vs_real_img_cov": ratio((img_ok & real_ok).sum(), real_ok.sum()),
        "recall_vs_sim_success": ratio((img_ok & sim_ok).sum(), sim_ok.sum()),
        "real_cov_at_hit_median": float(np.median(at_hit)) if len(rows) else None,
        "cov_gap_mean_by_round": [round(float(x), 4) for x in gap.mean(0)],
        "cov_gap_mean": float(gap.mean()),
        "cov_corr": float(np.corrcoef(img_cov.ravel(), real_cov.ravel())[0, 1]),
    }


def plot_fidelity(path: Path, img_cov, real_cov, sim_cov, thr_img, thr_sim, prov) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = img_cov.shape[1]
    x = np.arange(1, rounds + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    gap = np.abs(img_cov - real_cov)
    ax[0].plot(x, gap.mean(0), color="#2a6fdb", label="mean")
    ax[0].fill_between(x, *np.percentile(gap, [25, 75], axis=0), color="#2a6fdb", alpha=0.2, label="p25-p75")
    ax[0].set(
        xlabel="NanoWM round (5 sim steps)", ylabel="|imagined - real| image coverage", title="coverage gap"
    )
    ax[0].legend(frameon=False)
    ax[1].scatter(real_cov[:, -1], img_cov[:, -1], s=8, color="#2a6fdb", alpha=0.6)
    ax[1].plot([0, 1], [0, 1], color="#888", lw=0.8)
    ax[1].axhline(thr_img, color="#e8692e", lw=0.8)
    ax[1].axvline(thr_img, color="#e8692e", lw=0.8)
    ax[1].set(
        xlabel="real image coverage (last round)", ylabel="imagined coverage (last round)", title="last round"
    )
    ax[2].plot(x, (img_cov >= thr_img).mean(0), color="#2a6fdb", label="imagined >= thr")
    ax[2].plot(x, (real_cov >= thr_img).mean(0), color="#e8692e", label="real image >= thr")
    ax[2].plot(x, (sim_cov >= thr_sim).mean(0), color="#555", ls="--", label="sim coverage >= thr")
    ax[2].set(xlabel="NanoWM round", ylabel="fraction of branches", title="success by round")
    ax[2].legend(frameon=False)
    for axx in ax:
        axx.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"f={prov['f']} n_iter={prov['n_iter']}  ({len(img_cov)} replayed branches)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", type=Path, required=True, help="anchors.py source output dir")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--n-iter", type=int, required=True, help="policy queries after the anchor")
    r.add_argument("--f", type=int, required=True, help="NanoWM rounds per policy query")
    r.add_argument("--k", type=int, default=None, help="branches per anchor (default: rollout.k)")
    r.add_argument("--ckpt", default=None, help="policy (default: the source run's ckpt)")
    r.add_argument("--n-anchors", type=int, default=None, help="first N anchors only (smoke tests)")
    r.add_argument("--part-anchors", type=int, default=16)
    r.add_argument("--num-sampling-steps", type=int, default=30)
    r.add_argument("--wm-batch", type=int, default=64)
    r.add_argument("--branch-seed", type=int, default=None, help="default: rollout.branch_seed")
    r.add_argument("--wm-seed", type=int, default=None, help="default: rollout.wm_seed")
    r.add_argument("--inject-sigma", type=float, default=0.0, help="action noise on every chunk (0 = off)")
    r.add_argument("--max-z", type=float, default=3.0)
    r.add_argument("--noise-hold", type=int, default=1)
    r.add_argument("--noise-seed", type=int, default=0, help="numpy rng seed + part")
    fi = sub.add_parser("fidelity")
    fi.add_argument("--out", type=Path, required=True)
    fi.add_argument("--n", type=int, default=200, help="branches replayed in the simulator")
    a = ap.parse_args()
    {"run": cmd_run, "fidelity": cmd_fidelity}[a.cmd](a)


if __name__ == "__main__":
    main()
