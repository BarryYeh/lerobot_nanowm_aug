#!/usr/bin/env python
"""Stage 2 of the value/reward-model plan: EXACT-replay V labels for NanoWM-imagined branches.

Input: every branch npz that `eval/closed_loop_rollout.py --save-latents --accept-cov 0` wrote under
<stage2 root>/sigma*/branches/ (one per (anchor seed, candidate)). Each carries the NanoWM latents of
the whole rollout, the real lead-in actions, the executed action targets and the PD-tracked agent
state at every round boundary.

For every branch and every query round r in `stage2.query_rounds` (r=0 once per seed, not once per
branch: at r=0 all three frames are the real context, only the VAE sample noise differs) the label
is the outcome of the BASE policy from exactly the state the imagined branch is in after r rounds:

  1. Re-create the seed state the way the driver does (state convention: BODY origin, the same
     `seed_state` the driver placed): reset(seed=9000+seed_index), agent position, block.angle,
     block.position, zero block velocity, one `space.step`. Replay `lead_acts` (10 real actions) and
     ASSERT the sim equals the saved `anchor_state` (body origin, same form) within 1e-6 and the agent
     pos/vel equals `agent_state_hist[0]` within 1e-6 px. This proves the replay reproduces the run
     that produced the context frames.
  2. Replay `exec_abs_targets[: r*5]` and ASSERT agent pos/vel == `agent_state_hist[r]` within 1e-6 px
     (the driver's PD tracking of the agent equals the sim's). If the env terminated (success) during
     the replay the row is pre_success=True, y=True and no continuation is run.
  3. Otherwise continue with the base policy (n_action_steps=10, fresh `policy.reset()`) for B=200
     steps; y = any `info["is_success"]`. Continuations are batched across rows through
     `sim_rollout.rollout_batch(prepared_envs=...)`, so the rollout loop is not duplicated. The
     policy RNG is seeded from `label.cont_seed` (`--cont-seed` overrides; Table D re-runs use
     +1) and, when `label.cont_perturb` is set (T2 backup: noise everywhere), the WHOLE continuation
     is noised with that sigma / hold (`perturb_mode="prefix"`, `perturb_p = B`, noise rng seeded
     `cont_seed + 1`). Rows are sorted by (seed_id, sigma, cand, r) before batching (O9).
  4. wall_artifact = min `block_wall_gap` (CoG states) over the replay AND the continuation < -10 px.
     The row is kept and flagged; the gate/training code excludes flagged rows per the label rules.

This is NOT proto_filter_roc.py's replay, which starts from `anchor_state` with zero velocity; here
the agent's true velocity at the anchor and at every boundary is reproduced by construction.

Row fields (labels.npz, one row per (branch, r) with the r=0 exception above):
  latent_stack (N,3,4,32,32) f16   latents[r : r+3]; index r+2 (newest) is the frame of query time r*5
  frame_steps  (N,3) int64         [r*5-10, r*5-5, r*5] relative to the anchor (spacing 5, asserted)
  agent_state  (N,4)  f64          agent_state_hist[r] = [x, y, vx, vy]
  cov          (N,)   f32          RM label = sim_cov (real simulator coverage after the replay)
  cov_source   (N,)   str          always "sim"
  sim_cov      (N,)   f32          sim coverage after the replay
  cov_imagined (N,)   f32          diagnostic: worker cov of the imagined frame (r>=1: coverage[r-1];
                                   r=0: sim coverage at the anchor); cross-checked vs work/frames_r.npz
  success_step (N,)   i64          steps from the query state to the first success in the continuation
                                   (-1 = none within B; 0 = already a success during the replay)
  T            (N,)   i64          unified T (= success_step; kept distinct for stage-1/2 symmetry)
  cov_end      (N,)   f32          coverage at query + min(T, B) (sim_cov for replay successes)
  v_target     (N,)   f32          gamma**T * 1[0<=T<=B] (beta = 0; replay success -> 1)
  anchor_set (N,) str "train"|"test" (from --tag, cross-checked against the seeds file); src_seed,
  src_episode (N,) i64 (provenance of the anchor state); noised (N,) bool = sigma > 0;
  provenance_json (str).
  y, pre_success, wall_artifact (N,) bool;  wall_gap_min (N,) f32
  sigma (N,) f32 (branch inject_sigma), seed_id (N,) i64 (row in seeds_stage2.npz), cand (N,) i64,
  branch (N,) str (file name, sigma dir prefixed), r (N,) i64
  replay_cog_state (N,5) f64       CoG state [agent xy, CoG xy, theta] after the replay (oracle input)
Stored scalars: state_convention_replay_cog_state="cog", B, na, check counters.

    uv run --extra training python examples/nanowm_pusht_aug/critic/label_imagined_branches.py \
        --config configs/sailor_v1_reset.yaml [--tag train|test] [--smoke]

`--tag test` labels `test_stage2/` against `seeds_test.npz` (G6 anchors; train never reads it).
`--anchor-subset 3,7,...|ids.npz --cont-seed N --out-name f.npz` re-labels only those anchors into a
separate file (Table D noise-ceiling re-runs); `labels.npz` is never touched in that mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

_CRITIC = Path(__file__).resolve().parent
sys.path.insert(0, str(_CRITIC))
sys.path.insert(0, str(_CRITIC.parent))
sys.path.insert(0, str(_CRITIC.parent / "eval"))
sys.path.insert(0, str(_CRITIC.parent / "attribution"))
from config_io import load_config  # noqa: E402
from datagen.provenance import provenance_entry  # noqa: E402
from nwaug.pusht_state import load_seeds, place_body, read_body  # noqa: E402
from ood_failure_map import block_wall_gap, sim_state  # noqa: E402
from policy_io import load_policy  # noqa: E402
from sim_rollout import rollout_batch  # noqa: E402

from critic import labels  # noqa: E402

FI = 5  # sim steps per NanoWM round
LEAD = 10  # real lead-in steps (driver: (H-1)*fi)
TOL = 1e-6


def agent_pv(u) -> np.ndarray:
    return np.array([*u.agent.position, *u.agent.velocity], np.float64)


def place_seed(env, seed_index: int, st: np.ndarray) -> None:
    """The driver's seed placement (closed_loop_rollout lead-in, BODY-origin state)."""
    env.reset(seed=9000 + seed_index)
    place_body(env.unwrapped, st)


def replay_row(env, br: dict, r: int, name: str, stats: dict) -> dict:
    """Exact replay of one branch to round r on `env`. Asserts as documented in the module doc."""
    u = env.unwrapped
    place_seed(env, int(br["seed_index"]), br["seed_state"])
    wall_min = float(block_wall_gap(sim_state(u)[None])[0])
    for t in range(LEAD):
        env.step(br["lead_acts"][t])
        wall_min = min(wall_min, float(block_wall_gap(sim_state(u)[None])[0]))
    err_anchor = float(np.abs(read_body(u) - br["anchor_state"]).max())
    assert err_anchor <= TOL, f"{name} r={r}: replayed anchor differs from anchor_state by {err_anchor:.3e}"
    err0 = float(np.abs(agent_pv(u) - br["agent_state_hist"][0]).max())
    assert err0 <= TOL, f"{name} r={r}: agent state at anchor differs from agent_state_hist[0] by {err0:.3e}"
    stats["anchor_checks"] += 1
    stats["max_anchor_err"] = max(stats["max_anchor_err"], err_anchor)
    stats["agent_checks"] += 1
    stats["max_agent_err"] = max(stats["max_agent_err"], err0)
    cov_anchor = float(u._get_coverage())

    terminated, n_done = False, 0
    for t in range(r * FI):
        _o, _r, term, _tr, _info = env.step(br["exec_abs_targets"][t])
        n_done += 1
        wall_min = min(wall_min, float(block_wall_gap(sim_state(u)[None])[0]))
        if term:
            terminated = True
            break
    if r > 0 and n_done == r * FI:  # the whole replay ran (success on its last step still qualifies)
        errr = float(np.abs(agent_pv(u) - br["agent_state_hist"][r]).max())
        assert errr <= TOL, f"{name} r={r}: agent state differs from agent_state_hist[{r}] by {errr:.3e}"
        stats["agent_checks"] += 1
        stats["max_agent_err"] = max(stats["max_agent_err"], errr)
    return {
        "terminated": terminated,
        "wall_min": wall_min,
        "sim_cov": float(u._get_coverage()),
        "cov_anchor": cov_anchor,
        "cog": sim_state(u),
    }


def parse_anchor_subset(spec: str) -> np.ndarray:
    """`--anchor-subset`: comma-separated seed ids, or an .npz / .npy holding them (key `seed_id`,
    `anchors` or `ids` for npz). Returns a sorted unique int64 array."""
    path = Path(spec)
    if path.suffix in (".npz", ".npy") and path.exists():
        d = np.load(path)
        if path.suffix == ".npy":
            ids = d
        else:
            key = next(k for k in ("seed_id", "anchors", "ids") if k in d.files)
            ids = d[key]
    else:
        ids = [int(x) for x in spec.split(",") if x.strip()]
    return np.unique(np.asarray(ids, np.int64))


def assert_shared_lead_in(first: dict, br: dict, name: str) -> None:
    """Every branch of one anchor must start from the SAME decision point (O1/M2): the lead-in
    actions, the anchor state and the agent state at the anchor are bit-identical."""
    for key, a, b in (
        ("lead_acts", first["lead_acts"], br["lead_acts"]),
        ("anchor_state", first["anchor_state"], br["anchor_state"]),
        ("agent_state_hist[0]", first["agent_state_hist"][0], br["agent_state_hist"][0]),
    ):
        assert np.array_equal(a, b), f"{name}: {key} differs from {first['name']} (same seed_index)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--tag", choices=["train", "test"], default="train", help="stage2/ vs test_stage2/ (G6)")
    p.add_argument("--seeds", type=Path, default=None, help="anchor seeds npz (MUST exist); default by tag")
    p.add_argument(
        "--batch", type=int, default=None, help="rows continued together; default = label.batch (else 100)"
    )
    p.add_argument("--cont-seed", type=int, default=None, help="override label.cont_seed (Table D: 770001)")
    p.add_argument("--anchor-subset", default=None, help="only these seed ids: '3,7,9' or an npz/npy")
    p.add_argument("--out-name", default="labels.npz")
    args = p.parse_args()
    cfg = load_config(args.config)
    s2, lab = cfg["stage2"], cfg["label"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    base = Path(cfg["out_dir"]) / ("smoke" if args.smoke else "")
    root = base / ("stage2" if args.tag == "train" else "test_stage2")
    rounds = [int(r) for r in s2["query_rounds"]]
    B, na, wall_pen = int(lab["B"]), int(lab["na"]), float(lab["wall_pen_px"])
    gamma = float(lab.get("gamma", 0.995))
    h_max = int(s2["h_max"])
    batch_n = int(args.batch if args.batch is not None else lab.get("batch", 100))
    cont_seed = int(args.cont_seed if args.cont_seed is not None else lab.get("cont_seed", 0))
    cp = lab.get("cont_perturb")  # T2 backup: the continuation is noised too
    expect_R = int(s2.get("replan_every_rounds", 1))
    expect_lead_na = s2.get("lead_na")  # set <=> the paired-lead-in imagine pipeline produced the branches
    expect_H = int(cfg.get("stack", {}).get("H", 3))
    assert max(rounds) <= h_max and rounds[0] == 0, rounds
    assert h_max % expect_R == 0 and all(r % expect_R == 0 for r in rounds), (rounds, expect_R)
    if args.anchor_subset is not None:
        assert args.out_name != "labels.npz", "--anchor-subset must write to a separate --out-name"
    subset = parse_anchor_subset(args.anchor_subset) if args.anchor_subset is not None else None

    branch_files = []
    for d in sorted(root.glob("sigma*/branches")):
        branch_files += sorted(d.glob("seed*_cand*.npz"))
    assert branch_files, f"no branch npz under {root}/sigma*/branches"
    print(
        f"label[{args.tag}]: {len(branch_files)} branches, rounds {rounds}, B={B}, na={na}, "
        f"batch={batch_n}, cont_seed={cont_seed}, cont_perturb={cp} <- {root}"
    )

    seeds_path = args.seeds or (base / ("seeds_stage2.npz" if args.tag == "train" else "seeds_test.npz"))
    assert seeds_path.exists(), f"seeds file {seeds_path} must exist (run the anchors stage)"  # N18d
    seeds_body = load_seeds(seeds_path, "body")
    sd = np.load(seeds_path)
    seeds_set = str(sd["anchor_set"]) if "anchor_set" in sd.files else None
    if seeds_set is not None:
        assert seeds_set == args.tag, f"{seeds_path}: anchor_set={seeds_set!r} but --tag {args.tag}"
    src_seed = int(sd["src_seed"]) if "src_seed" in sd.files else -1
    src_episode = sd["src_episode"].astype(np.int64) if "src_episode" in sd.files else None

    # ---- load branches (small keys only) + per-branch asserts ----
    branches, first_of_seed = [], {}
    for f in branch_files:
        d = np.load(f)
        # skip the big per-branch image arrays (`frames`, `anchor_frame`): ~1.4 MB/branch, unused here
        br = {k: d[k] for k in d.files if k not in ("frames", "anchor_frame")}
        br["name"] = f"{f.parent.parent.name}/{f.name}"
        br["path"] = f
        sid = int(br["seed_index"])
        if subset is not None and sid not in subset:
            continue
        assert "latents" in br, f"{f} has no latents: rerun closed_loop_rollout.py with --save-latents"
        assert br["latents"].shape == (3 + h_max, 4, 32, 32), br["latents"].shape
        assert br["agent_state_hist"].shape == (h_max + 1, 4), br["agent_state_hist"].shape
        assert br["exec_abs_targets"].shape == (h_max * FI, 2)
        assert br["lead_acts"].shape == (LEAD, 2)
        # N18d: the branch was really generated from this anchor row (seed file MUST exist)
        assert sid < len(seeds_body), (f, sid, len(seeds_body))
        assert np.allclose(br["seed_state"], seeds_body[sid], atol=1e-9), f"{f}: seed_state != seeds row"
        # M7: cadence metadata written by the driver
        if "replan_every_rounds" in br or expect_lead_na is not None:
            assert int(br["fi"]) == FI and int(br["hist"]) == expect_H, (f, int(br["fi"]), int(br["hist"]))
            assert int(br["replan_every_rounds"]) == expect_R, (f, int(br["replan_every_rounds"]), expect_R)
            assert int(br["horizon"]) >= 10 + (expect_R - 1) * FI, (f, int(br["horizon"]))
            if expect_lead_na is not None:
                assert int(br["lead_na"]) == int(expect_lead_na), (f, int(br["lead_na"]), expect_lead_na)
        if expect_lead_na is not None:  # O1/M2: all branches of one anchor share the decision point
            if sid in first_of_seed:
                assert_shared_lead_in(first_of_seed[sid], br, br["name"])
            else:
                first_of_seed[sid] = br
        branches.append(br)
    assert branches, "no branch left after --anchor-subset"

    # ---- job list sorted by (seed_id, sigma, cand, r) (O9); r=0 only for the first branch of a seed ----
    keyed = sorted(
        (
            (int(b["seed_index"]), float(b["inject_sigma"]), int(b["path"].stem.split("_cand")[1]), r, bi)
            for bi, b in enumerate(branches)
            for r in rounds
        ),
    )
    jobs, seen_seed = [], set()
    for sid, _sig, _cand, r, bi in keyed:
        if r == 0 and sid in seen_seed:
            continue
        jobs.append((bi, r))
        seen_seed.add(sid)
    n_rows = len(jobs)
    print(f"label: {n_rows} rows ({len(seen_seed)} seeds)")

    policy, pre, post = load_policy(Path(cfg["ckpt"][cfg["ckpt_key"]]), device=device)
    max_steps = LEAD + h_max * FI + B + 60
    out = {
        k: []
        for k in [
            "latent_stack",
            "frame_steps",
            "agent_state",
            "cov_imagined",
            "sim_cov",
            "y",
            "pre_success",
            "wall_artifact",
            "wall_gap_min",
            "success_step",
            "T",
            "cov_end",
            "sigma",
            "seed_id",
            "cand",
            "branch",
            "r",
            "replay_cog_state",
        ]
    }
    stats = {"anchor_checks": 0, "max_anchor_err": 0.0, "agent_checks": 0, "max_agent_err": 0.0}
    t0, t_replay, t_cont = time.time(), 0.0, 0.0
    cont_noise_rng = np.random.default_rng(cont_seed + 1)  # advances through the (sorted) batches

    for c0 in range(0, n_rows, batch_n):
        chunk = jobs[c0 : c0 + batch_n]
        envs = [
            gym.make(
                "gym_pusht/PushT-v0",
                obs_type="pixels_agent_pos",
                observation_width=96,
                observation_height=96,
                max_episode_steps=max_steps,
            )
            for _ in chunk
        ]
        ta = time.time()
        rp = [
            replay_row(e, branches[bi], r, branches[bi]["name"], stats)
            for e, (bi, r) in zip(envs, chunk, strict=True)
        ]
        t_replay += time.time() - ta

        alive = [i for i, x in enumerate(rp) if not x["terminated"]]
        succ_step = np.zeros(len(chunk), np.int64)  # replay-time success -> 0 (T = 0, pre_success)
        cov_end = np.array([x["sim_cov"] for x in rp], np.float32)
        wall_gap = np.array([x["wall_min"] for x in rp], np.float64)
        if alive:
            ta = time.time()
            kw = {}
            if cp is not None:
                kw = {
                    "sigma": np.full(len(alive), float(cp["sigma"])),
                    "perturb_p": np.full(len(alive), B, np.int64),  # >= episode_steps: whole continuation
                    "perturb_mode": "prefix",
                    "max_z": float(cp["max_z"]),
                    "noise_hold": int(cp["noise_hold"]),
                    "noise_rng": cont_noise_rng,
                }
            res = rollout_batch(
                policy,
                pre,
                post,
                np.stack([rp[i]["cog"] for i in alive]),
                device=device,
                na=na,
                episode_steps=B,
                seed=cont_seed,
                batch_start=c0,
                lockstep224=False,
                wall_pen_px=wall_pen,
                prepared_envs=[envs[i] for i in alive],
                **kw,
            )
            assert res["queue_maxlen"] == na, res["queue_maxlen"]
            t_cont += time.time() - ta
            for j, i in enumerate(alive):
                s = int(res["success_step"][j])
                assert s == -1 or 1 <= s <= B, s
                succ_step[i] = s
                cov_end[i] = res["cov_per_step"][j, int(labels.cov_end_index(0, s, B))]
                wall_gap[i] = min(wall_gap[i], float(res["wall_gap_min"][j]))
        for e in envs:
            e.close()
        T_chunk = succ_step  # continuation start == query state, so T is the continuation success step

        for i, (bi, r) in enumerate(chunk):
            br = branches[bi]
            out["latent_stack"].append(br["latents"][r : r + 3])
            out["frame_steps"].append([r * FI - 10, r * FI - 5, r * FI])
            out["agent_state"].append(br["agent_state_hist"][r])
            out["cov_imagined"].append(float(br["coverage"][r - 1]) if r >= 1 else rp[i]["cov_anchor"])
            out["sim_cov"].append(rp[i]["sim_cov"])
            out["y"].append(0 <= int(T_chunk[i]) <= B)
            out["pre_success"].append(bool(rp[i]["terminated"]))
            out["wall_artifact"].append(bool(wall_gap[i] < -wall_pen))
            out["wall_gap_min"].append(float(wall_gap[i]))
            out["success_step"].append(int(T_chunk[i]))
            out["T"].append(int(T_chunk[i]))
            out["cov_end"].append(float(cov_end[i]))
            out["sigma"].append(float(br["inject_sigma"]))
            out["seed_id"].append(int(br["seed_index"]))
            out["cand"].append(int(br["path"].stem.split("_cand")[1]))
            out["branch"].append(br["name"])
            out["r"].append(r)
            out["replay_cog_state"].append(rp[i]["cog"])
        el = time.time() - t0
        print(
            f"  rows {c0 + len(chunk)}/{n_rows}  y={np.mean(out['y']):.3f}  "
            f"{el / (c0 + len(chunk)):.2f} s/row  (replay {t_replay:.0f}s, continuation {t_cont:.0f}s)",
            flush=True,
        )

    fs = np.asarray(out["frame_steps"], np.int64)
    assert (np.diff(fs, axis=1) == FI).all(), "frame_steps spacing != 5"

    # ---- stored imagined cov vs the worker's own frames_{r}.npz (only possible while the work dir survives) ----
    cov_cmp, cov_max = 0, 0.0
    by_name = {b["name"]: b for b in branches}
    rr, cov_img = np.asarray(out["r"]), np.asarray(out["cov_imagined"], np.float32)
    for i, name in enumerate(out["branch"]):
        r = int(rr[i])
        if r == 0:
            continue
        sig_dir = name.split("/")[0]
        wf = root / sig_dir / "work" / f"frames_{r}.npz"
        if not wf.exists():
            continue
        br = by_name[name]
        k = int(br["meta"][0])
        b_idx = int(br["seed_index"]) * k + int(out["cand"][i])
        w = np.load(wf)["cov"][b_idx]
        cov_max = max(cov_max, abs(float(w) - float(cov_img[i])))
        cov_cmp += 1

    T_arr = np.asarray(out["T"], np.int64)
    cov_end_arr = np.asarray(out["cov_end"], np.float32)
    sim_cov = np.asarray(out["sim_cov"], np.float32)
    v_t = np.asarray(labels.v_target(T_arr, cov_end_arr, B, gamma, 0.0), np.float32)
    seed_id_arr = np.asarray(out["seed_id"], np.int64)
    n = len(seed_id_arr)
    prov = provenance_entry(args, stage="label", config=str(args.config.resolve()), cont_seed=cont_seed)
    dst = root / args.out_name
    np.savez_compressed(
        dst,
        latent_stack=np.stack(out["latent_stack"]).astype(np.float16),
        frame_steps=fs,
        agent_state=np.asarray(out["agent_state"], np.float64),
        cov=sim_cov,  # RM label = REAL simulator coverage (pre-reg §3)
        cov_source=np.full(n, "sim"),
        sim_cov=sim_cov,
        cov_imagined=cov_img,
        y=np.asarray(out["y"], bool),
        pre_success=np.asarray(out["pre_success"], bool),
        wall_artifact=np.asarray(out["wall_artifact"], bool),
        wall_gap_min=np.asarray(out["wall_gap_min"], np.float32),
        success_step=T_arr,
        T=T_arr,
        cov_end=cov_end_arr,
        v_target=v_t,
        sigma=np.asarray(out["sigma"], np.float32),
        noised=np.asarray(out["sigma"], np.float32) > 0,
        seed_id=seed_id_arr,
        cand=np.asarray(out["cand"], np.int64),
        branch=np.asarray(out["branch"]),
        r=rr.astype(np.int64),
        anchor_set=np.full(n, args.tag),
        src_seed=np.full(n, src_seed, np.int64),
        src_episode=(src_episode[seed_id_arr] if src_episode is not None else np.full(n, -1, np.int64)),
        replay_cog_state=np.asarray(out["replay_cog_state"], np.float64),
        state_convention_replay_cog_state="cog",
        B=np.int64(B),
        na=np.int64(na),
        gamma=np.float64(gamma),
        cont_seed=np.int64(cont_seed),
        batch=np.int64(batch_n),
        provenance_json=json.dumps(prov),
    )
    y_, pre_, wa_ = (np.asarray(out[k], bool) for k in ("y", "pre_success", "wall_artifact"))
    print(
        f"\nasserts passed: anchor_state {stats['anchor_checks']} (max err {stats['max_anchor_err']:.2e}), "
        f"agent_state_hist {stats['agent_checks']} (max err {stats['max_agent_err']:.2e} px)"
    )
    if expect_lead_na is not None:
        print(f"lead-in identity across branches of one anchor: OK ({len(first_of_seed)} anchors)")
    print(f"imagined cov vs worker frames_r.npz cov: {cov_cmp} compared, max |diff| {cov_max:.2e}")
    print(f"rows {n_rows}: y {y_.mean():.3f}, pre_success {pre_.sum()}, wall_artifact {wa_.sum()}")
    for r in rounds:
        m = rr == r
        print(f"  r={r}: n={m.sum():4d} y={y_[m].mean():.3f} pre_success={pre_[m].sum()} wall={wa_[m].sum()}")
    print(f"wrote {dst} in {time.time() - t0:.1f}s ({(time.time() - t0) / n_rows:.2f} s/row)")


if __name__ == "__main__":
    main()
