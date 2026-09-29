#!/usr/bin/env python
"""T0.1 of the OOD-dream plan -- is there an OOD region of PushT start states where the policy fails?

The OOD-dream idea (dream recovery segments from high-fail / OOD locations with NanoWM) only has
headroom if such a region exists. Earlier runs showed 96% of random-reset episodes reach
coverage >= 0.9 and fail in the endgame, so this is measured, not assumed.

Three stages (lerobot venv), each writing into --work:

  density   lerobot/pusht frames -> per-frame state feature (agent xy, block CoG xy, block angle).
            lerobot/pusht stores only agent xy, so the block pose is ESTIMATED from the 96px image
            with datagen.coverage_from_frame (measured on gym renders at 96px: CoG err median 4.9 px
            of 512, angle err median 1.2 deg; theta_sim = theta_img + 90 deg).
            -> train_states.npz

  sweep     N gym_pusht random resets (the lerobot-eval start distribution) rolled in lock-step
            batches at native 10 Hz / 96px with n_action_steps from the checkpoint -- the same
            protocol as `lerobot-eval --env.type=pusht`, so overall success must land near the
            known lerobot-eval number (sanity). Records per episode: exact start state, success,
            max coverage, and every --v-stride-th visited state for the on-policy MC value table.
            -> sweep_<tag>.npz

  analyze   kNN distance of each start state to the training states (smaller = denser), success
            binned by density quantile, and the OOD split (lowest-density quantile) vs in-dist
            split (highest). Prints the T0.1 gate: OOD-split success <= --gate-success with
            >= --gate-min-inits starts. Writes the splits + V table and a figure.
            -> ood_split_<tag>.npz, value_table_<tag>.npz, ood_failure_map_<tag>.png

The OOD split is selected on DENSITY (a policy-agnostic feature), never on this sweep's outcomes,
so its success estimate is not biased by selection and the split is fair to every later arm.

    P=examples/nanowm_pusht_aug; W=$P/cl/ood
    uv run --extra training python $P/eval/ood_failure_map.py density --work $W
    uv run --extra training python $P/eval/ood_failure_map.py sweep --work $W --tag control \
        --ckpt outputs/train/cl_aug/a_real_only_d8_s1000/checkpoints/last/pretrained_model --n 1000
    uv run --extra training --with matplotlib python $P/eval/ood_failure_map.py analyze \
        --work $W --tag control
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dir that contains datagen/
from datagen.coverage_from_frame import block_mask, pose_from_mask  # noqa: E402

IMG_TO_SIM = 512 / 96
THETA_IMG_TO_SIM = np.pi / 2
# Angle enters the distance as a chord of this radius: 1 rad of block rotation ~ moves the T's
# extremities ~60 px, comparable in effect to a 60 px translation.
ANGLE_RADIUS_PX = 60.0


def feature(states: np.ndarray, mode: str = "full") -> np.ndarray:
    """(N,5) [agent x, agent y, cog x, cog y, theta] -> Euclidean-comparable feature.

    "block" drops the agent: on the discovery OOD-start sweep, block-only density separated
    success 69% -> 28% across quintiles while agent-only density ran the opposite way, so
    including the agent washed the signal out."""
    th = states[:, 4]
    ang = ANGLE_RADIUS_PX * np.stack([np.cos(th), np.sin(th)], 1)
    parts = [states[:, 2:4], ang] if mode == "block" else [states[:, :4], ang]
    return np.concatenate(parts, 1).astype(np.float32)


def sim_state(u) -> np.ndarray:
    cog = u.block.local_to_world(u.block.center_of_gravity)
    return np.array([*u.agent.position, cog.x, cog.y, u.block.angle % (2 * np.pi)], np.float64)


# ---------------------------------------------------------------------------------------- density
def stage_density(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4, shuffle=False)
    states, ep_idx, agent_img_err = [], [], []
    for batch in loader:
        imgs = (batch["observation.image"].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        agent = batch["observation.state"].numpy()
        for img, a in zip(imgs, agent, strict=True):
            cx, cy, th = pose_from_mask(block_mask(img))
            states.append(
                [a[0], a[1], cx * IMG_TO_SIM, cy * IMG_TO_SIM, (th + THETA_IMG_TO_SIM) % (2 * np.pi)]
            )
            # sanity: agent xy in observation.state must share the sim frame with the image
            ib = img.astype(np.int16)
            m = (ib[..., 2] - ib[..., 0] > 100) & (ib[..., 1] < 150)
            if m.sum() > 3:
                ys, xs = np.nonzero(m)
                agent_img_err.append(np.hypot(xs.mean() * IMG_TO_SIM - a[0], ys.mean() * IMG_TO_SIM - a[1]))
        ep_idx.append(batch["episode_index"].numpy())
        print(f"  {len(states)}/{len(ds)}", end="\r")
    st = np.asarray(states, np.float64)
    ok = np.isfinite(st).all(1)
    print(
        f"\n{len(st)} frames, {(~ok).sum()} without a block mask (dropped); "
        f"agent image-vs-state err median {np.median(agent_img_err):.1f} px (expect < ~8)"
    )
    args.work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.work / "train_states.npz", states=st[ok], episode_index=np.concatenate(ep_idx)[ok]
    )
    print(f"wrote {args.work / 'train_states.npz'}")


def set_state(u, state: np.ndarray):
    """Place agent + block from a CoG-based state. Angle before position (see eval_baseline.py:
    setting the angle rotates the body about its offset CoG), and the body origin is recovered
    from the CoG so the block lands where the density feature says it is."""
    import pymunk

    u.agent.position = (float(state[0]), float(state[1]))
    u.agent.velocity = (0.0, 0.0)
    u.block.angle = float(state[4])
    offset = pymunk.Vec2d(*u.block.center_of_gravity).rotated(float(state[4]))
    u.block.position = (float(state[2]) - offset.x, float(state[3]) - offset.y)
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)
    return u.get_obs()


TOPBAND_COG_Y = (100.0, 145.0)
# T outline in body-local coords with the CoG (0,45) as origin; walls' inner faces are at 7 / 504
T_OUTLINE = np.array(
    [(-60, 0), (60, 0), (60, 30), (-60, 30), (-15, 30), (-15, 120), (15, 120), (15, 30)], float
) - (0, 45)


def block_wall_gap(states: np.ndarray) -> np.ndarray:
    """(...,5) CoG states -> px between the T's outline and the nearest wall face (negative = through
    the wall). A gap below the agent's 30 px diameter means no room to get between block and wall."""
    s = states.reshape(-1, 5)
    c, sn = np.cos(s[:, 4])[:, None], np.sin(s[:, 4])[:, None]
    x = s[:, 2, None] + T_OUTLINE[None, :, 0] * c - T_OUTLINE[None, :, 1] * sn
    y = s[:, 3, None] + T_OUTLINE[None, :, 0] * sn + T_OUTLINE[None, :, 1] * c
    g = np.minimum.reduce([x.min(1) - 7, 504 - x.max(1), y.min(1) - 7, 504 - y.max(1)])
    return g.reshape(states.shape[:-1])


def sample_topband_starts(n: int, seed: int, min_wall_gap: float = 45.0) -> np.ndarray:
    """Top-wall band: block CoG y in [100,145) -- never produced by gym_pusht resets (their CoG y
    starts at 145, see --ood-range) nor seen in lerobot/pusht -- with CoG x [100,400], any angle,
    and the agent uniform over [50,450]^2 but > 90 px from the CoG so it starts clear of the T.
    No kNN filter: the band itself is the OOD definition (T0.1: base policy 37.9% vs 65.5%).

    Candidates whose block sits closer than `min_wall_gap` to a wall are resampled: below ~30 px
    (the agent's diameter) the agent physically cannot get between block and wall, so those states
    are unrecoverable by repositioning rather than out-of-distribution. 45 px leaves a margin."""
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    tried = 0
    while len(kept) < n:
        c = rng.uniform(
            [50, 50, 100, TOPBAND_COG_Y[0], 0], [450, 450, 400, TOPBAND_COG_Y[1], 2 * np.pi], size=(4 * n, 5)
        )
        tried += len(c)
        c = c[np.hypot(c[:, 0] - c[:, 2], c[:, 1] - c[:, 3]) > 90]  # agent clear of the T
        kept += list(c[block_wall_gap(c) >= min_wall_gap])
    print(f"topband starts: kept {n} of {tried} candidates (agent clearance + wall gap >= {min_wall_gap}px)")
    return np.asarray(kept[:n])


def sample_ood_starts(args) -> np.ndarray:
    """Rejection-sample starts sparser than the reset distribution's `--ood-quantile` of kNN distance.

    Random resets draw from the same ranges the human demos started from, so even their sparsest
    quintile may not be meaningfully OOD. Candidates are uniform over agent [50,450]^2 and any
    angle, rejected if the agent overlaps the block (pymunk would eject it on the first step) or if
    denser than the threshold.

    Block CoG range (--ood-range):
      reset  x [100,400], y [145,445] -- where gym_pusht resets actually put the CoG. Its legacy
             `_set_state` sets position THEN angle, and the rotation about the offset CoG shifts
             it +45 px in y. lerobot/pusht training frames span the same band (y q.001 = 141).
      wide   x [100,400], y [100,400] -- the nominal body-position range, applied to the CoG by
             mistake in the first sweeps. Its y < 145 strip is never seen in training or resets;
             kept only to reproduce `control_oodstart*`."""
    if args.ood_range == "topband":
        return sample_topband_starts(args.n, args.seed)

    import gym_pusht  # noqa: F401
    import gymnasium as gym

    tr = np.load(args.work / "train_states.npz")["states"]
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96, observation_height=96
    )
    ref = []
    for s in range(1000):
        env.reset(seed=10_000_000 + s)
        ref.append(sim_state(env.unwrapped))
    env.close()
    # sampler is pinned to the full feature / k=20 it was first run with, so every OOD-start
    # sweep (discovery and held-out) draws from the same distribution regardless of --k/--feature
    ood_cutoff = float(np.quantile(knn_dist(np.asarray(ref), tr, 20, args.device), args.ood_quantile))
    rng = np.random.default_rng(args.seed)
    kept = []
    while len(kept) < args.n:
        y_lo, y_hi = (145, 445) if args.ood_range == "reset" else (100, 400)
        c = rng.uniform([50, 50, 100, y_lo, 0], [450, 450, 400, y_hi, 2 * np.pi], size=(4096, 5))
        c = c[np.hypot(c[:, 0] - c[:, 2], c[:, 1] - c[:, 3]) > 90]  # agent clear of the T
        kept += list(c[knn_dist(c, tr, 20, args.device) > ood_cutoff])
    print(f"OOD starts: kNN dist > {ood_cutoff:.1f} (reset-distribution q{args.ood_quantile})")
    return np.asarray(kept[: args.n])


# ------------------------------------------------------------------------------------------ sweep
def stage_sweep(args) -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dev = args.device
    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    if args.n_action_steps:
        policy.config.n_action_steps = args.n_action_steps
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )
    B = args.batch
    envs = [
        gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96, observation_height=96
        )
        for _ in range(B)
    ]
    starts, success, max_cov, n_steps = [], [], [], []
    v_states, v_ep, v_t = [], [], []
    t0 = time.time()
    forced = sample_ood_starts(args) if args.start_source == "ood" else None
    for b0 in range(0, args.n, B):
        nb = min(B, args.n - b0)
        obs = []
        for i in range(nb):
            o, _ = envs[i].reset(seed=args.seed + b0 + i)
            if forced is not None:
                o = set_state(envs[i].unwrapped, forced[b0 + i])
            obs.append(o["pixels"])
            starts.append(sim_state(envs[i].unwrapped))
        torch.manual_seed(args.seed + b0)
        policy.reset()
        done = np.zeros(nb, bool)
        succ = np.zeros(nb, bool)
        mcov = np.zeros(nb)
        steps = np.full(nb, args.max_steps)
        for t in range(args.max_steps):
            if t % args.v_stride == 0:
                for i in np.nonzero(~done)[0]:
                    v_states.append(sim_state(envs[i].unwrapped))
                    v_ep.append(b0 + i)
                    v_t.append(t)
            img = torch.from_numpy(np.stack(obs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)
            with torch.no_grad():
                act = post(policy.select_action(pre({"observation.image": img}))).cpu().numpy()
            for i in range(nb):
                if done[i]:
                    continue  # finished envs are frozen; their stale obs still fills the batch slot
                o, _r, term, trunc, info = envs[i].step(act[i].astype(np.float64))
                obs[i] = o["pixels"]
                mcov[i] = max(mcov[i], float(info.get("coverage", 0.0)))
                if term or trunc:
                    done[i], succ[i], steps[i] = True, bool(info.get("is_success", term)), t + 1
            if done.all():
                break
        success += succ.tolist()
        max_cov += mcov.tolist()
        n_steps += steps.tolist()
        el = time.time() - t0
        print(
            f"  {b0 + nb}/{args.n}  success so far {np.mean(success):.3f}  "
            f"({el / (b0 + nb):.2f} s/ep, eta {el / (b0 + nb) * (args.n - b0 - nb) / 60:.1f} min)"
        )
    for e in envs:
        e.close()
    args.work.mkdir(parents=True, exist_ok=True)
    out = args.work / f"sweep_{args.tag}.npz"
    np.savez_compressed(
        out,
        starts=np.asarray(starts),
        success=np.asarray(success),
        max_cov=np.asarray(max_cov),
        n_steps=np.asarray(n_steps),
        v_states=np.asarray(v_states),
        v_ep=np.asarray(v_ep),
        v_t=np.asarray(v_t),
        ckpt=str(args.ckpt),
        n_action_steps=policy.config.n_action_steps,
        seed=args.seed,
    )
    print(f"\n{args.tag}: success {np.mean(success):.3f} over {args.n} episodes -> {out}")


# ---------------------------------------------------------------------------------------- analyze
def knn_dist(query: np.ndarray, ref: np.ndarray, k: int, device: str, mode: str = "full") -> np.ndarray:
    q = torch.from_numpy(feature(query, mode)).to(device)
    r = torch.from_numpy(feature(ref, mode)).to(device)
    out = []
    for chunk in q.split(512):
        d = torch.cdist(chunk, r)
        out.append(d.topk(k, largest=False).values.mean(1).cpu())
    return torch.cat(out).numpy()


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def stage_analyze(args) -> None:
    tr = np.load(args.work / "train_states.npz")["states"]
    sw = np.load(args.work / f"sweep_{args.tag}.npz")
    starts, success, max_cov = sw["starts"], sw["success"].astype(bool), sw["max_cov"]
    n = len(starts)
    dist = knn_dist(starts, tr, args.k, args.device, args.feature)

    print(
        f"[{args.tag}] n={n}  overall success {success.mean():.3f}  "
        f"(kNN k={args.k}, feature={args.feature}, over {len(tr)} train frames)"
    )
    q = np.quantile(dist, np.linspace(0, 1, args.bins + 1))
    bin_id = np.clip(np.searchsorted(q, dist, side="right") - 1, 0, args.bins - 1)
    print(
        f"{'density bin (0=densest)':>24} {'kNN dist':>14} {'n':>5} {'success':>8} {'95% CI':>15} {'reach.90':>9}"
    )
    rows = []
    for b in range(args.bins):
        m = bin_id == b
        p = success[m].mean()
        lo, hi = wilson(p, int(m.sum()))
        rows.append((b, dist[m].mean(), m.sum(), p, lo, hi, (max_cov[m] >= 0.9).mean()))
        print(
            f"{b:>24d} {q[b]:6.1f}-{q[b + 1]:<7.1f} {m.sum():5d} {p:8.3f}   [{lo:.2f},{hi:.2f}] {rows[-1][6]:9.3f}"
        )
    # Spearman of density vs outcome, without scipy
    rk = np.argsort(np.argsort(dist))
    rho = np.corrcoef(rk, max_cov)[0, 1]  # rank(dist) vs max_cov (Pearson on one ranked side)
    # A fixed --ood-min-dist is the pre-registered split (threshold chosen on a discovery sweep,
    # applied unchanged to a held-out one); without it, fall back to the sparsest quantile bin.
    ood = dist > args.ood_min_dist if args.ood_min_dist else bin_id == args.bins - 1
    ood_name = f"kNN dist > {args.ood_min_dist}" if args.ood_min_dist else f"sparsest 1/{args.bins}"
    ind = bin_id == 0
    p_ood, p_ind = success[ood].mean(), success[ind].mean()
    print(f"\ncorr(rank kNN dist, max_cov) = {rho:+.3f}  (negative = sparser starts do worse)")
    lo, hi = wilson(p_ood, int(ood.sum()))
    print(
        f"OOD split ({ood_name}): n={ood.sum()} success {p_ood:.3f} CI [{lo:.2f},{hi:.2f}]  "
        f"lost(max_cov<0.5) {(max_cov[ood] < 0.5).mean():.3f}"
    )
    lo, hi = wilson(p_ind, int(ind.sum()))
    print(f"in-dist split (densest 1/{args.bins}): n={ind.sum()} success {p_ind:.3f} CI [{lo:.2f},{hi:.2f}]")
    passed = p_ood <= args.gate_success and ood.sum() >= args.gate_min_inits
    print(
        f"T0.1 GATE (OOD success <= {args.gate_success} with >= {args.gate_min_inits} starts): "
        f"{'PASS' if passed else 'FAIL'}"
    )

    np.savez_compressed(
        args.work / f"ood_split_{args.tag}.npz",
        ood_starts=starts[ood],
        indist_starts=starts[ind],
        ood_knn=dist[ood],
        indist_knn=dist[ind],
        knn_all=dist,
        bin_edges=q,
        k=args.k,
        feature=args.feature,
        ood_min_dist=args.ood_min_dist,
        angle_radius_px=ANGLE_RADIUS_PX,
        state_layout="agent_x agent_y block_cog_x block_cog_y block_theta (sim px / rad)",
    )
    v_ep = sw["v_ep"]
    np.savez_compressed(
        args.work / f"value_table_{args.tag}.npz",
        states=sw["v_states"],
        t=sw["v_t"],
        mc_success=success[v_ep],
        episode=v_ep,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing -> skipped figure (run with `uv run --with matplotlib`)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    r = np.array(rows, dtype=float)
    ax[0].errorbar(r[:, 0], r[:, 3], yerr=[r[:, 3] - r[:, 4], r[:, 5] - r[:, 3]], fmt="o-", capsize=3)
    ax[0].set_xlabel("density bin (0 = densest)")
    ax[0].set_ylabel("success rate")
    ax[0].set_title(f"{args.tag}: success vs start-state density")
    ax[1].scatter(dist, max_cov, c=success, s=6, cmap="coolwarm_r")
    ax[1].set_xlabel(f"kNN dist to train states (k={args.k})")
    ax[1].set_ylabel("episode max coverage")
    ax[2].scatter(starts[:, 2], starts[:, 3], c=dist, s=8, cmap="viridis")
    ax[2].scatter(
        starts[ood & ~success, 2], starts[ood & ~success, 3], s=14, facecolors="none", edgecolors="r"
    )
    ax[2].set_title("start block CoG, color=kNN dist, red ring = OOD fail")
    ax[2].invert_yaxis()
    fig.tight_layout()
    png = args.work / f"ood_failure_map_{args.tag}.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["density", "sweep", "analyze"])
    p.add_argument("--work", type=Path, default=Path("examples/nanowm_pusht_aug/cl/ood"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # density
    p.add_argument("--repo-id", default="lerobot/pusht")
    # sweep
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--tag", default="control")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-action-steps", type=int, default=0, help="0 = keep the checkpoint's value")
    p.add_argument("--seed", type=int, default=500_000, help="reset seeds: seed + episode index")
    p.add_argument("--v-stride", type=int, default=10, help="record a V-table state every N sim steps")
    p.add_argument("--start-source", choices=["reset", "ood"], default="reset")
    p.add_argument("--ood-quantile", type=float, default=0.95)
    p.add_argument(
        "--ood-range",
        choices=["reset", "wide", "topband"],
        default="reset",
        help="block CoG sampling box; topband = CoG y in [100,145), no kNN filter",
    )
    # analyze
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--feature", choices=["full", "block"], default="full")
    p.add_argument("--ood-min-dist", type=float, default=0.0, help="fixed OOD threshold (0 = sparsest bin)")
    p.add_argument("--gate-success", type=float, default=0.40)
    p.add_argument("--gate-min-inits", type=int, default=100)
    args = p.parse_args()
    {"density": stage_density, "sweep": stage_sweep, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
