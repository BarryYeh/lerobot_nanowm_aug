#!/usr/bin/env python
"""E1 -- at a top-wall-band state, does the base diffusion policy still have a mode that works?

The reposition-and-hand-back idea only needs dreaming if the policy's own multimodality is already
gone at these states. So: take one OOD observation, sample K independent action chunks from it
(the policy is a diffusion model -- each sample is a mode), execute each chunk open loop, then hand
control back to the unmodified base policy and let it run to the episode limit.

  all modes fail          -> multimodality has collapsed here; new behaviour must come from
                             injection / repositioning
  some modes succeed      -> mode selection is enough, and no out-of-distribution action is needed
                             (chunks stay at |z| ~ 2.6 sigma, where T1 showed NanoWM tracks well)

Starts come from sample_topband_starts, which now rejects blocks closer than --min-wall-gap px to a
wall (the agent is 30 px across and cannot squeeze into a smaller gap, so those states are
unrecoverable by repositioning rather than merely OOD).

Every branch forces one of the K sampled chunks; there is no "let the policy pick" arm, since that
is just a (K+1)-th draw from the same distribution and adds no information the K modes don't carry.

    P=examples/nanowm_pusht_aug
    uv run --extra training python $P/eval/mode_probe.py --n-starts 4 --k-modes 8 \
        --rollouts-per-mode 3 --out $P/cl/e1_modes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_EVAL = Path(__file__).resolve().parent
_PROJ = _EVAL.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_EVAL))
from ood_failure_map import block_wall_gap, sample_topband_starts, set_state, sim_state  # noqa: E402

BASE_CKPT = Path("outputs/train/pusht_imgonly/obs1_s100000/checkpoints/last/pretrained_model")
IMG = 224
FI = 5  # frames stored every 5 sim steps, as everywhere else in this prototype


@torch.no_grad()
def sample_modes(policy, pre, post, frame96: np.ndarray, k: int, dev: str) -> np.ndarray:
    """One 96px observation -> (k, horizon, 2) absolute-target chunks, k independent diffusion samples.

    Goes through `conditional_sample` rather than `select_action` to get the whole horizon at once
    (same call as closed_loop_rollout.sample_chunks)."""
    from lerobot.utils.constants import OBS_IMAGES

    img = torch.from_numpy(frame96).float().permute(2, 0, 1).div(255.0)[None].to(dev)
    img = pre({"observation.image": img})["observation.image"]
    x = img[:, None, None].repeat_interleave(k, dim=0)  # (k, n_obs=1, n_cam=1, 3, 96, 96)
    gcond = policy.diffusion._prepare_global_conditioning({OBS_IMAGES: x})
    return post(policy.diffusion.conditional_sample(k, global_cond=gcond)).cpu().numpy().astype(np.float64)


def main() -> None:
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=BASE_CKPT)
    p.add_argument("--n-starts", type=int, default=4)
    p.add_argument("--k-modes", type=int, default=8)
    p.add_argument("--rollouts-per-mode", type=int, default=3, help="hand-back rollouts per mode")
    p.add_argument("--exec-steps", type=int, default=8, help="chunk steps executed before handing back")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--min-wall-gap", type=float, default=45.0)
    p.add_argument("--seed", type=int, default=400_000)
    p.add_argument("--out", type=Path, default=Path("examples/nanowm_pusht_aug/cl/e1_modes"))
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--video-max-starts", type=int, default=4, help="film rollout 0 of the first N starts")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev, K, Rp = args.device, args.k_modes, args.rollouts_per_mode
    args.out.mkdir(parents=True, exist_ok=True)

    starts = sample_topband_starts(args.n_starts, args.seed, args.min_wall_gap)
    policy = DiffusionPolicy.from_pretrained(args.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": dev}},
    )
    torch.manual_seed(args.seed)

    # one branch per (start, mode, rollout)
    modes = list(range(K))
    branches = [(s, m, r) for s in range(len(starts)) for m in modes for r in range(Rp)]
    bs = len(branches)
    b_start = np.array([b[0] for b in branches])
    b_mode = np.array([b[1] for b in branches])
    b_roll = np.array([b[2] for b in branches])
    print(f"starts={len(starts)} modes={K} rollouts/mode={Rp} branches={bs} exec_steps={args.exec_steps}")

    e96 = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", observation_width=96, observation_height=96,
                    max_episode_steps=args.max_steps + 10) for _ in range(bs)]  # fmt: skip
    film = args.video and (b_roll == 0) & (b_start < args.video_max_starts)
    e224 = [gym.make("gym_pusht/PushT-v0", obs_type="pixels", observation_width=IMG, observation_height=IMG,
                     max_episode_steps=args.max_steps + 10) if (args.video and film[b]) else None
            for b in range(bs)] if args.video else None  # fmt: skip
    obs = []
    for b in range(bs):
        e96[b].reset(seed=b)
        obs.append(set_state(e96[b].unwrapped, starts[b_start[b]])["pixels"])
        if e224 and e224[b] is not None:
            e224[b].reset(seed=b)
            set_state(e224[b].unwrapped, starts[b_start[b]])

    # K chunks per start, drawn once from that start's own observation and shared by its rollouts
    chunks = np.stack([sample_modes(policy, pre, post, obs[np.nonzero(b_start == s)[0][0]], K, dev)
                       for s in range(len(starts))])  # (S,K,horizon,2)  # fmt: skip

    n_frames = args.max_steps // FI + 1
    # only rollout 0 of the first --video-max-starts starts is filmed: one frame array per branch
    # would be ~2.5 GB at 10 starts x 9 modes x 3 rollouts
    vid_slot = np.full(bs, -1)
    if args.video:
        filmed = np.nonzero((b_roll == 0) & (b_start < args.video_max_starts))[0]
        vid_slot[filmed] = np.arange(len(filmed))
    frames = np.zeros((int((vid_slot >= 0).sum()), n_frames, IMG, IMG, 3), np.uint8) if args.video else None
    states = np.zeros((bs, args.max_steps + 1, 5))
    max_cov = np.zeros(bs)
    success = np.zeros(bs, bool)
    done = np.zeros(bs, bool)
    policy.reset()
    for t in range(args.max_steps + 1):
        for b in range(bs):
            states[b, t] = sim_state(e96[b].unwrapped)
        if frames is not None and t % FI == 0:
            for b in np.nonzero(vid_slot >= 0)[0]:
                v = vid_slot[b]
                if not done[b]:
                    frames[v, t // FI] = np.asarray(e224[b].unwrapped.get_obs())
                elif t // FI > 0:
                    frames[v, t // FI] = frames[v, t // FI - 1]  # freeze finished episodes
        if t == args.max_steps:
            break
        if t == args.exec_steps:
            policy.reset()  # hand back: the queue restarts from the post-chunk observation
        img = torch.from_numpy(np.stack(obs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)
        with torch.no_grad():
            a_pol = (
                post(policy.select_action(pre({"observation.image": img}))).cpu().numpy().astype(np.float64)
            )
        for b in range(bs):
            if done[b]:
                continue
            a = chunks[b_start[b], b_mode[b], t] if t < args.exec_steps else a_pol[b]
            o, _r, term, trunc, info = e96[b].step(a)
            obs[b] = o["pixels"]
            if e224 and e224[b] is not None:
                e224[b].step(a)
            max_cov[b] = max(max_cov[b], float(info.get("coverage", 0.0)))
            if term or trunc:
                done[b] = success[b] = bool(term)
        if (t + 1) % 50 == 0:
            print(f"  step {t + 1}/{args.max_steps}  done {done.sum()}/{bs}", flush=True)
    for e in e96 + [e for e in (e224 or []) if e is not None]:
        e.close()

    esc = states[:, -1, 3] >= 145  # block left the top-wall band
    res = {"ckpt": str(args.ckpt), "exec_steps": args.exec_steps, "k_modes": K, "rollouts_per_mode": Rp,
           "min_wall_gap": args.min_wall_gap, "starts": starts.tolist(), "per_start": []}  # fmt: skip
    print(f"\n{'start':>5} {'wall gap':>9} {'mean mode':>10} {'best mode':>10} {'modes >=1 success':>18} "
          f"{'mode success spread':>20} {'escape band':>12}")  # fmt: skip
    for s in range(len(starts)):
        ms = np.array([success[(b_start == s) & (b_mode == m)].mean() for m in range(K)])
        row = {"start": s, "wall_gap_px": float(block_wall_gap(starts[s][None])[0]), "mean_mode_success": float(ms.mean()),
               "mode_success": ms.round(3).tolist(), "best_mode_success": float(ms.max()),
               "frac_modes_with_success": float((ms > 0).mean()),
               "escape_rate": float(esc[b_start == s].mean()),
               "max_cov_best": float(max_cov[b_start == s].max()),
               "chunk_endpoint_spread_px": float(np.median(np.std(chunks[s][:, args.exec_steps - 1], axis=0)))}  # fmt: skip
        res["per_start"].append(row)
        print(f"{s:>5} {row['wall_gap_px']:>9.0f} {ms.mean():>10.2f} {ms.max():>10.2f} {(ms > 0).mean():>18.2f} "
              f"{np.round(ms, 2).tolist()!s:>20} {row['escape_rate']:>12.2f}")  # fmt: skip
    overall = {"any_mode_success_rate": float(np.mean([r["best_mode_success"] > 0 for r in res["per_start"]])),
               "mean_best_mode_success": float(np.mean([r["best_mode_success"] for r in res["per_start"]])),
               "all_modes_success": float(success.mean()),
               "escape_rate": float(esc.mean())}  # fmt: skip
    res["overall"] = overall
    print(f"\nall modes pooled {overall['all_modes_success']:.2f} | mean best-mode {overall['mean_best_mode_success']:.2f} | "
          f"starts with >=1 working mode {overall['any_mode_success_rate']:.2f} | block escapes band {overall['escape_rate']:.2f}")  # fmt: skip
    (args.out / "e1_modes.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(args.out / "e1_rollouts.npz", starts=starts, chunks=chunks, states=states, success=success,
                        max_cov=max_cov, b_start=b_start, b_mode=b_mode, b_roll=b_roll)  # fmt: skip

    if frames is not None:
        import cv2
        import imageio.v2 as imageio

        cols = int(np.ceil(np.sqrt(K + 1)))
        for s in range(min(len(starts), args.video_max_starts)):
            sel = [
                int(vid_slot[np.nonzero((b_start == s) & (b_mode == m) & (b_roll == 0))[0][0]]) for m in modes
            ]
            vid = []
            for f in range(n_frames):
                cells = []
                for b, m in zip(sel, modes, strict=True):
                    im = frames[b, f].copy()
                    tag = f"mode {m}"
                    cv2.putText(im, tag, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(im, f"{'SUCCESS' if success[b] else ''}", (5, IMG - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 140, 0), 2, cv2.LINE_AA)  # fmt: skip
                    cells.append(im)
                while len(cells) % cols:
                    cells.append(np.full((IMG, IMG, 3), 255, np.uint8))
                grid = np.vstack([np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)])
                vid.append(np.vstack([np.full((24, grid.shape[1], 3), 255, np.uint8), grid]))
                cv2.putText(vid[-1], f"start s{s}  t={f * FI / 10:.1f}s", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)  # fmt: skip
            vid = [v[: v.shape[0] // 2 * 2, : v.shape[1] // 2 * 2] for v in vid]
            imageio.mimwrite(args.out / f"e1_modes_s{s}.mp4", vid, fps=6, codec="libx264")
        print(f"wrote {min(len(starts), args.video_max_starts)} videos to {args.out}")


if __name__ == "__main__":
    main()
