#!/usr/bin/env python
"""Evaluate an image-only PushT diffusion policy at the correct 2 Hz control rate,
on one of two initial-state distributions, and dump per-episode coverage to JSON.

Why 2 Hz: the policy trained on `pusht_noise_224_fi5*` (fps 2 = 1 action per 5 sim-
steps, `--action-at end` absolute target). `lerobot-eval` drives gym_pusht at its
native 10 Hz -> 5x under-actuation -> misleading score. Here each policy action is
held for 5 `env.step` calls (5 x 10 Hz = 2 Hz).

Two eval distributions (pick one):
  --val-inits <states.pth> --rollouts-per-init N
        reset to each of the 21 held-out pusht_noise/val initial states, N rollouts
        each. Held out from BOTH the world model and the policy; the fairest test
        distribution. Paired: init i / rollout r uses a deterministic per-episode
        seed so `a` and `a+aug` face identical conditions.
  --random-inits N
        N fresh random resets (gym_pusht's own reset, same ranges as the pusht_noise
        env). Broad but higher variance. Paired by episode index via the seed.

`reset_to_pusht_state` corrects gym_pusht's `_set_state`: it sets body.position then
body.angle, and the T-block's offset centre-of-gravity makes the angle-set rotate
the body -> up to ~90px block displacement (angle-dependent). Setting angle first,
then position, then zeroing velocity + one space.step reproduces the state exactly
(verified: 0.0px offset, pixel MAE ~0.3 vs the pusht_noise render).

Output: <out> (JSON)
  {config, ckpt, distribution, seed, max_ticks, success_threshold,
   rollouts_per_init | n_episodes,
   episodes: [{init_id, rollout_id, max_cov, final_cov}, ...]}
init_id = -1 for random resets.

Run:
    uv run --extra training python examples/nanowm_pusht_aug/eval/eval_baseline.py \
        outputs/train/nanowm_aug/baseline_a_obs2_s100000/checkpoints/last/pretrained_model \
        --val-inits ~/Downloads/bmw48-osfstorage-datasets-archive/pusht_noise/val/states.pth \
        --rollouts-per-init 5 --out outputs/eval/nanowm_aug/a_valinit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gym_pusht  # noqa: F401  (registers gym_pusht/PushT-v0)
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

SIM_STEPS_PER_ACTION = 5  # 5 x 10 Hz = 2 Hz, matches --frame-interval 5


def reset_to_pusht_state(env, state5: np.ndarray, seed: int):
    """Reset then re-apply the pose in the right order (angle before position) so the
    T-block's offset-CoG rotation doesn't displace it. Returns the pixels obs."""
    env.reset(seed=seed)
    u = env.unwrapped
    u.agent.position = tuple(float(x) for x in state5[:2])
    u.block.angle = float(state5[4])
    u.block.position = tuple(float(x) for x in state5[2:4])
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)
    return u.get_obs()


def rollout_once(
    policy, pre, post, env, obs, max_ticks: int, device: str, frames: list | None = None
) -> tuple[float, float]:
    """One episode from a given initial obs. Returns (max_cov, final_cov).
    If `frames` is a list, appends the 10 Hz rendered frames to it (for video)."""
    policy.reset()
    max_cov, last_cov = 0.0, 0.0
    if frames is not None:
        frames.append(np.asarray(obs))
    for _ in range(max_ticks):
        img = torch.from_numpy(np.asarray(obs)).permute(2, 0, 1).float().div(255.0)[None]
        batch = pre({"observation.image": img})
        with torch.no_grad():
            raw = policy.select_action(batch)
        target = post(raw)[0].detach().cpu().numpy().astype(np.float64)

        stop = False
        for _ in range(SIM_STEPS_PER_ACTION):
            obs, _r, terminated, truncated, info = env.step(target)
            last_cov = float(info.get("coverage", last_cov))
            max_cov = max(max_cov, last_cov)
            if frames is not None:
                frames.append(np.asarray(obs))
            if terminated or truncated:
                stop = True
                break
        if stop:
            break
    return max_cov, last_cov


def run(
    ckpt: Path,
    distribution: str,
    seed: int,
    max_ticks: int,
    device: str,
    success_threshold: float,
    val_states: np.ndarray | None,
    rollouts_per_init: int,
    n_random: int,
    video_dir: Path | None = None,
    videos_per_config: int = 12,
) -> dict:
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels",
        observation_width=224,
        observation_height=224,
        max_episode_steps=max_ticks * SIM_STEPS_PER_ACTION + 10,
    )
    env.unwrapped.success_threshold = success_threshold

    config = ckpt.parents[2].name if len(ckpt.parents) >= 3 else "policy"
    vdir = (video_dir / f"{config}__{distribution}") if video_dir else None
    if vdir:
        vdir.mkdir(parents=True, exist_ok=True)
    n_vid = 0

    def maybe_save_video(tag: str, mx: float, frames: list) -> None:
        nonlocal n_vid
        if vdir and frames:
            imageio.mimwrite(vdir / f"{tag}_mx{mx:.2f}.mp4", frames, fps=15, codec="libx264")
            n_vid += 1

    episodes = []
    if distribution == "val_init":
        n_inits = len(val_states)
        for i in range(n_inits):
            for r in range(rollouts_per_init):
                ep_seed = seed * 100_000 + i * 100 + r
                torch.manual_seed(ep_seed)
                obs = reset_to_pusht_state(env, val_states[i], ep_seed)
                frames = [] if (vdir and n_vid < videos_per_config) else None
                mx, fn = rollout_once(policy, pre, post, env, obs, max_ticks, device, frames)
                maybe_save_video(f"i{i:02d}_r{r}", mx, frames or [])
                episodes.append({"init_id": i, "rollout_id": r, "max_cov": mx, "final_cov": fn})
            done = (i + 1) * rollouts_per_init
            print(
                f"  {done}/{n_inits * rollouts_per_init}  running mean max_cov="
                f"{np.mean([e['max_cov'] for e in episodes]):.3f}"
            )
    else:  # random
        for e in range(n_random):
            ep_seed = seed * 100_000 + e
            torch.manual_seed(ep_seed)
            obs, _ = env.reset(seed=ep_seed)
            frames = [] if (vdir and n_vid < videos_per_config) else None
            mx, fn = rollout_once(policy, pre, post, env, obs, max_ticks, device, frames)
            maybe_save_video(f"e{e:03d}", mx, frames or [])
            episodes.append({"init_id": -1, "rollout_id": e, "max_cov": mx, "final_cov": fn})
            if (e + 1) % 10 == 0:
                print(
                    f"  {e + 1}/{n_random}  running mean max_cov="
                    f"{np.mean([x['max_cov'] for x in episodes]):.3f}"
                )
    env.close()
    if vdir:
        print(f"  saved {n_vid} videos -> {vdir}")

    return {
        "config": ckpt.parents[2].name if len(ckpt.parents) >= 3 else str(ckpt),
        "ckpt": str(ckpt),
        "distribution": distribution,
        "seed": seed,
        "max_ticks": max_ticks,
        "success_threshold": success_threshold,
        "rollouts_per_init": rollouts_per_init if distribution == "val_init" else None,
        "n_episodes": len(episodes),
        "episodes": episodes,
    }


def summarize(res: dict) -> None:
    mx = np.array([e["max_cov"] for e in res["episodes"]])
    fn = np.array([e["final_cov"] for e in res["episodes"]])
    thresh = res["success_threshold"]
    print(f"\n{res['config']}  [{res['distribution']}]  n={res['n_episodes']}")
    print(
        f"  max_cov  : mean={mx.mean():.3f} median={np.median(mx):.3f} "
        f"p25/75/90={np.percentile(mx, 25):.2f}/{np.percentile(mx, 75):.2f}/{np.percentile(mx, 90):.2f}"
    )
    print(f"  final_cov: mean={fn.mean():.3f} median={np.median(fn):.3f}")
    for t in (0.5, 0.7, 0.9, 0.95):
        print(f"  frac(max_cov>={t}) = {(mx >= t).mean():.3f}")
    print(f"  success_rate(max_cov>={thresh}) = {(mx >= thresh).mean():.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path, help="<run>/checkpoints/<step>/pretrained_model")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--val-inits", type=Path, help="pusht_noise/val/states.pth")
    g.add_argument("--random-inits", type=int, help="number of random-reset episodes")
    p.add_argument("--rollouts-per-init", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-ticks", type=int, default=60, help="2 Hz ticks/episode (60 = 300 sim-steps)")
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--out", type=Path, required=True, help="output JSON path")
    p.add_argument(
        "--save-videos",
        type=Path,
        default=None,
        help="dir to write per-episode 10 Hz mp4s (first N episodes)",
    )
    p.add_argument("--videos-per-config", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.val_inits is not None:
        st = torch.load(args.val_inits, weights_only=True).float().numpy()
        val_states = st[:, 0, :5]
        res = run(
            args.ckpt,
            "val_init",
            args.seed,
            args.max_ticks,
            args.device,
            args.success_threshold,
            val_states,
            args.rollouts_per_init,
            0,
            args.save_videos,
            args.videos_per_config,
        )
    else:
        res = run(
            args.ckpt,
            "random",
            args.seed,
            args.max_ticks,
            args.device,
            args.success_threshold,
            None,
            args.rollouts_per_init,
            args.random_inits,
            args.save_videos,
            args.videos_per_config,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    summarize(res)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
