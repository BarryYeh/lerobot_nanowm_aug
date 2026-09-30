#!/usr/bin/env python
"""NanoWM <-> diffusion-policy action converter (Task 1).

The diffusion policy speaks ABSOLUTE agent target positions (pixels, [0,512]).
NanoWM speaks RELATIVE per-10Hz-step displacements: raw pixels /100, then
(a - a_mean) / a_std, reshaped to a (num_frames, frame_interval*2) window.
Conversion needs the agent position at each step: rel[t] = abs_target[t] - agent_pos[t]
(both envs do `pd_target = agent.position + a` for a relative action).

`pd_step` here is the block-INDEPENDENT kinematic PD used by imagined_rollout.py's
pd_replay_agent (explicit semi-implicit Euler). The real envs integrate the agent Body
through pymunk `space.step`, and while the agent is in contact with the T block the
block's reaction force perturbs it -- so this is an APPROXIMATION, only exact for a
free (non-contacting) agent. It is nonetheless the only tractable agent-position
estimate inside the closed loop (no real sim is running there); the Task-4 real-sim
calibration gate is what verifies the approximation is good enough.

Where an exact agent position IS available (saving the aug action label = the policy's
own target; calibration replay = feed targets straight to gym_pusht) no pd_step is used.

Pure numpy, no torch / no repo imports -> loads in either venv.

Sources:
  nano-world-model pusht_env.step (relative action)
  gym_pusht PushTEnv.step (absolute action)
  imagined_rollout.py pd_replay_agent
  imagined_rollout.py closed_loop action window
"""

from __future__ import annotations

import numpy as np

# --- constants (verified against pusht_env.py) ---
ACTION_SCALE = 100.0
KP, KV = 100.0, 20.0
SIM_HZ, CONTROL_HZ = 100, 10
DT = 1.0 / SIM_HZ  # 0.01
SUBSTEPS = SIM_HZ // CONTROL_HZ  # 10

# NanoWM checkpoint dataset action stats (rel_actions / ACTION_SCALE), from
# nano-world-model ckpt/config.yaml pusht_noise train split. Override via load_action_stats.
NANOWM_A_MEAN = np.array([-0.007812563329935074, 0.006860687397420406], dtype=np.float64)
NANOWM_A_STD = np.array([0.2084684818983078, 0.20674966275691986], dtype=np.float64)
NANOWM_NUM_FRAMES = 4
NANOWM_FRAME_INTERVAL = 5


def pd_step(pos: np.ndarray, vel: np.ndarray, abs_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One 10 Hz control step = SUBSTEPS sub-steps of the block-independent kinematic PD
    (explicit Euler), matching pd_replay_agent. Approximates the real pymunk integration."""
    pos = pos.astype(np.float64).copy()
    vel = vel.astype(np.float64).copy()
    for _ in range(SUBSTEPS):
        acc = KP * (abs_target - pos) + KV * (-vel)
        vel = vel + acc * DT
        pos = pos + vel * DT
    return pos, vel


def abs_targets_to_rel(
    abs_targets: np.ndarray, pos0: np.ndarray, vel0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """abs_targets (T,2) -> (rel_px (T,2) raw pixels, agent_pos (T,2), agent_vel (T,2)).

    rel_px[t] = abs_targets[t] - agent_pos[t]   (the env does `action = agent.position + a`)
    agent_pos[t] is the position BEFORE issuing target t (agent_pos[0] == pos0).
    """
    abs_targets = np.asarray(abs_targets, dtype=np.float64)
    T = len(abs_targets)
    agent_pos = np.empty((T, 2), dtype=np.float64)
    agent_vel = np.empty((T, 2), dtype=np.float64)
    p, v = np.asarray(pos0, np.float64).copy(), np.asarray(vel0, np.float64).copy()
    rel = np.empty((T, 2), dtype=np.float64)
    for t in range(T):
        agent_pos[t], agent_vel[t] = p, v
        rel[t] = abs_targets[t] - p
        p, v = pd_step(p, v, abs_targets[t])
    return rel, agent_pos, agent_vel


def rel_to_abs_targets(rel_px: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    """Inverse of the rel step: abs_targets[t] = agent_pos[t] + rel_px[t]."""
    return np.asarray(agent_pos, np.float64) + np.asarray(rel_px, np.float64)


def rel_to_nanowm(
    rel_px: np.ndarray,
    a_mean: np.ndarray = NANOWM_A_MEAN,
    a_std: np.ndarray = NANOWM_A_STD,
    num_frames: int = NANOWM_NUM_FRAMES,
    frame_interval: int = NANOWM_FRAME_INTERVAL,
) -> np.ndarray:
    """rel_px (raw pixels, T == num_frames*frame_interval, 2) -> (num_frames, frame_interval*2)
    normalized window, matching imagined_rollout.py:216-222.
    Pads by holding the last row when T < num_frames*frame_interval."""
    need = num_frames * frame_interval
    rel = np.asarray(rel_px, np.float64)
    if len(rel) < need:
        rel = np.concatenate([rel, np.repeat(rel[-1:], need - len(rel), axis=0)], axis=0)
    rel = rel[:need]
    norm = (rel / ACTION_SCALE - a_mean) / a_std
    return norm.reshape(num_frames, frame_interval * 2)


def nanowm_to_rel(
    win: np.ndarray,
    a_mean: np.ndarray = NANOWM_A_MEAN,
    a_std: np.ndarray = NANOWM_A_STD,
) -> np.ndarray:
    """(num_frames, frame_interval*2) normalized window -> rel_px (T,2) raw pixels."""
    win = np.asarray(win, np.float64).reshape(-1, 2)
    return (win * a_std + a_mean) * ACTION_SCALE


def load_action_stats(nanowm_root: str | None = None, cache_npz: str | None = None):
    """Read (a_mean, a_std, num_frames, frame_interval) from the NanoWM training dataset.
    Falls back to the hardcoded NANOWM_* constants. Only importable in the nanowm venv."""
    if cache_npz is not None:
        try:
            d = np.load(cache_npz)
            return d["a_mean"], d["a_std"], int(d["num_frames"]), int(d["frame_interval"])
        except Exception:  # nosec B110 -- optional cache; any failure falls through to the live read
            pass
    try:
        import os
        import sys
        from pathlib import Path

        vendored = Path(__file__).resolve().parents[3] / "third_party" / "nano-world-model"
        nr = Path(nanowm_root or vendored).expanduser()
        sys.path.insert(0, str(nr / "src"))
        os.environ.setdefault(
            "DATASET_DIR", str(Path("~/Downloads/bmw48-osfstorage-datasets-archive").expanduser())
        )
        from omegaconf import OmegaConf
        from wm_datasets import create_train_val_datasets

        cfg = OmegaConf.load(nr / "ckpt" / "config.yaml")
        nf, fi, img = cfg.model.num_frames, int(cfg.dataset.frame_interval), int(cfg.model.image_size)
        tr, va = create_train_val_datasets(
            dataset_name=cfg.dataset.name,
            num_frames=nf,
            frame_interval=fi,
            image_size=img,
            **OmegaConf.to_container(cfg.dataset.loader, resolve=True),
        )
        b = tr if tr.data_source.get_num_trajectories() >= va.data_source.get_num_trajectories() else va
        a_mean = b._raw_action_mean.cpu().numpy().astype(np.float64)
        a_std = b._raw_action_std.cpu().numpy().astype(np.float64)
        if cache_npz is not None:
            np.savez(cache_npz, a_mean=a_mean, a_std=a_std, num_frames=nf, frame_interval=fi)
        return a_mean, a_std, nf, fi
    except Exception as e:  # noqa: BLE001
        print(f"[load_action_stats] falling back to hardcoded constants ({e})")
        return NANOWM_A_MEAN, NANOWM_A_STD, NANOWM_NUM_FRAMES, NANOWM_FRAME_INTERVAL


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    pos0 = np.array([256.0, 300.0])
    vel0 = np.array([0.0, 0.0])
    abs_targets = pos0 + rng.normal(0, 40, size=(20, 2))

    # 1. round trip abs -> rel -> abs (given the agent trajectory)
    rel, agent_pos, _ = abs_targets_to_rel(abs_targets, pos0, vel0)
    back = rel_to_abs_targets(rel, agent_pos)
    err1 = np.abs(back - abs_targets).max()
    print(f"[1] abs->rel->abs max err = {err1:.2e}   {'OK' if err1 < 1e-6 else 'FAIL'}")

    # 1b. nanowm window round trip
    win = rel_to_nanowm(rel)
    rel_b = nanowm_to_rel(win)
    err1b = np.abs(rel_b - rel).max()
    print(
        f"[1b] rel->nanowm->rel max err = {err1b:.2e}   {'OK' if err1b < 1e-6 else 'FAIL'}   win shape {win.shape}"
    )

    # 2. free-PD approximation vs gym_pusht's real pymunk integration (agent in free space,
    #    block parked far away). Quantifies the Euler-vs-pymunk gap for a non-contacting agent.
    try:
        import gym_pusht  # noqa: F401
        import gymnasium as gym

        env = gym.make("gym_pusht/PushT-v0", obs_type="state")
        env.reset(seed=0)
        u = env.unwrapped
        u.agent.position = (float(pos0[0]), float(pos0[1]))
        u.agent.velocity = (0.0, 0.0)
        u.block.position = (30.0, 30.0)  # park the block out of the agent's path
        u.block.velocity = (0.0, 0.0)
        u.space.step(u.dt)
        start = np.array(u.agent.position)
        tgts = start + rng.normal(0, 30, size=(20, 2))
        env_pos = []
        for t in range(len(tgts)):
            env.step(tgts[t].astype(np.float64))  # gym_pusht action is ABSOLUTE
            env_pos.append(np.array(u.agent.position))
        env_pos = np.array(env_pos)

        p, v = start.copy(), np.zeros(2)
        ours_post = []
        for t in range(len(tgts)):
            p, v = pd_step(p, v, tgts[t])
            ours_post.append(p.copy())
        ours_post = np.array(ours_post)
        err2 = np.abs(ours_post - env_pos).max()
        drift = np.abs(ours_post - env_pos)[-1].max()
        print(
            f"[2] free-PD vs gym_pusht pymunk: max err {err2:.2f} px, final-step {drift:.2f} px "
            f"  {'OK' if err2 < 5 else 'CHECK (approx too loose?)'}"
        )
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[2] skipped ({e})")


if __name__ == "__main__":
    _selftest()
