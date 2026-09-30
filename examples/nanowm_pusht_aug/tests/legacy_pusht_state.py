"""Frozen verbatim placement/read-back copies from eddd92c3 (not collected by pytest)."""

from __future__ import annotations

import numpy as np


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/closed_loop_rollout.py:216-222
def legacy_closed_loop(env, seed, st) -> None:
    u = env.unwrapped
    env.reset(seed=seed)
    u.agent.position = (float(st[0]), float(st[1]))
    u.block.angle = float(st[4])
    u.block.position = (float(st[2]), float(st[3]))
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/closed_loop_rollout.py:234-240
def legacy_closed_loop_anchor_read(u) -> np.ndarray:
    return np.array(
        [
            u.agent.position[0],
            u.agent.position[1],
            u.block.position[0],
            u.block.position[1],
            u.block.angle,
        ],
        np.float64,
    )


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/make_topband_seeds.py:47-54
def legacy_place_like_closed_loop(u, st: np.ndarray) -> None:
    """Copy of closed_loop_rollout.py's seed placement (agent, angle, body position, one step)."""
    u.agent.position = (float(st[0]), float(st[1]))
    u.block.angle = float(st[4])
    u.block.position = (float(st[2]), float(st[3]))
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/critic/label_imagined_branches.py:84-93
def legacy_place_seed(env, seed_index: int, st: np.ndarray) -> None:
    """The driver's seed placement, verbatim (closed_loop_rollout.py lead-in, BODY-origin state)."""
    u = env.unwrapped
    env.reset(seed=9000 + seed_index)
    u.agent.position = (float(st[0]), float(st[1]))
    u.block.angle = float(st[4])
    u.block.position = (float(st[2]), float(st[3]))
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/critic/label_imagined_branches.py:75-77
def legacy_body_state(u) -> np.ndarray:
    """Body-origin sim state in the SAME form as the driver's `anchor_state`."""
    return np.array([*u.agent.position, u.block.position[0], u.block.position[1], u.block.angle], np.float64)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/proto_filter_roc.py:61-67
def legacy_proto_filter_roc(env, st) -> None:
    u = env.unwrapped
    env.reset(seed=0)
    u.agent.position = tuple(float(x) for x in st[:2])
    u.block.angle = float(st[4])
    u.block.position = tuple(float(x) for x in st[2:4])
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/eval_baseline.py:57-68
def legacy_reset_to_pusht_state(env, state5: np.ndarray, seed: int):
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


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/proto_make_seeds.py:74-76
def legacy_proto_make_seeds_read(u) -> np.ndarray:
    return np.array([*u.agent.position, *u.block.position, u.block.angle % (2 * np.pi)], dtype=np.float64)


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/ood_failure_map.py:113-127
def legacy_set_state(u, state: np.ndarray):
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


# verbatim from eddd92c3:examples/nanowm_pusht_aug/eval/ood_failure_map.py:72-74
def legacy_sim_state(u) -> np.ndarray:
    cog = u.block.local_to_world(u.block.center_of_gravity)
    return np.array([*u.agent.position, cog.x, cog.y, u.block.angle % (2 * np.pi)], np.float64)
