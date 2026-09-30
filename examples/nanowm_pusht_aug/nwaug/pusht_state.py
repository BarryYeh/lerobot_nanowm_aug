"""PushT sim-state placement and read-back (body-origin vs CoG conventions).

Two conventions for the 5-vector ``[agent_x, agent_y, block_x, block_y, angle]``:

* **body-origin** (``place_body`` / ``read_body``): ``block_x/y`` is the pymunk body
  origin. Used by closed-loop seed placement and replay asserts. Angle is NOT wrapped
  on read (matches ``label_imagined_branches.body_state`` and
  ``closed_loop_rollout`` ``anchor_state``); callers that need ``[0, 2pi)`` wrap at
  the call site (``proto_make_seeds``).
* **CoG** (``place_cog`` / ``read_cog``): ``block_x/y`` is the block centre of gravity
  in world coords (``ood_failure_map.set_state`` / ``sim_state``). Angle is wrapped
  ``% (2*pi)`` on read.

Mixing them silently shifts the T by about +45 px (the local CoG offset) -- see
``ood_failure_map.py`` ``--ood-range`` docstring. Angle is always assigned before
position because rotating an offset-CoG body about the origin would displace it.

Replaces: closed-loop / make_topband / label / proto_filter_roc / eval_baseline
inline placement (body); ``ood_failure_map.set_state`` / ``sim_state`` (CoG);
``label_imagined_branches.body_state`` and closed-loop anchor read-back (body);
seed ``np.savez`` / ``np.load`` of ``states`` / ``cog_states``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def place_body(u, s5, *, zero_agent_vel: bool = False) -> None:
    """Place agent + block from a body-origin 5-vector (closed_loop semantics).

    Does NOT reset; the caller resets with its own seed first (gym_pusht's reset
    steps the space once, so an overlapping reset can leave agent velocity non-zero).
    """
    u.agent.position = (float(s5[0]), float(s5[1]))
    if zero_agent_vel:
        u.agent.velocity = (0.0, 0.0)
    u.block.angle = float(s5[4])
    u.block.position = (float(s5[2]), float(s5[3]))
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)


def place_cog(u, s5) -> dict:
    """Place from a CoG-based 5-vector; returns ``u.get_obs()`` (pixels_agent_pos dict).

    Verbatim ``ood_failure_map.set_state``: agent vel zeroed, angle then body origin =
    CoG minus rotated local CoG offset, block vels zeroed, one space step.
    """
    import pymunk

    u.agent.position = (float(s5[0]), float(s5[1]))
    u.agent.velocity = (0.0, 0.0)
    u.block.angle = float(s5[4])
    offset = pymunk.Vec2d(*u.block.center_of_gravity).rotated(float(s5[4]))
    u.block.position = (float(s5[2]) - offset.x, float(s5[3]) - offset.y)
    u.block.velocity = (0.0, 0.0)
    u.block.angular_velocity = 0.0
    u.space.step(u.dt)
    return u.get_obs()


def read_body(u) -> np.ndarray:
    """Body-origin 5-vector; angle NOT wrapped."""
    return np.array(
        [*u.agent.position, u.block.position[0], u.block.position[1], u.block.angle],
        np.float64,
    )


def read_cog(u) -> np.ndarray:
    """CoG-world 5-vector; angle wrapped to ``[0, 2pi)``."""
    cog = u.block.local_to_world(u.block.center_of_gravity)
    return np.array([*u.agent.position, cog.x, cog.y, u.block.angle % (2 * np.pi)], np.float64)


def save_seeds(path, *, body=None, cog=None, **extra) -> None:
    """Write ``states`` (body) and/or ``cog_states`` via ``np.savez``."""
    if body is None and cog is None:
        raise ValueError("save_seeds requires at least one of body= or cog=")
    payload = dict(extra)
    if body is not None:
        body = np.asarray(body, np.float64)
        if body.ndim != 2 or body.shape[1] != 5:
            raise ValueError(f"body must be (N,5), got {body.shape}")
        payload["states"] = body
    if cog is not None:
        cog = np.asarray(cog, np.float64)
        if cog.ndim != 2 or cog.shape[1] != 5:
            raise ValueError(f"cog must be (N,5), got {cog.shape}")
        payload["cog_states"] = cog
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_seeds(path, convention: Literal["body", "cog"]) -> np.ndarray:
    """Load ``states`` (body) or ``cog_states`` (cog) as float64."""
    key = "states" if convention == "body" else "cog_states"
    d = np.load(path)
    if key not in d.files:
        raise KeyError(f"{key!r} missing in {path}; available keys: {list(d.files)}")
    return np.asarray(d[key], np.float64)
