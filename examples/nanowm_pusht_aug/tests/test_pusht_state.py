"""Bit-equivalence of nwaug.pusht_state vs frozen legacy placements."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")
import gymnasium as gym  # noqa: E402
import pymunk  # noqa: E402
from legacy_pusht_state import (  # noqa: E402
    legacy_body_state,
    legacy_closed_loop,
    legacy_closed_loop_anchor_read,
    legacy_place_like_closed_loop,
    legacy_place_seed,
    legacy_proto_filter_roc,
    legacy_proto_make_seeds_read,
    legacy_reset_to_pusht_state,
    legacy_set_state,
    legacy_sim_state,
)
from make_topband_seeds import cog_to_body  # noqa: E402
from nwaug.pusht_state import (  # noqa: E402
    load_seeds,
    place_body,
    place_cog,
    read_body,
    read_cog,
    save_seeds,
)
from ood_failure_map import sample_topband_starts  # noqa: E402


def full_state(u) -> np.ndarray:
    return np.array(
        [
            *u.agent.position,
            *u.agent.velocity,
            *u.block.position,
            u.block.angle,
            *u.block.velocity,
            u.block.angular_velocity,
        ],
        np.float64,
    )


def _make(obs_type: str, size: int | None = None):
    kw: dict = {"obs_type": obs_type}
    if size is not None:
        kw["observation_width"] = size
        kw["observation_height"] = size
    return gym.make("gym_pusht/PushT-v0", **kw)


def _eq_obs(a, b) -> None:
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            aa, bb = np.asarray(a[k]), np.asarray(b[k])
            assert np.array_equal(aa, bb), k
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b))


def _step_equal(env_a, env_b, n: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = rng.uniform(0, 512, size=2)
        env_a.step(a.copy())
        env_b.step(a.copy())
        assert np.array_equal(full_state(env_a.unwrapped), full_state(env_b.unwrapped))


def _body_states() -> list[tuple[str, np.ndarray]]:
    """Named sets of body-origin states."""
    out: list[tuple[str, np.ndarray]] = []
    # (a) reset read-backs
    env = _make("state")
    rows = []
    for s in range(100):
        env.reset(seed=s)
        rows.append(legacy_body_state(env.unwrapped))
    env.close()
    out.append(("reset", np.stack(rows)))
    # (b) topband -> body
    cog = sample_topband_starts(100, 123)
    # need cog_local from a live env
    env = _make("state")
    env.reset(seed=0)
    cog_local = pymunk.Vec2d(*env.unwrapped.block.center_of_gravity)
    env.close()
    out.append(("topband_body", cog_to_body(cog, cog_local)))
    # (c) uniform random incl overlap
    rng = np.random.default_rng(0)
    rnd = rng.uniform([0, 0, 0, 0, -1], [512, 512, 512, 512, 3 * np.pi], size=(100, 5))
    out.append(("random", rnd))
    # (d) angles outside [0, 2pi)
    ang = rnd.copy()
    ang[:, 4] = np.concatenate([rng.uniform(-4, -0.1, 50), rng.uniform(2 * np.pi + 0.1, 8, 50)])
    out.append(("angles", ang))
    return out


def _cog_states() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    out.append(("topband_cog", sample_topband_starts(100, 123)))
    rng = np.random.default_rng(1)
    rnd = rng.uniform([50, 50, 100, 100, -1], [450, 450, 400, 400, 8], size=(100, 5))
    out.append(("random_cog", rnd))
    ang = rnd.copy()
    ang[:, 4] = np.concatenate([rng.uniform(-3, -0.1, 50), rng.uniform(7, 12, 50)])
    out.append(("angles_cog", ang))
    return out


@pytest.fixture(scope="module")
def body_sets():
    return _body_states()


@pytest.fixture(scope="module")
def cog_sets():
    return _cog_states()


ENV_KINDS = [
    ("pixels_agent_pos", 96),
    ("pixels", 224),
    ("state", None),
]


@pytest.mark.parametrize("kind", ENV_KINDS, ids=["pap96", "pix224", "state"])
def test_place_body_vs_legacy_closed_loop(body_sets, kind) -> None:
    obs_type, size = kind
    for name, states in body_sets:
        for i, st in enumerate(states[:25] if name != "reset" else states):
            seed = [0, 9000 + i, 100000 + i][i % 3]
            ea, eb = _make(obs_type, size), _make(obs_type, size)
            legacy_closed_loop(ea, seed, st)
            eb.reset(seed=seed)
            place_body(eb.unwrapped, st)
            assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped)), (name, i)
            _eq_obs(ea.unwrapped.get_obs(), eb.unwrapped.get_obs())
            _step_equal(ea, eb, seed=i)
            ea.close()
            eb.close()


@pytest.mark.parametrize("kind", ENV_KINDS, ids=["pap96", "pix224", "state"])
def test_place_body_vs_place_like_and_label(body_sets, kind) -> None:
    obs_type, size = kind
    states = body_sets[0][1]  # reset set
    for i, st in enumerate(states[:30]):
        seed = 9000 + i
        # place_like (no reset in legacy helper) -- both reset first
        ea, eb = _make(obs_type, size), _make(obs_type, size)
        ea.reset(seed=seed)
        eb.reset(seed=seed)
        legacy_place_like_closed_loop(ea.unwrapped, st)
        place_body(eb.unwrapped, st)
        assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped))
        _step_equal(ea, eb, seed=i)
        ea.close()
        eb.close()
        # place_seed
        ea, eb = _make(obs_type, size), _make(obs_type, size)
        legacy_place_seed(ea, i, st)
        eb.reset(seed=9000 + i)
        place_body(eb.unwrapped, st)
        assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped))
        ea.close()
        eb.close()


@pytest.mark.parametrize("kind", ENV_KINDS, ids=["pap96", "pix224", "state"])
def test_place_body_vs_proto_filter_and_eval_baseline(body_sets, kind) -> None:
    obs_type, size = kind
    states = body_sets[0][1]
    for i, st in enumerate(states[:30]):
        ea, eb = _make(obs_type, size), _make(obs_type, size)
        legacy_proto_filter_roc(ea, st)
        eb.reset(seed=0)
        place_body(eb.unwrapped, st)
        assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped))
        ea.close()
        eb.close()
        ea, eb = _make(obs_type, size), _make(obs_type, size)
        legacy_reset_to_pusht_state(ea, st, seed=i)
        eb.reset(seed=i)
        place_body(eb.unwrapped, st)
        assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped))
        _eq_obs(ea.unwrapped.get_obs(), eb.unwrapped.get_obs())
        ea.close()
        eb.close()


@pytest.mark.parametrize("kind", ENV_KINDS, ids=["pap96", "pix224", "state"])
def test_place_cog_vs_legacy_set_state(cog_sets, kind) -> None:
    obs_type, size = kind
    for name, states in cog_sets:
        for i, st in enumerate(states[:30]):
            seed = [0, 9000 + i, 100000 + i][i % 3]
            ea, eb = _make(obs_type, size), _make(obs_type, size)
            ea.reset(seed=seed)
            eb.reset(seed=seed)
            obs_a = legacy_set_state(ea.unwrapped, st)
            obs_b = place_cog(eb.unwrapped, st)
            assert np.array_equal(full_state(ea.unwrapped), full_state(eb.unwrapped)), (name, i)
            _eq_obs(obs_a, obs_b)
            _step_equal(ea, eb, seed=i)
            ea.close()
            eb.close()


def test_read_body_and_cog(body_sets, cog_sets) -> None:
    env = _make("state")
    for i, st in enumerate(body_sets[0][1][:50]):
        env.reset(seed=i)
        place_body(env.unwrapped, st)
        u = env.unwrapped
        assert np.array_equal(read_body(u), legacy_body_state(u))
        assert np.array_equal(read_body(u), legacy_closed_loop_anchor_read(u))
        wrapped = read_body(u).copy()
        wrapped[4] %= 2 * np.pi
        assert np.array_equal(wrapped, legacy_proto_make_seeds_read(u))
    for i, st in enumerate(cog_sets[0][1][:50]):
        env.reset(seed=i)
        place_cog(env.unwrapped, st)
        u = env.unwrapped
        assert np.array_equal(read_cog(u), legacy_sim_state(u))
    env.close()


def test_save_load_seeds(tmp_path) -> None:
    body = np.arange(20, dtype=np.float64).reshape(4, 5)
    cog = body + 0.5
    path = tmp_path / "seeds.npz"
    save_seeds(path, body=body, cog=cog)
    assert np.array_equal(load_seeds(path, "body"), body)
    assert np.array_equal(load_seeds(path, "cog"), cog)
    # legacy np.savez
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, states=body, cog_states=cog)
    assert np.array_equal(load_seeds(legacy, "body"), body)
    assert np.array_equal(load_seeds(legacy, "cog"), cog)
    save_seeds(tmp_path / "only_body.npz", body=body)
    with pytest.raises(KeyError):
        load_seeds(tmp_path / "only_body.npz", "cog")
    with pytest.raises(ValueError):
        save_seeds(tmp_path / "bad.npz")
    with pytest.raises(ValueError):
        save_seeds(tmp_path / "bad2.npz", body=np.zeros((3, 4)))
