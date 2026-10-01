# ruff: noqa: N802, N803  -- T / B / success_T follow the pre-registration notation
"""Pure-numpy label / selection rules for the SAILOR v1 value model (pre-registration §3).

Deliberately free of torch / gym / lerobot imports so it imports in both the lerobot venv and the
NanoWM venv, and so the tests that pin the rules run fully offline.

Conventions
-----------
* `T`   steps from the query to success. `T = 0`: the query state is already a success state
        (`pre_success`). `T = -1`: never succeeded within the observed episode. `T > B`: succeeded,
        but after the label window (counts as a failure for the label).
* `B`   label window (200 steps), `gamma` discount (0.995).
* `cov_end` coverage read at `cov_end_index(t, T, B)` from the per-step coverage array.
"""

from __future__ import annotations

import numpy as np

PERTURB_MODES = ("point", "prefix")


def v_target(T, cov_end, B: int, gamma: float, beta: float = 0.0) -> np.ndarray:
    """`gamma**T * 1[0 <= T <= B] + beta * gamma**T_end * cov_end`, `T_end = B if (T<0 or T>B) else T`.

    Vectorised over `T` / `cov_end`. `T = -1` (never) and `T > B` (outside the window) make the
    first term 0; `T = 0` gives 1.0 (plus the beta term). The beta term is for search scores only;
    training the V head uses beta = 0."""
    T = np.asarray(T)
    cov_end = np.asarray(cov_end, np.float64)
    within = (T >= 0) & (T <= B)
    t_end = np.where(within, T, B)
    first = np.where(within, float(gamma) ** np.where(within, T, 0), 0.0)
    out = first + beta * float(gamma) ** t_end * cov_end
    return out[()] if out.ndim == 0 else out


def cov_end_index(t, T, B: int):
    """Absolute step index at which to read `cov_end`: `t + T` if `0 <= T <= B` else `t + B`.

    (A success later than the window, `T > B`, reads the coverage at `t + B`, NOT at the success
    step; an episode that never succeeded also reads `t + B`.)"""
    t = np.asarray(t)
    T = np.asarray(T)
    out = np.where((T >= 0) & (T <= B), t + T, t + B)
    return out[()] if out.ndim == 0 else out


def keep_query(t, p, sigma):
    """Prefix rule: a query at step `t` is kept iff the episode is clean (`sigma == 0`) or `t >= p`
    (every noised chunk starts before `p`, so the continuation from `t >= p` is clean base policy)."""
    out = (np.asarray(sigma) == 0) | (np.asarray(t) >= np.asarray(p))
    return out[()] if out.ndim == 0 else out


NOT_EXISTING = -2  # `success_T` return value: the query is AFTER the success, the row must be dropped


def success_T(success_step, t):
    """`T = success_step - t` for a query at absolute step `t`, given the absolute first-success
    step `success_step` (`-1` = never succeeded).

    Returns `-1` when the episode never succeeded, and `NOT_EXISTING` (= -2) when
    `0 <= success_step < t`: the episode was already over at `t`, so the query does not exist.
    Callers MUST filter `T == NOT_EXISTING` rows (-2 is not a valid label)."""
    s = np.asarray(success_step, np.int64)
    t = np.asarray(t, np.int64)
    out = np.where(s == -1, -1, np.where(s >= t, s - t, NOT_EXISTING))
    return out[()] if out.ndim == 0 else out


def chunks_active(t: int, done, sigma, perturb_p, mode: str) -> list[int]:
    """Indices of envs whose chunk STARTING at replan boundary `t` gets noise injected.

    `point`:  the single chunk with `perturb_p[i] == t`.
    `prefix`: every chunk with `t < perturb_p[i]` (so `perturb_p[i] >= episode_steps` noises the
              whole episode, `perturb_p[i] == 0` noises nothing).
    Finished (`done`) envs and `sigma == 0` envs are never selected."""
    if mode not in PERTURB_MODES:
        raise ValueError(f"perturb_mode must be one of {PERTURB_MODES}, got {mode!r}")
    done = np.asarray(done, bool)
    sigma = np.asarray(sigma)
    perturb_p = np.asarray(perturb_p)
    hit = (perturb_p == t) if mode == "point" else (t < perturb_p)
    return [int(i) for i in np.nonzero((~done) & (sigma > 0) & hit)[0]]


def forward_fill_cov(cov: np.ndarray, last_step) -> np.ndarray:
    """Forward-fill `cov` (nb, L) per row: entries after column `last_step[i]` (the last step that
    was actually executed / recorded for env i) are set to `cov[i, last_step[i]]`. Returns a copy."""
    cov = np.asarray(cov)
    last_step = np.asarray(last_step, np.int64)
    nb, L = cov.shape
    assert ((last_step >= 0) & (last_step <= L - 1)).all(), (last_step.min(), last_step.max(), L)
    fill = cov[np.arange(nb), last_step]
    after = np.arange(L)[None, :] > last_step[:, None]
    return np.where(after, fill[:, None], cov).astype(cov.dtype)
