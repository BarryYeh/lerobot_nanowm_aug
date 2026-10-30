"""Start-state and pair selection rules shared by several experiments (pure numpy).

qualifying   grid points of one episode that pass a stalled-coverage rule (heuristic last-mile, imagined_replay
             near-miss starts, early_anchors anchors)
water_fill   spread n picks as evenly as possible over groups with capacities (imagined_replay, early_anchors)
"""

from __future__ import annotations

import numpy as np


def qualifying(steps, est, success_step, *, t_min, cov_lo, cov_hi, stall, per_episode, gap, rng) -> list[int]:
    """Grid indices g (each >= 2) of one episode that qualify, at most per_episode, >= gap steps apart (random order)."""
    end = success_step if success_step > 0 else 10**9
    ok = [
        g
        for g in range(2, len(steps))
        if steps[g] >= t_min and steps[g] < end and cov_lo <= est[g] < cov_hi and est[g] - est[g - 2] <= stall
    ]
    picked = []
    for g in rng.permutation(ok) if ok else []:
        if all(abs(int(steps[g]) - int(steps[p])) >= gap for p in picked):
            picked.append(int(g))
        if len(picked) >= per_episode:
            break
    return sorted(picked)


def water_fill(cap: dict[int, int], n: int, rng: np.random.Generator) -> dict[int, int]:
    """Quota per group: equal shares, capped at each group's capacity; the remainder goes +1 to random open groups."""
    q = dict.fromkeys(cap, 0)
    left, open_ = n, sorted(cap)
    while left > 0 and open_:
        per = left // len(open_)
        if per == 0:  # remainder: +1 to randomly chosen starts that still have room
            for s in rng.choice(open_, size=left, replace=False):
                q[int(s)] += 1
            left = 0
            break
        for s in open_:
            take = min(per, cap[s] - q[s])
            q[s] += take
            left -= take
        open_ = [s for s in open_ if q[s] < cap[s]]
    if left:
        raise SystemExit(f"only {sum(cap.values())} passing pairs, {n} requested")
    return q
