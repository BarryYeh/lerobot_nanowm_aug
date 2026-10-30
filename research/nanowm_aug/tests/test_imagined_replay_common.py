"""imagined_replay / heuristic shared pieces: config-backed constants, start loading, pooled comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nwaug.attribution.config_io import seed_ns
from nwaug.attribution.paired_report import boot_idx

PROJ = Path(__file__).resolve().parents[1]


def test_constants_match_the_runs_on_disk():
    import ir_common as c

    assert (c.FI, c.H, c.NF, c.CTX, c.NEED, c.EP_LEN, c.NA) == (5, 3, 4, 15, 20, 9, 10)
    assert (c.NSS, c.WM_BATCH) == (30, 24)
    assert (c.SOURCE_SEED, c.BRANCH_SEED, c.WM_SEED, c.SELECT_SEED) == (920000, 930000, 990000, 995000)
    assert (c.RESET_BASE, c.EVAL_SEED) == (9000, 100000)


def test_heuristic_seeds_are_registered():
    assert seed_ns("heuristic_late_states") == 940000
    assert seed_ns("heuristic_lastmile_branch") == 950000
    assert seed_ns("heuristic_lastmile_select") == 960000
    with pytest.raises(KeyError):
        seed_ns("no_such_seed")


def test_load_starts_and_start_slice_the_prefix(tmp_path):
    import ir_common as c

    prefix = np.arange(2 * 5 * 2, dtype=float).reshape(2, 5, 2)
    np.savez(
        tmp_path / "starts.npz",
        env_seed=np.array([7, 9]),
        t_nm=np.array([3, 5]),
        cog_at_t_nm=np.ones((2, 3)),
        prefix_actions=prefix,
    )
    st = c.load_starts(tmp_path / "starts.npz")
    s0 = c.start(st, 0)
    assert (s0["env_seed"], s0["t_nm"]) == (7, 3) and isinstance(s0["env_seed"], int)
    np.testing.assert_array_equal(s0["prefix"], prefix[0, :3])
    np.testing.assert_array_equal(c.start(st, 1)["prefix"], prefix[1])


def _eval_dir(d: Path, s: np.ndarray, mr: np.ndarray) -> None:
    d.mkdir(parents=True)
    m = {"successes": s.tolist(), "max_rewards": mr.tolist()}
    (d / "eval_info.json").write_text(json.dumps({"per_task": [{"metrics": m}]}))


def test_lastmile_pooled_known_answer(tmp_path):
    rng = np.random.default_rng(3)
    n, seeds = 40, (1000, 1001)
    S = {(a, s): rng.random(n) < p for a, p in (("W", 0.7), ("R", 0.5)) for s in seeds}
    for (a, s), x in S.items():
        _eval_dir(tmp_path / f"{a}_s{s}", x, np.where(x, 1.0, 0.96))
    script = PROJ / "proj/pusht/heuristic/lastmile_pooled.py"
    cmd = [sys.executable, str(script), str(tmp_path), "--arms", "W", "R", "--seeds", "1000", "1001"]
    subprocess.run([*cmd, "--pairs", "W:R", "--draws", "500"], check=True, capture_output=True)
    out = json.loads((tmp_path / "pooled.json").read_text())
    d = sum(S["W", s].astype(float) - S["R", s].astype(float) for s in seeds) / 2
    lo, hi = np.percentile(d[boot_idx(n, 500)].mean(1), [2.5, 97.5])
    pr = out["pairs"]["W:R"]
    assert pr["pooled"] == pytest.approx(d.mean()) and pr["ci95"] == pytest.approx([lo, hi])
    assert pr["per_seed"] == pytest.approx([S["W", s].mean() - S["R", s].mean() for s in seeds])
    # every failure has max reward 0.96 >= 0.9 / 0.95: all of them are near misses
    assert out["arms"]["R"]["near_miss_failures_per_seed"] == [int((~S["R", s]).sum()) for s in seeds]
