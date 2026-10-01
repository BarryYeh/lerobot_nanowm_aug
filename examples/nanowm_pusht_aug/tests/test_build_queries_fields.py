# ruff: noqa: N802  -- T / B follow the pre-registration notation
"""`collect_value_states.build_queries` label fields on a fake rollout result (no sim, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gym_pusht")

from collect_value_states import build_queries  # noqa: E402
from critic.labels import v_target  # noqa: E402

B, G, L = 200, 21, 301
TS = list(range(10, 101, 10))


def _episode(success_step: int, sigma: float = 0.1, perturb_p: int = 300, wall: bool = False) -> dict:
    """One fake episode: coverage rises 0.001/step so cov_per_step[s] = s/1000 (forward-filled after success)."""
    cov = np.arange(L, dtype=np.float32) / 1000.0
    if success_step != -1:
        cov = np.minimum(cov, np.float32(success_step / 1000.0))
    return {
        "ep_ids": np.array([7]),
        "source": np.array(["reset"]),
        "sigma": np.array([sigma]),
        "perturb_p": np.array([perturb_p]),
        "success_step": np.array([success_step]),
        "wall_artifact": np.array([wall]),
        "grid_cog": np.zeros((1, G, 5)),
        "grid_agent_pos": np.zeros((1, G, 2)),
        "grid_agent_vel": np.zeros((1, G, 2)),
        "grid_coverage": (cov[::5][:G])[None].astype(np.float32),
        "b_steps": B,
        "cov_per_step": cov[None],
    }


def _rows(**kw):
    ep = _episode(kw.pop("success_step"), kw.pop("sigma", 0.1), kw.pop("perturb_p", 300))
    return build_queries(**ep, query_ts=TS, **kw)


def test_success_after_the_window_is_a_failure_with_true_T_and_cov_end_at_t_plus_B():
    q = _rows(success_step=250, query_keep="all")  # t=10 -> T=240 > B=200
    i = int(np.nonzero(q["t"] == 10)[0][0])
    assert q["T"][i] == 240 and q["T_succ"][i] == 250
    assert not q["y"][i] and q["v_target"][i] == 0.0
    assert q["cov_end"][i] == pytest.approx(0.210)  # coverage at t+B=210, NOT at the success step (0.250)
    j = int(np.nonzero(q["t"] == 100)[0][0])  # t=100 -> T=150 <= B
    assert q["T"][j] == 150 and q["y"][j]
    assert q["v_target"][j] == pytest.approx(0.995**150)
    assert q["cov_end"][j] == pytest.approx(0.250)  # coverage at the success step


def test_boundary_T_equal_B_succeeds_and_T_B_plus_1_fails():
    q = _rows(success_step=210, query_keep="all")  # t=10 -> T=200 ; (t=20 -> T=190)
    i = int(np.nonzero(q["t"] == 10)[0][0])
    assert q["T"][i] == 200 and q["y"][i] and q["v_target"][i] == pytest.approx(0.995**200)
    q = _rows(success_step=211, query_keep="all")
    i = int(np.nonzero(q["t"] == 10)[0][0])
    assert q["T"][i] == 201 and not q["y"][i] and q["v_target"][i] == 0.0


def test_pre_success_row_has_T0_target1_and_later_queries_do_not_exist():
    q = _rows(success_step=40, query_keep="all")
    assert q["t"].tolist() == [10, 20, 30, 40]  # t > 40 happens after the episode ended
    assert q["pre_success"].tolist() == [False, False, False, True]
    k = 3
    assert q["T"][k] == 0 and q["y"][k] and q["v_target"][k] == 1.0
    assert q["cov_end"][k] == pytest.approx(0.040)


def test_never_succeeded_has_T_minus1_and_cov_end_at_t_plus_B():
    q = _rows(success_step=-1, query_keep="all")
    assert (q["T"] == -1).all() and (q["T_succ"] == -1).all()
    assert not q["y"].any() and (q["v_target"] == 0).all()
    assert np.allclose(q["cov_end"], (q["t"] + B) / 1000.0)


def test_v_target_column_matches_labels_v_target_beta0():
    q = _rows(success_step=180, query_keep="all", gamma=0.99)
    assert np.allclose(q["v_target"], v_target(q["T"], q["cov_end"], B, 0.99, 0.0))


def test_noised_column_is_sigma_positive_and_perturb_p_positive():
    assert _rows(success_step=-1, query_keep="all", sigma=0.1, perturb_p=300)["noised"].all()
    assert not _rows(success_step=-1, query_keep="all", sigma=0.1, perturb_p=0)["noised"].any()
    assert not _rows(success_step=-1, query_keep="all", sigma=0.0, perturb_p=50)["noised"].any()
    q = _rows(success_step=-1, query_keep="all", sigma=0.1, perturb_p=50)
    assert q["perturb_p"].tolist() == [50] * len(q["t"]) and q["sigma"].dtype == np.float32


def test_query_keep_all_vs_after_switch_vs_legacy():
    allq = _rows(success_step=-1, query_keep="all", sigma=0.1, perturb_p=50)
    assert allq["t"].tolist() == TS  # nothing dropped
    after = _rows(success_step=-1, query_keep="after_switch", sigma=0.1, perturb_p=50)
    assert after["t"].tolist() == [t for t in TS if t >= 50]
    clean = _rows(success_step=-1, query_keep="after_switch", sigma=0.0, perturb_p=50)
    assert clean["t"].tolist() == TS  # sigma == 0: every query kept
    legacy = _rows(success_step=-1, sigma=0.1, perturb_p=50)  # default = legacy `t < p + 10` drop
    assert legacy["t"].tolist() == [t for t in TS if t >= 60]
    with pytest.raises(ValueError):
        _rows(success_step=-1, query_keep="bogus")


def test_wall_artifact_episode_emits_no_queries():
    ep = _episode(-1, wall=True)
    q = build_queries(**ep, query_ts=TS, query_keep="all")
    assert len(q["t"]) == 0 and q["T"].dtype == np.int64
