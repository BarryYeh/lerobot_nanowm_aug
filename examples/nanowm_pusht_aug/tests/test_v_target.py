"""Closed-form checks for critic/labels.py (numpy only, fully offline)."""

from __future__ import annotations

import numpy as np
import pytest
from critic.labels import NOT_EXISTING, cov_end_index, keep_query, success_T, v_target

G = 0.995
B = 200


def test_v_target_closed_form_success():
    assert v_target(50, 0.3, B, G) == pytest.approx(0.995**50)
    assert v_target(50, 0.3, B, G) == pytest.approx(0.77831, abs=1e-5)


def test_v_target_never_succeeded_is_beta_term_only():
    assert v_target(-1, 0.6, B, G, beta=0.0) == 0.0
    v = v_target(-1, 0.6, B, G, beta=0.1)
    assert v == pytest.approx(0.1 * 0.995**200 * 0.6)
    assert v == pytest.approx(0.02202, abs=1e-5)


def test_v_target_pre_success_is_one():
    assert v_target(0, 0.97, B, G) == 1.0
    # with beta the T_end is 0 -> cov_end counts fully
    assert v_target(0, 0.97, B, G, beta=0.1) == pytest.approx(1.0 + 0.1 * 0.97)


def test_v_target_outside_window_first_term_zero():
    assert v_target(201, 0.5, B, G, beta=0.0) == 0.0
    # T > B: T_end = B, so the beta term matches the never-succeeded case
    assert v_target(201, 0.5, B, G, beta=0.1) == pytest.approx(0.1 * 0.995**200 * 0.5)
    assert v_target(B, 0.97, B, G) == pytest.approx(0.995**200)  # boundary T == B still counts


def test_v_target_vectorised_matches_scalar():
    T = np.array([-1, 0, 1, 50, 200, 201, 299])
    cov = np.linspace(0.1, 0.99, len(T))
    for beta in (0.0, 0.3):
        vec = v_target(T, cov, B, G, beta=beta)
        assert vec.shape == T.shape
        for k in range(len(T)):
            assert vec[k] == pytest.approx(v_target(int(T[k]), float(cov[k]), B, G, beta=beta))
    assert ((v_target(T, cov, B, G) >= 0) & (v_target(T, cov, B, G) <= 1)).all()


def test_cov_end_index():
    assert cov_end_index(t=10, T=50, B=B) == 60
    assert cov_end_index(t=100, T=250, B=B) == 300  # succeeded, but after the window -> t + B
    assert cov_end_index(t=100, T=-1, B=B) == 300  # never
    assert cov_end_index(t=100, T=0, B=B) == 100  # pre_success
    assert cov_end_index(t=100, T=200, B=B) == 300
    np.testing.assert_array_equal(
        cov_end_index(np.array([10, 100, 100]), np.array([50, 250, -1]), B), [60, 300, 300]
    )


def test_keep_query_truth_table():
    # sigma == 0: always kept, regardless of p
    for t in (0, 10, 100):
        for p in (0, 50, 1000):
            assert keep_query(t, p, 0.0)
    # sigma > 0: kept iff t >= p
    assert keep_query(10, 10, 0.1)
    assert keep_query(20, 10, 0.1)
    assert not keep_query(10, 20, 0.1)
    assert not keep_query(0, 10, 0.1)
    # p == 0 keeps everything
    assert all(keep_query(t, 0, 0.1) for t in range(0, 101, 10))
    # vectorised
    t = np.array([10, 10, 50, 50])
    p = np.array([0, 20, 50, 60])
    s = np.array([0.1, 0.1, 0.1, 0.0])
    np.testing.assert_array_equal(keep_query(t, p, s), [True, False, True, True])


def test_success_t():
    assert success_T(-1, 50) == -1  # never
    assert success_T(80, 50) == 30
    assert success_T(50, 50) == 0  # pre_success: success exactly at the query step
    assert success_T(40, 50) == NOT_EXISTING == -2  # query is after the success: caller drops the row
    np.testing.assert_array_equal(success_T(np.array([-1, 80, 50, 40]), 50), [-1, 30, 0, -2])
