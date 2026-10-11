"""SAILOR expand: anchor headroom rule, BH, discovery classes, per-anchor pick, config namespaces."""

from __future__ import annotations

import math

import numpy as np
from config_io import load_config
from expand_select import bh, classify, pick_per_anchor
from select_improvable_anchors import keep_mask, summarize

from conftest import REPO

NAN = float("nan")


def test_keep_mask_drops_easy_and_hopeless_only():
    p0 = np.array([0.05, 0.10, 0.95, 0.95, 0.50, 1.00])
    t0 = np.array([150, 150, 40, 120, 60, NAN])
    keep = keep_mask(p0, t0, p0_min=0.1, easy_p0=0.9, easy_t0=80)
    # hopeless, kept at the boundary, easy+fast, reliable but slow, middling, all-success with no time info
    assert keep.tolist() == [False, True, False, True, True, True]


def test_summarize_median_over_successes():
    ss = np.array([[10, -1, 30, -1], [-1, -1, -1, -1]])
    p0, t0 = summarize(ss)
    assert p0.tolist() == [0.5, 0.0]
    assert t0[0] == 20 and math.isnan(t0[1])


def test_bh_matches_hand_computation_and_order():
    q = bh([0.01, 0.04, 0.03, 0.20])
    # sorted p 0.01, 0.03, 0.04, 0.20 -> p*n/rank 0.04, 0.06, 0.0533, 0.20 -> running min from the top
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert bh([np.nan, 0.5]).tolist() == [1.0, 1.0]


def test_classify_rate_first_then_time():
    a = 0.05
    assert classify(0.3, +0.4, 0.01, 0.9, 150, 100, a) == "rescue"
    assert classify(0.7, +0.3, 0.01, 0.9, 150, 100, a) == "more_reliable"
    assert classify(0.7, -0.3, 0.01, 0.01, 50, 150, a) == "worse_rate"
    assert classify(0.7, 0.0, 0.5, 0.01, 110, 57, a) == "faster"
    assert classify(0.7, 0.0, 0.5, 0.01, 40, 110, a) == "slower"
    assert classify(0.7, 0.0, 0.5, 0.01, NAN, 57, a) == "no_clear_diff"
    assert classify(0.7, 0.0, 0.5, 0.20, 110, 57, a) == "no_clear_diff"


def _row(seed, cand, sigma, cls, q):
    return {
        "seed": seed,
        "cand": cand,
        "sigma": sigma,
        "cls": cls,
        "q_rate": 1.0,
        "q_time": q,
        "branch": f"sigma{sigma}/seed{seed:04d}_cand{cand}.npz",
    }


def test_pick_per_anchor_distinct_cand_strongest_first():
    rows = [
        _row(1, 3, 0.0, "faster", 0.01),
        _row(1, 3, 0.1, "faster", 0.001),  # same cand: keep the stronger only
        _row(1, 5, 0.25, "faster", 0.02),
        _row(1, 6, 0.0, "faster", 0.03),  # cap 2 -> cand 6 dropped
        _row(2, 0, 0.0, "slower", 0.001),
        _row(2, 1, 0.0, "no_clear_diff", 1.0),  # nothing improved
        _row(3, 2, 0.1, "rescue", 0.5),
    ]
    got = [(r["seed"], r["cand"], r["sigma"]) for r in pick_per_anchor(rows, per_anchor=2)]
    assert got == [(1, 3, 0.1), (1, 5, 0.25), (3, 2, 0.1)]


def test_expand_config_namespaces_and_shape():
    cfg = load_config(REPO / "configs" / "sailor_expand.yaml", out_dir_copy=False)
    ns, s2, ex = cfg["seed_namespace"], cfg["stage2"], cfg["expand"]
    assert s2["anchors"]["seed"] == ns["sailor_expand_anchor"] == s2["torch_seed"]
    assert s2["noise_seed"] == ns["sailor_expand_anchor"] + 3
    assert ex["screen"]["seed"] == ns["sailor_expand_screen"]
    assert (
        ex["verify"]["seed"] == ns["sailor_expand_verify"]
        and ex["replicate"]["seed"] == ns["sailor_expand_replicate"]
    )
    # verification seeds use seed + 2i and the label's noise rng uses cont_seed + 1: the two sets never collide
    v = {ex["verify"]["seed"] + 2 * i + j for i in range(ex["verify"]["seeds"]) for j in (0, 1)}
    r = {ex["replicate"]["seed"] + 2 * i + j for i in range(ex["replicate"]["seeds"]) for j in (0, 1)}
    assert not v & r
    assert (
        len(set(ns.values())) == len(ns) - 1
    )  # the only shared value: agent_value_map == mode_probe (documented)
    assert s2["h_max"] % s2["replan_every_rounds"] == 0 and s2["inject_sigma"][0] == 0.0
    assert s2["query_rounds"] == [0, s2["h_max"]]
    assert ex["n_anchors"] <= s2["anchors"]["n"]
