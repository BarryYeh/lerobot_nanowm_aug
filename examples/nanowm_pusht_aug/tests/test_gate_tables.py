"""SAILOR v1 T5: 2x2 diagnostic deltas, G6 gap closed, noise ceiling, report fail-closed (all synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import gate_tables as gt  # noqa: E402
from train_value_model import roc_auc  # noqa: E402
from value_gate_report import compute_report, to_markdown  # noqa: E402


def make_preds(n_s1=40, n_anchor=30, k=8, seed=0):
    """Synthetic val_predictions: stage-1 reset episodes (3 queries each) + stage-2 anchors with k candidates
    at r=0 (1 row) and r=2, 4. Oracle scores are better than latent scores in every cell."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(**kw):
        rows.append(kw)

    def scores(y, noise):
        s = y + noise * rng.standard_normal(len(y))
        return 1 / (1 + np.exp(-(s - 0.5) * 3))

    for e in range(n_s1):
        for _t in range(3):
            y = bool(rng.random() < 0.5)
            add(
                stage=1,
                source="reset",
                episode=e,
                seed_id=-1,
                r=-1,
                sigma=0.1,
                cand=-1,
                y=y,
                cov=rng.random(),
            )
    for a in range(n_anchor):
        base = rng.random() < 0.5
        for r in (0, 2, 4):
            sigmas, cands = ((0.0,), (0,)) if r == 0 else ((0.0, 0.1), (0, 1, 2, 3))
            for sg in sigmas:
                for c in cands:
                    y = bool(base if r == 0 else rng.random() < (0.7 if base else 0.3))
                    add(
                        stage=2,
                        source="stage2",
                        episode=-1,
                        seed_id=a,
                        r=r,
                        sigma=sg,
                        cand=c,
                        y=y,
                        cov=rng.random(),
                    )
    n = len(rows)
    d = {k_: np.array([row[k_] for row in rows]) for k_ in rows[0]}
    y = d["y"].astype(float)
    d["v_mean_all"] = scores(y, 1.6)
    d["oracle_v_mean_all"] = scores(y, 0.7)
    d["v_std"] = rng.random(n) * 0.1
    d["oracle_v_std"] = rng.random(n) * 0.1
    d["rm_mean_all"] = np.clip(d["cov"] + 0.2 * rng.standard_normal(n), 0, 1)
    d["oracle_rm_mean_all"] = np.clip(d["cov"] + 0.05 * rng.standard_normal(n), 0, 1)
    d["rm_std"] = rng.random(n) * 0.1
    d["v_target"] = y * 0.9
    d["T"] = np.where(d["y"], 50, -1)
    d["T_succ"] = d["T"]
    d["cov_end"] = rng.random(n)
    d["anchor_id"] = d["seed_id"]
    d["pre_success"] = np.zeros(n, bool)
    d["wall_artifact"] = np.zeros(n, bool)
    d["pose_pred"] = rng.standard_normal((n, 4))
    d["cog_state"] = np.concatenate([rng.uniform(100, 400, (n, 4)), rng.uniform(-3, 3, (n, 1))], axis=1)
    d["source"] = d["source"].astype("<U8")
    return d


def test_fast_auc_matches_roc_auc_with_ties():
    rng = np.random.default_rng(1)
    y = rng.random(300) < 0.4
    s = np.round(rng.random(300), 1)  # many ties
    assert gt.fast_auc(y, s) == pytest.approx(roc_auc(y, s))
    assert np.isnan(gt.fast_auc(np.zeros(5, bool), np.arange(5.0)))


def test_2x2_deltas_are_exact_combinations_of_cells():
    d = make_preds()
    gm = ~d["pre_success"] & ~d["wall_artifact"]
    groups = np.where(d["stage"] == 2, d["seed_id"], d["episode"]).astype(np.int64)
    ta = gt.diagnostic_table(d, gm, groups, n_boot=60)
    for k in gt.METRICS:
        a, b = ta["cells"]["s1"]["value"][k]["value"], ta["cells"]["s2"]["value"][k]["value"]
        c, dd = ta["cells"]["s1"]["oracle"][k]["value"], ta["cells"]["s2"]["oracle"][k]["value"]
        dl = ta["deltas"][k]
        assert dl["c_minus_a"]["value"] == pytest.approx(c - a, nan_ok=True)
        assert dl["d_minus_b"]["value"] == pytest.approx(dd - b, nan_ok=True)
        assert dl["dmb_minus_cma"]["value"] == pytest.approx((dd - b) - (c - a), nan_ok=True)
        assert dl["c_minus_d"]["value"] == pytest.approx(c - dd, nan_ok=True)
        for name in dl:
            lo, hi = dl[name]["ci95"]
            assert lo <= hi or np.isnan(lo)
    # the oracle is built to be better, so the encoder-loss delta on AUROC is positive with CI above 0
    auc = ta["deltas"]["auroc"]
    assert auc["c_minus_a"]["value"] > 0 and auc["c_minus_a"]["ci95"][0] > 0
    # point AUROC of cell a equals the plain metric on its rows
    m = gm & (d["stage"] == 1)
    assert ta["cells"]["s1"]["value"]["auroc"]["value"] == pytest.approx(
        roc_auc(d["y"][m], d["v_mean_all"][m])
    )
    assert ta["cells"]["s1"]["n"] == int(m.sum())


def test_gap_closed_known_answer():
    # 4 anchors x 4 candidates; V picks the best candidate in 2 anchors, a bad one in the others
    y = np.array(
        [
            [1, 0, 0, 0],  # random 0.25, best 1
            [1, 1, 0, 0],  # 0.5, 1
            [0, 0, 0, 1],  # 0.25, 1
            [1, 1, 1, 0],  # 0.75, 1
        ],
        float,
    )
    v = np.array([[9, 1, 1, 1], [1, 1, 9, 1], [1, 1, 1, 9], [1, 1, 1, 9]], float)
    tab = {
        "anchor": np.arange(4),
        "y": y,
        "v": v,
        "rm": np.zeros_like(y),
        "ov": y.copy(),
        "orm": np.zeros_like(y),
        "n_wall_excluded": 0,
        "n_missing_excluded": 0,
    }
    res = gt.best_of_n(tab, [0.0], n_boot=500)
    s = res["per_beta"]["0.0"]["value"]
    sel = np.array([1, 0, 1, 0.0])
    assert s["S_random"] == pytest.approx(y.mean(1).mean())
    assert s["S_sel"] == pytest.approx(sel.mean())
    assert s["S_best"] == pytest.approx(1.0)
    assert s["delta_mean"] == pytest.approx(sel.mean() - y.mean(1).mean())
    assert s["gap_closed"] == pytest.approx((sel.mean() - 0.4375) / (1.0 - 0.4375))
    assert res["per_beta"]["0.0"]["oracle"]["gap_closed"] == pytest.approx(1.0)  # oracle picks the true best
    assert res["n_informative"] == 4 and res["underpowered"]  # < 40 anchors


def test_beta_selection_tie_break_and_candidate_table():
    d = make_preds(n_anchor=12)
    # drop one candidate of anchor 0 at r=2 -> the anchor leaves G6 as "missing"; wall-flag anchor 1
    drop = np.flatnonzero((d["seed_id"] == 0) & (d["r"] == 2))[0]
    keep = np.ones(len(d["y"]), bool)
    keep[drop] = False
    d = {k: v[keep] for k, v in d.items()}
    d["wall_artifact"][np.flatnonzero((d["seed_id"] == 1) & (d["r"] == 2))[0]] = True
    tab = gt.candidate_table(d, 8)
    assert tab["n_missing_excluded"] == 1 and tab["n_wall_excluded"] == 1
    assert tab["y"].shape == (10, 8)
    # flat scores -> every beta ties -> first beta wins; pick = first column (sigma 0, cand 0)
    flat = {
        **tab,
        "v": np.zeros_like(tab["v"]),
        "rm": np.zeros_like(tab["rm"]),
        "ov": tab["ov"],
        "orm": tab["orm"],
    }
    res = gt.best_of_n(flat, [0.0, 0.1, 0.3], n_boot=50)
    assert gt.select_beta(res, [0.0, 0.1, 0.3]) == 0.0
    assert res["per_beta"]["0.3"]["value"]["S_sel"] == pytest.approx(tab["y"][:, 0].mean())


def test_g6_skipped_without_test_anchors_and_runs_with_them():
    val, test = make_preds(seed=1), make_preds(seed=2)
    skipped = gt.g6_report(val, None, [0.0, 0.1, 0.3], 8, "no test yet", n_boot=100)
    assert skipped["status"] == "skipped" and skipped["meets_threshold"] is False
    assert "no test yet" in skipped["reason"] and skipped["val"]["n_used"] == 30
    full = gt.g6_report(val, test, [0.0, 0.1, 0.3], 8, n_boot=200)
    assert full["status"] in ("evaluated", "underpowered") and full["test"]["n_used"] == 30
    assert list(full["test"]["per_beta"]) == [str(full["selected_beta"])]  # ONLY the selected beta on test
    assert full["status"] == "underpowered"  # 30 anchors < 40 informative


def test_noise_ceiling_perfect_and_independent():
    d = make_preds(n_anchor=20, seed=3)
    m = d["stage"] == 2
    a = {k: v[m] for k, v in d.items()}
    same = gt.noise_ceiling(a, {k: v.copy() for k, v in a.items()}, n_boot=40)
    assert same["status"] == "ok" and same["y_agreement"] == 1.0 and same["within_group_auroc"] == 1.0
    rng = np.random.default_rng(0)
    b = {k: v.copy() for k, v in a.items()}
    b["y"] = rng.random(len(b["y"])) < 0.5
    ind = gt.noise_ceiling(a, b, n_boot=40)
    assert ind["y_agreement"] < 0.8 and ind["within_group_auroc"] < 0.75
    ids = gt.noise_ceiling_subset([1, 2, 3, 4], [5, 6], 3, seed=0)
    assert len(ids) == 3 and set(ids) <= {1, 2, 3, 4, 5, 6}


def test_report_fails_closed_with_empty_gate():
    d = make_preds()
    rep = compute_report(d, {}, None, "no test anchors", None, n_boot_table=20, g6_boot=100)
    assert rep["fail_closed"] and rep["g1_g5_all_meet_threshold"] is False
    assert set(rep["missing_thresholds"]) == {
        "auroc_s1_min",
        "ece_max",
        "auroc_s2_min",
        "within_anchor_auroc_min",
        "rm_mae_s1_max",
        "rm_mae_s2_max",
    }
    # threshold-dependent gates cannot pass; the fixed-rule G4 is allowed to be computed on its own
    for g in ("G1", "G2", "G3", "G5", "G6"):
        assert rep[g]["meets_threshold"] is False
    assert rep["G6"]["status"] == "skipped"
    assert rep["real_vs_imagined"]["oracle_imagined_r_ge2"]["n"] > 0
    assert "FAIL CLOSED" in to_markdown(rep)
    # G1 is evaluated on stage-1 reset rows
    assert rep["G1"]["n"] == int(((d["stage"] == 1) & (d["source"] == "reset")).sum())
    # and a full threshold set that is trivially satisfiable passes G1/G2/G3/G5 (so the failure above is the config)
    easy = {
        "auroc_s1_min": 0.0,
        "ece_max": 1.0,
        "auroc_s2_min": 0.0,
        "within_anchor_auroc_min": 0.0,
        "rm_mae_s1_max": 1.0,
        "rm_mae_s2_max": 1.0,
    }
    ok = compute_report(d, easy, None, "x", None, n_boot_table=20, g6_boot=100)
    assert not ok["fail_closed"] and all(ok[g]["meets_threshold"] for g in ("G1", "G2", "G3", "G5"))


def test_report_diagnostic_only_mode():
    d = make_preds()
    rep = compute_report(
        d, {"diagnostic_only": True}, None, "no test anchors", None, n_boot_table=20, g6_boot=100
    )
    assert rep["diagnostic_only"] and not rep["fail_closed"] and rep["missing_thresholds"] == []
    for g in ("G1", "G2", "G3", "G5"):
        assert rep[g]["meets_threshold"] is None
    assert np.isfinite(rep["G1"]["auroc"]) and np.isfinite(rep["G5"]["stage2_mae"])
    assert isinstance(rep["G4"]["meets_threshold"], bool) and rep["G6"]["meets_threshold"] is False
    md = to_markdown(rep)
    assert "Diagnostic-only mode" in md and "FAIL CLOSED" not in md
    with pytest.raises(AssertionError):
        compute_report(d, {"diagnostic_only": True, "auroc_s1_min": 0.7}, None, "x", None, n_boot_table=20)
