"""Pure-numpy evaluation tables for the SAILOR v1 gate report (pre-registration sailor_v1.md sections 5, 9).

  * Table A: 2x2 diagnostic (ValueNet / OracleNet x stage-1 real / stage-2 r>=2 imagined, + stage-2 r=0)
    with grouped-bootstrap CIs and the derived deltas (c-a, d-b, (d-b)-(c-a), c-d).
  * Table C / G6: offline Best-of-N over the 8 candidates of an anchor at the r=2 query.
  * Table D: noise ceiling (two independent clean-continuation relabels of the same anchors).

No torch / gym imports, so the tests run offline. Decisions that are not spelled out in the pre-registration
are listed in docs/decisions/sailor_v1_t5_changes.md.
"""

from __future__ import annotations

import numpy as np

N_BOOT = 1000
G6_BOOT = 10_000
MIN_INFORMATIVE = 40  # pre-registration section 9: fewer informative anchors -> "underpowered"
METRICS = ("auroc", "ece_soft", "within_anchor_auroc", "rm_mae")
MODELS = ("value", "oracle")


# ============================================================================================ primitives
def fast_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Tie-corrected Mann-Whitney AUROC, vectorised (same value as train_value_model.roc_auc)."""
    y = np.asarray(y_true, bool)
    s = np.asarray(score, np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(target: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """10 equal-width bins on [0, 1]; `target` may be soft (v_target -> ECE_soft) or binary (ECE_bin)."""
    target, prob = np.asarray(target, np.float64), np.asarray(prob, np.float64)
    if len(prob) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    total, e = len(prob), 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum() / total) * abs(target[m].mean() - prob[m].mean())
    return float(e)


def within_group_aucs(
    y: np.ndarray, score: np.ndarray, group_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """AUROC inside every group (rows sharing `group_code`) that has >= 2 rows and mixed y.
    Returns (aucs, positions) where positions[i] is the index of the group's first row."""
    aucs, first = [], []
    order = np.argsort(group_code, kind="stable")
    codes = group_code[order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    for part in np.split(order, bounds):
        if len(part) < 2:
            continue
        a = fast_auc(y[part], score[part])
        if np.isfinite(a):
            aucs.append(a)
            first.append(int(part[0]))
    return np.asarray(aucs, np.float64), np.asarray(first, np.int64)


def pct_ci(vals: np.ndarray) -> list[float]:
    vals = np.asarray(vals, np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


# ============================================================================================ Table A
def _column_data(d: dict, mask: np.ndarray, groups: np.ndarray, within_key: np.ndarray) -> dict:
    return {
        "y": d["y"][mask].astype(bool),
        "vt": d["v_target"][mask].astype(np.float64),
        "cov": d["cov"][mask].astype(np.float64),
        "super": groups[mask],
        "wkey": within_key[mask],
        "pred": {
            "value": (d["v_mean_all"][mask].astype(np.float64), d["rm_mean_all"][mask].astype(np.float64)),
            "oracle": (
                d["oracle_v_mean_all"][mask].astype(np.float64),
                d["oracle_rm_mean_all"][mask].astype(np.float64),
            ),
        },
    }


def _col_stats(c: dict, idx: np.ndarray | None, wa_cache: dict | None = None) -> dict:
    """{model: {metric: value}} on the rows `idx` (None = all). within_anchor handled by the caller."""
    sel = slice(None) if idx is None else idx
    out = {}
    for m, (v, rm) in c["pred"].items():
        out[m] = {
            "auroc": fast_auc(c["y"][sel], v[sel]),
            "ece_soft": ece(c["vt"][sel], v[sel]),
            "rm_mae": float(np.mean(np.abs(rm[sel] - c["cov"][sel]))) if len(c["y"][sel]) else float("nan"),
        }
    return out


def column_bootstrap(c: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Point estimates + `n_boot` grouped-bootstrap replicates for BOTH models on one column. The two
    models share every resample (paired Value - Oracle differences within a column, pre-registration D4).
    Returns {"n", "n_groups", "point": {model: {metric: v}}, "boot": {model: {metric: (n_boot,)}}}."""
    n = len(c["y"])
    nan = float("nan")
    if n == 0:
        pt = {m: dict.fromkeys(METRICS, nan) for m in MODELS}
        return {
            "n": 0,
            "n_groups": 0,
            "point": pt,
            "boot": {m: {k: np.full(n_boot, nan) for k in METRICS} for m in MODELS},
        }
    supers, sup_inv = np.unique(c["super"], return_inverse=True)
    n_g = len(supers)
    idx_by_super = [np.flatnonzero(sup_inv == g) for g in range(n_g)]

    # within-anchor AUROC: per (model) the group AUROCs and which super-group each belongs to
    wa = {}
    for m, (v, _) in c["pred"].items():
        aucs, first = within_group_aucs(c["y"], v, c["wkey"])
        wa[m] = (aucs, sup_inv[first] if len(first) else np.zeros(0, np.int64))

    def wa_mean(m: str, counts: np.ndarray) -> float:
        aucs, sg = wa[m]
        if len(aucs) == 0:
            return nan
        w = counts[sg]
        return float((aucs * w).sum() / w.sum()) if w.sum() > 0 else nan

    point_stats = _col_stats(c, None)
    ones = np.ones(n_g)
    point = {m: {**point_stats[m], "within_anchor_auroc": wa_mean(m, ones)} for m in MODELS}

    rng = np.random.default_rng(seed)
    boot = {m: {k: np.full(n_boot, nan) for k in METRICS} for m in MODELS}
    for b in range(n_boot):
        draw = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([idx_by_super[g] for g in draw])
        counts = np.bincount(draw, minlength=n_g)
        st = _col_stats(c, idx)
        for m in MODELS:
            for k in ("auroc", "ece_soft", "rm_mae"):
                boot[m][k][b] = st[m][k]
            boot[m]["within_anchor_auroc"][b] = wa_mean(m, counts)
    return {"n": int(n), "n_groups": int(n_g), "point": point, "boot": boot}


def _delta(point_fn, boot_fn) -> dict:
    boot = boot_fn()
    return {"value": float(point_fn()), "ci95": pct_ci(boot)}


def diagnostic_table(d: dict, gm: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Table A. `gm` = rows usable for gate metrics (pre_success / wall_artifact already excluded).
    Columns: s1 = stage-1 real (val), s2 = stage-2 r>=2 (imagined), s2_r0 = stage-2 r=0 (real frames,
    stage-2 state distribution; pre-registration D4). Within-anchor groups: stage-1 = episode,
    stage-2 = (seed_id, r, sigma). Every column bootstraps its own groups independently; Value/Oracle
    share the resamples inside a column."""
    wkey = np.zeros(len(d["y"]), np.int64)
    s2 = d["stage"] == 2
    if s2.any():
        key = np.stack([d["seed_id"][s2], d["r"][s2], np.round(d["sigma"][s2] * 1000)], axis=1)
        _, code = np.unique(key, axis=0, return_inverse=True)
        wkey[s2] = code.reshape(-1) + 1
    wkey[~s2] = -1 - groups[~s2]  # negative codes: one within-group per stage-1 episode
    cols = {
        "s1": gm & (d["stage"] == 1),
        "s2": gm & s2 & (d["r"] >= 2),
        "s2_r0": gm & s2 & (d["r"] == 0),
    }
    res = {
        name: column_bootstrap(_column_data(d, m, groups, wkey), n_boot, seed=6000 + 10 * i)
        for i, (name, m) in enumerate(cols.items())
    }

    cells = {}
    for name, r in res.items():
        cells[name] = {"n": r["n"], "n_groups": r["n_groups"]}
        for m in MODELS:
            cells[name][m] = {
                k: {"value": float(r["point"][m][k]), "ci95": pct_ci(r["boot"][m][k])} for k in METRICS
            }

    def diff(spec) -> dict:
        """spec: list of (sign, column, model). Independent columns => replicates add; same column pairs."""
        pt = sum(s * res[c]["point"][m][k_] for s, c, m, k_ in spec)
        bt = sum(s * res[c]["boot"][m][k_] for s, c, m, k_ in spec)
        return {"value": float(pt), "ci95": pct_ci(bt)}

    deltas = {}
    for k in METRICS:
        deltas[k] = {
            "c_minus_a": diff([(1, "s1", "oracle", k), (-1, "s1", "value", k)]),
            "d_minus_b": diff([(1, "s2", "oracle", k), (-1, "s2", "value", k)]),
            "dmb_minus_cma": diff(
                [
                    (1, "s2", "oracle", k),
                    (-1, "s2", "value", k),
                    (-1, "s1", "oracle", k),
                    (1, "s1", "value", k),
                ]
            ),
            "c_minus_d": diff([(1, "s1", "oracle", k), (-1, "s2", "oracle", k)]),
        }
    return {
        "desc": "cells a=ValueNet/s1, b=ValueNet/s2(r>=2), c=OracleNet/s1, d=OracleNet/s2(r>=2). "
        "c-a encoder loss; d-b encoder + block-pose WM hallucination; (d-b)-(c-a) ~ WM hallucination; "
        "c-d imagined distribution harder. Value-Oracle inside a column is a paired grouped bootstrap, "
        "cross-column differences use independent resamples.",
        "cells": cells,
        "deltas": deltas,
    }


# ============================================================================================ G6 / Table C
def candidate_table(d: dict, n_expected: int, r_query: int = 2) -> dict:
    """Group the r=`r_query` stage-2 rows of every anchor into (A, n_expected) arrays sorted by
    (sigma, cand) (the tie-break order). An anchor is dropped when it lacks exactly `n_expected`
    distinct (sigma, cand) rows or any of its candidates is a wall_artifact. pre_success candidates are
    kept with y = 1 (pre-registered G6 exception)."""
    m = (d["stage"] == 2) & (d["r"] == r_query)
    idx = np.flatnonzero(m)
    out = {k: [] for k in ("anchor", "y", "v", "rm", "ov", "orm")}
    n_wall = n_missing = 0
    for a in np.unique(d["seed_id"][idx]):
        rows = idx[d["seed_id"][idx] == a]
        sig = np.round(d["sigma"][rows].astype(np.float64), 4)
        order = np.lexsort((d["cand"][rows], sig))
        rows = rows[order]
        pairs = set(zip(sig[order].tolist(), d["cand"][rows].tolist(), strict=True))
        if len(rows) != n_expected or len(pairs) != n_expected:
            n_missing += 1
            continue
        if d["wall_artifact"][rows].any():
            n_wall += 1
            continue
        out["anchor"].append(int(a))
        out["y"].append((d["y"][rows] | d["pre_success"][rows]).astype(np.float64))
        out["v"].append(d["v_mean_all"][rows].astype(np.float64))
        out["rm"].append(d["rm_mean_all"][rows].astype(np.float64))
        out["ov"].append(d["oracle_v_mean_all"][rows].astype(np.float64))
        out["orm"].append(d["oracle_rm_mean_all"][rows].astype(np.float64))
    tab = {k: np.asarray(v) for k, v in out.items()}
    tab["n_wall_excluded"] = n_wall
    tab["n_missing_excluded"] = n_missing
    return tab


def _pick(score: np.ndarray) -> np.ndarray:
    """argmax over candidates; ties -> first column = lowest (sigma, cand)."""
    return np.argmax(score, axis=1)


def _bon_stats(y: np.ndarray, pick: np.ndarray, n_boot: int, seed: int) -> dict:
    """Selection success vs random / hindsight on (A, n) outcomes; all per-anchor quantities."""
    a = len(y)
    if a == 0:
        nan = float("nan")
        return {
            "n_anchors": 0,
            "S_random": nan,
            "S_sel": nan,
            "S_best": nan,
            "delta_mean": nan,
            "delta_sd": nan,
            "delta_ci95": [nan, nan],
            "delta_lower_one_sided_95": nan,
            "gap_closed": nan,
            "gap_closed_ci95": [nan, nan],
        }
    s_rand = y.mean(1)
    s_sel = y[np.arange(a), pick]
    s_best = y.max(1)
    delta = s_sel - s_rand
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, a, size=(n_boot, a))
    d_boot = delta[draw].mean(1)
    num = d_boot
    den = (s_best[draw] - s_rand[draw]).mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gc_boot = np.where(den > 1e-12, num / den, np.nan)
    den0 = float((s_best - s_rand).mean())
    return {
        "n_anchors": int(a),
        "S_random": float(s_rand.mean()),
        "S_sel": float(s_sel.mean()),
        "S_best": float(s_best.mean()),
        "delta_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)) if a > 1 else float("nan"),
        "delta_ci95": pct_ci(d_boot),
        "delta_lower_one_sided_95": float(np.percentile(d_boot, 5)),
        "gap_closed": float(delta.mean() / den0) if den0 > 1e-12 else float("nan"),
        "gap_closed_ci95": pct_ci(gc_boot),
    }


def best_of_n(tab: dict, betas: list[float], n_boot: int = G6_BOOT, seed: int = 7000) -> dict:
    """Per-beta selection stats for the ValueNet score V + beta*RM and the OracleNet analogue, plus the
    random / hindsight references. `tab` from `candidate_table`."""
    y = tab["y"]
    n_inf = int(((y.sum(1) > 0) & (y.sum(1) < y.shape[1])).sum()) if len(y) else 0
    res = {
        "n_used": int(len(y)),
        "n_informative": n_inf,
        "n_excluded_wall": int(tab["n_wall_excluded"]),
        "n_excluded_missing": int(tab["n_missing_excluded"]),
        "underpowered": bool(n_inf < MIN_INFORMATIVE),
        "per_beta": {},
    }
    for b in betas:
        if len(y) == 0:
            v = o = _bon_stats(y, np.zeros(0, int), n_boot, seed)
        else:
            v = _bon_stats(y, _pick(tab["v"] + b * tab["rm"]), n_boot, seed)
            o = _bon_stats(y, _pick(tab["ov"] + b * tab["orm"]), n_boot, seed + 1)
        res["per_beta"][str(b)] = {"value": v, "oracle": o}
    return res


def select_beta(val_res: dict, betas: list[float]) -> float | None:
    """Highest mean delta on val; ties -> first (smallest) beta in the pre-registered order."""
    best, best_b = -np.inf, None
    for b in betas:
        dm = val_res["per_beta"][str(b)]["value"]["delta_mean"]
        if np.isfinite(dm) and dm > best:
            best, best_b = dm, b
    return best_b


def g6_report(
    val: dict,
    test: dict | None,
    betas: list[float],
    n_expected: int,
    skip_reason: str | None = None,
    n_boot: int = G6_BOOT,
) -> dict:
    """G6. beta is chosen on val; ONLY the selected beta is evaluated on the test anchors. Skipped (not an
    error) if the test anchors / predictions do not exist yet."""
    val_res = best_of_n(candidate_table(val, n_expected), betas, n_boot)
    sel = select_beta(val_res, betas)
    out = {
        "desc": "offline Best-of-N at the r=2 query, 8 candidates per anchor; score = V + beta*RM; "
        "gate = one-sided 95% lower bound of mean_i(S_beta - S_random) > 0 on TEST anchors",
        "betas": betas,
        "n_expected_candidates": n_expected,
        "val": val_res,
        "selected_beta": sel,
        "meets_threshold": False,
    }
    if test is None:
        out.update(status="skipped", reason=skip_reason or "no test anchors", test=None)
        return out
    if sel is None:
        out.update(status="skipped", reason="no usable val anchors to select beta", test=None)
        return out
    t = best_of_n(candidate_table(test, n_expected), [sel], n_boot)
    out["test"] = t
    tv = t["per_beta"][str(sel)]["value"]
    lb = tv["delta_lower_one_sided_95"]
    out["status"] = "underpowered" if t["underpowered"] else "evaluated"
    out["meets_threshold"] = bool(np.isfinite(lb) and lb > 0)
    return out


# ============================================================================================ Table D
def noise_ceiling(a: dict, b: dict, n_boot: int = N_BOOT, seed: int = 8000) -> dict:
    """Two independent clean-continuation relabels (`a`, `b`: stage-2 row dicts with y / seed_id / r /
    sigma / cand / pre_success / wall_artifact) of the SAME anchors. Rows join on (seed_id, sigma, cand, r).
    Reports the marginal y agreement and the within-(seed_id, r, sigma)-group AUROC(y_a as label, y_b as
    score), averaged over groups with mixed y_a (the G3 grouping), with a bootstrap CI over anchors."""

    def key(d, i):
        return (int(d["seed_id"][i]), round(float(d["sigma"][i]), 4), int(d["cand"][i]), int(d["r"][i]))

    ib = {key(b, i): i for i in range(len(b["y"]))}
    pairs = [(i, ib[key(a, i)]) for i in range(len(a["y"])) if key(a, i) in ib]
    keep = [
        (i, j)
        for i, j in pairs
        if not (a["pre_success"][i] or b["pre_success"][j] or a["wall_artifact"][i] or b["wall_artifact"][j])
    ]
    if not keep:
        return {"status": "empty", "n_rows": 0}
    ia, jb = np.array([p[0] for p in keep]), np.array([p[1] for p in keep])
    y1, y2 = a["y"][ia].astype(bool), b["y"][jb].astype(bool)
    anchor = a["seed_id"][ia].astype(np.int64)
    sig = np.round(a["sigma"][ia].astype(np.float64) * 1000)
    _, code = np.unique(np.stack([anchor, a["r"][ia], sig], axis=1), axis=0, return_inverse=True)
    code = code.reshape(-1)
    aucs, first = within_group_aucs(y1, y2.astype(np.float64), code)
    supers, sup_inv = np.unique(anchor, return_inverse=True)
    sg = sup_inv[first] if len(first) else np.zeros(0, np.int64)
    rng = np.random.default_rng(seed)
    boot_auc, boot_agree = [], []
    idx_by = [np.flatnonzero(sup_inv == g) for g in range(len(supers))]
    for _ in range(n_boot):
        draw = rng.integers(0, len(supers), size=len(supers))
        counts = np.bincount(draw, minlength=len(supers))
        w = counts[sg]
        boot_auc.append((aucs * w).sum() / w.sum() if len(aucs) and w.sum() > 0 else np.nan)
        idx = np.concatenate([idx_by[g] for g in draw])
        boot_agree.append(float((y1[idx] == y2[idx]).mean()))
    return {
        "status": "ok",
        "n_rows": int(len(ia)),
        "n_anchors": int(len(supers)),
        "n_groups_mixed": int(len(aucs)),
        "y_agreement": float((y1 == y2).mean()),
        "y_agreement_ci95": pct_ci(boot_agree),
        "within_group_auroc": float(aucs.mean()) if len(aucs) else float("nan"),
        "within_group_auroc_ci95": pct_ci(boot_auc),
    }


def noise_ceiling_subset(
    fit_groups: list[int], es_groups: list[int], n: int, seed: int, splits=("fit", "es")
) -> np.ndarray:
    """Table D anchors: `n` random stage-2 seed ids from the fit + es groups (never val / test)."""
    pool = []
    if "fit" in splits:
        pool += list(fit_groups)
    if "es" in splits:
        pool += list(es_groups)
    pool = np.asarray(sorted(set(pool)), np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
