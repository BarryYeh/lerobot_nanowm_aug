#!/usr/bin/env python
"""Near-goal fidelity report for a wm_fidelity_ood.py work dir (made for --region late).

Question: can NanoWM tell success from a near miss where the base policy's failures are decided (late, coverage
0.7-0.95)? For every branch and round r the truth is the simulator's coverage at step (H+r)*FI; the same image
estimator (coverage_from_frame, calibrated) is applied to the imagined frame and to the real frame, so the report
separates the estimator's own error (real frame vs sim) from the world model's (imagined vs real frame).

  bands      MAE of imagined / real-frame estimates vs sim coverage, per sim-coverage band
  success    per (branch, round): sim >= 0.95 vs estimate >= thr, precision / recall, imagined and real-frame
  horizon    per branch: success anywhere in the horizon, and the sign / correlation of the coverage change
Autoregressive (wm_ar.npz) and, if present, teacher-forced (wm_tf.npz, one-step) predictions.
Output: <work>/late_report.json / .md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_fidelity_ood import FI, H, measure  # noqa: E402

BANDS = [(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]
THRS = (0.9, 0.93, 0.95)


def prf(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "n_true": int(truth.sum()),
        "n_pred": int(pred.sum()),
    }


def report(truth: np.ndarray, est_imag: np.ndarray, est_real: np.ndarray) -> dict:
    """truth / est_*: flat arrays over (branch, round)."""
    out = {"n": int(len(truth)), "bands": {}, "success": {}}
    for lo, hi in BANDS:
        m = (truth >= lo) & (truth < hi)
        if not m.any():
            continue
        out["bands"][f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {
            "n": int(m.sum()),
            "mae_imag_vs_sim": float(np.nanmean(np.abs(est_imag[m] - truth[m]))),
            "mae_realframe_vs_sim": float(np.nanmean(np.abs(est_real[m] - truth[m]))),
            "mae_imag_vs_realframe": float(np.nanmean(np.abs(est_imag[m] - est_real[m]))),
            "bias_imag_vs_sim": float(np.nanmean(est_imag[m] - truth[m])),
        }
    t = truth >= 0.95
    for thr in THRS:
        out["success"][f"imag>={thr}"] = prf(t, est_imag >= thr)
        out["success"][f"realframe>={thr}"] = prf(t, est_real >= thr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    d = np.load(a.work / "real.npz")
    frames, cov, pen = d["frames"], d["cov"], d["wall_pen"]
    keep = pen <= 10
    res = {"work": str(a.work), "branches": int(keep.sum())}
    ar = np.load(a.work / "wm_ar.npz")["imag"]
    bs, R = ar.shape[:2]
    rounds = np.arange(R)
    sim = np.array([[cov[b, (H + r) * FI] for r in rounds] for b in range(bs)])
    real_est = np.array([[measure(frames[b, H + r])[5] for r in rounds] for b in range(bs)])
    ar_est = np.array([[measure(ar[b, r])[5] for r in rounds] for b in range(bs)])
    ctx_est = np.array([measure(frames[b, H - 1])[5] for b in range(bs)])
    k = keep
    res["autoregressive"] = report(sim[k].ravel(), ar_est[k].ravel(), real_est[k].ravel())
    res["autoregressive"]["by_round_mae_imag_vs_sim"] = (
        np.nanmean(np.abs(ar_est[k] - sim[k]), 0).round(4).tolist()
    )
    # horizon level: success anywhere within the predicted rounds, and the net coverage change
    t_any = (sim[k] >= 0.95).any(1)
    hz = {f"imag_any>={thr}": prf(t_any, (ar_est[k] >= thr).any(1)) for thr in THRS}
    d_true = sim[k, -1] - cov[k, (H - 1) * FI]
    d_imag = ar_est[k, -1] - ctx_est[k]
    hz["dcov_pearson"] = float(np.corrcoef(d_true, d_imag)[0, 1])
    hz["dcov_sign_agree"] = float(np.mean(np.sign(d_true) == np.sign(d_imag)))
    hz["start_cov_sim_median"] = float(np.median(cov[k, (H - 1) * FI]))
    hz["success_within_horizon_rate"] = float(t_any.mean())
    res["horizon"] = hz
    tf_path = a.work / "wm_tf.npz"
    if tf_path.exists():
        tf = np.load(tf_path)
        rs = tf["rounds"]
        tf_est = np.array([[measure(tf["imag"][b, i])[5] for i in range(len(rs))] for b in range(bs)])
        res["teacher_forced"] = report(sim[k][:, rs].ravel(), tf_est[k].ravel(), real_est[k][:, rs].ravel())
    (a.work / "late_report.json").write_text(json.dumps(res, indent=1))

    def fmt(x):
        return "—" if x is None else f"{x:.2f}"

    lines = [f"# NanoWM near-goal fidelity ({res['branches']} branches x {R} rounds, {a.work.name})", ""]
    for mode in ("autoregressive", "teacher_forced"):
        if mode not in res:
            continue
        r = res[mode]
        lines += [
            f"## {mode}",
            "",
            "| sim coverage band | n | MAE imagined vs sim | MAE real-frame estimate vs sim | bias imagined |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {b} | {v['n']} | {v['mae_imag_vs_sim']:.3f} | {v['mae_realframe_vs_sim']:.3f} | {v['bias_imag_vs_sim']:+.3f} |"
            for b, v in r["bands"].items()
        ]
        lines += [
            "",
            "| success call (sim >= 0.95) | precision | recall | predicted / true |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {c} | {fmt(v['precision'])} | {fmt(v['recall'])} | {v['n_pred']} / {v['n_true']} |"
            for c, v in r["success"].items()
        ]
        lines.append("")
    hz = res["horizon"]
    lines += [
        "## horizon (autoregressive)",
        "",
        f"start sim coverage median {hz['start_cov_sim_median']:.3f}; success within horizon {hz['success_within_horizon_rate']:.2f}; "
        f"Δcov Pearson {hz['dcov_pearson']:.3f}, sign agreement {hz['dcov_sign_agree']:.2f}",
        "",
        "| success anywhere in horizon | precision | recall |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c} | {fmt(v['precision'])} | {fmt(v['recall'])} |" for c, v in hz.items() if isinstance(v, dict)
    ]
    (a.work / "late_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
