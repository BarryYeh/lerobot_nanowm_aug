# SAILOR v1 — T5 code changes (Phase 3) and open design questions

Range: `4ec4908b..HEAD` on `feat/sailor-v1-value`. Pre-registration: `sailor_v1.md` (§3, §5, §9). `gate: {}` in
`configs/sailor_v1_reset.yaml` is **unchanged (empty)**; no val metric was inspected (smoke and a 3-step shape check
only).

## What changed

`critic/train_value_model.py`
- V head: BCE with soft labels against `v_target` for stage-1 and stage-2 rows (`compute_loss`; `y_override` still
  replaces the whole target vector). Stage-1 shards use `q_v_target`; legacy shards without it fall back to the hard y.
- RM head regresses `cov` (stage 1: coverage(t); stage 2: `sim_cov`, already what `labels.npz["cov"]` holds).
- Shuffled-label control shuffles `v_target` (fit / es independently, as before). Overfit test keeps binary y (§9 D4).
- `val_predictions.npz` gains `v_target, T, T_succ, cov_end, anchor_id, cand`, plus `oracle_rm_mean_all` and
  `oracle_v_std` (needed for the OracleNet RM MAE and G6 oracle). `T_succ`: stage 1 = absolute success step (`q_T_succ`),
  stage 2 = steps from the query (`success_step`); `T` is the unified steps-to-success (-1 never). `anchor_id` = stage-2
  `seed_id` (-1 for stage 1), `cand` = -1 for stage 1.
- Prediction code factored into `predict_rows` (reused by the report for the test anchors); `_load_stage2` is now
  `load_stage2_rows`. Empty band bucket: split / class_balance already handled it; now pinned by a test.
- **D6 (pre-registered, §9)**: Normalizer is fit on the fit split only, without stage-2 `wall_artifact` rows (was
  fit + es). `OracleEnsemble` docstring corrected (it shares the normalizer object). This is the only change to the
  training procedure; split, ensemble-of-5 bootstrap, shuffled control, OracleNet are untouched.

`critic/gate_tables.py` (new, pure numpy) — Table A (2x2 + r=0 column, paired Value/Oracle bootstrap inside a column,
independent across columns, deltas c-a / d-b / (d-b)-(c-a) / c-d for AUROC, ECE_soft, within-anchor AUROC, RM MAE),
G6 candidate table + Best-of-N + beta selection + gap closed, Table D noise ceiling, vectorised `fast_auc`.

`critic/value_gate_report.py`
- G1 on stage-1 reset rows; ECE_soft (vs `v_target`) is the gated ECE, ECE_bin reported. G4 gated on |V − v_target|
  (binary version reported). G5 unchanged (cov is now real).
- Fail closed: missing thresholds → gate `meets_threshold=False`, `threshold: null`, `fail_closed: true`,
  `missing_thresholds` listed, report files still written, **process exits 2**. (Previously a KeyError.)
- Table A, G6 / Table C, Table D sections in json + md. `real_vs_imagined` gets OracleNet rows
  (`oracle_real_reset`, `oracle_imagined_r_ge2`, `oracle_stage2_r0_diagnostic`); the unpaired imagined-minus-real gap
  now references stage-1 **reset** (band is empty) and also reports Oracle.
- G6: beta chosen on val (max mean Δ, ties → smaller beta), only the selected beta evaluated on test anchors. Test
  predictions come from the saved ensembles on `test_stage2/labels.npz`; **if that file does not exist G6 is skipped**
  (`status: skipped`, `meets_threshold: false`, message printed). Gate = one-sided 95% lower bound of mean Δ > 0
  (10000 anchor bootstrap); gap closed descriptive; `underpowered` if n_informative < 40.
- Table D reads `stage2/labels_nc0.npz` / `labels_nc1.npz` (skipped with a message if absent).
  `--nc-subset-out ids.npz` writes the 25 fit+es anchor ids (seed 770003) for `label --anchor-subset ids.npz
  --cont-seed 770000|770001 --out-name labels_nc{0,1}.npz`. No shell stage was added for this.

Tests (`tests/test_value_train_targets.py`, `tests/test_gate_tables.py`): soft-label BCE target, empty band bucket,
loader/predict fields, 2x2 deltas on synthetic predictions, gap closed known answer, beta tie-break + candidate table,
G6 skip without test anchors, noise ceiling, fail-closed report.

## Open design questions for the orchestrator

1. **Thresholds** still empty: required keys are `auroc_s1_min, ece_max, auroc_s2_min, within_anchor_auroc_min,
   rm_mae_s1_max, rm_mae_s2_max` (read by `value_gate_report.py`). Smoke/full `report` exits 2 until they are set.
2. **ECE_soft semantics**: compares mean(v_target) to mean(V) per bin, so with soft targets the achievable level
   depends on the T distribution (v_target ≈ 0.37–1 for successes). E1 should be set with that in mind.
3. **Stage-1 within-anchor AUROC** (2x2 cell a/c): "anchor" = episode (10 queries at different t), so it mixes time
   and state; stage 2 uses (seed_id, r, sigma) as in G3. Confirm, or drop within-anchor for the stage-1 column.
4. **Oracle score in G6** is `oracle_V + beta*oracle_RM` at each beta (not V-only); hindsight-best is the gap-closed
   denominator as pre-registered.
5. **Table D plumbing**: run commands/file names are my convention (`labels_nc0/1.npz`); the cont-seed 770000 re-label
   duplicates `labels.npz` rows for 25 anchors only if batch composition matches (O9) — worth a check at T6.
6. **D8 t-stratified reporting** and D4's `noised` stratification are not implemented (val predictions carry no `t`;
   `frame_steps[:, -1]` is available if wanted).
7. **Legacy hard-label fallback** for stage-1 shards without `q_v_target` is silent apart from this note; T4 data has
   the field, so it is inert here.
8. Smoke needs `--with matplotlib` for `train` (T3 F5 still open: `matplotlib` is not in `--extra training`).
