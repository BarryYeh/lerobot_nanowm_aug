# early_anchors

imagined_replay generated each imagined frame by sliding the NanoWM context 1 sim step at a time near the goal. That breaks the
world model's native interval (fi = 5), and it showed no gain: Img, Real and ctrl all scored about the same.
This project starts instead from **early anchors**: real states that stalled at mid coverage and are still far from termination.
From each anchor the base policy runs closed-loop inside NanoWM at multiples of fi. Only the rollouts that succeed in
imagination are kept, and they are used to fine-tune a weaker base in two arms, **Img vs Real**.

Config: `configs/early_anchors.yaml` (`extends: common.yaml`; seed namespaces `early_anchors_*`). Output:
`~/datastorage/nanowm_pusht_aug/early_anchors/`. GPUs 6, 7.

## Files

| File | Step | Content |
|---|---|---|
| `ea_common.py` | - | Shared constants, rollout / source I/O, the exact anchor replay, pair repos, paired statistics |
| `anchors.py` | 1, 3 | `source`: deploy a policy, find each episode's anchor; `plot`: coverage panels |
| `train_base.sh`, `pick_base.py` | 2 | Retrain the base, evaluate every checkpoint, write `configs/base_policy_result.yaml` |
| `rollout.py` | 4, 5 | `run`: closed-loop NanoWM imagination from the anchors; `fidelity`: sim replay of the branches |
| `pack.py` | 6 | `select` / `pack` / `build` the Img and Real pair sets |
| `run_finetune.sh`, `report.py` | 7 | Fine-tune, evaluate, P1 / P2 report |
| `report_cov09.py`, `compare_ref.py`, `steps_cov09.py` | 7+ | Secondary analyses: success at coverage 0.9, arms vs a reference checkpoint, steps to 0.9 |
| `compare_video.py` | 7+ | One eval episode side by side for three runs |
| `select_check.py`, `wpick.py` | select study | Can NanoWM pick the better branch? (Step 0 and Step 1, see below) |

The scripts import `ea_common` as a sibling module (`proj/` is not a package); shared helpers from other experiments
live in the library (`nwaug.datagen.selection`, `nwaug_envs.pusht.deploy`). The drivers of the repeated runs
(`early_anchors_base100k`, `early_anchors_base40K_cos`, `early_anchors_select`) live in each run's `code/` dir.
They call these scripts with explicit arguments. Their `anchors_na10.py` wrapper is replaced by
`anchors.py source --na 10`.

## Early anchor (locked 2026-10-17)

On the 5-step grid of the image coverage estimate (`coverage_from_frame()["coverage_cal"]`), the anchor of an episode is the
**first** grid point t with

- `0.5 <= cov(t) < 0.8`
- `15 <= t <= 200`. t >= 15 gives NanoWM its 3 real context frames.
- `cov(t) - cov(t-10) <= 0.005` (no progress)
- the episode has not ended

Implemented by `anchors.py::first_anchor`, which wraps `nwaug.datagen.selection.qualifying` (the last-mile rule).

Old-base definition run (`define/`, 100 episodes, seeds 1050000+i):

| Metric | Value |
|---|---|
| Base success rate | 66% |
| Episodes with an anchor | 50 |
| Anchor time t | median 110 (p10 70, p90 170) |
| Success after the anchor | 58% |
| Anchor -> termination | median 114 steps (success only: 72) |

## Steps

1. **Anchor definition and coverage plots.** `anchors.py source --stage define` then `anchors.py plot`. Done.
2. **New base.** `train_base.sh tmux` retrains with the old base's recipe from scratch (100k steps, checkpoint every 10k,
   W&B project `early_anchors`) and evaluates every checkpoint with the shared eval protocol.
   `pick_base.py` picks the checkpoint whose success falls in `base_train.target_success` and writes
   `configs/base_policy_result.yaml`.
3. **Anchors from the new base.** `anchors.py source --stage source --ckpt <picked> --na 10` (`source_na10/`) reports
   the anchor -> termination steps as closed-loop iterations for f = 1, 2. The user sets `rollout.n_iter` from this
   report.
4. **Imagined rollout.** `rollout.py run` (NanoWM call and alignment from `nwaug/wm/closed_loop_rollout.py`):
   - The first query sees the real frame. `--lead-in` replays the real actions a[0:t].
   - Each later query sees the policy's own imagined frame. f NanoWM rounds run per query, for n_iter iterations.
   - A rollout is kept once its imagined coverage reaches `filter.success_cov` (0.9).
   - Sim replay is run only as a diagnostic, to report precision and recall.
5. **Real arm.** The same actions are replayed in sim to get real frames.
6. **Pack.** `pack.py select / pack / build`.
   - Filter: a branch is kept once its imagined coverage reaches 0.9. False positives stay; sim success is not used.
   - Pairs: imagined frame j (after round j) for j = 0 .. min(hit, 17), every round, re-plan or mid-chunk. The real
     anchor frame is excluded. Label = executed actions `exec_acts[5(j+1) : +9]`, ep_len 9 / drop 8.
   - Dose 5% = 1200 unique pairs, water-filled over anchors, j uniform inside a branch. Real arm = the same
     actions replayed in sim. The mix defaults to `balanced` (pool repeated 20x -> 50% of samples); `natural` is the ablation.
   - f2_n10: 2136/4288 branches kept, 256 anchors x 4-5 pairs, |img - real| cov at the pair frame mean 0.048.
     Repos `local/pusht_ea_f2n10_{img,real}_balanced`.
7. **Fine-tune and eval.**
   - Arms: Img and Real, seeds 1000 and 1001, via `BASE_CKPT=<new base> scripts/finetune_cl.sh`.
   - Evals: full-episode success, plus success from held-out anchors.
   - `criteria.md` is written before any fine-tune.

## Select study (`select_check.py`, `wpick.py`)

Both scripts replay branches in sim from the exact anchor state. The base policy then continues
(`select_check.na` = 10) until sim coverage >= `filter.success_cov` or the 300-step cap. Seeds are in the `select_check:`
and `wpick:` blocks of the config.

- **Step 0, `select_check.py run / report`.** Does the imagined coverage after the first chunk rank the k candidates
  of one anchor? The script compares W (the imagined argmax) with R (the mean of all k) in restricted-mean steps to 0.9.
- **Step 1, `wpick.py select / simcheck / pack / build / report`.** It starts from a noised rollout
  (`rollout.py run --k 4 --f 2 --inject-sigma 0.1`). W is the branch with the highest imagined peak, R is a random
  branch, and the experience comes from W's or R's branch up to W's peak. `simcheck` is the sim gate run before any
  fine-tune.
