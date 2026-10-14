# CUPID: influence scores for imagined pairs

## Purpose

NanoWM imagined rollouts give extra training pairs (a frame and an action chunk).
Some pairs help the diffusion policy.
Other pairs hurt it, because the world model makes errors.

This experiment uses CUPID to find the pairs that help.
CUPID scores each pair by its influence on the policy performance.
The scores come from TRAK, a gradient-based data-attribution method.

Two settings use the same scoring code:

| Setting                | Question                                                                                                   | Config                           |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Filtering              | If we remove the lowest-scoring pairs from a trained-on dataset, does the policy improve?                  | `cupid_filter_untrimmed464.yaml` |
| Selection (add-one-in) | If we add only the highest-scoring pairs to the real data, does the policy recover from the top-wall band? | `cupid_select_topband.yaml`      |

Each setting compares CUPID against a random choice of the same size.
The comparison uses the same evaluation episodes for every arm.

## Method

1. Record test rollouts of the policy in `gym_pusht`.
   Each policy query is one test sample.
   Successful and failed episodes get opposite weights.
2. Compute projected per-sample gradients with TRAK.
   This covers the real samples, the imagined pairs, and the test queries (`nwaug/attribution/featurize.py`).
3. Compute one score per pair (`nwaug/attribution/score_pairs.py`).
   A high score means the pair moves the policy toward successful behavior.
4. Check the rank stability.
   A second run with other seeds must give a similar ranking (Spearman correlation of at least 0.8).
5. Build the arms.
   - Filtering: remove the lowest-scoring pairs, or the same number of random pairs.
   - Selection: add the k highest-scoring pairs, or k random pairs, to the real data.
6. Fine-tune each arm from the same base policy with the same settings.
7. Evaluate each arm on the same episode seeds.
   Report the success rates, the paired McNemar test, and the paired bootstrap interval.

In the selection setting, the pool of imagined pairs is a holdout.
The pool is scored but never enters the X^T X matrix.
The test rollouts start in the top-wall band, where the policy fails most often.

## How to run

Run all commands from the repository root.
Each script takes the config as its first argument.
After the config, name the stages to run.
With no stage names, the script runs all stages.

1. Filtering:

   ```bash
   bash research/nanowm_aug/proj/pusht/cupid/run_cupid_filter.sh \
       research/nanowm_aug/configs/cupid_filter_untrimmed464.yaml
   ```

   Stages: `record`, `featurize`, `replicate`, `score`, `build`, `finetune`, `eval`, `report`.

2. Selection:

   ```bash
   bash research/nanowm_aug/proj/pusht/cupid/run_cupid_select.sh \
       research/nanowm_aug/configs/cupid_select_topband.yaml
   ```

   Stages: `wait_stage1`, `seeds`, `generate`, `merge`, `pack`, `record`, `featurize`, `score`, `build`, `finetune`, `eval`, `report`.

3. Run long jobs in a tmux session and write the log to `research/nanowm_aug/logs/`.

The filtering script stops after `score` if the rank stability is below 0.8.
The selection script does not stop.
It marks the report as "reliability uncertain".

## Outputs

| Setting   | Folder                          | Main files                                                                                                              |
| --------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Filtering | `outputs/cupid/untrimmed464/`   | `test_rollouts.npz`, `features/`, `features_r1/`, `pair_scores.csv`, `score_summary.json`, `eval/<tag>/`, `report.md`   |
| Selection | `outputs/cupid/select_topband/` | `seeds_{pool,test,eval}.npz`, `pool_raw/`, `test_rollouts.npz`, `pair_scores.csv`, `report_band.md`, `report_random.md` |

Fine-tuned checkpoints go to `outputs/train/cl_aug/<tag>_d<drop>_s<seed>/`.
Built datasets go to the local LeRobot cache with the name `local/<repo_id>_{cupid,random}_...`.

## Files

| File                                         | Role                                             |
| -------------------------------------------- | ------------------------------------------------ |
| `run_cupid_filter.sh`, `run_cupid_select.sh` | Stage drivers                                    |
| `record_policy_rollouts.py`                  | Test rollouts (one row per policy query)         |
| `band_eval.py`                               | Evaluation from fixed top-wall-band start states |

The TRAK scoring code is in `nwaug/attribution/`.
The fine-tune script is `proj/pusht/scripts/finetune_cl.sh`.
