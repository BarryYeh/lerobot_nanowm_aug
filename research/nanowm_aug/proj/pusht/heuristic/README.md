# Heuristic: last-mile selection by imagined coverage

## Purpose

The base policy often gets stuck close to the goal.
The block is almost in place, but the policy does not finish the push.
This is the "last mile".

This experiment tests a simple heuristic.
For each stuck state, NanoWM imagines several candidate action chunks.
The heuristic picks the chunk with the highest imagined coverage.
The question: do the picked chunks, added to the training data, raise the success rate more than random chunks do?

The heuristic needs no learned value model.
It uses only the coverage that NanoWM imagines.

## Method

The success criterion is fixed before any data exists.
It is in `~/datastorage/nanowm_pusht_aug/lastmile/criteria.md`.

1. Collect stuck late states.
   Run the base policy without noise.
   Keep a state when the image-based coverage estimate is in a fixed band and has stopped rising.
2. Imagine candidates.
   For each stuck state, sample 4 chunks at each of 3 noise levels (sigma 0, 0.1, 0.25).
   This gives 12 candidates per state.
   NanoWM imagines 2 rounds for each candidate.
3. Select.
   Define the score of a candidate as its maximum imagined coverage.
   - W: the candidate with the highest score.
   - R: a random candidate of the same state.
4. Apply gate 1.
   At least 100 states must be accepted.
5. Apply gate 2 (diagnostic only).
   Replay all candidates in the simulator.
   The true success rate of W minus that of R must be at least 10 percentage points.
   This gate never selects or labels training data.
6. Pack the real anchor frame and the executed chunk of each pick.
   The pairs make up about 5% of the training samples.
7. Fine-tune from the base policy: one arm with W pairs, one with R pairs.
   The control arm (ctrl) uses real data only.
8. Evaluate W, R, and ctrl on the same 2000 episodes.
   Compare the arms with paired statistics.

`run_lastmile_seeds.sh` repeats steps 7 and 8 for a second training seed (1001) on the same datasets.
`lastmile_pooled.py` pools the seeds.
Its confidence interval comes from a paired bootstrap over episodes.

## How to run

Run all commands from the repository root.

1. Run a smoke test.
   It uses small sizes, applies no gates, and stops after `build`.

   ```bash
   bash research/nanowm_aug/proj/pusht/heuristic/run_lastmile.sh --smoke
   ```

2. Run the full experiment in a detached tmux session.
   The default GPUs are `6 7`.
   Change them with the `GPUS` variable.

   ```bash
   GPUS="6 7" bash research/nanowm_aug/proj/pusht/heuristic/run_lastmile.sh tmux
   ```

3. Replicate with a second training seed.

   ```bash
   GPUS="6 7" bash research/nanowm_aug/proj/pusht/heuristic/run_lastmile_seeds.sh tmux
   ```

The driver is resumable.
A finished stage leaves a marker in `<out>/.done/`.
To resume after a failure, run the same command again.
The stages are: `collect`, `imagine`, `select`, `simcheck`, `pack`, `build`, `finetune`, `eval`, `compare`.

## Outputs

All outputs go to `~/datastorage/nanowm_pusht_aug/lastmile/`.

| Path                    | Content                                              |
| ----------------------- | ---------------------------------------------------- |
| `seeds_late.npz`        | Stuck start states                                   |
| `imag/sigma*/`          | Imagined branches for each noise level               |
| `sel/`                  | `W` and `R` picks, `selection.json`, `simcheck.json` |
| `eval/<arm>/`           | Evaluation results for W, R, and ctrl                |
| `eval_seeds/`           | Evaluation results for the second seed               |
| `status.txt`, `run.log` | Progress and log                                     |

## Files

| File                                       | Role                                  |
| ------------------------------------------ | ------------------------------------- |
| `run_lastmile.sh`, `run_lastmile_seeds.sh` | Stage drivers                         |
| `collect_late_states.py`                   | Step 1: stuck late states             |
| `select_lastmile.py`                       | Step 3 and gate 1: W and R picks      |
| `lastmile_simcheck.py`                     | Gate 2: simulator replay (diagnostic) |
| `lastmile_pooled.py`                       | Pooled multi-seed comparison          |

The Method 1 experiment reuses the accepted states from this experiment (see `proj/pusht/method1/`).
