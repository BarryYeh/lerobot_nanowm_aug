# Method 1: imagined-frame harm test

## Purpose

Imagined frames are not exact.
NanoWM makes small errors in the block pose and in the image.
This experiment asks a narrow question.
Do the imagined frames, by themselves, harm the policy?

The test removes every other difference.
Both arms use the same states and the same action labels.
Only the frame is different: imagined in one arm, real in the other.
The result is a non-inferiority test.
The imagined arm must not be worse than the real-frame arm by more than 2 percentage points.

## Method

The success criterion is fixed before any data exists.
It is in `~/datastorage/nanowm_pusht_aug/lastmile/method1/criteria.md`.

1. Pick starts.
   Take 121 of the 179 accepted last-mile starts (see `proj/pusht/heuristic/`).
   Keep one random candidate for each start.
2. Realize the branches.
   Replay each picked branch in the simulator.
   Capture the real frame at the step where NanoWM predicted each imagined frame.
3. Label action chunk with the base policy.
   The base policy samples a 16-step chunk on each frame.
   The imagined frame and the real frame use the same random seed (common random numbers).
   This gives the pair sets `img_self` and `real_self`.
4. Pack both pair sets with the same layout as the last-mile experiment.
   Each pair set has 121 pairs.
   The build stage adds each set to the real dataset.
5. Fine-tune each arm from the base policy with training seeds 1000 and 1001.
6. Evaluate on the same 2000 episodes.
   Compare `img_self`, `real_self`, and the control arm (ctrl) with the pooled paired bootstrap.
   Use the margin of 0.02.

An optional offline check measures the spread of NanoWM.
It re-imagines each last-mile candidate with 8 NanoWM seeds.
This check is diagnostic only.
It creates no training data.

## How to run

Run all commands from the repository root.

1. Run the manual preparation steps once.
   Use the paths from the criteria file for `<dir>` and `<base>`.

   ```bash
   uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py pick --out <dir>
   uv run --extra training python research/nanowm_aug/proj/pusht/method1/realize_branches.py \
       --src <dir>/src --dst <dir>/realized
   uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py label \
       --out <dir> --ckpt <base>
   ```

2. Run the experiment in a detached tmux session.
   The default GPUs are `6 7`.

   ```bash
   GPUS="6 7" bash research/nanowm_aug/proj/pusht/method1/run_method1.sh tmux
   ```

   The stages are: `pack`, `build`, `finetune`, `eval`, `pooled`.
   A finished stage leaves a marker in `<out>/.done/`.
   To resume after a failure, run the same command again.

3. The driver runs the offline NanoWM-seed check last.
   A failure of this check never stops the main result.
   To run it alone, use `run_method1_multiseed.sh`.

## Round 2

Round 2 asks a follow-up question.
What happens when NanoWM imagines one more round, and the imagined data doubles?

- Each picked branch already holds a second frame at step 10, imagined and real.
  Nothing is re-imagined or re-realized.
- `img_r12` and `real_r12` each have 242 pairs.
  The first 121 are the method 1 pairs (symlinks, same labels).
  The other 121 are new pairs on the round-2 frame, labeled in the same way.
- The dose is about 1.0% of sampleable frames.
- The primary test is the same: `img_r12 - real_r12`, margin 0.02.
  The pooled comparison also links in the method 1 evaluations, to compare `r12` against `self` (round 1 only).
- The criterion is in `~/datastorage/nanowm_pusht_aug/lastmile/method1_r2/criteria.md`.
  All outputs go to the same directory.

```bash
uv run --extra training python research/nanowm_aug/proj/pusht/method1/method1_prep.py label_r2 \
    --out ~/datastorage/nanowm_pusht_aug/lastmile/method1_r2 --ckpt <base>
M1_DIR=method1_r2 M1_TAG=m1r2 M1_ARMS="img_r12 real_r12" M1_N=242 M1_EXTRA_EVAL="img_self real_self" \
    M1_POOLED_PAIRS="img_r12:real_r12 img_r12:img_self real_r12:real_self img_r12:ctrl real_r12:ctrl" \
    M1_MULTISEED=0 GPUS="6 7" bash research/nanowm_aug/proj/pusht/method1/run_method1.sh tmux
```

`offline.json` reports the action gap, coverage and block CoG error for each round.

## Outputs

All outputs go to `~/datastorage/nanowm_pusht_aug/lastmile/method1/`.

| Path                                  | Content                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `src/`, `realized/`                   | Picked branches and the same branches with real frames                       |
| `pairs_img_self/`, `pairs_real_self/` | Packable pair sets                                                           |
| `offline.json`                        | Action gap (imagined vs real) and image coverage gap                         |
| `eval/`                               | Evaluation results for each arm and seed, plus `pooled.json` and `pooled.md` |
| `multiseed/`                          | Output of the offline NanoWM-seed check                                      |
| `status.txt`, `run.log`               | Progress and log                                                             |

## Files

| File                       | Role                                               |
| -------------------------- | -------------------------------------------------- |
| `run_method1.sh`           | Stage driver                                       |
| `run_method1_multiseed.sh` | Offline NanoWM-seed check (two parts, one per GPU) |
| `method1_prep.py`          | `pick` and `label` steps                           |
| `realize_branches.py`      | Replays branches to get real simulator frames      |
| `method1_multiseed.py`     | NanoWM seed spread (`run` and `analyze`)           |

The pooled comparison uses `proj/pusht/heuristic/lastmile_pooled.py`.
