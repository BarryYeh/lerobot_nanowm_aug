# SAILOR: value model on NanoWM latents

## Purpose

This experiment tests one question.
Can a value model, trained on NanoWM latents, rank candidate action chunks inside imagined rollouts?

The value model scores a candidate chunk before the robot executes it.
A good ranking lets the policy pick the best of N chunks.
The idea comes from SAILOR (arXiv 2506.05294).

The test has one gate.
Offline best-of-N must raise the success rate.
The lower bound of the one-sided 95% interval must be above zero.
Online best-of-N is out of scope.

## Method

1. The base policy plays PushT from random resets.
   This gives real frames and real outcomes (stage 1).
2. A second set of resets gives anchor states (stage 2).
   NanoWM imagines 4 candidate branches from each anchor.
   Two noise levels are used: sigma 0 and sigma 0.1.
3. The simulator replays each imagined branch exactly.
   The base policy then continues for 200 steps.
   This gives the true outcome label of every branch.
4. An ensemble of 5 value models trains on SD-VAE latents of 3 frames.
   Each model has a value head, a reward head (coverage), and a pose head.
5. An oracle ensemble trains on the true 8-dimensional state.
   It sets the upper bound.
   The gap between the two ensembles shows the loss from the encoder and from world-model errors.
6. The gate report scores held-out anchors.
   The search score is `V + beta * RM`, with beta in {0, 0.1, 0.3}.

The config is `research/nanowm_aug/configs/sailor_v1_reset.yaml`.
The pre-registration is `research/nanowm_aug/docs/decisions/sailor_v1.md`.
The overview is `research/nanowm_aug/docs/sailor_v1_overview_en.md`.

## How to run

Run all commands from the repository root.

1. Run a smoke test first.
   It writes only to `<out_dir>/smoke/`.

   ```bash
   bash research/nanowm_aug/proj/pusht/sailor/run_value_model.sh \
       research/nanowm_aug/configs/sailor_v1_reset.yaml --smoke
   ```

2. Run the full pipeline in a tmux session.

   ```bash
   mkdir -p research/nanowm_aug/logs
   tmux new-session -d -s sailor_v1 'bash research/nanowm_aug/proj/pusht/sailor/run_value_model.sh \
       research/nanowm_aug/configs/sailor_v1_reset.yaml 2>&1 | tee research/nanowm_aug/logs/sailor_v1.log'
   ```

3. To run one stage, add its name after the config path.
   The stages, in order, are:
   `collect`, `encode`, `anchors`, `imagine`, `label`, `test_anchors`, `test_imagine`, `test_label`, `train`, `report`.

4. To record videos of the top-scored branches, run `record_sailor_v1_videos.sh` or `topscore_video.py`.

## Outputs

All outputs go to `out_dir` in the config.
The default is `~/datastorage/nanowm_pusht_aug/value_model/sailor_v1`.
The repository `outputs/` folder is never used.

| Path under `out_dir`                     | Content                                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| `stage1/`                                | Real-frame value states and SD-VAE latents               |
| `seeds_stage2.npz`                       | Training anchors                                         |
| `stage2/`                                | Imagined branches and exact-replay labels                |
| `seeds_test.npz`, `test_stage2/`         | Test anchors and their labels (never used for training)  |
| `config.yaml`, `config_provenance.jsonl` | Resolved config and provenance of each run               |
| Gate report                              | Gate tables (G1 to G6) written by `value_gate_report.py` |

## Files

| File                                              | Role                                          |
| ------------------------------------------------- | --------------------------------------------- |
| `run_value_model.sh`                              | Stage driver                                  |
| `collect_value_states.py`                         | Stage 1 rollouts and query labels             |
| `make_reset_anchors.py`                           | Stage 2 and test anchors                      |
| `train_value_model.py`                            | Value and oracle ensemble training            |
| `value_gate_report.py`                            | Held-out gate report                          |
| `finetune_pool.py`                                | Selects branches and builds fine-tuning pools |
| `t2_checks.py`                                    | Simulator checks (noise and failure timing)   |
| `topscore_video.py`, `record_sailor_v1_videos.sh` | Videos of top-scored branches                 |

Shared code is in `nwaug/critic/` (value model, labels, replay) and `nwaug/wm/` (world-model rollouts).
