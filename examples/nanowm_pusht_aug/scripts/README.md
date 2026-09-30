# scripts/ -- shell recipes

Every `.sh` in this project lives here. **All of them are run from the repo root**, not from this
directory; the ones that need it `cd` there themselves from `$BASH_SOURCE`. Parameter defaults are
audited in `../eval/README.md`; the shared values live in `configs/nanowm_pusht_common.yaml`.

## Building blocks -- called by the pipelines below, or directly

| script                                                         | what                                                                                                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `finetune_cl.sh <arm\|repo=local/<id>> <seed> <steps> <drop>`  | fine-tune one arm from `ckpt.base`. `repo=local/<id>` takes any dataset; `SAVE_FREQ=20000` keeps only the final checkpoint (~2 GB instead of ~6 GB) |
| `proto_eval_pusht.sh <ckpt> <out> [n] [seed] [n_action_steps]` | `lerobot-eval` on gym_pusht at fps 10. Omitting `n_action_steps` keeps the checkpoint's own                                                         |
| `proto_train_pusht.sh [n_obs] [seed]`                          | train from scratch on `local/pusht_imgonly`                                                                                                         |
| `proto_train_pushtnoise.sh [seed]`                             | train from scratch on `local/pushtnoise_clean185_imgonly`                                                                                           |
| `train_baseline.sh [seed] [n_obs]`                             | the fps-2 `a` baseline on `local/pusht_noise_224_fi5_imgonly`                                                                                       |

## Pipelines -- each runs a whole comparison end to end

| script                                                   | what                                                                                                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_cupid_filter.sh <config.yaml> [stage ...]`          | CUPID filtering: record / featurize / replicate / score / build / finetune / eval / report                                                           |
| `run_cupid_select.sh <config.yaml> [stage ...]`          | CUPID add-one-in in the top-wall band: wait_stage1 / seeds / generate / merge / pack / record / featurize / score / build / finetune / eval / report |
| `run_cupid_bidir.sh <shared_config.yaml> [stage ...]`    | CUPID LOO vs AOI on one pool: sidecar / record / featurize / replicate / score / gate / build / finetune / eval / report                             |
| `run_dose464.sh`                                         | equal-dose (N=464) untrimmed vs trimmed vs verified                                                                                                  |
| `run_dose500.sh`                                         | generate a fresh pool, then the same three arms at N=500                                                                                             |
| `run_pad_hold.sh`                                        | pad-mode and `n_action_steps` sweep at drop=9                                                                                                        |
| `run_hesitation_validation.sh`                           | whether the augmented policy hesitates near the goal                                                                                                 |
| `run_eval_suite.sh [rollouts_per_init] [n_random]`       | headline `a` vs `a+aug` at the fps-2 control rate, both init distributions                                                                           |
| `run_value_model.sh <config.yaml> [stage ...] [--smoke]` | value/reward-model (V/RM) on NanoWM latents: seeds / collect / encode / imagine / label / train / report                                             |

## Note on the `proto_` prefix

`proto_eval_pusht.sh`, `proto_train_pusht.sh` and `proto_train_pushtnoise.sh` keep it only because
renaming them would touch every log and output directory that already references them. New scripts
should be named for what they do, without a stage prefix.
