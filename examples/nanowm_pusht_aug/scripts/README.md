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
| `train_latent_base.sh [n_obs] [seed]`                          | base v2: train from scratch on SD-VAE latents, `local/pusht_latent` (eval: `eval/eval_latent_policy.py`)                                            |
| `train_baseline.sh [seed] [n_obs]`                             | the fps-2 `a` baseline on `local/pusht_noise_224_fi5_imgonly`                                                                                       |

## Pipelines -- each runs a whole comparison end to end

| script                                                   | what                                                                                                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_cupid_filter.sh <config.yaml> [stage ...]`          | CUPID filtering: record / featurize / replicate / score / build / finetune / eval / report                                                           |
| `run_cupid_select.sh <config.yaml> [stage ...]`          | CUPID add-one-in in the top-wall band: wait_stage1 / seeds / generate / merge / pack / record / featurize / score / build / finetune / eval / report |
| `run_pad_hold.sh`                                        | pad-mode and `n_action_steps` sweep at drop=9                                                                                                        |
| `run_hesitation_validation.sh`                           | whether the augmented policy hesitates near the goal                                                                                                 |
| `run_eval_suite.sh [rollouts_per_init] [n_random]`       | headline `a` vs `a+aug` at the fps-2 control rate, both init distributions                                                                           |
| `run_value_model.sh <config.yaml> [stage ...] [--smoke]` | value/reward-model (V/RM) on NanoWM latents: seeds / collect / encode / imagine / label / train / report                                             |

Ridge scoring (`cupid_select_topband_ridge.yaml`) and the bidirectional LOO/AOI driver (`run_cupid_bidir.sh` and `cupid_bidir_verified464*.yaml`) were removed after tag `archive/pre-cleanup`. `attribution/build_bidir_datasets.py` remains and can be called by hand. SAILOR v1 variants `sailor_v1_r6.yaml`, `sailor_v1_s025.yaml`, `run_sailor_v1_s025.sh`, and `value_model_topband.yaml` were removed in the same cleanup; the kept entry is `sailor_v1_reset.yaml` with `run_value_model.sh`. Equal-dose drivers `run_dose464.sh` and `run_dose500.sh`, plus `eval/proto_filter_roc.py` and `eval/proto_make_seeds.py`, are at that tag too. `run_pad_hold.sh` and `run_hesitation_validation.sh` stay. SAILOR expand (`sailor_expand.yaml`, `run_sailor_expand.sh`, `run_sailor_expand_dose.sh`, `run_traj_ft.sh`, and the expand select/score/build/diag scripts) is at that tag as well. `run_lastmile.sh` reads `nanowm.num_sampling_steps` and `nanowm.wm_batch` from `configs/nanowm_pusht_common.yaml`. `critic/realize_branches.py` stays for Method 1.

## Note on the `proto_` prefix

`proto_eval_pusht.sh`, `proto_train_pusht.sh` and `proto_train_pushtnoise.sh` keep it only because
renaming them would touch every log and output directory that already references them. New scripts
should be named for what they do, without a stage prefix.
