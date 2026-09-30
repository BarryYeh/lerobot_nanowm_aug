# eval/ -- parameter audit

Every `.py` and `.sh` here was scanned for argparse defaults, module constants and shell variables.
Values that appeared **identically in two or more files** moved to `configs/nanowm_pusht_common.yaml`;
per-experiment values live in each script's argparse defaults (the only source any script reads), not
in per-script yaml.

Configs resolve `extends:` via `attribution/config_io.py`, which merges the chain and snapshots the
RESOLVED config into the run's `out_dir`. It deliberately imports no lerobot, so a shell recipe can
read one value without paying for `import lerobot`.

## Shared -- same value in 2+ files -> `nanowm_pusht_common.yaml`

| parameter                      | where it was repeated                                                        | value                                        |
| ------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------- |
| `--device`                     | 7 py files                                                                   | `cuda`                                       |
| `--max-steps` / episode length | agent_value_map, make_seeds, mode_probe, ood_failure_map                     | 300                                          |
| `BASE_CKPT`                    | agent_value_map, mode_probe, wm_fidelity_ood, finetune_cl.sh, run_dose500.sh | `pusht_imgonly/obs1_s100000`                 |
| `T_OUTLINE`                    | ood_failure_map, wm_fidelity_ood (**byte-identical**)                        | 8-point T outline                            |
| `AGENT_BOX`                    | agent_value_map, wm_fidelity_ood                                             | `(22.0, 489.0)`                              |
| `IMG` / render size            | mode_probe, wm_fidelity_ood                                                  | 224                                          |
| `--fi` / frame_interval        | closed_loop_rollout, filter_roc, every converter                             | 5                                            |
| success threshold              | filter_roc `SUCCESS`, eval_baseline `--success-threshold`, `--accept-cov`    | 0.95                                         |
| `--num-sampling-steps`         | closed_loop_rollout, wm_fidelity_ood                                         | 30                                           |
| `--wm-batch`                   | closed_loop_rollout, wm_fidelity_ood                                         | 24                                           |
| `NANOWM_PY`                    | closed_loop_rollout, wm_fidelity_ood                                         | `third_party/nano-world-model/.venv/bin/python` |
| eval protocol                  | proto_eval_pusht.sh, run_pad_hold.sh, run_dose\*.sh                          | see note below                               |
| finetune hyperparams           | finetune_cl.sh, run_dose464/500.sh                                           | seed 1000, 20k steps, lr 1e-5, bs 64, drop 8 |
| train hyperparams              | proto_train_pusht.sh, proto_train_pushtnoise.sh, train_baseline.sh           | 100k steps, bs 64, horizon 16, na 8          |
| `lerobot_home`                 | finetune_cl.sh, all three train scripts, run_dose\*.sh                       | `~/.cache/huggingface/lerobot`               |
| `dataset_dir`                  | train_baseline.sh, run_eval_suite.sh, imagined_rollout.py                    | the DINO-WM archive                          |

Eval protocol facts: `proto_eval_pusht.sh` defaults to N=200 and leaves `n_action_steps` unset (the
checkpoint's own value is used); `run_dose464.sh` and `run_dose500.sh` pass 200 and no NA. The
`eval:` block of `nanowm_pusht_common.yaml` (n=1000, na=10) is the protocol for comparison runs and
must be passed explicitly.

## Per experiment -- differs by design, named separately

| parameter          | values                                                                 | why not shared                                                                                                                                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--seed`           | 0 / 9000 / 100000 / 200000 / 300000 / 400000 / 500000                  | **deliberately disjoint namespaces** so experiments never share initial states. Recorded as `seed_namespace` in the common config: read it before picking a new one. agent_value_map reusing mode_probe's 400000 is intentional -- it branches from those starts. |
| `--k`              | closed_loop_rollout 4 (branches/seed) vs ood_failure_map 20 (clusters) | same flag name, unrelated quantities                                                                                                                                                                                                                              |
| `--n-starts`       | mode_probe 4, agent_value_map 10, wm_fidelity_ood 48                   | cost per start differs by orders of magnitude                                                                                                                                                                                                                     |
| `--n-seeds`        | closed_loop_rollout 32, make_seeds 200                                 | different stages                                                                                                                                                                                                                                                  |
| `--batch`          | agent_value_map 250, ood_failure_map 50                                | rollout vs WM memory footprint                                                                                                                                                                                                                                    |
| `--out` / `--work` | one path per experiment                                                | must not collide                                                                                                                                                                                                                                                  |

## Configs

| file                              | drives / who reads it via `--config`                                                                                                      |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `nanowm_pusht_common.yaml`        | the shared base; every other file `extends:` it                                                                                           |
| `cupid_filter_untrimmed464.yaml`  | `run_cupid_filter.sh` → record / featurize / score / build_filtered_datasets / diffusion_influence_loss                                   |
| `cupid_select_topband.yaml`       | `run_cupid_select.sh` → make_topband_seeds / record / featurize / score / build_selected_datasets (+ ridge override)                      |
| `cupid_select_topband_ridge.yaml` | `run_cupid_select.sh` ridge score path                                                                                                    |
| `cupid_bidir_verified464.yaml`    | `run_cupid_bidir.sh` → build_bidir_datasets                                                                                               |
| `cupid_bidir_verified464_loo.yaml`| `run_cupid_bidir.sh` LOO direction (record / featurize / score)                                                                           |
| `cupid_bidir_verified464_aoi.yaml`| `run_cupid_bidir.sh` AOI direction (record / featurize / score)                                                                           |
| `value_model_topband.yaml`        | `run_value_model.sh` → collect_value_states / make_topband_seeds / label_imagined_branches / train_value_model / value_gate_report        |

The six unread per-script yaml (closed_loop_rollout, mode_probe, agent_value_map, ood_failure_map,
wm_fidelity_ood, baseline_eval under `configs/`) were removed because nothing read them and several
disagreed with the code (review H1). Migrating a script to read a config is a later task.
