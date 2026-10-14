# Parameter audit

Every `.py` and `.sh` in `nwaug/`, `nwaug_envs/` and `proj/` was scanned for argparse defaults, module constants and shell variables.
Values that appeared **identically in two or more files** moved to `research/nanowm_aug/configs/common.yaml`;
per-experiment values live in each script's argparse defaults (the only source any script reads), not
in per-script yaml.

Configs resolve `extends:` via `nwaug/attribution/config_io.py`, which merges the chain and snapshots the
RESOLVED config into the run's `out_dir`. It deliberately imports no lerobot, so a shell recipe can
read one value without paying for `import lerobot`.

## Shared -- same value in 2+ files -> `common.yaml`

| parameter                      | where it was repeated                                 | value                                           |
| ------------------------------ | ----------------------------------------------------- | ----------------------------------------------- |
| `--device`                     | 7 py files                                            | `cuda`                                          |
| `--max-steps` / episode length | make_topband_seeds, ood_failure_map                   | 300                                             |
| `BASE_CKPT`                    | wm_fidelity_ood, proj/pusht/scripts/finetune_cl.sh    | `pusht_imgonly/obs1_s100000`                    |
| `T_OUTLINE`                    | ood_failure_map, wm_fidelity_ood (**byte-identical**) | 8-point T outline                               |
| `AGENT_BOX`                    | wm_fidelity_ood                                       | `(22.0, 489.0)`                                 |
| `IMG` / render size            | wm_fidelity_ood, closed_loop_rollout                  | 224                                             |
| `--fi` / frame_interval        | closed_loop_rollout, every converter                  | 5                                               |
| success threshold              | `--accept-cov`, gym_pusht default                     | 0.95                                            |
| `--num-sampling-steps`         | closed_loop_rollout, wm_fidelity_ood                  | 30                                              |
| `--wm-batch`                   | closed_loop_rollout, wm_fidelity_ood                  | 24                                              |
| `NANOWM_PY`                    | closed_loop_rollout, wm_fidelity_ood                  | `third_party/nano-world-model/.venv/bin/python` |
| eval protocol                  | run*cupid*\*.sh `eval_pusht`                          | see note below                                  |
| finetune hyperparams           | proj/pusht/scripts/finetune_cl.sh                     | seed 1000, 20k steps, lr 1e-5, bs 64, drop 8    |
| `lerobot_home`                 | proj/pusht/scripts/finetune_cl.sh                     | `~/.cache/huggingface/lerobot`                  |
| `dataset_dir`                  | nwaug/wm/imagined_rollout.py                          | the DINO-WM archive                             |

Eval protocol facts: CUPID `eval_pusht` leaves `n_action_steps` unset unless the config passes
one (the checkpoint's own value is used). The `eval:` block of `common.yaml`
(n=1000, na=10) is the protocol for comparison runs and must be passed explicitly.

## Per experiment -- differs by design, named separately

| parameter          | values                                                                 | why not shared                                                                                                                                                       |
| ------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--seed`           | 0 / 9000 / 100000 / 200000 / 300000 / 400000 / 500000                  | **deliberately disjoint namespaces** so experiments never share initial states. Recorded as `seed_namespace` in the common config: read it before picking a new one. |
| `--k`              | closed_loop_rollout 4 (branches/seed) vs ood_failure_map 20 (clusters) | same flag name, unrelated quantities                                                                                                                                 |
| `--n-starts`       | wm_fidelity_ood 48                                                     | cost per start differs by orders of magnitude                                                                                                                        |
| `--n-seeds`        | closed_loop_rollout 32, make_seeds 200                                 | different stages                                                                                                                                                     |
| `--batch`          | ood_failure_map 50                                                     | rollout vs WM memory footprint                                                                                                                                       |
| `--out` / `--work` | one path per experiment                                                | must not collide                                                                                                                                                     |

## Configs

| file                             | drives / who reads it via `--config`                                                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `common.yaml`                    | the shared base; every other file `extends:` it                                                                                                      |
| `cupid_filter_untrimmed464.yaml` | `proj/pusht/cupid/run_cupid_filter.sh` → record / featurize / score / build_filtered_datasets / diffusion_influence_loss                             |
| `cupid_select_topband.yaml`      | `proj/pusht/cupid/run_cupid_select.sh` → make_topband_seeds / record / featurize / score / build_selected_datasets                                   |
| `sailor_v1_reset.yaml`           | `proj/pusht/sailor/run_value_model.sh` → collect_value_states / make_topband_seeds / label_imagined_branches / train_value_model / value_gate_report |
