# scripts/ -- shared shell recipes

`finetune_cl.sh` is the only script here.
Each experiment folder calls it.
Run it from the repository root.

```bash
bash research/nanowm_aug/proj/pusht/scripts/finetune_cl.sh <arm|repo=local/<id>> <seed> <steps> <drop>
```

| Argument                   | Meaning                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `arm` or `repo=local/<id>` | Training dataset. `repo=` takes any dataset.                     |
| `seed`, `steps`            | Training seed and number of steps.                               |
| `drop`                     | `drop_n_last_frames`. It must equal the pack `--ep-len` minus 1. |

Set `SAVE_FREQ=20000` to keep only the final checkpoint (about 2 GB instead of about 6 GB).
All arms of one comparison must use the same base checkpoint, seed, learning rate, steps, and `drop`.

The experiment drivers are in the experiment folders:

| Experiment | Drivers                                                              |
| ---------- | -------------------------------------------------------------------- |
| SAILOR     | `../sailor/run_value_model.sh`                                       |
| CUPID      | `../cupid/run_cupid_filter.sh`, `../cupid/run_cupid_select.sh`       |
| Heuristic  | `../heuristic/run_lastmile.sh`, `../heuristic/run_lastmile_seeds.sh` |
| Method 1   | `../method1/run_method1.sh`, `../method1/run_method1_multiseed.sh`   |

Shared parameter values are in `research/nanowm_aug/configs/common.yaml`.
The audit of every default is in `research/nanowm_aug/docs/parameter_audit.md`.
