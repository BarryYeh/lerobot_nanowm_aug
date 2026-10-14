# `nwaug/datagen/` — dataset generation helpers

The package is named `datagen`, **not `utils`**. A top-level `utils` package would shadow
nano-world-model's own `src/utils` (used by `nwaug/wm/imagined_rollout.py`) and break the NanoWM worker.
For the same reason, no top-level package in this project is named `envs`.

## Modules

| module                 | venv    | what                                                                                                                                    |
| ---------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `provenance.py`        | both    | `provenance_entry`: records time, argv, git sha, branch, dirty flag. Stdlib only.                                                       |
| `dataset_converter.py` | lerobot | `LeRobotDatasetConverter` (ABC) and `ConverterConfig`. Owns the `create → per-episode {add_frame*; save_episode} → finalize` lifecycle. |
| `run_converter.py`     | lerobot | Single entry point for the converter registry.                                                                                          |
| `pack_closed_loop.py`  | lerobot | Packs accepted closed-loop rollouts into a `LeRobotDataset`.                                                                            |

The PushT-specific transforms are in `nwaug_envs/pusht/`:

| module                     | venv    | what                                                                                                   |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| `action_convert.py`        | both    | NanoWM ⇄ diffusion-policy action converter. Pure numpy.                                                |
| `coverage.py`              | both    | Coverage and block pose from an RGB frame. numpy and cv2. `goal_mask_224.npy` is the cached goal mask. |
| `convert_pusht_imgonly.py` | lerobot | The `pusht-imgonly` converter.                                                                         |

Both-venv modules must never import `lerobot` or `gym_pusht`.

## Converters

```bash
uv run --extra training python research/nanowm_aug/nwaug/datagen/run_converter.py --list
uv run --extra training python research/nanowm_aug/nwaug/datagen/run_converter.py <name> --help
```

| name            | module                                      | source → output                                                    |
| --------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| `pusht-imgonly` | `nwaug_envs/pusht/convert_pusht_imgonly.py` | `lerobot/pusht` → `local/pusht_imgonly` (drop `observation.state`) |

`DATASET_DIR` must be the parent of `pusht_noise/` (for example `~/Downloads/bmw48-osfstorage-datasets-archive`).
Open-loop mode and a live `action_convert.load_action_stats` call make the vendored NanoWM loader write
`wm_stats_cache/*.pt` (about 2 KB, deterministic action stats) into `pusht_noise/{train,val}/`.
Keep those files. Deleting them only forces the next run to recreate them.

`pack_closed_loop.py` is deliberately **not** in the registry.
It packs one 10-frame episode per imagined-frame and action-chunk pair, trained with `drop_n_last_frames=9`.
It has its own CLI. See its module docstring for the reason.
