# nanowm_aug

Research code that uses NanoWM imagined rollouts as data augmentation for a PushT diffusion policy.
The code is a prototype.
It is not part of the `lerobot` package.

## Experiments

Each experiment has its own folder and README.
Read the README first.

| Experiment | Folder                                                    | Question                                                                         |
| ---------- | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| SAILOR     | [`proj/pusht/sailor/`](proj/pusht/sailor/README.md)       | Can a value model on NanoWM latents rank imagined action chunks?                 |
| CUPID      | [`proj/pusht/cupid/`](proj/pusht/cupid/README.md)         | Do influence scores find the imagined pairs that help the policy?                |
| Heuristic  | [`proj/pusht/heuristic/`](proj/pusht/heuristic/README.md) | Does picking the chunk with the highest imagined coverage help at the last mile? |
| Method 1   | [`proj/pusht/method1/`](proj/pusht/method1/README.md)     | Do imagined frames harm the policy?                                              |
| Imagined replay | [`proj/pusht/imagined_replay/`](proj/pusht/imagined_replay/README.md) | With real successful actions as labels, can round-1 imagined frames replace the real frames? |
| Early anchors | [`proj/pusht/early_anchors/`](proj/pusht/early_anchors/README.md) | Do rollouts that succeed in imagination from stalled mid-coverage states help a weaker base? |

## Layout

| Path                | Content                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `nwaug/`            | Library without environment-specific code: `wm`, `datagen`, `critic`, `attribution`, `eval` |
| `nwaug_envs/pusht/` | PushT adapter: action conversion, coverage, state placement, dataset converter              |
| `proj/pusht/`       | One folder per experiment, plus `scripts/finetune_cl.sh` (shared fine-tune script)          |
| `configs/`          | `common.yaml` (shared values), `envs/pusht.yaml` (PushT facts), one config per experiment   |
| `tests/`            | Offline tests                                                                               |
| `docs/`             | Background, parameter audit, results, and decision records                                  |

## Rules

1. The import root is `research/nanowm_aug`.
   Import code as `nwaug.*` or `nwaug_envs.pusht.*`.
   Do not add other `sys.path` entries.
2. `proj/` is not a package.
   Run its scripts directly.
3. There are two uv environments.
   They cannot be merged.
   - The lerobot environment (Python 3.12 or newer) runs everything except the NanoWM worker.
   - The NanoWM environment (`third_party/nano-world-model/.venv`, Python 3.11) runs `nwaug/wm/imagined_rollout.py`.
4. The two environments exchange data only through `.npz` files.
5. These modules load in both environments.
   They must not import `lerobot` or `gym_pusht`:
   `nwaug_envs/pusht/action_convert.py`, `nwaug_envs/pusht/coverage.py`, `nwaug/datagen/provenance.py`,
   `nwaug/attribution/config_io.py`, `nwaug/critic/value_model.py`, `nwaug/critic/labels.py`.
   The first four use only numpy, cv2, yaml, and the standard library.
6. Do not edit `third_party/nano-world-model`.
   Update it with `git subtree pull`.
7. Generated folders (`imagined/`, `cl/`, `diag/`, `logs/`, `output/`) are ignored by git.
   Experiment outputs go to `~/datastorage/nanowm_pusht_aug/` or to `outputs/`.

## How to run

1. Install the NanoWM environment and the weights.
   See `docs/background.md` for the commands.
2. Run commands from the repository root with `uv run`.
3. Run the tests:

   ```bash
   uv run pytest research/nanowm_aug/tests -q
   ```

4. Run a long job in a tmux session.
   Write its log to `research/nanowm_aug/logs/`.

Shared parameters are in `configs/common.yaml`.
The audit of every default is in `docs/parameter_audit.md`.
