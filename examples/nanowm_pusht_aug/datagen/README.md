# `datagen/` — reusable helpers for the NanoWM PushT augmentation prototype

Named `datagen`, **not `utils`**: `utils` collides with nano-world-model's own top-level
`src/utils` package (imported by `imagined_rollout`) and with the repo-root `utils/`
doc-check scripts. A `utils/` package here would shadow nano-world-model's and break the
NanoWM worker.

Two independent groups. Import only what you need so each stays usable from the venv that
needs it.

## Stateless transforms (load in BOTH venvs — nanowm conda env + lerobot uv venv)

| module                   | what                                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `action_convert.py`      | NanoWM ⇄ diffusion-policy action converter (`pd_step`, `abs_targets_to_rel`, `rel_to_nanowm`, `nanowm_to_rel`, `rel_to_abs_targets`, `load_action_stats`). Pure numpy. |
| `coverage_from_frame.py` | coverage + block pose from an RGB frame (color-mask + moments). numpy + cv2. `goal_mask_224.npy` is the cached goal mask (rebuilt by `_sanity()` in the lerobot venv). |

```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))          # dir that contains datagen/
from datagen import action_convert as ac
from datagen.coverage_from_frame import coverage_from_frame
```

## Dataset converters (lerobot venv only — need `LeRobotDataset`)

`dataset_converter.py` holds `LeRobotDatasetConverter` (ABC) + `ConverterConfig`. It owns
the `create → per-episode {add_frame*; save_episode} → finalize` lifecycle; a subclass
supplies the schema (`from_args`), the source (`load_source`), and `iter_episodes()`.

`run_converter.py` is the single entry point:

```bash
uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py --list
uv run --extra training python examples/nanowm_pusht_aug/datagen/run_converter.py <name> --help
```

| name                 | module                          | source → output                                                                             |
| -------------------- | ------------------------------- | ------------------------------------------------------------------------------------------- |
| `pusht-noise`        | `convert_pusht_noise.py`        | `pusht_noise/{train,val}` archive → `local/pusht_noise_224_fi<N>[_imgonly]`                 |
| `pusht-imgonly`      | `convert_pusht_imgonly.py`      | `lerobot/pusht` → `local/pusht_imgonly` (drop `observation.state`)                          |
| `pushtnoise-denoise` | `convert_pushtnoise_denoise.py` | `pusht_noise` train → `local/pushtnoise_clean185_imgonly` (185 min-residual denoised demos) |
| `imagined`           | `pack_imagined.py`              | `imagined_rollout.py` `.npz` shards → `local/pusht_noise_224_imgonly_nanoWM_fi5`             |

Each converter module also runs standalone (`python datagen/convert_pusht_imgonly.py …`).

`pack_closed_loop.py` is deliberately **not** in that registry: it packs accepted
closed-loop rollouts with a different episode layout (one 10-frame episode per
imagined-frame/action-chunk pair, trained with `drop_n_last_frames=9`) and has its own
CLI. See its module docstring for why the layout differs.
